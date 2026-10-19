"""
Base classes and utilities for BBN Annotation Agents
"""

from typing import Any
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum
import json


class Speaker(str, Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"


class AnnotationSource(str, Enum):
    AGENT = "agent"
    EXPERT = "expert"


@dataclass
class SpanAnnotation:
    """A single span annotation"""
    span_id: str
    text: str
    start: int
    end: int
    label: str
    reasoning: str = ""
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "span_id": self.span_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "reasoning": getattr(self, "reasoning", "") or "",
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanAnnotation":
        """Reconstruct SpanAnnotation from dictionary with resilient field mapping."""
        start = data.get("start", data.get("start_pos", 0))
        end = data.get("end", data.get("end_pos", len(data.get("text", ""))))
        return cls(
            span_id=str(data.get("span_id", "")),
            text=str(data.get("text", "")),
            start=int(start) if start is not None else 0,
            end=int(end) if end is not None else 0,
            label=str(data.get("label", "")),
            reasoning=str(data.get("reasoning", "") or ""),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
        )


@dataclass
class RelationAnnotation:
    """A relation between two annotations"""
    from_span_id: str
    to_span_id: str
    relation_type: Literal["response_to", "elicitation_of"]
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "from": self.from_span_id,
            "to": self.to_span_id,
            "type": self.relation_type,
        }
        if getattr(self, "reasoning", ""):
            data["reasoning"] = self.reasoning
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationAnnotation":
        """Reconstruct RelationAnnotation from dictionary."""
        return cls(
            from_span_id=str(data.get("from") or data.get("from_span_id", "")),
            to_span_id=str(data.get("to") or data.get("to_span_id", "")),
            relation_type=data.get("type") or data.get("relation_type", "response_to"),
            reasoning=str(data.get("reasoning", "") or ""),
        )


@dataclass
class TurnAnnotation:
    """Complete annotation for a single turn"""
    turn_id: int
    speaker: Speaker
    text: str
    spikes_stage: Optional[str] = None
    spans: List[SpanAnnotation] = field(default_factory=list)
    relations: List[RelationAnnotation] = field(default_factory=list)
    reasoning_steps: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker.value if isinstance(self.speaker, Speaker) else str(self.speaker),
            "text": self.text,
            "annotations": {
                "spikes_stage": self.spikes_stage,
                "spans": [s.to_dict() for s in self.spans],
                "relations": [r.to_dict() for r in self.relations],
            },
            "reasoning_steps": [
                step.to_dict() if hasattr(step, "to_dict") else step
                for step in self.reasoning_steps
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnAnnotation":
        """Reconstruct TurnAnnotation from dictionary."""
        annotations = data.get("annotations", {})
        spans_raw = annotations.get("spans", []) if isinstance(annotations, dict) else []
        relations_raw = annotations.get("relations", []) if isinstance(annotations, dict) else []
        spikes_stage = annotations.get("spikes_stage") if isinstance(annotations, dict) else data.get("spikes_stage")

        speaker_val = data.get("speaker", Speaker.PATIENT.value)
        if isinstance(speaker_val, str) and speaker_val.lower() == "clinician":
            speaker = Speaker.CLINICIAN
        else:
            speaker = Speaker.PATIENT

        return cls(
            turn_id=int(data.get("turn_id", 0)),
            speaker=speaker,
            text=str(data.get("text", "")),
            spikes_stage=spikes_stage,
            spans=[SpanAnnotation.from_dict(s) for s in spans_raw if isinstance(s, dict)],
            relations=[RelationAnnotation.from_dict(r) for r in relations_raw if isinstance(r, dict)],
            reasoning_steps=data.get("reasoning_steps", []),
        )


@dataclass
class AnnotationResult:
    """Complete annotation result for a conversation"""
    conversation_id: str
    turns: List[TurnAnnotation]
    agent_type: str  # 'react' or 'multi_agent'
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "agent_type": self.agent_type,
            "metadata": self.metadata,
            "turns": [t.to_dict() for t in self.turns],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationResult":
        """Reconstruct AnnotationResult from dictionary."""
        return cls(
            conversation_id=str(data.get("conversation_id", data.get("id", ""))),
            agent_type=str(data.get("agent_type", "react")),
            metadata=data.get("metadata", {}),
            turns=[TurnAnnotation.from_dict(t) for t in data.get("turns", []) if isinstance(t, dict)],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AnnotationResult":
        """Reconstruct AnnotationResult from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AgentConfig:
    """Configuration for annotation agents"""
    model: str = "gpt-4o"  # or "claude-3-5-sonnet-20241022", etc.
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: Optional[str] = None
    provider: Literal["openai", "anthropic"] = "openai"

    # Agent behavior
    include_reasoning: bool = True

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": self.provider,
            "include_reasoning": self.include_reasoning,
        }


# Label definitions from schema
PATIENT_LABELS = {
    # Empathic Opportunities - Feelings
    "explicit_feeling": "Direct expression of emotion, emotive behavior, or mental state",
    "implicit_feeling": "Indirect expression of feeling through references to experiences",
    # Empathic Opportunities - Appreciation
    "explicit_appreciation": "Direct attitude toward things, events, actions, or behaviors",
    "implicit_appreciation": "Indirect attitude toward things, events, actions, or behaviors",
    # Empathic Opportunities - Judgement
    "explicit_judgement": "Direct attitude toward self or others",
    "implicit_judgement": "Indirect attitude toward self or others",
}

CLINICIAN_LABELS = {
    # Elicitations - Feeling
    "direct_elicitation_feeling": "Direct inquiry about patient's emotions or mental state",
    "indirect_elicitation_feeling": "Indirect inquiry about experiences or emotive behaviors",
    # Elicitations - Appreciation
    "direct_elicitation_appreciation": "Direct inquiry about patient's appreciation of things/events",
    "indirect_elicitation_appreciation": "Indirect exploration of preferences",
    # Elicitations - Judgement
    "direct_elicitation_judgement": "Direct inquiry about patient's judgement of self or others",
    "indirect_elicitation_judgement": "Indirect inquiry about behaviors or interpretations",
    # Responses - Acceptance
    "acceptance_positive_regard_explicit_judgement": "Expression of positive judgment of the patient as a person",
    "acceptance_positive_regard_implicit_judgement": "Expression of judgement of patient's thoughts or feelings",
    "acceptance_positive_regard_repetition": "Repeating or paraphrasing patient's words without countering",
    "acceptance_positive_regard_allowing": "Allowing patients to express feelings fully",
    "acceptance_neutral_support_appreciation": "Appreciation of ideas, feelings, or behaviors regarding normality",
    "acceptance_neutral_support_judgement": "Denying negative self-assessment by the patient",
    # Responses - Sharing
    "sharing_feeling": "Sharing patient feelings through expressed agreement",
    "sharing_appreciation": "Sharing patient views through expressed agreement",
    "sharing_judgement": "Sharing patient judgements through expressed agreement",
    # Responses - Understanding
    "understanding_feeling": "Understanding and acknowledgement of patient's feelings",
    "understanding_appreciation": "Understanding and acknowledgement of patient's views",
    "understanding_judgement": "Understanding and acknowledgement of patient's judgements",
}

SPIKES_STAGES = {
    "setting": "Establishing time, place, and environment",
    "perception": "Understanding patient's current knowledge",
    "invitation": "Assessing how much information patient wants",
    "knowledge": "Delivering medical information",
    "empathy": "Responding to patient emotions",
    "strategy": "Discussing treatment plans",
}

ALL_LABELS = {**PATIENT_LABELS, **CLINICIAN_LABELS}


def get_labels_for_speaker(speaker: Speaker) -> Dict[str, str]:
    """Get applicable labels based on speaker role"""
    if speaker == Speaker.PATIENT:
        return PATIENT_LABELS
    return CLINICIAN_LABELS


def generate_span_id(turn_id: int, index: int) -> str:
    """Generate a unique span ID"""
    return f"span_t{turn_id}_{index}"
