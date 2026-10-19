"""Multi-Agent Annotation System with coordinated specialist agents."""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .base import (
    AgentConfig,
    AnnotationResult,
    TurnAnnotation,
    SpanAnnotation,
    RelationAnnotation,
    Speaker,
    generate_span_id,
    PATIENT_LABELS,
    CLINICIAN_LABELS,
    SPIKES_STAGES,
)
from .prompts import (
    EO_DETECTOR_PROMPT,
    RESPONSE_CLASSIFIER_PROMPT,
    SPIKES_TAGGER_PROMPT,
    RELATION_LINKER_PROMPT,
    MULTI_AGENT_USER_PROMPT_TEMPLATE,
    SPIKES_USER_PROMPT_TEMPLATE,
    RELATION_USER_PROMPT_TEMPLATE,
)
from .react_agent import extract_json_from_text, resolve_span_coordinates

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles for specialized agents."""
    COORDINATOR = "coordinator"
    EO_DETECTOR = "eo_detector"
    RESPONSE_CLASSIFIER = "response_classifier"
    SPIKES_TAGGER = "spikes_tagger"
    RELATION_LINKER = "relation_linker"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: AgentRole
    to_agent: AgentRole
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMAgent:
    """Base class for specialist LLM agents."""

    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        config: AgentConfig
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.config.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.api_key)
            elif self.config.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.config.api_key)
        return self._client

    def _call_llm(self, user_prompt: str) -> str:
        """Execute LLM call."""
        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""

        elif self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.content[0].text if response.content else ""

        raise ValueError(f"Unknown provider: {self.config.provider}")

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse structured JSON from response."""
        return extract_json_from_text(response)

    def process(self, message: AgentMessage) -> AgentMessage:
        """Process an agent message."""
        raise NotImplementedError


class EODetectorAgent(BaseLLMAgent):
    """Specialist agent for detecting patient Empathic Opportunities (EOs)."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.EO_DETECTOR,
            system_prompt=EO_DETECTOR_PROMPT,
            config=config
        )

    def process(self, message: AgentMessage) -> AgentMessage:
        turn = message.content.get("turn", {})
        context = message.content.get("context", "")

        if turn.get("speaker", "").lower() != "patient":
            return AgentMessage(
                from_agent=self.role,
                to_agent=AgentRole.COORDINATOR,
                content={"annotations": [], "turn_id": turn.get("turn_id")}
            )

        user_prompt = MULTI_AGENT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0),
            speaker="patient",
            text=turn.get("text", ""),
            context=context
        )

        response = self._call_llm(user_prompt)
        parsed = self._parse_json(response)

        annotations = []
        for ann in parsed.get("annotations", []):
            label = ann.get("label", "")
            if label in PATIENT_LABELS:
                annotations.append(ann)

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            content={
                "annotations": annotations,
                "turn_id": turn.get("turn_id")
            }
        )


class ResponseClassifierAgent(BaseLLMAgent):
    """Specialist agent for classifying clinician responses and elicitations."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.RESPONSE_CLASSIFIER,
            system_prompt=RESPONSE_CLASSIFIER_PROMPT,
            config=config
        )

    def process(self, message: AgentMessage) -> AgentMessage:
        turn = message.content.get("turn", {})
        context = message.content.get("context", "")

        if turn.get("speaker", "").lower() != "clinician":
            return AgentMessage(
                from_agent=self.role,
                to_agent=AgentRole.COORDINATOR,
                content={"annotations": [], "turn_id": turn.get("turn_id")}
            )

        user_prompt = MULTI_AGENT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0),
            speaker="clinician",
            text=turn.get("text", ""),
            context=context
        )

        response = self._call_llm(user_prompt)
        parsed = self._parse_json(response)

        annotations = []
        for ann in parsed.get("annotations", []):
            label = ann.get("label", "")
            if label in CLINICIAN_LABELS:
                annotations.append(ann)

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            content={
                "annotations": annotations,
                "turn_id": turn.get("turn_id")
            }
        )


class SPIKESTaggerAgent(BaseLLMAgent):
    """Specialist agent for assigning SPIKES protocol stages."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.SPIKES_TAGGER,
            system_prompt=SPIKES_TAGGER_PROMPT,
            config=config
        )

    def process(self, message: AgentMessage) -> AgentMessage:
        turn = message.content.get("turn", {})
        context = message.content.get("context", "")

        if turn.get("speaker", "").lower() != "clinician":
            return AgentMessage(
                from_agent=self.role,
                to_agent=AgentRole.COORDINATOR,
                content={"spikes_stage": None, "turn_id": turn.get("turn_id")}
            )

        user_prompt = SPIKES_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0),
            text=turn.get("text", ""),
            context=context
        )

        response = self._call_llm(user_prompt)
        parsed = self._parse_json(response)

        spikes_stage = parsed.get("spikes_stage")
        if spikes_stage and str(spikes_stage).lower() in SPIKES_STAGES:
            spikes_stage = str(spikes_stage).lower()
        else:
            spikes_stage = None

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            content={
                "spikes_stage": spikes_stage,
                "reasoning": parsed.get("reasoning", ""),
                "turn_id": turn.get("turn_id")
            }
        )


class RelationLinkerAgent(BaseLLMAgent):
    """Specialist agent for linking EOs to clinician responses."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.RELATION_LINKER,
            system_prompt=RELATION_LINKER_PROMPT,
            config=config
        )

    def process(self, message: AgentMessage) -> AgentMessage:
        patient_eos = message.content.get("patient_eos", [])
        clinician_responses = message.content.get("clinician_responses", [])
        conversation = message.content.get("conversation", "")

        if not patient_eos or not clinician_responses:
            return AgentMessage(
                from_agent=self.role,
                to_agent=AgentRole.COORDINATOR,
                content={"relations": []}
            )

        user_prompt = RELATION_USER_PROMPT_TEMPLATE.format(
            patient_eos=json.dumps(patient_eos, indent=2),
            clinician_responses=json.dumps(clinician_responses, indent=2),
            conversation=conversation
        )

        response = self._call_llm(user_prompt)
        parsed = self._parse_json(response)

        return AgentMessage(
            from_agent=self.role,
            to_agent=AgentRole.COORDINATOR,
            content={"relations": parsed.get("relations", [])}
        )


class MultiAgentSystem:
    """Coordinator that orchestrates specialist agents for BBN annotation."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.eo_detector = EODetectorAgent(self.config)
        self.response_classifier = ResponseClassifierAgent(self.config)
        self.spikes_tagger = SPIKESTaggerAgent(self.config)
        self.relation_linker = RelationLinkerAgent(self.config)

    def annotate_conversation(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True,
        parallel: bool = True
    ) -> AnnotationResult:
        """Annotate an entire conversation using specialized agents."""
        conversation_id = conversation.get("id", "unknown")
        turns = conversation.get("turns", [])

        annotated_turns: List[TurnAnnotation] = []
        all_patient_eos: List[Dict] = []
        all_clinician_responses: List[Dict] = []

        for i, turn in enumerate(turns):
            context = self._build_context(turns[:i])
            turn_annotation = self._process_turn(turn, context, parallel)
            annotated_turns.append(turn_annotation)

            for span in turn_annotation.spans:
                span_info = {
                    "turn_id": turn_annotation.turn_id,
                    "span_id": span.span_id,
                    "text": span.text,
                    "label": span.label,
                }
                if turn_annotation.speaker == Speaker.PATIENT:
                    all_patient_eos.append(span_info)
                else:
                    all_clinician_responses.append(span_info)

        if include_relations and all_patient_eos and all_clinician_responses:
            relations = self._link_relations(all_patient_eos, all_clinician_responses, turns)
            self._attach_relations(annotated_turns, relations)

        return AnnotationResult(
            conversation_id=conversation_id,
            turns=annotated_turns,
            agent_type="multi_agent",
            metadata={
                "model": self.config.model,
                "provider": self.config.provider,
                "agents": ["eo_detector", "response_classifier", "spikes_tagger", "relation_linker"]
            }
        )

    def _process_turn(
        self,
        turn: Dict[str, Any],
        context: str,
        parallel: bool = True
    ) -> TurnAnnotation:
        """Process a single turn with appropriate specialist agents."""
        turn_id = turn.get("turn_id", 0)
        speaker_str = turn.get("speaker", "patient").lower()
        speaker = Speaker.CLINICIAN if "clinician" in speaker_str else Speaker.PATIENT
        text = turn.get("text", "")

        message = AgentMessage(
            from_agent=AgentRole.COORDINATOR,
            to_agent=AgentRole.EO_DETECTOR,
            content={"turn": turn, "context": context}
        )

        annotations: List[SpanAnnotation] = []
        spikes_stage = None

        if speaker == Speaker.PATIENT:
            result = self.eo_detector.process(message)
            annotations = self._convert_annotations(
                result.content.get("annotations", []),
                turn_id,
                text
            )
        else:
            if parallel:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    f_resp = executor.submit(self.response_classifier.process, message)
                    f_spikes = executor.submit(self.spikes_tagger.process, message)
                    response_result = f_resp.result()
                    spikes_result = f_spikes.result()
            else:
                response_result = self.response_classifier.process(message)
                spikes_result = self.spikes_tagger.process(message)

            annotations = self._convert_annotations(
                response_result.content.get("annotations", []),
                turn_id,
                text
            )
            spikes_stage = spikes_result.content.get("spikes_stage")

        return TurnAnnotation(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            spikes_stage=spikes_stage,
            spans=annotations,
            relations=[],
        )

    def _convert_annotations(
        self,
        raw_annotations: List[Dict],
        turn_id: int,
        text: str
    ) -> List[SpanAnnotation]:
        """Convert raw annotation dictionaries to validated SpanAnnotation objects."""
        spans = []
        for i, ann in enumerate(raw_annotations):
            span_text = ann.get("text", "").strip()
            if not span_text or span_text not in text:
                continue

            raw_start = ann.get("start", 0)
            raw_end = ann.get("end", len(span_text))
            start, end = resolve_span_coordinates(text, span_text, raw_start, raw_end)

            confidence = ann.get("confidence", 0.9)
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
            else:
                confidence = None

            spans.append(SpanAnnotation(
                span_id=generate_span_id(turn_id, i),
                text=text[start:end] if 0 <= start < end <= len(text) else span_text,
                start=start,
                end=end,
                label=ann.get("label", "unknown"),
                reasoning=ann.get("reasoning", ""),
                confidence=confidence,
            ))

        return spans

    def _link_relations(
        self,
        patient_eos: List[Dict],
        clinician_responses: List[Dict],
        turns: List[Dict]
    ) -> List[RelationAnnotation]:
        """Link patient EOs with clinician responses."""
        conversation_summary = "\n".join([
            f"Turn {t.get('turn_id', 0)} ({t.get('speaker', 'unknown')}): {t.get('text', '')[:120]}..."
            for t in turns
        ])

        message = AgentMessage(
            from_agent=AgentRole.COORDINATOR,
            to_agent=AgentRole.RELATION_LINKER,
            content={
                "patient_eos": patient_eos,
                "clinician_responses": clinician_responses,
                "conversation": conversation_summary
            }
        )

        result = self.relation_linker.process(message)
        relations = []
        for rel in result.content.get("relations", []):
            from_id = rel.get("from_span_id") or rel.get("from", "")
            to_id = rel.get("to_span_id") or rel.get("to", "")
            rel_type = rel.get("relation_type") or rel.get("type", "response_to")
            if from_id and to_id:
                relations.append(RelationAnnotation(
                    from_span_id=from_id,
                    to_span_id=to_id,
                    relation_type=rel_type,
                ))

        return relations

    def _attach_relations(
        self,
        turns: List[TurnAnnotation],
        relations: List[RelationAnnotation]
    ):
        span_to_turn = {}
        for turn in turns:
            for span in turn.spans:
                span_to_turn[span.span_id] = turn

        for relation in relations:
            if relation.from_span_id in span_to_turn:
                turn = span_to_turn[relation.from_span_id]
                turn.relations.append(relation)

    def _build_context(self, previous_turns: List[Dict], max_turns: int = 5) -> str:
        if not previous_turns:
            return "No previous context."

        recent = previous_turns[-max_turns:]
        parts = []
        for turn in recent:
            speaker = turn.get("speaker", "unknown").upper()
            text = turn.get("text", "")[:200]
            parts.append(f"{speaker}: {text}")

        return "\n".join(parts)


def create_multi_agent_system(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> MultiAgentSystem:
    """Create a configured multi-agent system."""
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"

    config = AgentConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return MultiAgentSystem(config)
