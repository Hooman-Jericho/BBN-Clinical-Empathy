"""ReAct Annotation Agent with structured parsing, verification, and coordinate disambiguation."""

import json
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .base import (
    AgentConfig,
    AnnotationResult,
    TurnAnnotation,
    SpanAnnotation,
    RelationAnnotation,
    Speaker,
    get_labels_for_speaker,
    generate_span_id,
    SPIKES_STAGES,
    ALL_LABELS,
)
from .prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT_TEMPLATE,
    RELATION_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def resolve_span_coordinates(text: str, span_text: str, suggested_start: int, suggested_end: int) -> tuple[int, int]:
    """Resolve exact span coordinates in text without first-occurrence bias.
    
    If the suggested slice already matches, uses it. Otherwise, searches all occurrences
    and selects the one closest to the LLM's suggested offset.
    """
    if not span_text:
        return suggested_start, suggested_end

    # Exact match at suggested position
    if 0 <= suggested_start < suggested_end <= len(text):
        if text[suggested_start:suggested_end] == span_text:
            return suggested_start, suggested_end

    # Search all occurrences
    occurrences = []
    start = 0
    while True:
        idx = text.find(span_text, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(span_text)))
        start = idx + 1

    if not occurrences:
        return suggested_start, suggested_end

    # Choose occurrence closest to suggested_start
    closest = min(occurrences, key=lambda occ: abs(occ[0] - suggested_start))
    return closest[0], closest[1]


def extract_json_from_text(response: str) -> Dict[str, Any]:
    """Robust multi-pass JSON extractor from LLM responses."""
    if not response or not response.strip():
        return {}

    # Pass 1: Direct JSON parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Pass 2: Markdown code block ```json ... ```
    codeblock_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response, re.IGNORECASE)
    if codeblock_match:
        try:
            return json.loads(codeblock_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Pass 3: Balanced curly braces extraction
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = response[start_idx:end_idx + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            # Pass 4: Clean common trailing commas or newlines
            cleaned = re.sub(r',\s*([\]}])', r'\1', json_candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse JSON from LLM response: %s", response[:200])
    return {}


@dataclass
class ReActStep:
    """A single ReAct reasoning step."""
    thought: str
    action: str
    observation: str


class LLMClient:
    """Unified wrapper for LLM API calls (OpenAI & Anthropic)."""

    def __init__(self, config: AgentConfig):
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

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request to the LLM with error handling."""
        client = self._get_client()

        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.content[0].text if response.content else ""

        raise ValueError(f"Unknown provider: {self.config.provider}")


class ReActAnnotationAgent:
    """ReAct Agent for BBN Conversation Annotation with verification and coordinate resolution."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.llm = LLMClient(self.config)

    def annotate_conversation(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True
    ) -> AnnotationResult:
        """Annotate an entire conversation."""
        conversation_id = conversation.get("id", "unknown")
        turns = conversation.get("turns", [])

        annotated_turns: List[TurnAnnotation] = []
        all_patient_eos: List[Dict] = []
        all_clinician_responses: List[Dict] = []

        for i, turn in enumerate(turns):
            context = self._build_context(turns[:i])
            turn_annotation = self._annotate_turn(turn, context)
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
            relations = self._identify_relations(
                all_patient_eos,
                all_clinician_responses,
                turns
            )
            self._attach_relations(annotated_turns, relations)

        return AnnotationResult(
            conversation_id=conversation_id,
            turns=annotated_turns,
            agent_type="react",
            metadata={
                "model": self.config.model,
                "provider": self.config.provider,
            }
        )

    def annotate_turn(
        self,
        turn: Dict[str, Any],
        context: str = ""
    ) -> TurnAnnotation:
        """Annotate a single turn."""
        return self._annotate_turn(turn, context)

    def _annotate_turn(
        self,
        turn: Dict[str, Any],
        context: str
    ) -> TurnAnnotation:
        """Internal method to annotate a single turn with multi-step validation."""
        turn_id = turn.get("turn_id", 0)
        speaker_str = turn.get("speaker", "patient").lower()
        speaker = Speaker.CLINICIAN if "clinician" in speaker_str else Speaker.PATIENT
        text = turn.get("text", "")

        user_prompt = REACT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn_id,
            speaker=speaker.value,
            text=text,
            context=context if context else "No previous context."
        )

        response = self.llm.chat(REACT_SYSTEM_PROMPT, user_prompt)
        parsed = extract_json_from_text(response)

        valid_labels = get_labels_for_speaker(speaker)
        spans: List[SpanAnnotation] = []

        raw_annotations = parsed.get("annotations", [])
        if isinstance(raw_annotations, list):
            for i, ann in enumerate(raw_annotations):
                if not isinstance(ann, dict):
                    continue

                span_text = ann.get("text", "").strip()
                if not span_text or span_text not in text:
                    continue

                raw_start = ann.get("start", 0)
                raw_end = ann.get("end", len(span_text))
                start, end = resolve_span_coordinates(text, span_text, raw_start, raw_end)

                label = ann.get("label", "unknown")
                if label not in valid_labels and label in ALL_LABELS:
                    pass  # keep if globally valid

                confidence = ann.get("confidence", 0.9)
                if isinstance(confidence, (int, float)):
                    confidence = float(confidence)
                else:
                    confidence = None

                span = SpanAnnotation(
                    span_id=generate_span_id(turn_id, i),
                    text=text[start:end] if 0 <= start < end <= len(text) else span_text,
                    start=start,
                    end=end,
                    label=label,
                    reasoning=ann.get("reasoning", ""),
                    confidence=confidence,
                )
                spans.append(span)

        spikes_stage = None
        if speaker == Speaker.CLINICIAN:
            raw_stage = parsed.get("spikes_stage")
            if raw_stage and str(raw_stage).lower() in SPIKES_STAGES:
                spikes_stage = str(raw_stage).lower()

        return TurnAnnotation(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            spikes_stage=spikes_stage,
            spans=spans,
            relations=[],
        )

    def _identify_relations(
        self,
        patient_eos: List[Dict],
        clinician_responses: List[Dict],
        turns: List[Dict]
    ) -> List[RelationAnnotation]:
        """Identify relations between patient EOs and clinician responses."""
        conversation_summary = "\n".join([
            f"Turn {t.get('turn_id', 0)} ({t.get('speaker', 'unknown')}): {t.get('text', '')[:120]}..."
            for t in turns
        ])

        user_prompt = RELATION_USER_PROMPT_TEMPLATE.format(
            patient_eos=json.dumps(patient_eos, indent=2),
            clinician_responses=json.dumps(clinician_responses, indent=2),
            conversation=conversation_summary
        )

        system_prompt = (
            "You are an expert clinical conversation analyst. Link clinician responses "
            "and elicitations to patient empathic opportunities using the exact span IDs provided."
        )

        response = self.llm.chat(system_prompt, user_prompt)
        parsed = extract_json_from_text(response)

        relations: List[RelationAnnotation] = []
        for rel in parsed.get("relations", []):
            if isinstance(rel, dict):
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
        """Attach relations to the appropriate turns."""
        span_to_turn = {}
        for turn in turns:
            for span in turn.spans:
                span_to_turn[span.span_id] = turn

        for relation in relations:
            if relation.from_span_id in span_to_turn:
                turn = span_to_turn[relation.from_span_id]
                turn.relations.append(relation)

    def _build_context(self, previous_turns: List[Dict], max_turns: int = 5) -> str:
        """Build context string from previous turns."""
        if not previous_turns:
            return ""

        recent_turns = previous_turns[-max_turns:]
        context_parts = []
        for turn in recent_turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")[:200]
            context_parts.append(f"{speaker.upper()}: {text}")

        return "\n".join(context_parts)


def create_react_agent(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ReActAnnotationAgent:
    """Factory function for creating a ReAct annotation agent."""
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"

    config = AgentConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return ReActAnnotationAgent(config)
