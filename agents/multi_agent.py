"""Multi-Agent Annotation System with Coordinator, Specialists, and Consensus Critic."""

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
    MULTI_AGENT_COORDINATOR_PROMPT,
    MULTI_AGENT_USER_PROMPT_TEMPLATE,
    SPIKES_USER_PROMPT_TEMPLATE,
    RELATION_USER_PROMPT_TEMPLATE,
)
from .schemas import (
    CoordinatorPlanSchema,
    CriticReviewSchema,
    SpanPredictionSchema,
    RelationListSchema,
)
from .react_agent import extract_json_from_text, resolve_span_coordinates

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles for specialized agents."""
    COORDINATOR = "coordinator"
    EO_DETECTOR = "eo_detector"
    RESPONSE_CLASSIFIER = "response_classifier"
    SPIKES_TAGGER = "spikes_tagger"
    CRITIC_CONSENSUS = "critic_consensus"
    RELATION_LINKER = "relation_linker"


@dataclass
class AgentMessage:
    """Message passed between agents in the collaboration network."""
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
        return extract_json_from_text(response)


class CoordinatorAgent(BaseLLMAgent):
    """Coordinator that orchestrates specialist dispatching based on conversation dynamics."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.COORDINATOR,
            system_prompt=MULTI_AGENT_COORDINATOR_PROMPT,
            config=config,
        )

    def plan_turn(self, turn: Dict[str, Any], context: str) -> CoordinatorPlanSchema:
        """Analyze turn and return structured dispatch plan."""
        speaker = turn.get("speaker", "patient").lower()
        prompt = (
            f"Formulate an orchestration plan for this turn:\n"
            f"Speaker: {speaker}\nText: \"{turn.get('text', '')}\"\n"
            f"Context: {context if context else 'Initial turn.'}\n\n"
            f"Return JSON adhering to schema:\n"
            f'{{"speaker_role": "{speaker}", "active_specialists": ["..."], "coordination_strategy": "..."}}'
        )
        resp = self._call_llm(prompt)
        parsed = self._parse_json(resp)
        try:
            return CoordinatorPlanSchema.model_validate(parsed)
        except Exception:
            specialists = ["eo_detector"] if "patient" in speaker else ["response_classifier", "spikes_tagger"]
            return CoordinatorPlanSchema(
                speaker_role=speaker,
                active_specialists=specialists,
                coordination_strategy="Standard clinical analysis dispatch.",
            )


class EODetectorAgent(BaseLLMAgent):
    """Specialist agent for detecting patient Empathic Opportunities (EOs)."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.EO_DETECTOR,
            system_prompt=EO_DETECTOR_PROMPT,
            config=config,
        )

    def process(self, turn: Dict[str, Any], context: str) -> List[Dict[str, Any]]:
        user_prompt = MULTI_AGENT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0),
            speaker="patient",
            text=turn.get("text", ""),
            context=context,
        )
        resp = self._call_llm(user_prompt)
        parsed = self._parse_json(resp)
        annotations = []
        for ann in parsed.get("annotations", []):
            if isinstance(ann, dict) and ann.get("label") in PATIENT_LABELS:
                annotations.append(ann)
        return annotations


class ResponseClassifierAgent(BaseLLMAgent):
    """Specialist agent for classifying clinician responses and elicitations."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.RESPONSE_CLASSIFIER,
            system_prompt=RESPONSE_CLASSIFIER_PROMPT,
            config=config,
        )

    def process(self, turn: Dict[str, Any], context: str) -> List[Dict[str, Any]]:
        user_prompt = MULTI_AGENT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0),
            speaker="clinician",
            text=turn.get("text", ""),
            context=context,
        )
        resp = self._call_llm(user_prompt)
        parsed = self._parse_json(resp)
        annotations = []
        for ann in parsed.get("annotations", []):
            if isinstance(ann, dict) and ann.get("label") in CLINICIAN_LABELS:
                annotations.append(ann)
        return annotations


class SPIKESTaggerAgent(BaseLLMAgent):
    """Specialist agent for identifying SPIKES protocol stages."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.SPIKES_TAGGER,
            system_prompt=SPIKES_TAGGER_PROMPT,
            config=config,
        )

    def process(self, turn: Dict[str, Any], context: str) -> Optional[str]:
        user_prompt = SPIKES_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0),
            text=turn.get("text", ""),
            context=context,
        )
        resp = self._call_llm(user_prompt)
        parsed = self._parse_json(resp)
        stage = parsed.get("spikes_stage")
        if stage and str(stage).lower() in SPIKES_STAGES:
            return str(stage).lower()
        return None


class CriticConsensusAgent(BaseLLMAgent):
    """Critic and consensus agent that cross-examines specialist proposals to eliminate errors and overlaps."""

    def __init__(self, config: AgentConfig):
        system_prompt = (
            "You are the Lead Clinical Consensus Critic in a multi-agent annotation system.\n"
            "Your role is to rigorously evaluate candidate span annotations proposed by specialist agents.\n"
            "Tasks:\n"
            "1. Eliminate false positives that do not strictly meet appraisal criteria.\n"
            "2. Resolve overlapping spans and ensure boundaries align with exact semantic units.\n"
            "3. Confirm label consistency with the speaker's role.\n"
            "Output JSON with approved_spans, rejected_spans, and critic_notes."
        )
        super().__init__(
            role=AgentRole.CRITIC_CONSENSUS,
            system_prompt=system_prompt,
            config=config,
        )

    def evaluate_proposals(
        self,
        turn: Dict[str, Any],
        proposed_spans: List[Dict[str, Any]],
        context: str,
    ) -> List[Dict[str, Any]]:
        """Cross-examine candidate spans and return verified consensus list."""
        if not proposed_spans:
            return []

        prompt = (
            f"Turn {turn.get('turn_id', 0)} ({turn.get('speaker', '')}): \"{turn.get('text', '')}\"\n"
            f"Context: {context}\n"
            f"Specialist Proposed Spans: {json.dumps(proposed_spans, indent=2)}\n\n"
            f"Evaluate proposals and return JSON: {{\"approved_spans\": [...], \"rejected_spans\": [...], \"critic_notes\": \"...\"}}"
        )

        resp = self._call_llm(prompt)
        parsed = self._parse_json(resp)
        try:
            review = CriticReviewSchema.model_validate(parsed)
            if review.approved_spans:
                return [s.model_dump() for s in review.approved_spans]
        except Exception:
            pass

        # Fallback to proposed if critic was unavailable
        return proposed_spans


class RelationLinkerAgent(BaseLLMAgent):
    """Specialist agent for linking EOs to clinician responses."""

    def __init__(self, config: AgentConfig):
        super().__init__(
            role=AgentRole.RELATION_LINKER,
            system_prompt=RELATION_LINKER_PROMPT,
            config=config,
        )

    def link(
        self,
        patient_eos: List[Dict],
        clinician_responses: List[Dict],
        turns: List[Dict],
    ) -> List[RelationAnnotation]:
        if not patient_eos or not clinician_responses:
            return []

        summary = "\n".join([f"T{t.get('turn_id')}: {t.get('text')[:100]}" for t in turns])
        prompt = RELATION_USER_PROMPT_TEMPLATE.format(
            patient_eos=json.dumps(patient_eos, indent=2),
            clinician_responses=json.dumps(clinician_responses, indent=2),
            conversation=summary,
        )

        resp = self._call_llm(prompt)
        parsed = self._parse_json(resp)
        relations = []
        try:
            validated = RelationListSchema.model_validate(parsed)
            for r in validated.relations:
                relations.append(RelationAnnotation(
                    from_span_id=r.from_span_id,
                    to_span_id=r.to_span_id,
                    relation_type=r.relation_type,
                ))
        except Exception:
            pass
        return relations


class MultiAgentSystem:
    """True Multi-Agent System orchestrating Coordinator, Specialists, and Consensus Critic."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.coordinator = CoordinatorAgent(self.config)
        self.eo_detector = EODetectorAgent(self.config)
        self.response_classifier = ResponseClassifierAgent(self.config)
        self.spikes_tagger = SPIKESTaggerAgent(self.config)
        self.critic = CriticConsensusAgent(self.config)
        self.relation_linker = RelationLinkerAgent(self.config)

    def _process_turn(
        self,
        turn: Dict[str, Any],
        context: str,
        parallel: bool = True,
    ) -> TurnAnnotation:
        turn_id = turn.get("turn_id", 0)
        speaker_str = turn.get("speaker", "patient").lower()
        speaker = Speaker.CLINICIAN if "clinician" in speaker_str else Speaker.PATIENT
        text = turn.get("text", "")

        # 1. Coordinator plans dispatch
        plan = self.coordinator.plan_turn(turn, context)

        proposals = []
        spikes_stage = None

        # 2. Specialist execution
        if speaker == Speaker.PATIENT:
            proposals = self.eo_detector.process(turn, context)
        else:
            if parallel:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    f_resp = executor.submit(self.response_classifier.process, turn, context)
                    f_spikes = executor.submit(self.spikes_tagger.process, turn, context)
                    proposals = f_resp.result()
                    spikes_stage = f_spikes.result()
            else:
                proposals = self.response_classifier.process(turn, context)
                spikes_stage = self.spikes_tagger.process(turn, context)

        # 3. Consensus Critic evaluates and filters proposals
        consensus_spans = self.critic.evaluate_proposals(turn, proposals, context)

        # 4. Convert and disambiguate coordinates
        validated_spans = self._convert_annotations(consensus_spans, turn_id, text)

        return TurnAnnotation(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            spikes_stage=spikes_stage,
            spans=validated_spans,
            relations=[],
            reasoning_steps=[f"Coordinator Plan: {plan.coordination_strategy}"],
        )

    def _convert_annotations(
        self,
        raw_annotations: List[Dict],
        turn_id: int,
        text: str,
    ) -> List[SpanAnnotation]:
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

    def annotate_conversation(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True,
        parallel: bool = True,
    ) -> AnnotationResult:
        conversation_id = conversation.get("id", "unknown")
        turns = conversation.get("turns", [])

        annotated_turns: List[TurnAnnotation] = []
        all_patient_eos: List[Dict] = []
        all_clinician_responses: List[Dict] = []

        for i, turn in enumerate(turns):
            context = self._build_context(turns[:i])
            turn_ann = self._process_turn(turn, context, parallel)
            annotated_turns.append(turn_ann)

            for span in turn_ann.spans:
                s_dict = {
                    "turn_id": turn_ann.turn_id,
                    "span_id": span.span_id,
                    "text": span.text,
                    "label": span.label,
                }
                if turn_ann.speaker == Speaker.PATIENT:
                    all_patient_eos.append(s_dict)
                else:
                    all_clinician_responses.append(s_dict)

        if include_relations and all_patient_eos and all_clinician_responses:
            relations = self.relation_linker.link(all_patient_eos, all_clinician_responses, turns)
            self._attach_relations(annotated_turns, relations)

        return AnnotationResult(
            conversation_id=conversation_id,
            turns=annotated_turns,
            agent_type="multi_agent",
            metadata={
                "model": self.config.model,
                "provider": self.config.provider,
                "agents": ["coordinator", "eo_detector", "response_classifier", "spikes_tagger", "critic_consensus", "relation_linker"],
            },
        )

    def _attach_relations(self, turns: List[TurnAnnotation], relations: List[RelationAnnotation]):
        span_to_turn = {s.span_id: turn for turn in turns for s in turn.spans}
        for rel in relations:
            if rel.from_span_id in span_to_turn:
                span_to_turn[rel.from_span_id].relations.append(rel)

    def _build_context(self, previous_turns: List[Dict], max_turns: int = 5) -> str:
        if not previous_turns:
            return "No previous context."
        return "\n".join([f"{t.get('speaker', '').upper()}: {t.get('text', '')[:180]}" for t in previous_turns[-max_turns:]])


def create_multi_agent_system(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> MultiAgentSystem:
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"
    return MultiAgentSystem(AgentConfig(provider=provider, model=model, api_key=api_key, **kwargs))
