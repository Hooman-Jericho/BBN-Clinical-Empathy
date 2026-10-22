"""Multi-agent annotation system: specialists, optional coordinator, optional critic.

Changes: all components share one :class:`LLMClient` (single retry/usage implementation), the
critic can reject *every* proposal, unusable critic output is flagged instead of silently
replaced, coordinator guidance is actually passed to the specialists (and the coordinator call is
optional - it used to cost one LLM call per turn for nothing), and every parse problem is counted.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import (
    SPIKES_STAGES,
    AgentConfig,
    AnnotationResult,
    RelationAnnotation,
    Speaker,
    TurnAnnotation,
)
from .jsonutil import extract_json_from_text
from .llm import LLMClient
from .prompts import (
    EO_DETECTOR_PROMPT,
    MULTI_AGENT_COORDINATOR_PROMPT,
    MULTI_AGENT_USER_PROMPT_TEMPLATE,
    RELATION_LINKER_PROMPT,
    RELATION_USER_PROMPT_TEMPLATE,
    RESPONSE_CLASSIFIER_PROMPT,
    SPIKES_TAGGER_PROMPT,
    SPIKES_USER_PROMPT_TEMPLATE,
)
from .schemas import CoordinatorPlanSchema
from .validation import (
    ParseStats,
    build_relations,
    build_spans,
    select_critic_spans,
)

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
    """Base class for specialist LLM agents (shares one LLMClient per system)."""

    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        config: AgentConfig,
        llm: Optional[LLMClient] = None,
    ):
        self.role = role
        self.system_prompt = system_prompt
        self.config = config
        self.llm = llm or LLMClient(config)

    def _call_llm(self, user_prompt: str, max_retries: int = 2) -> str:
        try:
            return self.llm.chat(self.system_prompt, user_prompt, max_retries=max_retries)
        except RuntimeError as exc:
            raise RuntimeError(f"[{self.role.value}] {exc}") from exc

    def _parse_json(self, response: str) -> Dict[str, Any]:
        return extract_json_from_text(response)

    def _annotations_from(self, resp: str, stats: Optional[ParseStats]) -> List[Dict[str, Any]]:
        """Raw span dicts from a specialist; label/speaker checks happen in ``build_spans``."""
        parsed = self._parse_json(resp)
        if not parsed:
            if stats is not None:
                stats.json_failures += 1
            return []
        raw = parsed.get("annotations", [])
        if not isinstance(raw, list):
            if stats is not None:
                stats.schema_failures += 1
            return []
        return [a for a in raw if isinstance(a, dict)]


class CoordinatorAgent(BaseLLMAgent):
    """Optional coordinator that produces guidance for the specialists."""

    def __init__(self, config: AgentConfig, llm: Optional[LLMClient] = None):
        super().__init__(AgentRole.COORDINATOR, MULTI_AGENT_COORDINATOR_PROMPT, config, llm)

    def plan_turn(
        self, turn: Dict[str, Any], context: str, stats: Optional[ParseStats] = None
    ) -> CoordinatorPlanSchema:
        speaker = turn.get("speaker", "patient").lower()
        prompt = (
            "Formulate an orchestration plan for this turn:\n"
            f"Speaker: {speaker}\nText: \"{turn.get('text', '')}\"\n"
            f"Context: {context if context else 'Initial turn.'}\n\n"
            "Return JSON adhering to schema:\n"
            f'{{"speaker_role": "{speaker}", "active_specialists": ["..."], '
            '"coordination_strategy": "..."}'
        )
        parsed = self._parse_json(self._call_llm(prompt))
        try:
            return CoordinatorPlanSchema.model_validate(parsed)
        except Exception:  # noqa: BLE001 - pydantic ValidationError or non-dict input
            if stats is not None:
                stats.schema_failures += 1
            logger.warning("Coordinator returned an unusable plan; using the default plan.")
            specialists = (["eo_detector"] if "patient" in speaker
                           else ["response_classifier", "spikes_tagger"])
            return CoordinatorPlanSchema(
                speaker_role=speaker,
                active_specialists=specialists,
                coordination_strategy="Standard clinical analysis dispatch.",
            )


class EODetectorAgent(BaseLLMAgent):
    """Specialist agent for detecting patient Empathic Opportunities (EOs)."""

    def __init__(self, config: AgentConfig, llm: Optional[LLMClient] = None):
        super().__init__(AgentRole.EO_DETECTOR, EO_DETECTOR_PROMPT, config, llm)

    def process(self, turn: Dict[str, Any], context: str,
                stats: Optional[ParseStats] = None) -> List[Dict[str, Any]]:
        prompt = MULTI_AGENT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0), speaker="patient",
            text=turn.get("text", ""), context=context)
        return self._annotations_from(self._call_llm(prompt), stats)


class ResponseClassifierAgent(BaseLLMAgent):
    """Specialist agent for classifying clinician responses and elicitations."""

    def __init__(self, config: AgentConfig, llm: Optional[LLMClient] = None):
        super().__init__(AgentRole.RESPONSE_CLASSIFIER, RESPONSE_CLASSIFIER_PROMPT, config, llm)

    def process(self, turn: Dict[str, Any], context: str,
                stats: Optional[ParseStats] = None) -> List[Dict[str, Any]]:
        prompt = MULTI_AGENT_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0), speaker="clinician",
            text=turn.get("text", ""), context=context)
        return self._annotations_from(self._call_llm(prompt), stats)


class SPIKESTaggerAgent(BaseLLMAgent):
    """Specialist agent for identifying SPIKES protocol stages."""

    def __init__(self, config: AgentConfig, llm: Optional[LLMClient] = None):
        super().__init__(AgentRole.SPIKES_TAGGER, SPIKES_TAGGER_PROMPT, config, llm)

    def process(self, turn: Dict[str, Any], context: str,
                stats: Optional[ParseStats] = None) -> Optional[str]:
        prompt = SPIKES_USER_PROMPT_TEMPLATE.format(
            turn_id=turn.get("turn_id", 0), text=turn.get("text", ""), context=context)
        parsed = self._parse_json(self._call_llm(prompt))
        if not parsed:
            if stats is not None:
                stats.json_failures += 1
            return None
        stage = parsed.get("spikes_stage")
        if stage is None:
            return None
        if str(stage).lower() in SPIKES_STAGES:
            return str(stage).lower()
        if stats is not None:
            stats.spikes_invalid += 1
        return None


class CriticConsensusAgent(BaseLLMAgent):
    """Cross-examines specialist proposals. It may reject all of them."""

    def __init__(self, config: AgentConfig, llm: Optional[LLMClient] = None):
        system_prompt = (
            "You are the Lead Clinical Consensus Critic in a multi-agent annotation system.\n"
            "Rigorously evaluate candidate span annotations proposed by specialist agents.\n"
            "1. Remove false positives that do not strictly meet the appraisal criteria.\n"
            "2. Resolve overlapping spans; boundaries must be exact semantic units.\n"
            "3. Confirm the label is consistent with the speaker's role.\n"
            "You may ONLY approve spans that were proposed. Returning an empty approved_spans "
            "list is correct when no proposal is valid.\n"
            'Output JSON: {"approved_spans": [...], "rejected_spans": [...], "critic_notes": "..."}'
        )
        super().__init__(AgentRole.CRITIC_CONSENSUS, system_prompt, config, llm)

    def evaluate_proposals(
        self,
        turn: Dict[str, Any],
        proposed_spans: List[Dict[str, Any]],
        context: str,
        stats: Optional[ParseStats] = None,
    ) -> List[Dict[str, Any]]:
        """Return the approved spans. Unusable critic output falls back to the proposals but is
        counted in ``stats.critic_fallbacks`` so it is visible in the diagnostics."""
        if not proposed_spans:
            return []
        stats = stats if stats is not None else ParseStats()

        prompt = (
            f"Turn {turn.get('turn_id', 0)} ({turn.get('speaker', '')}): \"{turn.get('text', '')}\"\n"
            f"Context: {context}\n"
            f"Specialist Proposed Spans: {json.dumps(proposed_spans, indent=2)}\n\n"
            'Return JSON: {"approved_spans": [...], "rejected_spans": [...], "critic_notes": "..."}'
        )
        parsed = self._parse_json(self._call_llm(prompt))
        approved = select_critic_spans(parsed, proposed_spans, stats)
        if approved is None:
            logger.warning("Critic output unusable for turn %s; keeping specialist proposals.",
                           turn.get("turn_id"))
            return proposed_spans
        return approved


class RelationLinkerAgent(BaseLLMAgent):
    """Specialist agent for linking clinician spans to patient EOs."""

    def __init__(self, config: AgentConfig, llm: Optional[LLMClient] = None):
        super().__init__(AgentRole.RELATION_LINKER, RELATION_LINKER_PROMPT, config, llm)

    def link(
        self,
        patient_eos: List[Dict],
        clinician_responses: List[Dict],
        turns: List[Dict],
        stats: Optional[ParseStats] = None,
    ) -> List[RelationAnnotation]:
        if not patient_eos or not clinician_responses:
            return []
        stats = stats if stats is not None else ParseStats()
        summary = "\n".join(f"T{t.get('turn_id')}: {str(t.get('text', ''))[:100]}" for t in turns)
        prompt = RELATION_USER_PROMPT_TEMPLATE.format(
            patient_eos=json.dumps(patient_eos, indent=2),
            clinician_responses=json.dumps(clinician_responses, indent=2),
            conversation=summary)
        parsed = self._parse_json(self._call_llm(prompt))
        return build_relations(
            parsed,
            {e["span_id"] for e in patient_eos},
            {c["span_id"] for c in clinician_responses},
            stats,
        )


class MultiAgentSystem:
    """Specialists + (optional) coordinator + (optional) critic.

    ``use_critic`` / ``use_coordinator`` exist so that each component can be ablated and its
    cost/benefit measured. The coordinator is OFF by default: its plan used to be discarded.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        use_critic: bool = True,
        use_coordinator: bool = False,
    ):
        self.config = config or AgentConfig()
        self.use_critic = use_critic
        self.use_coordinator = use_coordinator
        self.llm = LLMClient(self.config)  # one client => one usage counter for the whole system
        self.coordinator = CoordinatorAgent(self.config, self.llm)
        self.eo_detector = EODetectorAgent(self.config, self.llm)
        self.response_classifier = ResponseClassifierAgent(self.config, self.llm)
        self.spikes_tagger = SPIKESTaggerAgent(self.config, self.llm)
        self.critic = CriticConsensusAgent(self.config, self.llm)
        self.relation_linker = RelationLinkerAgent(self.config, self.llm)

    def usage(self) -> Dict[str, int]:
        return self.llm.usage()

    def _process_turn(
        self,
        turn: Dict[str, Any],
        context: str,
        parallel: bool = True,
        stats: Optional[ParseStats] = None,
    ) -> TurnAnnotation:
        turn_id = turn.get("turn_id", 0)
        speaker_str = turn.get("speaker", "patient").lower()
        speaker = Speaker.CLINICIAN if "clinician" in speaker_str else Speaker.PATIENT
        text = turn.get("text", "")
        turn_stats = ParseStats()
        notes: List[str] = []

        spec_context = context
        if self.use_coordinator:
            plan = self.coordinator.plan_turn(turn, context, turn_stats)
            spec_context = f"{context}\nCoordinator guidance: {plan.coordination_strategy}"
            notes.append(f"Coordinator Plan: {plan.coordination_strategy}")

        spikes_stage: Optional[str] = None
        if speaker == Speaker.PATIENT:
            proposals = self.eo_detector.process(turn, spec_context, turn_stats)
        else:
            s_resp, s_spikes = ParseStats(), ParseStats()  # per-thread counters, merged below
            if parallel:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    f_resp = executor.submit(self.response_classifier.process, turn, spec_context, s_resp)
                    f_spikes = executor.submit(self.spikes_tagger.process, turn, spec_context, s_spikes)
                    proposals = f_resp.result()
                    spikes_stage = f_spikes.result()
            else:
                proposals = self.response_classifier.process(turn, spec_context, s_resp)
                spikes_stage = self.spikes_tagger.process(turn, spec_context, s_spikes)
            turn_stats.merge(s_resp).merge(s_spikes)

        if self.use_critic:
            consensus = self.critic.evaluate_proposals(turn, proposals, spec_context, turn_stats)
        else:
            consensus = proposals

        spans = build_spans(consensus, turn_id, text, speaker, turn_stats)

        status = "parse_failed" if turn_stats.json_failures and not proposals else "ok"
        if status != "ok" or turn_stats.problems:
            logger.warning("Turn %s: status=%s problems=%s", turn_id, status, turn_stats.nonzero())
        if stats is not None:
            stats.merge(turn_stats)

        return TurnAnnotation(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            spikes_stage=spikes_stage,
            spans=spans,
            relations=[],
            reasoning_steps=notes,
            diagnostics={"status": status, "stats": turn_stats.nonzero()},
        )

    def annotate_conversation(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True,
        parallel: bool = True,
    ) -> AnnotationResult:
        conversation_id = conversation.get("id", "unknown")
        turns = conversation.get("turns", [])
        conv_stats = ParseStats()

        annotated_turns: List[TurnAnnotation] = []
        patient_eos: List[Dict] = []
        clinician_spans: List[Dict] = []

        for i, turn in enumerate(turns):
            context = self._build_context(turns[:i])
            turn_ann = self._process_turn(turn, context, parallel, conv_stats)
            annotated_turns.append(turn_ann)
            for span in turn_ann.spans:
                item = {"turn_id": turn_ann.turn_id, "span_id": span.span_id,
                        "text": span.text, "label": span.label}
                (patient_eos if turn_ann.speaker == Speaker.PATIENT else clinician_spans).append(item)

        if include_relations and patient_eos and clinician_spans:
            relations = self.relation_linker.link(patient_eos, clinician_spans, turns, conv_stats)
            self._attach_relations(annotated_turns, relations)

        not_ok = sum(1 for t in annotated_turns if t.diagnostics.get("status") != "ok")
        agents = ["eo_detector", "response_classifier", "spikes_tagger", "relation_linker"]
        if self.use_critic:
            agents.append("critic_consensus")
        if self.use_coordinator:
            agents.insert(0, "coordinator")
        return AnnotationResult(
            conversation_id=conversation_id,
            turns=annotated_turns,
            agent_type="multi_agent",
            metadata={
                "model": self.config.model,
                "provider": self.config.provider,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "agents": agents,
                "diagnostics": {
                    "turns_total": len(annotated_turns),
                    "turns_not_ok": not_ok,
                    "parse": conv_stats.nonzero(),
                    "usage": self.llm.usage(),
                },
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
        return "\n".join(
            f"{t.get('speaker', '').upper()}: {t.get('text', '')[:180]}"
            for t in previous_turns[-max_turns:]
        )


def create_multi_agent_system(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    use_critic: bool = True,
    use_coordinator: bool = False,
    **kwargs,
) -> MultiAgentSystem:
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"
    return MultiAgentSystem(
        AgentConfig(provider=provider, model=model, api_key=api_key, **kwargs),
        use_critic=use_critic,
        use_coordinator=use_coordinator,
    )
