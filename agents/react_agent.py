"""ReAct annotation agent: iterative tool use with strict, observable output validation."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import (
    ALL_LABELS,
    SPIKES_STAGES,
    AgentConfig,
    AnnotationResult,
    RelationAnnotation,
    Speaker,
    TurnAnnotation,
    get_labels_for_speaker,
)
from .jsonutil import extract_json_from_text, resolve_span_coordinates  # noqa: F401 (re-exported)
from .llm import LLMClient  # noqa: F401 (re-exported for backward compatibility)
from .validation import ParseStats, build_relations, build_spans, parse_turn_payload

logger = logging.getLogger(__name__)

TOOL_NAMES = ("schema_lookup", "text_search", "validate_span")


@dataclass
class ReActStep:
    """A ReAct execution step capturing Thought, Action, and Environment Observation."""
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


def build_react_system_prompt(speaker: Speaker) -> str:
    """System prompt that lists the *allowed* labels for this speaker (the old one did not)."""
    labels = get_labels_for_speaker(speaker)
    label_lines = "\n".join(f"- {name}: {desc}" for name, desc in labels.items())
    stage_block = ""
    if speaker == Speaker.CLINICIAN:
        stage_lines = "\n".join(f"- {name}: {desc}" for name, desc in SPIKES_STAGES.items())
        stage_block = f"SPIKES stages (use one for spikes_stage, or null):\n{stage_lines}\n\n"
    return (
        "You are an expert clinical empathy annotation agent using the ReAct "
        "(Reason, Act, Observe) pattern.\n"
        f"You annotate one {speaker.value.upper()} turn of a breaking-bad-news conversation.\n\n"
        f"Allowed span labels for a {speaker.value} turn - use EXACTLY these strings:\n"
        f"{label_lines}\n\n"
        f"{stage_block}"
        "Rules:\n"
        "- 'text' must be an exact, contiguous substring of the turn text.\n"
        "- Annotate only spans that clearly satisfy a label definition. "
        "An empty list is a valid answer.\n"
        "- Never invent labels. Use text_search to obtain character offsets.\n\n"
        "Tools:\n"
        "1. schema_lookup(category): look up label definitions.\n"
        "2. text_search(phrase): exact character indices of a phrase in the turn.\n"
        "3. validate_span(text, start, end, label): check boundaries and label.\n"
        "4. finish(spikes_stage, annotations): finalise.\n\n"
        "At every step output exactly ONE JSON object:\n"
        '{"thought": "...", "action": "<tool>", "action_input": {...}}\n'
        'Final step: {"thought": "...", "action": "finish", "action_input": '
        '{"spikes_stage": null, "annotations": [{"text": "...", "start": 0, "end": 0, '
        '"label": "...", "reasoning": "..."}]}}'
    )


class ClinicalToolEnvironment:
    """Environment providing callable clinical appraisal tools to the ReAct agent."""

    def __init__(self, text: str, speaker: Speaker):
        self.text = text
        self.speaker = speaker

    def schema_lookup(self, category: str) -> str:
        """Query clinical definitions for a given label category."""
        category_clean = category.strip().lower()
        valid_labels = get_labels_for_speaker(self.speaker)

        matches = {k: v for k, v in valid_labels.items()
                   if category_clean in k or category_clean in v.lower()}
        if matches:
            lines = [f"- **{k}**: {v}" for k, v in matches.items()]
            return "Found matching labels in Appraisal Framework:\n" + "\n".join(lines)

        all_matches = {k: v for k, v in ALL_LABELS.items()
                       if category_clean in k or category_clean in v.lower()}
        if all_matches:
            lines = [f"- **{k}** (Note: strictly for opposite speaker): {v}"
                     for k, v in all_matches.items()]
            return ("Labels found in framework, but check speaker role applicability:\n"
                    + "\n".join(lines))

        return (f"Category '{category}' not found in Clinical Appraisal Schema. Available "
                "categories: Feelings, Appreciation, Judgement, Acceptance, Sharing, Understanding.")

    def text_search(self, phrase: str) -> str:
        """Search exact occurrences, boundary indices, and surrounding context in the turn text."""
        phrase_clean = phrase.strip()
        if not phrase_clean:
            return "Error: Empty query phrase."

        occurrences = []
        start = 0
        while True:
            idx = self.text.find(phrase_clean, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(phrase_clean)))
            start = idx + 1

        if not occurrences:
            return f"Phrase '{phrase_clean}' NOT found in turn text."

        results = []
        for i, (s, e) in enumerate(occurrences):
            before = self.text[max(0, s - 15):s]
            after = self.text[e:min(len(self.text), e + 15)]
            results.append(
                f"Occurrence {i + 1}: index [{s}:{e}], context: '...{before}[{phrase_clean}]{after}...'"
            )

        return f"Found {len(occurrences)} occurrence(s):\n" + "\n".join(results)

    def validate_span(self, text_slice: str, start: int, end: int, label: str) -> str:
        """Validate candidate span coordinates and clinical category validity."""
        if start < 0 or end > len(self.text) or start >= end:
            return f"Validation FAILED: Invalid range [{start}:{end}] for text length {len(self.text)}."

        actual_slice = self.text[start:end]
        if actual_slice != text_slice:
            return (f"Validation MISMATCH: text[{start}:{end}] is '{actual_slice}', "
                    f"expected '{text_slice}'.")

        valid_labels = get_labels_for_speaker(self.speaker)
        if label not in valid_labels:
            return (f"Validation WARNING: Label '{label}' is not standard for {self.speaker.value}. "
                    f"Valid: {list(valid_labels.keys())}")

        return (f"Validation SUCCESS: Span '{actual_slice}' [{start}:{end}] perfectly matches "
                f"'{label}' for {self.speaker.value}.")

    def execute_tool(self, action: str, action_input: Dict[str, Any]) -> str:
        """Execute a tool action and return environment observation."""
        try:
            if action == "schema_lookup":
                return self.schema_lookup(action_input.get("category", ""))
            if action == "text_search":
                return self.text_search(action_input.get("phrase", ""))
            if action == "validate_span":
                return self.validate_span(
                    text_slice=action_input.get("text", ""),
                    start=int(action_input.get("start", 0)),
                    end=int(action_input.get("end", 0)),
                    label=action_input.get("label", ""),
                )
            return f"Unknown tool action: {action}"
        except Exception as exc:  # noqa: BLE001 - returned to the model as an observation
            return f"Tool execution error: {exc}"


class ReActAnnotationAgent:
    """ReAct agent with a bounded tool-calling loop and per-span output validation."""

    def __init__(self, config: Optional[AgentConfig] = None, max_steps: int = 4):
        self.config = config or AgentConfig()
        self.max_steps = max_steps
        self.llm = LLMClient(self.config)

    # ------------------------------------------------------------------ one turn
    def annotate_turn(
        self,
        turn: Dict[str, Any],
        context: str = "",
        stats: Optional[ParseStats] = None,
    ) -> TurnAnnotation:
        """Run the ReAct loop on one turn. Never fails silently: see ``diagnostics``."""
        turn_id = turn.get("turn_id", 0)
        speaker_str = turn.get("speaker", "patient").lower()
        speaker = Speaker.CLINICIAN if "clinician" in speaker_str else Speaker.PATIENT
        text = turn.get("text", "")

        env = ClinicalToolEnvironment(text, speaker)
        reasoning_steps: List[ReActStep] = []
        turn_stats = ParseStats()
        system_prompt = build_react_system_prompt(speaker)

        history = [
            f"Turn {turn_id} ({speaker.value}): \"{text}\"\n"
            f"Context: {context if context else 'No previous context.'}"
        ]

        final_parsed: Optional[Dict[str, Any]] = None

        for step_idx in range(1, self.max_steps + 1):
            user_prompt = ("\n\n".join(history)
                           + f"\n\nStep {step_idx}: Output your next ReAct Thought and Action JSON.")
            response = self.llm.chat(system_prompt, user_prompt)
            parsed_action = extract_json_from_text(response)

            if not parsed_action:
                # Previously this silently became "finish with nothing".
                turn_stats.json_failures += 1
                reasoning_steps.append(ReActStep(
                    step_idx, "", "invalid_output", {}, "Model output was not valid JSON."))
                history.append(
                    f"Observation {step_idx}: Your previous output was not valid JSON. "
                    "Reply with exactly one JSON object."
                )
                continue

            action = parsed_action.get("action")
            thought = str(parsed_action.get("thought", "") or "")
            action_input = parsed_action.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            if action == "finish" or "annotations" in action_input or "annotations" in parsed_action:
                reasoning_steps.append(ReActStep(
                    step_idx, thought, "finish", action_input, "Finalizing annotations."))
                final_parsed = action_input if "annotations" in action_input else parsed_action
                break

            if action not in TOOL_NAMES:
                turn_stats.schema_failures += 1
                obs = (f"Unknown or missing action {action!r}. "
                       f"Valid actions: {', '.join(TOOL_NAMES)}, finish.")
            else:
                obs = env.execute_tool(action, action_input)

            reasoning_steps.append(ReActStep(
                step_idx, thought, str(action), action_input, obs))
            history.append(
                f"Thought {step_idx}: {thought}\n"
                f"Action {step_idx}: {action}({json.dumps(action_input)})\n"
                f"Observation {step_idx}: {obs}"
            )

        if final_parsed is None:
            final_prompt = (
                "Based on your investigation:\n" + "\n".join(history[-3:])
                + '\nOutput the final annotations in exact JSON format: '
                  '{"spikes_stage": null, "annotations": [...]}'
            )
            final_parsed = extract_json_from_text(self.llm.chat(system_prompt, final_prompt))

        spikes_stage, raw_spans, status = parse_turn_payload(final_parsed, speaker, turn_stats)
        spans = build_spans(raw_spans, turn_id, text, speaker, turn_stats)

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
            reasoning_steps=reasoning_steps,
            diagnostics={"status": status, "stats": turn_stats.nonzero()},
        )

    # ------------------------------------------------------------------ conversation
    def annotate_conversation(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True,
    ) -> AnnotationResult:
        """Annotate an entire conversation sequentially."""
        conversation_id = conversation.get("id", "unknown")
        turns = conversation.get("turns", [])
        conv_stats = ParseStats()

        annotated_turns: List[TurnAnnotation] = []
        patient_eos: List[Dict] = []
        clinician_spans: List[Dict] = []

        for i, turn in enumerate(turns):
            context = self._build_context(turns[:i])
            turn_ann = self.annotate_turn(turn, context, stats=conv_stats)
            annotated_turns.append(turn_ann)

            for span in turn_ann.spans:
                item = {"turn_id": turn_ann.turn_id, "span_id": span.span_id,
                        "text": span.text, "label": span.label}
                (patient_eos if turn_ann.speaker == Speaker.PATIENT else clinician_spans).append(item)

        if include_relations and patient_eos and clinician_spans:
            relations = self._identify_relations(patient_eos, clinician_spans, turns, conv_stats)
            self._attach_relations(annotated_turns, relations)

        failed = sum(1 for t in annotated_turns if t.diagnostics.get("status") != "ok")
        return AnnotationResult(
            conversation_id=conversation_id,
            turns=annotated_turns,
            agent_type="react",
            metadata={
                "model": self.config.model,
                "provider": self.config.provider,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "max_steps": self.max_steps,
                "diagnostics": {
                    "turns_total": len(annotated_turns),
                    "turns_not_ok": failed,
                    "parse": conv_stats.nonzero(),
                    "usage": self.llm.usage(),
                },
            },
        )

    def _identify_relations(
        self,
        patient_eos: List[Dict],
        clinician_responses: List[Dict],
        turns: List[Dict],
        stats: Optional[ParseStats] = None,
    ) -> List[RelationAnnotation]:
        """Identify cross-turn relations; endpoints are validated against known span ids."""
        stats = stats if stats is not None else ParseStats()
        summary = "\n".join(
            f"Turn {t.get('turn_id', 0)} ({t.get('speaker', '')}): {t.get('text', '')[:100]}"
            for t in turns
        )
        prompt = (
            "Link clinician responses/elicitations to patient EOs.\n"
            f"Patient EOs: {json.dumps(patient_eos, indent=2)}\n"
            f"Clinician Responses: {json.dumps(clinician_responses, indent=2)}\n"
            f"Dialogue Flow:\n{summary}\n\n"
            'Return JSON: {"relations": [{"from_span_id": "...", "to_span_id": "...", '
            '"relation_type": "response_to" | "elicitation_of", "reasoning": "..."}]}'
        )
        resp = self.llm.chat("You are an expert relation linker for clinical dialogue.", prompt)
        return build_relations(
            extract_json_from_text(resp),
            {e["span_id"] for e in patient_eos},
            {c["span_id"] for c in clinician_responses},
            stats,
        )

    def _attach_relations(self, turns: List[TurnAnnotation], relations: List[RelationAnnotation]):
        span_to_turn = {s.span_id: turn for turn in turns for s in turn.spans}
        for rel in relations:
            if rel.from_span_id in span_to_turn:
                span_to_turn[rel.from_span_id].relations.append(rel)

    def _build_context(self, previous_turns: List[Dict], max_turns: int = 5) -> str:
        if not previous_turns:
            return ""
        return "\n".join(
            f"{t.get('speaker', '').upper()}: {t.get('text', '')[:180]}"
            for t in previous_turns[-max_turns:]
        )

    def usage(self) -> Dict[str, int]:
        return self.llm.usage()


def create_react_agent(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> ReActAnnotationAgent:
    """Create configured ReAct Agent."""
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"
    return ReActAnnotationAgent(AgentConfig(provider=provider, model=model, api_key=api_key, **kwargs))
