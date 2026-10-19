"""ReAct Annotation Agent with genuine iterative tool execution, observation loop, and Pydantic validation."""

import json
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

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
    PATIENT_LABELS,
    CLINICIAN_LABELS,
)
from .schemas import (
    SpanPredictionSchema,
    TurnAnnotationSchema,
    ReActToolCallSchema,
    RelationPredictionSchema,
    RelationListSchema,
)

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """A genuine ReAct execution step capturing Thought, Action, and Environment Observation."""
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


def resolve_span_coordinates(text: str, span_text: str, suggested_start: int, suggested_end: int) -> tuple[int, int]:
    """Resolve exact span coordinates in text without first-occurrence bias."""
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

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    codeblock_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response, re.IGNORECASE)
    if codeblock_match:
        try:
            return json.loads(codeblock_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start_idx = response.find('{')
    end_idx = response.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = response[start_idx:end_idx + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r',\s*([\]}])', r'\1', json_candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to extract JSON from LLM response: %s", response[:200])
    return {}


class ClinicalToolEnvironment:
    """Environment providing callable clinical appraisal tools to the ReAct agent."""

    def __init__(self, text: str, speaker: Speaker):
        self.text = text
        self.speaker = speaker

    def schema_lookup(self, category: str) -> str:
        """Query clinical definitions, boundaries, and examples for a given label category."""
        category_clean = category.strip().lower()
        valid_labels = get_labels_for_speaker(self.speaker)

        matches = {k: v for k, v in valid_labels.items() if category_clean in k or category_clean in v.lower()}
        if matches:
            lines = [f"- **{k}**: {v}" for k, v in matches.items()]
            return "Found matching labels in Appraisal Framework:\n" + "\n".join(lines)

        all_matches = {k: v for k, v in ALL_LABELS.items() if category_clean in k or category_clean in v.lower()}
        if all_matches:
            lines = [f"- **{k}** (Note: strictly for opposite speaker): {v}" for k, v in all_matches.items()]
            return "Labels found in framework, but check speaker role applicability:\n" + "\n".join(lines)

        return f"Category '{category}' not found in Clinical Appraisal Schema. Available categories: Feelings, Appreciation, Judgement, Acceptance, Sharing, Understanding."

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
            results.append(f"Occurrence {i + 1}: index [{s}:{e}], context: '...{before}[{phrase_clean}]{after}...'")

        return f"Found {len(occurrences)} occurrence(s):\n" + "\n".join(results)

    def validate_span(self, text_slice: str, start: int, end: int, label: str) -> str:
        """Validate candidate span coordinates and clinical category validity."""
        if start < 0 or end > len(self.text) or start >= end:
            return f"Validation FAILED: Invalid range [{start}:{end}] for text length {len(self.text)}."

        actual_slice = self.text[start:end]
        if actual_slice != text_slice:
            return f"Validation MISMATCH: text[{start}:{end}] is '{actual_slice}', expected '{text_slice}'."

        valid_labels = get_labels_for_speaker(self.speaker)
        if label not in valid_labels:
            return f"Validation WARNING: Label '{label}' is not standard for {self.speaker.value}. Valid: {list(valid_labels.keys())[:5]}..."

        return f"Validation SUCCESS: Span '{actual_slice}' [{start}:{end}] perfectly matches '{label}' for {self.speaker.value}."

    def execute_tool(self, action: str, action_input: Dict[str, Any]) -> str:
        """Execute a tool action and return environment observation."""
        try:
            if action == "schema_lookup":
                category = action_input.get("category", "")
                return self.schema_lookup(category)
            elif action == "text_search":
                phrase = action_input.get("phrase", "")
                return self.text_search(phrase)
            elif action == "validate_span":
                return self.validate_span(
                    text_slice=action_input.get("text", ""),
                    start=int(action_input.get("start", 0)),
                    end=int(action_input.get("end", 0)),
                    label=action_input.get("label", ""),
                )
            else:
                return f"Unknown tool action: {action}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"


class LLMClient:
    """Unified wrapper for LLM API calls."""

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
        """Send chat request."""
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
    """Genuine ReAct Agent with iterative tool-calling loop, observation feedback, and Pydantic validation."""

    def __init__(self, config: Optional[AgentConfig] = None, max_steps: int = 4):
        self.config = config or AgentConfig()
        self.max_steps = max_steps
        self.llm = LLMClient(self.config)

    def annotate_turn(self, turn: Dict[str, Any], context: str = "") -> TurnAnnotation:
        """Execute full ReAct loop on a turn."""
        turn_id = turn.get("turn_id", 0)
        speaker_str = turn.get("speaker", "patient").lower()
        speaker = Speaker.CLINICIAN if "clinician" in speaker_str else Speaker.PATIENT
        text = turn.get("text", "")

        env = ClinicalToolEnvironment(text, speaker)
        reasoning_steps: List[ReActStep] = []

        system_prompt = (
            "You are an expert clinical empathy annotation agent using the ReAct (Reasoning + Acting) framework.\n"
            "You have access to the following real tools:\n"
            "1. schema_lookup(category): Look up clinical criteria and examples.\n"
            "2. text_search(phrase): Search exact character indices and occurrences in the turn.\n"
            "3. validate_span(text, start, end, label): Validate candidate boundary and label compatibility.\n"
            "4. finish(annotations, spikes_stage): Finalize annotations.\n\n"
            "At each step, output a single JSON object with:\n"
            '{"thought": "...", "action": "tool_name", "action_input": {"arg": "val"}}\n'
            "When ready to finalize, call 'finish' with the list of annotations."
        )

        history = [
            f"Turn {turn_id} ({speaker.value}): \"{text}\"\n"
            f"Context: {context if context else 'No previous context.'}"
        ]

        final_parsed = None

        # Execute ReAct Tool-Calling Loop
        for step_idx in range(1, self.max_steps + 1):
            user_prompt = "\n\n".join(history) + f"\n\nStep {step_idx}: Output your next ReAct Thought and Action JSON."
            response = self.llm.chat(system_prompt, user_prompt)
            parsed_action = extract_json_from_text(response)

            action = parsed_action.get("action", "finish")
            thought = parsed_action.get("thought", "Analyzing turn.")
            action_input = parsed_action.get("action_input", {})

            # Check if finish is called or if max steps reached
            if action == "finish" or "annotations" in action_input or "annotations" in parsed_action:
                obs = "Finalizing annotations."
                reasoning_steps.append(ReActStep(
                    step_number=step_idx,
                    thought=thought,
                    action="finish",
                    action_input=action_input,
                    observation=obs,
                ))
                final_parsed = action_input if "annotations" in action_input else parsed_action
                break

            # Execute tool in environment
            obs = env.execute_tool(action, action_input)
            reasoning_steps.append(ReActStep(
                step_number=step_idx,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=obs,
            ))

            # Feed observation back into conversation history
            history.append(
                f"Thought {step_idx}: {thought}\n"
                f"Action {step_idx}: {action}({json.dumps(action_input)})\n"
                f"Observation {step_idx}: {obs}"
            )

        # Fallback if loop completed without explicit finish
        if not final_parsed:
            final_prompt = (
                f"Based on your investigation:\n" + "\n".join(history[-3:]) +
                f'\nOutput the final annotations in exact JSON format: {{"spikes_stage": null, "annotations": [...]}}'
            )
            final_resp = self.llm.chat(system_prompt, final_prompt)
            final_parsed = extract_json_from_text(final_resp)

        # Validate with Pydantic Schema
        try:
            validated = TurnAnnotationSchema.model_validate(final_parsed)
            raw_spans = validated.annotations
            spikes_stage = validated.spikes_stage
        except Exception:
            raw_spans = []
            spikes_stage = final_parsed.get("spikes_stage")

        spans: List[SpanAnnotation] = []
        for i, s_model in enumerate(raw_spans):
            span_text = s_model.text.strip()
            if not span_text or span_text not in text:
                continue

            start, end = resolve_span_coordinates(text, span_text, s_model.start, s_model.end)
            spans.append(SpanAnnotation(
                span_id=generate_span_id(turn_id, i),
                text=text[start:end] if 0 <= start < end <= len(text) else span_text,
                start=start,
                end=end,
                label=s_model.label,
                reasoning=s_model.reasoning or "",
                confidence=s_model.confidence,
            ))

        if speaker != Speaker.CLINICIAN:
            spikes_stage = None
        elif spikes_stage and str(spikes_stage).lower() not in SPIKES_STAGES:
            spikes_stage = None

        return TurnAnnotation(
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            spikes_stage=spikes_stage,
            spans=spans,
            relations=[],
            reasoning_steps=reasoning_steps,
        )

    def annotate_conversation(
        self,
        conversation: Dict[str, Any],
        include_relations: bool = True
    ) -> AnnotationResult:
        """Annotate entire conversation sequentially with ReAct."""
        conversation_id = conversation.get("id", "unknown")
        turns = conversation.get("turns", [])

        annotated_turns: List[TurnAnnotation] = []
        all_patient_eos: List[Dict] = []
        all_clinician_responses: List[Dict] = []

        for i, turn in enumerate(turns):
            context = self._build_context(turns[:i])
            turn_ann = self.annotate_turn(turn, context)
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
            relations = self._identify_relations(all_patient_eos, all_clinician_responses, turns)
            self._attach_relations(annotated_turns, relations)

        return AnnotationResult(
            conversation_id=conversation_id,
            turns=annotated_turns,
            agent_type="react",
            metadata={
                "model": self.config.model,
                "provider": self.config.provider,
                "framework": "True ReAct with Tool Environment & Pydantic Validation",
            }
        )

    def _identify_relations(
        self,
        patient_eos: List[Dict],
        clinician_responses: List[Dict],
        turns: List[Dict]
    ) -> List[RelationAnnotation]:
        """Identify cross-turn relations validated with Pydantic."""
        summary = "\n".join([f"Turn {t.get('turn_id', 0)} ({t.get('speaker', '')}): {t.get('text', '')[:100]}" for t in turns])
        prompt = (
            f"Link clinician responses/elicitations to patient EOs.\n"
            f"Patient EOs: {json.dumps(patient_eos, indent=2)}\n"
            f"Clinician Responses: {json.dumps(clinician_responses, indent=2)}\n"
            f"Dialogue Flow:\n{summary}\n\n"
            f"Return JSON: {{\"relations\": [{{\"from_span_id\": \"...\", \"to_span_id\": \"...\", \"relation_type\": \"response_to\", \"reasoning\": \"...\"}}]}}"
        )
        resp = self.llm.chat("You are an expert relation linker for clinical dialogue.", prompt)
        parsed = extract_json_from_text(resp)

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

    def _attach_relations(self, turns: List[TurnAnnotation], relations: List[RelationAnnotation]):
        span_to_turn = {s.span_id: turn for turn in turns for s in turn.spans}
        for rel in relations:
            if rel.from_span_id in span_to_turn:
                span_to_turn[rel.from_span_id].relations.append(rel)

    def _build_context(self, previous_turns: List[Dict], max_turns: int = 5) -> str:
        if not previous_turns:
            return ""
        return "\n".join([f"{t.get('speaker', '').upper()}: {t.get('text', '')[:180]}" for t in previous_turns[-max_turns:]])


def create_react_agent(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ReActAnnotationAgent:
    """Create configured ReAct Agent."""
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"
    return ReActAnnotationAgent(AgentConfig(provider=provider, model=model, api_key=api_key, **kwargs))
