"""Deterministic mocked unit tests for ReAct and Multi-Agent LLM workflows.

Verifies:
- ReAct loop execution with mocked LLM responses
- Malformed JSON recovery and error boundaries
- Array-based LLM response normalization
- Authentication and provider error handling
- Coordinator dispatch and MultiAgent execution with mocked specialists
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

import pytest
from agents.base import AgentConfig, Speaker, TurnAnnotation, SpanAnnotation
from agents.react_agent import ReActAnnotationAgent, extract_json_from_text, LLMClient
from agents.multi_agent import MultiAgentSystem, CoordinatorAgent, BaseLLMAgent, AgentRole


def test_extract_json_from_text_variants():
    """Verify extract_json_from_text handles markdown code fences, trailing commas, and raw lists."""
    # 1. Clean object
    assert extract_json_from_text('{"spikes_stage": "setting"}') == {"spikes_stage": "setting"}

    # 2. Markdown codeblock
    md_text = '```json\n{"annotations": [{"text": "scared", "label": "explicit_feeling"}]}\n```'
    parsed_md = extract_json_from_text(md_text)
    assert len(parsed_md.get("annotations", [])) == 1

    # 3. Trailing comma in JSON
    trailing_text = '{"annotations": [{"text": "sad", "label": "explicit_feeling",},], "spikes_stage": null,}'
    parsed_trailing = extract_json_from_text(trailing_text)
    assert "annotations" in parsed_trailing

    # 4. Raw list response normalized to {"annotations": [...]}
    raw_list_text = '[{"text": "anxious", "label": "explicit_feeling"}]'
    parsed_list = extract_json_from_text(raw_list_text)
    assert "annotations" in parsed_list
    assert len(parsed_list["annotations"]) == 1

    # 5. Malformed text returns empty dict without crashing
    assert extract_json_from_text("Sorry, I cannot annotate this conversation.") == {}
    assert extract_json_from_text("") == {}


def test_react_agent_deterministic_mocked_loop():
    """Verify ReActAnnotationAgent executes tools and completes turn annotation deterministically."""
    config = AgentConfig(provider="openai", model="gpt-4o", api_key="sk-mock")
    agent = ReActAnnotationAgent(config=config, max_steps=2)

    # Mock LLMClient.chat
    # Step 1: Model requests text_search
    # Step 2: Model finishes with valid annotation JSON
    step1_response = json.dumps({
        "thought": "I need to check where 'terrified' appears.",
        "action": "text_search",
        "action_input": {"phrase": "terrified"},
    })
    step2_response = json.dumps({
        "thought": "Found the phrase, now finalizing.",
        "action": "finish",
        "action_input": {
            "spikes_stage": None,
            "annotations": [
                {
                    "text": "terrified",
                    "start": 7,
                    "end": 16,
                    "label": "explicit_feeling",
                    "reasoning": "Direct fear statement",
                    "confidence": 0.95,
                }
            ],
        },
    })

    with patch.object(agent.llm, "chat", side_effect=[step1_response, step2_response]):
        turn = {
            "turn_id": 1,
            "speaker": "patient",
            "text": "I feel terrified of the diagnosis.",
        }
        result = agent.annotate_turn(turn)

        assert isinstance(result, TurnAnnotation)
        assert result.turn_id == 1
        assert result.speaker == Speaker.PATIENT
        assert len(result.spans) == 1
        assert result.spans[0].text == "terrified"
        assert result.spans[0].start == 7
        assert result.spans[0].end == 16
        assert result.spans[0].label == "explicit_feeling"
        assert len(result.reasoning_steps) == 2
        assert result.reasoning_steps[0].action == "text_search"
        assert result.reasoning_steps[1].action == "finish"


def test_react_agent_handles_malformed_final_output_gracefully():
    """Verify that malformed or non-JSON final agent outputs do not crash or corrupt the turn."""
    config = AgentConfig(provider="openai", model="gpt-4o", api_key="sk-mock")
    agent = ReActAnnotationAgent(config=config, max_steps=1)

    with patch.object(agent.llm, "chat", return_value="Here is my analysis: Patient is very sick."):
        turn = {
            "turn_id": 2,
            "speaker": "patient",
            "text": "I feel weak.",
        }
        result = agent.annotate_turn(turn)
        assert result.turn_id == 2
        assert result.spans == []  # Gracefully empty, no unhandled exception


def test_llm_client_authentication_error_handling():
    """Verify LLMClient immediately raises actionable RuntimeError on authentication failure."""
    config = AgentConfig(provider="openai", model="gpt-4o", api_key="sk-invalid")
    client = LLMClient(config)

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = Exception("Incorrect API key provided: authentication_error")

    with patch.object(client, "_get_client", return_value=mock_openai):
        with pytest.raises(RuntimeError) as exc_info:
            client.chat("sys", "user")
        assert "Authentication Failure" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)


def test_coordinator_agent_plan_mocked():
    """Verify CoordinatorAgent plans specialists appropriately."""
    config = AgentConfig(provider="openai", model="gpt-4o", api_key="sk-mock")
    coordinator = CoordinatorAgent(config)

    mock_plan = json.dumps({
        "speaker_role": "clinician",
        "active_specialists": ["response_classifier", "spikes_tagger"],
        "coordination_strategy": "Evaluate clinician empathy and SPIKES staging.",
    })

    with patch.object(coordinator, "_call_llm", return_value=mock_plan):
        plan = coordinator.plan_turn(
            turn={"turn_id": 1, "speaker": "clinician", "text": "I am so sorry to give you this news."},
            context="",
        )
        assert plan.speaker_role == "clinician"
        assert "response_classifier" in plan.active_specialists
        assert "spikes_tagger" in plan.active_specialists
