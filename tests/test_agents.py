"""Unit tests for agent coordinate resolution, JSON extraction, Bipartite Span Matching, and ReAct tools."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

from agents.react_agent import (
    resolve_span_coordinates,
    extract_json_from_text,
    ClinicalToolEnvironment,
    ReActStep,
)
from agents.runner import AnnotationRunner
from agents.base import AnnotationResult, TurnAnnotation, SpanAnnotation, Speaker
from agents.schemas import (
    SpanPredictionSchema,
    TurnAnnotationSchema,
    CoordinatorPlanSchema,
    CriticReviewSchema,
)


def test_resolve_span_coordinates_disambiguates_multiple_occurrences():
    """Verify that closest occurrence to suggested_start is selected, not always the first."""
    text = "I know that you are anxious, but I also know you can overcome this."
    start, end = resolve_span_coordinates(
        text=text,
        span_text="know",
        suggested_start=38,
        suggested_end=42,
    )
    assert start == 40
    assert end == 44
    assert text[start:end] == "know"

    start1, end1 = resolve_span_coordinates(
        text=text,
        span_text="know",
        suggested_start=0,
        suggested_end=4,
    )
    assert start1 == 2
    assert end1 == 6


def test_clinical_tool_environment_react_execution():
    """Verify that ClinicalToolEnvironment provides authentic tool observations."""
    text = "I feel terrified about what might happen next, doctor."
    env = ClinicalToolEnvironment(text, Speaker.PATIENT)

    # 1. Test schema_lookup
    obs_schema = env.execute_tool("schema_lookup", {"category": "feeling"})
    assert "explicit_feeling" in obs_schema
    assert "implicit_feeling" in obs_schema

    # 2. Test text_search
    obs_search = env.execute_tool("text_search", {"phrase": "terrified"})
    assert "Found 1 occurrence" in obs_search
    assert "[7:16]" in obs_search

    # 3. Test validate_span
    obs_val = env.execute_tool("validate_span", {"text": "terrified", "start": 7, "end": 16, "label": "explicit_feeling"})
    assert "Validation SUCCESS" in obs_val


def test_react_step_instantiation_and_serialization():
    """Verify that ReActStep objects are properly instantiated and serialized."""
    step = ReActStep(
        step_number=1,
        thought="The patient expressed direct fear.",
        action="validate_span",
        action_input={"text": "scared", "start": 5, "end": 11, "label": "explicit_feeling"},
        observation="Validation SUCCESS: Span matches.",
    )
    d = step.to_dict()
    assert d["step"] == 1
    assert d["thought"] == "The patient expressed direct fear."
    assert d["action"] == "validate_span"
    assert "Validation SUCCESS" in d["observation"]


def test_pydantic_schemas_validation():
    """Verify that Pydantic v2 schemas properly validate agent and specialist outputs."""
    # Span schema
    span_data = {"text": "nervous", "start": 2, "end": 9, "label": "explicit_feeling", "confidence": 0.95}
    span_obj = SpanPredictionSchema.model_validate(span_data)
    assert span_obj.text == "nervous"
    assert span_obj.confidence == 0.95

    # Coordinator Plan Schema
    plan_data = {
        "speaker_role": "patient",
        "active_specialists": ["eo_detector"],
        "coordination_strategy": "Focus on subtle indirect feelings.",
    }
    plan_obj = CoordinatorPlanSchema.model_validate(plan_data)
    assert plan_obj.speaker_role == "patient"
    assert len(plan_obj.active_specialists) == 1

    # Critic Review Schema
    critic_data = {
        "approved_spans": [span_data],
        "rejected_spans": [],
        "critic_notes": "All boundaries verified against clinical appraisal criteria.",
    }
    critic_obj = CriticReviewSchema.model_validate(critic_data)
    assert len(critic_obj.approved_spans) == 1
    assert "verified" in critic_obj.critic_notes


def test_bipartite_matching_with_multiple_identical_labels(tmp_path):
    """CRITICAL TEST: Verify that 1-to-1 Bipartite Matching calculates 100% Precision (not 50%)
    when multiple spans in the same turn share the same label category."""
    runner = AnnotationRunner(agent_type="react", provider="openai")

    # Expert has 2 distinct spans with the same label 'explicit_feeling'
    expert_file = tmp_path / "expert_multi.json"
    expert_file.write_text("""{
      "id": "conv_test",
      "turns": [
        {
          "turn_id": 1,
          "annotations": {
            "spans": [
              {"span_id": "e1", "text": "scared", "start": 5, "end": 11, "label": "explicit_feeling"},
              {"span_id": "e2", "text": "hopeless", "start": 20, "end": 28, "label": "explicit_feeling"}
            ]
          }
        }
      ]
    }""", encoding="utf-8")

    # Agent also correctly predicts BOTH spans with label 'explicit_feeling'
    agent_turn = TurnAnnotation(
        turn_id=1,
        speaker=Speaker.PATIENT,
        text="I am scared and feel hopeless today.",
        spans=[
            SpanAnnotation(span_id="a1", text="scared", start=5, end=11, label="explicit_feeling"),
            SpanAnnotation(span_id="a2", text="hopeless", start=20, end=28, label="explicit_feeling"),
        ]
    )
    agent_result = AnnotationResult(
        conversation_id="conv_test",
        turns=[agent_turn],
        agent_type="react"
    )

    metrics = runner.compare_with_expert(agent_result, str(expert_file))

    # Bipartite matching must yield 2 true positives, 0 false positives, 0 false negatives
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0

    # In the old buggy code, precision was 50% due to set compression. Here it must be 100%!
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["avg_span_iou"] == 1.0


def test_bipartite_matching_avoids_cartesian_product_zero_pollution(tmp_path):
    """Verify that non-overlapping spans with the same label do not pollute average IoU with zeros."""
    runner = AnnotationRunner(agent_type="react", provider="openai")

    expert_file = tmp_path / "expert_iou.json"
    expert_file.write_text("""{
      "id": "c2",
      "turns": [
        {
          "turn_id": 1,
          "annotations": {
            "spans": [
              {"span_id": "e1", "text": "pain", "start": 5, "end": 9, "label": "explicit_feeling"},
              {"span_id": "e2", "text": "fear", "start": 30, "end": 34, "label": "explicit_feeling"}
            ]
          }
        }
      ]
    }""", encoding="utf-8")

    # Agent matches pain exactly, but misses fear
    agent_turn = TurnAnnotation(
        turn_id=1,
        speaker=Speaker.PATIENT,
        text="Some pain was there, followed by fear.",
        spans=[
            SpanAnnotation(span_id="a1", text="pain", start=5, end=9, label="explicit_feeling"),
        ]
    )
    agent_result = AnnotationResult(
        conversation_id="c2",
        turns=[agent_turn],
        agent_type="react"
    )

    metrics = runner.compare_with_expert(agent_result, str(expert_file))
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    # The matched pair has exact IoU = 1.0; Cartesian IoU zeros are not added
    assert metrics["avg_span_iou"] == 1.0
