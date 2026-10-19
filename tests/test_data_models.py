"""Regression tests for SpanAnnotation and core data model contracts.

Covers:
- Default construction of SpanAnnotation
- Explicit reasoning values
- to_dict() without optional values (confidence, reasoning defaults)
- Serialization round trips (to_dict -> from_dict)
- Backward compatibility with legacy database records and alternative key names
- JSON serialization of the resulting dictionary
- TurnAnnotation and AnnotationResult serialization round-trips
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

import pytest
from agents.base import SpanAnnotation, RelationAnnotation, TurnAnnotation, AnnotationResult, Speaker
from database.models import SpanAnnotation as DbSpanAnnotation


def test_span_annotation_default_construction():
    """Verify SpanAnnotation constructs with default reasoning and optional confidence."""
    span = SpanAnnotation(
        span_id="span_1",
        text="worried",
        start=10,
        end=17,
        label="explicit_feeling",
    )
    assert span.span_id == "span_1"
    assert span.text == "worried"
    assert span.start == 10
    assert span.end == 17
    assert span.label == "explicit_feeling"
    assert span.reasoning == ""
    assert span.confidence is None


def test_span_annotation_explicit_reasoning():
    """Verify SpanAnnotation accepts and preserves explicit reasoning and confidence."""
    span = SpanAnnotation(
        span_id="span_2",
        text="I am afraid",
        start=0,
        end=11,
        label="explicit_feeling",
        reasoning="Patient directly articulates emotional dread.",
        confidence=0.98,
    )
    assert span.reasoning == "Patient directly articulates emotional dread."
    assert span.confidence == 0.98


def test_span_annotation_to_dict_without_optional_values():
    """Verify to_dict() works reliably when confidence is None and reasoning is empty."""
    span = SpanAnnotation(
        span_id="span_3",
        text="fear",
        start=2,
        end=6,
        label="explicit_feeling",
    )
    data = span.to_dict()
    assert data["span_id"] == "span_3"
    assert data["text"] == "fear"
    assert data["start"] == 2
    assert data["end"] == 6
    assert data["label"] == "explicit_feeling"
    assert data["reasoning"] == ""
    assert "confidence" not in data


def test_span_annotation_serialization_round_trip():
    """Verify to_dict() -> from_dict() round trip preserves all fields."""
    original = SpanAnnotation(
        span_id="span_rt",
        text="hopeless",
        start=15,
        end=23,
        label="explicit_feeling",
        reasoning="Implicit appraisal of negative trajectory.",
        confidence=0.91,
    )
    d = original.to_dict()
    reconstructed = SpanAnnotation.from_dict(d)
    assert reconstructed.span_id == original.span_id
    assert reconstructed.text == original.text
    assert reconstructed.start == original.start
    assert reconstructed.end == original.end
    assert reconstructed.label == original.label
    assert reconstructed.reasoning == original.reasoning
    assert reconstructed.confidence == original.confidence


def test_span_annotation_backward_compatibility_with_legacy_records():
    """Verify from_dict handles database column names (start_pos, end_pos) gracefully."""
    legacy_data = {
        "span_id": "span_legacy",
        "text": "tired",
        "start_pos": 4,
        "end_pos": 9,
        "label": "implicit_feeling",
        "source": "manual",
    }
    span = SpanAnnotation.from_dict(legacy_data)
    assert span.span_id == "span_legacy"
    assert span.start == 4
    assert span.end == 9
    assert span.label == "implicit_feeling"
    assert span.reasoning == ""
    assert span.confidence is None


def test_span_annotation_json_serialization():
    """Verify resulting dictionary is valid JSON and round-trips via json.dumps/loads."""
    span = SpanAnnotation(
        span_id="span_json",
        text="pain",
        start=0,
        end=4,
        label="explicit_feeling",
        reasoning="Direct physical and emotional symptom.",
        confidence=0.88,
    )
    json_str = json.dumps(span.to_dict())
    loaded = json.loads(json_str)
    assert loaded["text"] == "pain"
    assert loaded["reasoning"] == "Direct physical and emotional symptom."
    reconstructed = SpanAnnotation.from_dict(loaded)
    assert reconstructed == span


def test_turn_and_conversation_annotation_round_trip():
    """Verify TurnAnnotation and AnnotationResult serialization and deserialization."""
    turn = TurnAnnotation(
        turn_id=1,
        speaker=Speaker.PATIENT,
        text="I am scared about the future.",
        spikes_stage=None,
        spans=[
            SpanAnnotation(span_id="s1", text="scared", start=5, end=11, label="explicit_feeling", reasoning="Fear"),
        ],
        relations=[
            RelationAnnotation(from_span_id="s1", to_span_id="s2", relation_type="response_to", reasoning="Link"),
        ],
    )
    result = AnnotationResult(
        conversation_id="conv_full",
        turns=[turn],
        agent_type="react",
        metadata={"annotator": "agent"},
    )
    json_repr = result.to_json()
    reconstructed = AnnotationResult.from_json(json_repr)
    assert reconstructed.conversation_id == "conv_full"
    assert reconstructed.agent_type == "react"
    assert len(reconstructed.turns) == 1
    t0 = reconstructed.turns[0]
    assert t0.turn_id == 1
    assert t0.speaker == Speaker.PATIENT
    assert len(t0.spans) == 1
    assert t0.spans[0].text == "scared"
    assert t0.spans[0].reasoning == "Fear"
    assert len(t0.relations) == 1
    assert t0.relations[0].reasoning == "Link"


def test_db_span_annotation_model_reasoning_support():
    """Verify database model SpanAnnotation supports reasoning seamlessly."""
    db_span = DbSpanAnnotation(
        span_id="db_s1",
        text="hurts",
        start_pos=1,
        end_pos=6,
        label="explicit_feeling",
        reasoning="Physical pain expression",
        confidence=0.85,
    )
    d = db_span.to_dict()
    assert d["reasoning"] == "Physical pain expression"
    reconstructed = DbSpanAnnotation.from_dict(d)
    assert reconstructed.reasoning == "Physical pain expression"
    assert reconstructed.start_pos == 1
    assert reconstructed.end_pos == 6
