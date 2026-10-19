"""Regression test suite for expert evaluation and Maximum Bipartite Matching algorithms.

Covers all 10 mandated scenarios in Phase 3 of the Master Audit Prompt:
1. Two exact predictions and two exact expert spans with identical labels.
2. One prediction against two expert spans (1-to-1 matching constraint).
3. Two duplicate predictions against one expert span.
4. Correct character span but incorrect clinical label.
5. Partial overlap below and above defined min_iou_threshold.
6. Empty predictions against non-empty expert spans.
7. Non-empty predictions against empty expert spans.
8. Both predictions and expert spans empty (zero-denominator safety).
9. Identical labels on distinct character positions (disambiguation).
10. Maximum bipartite matching vs greedy assignment optimality.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

import pytest
from agents.runner import AnnotationRunner
from agents.base import AnnotationResult, TurnAnnotation, SpanAnnotation, Speaker


@pytest.fixture
def runner():
    return AnnotationRunner(agent_type="react", provider="openai")


def test_scenario_1_exact_matches_identical_labels(runner):
    """Scenario 1: Two exact predictions and two exact expert spans with same label:
    TP = 2, FP = 0, FN = 0, P = 1.0, R = 1.0, F1 = 1.0.
    """
    expert_data = {
        "id": "c1",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 0, "end": 5, "label": "explicit_feeling"},
                        {"span_id": "e2", "start": 10, "end": 15, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c1",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="scared and happy",
                spans=[
                    SpanAnnotation(span_id="a1", text="scared", start=0, end=5, label="explicit_feeling"),
                    SpanAnnotation(span_id="a2", text="happy", start=10, end=15, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["avg_span_iou"] == 1.0


def test_scenario_2_one_prediction_against_two_experts(runner):
    """Scenario 2: One prediction against two expert spans: at most one TP."""
    expert_data = {
        "id": "c2",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 0, "end": 6, "label": "explicit_feeling"},
                        {"span_id": "e2", "start": 10, "end": 16, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c2",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="scared ... afraid",
                spans=[
                    SpanAnnotation(span_id="a1", text="scared", start=0, end=6, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(0.6667, abs=1e-3)


def test_scenario_3_two_duplicate_predictions_against_one_expert(runner):
    """Scenario 3: Two duplicate predictions against one expert span: at most one TP."""
    expert_data = {
        "id": "c3",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 0, "end": 6, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c3",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="scared",
                spans=[
                    SpanAnnotation(span_id="a1", text="scared", start=0, end=6, label="explicit_feeling"),
                    SpanAnnotation(span_id="a2", text="scared", start=0, end=6, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.6667, abs=1e-3)


def test_scenario_4_correct_span_wrong_label(runner):
    """Scenario 4: A prediction with the correct span but wrong label: TP = 0, FP = 1, FN = 1."""
    expert_data = {
        "id": "c4",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 5, "end": 12, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c4",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="I feel anxious.",
                spans=[
                    SpanAnnotation(span_id="a1", text="anxious", start=5, end=12, label="implicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_scenario_5_partial_overlap_threshold(runner):
    """Scenario 5: Partial overlap below and above threshold."""
    # Span 1: [0:10], Span 2: [5:15]. Intersection is [5:10] (5 chars), Union is [0:15] (15 chars) -> IoU = 5/15 = 0.333
    expert_data = {
        "id": "c5",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 0, "end": 10, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c5",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="0123456789012345",
                spans=[
                    SpanAnnotation(span_id="a1", text="5678901234", start=5, end=15, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    # Threshold 0.5: IoU 0.333 < 0.5 -> No match
    strict_metrics = runner.compare_with_expert(agent_result, expert_data, min_iou_threshold=0.5)
    assert strict_metrics["true_positives"] == 0
    assert strict_metrics["false_positives"] == 1
    assert strict_metrics["false_negatives"] == 1

    # Threshold 0.2: IoU 0.333 >= 0.2 -> Matches!
    lenient_metrics = runner.compare_with_expert(agent_result, expert_data, min_iou_threshold=0.2)
    assert lenient_metrics["true_positives"] == 1
    assert lenient_metrics["false_positives"] == 0
    assert lenient_metrics["false_negatives"] == 0
    assert lenient_metrics["avg_span_iou"] == pytest.approx(0.3333, abs=1e-3)


def test_scenario_6_empty_predictions_non_empty_expert(runner):
    """Scenario 6: Empty predictions and non-empty expert: TP = 0, FP = 0, FN = N."""
    expert_data = {
        "id": "c6",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 0, "end": 5, "label": "explicit_feeling"},
                        {"span_id": "e2", "start": 10, "end": 15, "label": "implicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c6",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="hello world",
                spans=[],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 2
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_scenario_7_non_empty_predictions_empty_expert(runner):
    """Scenario 7: Non-empty predictions and empty expert: TP = 0, FP = N, FN = 0."""
    expert_data = {
        "id": "c7",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {"spans": []},
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c7",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="hello world",
                spans=[
                    SpanAnnotation(span_id="a1", text="hello", start=0, end=5, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_scenario_8_both_empty_zero_denominator_safe(runner):
    """Scenario 8: Both inputs empty: TP = 0, FP = 0, FN = 0, Precision/Recall/F1 = 0.0."""
    expert_data = {
        "id": "c8",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {"spans": []},
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c8",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="clean text",
                spans=[],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["avg_span_iou"] == 0.0


def test_scenario_9_identical_labels_distinct_positions(runner):
    """Scenario 9: Identical labels on distinct text positions correctly disambiguated."""
    expert_data = {
        "id": "c9",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 0, "end": 4, "label": "explicit_feeling"},
                        {"span_id": "e2", "start": 30, "end": 34, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c9",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="pain is at the start and then pain is at the end",
                spans=[
                    # Order in agent spans is reversed to test coordinate matching, not order matching
                    SpanAnnotation(span_id="a2", text="pain", start=30, end=34, label="explicit_feeling"),
                    SpanAnnotation(span_id="a1", text="pain", start=0, end=4, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    metrics = runner.compare_with_expert(agent_result, expert_data)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["avg_span_iou"] == 1.0


def test_scenario_10_maximum_bipartite_matching_beats_greedy(runner):
    """Scenario 10: Case in which greedy matching produces a worse assignment than maximum matching."""
    expert_data = {
        "id": "c10",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e0", "start": 10, "end": 30, "label": "explicit_feeling"},
                        {"span_id": "e1", "start": 20, "end": 40, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    agent_result = AnnotationResult(
        conversation_id="c10",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="." * 50,
                spans=[
                    SpanAnnotation(span_id="a0", text="." * 23, start=12, end=35, label="explicit_feeling"),
                    SpanAnnotation(span_id="a1", text="." * 20, start=5, end=25, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )

    metrics = runner.compare_with_expert(agent_result, expert_data, min_iou_threshold=0.3)
    # Under maximum bipartite matching, both spans MUST be matched!
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_exact_match_flag(runner):
    """Test exact_match=True requires 100% boundary alignment."""
    expert_data = {
        "id": "c_exact",
        "turns": [
            {
                "turn_id": 1,
                "annotations": {
                    "spans": [
                        {"span_id": "e1", "start": 5, "end": 15, "label": "explicit_feeling"},
                    ]
                },
            }
        ],
    }
    # Slight boundary shift [5:16]
    agent_result = AnnotationResult(
        conversation_id="c_exact",
        turns=[
            TurnAnnotation(
                turn_id=1,
                speaker=Speaker.PATIENT,
                text="." * 30,
                spans=[
                    SpanAnnotation(span_id="a1", text="." * 11, start=5, end=16, label="explicit_feeling"),
                ],
            )
        ],
        agent_type="react",
    )
    # exact_match=True must reject the slight boundary shift
    exact_metrics = runner.compare_with_expert(agent_result, expert_data, exact_match=True)
    assert exact_metrics["true_positives"] == 0
    assert exact_metrics["false_positives"] == 1
    assert exact_metrics["false_negatives"] == 1

    # exact_match=False accepts it with high IoU
    overlap_metrics = runner.compare_with_expert(agent_result, expert_data, exact_match=False)
    assert overlap_metrics["true_positives"] == 1
    assert overlap_metrics["false_positives"] == 0
    assert overlap_metrics["false_negatives"] == 0
