"""Unit tests for agent coordinate resolution, JSON extraction, and runner comparison metrics."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

from agents.react_agent import resolve_span_coordinates, extract_json_from_text
from agents.runner import AnnotationRunner
from agents.base import AnnotationResult, TurnAnnotation, SpanAnnotation, Speaker


def test_resolve_span_coordinates_disambiguates_multiple_occurrences():
    """Verify that closest occurrence to suggested_start is selected, not always the first."""
    text = "I know that you are anxious, but I also know you can overcome this."
    # "know" appears at index 2 ("I know") and at index 40 ("also know")

    # Target the second occurrence
    start, end = resolve_span_coordinates(
        text=text,
        span_text="know",
        suggested_start=38,  # approximate offset from LLM
        suggested_end=42,
    )
    assert start == 40
    assert end == 44
    assert text[start:end] == "know"

    # Target the first occurrence
    start1, end1 = resolve_span_coordinates(
        text=text,
        span_text="know",
        suggested_start=0,
        suggested_end=4,
    )
    assert start1 == 2
    assert end1 == 6


def test_extract_json_from_text_variants():
    """Verify multi-pass JSON extraction handles raw, markdown-fenced, and surrounding text."""
    # 1. Pure JSON
    assert extract_json_from_text('{"annotations": []}') == {"annotations": []}

    # 2. Markdown fenced JSON
    fenced = """Here is the result:
```json
{
  "spikes_stage": "setting",
  "annotations": [{"text": "hello", "start": 0, "end": 5, "label": "setting"}]
}
```
Hope this helps!"""
    parsed = extract_json_from_text(fenced)
    assert parsed.get("spikes_stage") == "setting"
    assert len(parsed.get("annotations", [])) == 1

    # 3. Surrounding text without fences
    unfenced = 'Sure! Here: {"spikes_stage": "empathy", "annotations": []} Thank you.'
    parsed2 = extract_json_from_text(unfenced)
    assert parsed2.get("spikes_stage") == "empathy"

    # 4. Invalid input returns empty dict safely
    assert extract_json_from_text("no json here at all") == {}


def test_runner_compare_with_expert_metrics(tmp_path):
    """Verify AnnotationRunner.compare_with_expert calculates precision, recall, and F1 accurately."""
    runner = AnnotationRunner(agent_type="react", provider="openai")

    # Create dummy expert file
    expert_file = tmp_path / "expert.json"
    expert_file.write_text("""{
      "id": "c1",
      "turns": [
        {
          "turn_id": 1,
          "annotations": {
            "spans": [
              {"span_id": "e1", "text": "scared", "start": 5, "end": 11, "label": "explicit_feeling"}
            ]
          }
        }
      ]
    }""", encoding="utf-8")

    # Agent result with matching label and identical span
    agent_turn = TurnAnnotation(
        turn_id=1,
        speaker=Speaker.PATIENT,
        text="I am scared today.",
        spans=[
            SpanAnnotation(span_id="a1", text="scared", start=5, end=11, label="explicit_feeling")
        ]
    )
    agent_result = AnnotationResult(
        conversation_id="c1",
        turns=[agent_turn],
        agent_type="react"
    )

    metrics = runner.compare_with_expert(agent_result, str(expert_file))
    assert metrics["label_matches"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["avg_span_iou"] == 1.0
