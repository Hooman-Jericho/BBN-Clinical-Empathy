"""Every committed sample must be valid; the validator must catch real defects."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agents.data_validation import (repair_offsets, strip_annotations, validate_conversation,
                                    validate_file)

SAMPLES = sorted((Path(__file__).parent.parent / "data").rglob("*.json"))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_committed_data_is_valid(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "turns" in data:
        assert validate_file(path) == []


def _conv():
    return {"id": "c", "turns": [
        {"turn_id": 1, "speaker": "patient", "text": "I am scared today", "annotations": {"spans": [
            {"span_id": "eo_1", "label": "explicit_feeling", "start": 5, "end": 11, "text": "scared"}]}},
        {"turn_id": 2, "speaker": "clinician", "text": "That sounds hard", "annotations": {
            "spikes_stage": "empathy",
            "spans": [{"span_id": "el_1", "label": "understanding_feeling", "start": 0,
                       "end": 11, "text": "That sounds"}],
            "relations": [{"from": "el_1", "to": "eo_1", "type": "response_to"}]}}]}


def test_clean_conversation_ok():
    assert validate_conversation(_conv()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["turns"][0]["annotations"]["spans"][0].update(start=6, end=12), "offsets"),
    (lambda c: c["turns"][0]["annotations"]["spans"][0].update(label="nope"), "unknown label"),
    (lambda c: c["turns"][0]["annotations"]["spans"][0].update(label="direct_elicitation_feeling"),
     "not a patient label"),
    (lambda c: c["turns"][1]["annotations"].update(spikes_stage="xx"), "spikes_stage"),
    (lambda c: c["turns"][1]["annotations"]["relations"][0].update(to="zz"), "unknown 'to'"),
    (lambda c: c["turns"][1]["annotations"]["relations"][0].update(**{"from": "eo_1"}), "clinician"),
    (lambda c: c["turns"][1].update(turn_id=1), "duplicate turn_id"),
    (lambda c: c["turns"][0].update(speaker="doctor"), "speaker"),
])
def test_validator_catches_defect(mutate, needle):
    c = copy.deepcopy(_conv())
    mutate(c)
    assert any(needle in i for i in validate_conversation(c))


def test_repair_and_strip():
    c = _conv()
    c["turns"][0]["annotations"]["spans"][0].update(start=6, end=12)
    fixed, log = repair_offsets(c)
    assert validate_conversation(fixed) == [] and len(log) == 1
    stripped = strip_annotations(fixed)
    assert "annotations" not in json.dumps(stripped)
