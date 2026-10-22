"""Tests for the evaluation protocol (agents/evaluation.py)."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from agents.evaluation import (bootstrap_ci, cohen_kappa, count_conversation, evaluate_dataset,
                               hungarian_fallback, match_spans, prf, solve_max_weight_matching,
                               span_iou, summarize, summarize_runs)


def span(sid, label, s, e):
    return {"span_id": sid, "label": label, "start": s, "end": e}


def gold_conv(cid="c1"):
    return {"id": cid, "turns": [
        {"turn_id": 1, "speaker": "patient", "text": "x" * 60, "annotations": {"spans": [
            {"span_id": "eo_1", "label": "explicit_feeling", "start": 0, "end": 10, "text": "x" * 10},
            {"span_id": "eo_2", "label": "implicit_feeling", "start": 20, "end": 40, "text": "x" * 20}]}},
        {"turn_id": 2, "speaker": "clinician", "text": "y" * 60, "annotations": {
            "spikes_stage": "empathy",
            "spans": [{"span_id": "el_1", "label": "understanding_feeling", "start": 0, "end": 30,
                       "text": "y" * 30}],
            "relations": [{"from": "el_1", "to": "eo_1", "type": "response_to"}]}},
    ]}


def pred_conv(spans1, spans2, stage="empathy", rels=()):
    def sp(lst):
        return [{"span_id": i, "text": "t", "start": s, "end": e, "label": l}
                for i, l, s, e in lst]
    return {"conversation_id": "c1", "turns": [
        {"turn_id": 1, "speaker": "patient", "text": "x" * 60, "annotations": {"spans": sp(spans1), "relations": []}},
        {"turn_id": 2, "speaker": "clinician", "text": "y" * 60, "annotations": {"spikes_stage": stage, "spans": sp(spans2),
         "relations": [{"from": f, "to": t, "type": ty} for f, t, ty in rels]}}]}


PERFECT1 = [("a", "explicit_feeling", 0, 10), ("b", "implicit_feeling", 20, 40)]
PERFECT2 = [("c", "understanding_feeling", 0, 30)]


def test_iou_basic():
    assert span_iou((0, 10), (0, 10)) == 1.0
    assert span_iou((0, 10), (10, 20)) == 0.0
    assert span_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert span_iou((5, 5), (0, 10)) == 0.0


def test_perfect_prediction():
    c = count_conversation(pred_conv(PERFECT1, PERFECT2, rels=[("c", "a", "response_to")]),
                           gold_conv())
    s = summarize(c)
    assert s["span"]["exact"]["f1"] == 1.0
    assert s["spikes"]["accuracy"] == 1.0
    assert s["relations"]["f1"] == 1.0
    assert s["label_agreement"]["accuracy"] == 1.0


def test_boundary_error_separates_strictness_levels():
    # (0,10) predicted as (0,6): IoU .6  -> hit at .5, miss at .75 and exact
    p = pred_conv([("a", "explicit_feeling", 0, 6), ("b", "implicit_feeling", 20, 40)], PERFECT2)
    s = summarize(count_conversation(p, gold_conv()))
    assert s["span"]["iou>=0.5"]["tp"] == 3
    assert s["span"]["iou>=0.75"]["tp"] == 2
    assert s["span"]["exact"]["tp"] == 2


def test_wrong_label_hurts_span_f1_but_not_detection():
    p = pred_conv([("a", "implicit_feeling", 0, 10), ("b", "implicit_feeling", 20, 40)], PERFECT2)
    s = summarize(count_conversation(p, gold_conv()))
    assert s["span"]["iou>=0.5"]["tp"] == 2
    assert s["detection"]["iou>=0.5"]["f1"] == 1.0
    assert s["label_agreement"]["accuracy"] == pytest.approx(2 / 3)


def test_one_prediction_cannot_match_two_gold_spans():
    g = gold_conv()
    g["turns"][0]["annotations"]["spans"][1] = {
        "span_id": "eo_2", "label": "explicit_feeling", "start": 0, "end": 10, "text": "x" * 10}
    p = pred_conv([("a", "explicit_feeling", 0, 10)], PERFECT2)
    s = summarize(count_conversation(p, g))
    assert s["span"]["exact"]["tp"] == 2 and s["span"]["exact"]["fn"] == 1


def test_relation_requires_matching_endpoints_and_direction():
    g = gold_conv()
    bad = pred_conv(PERFECT1, PERFECT2, rels=[("c", "b", "response_to")])  # wrong target
    assert summarize(count_conversation(bad, g))["relations"]["f1"] == 0.0
    wrong_type = pred_conv(PERFECT1, PERFECT2, rels=[("c", "a", "elicitation_of")])
    assert summarize(count_conversation(wrong_type, g))["relations"]["tp"] == 0


def test_spikes_confusion_and_missing():
    s = summarize(count_conversation(pred_conv(PERFECT1, PERFECT2, stage="setting"), gold_conv()))
    assert s["spikes"]["accuracy"] == 0.0
    assert s["spikes"]["confusion"] == {"empathy": {"setting": 1}}
    s = summarize(count_conversation(pred_conv(PERFECT1, PERFECT2, stage=None), gold_conv()))
    assert s["spikes"]["agent_missing"] == 1


def test_empty_prediction_gives_zero_recall_and_undefined_precision():
    p = pred_conv([], [])
    s = summarize(count_conversation(p, gold_conv()))
    assert s["span"]["exact"]["recall"] == 0.0
    assert s["span"]["exact"]["precision"] is None  # undefined, not a fake 0 or 1
    assert s["span"]["exact"]["f1"] == 0.0


def test_kappa_known_values():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("baba")) == pytest.approx(-1.0)
    assert cohen_kappa(list("aaaa"), list("aaaa")) == 1.0
    assert math.isnan(cohen_kappa([], []))
    # sklearn-verified: po=.75, pe=.5 -> .5
    assert cohen_kappa(list("aabb"), list("aaba")) == pytest.approx(0.5)


def test_prf():
    assert prf(2, 1, 1) == {"precision": 2 / 3, "recall": 2 / 3, "f1": pytest.approx(2 / 3)}
    assert prf(0, 0, 0)["f1"] is None


def test_fallback_matcher_equals_scipy():
    rng = random.Random(3)
    for _ in range(60):
        n, m = rng.randint(1, 7), rng.randint(1, 7)
        w = [[(1000 + rng.random()) if rng.random() < 0.5 else 0.0 for _ in range(m)]
             for _ in range(n)]
        a = solve_max_weight_matching(w, use_scipy=True)
        b = hungarian_fallback(w)
        score = lambda ms: sum(w[i][j] for i, j in ms)
        assert len(a) == len(b)
        assert score(a) == pytest.approx(score(b))


def test_match_spans_global_optimum_beats_greedy():
    pred = [span("p1", "L", 0, 10), span("p2", "L", 0, 4)]
    gold = [span("g1", "L", 0, 4), span("g2", "L", 0, 10)]
    ms = match_spans(pred, gold, 0.3)
    assert sorted((i, j) for i, j, _ in ms) == [(0, 1), (1, 0)]


def test_bootstrap_is_seeded_and_brackets_estimate():
    pairs = [(pred_conv(PERFECT1, PERFECT2), gold_conv("c1")) for _ in range(3)]
    pairs += [(pred_conv([], []), gold_conv("c1"))]
    rep = evaluate_dataset(pairs, n_boot=200, seed=1)
    rep2 = evaluate_dataset(pairs, n_boot=200, seed=1)
    assert rep["bootstrap"] == rep2["bootstrap"]
    lo, hi = rep["bootstrap"]["span"]["f1"]
    assert lo <= rep["headline"]["span_f1@iou>=0.5"] <= hi
    assert rep["bootstrap"]["span"]["unit"] == "turn"  # <5 conversations -> turn level


def test_bootstrap_uses_conversations_when_enough():
    c = count_conversation(pred_conv(PERFECT1, PERFECT2), gold_conv("a"))
    for k in "bcdef":
        c.merge(count_conversation(pred_conv(PERFECT1, PERFECT2), gold_conv(k)))
    assert bootstrap_ci(c, n_boot=50)["unit"] == "conversation"


def test_summarize_runs():
    out = summarize_runs([{"f1": 0.5}, {"f1": 0.7}, {"f1": None}])
    assert out["f1"]["n"] == 2 and out["f1"]["mean"] == pytest.approx(0.6)
    assert out["f1"]["std"] == pytest.approx(0.1414, abs=1e-3)
