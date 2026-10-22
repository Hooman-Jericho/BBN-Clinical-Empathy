"""Evaluation protocol for BBN annotations: spans, labels, SPIKES stages and relations.

What this module reports (and why):

* **Span P/R/F1 at several strictness levels** - ``exact`` boundaries, ``IoU >= 0.5`` and
  ``IoU >= 0.75`` - with 1-to-1 Hungarian matching. The old default (any 1-character overlap
  counts) hid boundary errors.
* **Per-label P/R/F1 and macro-F1** (label must match).
* **Label-agnostic detection F1** - separates "found the span" from "named it correctly".
* **Cohen's kappa** between agent and expert labels on label-agnostically matched spans -
  comparable with the human inter-annotator agreement reported in the literature.
* **SPIKES accuracy + confusion** on clinician turns the expert staged.
* **Relation P/R/F1** - a predicted relation is correct only if both endpoints match expert
  spans and the expert has that relation.
* **Bootstrap confidence intervals** (resampling conversations when there are >= 5 of them,
  turns otherwise) and **multi-run aggregation** (LLMs are non-deterministic).
"""

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .base import AnnotationResult, TurnAnnotation

CARDINAL_WEIGHT = 1000.0  # maximise the number of matches first, total IoU second
ALL = "__all__"
Setting = Union[str, float]


# ============================================================================ matching
def span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Intersection-over-union of two half-open character ranges."""
    (s1, e1), (s2, e2) = a, b
    if s1 >= e1 or s2 >= e2:
        return 0.0
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def hungarian_fallback(weights: List[List[float]]) -> List[Tuple[int, int]]:
    """Pure-Python Kuhn-Munkres (no scipy). Maximises total weight; only weight>0 pairs kept."""
    if not weights or not weights[0]:
        return []
    n, m = len(weights), len(weights[0])
    dim = max(n, m)
    neg_cost = [[0.0] * dim for _ in range(dim)]
    for i in range(n):
        for j in range(m):
            neg_cost[i][j] = -float(weights[i][j])

    u = [0.0] * (dim + 1)
    v = [0.0] * (dim + 1)
    p = [0] * (dim + 1)
    way = [0] * (dim + 1)
    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (dim + 1)
        used = [False] * (dim + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, dim + 1):
                if not used[j]:
                    cur = neg_cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matches = []
    for j in range(1, m + 1):
        i = p[j]
        if i != 0 and i - 1 < n and weights[i - 1][j - 1] > 0:
            matches.append((i - 1, j - 1))
    return sorted(matches)


def solve_max_weight_matching(
    weights: List[List[float]], use_scipy: bool = True
) -> List[Tuple[int, int]]:
    """Globally optimal 1-to-1 assignment (scipy if available, else the pure-Python fallback)."""
    if not weights or not weights[0]:
        return []
    if use_scipy:
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-np.array(weights, dtype=float))
            return sorted((int(r), int(c)) for r, c in zip(rows, cols) if weights[r][c] > 0)
        except ImportError:
            pass
    return hungarian_fallback(weights)


def setting_name(setting: Setting) -> str:
    return "exact" if setting == "exact" else f"iou>={float(setting):g}"


def match_spans(
    pred: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    setting: Setting,
    require_label: bool = True,
    use_scipy: bool = True,
) -> List[Tuple[int, int, float]]:
    """1-to-1 matching of predicted to gold spans -> ``[(pred_idx, gold_idx, iou)]``."""
    if not pred or not gold:
        return []
    weights = [[0.0] * len(gold) for _ in pred]
    ious = [[0.0] * len(gold) for _ in pred]
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if require_label and p["label"] != g["label"]:
                continue
            iou = span_iou((p["start"], p["end"]), (g["start"], g["end"]))
            ious[i][j] = iou
            if setting == "exact":
                ok = p["start"] == g["start"] and p["end"] == g["end"]
            else:
                thr = float(setting)
                ok = iou > 0.0 and iou >= thr
            if ok:
                weights[i][j] = CARDINAL_WEIGHT + iou
    return [(i, j, ious[i][j]) for i, j in solve_max_weight_matching(weights, use_scipy)]


# ============================================================================ statistics
def prf(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    """Precision/recall/F1; ``None`` where undefined (instead of a misleading 0 or 1)."""
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * p * r / (p + r) if (p is not None and r is not None and (p + r) > 0) else (
        0.0 if (tp + fp + fn) else None)
    return {"precision": p, "recall": r, "f1": f1}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items (nan for no items).

    Degenerate case: if chance agreement is 1 (both raters always use one single category)
    kappa is undefined; perfect agreement is then reported as 1.0.
    """
    n = len(a)
    if n == 0 or n != len(b):
        return math.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else math.nan
    return (po - pe) / (1 - pe)


# ============================================================================ counting
def _add(d: Dict[str, Dict[str, int]], key: str, tp: int = 0, fp: int = 0, fn: int = 0) -> None:
    cell = d.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
    cell["tp"] += tp
    cell["fp"] += fp
    cell["fn"] += fn


class EvalCounts:
    """Additive raw counts; summing instances == evaluating the concatenated data."""

    def __init__(self) -> None:
        self.span: Dict[str, Dict[str, Dict[str, int]]] = {}  # setting -> label|ALL -> counts
        self.detection: Dict[str, Dict[str, int]] = {}  # setting -> counts
        self.relations: Dict[str, int] = {"tp": 0, "fp": 0, "fn": 0}
        self.label_pairs: List[Tuple[str, str]] = []  # (agent label, expert label)
        self.spikes_pairs: List[Tuple[str, Optional[str]]] = []  # (expert stage, agent stage)
        self.per_turn: List[Dict[str, Any]] = []  # primary-setting counts, for the bootstrap
        self.turns_only_in_pred = 0

    def merge(self, other: "EvalCounts") -> "EvalCounts":
        for st, labels in other.span.items():
            for label, c in labels.items():
                _add(self.span.setdefault(st, {}), label, **c)
        for st, c in other.detection.items():
            _add(self.detection, st, **c)
        for k in ("tp", "fp", "fn"):
            self.relations[k] += other.relations[k]
        self.label_pairs += other.label_pairs
        self.spikes_pairs += other.spikes_pairs
        self.per_turn += other.per_turn
        self.turns_only_in_pred += other.turns_only_in_pred
        return self


# ============================================================================ normalisation
def _as_result(pred: Union[AnnotationResult, Dict[str, Any]]) -> AnnotationResult:
    return pred if isinstance(pred, AnnotationResult) else AnnotationResult.from_dict(pred)


def _pred_spans(turn: Optional[TurnAnnotation]) -> List[Dict[str, Any]]:
    if turn is None:
        return []
    return [{"span_id": s.span_id, "label": s.label, "start": s.start, "end": s.end}
            for s in turn.spans]


def _gold_spans(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for s in (turn.get("annotations") or {}).get("spans", []) or []:
        out.append({
            "span_id": s.get("span_id"),
            "label": s.get("label", ""),
            "start": s.get("start", s.get("start_pos", 0)),
            "end": s.get("end", s.get("end_pos", 0)),
        })
    return out


def _gold_relations(gold: Dict[str, Any]) -> set:
    rels = set()
    for t in gold.get("turns", []):
        for r in (t.get("annotations") or {}).get("relations", []) or []:
            rels.add((str(r.get("from") or r.get("from_span_id")),
                      str(r.get("to") or r.get("to_span_id")),
                      str(r.get("type") or r.get("relation_type"))))
    return rels


# ============================================================================ evaluation
def count_conversation(
    pred: Union[AnnotationResult, Dict[str, Any]],
    gold: Dict[str, Any],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    primary_iou: float = 0.5,
    use_scipy: bool = True,
) -> EvalCounts:
    """Raw counts for one (prediction, gold) pair."""
    result = _as_result(pred)
    pred_turns = {t.turn_id: t for t in result.turns}
    gold_turns = {t["turn_id"]: t for t in gold.get("turns", [])
                  if isinstance(t, dict) and "turn_id" in t}
    settings: List[Setting] = ["exact"] + list(iou_thresholds)
    primary = primary_iou
    conv_id = str(gold.get("id", result.conversation_id))

    counts = EvalCounts()
    counts.turns_only_in_pred = len(set(pred_turns) - set(gold_turns))
    pred_to_gold: Dict[str, str] = {}

    for turn_id in sorted(set(pred_turns) | set(gold_turns)):
        p_spans = _pred_spans(pred_turns.get(turn_id))
        g_turn = gold_turns.get(turn_id, {})
        g_spans = _gold_spans(g_turn)

        for st in settings:
            name = setting_name(st)
            matches = match_spans(p_spans, g_spans, st, True, use_scipy)
            mp = {i for i, _, _ in matches}
            mg = {j for _, j, _ in matches}
            bucket = counts.span.setdefault(name, {})
            _add(bucket, ALL, tp=len(matches), fp=len(p_spans) - len(matches),
                 fn=len(g_spans) - len(matches))
            for i, j, _ in matches:
                _add(bucket, p_spans[i]["label"], tp=1)
            for i, s in enumerate(p_spans):
                if i not in mp:
                    _add(bucket, s["label"], fp=1)
            for j, s in enumerate(g_spans):
                if j not in mg:
                    _add(bucket, s["label"], fn=1)

            det = match_spans(p_spans, g_spans, st, False, use_scipy)
            _add(counts.detection, name, tp=len(det), fp=len(p_spans) - len(det),
                 fn=len(g_spans) - len(det))

            if st == primary:
                counts.per_turn.append({
                    "conv": conv_id, "turn_id": turn_id,
                    "span": {"tp": len(matches), "fp": len(p_spans) - len(matches),
                             "fn": len(g_spans) - len(matches)},
                    "detection": {"tp": len(det), "fp": len(p_spans) - len(det),
                                  "fn": len(g_spans) - len(det)},
                })
                for i, j, _ in det:
                    counts.label_pairs.append((p_spans[i]["label"], g_spans[j]["label"]))
                for i, j, _ in matches:
                    if p_spans[i]["span_id"] and g_spans[j]["span_id"]:
                        pred_to_gold[p_spans[i]["span_id"]] = g_spans[j]["span_id"]

        gold_stage = (g_turn.get("annotations") or {}).get("spikes_stage") if g_turn else None
        if gold_stage:
            pt = pred_turns.get(turn_id)
            counts.spikes_pairs.append((str(gold_stage), pt.spikes_stage if pt else None))

    # Relations: correct iff both endpoints matched expert spans and the expert has the relation.
    gold_rels = _gold_relations(gold)
    pred_rels = {(r.from_span_id, r.to_span_id, r.relation_type)
                 for t in result.turns for r in t.relations}
    found = set()
    tp = 0
    for f, t, ty in pred_rels:
        key = (pred_to_gold.get(f), pred_to_gold.get(t), ty)
        if key in gold_rels and key not in found:
            found.add(key)
            tp += 1
    counts.relations = {"tp": tp, "fp": len(pred_rels) - tp, "fn": len(gold_rels) - tp}
    return counts


def summarize(counts: EvalCounts, primary_iou: float = 0.5) -> Dict[str, Any]:
    """Turn raw counts into the metric dictionary."""
    out: Dict[str, Any] = {"span": {}, "detection": {}}
    for st, labels in counts.span.items():
        total = labels.get(ALL, {"tp": 0, "fp": 0, "fn": 0})
        per_label = {}
        for label, c in labels.items():
            if label == ALL:
                continue
            per_label[label] = {**c, **prf(**c), "support": c["tp"] + c["fn"]}
        f1s = [v["f1"] for v in per_label.values() if v["f1"] is not None]
        out["span"][st] = {**total, **prf(**total),
                           "macro_f1": sum(f1s) / len(f1s) if f1s else None,
                           "per_label": dict(sorted(per_label.items()))}
    for st, c in counts.detection.items():
        out["detection"][st] = {**c, **prf(**c)}

    pairs = counts.label_pairs
    agent, expert = [p for p, _ in pairs], [g for _, g in pairs]
    out["label_agreement"] = {
        "matched_spans": len(pairs),
        "accuracy": (sum(a == e for a, e in pairs) / len(pairs)) if pairs else None,
        "kappa": cohen_kappa(agent, expert) if pairs else None,
        "setting": f"iou>={primary_iou:g} (label-agnostic matching)",
    }

    sp = counts.spikes_pairs
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gold_stage, pred_stage in sp:
        confusion[gold_stage][str(pred_stage)] += 1
    out["spikes"] = {
        "n_gold_staged_turns": len(sp),
        "accuracy": (sum(g == p for g, p in sp) / len(sp)) if sp else None,
        "agent_missing": sum(p is None for _, p in sp),
        "confusion": {g: dict(v) for g, v in confusion.items()},
    }
    out["relations"] = {**counts.relations, **prf(**counts.relations)}
    out["turns_only_in_prediction"] = counts.turns_only_in_pred
    return out


def headline(summary: Dict[str, Any], primary_iou: float = 0.5) -> Dict[str, Optional[float]]:
    """Flat dictionary of the numbers worth reporting / aggregating over runs."""
    key = setting_name(primary_iou)
    span = summary["span"].get(key, {})
    det = summary["detection"].get(key, {})
    exact = summary["span"].get("exact", {})
    return {
        f"span_f1@{key}": span.get("f1"),
        f"span_precision@{key}": span.get("precision"),
        f"span_recall@{key}": span.get("recall"),
        "span_f1@exact": exact.get("f1"),
        "span_macro_f1": span.get("macro_f1"),
        f"detection_f1@{key}": det.get("f1"),
        "label_kappa": summary["label_agreement"]["kappa"],
        "spikes_accuracy": summary["spikes"]["accuracy"],
        "relation_f1": summary["relations"]["f1"],
    }


# ============================================================================ uncertainty
def bootstrap_ci(
    counts: EvalCounts,
    kind: str = "span",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    unit: str = "auto",
) -> Dict[str, Any]:
    """Percentile bootstrap CI for micro P/R/F1 at the primary setting.

    ``unit="auto"`` resamples whole conversations when there are at least 5 (turns inside a
    conversation are correlated) and turns otherwise; the unit used is reported.
    """
    convs = sorted({t["conv"] for t in counts.per_turn})
    use_conv = unit == "conversation" or (unit == "auto" and len(convs) >= 5)
    if use_conv:
        grouped: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in convs}
        for t in counts.per_turn:
            for k in ("tp", "fp", "fn"):
                grouped[t["conv"]][k] += t[kind][k]
        units = list(grouped.values())
    else:
        units = [t[kind] for t in counts.per_turn]
    if not units:
        return {"unit": "none", "n_units": 0, "n_boot": 0}

    rng = random.Random(seed)
    samples: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        draw = [units[rng.randrange(len(units))] for _ in units]
        m = prf(sum(u["tp"] for u in draw), sum(u["fp"] for u in draw), sum(u["fn"] for u in draw))
        for name in samples:
            if m[name] is not None:
                samples[name].append(m[name])

    def interval(values: List[float]) -> Optional[List[float]]:
        if not values:
            return None
        values = sorted(values)
        lo = values[int((alpha / 2) * (len(values) - 1))]
        hi = values[int((1 - alpha / 2) * (len(values) - 1))]
        return [lo, hi]

    return {"unit": "conversation" if use_conv else "turn", "n_units": len(units),
            "n_boot": n_boot, "alpha": alpha, **{k: interval(v) for k, v in samples.items()}}


def summarize_runs(runs: Iterable[Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Any]]:
    """Mean / sample-std / min / max of each headline metric over repeated runs."""
    runs = list(runs)
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted({k for r in runs for k in r}):
        vals = [r[key] for r in runs if r.get(key) is not None and not math.isnan(r[key])]
        if not vals:
            out[key] = {"n": 0, "mean": None, "std": None, "min": None, "max": None}
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        out[key] = {"n": len(vals), "mean": mean, "std": std, "min": min(vals), "max": max(vals)}
    return out


def evaluate_dataset(
    pairs: Sequence[Tuple[Union[AnnotationResult, Dict[str, Any]], Dict[str, Any]]],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    primary_iou: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
    use_scipy: bool = True,
) -> Dict[str, Any]:
    """Evaluate ``[(prediction, gold), ...]`` and return summary, CIs and per-conversation rows."""
    total = EvalCounts()
    rows = []
    for pred, gold in pairs:
        c = count_conversation(pred, gold, iou_thresholds, primary_iou, use_scipy)
        total.merge(c)
        s = summarize(c, primary_iou)
        rows.append({"conversation_id": str(gold.get("id", "")), "headline": headline(s, primary_iou)})
    summary = summarize(total, primary_iou)
    return {
        "n_conversations": len(pairs),
        "primary_iou": primary_iou,
        "summary": summary,
        "headline": headline(summary, primary_iou),
        "bootstrap": {
            "span": bootstrap_ci(total, "span", n_boot, seed),
            "detection": bootstrap_ci(total, "detection", n_boot, seed),
        } if n_boot > 0 else {},
        "per_conversation": rows,
    }
