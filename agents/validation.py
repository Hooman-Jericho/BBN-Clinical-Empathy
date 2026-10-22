"""Strict, observable validation of LLM output.

Design rules (each one fixes a concrete problem found in the review):

* Spans are validated **one by one** - a single malformed span never discards its siblings.
* A label must be a *known* label **and** belong to the turn's speaker; anything else is rejected
  and counted, never silently accepted.
* Nothing fails silently: every rejection increments a counter in :class:`ParseStats`, which is
  stored in the turn / conversation diagnostics. "Parse failed" is therefore distinguishable
  from "model legitimately found nothing".
* ``confidence`` is only kept if the model really supplied a value in [0, 1]; there is no
  fabricated default.
* Relations must connect a clinician span to a patient span that exist in this conversation.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pydantic import ValidationError

from .base import (
    ALL_LABELS,
    SPIKES_STAGES,
    RelationAnnotation,
    Speaker,
    SpanAnnotation,
    generate_span_id,
    get_labels_for_speaker,
)
from .jsonutil import resolve_span_coordinates
from .schemas import RelationPredictionSchema

logger = logging.getLogger(__name__)


@dataclass
class ParseStats:
    """Counters describing everything that was rejected or repaired while parsing."""

    json_failures: int = 0  # model output was not parseable JSON
    schema_failures: int = 0  # JSON parsed, but structure/protocol was wrong
    spans_proposed: int = 0
    spans_accepted: int = 0
    spans_rejected_schema: int = 0
    spans_rejected_label: int = 0  # label not in the framework at all
    spans_rejected_speaker: int = 0  # valid label, but for the other speaker role
    spans_rejected_text: int = 0  # empty text, or text not found in the turn
    spans_rejected_duplicate: int = 0
    spans_boundary_corrected: int = 0  # offsets re-derived from the span text
    spikes_invalid: int = 0
    relations_proposed: int = 0
    relations_accepted: int = 0
    relations_rejected: int = 0
    critic_fallbacks: int = 0  # critic output unusable -> proposals kept (flagged!)
    critic_invented: int = 0  # critic "approved" a span nobody proposed
    critic_rejected_all: int = 0  # critic legitimately rejected every proposal

    def merge(self, other: "ParseStats") -> "ParseStats":
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)
        return self

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    def nonzero(self) -> Dict[str, int]:
        return {k: v for k, v in asdict(self).items() if v}

    @property
    def problems(self) -> int:
        """Number of things that went wrong (excludes benign counters)."""
        d = asdict(self)
        benign = {"spans_proposed", "spans_accepted", "relations_proposed", "relations_accepted",
                  "critic_rejected_all", "spans_boundary_corrected"}
        return sum(v for k, v in d.items() if k not in benign)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _clean_confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0.0 <= float(value) <= 1.0 else None


def parse_turn_payload(
    parsed: Any, speaker: Speaker, stats: ParseStats
) -> Tuple[Optional[str], List[Any], str]:
    """Split a model's final answer into ``(spikes_stage, raw_spans, status)``.

    ``status`` is ``"ok"``, ``"parse_failed"`` (nothing usable) or ``"schema_failed"``.
    An empty ``annotations`` list with status ``"ok"`` is a legitimate "nothing found".
    """
    if not isinstance(parsed, dict) or not parsed:
        stats.json_failures += 1
        return None, [], "parse_failed"

    status = "ok"
    raw = parsed.get("annotations", [])
    if not isinstance(raw, list):
        stats.schema_failures += 1
        raw, status = [], "schema_failed"

    stage = parsed.get("spikes_stage")
    stage_out: Optional[str] = None
    if stage is not None:
        if speaker == Speaker.CLINICIAN and str(stage).lower() in SPIKES_STAGES:
            stage_out = str(stage).lower()
        else:
            stats.spikes_invalid += 1
    return stage_out, raw, status


def build_spans(
    raw_spans: Iterable[Any],
    turn_id: int,
    text: str,
    speaker: Speaker,
    stats: ParseStats,
) -> List[SpanAnnotation]:
    """Validate raw model spans one at a time and return the accepted ones."""
    valid_for_speaker = get_labels_for_speaker(speaker)
    out: List[SpanAnnotation] = []
    seen: Set[Tuple[int, int, str]] = set()

    for raw in raw_spans or []:
        stats.spans_proposed += 1
        if not isinstance(raw, dict):
            stats.spans_rejected_schema += 1
            continue

        label = raw.get("label")
        if label not in ALL_LABELS:
            logger.warning("Rejected span with unknown label %r in turn %s", label, turn_id)
            stats.spans_rejected_label += 1
            continue
        if label not in valid_for_speaker:
            logger.warning("Rejected %r: not a %s label (turn %s)", label, speaker.value, turn_id)
            stats.spans_rejected_speaker += 1
            continue

        span_text = str(raw.get("text", "")).strip()
        if not span_text or span_text not in text:
            stats.spans_rejected_text += 1
            continue

        hint_start, hint_end = _coerce_int(raw.get("start")), _coerce_int(raw.get("end"))
        h_s = hint_start if hint_start is not None and hint_start >= 0 else 0
        h_e = hint_end if hint_end is not None and hint_end >= 0 else 0
        start, end = resolve_span_coordinates(text, span_text, h_s, h_e)
        if not (0 <= start < end <= len(text)):
            stats.spans_rejected_text += 1
            continue
        if hint_start is not None and hint_end is not None and (hint_start, hint_end) != (start, end):
            stats.spans_boundary_corrected += 1

        key = (start, end, label)
        if key in seen:
            stats.spans_rejected_duplicate += 1
            continue
        seen.add(key)

        out.append(
            SpanAnnotation(
                span_id=generate_span_id(turn_id, len(out)),
                text=text[start:end],
                start=start,
                end=end,
                label=label,
                reasoning=str(raw.get("reasoning", "") or ""),
                confidence=_clean_confidence(raw.get("confidence")),
            )
        )
        stats.spans_accepted += 1
    return out


def build_relations(
    parsed: Any,
    patient_ids: Set[str],
    clinician_ids: Set[str],
    stats: ParseStats,
) -> List[RelationAnnotation]:
    """Validate relations: known endpoints, clinician -> patient direction, no duplicates."""
    if not isinstance(parsed, dict) or not parsed:
        stats.json_failures += 1
        return []
    raw = parsed.get("relations", [])
    if not isinstance(raw, list):
        stats.schema_failures += 1
        return []

    relations: List[RelationAnnotation] = []
    seen: Set[Tuple[str, str, str]] = set()
    for item in raw:
        stats.relations_proposed += 1
        try:
            rel = RelationPredictionSchema.model_validate(item)
        except ValidationError:
            stats.relations_rejected += 1
            continue
        if rel.from_span_id not in clinician_ids or rel.to_span_id not in patient_ids:
            logger.warning("Rejected relation with invalid endpoints: %s -> %s",
                           rel.from_span_id, rel.to_span_id)
            stats.relations_rejected += 1
            continue
        key = (rel.from_span_id, rel.to_span_id, rel.relation_type)
        if key in seen:
            stats.relations_rejected += 1
            continue
        seen.add(key)
        relations.append(
            RelationAnnotation(
                from_span_id=rel.from_span_id,
                to_span_id=rel.to_span_id,
                relation_type=rel.relation_type,
                reasoning=rel.reasoning or "",
            )
        )
        stats.relations_accepted += 1
    return relations


def _overlaps_a_proposal(approved: Dict[str, Any], proposals: List[Dict[str, Any]]) -> bool:
    a_text = str(approved.get("text", "")).strip()
    for prop in proposals:
        if prop.get("label") != approved.get("label"):
            continue
        p_text = str(prop.get("text", "")).strip()
        if a_text and p_text and (a_text in p_text or p_text in a_text):
            return True
    return False


def select_critic_spans(
    parsed: Any, proposals: List[Dict[str, Any]], stats: ParseStats
) -> Optional[List[Dict[str, Any]]]:
    """Return the critic's approved spans, or ``None`` if the critic output is unusable.

    * an explicit empty ``approved_spans`` list is honoured (the critic *can* reject everything);
    * a span the specialists never proposed is dropped and counted (``critic_invented``);
    * ``None`` tells the caller to fall back to the proposals **and** records ``critic_fallbacks``.
    """
    if not isinstance(parsed, dict) or not isinstance(parsed.get("approved_spans"), list):
        stats.critic_fallbacks += 1
        return None
    approved: List[Dict[str, Any]] = []
    for span in parsed["approved_spans"]:
        if not isinstance(span, dict):
            continue
        if _overlaps_a_proposal(span, proposals):
            approved.append(span)
        else:
            stats.critic_invented += 1
    if proposals and not approved:
        stats.critic_rejected_all += 1
    return approved
