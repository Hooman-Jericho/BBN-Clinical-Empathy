"""Validation and repair of conversation / gold-annotation JSON files.

Used by CI (so a mis-aligned sample can never be committed again), by the ``validate`` CLI
command and before every evaluation run.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import ALL_LABELS, CLINICIAN_LABELS, PATIENT_LABELS, SPIKES_STAGES

RELATION_TYPES = {"response_to", "elicitation_of"}
SPEAKERS = {"patient", "clinician"}


def validate_conversation(conv: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    issues: List[str] = []
    if not isinstance(conv, dict):
        return ["top level must be a JSON object"]
    if not conv.get("id"):
        issues.append("missing conversation 'id'")
    turns = conv.get("turns")
    if not isinstance(turns, list) or not turns:
        return issues + ["'turns' must be a non-empty list"]

    span_owner: Dict[str, str] = {}  # span_id -> speaker
    seen_turn_ids = set()
    last_id = None

    for idx, turn in enumerate(turns):
        where = f"turn[{idx}]"
        if not isinstance(turn, dict):
            issues.append(f"{where}: not an object")
            continue
        tid = turn.get("turn_id")
        where = f"turn {tid}"
        if not isinstance(tid, int):
            issues.append(f"{where}: turn_id must be an integer")
        elif tid in seen_turn_ids:
            issues.append(f"{where}: duplicate turn_id")
        elif last_id is not None and tid <= last_id:
            issues.append(f"{where}: turn_ids must be strictly increasing")
        if isinstance(tid, int):
            seen_turn_ids.add(tid)
            last_id = tid

        speaker = turn.get("speaker")
        if speaker not in SPEAKERS:
            issues.append(f"{where}: speaker must be one of {sorted(SPEAKERS)}, got {speaker!r}")
            continue
        text = turn.get("text")
        if not isinstance(text, str) or not text:
            issues.append(f"{where}: missing text")
            continue

        ann = turn.get("annotations") or {}
        allowed = PATIENT_LABELS if speaker == "patient" else CLINICIAN_LABELS
        for span in ann.get("spans", []) or []:
            sid = span.get("span_id")
            label = span.get("label")
            if not sid:
                issues.append(f"{where}: span without span_id")
            elif sid in span_owner:
                issues.append(f"{where}: duplicate span_id {sid!r}")
            else:
                span_owner[sid] = speaker
            if label not in ALL_LABELS:
                issues.append(f"{where} {sid}: unknown label {label!r}")
            elif label not in allowed:
                issues.append(f"{where} {sid}: label {label!r} is not a {speaker} label")
            s, e = span.get("start"), span.get("end")
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)):
                issues.append(f"{where} {sid}: invalid offsets [{s}:{e}] for text length {len(text)}")
            elif text[s:e] != span.get("text"):
                issues.append(
                    f"{where} {sid}: offsets [{s}:{e}] give {text[s:e]!r} but text is {span.get('text')!r}")

        stage = ann.get("spikes_stage")
        if stage is not None:
            if speaker != "clinician":
                issues.append(f"{where}: spikes_stage is only valid on clinician turns")
            elif stage not in SPIKES_STAGES:
                issues.append(f"{where}: unknown spikes_stage {stage!r}")

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        for rel in (turn.get("annotations") or {}).get("relations", []) or []:
            f, t, ty = rel.get("from"), rel.get("to"), rel.get("type")
            where = f"turn {turn.get('turn_id')} relation {f}->{t}"
            if ty not in RELATION_TYPES:
                issues.append(f"{where}: unknown type {ty!r}")
            if f not in span_owner:
                issues.append(f"{where}: unknown 'from' span")
            elif span_owner[f] != "clinician":
                issues.append(f"{where}: 'from' must be a clinician span")
            if t not in span_owner:
                issues.append(f"{where}: unknown 'to' span")
            elif span_owner[t] != "patient":
                issues.append(f"{where}: 'to' must be a patient span")
    return issues


def repair_offsets(conv: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Re-derive span offsets from the span text (nearest occurrence to the given start)."""
    fixed = copy.deepcopy(conv)
    log: List[str] = []
    for turn in fixed.get("turns", []):
        text = turn.get("text", "")
        for span in (turn.get("annotations") or {}).get("spans", []) or []:
            s, e = span.get("start"), span.get("end")
            if isinstance(s, int) and isinstance(e, int) and text[s:e] == span.get("text"):
                continue
            needle = span.get("text", "")
            occ, pos = [], text.find(needle) if needle else -1
            while pos != -1:
                occ.append(pos)
                pos = text.find(needle, pos + 1)
            if not occ:
                log.append(f"turn {turn.get('turn_id')} {span.get('span_id')}: text not found; unfixable")
                continue
            new_s = min(occ, key=lambda p: abs(p - (s if isinstance(s, int) else 0)))
            log.append(f"turn {turn.get('turn_id')} {span.get('span_id')}: "
                       f"[{s}:{e}] -> [{new_s}:{new_s + len(needle)}]")
            span["start"], span["end"] = new_s, new_s + len(needle)
    return fixed, log


def strip_annotations(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Remove every annotation so gold labels can never leak into an agent's input."""
    return {
        "id": conv.get("id", "unknown"),
        "metadata": {k: v for k, v in (conv.get("metadata") or {}).items()},
        "turns": [{"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]}
                  for t in conv.get("turns", [])],
    }


def validate_file(path: Path) -> List[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            conv = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {path}: {exc}"]
    return validate_conversation(conv)
