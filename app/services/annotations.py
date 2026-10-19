"""Annotation CRUD and state management operations."""

import uuid
import streamlit as st
from state import save_to_undo_history


def add_span_annotation(
    turn_id: int,
    text: str,
    start: int,
    end: int,
    label: str,
    confidence: float = None,
    source: str = "manual",
) -> str | None:
    """Add a span annotation. Returns span_id or None if invalid or duplicate."""
    if start < 0 or end <= start:
        return None

    if turn_id not in st.session_state.current_annotations:
        st.session_state.current_annotations[turn_id] = {
            "spans": [],
            "relations": [],
            "spikes_stage": None,
        }

    existing_spans = st.session_state.current_annotations[turn_id].get("spans", [])

    # Exact boundary and label match duplicate check (preserves identical words at different positions)
    for existing in existing_spans:
        if existing["start"] == start and existing["end"] == end and existing["label"] == label:
            return None

    save_to_undo_history()

    span_id = f"span_{uuid.uuid4().hex[:8]}"
    span = {
        "span_id": span_id,
        "text": text,
        "start": start,
        "end": end,
        "label": label,
        "source": source,
    }
    if confidence is not None:
        span["confidence"] = confidence

    existing_spans.append(span)
    st.session_state.current_annotations[turn_id]["spans"] = existing_spans
    return span_id


def remove_span_annotation(turn_id: int, span_id: str):
    """Remove a span annotation and cascade delete any referencing relations."""
    save_to_undo_history()

    if turn_id in st.session_state.current_annotations:
        st.session_state.current_annotations[turn_id]["spans"] = [
            s for s in st.session_state.current_annotations[turn_id].get("spans", [])
            if s["span_id"] != span_id
        ]

    # Cascade remove relations referencing this span from ALL turns
    for tid in list(st.session_state.current_annotations.keys()):
        if "relations" in st.session_state.current_annotations[tid]:
            st.session_state.current_annotations[tid]["relations"] = [
                r for r in st.session_state.current_annotations[tid]["relations"]
                if r["from"] != span_id and r["to"] != span_id
            ]


def add_relation(
    from_turn_id: int,
    from_span_id: str,
    to_turn_id: int,
    to_span_id: str,
    relation_type: str,
) -> str:
    """Add a semantic relation between two spans."""
    save_to_undo_history()

    if from_turn_id not in st.session_state.current_annotations:
        st.session_state.current_annotations[from_turn_id] = {
            "spans": [],
            "relations": [],
            "spikes_stage": None,
        }

    rel_id = f"rel_{uuid.uuid4().hex[:8]}"
    relation = {
        "relation_id": rel_id,
        "from": from_span_id,
        "to": to_span_id,
        "to_turn_id": to_turn_id,
        "type": relation_type,
    }

    st.session_state.current_annotations[from_turn_id]["relations"].append(relation)
    return rel_id


def remove_relation(turn_id: int, relation_id: str) -> bool:
    """Remove a specific relation by ID."""
    save_to_undo_history()

    removed = False
    if turn_id in st.session_state.current_annotations:
        rels = st.session_state.current_annotations[turn_id].get("relations", [])
        initial_len = len(rels)
        st.session_state.current_annotations[turn_id]["relations"] = [
            r for r in rels if r.get("relation_id") != relation_id
        ]
        removed = len(st.session_state.current_annotations[turn_id]["relations"]) < initial_len

    # Also check if it was attached to any other turn
    if not removed:
        for tid in list(st.session_state.current_annotations.keys()):
            rels = st.session_state.current_annotations[tid].get("relations", [])
            initial_len = len(rels)
            st.session_state.current_annotations[tid]["relations"] = [
                r for r in rels if r.get("relation_id") != relation_id
            ]
            if len(st.session_state.current_annotations[tid]["relations"]) < initial_len:
                removed = True

    return removed


def set_spikes_stage(turn_id: int, stage: str | None):
    """Set or clear SPIKES stage for a clinician turn."""
    save_to_undo_history()

    if turn_id not in st.session_state.current_annotations:
        st.session_state.current_annotations[turn_id] = {
            "spans": [],
            "relations": [],
            "spikes_stage": None,
        }

    if stage and stage.lower() in ("none", ""):
        stage = None

    st.session_state.current_annotations[turn_id]["spikes_stage"] = stage
