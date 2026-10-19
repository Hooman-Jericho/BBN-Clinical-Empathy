"""Unit tests for annotation operations, boundary deduplication, and relations."""

import pytest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from services.annotations import (
    add_span_annotation,
    remove_span_annotation,
    add_relation,
    remove_relation,
    set_spikes_stage,
)
from state import init_session_state, undo_last_action


@pytest.fixture(autouse=True)
def reset_session():
    """Reset streamlit session state before each test."""
    st.session_state.clear()
    init_session_state()


def test_add_span_allows_identical_words_at_different_positions():
    """Verify that identical text phrases at distinct character offsets are both preserved."""
    turn_id = 1
    # First occurrence of "fine"
    span1_id = add_span_annotation(
        turn_id=turn_id,
        text="fine",
        start=5,
        end=9,
        label="explicit_feeling",
    )
    assert span1_id is not None

    # Second occurrence of "fine" at a different position in the same turn
    span2_id = add_span_annotation(
        turn_id=turn_id,
        text="fine",
        start=45,
        end=49,
        label="explicit_feeling",
    )
    assert span2_id is not None
    assert span1_id != span2_id

    # Verify both exist in state
    turn_spans = st.session_state.current_annotations[turn_id]["spans"]
    assert len(turn_spans) == 2
    assert turn_spans[0]["start"] == 5
    assert turn_spans[1]["start"] == 45


def test_exact_duplicate_span_is_rejected():
    """Verify that identical start, end, and label are rejected as duplicate."""
    turn_id = 1
    span1_id = add_span_annotation(turn_id, "anxious", 10, 17, "explicit_feeling")
    assert span1_id is not None

    # Exact duplicate
    span2_id = add_span_annotation(turn_id, "anxious", 10, 17, "explicit_feeling")
    assert span2_id is None

    # Different label at same position is allowed (multi-labeling)
    span3_id = add_span_annotation(turn_id, "anxious", 10, 17, "implicit_feeling")
    assert span3_id is not None


def test_remove_span_cascades_to_relations():
    """Verify that deleting a span also cleans up relations pointing to or from it."""
    span_a = add_span_annotation(1, "worried", 0, 7, "explicit_feeling")
    span_b = add_span_annotation(2, "I hear you", 0, 10, "understanding_feeling")

    rel_id = add_relation(
        from_turn_id=2,
        from_span_id=span_b,
        to_turn_id=1,
        to_span_id=span_a,
        relation_type="response_to",
    )
    assert rel_id is not None
    assert len(st.session_state.current_annotations[2]["relations"]) == 1

    # Remove span_a from turn 1
    remove_span_annotation(1, span_a)

    # Relation referencing span_a should be automatically purged
    assert len(st.session_state.current_annotations[2]["relations"]) == 0


def test_remove_relation_by_id():
    """Verify that relations can be deleted explicitly by relation_id."""
    span_a = add_span_annotation(1, "worried", 0, 7, "explicit_feeling")
    span_b = add_span_annotation(2, "I hear you", 0, 10, "understanding_feeling")

    rel_id = add_relation(2, span_b, 1, span_a, "response_to")
    assert len(st.session_state.current_annotations[2]["relations"]) == 1

    deleted = remove_relation(2, rel_id)
    assert deleted is True
    assert len(st.session_state.current_annotations[2]["relations"]) == 0


def test_set_and_clear_spikes_stage():
    """Verify setting SPIKES stage and clearing with None."""
    set_spikes_stage(1, "empathy")
    assert st.session_state.current_annotations[1]["spikes_stage"] == "empathy"

    set_spikes_stage(1, None)
    assert st.session_state.current_annotations[1]["spikes_stage"] is None

    set_spikes_stage(1, "None")
    assert st.session_state.current_annotations[1]["spikes_stage"] is None
