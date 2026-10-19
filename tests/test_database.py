"""Unit and integration tests for SQLite database layer and state synchronization."""

import pytest
import sqlite3
import tempfile
from pathlib import Path
import sys

# Add app and root to sys.path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

import database.connection as db_conn
from database import (
    init_db,
    create_expert,
    get_expert,
    get_all_experts,
    create_conversation,
    get_conversation_by_external_id,
    create_turn,
    get_turns_by_conversation,
    create_span_annotation,
    get_span_annotations,
    delete_span_annotation,
    create_relation,
    get_relations,
    delete_relation,
    set_spikes_annotation,
    get_spikes_annotation,
    create_ai_suggestion,
    update_ai_suggestion_status,
    get_ai_suggestions_for_turn,
    get_full_conversation_with_annotations,
    sync_session_annotations,
)


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch, tmp_path):
    """Use an isolated SQLite database file for each test."""
    test_db_file = tmp_path / "test_annotations.db"
    monkeypatch.setattr(db_conn, "DB_PATH", test_db_file)
    monkeypatch.setattr(db_conn, "DB_DIR", tmp_path)
    init_db()
    yield test_db_file


def test_expert_crud():
    """Test expert creation and retrieval."""
    expert = create_expert("Dr. Sarah", "sarah@clinic.org", "reviewer")
    assert expert.id is not None
    assert expert.name == "Dr. Sarah"
    assert expert.role == "reviewer"

    fetched = get_expert(expert.id)
    assert fetched is not None
    assert fetched.name == "Dr. Sarah"

    all_experts = get_all_experts()
    assert len(all_experts) == 1
    assert all_experts[0].name == "Dr. Sarah"


def test_conversation_and_turn_batch_retrieval():
    """Test batch conversation retrieval without N+1 query penalty."""
    conv = create_conversation(
        external_id="conv_test_001",
        scenario="Pancreatic mass",
        language="en",
        date="2026-10-19",
    )
    assert conv.id is not None

    turn1 = create_turn(conv.id, 1, "clinician", "Hello, please sit down.")
    turn2 = create_turn(conv.id, 2, "patient", "I am worried about my scans.")

    # Add annotations
    span1 = create_span_annotation(turn1.id, "Hello", 0, 5, "setting", span_id="s_1")
    span2 = create_span_annotation(turn2.id, "worried", 5, 12, "explicit_feeling", span_id="s_2")
    rel1 = create_relation(turn1.id, "s_1", "s_2", "elicitation_of", relation_id="r_1")
    set_spikes_annotation(turn1.id, "setting")

    # Fetch full conversation in single batch
    full_conv = get_full_conversation_with_annotations(conv.id)
    assert full_conv is not None
    assert len(full_conv.turns) == 2

    # Check turn 1 annotations
    t1 = full_conv.turns[0]
    assert t1.turn_number == 1
    assert t1.annotations["spikes_stage"] == "setting"
    assert len(t1.annotations["spans"]) == 1
    assert t1.annotations["spans"][0]["span_id"] == "s_1"
    assert len(t1.annotations["relations"]) == 1

    # Check turn 2 annotations
    t2 = full_conv.turns[1]
    assert t2.turn_number == 2
    assert len(t2.annotations["spans"]) == 1
    assert t2.annotations["spans"][0]["span_id"] == "s_2"


def test_sync_session_annotations_deletes_zombie_spans():
    """Test that sync_session_annotations deletes spans removed in session (no zombies!)."""
    conv = create_conversation(external_id="conv_sync_001")
    t1 = create_turn(conv.id, 1, "patient", "I am very scared and frightened.")

    # Initially insert 2 spans
    create_span_annotation(t1.id, "scared", 10, 16, "explicit_feeling", span_id="span_scared")
    create_span_annotation(t1.id, "frightened", 21, 31, "explicit_feeling", span_id="span_frightened")

    # Verify both exist initially
    initial_spans = get_span_annotations(t1.id)
    assert len(initial_spans) == 2

    # User deleted 'span_scared' in UI; session now only has 'span_frightened'
    session_annotations = {
        1: {
            "spans": [
                {
                    "span_id": "span_frightened",
                    "text": "frightened",
                    "start": 21,
                    "end": 31,
                    "label": "explicit_feeling",
                }
            ],
            "relations": [],
            "spikes_stage": None,
        }
    }

    # Execute Diff & Sync
    success = sync_session_annotations(conv.id, session_annotations)
    assert success is True

    # Verify 'span_scared' was truly deleted from database!
    updated_spans = get_span_annotations(t1.id)
    assert len(updated_spans) == 1
    assert updated_spans[0].span_id == "span_frightened"


def test_ai_suggestions_tracking():
    """Test creating, querying, and updating human review status on AI suggestions."""
    conv = create_conversation(external_id="conv_ai_001")
    t1 = create_turn(conv.id, 1, "patient", "I just feel numb.")

    sug = create_ai_suggestion(
        turn_id=t1.id,
        span_id="ai_span_01",
        text="feel numb",
        start_pos=7,
        end_pos=16,
        suggested_label="explicit_feeling",
        confidence=0.94,
        reasoning="Direct expression of emotional numbness",
        agent_type="react",
        model="gpt-4o",
    )
    assert sug.id is not None
    assert sug.status == "pending"

    # Query suggestions
    pending_list = get_ai_suggestions_for_turn(t1.id, status="pending")
    assert len(pending_list) == 1
    assert pending_list[0].span_id == "ai_span_01"

    # Expert reviews and accepts suggestion
    expert = create_expert("Reviewer 1")
    ok = update_ai_suggestion_status(t1.id, "ai_span_01", "accepted", expert_id=expert.id)
    assert ok is True

    # Verify status changed to accepted
    all_sugs = get_ai_suggestions_for_turn(t1.id)
    assert len(all_sugs) == 1
    assert all_sugs[0].status == "accepted"
    assert all_sugs[0].expert_id == expert.id


def test_spikes_annotation_null_expert_upsert_no_duplicate_rows():
    """Regression Test: Setting spikes stage with expert_id=None must not create duplicate zombie rows."""
    conv = create_conversation(external_id="conv_spikes_null_001")
    t1 = create_turn(conv.id, 1, "clinician", "Let's review your scan results.")

    # 1. Set stage to 'setting'
    s1 = set_spikes_annotation(t1.id, "setting", expert_id=None)
    assert s1 is not None
    assert s1.stage == "setting"

    # 2. Update stage to 'knowledge' (with expert_id=None)
    s2 = set_spikes_annotation(t1.id, "knowledge", expert_id=None)
    assert s2 is not None
    assert s2.stage == "knowledge"

    # Query DB directly to count rows
    with db_conn.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM spikes_annotations WHERE turn_id = ?", (t1.id,))
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["stage"] == "knowledge"

    # 3. Clear stage with None
    set_spikes_annotation(t1.id, None, expert_id=None)
    with db_conn.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM spikes_annotations WHERE turn_id = ?", (t1.id,))
        rows = cursor.fetchall()
        assert len(rows) == 0


def test_database_transaction_rollback_on_failure():
    """Verify that exceptions raised inside get_db() trigger rollback and release the connection."""
    conv = create_conversation(external_id="conv_rollback_001")
    turn = create_turn(conv.id, 1, "patient", "Initial text")

    try:
        with db_conn.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO span_annotations (turn_id, span_id, text, start_pos, end_pos, label) VALUES (?, ?, ?, ?, ?, ?)",
                (turn.id, "s_temp", "text", 0, 4, "explicit_feeling"),
            )
            # Intentionally raise an exception mid-transaction
            raise RuntimeError("Simulated mid-transaction failure")
    except RuntimeError:
        pass

    # Verify that 's_temp' was rolled back and is not in the database
    spans = get_span_annotations(turn.id)
    assert len(spans) == 0


def test_duplicate_conversation_external_id_rejected():
    """Verify that creating two conversations with the same external_id raises IntegrityError."""
    create_conversation(external_id="unique_conv_001")
    with pytest.raises(sqlite3.IntegrityError):
        create_conversation(external_id="unique_conv_001")


def test_cascade_delete_conversation_removes_turns_and_spans():
    """Verify that deleting a conversation cascades to turns and spans via foreign keys."""
    conv = create_conversation(external_id="conv_cascade_001")
    turn = create_turn(conv.id, 1, "patient", "My test turn")
    create_span_annotation(turn.id, "test", 3, 7, "explicit_feeling", span_id="span_casc")

    with db_conn.get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conv.id,))

    # Spans and turns must be gone
    turns = get_turns_by_conversation(conv.id)
    assert len(turns) == 0
    spans = get_span_annotations(turn.id)
    assert len(spans) == 0


def test_get_full_conversation_with_missing_and_empty_conversation():
    """Verify get_full_conversation_with_annotations returns None for missing and empty turns list for empty."""
    # Nonexistent conversation
    assert get_full_conversation_with_annotations(999999) is None

    # Empty conversation (no turns)
    conv = create_conversation(external_id="conv_empty_001")
    full_conv = get_full_conversation_with_annotations(conv.id)
    assert full_conv is not None
    assert full_conv.turns == []

