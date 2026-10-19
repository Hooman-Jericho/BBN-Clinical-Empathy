"""Unified Data Service for JSON and SQLite database backends.

Provides transparent persistence, full state synchronization, and schema management.
"""

import copy
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import streamlit as st

from config import USE_DATABASE, SAMPLES_DIR, SCHEMA_PATH


def _use_db() -> bool:
    """Check if database mode is enabled."""
    return USE_DATABASE


# ============== SCHEMA ==============

def load_schema() -> Optional[dict]:
    """Load annotation schema from JSON file."""
    if SCHEMA_PATH.exists():
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading schema: {e}")
    return None


# ============== CONVERSATIONS ==============

def load_conversations() -> List[dict]:
    """Load all conversations (from database or JSON files)."""
    if _use_db():
        return _load_conversations_from_db()
    return _load_conversations_from_json()


def _load_conversations_from_db() -> List[dict]:
    """Load conversations from SQLite database using batch queries."""
    try:
        from database import init_db, get_all_conversations, get_turns_by_conversation

        init_db()
        conversations = []

        for conv in get_all_conversations():
            turns = get_turns_by_conversation(conv.id)
            conv.turns = turns
            conv_dict = conv.to_dict()
            conv_dict["filepath"] = conv.source_file
            conv_dict["filename"] = Path(conv.source_file).name if conv.source_file else f"{conv.external_id}.json"
            conversations.append(conv_dict)

        # Fallback to json if DB is empty
        if not conversations:
            json_convs = _load_conversations_from_json()
            if json_convs:
                from database import import_conversation_from_json
                for jc in json_convs:
                    import_conversation_from_json(jc, source_file=jc.get("filepath"))
                return _load_conversations_from_db()

        return conversations
    except Exception as e:
        st.warning(f"Database error, falling back to JSON: {e}")
        return _load_conversations_from_json()


def _load_conversations_from_json() -> List[dict]:
    """Load conversations from JSON files."""
    conversations = []
    if SAMPLES_DIR.exists():
        for file in sorted(SAMPLES_DIR.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    conv = json.load(f)
                    conv["filename"] = file.name
                    conv["filepath"] = str(file)
                    conversations.append(conv)
            except (json.JSONDecodeError, IOError) as e:
                st.warning(f"Skipping {file.name}: {e}")
    return conversations


def get_conversation_by_id(conv_id: str) -> Optional[dict]:
    """Get a specific conversation by external ID."""
    if _use_db():
        try:
            from database import get_conversation_by_external_id, get_full_conversation_with_annotations

            conv = get_conversation_by_external_id(conv_id)
            if conv:
                full_conv = get_full_conversation_with_annotations(conv.id)
                return full_conv.to_dict() if full_conv else None
        except Exception:
            pass

    for conv in _load_conversations_from_json():
        if conv.get("id") == conv_id:
            return conv
    return None


# ============== SAVE & MERGE ANNOTATIONS ==============

def save_annotations(conversation: dict, annotations: dict, expert_id: int = None) -> bool:
    """Save annotations to database and JSON file with complete state synchronization."""
    db_success = True
    if _use_db():
        db_success = _save_annotations_to_db(conversation, annotations, expert_id)

    json_success = _save_annotations_to_json(conversation, annotations)
    return db_success and json_success


def _save_annotations_to_db(conversation: dict, annotations: dict, expert_id: int = None) -> bool:
    """Synchronize session annotations with SQLite database."""
    try:
        from database import (
            get_conversation_by_external_id,
            sync_session_annotations,
            import_conversation_from_json,
        )

        conv_id = conversation.get("id")
        db_conv = get_conversation_by_external_id(conv_id)

        if not db_conv:
            db_conv = import_conversation_from_json(conversation)

        if db_conv:
            return sync_session_annotations(db_conv.id, annotations, expert_id)
    except Exception as e:
        st.error(f"Database save error: {e}")

    return False


def _save_annotations_to_json(conversation: dict, annotations: dict) -> bool:
    """Save annotations to JSON file ensuring removals and updates are fully reflected."""
    try:
        merged = _merge_annotations(conversation, annotations)
        filepath = conversation.get("filepath")
        if not filepath:
            filepath = SAMPLES_DIR / f"{conversation.get('id', 'conversation')}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        st.error(f"JSON save error: {e}")
        return False


def _merge_annotations(conversation: dict, annotations: dict) -> dict:
    """Synchronize session annotations into conversation dictionary.
    
    Replaces each turn's annotations with the active session state,
    correctly handling span additions, deletions, updates, and SPIKES stage clearing.
    """
    updated_conv = copy.deepcopy(conversation)

    for turn in updated_conv.get("turns", []):
        turn_id = turn.get("turn_id")
        if turn_id in annotations:
            session_turn = annotations[turn_id]
            turn["annotations"] = {
                "spans": [copy.deepcopy(s) for s in session_turn.get("spans", [])],
                "relations": [copy.deepcopy(r) for r in session_turn.get("relations", [])],
                "spikes_stage": session_turn.get("spikes_stage"),
            }

    return updated_conv


merge_annotations_to_conversation = _merge_annotations


# ============== LOAD EXISTING ANNOTATIONS ==============

def load_existing_annotations(conversation: dict, expert_id: int = None) -> dict:
    """Load existing annotations for a conversation."""
    if _use_db():
        annotations = _load_annotations_from_db(conversation, expert_id)
        if annotations:
            return annotations

    return _load_annotations_from_json(conversation)


def _load_annotations_from_db(conversation: dict, expert_id: int = None) -> dict:
    """Load annotations from database."""
    try:
        from database import (
            get_conversation_by_external_id,
            get_full_conversation_with_annotations,
        )

        conv_id = conversation.get("id")
        db_conv = get_conversation_by_external_id(conv_id)
        if not db_conv:
            return {}

        full_conv = get_full_conversation_with_annotations(db_conv.id, expert_id)
        if not full_conv:
            return {}

        annotations = {}
        for turn in full_conv.turns:
            annotations[turn.turn_number] = {
                "spans": [s.copy() for s in turn.annotations.get("spans", [])],
                "relations": [r.copy() for r in turn.annotations.get("relations", [])],
                "spikes_stage": turn.annotations.get("spikes_stage"),
            }
        return annotations
    except Exception as e:
        st.warning(f"Error loading from database: {e}")
        return {}


def _load_annotations_from_json(conversation: dict) -> dict:
    """Load annotations from JSON conversation."""
    annotations = {}
    for turn in conversation.get("turns", []):
        turn_id = turn.get("turn_id")
        turn_annotations = turn.get("annotations", {})
        annotations[turn_id] = {
            "spans": [s.copy() for s in turn_annotations.get("spans", [])],
            "relations": [r.copy() for r in turn_annotations.get("relations", [])],
            "spikes_stage": turn_annotations.get("spikes_stage"),
        }
    return annotations


# ============== EXPERTS ==============

def get_current_expert() -> Optional[dict]:
    """Get current expert from session state or database."""
    if "current_expert" not in st.session_state:
        if _use_db():
            try:
                from database import get_or_create_expert
                expert = get_or_create_expert("Default Expert")
                st.session_state.current_expert = {
                    "id": expert.id,
                    "name": expert.name,
                    "role": expert.role,
                }
            except Exception:
                st.session_state.current_expert = {"id": None, "name": "Default", "role": "annotator"}
        else:
            st.session_state.current_expert = {"id": None, "name": "Default", "role": "annotator"}

    return st.session_state.current_expert


def get_all_experts() -> List[dict]:
    """Get all experts."""
    if _use_db():
        try:
            from database import get_all_experts as db_get_all_experts
            experts = db_get_all_experts()
            if experts:
                return [{"id": e.id, "name": e.name, "role": e.role} for e in experts]
        except Exception:
            pass

    return [{"id": None, "name": "Default", "role": "annotator"}]
