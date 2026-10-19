"""Session state management for BBN Annotation Tool."""

import os
import copy
import streamlit as st
from config import MAX_UNDO_HISTORY, DEFAULT_AGENT_CONFIG


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "current_annotations": {},
        "selected_turn": None,
        "selected_text": "",
        "pending_spans": [],
        "pending_relations": [],
        "ai_suggestions": [],
        "annotation_mode": "view",
        "open_dialog_turn_id": None,
        "undo_history": [],
        "last_conv_id": None,
        "agent_config": {
            **DEFAULT_AGENT_CONFIG,
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_to_undo_history():
    """Save current annotation state to undo history."""
    if "undo_history" not in st.session_state:
        st.session_state.undo_history = []

    if len(st.session_state.undo_history) >= MAX_UNDO_HISTORY:
        st.session_state.undo_history.pop(0)

    st.session_state.undo_history.append(
        copy.deepcopy(st.session_state.current_annotations)
    )


def undo_last_action() -> bool:
    """Restore the previous annotation state. Returns True if successful."""
    if st.session_state.get("undo_history"):
        st.session_state.current_annotations = st.session_state.undo_history.pop()
        return True
    return False


def update_agent_config(provider: str, model: str, agent_type: str):
    """Update agent configuration in session state with normalized agent_type."""
    normalized_agent_type = "multi_agent" if agent_type in ("multi", "multi_agent") else "react"
    api_key_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    st.session_state.agent_config = {
        "provider": provider,
        "model": model,
        "agent_type": normalized_agent_type,
        "api_key": os.getenv(api_key_env, ""),
    }


def clear_ai_suggestions():
    """Clear all AI suggestions."""
    st.session_state.ai_suggestions = []


def set_open_dialog(turn_id: int):
    """Set which dialog is currently open."""
    st.session_state.open_dialog_turn_id = turn_id


def close_dialog(turn_id=None):
    """Close the current dialog and clear modal-specific keys while preserving undo history."""
    if turn_id is None:
        turn_id = st.session_state.get("open_dialog_turn_id")

    if turn_id is not None:
        keys_to_clear = [
            f"modal_text_{turn_id}",
            f"modal_label_{turn_id}",
            f"modal_spikes_{turn_id}",
            f"modal_occ_{turn_id}",
            f"rel_from_{turn_id}",
            f"rel_to_{turn_id}",
            f"rel_type_{turn_id}",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    st.session_state.open_dialog_turn_id = None


def get_turn_annotations(turn_id: int) -> dict:
    """Get annotations for a specific turn."""
    return st.session_state.current_annotations.get(
        turn_id, {"spans": [], "relations": [], "spikes_stage": None}
    )
