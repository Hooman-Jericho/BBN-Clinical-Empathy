"""BBN Annotation Tool - Main Application Entry Point."""

import sys
from pathlib import Path

# Add app directory and parent directory to path for imports
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import STYLES_PATH, USE_DATABASE
from state import init_session_state
from services.data_service import load_schema
from components import render_sidebar, render_turn_card
from components.dialog import annotation_dialog

# Initialize database if enabled
if USE_DATABASE:
    from database import init_db
    init_db()


def load_styles():
    """Load CSS styles from external file."""
    if STYLES_PATH.exists():
        with open(STYLES_PATH, "r") as f:
            return f.read()
    return ""


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="BBN Annotation Tool",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Load external CSS
    st.markdown(f"<style>{load_styles()}</style>", unsafe_allow_html=True)


def render_conversation_header(current_conv):
    """Render conversation metadata and stats."""
    metadata = current_conv.get("metadata", {})
    st.info(
        f"**Scenario:** {metadata.get('scenario', 'N/A')} | "
        f"**Language:** {metadata.get('language', 'N/A')} | "
        f"**Date:** {metadata.get('date', 'N/A')}"
    )

    # Annotation stats
    total_spans = sum(
        len(st.session_state.current_annotations.get(t["turn_id"], {}).get("spans", []))
        for t in current_conv.get("turns", [])
    )
    st.caption(f"Total annotations: {total_spans}")


def render_conversation_turns(current_conv, schema, show_annotations):
    """Render all conversation turns with annotate buttons."""
    for turn in current_conv.get("turns", []):
        # Render the turn card
        html = render_turn_card(turn, schema, show_annotations)
        st.markdown(html, unsafe_allow_html=True)

        # Annotate button and relation count
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("Annotate", key=f"annotate_btn_{turn['turn_id']}", use_container_width=True):
                st.session_state.open_dialog_turn_id = turn["turn_id"]
                st.rerun()
        with col2:
            turn_annotations = st.session_state.current_annotations.get(
                turn["turn_id"], {"relations": []}
            )
            relations = turn_annotations.get("relations", [])
            if relations:
                st.markdown(
                    f'<div class="relations-badge">{len(relations)} relation{"s" if len(relations) > 1 else ""}</div>',
                    unsafe_allow_html=True,
                )


def main():
    """Main application entry point."""
    setup_page()
    init_session_state()

    # Load schema
    schema = load_schema()

    # Render sidebar and get current conversation
    current_conv = render_sidebar(schema)

    # Main content area
    title_col, toggle_col = st.columns([4, 1])
    with title_col:
        st.title("Conversation Viewer")
    with toggle_col:
        st.markdown("<div style='height: 42px'></div>", unsafe_allow_html=True)
        show_annotations = st.toggle("Show Annotations", value=True)

    if current_conv:
        # Handle dialog opening
        if st.session_state.open_dialog_turn_id is not None:
            open_turn_id = st.session_state.open_dialog_turn_id
            open_turn = next(
                (t for t in current_conv.get("turns", []) if t["turn_id"] == open_turn_id),
                None,
            )
            if open_turn:
                annotation_dialog(open_turn, schema, current_conv)

        # Render conversation content
        render_conversation_header(current_conv)
        st.markdown("---")
        render_conversation_turns(current_conv, schema, show_annotations)
    else:
        st.warning("No conversation selected or available.")


if __name__ == "__main__":
    main()
