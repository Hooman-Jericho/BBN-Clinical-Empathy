"""Main annotation dialog component."""

import streamlit as st
from state import close_dialog
from components.dialog.manual_panel import render_manual_panel
from components.dialog.ai_panel import render_ai_panel
from components.dialog.annotations_panel import render_annotations_panel
from components.dialog.relations_panel import render_relations_panel


@st.dialog("Annotate Turn", width="large")
def annotation_dialog(turn, schema, conversation):
    """Modal dialog for annotating a single turn."""
    turn_id = turn["turn_id"]
    speaker = turn["speaker"]
    text = turn["text"]

    # Header with speaker info
    _render_header(speaker, turn_id)

    # Display the turn text
    st.markdown(
        f'<div class="selectable-text">{text}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Select text above and paste it below to annotate")

    # Two columns: Manual | AI Suggestions
    col1, col2 = st.columns([1, 1])

    with col1:
        render_manual_panel(turn, turn_id, speaker, text)

    with col2:
        render_ai_panel(turn, turn_id, conversation)

    # Current annotations section
    st.markdown("---")
    render_annotations_panel(turn_id)

    # Relations section
    st.markdown("---")
    turn_annotations = st.session_state.current_annotations.get(
        turn_id, {"spans": [], "relations": []}
    )
    render_relations_panel(turn_id, conversation, turn_annotations)

    # Close button
    st.markdown("---")
    if st.button("Done", type="primary", use_container_width=True):
        close_dialog()
        st.rerun()


def _render_header(speaker, turn_id):
    """Render the dialog header."""
    speaker_icon = "🧑‍🦱" if speaker == "patient" else "👨‍⚕️"
    speaker_color = "#8b5cf6" if speaker == "patient" else "#22c55e"
    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">'
        f'<span style="font-size: 1.5em;">{speaker_icon}</span>'
        f'<span style="font-size: 1.2em; font-weight: 600; color: {speaker_color};">{speaker.title()}</span>'
        f'<span style="color: #64748b; font-size: 0.9em;">Turn {turn_id}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
