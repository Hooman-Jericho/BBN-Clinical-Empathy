"""Current annotations panel component."""

import streamlit as st
from utils import get_label_color, format_label_name
from services import remove_span_annotation
from state import undo_last_action


def render_annotations_panel(turn_id):
    """Render the current annotations panel."""
    # Header with undo button
    header_col1, _, header_col3 = st.columns([3, 2, 1])
    with header_col1:
        st.markdown("#### Current Annotations")
    with header_col3:
        if st.session_state.undo_history:
            if st.button("Undo", key=f"modal_undo_{turn_id}"):
                undo_last_action()
                st.rerun()

    turn_annotations = st.session_state.current_annotations.get(
        turn_id, {"spans": [], "relations": []}
    )

    if turn_annotations.get("spans"):
        for span in turn_annotations["spans"]:
            col_text, col_label, col_del = st.columns([3, 2, 1])
            with col_text:
                color = get_label_color(span["label"])
                display_text = span["text"][:40] + ("..." if len(span["text"]) > 40 else "")
                st.markdown(
                    f'<span style="background-color: {color}; color: #1e293b; padding: 2px 8px; border-radius: 4px;">"{display_text}"</span>',
                    unsafe_allow_html=True,
                )
            with col_label:
                st.caption(format_label_name(span["label"]))
            with col_del:
                if st.button("Del", key=f"modal_del_{span['span_id']}"):
                    remove_span_annotation(turn_id, span["span_id"])
                    st.rerun()
    else:
        st.caption("No annotations for this turn yet.")
