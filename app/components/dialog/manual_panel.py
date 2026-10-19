"""Interactive Selection-based manual annotation panel with Word Ribbon, JS Selection bridge, and Quick Labels."""

import html
import re
import streamlit as st
import streamlit.components.v1 as components
from utils import get_label_color, format_label_name
from utils.colors import get_labels_for_speaker, get_all_labels_flat
from services import add_span_annotation, set_spikes_stage
from config import SPIKES_STAGES


def tokenize_with_offsets(text: str) -> list[dict]:
    """Tokenize text into words with exact character start and end offsets."""
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append({
            "word": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def find_all_occurrences(text: str, query: str) -> list[tuple[int, int]]:
    """Find all (start, end) occurrence indices of query inside text."""
    if not query:
        return []
    occurrences = []
    start = 0
    while True:
        idx = text.find(query, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(query)))
        start = idx + 1
    return occurrences


def render_manual_panel(turn: dict, turn_id: int, speaker: str, text: str):
    """Render the modern selection-based annotation panel."""
    st.markdown("#### Interactive Text Annotation")

    # Mode toggle: Interactive Word Picker (Direct Click) vs Custom Range
    mode = st.radio(
        "Selection Mode:",
        ["Direct Word Selection (No Copy-Paste)", "Custom Character Offset"],
        horizontal=True,
        key=f"mode_select_{turn_id}",
    )

    tokens = tokenize_with_offsets(text)
    start_idx = 0
    end_idx = 0
    selected_text = ""

    if mode == "Direct Word Selection (No Copy-Paste)":
        if tokens:
            st.caption("Select start and end words from the turn below to instantly span text:")
            col_start, col_end = st.columns(2)

            token_display_options = {
                f"{i + 1}. {t['word']} [{t['start']}:{t['end']}]": i
                for i, t in enumerate(tokens)
            }
            token_keys = list(token_display_options.keys())

            with col_start:
                chosen_start_label = st.selectbox(
                    "Start Word:",
                    options=token_keys,
                    index=0,
                    key=f"word_start_{turn_id}",
                )
                start_token_idx = token_display_options[chosen_start_label]

            with col_end:
                # Default end word to start word
                chosen_end_label = st.selectbox(
                    "End Word:",
                    options=token_keys,
                    index=min(start_token_idx, len(token_keys) - 1),
                    key=f"word_end_{turn_id}",
                )
                end_token_idx = token_display_options[chosen_end_label]

            # Ensure start <= end
            actual_start_idx = min(start_token_idx, end_token_idx)
            actual_end_idx = max(start_token_idx, end_token_idx)

            start_idx = tokens[actual_start_idx]["start"]
            end_idx = tokens[actual_end_idx]["end"]
            selected_text = text[start_idx:end_idx]

            # Live preview box
            st.markdown(
                f"""<div style="background:#f1f5f9; border-left:4px solid #3b82f6; padding:8px 12px; border-radius:4px; margin:8px 0; font-size:0.95em;">
                <strong>Selected Span [{start_idx}:{end_idx}]:</strong> <span style="background:#fef08a; padding:2px 6px; border-radius:3px;">{html.escape(selected_text)}</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        # Custom Character Offset / Text Input
        input_text = st.text_input(
            "Text to annotate:",
            key=f"modal_text_{turn_id}",
            placeholder="Type or paste exact substring...",
        ).strip()

        if input_text:
            occurrences = find_all_occurrences(text, input_text)
            if occurrences:
                if len(occurrences) == 1:
                    start_idx, end_idx = occurrences[0]
                    selected_text = input_text
                else:
                    occ_options = {
                        f"Occurrence {i + 1} [{s}:{e}]": (s, e)
                        for i, (s, e) in enumerate(occurrences)
                    }
                    chosen_occ = st.selectbox(
                        "Multiple matches found. Choose occurrence:",
                        options=list(occ_options.keys()),
                        key=f"modal_occ_{turn_id}",
                    )
                    start_idx, end_idx = occ_options[chosen_occ]
                    selected_text = input_text
            else:
                st.error("Text not found in this turn.")

    # Label selection & One-Click Quick Action Chips
    if selected_text:
        all_labels = get_all_labels_flat(speaker)
        st.markdown("**Choose Label:**")

        # Top 4 quick-action chips for fast 1-click annotation
        quick_labels = all_labels[:4]
        quick_cols = st.columns(len(quick_labels))
        for col, q_label in zip(quick_cols, quick_labels):
            with col:
                btn_name = format_label_name(q_label)
                if len(btn_name) > 18:
                    btn_name = btn_name[:16] + "..."
                if st.button(btn_name, key=f"quick_{q_label}_{turn_id}", use_container_width=True):
                    res = add_span_annotation(
                        turn_id=turn_id,
                        text=selected_text,
                        start=start_idx,
                        end=end_idx,
                        label=q_label,
                        source="manual",
                    )
                    if res:
                        st.success(f"Added {format_label_name(q_label)}!")
                        st.rerun()
                    else:
                        st.warning("Annotation already exists!")

        # Full dropdown for all clinical appraisal categories
        selected_label = st.selectbox(
            "Or select from all appraisal categories:",
            options=all_labels,
            format_func=format_label_name,
            key=f"modal_label_{turn_id}",
        )

        color = get_label_color(selected_label)
        st.markdown(
            f'<span style="background-color: {color}; color: #1a1a2e; padding: 4px 12px; border-radius: 20px; font-weight: 600;">{html.escape(format_label_name(selected_label))}</span>',
            unsafe_allow_html=True,
        )

        if st.button("Add Annotation", type="primary", key=f"modal_add_{turn_id}"):
            result = add_span_annotation(
                turn_id=turn_id,
                text=selected_text,
                start=start_idx,
                end=end_idx,
                label=selected_label,
                source="manual",
            )
            if result:
                st.success("Annotation added!")
                st.rerun()
            else:
                st.warning("This exact span and label already exists!")

    # SPIKES stage for clinician turns
    if speaker == "clinician":
        st.markdown("---")
        st.markdown("**SPIKES Protocol Stage:**")
        spikes_options = ["None"] + SPIKES_STAGES
        current_spikes = st.session_state.current_annotations.get(turn_id, {}).get("spikes_stage")

        selected_spikes = st.selectbox(
            "Stage:",
            spikes_options,
            index=spikes_options.index(current_spikes) if current_spikes in spikes_options else 0,
            key=f"modal_spikes_{turn_id}",
        )

        if st.button("Update SPIKES Stage", key=f"modal_spikes_btn_{turn_id}"):
            stage_to_set = None if selected_spikes == "None" else selected_spikes
            set_spikes_stage(turn_id, stage_to_set)
            st.success("SPIKES stage updated!")
            st.rerun()
