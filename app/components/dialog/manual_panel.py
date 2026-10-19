"""Manual annotation panel component with exact multi-occurrence disambiguation."""

import streamlit as st
from utils import get_label_color, format_label_name
from utils.colors import get_labels_for_speaker, get_all_labels_flat
from services import add_span_annotation, set_spikes_stage
from config import SPIKES_STAGES


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
    """Render the manual annotation panel with multi-occurrence disambiguation."""
    st.markdown("#### Manual Annotation")

    # Text selection input
    selected_text = st.text_input(
        "Text to annotate:",
        key=f"modal_text_{turn_id}",
        placeholder="Paste or type selected text here...",
    ).strip()

    if selected_text:
        occurrences = find_all_occurrences(text, selected_text)

        if not occurrences:
            st.error("Text not found in this turn. Please check spelling or select directly.")
        else:
            # Multi-occurrence disambiguation
            if len(occurrences) == 1:
                start_idx, end_idx = occurrences[0]
                st.success(f"Matched 1 occurrence at character index [{start_idx}:{end_idx}]")
            else:
                st.info(f"Found {len(occurrences)} occurrences of '{selected_text}'. Select which one to annotate:")
                occ_options = {}
                for i, (s_idx, e_idx) in enumerate(occurrences):
                    before = text[max(0, s_idx - 20):s_idx]
                    after = text[e_idx:min(len(text), e_idx + 20)]
                    label_desc = f"Occurrence {i + 1} (pos {s_idx}-{e_idx}): ...{before}[{selected_text}]{after}..."
                    occ_options[label_desc] = (s_idx, e_idx)

                chosen_occ = st.selectbox(
                    "Select occurrence:",
                    options=list(occ_options.keys()),
                    key=f"modal_occ_{turn_id}",
                )
                start_idx, end_idx = occ_options[chosen_occ]

            # Label selection
            st.markdown("**Choose label:**")
            all_labels = get_all_labels_flat(speaker)

            selected_label = st.selectbox(
                "Label:",
                options=all_labels,
                format_func=format_label_name,
                key=f"modal_label_{turn_id}",
            )

            # Show color preview
            color = get_label_color(selected_label)
            st.markdown(
                f'<span style="background-color: {color}; color: #1a1a2e; padding: 4px 12px; border-radius: 20px; font-weight: 600;">{format_label_name(selected_label)}</span>',
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
                    st.warning("This exact span and label already exists on this turn!")

    # SPIKES stage for clinician turns
    if speaker == "clinician":
        st.markdown("---")
        st.markdown("**SPIKES Stage:**")
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
