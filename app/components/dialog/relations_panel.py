"""Relations panel component with full CRUD and deletion support."""

import streamlit as st
from utils import format_label_name
from services import add_relation, remove_relation

# Available relation types from schema
RELATION_TYPES = ["response_to", "elicitation_of"]


def render_relations_panel(turn_id: int, conversation: dict, turn_annotations: dict):
    """Render the relations panel for linking and managing cross-turn annotations."""
    st.markdown("#### Link Cross-Turn Relations")

    # Collect all patient and clinician spans across the entire conversation
    all_patient_spans = []
    all_clinician_spans = []

    for t in conversation.get("turns", []):
        t_id = t["turn_id"]
        t_speaker = t["speaker"]
        t_annotations = st.session_state.current_annotations.get(t_id, {"spans": []})
        for span in t_annotations.get("spans", []):
            span_text = span.get("text", "")
            truncated = (span_text[:30] + "...") if len(span_text) > 30 else span_text
            span_info = {
                "turn_id": t_id,
                "span_id": span["span_id"],
                "text": span_text,
                "label": span["label"],
                "display": f"T{t_id} ({t_speaker[:3].upper()}): \"{truncated}\" [{format_label_name(span['label'])}]",
            }
            if t_speaker == "patient":
                all_patient_spans.append(span_info)
            else:
                all_clinician_spans.append(span_info)

    if all_patient_spans and all_clinician_spans:
        rel_col1, rel_col2, rel_col3 = st.columns(3)
        with rel_col1:
            clinician_options = ["Select Clinician span..."] + [s["display"] for s in all_clinician_spans]
            selected_clinician = st.selectbox(
                "Clinician Span (Source):", clinician_options, key=f"rel_clinician_{turn_id}"
            )
        with rel_col2:
            rel_type_options = RELATION_TYPES
            selected_rel_type = st.selectbox(
                "Relation Type:",
                rel_type_options,
                format_func=lambda x: x.replace("_", " ").title(),
                key=f"rel_type_{turn_id}",
            )
        with rel_col3:
            patient_options = ["Select Patient EO..."] + [s["display"] for s in all_patient_spans]
            selected_patient = st.selectbox(
                "Patient EO (Target):", patient_options, key=f"rel_patient_{turn_id}"
            )

        if selected_clinician != "Select Clinician span..." and selected_patient != "Select Patient EO...":
            if st.button("Create Relation Link", key=f"create_rel_{turn_id}", type="primary"):
                c_span = next(s for s in all_clinician_spans if s["display"] == selected_clinician)
                p_span = next(s for s in all_patient_spans if s["display"] == selected_patient)

                add_relation(
                    from_turn_id=c_span["turn_id"],
                    from_span_id=c_span["span_id"],
                    to_turn_id=p_span["turn_id"],
                    to_span_id=p_span["span_id"],
                    relation_type=selected_rel_type,
                )
                st.success("Relation linked successfully!")
                st.rerun()
    else:
        st.caption("Add both Patient EOs and Clinician Responses in this conversation to create relations.")

    # Show existing relations touching this turn or spans in this turn
    current_turn_spans = {s["span_id"] for s in turn_annotations.get("spans", [])}
    relations_for_this_turn = []

    # Check relations stored on all turns
    for tid, t_ann in st.session_state.current_annotations.items():
        for r in t_ann.get("relations", []):
            if tid == turn_id or r.get("from") in current_turn_spans or r.get("to") in current_turn_spans:
                if r not in relations_for_this_turn:
                    relations_for_this_turn.append((tid, r))

    if relations_for_this_turn:
        st.markdown("**Existing Relations Involving This Turn:**")
        span_lookup = {s["span_id"]: s["text"] for s in (all_patient_spans + all_clinician_spans)}

        for host_turn_id, rel in relations_for_this_turn:
            rel_id = rel.get("relation_id", "")
            from_text = span_lookup.get(rel["from"], rel["from"])
            to_text = span_lookup.get(rel["to"], rel["to"])
            from_trunc = (from_text[:25] + "...") if len(from_text) > 25 else from_text
            to_trunc = (to_text[:25] + "...") if len(to_text) > 25 else to_text

            col_desc, col_del = st.columns([5, 1])
            with col_desc:
                st.markdown(
                    f'<div class="relation-item">'
                    f'<span style="color:#16a34a; font-weight:600;">Clinician: "{from_trunc}"</span>'
                    f'<span class="rel-arrow"> ➔ </span>'
                    f'<span class="rel-type">[{rel["type"].replace("_", " ")}]</span>'
                    f'<span class="rel-arrow"> ➔ </span>'
                    f'<span style="color:#7c3aed; font-weight:600;">Patient: "{to_trunc}"</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("Delete", key=f"del_rel_btn_{rel_id}"):
                    remove_relation(host_turn_id, rel_id)
                    st.success("Relation removed!")
                    st.rerun()
    else:
        st.caption("No relations currently linked to this turn.")
