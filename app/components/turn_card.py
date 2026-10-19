"""Turn card rendering component with robust, XSS-safe span highlight partitioning."""

import html
import streamlit as st
from utils import get_label_color, format_label_name


def build_highlighted_text(text: str, spans: list) -> str:
    """Build XSS-safe highlighted text using character partition intervals.
    
    This interval-slicing algorithm completely prevents HTML tag breakage,
    overlapping span corruption, and index shifts.
    """
    if not text:
        return ""

    if not spans:
        return html.escape(text)

    # Validate and filter spans
    text_len = len(text)
    valid_spans = []
    for s in spans:
        start = max(0, min(s.get("start", 0), text_len))
        end = max(0, min(s.get("end", text_len), text_len))
        if start < end:
            valid_spans.append({
                "start": start,
                "end": end,
                "label": s.get("label", ""),
                "span_id": s.get("span_id", ""),
            })

    if not valid_spans:
        return html.escape(text)

    # Collect unique boundary points
    cut_points = {0, text_len}
    for s in valid_spans:
        cut_points.add(s["start"])
        cut_points.add(s["end"])

    sorted_cuts = sorted(cut_points)
    output_parts = []

    # Process each non-overlapping slice
    for i in range(len(sorted_cuts) - 1):
        seg_start = sorted_cuts[i]
        seg_end = sorted_cuts[i + 1]
        seg_text = text[seg_start:seg_end]
        escaped_seg = html.escape(seg_text)

        # Find which spans cover this exact segment
        matching_spans = [
            s for s in valid_spans
            if s["start"] <= seg_start and s["end"] >= seg_end
        ]

        if matching_spans:
            # Primary span determines the background color and label title
            primary = matching_spans[0]
            color = get_label_color(primary["label"])
            label_title = format_label_name(primary["label"])
            output_parts.append(
                f'<span class="highlight-span" style="background-color: {color};" title="{html.escape(label_title)}">{escaped_seg}</span>'
            )
        else:
            output_parts.append(escaped_seg)

    return "".join(output_parts)


def render_turn_card(turn: dict, schema: dict, show_annotations: bool = True) -> str:
    """Render a conversation turn card. Returns safe HTML string."""
    text = turn.get("text", "")
    speaker = turn.get("speaker", "patient")
    turn_id = turn.get("turn_id", 0)

    # Source of truth: Session state annotations
    session_data = st.session_state.current_annotations.get(turn_id)
    if session_data is not None:
        spans = session_data.get("spans", [])
        spikes_stage = session_data.get("spikes_stage")
    else:
        # Fallback to turn original annotations on first load
        orig = turn.get("annotations", {})
        spans = orig.get("spans", [])
        spikes_stage = orig.get("spikes_stage")

    highlighted_text = build_highlighted_text(text, spans if show_annotations else [])

    is_patient = str(speaker).lower() == "patient"
    turn_class = "patient-turn" if is_patient else "clinician-turn"
    speaker_icon = "🧑‍🦱" if is_patient else "👨‍⚕️"
    speaker_label = "Patient" if is_patient else "Clinician"
    turn_id_display = html.escape(str(turn_id))

    html_card = f"""
    <div class="{turn_class}">
        <div class="speaker-label">{speaker_icon} {speaker_label} · Turn {turn_id_display}</div>
    """

    if spikes_stage and not is_patient:
        spikes_color = html.escape(get_label_color(spikes_stage))
        html_card += f'<span class="annotation-badge" style="background-color: {spikes_color};">SPIKES: {html.escape(str(spikes_stage).upper())}</span><br><br>'

    html_card += f'<div class="turn-text">{highlighted_text}</div>'

    # Show annotation badges
    if show_annotations and spans:
        html_card += '<div class="annotation-section">'
        for span in spans:
            label = span.get("label", "")
            color = html.escape(get_label_color(label))
            span_text = span.get("text", "")
            badge_title = f"{format_label_name(label)}: '{span_text}'"
            html_card += f'<span class="annotation-badge" style="background-color: {color}; color: #1a1a2e;" title="{html.escape(badge_title)}">{html.escape(format_label_name(label))}</span> '
        html_card += "</div>"

    html_card += "</div>"
    return html_card
