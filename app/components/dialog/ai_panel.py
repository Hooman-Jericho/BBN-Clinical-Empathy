"""AI suggestions panel component with human-in-the-loop review persistence."""

import os
import html
import streamlit as st
from utils import get_label_color, format_label_name
from services import run_agent_on_turn, agent_result_to_suggestions, add_span_annotation
from config import USE_DATABASE


def render_ai_panel(turn: dict, turn_id: int, conversation: dict):
    """Render the AI suggestions panel with review tracking."""
    st.markdown("#### AI Suggestions")

    # Build context from previous turns
    context_parts = []
    for t in conversation.get("turns", []):
        if t["turn_id"] < turn["turn_id"]:
            context_parts.append(f"{t['speaker'].upper()}: {t['text'][:120]}")
    context = "\n".join(context_parts[-5:])

    # Check API key
    config = st.session_state.get("agent_config", {})
    provider = config.get("provider", "openai")
    api_key_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    has_api_key = bool(config.get("api_key")) or bool(os.getenv(api_key_env))

    if not has_api_key:
        st.warning(f"Please set your {provider.upper()} API key in the sidebar or environment.")
    else:
        st.caption(f"Engine: **{config.get('agent_type', 'react').upper()}** · Model: **{config.get('model', '')}**")

        if st.button("Generate Suggestions", key=f"modal_ai_{turn_id}", type="secondary"):
            with st.spinner("Analyzing clinical empathy markers..."):
                result = run_agent_on_turn(turn, context)
                if result:
                    new_suggestions = agent_result_to_suggestions(result, turn_id)
                    # Merge with existing suggestions
                    existing_tuples = {(s["turn_id"], s["text"], s["suggested_label"]) for s in st.session_state.ai_suggestions}
                    for ns in new_suggestions:
                        if (ns["turn_id"], ns["text"], ns["suggested_label"]) not in existing_tuples:
                            st.session_state.ai_suggestions.append(ns)
                    st.rerun()

    # Display suggestions for this turn
    suggestions_for_turn = [
        s for s in st.session_state.get("ai_suggestions", []) if s.get("turn_id") == turn_id
    ]

    if suggestions_for_turn:
        expert_id = st.session_state.get("current_expert", {}).get("id")

        for i, suggestion in enumerate(suggestions_for_turn):
            color = get_label_color(suggestion["suggested_label"])
            display_text = suggestion.get("text", "")
            if len(display_text) > 60:
                display_text = display_text[:60] + "..."

            reasoning = suggestion.get("reasoning", "")
            confidence = suggestion.get("confidence")
            conf_str = f" · Conf: {confidence:.0%}" if confidence is not None else ""

            st.markdown(
                f"""<div style="background: linear-gradient(135deg, #fef3c7 0%, #fee2e2 100%); padding: 12px; border-radius: 8px; margin: 8px 0; border: 1px solid #fcd34d; color: #1e293b;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color: #0f172a;">{html.escape(format_label_name(suggestion['suggested_label']))}</strong>
                    <span style="font-size:0.8em; color:#64748b;">{conf_str}</span>
                </div>
                <div style="margin-top:4px; font-style:italic;">"{html.escape(display_text)}"</div>
                {f'<div style="margin-top:6px; font-size:0.82em; color:#475569;">💡 {html.escape(reasoning)}</div>' if reasoning else ''}
                </div>""",
                unsafe_allow_html=True,
            )

            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                if st.button("Accept", key=f"modal_accept_{turn_id}_{i}", type="primary"):
                    result = add_span_annotation(
                        turn_id=suggestion["turn_id"],
                        text=suggestion["text"],
                        start=suggestion.get("start", 0),
                        end=suggestion.get("end", len(suggestion["text"])),
                        label=suggestion["suggested_label"],
                        confidence=confidence,
                        source="ai_accepted",
                    )
                    if result:
                        if USE_DATABASE:
                            try:
                                from database import update_ai_suggestion_status
                                update_ai_suggestion_status(turn_id, suggestion.get("span_id", ""), "accepted", expert_id)
                            except Exception:
                                pass

                        st.session_state.ai_suggestions = [
                            s for s in st.session_state.ai_suggestions
                            if not (s["turn_id"] == suggestion["turn_id"] and s["text"] == suggestion["text"] and s["suggested_label"] == suggestion["suggested_label"])
                        ]
                        st.rerun()
                    else:
                        st.warning("This annotation already exists!")

            with c2:
                if st.button("Reject", key=f"modal_reject_{turn_id}_{i}"):
                    if USE_DATABASE:
                        try:
                            from database import update_ai_suggestion_status
                            update_ai_suggestion_status(turn_id, suggestion.get("span_id", ""), "rejected", expert_id)
                        except Exception:
                            pass

                    st.session_state.ai_suggestions = [
                        s for s in st.session_state.ai_suggestions
                        if not (s["turn_id"] == suggestion["turn_id"] and s["text"] == suggestion["text"] and s["suggested_label"] == suggestion["suggested_label"])
                    ]
                    st.rerun()
    else:
        st.caption("No suggestions generated for this turn yet. Click 'Generate Suggestions' above.")
