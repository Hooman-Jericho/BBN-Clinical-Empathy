"""AI agent integration services with database tracking."""

import os
import streamlit as st
from config import USE_DATABASE


def get_agent_runner():
    """Get configured agent runner based on session state."""
    try:
        from agents.runner import AnnotationRunner

        config = st.session_state.agent_config

        # Determine API key based on provider
        api_key = config.get("api_key", "")
        if not api_key:
            if config["provider"] == "openai":
                api_key = os.getenv("OPENAI_API_KEY", "")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY", "")

        runner = AnnotationRunner(
            agent_type=config["agent_type"],
            provider=config["provider"],
            model=config["model"],
            api_key=api_key,
        )
        return runner
    except ImportError as e:
        st.error(f"Failed to import agents: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to create agent runner: {e}")
        return None


def run_agent_on_turn(turn, context=""):
    """Run the agent on a single turn and record suggestions into database."""
    runner = get_agent_runner()
    if runner is None:
        return None

    try:
        result = runner.annotate_turn(turn, context)
        return result
    except Exception as e:
        st.error(f"Agent error: {e}")
        return None


def agent_result_to_suggestions(agent_result, turn_id: int):
    """Convert agent result to suggestion list and persist to SQLite if enabled."""
    suggestions = []

    if agent_result is None:
        return suggestions

    config = st.session_state.get("agent_config", {})
    agent_type = config.get("agent_type", "react")
    model = config.get("model", "")

    # Extract spans from result
    spans = []
    if hasattr(agent_result, "spans"):
        spans = agent_result.spans
    elif hasattr(agent_result, "turns"):
        for t in agent_result.turns:
            if t.turn_id == turn_id:
                spans = t.spans
                break

    for span in spans:
        span_id = getattr(span, "span_id", "")
        text = getattr(span, "text", "")
        start = getattr(span, "start", 0)
        end = getattr(span, "end", len(text))
        label = getattr(span, "label", "")
        reasoning = getattr(span, "reasoning", "")
        confidence = getattr(span, "confidence", None)

        item = {
            "turn_id": turn_id,
            "span_id": span_id,
            "text": text,
            "start": start,
            "end": end,
            "suggested_label": label,
            "reasoning": reasoning,
            "confidence": confidence,
        }
        suggestions.append(item)

        # Persist to database for performance tracking
        if USE_DATABASE:
            try:
                from database import create_ai_suggestion, get_conversation_by_external_id, get_turns_by_conversation
                conv_id = st.session_state.get("last_conv_id")
                if conv_id:
                    db_conv = get_conversation_by_external_id(conv_id)
                    if db_conv:
                        turns = get_turns_by_conversation(db_conv.id)
                        matching_turn = next((t for t in turns if t.turn_number == turn_id), None)
                        if matching_turn:
                            create_ai_suggestion(
                                turn_id=matching_turn.id,
                                span_id=span_id,
                                text=text,
                                start_pos=start,
                                end_pos=end,
                                suggested_label=label,
                                confidence=confidence,
                                reasoning=reasoning,
                                agent_type=agent_type,
                                model=model,
                            )
            except Exception:
                pass

    return suggestions
