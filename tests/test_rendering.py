"""Unit tests for XSS escaping, HTML interval partitioning, and span highlights."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))
sys.path.insert(0, str(BASE_DIR))

from components.turn_card import build_highlighted_text


def test_build_highlighted_text_escapes_xss():
    """Verify that dangerous HTML characters in clinical text are escaped."""
    raw_text = "Patient said <script>alert(1)</script> and WBC count < 4000."
    html_out = build_highlighted_text(raw_text, [])
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "&lt; 4000" in html_out


def test_build_highlighted_text_single_span():
    """Verify clean highlighting of a single span."""
    text = "I am feeling scared about the results."
    spans = [
        {"start": 13, "end": 19, "label": "explicit_feeling", "span_id": "s1"}
    ]
    html_out = build_highlighted_text(text, spans)
    assert '<span class="highlight-span"' in html_out
    assert "scared</span>" in html_out
    assert "I am feeling " in html_out
    assert " about the results." in html_out


def test_build_highlighted_text_adjacent_and_overlapping_spans():
    """Verify interval partitioning does not produce malformed or broken HTML on overlapping spans."""
    text = "very sad and hopeless"
    # Spans: [0:8] 'very sad', [5:21] 'sad and hopeless' (overlap at 'sad')
    spans = [
        {"start": 0, "end": 8, "label": "explicit_feeling", "span_id": "s1"},
        {"start": 5, "end": 21, "label": "implicit_feeling", "span_id": "s2"},
    ]
    html_out = build_highlighted_text(text, spans)

    # Check balanced tags
    open_tags = html_out.count("<span")
    close_tags = html_out.count("</span>")
    assert open_tags == close_tags
    assert open_tags >= 2


def test_build_highlighted_text_event_handlers_and_quotes():
    """Verify that quotes, double quotes, and event handler vectors remain inert."""
    malicious_text = 'Patient said "feeling bad" & <img src=x onerror="alert(\'XSS\')">'
    spans = [
        {"start": 14, "end": 25, "label": "explicit_feeling", "span_id": "s1"}
    ]
    html_out = build_highlighted_text(malicious_text, spans)
    assert '<img' not in html_out
    assert '&lt;img src=x onerror=&quot;alert(&#x27;XSS&#x27;)&quot;&gt;' in html_out
    assert '&amp;' in html_out


def test_build_highlighted_text_unicode_and_emoji_character_offsets():
    """Verify character offset precision with unicode text, accents, and emojis."""
    text = "Feeling 😢 and naïve about treatments."
    # "Feeling " is 8 chars, "😢" is 1 char (Python len 1), " and " is 5 chars
    # "naïve" starts at 14, ends at 19
    spans = [
        {"start": 14, "end": 19, "label": "explicit_feeling", "span_id": "s_naive"}
    ]
    html_out = build_highlighted_text(text, spans)
    assert "naïve</span>" in html_out
    assert "Feeling 😢 and " in html_out


def test_build_highlighted_text_out_of_bounds_and_inverted_spans():
    """Verify that invalid span offsets (negative, inverted, out of range) do not crash."""
    import re
    text = "Short text."
    spans = [
        {"start": -5, "end": 2, "label": "explicit_feeling"},
        {"start": 8, "end": 50, "label": "implicit_feeling"},
        {"start": 6, "end": 4, "label": "invalid_inverted"},
    ]
    html_out = build_highlighted_text(text, spans)
    # Tags should remain balanced and text should not be duplicated or truncated
    assert html_out.count("<span") == html_out.count("</span>")
    assert re.sub(r'<[^>]+>', '', html_out) == "Short text."


def test_render_turn_card_escapes_xss():
    """Verify render_turn_card escapes malicious script tags in text, turn_id, and spikes stage."""
    import streamlit as st
    from components.turn_card import render_turn_card
    from state import init_session_state

    if "current_annotations" not in st.session_state:
        init_session_state()

    turn = {
        "turn_id": "1<script>alert('turn')</script>",
        "speaker": "clinician",
        "text": "Hello <script>alert('text')</script>",
        "annotations": {
            "spans": [
                {
                    "span_id": "s1",
                    "text": "<script>alert('span')</script>",
                    "start": 6,
                    "end": 37,
                    "label": "acceptance_positive_regard_allowing",
                }
            ],
            "spikes_stage": "setting\"><script>alert('spikes')</script>",
        },
    }
    card_html = render_turn_card(turn, {}, show_annotations=True)
    assert "<script>" not in card_html
    assert "</script>" not in card_html
    assert "&lt;script&gt;" in card_html
