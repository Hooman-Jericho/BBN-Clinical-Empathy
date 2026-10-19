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
