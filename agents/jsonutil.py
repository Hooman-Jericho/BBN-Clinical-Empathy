"""Text/JSON helpers shared by the agents (kept free of agent imports to avoid cycles)."""

import json
import logging
import re
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def resolve_span_coordinates(
    text: str, span_text: str, suggested_start: int, suggested_end: int
) -> Tuple[int, int]:
    """Resolve exact span coordinates in text without first-occurrence bias."""
    if not span_text:
        return suggested_start, suggested_end

    # Exact match at suggested position
    if 0 <= suggested_start < suggested_end <= len(text):
        if text[suggested_start:suggested_end] == span_text:
            return suggested_start, suggested_end

    # Search all occurrences
    occurrences = []
    start = 0
    while True:
        idx = text.find(span_text, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(span_text)))
        start = idx + 1

    if not occurrences:
        return suggested_start, suggested_end

    # Choose occurrence closest to suggested_start
    closest = min(occurrences, key=lambda occ: abs(occ[0] - suggested_start))
    return closest[0], closest[1]


def extract_json_from_text(response: str) -> Dict[str, Any]:
    """Robust multi-pass JSON extractor from LLM responses.

    Handles direct JSON, markdown code fences, JSON embedded in prose, trailing commas and
    top-level arrays (normalised into ``{"annotations": [...]}``). Returns ``{}`` on failure;
    callers MUST treat ``{}`` as a parse failure and count it (see ``validation.ParseStats``).
    """
    if not response or not str(response).strip():
        return {}

    text = str(response).strip()

    def _load(candidate: str):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def _normalise(data):
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"annotations": data}
        return None

    # Pass 1: direct load
    out = _normalise(_load(text))
    if out is not None:
        return out

    # Pass 2: markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        out = _normalise(_load(match.group(1).strip()))
        if out is not None:
            return out

    # Pass 3: outer braces
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        data = _load(text[a : b + 1])
        if isinstance(data, dict):
            return data

    # Pass 4: outer brackets
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b > a:
        data = _load(text[a : b + 1])
        if isinstance(data, list):
            return {"annotations": data}

    logger.warning("Failed to extract JSON from LLM response: %s", text[:200])
    return {}
