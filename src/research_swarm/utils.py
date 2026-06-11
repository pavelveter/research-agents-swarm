"""Shared utilities used across the research-swarm codebase."""

from __future__ import annotations

import json
from typing import Any


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    return cleaned


def safe_json(text: str) -> dict[str, Any]:
    """Strip ```json fences and parse JSON from an LLM response.

    Handles common LLM formatting quirks:
    - Markdown code fences (```json ... ```)
    - Plain code fences (``` ... ```)
    - Leading/trailing whitespace
    - Multiple backtick variations
    - Prose before/after a JSON object
    """
    cleaned = _strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    return parsed
