"""Shared utilities used across the research-swarm codebase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Template

from graph.state import ResearchState

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_shared_http_client: httpx.AsyncClient | None = None


def get_shared_http_client() -> httpx.AsyncClient:
    """Return a process-wide shared httpx.AsyncClient with tuned connection limits."""
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _shared_http_client


async def close_shared_http_client() -> None:
    """Gracefully close the shared httpx.AsyncClient."""
    global _shared_http_client
    if _shared_http_client is not None and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
        _shared_http_client = None


def render_prompt(template_name: str, **kwargs) -> str:
    """Load and render a prompt from the prompts directory."""
    path = _PROMPTS_DIR / template_name
    with open(path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    return template.render(**kwargs)


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


def merge_state(current: ResearchState, event: dict) -> ResearchState:
    """Merge a LangGraph event update into the current ResearchState.

    Used by both ``main.py`` and ``news_sender.py`` to accumulate state across
    streaming workflow iterations.
    """
    merged = current.model_dump()
    for update in event.values():
        if isinstance(update, ResearchState):
            merged.update(update.model_dump())
        elif isinstance(update, dict):
            merged.update(update)
    return ResearchState(**merged)
