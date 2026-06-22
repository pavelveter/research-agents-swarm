"""Shared utilities used across the research-swarm codebase."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from graph.state import ResearchState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


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


async def run_workflow(
    query: str,
    session_id: str,
    *,
    node_logger: logging.Logger | None = None,
) -> tuple[ResearchState, dict]:
    """Execute the research workflow and return the final state + metadata.

    Extracts the duplicated streaming loop from ``main.py`` and
    ``news_sender.py`` into a single reusable entry point.

    Returns ``(final_state, metadata)`` where *metadata* contains:
    - ``"nodes_completed"``: list of node names that finished.
    - ``"final_state"``: the accumulated ``ResearchState``.
    """
    from graph.workflow import build_workflow

    log = node_logger or logger
    state = ResearchState(query=query, session_id=session_id)
    workflow = build_workflow()

    nodes_completed: list[str] = []
    result = state
    async for event in workflow.astream(state, stream_mode="updates"):
        for node, _update in event.items():
            log.info("Finished node: %s", node)
            nodes_completed.append(node)
        result = merge_state(result, event)

    return result, {"nodes_completed": nodes_completed, "final_state": result}
