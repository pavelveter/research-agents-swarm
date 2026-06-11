from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.llm.client import invoke_messages
from research_swarm.logging_config import preview
from research_swarm.observability.langfuse import trace_agent
from research_swarm.graph.state import ResearchState, SearchResult

logger = logging.getLogger(__name__)

SEARCH_SYSTEM = (
    "You are a research search assistant. Given a question, produce a short list of "
    "evidence tuples: [\"quote or fact\", \"source name or URL\"] for each item. "
    "Return ONLY valid JSON: { \"question_id\": \"<id>\", "
    "\"evidence\": [[\"fact\", \"source\"], ...] }."
)


async def search(state: ResearchState) -> ResearchState:
    """Gather evidence for each research question."""
    if state.plan is None:
        raise RuntimeError("Planner must run before Searcher")
    with trace_agent(
        "searcher", input_data={"questions": state.plan.research_questions}
    ) as tracer:
        question = state.plan.research_questions[0]
        logger.info("Searching | question=%s", preview(question, 100))
        messages = [
            SystemMessage(content=SEARCH_SYSTEM),
            HumanMessage(content=f"Question: {question}"),
        ]
        raw = await invoke_messages(messages)
        parsed = _safe_json(raw)
        state.search_results.append(
            SearchResult(
                question_id=str(parsed.get("question_id", question)),
                evidence=[
                    f"{fact} ({source})" for fact, source in parsed.get("evidence", [])
                ],
            )
        )
        logger.info(
            "Search done | evidence_items=%s",
            len(state.search_results[-1].evidence),
        )
        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={"question_id": state.search_results[-1].question_id}
            )
    return state


def _safe_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    return json.loads(cleaned)
