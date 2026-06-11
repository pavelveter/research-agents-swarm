from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.llm.client import invoke_messages
from research_swarm.logging_config import preview
from research_swarm.observability.langfuse import trace_agent
from research_swarm.graph.state import ResearchPlan, ResearchState

logger = logging.getLogger(__name__)

PLAN_SYSTEM = (
    "You are a research planner. Given a user research topic, decompose it into "
    "3-7 focused research questions. Return ONLY valid JSON with keys: "
    "goal (string) and research_questions (list of strings)."
)


async def plan(state: ResearchState) -> ResearchState:
    """Analyze the user query and produce a research plan."""
    logger.info("Planning research for: %s", preview(state.query, 80))
    with trace_agent("planner", input_data={"query": state.query}) as tracer:
        messages = [
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(content=f"Topic: {state.query}"),
        ]
        raw = await invoke_messages(messages)
        plan_data = _safe_json(raw)
        if not isinstance(plan_data, dict) or "research_questions" not in plan_data:
            raise ValueError("Planner did not return a valid research plan")
        state.plan = ResearchPlan(
            goal=str(plan_data.get("goal", state.query)),
            research_questions=[str(q) for q in plan_data["research_questions"]],
        )
        logger.info(
            "Plan ready | goal=%r questions=%s",
            preview(state.plan.goal, 80),
            len(state.plan.research_questions),
        )
        if hasattr(tracer, "update_observation"):
            tracer.update_observation(output=state.plan.model_dump())
    return state


def _safe_json(text: str) -> dict[str, Any]:
    """Strip ```json fences and parse JSON from an LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    return json.loads(cleaned)
