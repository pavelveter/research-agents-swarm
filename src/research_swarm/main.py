#!/usr/bin/env python3
"""Run the research-swarm workflow from the command line."""

from __future__ import annotations

import asyncio
import logging

from langgraph.graph import END, StateGraph

from research_swarm.agents.fact_checker import fact_check
from research_swarm.agents.judge import judge
from research_swarm.agents.planner import plan
from research_swarm.agents.searcher import search
from research_swarm.agents.summarizer import summarize
from research_swarm.config.settings import get_settings
from research_swarm.graph.state import ResearchState
from research_swarm.logging_config import setup_terminal_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_terminal_logging()
    settings = get_settings()

    if True:
        query = "Latest trends in AI coding assistants."
    else:
        query = input("Enter research topic: ")

    logger.info("Starting research swarm")
    logger.info("Query: %s", query)
    logger.info("Model: %s", settings.openai_model)
    if settings.openai_base_url:
        logger.info("API base: %s", settings.openai_base_url)

    state = ResearchState(query=query)

    builder = StateGraph(ResearchState)
    builder.add_node("planner", plan)
    builder.add_node("searcher", search)
    builder.add_node("fact_checker", fact_check)
    builder.add_node("summarizer", summarize)
    builder.add_node("judge", judge)

    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "fact_checker")
    builder.add_edge("fact_checker", "summarizer")
    builder.add_edge("summarizer", "judge")

    def next_node(state: ResearchState) -> str:
        if state.judge_score >= 80:
            logger.info("Judge score %s >= 80 — done", state.judge_score)
            return END
        logger.info("Judge score %s < 80 — looping back to searcher", state.judge_score)
        return "searcher"

    builder.add_conditional_edges("judge", next_node)

    builder.set_entry_point("planner")
    workflow = builder.compile()

    result = state
    async for event in workflow.astream(state, stream_mode="updates"):
        for node, _update in event.items():
            logger.info("Finished node: %s", node)
        result = _merge_state(result, event)

    logger.info("Workflow complete | score=%s", result.judge_score)
    if result.final_report:
        logger.info("Report preview: %s", result.final_report.summary[:200])
    print("\n--- Final score ---")
    print(result.judge_score)
    print("\n--- Report ---")
    print(result.final_report.summary if result.final_report else "(none)")


def _merge_state(current: ResearchState, event: dict) -> ResearchState:
    merged = current.model_dump()
    for update in event.values():
        if isinstance(update, ResearchState):
            merged.update(update.model_dump())
        elif isinstance(update, dict):
            merged.update(update)
    return ResearchState(**merged)


if __name__ == "__main__":
    asyncio.run(main())
