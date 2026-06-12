#!/usr/bin/env python3
"""Run the research-swarm workflow from the command line."""

from __future__ import annotations

import asyncio
import logging

from research_swarm.config.settings import get_settings
from research_swarm.graph.state import ResearchState
from research_swarm.graph.workflow import build_workflow
from research_swarm.llm.client import shutdown_llm_client
from research_swarm.logging_config import setup_terminal_logging
from research_swarm.observability.langfuse import shutdown_observability
from research_swarm.utils import merge_state

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_terminal_logging()
    settings = get_settings()

    try:
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

        workflow = build_workflow()

        result = state
        async for event in workflow.astream(state, stream_mode="updates"):
            for node, _update in event.items():
                logger.info("Finished node: %s", node)
            result = merge_state(result, event)

        _log_final_result(result)
        _print_result(result)

        # Log search health summary
        from research_swarm.search import get_orchestrator
        get_orchestrator().health.log_summary()
    finally:
        await shutdown_llm_client()
        shutdown_observability()



def _log_final_result(state: ResearchState) -> None:
    logger.info("=" * 60)
    logger.info("Workflow complete")
    logger.info("  Iterations:  %s", state.iteration)
    logger.info("  Final score: %s", state.judge_score)
    logger.info("  Stop reason: %s", state.stop_reason)
    logger.info("  Score delta: %s",
                 f"+{state.score_delta}" if state.score_delta is not None
                 and state.score_delta >= 0 else state.score_delta)
    logger.info("  No progress: %s", state.no_progress)
    logger.info("  Retrieval failed: %s", state.retrieval_failed)
    logger.info("  Search provider: %s", state.search_provider_used or "none")
    logger.info("  Providers tried: %s", state.search_providers_tried)
    logger.info("  New evidence count: %s", state.new_evidence_count)
    logger.info("  Total search results: %s", len(state.search_results))
    logger.info("  Total validated results: %s", len(state.validated_results))
    if state.final_report:
        logger.info("  Report preview: %s", state.final_report.summary[:200])
    logger.info("=" * 60)


def _print_result(state: ResearchState) -> None:
    print("\n--- Final score ---")
    print(state.judge_score)
    print(f"\n--- Component scores ---")
    print(f"  Coverage:    {state.coverage_score}/30")
    print(f"  Evidence:    {state.evidence_score}/20")
    print(f"  Sources:     {state.source_score}/20")
    print(f"  Depth:       {state.depth_score}/15")
    print(f"  Completeness:{state.completeness_score}/15")
    print(f"\n--- Stop reason ---")
    print(state.stop_reason)
    print(f"\n--- Retrieval ---")
    print(f"  Failed:     {state.retrieval_failed}")
    print(f"  Provider:   {state.search_provider_used or 'none'}")
    print(f"  Tried:      {state.search_providers_tried}")
    print(f"\n--- Iterations ---")
    print(state.iteration)
    print("\n--- Report ---")
    print(state.final_report.summary if state.final_report else "(none)")
    if state.missing_topics:
        print("\n--- Missing topics ---")
        for t in state.missing_topics:
            print(f"  - {t}")


if __name__ == "__main__":
    asyncio.run(main())
