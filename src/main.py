#!/usr/bin/env python3
"""Run the research-swarm workflow from the command line."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

from config.settings import get_settings
from graph.state import ResearchState
from graph.workflow import build_workflow
from llm.client import shutdown_llm_client
from logging_config import setup_terminal_logging
from observability.langfuse import shutdown_observability
from utils import merge_state

logger = logging.getLogger(__name__)

QUERY_FILE = Path("theme-of-the-news.txt")


async def async_main() -> None:
    setup_terminal_logging()
    settings = get_settings()

    try:
        query = _read_query_from_file()

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
        from search import get_orchestrator
        get_orchestrator().health.log_summary()
    finally:
        await shutdown_llm_client()
        shutdown_observability()



def _read_query_from_file() -> str:
    path = QUERY_FILE
    if not path.exists():
        raise SystemExit(f"Query file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Query file is empty: {path}")
    return text


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
    print("\n--- Component scores ---")
    print(f"  Coverage:    {state.coverage_score}/30")
    print(f"  Evidence:    {state.evidence_score}/20")
    print(f"  Sources:     {state.source_score}/20")
    print(f"  Depth:       {state.depth_score}/15")
    print(f"  Completeness:{state.completeness_score}/15")
    print("\n--- Stop reason ---")
    print(state.stop_reason)
    print("\n--- Retrieval ---")
    print(f"  Failed:     {state.retrieval_failed}")
    print(f"  Provider:   {state.search_provider_used or 'none'}")
    print(f"  Tried:      {state.search_providers_tried}")
    print("\n--- Iterations ---")
    print(state.iteration)
    print("\n--- Report ---")
    print(state.final_report.summary if state.final_report else "(none)")
    if state.final_report and state.final_report.sources:
        print("\n--- Sources ---")
        seen: set[str] = set()
        unique_sources = [u for u in state.final_report.sources
                          if not (u in seen or seen.add(u))]
        for i, url in enumerate(unique_sources, start=1):
            domain = urlparse(url).netloc or url
            print(f"  {i}. {domain:<30s} → {url}")
    if state.missing_topics:
        print("\n--- Missing topics ---")
        for t in state.missing_topics:
            print(f"  - {t}")

def main() -> None:
    """Synchronous entry point for the hatch/pip CLI script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
