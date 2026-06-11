from __future__ import annotations

import logging
from typing import Literal

from research_swarm.graph.state import ResearchState

logger = logging.getLogger(__name__)

MIN_DELTA = 2
PASS_THRESHOLD = 80


def _compute_delta(state: ResearchState) -> None:
    """Update score_delta and previous_score from the current judge score."""
    if state.previous_score is None:
        state.score_delta = None
    else:
        state.score_delta = state.judge_score - state.previous_score
    state.previous_score = state.judge_score


def _evaluate_stop_conditions(state: ResearchState) -> str | None:
    if state.judge_score >= PASS_THRESHOLD:
        return "score_threshold_met"

    # Проверяем строго текущую итерацию
    if state.iteration >= state.max_iterations:
        return "max_iterations_reached"

    # score_delta проверяем только если это не первая итерация (iteration > 0)
    if (
        state.iteration > 0
        and state.score_delta is not None
        and state.score_delta < MIN_DELTA
    ):
        return "insufficient_progress"

    if not state.missing_topics:
        return "no_missing_topics"

    if not state.new_evidence_found and state.new_evidence_count == 0:
        return "no_new_evidence"

    return None


def _log_component_scores(state: ResearchState) -> None:
    if any(
        [
            state.coverage_score,
            state.evidence_score,
            state.source_score,
            state.depth_score,
            state.completeness_score,
        ]
    ):
        logger.info(
            "  Coverage: %s/30  Evidence: %s/20  Sources: %s/20  Depth: %s/15  Completeness: %s/15",
            state.coverage_score,
            state.evidence_score,
            state.source_score,
            state.depth_score,
            state.completeness_score,
        )


def route_from_judge(
    state: ResearchState,
) -> Literal["searcher", "__end__"]:
    """Pure routing function — reads state, returns next node id.

    When retrieval_failed is True, routes to END instead of looping
    infinitely — the infrastructure cannot recover within this workflow.
    """
    stop_reason = _evaluate_stop_conditions(state)
    return "__end__" if stop_reason is not None else "searcher"


async def route_node(state: ResearchState) -> ResearchState:
    """Persist routing decisions and iteration updates into graph state."""
    updated = state.model_copy()
    _compute_delta(updated)
    stop_reason = _evaluate_stop_conditions(updated)

    if stop_reason is not None:
        updated.stop_reason = stop_reason
        updated.no_progress = (
            updated.score_delta is not None and updated.score_delta < MIN_DELTA
        )

        logger.info("=" * 50)
        logger.info(
            "Decision: STOP | Reason: %s | iteration=%s score=%s delta=%s retrieval_failed=%s",
            stop_reason,
            updated.iteration,
            updated.judge_score,
            updated.score_delta,
            updated.retrieval_failed,
        )
        _log_component_scores(updated)
        logger.info("=" * 50)
        return updated

    updated.iteration += 1

    # Set search mode: after first iteration, switch to targeted mode
    if updated.iteration >= 1:
        updated.search_mode = "targeted"

    logger.info("-" * 40)
    logger.info(
        "Iteration: %s -> %s | Score: %s | Delta: %s | Mode: %s",
        state.iteration,
        updated.iteration,
        updated.judge_score,
        updated.score_delta,
        updated.search_mode,
    )
    _log_component_scores(updated)

    if updated.missing_topics:
        logger.info("Missing topics:")
        for topic in updated.missing_topics:
            logger.info("  - %s", topic)

    logger.info("Decision: CONTINUE")
    logger.info("-" * 40)
    return updated
