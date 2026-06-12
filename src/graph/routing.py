from __future__ import annotations

import logging
from typing import Literal

from graph.state import ResearchState

logger = logging.getLogger(__name__)


# Graph limit infrastructure constants
PASS_THRESHOLD = 80  # Target score for successful research completion
MAX_ITERATIONS = 5  # Safety net against infinite token consumption
MIN_DELTA = 5  # Minimum score progress between iterations


def _compute_delta(state: ResearchState) -> None:
    """Update score_delta and previous_score from the current judge score."""
    if state.previous_score is None:
        state.score_delta = None
    else:
        state.score_delta = state.judge_score - state.previous_score
    state.previous_score = state.judge_score


def _evaluate_stop_conditions(state: ResearchState) -> str | None:
    """Deterministic evaluation of cyclic graph stop conditions.

    Analyzes the current LangGraph Shared State and returns a string
    identifier of the terminal node (route), or None if it should continue.
    """

    # 1. Check if target report quality is achieved (Successful finish)
    if state.judge_score >= PASS_THRESHOLD:
        logger.info(
            "Stop condition triggered: TARGET SCORE MET | Score: %s >= %s",
            state.judge_score,
            PASS_THRESHOLD,
        )
        return "score_threshold_met"

    # 2. Check hard iteration limit (Safety net / Hard Limit)
    # Use max_iterations from state, or fallback to global constant
    max_iter = getattr(state, "max_iterations", MAX_ITERATIONS)
    if state.iteration >= max_iter:
        logger.warning(
            "Stop condition triggered: HARD LIMIT REACHED | Iteration: %s >= %s",
            state.iteration,
            max_iter,
        )
        return "max_iterations_reached"

    # 2.5. T11: Judge retry on weak evidence — allow one more iteration
    # when evidence count is meaningful but coverage is still low
    if state.iteration == 0:
        total_evidence = sum(len(sr.evidence) for sr in state.search_results)
        if total_evidence >= 5 and state.coverage_score <= 10:
            logger.info(
                "Retry granted: meaningful evidence (%d items) but low coverage (%d/30). "
                "Allowing one more iteration to improve coverage.",
                total_evidence,
                state.coverage_score,
            )
            return None  # Continue — grant one more iteration

    # 3. Check for empty-evidence iteration — stop after 1 iteration with zero evidence
    # Only triggers when search was actually attempted (providers_tried is non-empty)
    if state.iteration >= 1 and state.search_providers_tried:
        total_evidence = sum(len(sr.evidence) for sr in state.search_results)
        if total_evidence == 0:
            logger.warning(
                "Stop condition triggered: NO EVIDENCE AFTER 1 ITERATION | "
                "No retrievable evidence found for this topic."
            )
            return "no_evidence_found"

    # 3.5. Substance-aware stop: retrieval_quality_low persists → stop after iteration 1
    if getattr(state, "retrieval_quality_low", False) and state.iteration >= 2:
        logger.warning(
            "Stop condition triggered: LOW SUBSTANCE RETRIEVAL | "
            "retrieval_quality_low=True, iteration=%s",
            state.iteration,
        )
        return "low_substance_retrieval"

    # 3.7. Deep-Dive Grind: when score is plateaued (55-79) with real analytical
    # questions remaining, force continuation regardless of delta. Overrides
    # insufficient_progress and no_new_evidence but respects max_iterations,
    # retrieval_failed, and empty-evidence stops (those fire earlier at 2-3.5).
    if (
        55 <= state.judge_score < PASS_THRESHOLD
        and state.iteration < MAX_ITERATIONS
        and state.missing_topics
        and any(not str(t).startswith("[SYSTEM]") for t in state.missing_topics)
        and not getattr(state, "retrieval_failed", False)
    ):
        logger.info(
            "Deep-Dive Grind: score=%s in plateau zone, iteration=%s < %s, "
            "missing_topics=%d — forcing continuation for analytical depth",
            state.judge_score,
            state.iteration,
            MAX_ITERATIONS,
            len(state.missing_topics),
        )
        return None  # Continue — grind through the plateau

    # 4. Check for mathematical score stagnation by Judge (Metric downgrade)
    # T8: allow negative delta when score > 55 — digging deeper
    # T9: allow negative delta when quality is good and iteration hasn't maxed out
    if (
        state.iteration > 0
        and state.score_delta is not None
        and state.score_delta < MIN_DELTA
    ):
        # Negative delta = score regressed (digging deeper at cost of surface metrics)
        if state.score_delta < 0:
            if state.judge_score > 55:
                logger.info(
                    "Allowing continuation despite negative delta (%s) — "
                    "score=%s > 55, digging deeper into analytical depth",
                    state.score_delta,
                    state.judge_score,
                )
            else:
                quality_ok = not getattr(state, "retrieval_quality_low", False)
                if quality_ok and state.iteration < MAX_ITERATIONS:
                    logger.info(
                        "Allowing continuation despite negative delta (%s) — "
                        "retrieval quality OK, iteration=%s < %s",
                        state.score_delta,
                        state.iteration,
                        MAX_ITERATIONS,
                    )
                else:
                    logger.warning(
                        "Stop condition triggered: INSUFFICIENT PROGRESS | Score Delta: %s < %s",
                        state.score_delta,
                        MIN_DELTA,
                    )
                    return "insufficient_progress"
        else:
            # Small positive delta (0-4): stagnation, not regression
            logger.warning(
                "Stop condition triggered: INSUFFICIENT PROGRESS | Score Delta: %s < %s",
                state.score_delta,
                MIN_DELTA,
            )
            return "insufficient_progress"

    # 5. Check if Judge has any unresolved topics left
    if not state.missing_topics:
        logger.info(
            "Stop condition triggered: NO MISSING TOPICS | Research queue is empty."
        )
        return "no_missing_topics"

    # 6. Check if retrieval infrastructure failed (E) — NOT "nothing found"
    if getattr(state, "retrieval_failed", False):
        logger.warning(
            "Stop condition triggered: RETRIEVAL FAILED | All search providers exhausted."
        )
        return "retrieval_failed"

    # 7. Check for vector looping / semantic stagnation (F)
    # If Qdrant filtered all new facts as duplicates from previous steps.
    # Only fire after iteration >= 2 — give the swarm at least one more pass
    # to refine questions and find fresh evidence.
    if (
        state.iteration >= 2
        and hasattr(state, "new_evidence_found")
        and not state.new_evidence_found
    ):
        logger.info(
            "Stop condition triggered: SEMANTIC STAGNATION | Qdrant filtered all facts as duplicates."
        )
        return "no_new_evidence"

    # No stop condition met -> Continue traversing the graph (Decision: CONTINUE)
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
