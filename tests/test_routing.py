from __future__ import annotations

import pytest

from research_swarm.graph.routing import (
    MIN_DELTA,
    PASS_THRESHOLD,
    _compute_delta,
    _evaluate_stop_conditions,
    route_from_judge,
    route_node,
)
from research_swarm.graph.state import ResearchState


class TestRouteFromJudge:
    """Tests for the PURE conditional routing function (no side effects).

    route_from_judge() reads state and returns END or "searcher".
    It does NOT mutate state — that's route_node()'s job.
    """

    def _state(self, **overrides) -> ResearchState:
        defaults = {
            "query": "test",
            "judge_score": 70,
            "iteration": 0,
            "max_iterations": 3,
            "score_delta": None,
            "missing_topics": ["topic A"],
            "new_evidence_found": True,
            "retrieval_failed": False,
        }
        defaults.update(overrides)
        return ResearchState(**defaults)

    # Condition A: score >= threshold
    def test_score_above_threshold_returns_end(self) -> None:
        state = self._state(judge_score=85)
        assert route_from_judge(state) == "__end__"

    # Condition B: max iterations
    def test_max_iterations_returns_end(self) -> None:
        state = self._state(judge_score=70, iteration=3, max_iterations=3)
        assert route_from_judge(state) == "__end__"

    # Condition C: insufficient progress
    def test_insufficient_progress_returns_end(self) -> None:
        state = self._state(
            judge_score=64, score_delta=2, missing_topics=["t"],
            iteration=1, max_iterations=5,
        )
        assert route_from_judge(state) == "__end__"

    # Condition D: no missing topics
    def test_no_missing_topics_returns_end(self) -> None:
        state = self._state(judge_score=70, missing_topics=[])
        assert route_from_judge(state) == "__end__"

    # Condition E: retrieval_failed (NEW)
    def test_retrieval_failed_returns_end(self) -> None:
        state = self._state(judge_score=70, retrieval_failed=True,
                            retrieval_failure_reason="All providers down")
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "retrieval_failed"

    # Condition F: no new evidence (but only when retrieval did NOT fail)
    def test_no_new_evidence_returns_end(self) -> None:
        state = self._state(judge_score=70, new_evidence_found=False,
                            new_evidence_count=0, retrieval_failed=False)
        assert route_from_judge(state) == "__end__"

    # Continue
    def test_continues_when_all_checks_pass(self) -> None:
        state = self._state(
            judge_score=70, score_delta=10, iteration=1,
            missing_topics=["security"], new_evidence_found=True,
            retrieval_failed=False,
        )
        assert route_from_judge(state) == "searcher"

    # Pure function: does NOT mutate state
    def test_does_not_mutate_state(self) -> None:
        state = self._state(iteration=0)
        orig_iter = state.iteration
        route_from_judge(state)
        assert state.iteration == orig_iter  # unchanged!
        assert state.stop_reason == ""


class TestRouteNode:
    """Tests for route_node() — the LangGraph node that persists state mutations."""

    def _state(self, **overrides) -> ResearchState:
        defaults = {
            "query": "test",
            "judge_score": 70,
            "iteration": 0,
            "max_iterations": 3,
            "previous_score": None,
            "score_delta": None,
            "missing_topics": ["topic A"],
            "new_evidence_found": True,
            "retrieval_failed": False,
            "search_mode": "full",
        }
        defaults.update(overrides)
        return ResearchState(**defaults)

    @pytest.mark.asyncio
    async def test_stops_on_score_threshold(self) -> None:
        state = self._state(judge_score=85)
        result = await route_node(state)
        assert result.stop_reason == "score_threshold_met"
        assert result.iteration == 0  # not incremented on stop

    @pytest.mark.asyncio
    async def test_stops_on_max_iterations(self) -> None:
        state = self._state(judge_score=70, iteration=3, max_iterations=3)
        result = await route_node(state)
        assert result.stop_reason == "max_iterations_reached"

    @pytest.mark.asyncio
    async def test_stops_on_insufficient_progress(self) -> None:
        state = self._state(
            judge_score=64, previous_score=62,
            iteration=1, max_iterations=5,
        )
        result = await route_node(state)
        assert result.stop_reason == "insufficient_progress"
        assert result.no_progress is True
        assert result.score_delta == 2

    @pytest.mark.asyncio
    async def test_stops_on_no_missing_topics(self) -> None:
        state = self._state(judge_score=70, missing_topics=[])
        result = await route_node(state)
        assert result.stop_reason == "no_missing_topics"

    @pytest.mark.asyncio
    async def test_stops_on_no_new_evidence(self) -> None:
        state = self._state(judge_score=70, new_evidence_found=False,
                            new_evidence_count=0)
        result = await route_node(state)
        assert result.stop_reason == "no_new_evidence"

    @pytest.mark.asyncio
    async def test_stops_on_retrieval_failed(self) -> None:
        state = self._state(judge_score=70, retrieval_failed=True,
                            retrieval_failure_reason="All providers down")
        result = await route_node(state)
        assert result.stop_reason == "retrieval_failed"

    @pytest.mark.asyncio
    async def test_continues_and_increments_iteration(self) -> None:
        state = self._state(
            judge_score=70, score_delta=10, iteration=1,
            missing_topics=["security"], new_evidence_found=True,
        )
        result = await route_node(state)
        assert result.iteration == 2  # INCREMENTED — persisted!
        assert result.stop_reason == ""

    @pytest.mark.asyncio
    async def test_first_iteration_increments_to_1(self) -> None:
        state = self._state(judge_score=70, iteration=0)
        result = await route_node(state)
        assert result.iteration == 1

    @pytest.mark.asyncio
    async def test_switches_to_targeted_mode_on_continue(self) -> None:
        """After first iteration, search_mode should become 'targeted'."""
        state = self._state(judge_score=70, iteration=0, search_mode="full")
        result = await route_node(state)
        assert result.iteration == 1
        assert result.search_mode == "targeted"

    @pytest.mark.asyncio
    async def test_computes_delta(self) -> None:
        state = self._state(
            judge_score=75, previous_score=70, iteration=1,
        )
        result = await route_node(state)
        assert result.score_delta == 5
        assert result.previous_score == 75

    @pytest.mark.asyncio
    async def test_delta_none_on_first_call(self) -> None:
        state = self._state(judge_score=70, previous_score=None)
        result = await route_node(state)
        assert result.score_delta is None
        assert result.previous_score == 70


class TestStopConditionPriority:
    """Verify priority order: A > B > C > D > E > F."""

    def _state(self, **overrides) -> ResearchState:
        defaults = {
            "query": "test",
            "judge_score": 70,
            "iteration": 0,
            "max_iterations": 3,
            "score_delta": None,
            "missing_topics": ["t"],
            "new_evidence_found": True,
            "retrieval_failed": False,
        }
        defaults.update(overrides)
        return ResearchState(**defaults)

    def test_score_beats_max_iterations(self) -> None:
        state = self._state(judge_score=85, iteration=5, max_iterations=3)
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "score_threshold_met"

    def test_max_iterations_beats_insufficient_progress(self) -> None:
        state = self._state(
            judge_score=64, score_delta=2,
            iteration=3, max_iterations=3,
        )
        reason = _evaluate_stop_conditions(state)
        assert reason == "max_iterations_reached"

    def test_insufficient_progress_beats_no_missing_topics(self) -> None:
        state = self._state(
            judge_score=64, score_delta=2,
            missing_topics=["t"], iteration=1,
        )
        reason = _evaluate_stop_conditions(state)
        assert reason == "insufficient_progress"

    def test_no_missing_topics_beats_retrieval_failed(self) -> None:
        state = self._state(
            judge_score=70, missing_topics=[],
            retrieval_failed=True,
        )
        reason = _evaluate_stop_conditions(state)
        assert reason == "no_missing_topics"

    def test_retrieval_failed_beats_no_new_evidence(self) -> None:
        state = self._state(
            judge_score=70, retrieval_failed=True,
            new_evidence_found=False, new_evidence_count=0,
            missing_topics=["t"],
        )
        reason = _evaluate_stop_conditions(state)
        assert reason == "retrieval_failed"


class TestComputeDelta:
    """Tests for the _compute_delta helper."""

    def test_score_delta_none_when_no_previous(self) -> None:
        state = ResearchState(query="t", judge_score=70, previous_score=None)
        _compute_delta(state)
        assert state.score_delta is None
        assert state.previous_score == 70

    def test_score_delta_computed(self) -> None:
        state = ResearchState(query="t", judge_score=75, previous_score=70)
        _compute_delta(state)
        assert state.score_delta == 5
        assert state.previous_score == 75

    def test_score_delta_negative(self) -> None:
        state = ResearchState(query="t", judge_score=60, previous_score=70)
        _compute_delta(state)
        assert state.score_delta == -10
