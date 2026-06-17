from __future__ import annotations

import pytest

from graph.routing import (
    _compute_delta,
    _evaluate_stop_conditions,
    route_from_judge,
    route_node,
)
from graph.state import ResearchState


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

    # Condition C: insufficient progress (score <= 55 so T8 bypass doesn't apply)
    def test_insufficient_progress_returns_end(self) -> None:
        state = self._state(
            judge_score=45, score_delta=2, missing_topics=["t"],
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

    # Condition F: no new evidence (but only when iteration >= 2 AND retrieval did NOT fail)
    # Uses score=45 (< 55) to avoid the Deep-Dive Grind override
    def test_no_new_evidence_returns_end(self) -> None:
        state = self._state(judge_score=45, new_evidence_found=False,
                            new_evidence_count=0, retrieval_failed=False,
                            iteration=2)
        assert route_from_judge(state) == "__end__"

    # Deep-Dive Grind: score 55-79 + real topics + healthy retrieval → force continue
    def test_deep_dive_grind_forces_continuation(self) -> None:
        """Score 58 (plateau zone), iteration 2 < 5, real missing topics,
        retrieval healthy → grind fires, overrides insufficient_progress
        and no_new_evidence, forcing continuation."""
        state = self._state(
            judge_score=58, score_delta=2, iteration=2,
            missing_topics=["vicarious liability of AI", "ZKP proof of eligibility"],
            new_evidence_found=False,  # would trigger position 7
            retrieval_failed=False,
        )
        assert route_from_judge(state) == "searcher"
        reason = _evaluate_stop_conditions(state)
        assert reason is None  # grind returns None = continue

    # Grind blocked: retrieval_failed → grind doesn't fire, falls to position 6
    def test_deep_dive_grind_blocked_by_retrieval_failed(self) -> None:
        """Score 58, real topics, but retrieval_failed=True → grind skips,
        position 6 (retrieval_failed) fires instead."""
        state = self._state(
            judge_score=58, iteration=2,
            missing_topics=["vicarious liability of AI"],
            retrieval_failed=True,
            retrieval_failure_reason="All providers down",
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "retrieval_failed"

    # Grind blocked: only [SYSTEM] topics → no real analytical work remains
    def test_deep_dive_grind_blocked_by_system_topics_only(self) -> None:
        """Score 70, but all missing_topics are [SYSTEM] prefixed —
        grind skips, position 7 (no_new_evidence) fires since there are
        no real analytical questions to answer."""
        state = self._state(
            judge_score=70, iteration=2,
            missing_topics=["[SYSTEM] low coverage", "[SYSTEM] weak sources"],
            new_evidence_found=False,
            retrieval_failed=False,
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "no_new_evidence"

    # Grind blocked: at max iterations → hard limit fires first
    def test_deep_dive_grind_blocked_by_max_iterations(self) -> None:
        """Score 65, real topics, but iteration=5 >= max_iterations=5 →
        position 2 (max_iterations_reached) fires before grind."""
        state = self._state(
            judge_score=65, iteration=5, max_iterations=5,
            missing_topics=["something unresolved"],
            retrieval_failed=False,
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "max_iterations_reached"

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
            judge_score=45, previous_score=43,
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
        # Score 45 (< 55) avoids Deep-Dive Grind override
        state = self._state(judge_score=45, new_evidence_found=False,
                            new_evidence_count=0, iteration=2)
        result = await route_node(state)
        assert result.stop_reason == "no_new_evidence"

    @pytest.mark.asyncio
    async def test_stops_on_retrieval_failed(self) -> None:
        state = self._state(judge_score=70, retrieval_failed=True,
                            retrieval_failure_reason="All providers down")
        result = await route_node(state)
        assert result.stop_reason == "retrieval_failed"

    @pytest.mark.asyncio
    async def test_deep_dive_grind_forces_continuation(self) -> None:
        """Score 58, iteration 2 < 5, real topics, healthy retrieval —
        grind fires, iterations increment, stop_reason stays empty."""
        state = self._state(
            judge_score=58, iteration=2, max_iterations=5,
            missing_topics=["vicarious liability of AI"],
            new_evidence_found=False,
            retrieval_failed=False,
        )
        result = await route_node(state)
        assert result.stop_reason == ""
        assert result.iteration == 3  # grind continued, iteration incremented
        assert result.no_progress is False

    @pytest.mark.asyncio
    async def test_deep_dive_grind_blocked_by_retrieval_failed(self) -> None:
        """Score 58, real topics, but retrieval_failed=True →
        grind skips, retrieval_failed stop fires."""
        state = self._state(
            judge_score=58, iteration=2, max_iterations=5,
            missing_topics=["vicarious liability of AI"],
            retrieval_failed=True,
            retrieval_failure_reason="All providers down",
        )
        result = await route_node(state)
        assert result.stop_reason == "retrieval_failed"
        assert result.iteration == 2  # not incremented on stop

    @pytest.mark.asyncio
    async def test_deep_dive_grind_blocked_by_system_topics_only(self) -> None:
        """All missing_topics are [SYSTEM] prefixed → grind skips,
        position 7 (no_new_evidence) fires since no real analytical
        work remains."""
        state = self._state(
            judge_score=70, iteration=2, max_iterations=5,
            missing_topics=["[SYSTEM] low coverage", "[SYSTEM] weak sources"],
            new_evidence_found=False,
            retrieval_failed=False,
        )
        result = await route_node(state)
        assert result.stop_reason == "no_new_evidence"

    @pytest.mark.asyncio
    async def test_deep_dive_grind_blocked_by_max_iterations(self) -> None:
        """Score 65, real topics, but iteration >= max_iterations →
        max_iterations_reached fires before grind."""
        state = self._state(
            judge_score=65, iteration=5, max_iterations=5,
            missing_topics=["something unresolved"],
            retrieval_failed=False,
        )
        result = await route_node(state)
        assert result.stop_reason == "max_iterations_reached"

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
            judge_score=45, score_delta=2,
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


class TestNegativeDeltaBypass:
    """T8 + T9: negative delta should NOT trigger insufficient_progress
    when the score is high (T8) or quality is OK within iteration budget (T9).

    Small positive delta (0-4) still triggers insufficient_progress —
    that is covered by the existing test_insufficient_progress_* tests.
    """

    def _state(self, **overrides) -> ResearchState:
        defaults = {
            "query": "test",
            "judge_score": 70,
            "iteration": 1,
            "max_iterations": 5,
            "score_delta": None,
            "missing_topics": ["topic A"],
            "new_evidence_found": True,
            "retrieval_failed": False,
            "retrieval_quality_low": False,
        }
        defaults.update(overrides)
        return ResearchState(**defaults)

    # ── T8: negative delta + score > 55 → continue (digging deeper) ──

    def test_t8_negative_delta_high_score_continues(self) -> None:
        """Score 78 > 55, delta -3 — regression but quality is high."""
        state = self._state(judge_score=78, score_delta=-3)
        assert route_from_judge(state) == "searcher"
        reason = _evaluate_stop_conditions(state)
        assert reason is None

    def test_t8_negative_delta_at_boundary_score_56_continues(self) -> None:
        """Score 56 > 55, delta -1 — just above the T8 threshold."""
        state = self._state(judge_score=56, score_delta=-1)
        assert route_from_judge(state) == "searcher"

    def test_t8_boundary_score_55_falls_through_to_t9(self) -> None:
        """Score 55 (not > 55), delta -1, quality OK — T9 applies instead."""
        state = self._state(judge_score=55, score_delta=-1)
        # T8 doesn't fire (55 not > 55), T9 fires (quality OK, iter 1 < 5)
        assert route_from_judge(state) == "searcher"

    # ── T9: negative delta + quality OK + iteration < max → continue ──

    def test_t9_negative_delta_quality_ok_continues(self) -> None:
        """Score 42, delta -5, quality OK, iteration 2 < 5."""
        state = self._state(
            judge_score=42, score_delta=-5, iteration=2,
            retrieval_quality_low=False,
        )
        assert route_from_judge(state) == "searcher"
        reason = _evaluate_stop_conditions(state)
        assert reason is None

    def test_t9_negative_delta_iteration_4_continues(self) -> None:
        """Iteration 4 < 5 — still within budget."""
        state = self._state(
            judge_score=42, score_delta=-5, iteration=4,
        )
        assert route_from_judge(state) == "searcher"

    def test_t9_negative_delta_low_score_still_continues(self) -> None:
        """Score 30, delta -8 — low score but quality OK, should continue."""
        state = self._state(
            judge_score=30, score_delta=-8, iteration=2,
        )
        assert route_from_judge(state) == "searcher"

    # ── Edge: negative delta still stops when quality is low ──

    def test_negative_delta_quality_low_stops(self) -> None:
        """Quality flagged → bypass denied, insufficient_progress fires.

        Uses iteration=1 so position 3.5 (low_substance_retrieval) doesn't
        fire first — we specifically test position 4's quality gate."""
        state = self._state(
            judge_score=42, score_delta=-5, iteration=1,
            retrieval_quality_low=True,
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "insufficient_progress"

    # ── Edge: negative delta still stops at max iterations ──

    def test_negative_delta_at_max_iterations_stops_on_hard_limit(self) -> None:
        """Iteration 5 >= 5 → max_iterations beats insufficient_progress."""
        state = self._state(
            judge_score=42, score_delta=-5, iteration=5,
            max_iterations=5,
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "max_iterations_reached"

    # ── Small positive delta still stops (regression check) ──

    def test_small_positive_delta_still_stops(self) -> None:
        """Delta +2 is not negative → insufficient_progress regardless.
        Score 45 (< 55) avoids Deep-Dive Grind override."""
        state = self._state(
            judge_score=45, score_delta=2, iteration=2,
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "insufficient_progress"

    def test_zero_delta_stops(self) -> None:
        """Delta 0 is not negative → treated as stagnation."""
        state = self._state(
            judge_score=42, score_delta=0, iteration=2,
        )
        assert route_from_judge(state) == "__end__"
        reason = _evaluate_stop_conditions(state)
        assert reason == "insufficient_progress"

    # ── iteration=0 guard: position 4 requires iteration > 0 ──

    def test_negative_delta_at_iteration_zero_skips_insufficient_progress(self) -> None:
        """Iteration 0 + negative delta — guard clause 'iteration > 0' prevents
        insufficient_progress from firing. Falls through to other checks."""
        state = self._state(
            judge_score=42, score_delta=-10, iteration=0,
            missing_topics=["t"],  # has topics, so won't stop on that
        )
        # Not insufficient_progress — the iteration>0 guard skips position 4
        assert route_from_judge(state) == "searcher"
        reason = _evaluate_stop_conditions(state)
        assert reason is None

    # ── route_node T8 bypass: verify state mutations ──

    @pytest.mark.asyncio
    async def test_t8_bypass_route_node_continues(self) -> None:
        """route_node with T8 bypass should increment and clear stop_reason.

        Sets previous_score so _compute_delta produces delta = 78-81 = -3."""
        state = self._state(
            judge_score=78, previous_score=81, iteration=1,
            search_mode="targeted",
        )
        result = await route_node(state)
        assert result.stop_reason == ""
        assert result.iteration == 2
        assert result.no_progress is False  # bypassed, so no "no_progress" stop

    @pytest.mark.asyncio
    async def test_t9_bypass_route_node_continues(self) -> None:
        """route_node with T9 bypass should increment and clear stop_reason.

        Sets previous_score so _compute_delta produces delta = 42-47 = -5."""
        state = self._state(
            judge_score=42, previous_score=47, iteration=2,
            retrieval_quality_low=False, search_mode="targeted",
        )
        result = await route_node(state)
        assert result.stop_reason == ""
        assert result.iteration == 3
        assert result.no_progress is False  # bypassed, so no "no_progress" stop

    @pytest.mark.asyncio
    async def test_iteration_zero_guard_route_node_skips_insufficient_progress(self) -> None:
        """route_node at iteration=0: _compute_delta produces a negative delta,
        but the iteration>0 guard in position 4 prevents insufficient_progress.
        The node should continue normally — increment to 1, clear stop_reason."""
        state = self._state(
            judge_score=42, previous_score=52, iteration=0,
            search_mode="full",
        )
        result = await route_node(state)
        assert result.stop_reason == ""
        assert result.iteration == 1
        assert result.score_delta == -10
        assert result.no_progress is False  # guard prevented stop


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
