from __future__ import annotations

import pytest

from research_swarm.graph.routing import route_from_judge
from research_swarm.graph.state import ResearchState


class TestRouteFromJudge:
    """Tests for the conditional routing function."""

    def test_returns_end_when_score_above_threshold(self) -> None:
        """Score >= 80 should route to END."""
        state = ResearchState(query="test", judge_score=80)
        result = route_from_judge(state)
        assert result == "__end__"

    def test_returns_end_when_score_equal_to_threshold(self) -> None:
        state = ResearchState(query="test", judge_score=80)
        result = route_from_judge(state)
        assert result == "__end__"

    def test_returns_searcher_when_score_below_threshold(self) -> None:
        state = ResearchState(query="test", judge_score=79)
        result = route_from_judge(state)
        assert result == "searcher"

    def test_returns_searcher_when_score_zero(self) -> None:
        state = ResearchState(query="test", judge_score=0)
        result = route_from_judge(state)
        assert result == "searcher"

    def test_returns_searcher_when_score_negative(self) -> None:
        state = ResearchState(query="test", judge_score=-1)
        result = route_from_judge(state)
        assert result == "searcher"

    def test_returns_end_when_score_exactly_100(self) -> None:
        state = ResearchState(query="test", judge_score=100)
        result = route_from_judge(state)
        assert result == "__end__"

    def test_returns_searcher_when_score_50(self) -> None:
        state = ResearchState(query="test", judge_score=50)
        result = route_from_judge(state)
        assert result == "searcher"

    def test_returns_end_when_score_99(self) -> None:
        state = ResearchState(query="test", judge_score=99)
        result = route_from_judge(state)
        assert result == "__end__"

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0, "searcher"),
            (1, "searcher"),
            (50, "searcher"),
            (79, "searcher"),
            (80, "__end__"),
            (85, "__end__"),
            (100, "__end__"),
        ],
    )
    def test_parametrized_routing(self, score: int, expected: str) -> None:
        state = ResearchState(query="test", judge_score=score)
        assert route_from_judge(state) == expected
