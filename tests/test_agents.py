from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.agents.planner import _safe_json as planner_safe_json
from research_swarm.agents.searcher import _safe_json as searcher_safe_json
from research_swarm.agents.fact_checker import _safe_json as fact_checker_safe_json
from research_swarm.agents.judge import _safe_json as judge_safe_json
from research_swarm.agents.summarizer import _safe_json as summarizer_safe_json


class TestSafeJson:
    """Tests for the _safe_json helper used across all agents.

    Since the function is duplicated in each agent module, verify
    that all copies behave identically.
    """

    SAFE_JSON_FUNCTIONS = [
        ("planner", planner_safe_json),
        ("searcher", searcher_safe_json),
        ("fact_checker", fact_checker_safe_json),
        ("judge", judge_safe_json),
        ("summarizer", summarizer_safe_json),
    ]

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_plain_json(self, name: str, func) -> None:
        """All implementations should parse plain JSON strings."""
        result = func('{"key": "value"}')
        assert result == {"key": "value"}

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_json_with_markdown_fence(self, name: str, func) -> None:
        """All implementations should strip ```json fences."""
        result = func('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_json_with_plain_fence(self, name: str, func) -> None:
        """All implementations should strip ``` fences (no language)."""
        result = func('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_json_with_trailing_whitespace(self, name: str, func) -> None:
        """Trailing whitespace should be handled."""
        result = func('{"key": "value"}  \n\t')
        assert result == {"key": "value"}

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_json_with_arrays(self, name: str, func) -> None:
        """Should handle JSON arrays."""
        result = func('[1, 2, 3]')
        assert result == [1, 2, 3]

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_nested_json(self, name: str, func) -> None:
        """Should handle nested JSON objects."""
        result = func('{"outer": {"inner": [1, 2]}}')
        assert result == {"outer": {"inner": [1, 2]}}

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_raises_on_invalid_json(self, name: str, func) -> None:
        """Should raise json.JSONDecodeError on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            func("not json")

    @pytest.mark.parametrize("name,func", SAFE_JSON_FUNCTIONS)
    def test_parses_json_with_multiple_backticks(self, name: str, func) -> None:
        """Should handle edge case with extra backticks."""
        result = func('````json\n{"a": 1}\n````')
        assert result == {"a": 1}


class TestPlannerAgent:
    """Tests for the planner agent."""

    @pytest.mark.asyncio
    @patch("research_swarm.agents.planner.invoke_messages")
    @patch("research_swarm.agents.planner.trace_agent")
    async def test_plan_creates_research_plan(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.planner import plan
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({
            "goal": "Analyze AI coding assistants",
            "research_questions": ["Q1", "Q2", "Q3"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI coding assistants")
        result = await plan(state)

        assert result.plan is not None
        assert result.plan.goal == "Analyze AI coding assistants"
        assert len(result.plan.research_questions) == 3
        assert result.plan.research_questions == ["Q1", "Q2", "Q3"]

    @pytest.mark.asyncio
    @patch("research_swarm.agents.planner.invoke_messages")
    @patch("research_swarm.agents.planner.trace_agent")
    async def test_plan_handles_missing_goal(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.planner import plan
        from research_swarm.graph.state import ResearchState

        # Goal field missing, uses query as fallback
        mock_invoke.return_value = json.dumps({
            "research_questions": ["Q1"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI trends")
        result = await plan(state)

        assert result.plan is not None
        assert result.plan.goal == "AI trends"  # fallback to query

    @pytest.mark.asyncio
    @patch("research_swarm.agents.planner.invoke_messages")
    @patch("research_swarm.agents.planner.trace_agent")
    async def test_plan_raises_on_invalid_response(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.planner import plan
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({"unexpected": "data"})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="test")

        with pytest.raises(ValueError, match="Planner did not return a valid research plan"):
            await plan(state)

    @pytest.mark.asyncio
    @patch("research_swarm.agents.planner.invoke_messages")
    @patch("research_swarm.agents.planner.trace_agent")
    async def test_plan_handles_json_with_fences(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.planner import plan
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = '```json\n{"goal": "Test", "research_questions": ["Q"]}\n```'

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="test")
        result = await plan(state)

        assert result.plan is not None
        assert result.plan.goal == "Test"


class TestSearcherAgent:
    """Tests for the searcher agent."""

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.invoke_messages")
    @patch("research_swarm.agents.searcher.trace_agent")
    async def test_search_appends_result(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState

        mock_invoke.return_value = json.dumps({
            "question_id": "Q1",
            "evidence": [["AI is growing", "nature.com"], ["Robots are coming", "arxiv.org"]],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            plan=ResearchPlan(goal="AI", research_questions=["Q1"]),
        )
        result = await search(state)

        assert len(result.search_results) == 1
        assert result.search_results[0].question_id == "Q1"
        assert len(result.search_results[0].evidence) == 2

    @pytest.mark.asyncio
    async def test_search_requires_plan(self) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchState

        state = ResearchState(query="AI")  # no plan
        with pytest.raises(RuntimeError, match="Planner must run before Searcher"):
            await search(state)

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.invoke_messages")
    @patch("research_swarm.agents.searcher.trace_agent")
    async def test_search_handles_empty_evidence(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState

        mock_invoke.return_value = json.dumps({
            "question_id": "Q1",
            "evidence": [],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            plan=ResearchPlan(goal="AI", research_questions=["Q1"]),
        )
        result = await search(state)

        assert len(result.search_results) == 1
        assert result.search_results[0].evidence == []

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.invoke_messages")
    @patch("research_swarm.agents.searcher.trace_agent")
    async def test_search_falls_back_question_id(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState

        # No question_id in response — should use question as fallback
        mock_invoke.return_value = json.dumps({"evidence": [["fact", "src"]]})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            plan=ResearchPlan(goal="AI", research_questions=["What is AI?"]),
        )
        result = await search(state)

        assert result.search_results[0].question_id == "What is AI?"


class TestFactCheckerAgent:
    """Tests for the fact checker agent."""

    @pytest.mark.asyncio
    @patch("research_swarm.agents.fact_checker.invoke_messages")
    @patch("research_swarm.agents.fact_checker.trace_agent")
    async def test_fact_check_validates_evidence(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.fact_checker import fact_check
        from research_swarm.graph.state import ResearchState, SearchResult

        mock_invoke.return_value = json.dumps({
            "validated_facts": ["fact 1 confirmed", "fact 2 confirmed"],
            "rejected_facts": ["bad fact"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            search_results=[
                SearchResult(question_id="q1", evidence=["fact 1", "fact 2"]),
                SearchResult(question_id="q2", evidence=["bad fact"]),
            ],
        )
        result = await fact_check(state)

        assert len(result.validated_results) == 1
        assert len(result.validated_results[0].validated_facts) == 2
        assert len(result.validated_results[0].rejected_facts) == 1

    @pytest.mark.asyncio
    @patch("research_swarm.agents.fact_checker.invoke_messages")
    @patch("research_swarm.agents.fact_checker.trace_agent")
    async def test_fact_check_no_evidence(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.fact_checker import fact_check
        from research_swarm.graph.state import ResearchState

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", search_results=[])
        result = await fact_check(state)

        # Should not call invoke_messages when no evidence
        mock_invoke.assert_not_called()
        assert len(result.validated_results) == 1
        assert result.validated_results[0].validated_facts == []
        assert result.validated_results[0].rejected_facts == []

    @pytest.mark.asyncio
    @patch("research_swarm.agents.fact_checker.invoke_messages")
    @patch("research_swarm.agents.fact_checker.trace_agent")
    async def test_fact_check_handles_missing_fields_in_response(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.fact_checker import fact_check
        from research_swarm.graph.state import ResearchState, SearchResult

        # No rejected_facts key
        mock_invoke.return_value = json.dumps({"validated_facts": ["fact 1"]})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            search_results=[SearchResult(question_id="q1", evidence=["fact 1"])],
        )
        result = await fact_check(state)

        assert len(result.validated_results[0].validated_facts) == 1
        assert result.validated_results[0].rejected_facts == []


class TestSummarizerAgent:
    """Tests for the summarizer agent."""

    @pytest.mark.asyncio
    @patch("research_swarm.agents.summarizer.invoke_messages")
    @patch("research_swarm.agents.summarizer.trace_agent")
    async def test_summarize_creates_report(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.summarizer import summarize
        from research_swarm.graph.state import (
            ResearchState,
            ValidatedResult,
        )

        mock_invoke.return_value = json.dumps({
            "summary": "AI is an important field of study.",
            "sources": ["nature.com", "arxiv.org"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            validated_results=[
                ValidatedResult(
                    validated_facts=["AI is important", "AI is growing"],
                    rejected_facts=[],
                ),
            ],
        )
        result = await summarize(state)

        assert result.final_report is not None
        assert result.final_report.summary == "AI is an important field of study."
        assert result.final_report.sources == ["nature.com", "arxiv.org"]

    @pytest.mark.asyncio
    @patch("research_swarm.agents.summarizer.invoke_messages")
    @patch("research_swarm.agents.summarizer.trace_agent")
    async def test_summarize_no_validated_facts(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.summarizer import summarize
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({
            "summary": "No information available.",
            "sources": [],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")  # no validated_results
        result = await summarize(state)

        assert result.final_report is not None
        assert result.final_report.summary == "No information available."

    @pytest.mark.asyncio
    @patch("research_swarm.agents.summarizer.invoke_messages")
    @patch("research_swarm.agents.summarizer.trace_agent")
    async def test_summarize_handles_missing_sources(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.summarizer import summarize
        from research_swarm.graph.state import (
            ResearchState,
            ValidatedResult,
        )

        mock_invoke.return_value = json.dumps({"summary": "A summary."})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            validated_results=[
                ValidatedResult(validated_facts=["fact 1"], rejected_facts=[]),
            ],
        )
        result = await summarize(state)

        assert result.final_report is not None
        assert result.final_report.summary == "A summary."
        assert result.final_report.sources == []


class TestJudgeAgent:
    """Tests for the judge agent."""

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_sets_high_score(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import (
            ResearchPlan,
            ResearchReport,
            ResearchState,
        )

        mock_invoke.return_value = json.dumps({
            "score": 85,
            "needs_research": False,
            "missing_topics": [],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            plan=ResearchPlan(goal="AI", research_questions=["Q1"]),
            final_report=ResearchReport(summary="Great report", sources=["src"]),
        )
        result = await judge(state)

        assert result.judge_score == 85

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_clamps_score_to_100(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({"score": 150})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")
        result = await judge(state)

        assert result.judge_score == 100

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_clamps_negative_score_to_0(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({"score": -50})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")
        result = await judge(state)

        assert result.judge_score == 0

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_handles_no_report(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({
            "score": 30,
            "needs_research": True,
            "missing_topics": ["topic A"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", final_report=None)
        result = await judge(state)

        assert result.judge_score == 30

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_handles_missing_score(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")
        result = await judge(state)

        assert result.judge_score == 0

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_captures_missing_topics(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({
            "score": 60,
            "needs_research": True,
            "missing_topics": ["ethics", "regulation"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")
        result = await judge(state)

        assert result.judge_score == 60
