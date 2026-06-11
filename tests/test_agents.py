from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.utils import safe_json


class TestSafeJson:
    """Tests for the shared safe_json helper used across all agents."""

    @pytest.mark.parametrize("text,expected", [
        ('{"key": "value"}', {"key": "value"}),
        ('```json\n{"key": "value"}\n```', {"key": "value"}),
        ('```\n{"key": "value"}\n```', {"key": "value"}),
        ('{"key": "value"}  \n\t', {"key": "value"}),
        ('[1, 2, 3]', [1, 2, 3]),
        ('{"outer": {"inner": [1, 2]}}', {"outer": {"inner": [1, 2]}}),
        ('````json\n{"a": 1}\n````', {"a": 1}),
        ('## Judge\n{"score": 5, "needs_research": true}', {"score": 5, "needs_research": True}),
    ])
    def test_parses_json_variants(self, text: str, expected) -> None:
        result = safe_json(text)
        assert result == expected

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            safe_json("not json")


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

    @pytest.mark.asyncio
    @patch("research_swarm.agents.planner.invoke_messages")
    @patch("research_swarm.agents.planner.trace_agent")
    async def test_plan_handles_missing_goal(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.planner import plan
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({"research_questions": ["Q1"]})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI trends")
        result = await plan(state)
        assert result.plan is not None
        assert result.plan.goal == "AI trends"

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
    """Tests for the searcher agent — uses SearchOrchestrator (no trace_agent)."""

    def _make_orch_mock(self, results: list, metadata: dict | None = None) -> MagicMock:
        if metadata is None:
            metadata = {
                "providers_tried": ["duckduckgo"],
                "successful_provider": "duckduckgo",
                "all_failed": False,
                "attempts": [{"provider": "duckduckgo", "success": True, "result_count": len(results)}],
            }
        mock = MagicMock()
        mock.search = AsyncMock(return_value=(results, metadata))
        mock.health = MagicMock()
        return mock

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.get_orchestrator")
    @patch("research_swarm.agents.searcher.invoke_messages")
    async def test_full_search_uses_orchestrator(
        self, mock_invoke: AsyncMock, mock_get_orch: MagicMock,
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState
        from research_swarm.search.base import SearchResultItem

        items = [SearchResultItem(title="AI growing", url="https://nature.com",
                                   snippet="AI adoption growing", provider="tavily")]
        mock_get_orch.return_value = self._make_orch_mock(
            results=items,
            metadata={"providers_tried": ["tavily"], "successful_provider": "tavily",
                      "all_failed": False, "attempts": []},
        )
        mock_invoke.return_value = json.dumps({"evidence": [["AI is growing", "nature.com"]]})

        state = ResearchState(query="AI", iteration=0, search_mode="full",
                              plan=ResearchPlan(goal="AI", research_questions=["Q1"]))
        result = await search(state)

        assert len(result.search_results) == 1
        assert result.new_evidence_found is True
        assert result.search_provider_used == "tavily"
        assert result.retrieval_failed is False

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.get_orchestrator")
    @patch("research_swarm.agents.searcher.invoke_messages")
    async def test_targeted_search_for_missing_topics(
        self, mock_invoke: AsyncMock, mock_get_orch: MagicMock,
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchState
        from research_swarm.search.base import SearchResultItem

        items = [SearchResultItem(title="Risk", url="https://s.com",
                                   snippet="Risk", provider="brave")]
        mock_get_orch.return_value = self._make_orch_mock(
            results=items,
            metadata={"providers_tried": ["brave"], "successful_provider": "brave",
                      "all_failed": False, "attempts": []},
        )
        mock_invoke.return_value = json.dumps({"evidence": [["Risk found", "sec-db.com"]]})

        state = ResearchState(query="AI", iteration=1, search_mode="targeted",
                              missing_topics=["security risks", "cost analysis"])
        result = await search(state)

        assert mock_get_orch.return_value.search.call_count == 2
        assert result.new_evidence_found is True

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.get_orchestrator")
    async def test_all_providers_fail_marks_retrieval_failed(
        self, mock_get_orch: MagicMock,
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState

        mock_get_orch.return_value = self._make_orch_mock(
            results=[],
            metadata={
                "providers_tried": ["tavily", "brave", "duckduckgo"],
                "successful_provider": None,
                "all_failed": True,
                "attempts": [
                    {"provider": "tavily", "success": False, "error": "timeout"},
                    {"provider": "brave", "success": False, "error": "timeout"},
                    {"provider": "duckduckgo", "success": False, "error": "empty"},
                ],
            },
        )
        state = ResearchState(query="AI", iteration=0, search_mode="full",
                              plan=ResearchPlan(goal="AI", research_questions=["Q1"]))
        result = await search(state)

        assert result.retrieval_failed is True
        assert result.new_evidence_found is False
        assert "All providers failed" in result.retrieval_failure_reason

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.get_orchestrator")
    @patch("research_swarm.agents.searcher.invoke_messages")
    async def test_search_deduplicates_evidence(
        self, mock_invoke: AsyncMock, mock_get_orch: MagicMock,
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState
        from research_swarm.search.base import SearchResultItem
        import hashlib

        items = [SearchResultItem(title="AI", url="https://n.com",
                                   snippet="AI is growing", provider="tavily")]
        mock_get_orch.return_value = self._make_orch_mock(
            results=items,
            metadata={"providers_tried": ["tavily"], "successful_provider": "tavily",
                      "all_failed": False, "attempts": []},
        )
        mock_invoke.return_value = json.dumps({
            "evidence": [["AI is growing", "nature.com"], ["OLD FACT", "arxiv.org"]],
        })

        old_hash = hashlib.sha256("old fact (arxiv.org)".lower().encode()).hexdigest()[:16]
        state = ResearchState(query="AI", iteration=0, search_mode="full",
                              plan=ResearchPlan(goal="AI", research_questions=["Q1"]),
                              known_evidence_hashes=[old_hash])
        result = await search(state)

        assert result.new_evidence_found is True
        assert result.new_evidence_count == 1

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.get_orchestrator")
    @patch("research_swarm.agents.searcher.invoke_messages")
    async def test_evidence_quality_is_tracked(
        self, mock_invoke: AsyncMock, mock_get_orch: MagicMock,
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState
        from research_swarm.search.base import SearchResultItem

        items = [SearchResultItem(title="AI fact", url="https://e.com",
                                   snippet="AI important", provider="brave")]
        mock_get_orch.return_value = self._make_orch_mock(
            results=items,
            metadata={"providers_tried": ["brave"], "successful_provider": "brave",
                      "all_failed": False, "attempts": []},
        )
        mock_invoke.return_value = json.dumps({"evidence": [["AI is important", "example.com"]]})

        state = ResearchState(query="AI", iteration=0, search_mode="full",
                              plan=ResearchPlan(goal="AI", research_questions=["Q1"]))
        result = await search(state)

        assert len(result.evidence_quality) > 0
        assert result.evidence_quality[0]["provider"] == "brave"
        assert result.evidence_quality[0]["search_mode"] == "full"
        assert "confidence" in result.evidence_quality[0]
        assert "retrieved_at" in result.evidence_quality[0]

    @pytest.mark.asyncio
    @patch("research_swarm.agents.searcher.get_orchestrator")
    async def test_fallback_provider_success(
        self, mock_get_orch: MagicMock,
    ) -> None:
        from research_swarm.agents.searcher import search
        from research_swarm.graph.state import ResearchPlan, ResearchState
        from research_swarm.search.base import SearchResultItem

        items = [SearchResultItem(title="Fallback", url="https://ddg.com",
                                   snippet="Result", provider="duckduckgo")]
        mock_get_orch.return_value = self._make_orch_mock(
            results=items,
            metadata={
                "providers_tried": ["tavily", "duckduckgo"],
                "successful_provider": "duckduckgo",
                "all_failed": False,
                "attempts": [
                    {"provider": "tavily", "success": False, "error": "timeout"},
                    {"provider": "duckduckgo", "success": True, "result_count": 1},
                ],
            },
        )
        state = ResearchState(query="AI", iteration=0, search_mode="full",
                              plan=ResearchPlan(goal="AI", research_questions=["Q1"]))
        result = await search(state)

        assert result.retrieval_failed is False
        assert result.search_provider_used == "duckduckgo"
        assert "tavily" in result.search_providers_tried


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

        state = ResearchState(query="AI", search_results=[
            SearchResult(question_id="q1", evidence=["fact 1", "fact 2"]),
            SearchResult(question_id="q2", evidence=["bad fact"]),
        ])
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
        mock_invoke.assert_not_called()
        assert len(result.validated_results) == 1
        assert result.validated_results[0].validated_facts == []

    @pytest.mark.asyncio
    @patch("research_swarm.agents.fact_checker.invoke_messages")
    @patch("research_swarm.agents.fact_checker.trace_agent")
    async def test_fact_check_handles_missing_fields(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.fact_checker import fact_check
        from research_swarm.graph.state import ResearchState, SearchResult

        mock_invoke.return_value = json.dumps({"validated_facts": ["fact 1"]})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", search_results=[
            SearchResult(question_id="q1", evidence=["fact 1"]),
        ])
        result = await fact_check(state)
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
        from research_swarm.graph.state import ResearchState, ValidatedResult

        mock_invoke.return_value = json.dumps({
            "summary": "AI is an important field of study.",
            "sources": ["nature.com", "arxiv.org"],
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", validated_results=[
            ValidatedResult(validated_facts=["AI is important", "AI is growing"], rejected_facts=[]),
        ])
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

        mock_invoke.return_value = json.dumps({"summary": "No information available.", "sources": []})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")
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
        from research_swarm.graph.state import ResearchState, ValidatedResult

        mock_invoke.return_value = json.dumps({"summary": "A summary."})

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", validated_results=[
            ValidatedResult(validated_facts=["fact 1"], rejected_facts=[]),
        ])
        result = await summarize(state)
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
        from research_swarm.graph.state import ResearchPlan, ResearchReport, ResearchState

        mock_invoke.return_value = json.dumps({
            "score": 85, "needs_research": False, "missing_topics": [],
            "strengths": ["thorough"], "weaknesses": [], "reasoning": "Complete.",
            "coverage_score": 28, "evidence_score": 18, "source_score": 17,
            "depth_score": 12, "completeness_score": 10,
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", plan=ResearchPlan(goal="AI", research_questions=["Q1"]),
                              final_report=ResearchReport(summary="Great report", sources=["src"]))
        result = await judge(state)
        assert result.judge_score == 85
        assert result.coverage_score == 28

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_sets_missing_topics(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({
            "score": 60, "needs_research": True, "missing_topics": ["security", "cost analysis"],
            "strengths": [], "weaknesses": ["missing cost data"], "reasoning": "Needs more data.",
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI")
        result = await judge(state)
        assert result.judge_score == 60
        assert result.missing_topics == ["security", "cost analysis"]

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_clamps_score_to_100(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({"score": 150, "needs_research": False, "missing_topics": []})

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
    async def test_judge_handles_no_report(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({"score": 30, "needs_research": True, "missing_topics": ["topic A"]})

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
    async def test_judge_handles_retrieval_failure(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchState

        mock_invoke.return_value = json.dumps({
            "score": 0, "needs_research": False, "missing_topics": [],
            "strengths": [], "weaknesses": ["No evidence available"],
            "reasoning": "All search providers failed.", "retrieval_failed": True,
        })

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(query="AI", retrieval_failed=True,
                              search_providers_tried=["tavily", "brave", "duckduckgo"], judge_score=0)
        result = await judge(state)
        assert result.judge_score == 0

    @pytest.mark.asyncio
    @patch("research_swarm.agents.judge.invoke_messages")
    @patch("research_swarm.agents.judge.trace_agent")
    async def test_judge_handles_markdown_response(
        self, mock_trace: MagicMock, mock_invoke: AsyncMock
    ) -> None:
        from research_swarm.agents.judge import judge
        from research_swarm.graph.state import ResearchPlan, ResearchReport, ResearchState

        mock_invoke.return_value = (
            "## Judge's Evaluation **Score: 12/100** "
            "Report is too thin for production review."
        )

        mock_tracer = MagicMock()
        mock_tracer.update_observation = MagicMock()
        mock_trace.return_value.__enter__ = MagicMock(return_value=mock_tracer)
        mock_trace.return_value.__exit__ = MagicMock(return_value=False)

        state = ResearchState(
            query="AI",
            plan=ResearchPlan(goal="AI", research_questions=["Q1", "Q2"]),
            final_report=ResearchReport(summary="Thin report", sources=["src"]),
        )
        result = await judge(state)
        assert result.judge_score == 12
        assert result.missing_topics == ["Q1", "Q2"]
        assert "unstructured prose" in result.weaknesses[0]
