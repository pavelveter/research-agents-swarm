from __future__ import annotations

import pytest

from research_swarm.graph.state import (
    JudgeResult,
    ResearchPlan,
    ResearchReport,
)
from research_swarm.graph.state import AgentIO


class TestSchemataModels:
    """Tests for the schemas.py models (inheriting from AgentIO)."""

    def test_research_plan_extends_agentio(self) -> None:
        """ResearchPlan in schemas.py should be a subclass of AgentIO."""
        assert issubclass(ResearchPlan, AgentIO)

    def test_research_plan_has_correct_fields(self) -> None:
        rp = ResearchPlan(goal="Test goal", research_questions=["Q1", "Q2"])
        assert rp.goal == "Test goal"
        assert rp.research_questions == ["Q1", "Q2"]

    def test_research_plan_serialization(self) -> None:
        rp = ResearchPlan(goal="Test", research_questions=["Q"])
        data = rp.model_dump()
        assert data == {"goal": "Test", "research_questions": ["Q"]}

    def test_research_report_extends_agentio(self) -> None:
        """ResearchReport in schemas.py should be a subclass of AgentIO."""
        assert issubclass(ResearchReport, AgentIO)

    def test_research_report_has_correct_fields(self) -> None:
        rr = ResearchReport(summary="A summary", sources=["src1", "src2"])
        assert rr.summary == "A summary"
        assert rr.sources == ["src1", "src2"]

    def test_research_report_serialization(self) -> None:
        rr = ResearchReport(summary="Test", sources=["s1"])
        data = rr.model_dump()
        assert data == {"summary": "Test", "sources": ["s1"]}

    def test_judge_result_extends_agentio(self) -> None:
        """JudgeResult in schemas.py should be a subclass of AgentIO."""
        assert issubclass(JudgeResult, AgentIO)

    def test_judge_result_has_correct_fields(self) -> None:
        jr = JudgeResult(score=80, needs_research=True, missing_topics=["t"])
        assert jr.score == 80
        assert jr.needs_research is True
        assert jr.missing_topics == ["t"]

    def test_judge_result_serialization(self) -> None:
        jr = JudgeResult(score=90, needs_research=False, missing_topics=[])
        data = jr.model_dump()
        assert data["score"] == 90
        assert data["needs_research"] is False
        assert data["missing_topics"] == []
        assert data["strengths"] == []
        assert data["weaknesses"] == []
        assert data["reasoning"] == ""

    def test_judge_result_all_fields_required(self) -> None:
        """JudgeResult in schemas.py requires score, needs_research, missing_topics."""
        with pytest.raises(Exception):
            JudgeResult(score=50)  # type: ignore[arg-type]

    def test_judge_result_optional_fields_default(self) -> None:
        """strengths, weaknesses, reasoning have empty defaults."""
        jr = JudgeResult(score=50, needs_research=True, missing_topics=["t"])
        assert jr.strengths == []
        assert jr.weaknesses == []
        assert jr.reasoning == ""
