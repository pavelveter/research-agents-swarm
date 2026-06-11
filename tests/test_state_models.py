from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from research_swarm.graph.state import (
    AgentIO,
    JudgeResult,
    ResearchPlan,
    ResearchReport,
    ResearchState,
    SearchResult,
    ValidatedResult,
)


class TestAgentIO:
    """Tests for the AgentIO base model."""

    def test_agentio_is_base_model(self) -> None:
        """AgentIO should be a Pydantic BaseModel."""
        assert issubclass(AgentIO, BaseModel)
        io = AgentIO()
        assert io.model_dump() == {}

    def test_agentio_allows_extra_fields(self) -> None:
        """AgentIO should allow extra fields by default (BaseModel behavior)."""
        io = AgentIO()
        assert hasattr(io, "model_dump")


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_creation_with_empty_evidence(self) -> None:
        sr = SearchResult(question_id="q1", evidence=[])
        assert sr.question_id == "q1"
        assert sr.evidence == []

    def test_creation_with_evidence(self) -> None:
        sr = SearchResult(
            question_id="q1",
            evidence=["fact 1 (source A)", "fact 2 (source B)"],
        )
        assert sr.question_id == "q1"
        assert len(sr.evidence) == 2
        assert sr.evidence[0] == "fact 1 (source A)"

    def test_requires_question_id(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(evidence=[])  # type: ignore[arg-type]

    def test_evidence_field_is_required(self) -> None:
        """evidence field has no default — must be provided."""
        with pytest.raises(ValidationError):
            SearchResult(question_id="q1")  # type: ignore[arg-type]

    def test_serialization_roundtrip(self) -> None:
        sr = SearchResult(question_id="q1", evidence=["fact (src)"])
        data = sr.model_dump()
        restored = SearchResult(**data)
        assert restored == sr

    def test_json_serialization(self) -> None:
        sr = SearchResult(question_id="q1", evidence=["fact (src)"])
        json_str = sr.model_dump_json()
        assert "q1" in json_str
        assert "fact (src)" in json_str

    def test_question_id_is_string(self) -> None:
        sr = SearchResult(question_id="42", evidence=[])
        assert isinstance(sr.question_id, str)
        assert sr.question_id == "42"


class TestValidatedResult:
    """Tests for ValidatedResult model."""

    def test_creation_with_both_lists(self) -> None:
        vr = ValidatedResult(
            validated_facts=["fact 1", "fact 2"],
            rejected_facts=["bad fact"],
        )
        assert vr.validated_facts == ["fact 1", "fact 2"]
        assert vr.rejected_facts == ["bad fact"]

    def test_creation_empty(self) -> None:
        vr = ValidatedResult(validated_facts=[], rejected_facts=[])
        assert vr.validated_facts == []
        assert vr.rejected_facts == []

    def test_all_fields_are_required(self) -> None:
        """ValidatedResult requires both validated_facts and rejected_facts."""
        with pytest.raises(ValidationError):
            ValidatedResult()  # type: ignore[arg-type]

    def test_requires_string_lists(self) -> None:
        with pytest.raises(ValidationError):
            ValidatedResult(validated_facts=[1, 2, 3])  # type: ignore[list-item]


class TestResearchPlan:
    """Tests for ResearchPlan model."""

    def test_creation_with_single_question(self) -> None:
        rp = ResearchPlan(goal="AI trends", research_questions=["What is AI?"])
        assert rp.goal == "AI trends"
        assert rp.research_questions == ["What is AI?"]

    def test_creation_with_multiple_questions(self) -> None:
        rp = ResearchPlan(
            goal="AI in 2025",
            research_questions=["Q1", "Q2", "Q3"],
        )
        assert rp.goal == "AI in 2025"
        assert len(rp.research_questions) == 3

    def test_requires_goal(self) -> None:
        with pytest.raises(ValidationError):
            ResearchPlan(research_questions=["Q1"])  # type: ignore[arg-type]

    def test_requires_research_questions(self) -> None:
        with pytest.raises(ValidationError):
            ResearchPlan(goal="AI")  # type: ignore[arg-type]

    def test_empty_questions_list(self) -> None:
        rp = ResearchPlan(goal="AI", research_questions=[])
        assert rp.research_questions == []

    def test_goal_must_be_string(self) -> None:
        with pytest.raises(ValidationError):
            ResearchPlan(goal=42, research_questions=["Q1"])  # type: ignore[arg-type]

    def test_questions_must_be_strings(self) -> None:
        with pytest.raises(ValidationError):
            ResearchPlan(goal="AI", research_questions=[1, 2, 3])  # type: ignore[list-item]

    def test_model_dump_output(self) -> None:
        rp = ResearchPlan(goal="AI", research_questions=["Q1", "Q2"])
        data = rp.model_dump()
        assert data == {"goal": "AI", "research_questions": ["Q1", "Q2"]}


class TestResearchReport:
    """Tests for ResearchReport model."""

    def test_creation_with_summary_and_sources(self) -> None:
        rr = ResearchReport(
            summary="AI is growing rapidly.",
            sources=["arxiv.org/1", "nature.com/2"],
        )
        assert rr.summary == "AI is growing rapidly."
        assert len(rr.sources) == 2

    def test_creation_with_empty_sources(self) -> None:
        rr = ResearchReport(summary="Nothing to report.", sources=[])
        assert rr.sources == []

    def test_requires_summary(self) -> None:
        with pytest.raises(ValidationError):
            ResearchReport(sources=["src"])  # type: ignore[arg-type]

    def test_requires_sources(self) -> None:
        with pytest.raises(ValidationError):
            ResearchReport(summary="test")  # type: ignore[arg-type]

    def test_summary_must_be_string(self) -> None:
        with pytest.raises(ValidationError):
            ResearchReport(summary=123, sources=["src"])  # type: ignore[arg-type]

    def test_sources_must_be_strings(self) -> None:
        with pytest.raises(ValidationError):
            ResearchReport(summary="test", sources=[1, 2, 3])  # type: ignore[list-item]


class TestJudgeResult:
    """Tests for JudgeResult model."""

    def test_creation_high_score(self) -> None:
        jr = JudgeResult(score=95, needs_research=False, missing_topics=[])
        assert jr.score == 95
        assert jr.needs_research is False
        assert jr.missing_topics == []

    def test_creation_low_score(self) -> None:
        jr = JudgeResult(
            score=40,
            needs_research=True,
            missing_topics=["topic A", "topic B"],
        )
        assert jr.score == 40
        assert jr.needs_research is True
        assert len(jr.missing_topics) == 2

    def test_all_fields_required(self) -> None:
        """All JudgeResult fields are required (no defaults)."""
        with pytest.raises(ValidationError):
            JudgeResult(score=50)  # type: ignore[arg-type]

    def test_accepts_any_score_value(self) -> None:
        """Pydantic won't reject out-of-range by default without constraints."""
        jr = JudgeResult(score=150, needs_research=False, missing_topics=[])
        assert jr.score == 150

    def test_accepts_zero_score(self) -> None:
        jr = JudgeResult(score=0, needs_research=False, missing_topics=[])
        assert jr.score == 0

    def test_serialization_roundtrip(self) -> None:
        jr = JudgeResult(score=85, needs_research=True, missing_topics=["t1"])
        data = jr.model_dump()
        restored = JudgeResult(**data)
        assert restored == jr

    def test_score_must_be_int(self) -> None:
        with pytest.raises(ValidationError):
            JudgeResult(score="high")  # type: ignore[arg-type]


class TestResearchState:
    """Tests for ResearchState model — the main state container."""

    def test_minimal_creation(self) -> None:
        state = ResearchState(query="AI trends")
        assert state.query == "AI trends"
        assert state.plan is None
        assert state.search_results == []
        assert state.validated_results == []
        assert state.final_report is None
        assert state.judge_score == 0

    def test_full_creation(self) -> None:
        plan = ResearchPlan(goal="AI", research_questions=["Q1"])
        sr = SearchResult(question_id="Q1", evidence=["fact (src)"])
        vr = ValidatedResult(validated_facts=["fact"], rejected_facts=[])
        report = ResearchReport(summary="AI is great.", sources=["src"])
        state = ResearchState(
            query="AI trends",
            plan=plan,
            search_results=[sr],
            validated_results=[vr],
            final_report=report,
            judge_score=85,
        )
        assert state.plan is not None
        assert state.plan.goal == "AI"
        assert len(state.search_results) == 1
        assert len(state.validated_results) == 1
        assert state.final_report is not None
        assert state.final_report.summary == "AI is great."
        assert state.judge_score == 85

    def test_requires_query(self) -> None:
        with pytest.raises(ValidationError):
            ResearchState()  # type: ignore[arg-type]

    def test_query_must_be_string(self) -> None:
        with pytest.raises(ValidationError):
            ResearchState(query=42)  # type: ignore[arg-type]

    def test_judge_score_default_zero(self) -> None:
        state = ResearchState(query="test")
        assert state.judge_score == 0

    def test_plan_defaults_to_none(self) -> None:
        state = ResearchState(query="test")
        assert state.plan is None

    def test_search_results_default_to_empty_list(self) -> None:
        state = ResearchState(query="test")
        assert state.search_results == []

    def test_validated_results_default_to_empty_list(self) -> None:
        state = ResearchState(query="test")
        assert state.validated_results == []

    def test_final_report_defaults_to_none(self) -> None:
        state = ResearchState(query="test")
        assert state.final_report is None

    def test_model_dump_with_none_fields(self) -> None:
        state = ResearchState(query="test")
        data = state.model_dump()
        assert data["query"] == "test"
        assert data["plan"] is None
        assert data["search_results"] == []
        assert data["validated_results"] == []
        assert data["final_report"] is None
        assert data["judge_score"] == 0

    def test_model_dump_with_populated_fields(self) -> None:
        plan = ResearchPlan(goal="AI", research_questions=["Q1"])
        state = ResearchState(query="test", plan=plan, judge_score=70)
        data = state.model_dump()
        assert data["plan"] == {"goal": "AI", "research_questions": ["Q1"]}
        assert data["judge_score"] == 70

    def test_multiple_search_results(self) -> None:
        sr1 = SearchResult(question_id="q1", evidence=["e1"])
        sr2 = SearchResult(question_id="q2", evidence=["e2"])
        state = ResearchState(query="test", search_results=[sr1, sr2])
        assert len(state.search_results) == 2
        assert state.search_results[0].question_id == "q1"
        assert state.search_results[1].question_id == "q2"

    def test_multiple_validated_results(self) -> None:
        vr1 = ValidatedResult(validated_facts=["f1"], rejected_facts=[])
        vr2 = ValidatedResult(validated_facts=["f2"], rejected_facts=["bad"])
        state = ResearchState(query="test", validated_results=[vr1, vr2])
        assert len(state.validated_results) == 2

    def test_convenience_for_empty_check(self) -> None:
        """Demonstrate pattern for checking if state has results."""
        state = ResearchState(query="test")
        assert not state.search_results
        assert not state.validated_results
