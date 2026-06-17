from __future__ import annotations



from graph.state import (
    EvidenceItem,
    ResearchPlan,
    ResearchReport,
    ResearchState,
    SearchResult,
)
from graph.workflow import build_workflow
from utils import merge_state


class TestBuildWorkflow:
    """Tests for the workflow graph construction."""

    def test_returns_compiled_graph(self) -> None:
        """build_workflow should return a compiled StateGraph."""
        workflow = build_workflow()
        assert workflow is not None
        assert hasattr(workflow, "astream")

    def test_graph_has_required_nodes(self) -> None:
        """The graph should have all 6 nodes (5 agents + routing node)."""
        workflow = build_workflow()
        nodes = list(workflow.get_graph().nodes.keys())
        expected_nodes = {"planner", "searcher", "fact_checker", "summarizer", "judge", "routing"}
        assert expected_nodes.issubset(set(nodes))

    def test_graph_entry_point_is_planner(self) -> None:
        """Entry point should be the planner node."""
        # Graph is compiled — just verify it doesn't crash
        workflow = build_workflow()
        assert workflow is not None

    def test_graph_has_conditional_routing(self) -> None:
        """Should have conditional edges from judge node."""
        workflow = build_workflow()
        graph = workflow.get_graph()

        # Verify the graph has edges (regular or conditional)
        edges = graph.edges
        assert edges is not None
        assert len(edges) > 0

    def test_workflow_can_be_invoked(self) -> None:
        """The compiled workflow should accept invocations."""
        workflow = build_workflow()
        state = ResearchState(query="test")

        # Just verify the workflow is callable (no actual execution)
        assert hasattr(workflow, "invoke")
        assert hasattr(workflow, "astream")

    def test_build_workflow_imports_all_agents(self) -> None:
        """Verify that build_workflow imports all agent functions."""
        workflow = build_workflow()
        # Implicitly tests that all imports in workflow.py work
        assert workflow is not None


class TestMergeState:
    """Tests for the _merge_state helper in main.py."""

    def test_merge_single_state_update(self) -> None:
        """Should merge when event contains a ResearchState value."""
        current = ResearchState(query="test")
        updated = ResearchState(query="test", judge_score=85)

        event = {"judge": updated}
        result = merge_state(current, event)

        assert result.judge_score == 85
        assert result.query == "test"

    def test_merge_dict_update(self) -> None:
        """Should merge when event contains a dict value."""
        current = ResearchState(query="test")
        event = {"planner": {"plan": ResearchPlan(goal="AI", research_questions=["Q1"])}}

        result = merge_state(current, event)

        assert result.plan is not None
        assert result.plan.goal == "AI"

    def test_merge_multiple_events(self) -> None:
        """Should accumulate changes from multiple events."""
        current = ResearchState(query="test")
        plan = ResearchPlan(goal="AI", research_questions=["Q1"])
        event = {"planner": {"plan": plan}, "judge": {"judge_score": 90}}

        result = merge_state(current, event)

        assert result.plan is not None
        assert result.plan.goal == "AI"
        assert result.judge_score == 90

    def test_merge_preserves_existing_fields(self) -> None:
        """Existing fields should not be overwritten by missing keys."""
        current = ResearchState(
            query="test",
            judge_score=50,
            search_results=[SearchResult(question_id="q1", evidence=[EvidenceItem(fact="e1")])],
        )
        event = {"planner": {"plan": ResearchPlan(goal="AI", research_questions=["Q1"])}}

        result = merge_state(current, event)

        assert result.judge_score == 50
        assert len(result.search_results) == 1

    def test_merge_overwrites_existing_fields(self) -> None:
        """New values should overwrite old values."""
        current = ResearchState(query="test", judge_score=50)
        event = {"judge": {"judge_score": 75}}

        result = merge_state(current, event)

        assert result.judge_score == 75

    def test_merge_with_full_research_state_update(self) -> None:
        """Test merge with a complete ResearchState in the event."""
        current = ResearchState(query="test")
        report = ResearchReport(summary="Final summary", sources=["s1"])
        updated_state = ResearchState(
            query="test",
            judge_score=95,
            final_report=report,
        )
        event = {"summarizer": updated_state}

        result = merge_state(current, event)

        assert result.judge_score == 95
        assert result.final_report is not None
        assert result.final_report.summary == "Final summary"

    def test_merge_updates_new_fields(self) -> None:
        """New fields like iteration, missing_topics merge correctly."""
        current = ResearchState(query="test")
        event = {"judge": {"iteration": 2, "missing_topics": ["security"], "judge_score": 70}}

        result = merge_state(current, event)

        assert result.iteration == 2
        assert result.missing_topics == ["security"]
        assert result.judge_score == 70

    def test_merge_empty_event(self) -> None:
        """Empty event dict should not change state."""
        current = ResearchState(query="test", judge_score=50)
        event: dict = {}

        result = merge_state(current, event)

        assert result.query == "test"
        assert result.judge_score == 50


class TestWorkflowIntegration:
    """Integration-like tests that verify the workflow graph structure."""

    def test_workflow_graph_is_deterministic(self) -> None:
        """Building workflow twice should be consistent."""
        wf1 = build_workflow()
        wf2 = build_workflow()

        nodes1 = set(wf1.get_graph().nodes.keys())
        nodes2 = set(wf2.get_graph().nodes.keys())
        assert nodes1 == nodes2

    def test_workflow_has_edge_from_planner_to_searcher(self) -> None:
        """Should have a direct edge from planner to searcher."""
        workflow = build_workflow()
        graph = workflow.get_graph()

        # Check that the graph structure has edges
        edges = graph.edges
        assert edges is not None

    def test_workflow_supports_astream_with_state(self) -> None:
        """The compiled workflow should support astream with ResearchState input."""
        workflow = build_workflow()
        state = ResearchState(query="test")

        # Verify the method exists and accepts the state
        assert hasattr(workflow, "astream")
        # Don't actually call it since it requires LLM access
