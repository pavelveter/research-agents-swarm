from __future__ import annotations

import pytest

from research_swarm.graph.state import ResearchPlan, ResearchState


def test_research_state_creation() -> None:
    state = ResearchState(query="AI trends")
    assert state.query == "AI trends"
    assert state.plan is None


def test_research_plan_schema() -> None:
    plan = ResearchPlan(goal="AI trends", research_questions=["What is AI?"])
    assert plan.goal == "AI trends"
    assert len(plan.research_questions) == 1
