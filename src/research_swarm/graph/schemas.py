from __future__ import annotations

from research_swarm.graph.state import AgentIO


class ResearchPlan(AgentIO):
    goal: str
    research_questions: list[str]


class ResearchReport(AgentIO):
    summary: str
    sources: list[str]


class JudgeResult(AgentIO):
    score: int
    needs_research: bool
    missing_topics: list[str]
