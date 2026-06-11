from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentIO(BaseModel):
    "Base Pydantic model reused across agent contracts."


class SearchResult(BaseModel):
    question_id: str
    evidence: list[str]


class ValidatedResult(BaseModel):
    validated_facts: list[str]
    rejected_facts: list[str]


class ResearchPlan(BaseModel):
    goal: str
    research_questions: list[str]


class ResearchReport(BaseModel):
    summary: str
    sources: list[str]


class JudgeResult(BaseModel):
    score: int
    needs_research: bool
    missing_topics: list[str]


class ResearchState(BaseModel):
    query: str
    plan: ResearchPlan | None = None
    search_results: list[SearchResult] = []
    validated_results: list[ValidatedResult] = []
    final_report: ResearchReport | None = None
    judge_score: int = 0
