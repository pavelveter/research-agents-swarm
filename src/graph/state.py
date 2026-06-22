from __future__ import annotations

from pydantic import BaseModel


class AgentIO(BaseModel):
    "Base Pydantic model reused across agent contracts."


class EvidenceItem(BaseModel):
    """Structured evidence with source traceability."""
    fact: str
    source: str = ""
    url: str = ""
    iteration_added: int = 0


class SearchResult(BaseModel):
    question_id: str
    evidence: list[EvidenceItem] = []


class ValidatedResult(BaseModel):
    validated_facts: list[str]
    rejected_facts: list[str]


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
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasoning: str = ""


class ResearchState(BaseModel):
    query: str
    plan: ResearchPlan | None = None
    search_results: list[SearchResult] = []
    validated_results: list[ValidatedResult] = []
    final_report: ResearchReport | None = None
    judge_score: int = 0
    iteration: int = 0
    max_iterations: int = 5
    previous_score: int | None = None
    score_delta: int | None = None
    missing_topics: list[str] = []
    no_progress: bool = False
    stop_reason: str = ""
    new_evidence_found: bool = True
    new_evidence_count: int = 0
    known_evidence_hashes: list[str] = []
    # Weighted component scores from judge
    coverage_score: int = 0
    evidence_score: int = 0
    source_score: int = 0
    depth_score: int = 0
    completeness_score: int = 0
    strengths: list[str] = []
    weaknesses: list[str] = []
    reasoning: str = ""
    # --- Retrieval architecture fields ---
    retrieval_failed: bool = False
    retrieval_failure_reason: str = ""
    search_providers_tried: list[str] = []
    search_provider_used: str = ""
    evidence_quality: list[dict] = []
    # Search mode: "full" (iteration 0) or "targeted" (subsequent)
    search_mode: str = "full"
    # Previous report for incremental rewrite loop
    previous_report: ResearchReport | None = None
    # Substance quality flag — set by searcher when fluff dominates results
    retrieval_quality_low: bool = False
    # Source diversity flag — set by summarizer when sources collapse to single domain
    source_diversity_low: bool = False
    # Langfuse session_id — generated per workflow run in main.py/news_sender.py
    # and propagated to all agent observations so they share one session.
    session_id: str | None = None
