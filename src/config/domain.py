"""Domain-aware configuration — maps domain slugs to strictness, scoring weights,
and evidence expectations so the research swarm adapts without code edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from config.settings import get_settings


@dataclass(frozen=True)
class DomainConfig:
    """Per-domain knobs that feed into prompt templates and agent behaviour."""

    slug: str
    label: str = ""

    # Strictness (0..1) — maps to inverse temperature for planner/judge/summarizer
    strictness: float = 0.5

    # Component weights for the Judge rubric (sum should ≈ 1.0)
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "coverage": 0.30,
        "evidence": 0.20,
        "sources": 0.20,
        "depth": 0.15,
        "completeness": 0.15,
    })

    # What counts as quality evidence in this domain (injected into Judge prompt)
    evidence_types: list[str] = field(default_factory=lambda: [
        "peer-reviewed publications",
        "official documentation",
        "benchmarks & datasets",
        "technical post-mortems",
    ])

    # Optional inference synthesis block — injected into the summarizer system prompt
    # when the domain is explicitly configured (via RESEARCH_DOMAIN env var).
    # Left empty for "general" and auto-detected domains — only explicit config triggers it.
    inference_synthesis: str = ""


# ── Pre-defined domains ────────────────────────────────────────

_DOMAINS: dict[str, DomainConfig] = {
    "ai": DomainConfig(
        slug="ai",
        label="AI & ML",
        strictness=0.7,
        evidence_types=[
            "peer-reviewed papers (arXiv, NeurIPS, ICML)",
            "official model cards & system cards",
            "benchmark leaderboards (SWE-bench, MMLU, LiveCodeBench)",
            "technical post-mortems & engineering blogs",
        ],
    ),
    "policy": DomainConfig(
        slug="policy",
        label="Policy & Regulation",
        strictness=0.6,
        evidence_types=[
            "government publications & legislative texts",
            "regulatory impact assessments",
            "think-tank policy briefs",
            "official implementation timelines",
        ],
        scoring_weights={
            "coverage": 0.30,
            "evidence": 0.15,
            "sources": 0.25,
            "depth": 0.15,
            "completeness": 0.15,
        },
        inference_synthesis=(
            "## Domain Synthesis Authority\n"
            "When a missing topic is not addressed by provided facts, you may synthesize "
            "by inference using established policy-analysis principles:\n"
            "- Trace regulatory lineage: if a topic references a directive or act, note "
            "its jurisdictional scope and implementation timeline even without explicit facts.\n"
            "- Identify stakeholder positions: infer likely positions of industry, civil society, "
            "and regulators based on the policy's stated objectives.\n"
            "- Surface precedent: reference analogous regulatory frameworks from other jurisdictions "
            "when the gap concerns implementation guidance.\n"
            "Clearly mark synthesized content with '[Inferred]' so readers can distinguish "
            "factual evidence from analytical inference."
        ),
    ),
    "biotech": DomainConfig(
        slug="biotech",
        label="Biotech & Medicine",
        strictness=0.8,
        evidence_types=[
            "clinical trial registrations & results",
            "peer-reviewed medical journals",
            "FDA/EMA/WHO guidance documents",
            "meta-analyses & systematic reviews",
        ],
        scoring_weights={
            "coverage": 0.25,
            "evidence": 0.30,
            "sources": 0.20,
            "depth": 0.10,
            "completeness": 0.15,
        },
        inference_synthesis=(
            "## Domain Synthesis Authority\n"
            "When a missing topic is not addressed by provided facts, you may synthesize "
            "by inference using established biomedical principles:\n"
            "- Apply mechanism-of-action reasoning: if facts describe a drug class or pathway, "
            "infer likely safety/efficacy considerations from pharmacological principles.\n"
            "- Reference regulatory frameworks: note which agency (FDA/EMA/WHO) governs the topic "
            "and what approval pathway (IND, NDA, 510(k), PMA) would apply.\n"
            "- Identify evidence hierarchy gaps: if only preclinical or Phase I data is available, "
            "flag the absence of confirmatory Phase III or real-world evidence.\n"
            "Clearly mark synthesized content with '[Inferred]' so readers can distinguish "
            "factual evidence from analytical inference."
        ),
    ),
    "finance": DomainConfig(
        slug="finance",
        label="Finance & Economics",
        strictness=0.65,
        evidence_types=[
            "SEC filings & regulatory disclosures",
            "central bank statements & minutes",
            "market data from Bloomberg/Reuters/FRED",
            "audited financial statements",
        ],
        scoring_weights={
            "coverage": 0.25,
            "evidence": 0.25,
            "sources": 0.25,
            "depth": 0.10,
            "completeness": 0.15,
        },
    ),
    "general": DomainConfig(
        slug="general",
        label="General Research",
        strictness=0.5,
        evidence_types=[
            "peer-reviewed publications",
            "official documentation",
            "benchmarks & datasets",
            "expert analysis & commentary",
        ],
    ),
}


# ── Detection & lookup ─────────────────────────────────────────

def _detect_domain(query: str) -> str:
    """Heuristic domain detection from query keywords."""
    q = query.lower()
    if any(kw in q for kw in ("ai", "ml", "llm", "gpt", "claude", "model", "neural",
                                "transformer", "benchmark", "swe-bench", "coding assistant")):
        return "ai"
    if any(kw in q for kw in ("regulation", "act", "law", "legislation", "eu", "policy",
                                "compliance", "gdpr", "directive")):
        return "policy"
    if any(kw in q for kw in ("vaccin", "clinical", "trial", "drug", "disease", "covid",
                                "patient", "therapy", "treatment", "biotech")):
        return "biotech"
    if any(kw in q for kw in ("stock", "market", "fed", "interest rate", "inflation",
                                "gdp", "revenue", "financial", "investment", "sec")):
        return "finance"
    return "general"


def get_domain(query: str | None = None) -> DomainConfig:
    """Return the DomainConfig for the current query.

    Priority: explicit ``RESEARCH_DOMAIN`` env var > keyword detection > 'general'.
    """
    settings = get_settings()
    explicit = settings.research_domain.strip().lower() if settings.research_domain else ""
    slug = explicit if explicit in _DOMAINS else _detect_domain(query or "")
    return _DOMAINS.get(slug, _DOMAINS["general"])


@lru_cache(maxsize=8)
def get_domain_cached(slug: str) -> DomainConfig:
    """Look up a domain by slug (cached)."""
    return _DOMAINS.get(slug, _DOMAINS["general"])
