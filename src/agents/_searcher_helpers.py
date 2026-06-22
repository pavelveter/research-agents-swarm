"""Internal helpers for the Searcher agent."""

from __future__ import annotations

import datetime
import hashlib
import logging
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import EvidenceItem, ResearchState
from llm.client import invoke_messages
from utils import render_prompt

logger = logging.getLogger(__name__)

_ANALYTICAL_TERMS: frozenset[str] = frozenset(
    {
        "theory",
        "analysis",
        "mechanism",
        "framework",
        "model",
        "evidence",
        "study",
        "research",
        "method",
        "approach",
        "finding",
        "result",
        "hypothesis",
        "experiment",
        "observation",
        "implication",
        "critique",
        "review",
        "comparison",
        "evaluation",
        "methodology",
        "taxonomy",
        "synthesis",
        "paradigm",
        "principle",
        "correlation",
        "causation",
        "phenomenon",
        "dynamic",
        "structure",
        "function",
        "evolution",
        "behavior",
        "cognitive",
        "psychological",
        "sociological",
        "anthropological",
    }
)


def _substance_score(snippet: str) -> float:
    """Score a search snippet for analytical substance (0.0–1.0).

    Rewards length (>120 chars) and presence of analytical vocabulary.
    Penalises listicle/pun markers.
    """
    text = snippet.lower()
    score = 0.0
    # Length bonus: 120+ chars gets 0.35, scales down below that
    if len(snippet) >= 120:
        score += 0.35
    else:
        score += (len(snippet) / 120.0) * 0.35
    # Analytical term bonus: each match adds 0.08, capped at 0.40
    term_hits = sum(1 for t in _ANALYTICAL_TERMS if t in text)
    score += min(term_hits * 0.08, 0.40)
    # Fluff penalty: listicle/pun markers
    fluff_markers = [
        "top 10",
        "top 5",
        "best of",
        "jokes",
        "puns",
        "funny",
        "hilarious",
        "listicle",
        "you won't believe",
    ]
    if any(m in text for m in fluff_markers):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _extract_source_name(url_or_name: str) -> str:
    """Extract a human-readable source name from a URL or raw name string."""
    raw = url_or_name.strip()
    # If it's a URL, extract domain
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            parsed = urlparse(raw)
            domain = parsed.netloc or parsed.hostname or ""
            # Strip www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain or raw
        except Exception:
            return raw
    return raw


def _deduplicate_evidence(
    evidence: list[EvidenceItem], state: ResearchState
) -> tuple[list[EvidenceItem], int]:
    """Deduplicate ``EvidenceItem`` list by hashing fact text.

    Returns the deduped list (in encounter order) and a count of newly
    added items (i.e. those whose hash was not previously known).
    """
    kept: list[EvidenceItem] = []
    known = set(state.known_evidence_hashes)
    new_item_count = 0

    for item in evidence:
        normalized = " ".join(item.fact.lower().split())
        ev_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        if ev_hash not in known:
            kept.append(item)
            known.add(ev_hash)
            new_item_count += 1

    state.known_evidence_hashes = sorted(known)
    return kept, new_item_count


async def _optimize_query_with_llm(question: str) -> str:
    """Transform a requirement into a specific technical search query anchored to today's date.

    Uses the planner Jinja prompt with current month/year context so the
    query is temporally relevant. Falls back to appending the current year
    to salient keywords if the LLM call fails.
    """
    # Dynamically calculate current date so agent knows current time
    now = datetime.date.today()
    current_year = now.year
    current_month_year = now.strftime("%B %Y")

    system_prompt = render_prompt(
        "searcher_system_prompt.jinja",
        current_month_year=current_month_year,
        current_year=current_year,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic to optimize: {question}"),
    ]

    try:
        llm_resp = await invoke_messages(messages, max_tokens=25, temperature=0.1)
        cleaned = llm_resp.content.strip().lower().replace('"', "").replace("'", "")
        if cleaned and len(cleaned.split()) <= 12:
            # T7: Sniper mode — append academic-source hints for analytical queries
            question_lower = question.lower()
            if any(
                t in question_lower
                for t in (
                    "analysis",
                    "framework",
                    "guidance",
                    "regulation",
                    "policy",
                    "clinical",
                    "legal",
                    "compliance",
                    "mechanism",
                    "methodology",
                    "theory",
                    "evaluation",
                )
            ):
                return f"{cleaned} analysis framework"
            return cleaned
    except Exception as exc:
        logger.warning("LLM query temporal optimization failed: %s", exc)

    # Smart fallback: if model fails, dynamically append current year to words
    words = [w for w in question.strip().rstrip("?").lower().split() if len(w) > 3]
    return f"{' '.join(words[:5])} {current_year}"
