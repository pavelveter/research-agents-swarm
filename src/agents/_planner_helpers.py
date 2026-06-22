"""Internal helpers for the Planner agent.s
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from llm.client import invoke_messages
from utils import safe_json

logger = logging.getLogger(__name__)

_MIN_KEYWORD_OVERLAP = 2


def _build_search_packet(missing_topics: list[str]) -> dict[str, Any]:
    """T17: Package judge's ``missing_topics`` into a focused search packet.

    Extracts keywords (>3 chars) and infers source-type hints (regulatory,
    clinical, technical, academic) so the planner receives actionable input,
    not raw topic strings.
    """
    topics = [t for t in missing_topics if not str(t).startswith("[SYSTEM]")]
    if not topics:
        return {}
    keywords: list[str] = []
    for t in topics[:5]:
        words = [w.lower() for w in str(t).split() if len(w) > 3]
        keywords.extend(words)
    unique_keywords = list(dict.fromkeys(keywords))[:15]

    # Source-type hints — always include academic fallbacks for deep-dive queries
    text = " ".join(topics).lower()
    hints: list[str] = [
        "site:ssrn.com",
        "site:arxiv.org",
        "filetype:pdf",
        "technical whitepaper",
    ]
    if any(w in text for w in ("regulation", "act", "law", "compliance", "guideline")):
        hints.append("regulatory")
    if any(w in text for w in ("clinical", "trial", "drug", "patient", "disease")):
        hints.append("clinical")
    if any(w in text for w in ("algorithm", "model", "benchmark", "architecture")):
        hints.append("technical")
    if any(w in text for w in ("study", "research", "paper", "review", "meta")):
        hints.append("academic")

    return {
        "keywords": unique_keywords,
        "source_hints": hints,
        "topics": topics,
    }


async def _adversarial_challenge(
    plan_data: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    """T15: Devil's Advocate step — generate 3 challenge questions.

    Returns a dict with keys: ``"passed"`` (bool), ``"challenges"`` (list[str]),
    ``"risk_flags"`` (list[str]). If ``passed=False``, the plan should be refined.
    """
    questions = plan_data.get("research_questions", [])
    if not questions:
        return {"passed": True, "challenges": [], "risk_flags": []}

    challenge_prompt = (
        "You are a Devil's Advocate for research planning. Review the proposed "
        "research questions and identify weaknesses. Generate exactly 3 challenge "
        "questions covering: (1) data-bias risk — could the evidence be skewed by "
        "source selection or recency bias?, (2) alternative academic views — what "
        "contrary position or framework is NOT being investigated?, (3) untested "
        "assumptions — which core assumption in the questions might be false?. "
        "Return ONLY valid JSON with keys: 'passed' (true if plan is robust, false "
        "if it needs refinement), 'challenges' (list of 3 strings), 'risk_flags' "
        "(list of specific weaknesses to address)."
    )

    messages = [
        SystemMessage(content=challenge_prompt),
        HumanMessage(
            content=(
                f"Query: {query}\n"
                f"Research Questions:\n"
                + "\n".join(f"- {q}" for q in questions)
            )
        ),
    ]

    try:
        response = await invoke_messages(messages, max_tokens=300, temperature=0.4)
        result = safe_json(response.content)
        passed = bool(result.get("passed", True))
        challenges = [str(c) for c in result.get("challenges", [])]
        risk_flags = [str(r) for r in result.get("risk_flags", [])]
        if not passed:
            logger.info(
                "Adversarial challenge: plan FAILED — risks=%s",
                ", ".join(risk_flags[:3]) if risk_flags else "none",
            )
        else:
            logger.info("Adversarial challenge: plan passed")
        return {"passed": passed, "challenges": challenges, "risk_flags": risk_flags}
    except Exception as exc:
        logger.warning("Adversarial challenge LLM call failed: %s", exc)
        return {"passed": True, "challenges": [], "risk_flags": []}


def _enforce_traceability(questions: list[str], topics: list[str]) -> list[str]:
    """T17: Verify at least 2 of 3 refined questions share keywords with topics."""
    topic_words: set[str] = set()
    for t in topics:
        topic_words.update(w.lower() for w in str(t).split() if len(w) > 3)
    if not topic_words:
        return questions
    traceable: list[str] = []
    for q in questions:
        q_words = set(w.lower() for w in q.split() if len(w) > 3)
        if len(q_words & topic_words) >= _MIN_KEYWORD_OVERLAP:
            traceable.append(q)
    return traceable if len(traceable) >= 2 else questions
