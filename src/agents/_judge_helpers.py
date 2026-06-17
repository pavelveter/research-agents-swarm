"""Internal helpers for the Judge agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from graph.state import ResearchState
from memory.vector_storage import get_memory_bank
from utils import safe_json

logger = logging.getLogger(__name__)

_MIN_TOPIC_SIMILARITY = (
    0.25  # lower to reduce false rejections of domain-specific topics
)

_AI_KEYWORDS: frozenset[str] = frozenset(
    {
        "ai",
        "artificial intelligence",
        "machine learning",
        "llm",
        "gpt",
        "chatgpt",
        "copilot",
        "claude",
        "gemini",
        "deep learning",
        "neural network",
        "transformer",
        "large language model",
        "cursor",
        "windsurf",
        "openai",
        "anthropic",
        "coding assistant",
        "code assistant",
        "ml model",
        "foundation model",
        "diffusion model",
        "stable diffusion",
        "langchain",
        "langgraph",
        "qdrant",
        "pinecone",
        "vector database",
        "embedding",
        "rag",
        "retrieval augmented",
        "fine-tuning",
        "fine tuning",
    }
)


def _query_is_ai_related(query: str) -> bool:
    """Return True if query contains AI/tech keywords that warrant domain-specific examples."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in _AI_KEYWORDS)


def _strip_tech_few_shot(prompt: str) -> str:
    """Remove the Technology (generic) calibration block for non-AI queries."""
    prompt = re.sub(
        r"Technology \(generic\):\n- .*?\n- .*?\n\n?",
        "",
        prompt,
        flags=re.DOTALL,
    )
    return prompt


async def _filter_missing_topics(
    topics: list[str],
    research_questions: list[str],
) -> list[str]:
    """Drop missing topics that are semantically unrelated to any research question.

    Uses Qdrant embeddings (Ollama nomic-embed-text, normalized vectors) to
    compute cosine similarity. Topics whose max similarity to any research
    question is below ``_MIN_TOPIC_SIMILARITY`` are dropped as off-domain
    hallucinations. ``[SYSTEM]`` flags always pass through unchanged.
    """
    try:
        memory = get_memory_bank()
        question_vectors = [await memory._embed_string(q) for q in research_questions]

        filtered: list[str] = []
        for topic in topics:
            if str(topic).startswith("[SYSTEM]"):
                filtered.append(topic)  # operational flags always pass through
                continue
            topic_vec = await memory._embed_string(topic)
            # Cosine similarity via dot product (nomic-embed-text vectors are normalized)
            max_sim = max(
                sum(a * b for a, b in zip(topic_vec, qv)) for qv in question_vectors
            )
            if max_sim >= _MIN_TOPIC_SIMILARITY:
                filtered.append(topic)
            else:
                logger.info(
                    "Filtered off-domain missing topic (max_sim=%.3f < %.2f): %s",
                    max_sim,
                    _MIN_TOPIC_SIMILARITY,
                    topic[:80],
                )

        dropped = len(topics) - len(filtered)
        if dropped:
            logger.info(
                "Missing topics semantic filter: dropped %d/%d off-domain topics",
                dropped,
                len(topics),
            )
        return filtered
    except Exception as exc:
        logger.warning(
            "Missing topics semantic filter failed (Qdrant/Ollama may be down): %s", exc
        )
        return topics


def _extract_score_from_prose(text: str) -> int | None:
    """Fallback parser to extract numeric score if LLM ignores JSON instructions."""
    match = re.search(r"score[:\s*]*(\d{1,3})", text, flags=re.IGNORECASE)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def _parse_judge_response(raw: str, state: ResearchState) -> dict[str, Any]:
    """Safely decode judge response into a structured dict.

    On non-JSON output, synthesise a stub response that flags missing topics
    so the router can still decide whether to continue researching.
    """
    try:
        return safe_json(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Judge returned non-JSON response; using fallback parse | preview=%r",
            raw[:200],
        )
        missing_topics = (
            list(state.plan.research_questions)
            if state.plan and state.plan.research_questions
            else ["Judge response was not valid JSON"]
        )
        return {
            "score": _extract_score_from_prose(raw) or 0,
            "needs_research": True,
            "missing_topics": missing_topics,
            "strengths": [],
            "weaknesses": ["Judge model returned unstructured prose instead of JSON"],
            "reasoning": raw[:500],
        }
