from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.domain import get_domain
from graph.state import ResearchState
from llm.client import invoke_messages
from observability.langfuse import trace_agent
from utils import safe_json, render_prompt

logger = logging.getLogger(__name__)

_MIN_TOPIC_SIMILARITY = 0.25  # lower to reduce false rejections of domain-specific topics

# Keywords that indicate an AI/tech query (if absent, strip tech few-shot from prompt)
_AI_KEYWORDS = frozenset([
    "ai", "artificial intelligence", "machine learning", "llm", "gpt",
    "chatgpt", "copilot", "claude", "gemini", "deep learning",
    "neural network", "transformer", "large language model", "cursor",
    "windsurf", "openai", "anthropic", "coding assistant", "code assistant",
    "ml model", "foundation model", "diffusion model", "stable diffusion",
    "langchain", "langgraph", "qdrant", "pinecone", "vector database",
    "embedding", "rag", "retrieval augmented", "fine-tuning", "fine tuning",
])


def _query_is_ai_related(query: str) -> bool:
    """Return True if query contains AI/tech keywords that warrant domain-specific examples."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in _AI_KEYWORDS)


def _strip_tech_few_shot(prompt: str) -> str:
    """Remove the Technology (generic) calibration block for non-AI queries."""
    # Remove the Technology calibration block from the few-shot section
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
    """Filter out missing topics that are semantically unrelated to any research question.

    Uses Qdrant embeddings to compute cosine similarity. Topics with max similarity
    below ``_MIN_TOPIC_SIMILARITY`` are dropped as off-domain hallucinations.
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
                sum(a * b for a, b in zip(topic_vec, qv))
                for qv in question_vectors
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
    """Safely decode judge response into structured dataset."""
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


async def judge(state: ResearchState) -> ResearchState:
    """Evaluate the report with dynamic temporal awareness and weighted component scoring.

    Strips hardcoded model indices to maintain production validity across years.
    """
    with trace_agent(
        "judge",
        input_data={
            "has_report": state.final_report is not None,
            "iteration": state.iteration,
        },
    ) as tracer:
        # Extract precise system time to build sliding window of relevance
        now = datetime.date.today()
        current_context_time = now.strftime("%B %Y")
        current_year = now.year

        # Fully dynamic prompt without hardcoded specific models
        domain = get_domain(state.query)
        evidence_hint = "\n".join(f"- {t}" for t in domain.evidence_types)
        judge_system_prompt = render_prompt(
            "judge_judge_system_prompt.jinja",
            current_context_time=current_context_time,
            current_year=current_year,
            evidence_types=evidence_hint,
        )

        # Strip AI-specific few-shot examples for non-AI queries (P0-4)
        if not _query_is_ai_related(state.query):
            judge_system_prompt = _strip_tech_few_shot(judge_system_prompt)
            logger.info("Query has no AI keywords — stripped Technology calibration block from judge prompt")

        report_text = state.final_report.summary if state.final_report else ""
        sources = (
            "\n".join(f"- {s}" for s in state.final_report.sources[:20])
            if state.final_report
            else "(none)"
        )
        questions = (
            "\n".join(f"- {q}" for q in state.plan.research_questions)
            if state.plan
            else "(none)"
        )
        evidence_count = sum(len(sr.evidence) for sr in state.search_results)
        provider_info = (
            (
                f"Search provider: {state.search_provider_used or 'none'}\n"
                f"Providers tried: {state.search_providers_tried or ['none']}\n"
                f"Retrieval failed: {state.retrieval_failed}\n"
                f"Retrieval failure reason: {state.retrieval_failure_reason or 'N/A'}\n"
            )
            if state.search_providers_tried
            else ""
        )

        messages = [
            SystemMessage(content=judge_system_prompt),
            HumanMessage(
                content=(
                    f"Iteration: {state.iteration}\n"
                    f"Evidence items gathered: {evidence_count}\n"
                    f"{provider_info}"
                    "Report:\n"
                    f"{report_text}\n\n"
                    "Sources:\n"
                    f"{sources}\n\n"
                    "Research questions:\n"
                    f"{questions}"
                )
            ),
        ]
        logger.info(
            "Judging report | iteration=%s evidence_items=%s context_time=%s",
            state.iteration,
            evidence_count,
            current_context_time,
        )

        # Domain-aware temperature: higher strictness → lower temperature
        judge_temp = max(0.0, 1.0 - domain.strictness)
        raw = await invoke_messages(messages, max_tokens=800, temperature=0)
        parsed = _parse_judge_response(raw, state)

        # Domain-aware component maxes from scoring_weights (each weight × 100)
        w = domain.scoring_weights
        max_cov = int(w.get("coverage", 0.30) * 100)
        max_ev = int(w.get("evidence", 0.20) * 100)
        max_src = int(w.get("sources", 0.20) * 100)
        max_dep = int(w.get("depth", 0.15) * 100)
        max_cmp = int(w.get("completeness", 0.15) * 100)

        coverage = max(0, min(max_cov, int(parsed.get("coverage_score", 0))))
        evidence_s = max(0, min(max_ev, int(parsed.get("evidence_score", 0))))
        source_s = max(0, min(max_src, int(parsed.get("source_score", 0))))
        depth = max(0, min(max_dep, int(parsed.get("depth_score", 0))))
        completeness = max(0, min(max_cmp, int(parsed.get("completeness_score", 0))))

        computed_total = coverage + evidence_s + source_s + depth + completeness
        if computed_total > 0:
            state.judge_score = max(0, min(100, computed_total))
        else:
            llm_score = int(parsed.get("score", 0))
            state.judge_score = max(0, min(100, llm_score))

        # --- Hard guard rails: deterministic post-parse score caps ---
        sources_count = len(state.final_report.sources) if state.final_report else 0
        total_facts = sum(len(sr.evidence) for sr in state.search_results)

        if total_facts < 10:
            state.judge_score = min(state.judge_score, 70)
            state.missing_topics = [
                str(t) for t in state.missing_topics
            ] + ["[SYSTEM] Insufficient evidence: fewer than 10 facts retrieved"]
            logger.info(
                "Guard rail: score capped at 70 (total_evidence_items=%s < 10)",
                total_facts,
            )
        elif sources_count < 5:
            state.judge_score = min(state.judge_score, 75)
            state.missing_topics = [
                str(t) for t in state.missing_topics
            ] + ["[SYSTEM] Insufficient sources: fewer than 5 unique sources"]
            logger.info(
                "Guard rail: score capped at 75 (sources=%s < 5)",
                sources_count,
            )

        state.coverage_score = coverage
        state.evidence_score = evidence_s
        state.source_score = source_s
        state.depth_score = depth
        state.completeness_score = completeness

        needs_research = bool(parsed.get("needs_research", False))
        state.missing_topics = [str(t) for t in parsed.get("missing_topics", [])]
        state.strengths = [str(s) for s in parsed.get("strengths", [])]
        state.weaknesses = [str(w) for w in parsed.get("weaknesses", [])]
        state.reasoning = str(parsed.get("reasoning", ""))

        logger.info(
            "Judge score=%s | components: cov=%s/%s ev=%s/%s src=%s/%s "
            "dep=%s/%s cmp=%s/%s | needs_research=%s missing_topics=%s",
            state.judge_score,
            coverage, max_cov,
            evidence_s, max_ev,
            source_s, max_src,
            depth, max_dep,
            completeness, max_cmp,
            needs_research,
            len(state.missing_topics),
        )
        if state.missing_topics:
            logger.info("Missing topics generated by judge: %s", state.missing_topics)

        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "score": state.judge_score,
                    "coverage": coverage,
                    "evidence": evidence_s,
                    "source": source_s,
                    "depth": depth,
                    "completeness": completeness,
                    "needs_research": needs_research,
                    "missing_topics": state.missing_topics,
                    "strengths": state.strengths,
                    "weaknesses": state.weaknesses,
                    "reasoning": state.reasoning,
                    "iteration": state.iteration,
                }
            )
    return state
