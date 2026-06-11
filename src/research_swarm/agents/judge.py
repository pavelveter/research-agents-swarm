from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchState
from research_swarm.llm.client import invoke_messages
from research_swarm.observability.langfuse import trace_agent
from research_swarm.utils import safe_json

logger = logging.getLogger(__name__)


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
        # Извлекаем точное системное время для построения скользящего окна актуальности
        now = datetime.date.today()
        current_context_time = now.strftime("%B %Y")
        current_year = now.year
        cutoff_year = current_year - 2

        # Полностью динамический промпт без хардкода конкретных моделей
        judge_system_prompt = (
            f"You are a cynical, hardcore AI Infrastructure Judge evaluating engineering data in {current_context_time}.\n"
            "Strictly evaluate the report for technical, architectural, and production-grade validity.\n"
            f"Reject historical marketing fluff and baseline tech older than {cutoff_year} as 'current trends'.\n\n"
            "TEMPORAL ANCHORING & VERSIONING:\n"
            f"1. Evaluate all findings relative to the current baseline of {current_context_time}.\n"
            f"2. You expect the report to cover the absolute frontier models, developer tools, and agentic runtimes active in {current_year}.\n"
            "3. Do NOT invent or extrapolate future version numbers (e.g., if a model family is currently at v3, do not demand v5.5 unless it actually exists in the wild).\n"
            "4. Base your missing topics on active engineering ecosystems (e.g., state-of-the-art proprietary and open-source models, agent runtimes, and active IDE extensions).\n\n"
            "Component scoring guidelines:\n"
            "- Coverage (0-30): Explicit coverage of current state-of-the-art agentic runtimes, IDE integrations, and active legal/copyright boundaries.\n"
            "- Evidence Quality (0-20): Real, recent industry-standard benchmarks (e.g., SWE-bench variants, LiveCodeBench, or current leading evals) vs commercial claims.\n"
            f"- Source Diversity (0-20): Academic papers, technical post-mortems, engineering blogs, and CVE databases targeting vulnerabilities discovered up to {current_year}.\n"
            "- Technical Depth (0-15): Low-level mechanics (context window retrieval engineering, chunk overlap strategies, production re-ranking pipelines, multi-agent state persistence).\n"
            "- Completeness (0-15): Deep analysis of operational risks, data poisoning, RCE exploits via tool usage, and enterprise license compliance tools.\n\n"
            "CRITICAL DIRECTIONS FOR MISSING TOPICS:\n"
            "Formulate items in 'missing_topics' as explicit, highly targeted tasks containing specific engineering keywords, "
            f"focusing on gaps relevant to the {current_year} infrastructure landscape.\n"
            "Example format: 'SWE-bench scores for latest frontier models vs leading open-source models'.\n\n"
            "Return ONLY valid JSON with this exact shape:\n"
            '{ "coverage_score": 0, "evidence_score": 0, "source_score": 0, "depth_score": 0, '
            '"completeness_score": 0, "score": 0, "needs_research": true, "missing_topics": [], '
            '"strengths": [], "weaknesses": [], "reasoning": "..." }'
        )

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

        raw = await invoke_messages(messages)
        parsed = _parse_judge_response(raw, state)

        coverage = max(0, min(30, int(parsed.get("coverage_score", 0))))
        evidence_s = max(0, min(20, int(parsed.get("evidence_score", 0))))
        source_s = max(0, min(20, int(parsed.get("source_score", 0))))
        depth = max(0, min(15, int(parsed.get("depth_score", 0))))
        completeness = max(0, min(15, int(parsed.get("completeness_score", 0))))

        computed_total = coverage + evidence_s + source_s + depth + completeness
        if computed_total > 0:
            state.judge_score = max(0, min(100, computed_total))
        else:
            llm_score = int(parsed.get("score", 0))
            state.judge_score = max(0, min(100, llm_score))

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
            "Judge score=%s | components: cov=%s/30 ev=%s/20 src=%s/20 "
            "dep=%s/15 cmp=%s/15 | needs_research=%s missing_topics=%s",
            state.judge_score,
            coverage,
            evidence_s,
            source_s,
            depth,
            completeness,
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
