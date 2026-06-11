from __future__ import annotations

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

JUDGE_SYSTEM = (
    "You are a cynical, hardcore AI Infrastructure Judge. Evaluate the report strictly "
    "for technical, architectural, and production-grade validity. Reject high-level business "
    "fluff, market growth percentages ($B), and marketing metrics.\n\n"
    "Component scoring guidelines:\n"
    "- Coverage (0-30): Explicit coverage of specific models (Claude 3.5 Sonnet, GPT-4o), frameworks, and legal bounds.\n"
    "- Evidence Quality (0-20): Real benchmarks (SWE-bench, HumanEval) vs marketing claims.\n"
    "- Source Diversity (0-20): Academic papers, CVE databases, post-mortems, and engineering blogs.\n"
    "- Technical Depth (0-15): Low-level mechanics (context window retrieval, RAG strategies, multi-agent state management).\n"
    "- Completeness (0-15): Zero tolerance for missing failure modes, risks, and license vectors.\n\n"
    "If the report contains commercial boilerplate text without architectural diagrams/patterns descriptions, "
    "tank the Technical Depth score below 5. Demand specific missing details in 'missing_topics'.\n\n"
    "Return ONLY valid JSON with this exact shape:\n"
    '{ "coverage_score": 0, "evidence_score": 0, "source_score": 0, "depth_score": 0, '
    '"completeness_score": 0, "score": 0, "needs_research": true, "missing_topics": [], '
    '"strengths": [], "weaknesses": [], "reasoning": "..." }'
)


def _extract_score_from_prose(text: str) -> int | None:
    match = re.search(r"score[:\s*]*(\d{1,3})", text, flags=re.IGNORECASE)
    if not match:
        return None
    return max(0, min(100, int(match.group(1))))


def _parse_judge_response(raw: str, state: ResearchState) -> dict[str, Any]:
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
    """Evaluate the report with weighted component scoring.

    Produces 5 component scores (0-30/0-20/0-20/0-15/0-15), a total score,
    missing topics for targeted research, and stores all on state.
    """
    with trace_agent(
        "judge",
        input_data={
            "has_report": state.final_report is not None,
            "iteration": state.iteration,
        },
    ) as tracer:
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
            SystemMessage(content=JUDGE_SYSTEM),
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
            "Judging report | iteration=%s evidence_items=%s",
            state.iteration,
            evidence_count,
        )
        raw = await invoke_messages(messages)
        parsed = _parse_judge_response(raw, state)

        # Extract component scores
        coverage = max(0, min(30, int(parsed.get("coverage_score", 0))))
        evidence_s = max(0, min(20, int(parsed.get("evidence_score", 0))))
        source_s = max(0, min(20, int(parsed.get("source_score", 0))))
        depth = max(0, min(15, int(parsed.get("depth_score", 0))))
        completeness = max(0, min(15, int(parsed.get("completeness_score", 0))))

        # Total score: use component sum as authoritative.
        # Only fall back to the LLM-provided "score" field if no
        # component scores were provided (sum is 0 — LLM didn't comply).
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
            logger.info("Missing topics: %s", state.missing_topics)

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
