"""Fact checker agent — eliminated hardcoded slicing bugs."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchState, ValidatedResult
from research_swarm.llm.client import invoke_messages
from research_swarm.observability.langfuse import trace_agent

logger = logging.getLogger(__name__)

FACTCHECK_SYSTEM = (
    "You are a critical, zero-trust Technical Fact Checker. Validate each evidence item "
    "against technical reality. Kill marketing exaggerations, non-existent models (e.g., GPT-5), "
    "and invalid pricing claims. Every fact MUST contain concrete verifiable metrics or "
    "direct references to active tools (Cursor, Copilot, Windsurf, Claude Code).\n"
    "If an evidence item mentions speculative or future tech as currently available, "
    "you MUST put it into 'rejected_facts'.\n"
    'Return ONLY valid JSON: { "validated_facts": ["fact"], "rejected_facts": ["fact"] }'
)


async def fact_check(state: ResearchState) -> ResearchState:
    """Validate gathered evidence."""
    with trace_agent(
        "fact_checker", input_data={"evidence_count": len(state.search_results)}
    ) as tracer:
        evidence_flat: list[str] = []
        for sr in state.search_results:
            evidence_flat.extend(sr.evidence)

        logger.info("Fact checking | total_evidence_items=%s", len(evidence_flat))

        if not evidence_flat:
            logger.warning("No evidence to validate")
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            return state

        # Берем плоский список улик (ограничим топ-20, как и было)
        evidence_block = "\n".join(f"- {item}" for item in evidence_flat[:20])

        # Модифицируем HumanMessage: даем четкий контекст, К ЧЕМУ эти факты вообще относятся
        messages = [
            SystemMessage(content=FACTCHECK_SYSTEM),
            HumanMessage(
                content=(
                    f"Research Goal/Topic: {state.query}\n\n"
                    f"Evaluate if the following evidence items are relevant, cohesive, "
                    f"and free of internal contradictions relative to the topic.\n"
                    f"Evidence Items:\n{evidence_block}"
                )
            ),
        ]

        raw = await invoke_messages(messages)
        parsed = _safe_json(raw)

        state.validated_results.append(
            ValidatedResult(
                validated_facts=[str(x) for x in parsed.get("validated_facts", [])],
                rejected_facts=[str(x) for x in parsed.get("rejected_facts", [])],
            )
        )

        logger.info(
            "Fact check done | validated=%s rejected=%s",
            len(state.validated_results[-1].validated_facts),
            len(state.validated_results[-1].rejected_facts),
        )

        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "validated_count": len(state.validated_results[-1].validated_facts),
                    "rejected_count": len(state.validated_results[-1].rejected_facts),
                }
            )
    return state


def _safe_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from LLM: %s", text)
        return {}
