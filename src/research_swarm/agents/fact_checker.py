from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.llm.client import invoke_messages
from research_swarm.observability.langfuse import trace_agent
from research_swarm.graph.state import ResearchState, ValidatedResult

logger = logging.getLogger(__name__)

FACTCHECK_SYSTEM = (
    "You are a strict fact checker. Validate each evidence item against the claim. "
    "Mark unsupported or contradictory claims as rejected. "
    "Return ONLY valid JSON: { \"validated_facts\": [\"fact\"], \"rejected_facts\": [\"fact\"] }."
)


async def fact_check(state: ResearchState) -> ResearchState:
    """Validate gathered evidence."""
    with trace_agent(
        "fact_checker", input_data={"evidence_count": len(state.search_results)}
    ) as tracer:
        evidence_flat: list[str] = []
        for sr in state.search_results:
            evidence_flat.extend(sr.evidence)
        logger.info("Fact checking | evidence_items=%s", len(evidence_flat))
        if not evidence_flat:
            logger.warning("No evidence to validate")
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            return state
        evidence_block = "\n".join(
            f"- {item}" for item in evidence_flat[:20]
        )
        messages = [
            SystemMessage(content=FACTCHECK_SYSTEM),
            HumanMessage(content=f"Evidence:\n{evidence_block}"),
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
    return json.loads(cleaned)
