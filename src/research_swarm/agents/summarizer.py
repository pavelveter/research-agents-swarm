from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.llm.client import invoke_messages
from research_swarm.logging_config import preview
from research_swarm.observability.langfuse import trace_agent
from research_swarm.graph.state import ResearchReport, ResearchState

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM = (
    "You are a research summarizer. Produce a concise summary using ONLY the "
    "validated facts. Include inline citations like [1], [2] referencing the "
    "provided sources. Return ONLY valid JSON: { \"summary\": \"...\", "
    "\"sources\": [\"source string\"], \"validated_facts\": [\"fact\"] }."
)


async def summarize(state: ResearchState) -> ResearchState:
    """Produce final report from validated facts."""
    with trace_agent(
        "summarizer", input_data={"validated_results": len(state.validated_results)}
    ) as tracer:
        facts: list[str] = []
        for vr in state.validated_results:
            facts.extend(vr.validated_facts)
        logger.info("Summarizing | validated_facts=%s", len(facts))
        facts_block = "\n".join(f"- {f}" for f in facts[:40]) or "(no facts)"
        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM),
            HumanMessage(content=f"Validated facts:\n{facts_block}"),
        ]
        raw = await invoke_messages(messages)
        parsed = _safe_json(raw)
        state.final_report = ResearchReport(
            summary=str(parsed.get("summary", "")),
            sources=[str(s) for s in parsed.get("sources", [])],
        )
        logger.info(
            "Summary ready | chars=%s sources=%s",
            len(state.final_report.summary),
            len(state.final_report.sources),
        )
        logger.info("Summary preview: %s", preview(state.final_report.summary))
        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "summary": state.final_report.summary,
                    "sources_count": len(state.final_report.sources),
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
