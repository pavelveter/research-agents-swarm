from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchReport, ResearchState
from research_swarm.llm.client import invoke_messages
from research_swarm.logging_config import preview
from research_swarm.observability.langfuse import trace_agent
from research_swarm.utils import safe_json

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM = (
    "You are a hard-boiled Technical Core Report Compiler. Combine the validated facts into "
    "a compressed engineering digest. \n"
    "CRITICAL CRITERIA:\n"
    "1. Do NOT mention what is missing, weak, or not provided. Never write phrases like 'No specific details are provided'.\n"
    "2. If data on a topic doesn't exist in the validated facts, omit the topic entirely.\n"
    "3. Focus exclusively on raw architecture, benchmarks, configuration flags, and explicit version upgrades.\n"
    'Return ONLY valid JSON: { "summary": "...", "sources": ["..."] }'
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
        facts_block = "\n".join(f"- {f}" for f in facts) or "(no facts)"
        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM),
            HumanMessage(content=f"Validated facts:\n{facts_block}"),
        ]
        raw = await invoke_messages(messages)
        parsed = safe_json(raw)
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
