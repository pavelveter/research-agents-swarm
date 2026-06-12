"""Summarizer agent — re-architected to pull clean context from Qdrant vector storage."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchReport, ResearchState
from research_swarm.llm.client import invoke_messages
from research_swarm.logging_config import preview
from research_swarm.memory.vector_storage import get_memory_bank
from research_swarm.observability.langfuse import trace_agent
from research_swarm.utils import safe_json, render_prompt

logger = logging.getLogger(__name__)

# Prompts moved to jinja


async def summarize(state: ResearchState) -> ResearchState:
    """Produce final comprehensive report pulling clean context out of Qdrant memory layer."""
    with trace_agent("summarizer", input_data={"query": state.query}) as tracer:

        # Copy previous report before overwriting (for incremental rewrite loop)
        # Use shallow copy for payloads to keep Qdrant dedup intact
        if state.final_report is not None:
            state.previous_report = state.final_report

        # Connect our vector memory bank
        memory = get_memory_bank()

        # Extract only the most important and deduplicated
        facts = await memory.retrieve_context(query=state.query, limit=50)
        logger.info(
            "Summarizer retrieved %d top unique semantic facts from Qdrant", len(facts)
        )

        if not facts:
            facts_block = "(No verified facts collected across graph iterations.)"
        else:
            facts_block = "\n".join(f"- {f}" for f in facts)

        # Build prompt with rewrite instruction if we have a previous report
        system_content = render_prompt("summarizer_summarizer_system.jinja")
        human_extra = ""
        if state.previous_report is not None:
            system_content += (
                "\n\nINCREMENTAL REDESIGN: This is a rewrite iteration. "
                "Preserve strong sections from the previous report. "
                "Address weaknesses and missing topics using the new facts below. "
                "Do NOT restart from scratch — extend and improve the existing report."
            )
            human_extra = (
                f"\n\nPrevious Report (preserve strong parts, improve weak parts):\n{state.previous_report.summary}\n"
                f"Previous Sources: {state.previous_report.sources}"
            )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(
                content=(
                    f"Global Research Topic: {state.query}\n"
                    f"Top Validated Contextual Facts:\n{facts_block}"
                    f"{human_extra}"
                )
            ),
        ]

        raw = await invoke_messages(messages)
        parsed = safe_json(raw)

        # Fallback: if parsed JSON is empty, merge previous report with new facts via a secondary LLM call
        if not parsed or (not parsed.get("summary") and not parsed.get("sources")):
            logger.warning(
                "Summarizer returned empty JSON, attempting fallback merge with previous report"
            )
            if state.previous_report and state.previous_report.summary:
                merge_messages = [
                    SystemMessage(
                        content=(
                            "You are a report merger. Combine the previous report with new facts. "
                            "Preserve strong sections from the previous report. Add new facts where they fit naturally. "
                            "Return ONLY valid JSON with keys 'summary' and 'sources' and 'citations'."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Previous Report:\n{state.previous_report.summary}\n\n"
                            f"Previous Sources: {', '.join(str(s) for s in state.previous_report.sources)}\n\n"
                            f"New Facts:\n{facts_block}"
                        )
                    ),
                ]
                raw = await invoke_messages(merge_messages, temperature=0.2)
                parsed = safe_json(raw)
            else:
                logger.warning("No previous report available for fallback merge")

        # Write report back to graph state
        state.final_report = ResearchReport(
            summary=str(parsed.get("summary", "")),
            sources=[str(s) for s in parsed.get("sources", [])],
        )

        # Log citations if present in parsed response
        citations = parsed.get("citations", [])
        if citations:
            logger.info("Report includes %d citations with fact-to-source mappings", len(citations))

        logger.info(
            "Summary ready | chars=%s sources=%s",
            len(state.final_report.summary),
            len(state.final_report.sources),
        )
        logger.info("Summary preview: %s", preview(state.final_report.summary))

        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "summary_length": len(state.final_report.summary),
                    "sources_count": len(state.final_report.sources),
                }
            )

    return state
