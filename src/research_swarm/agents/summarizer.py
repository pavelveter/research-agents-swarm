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

        # Подключаем наш векторный банк памяти
        memory = get_memory_bank()

        # Вытаскиваем только самое важное и дедуплицированное
        facts = await memory.retrieve_context(query=state.query, limit=25)
        logger.info(
            "Summarizer retrieved %d top unique semantic facts from Qdrant", len(facts)
        )

        if not facts:
            facts_block = "(No verified facts collected across graph iterations.)"
        else:
            facts_block = "\n".join(f"- {f}" for f in facts)

        messages = [
            SystemMessage(content=render_prompt("summarizer_summarizer_system.jinja")),
            HumanMessage(
                content=(
                    f"Global Research Topic: {state.query}\n"
                    f"Top Validated Contextual Facts:\n{facts_block}"
                )
            ),
        ]

        raw = await invoke_messages(messages)
        parsed = safe_json(raw)

        # Записываем отчет обратно в стейт графа
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
                    "summary_length": len(state.final_report.summary),
                    "sources_count": len(state.final_report.sources),
                }
            )

    return state
