"""Fact checker agent — hardened with Qdrant vector memory layer."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchState, ValidatedResult
from research_swarm.llm.client import invoke_messages
from research_swarm.memory.vector_storage import get_memory_bank
from research_swarm.observability.langfuse import trace_agent
from research_swarm.utils import render_prompt

logger = logging.getLogger(__name__)

# Prompts moved to jinja


async def fact_check(state: ResearchState) -> ResearchState:
    """Validate ONLY the newest batch of gathered evidence against vector long-term storage."""
    with trace_agent(
        "fact_checker", input_data={"evidence_count": len(state.search_results)}
    ) as tracer:

        if not state.search_results:
            logger.warning("No search results found in state.")
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            return state

        # Берем ТОЛЬКО свежий батч улик из последнего поиска
        latest_search = state.search_results[-1]
        raw_evidence = latest_search.evidence

        if not raw_evidence:
            logger.warning("Latest search batch contains zero evidence items.")
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            return state

        logger.info("Fact checking | incoming_batch_size=%s", len(raw_evidence))

        # Инициализируем наш Qdrant синглтон
        memory = get_memory_bank()

        # ФИКС: Динамический порог схожести. Чем дальше итерация, тем деликатнее фильтруем.
        # Это позволит протиснуть в отчёт близкие, но семантически разные хардкорные факты.
        current_threshold = 0.92 if state.iteration < 2 else 0.86
        logger.info(
            "Fact checking | iteration=%s, dynamic_threshold=%s",
            state.iteration,
            current_threshold,
        )

        # Слой 1: Первичная семантическая фильтрация перед отправкой в LLM
        unique_incoming: list[str] = []
        for item in raw_evidence:
            if await memory._is_semantic_duplicate(item, threshold=current_threshold):
                logger.debug(
                    "Pre-filtered semantic duplicate from batch: %s", item[:50]
                )
                continue
            unique_incoming.append(item)

        if not unique_incoming:
            logger.info(
                "All incoming items filtered out as semantic duplicates. Skipping LLM validation."
            )
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            state.new_evidence_count = 0
            state.new_evidence_found = False
            return state

        # Слой 2: LLM Валидация оставшихся уникальных улик
        evidence_block = "\n".join(f"- {item}" for item in unique_incoming)
        messages = [
            SystemMessage(content=render_prompt("fact_checker_factcheck_system.jinja")),
            HumanMessage(
                content=(
                    f"Task context: Validate new evidence gathered for question: {latest_search.question_id}\n"
                    f"Evidence Items:\n{evidence_block}"
                )
            ),
        ]

        raw = await invoke_messages(messages)
        parsed = _safe_json(raw)

        # Синхронизируем парсинг с форматом твоего state.py (массив строк)
        validated_list = [str(x) for x in parsed.get("validated_facts", [])]
        rejected_list = [str(x) for x in parsed.get("rejected_facts", [])]

        # Слой 3: Финальный коммит чистых фактов в Qdrant
        current_task = latest_search.question_id
        new_stored_count = await memory.upsert_facts(
            facts=validated_list, iteration=state.iteration, task=current_task
        )

        # Сохраняем шаг в историю LangGraph
        state.validated_results.append(
            ValidatedResult(
                validated_facts=validated_list,
                rejected_facts=rejected_list,
            )
        )

        # Управляем флагами роутера на основе РЕАЛЬНО добавленного нового веса
        state.new_evidence_count = new_stored_count
        state.new_evidence_found = new_stored_count > 0

        logger.info(
            "Fact check done | raw_incoming=%s -> unique_filtered=%s | verified=%s -> newly_stored_in_qdrant=%s",
            len(raw_evidence),
            len(unique_incoming),
            len(validated_list),
            new_stored_count,
        )

        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "incoming_count": len(raw_evidence),
                    "after_semantic_filter": len(unique_incoming),
                    "validated_count": len(validated_list),
                    "newly_stored_count": new_stored_count,
                    "rejected_count": len(rejected_list),
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
