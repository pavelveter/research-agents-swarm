"""Searcher agent — production-grade version using SearchOrchestrator fallback chain."""

from __future__ import annotations

import hashlib
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchState, SearchResult
from research_swarm.llm.client import invoke_messages

# Импортируем правильный оркестратор вместо мертвого mcp_client
from research_swarm.search.orchestrator import get_orchestrator
from research_swarm.utils import safe_json

logger = logging.getLogger(__name__)

FORMAT_SYSTEM = (
    "You are a research evidence formatter. You receive raw search results.\n"
    "Your ONLY job is to extract key findings into a JSON object with an 'evidence' key.\n"
    "Each item inside 'evidence' MUST be a list containing exactly two strings:\n"
    "['fact or key finding', 'source name or URL'].\n\n"
    "Example output:\n"
    "{\n"
    "  'evidence': [\n"
    "    ['AI assistants boost productivity by 25%', 'https://example.com']\n"
    "  ]\n"
    "}"
)


async def search(state: ResearchState) -> ResearchState:
    """Execute search query via orchestrator fallback chain and guarantee evidence extraction."""
    # Обрабатываем топ-1 недостающую тему для этого шага агента
    current_question = (
        state.missing_topics[0]
        if state.missing_topics
        else (
            state.plan.research_questions[0]
            if state.plan and state.plan.research_questions
            else state.query
        )
    )

    # Оптимизируем поисковый запрос с помощью LLM (сжимаем до ключевых слов)
    search_query = await _optimize_query_with_llm(current_question)
    logger.info(
        "Optimized search term: '%s' (was: '%s')", search_query, current_question
    )

    # Получаем глобальный оркестратор
    orchestrator = get_orchestrator()

    # Вызываем поиск по цепочке Tavily -> Brave -> SerpAPI
    # Метод возвращает tuple[list[SearchResultItem], dict]
    raw_hits, meta = await orchestrator.search(query=search_query, max_results=5)
    logger.info("Orchestrator results | raw_items=%s meta=%s", len(raw_hits), meta)

    evidence_list: list[str] = []

    if raw_hits:
        # Сериализуем SearchResultItem в текстовый блок для LLM
        raw_txt_entries = [
            f"Title: {hit.title}\nContent: {hit.snippet}\nURL: {hit.url}"
            for hit in raw_hits
        ]

        messages = [
            SystemMessage(content=FORMAT_SYSTEM),
            HumanMessage(
                content=f"Question: {current_question}\nRaw Results:\n"
                + "\n\n".join(raw_txt_entries)
            ),
        ]

        try:
            raw_response = await invoke_messages(messages)
            parsed = safe_json(raw_response)

            raw_evidence = (
                parsed.get("evidence", []) if isinstance(parsed, dict) else []
            )
            for item in raw_evidence:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    evidence_list.append(f"{item[0]} ({item[1]})")
                elif isinstance(item, str):
                    evidence_list.append(item)
        except Exception as exc:
            logger.warning("LLM formatting failed: %s. Using raw fallback.", exc)

    # КРИТИЧЕСКИЙ ФОЛБЕК: Спасаем граф от zero-evidence стоппера, если LLM выдала мусор
    if not evidence_list and raw_hits:
        logger.warning(
            "LLM returned empty or invalid evidence. Applying RAW fallback for %s items.",
            len(raw_hits),
        )
        for i, hit in enumerate(raw_hits[:10]):
            source = hit.url or hit.title or "Web Search"
            evidence_list.append(f"Discovery {i+1}: {hit.snippet.strip()} ({source})")

    # Дедупликация собранных данных относительно глобального стейта
    kept_evidence, new_count = _deduplicate_evidence(evidence_list, state)

    # Аппендим в историю, сохраняя результаты прошлых итераций
    state.search_results.append(
        SearchResult(question_id=current_question, evidence=kept_evidence)
    )

    state.new_evidence_count = new_count
    state.new_evidence_found = new_count > 0

    logger.info(
        "Search node done | evidence_items=%s new_evidence=%s (count=%s)",
        len(kept_evidence),
        state.new_evidence_found,
        new_count,
    )
    return state


def _optimize_search_query(question: str) -> str:
    """Strip fluff from the query to make it friendly for search providers."""
    clean = question.strip().rstrip("?").lower()
    if "traction in 2024" in clean:
        return "top ai coding assistants 2024 key features traction"
    if "adoption" in clean:
        return "developer adoption trends ai coding assistants 2024 2025"
    if "capabilities" in clean or "multi-file" in clean:
        return (
            "ai coding assistants multi file editing autonomous bug fixing capabilities"
        )
    if "productivity" in clean or "quality" in clean:
        return "ai coding assistants metrics productivity code quality statistics"
    if "challenges" in clean or "limitation" in clean:
        return "ai coding assistants security vulnerabilities accuracy limitations"
    if "workflows" in clean or "toolchains" in clean:
        return "integrate ai coding assistants software toolchain workflow enterprise"

    words = [w for w in question.split() if len(w) > 2]
    return " ".join(words[:8])


def _deduplicate_evidence(
    evidence: list[str], state: ResearchState
) -> tuple[list[str], int]:
    kept: list[str] = []
    known = set(state.known_evidence_hashes)
    new_item_count = 0

    for item in evidence:
        normalized = " ".join(item.lower().split())
        ev_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        if ev_hash not in known:
            kept.append(item)
            known.add(ev_hash)
            new_item_count += 1

    state.known_evidence_hashes = sorted(known)
    return kept, new_item_count


async def _optimize_query_with_llm(question: str) -> str:
    """Degrade and compress human questions into search-friendly keywords using LLM."""
    system_prompt = (
        "You are a search query optimizer. Compress human questions into 3-6 raw keywords.\n"
        "Strip question words (what, how, which), fluff, and punctuation.\n"
        "Output ONLY the raw keywords, lowercase, no quotes."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}"),
    ]

    try:
        raw_response = await invoke_messages(messages, max_tokens=20, temperature=0.1)
        cleaned = raw_response.strip().lower().replace('"', "").replace("'", "")
        if cleaned and len(cleaned.split()) <= 10:
            return cleaned
    except Exception as exc:
        logger.warning(
            "LLM query degradation failed: %s. Using naive word-cut fallback.", exc
        )

    words = [w for w in question.strip().rstrip("?").lower().split() if len(w) > 3]
    return " ".join(words[:6])
