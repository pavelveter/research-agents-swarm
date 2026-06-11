"""Searcher agent — production-grade version using SearchOrchestrator fallback chain."""

from __future__ import annotations

import datetime
import hashlib
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchState, SearchResult
from research_swarm.llm.client import invoke_messages
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
    # ФИКС: Вместо застревания на missing_topics[0], берём срез топ-3 тем
    topics_to_research = []
    if state.missing_topics:
        topics_to_research = state.missing_topics[:3]
    elif state.plan and state.plan.research_questions:
        topics_to_research = state.plan.research_questions[:1]
    else:
        topics_to_research = [state.query]

    orchestrator = get_orchestrator()
    total_new_count = 0
    all_kept_evidence = []

    # Веерный поиск по критическим лакунам отчёта
    for current_question in topics_to_research:
        search_query = await _optimize_query_with_llm(current_question)
        logger.info(
            "Targeted technical query: '%s' (for topic: '%s')",
            search_query,
            current_question[:40],
        )

        # Расширяем выборку до 7 результатов, чтобы зацепить глубокие доки/блоги
        raw_hits, meta = await orchestrator.search(query=search_query, max_results=7)
        logger.info(
            "Orchestrator fetched %s items for query: %s", len(raw_hits), search_query
        )

        evidence_list: list[str] = []

        if raw_hits:
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

        # Критический фолбек
        if not evidence_list and raw_hits:
            logger.warning("LLM mapping failed, injecting raw snippet fallbacks.")
            for i, hit in enumerate(raw_hits[:4]):
                source = hit.url or hit.title or "Search Engine"
                evidence_list.append(
                    f"Technical Data: {hit.snippet.strip()} ({source})"
                )

        kept_evidence, new_count = _deduplicate_evidence(evidence_list, state)
        total_new_count += new_count
        all_kept_evidence.extend(kept_evidence)

        state.search_results.append(
            SearchResult(question_id=current_question, evidence=kept_evidence)
        )

    state.new_evidence_count = total_new_count
    state.new_evidence_found = total_new_count > 0

    logger.info(
        "Multi-query search step complete | Found %s new unique records across %s paths",
        total_new_count,
        len(topics_to_research),
    )
    return state


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
    """Transform requirements into highly specific technical search queries anchored to the actual execution date."""
    # Динамически вычисляем текущую дату, чтобы агент знал, где он находится
    now = datetime.date.today()
    current_year = now.year

    # Формируем человекочитаемый контекст (например, "June 2026")
    current_month_year = now.strftime("%B %Y")

    system_prompt = (
        "You are an expert infrastructure search query optimizer.\n"
        f"CRITICAL TEMPORAL CONTEXT: The current time is {current_month_year}. You are investigating the absolute "
        f"latest data, breakthroughs, critical incidents, and benchmarks up to {current_month_year}.\n"
        "Convert the input topic into a highly targeted, keyword-dense search query for a technical web search.\n"
        "Preserve exact metrics, product names, and architecture keywords (e.g., 'SWE-bench', 'RAG', 'CVE').\n"
        f"If searching for trends or vulnerabilities, implicitly pivot towards recent {current_year} data.\n"
        "DO NOT use outdated chronological markers or past target years unless explicitly required.\n"
        "Output ONLY the raw keywords, lowercase, no quotes, no punctuation. Maximum 8 words."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic to optimize: {question}"),
    ]

    try:
        raw_response = await invoke_messages(messages, max_tokens=25, temperature=0.1)
        cleaned = raw_response.strip().lower().replace('"', "").replace("'", "")
        if cleaned and len(cleaned.split()) <= 12:
            return cleaned
    except Exception as exc:
        logger.warning("LLM query temporal optimization failed: %s", exc)

    # Умный фолбек: если модель легла, динамически подливаем текущий год к словам
    words = [w for w in question.strip().rstrip("?").lower().split() if len(w) > 3]
    return f"{' '.join(words[:5])} {current_year}"
