"""Searcher agent — production-grade version using SearchOrchestrator fallback chain."""

from __future__ import annotations

import datetime
import hashlib
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from research_swarm.graph.state import ResearchState, SearchResult
from research_swarm.llm.client import invoke_messages
from research_swarm.search.orchestrator import get_orchestrator
from research_swarm.utils import safe_json, render_prompt

logger = logging.getLogger(__name__)

# FORMAT_SYSTEM moved to jinja


async def search(state: ResearchState) -> ResearchState:
    """Execute search query via orchestrator fallback chain and guarantee evidence extraction."""
    # Take up to 5 missing topics (was hardcoded to 3)
    # Filter out [SYSTEM] operational flags — these are not real search topics
    topics_to_research = []
    if state.missing_topics:
        topics_to_research = [
            t for t in state.missing_topics[:5]
            if not str(t).startswith("[SYSTEM]")
        ]
        # Fallback: if all topics were [SYSTEM] flags, fall back to research_questions
        if not topics_to_research:
            logger.info(
                "All missing_topics are [SYSTEM] operational flags — falling back to research_questions"
            )
            if state.plan and state.plan.research_questions:
                topics_to_research = state.plan.research_questions[:1]
            else:
                topics_to_research = [state.query]
    elif state.plan and state.plan.research_questions:
        topics_to_research = state.plan.research_questions[:1]
    else:
        topics_to_research = [state.query]

    orchestrator = get_orchestrator()
    total_new_count = 0
    all_kept_evidence = []

    # Fan-out search for critical report gaps
    for current_question in topics_to_research:
        search_query = await _optimize_query_with_llm(current_question)
        logger.info(
            "Targeted technical query: '%s' (for topic: '%s')",
            search_query,
            current_question[:40],
        )

        # Adaptive max_results: iteration 0 gets 12-15 (broad research), iteration >= 1 gets 7 (targeted)
        max_results = 12 if state.iteration == 0 else 7
        raw_hits, meta = await orchestrator.search(query=search_query, max_results=max_results)
        logger.info(
            "Orchestrator fetched %s items for query: %s", len(raw_hits), search_query
        )

        evidence_list: list[str] = []

        if raw_hits:
            raw_txt_entries = [
                f"Title: {hit.title}\nContent: {hit.snippet}\nURL: {hit.url}"
                for hit in raw_hits
            ]

            # Batch format: process in groups of 10-15 to avoid token blow-up
            evidence_list = []
            batch_size = 12
            for batch_start in range(0, len(raw_txt_entries), batch_size):
                batch = raw_txt_entries[batch_start:batch_start + batch_size]

                messages = [
                    SystemMessage(content=render_prompt("searcher_format_system.jinja")),
                    HumanMessage(
                        content=f"Question: {current_question}\nRaw Results:\n"
                        + "\n\n".join(batch)
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
                    logger.warning("LLM formatting failed for batch %s: %s. Using raw fallback.", batch_start, exc)

        # Critical fallback
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
    # Dynamically calculate current date so agent knows current time
    now = datetime.date.today()
    current_year = now.year

    # Form human-readable context (e.g., "June 2026")
    current_month_year = now.strftime("%B %Y")

    system_prompt = render_prompt(
        "searcher_system_prompt.jinja",
        current_month_year=current_month_year,
        current_year=current_year,
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

    # Smart fallback: if model fails, dynamically append current year to words
    words = [w for w in question.strip().rstrip("?").lower().split() if len(w) > 3]
    return f"{' '.join(words[:5])} {current_year}"
