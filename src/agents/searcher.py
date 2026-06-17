"""Searcher agent — production-grade version using SearchOrchestrator fallback chain."""

from __future__ import annotations

import datetime
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agents._searcher_helpers import (
    _deduplicate_evidence,
    _extract_source_name,
    _optimize_query_with_llm,
    _substance_score,
)
from graph.state import EvidenceItem, ResearchState, SearchResult
from llm.client import invoke_messages
from search.orchestrator import get_orchestrator
from utils import safe_json, render_prompt

logger = logging.getLogger(__name__)


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

    # T16: Bias toward academic sources for analytical queries
    combined_topics = " ".join(topics_to_research).lower()
    academic_terms = ("analysis", "framework", "mechanism", "theory", "study",
                      "research", "clinical", "regulation", "policy", "methodology")
    if any(t in combined_topics for t in academic_terms):
        logger.info("Analytical query detected — academic provider will be tried as fallback")

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

        evidence_list: list[EvidenceItem] = []

        # Propagate orchestrator metadata to state (per-iteration)
        if meta:
            providers_tried = meta.get("providers_tried", [])
            if providers_tried:
                state.search_providers_tried = list(dict.fromkeys(
                    state.search_providers_tried + providers_tried
                ))
            if meta.get("successful_provider"):
                state.search_provider_used = meta["successful_provider"]
            if meta.get("all_failed"):
                state.retrieval_failed = True
                state.retrieval_failure_reason = (
                    f"All providers failed across {len(topics_to_research)} queries. "
                    f"Tried: {', '.join(providers_tried)}"
                )
        # Track evidence quality from original hits before substance filter
        for hit in raw_hits:
            state.evidence_quality.append({
                "provider": getattr(hit, "provider", "unknown"),
                "search_mode": state.search_mode,
                "confidence": getattr(hit, "confidence", 0.5),
                "retrieved_at": int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000),
            })

        if raw_hits:
            # Ticket 7: Substance filter — score BEFORE building raw_txt_entries
            scores = [_substance_score(hit.snippet) for hit in raw_hits]
            substantial_count = sum(1 for s in scores if s >= 0.4)
            logger.info(
                "Substance filter: substantial=%d/%d hits",
                substantial_count, len(raw_hits),
            )
            # Flag low quality: ≥8 results but <30% pass substance threshold
            if len(raw_hits) >= 8 and (substantial_count / len(raw_hits)) < 0.30:
                state.retrieval_quality_low = True
                logger.warning(
                    "retrieval_quality_low=True — only %d/%d hits passed substance filter",
                    substantial_count, len(raw_hits),
                )
            # Drop fluff BEFORE building entries for LLM formatting
            raw_hits = [h for h, s in zip(raw_hits, scores) if s >= 0.4]
            logger.info(
                "Substance filter: kept %d/%d hits after dropping fluff",
                len(raw_hits), len(scores),
            )

            raw_txt_entries = [
                f"Title: {hit.title}\nContent: {hit.snippet}\nURL: {hit.url}"
                for hit in raw_hits
            ]

            # Batch format: process in groups of 4 to maximize evidence diversity
            batch_size = 4
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
                    raw_response = await invoke_messages(messages, temperature=0.1)
                    parsed = safe_json(raw_response)

                    raw_evidence = (
                        parsed.get("evidence", []) if isinstance(parsed, dict) else []
                    )
                    for item in raw_evidence:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            source_str = str(item[1])
                            evidence_list.append(EvidenceItem(
                                fact=str(item[0]),
                                source=_extract_source_name(source_str),
                                url=source_str if source_str.startswith("http") else "",
                                iteration_added=state.iteration,
                            ))
                        elif isinstance(item, str):
                            evidence_list.append(EvidenceItem(
                                fact=item,
                                source="search result",
                                iteration_added=state.iteration,
                            ))

                    # Retry with simpler prompt if formatter returns empty
                    if not raw_evidence:
                        logger.warning(
                            "Formatter returned empty evidence for batch %s. Retrying with simpler prompt.",
                            batch_start,
                        )
                        retry_messages = [
                            SystemMessage(
                                content=(
                                    "Extract key facts from the results below. "
                                    "Return ONLY valid JSON with key 'evidence' containing "
                                    "a list of [fact, source] pairs. Keep it simple."
                                )
                            ),
                            HumanMessage(
                                content=f"Question: {current_question}\nRaw Results:\n"
                                + "\n\n".join(batch)
                            ),
                        ]
                        retry_raw = await invoke_messages(retry_messages, max_tokens=200, temperature=0.0)
                        retry_parsed = safe_json(retry_raw)
                        retry_evidence = (
                            retry_parsed.get("evidence", [])
                            if isinstance(retry_parsed, dict)
                            else []
                        )
                        for item in retry_evidence:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                source_str = str(item[1])
                                evidence_list.append(EvidenceItem(
                                    fact=str(item[0]),
                                    source=_extract_source_name(source_str),
                                    url=source_str if source_str.startswith("http") else "",
                                    iteration_added=state.iteration,
                                ))
                            elif isinstance(item, str):
                                evidence_list.append(EvidenceItem(
                                    fact=item,
                                    source="search result",
                                    iteration_added=state.iteration,
                                ))
                except Exception as exc:
                    logger.warning("LLM formatting failed for batch %s: %s. Using raw fallback.", batch_start, exc)

        # Critical fallback
        if not evidence_list and raw_hits:
            logger.warning("LLM mapping failed, injecting raw snippet fallbacks.")
            for i, hit in enumerate(raw_hits[:4]):
                source = hit.url or hit.title or "Search Engine"
                evidence_list.append(EvidenceItem(
                    fact=f"Technical Data: {hit.snippet.strip()}",
                    source=_extract_source_name(source),
                    url=hit.url or "",
                    iteration_added=state.iteration,
                ))

        kept_evidence, new_count = _deduplicate_evidence(evidence_list, state)
        total_new_count += new_count

        state.search_results.append(
            SearchResult(question_id=current_question, evidence=kept_evidence)
        )

    # Ticket 7: persist quality-low state for routing to consume
    if state.retrieval_quality_low:
        logger.info("Searcher complete | retrieval_quality_low flag set")

    state.new_evidence_count = total_new_count
    state.new_evidence_found = total_new_count > 0

    logger.info(
        "Multi-query search step complete | Found %s new unique records across %s paths",
        total_new_count,
        len(topics_to_research),
    )
    return state
