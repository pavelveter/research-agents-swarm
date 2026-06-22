"""Fact checker agent — hardened with Qdrant vector memory layer."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_settings
from graph.state import EvidenceItem, ResearchState, ValidatedResult
from llm.client import invoke_messages
from memory.vector_storage import get_memory_bank
from observability.langfuse import _hash_query, trace_agent
from utils import render_prompt

logger = logging.getLogger(__name__)


async def fact_check(state: ResearchState) -> ResearchState:
    """Validate ONLY the newest batch of gathered evidence against vector long-term storage."""
    settings = get_settings()
    with trace_agent(
        "fact_checker",
        input_data={"evidence_count": len(state.search_results)},
        session_id=state.session_id,
        query_hash=_hash_query(state.query),
        model=settings.openai_model,
    ) as tracer:

        if not state.search_results:
            logger.warning("No search results found in state.")
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            return state

        # Take ONLY the fresh batch of evidence from the last search
        latest_search = state.search_results[-1]
        raw_evidence: list[EvidenceItem] = latest_search.evidence

        if not raw_evidence:
            logger.warning("Latest search batch contains zero evidence items.")
            state.validated_results.append(
                ValidatedResult(validated_facts=[], rejected_facts=[])
            )
            return state

        logger.info("Fact checking | incoming_batch_size=%s", len(raw_evidence))

        # Initialize our Qdrant singleton
        memory = get_memory_bank()

        # Dynamic similarity threshold with gradual decay:
        # iter 0 → 0.94 (strict), iter 1 → 0.90 (moderate), iter 2+ → 0.82 (relaxed)
        if state.iteration == 0:
            current_threshold = 0.94
        elif state.iteration == 1:
            current_threshold = 0.90
        else:
            current_threshold = 0.82
        # Ticket 9: relax threshold when retrieval quality is low to rescue niche academic findings
        if getattr(state, "retrieval_quality_low", False):
            current_threshold = max(0.70, current_threshold - 0.05)
            logger.info(
                "Fact checking | retrieval_quality_low=True — relaxing threshold by -0.05 to %.2f",
                current_threshold,
            )
        logger.info(
            "Fact checking | iteration=%s, dynamic_threshold=%s",
            state.iteration,
            current_threshold,
        )

        # Layer 1: Primary semantic filtering before sending to LLM
        unique_incoming: list[EvidenceItem] = []
        for item in raw_evidence:
            if await memory._is_semantic_duplicate(
                item.fact, threshold=current_threshold
            ):
                logger.debug(
                    "Pre-filtered semantic duplicate from batch: %s", item.fact[:50]
                )
                continue
            unique_incoming.append(item)

        if not unique_incoming:
            # T3: Desperation mode — if incoming evidence exists but all filtered, promote top-1
            if raw_evidence:
                logger.info(
                    "All %d items filtered as semantic duplicates. "
                    "Desperation mode: promoting top-1 result without strict threshold.",
                    len(raw_evidence),
                )
                unique_incoming = [raw_evidence[0]]
            else:
                logger.info(
                    "All incoming items filtered out as semantic duplicates. Skipping LLM validation."
                )
                state.validated_results.append(
                    ValidatedResult(validated_facts=[], rejected_facts=[])
                )
                state.new_evidence_count = 0
                state.new_evidence_found = False
                return state

        # Layer 2: LLM Validation of remaining unique evidence
        # Build enriched evidence block with source traceability
        evidence_parts: list[str] = []
        for i, item in enumerate(unique_incoming):
            src_tag = f" [source: {item.source}]" if item.source else ""
            url_tag = f" [{item.url}]" if item.url else ""
            evidence_parts.append(f"- [{i}] {item.fact}{src_tag}{url_tag}")
        evidence_block = "\n".join(evidence_parts)

        messages = [
            SystemMessage(content=render_prompt("fact_checker_factcheck_system.jinja")),
            HumanMessage(
                content=(
                    f"Task context: Validate new evidence gathered for question: {latest_search.question_id}\n"
                    f"Evidence Items:\n{evidence_block}"
                )
            ),
        ]

        response = await invoke_messages(messages)
        tracer.record_llm_response(
            response,
            prompt_version="fact_checker_factcheck_system",
            extra={"incoming_batch_size": len(raw_evidence)},
        )
        # Token usage metrics are pushed to Langfuse via ``record_llm_response``
        # above; fall through with parsed payload for downstream consumption.
        parsed = _safe_json(response.content)

        # Synchronize parsing with the state.py format (array of strings)
        validated_list = [str(x) for x in parsed.get("validated_facts", [])]
        rejected_list = [str(x) for x in parsed.get("rejected_facts", [])]

        # Attach source metadata from original evidence items to validated facts
        # where we can match them (P1-6: source→fact traceability)
        source_map: dict[str, tuple[str, str]] = {}
        for item in unique_incoming:
            key = " ".join(item.fact.lower().split())[:100]
            source_map[key] = (item.source, item.url)

        validated_with_source: list[str] = []
        for fact in validated_list:
            fact_key = " ".join(fact.lower().split())[:100]
            src_info = source_map.get(fact_key)
            if src_info:
                src, url = src_info
                if url:
                    validated_with_source.append(f"{fact} — {src} ({url})")
                else:
                    validated_with_source.append(f"{fact} — {src}")
            else:
                validated_with_source.append(fact)

        # T10: Evidence independence check — flag single-source claims
        source_counts: dict[str, int] = {}
        for item in unique_incoming:
            src = item.source or "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1
        dominant_source = (
            max(source_counts, key=source_counts.get) if source_counts else "unknown"
        )
        single_source_ratio = (
            source_counts[dominant_source] / len(unique_incoming)
            if unique_incoming and source_counts
            else 0.0
        )
        if single_source_ratio > 0.5 and len(unique_incoming) >= 2:
            logger.warning(
                "Evidence independence: %d/%d facts (%.0f%%) from single source '%s' — "
                "flagging low-confidence cluster",
                source_counts[dominant_source],
                len(unique_incoming),
                single_source_ratio * 100,
                dominant_source[:40],
            )
            # Inject low-confidence note into validated facts
            validated_list.append(
                f"[LOW-CONFIDENCE: {source_counts[dominant_source]}/{len(unique_incoming)} "
                f"facts sourced from '{dominant_source[:30]}']"
            )

        # Layer 3: Final commit of clean facts to Qdrant
        current_task = latest_search.question_id
        # T14: Per-fact layer tagging — split into principles vs analysis groups
        _REGULATORY_TERMS = frozenset(
            [
                "regulation",
                "directive",
                "act",
                "guideline",
                "guidance",
                "fda",
                "ema",
                "who",
                "law",
                "legislation",
                "compliance",
                "official",
                "standard",
                "framework",
                "policy",
            ]
        )
        principles_facts: list[str] = []
        analysis_facts: list[str] = []
        for fact in validated_with_source:
            fact_lower = fact.lower()
            if any(w in fact_lower for w in _REGULATORY_TERMS):
                principles_facts.append(fact)
            else:
                analysis_facts.append(fact)

        new_stored_count = 0
        if principles_facts:
            new_stored_count += await memory.upsert_facts(
                facts=principles_facts,
                iteration=state.iteration,
                task=current_task,
                threshold=current_threshold,
                layer="principles",
            )
        if analysis_facts:
            new_stored_count += await memory.upsert_facts(
                facts=analysis_facts,
                iteration=state.iteration,
                task=current_task,
                threshold=current_threshold,
                layer="analysis",
            )

        # Save step to LangGraph history
        state.validated_results.append(
            ValidatedResult(
                validated_facts=validated_with_source,
                rejected_facts=rejected_list,
            )
        )

        # Control router flags based on ACTUALLY added new weight
        state.new_evidence_count = new_stored_count
        state.new_evidence_found = new_stored_count > 0

        logger.info(
            "Fact check done | raw_incoming=%s -> unique_filtered=%s | verified=%s -> newly_stored_in_qdrant=%s",
            len(raw_evidence),
            len(unique_incoming),
            len(validated_with_source),
            new_stored_count,
        )

        if hasattr(tracer, "update_observation"):
            tracer.update_observation(
                output={
                    "incoming_count": len(raw_evidence),
                    "after_semantic_filter": len(unique_incoming),
                    "validated_count": len(validated_with_source),
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
