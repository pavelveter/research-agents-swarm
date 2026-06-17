"""Summarizer agent — re-architected to pull clean context from Qdrant vector storage."""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage

from config.domain import get_domain
from config.settings import get_settings
from graph.state import ResearchReport, ResearchState
from llm.client import invoke_messages
from logging_config import preview
from memory.vector_storage import get_memory_bank
from observability.langfuse import trace_agent
from utils import safe_json, render_prompt
from agents._summarizer_helpers import (
    _check_source_diversity,
    _dedupe_sources_and_remap_citations,
    _empty_report,
    _export_data_gaps,
    _validate_citations,
    _validate_summary_has_query_signals,
)

logger = logging.getLogger(__name__)

# ── Debug instrumentation ──────────────────────────────────────────
_DEBUG = os.environ.get("DEBUG_SUMMARIZER", "").strip() in ("1", "true", "yes")


def _debug_log(msg: str, *args: object) -> None:
    """Log when DEBUG_SUMMARIZER is enabled."""
    if _DEBUG:
        logger.info("[DEBUG_SUMMARIZER] " + msg, *args)

# ── Validation constants ───────────────────────────────────────────
_MIN_CONTEXT_CHARS = 150  # below this, emit empty report (no evidence)
# ────────────────────────────────────────────────────────────────────


async def summarize(state: ResearchState) -> ResearchState:
    """Produce final comprehensive report pulling clean context out of Qdrant memory layer."""
    with trace_agent("summarizer", input_data={"query": state.query}) as tracer:

        # Copy previous report before overwriting (for incremental rewrite loop)
        if state.final_report is not None:
            state.previous_report = state.final_report

        # ── 1. Retrieve context (T11: context isolation — prefer recent/matching facts)
        memory = get_memory_bank()
        # Build a focused query from current missing_topics to bias retrieval
        if state.missing_topics:
            focused_query = f"{state.query} {' '.join(state.missing_topics[:3])}"
        else:
            focused_query = state.query
        facts = await memory.retrieve_context(
            query=focused_query, limit=50, layer="principles",
        )
        logger.info(
            "Summarizer retrieved %d top unique semantic facts from Qdrant", len(facts)
        )

        if not facts:
            facts_block = "(No verified facts collected across graph iterations.)"
        else:
            facts_block = "\n".join(f"- {f}" for f in facts)

        _debug_log("Context sent to LLM: chars=%d preview=%s",
                    len(facts_block), preview(facts_block[:200]))

        # ── 2. Low-context / empty-evidence guard ────────────────
        total_evidence = sum(len(sr.evidence) for sr in state.search_results)
        sources_count = len(state.final_report.sources) if state.final_report else 0

        if not facts or len(facts_block) < _MIN_CONTEXT_CHARS or total_evidence == 0:
            logger.info(
                "Low-context or empty evidence: facts=%d chars=%d evidence_items=%d "
                "— emitting empty report",
                len(facts), len(facts_block), total_evidence,
            )
            if sources_count == 0 and total_evidence == 0:
                state.final_report = _empty_report()
            else:
                # Some evidence exists but context is thin — still report what we have
                state.final_report = ResearchReport(
                    summary="Insufficient evidence to produce a substantive report.",
                    sources=state.final_report.sources if state.final_report else [],
                )
            if hasattr(tracer, "update_observation"):
                tracer.update_observation(
                    output={"summary_length": len(state.final_report.summary),
                            "sources_count": len(state.final_report.sources),
                            "empty_report": True}
                )
            return state

        # ── 3. Build prompt with rewrite instruction ─────────────
        domain = get_domain(state.query)
        # T13: Only inject inference synthesis for explicitly configured domains
        # (RESEARCH_DOMAIN env var), never for auto-detected ones.
        settings = get_settings()
        explicit = settings.research_domain.strip().lower() if settings.research_domain else ""
        inference_block = domain.inference_synthesis if explicit else ""
        if inference_block:
            logger.info("Domain inference synthesis active: domain=%s", domain.slug)
        system_content = render_prompt(
            "summarizer_summarizer_system.jinja",
            domain_inference_synthesis=inference_block,
        )
        human_extra = ""
        if state.previous_report is not None:
            system_content += (
                "\n\nINCREMENTAL REDESIGN: This is a rewrite iteration. "
                "Preserve strong sections from the previous report. "
                "Address weaknesses and missing topics using the new facts below. "
                "Do NOT restart from scratch — extend and improve the existing report."
            )
            human_extra = (
                f"\n\nPrevious Report (preserve strong parts, improve weak parts):\n"
                f"{state.previous_report.summary}\n"
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

        # ── 3.4. Synthetic Analysis: when deep-dive mode is active and
        # missing_topics persist, instruct the LLM to derive answers from
        # existing principles (shift from Search Engine → Expert System).
        if state.iteration >= 2 and state.missing_topics:
            real_topics = [t for t in state.missing_topics if not str(t).startswith("[SYSTEM]")]
            if real_topics:
                synthesis_instruction = (
                    "\n\n## SYNTHETIC ANALYSIS MODE (overrides no-synthesis rule)\n"
                    "The hard rule against synthesis is RELAXED for the uncovered topics below. "
                    "These topics were NOT found in search despite deep-dive iterations — "
                    "public sources are insufficient. You MUST now perform deductive derivation "
                    "from the principles you have already cited:\n"
                    + "\n".join(f"- {t}" for t in real_topics[:5])
                    + "\n\nFor each uncovered topic: "
                    "(1) State which existing principles/frameworks from the cited facts apply, "
                    "(2) Apply deductive reasoning to derive the logical implication, "
                    "(3) Mark derived content with '[Inferred]' to distinguish from factual evidence. "
                    "This is structured expert analysis grounded in cited principles — NOT free hallucination."
                )
                messages[0] = SystemMessage(
                    content=system_content + synthesis_instruction
                )
                logger.info(
                    "Synthetic analysis mode active: %d missing_topics for deep-dive derivation",
                    len(real_topics),
                )

        # ── 3.5. T3: Chain-of-Thought reasoning pass ─────────────
        # Pass 1 (hidden): generate structured comparison of facts
        cot_reasoning: dict = {}
        try:
            cot_messages = [
                SystemMessage(
                    content=(
                        "You are an analytical reasoning engine. Perform a structured comparison "
                        "of the provided facts. Identify: (1) agreements — facts that reinforce "
                        "each other, (2) conflicts — contradictory or diverging claims, "
                        "(3) evidence strength — which facts are most credible based on source "
                        "diversity and specificity. Output ONLY a JSON object with keys "
                        "'agreements', 'conflicts', 'evidence_strength', and 'synthesis_direction' "
                        "(1-2 sentences on the report's analytical arc). This reasoning will NOT "
                        "appear in the final report — it feeds the next compilation step."
                    )
                ),
                HumanMessage(content=f"Facts to analyze:\n{facts_block}"),
            ]
            reasoning_raw = await invoke_messages(cot_messages, temperature=0.3, max_tokens=400)
            cot_reasoning = safe_json(reasoning_raw)
            _debug_log("CoT reasoning produced keys=%s", list(cot_reasoning.keys()) if cot_reasoning else "(none)")
        except Exception as exc:
            logger.warning("CoT reasoning pass failed (non-fatal): %s", exc)

        # ── 4. Primary LLM call ──────────────────────────────────
        temperature = max(0.0, 1.0 - domain.strictness)
        logger.info("Summarizer temperature=%.2f (domain=%s strictness=%.2f)", temperature, domain.slug, domain.strictness)
        raw = await invoke_messages(messages, temperature=temperature)
        _debug_log("Raw LLM output (first 500 chars): %s", raw[:500])
        parsed = safe_json(raw)

        # ── 5. Fallback: merge with previous report if JSON empty ─
        if not parsed or (not parsed.get("summary") and not parsed.get("sources")):
            logger.warning(
                "Summarizer returned empty JSON, attempting fallback merge with previous report"
            )
            if state.previous_report and state.previous_report.summary:
                merge_messages = [
                    SystemMessage(
                        content=(
                            "You are a report merger. Combine the previous report with new facts. "
                            "Preserve strong sections from the previous report. Add new facts where "
                            "they fit naturally. Return ONLY valid JSON with keys 'summary', "
                            "'sources', and 'citations'."
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
                state.final_report = _empty_report()
                if hasattr(tracer, "update_observation"):
                    tracer.update_observation(
                        output={"summary_length": 0, "sources_count": 0, "empty_report": True}
                    )
                return state

        # ── 6. Extract parsed fields ─────────────────────────────
        summary = str(parsed.get("summary", ""))
        sources = [str(s) for s in parsed.get("sources", [])]
        citations = parsed.get("citations", [])
        # Deduplicate sources & remap citation indices to first occurrence
        sources, citations = _dedupe_sources_and_remap_citations(sources, citations)

        # ── 7. Post-parse validation ─────────────────────────────
        # 7a. Validate citation source_index bounds
        valid_citations, citation_warnings = _validate_citations(citations, sources)
        if citation_warnings:
            logger.warning(
                "Citation validation: %d/%d citations dropped due to out-of-bounds source_index",
                len(citations) - len(valid_citations), len(citations),
            )

        # 7b. Validate summary has query signals
        has_signals = _validate_summary_has_query_signals(summary, state.query)
        _debug_log(
            "Post-parse: summary_chars=%d sources=%d citations=%d/%d valid "
            "has_query_signals=%s",
            len(summary), len(sources), len(valid_citations), len(citations),
            has_signals,
        )

        needs_retry = not has_signals or (
            citations and not valid_citations and not summary
        )

        # ── 8. Retry with stricter prompt if validation fails ─────
        if needs_retry:
            logger.warning(
                "Summarizer validation failed (has_signals=%s valid_citations=%d/%d). "
                "Retrying with stricter prompt.",
                has_signals, len(valid_citations), len(citations),
            )
            stricter_system = system_content + (
                "\n\n⚠️ PREVIOUS ATTEMPT FAILED VALIDATION. Your response was rejected because:\n"
                + ("- Summary must contain query-related keywords or [N] citation markers\n"
                   if not has_signals else "")
                + (f"- source_index values must be valid indices into sources array (0..{len(sources)-1 if sources else 0})\n"
                   if citation_warnings else "")
                + "Please fix these issues and return ONLY valid JSON."
            )
            retry_messages = [
                SystemMessage(content=stricter_system),
                HumanMessage(
                    content=(
                        f"Global Research Topic: {state.query}\n"
                        f"Top Validated Contextual Facts:\n{facts_block}"
                        f"{human_extra}"
                    )
                ),
            ]
            retry_raw = await invoke_messages(retry_messages, temperature=0.1)
            _debug_log("Retry raw output (first 500 chars): %s", retry_raw[:500])
            retry_parsed = safe_json(retry_raw)

            if retry_parsed and retry_parsed.get("summary"):
                summary = str(retry_parsed.get("summary", summary))
                sources = [str(s) for s in retry_parsed.get("sources", sources)]
                citations = retry_parsed.get("citations", citations)
                sources, citations = _dedupe_sources_and_remap_citations(sources, citations)
                valid_citations, _ = _validate_citations(citations, sources)
                logger.info("Retry succeeded: summary_chars=%d sources=%d citations=%d",
                            len(summary), len(sources), len(valid_citations))
            else:
                logger.warning("Retry also failed — emitting empty report")
                state.final_report = _empty_report()
                if hasattr(tracer, "update_observation"):
                    tracer.update_observation(
                        output={"summary_length": 0, "sources_count": 0,
                                "empty_report": True, "retry_failed": True}
                    )
                return state

        # ── 9. Write final report ────────────────────────────────
        # T2: Gap handling — mark uncovered missing_topics as Data gaps
        gap_notes: list[str] = []
        if state.missing_topics:
            summary_lower = summary.lower()
            for topic in state.missing_topics:
                if str(topic).startswith("[SYSTEM]"):
                    continue
                # Check if topic keywords appear in summary (simple heuristic)
                topic_words = [w.lower() for w in str(topic).split() if len(w) > 3]
                if topic_words and not any(w in summary_lower for w in topic_words):
                    gap_notes.append(f"Data gap: {topic}")
            if gap_notes:
                logger.info(
                    "Gap handling: %d/%d missing_topics not covered — marking as Data gaps",
                    len(gap_notes),
                    len([t for t in state.missing_topics if not str(t).startswith("[SYSTEM]")]),
                )
                summary += "\n\n## Unresolved Data Gaps\n" + "\n".join(f"- {g}" for g in gap_notes)
                # T18: Persist gaps to machine-readable JSONL log
                _export_data_gaps(gap_notes, state.query)

        # T19: Source diversity enforcement
        diversity_warning = _check_source_diversity(sources)
        if diversity_warning:
            logger.warning("Source diversity low: %s", diversity_warning[:80])
            summary = f"{diversity_warning}\n\n{summary}"
            state.source_diversity_low = True

        state.final_report = ResearchReport(
            summary=summary,
            sources=sources,
        )

        if valid_citations:
            logger.info("Report includes %d valid citations with fact-to-source mappings",
                        len(valid_citations))

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
                    "citations_count": len(valid_citations),
                    "retried": needs_retry,
                    # T3: CoT reasoning stored for observability (Langfuse)
                    "reasoning": cot_reasoning if cot_reasoning else {},
                }
            )

    return state
