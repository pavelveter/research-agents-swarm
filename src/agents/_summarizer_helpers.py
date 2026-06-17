"""Internal helpers for the Summarizer agent."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from graph.state import ResearchReport

logger = logging.getLogger(__name__)

_GAP_LOG_PATH: Final[Path] = Path("data_gaps.jsonl")


def _export_data_gaps(
    gaps: list[str], query: str, timestamp: float | None = None
) -> None:
    """Persist uncovered missing_topics as machine-readable JSONL.

    Each line is a JSON object with query, missing_topic, and timestamp.
    Appends to ``data_gaps.jsonl`` so the file grows across runs.
    """
    ts = timestamp or _dt.datetime.now(_dt.timezone.utc).timestamp()
    try:
        with open(_GAP_LOG_PATH, "a", encoding="utf-8") as f:
            for gap in gaps:
                f.write(
                    json.dumps(
                        {
                            "query": query,
                            "missing_topic": gap,
                            "timestamp": ts,
                        }
                    )
                    + "\n"
                )
        logger.info("Exported %d data gaps to %s", len(gaps), _GAP_LOG_PATH)
    except OSError as exc:
        logger.warning("Failed to write data-gap log: %s", exc)


def _check_source_diversity(sources: list[str]) -> str | None:
    """Return a warning string if sources collapse to a single domain.

    Returns None if diversity is acceptable (≥2 domains or <3 sources).
    """
    if len(sources) < 3:
        return None
    domains: dict[str, int] = {}
    for s in sources:
        try:
            netloc = urlparse(s).netloc or s
            # Extract root domain (last two parts)
            parts = netloc.split(".")
            root = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
        except Exception:
            root = s
        domains[root] = domains.get(root, 0) + 1
    if not domains:
        return None
    top_domain = max(domains, key=domains.get)
    top_ratio = domains[top_domain] / len(sources)
    if top_ratio > 0.5:
        return (
            f"[SOURCE DIVERSITY WARNING: {domains[top_domain]}/{len(sources)} "
            f"sources ({top_ratio:.0%}) originate from '{top_domain}'. "
            f"Verify findings against independent sources.]"
        )
    return None


def _validate_citations(
    citations: list[dict],
    sources: list[str],
) -> tuple[list[dict], list[str]]:
    """Filter out citations with out-of-bounds source_index.

    Returns (valid_citations, rejection_messages).
    """
    if not citations or not sources:
        return [], []
    valid: list[dict] = []
    rejected: list[str] = []
    for i, cit in enumerate(citations):
        idx = cit.get("source_index", -1)
        # LLMs sometimes return string indices — attempt conversion
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(sources):
            valid.append(cit)
        else:
            reason = (
                f"Citation #{i} source_index={cit.get('source_index')} out of bounds "
                f"(sources count={len(sources)}) — dropped"
            )
            rejected.append(reason)
            logger.warning(reason)
    return valid, rejected


def _validate_summary_has_query_signals(summary: str, query: str) -> bool:
    """Return True if summary has BOTH citation markers ``[N]`` AND query keywords."""
    has_citations = bool(re.search(r"\[\d+\]", summary))
    # Check if at least one substantive query keyword appears in summary
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    summary_lower = summary.lower()
    hits = sum(1 for w in query_words if w in summary_lower)
    has_keywords = hits >= 1
    # Require BOTH: citations AND query-relevant content
    return has_citations and has_keywords


def _dedupe_sources_and_remap_citations(
    sources: list[str],
    citations: list[dict],
) -> tuple[list[str], list[dict]]:
    """Deduplicate sources preserving first-occurrence order and remap citation indices.

    Returns (deduped_sources, citations_with_remapped_indices).
    Citations whose source_index falls on a duplicate URL are remapped to the
    URL's first occurrence index so they stay valid after dedup.
    """
    if not sources:
        return [], citations
    # Build deduped list and index map: old_index -> new_index (first occurrence)
    deduped: list[str] = []
    index_map: dict[int, int] = {}
    first_seen: dict[str, int] = {}  # url -> first new_index
    for old_idx, url in enumerate(sources):
        if url not in first_seen:
            new_idx = len(deduped)
            first_seen[url] = new_idx
            deduped.append(url)
        else:
            new_idx = first_seen[url]
        index_map[old_idx] = new_idx
    # Remap citation source_index values
    remapped_citations: list[dict] = []
    for cit in citations:
        cit = dict(cit)  # shallow copy to avoid mutating originals
        idx = cit.get("source_index", -1)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = -1
        if idx in index_map:
            cit["source_index"] = index_map[idx]
        # else: leave as-is (out of bounds, will be caught by _validate_citations)
        remapped_citations.append(cit)
    return deduped, remapped_citations


def _empty_report() -> ResearchReport:
    """Return a trivial report for no-evidence / low-context scenarios."""
    return ResearchReport(
        summary="No verifiable evidence was found for this query.",
        sources=[],
    )
