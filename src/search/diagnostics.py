"""Retrieval failure diagnostics — optional debug logging for search failures.

When a search provider returns zero results, the diagnostics module saves
diagnostic snapshots to ``logs/retrieval_failures/`` so operators can
inspect what went wrong without sifting through full application logs.

Each failure snapshot is a JSON file containing:

* query
* provider slug
* raw response metadata (HTTP status, latency, error messages)
* optional HTML snapshot (for HTML-scraping providers like DDG)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FAILURE_LOG_DIR: Path = Path("logs/retrieval_failures")


def _ensure_dir() -> None:
    FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def save_failure_diagnostic(
    query: str,
    provider_slug: str,
    metadata: dict | None = None,
    html_snapshot: str | None = None,
) -> Path | None:
    """Persist a diagnostic file for a single retrieval failure.

    Parameters
    ----------
    query : str
        The search query that failed.
    provider_slug : str
        Provider identifier (e.g. ``"duckduckgo"``).
    metadata : dict | None
        Raw metadata from the provider's ``SearchResponse``.
    html_snapshot : str | None
        Raw HTML response body (only for HTML-scraping providers).

    Returns
    -------
    Path | None
        Path to the saved file, or ``None`` if saving was skipped.
    """
    try:
        _ensure_dir()
    except OSError as exc:
        logger.error("Cannot create diagnostics directory: %s", exc)
        return None

    timestamp = int(time.time() * 1000)
    safe_provider = provider_slug.replace("/", "_").replace(" ", "_")
    filename = f"failure_{timestamp}_{safe_provider}.json"
    filepath = FAILURE_LOG_DIR / filename

    payload: dict = {
        "timestamp": timestamp,
        "query": query,
        "provider": provider_slug,
        "metadata": metadata or {},
    }

    if html_snapshot:
        payload["html_snapshot"] = html_snapshot[:10_000]

    try:
        filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved retrieval failure diagnostic: %s", filepath)
        return filepath
    except OSError as exc:
        logger.error("Failed to write diagnostic file: %s", exc)
        return None
