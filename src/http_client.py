"""Shared httpx.AsyncClient singleton for the entire process.

Reuses TCP/TLS connections across all modules (vector_storage, news_sender,
search providers) instead of creating a new client per request. Close via
``shutdown_http_client()`` at process exit.
"""

from __future__ import annotations

import atexit
import logging

import httpx

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide shared ``httpx.AsyncClient``."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )
        logger.debug("Created shared httpx.AsyncClient (max_connections=50)")
    return _client


async def shutdown_http_client() -> None:
    """Gracefully close the shared HTTP client."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        logger.debug("Closed shared httpx.AsyncClient")
    _client = None


def _atexit_close() -> None:
    """Best-effort sync close for atexit (cannot await)."""
    global _client
    if _client is not None and not _client.is_closed:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(_atexit_close)
