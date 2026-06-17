"""Shared HTTP helpers for search providers.

Underscore-prefixed: internal to the search package, not part of the public
API exported from ``search.__init__``.

Provides:

* ``DEFAULT_TIMEOUT`` / ``DDG_TIMEOUT`` — shared httpx timeout presets.
* ``Latency`` — zero-BOI elapsed-time tracker for one HTTP call.
* ``failure`` / ``success`` — uniform ``SearchResponse`` builders that
  always include ``latency_s`` in ``raw_metadata``.
* ``safe_request`` — runs a single ``client.{get,post}(...)`` call with the
  standard error chain (timeout, transport error, non-200). Returns the raw
  ``httpx.Response`` so callers can parse HTML/JSON/XML/etc. themselves.
* ``safe_json_request`` — like ``safe_request`` but additionally parses and
  validates JSON.

These helpers exist to remove ~120 lines of duplicated boilerplate across
all five HTTP-based search providers. Academic does not fit this pattern
(dual-source aggregation with shared client).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from search.base import SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: httpx.Timeout = httpx.Timeout(15.0, connect=5.0)
DDG_TIMEOUT: httpx.Timeout = httpx.Timeout(10.0, connect=5.0)


@dataclass(slots=True)
class Latency:
    """Elapsed-time tracker for a single HTTP call.

    Ponytail: ``elapsed`` is recomputed per access (cheap, monotonic).
    Variants that cache are unnecessary for sub-second search calls.
    """

    start: float

    @classmethod
    def start_now(cls) -> Latency:
        return cls(start=time.monotonic())

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start


def failure(slug: str, latency: Latency, **meta: Any) -> SearchResponse:
    """Build an empty ``SearchResponse`` with uniform error metadata.

    Always includes ``latency_s``; additional keys (typically ``error=`` or
    ``http_status=``) are merged in. Never raises.
    """
    md: dict[str, Any] = {"latency_s": latency.elapsed, **meta}
    return SearchResponse(results=[], provider=slug, raw_metadata=md)


def success(
    slug: str, items: list[SearchResultItem], latency: Latency
) -> SearchResponse:
    """Build a successful ``SearchResponse`` with latency + result count."""
    return SearchResponse(
        results=items,
        provider=slug,
        raw_metadata={"result_count": len(items), "latency_s": latency.elapsed},
    )


async def _safe_http_call(
    client: httpx.AsyncClient,
    slug: str,
    method: str,
    url: str,
    latency: Latency,
    query: str | None = None,
    logger_: logging.Logger | None = None,
    **kwargs: Any,
) -> tuple[httpx.Response, None] | tuple[None, SearchResponse]:
    """Run ``client.{get,post}(...)``, returning ``(resp, None)`` or error.

    Shared error chain for ``safe_request`` and ``safe_json_request``:

    1. ``httpx.TimeoutException`` → ``failure(... error="timeout")``
    2. ``httpx.HTTPError`` → ``failure(... error=str(exc))``
    3. ``resp.status_code != 200`` → ``failure(... http_status=...)``
    """
    log = logger_ or logger
    q = f" for: {query}" if query else ""
    try:
        resp = await getattr(client, method)(url, **kwargs)
    except httpx.TimeoutException:
        log.error("%s search timed out%s", slug, q)
        return None, failure(slug, latency, error="timeout")
    except httpx.HTTPError as exc:
        log.error("%s transport error for '%s': %s", slug, query or "?", exc)
        return None, failure(slug, latency, error=str(exc))

    if resp.status_code != 200:
        log.error("%s returned HTTP %d%s", slug, resp.status_code, q)
        return None, failure(slug, latency, http_status=resp.status_code)

    return resp, None


async def safe_request(
    client: httpx.AsyncClient,
    slug: str,
    method: str,
    url: str,
    latency: Latency,
    query: str | None = None,
    logger_: logging.Logger | None = None,
    **kwargs: Any,
) -> tuple[httpx.Response, None] | tuple[None, SearchResponse]:
    """Run an HTTP request with the standard error chain, returning raw response.

    Returns ``(resp, None)`` on success so the caller can parse ``resp.text``,
    ``resp.content``, or any other payload form. Returns ``(None, error_response)``
    on timeout, transport error, or non-200 status.

    Used by HTML-scraping providers (DuckDuckGo) — JSON-API providers use
    ``safe_json_request`` instead.

    Parameters
    ----------
    client :
        Open ``httpx.AsyncClient`` (caller manages the context).
    slug :
        Provider identifier, used for logging and ``SearchResponse.provider``.
    method :
        ``"get"`` or ``"post"`` — passed to ``getattr(client, method)(url, ...)``.
    url :
        Request URL.
    latency :
        Started ``Latency`` instance.
    query :
        Optional query string for log context.
    logger_ :
        Logger for error messages; falls back to this module's logger.
    **kwargs :
        Forwarded to ``client.{get,post}`` (typically ``params=`` or ``headers=``).
    """
    return await _safe_http_call(
        client, slug, method, url, latency, query, logger_, **kwargs
    )


async def safe_json_request(
    client: httpx.AsyncClient,
    slug: str,
    method: str,
    url: str,
    latency: Latency,
    query: str | None = None,
    logger_: logging.Logger | None = None,
    **kwargs: Any,
) -> tuple[dict, None] | tuple[None, SearchResponse]:
    """Run a JSON-API request with the standard error chain.

    On top of the ``safe_request`` chain, additionally handles:

    * JSON parse error → ``failure(... error=str(exc))``

    Returns ``(data, None)`` on success or ``(None, error_response)`` on any
    failure. The caller is responsible for constructing items from ``data``.
    """
    log = logger_ or logger
    resp, err = await _safe_http_call(
        client, slug, method, url, latency, query, logger_, **kwargs
    )
    if err is not None:
        return None, err

    try:
        data = resp.json()
    except Exception as exc:
        log.error("%s JSON parse error: %s", slug, exc)
        return None, failure(slug, latency, error=str(exc))

    # JSON literal null → treat as empty so callers always see a dict.
    if data is None:
        data = {}
    return data, None
