"""Brave Search API provider.

Requires ``BRAVE_API_KEY`` environment variable.

Uses the Brave Search API via ``httpx``.
"""

from __future__ import annotations

import logging
import time
from typing import Final

import httpx

from research_swarm.config.settings import get_settings
from research_swarm.search.base import BaseSearchProvider, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

_BRAVE_BASE: Final[str] = "https://api.search.brave.com/res/v1/web/search"


class BraveProvider(BaseSearchProvider):
    """Brave Search API provider — privacy-respecting web + news search."""

    @property
    def slug(self) -> str:
        return "brave"

    @property
    def is_available(self) -> bool:
        settings = get_settings()
        return bool(settings.brave_api_key)

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        settings = get_settings()
        t0 = time.monotonic()

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": settings.brave_api_key,
        }
        params = {"q": query, "count": min(max_results, 20)}

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            try:
                resp = await client.get(_BRAVE_BASE, headers=headers, params=params)
            except httpx.TimeoutException:
                logger.error("Brave search timed out for: %s", query)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": "timeout", "latency_s": time.monotonic() - t0},
                )
            except httpx.HTTPError as exc:
                logger.error("Brave transport error for '%s': %s", query, exc)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
                )

        if resp.status_code != 200:
            logger.error("Brave returned HTTP %d for: %s", resp.status_code, query)
            return SearchResponse(
                results=[], provider=self.slug,
                raw_metadata={
                    "http_status": resp.status_code,
                    "latency_s": time.monotonic() - t0,
                },
            )

        try:
            data = resp.json()
        except Exception as exc:
            logger.error("Brave JSON parse error: %s", exc)
            return SearchResponse(
                results=[], provider=self.slug,
                raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
            )

        items: list[SearchResultItem] = []
        web_results = data.get("web", {}).get("results", [])
        for r in web_results[:max_results]:
            items.append(SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                provider=self.slug,
                confidence=0.7,
            ))

        logger.info("Brave produced %d result(s) for: %s", len(items), query)
        return SearchResponse(
            results=items, provider=self.slug,
            raw_metadata={"result_count": len(items), "latency_s": time.monotonic() - t0},
        )
