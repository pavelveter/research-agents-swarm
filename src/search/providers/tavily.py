"""Tavily Search API provider.

Requires ``TAVILY_API_KEY`` environment variable.

Uses the Tavily Search API directly via ``httpx`` — no additional
package dependency required.
"""

from __future__ import annotations

import logging
import time
from typing import Final

import httpx

from config.settings import get_settings
from search.base import BaseSearchProvider, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

_TAVILY_BASE: Final[str] = "https://api.tavily.com/search"


class TavilyProvider(BaseSearchProvider):
    """Tavily Search API provider — AI-optimised search results."""

    @property
    def slug(self) -> str:
        return "tavily"

    @property
    def is_available(self) -> bool:
        settings = get_settings()
        return bool(settings.tavily_api_key)

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        settings = get_settings()
        t0 = time.monotonic()

        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            try:
                resp = await client.post(_TAVILY_BASE, json=payload)
            except httpx.TimeoutException:
                logger.error("Tavily search timed out for: %s", query)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": "timeout", "latency_s": time.monotonic() - t0},
                )
            except httpx.HTTPError as exc:
                logger.error("Tavily transport error for '%s': %s", query, exc)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
                )

        if resp.status_code != 200:
            logger.error("Tavily returned HTTP %d for: %s", resp.status_code, query)
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
            logger.error("Tavily JSON parse error: %s", exc)
            return SearchResponse(
                results=[], provider=self.slug,
                raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
            )

        items: list[SearchResultItem] = []
        for r in data.get("results", [])[:max_results]:
            items.append(SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                provider=self.slug,
                confidence=r.get("score", 0.7),
            ))

        logger.info("Tavily produced %d result(s) for: %s", len(items), query)
        return SearchResponse(
            results=items, provider=self.slug,
            raw_metadata={"result_count": len(items), "latency_s": time.monotonic() - t0},
        )
