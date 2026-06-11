"""SerpAPI provider.

Requires ``SERPAPI_API_KEY`` environment variable.

Uses the SerpAPI Google Search API via ``httpx``.
"""

from __future__ import annotations

import logging
import time
from typing import Final

import httpx

from research_swarm.config.settings import get_settings
from research_swarm.search.base import BaseSearchProvider, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

_SERPAPI_BASE: Final[str] = "https://serpapi.com/search"


class SerpAPIProvider(BaseSearchProvider):
    """SerpAPI provider — Google Search results via API."""

    @property
    def slug(self) -> str:
        return "serpapi"

    @property
    def is_available(self) -> bool:
        settings = get_settings()
        return bool(settings.serpapi_api_key)

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        settings = get_settings()
        t0 = time.monotonic()

        params = {
            "api_key": settings.serpapi_api_key,
            "q": query,
            "engine": "google",
            "num": min(max_results, 10),
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            try:
                resp = await client.get(_SERPAPI_BASE, params=params)
            except httpx.TimeoutException:
                logger.error("SerpAPI search timed out for: %s", query)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": "timeout", "latency_s": time.monotonic() - t0},
                )
            except httpx.HTTPError as exc:
                logger.error("SerpAPI transport error for '%s': %s", query, exc)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
                )

        if resp.status_code != 200:
            logger.error("SerpAPI returned HTTP %d for: %s", resp.status_code, query)
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
            logger.error("SerpAPI JSON parse error: %s", exc)
            return SearchResponse(
                results=[], provider=self.slug,
                raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
            )

        items: list[SearchResultItem] = []
        organic = data.get("organic_results", [])
        for r in organic[:max_results]:
            items.append(SearchResultItem(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                provider=self.slug,
                confidence=0.7,
            ))

        logger.info("SerpAPI produced %d result(s) for: %s", len(items), query)
        return SearchResponse(
            results=items, provider=self.slug,
            raw_metadata={"result_count": len(items), "latency_s": time.monotonic() - t0},
        )
