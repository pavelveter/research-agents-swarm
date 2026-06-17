"""SerpAPI provider.

Requires ``SERPAPI_API_KEY`` environment variable.

Uses the SerpAPI Google Search API via ``httpx``.
"""

from __future__ import annotations

import logging
from typing import Final

import httpx

from config.settings import get_settings
from search._http import DEFAULT_TIMEOUT, Latency, safe_json_request, success
from search.base import BaseSearchProvider, SearchResponse, SearchResultItem

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
        latency = Latency.start_now()

        params = {
            "api_key": settings.serpapi_api_key,
            "q": query,
            "engine": "google",
            "num": min(max_results, 10),
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            data, err = await safe_json_request(
                client, self.slug, "get", _SERPAPI_BASE,
                latency, query=query, logger_=logger, params=params,
            )
        if err is not None:
            return err

        organic = data.get("organic_results", [])
        items = [
            SearchResultItem(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                provider=self.slug,
                confidence=0.7,
            )
            for r in organic[:max_results]
        ]
        logger.info("SerpAPI produced %d result(s) for: %s", len(items), query)
        return success(self.slug, items, latency)
