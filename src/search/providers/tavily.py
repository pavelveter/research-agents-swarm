"""Tavily Search API provider.

Requires ``TAVILY_API_KEY`` environment variable.

Uses the Tavily Search API directly via ``httpx`` — no additional
package dependency required.
"""

from __future__ import annotations

import logging
from typing import Final

import httpx

from config.settings import get_settings
from search._http import DEFAULT_TIMEOUT, Latency, safe_json_request, success
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
        latency = Latency.start_now()
        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            data, err = await safe_json_request(
                client, self.slug, "post", _TAVILY_BASE,
                latency, query=query, logger_=logger, json=payload,
            )
        if err is not None:
            return err

        items = [
            SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                provider=self.slug,
                confidence=r.get("score", 0.7),
            )
            for r in data.get("results", [])[:max_results]
        ]
        logger.info("Tavily produced %d result(s) for: %s", len(items), query)
        return success(self.slug, items, latency)
