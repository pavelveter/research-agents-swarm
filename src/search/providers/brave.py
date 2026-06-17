"""Brave Search API provider.

Requires ``BRAVE_API_KEY`` environment variable.

Uses the Brave Search API via ``httpx``.
"""

from __future__ import annotations

import logging
from typing import Final

import httpx

from config.settings import get_settings
from search._http import DEFAULT_TIMEOUT, Latency, safe_json_request, success
from search.base import BaseSearchProvider, SearchResponse, SearchResultItem

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
        latency = Latency.start_now()

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": settings.brave_api_key,
        }
        params = {"q": query, "count": min(max_results, 20)}

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            data, err = await safe_json_request(
                client, self.slug, "get", _BRAVE_BASE,
                latency, query=query, logger_=logger, headers=headers, params=params,
            )
        if err is not None:
            return err

        web_results = data.get("web", {}).get("results", [])
        items = [
            SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                provider=self.slug,
                confidence=0.7,
            )
            for r in web_results[:max_results]
        ]
        logger.info("Brave produced %d result(s) for: %s", len(items), query)
        return success(self.slug, items, latency)
