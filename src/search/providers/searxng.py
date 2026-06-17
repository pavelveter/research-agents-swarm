"""SearXNG provider.

Requires ``SEARXNG_BASE_URL`` environment variable pointing to a
self-hosted SearXNG instance.

Uses the SearXNG JSON API via ``httpx``.
"""

from __future__ import annotations

import logging

import httpx

from config.settings import get_settings
from search._http import DEFAULT_TIMEOUT, Latency, safe_json_request, success
from search.base import BaseSearchProvider, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)


class SearXNGProvider(BaseSearchProvider):
    """SearXNG metasearch provider — self-hosted, privacy-respecting.

    Requires a running SearXNG instance accessible at ``SEARXNG_BASE_URL``.
    """

    @property
    def slug(self) -> str:
        return "searxng"

    @property
    def is_available(self) -> bool:
        settings = get_settings()
        return bool(settings.searxng_base_url)

    def _api_url(self) -> str:
        settings = get_settings()
        base = settings.searxng_base_url.rstrip("/")
        return f"{base}/search"

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        latency = Latency.start_now()
        url = self._api_url()
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "pageno": 1,
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            data, err = await safe_json_request(
                client, self.slug, "get", url,
                latency, query=query, logger_=logger, params=params,
            )
        if err is not None:
            return err

        items = [
            SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", r.get("snippet", "")),
                provider=self.slug,
                confidence=0.6,
            )
            for r in data.get("results", [])[:max_results]
        ]
        logger.info("SearXNG produced %d result(s) for: %s", len(items), query)
        return success(self.slug, items, latency)
