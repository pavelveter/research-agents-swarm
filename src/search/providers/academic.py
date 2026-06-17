"""T16: Academic-source provider for Semantic Scholar and arXiv.

Uses Semantic Scholar's free REST API (no key required) as the primary
academic backend, with arXiv as a keyword fallback via HTTP.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from search.base import BaseSearchProvider, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)


class AcademicProvider(BaseSearchProvider):
    """Searches Semantic Scholar and arXiv for academic/analytical queries.

    Priority is set lower than DuckDuckGo (99) so it only runs
    when the orchestrator explicitly biases toward academic sources.
    """

    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_URL = "http://export.arxiv.org/api/query"

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=15.0)

    @property
    def slug(self) -> str:
        return "academic"

    @property
    def is_available(self) -> bool:
        # Always available — Semantic Scholar is free, no key
        return True

    @property
    def priority(self) -> int:
        return 5  # T16: after Tavily(0)/Brave(1), before SerpAPI(3)

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        results: list[SearchResultItem] = []

        # Try Semantic Scholar first
        try:
            params: dict[str, Any] = {
                "query": query,
                "limit": min(max_results, 10),
                "fields": "title,url,abstract",
            }
            resp = await self._http.get(
                self.SEMANTIC_SCHOLAR_URL, params=params
            )
            if resp.status_code == 200:
                data = resp.json()
                for paper in data.get("data", []):
                    results.append(SearchResultItem(
                        title=str(paper.get("title", "")),
                        url=str(paper.get("url", "")),
                        snippet=str(paper.get("abstract", ""))[:300],
                        provider=self.slug,
                        confidence=0.75,
                    ))
        except Exception as exc:
            logger.debug("Semantic Scholar query failed: %s", exc)

        # Fallback to arXiv if Semantic Scholar returned nothing
        if not results:
            try:
                arxiv_params = {
                    "search_query": f"all:{query}",
                    "max_results": str(min(max_results, 5)),
                    "sortBy": "relevance",
                }
                resp = await self._http.get(self.ARXIV_URL, params=arxiv_params)
                if resp.status_code == 200:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(resp.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall("atom:entry", ns):
                        title_el = entry.find("atom:title", ns)
                        summary_el = entry.find("atom:summary", ns)
                        link_el = entry.find("atom:id", ns)
                        results.append(SearchResultItem(
                            title=(title_el.text or "").strip() if title_el is not None else "",
                            url=(link_el.text or "").strip() if link_el is not None else "",
                            snippet=(summary_el.text or "")[:300].strip() if summary_el is not None else "",
                            provider="arxiv",
                            confidence=0.65,
                        ))
            except Exception as exc:
                logger.debug("arXiv fallback failed: %s", exc)

        return SearchResponse(
            results=results,
            provider=self.slug,
            raw_metadata={"result_count": len(results)},
        )
