"""DuckDuckGo HTML search provider — zero-dependency scraping.

Moves the existing DDG HTML scraping logic from ``mcp_client.py`` into the
provider abstraction. Uses ``httpx.AsyncClient`` + ``beautifulsoup4``.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Final

import httpx
from bs4 import BeautifulSoup

from search.base import BaseSearchProvider, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

HTTP_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_DDG_HTML_BASE: Final[str] = "https://html.duckduckgo.com/html/"
_REDIRECT_PREFIX: Final[str] = "/l/?kh=-1&uddg="

_RESULT_SELECTORS: Final[tuple[str, ...]] = (
    "div.result__body",
    "div.result",
)
_TITLE_SELECTORS: Final[tuple[str, ...]] = ("a.result__a",)
_SNIPPET_SELECTORS: Final[tuple[str, ...]] = (
    "a.result__snippet",
    "div.result__snippet",
)
_URL_SELECTOR: Final[str] = "a.result__url"


def _resolve_redirect(href: str) -> str:
    if _REDIRECT_PREFIX not in href:
        return href
    try:
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        uddg_values = params.get("uddg", [])
        if uddg_values:
            return urllib.parse.unquote(uddg_values[0])
    except Exception:
        logger.debug("Failed to decode DDG redirect URL: %s", href)
    return href


def _first(soup: BeautifulSoup, selectors: tuple[str, ...], default: str = "") -> str:
    for sel in selectors:
        tag_name, _, class_name = sel.partition(".")
        el = soup.find(tag_name, class_=class_name) if class_name else soup.find(tag_name)
        if el:
            return el.get_text(strip=True)
    return default


def _first_href(block: BeautifulSoup, selectors: tuple[str, ...]) -> str:
    for sel in selectors:
        tag_name, _, class_name = sel.partition(".")
        el = block.find(tag_name, class_=class_name) if class_name else block.find(tag_name)
        if el and el.name == "a":
            href = el.get("href", "")
            if href:
                return href.strip()
    return ""


def _parse_results(html: str, max_results: int) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    blocks: list = []

    for sel in _RESULT_SELECTORS:
        _, _, class_name = sel.partition(".")
        blocks = soup.find_all("div", class_=class_name)
        if blocks:
            break

    if not blocks:
        logger.warning("No search result containers found in DDG HTML.")
        return []

    parsed: list[dict[str, str]] = []
    for block in blocks[:max_results]:
        title = _first(block, _TITLE_SELECTORS)
        snippet = _first(block, _SNIPPET_SELECTORS)
        raw_href = _first_href(block, _TITLE_SELECTORS)
        if not raw_href:
            raw_href = _first_href(block, (_URL_SELECTOR,))
        actual_url = _resolve_redirect(raw_href) if raw_href else ""
        if title and snippet:
            parsed.append({"title": title, "url": actual_url, "snippet": snippet})

    return parsed


class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search via server-side HTML endpoint (no API key required)."""

    @property
    def slug(self) -> str:
        return "duckduckgo"

    @property
    def is_available(self) -> bool:
        return True  # Always available — free, no API key needed

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        t0 = time.monotonic()
        encoded = urllib.parse.quote_plus(query)
        url = f"{_DDG_HTML_BASE}?q={encoded}"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
            headers=HTTP_HEADERS,
        ) as client:
            try:
                response = await client.get(url)
            except httpx.TimeoutException:
                logger.error("DDG search timed out for query: %s", query)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": "timeout", "latency_s": time.monotonic() - t0},
                )
            except httpx.HTTPError as exc:
                logger.error("DDG transport error for '%s': %s", query, exc)
                return SearchResponse(
                    results=[], provider=self.slug,
                    raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
                )

        if response.status_code != 200:
            logger.error("DDG returned HTTP %d for: %s", response.status_code, query)
            return SearchResponse(
                results=[], provider=self.slug,
                raw_metadata={
                    "http_status": response.status_code,
                    "latency_s": time.monotonic() - t0,
                },
            )

        try:
            parsed = _parse_results(response.text, max_results)
        except Exception as exc:
            logger.error("Failed to parse DDG HTML for '%s': %s", query, exc, exc_info=True)
            return SearchResponse(
                results=[], provider=self.slug,
                raw_metadata={"error": str(exc), "latency_s": time.monotonic() - t0},
            )

        items = [
            SearchResultItem(
                title=r["title"], url=r["url"], snippet=r["snippet"],
                provider=self.slug, confidence=0.6,
            )
            for r in parsed
        ]
        logger.info("DDG produced %d result(s) for: %s", len(items), query)
        return SearchResponse(
            results=items, provider=self.slug,
            raw_metadata={"result_count": len(items), "latency_s": time.monotonic() - t0},
        )
