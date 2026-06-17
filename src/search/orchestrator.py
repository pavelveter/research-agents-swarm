"""Search orchestrator — manages provider priority fallback chain.

The orchestrator tries providers in priority order (Tavily → Brave →
SerpAPI → SearXNG → DuckDuckGo). Each provider is skipped silently
if it is not available (missing credentials). Results from each attempt
are accumulated and deduplicated.

The orchestrator reports the combined result along with per-provider
diagnostics through the ``SearchHealthMonitor``.
"""

from __future__ import annotations

import logging
from typing import Sequence

from search.base import BaseSearchProvider, SearchResultItem
from search.health import SearchHealthMonitor
from search.providers.academic import AcademicProvider
from search.providers.brave import BraveProvider
from search.providers.duckduckgo import DuckDuckGoProvider
from search.providers.searxng import SearXNGProvider
from search.providers.serpapi import SerpAPIProvider
from search.providers.tavily import TavilyProvider

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: tuple[type[BaseSearchProvider], ...] = (
    TavilyProvider,
    BraveProvider,
    SerpAPIProvider,
    SearXNGProvider,
    DuckDuckGoProvider,
    AcademicProvider,
)


class SearchOrchestrator:
    """Coordinates searches across multiple providers with priority fallback.

    Public API — the searcher agent should depend on this class, not on
    individual providers.

    Parameters
    ----------
    providers : Sequence[BaseSearchProvider] | None
        Custom provider ordering. Defaults to the standard priority chain.
    health : SearchHealthMonitor | None
        Health monitor instance for tracking metrics.
    """

    def __init__(
        self,
        providers: Sequence[BaseSearchProvider] | None = None,
        health: SearchHealthMonitor | None = None,
    ) -> None:
        if providers is not None:
            self._providers = list(providers)
        else:
            self._providers = [cls() for cls in DEFAULT_PROVIDERS]

        # Sort by priority (lower = higher priority)
        self._providers.sort(key=lambda p: p.priority)

        self._health = health or SearchHealthMonitor()

    @property
    def providers(self) -> list[BaseSearchProvider]:
        return list(self._providers)

    @property
    def health(self) -> SearchHealthMonitor:
        return self._health

    async def search(
        self, query: str, max_results: int = 5
    ) -> tuple[list[SearchResultItem], dict]:
        """Execute a search across the fallback chain.

        Tries each available provider in priority order. Stops early if
        any provider returns results — but records all attempts for health
        monitoring.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int
            Maximum results to return per provider.

        Returns
        -------
        tuple[list[SearchResultItem], dict]
            Accumulated results and a metadata dict with keys:
            ``"providers_tried"``, ``"successful_provider"``,
            ``"all_failed"``, ``"attempts"``.
        """
        all_results: list[SearchResultItem] = []
        providers_tried: list[str] = []
        successful_provider: str | None = None
        attempts: list[dict] = []
        all_failed = True

        for provider in self._providers:
            if not provider.is_available:
                logger.debug("Skipping unavailable provider: %s", provider.slug)
                continue

            providers_tried.append(provider.slug)
            self._health.record_attempt(provider.slug)
            logger.info("Trying provider: %s for query: %s", provider.slug, query)

            try:
                response = await provider.search(query, max_results=max_results)
            except Exception as exc:
                logger.error(
                    "Provider %s raised unhandled exception for '%s': %s",
                    provider.slug, query, exc, exc_info=True,
                )
                self._health.record_failure(provider.slug, str(exc))
                attempts.append({
                    "provider": provider.slug,
                    "success": False,
                    "error": str(exc),
                })
                continue

            attempts.append({
                "provider": provider.slug,
                "success": len(response.results) > 0,
                "result_count": len(response.results),
                "metadata": response.raw_metadata,
            })

            if response.results:
                self._health.record_success(
                    provider.slug,
                    result_count=len(response.results),
                    latency_s=(
                        response.raw_metadata.get("latency_s", 0)
                        if response.raw_metadata
                        else 0
                    ),
                )
                all_results.extend(response.results)
                successful_provider = provider.slug
                all_failed = False
                logger.info(
                    "Provider %s succeeded with %d results. Stopping fallback.",
                    provider.slug, len(response.results),
                )
                break
            else:
                self._health.record_failure(
                    provider.slug,
                    f"empty results (metadata: {response.raw_metadata})",
                )
                logger.warning(
                    "Provider %s returned 0 results for: %s",
                    provider.slug, query,
                )

        if all_failed:
            logger.error(
                "ALL providers failed for query: %s (tried: %s)",
                query, providers_tried,
            )

        metadata = {
            "providers_tried": providers_tried,
            "successful_provider": successful_provider,
            "all_failed": all_failed,
            "attempts": attempts,
        }

        return all_results, metadata


# Module-level singleton for convenience (recreated per process).
_orchestrator: SearchOrchestrator | None = None


def get_orchestrator() -> SearchOrchestrator:
    """Return the module-level orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SearchOrchestrator()
    return _orchestrator
