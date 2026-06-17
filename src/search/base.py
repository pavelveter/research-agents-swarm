"""Abstract base search provider and result types.

All search providers MUST inherit from ``BaseSearchProvider`` and implement
the ``search`` async method. This abstraction allows the searcher agent to be
completely decoupled from any specific search implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final


@dataclass(slots=True)
class SearchResultItem:
    """A single search result from any provider, normalized to a common shape.

    Attributes
    ----------
    title : str
        The result title / heading.
    url : str
        The resolved destination URL.
    snippet : str
        A short text excerpt or description.
    provider : str
        The provider slug that produced this result (e.g. ``"tavily"``).
    confidence : float
        Provider-reported confidence score (0.0–1.0). Default 0.5.
    """

    title: str
    url: str
    snippet: str
    provider: str
    confidence: float = 0.5


@dataclass(slots=True)
class SearchResponse:
    """Normalised search response returned by every provider.

    Attributes
    ----------
    results : list[SearchResultItem]
        Zero or more result items.
    provider : str
        Slug of the provider that produced this response.
    raw_metadata : dict | None
        Optional provider-specific diagnostics (e.g. latency, raw HTML size,
        HTTP status, cache hit flag).
    """

    results: list[SearchResultItem]
    provider: str
    raw_metadata: dict | None = None


class BaseSearchProvider(ABC):
    """Abstract base class for all search providers.

    Subclasses must implement:

    * ``slug`` — a short unique provider identifier (e.g. ``"tavily"``).
    * ``search(query, max_results)`` — perform the search and return a
      ``SearchResponse``.
    * ``is_available`` — whether the provider can be used (credentials
      present, endpoint reachable).
    """

    # Priority constants used by the orchestrator:
    PRIORITY_TIERS: Final[dict[str, int]] = {
        "tavily": 0,
        "brave": 1,
        "serpapi": 2,
        "searxng": 3,
        "duckduckgo": 4,
    }

    @property
    @abstractmethod
    def slug(self) -> str:
        """Unique provider identifier, e.g. ``"tavily"``."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Execute a search query and return normalised results.

        Parameters
        ----------
        query : str
            The search query string.
        max_results : int
            Maximum number of results to return (default 5).

        Returns
        -------
        SearchResponse
            Never raises; on failure returns an empty ``SearchResponse``.
        """

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is configured and ready to use."""

    @property
    def priority(self) -> int:
        """Lower value = higher priority in the fallback chain."""
        return self.PRIORITY_TIERS.get(self.slug, 99)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} slug={self.slug!r} available={self.is_available}>"
