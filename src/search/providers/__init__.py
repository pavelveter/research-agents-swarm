"""Search providers package — exports all built-in providers."""

from search.providers.academic import AcademicProvider
from search.providers.brave import BraveProvider
from search.providers.duckduckgo import DuckDuckGoProvider
from search.providers.searxng import SearXNGProvider
from search.providers.serpapi import SerpAPIProvider
from search.providers.tavily import TavilyProvider

__all__ = [
    "TavilyProvider",
    "BraveProvider",
    "SerpAPIProvider",
    "SearXNGProvider",
    "DuckDuckGoProvider",
    "AcademicProvider",
]
