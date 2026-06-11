"""Search layer — provider abstraction, fallback orchestration, health monitoring.

Public API
----------
* ``SearchOrchestrator`` — the main entry point for searches with
  automatic provider fallback.
* ``get_orchestrator()`` — singleton factory.
* ``SearchHealthMonitor`` — per-provider metrics tracking.
* ``BaseSearchProvider`` — abstract base for custom providers.
* ``SearchResultItem`` / ``SearchResponse`` — data types.
* ``save_failure_diagnostic()`` — optional debug persistence.
"""

from research_swarm.search.base import BaseSearchProvider, SearchResponse, SearchResultItem
from research_swarm.search.diagnostics import save_failure_diagnostic
from research_swarm.search.health import SearchHealthMonitor
from research_swarm.search.orchestrator import SearchOrchestrator, get_orchestrator

__all__ = [
    "BaseSearchProvider",
    "SearchResponse",
    "SearchResultItem",
    "SearchOrchestrator",
    "get_orchestrator",
    "SearchHealthMonitor",
    "save_failure_diagnostic",
]
