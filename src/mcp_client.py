"""DEPRECATED — DuckDuckGo HTML search client.

This module has been superseded by the provider abstraction layer in
``search.providers.duckduckgo``.

The ``search_web()`` and ``_parse_results()`` functions have moved to
``DuckDuckGoProvider`` in the new architecture. This file is kept as
a compatibility shim but should not be used in new code.

Use ``from search import SearchOrchestrator`` instead.
"""

from __future__ import annotations

import warnings

from search.providers.duckduckgo import DuckDuckGoProvider

warnings.warn(
    "mcp_client.py is deprecated. Use search.providers.duckduckgo "
    "or search.SearchOrchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

_ddg = DuckDuckGoProvider()


async def search_web(query: str, max_results: int = 5) -> list[str]:
    """DEPRECATED — use ``SearchOrchestrator.search()`` instead.

    Thin compatibility wrapper around ``DuckDuckGoProvider.search()``.
    """
    import warnings as _w

    _w.warn(
        "search_web() is deprecated. Use SearchOrchestrator.search() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    response = await _ddg.search(query, max_results=max_results)
    return [f"{r.title} - {r.snippet} ({r.url})" for r in response.results]
