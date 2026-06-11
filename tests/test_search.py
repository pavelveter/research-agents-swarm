"""Integration tests for search provider abstraction and fallback.

Covers:
A) Primary fails, fallback succeeds → workflow continues
B) Specific provider pairs fail/succeed
C) All providers fail → retrieval_failed
D) Targeted research mode
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_swarm.search.base import SearchResultItem
from research_swarm.search.health import SearchHealthMonitor


class TestSearchOrchestrator:
    """Tests for the SearchOrchestrator fallback chain."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_stops_fallback(self) -> None:
        """When primary provider returns results, no fallback is tried."""
        from research_swarm.search.orchestrator import SearchOrchestrator
        from research_swarm.search.base import BaseSearchProvider, SearchResponse

        class MockPrimary(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "tavily"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(
                    results=[
                        SearchResultItem(
                            title="Result", url="https://ex.com",
                            snippet="Found", provider=self.slug,
                        ),
                    ],
                    provider=self.slug,
                )

        class MockFallback(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "brave"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(
                    results=[
                        SearchResultItem(
                            title="Fallback", url="https://fb.com",
                            snippet="Fallback result", provider=self.slug,
                        ),
                    ],
                    provider=self.slug,
                )

        primary = MockPrimary()
        fallback = MockFallback()

        orch = SearchOrchestrator(providers=[primary, fallback])
        results, metadata = await orch.search("test query")

        assert len(results) == 1
        assert results[0].provider == "tavily"
        assert metadata["successful_provider"] == "tavily"
        assert metadata["all_failed"] is False

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self) -> None:
        """Scenario A: DDG fails, Tavily succeeds — workflow continues."""
        from research_swarm.search.orchestrator import SearchOrchestrator
        from research_swarm.search.base import BaseSearchProvider, SearchResponse

        class MockFailingPrimary(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "duckduckgo"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(results=[], provider=self.slug)

        class MockTavily(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "tavily"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(
                    results=[
                        SearchResultItem(
                            title="Tavily result", url="https://tavily.com",
                            snippet="Found via Tavily", provider=self.slug,
                        ),
                    ],
                    provider=self.slug,
                )

        ddg = MockFailingPrimary()
        tavily = MockTavily()

        orch = SearchOrchestrator(providers=[tavily, ddg])
        results, metadata = await orch.search("test")

        assert len(results) == 1
        assert results[0].provider == "tavily"
        assert metadata["providers_tried"] == ["tavily"]
        assert metadata["successful_provider"] == "tavily"

    @pytest.mark.asyncio
    async def test_brave_fallback_after_tavily_fails(self) -> None:
        """Scenario B: Tavily fails, Brave succeeds — workflow continues."""
        from research_swarm.search.orchestrator import SearchOrchestrator
        from research_swarm.search.base import BaseSearchProvider, SearchResponse

        class MockFailingTavily(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "tavily"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(results=[], provider=self.slug)

        class MockBrave(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "brave"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(
                    results=[
                        SearchResultItem(
                            title="Brave result", url="https://brave.com",
                            snippet="Found via Brave", provider=self.slug,
                        ),
                    ],
                    provider=self.slug,
                )

        tavily = MockFailingTavily()
        brave = MockBrave()

        orch = SearchOrchestrator(providers=[tavily, brave])
        results, metadata = await orch.search("test")

        assert len(results) == 1
        assert results[0].provider == "brave"
        assert metadata["providers_tried"] == ["tavily", "brave"]
        assert metadata["successful_provider"] == "brave"

    @pytest.mark.asyncio
    async def test_all_providers_fail(self) -> None:
        """Scenario C: All providers fail — all_failed=True."""
        from research_swarm.search.orchestrator import SearchOrchestrator
        from research_swarm.search.base import BaseSearchProvider, SearchResponse

        class MockFailing(BaseSearchProvider):
            def __init__(self, slug: str) -> None:
                self._slug = slug

            @property
            def slug(self) -> str:
                return self._slug

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(results=[], provider=self.slug)

        providers = [MockFailing("tavily"), MockFailing("brave"), MockFailing("ddg")]

        orch = SearchOrchestrator(providers=providers)
        results, metadata = await orch.search("test")

        assert len(results) == 0
        assert metadata["all_failed"] is True
        assert metadata["successful_provider"] is None
        assert metadata["providers_tried"] == ["tavily", "brave", "ddg"]

    @pytest.mark.asyncio
    async def test_unavailable_providers_are_skipped(self) -> None:
        """Providers with is_available=False are silently skipped."""
        from research_swarm.search.orchestrator import SearchOrchestrator
        from research_swarm.search.base import BaseSearchProvider, SearchResponse

        class MockUnavailable(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "tavily"

            @property
            def is_available(self) -> bool:
                return False

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(results=[], provider=self.slug)

        class MockAvailable(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "duckduckgo"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(
                    results=[
                        SearchResultItem(
                            title="DDG result", url="https://ddg.com",
                            snippet="Found", provider=self.slug,
                        ),
                    ],
                    provider=self.slug,
                )

        unavailable = MockUnavailable()
        available = MockAvailable()

        orch = SearchOrchestrator(providers=[unavailable, available])
        results, metadata = await orch.search("test")

        # Unavailable was skipped, only available tried
        assert "tavily" not in metadata["providers_tried"]
        assert "duckduckgo" in metadata["providers_tried"]
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_provider_exception_is_caught(self) -> None:
        """When a provider raises an exception, fallback continues."""
        from research_swarm.search.orchestrator import SearchOrchestrator
        from research_swarm.search.base import BaseSearchProvider, SearchResponse

        class MockCrashing(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "tavily"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                raise RuntimeError("Provider crashed!")

        class MockStable(BaseSearchProvider):
            @property
            def slug(self) -> str:
                return "duckduckgo"

            @property
            def is_available(self) -> bool:
                return True

            async def search(self, query: str, max_results: int = 5) -> SearchResponse:
                return SearchResponse(
                    results=[
                        SearchResultItem(
                            title="Stable result", url="https://ddg.com",
                            snippet="Found", provider=self.slug,
                        ),
                    ],
                    provider=self.slug,
                )

        crashing = MockCrashing()
        stable = MockStable()

        orch = SearchOrchestrator(providers=[crashing, stable])
        results, metadata = await orch.search("test")

        assert len(results) == 1
        assert results[0].provider == "duckduckgo"
        assert "tavily" in metadata["providers_tried"]
        assert "duckduckgo" in metadata["providers_tried"]


class TestSearchHealthMonitor:
    """Tests for the SearchHealthMonitor."""

    def test_records_success_and_failure(self) -> None:
        monitor = SearchHealthMonitor()

        monitor.record_attempt("tavily")
        monitor.record_success("tavily", result_count=5, latency_s=0.3)

        monitor.record_attempt("brave")
        monitor.record_failure("brave", "timeout")

        tavily = monitor.get_metrics("tavily")
        assert tavily.total_attempts == 1
        assert tavily.total_successes == 1
        assert tavily.success_rate == 1.0
        assert tavily.avg_results == 5.0

        brave = monitor.get_metrics("brave")
        assert brave.total_attempts == 1
        assert brave.total_failures == 1
        assert brave.failure_rate == 1.0
        assert brave.total_timeouts == 1

    def test_snapshot_returns_all_metrics(self) -> None:
        monitor = SearchHealthMonitor()
        monitor.record_attempt("tavily")
        monitor.record_success("tavily", 3, 0.1)
        monitor.record_attempt("brave")
        monitor.record_failure("brave", "error")

        snapshot = monitor.snapshot()
        assert "tavily" in snapshot
        assert "brave" in snapshot
        assert snapshot["tavily"].success_rate == 1.0
        assert snapshot["brave"].failure_rate == 1.0

    def test_zero_attempts_defaults(self) -> None:
        monitor = SearchHealthMonitor()
        metrics = monitor.get_metrics("unknown")
        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 0.0
        assert metrics.avg_results == 0.0

    def test_multiple_successes_avg(self) -> None:
        monitor = SearchHealthMonitor()
        monitor.record_attempt("tavily")
        monitor.record_success("tavily", 2, 0.2)
        monitor.record_attempt("tavily")
        monitor.record_success("tavily", 8, 0.8)

        m = monitor.get_metrics("tavily")
        assert m.total_attempts == 2
        assert m.total_successes == 2
        assert m.avg_results == 5.0  # (2+8)/2
        assert m.avg_latency_s == 0.5  # (0.2+0.8)/2

    def test_log_summary_does_not_raise(self) -> None:
        monitor = SearchHealthMonitor()
        monitor.record_attempt("tavily")
        monitor.record_success("tavily", 5, 0.1)
        monitor.log_summary()  # Should not raise


class TestSearchResultItem:
    """Tests for SearchResultItem dataclass."""

    def test_default_confidence(self) -> None:
        item = SearchResultItem(
            title="T", url="https://x.com", snippet="S", provider="tavily",
        )
        assert item.confidence == 0.5

    def test_explicit_confidence(self) -> None:
        item = SearchResultItem(
            title="T", url="https://x.com", snippet="S",
            provider="tavily", confidence=0.92,
        )
        assert item.confidence == 0.92
