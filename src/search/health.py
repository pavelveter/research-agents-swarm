"""Search health monitoring — in-memory metrics for retrieval reliability.

Tracks per-provider success rate, failure rate, average results returned,
average latency, and timeout frequency. All metrics are in-process (no
external database) and available via the ``.snapshot()`` method for logging
or observability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Per-provider health statistics."""

    slug: str
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_timeouts: int = 0
    total_results: int = 0
    total_latency_s: float = 0.0
    last_error: str = ""
    last_error_at: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.total_successes / self.total_attempts

    @property
    def failure_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.total_failures / self.total_attempts

    @property
    def avg_results(self) -> float:
        if self.total_successes == 0:
            return 0.0
        return self.total_results / self.total_successes

    @property
    def avg_latency_s(self) -> float:
        if self.total_successes == 0:
            return 0.0
        return self.total_latency_s / self.total_successes


class SearchHealthMonitor:
    """In-memory retrieval health tracker.

    Usage::

        monitor = SearchHealthMonitor()
        monitor.record_attempt("duckduckgo")
        monitor.record_success("duckduckgo", result_count=5, latency_s=0.3)
        snapshot = monitor.snapshot()
    """

    def __init__(self) -> None:
        self._metrics: dict[str, ProviderMetrics] = {}
        self._started_at = time.monotonic()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_attempt(self, slug: str) -> None:
        self._ensure(slug).total_attempts += 1

    def record_success(self, slug: str, result_count: int, latency_s: float) -> None:
        m = self._ensure(slug)
        m.total_successes += 1
        m.total_results += result_count
        m.total_latency_s += latency_s

    def record_failure(self, slug: str, error: str) -> None:
        m = self._ensure(slug)
        m.total_failures += 1
        m.last_error = error
        m.last_error_at = time.monotonic()
        if "timeout" in error.lower():
            m.total_timeouts += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_metrics(self, slug: str) -> ProviderMetrics:
        """Return metrics for a specific provider."""
        return self._ensure(slug)

    def snapshot(self) -> dict[str, ProviderMetrics]:
        """Return a copy of all per-provider metrics."""
        return {slug: self._metrics[slug] for slug in sorted(self._metrics)}

    def log_summary(self) -> None:
        """Log a human-readable health summary at INFO level."""
        logger.info("=== Search Health Summary ===")
        for slug, m in sorted(self._metrics.items()):
            logger.info(
                "  %-14s attempts=%3d success=%.0f%% fail=%.0f%% "
                "avg_results=%.1f avg_lat=%.2fs timeouts=%d",
                slug,
                m.total_attempts,
                m.success_rate * 100,
                m.failure_rate * 100,
                m.avg_results,
                m.avg_latency_s,
                m.total_timeouts,
            )
        logger.info("==============================")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure(self, slug: str) -> ProviderMetrics:
        if slug not in self._metrics:
            self._metrics[slug] = ProviderMetrics(slug=slug)
        return self._metrics[slug]
