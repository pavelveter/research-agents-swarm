"""Unit tests for ``search._http`` shared helpers.

Covers the extracted boilerplate now used by tavily/brave/serpapi/searxng:

* ``Latency`` elapsed-time tracker
* ``failure`` / ``success`` uniform ``SearchResponse`` builders
* ``safe_json_request`` error chain — timeout, HTTPError, non-200, JSON
  parse error, and the happy path

httpx responses are simulated via ``MockTransport`` so no network I/O.
"""

from __future__ import annotations

import httpx
import pytest

from search._http import (
    DEFAULT_TIMEOUT,
    DDG_TIMEOUT,
    Latency,
    failure,
    safe_json_request,
    safe_request,
    success,
)
from search.base import SearchResultItem


class TestLatency:
    """``Latency`` time tracker — sync portion."""

    def test_elapsed_is_positive_after_work(self) -> None:
        latency = Latency.start_now()
        # elapsed() recomputes each call — caller captures the value
        e = latency.elapsed
        assert e >= 0.0

    def test_start_now_distinct_instances(self) -> None:
        a = Latency.start_now()
        b = Latency.start_now()
        assert a is not b


class TestFailure:
    """``failure(slug, latency, **meta)`` empty response builder."""

    def test_includes_latency_and_meta(self) -> None:
        latency = Latency.start_now()
        resp = failure("tavily", latency, error="timeout")
        assert resp.results == []
        assert resp.provider == "tavily"
        assert resp.raw_metadata is not None
        assert "latency_s" in resp.raw_metadata
        assert resp.raw_metadata["error"] == "timeout"

    def test_meta_does_not_overwrite_latency(self) -> None:
        latency = Latency.start_now()
        resp = failure("brave", latency, http_status=429)
        assert resp.raw_metadata is not None
        assert resp.raw_metadata["http_status"] == 429
        assert "latency_s" in resp.raw_metadata


class TestSuccess:
    """``success(slug, items, latency)`` builder."""

    def test_includes_result_count_and_latency(self) -> None:
        latency = Latency.start_now()
        items = [
            SearchResultItem(title="t", url="u", snippet="s", provider="x"),
            SearchResultItem(title="t2", url="u2", snippet="s2", provider="x"),
        ]
        resp = success("brave", items, latency)
        assert resp.results == items
        assert resp.provider == "brave"
        assert resp.raw_metadata is not None
        assert resp.raw_metadata["result_count"] == 2
        assert "latency_s" in resp.raw_metadata

    def test_empty_items_returns_zero_count(self) -> None:
        latency = Latency.start_now()
        resp = success("duckduckgo", [], latency)
        assert resp.results == []
        assert resp.raw_metadata is not None
        assert resp.raw_metadata["result_count"] == 0


def _mock_client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, transport=handler)


class TestSafeJsonRequest:
    """``safe_json_request`` — runs an httpx call through the standard error chain."""

    @pytest.mark.asyncio
    async def test_success_returns_data(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"results": [{"k": "v"}]})
        )
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "tavily", "get", "https://t.test",
                Latency.start_now(), params={"q": "x"},
            )
        assert err is None
        assert data == {"results": [{"k": "v"}]}

    @pytest.mark.asyncio
    async def test_timeout_returns_error_response(self) -> None:
        def boom(_req: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("simulated timeout")

        transport = httpx.MockTransport(boom)
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "tavily", "get", "https://t.test",
                Latency.start_now(),
            )
        assert data is None
        assert err is not None
        assert err.results == []
        assert err.provider == "tavily"
        md = err.raw_metadata or {}
        assert md["error"] == "timeout"
        assert "latency_s" in md

    @pytest.mark.asyncio
    async def test_http_error_returns_error_response(self) -> None:
        def boom(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated connection error")

        transport = httpx.MockTransport(boom)
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "brave", "get", "https://b.test",
                Latency.start_now(),
            )
        assert data is None
        assert err is not None
        md = err.raw_metadata or {}
        assert "connection error" in md["error"]
        assert "latency_s" in md

    @pytest.mark.asyncio
    async def test_non_200_returns_error_response(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(429, text="rate limited")
        )
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "serpapi", "get", "https://s.test",
                Latency.start_now(),
            )
        assert data is None
        assert err is not None
        md = err.raw_metadata or {}
        assert md["http_status"] == 429
        assert "latency_s" in md

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error_response(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, text="<not json>")
        )
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "searxng", "get", "https://sx.test",
                Latency.start_now(),
            )
        assert data is None
        assert err is not None
        md = err.raw_metadata or {}
        assert "error" in md
        assert "latency_s" in md

    @pytest.mark.asyncio
    async def test_json_null_returns_empty_dict(self) -> None:
        """Server-returned JSON ``null`` → success branch returns ``({}, None)``."""
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, text="null")
        )
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "serpapi", "get", "https://s.test",
                Latency.start_now(),
            )
        assert err is None
        assert data == {}

    @pytest.mark.asyncio
    async def test_query_appears_in_error_logs(self, caplog) -> None:
        """Error log messages include the query string for debugging.

        Triggers the non-200 path (which is one of the paths that interpolates
        ``query`` into the log message). The helper falls back to
        ``search._http``'s logger when ``logger_`` is not supplied, so
        ``caplog`` targets that name.
        """
        import logging
        transport = httpx.MockTransport(
            lambda req: httpx.Response(429, text="rate limited")
        )
        with caplog.at_level(logging.ERROR, logger="search._http"):
            async with _mock_client(transport) as client:
                await safe_json_request(
                    client, "tavily", "get", "https://t.test",
                    Latency.start_now(), query="my search phrase",
                )
        assert any("my search phrase" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_post_method_is_forwarded(self) -> None:
        seen: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["method"] = req.method
            seen["body"] = req.content.decode()
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with _mock_client(transport) as client:
            data, err = await safe_json_request(
                client, "tavily", "post", "https://t.test",
                Latency.start_now(), json={"q": "hello"},
            )
        assert err is None
        assert data == {"ok": True}
        assert seen["method"] == "POST"
        assert "hello" in seen["body"]


def test_default_timeouts_are_configured() -> None:
    """Sanity: constants are real httpx.Timeout instances with sane values."""
    assert isinstance(DEFAULT_TIMEOUT, httpx.Timeout)
    assert isinstance(DDG_TIMEOUT, httpx.Timeout)
    # DDG_TIMEOUT uses a shorter read timeout than DEFAULT_TIMEOUT
    # connect timeouts match (both 5s).
    assert DDG_TIMEOUT.connect == DEFAULT_TIMEOUT.connect == 5.0


class TestSafeRequest:
    """``safe_request`` — same error chain as ``safe_json_request``, returns raw response."""

    @pytest.mark.asyncio
    async def test_success_returns_raw_response(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, text="<html>ok</html>")
        )
        async with _mock_client(transport) as client:
            resp, err = await safe_request(
                client, "duckduckgo", "get", "https://ddg.test",
                Latency.start_now(),
            )
        assert err is None
        assert resp is not None
        assert resp.status_code == 200
        assert resp.text == "<html>ok</html>"

    @pytest.mark.asyncio
    async def test_timeout_returns_error_response(self) -> None:
        def boom(_req: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("read timeout")

        transport = httpx.MockTransport(boom)
        async with _mock_client(transport) as client:
            resp, err = await safe_request(
                client, "duckduckgo", "get", "https://ddg.test",
                Latency.start_now(),
            )
        assert resp is None
        assert err is not None
        assert err.results == []
        assert err.provider == "duckduckgo"
        md = err.raw_metadata or {}
        assert md["error"] == "timeout"
        assert "latency_s" in md

    @pytest.mark.asyncio
    async def test_http_error_returns_error_response(self) -> None:
        def boom(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("network down")

        transport = httpx.MockTransport(boom)
        async with _mock_client(transport) as client:
            resp, err = await safe_request(
                client, "duckduckgo", "get", "https://ddg.test",
                Latency.start_now(),
            )
        assert resp is None
        assert err is not None
        md = err.raw_metadata or {}
        assert "network down" in md["error"]

    @pytest.mark.asyncio
    async def test_non_200_returns_error_response(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(503, text="service unavailable")
        )
        async with _mock_client(transport) as client:
            resp, err = await safe_request(
                client, "duckduckgo", "get", "https://ddg.test",
                Latency.start_now(),
            )
        assert resp is None
        assert err is not None
        md = err.raw_metadata or {}
        assert md["http_status"] == 503

    @pytest.mark.asyncio
    async def test_query_appears_in_error_logs(self, caplog) -> None:
        """Error log messages include the query string for debugging."""
        import logging
        transport = httpx.MockTransport(
            lambda req: httpx.Response(503, text="no service")
        )
        with caplog.at_level(logging.ERROR, logger="search.providers.duckduckgo"):
            async with _mock_client(transport) as client:
                await safe_request(
                    client, "duckduckgo", "get", "https://ddg.test",
                    Latency.start_now(), query="frontier ai",
                    logger_=logging.getLogger("search.providers.duckduckgo"),
                )
        # Logger passed explicitly here so caplog captures at provider scope.
        assert any(
            getattr(rec, "name", "") == "search.providers.duckduckgo"
            and "frontier ai" in rec.message
            for rec in caplog.records
        )
