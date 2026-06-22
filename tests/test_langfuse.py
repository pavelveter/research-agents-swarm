from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm.client import LLMResponse
from observability.langfuse import (
    _LangfuseTracer,
    _NoopTracer,
    _configure_langfuse,
    _hash_query,
    generate_session_id,
    trace_agent,
    trace_routing_decision,
    session_context,
)


def _make_mock_client() -> tuple[MagicMock, MagicMock, MagicMock]:
    "Build a mock Langfuse client + observation + span trio for tests."
    mock_client = MagicMock()
    mock_span = MagicMock()
    mock_observation = MagicMock()
    mock_observation.__enter__ = MagicMock(return_value=mock_span)
    mock_observation.__exit__ = MagicMock(return_value=False)
    mock_client.start_as_current_observation.return_value = mock_observation
    return mock_client, mock_observation, mock_span


class TestNoopTracer:
    """Tests for the no-op tracer used when Langfuse is unavailable."""

    def test_update_observation_does_nothing(self) -> None:
        tracer = _NoopTracer()
        result = tracer.update_observation(key="value")
        assert result is None

    def test_end_does_nothing(self) -> None:
        tracer = _NoopTracer()
        result = tracer.end(key="value")
        assert result is None

    def test_record_llm_response_does_nothing(self) -> None:
        tracer = _NoopTracer()
        response = LLMResponse(content="hi", model="gpt-4o")
        # Should not raise and should return None
        assert (
            tracer.record_llm_response(
                response, temperature=0.2, agent_name="planner"
            )
            is None
        )

    def test_multiple_calls_safe(self) -> None:
        tracer = _NoopTracer()
        for _ in range(10):
            tracer.update_observation(a=1)
            tracer.end(b=2)


class TestLangfuseTracer:
    """Tests for the real Langfuse tracer wrapper."""

    def test_update_observation_delegates_to_span(self) -> None:
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span)
        tracer.update_observation(output={"score": 90})
        mock_span.update.assert_called_once_with(output={"score": 90})

    def test_end_delegates_to_span(self) -> None:
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span)
        tracer.end(status="completed")
        mock_span.update.assert_called_once_with(status="completed")

    def test_init_stores_span(self) -> None:
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span)
        assert tracer._span is mock_span

    def test_record_llm_response_writes_metadata(self) -> None:
        """TTFT, token usage and cost all land in the span's metadata."""
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span)
        response = LLMResponse(
            content="hello",
            model="gpt-4o",
            ttft_s=0.42,
            duration_s=1.5,
            token_usage={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            cost_usd=0.0005,
        )
        tracer.record_llm_response(
            response, temperature=0.3, prompt_version="v1", agent_name="planner"
        )
        kwargs = mock_span.update.call_args.kwargs
        md = kwargs["metadata"]
        assert md["ttft_s"] == 0.42
        assert md["duration_s"] == 1.5
        assert md["model"] == "gpt-4o"
        assert md["input_tokens"] == 10
        assert md["output_tokens"] == 20
        assert md["total_tokens"] == 30
        assert md["temperature"] == 0.3
        assert md["prompt_version"] == "v1"
        assert md["agent_name"] == "planner"
        # input cost split priced against gpt-4o ($0.0025 / $0.01 per 1k)
        # input: 10/1000 * 0.0025 ≈ 0.000025
        # output: 20/1000 * 0.01 ≈ 0.0002
        # total ≈ 0.000225
        assert md["cost_usd_input"] == round(0.000025, 6)
        assert md["cost_usd_output"] == round(0.0002, 6)
        assert md["cost_usd_total"] == round(0.000225, 6)
        # The langfuse-shaped sub-dict is also present.
        assert md["langfuse"]["usage_details"]["input"] == 10
        assert md["langfuse"]["cost_details"]["total"] == round(0.000225, 6)

    def test_record_llm_response_omits_token_split_when_no_usage(self) -> None:
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span)
        response = LLMResponse(content="hi", model="gpt-4o")
        tracer.record_llm_response(response, agent_name="x")
        md = mock_span.update.call_args.kwargs["metadata"]
        # All zeros when no usage is returned, cost split stays coherent.
        assert md["input_tokens"] == 0
        assert md["output_tokens"] == 0
        assert md["total_tokens"] == 0
        assert md["cost_usd_total"] == 0.0

    def test_record_llm_response_swallows_span_errors(self) -> None:
        """If span.update raises, the safe wrapper degrades to no-op."""
        mock_span = MagicMock()
        mock_span.update.side_effect = Exception("boom")
        tracer = _LangfuseTracer(mock_span)
        response = LLMResponse(
            content="hi",
            model="gpt-4o",
            token_usage={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        )
        # Must not raise
        tracer.record_llm_response(response)

    # ── agent_name auto-derivation ──────────────────────────────

    def test_record_llm_response_auto_derives_agent_name_from_tracer(self) -> None:
        """When agent_name kwarg is omitted on record_llm_response, the value
        from the tracer (set by trace_agent(...)) flows into metadata.
        """
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span, agent_name="summarizer")
        response = LLMResponse(content="hi", model="gpt-4o")
        tracer.record_llm_response(response)  # NO agent_name kwarg

        md = mock_span.update.call_args.kwargs["metadata"]
        assert md["agent_name"] == "summarizer"

    def test_record_llm_response_kwarg_overrides_tracer_default(self) -> None:
        """Explicit agent_name kwarg wins over the tracer-level default."""
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span, agent_name="summarizer")
        response = LLMResponse(content="hi", model="gpt-4o")
        tracer.record_llm_response(response, agent_name="summarizer_cot_reasoning")

        md = mock_span.update.call_args.kwargs["metadata"]
        assert md["agent_name"] == "summarizer_cot_reasoning"

    def test_record_llm_response_kwargs_missing_falls_back_to_unknown(self) -> None:
        """Both tracer attr and kwarg missing → metadata uses 'unknown'."""
        mock_span = MagicMock()
        tracer = _LangfuseTracer(mock_span)  # no agent_name set
        response = LLMResponse(content="hi", model="gpt-4o")
        tracer.record_llm_response(response)  # no agent_name kwarg

        md = mock_span.update.call_args.kwargs["metadata"]
        assert md["agent_name"] == "unknown"

    def test_trace_agent_propagates_agent_name_to_tracer(self) -> None:
        """trace_agent('planner', ...) yields a tracer whose record_llm_response
        writes planner-named metadata without an explicit kwarg.
        """
        mock_client, _mock_obs, mock_span = _make_mock_client()
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_agent("planner", {"query": "x"}) as tracer:
                response = LLMResponse(content="hi", model="gpt-4o")
                tracer.record_llm_response(response)  # auto-derived

        md = mock_span.update.call_args.kwargs["metadata"]
        assert md["agent_name"] == "planner"

    def test_noop_tracer_stores_agent_name(self) -> None:
        """_NoopTracer also accepts agent_name so callers can rely on
        a uniform interface between active and degraded modes.
        """
        tracer = _NoopTracer(agent_name="planner")
        assert tracer._agent_name == "planner"


class TestConfigureLangfuse:
    """Tests for the _configure_langfuse function."""

    def test_no_credentials_does_not_create_client(self) -> None:
        """When keys are missing, no client should be created."""
        with patch("observability.langfuse.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                langfuse_public_key=None,
                langfuse_secret_key=None,
                langfuse_host="https://cloud.langfuse.com",
                langfuse_debug=False,
            )
            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")

            _configure_langfuse()

            assert getattr(_configure_langfuse, "_client", None) is None

    def test_valid_credentials_creates_client(self) -> None:
        """When both keys are present, a Langfuse client should be created."""
        mock_langfuse_module = MagicMock()
        mock_client = MagicMock()
        mock_langfuse_module.Langfuse = MagicMock(return_value=mock_client)

        with (
            patch("observability.langfuse.get_settings") as mock_settings,
            patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
        ):
            mock_settings.return_value = MagicMock(
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
                langfuse_host="https://cloud.langfuse.com",
                langfuse_debug=False,
            )

            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")

            _configure_langfuse()

            assert _configure_langfuse._client is mock_client  # type: ignore[attr-defined]
            # Verify python kwargs: include debug + should_export_span
            call_kwargs = mock_langfuse_module.Langfuse.call_args.kwargs
            assert call_kwargs["public_key"] == "pk-test"
            assert call_kwargs["secret_key"] == "sk-test"
            assert call_kwargs["host"] == "https://cloud.langfuse.com"
            assert call_kwargs["debug"] is False
            assert "should_export_span" in call_kwargs

    def test_initialization_error_is_handled_after_retries(self) -> None:
        """If Langfuse init fails repeatedly, it should be caught gracefully."""
        mock_langfuse_module = MagicMock()
        mock_langfuse_module.Langfuse = MagicMock(
            side_effect=Exception("Connection refused")
        )

        with (
            patch("observability.langfuse.get_settings") as mock_settings,
            patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
            patch("observability.langfuse.time.sleep"),
        ):
            mock_settings.return_value = MagicMock(
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
                langfuse_host="https://cloud.langfuse.com",
                langfuse_debug=False,
            )
            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")

            # Should not raise
            _configure_langfuse()

            # 3 attempts were made
            assert mock_langfuse_module.Langfuse.call_count == 3
            assert getattr(_configure_langfuse, "_client", None) is None

    def test_init_succeeds_on_third_attempt(self) -> None:
        """If first 2 attempts fail, the third should still be used."""
        mock_langfuse_module = MagicMock()
        mock_client = MagicMock()
        mock_langfuse_module.Langfuse = MagicMock(
            side_effect=[Exception("timeout"), Exception("reset"), mock_client]
        )

        with (
            patch("observability.langfuse.get_settings") as mock_settings,
            patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
            patch("observability.langfuse.time.sleep"),
        ):
            mock_settings.return_value = MagicMock(
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
                langfuse_host="https://cloud.langfuse.com",
                langfuse_debug=False,
            )
            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")

            _configure_langfuse()

            assert mock_langfuse_module.Langfuse.call_count == 3
            assert _configure_langfuse._client is mock_client  # type: ignore[attr-defined]

    def test_debug_flag_passed_through(self) -> None:
        """langfuse_debug=True must propagate to Langfuse(...)."""
        mock_langfuse_module = MagicMock()
        mock_client = MagicMock()
        mock_langfuse_module.Langfuse = MagicMock(return_value=mock_client)

        with (
            patch("observability.langfuse.get_settings") as mock_settings,
            patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
        ):
            mock_settings.return_value = MagicMock(
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
                langfuse_host="https://cloud.langfuse.com",
                langfuse_debug=True,
            )
            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")
            _configure_langfuse()

            assert mock_langfuse_module.Langfuse.call_args.kwargs["debug"] is True


class TestTraceAgent:
    """Tests for the trace_agent context manager."""

    def test_returns_noop_when_no_client(self) -> None:
        """When no Langfuse client, should yield a _NoopTracer."""
        with patch.object(_configure_langfuse, "_client", None, create=True):
            with trace_agent("test_agent", {"input": "test"}) as tracer:
                assert isinstance(tracer, _NoopTracer)

    def test_passes_agent_name_to_observation(self) -> None:
        """Should create observation with correct agent name and metadata."""
        mock_client, _mock_obs, _mock_span = _make_mock_client()
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_agent("planner", {"query": "AI"}):
                pass

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["name"] == "planner"
        assert call_kwargs["as_type"] == "agent"
        assert call_kwargs["input"] == {"query": "AI"}
        assert call_kwargs["metadata"]["agent_name"] == "planner"

    def test_passes_session_id_via_metadata(self) -> None:
        """session_id appears in metadata so propagate_attributes can find it
        and ensure all sibling observations share one session.
        """
        mock_client, _mock_obs, _mock_span = _make_mock_client()
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_agent(
                "planner",
                {"query": "x"},
                session_id="research-swarm-abc123",
            ):
                pass

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        # v4.x doesn't accept session_id as a kwarg; we mirror it in
        # metadata and let propagate_attributes drive propagation.
        assert "session_id" not in call_kwargs
        assert call_kwargs["metadata"]["session_id"] == "research-swarm-abc123"
        assert call_kwargs["metadata"]["agent_name"] == "planner"

    def test_enriched_metadata_keys(self) -> None:
        """All observability kwargs (query_hash, model, temperature, prompt_version) appear in the metadata passed to the observation."""
        mock_client, _mock_obs, _mock_span = _make_mock_client()
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_agent(
                "summarizer",
                {"query": "AI"},
                query_hash="qhash01",
                model="gpt-4o",
                temperature=0.2,
                prompt_version="summarizer_v1",
            ):
                pass

        md = mock_client.start_as_current_observation.call_args.kwargs["metadata"]
        assert md["agent_name"] == "summarizer"
        assert md["query_hash"] == "qhash01"
        assert md["model"] == "gpt-4o"
        assert md["temperature"] == 0.2
        assert md["prompt_version"] == "summarizer_v1"

    def test_without_kwargs_metadata_only_has_agent_name(self) -> None:
        """When no extras are passed, only agent_name lands in metadata."""
        mock_client, _mock_obs, _mock_span = _make_mock_client()
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_agent("planner", {}):
                pass

        md = mock_client.start_as_current_observation.call_args.kwargs["metadata"]
        assert md == {"agent_name": "planner"}

    def test_handles_observation_start_failure(self) -> None:
        """If start_as_current_observation raises, should fall back to noop."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = Exception("boom")
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_agent("test_agent", {"input": "test"}) as tracer:
                assert isinstance(tracer, _NoopTracer)


class TestSessionContext:
    """Tests for the session_context manager — drives session_id via propagate_attributes."""

    def test_noop_when_session_id_empty(self) -> None:
        """Empty session_id means no propagate_attributes call is made."""
        with patch.object(_configure_langfuse, "_client", MagicMock()) as mc:
            with session_context(None):
                # propagate_attributes must NOT be called
                assert not mc.propagate_attributes.called
            assert not mc.propagate_attributes.called

    def test_noop_when_no_client(self) -> None:
        """Missing Langfuse client → contextmanager is a harmless no-op."""
        with patch.object(_configure_langfuse, "_client", None, create=True):
            with session_context("session-x"):
                pass  # Should not raise

    def test_calls_propagate_attributes_when_available(self) -> None:
        """When client exists + session_id given, session is propagated."""
        mock_client = MagicMock()
        with patch.object(_configure_langfuse, "_client", mock_client):
            with session_context("session-x"):
                pass

        mock_client.propagate_attributes.assert_called_once_with(
            session_id="session-x"
        )


class TestTraceRoutingDecision:
    """Tests for the trace_routing_decision context manager."""

    def test_returns_noop_when_no_client(self) -> None:
        from graph.state import ResearchState

        state = ResearchState(query="test", judge_score=70, iteration=1)
        with patch.object(_configure_langfuse, "_client", None, create=True):
            with trace_routing_decision("routing", state) as tracer:
                assert isinstance(tracer, _NoopTracer)

    def test_passes_routing_metadata_with_session_id(self) -> None:
        """session_id from state flows into the metadata of routing observation."""
        from graph.state import ResearchState

        mock_client, _mock_obs, _mock_span = _make_mock_client()
        state = ResearchState(
            query="test",
            judge_score=70,
            iteration=2,
            session_id="research-swarm-xyz",
        )
        with patch.object(_configure_langfuse, "_client", mock_client):
            with trace_routing_decision("routing_decision", state):
                pass

        call_kwargs = mock_client.start_as_current_observation.call_args.kwargs
        assert call_kwargs["name"] == "routing_decision"
        assert call_kwargs["as_type"] == "chain"
        assert call_kwargs["metadata"]["session_id"] == "research-swarm-xyz"


class TestSessionIdHelpers:
    """Tests for the small helper utilities."""

    def test_generate_session_id_returns_unique(self) -> None:
        ids = {generate_session_id() for _ in range(10)}
        assert len(ids) == 10  # all different

    def test_generate_session_id_with_prefix(self) -> None:
        sid = generate_session_id(prefix="news-sender")
        assert sid.startswith("news-sender-")

    def test_hash_query_returns_12_chars(self) -> None:
        h = _hash_query("hello world")
        assert len(h) == 12
        assert h.isalnum()

    def test_hash_query_deterministic(self) -> None:
        assert _hash_query("x") == _hash_query("x")

    def test_hash_query_handles_empty(self) -> None:
        # Empty input should still produce a stable 12-char hash (not error)
        assert len(_hash_query("")) == 12
