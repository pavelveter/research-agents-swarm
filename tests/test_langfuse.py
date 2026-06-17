from __future__ import annotations

from unittest.mock import MagicMock, patch


from observability.langfuse import (
    _LangfuseTracer,
    _NoopTracer,
    _configure_langfuse,
    trace_agent,
    trace_routing_decision,
)


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


class TestConfigureLangfuse:
    """Tests for the _configure_langfuse function."""

    def test_no_credentials_does_not_create_client(self) -> None:
        """When keys are missing, no client should be created."""
        with patch("observability.langfuse.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                langfuse_public_key=None,
                langfuse_secret_key=None,
                langfuse_host="https://cloud.langfuse.com",
            )
            # Clear cached client attribute
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
            )

            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")

            _configure_langfuse()

            assert _configure_langfuse._client is mock_client  # type: ignore[attr-defined]
            mock_langfuse_module.Langfuse.assert_called_once_with(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://cloud.langfuse.com",
            )

    def test_initialization_error_is_handled(self) -> None:
        """If Langfuse init fails, it should be caught gracefully."""
        mock_langfuse_module = MagicMock()
        mock_langfuse_module.Langfuse = MagicMock(side_effect=Exception("Connection refused"))

        with (
            patch("observability.langfuse.get_settings") as mock_settings,
            patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
        ):
            mock_settings.return_value = MagicMock(
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
                langfuse_host="https://cloud.langfuse.com",
            )
            if hasattr(_configure_langfuse, "_client"):
                delattr(_configure_langfuse, "_client")

            # Should not raise
            _configure_langfuse()

            assert getattr(_configure_langfuse, "_client", None) is None


class TestTraceAgent:
    """Tests for the trace_agent context manager."""

    def test_returns_noop_when_no_client(self) -> None:
        """When no Langfuse client, should yield a _NoopTracer."""
        with patch.object(_configure_langfuse, "_client", None, create=True):
            with trace_agent("test_agent", {"input": "test"}) as tracer:
                assert isinstance(tracer, _NoopTracer)

    def test_returns_noop_when_client_is_none(self) -> None:
        """Should yield _NoopTracer when _client attribute is None."""
        old_client = getattr(_configure_langfuse, "_client", None)
        try:
            _configure_langfuse._client = None  # type: ignore[attr-defined]
            with trace_agent("test_agent", {"input": "test"}) as tracer:
                assert isinstance(tracer, _NoopTracer)
        finally:
            if hasattr(_configure_langfuse, "_client"):
                _configure_langfuse._client = old_client  # type: ignore[attr-defined]

    @patch("observability.langfuse._configure_langfuse")
    def test_returns_langfuse_tracer_when_available(self, mock_configure: MagicMock) -> None:
        """When client is available, should yield a _LangfuseTracer."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_span)
        mock_observation.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_observation

        mock_configure._client = mock_client

        with trace_agent("test_agent", {"query": "test"}) as tracer:
            assert isinstance(tracer, _LangfuseTracer)
            assert tracer._span is mock_span

    @patch("observability.langfuse._configure_langfuse")
    def test_passes_agent_name_to_observation(self, mock_configure: MagicMock) -> None:
        """Should create observation with correct agent name and metadata."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_span)
        mock_observation.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_observation

        mock_configure._client = mock_client

        with trace_agent("planner", {"query": "AI"}):
            pass

        mock_client.start_as_current_observation.assert_called_once_with(
            name="planner",
            as_type="agent",
            input={"query": "AI"},
            metadata={"agent_name": "planner"},
        )

    @patch("observability.langfuse._configure_langfuse")
    def test_handles_observation_start_failure(self, mock_configure: MagicMock) -> None:
        """If start_as_current_observation raises, should fall back to noop."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = Exception("Langfuse error")

        mock_configure._client = mock_client

        with trace_agent("test_agent", {"input": "test"}) as tracer:
            assert isinstance(tracer, _NoopTracer)


class TestTraceRoutingDecision:
    """Tests for the trace_routing_decision context manager."""

    def test_returns_noop_when_no_client(self) -> None:
        """When no Langfuse client, should yield a _NoopTracer."""
        from graph.state import ResearchState

        state = ResearchState(query="test", judge_score=70, iteration=1)
        with patch.object(_configure_langfuse, "_client", None, create=True):
            with trace_routing_decision("routing", state) as tracer:
                assert isinstance(tracer, _NoopTracer)

    @patch("observability.langfuse._configure_langfuse")
    def test_returns_langfuse_tracer_when_available(
        self, mock_configure: MagicMock
    ) -> None:
        """When client is available, should yield a _LangfuseTracer."""
        from graph.state import ResearchState

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_span)
        mock_observation.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_observation

        mock_configure._client = mock_client

        state = ResearchState(query="test", judge_score=70, iteration=1)
        with trace_routing_decision("routing", state) as tracer:
            assert isinstance(tracer, _LangfuseTracer)
            assert tracer._span is mock_span

    @patch("observability.langfuse._configure_langfuse")
    def test_passes_routing_metadata_to_observation(
        self, mock_configure: MagicMock
    ) -> None:
        """Should create observation with routing-specific metadata."""
        from graph.state import ResearchState

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=mock_span)
        mock_observation.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_observation

        mock_configure._client = mock_client

        state = ResearchState(
            query="test",
            judge_score=70,
            iteration=2,
            score_delta=5,
            missing_topics=["security"],
            max_iterations=3,
            new_evidence_found=True,
        )
        with trace_routing_decision("routing_decision", state):
            pass

        mock_client.start_as_current_observation.assert_called_once_with(
            name="routing_decision",
            as_type="chain",
            input={
                "iteration": 2,
                "score": 70,
                "score_delta": 5,
                "missing_topics": ["security"],
                "max_iterations": 3,
                "new_evidence_found": True,
            },
            metadata={"routing_name": "routing_decision"},
        )

    @patch("observability.langfuse._configure_langfuse")
    def test_handles_observation_start_failure(
        self, mock_configure: MagicMock
    ) -> None:
        """If start_as_current_observation raises, should fall back to noop."""
        from graph.state import ResearchState

        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = Exception("Langfuse error")

        mock_configure._client = mock_client

        state = ResearchState(query="test", judge_score=70, iteration=1)
        with trace_routing_decision("routing", state) as tracer:
            assert isinstance(tracer, _NoopTracer)
