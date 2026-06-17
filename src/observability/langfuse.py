from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from config.settings import get_settings
from graph.state import ResearchState

logger = logging.getLogger(__name__)


def _configure_langfuse() -> None:
    "Best-effort configuration of Langfuse from settings."
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse credentials are not configured; tracing will be disabled"
        )
        return
    try:
        from langfuse import Langfuse

        _configure_langfuse._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover - optional integration
        logger.error("Failed to initialize Langfuse: %s", exc)


_configure_langfuse()


def shutdown_observability() -> None:
    "Flush and close Langfuse background workers before process exit."
    client = getattr(_configure_langfuse, "_client", None)
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.debug("Langfuse shutdown failed", exc_info=True)
    finally:
        _configure_langfuse._client = None


class _LangfuseTracer:
    def __init__(self, span: Any) -> None:
        self._span = span

    def update_observation(self, **kwargs: Any) -> None:
        self._span.update(**kwargs)

    def end(self, **kwargs: Any) -> None:
        self._span.update(**kwargs)


class _NoopTracer:
    "Minimal no-op tracer used when Langfuse is unavailable."

    def update_observation(self, **kwargs: Any) -> None:
        return None

    def end(self, **kwargs: Any) -> None:
        return None


@contextmanager
def trace_agent(agent_name: str, input_data: dict[str, Any]) -> Iterator[Any]:
    "Context manager providing Langfuse tracing for agent execution."
    client = getattr(_configure_langfuse, "_client", None)
    if client is None:
        yield _NoopTracer()
        return
    try:
        observation = client.start_as_current_observation(
            name=agent_name,
            as_type="agent",
            input=input_data or None,
            metadata={"agent_name": agent_name},
        )
    except Exception:
        logger.debug("Langfuse tracing is not active for %s", agent_name, exc_info=True)
        yield _NoopTracer()
        return

    with observation as span:
        yield _LangfuseTracer(span)


@contextmanager
def trace_routing_decision(
    routing_name: str, state: ResearchState
) -> Iterator[Any]:
    """Context manager providing Langfuse tracing for routing decisions."""
    client = getattr(_configure_langfuse, "_client", None)
    if client is None:
        yield _NoopTracer()
        return
    try:
        observation = client.start_as_current_observation(
            name=routing_name,
            as_type="chain",
            input={
                "iteration": state.iteration,
                "score": state.judge_score,
                "score_delta": state.score_delta,
                "missing_topics": state.missing_topics,
                "max_iterations": state.max_iterations,
                "new_evidence_found": state.new_evidence_found,
            },
            metadata={"routing_name": routing_name},
        )
    except Exception:
        logger.debug(
            "Langfuse tracing is not active for %s", routing_name, exc_info=True
        )
        yield _NoopTracer()
        return

    with observation as span:
        tracer = _LangfuseTracer(span)
        yield tracer
        tracer.update_observation(
            output={
                "stop_reason": state.stop_reason,
                "score": state.judge_score,
                "delta": state.score_delta,
                "iteration": state.iteration,
            }
        )
