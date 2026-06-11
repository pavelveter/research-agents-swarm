from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from research_swarm.config.settings import get_settings

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
