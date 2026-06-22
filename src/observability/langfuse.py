"""Langfuse observability for research-swarm.

Improvements over the previous minimal module:

* ``should_export_span`` is wired to ``is_default_export_span`` so only
  Langfuse-originated spans and known LLM instrumentation scopes (incl.
  ``gen_ai.*``) are exported. The same filter is reused as a callback
  passed to ``Langfuse(...)``.
* Up to 3 init attempts with 1s sleep between them; on final failure a
  clear, actionable warning is logged (check keys / network).
* The LLM client returns an ``LLMResponse`` with token usage, cost and
  TTFT. ``record_llm_response`` renders that into the active
  observation's metadata so the Langfuse UI shows ``ttft_s``, model,
  prompt version and cost per agent.
* ``session_context`` wraps ``client.propagate_attributes(session_id=...)``
  so every observation in a workflow run shares a session without
  needing to thread ``session_id`` through the SDK kwargs (the v4.x API
  surfaced during SDK review). Tracing calls degrade to no-op whenever
  Langfuse is unconfigured or any SDK call raises.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextlib import contextmanager, suppress
from typing import Any, Iterator, Optional

from config.settings import get_settings
from graph.state import ResearchState
from llm.client import LLMResponse, split_cost_usd

logger = logging.getLogger(__name__)

_DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
_INIT_RETRY_ATTEMPTS = 3
_INIT_RETRY_BACKOFF_S = 1.0


# ── Helpers ─────────────────────────────────────────────────────────


def _hash_query(query: str) -> str:
    """Return a 12-char sha1 hash of the query for cheap de-duplication in traces."""
    return hashlib.sha1((query or "").encode("utf-8")).hexdigest()[:12]


def generate_session_id(prefix: str = "research-swarm") -> str:
    """Return a unique session id for grouping all observations of one workflow run."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _calc_cost_split(
    model: str, token_usage: dict[str, int]
) -> tuple[float, float, float]:
    """Return (input_cost, output_cost, total_cost) in USD using the pricing table."""
    return split_cost_usd(model, token_usage)


# ── Langfuse client setup with retry + span filter ──────────────────


def _configure_langfuse() -> None:
    """Configure the Langfuse client with retries and a default span export filter.

    Best-effort: never raises. Errors are logged so observability failures
    never break the actual workflow. Retries ``_INIT_RETRY_ATTEMPTS``
    times with ``_INIT_RETRY_BACKOFF_S`` seconds between attempts (the
    spec is 3 attempts with 1s backoff, per ``TODO.md``).
    """
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse credentials are not configured; tracing will be disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to enable."
        )
        _configure_langfuse._client = None
        return

    last_error: Optional[BaseException] = None
    for attempt in range(1, _INIT_RETRY_ATTEMPTS + 1):
        try:
            from langfuse import Langfuse

            try:
                from langfuse.span_filter import is_default_export_span

                should_export = is_default_export_span
            except ImportError:  # pragma: no cover
                should_export = None

            _configure_langfuse._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host or _DEFAULT_LANGFUSE_HOST,
                debug=bool(settings.langfuse_debug),
                should_export_span=should_export,
            )
            logger.info(
                "Langfuse initialised | host=%s attempt=%d/%d debug=%s",
                settings.langfuse_host,
                attempt,
                _INIT_RETRY_ATTEMPTS,
                settings.langfuse_debug,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Langfuse init attempt %d/%d failed: %s",
                attempt,
                _INIT_RETRY_ATTEMPTS,
                exc,
            )
            if attempt < _INIT_RETRY_ATTEMPTS:
                time.sleep(_INIT_RETRY_BACKOFF_S)

    if last_error is not None:
        logger.error(
            "Langfuse tracing is DISABLED after %d init attempts: %s. "
            "Action: check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / network.",
            _INIT_RETRY_ATTEMPTS,
            last_error,
        )
        _configure_langfuse._client = None


# Eagerly initialise so the client is ready before agents run.
_configure_langfuse()


def shutdown_observability() -> None:
    """Flush and close Langfuse background workers before process exit."""
    client = getattr(_configure_langfuse, "_client", None)
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.debug("Langfuse shutdown failed", exc_info=True)
    finally:
        _configure_langfuse._client = None


# ── Tracer wrappers ─────────────────────────────────────────────────


class _LangfuseTracer:
    """Rich wrapper around the Langfuse observation returned by ``start_as_current_observation``.

    ``agent_name`` is captured from the enclosing ``trace_agent(...)`` call
    so individual ``record_llm_response(...)`` invocations don't have to
    pass it. Per-call overrides stay available for sub-phases
    (``planner_initial`` vs ``planner_refine``).
    """

    def __init__(self, span: Any, *, agent_name: Optional[str] = None) -> None:
        self._span = span
        self._agent_name = agent_name

    def update_observation(self, **kwargs: Any) -> None:
        with suppress(Exception):
            self._span.update(**kwargs)

    def end(self, **kwargs: Any) -> None:
        with suppress(Exception):
            self._span.update(**kwargs)

    def record_llm_response(
        self,
        response: LLMResponse,
        *,
        temperature: Optional[float] = None,
        prompt_version: Optional[str] = None,
        agent_name: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Attach token usage, latency and cost to the active observation's metadata.

        Agent-shaped observations in Langfuse v4.x don't accept native
        ``usage_details``/``cost_details``/``model`` kwargs (those are
        generation-only). We therefore write everything into ``metadata``
        so ttft_s / model / token counts show up in the UI consistently.
        Token counts are echoed as both flat fields and a ``langfuse``
        sub-dict for downstream tooling that prefers nested shapes.

        ``agent_name`` resolution order: explicit kwarg > tracer-level
        attribute set by ``trace_agent(...)`` > ``"unknown"`` fallback.
        """
        if response is None:
            return
        input_tokens = response.token_usage.get("input_tokens", 0) or 0
        output_tokens = response.token_usage.get("output_tokens", 0) or 0
        total_tokens = response.token_usage.get("total_tokens") or (
            input_tokens + output_tokens
        )
        input_cost, output_cost, total_cost = _calc_cost_split(
            response.model or "", response.token_usage
        )
        # If the LLMResponse pre-computed cost, prefer it (it may use a
        # model name match we don't know about here).
        if response.cost_usd and not input_cost and not output_cost:
            total_cost = round(float(response.cost_usd), 6)

        resolved_agent = agent_name or self._agent_name or "unknown"
        metadata: dict[str, Any] = {
            "agent_name": resolved_agent,
            "model": response.model,
            "ttft_s": round(float(response.ttft_s or 0.0), 4),
            "duration_s": round(float(response.duration_s or 0.0), 4),
            "ttft_measured": bool(
                response.completion_start_time_ns
                and response.ttft_s > 0.0
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd_total": total_cost,
            "cost_usd_input": input_cost,
            "cost_usd_output": output_cost,
            "langfuse": {
                "usage_details": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                },
                "cost_details": {
                    "input": input_cost,
                    "output": output_cost,
                    "total": total_cost,
                },
                "model": response.model,
            },
        }
        if temperature is not None:
            metadata["temperature"] = float(temperature)
        if prompt_version:
            metadata["prompt_version"] = prompt_version
        if extra:
            metadata.update(extra)
        with suppress(Exception):
            self._span.update(metadata=metadata)


class _NoopTracer:
    """Minimal no-op tracer used when Langfuse is unavailable.

    Mirrors ``_LangfuseTracer``'s surface so callers don't have to
    special-case degraded mode. All record calls are inert.
    """

    def __init__(self, *, agent_name: Optional[str] = None) -> None:
        self._agent_name = agent_name

    def update_observation(self, **kwargs: Any) -> None:
        return None

    def end(self, **kwargs: Any) -> None:
        return None

    def record_llm_response(
        self,
        response: LLMResponse,
        *,
        temperature: Optional[float] = None,
        prompt_version: Optional[str] = None,
        agent_name: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        return None


# ── Context managers ────────────────────────────────────────────────


@contextmanager
def session_context(session_id: Optional[str]):
    """Apply ``client.propagate_attributes(session_id=...)`` for the workflow's lifetime.

    Langfuse v4.x groups observations by ``session_id`` only when the
    attribute has been propagated through ``propagate_attributes`` —
    there is no ``session_id`` kwarg on
    ``start_as_current_observation`` in v4.7.1. ``session_context`` is
    a no-op when Langfuse is disabled or ``session_id`` is empty/None.
    """
    if not session_id:
        yield
        return
    client = getattr(_configure_langfuse, "_client", None)
    if client is None:
        yield
        return
    try:
        with client.propagate_attributes(session_id=session_id):
            yield
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("session_context failed for %s: %s", session_id, exc)
        yield


@contextmanager
def trace_agent(
    agent_name: str,
    input_data: Optional[dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    query_hash: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    prompt_version: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
):
    """Context manager providing Langfuse tracing for agent execution.

    All observability hints are merged into the observation's
    ``metadata`` so the Langfuse UI shows them next to the existing
    ``agent_name`` key. ``session_id`` is forwarded via
    ``propagate_attributes`` (not as a kwarg on
    ``start_as_current_observation``) — wrap the workflow with
    ``session_context(session_id)`` for grouping.
    """
    client = getattr(_configure_langfuse, "_client", None)
    if client is None:
        yield _NoopTracer()
        return

    metadata: dict[str, Any] = {"agent_name": agent_name}
    if query_hash:
        metadata["query_hash"] = query_hash
    if model:
        metadata["model"] = model
    if temperature is not None:
        metadata["temperature"] = float(temperature)
    if prompt_version:
        metadata["prompt_version"] = prompt_version
    if session_id:
        metadata["session_id"] = session_id
    if extra_metadata:
        metadata.update(extra_metadata)

    try:
        observation = client.start_as_current_observation(
            name=agent_name,
            as_type="agent",
            input=input_data or None,
            metadata=metadata,
        )
    except Exception as exc:
        logger.debug(
            "Langfuse tracing is not active for %s: %s", agent_name, exc,
            exc_info=True,
        )
        yield _NoopTracer()
        return

    with observation as span:
        yield _LangfuseTracer(span, agent_name=agent_name)


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
            metadata={
                "routing_name": routing_name,
                "session_id": state.session_id,
            },
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
