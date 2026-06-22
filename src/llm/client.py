from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings
from logging_config import preview

logger = logging.getLogger(__name__)
_settings = get_settings()
_default_llm: ChatOpenAI | None = None

# ── Cost estimation table ──────────────────────────────────────────
# Prices are USD per 1k tokens. Source: OpenAI public pricing (2024-2025)
# — values are best-effort approximations that drive totalCost UX, NOT
# invoicing. Extend ``_BUILTIN_PRICING_TABLE`` or ``data/pricing.json``
# when new models are introduced.
_BUILTIN_PRICING_TABLE: dict[str, tuple[float, float]] = {
    # model name → (input_cost_usd_per_1k, output_cost_usd_per_1k)
    # ── OpenAI ──────────────────────────────────────────────────────────
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-2024-08-06": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-turbo-preview": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),
    "o1-preview": (0.015, 0.06),
    "o1-mini": (0.003, 0.012),
    "o3-mini": (0.0011, 0.0044),
    # ── Anthropic Claude ─────────────────────────────────────────────────
    "claude-3-5-sonnet-latest": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-latest": (0.001, 0.005),
    "claude-3-opus-latest": (0.005, 0.025),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    # ── Google Gemini (default tiers: ≤200k for 1.5 Pro, ≤128k for 1.5 Flash) ─
    "gemini-2.5-pro": (0.00125, 0.01),
    "gemini-2.5-flash": (0.0003, 0.0025),
    "gemini-2.5-flash-lite": (0.0001, 0.0004),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "gemini-2.0-flash-lite": (0.000075, 0.0003),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-flash-8b": (0.0000375, 0.00015),
    # ── Mistral AI ────────────────────────────────────────────────────────
    "mistral-large-latest": (0.0005, 0.0015),
    "mistral-large-2407": (0.0005, 0.0015),
    "mistral-medium-latest": (0.0004, 0.002),
    "mistral-small-latest": (0.00015, 0.0006),
    "codestral-latest": (0.0003, 0.0009),
    # Conservative fallback for unknown models — schema-required key.
    "_default": (0.001, 0.003),
}

_DEFAULT_PRICING_TABLE_RELATIVE_PATH = Path("data") / "pricing.json"


def _resolve_pricing_table_path() -> Path:
    """Return the active pricing-table path, honouring ``PRICING_TABLE_PATH``.

    Falls back to ``<project_root>/data/pricing.json`` when the env var is
    empty or unset. ``project_root`` is the grandparent of this module's
    package directory (``src/llm/client.py`` → ``src/llm`` → ``src`` →
    ``project_root``).
    """
    configured = (get_settings().pricing_table_path or "").strip()
    if configured:
        return Path(configured).expanduser()
    # parents[0]=src/llm, parents[1]=src, parents[2]=project_root.
    project_root = Path(__file__).resolve().parents[2]
    return project_root / _DEFAULT_PRICING_TABLE_RELATIVE_PATH


def _load_pricing_table_from_disk(path: Path) -> dict[str, tuple[float, float]]:
    """Parse ``path`` as a JSON flat-dict pricing table with eager validation.

    Schema: ``{ "<model>": [<input_usd_per_1k>, <output_usd_per_1k>], ... }``.
    The ``_default`` key is required. Rows missing ``_default`` are merged
    over built-in rates; rows present in the file override the
    corresponding built-in keys.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pricing table root must be a JSON object")
    parsed: dict[str, tuple[float, float]] = {}
    for model, rates in raw.items():
        # Leading-underscore keys are reserved for metadata (`_comment`,
        # future `_version`, etc.) and are silently skipped — only `_default`
        # is consumed below.
        if model.startswith("_") and model != "_default":
            continue
        if not isinstance(rates, (list, tuple)) or len(rates) != 2:
            raise ValueError(
                f"pricing row for {model!r} must be a 2-element [input, output] array"
            )
        input_rate, output_rate = rates
        try:
            input_f = float(input_rate)
            output_f = float(output_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"pricing rates for {model!r} must be numeric (got {rates!r})"
            ) from exc
        if input_f < 0 or output_f < 0:
            raise ValueError(
                f"pricing rates for {model!r} must be non-negative"
            )
        parsed[model] = (input_f, output_f)
    if "_default" not in parsed:
        raise ValueError("pricing table must include a '_default' key")
    return parsed


@lru_cache(maxsize=1)
def _get_pricing_table() -> dict[str, tuple[float, float]]:
    """Return the merged pricing table used by ``split_cost_usd``/``estimate_cost_usd``.

    Looks up ``PRICING_TABLE_PATH`` (via settings), parses the JSON file,
    and overlays it on a copy of ``_BUILTIN_PRICING_TABLE`` so unseen
    models still resolve via the built-in rates. Falls back to the
    built-in table when the file is missing or invalid — degraded
    behaviour is logged but never raises, keeping the workflow running.
    """
    path = _resolve_pricing_table_path()
    try:
        from_disk = _load_pricing_table_from_disk(path)
    except FileNotFoundError:
        logger.info(
            "Pricing table %s not found — using built-in defaults", path
        )
        return dict(_BUILTIN_PRICING_TABLE)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Pricing table %s is invalid (%s) — using built-in defaults", path, exc
        )
        return dict(_BUILTIN_PRICING_TABLE)
    except OSError as exc:
        logger.warning(
            "Pricing table %s could not be read (%s) — using built-in defaults",
            path,
            exc,
        )
        return dict(_BUILTIN_PRICING_TABLE)

    merged: dict[str, tuple[float, float]] = dict(_BUILTIN_PRICING_TABLE)
    merged.update(from_disk)
    logger.info(
        "Pricing table loaded from %s (models=%d, builtins preserved=%d)",
        path,
        len(from_disk),
        len(_BUILTIN_PRICING_TABLE) - len(set(merged) - set(from_disk)),
    )
    return merged


def reset_pricing_cache() -> None:
    """Clear the cached pricing table so the next lookup reloads from disk.

    Useful for tests and operators who monkey-patch the file content at runtime.
    """
    _get_pricing_table.cache_clear()


@dataclass
class LLMResponse:
    """Rich return type for LLM calls — backward-compatible via .content.

    Carries everything needed for Langfuse observability: token usage,
    model identity, latency (TTFT + total duration), and an estimated cost
    in USD so the Langfuse UI's totalCost reflects real usage.
    """

    content: str
    """Raw text content from the model. Used as a string in all existing call sites."""

    model: str
    """Effective model name (overrides take precedence)."""

    ttft_s: float = 0.0
    """Seconds elapsed before the first streamed token arrived. 0.0 when
    the model returned non-streamingly or no chunks were received."""

    duration_s: float = 0.0
    """Total wall-clock seconds for the LLM call (request → full response)."""

    token_usage: dict[str, int] = field(default_factory=dict)
    """Standard {input_tokens, output_tokens, total_tokens} dict — empty
    when the upstream provider didn't return usage metadata."""

    cost_usd: float = 0.0
    """Estimated cost in USD, derived from _PRICING_PER_1K_TOKENS."""

    completion_start_time_ns: Optional[int] = None
    """Nanosecond timestamp of the first chunk — useful for observability
    backends that compute TTFT from first-event timing."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Reservoir for provider-specific fields (finish_reason, id, etc.)."""


# ── Helpers ────────────────────────────────────────────────────────


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute an estimated cost in USD from a token-usage breakdown.

    Uses an exact-match lookup against ``_PRICING_PER_1K_TOKENS`` and falls
    back to the conservative ``_default`` rate. Sub-cent amounts are
    rounded to 6 decimals so a noisy stream of micro-costs still has
    meaningful precision when summed.
    """
    input_cost, output_cost, total_cost = split_cost_usd(
        model, {"input_tokens": input_tokens, "output_tokens": output_tokens}
    )
    return total_cost


def split_cost_usd(
    model: str,
    token_usage: dict[str, int],
) -> tuple[float, float, float]:
    """Return ``(input_cost, output_cost, total_cost)`` in USD.

    Public helper that wraps the pricing lookup so other modules (notably
    ``observability/langfuse.py``) don't have to reach for the private
    pricing table. Reads from the cached, file-driven
    ``_get_pricing_table()`` so operators can override rates via
    ``PRICING_TABLE_PATH`` without code changes.
    """
    input_tokens = int(token_usage.get("input_tokens", 0) or 0)
    output_tokens = int(token_usage.get("output_tokens", 0) or 0)
    table = _get_pricing_table()
    input_rate, output_rate = table.get(model, table["_default"])
    input_cost = (input_tokens / 1000.0) * input_rate
    output_cost = (output_tokens / 1000.0) * output_rate
    return (
        round(input_cost, 6),
        round(output_cost, 6),
        round(input_cost + output_cost, 6),
    )


def _extract_usage(response: AIMessage) -> dict[str, int]:
    """Extract a normalised ``{input, output, total}`` token-usage dict.

    The shape varies by provider:
    - OpenAI-compatible (langchain-openai): ``response.usage_metadata`` provides
      input_tokens / output_tokens / total_tokens.
    - Older surfaces expose ``response_metadata['token_usage']`` /
      ``usage['prompt_tokens']`` etc. Both are inspected defensively.
    - Some proxies omit usage entirely — returns an empty dict in that case.
    """
    if response is None:
        return {}
    usage: dict[str, int] = {}
    metadata = getattr(response, "usage_metadata", None)
    if isinstance(metadata, dict):
        for key_in, key_out in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = metadata.get(key_in)
            if isinstance(value, int):
                usage[key_out] = value
    if not usage:
        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get(
            "usage"
        )
        if isinstance(token_usage, dict):
            for key_in, key_out in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = token_usage.get(key_in)
                if isinstance(value, int):
                    usage[key_out] = value
    # Derive total when only parts are present, so cost estimation still works.
    if "total_tokens" not in usage and {"input_tokens", "output_tokens"} <= usage.keys():
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors worth retrying.

    Covers OS-level errors, asyncio timeouts, httpx HTTP errors, and
    OpenAI API errors (rate limits, server errors, timeouts).
    """
    if isinstance(exc, (OSError, TimeoutError)):
        return True
    try:
        from httpx import HTTPError as HttpxError

        if isinstance(exc, HttpxError):
            return True
    except ImportError:
        pass
    try:
        from openai import APIError as OpenaiError

        if isinstance(exc, OpenaiError):
            return True
    except ImportError:
        pass
    return False


def _build_llm(**overrides: Any) -> ChatOpenAI:
    params: dict[str, Any] = {
        "model": _settings.openai_model,
        "api_key": _settings.openai_api_key,
    }
    if _settings.openai_base_url:
        params["base_url"] = _settings.openai_base_url
    if overrides:
        params.update(overrides)
    return ChatOpenAI(**params)


def get_llm(**overrides: Any) -> ChatOpenAI:
    "Return a configured ChatOpenAI instance using project settings."
    return _build_llm(**overrides)


def _get_shared_llm() -> ChatOpenAI:
    global _default_llm
    if _default_llm is None:
        _default_llm = get_llm()
    return _default_llm


async def shutdown_llm_client() -> None:
    "Close the shared async HTTP client before the event loop shuts down."
    global _default_llm
    if _default_llm is None:
        return
    async_client = getattr(_default_llm, "root_async_client", None)
    if async_client is not None and not getattr(async_client, "is_closed", True):
        await async_client.aclose()
    _default_llm = None


def _coerce_response(
    response: AIMessage,
    start_ns: int,
    end_ns: int,
    *,
    ttft_s_override: Optional[float] = None,
) -> LLMResponse:
    """Build an ``LLMResponse`` from a non-streaming ``ainvoke`` result.

    ``ttft_s_override`` exists because we cannot measure the true
    start-to-first-token window for non-streaming calls — the caller
    passes an explicit ``ttft_s`` value (typically ``0.0``) instead of
    having us conflate the whole-call duration with TTFT.
    """
    duration_s = max(0.0, (end_ns - start_ns) / 1_000_000_000.0)
    content = str(getattr(response, "content", "") or "")
    model_name = str(getattr(response, "response_metadata", {}).get("model_name")
                      or getattr(response, "type", "") or "")
    usage = _extract_usage(response)
    cost = estimate_cost_usd(
        model_name,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return LLMResponse(
        content=content,
        model=model_name,
        ttft_s=ttft_s_override if ttft_s_override is not None else duration_s,
        duration_s=duration_s,
        token_usage=usage,
        cost_usd=cost,
        completion_start_time_ns=None,
    )


@retry(
    retry=_is_retryable,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    reraise=True,
)
async def _invoke_ainvoke(llm: ChatOpenAI, messages: list[BaseMessage]) -> AIMessage:
    """Fallback non-streaming path — kept for proxies that don't support astream."""
    return await llm.ainvoke(messages)


@retry(
    retry=_is_retryable,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    reraise=True,
)
async def _invoke_astream(llm: ChatOpenAI, messages: list[BaseMessage]) -> AIMessage:
    """Streaming path — assembles chunks into a final AIMessage while measuring TTFT."""
    chunks: list[AIMessage] = []
    content_parts: list[str] = []
    async for chunk in llm.astream(messages):
        chunks.append(chunk)
        content_parts.append(str(getattr(chunk, "content", "") or ""))
    # Merge chunk list into a single AIMessage so callers see one response object
    merged_content = "".join(content_parts)
    last = chunks[-1] if chunks else AIMessage(content="")
    usage_metadata = getattr(last, "usage_metadata", None)
    response_metadata = dict(getattr(last, "response_metadata", {}) or {})
    return AIMessage(
        content=merged_content,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
        id=getattr(last, "id", None),
    )


async def invoke_messages(
    messages: list[BaseMessage], **overrides: Any
) -> LLMResponse:
    """Call the LLM and return a rich ``LLMResponse``.

    Uses ``astream`` so we can measure the real time-to-first-token (TTFT).
    Implements exponential backoff for transient failures and falls back
    to ``ainvoke`` for proxies that don't support streaming. Token usage
    and an estimated USD cost are attached when the provider returns them,
    so Langfuse's totalCost reflects real usage.
    """
    llm = get_llm(**overrides) if overrides else _get_shared_llm()
    logger.info(
        "LLM request | model=%s messages=%d overrides=%s",
        llm.model_name,
        len(messages),
        sorted(overrides.keys()) if overrides else "{}",
    )
    start_ns = time.monotonic_ns()
    first_chunk_ns: Optional[int] = None
    chunks: list[AIMessage] = []
    content_parts: list[str] = []
    final_message: AIMessage | None = None
    try:
        async for chunk in llm.astream(messages):
            if first_chunk_ns is None:
                first_chunk_ns = time.monotonic_ns()
            chunks.append(chunk)
            content_parts.append(str(getattr(chunk, "content", "") or ""))
    except Exception as stream_exc:
        # NB: TTFT cannot be measured when the proxy falls back to ainvoke —
        # we deliberately leave ``ttft_s = 0.0`` rather than misreporting
        # whole-call latency as TTFT.
        if _is_retryable(stream_exc):
            logger.debug(
                "astream failed with retryable error (%s); falling back to ainvoke",
                stream_exc,
            )
            response = await _invoke_ainvoke(llm, messages)
            end_ns = time.monotonic_ns()
            result = _coerce_response(response, start_ns, end_ns, ttft_s_override=0.0)
        else:
            logger.debug(
                "astream failed with non-retryable error (%s); falling back to ainvoke",
                stream_exc,
            )
            try:
                response = await _invoke_ainvoke(llm, messages)
            except Exception:
                logger.error("LLM invocation failed: %s", stream_exc, exc_info=True)
                raise
            end_ns = time.monotonic_ns()
            result = _coerce_response(response, start_ns, end_ns, ttft_s_override=0.0)
        logger.info(
            "LLM response (ainvoke fallback) | model=%s chars=%d "
            "ttft_s=0.000 (non-streaming) duration_s=%.3f preview=%r",
            result.model,
            len(result.content),
            result.duration_s,
            preview(result.content),
        )
        return result

    # Stream succeeded — assemble chunks and measure TTFT.
    final_message = chunks[-1] if chunks else AIMessage(content="")
    # TTFT can only be measured when at least one streaming chunk arrived.
    # If the proxy returns zero chunks (empty model response), we set
    # ttft_s = 0.0 explicitly so the metric isn't misreported as the
    # whole-call duration — same discipline as the ainvoke fallback path.
    ttft_s = (
        (first_chunk_ns - start_ns) / 1_000_000_000.0
        if first_chunk_ns is not None
        else 0.0
    )
    end_ns = time.monotonic_ns()
    ttft_s = (first_chunk_ns - start_ns) / 1_000_000_000.0
    duration_s = (end_ns - start_ns) / 1_000_000_000.0
    content = "".join(content_parts)
    usage_metadata = getattr(final_message, "usage_metadata", None)
    usage = _extract_usage(final_message)
    if not usage and isinstance(usage_metadata, dict):
        # Some providers embed only usage_metadata on the last chunk.
        usage = _extract_usage(
            AIMessage(content="", usage_metadata=usage_metadata)
        )
    model_name = (
        str(getattr(final_message, "response_metadata", {}).get("model_name"))
        or llm.model_name
    )
    cost = estimate_cost_usd(
        model_name,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    logger.info(
        "LLM response (astream) | model=%s chunks=%d chars=%d ttft_s=%.3f "
        "duration_s=%.3f cost_usd=%.6f preview=%r",
        model_name,
        len(chunks),
        len(content),
        ttft_s,
        duration_s,
        cost,
        preview(content),
    )
    return LLMResponse(
        content=content,
        model=model_name,
        ttft_s=ttft_s,
        duration_s=duration_s,
        token_usage=usage,
        cost_usd=cost,
        completion_start_time_ns=first_chunk_ns,
    )



@retry(
    retry=_is_retryable,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    reraise=True,
)
async def ainvoke(messages: list[BaseMessage], **overrides: Any) -> LLMResponse:
    """Backward-compatible alias for ``invoke_messages`` (returns ``LLMResponse``)."""
    return await invoke_messages(messages, **overrides)
