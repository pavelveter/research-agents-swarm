from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from research_swarm.config.settings import get_settings
from research_swarm.logging_config import preview

logger = logging.getLogger(__name__)
_settings = get_settings()
_default_llm: ChatOpenAI | None = None


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


@retry(
    retry=_is_retryable,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    reraise=True,
)
async def invoke_messages(messages: list[BaseMessage], **overrides: Any) -> str:
    """Call the LLM with exponential backoff for transient failures.

    Retries up to 4 times on network/timeout errors with 1s→2s→4s→8s backoff.
    Non-retryable errors (e.g. auth failures) propagate immediately.
    """
    llm = get_llm(**overrides) if overrides else _get_shared_llm()
    logger.info("LLM request | model=%s messages=%s", llm.model_name, len(messages))
    response = await llm.ainvoke(messages)
    content = str(response.content or "")
    logger.info(
        "LLM response | chars=%s preview=%r",
        len(content),
        preview(content),
    )
    return content


@retry(
    retry=_is_retryable,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    reraise=True,
)
async def ainvoke(messages: list[BaseMessage], **overrides: Any) -> str:
    return await invoke_messages(messages, **overrides)
