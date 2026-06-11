from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from research_swarm.config.settings import get_settings
from research_swarm.logging_config import preview

logger = logging.getLogger(__name__)
_settings = get_settings()


def get_llm(**overrides: Any) -> ChatOpenAI:
    "Return a configured ChatOpenAI instance using project settings."
    params: dict[str, Any] = {
        "model": _settings.openai_model,
        "api_key": _settings.openai_api_key,
    }
    if _settings.openai_base_url:
        params["base_url"] = _settings.openai_base_url
    if overrides:
        params.update(overrides)
    return ChatOpenAI(**params)


async def invoke_messages(messages: list[BaseMessage], **overrides: Any) -> str:
    llm = get_llm(**overrides)
    logger.info("LLM request | model=%s messages=%s", llm.model_name, len(messages))
    response = await llm.ainvoke(messages)
    content = str(response.content or "")
    logger.info(
        "LLM response | chars=%s preview=%r",
        len(content),
        preview(content),
    )
    return content


async def ainvoke(messages: list[BaseMessage], **overrides: Any) -> str:
    return await invoke_messages(messages, **overrides)
