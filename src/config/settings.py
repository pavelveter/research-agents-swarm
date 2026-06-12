from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI / LLM configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def normalize_openai_base_url(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = value.rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")].rstrip("/")
        return normalized or None

    # Langfuse configuration
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # MCP server configuration
    mcp_host: str = Field(default="127.0.0.1", alias="MCP_HOST")
    mcp_port: int = Field(default=8765, alias="MCP_PORT")

    # Search provider configuration
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    serpapi_api_key: str = Field(default="", alias="SERPAPI_API_KEY")
    searxng_base_url: str = Field(default="", alias="SEARXNG_BASE_URL")

    # ── News sender channel configuration ──
    # Email (Resend)
    resend_api_key: str = Field(default="", alias="RESEND_API_KEY")
    resend_from: str = Field(default="", alias="RESEND_FROM")
    resend_to: str = Field(default="", alias="RESEND_TO")
    news_send_email: bool = Field(default=False, alias="NEWS_SEND_EMAIL")
    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    news_send_telegram: bool = Field(default=False, alias="NEWS_SEND_TELEGRAM")
    # Discord
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")
    news_send_discord: bool = Field(default=False, alias="NEWS_SEND_DISCORD")
    # File output
    news_output_file: str = Field(default="", alias="NEWS_OUTPUT_FILE")
    # Log file output (plain text, no ANSI codes)
    news_log_file: str = Field(default="", alias="NEWS_LOG_FILE")
    # Domain override (ai, policy, biotech, finance, general)
    research_domain: str = Field(default="", alias="RESEARCH_DOMAIN")

    @property
    def llm_api_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        return self.openai_api_key

    @property
    def llm_base_url(self) -> Optional[str]:
        return self.openai_base_url or None

    @property
    def mcp_url(self) -> str:
        return f"http://{self.mcp_host}:{self.mcp_port}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
