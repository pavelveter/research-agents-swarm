"""Agentic news sender — runs research-swarm on a theme and pushes the report to
configurable channels (email, Telegram, Discord).

Theme is read from ``theme-of-the-news.txt`` in the project root.
Each channel is independently gated by a ``NEWS_SEND_*`` env-var flag.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config.settings import get_settings
from graph.state import ResearchState
from graph.workflow import build_workflow
from llm.client import shutdown_llm_client
from logging_config import separator, setup_terminal_logging
from observability.langfuse import shutdown_observability
from utils import merge_state

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Channel abstraction
# ─────────────────────────────────────────────────────────────────


class NewsChannel(ABC):
    """Abstract channel for delivering a research report."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def send(self, subject: str, report: str) -> bool:
        """Deliver the report. Return True on success."""
        ...


# ─────────────────────────────────────────────────────────────────
# Email (Resend)
# ─────────────────────────────────────────────────────────────────

RESEND_API = "https://api.resend.com/emails"


class EmailChannel(NewsChannel):
    def __init__(self) -> None:
        s = get_settings()
        self._api_key = s.resend_api_key
        self._from = s.resend_from
        self._to = s.resend_to

    @property
    def name(self) -> str:
        return "email"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._from and self._to)

    async def send(self, subject: str, report: str) -> bool:
        if not self.is_configured:
            logger.warning("Email channel not configured — skipping")
            return False
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    RESEND_API,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "from": self._from,
                        "to": [self._to],
                        "subject": subject,
                        "html": report,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                email_id = data.get("id", "unknown")
                logger.info("Report sent via Resend to %s (id=%s)", self._to, email_id)
            return True
        except Exception as exc:
            logger.error("Resend send failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────────────────────────


class TelegramChannel(NewsChannel):
    def __init__(self) -> None:
        s = get_settings()
        self._token = s.telegram_bot_token
        self._chat_id = s.telegram_chat_id

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    async def send(self, subject: str, report: str) -> bool:
        if not self.is_configured:
            logger.warning("Telegram channel not configured — skipping")
            return False
        try:
            # Telegram limits messages to 4096 chars; truncate with ellipsis
            max_len = 4000
            body = f"{subject}\n\n{report}"
            if len(body) > max_len:
                body = body[: max_len - 30] + "\n\n…\n(report truncated)"

            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": body,
                        "disable_web_page_preview": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    logger.error("Telegram API error: %s", data)
                    return False
            logger.info("Report sent via Telegram to chat %s", self._chat_id)
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────
# Discord
# ─────────────────────────────────────────────────────────────────


class DiscordChannel(NewsChannel):
    def __init__(self) -> None:
        s = get_settings()
        self._webhook = s.discord_webhook_url

    @property
    def name(self) -> str:
        return "discord"

    @property
    def is_configured(self) -> bool:
        return bool(self._webhook)

    async def send(self, subject: str, report: str) -> bool:
        if not self.is_configured:
            logger.warning("Discord channel not configured — skipping")
            return False
        try:
            # Discord embeds have a 4096 char description limit
            max_len = 4000
            if len(report) > max_len:
                report = report[: max_len - 30] + "\n\n…\n*(report truncated)*"

            payload = {
                "embeds": [
                    {
                        "title": subject,
                        "description": report,
                        "color": 0x3B82F6,  # blue
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(self._webhook, json=payload)
                resp.raise_for_status()
            logger.info("Report sent via Discord webhook")
            return True
        except Exception as exc:
            logger.error("Discord send failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────
# Channel registry
# ─────────────────────────────────────────────────────────────────


def _enabled_channels() -> list[NewsChannel]:
    s = get_settings()
    channels: list[NewsChannel] = []
    if s.news_send_email:
        channels.append(EmailChannel())
    if s.news_send_telegram:
        channels.append(TelegramChannel())
    if s.news_send_discord:
        channels.append(DiscordChannel())
    return channels


# ─────────────────────────────────────────────────────────────────
# Theme reader
# ─────────────────────────────────────────────────────────────────

THEME_FILE = Path("theme-of-the-news.txt")


def read_theme(path: Path | None = None) -> str:
    """Read the news theme from the theme file."""
    target = path or THEME_FILE
    if not target.exists():
        msg = f"Theme file not found: {target.resolve()}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Theme file is empty: {target.resolve()}")
    logger.info("Theme loaded (%d chars): %s", len(text), text[:80])
    return text


# ─────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────


def _build_subject(theme: str, score: int) -> str:
    return f"[News Digest] {theme[:60]} (score: {score}/100)"


async def run_news_sender(theme_path: Path | None = None) -> dict[str, bool]:
    """Entry point: read theme, run research-swarm, send to enabled channels.

    Returns ``{channel_name: success_bool}`` for each channel.
    """
    setup_terminal_logging()
    settings = get_settings()

    # 1. Read theme
    theme = read_theme(theme_path)
    logger.info(separator("News Sender Starting"))
    logger.info("Theme:  %s", theme)
    logger.info("Model:  %s", settings.openai_model)

    # 2. Determine channels (file output counts as a valid sink)
    channels = _enabled_channels()
    has_file_output = bool(settings.news_output_file.strip())
    if not channels and not has_file_output:
        logger.warning(
            "No channels or file output enabled — set NEWS_SEND_EMAIL / NEWS_SEND_TELEGRAM / NEWS_SEND_DISCORD / NEWS_OUTPUT_FILE"
        )
        return {}
    if channels:
        logger.info("Channels: %s", [c.name for c in channels])
    if has_file_output:
        logger.info("File output: %s", settings.news_output_file)

    try:
        # 3. Run research-swarm
        state = ResearchState(query=theme)
        workflow = build_workflow()
        result = state
        async for event in workflow.astream(state, stream_mode="updates"):
            for node, _update in event.items():
                logger.info("Finished node: %s", node)
            result = merge_state(result, event)

        # 4. Build report
        report_body = (
            result.final_report.summary
            if result.final_report
            else "(no report produced)"
        )
        subject = _build_subject(theme, result.judge_score)
        logger.info(
            "Report ready | chars=%s score=%s", len(report_body), result.judge_score
        )

        # 4b. Write report to file if output path configured
        output_file = settings.news_output_file.strip()
        if output_file:
            try:
                out_path = Path(output_file)
                # Insert YYYY-MM-DD date before the filename stem
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                dated_name = f"{today}-{out_path.name}"
                out_path = out_path.with_name(dated_name)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    f"{subject}\n\n{report_body}",
                    encoding="utf-8",
                )
                logger.info(
                    "Report written to %s (%d bytes)",
                    out_path.resolve(),
                    out_path.stat().st_size,
                )
            except Exception as exc:
                logger.error("Failed to write output file: %s", exc)

        # 5. Send to all enabled channels
        results: dict[str, bool] = {}
        for channel in channels:
            logger.info("Sending via %s ...", channel.name)
            ok = await channel.send(subject, report_body)
            results[channel.name] = bool(ok)

        # 6. Health summary
        from search import get_orchestrator

        orchestrator = get_orchestrator()
        if orchestrator.health is not None:
            orchestrator.health.log_summary()

        logger.info(separator("News Sender Complete"))
        for ch, ok in results.items():
            logger.info("  %s: %s", ch, "✓" if ok else "✗")
        return results

    finally:
        await shutdown_llm_client()
        shutdown_observability()


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for ``news-sender`` command."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else THEME_FILE
    results = asyncio.run(run_news_sender(path))
    if not results:
        settings = get_settings()
        output_file = settings.news_output_file.strip()
        if output_file:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            dated = Path(output_file).with_name(f"{today}-{Path(output_file).name}")
            print(f"Report written to file: {dated}")
            sys.exit(0)
        print("No channels were enabled or all failed.")
        sys.exit(1)
    failed = [ch for ch, ok in results.items() if not ok]
    if failed:
        print(f"Some channels failed: {', '.join(failed)}")
        sys.exit(1)
    print(f"News sent successfully via: {', '.join(results.keys())}")


if __name__ == "__main__":
    main()
