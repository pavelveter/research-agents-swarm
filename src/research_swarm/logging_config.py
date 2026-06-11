from __future__ import annotations

import logging
import sys

_QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "langfuse",
    "urllib3",
)


def setup_terminal_logging(level: int = logging.INFO) -> None:
    """Configure readable logs for the CLI demo."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-5s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def preview(text: str, limit: int = 160) -> str:
    """Truncate long strings for log lines."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."
