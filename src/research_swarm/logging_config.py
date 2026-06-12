from __future__ import annotations

import logging
import re
import sys
import textwrap

# ── ANSI color palette ───────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_COLORS: dict[str, str] = {
    "grey":   "\033[90m",
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "magenta":"\033[95m",
    "cyan":   "\033[96m",
    "white":  "\033[97m",
}

_BG_RED = "\033[41m"

# Map levelno → (label, color)
_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG:    ("DEBUG", _COLORS["grey"]),
    logging.INFO:     ("INFO ", _COLORS["green"]),
    logging.WARNING:  ("WARN ", _COLORS["yellow"]),
    logging.ERROR:    ("ERROR", _COLORS["red"]),
    logging.CRITICAL: ("CRIT ", _BG_RED + _BOLD + _COLORS["white"]),
}

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

_MAX_NAME_WIDTH = 26  # truncate long logger names for readability


def _shorten_name(name: str) -> str:
    """Abbreviate dotted logger names, e.g. research_swarm.agents.searcher → r.s.agents.searcher"""
    if len(name) <= _MAX_NAME_WIDTH:
        return name
    parts = name.split(".")
    # Keep the last 2 segments intact, abbreviate earlier ones
    if len(parts) <= 2:
        return textwrap.shorten(name, width=_MAX_NAME_WIDTH, placeholder="…")
    abbr = ".".join(p[0] for p in parts[:-2])
    tail = ".".join(parts[-2:])
    short = f"{abbr}.{tail}"
    if len(short) > _MAX_NAME_WIDTH:
        short = f"…{tail}"
    return short


class _ColoredFormatter(logging.Formatter):
    """ANSI-colored log formatter with aligned columns and key=value highlighting."""

    _KV_RE = re.compile(r"(\w[\w_]*)=(" r"[^\s,]+" r")")

    def format(self, record: logging.LogRecord) -> str:
        style = _LEVEL_STYLES.get(record.levelno, ("????", ""))
        label, color = style
        level_str = f"{color}{_BOLD}{label}{_RESET}"

        # dim grey timestamp
        ts = self.formatTime(record, "%H:%M:%S")
        ts_str = f"{_COLORS['grey']}{ts}{_RESET}"

        # cyan logger name, right-padded
        name = _shorten_name(record.name)
        name_str = f"{_COLORS['cyan']}{name:<{_MAX_NAME_WIDTH}}{_RESET}"

        # message with key=value highlighting (skip for errors to preserve uniform color)
        if record.levelno >= logging.ERROR:
            msg = f"{color}{record.getMessage()}{_RESET}"
        else:
            msg = self._highlight_kv(str(record.getMessage()))

        return f"{ts_str} {level_str} {name_str} {msg}"

    def _highlight_kv(self, text: str) -> str:
        """Highlight key=value pairs: grey key, bold white value."""
        def _repl(m: re.Match) -> str:
            key = m.group(1)
            val = m.group(2)
            return f"{_COLORS['grey']}{key}={_RESET}{_BOLD}{val}{_RESET}"
        return self._KV_RE.sub(_repl, text)


def setup_terminal_logging(level: int = logging.INFO) -> None:
    """Configure colorful, human-readable logs for the CLI."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColoredFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def separator(title: str = "", char: str = "─", width: int = 60) -> str:
    """Return a styled section separator string for use in log messages.

    Example:
        logger.info(separator("Search Health Summary"))
        # prints: ────── Search Health Summary ────────────────────
    """
    if not title:
        return f"{_COLORS['grey']}{_DIM}{char * width}{_RESET}"
    side = (width - len(title) - 2) // 2
    left = char * max(side, 1)
    right = char * max(width - len(title) - len(left) - 2, 1)
    return f"{_COLORS['grey']}{_DIM}{left} {_COLORS['magenta']}{_BOLD}{title}{_RESET}{_COLORS['grey']}{_DIM} {right}{_RESET}"


def preview(text: str, limit: int = 160) -> str:
    """Truncate long strings for log lines."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."
