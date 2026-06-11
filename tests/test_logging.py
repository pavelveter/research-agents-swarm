from __future__ import annotations

import logging
import sys

import pytest

from research_swarm.logging_config import preview, setup_terminal_logging


class TestSetupTerminalLogging:
    """Tests for logging configuration."""

    def test_adds_handler_to_root_logger(self) -> None:
        """First call should add a StreamHandler to the root logger."""
        # Get root logger and clear existing handlers
        root = logging.getLogger()
        root.handlers.clear()

        setup_terminal_logging(level=logging.INFO)
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert root.level == logging.INFO

    def test_second_call_does_not_add_duplicate_handler(self) -> None:
        """Second call should return early and not add duplicate handlers."""
        root = logging.getLogger()
        root.handlers.clear()

        setup_terminal_logging(level=logging.DEBUG)
        initial_count = len(root.handlers)

        setup_terminal_logging(level=logging.INFO)
        assert len(root.handlers) == initial_count

    def test_quiet_loggers_are_set_to_warning(self) -> None:
        """Third-party loggers should be silenced to WARNING level."""
        root = logging.getLogger()
        root.handlers.clear()

        setup_terminal_logging()

        quiet_loggers = [
            "httpx",
            "httpcore",
            "openai",
            "langchain",
            "langchain_core",
            "langchain_openai",
            "langgraph",
            "langfuse",
            "urllib3",
        ]
        for name in quiet_loggers:
            logger = logging.getLogger(name)
            assert logger.level == logging.WARNING, f"Logger {name} should be WARNING, got {logger.level}"

    def test_custom_level(self) -> None:
        """Should accept custom log level."""
        root = logging.getLogger()
        root.handlers.clear()

        setup_terminal_logging(level=logging.DEBUG)
        assert root.level == logging.DEBUG

    def test_handler_stream_is_stderr(self) -> None:
        """Handler should write to stderr."""
        root = logging.getLogger()
        root.handlers.clear()

        setup_terminal_logging()
        handler = root.handlers[0]
        assert hasattr(handler, "stream")
        assert handler.stream is sys.stderr  # type: ignore[union-attr]

    def test_formatter_is_configured(self) -> None:
        """Custom formatter should be set on the handler."""
        root = logging.getLogger()
        root.handlers.clear()

        setup_terminal_logging()
        handler = root.handlers[0]
        assert handler.formatter is not None
        # Verify it's a logging.Formatter instance
        assert isinstance(handler.formatter, logging.Formatter)


class TestPreview:
    """Tests for the preview helper function."""

    def test_short_text_unchanged(self) -> None:
        assert preview("hello") == "hello"

    def test_exact_limit_not_truncated(self) -> None:
        text = "a" * 160
        assert preview(text) == text

    def test_long_text_truncated(self) -> None:
        text = "a" * 200
        result = preview(text)
        assert len(result) <= 160
        assert result.endswith("...")

    def test_custom_limit(self) -> None:
        text = "a" * 50
        result = preview(text, limit=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_whitespace_normalized(self) -> None:
        text = "hello    world\n\n\ttest"
        result = preview(text)
        assert result == "hello world test"

    def test_very_long_text_truncated_properly(self) -> None:
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 20
        result = preview(text, limit=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_empty_string(self) -> None:
        assert preview("") == ""

    def test_only_whitespace(self) -> None:
        assert preview("   \n\t  ") == ""

    def test_text_exactly_at_limit(self) -> None:
        text = "x" * 40
        result = preview(text, limit=40)
        assert result == text
        assert not result.endswith("...")

    def test_text_one_over_limit(self) -> None:
        text = "x" * 41
        result = preview(text, limit=40)
        assert result.endswith("...")
        assert len(result) == 40
