from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from research_swarm.llm.client import ainvoke, get_llm, invoke_messages


class TestGetLLM:
    """Tests for the get_llm factory function."""

    @patch("research_swarm.llm.client.ChatOpenAI")
    def test_uses_settings_model(self, mock_chat_openai: MagicMock) -> None:
        """Should create ChatOpenAI with model from settings."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        llm = get_llm()

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert "model" in call_kwargs
        assert "api_key" in call_kwargs

    @patch("research_swarm.llm.client.ChatOpenAI")
    def test_accepts_overrides(self, mock_chat_openai: MagicMock) -> None:
        """Should pass override parameters to ChatOpenAI."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        llm = get_llm(model="gpt-4o-mini", temperature=0.5)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs.get("model") == "gpt-4o-mini"
        assert call_kwargs.get("temperature") == 0.5

    @patch("research_swarm.llm.client.ChatOpenAI")
    def test_returns_chat_openai_instance(self, mock_chat_openai: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        llm = get_llm()
        assert llm is mock_instance


class TestInvokeMessages:
    """Tests for the invoke_messages async function."""

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.get_llm")
    async def test_returns_content_string(self, mock_get_llm: MagicMock) -> None:
        """Should return the content as a string."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="Hello, world!")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        messages = [SystemMessage(content="You are helpful"), HumanMessage(content="Hi")]
        result = await invoke_messages(messages)

        assert result == "Hello, world!"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.get_llm")
    async def test_handles_empty_content(self, mock_get_llm: MagicMock) -> None:
        """Should handle empty string content from the LLM."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        messages = [HumanMessage(content="Hi")]
        result = await invoke_messages(messages)

        assert result == ""

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.get_llm")
    async def test_passes_overrides_to_get_llm(self, mock_get_llm: MagicMock) -> None:
        """Should forward override kwargs to get_llm."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="ok")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        messages = [HumanMessage(content="Hi")]
        await invoke_messages(messages, temperature=0.3)

        mock_get_llm.assert_called_once_with(temperature=0.3)

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.get_llm")
    async def test_ainvoke_called_with_messages(self, mock_get_llm: MagicMock) -> None:
        """Should call ainvoke with the provided messages."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="ok")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        messages = [SystemMessage(content="You are helpful"), HumanMessage(content="What is AI?")]
        await invoke_messages(messages)

        mock_llm.ainvoke.assert_called_once()
        call_args = mock_llm.ainvoke.call_args[0]
        assert call_args[0] == messages

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.get_llm")
    async def test_logs_request_and_response(self, mock_get_llm: MagicMock) -> None:
        """Should log info about the request and response (no exception)."""
        mock_llm = MagicMock()
        mock_response = AIMessage(content="AI response here")
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.model_name = "test-model"
        mock_get_llm.return_value = mock_llm

        messages = [HumanMessage(content="Hello")]
        result = await invoke_messages(messages)

        assert result == "AI response here"


class TestAinvoke:
    """Tests for the ainvoke alias function."""

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.invoke_messages")
    async def test_calls_invoke_messages(self, mock_invoke: AsyncMock) -> None:
        """ainvoke should delegate to invoke_messages."""
        mock_invoke.return_value = "test response"

        messages = [HumanMessage(content="Hello")]
        result = await ainvoke(messages, temperature=0.7)

        assert result == "test response"
        mock_invoke.assert_called_once_with(messages, temperature=0.7)

    @pytest.mark.asyncio
    @patch("research_swarm.llm.client.invoke_messages")
    async def test_ainvoke_without_overrides(self, mock_invoke: AsyncMock) -> None:
        """ainvoke should work without override kwargs."""
        mock_invoke.return_value = "ok"

        messages = [HumanMessage(content="Hi")]
        result = await ainvoke(messages)

        assert result == "ok"
        mock_invoke.assert_called_once_with(messages)
