from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm.client import (
    LLMResponse,
    ainvoke,
    estimate_cost_usd,
    get_llm,
    invoke_messages,
    reset_pricing_cache,
    split_cost_usd,
    _BUILTIN_PRICING_TABLE,
    _extract_usage,
    _get_pricing_table,
    _load_pricing_table_from_disk,
    _resolve_pricing_table_path,
)


class TestGetLLM:
    """Tests for the get_llm factory function."""

    @patch("llm.client.ChatOpenAI")
    def test_uses_settings_model(self, mock_chat_openai: MagicMock) -> None:
        """Should create ChatOpenAI with model from settings."""
        mock_chat_openai.return_value = MagicMock()

        get_llm()

        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert "model" in call_kwargs
        assert "api_key" in call_kwargs

    @patch("llm.client.ChatOpenAI")
    def test_accepts_overrides(self, mock_chat_openai: MagicMock) -> None:
        """Should pass override parameters to ChatOpenAI."""
        mock_chat_openai.return_value = MagicMock()

        get_llm(model="gpt-4o-mini", temperature=0.5)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs.get("model") == "gpt-4o-mini"
        assert call_kwargs.get("temperature") == 0.5

    @patch("llm.client.ChatOpenAI")
    def test_returns_chat_openai_instance(self, mock_chat_openai: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        assert get_llm() is mock_instance


class TestEstimateCost:
    """Tests for estimate_cost_usd() — pure cost math."""

    def test_known_model_zero_tokens_returns_zero(self) -> None:
        assert estimate_cost_usd("gpt-4o", 0, 0) == 0.0

    def test_known_model_input_only(self) -> None:
        # gpt-4o: $0.0025 per 1k input
        # 1000 input tokens → $0.0025
        assert estimate_cost_usd("gpt-4o", 1000, 0) == 0.0025

    def test_known_model_output_only(self) -> None:
        # gpt-4o: $0.01 per 1k output
        # 1000 output tokens → $0.01
        assert estimate_cost_usd("gpt-4o", 0, 1000) == 0.01

    def test_known_model_combined(self) -> None:
        # 1000 in + 1000 out → 0.0025 + 0.01 = 0.0125
        assert estimate_cost_usd("gpt-4o", 1000, 1000) == 0.0125

    def test_unknown_model_uses_default_pricing(self) -> None:
        # default: $0.001 / $0.003 per 1k
        # 1000 in + 1000 out → 0.001 + 0.003 = 0.004
        assert estimate_cost_usd("unknown-future-model", 1000, 1000) == 0.004

    def test_returns_rounded_value(self) -> None:
        # gpt-3.5-turbo: $0.0005 / $0.0015 — rounding to 6 decimals
        cost = estimate_cost_usd("gpt-3.5-turbo", 3, 7)
        assert isinstance(cost, float)
        assert cost >= 0.0


class TestExtractUsage:
    """Tests for _extract_usage() — normalises token usage across providers."""

    def test_usage_metadata_shapes(self) -> None:
        msg = AIMessage(
            content="hello",
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 7,
                "total_tokens": 12,
            },
        )
        usage = _extract_usage(msg)
        assert usage == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}

    def test_response_metadata_token_usage_alias(self) -> None:
        msg = AIMessage(
            content="hi",
            response_metadata={"token_usage": {"prompt_tokens": 4, "completion_tokens": 6}},
        )
        usage = _extract_usage(msg)
        assert usage == {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}

    def test_no_usage_data_returns_empty(self) -> None:
        msg = AIMessage(content="hi")
        usage = _extract_usage(msg)
        assert usage == {}

    def test_none_message_returns_empty(self) -> None:
        assert _extract_usage(None) == {}  # type: ignore[arg-type]

    def test_partial_usage_derives_total(self) -> None:
        # AIMessage requires total_tokens in usage_metadata, so we exercise
        # the partial-derive path via response_metadata (which is more
        # realistic for proxies that omit totals).
        msg = AIMessage(
            content="hi",
            response_metadata={
                "token_usage": {"prompt_tokens": 2, "completion_tokens": 3}
            },
        )
        usage = _extract_usage(msg)
        assert usage["input_tokens"] == 2
        assert usage["output_tokens"] == 3
        assert usage["total_tokens"] == 5


class TestInvokeMessages:
    """Tests for the invoke_messages async function."""

    @pytest.mark.asyncio
    async def test_returns_llm_response_object(self) -> None:
        """invoke_messages returns an LLMResponse, not a raw string."""
        # Build two AIMessage chunks so streaming returns content
        chunk_a = AIMessage(content="Hello, ")
        chunk_b = AIMessage(
            content="world!",
            usage_metadata={
                "input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 5,
            },
            response_metadata={"model_name": "gpt-4o"},
        )

        async def fake_astream(_messages: list[HumanMessage]):
            yield chunk_a
            yield chunk_b

        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm.model_name = "gpt-4o"

        with patch("llm.client.get_llm", return_value=mock_llm):
            messages = [SystemMessage(content="You are helpful"), HumanMessage(content="Hi")]
            result = await invoke_messages(messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello, world!"
        assert result.model == "gpt-4o"
        assert result.token_usage["input_tokens"] == 2
        assert result.token_usage["output_tokens"] == 3
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_handles_empty_content(self) -> None:
        """Should handle empty string content from the LLM."""
        empty_chunk = AIMessage(content="")

        async def fake_astream(_messages: list[HumanMessage]):
            yield empty_chunk

        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm.model_name = "gpt-4o"

        with patch("llm.client.get_llm", return_value=mock_llm):
            messages = [HumanMessage(content="Hi")]
            result = await invoke_messages(messages)

        assert isinstance(result, LLMResponse)
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_passes_overrides_to_get_llm(self) -> None:
        """Should forward override kwargs to get_llm."""
        chunk = AIMessage(content="ok")

        async def fake_astream(_messages: list[HumanMessage]):
            yield chunk

        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm.model_name = "gpt-4o"

        with patch("llm.client.get_llm", return_value=mock_llm) as mock_get:
            messages = [HumanMessage(content="Hi")]
            await invoke_messages(messages, temperature=0.3)

        mock_get.assert_called_once_with(temperature=0.3)

    @pytest.mark.asyncio
    async def test_falls_back_to_ainvoke_on_stream_failure(self) -> None:
        """If astream raises a non-retryable error, fall back to ainvoke."""
        mock_response = AIMessage(
            content="fallback",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(
            side_effect=Exception("streaming not supported")
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.model_name = "gpt-4o"

        with (
            patch("llm.client.get_llm", return_value=mock_llm),
            patch("llm.client.time.sleep"),  # silence tenacity backoff
        ):
            messages = [HumanMessage(content="Hi")]
            result = await invoke_messages(messages)

        assert result.content == "fallback"
        mock_llm.ainvoke.assert_called_once()


class TestAinvoke:
    """Tests for the ainvoke alias function — returns LLMResponse."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_llm_response(self) -> None:
        chunk = AIMessage(content="test response")

        async def fake_astream(_messages: list[HumanMessage]):
            yield chunk

        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm.model_name = "gpt-4o"

        with patch("llm.client.get_llm", return_value=mock_llm):
            messages = [HumanMessage(content="Hello")]
            result = await ainvoke(messages, temperature=0.7)

        assert isinstance(result, LLMResponse)
        assert result.content == "test response"


class TestPricingTableLoader:
    """Tests for the file-driven pricing-table loader."""

    def setup_method(self) -> None:
        # Always start from a clean cache so per-test mutations don't bleed.
        reset_pricing_cache()

    def teardown_method(self) -> None:
        reset_pricing_cache()

    def test_resolve_pricing_table_path_default_is_data_pricing_json(self) -> None:
        with patch("llm.client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(pricing_table_path="")
            resolved = _resolve_pricing_table_path()
        # Default file lives at <project_root>/data/pricing.json
        assert resolved.name == "pricing.json"
        assert resolved.parent.name == "data"
        # CRITICAL: must NOT be inside src/ — that would silently fall back to
        # built-in defaults. Walk 3 levels up from src/llm/client.py.
        assert "src" not in resolved.parts, (
            f"default pricing table resolved under src/: {resolved} — "
            "parents[2] indexing is broken"
        )

    def test_resolve_pricing_table_path_default_file_actually_exists(self) -> None:
        """Smoke check: with no PRICING_TABLE_PATH override the resolver
        points at a real on-disk file shipped with the repo. Catches
        regressions in the project-root calculation.
        """
        with patch("llm.client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(pricing_table_path="")
            resolved = _resolve_pricing_table_path()
        assert resolved.is_file(), (
            f"bundled pricing file not found at {resolved} — check "
            "_resolve_pricing_table_path() depth"
        )

    def test_resolve_pricing_table_path_env_override(self) -> None:
        with patch("llm.client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock(
                pricing_table_path="/srv/secrets/openai_rates.json"
            )
            resolved = _resolve_pricing_table_path()
        assert str(resolved) == "/srv/secrets/openai_rates.json"

    def test_load_pricing_table_from_disk_parses_flat_dict(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "_comment": "test fixture",
            "gpt-4o": [0.99, 0.88],
            "_default": [0.001, 0.003],
        }
        path = tmp_path / "rates.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = _load_pricing_table_from_disk(path)
        assert loaded == {
            "gpt-4o": (0.99, 0.88),
            "_default": (0.001, 0.003),
        }

    def test_load_pricing_table_missing_default_raises(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"gpt-4o": [0.0, 0.0]}), encoding="utf-8")
        with pytest.raises(ValueError, match="_default"):
            _load_pricing_table_from_disk(path)

    def test_load_pricing_table_invalid_row_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"_default": [0.0, 0.0], "gpt-4o": [0.01]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="2-element"):
            _load_pricing_table_from_disk(path)

    def test_get_pricing_table_falls_back_when_file_missing(
        self, tmp_path: Path
    ) -> None:
        # tmp_path is *not* the configured location → FileNotFoundError path
        bogus = tmp_path / "does_not_exist.json"
        with (
            patch("llm.client._resolve_pricing_table_path", return_value=bogus),
            patch("llm.client.logger") as mock_logger,
        ):
            table = _get_pricing_table()
            assert table == dict(_BUILTIN_PRICING_TABLE)
            assert mock_logger.info.called  # degraded-mode log line

    def test_get_pricing_table_falls_back_on_invalid_json(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "garbage.json"
        path.write_text("{this is not json", encoding="utf-8")
        with (
            patch("llm.client._resolve_pricing_table_path", return_value=path),
            patch("llm.client.logger") as mock_logger,
        ):
            table = _get_pricing_table()
            assert table == dict(_BUILTIN_PRICING_TABLE)
            assert mock_logger.warning.called  # warning path for invalid file

    def test_get_pricing_table_merges_file_over_builtins(
        self, tmp_path: Path
    ) -> None:
        # File only overrides gpt-4o; other builtins must remain.
        path = tmp_path / "rates.json"
        path.write_text(
            json.dumps(
                {"gpt-4o": [0.111, 0.222], "_default": [0.05, 0.5]}
            ),
            encoding="utf-8",
        )
        # Reset cache before loading fresh file.
        with patch("llm.client._resolve_pricing_table_path", return_value=path):
            reset_pricing_cache()
            table = _get_pricing_table()
        assert table["gpt-4o"] == (0.111, 0.222)
        assert table["_default"] == (0.05, 0.5)
        # Built-ins preserved for models not in the file.
        assert table["gpt-4o-mini"] == _BUILTIN_PRICING_TABLE["gpt-4o-mini"]
        assert table["o3-mini"] == _BUILTIN_PRICING_TABLE["o3-mini"]

    def test_reset_pricing_cache_reloads_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "rates.json"
        path.write_text(
            json.dumps({"_default": [0.001, 0.003], "gpt-4o": [0.5, 1.5]}),
            encoding="utf-8",
        )
        with patch("llm.client._resolve_pricing_table_path", return_value=path):
            reset_pricing_cache()
            table_a = _get_pricing_table()
            assert table_a["gpt-4o"] == (0.5, 1.5)
            assert _get_pricing_table() is table_a  # cached — same object

            # Mutate the file and reset; new table loads fresh data.
            path.write_text(
                json.dumps(
                    {"_default": [0.001, 0.003], "gpt-4o": [0.7, 1.9]}
                ),
                encoding="utf-8",
            )
            reset_pricing_cache()
            table_b = _get_pricing_table()
            assert table_b["gpt-4o"] == (0.7, 1.9)
            assert table_b is not table_a

    def test_split_cost_usd_uses_overridden_rate(self, tmp_path: Path) -> None:
        # End-to-end: operator overrides gpt-4o via on-disk table;
        # public API picks it up automatically.
        path = tmp_path / "rates.json"
        path.write_text(
            json.dumps({"_default": [0.001, 0.003], "gpt-4o": [0.444, 0.555]}),
            encoding="utf-8",
        )
        with patch("llm.client._resolve_pricing_table_path", return_value=path):
            reset_pricing_cache()
            # 1000 input + 1000 output tokens at overridden rates.
            in_c, out_c, total = split_cost_usd(
                "gpt-4o",
                {"input_tokens": 1000, "output_tokens": 1000},
            )
        assert in_c == pytest.approx(0.444, abs=1e-6)
        assert out_c == pytest.approx(0.555, abs=1e-6)
        assert total == pytest.approx(0.999, abs=1e-6)


class TestMultiProviderPricing:
    """Cost math for non-OpenAI providers — Anthropic / Google Gemini / Mistral.

    These exercise both the bundled ``data/pricing.json`` and the
    ``_BUILTIN_PRICING_TABLE`` fallback so any rate drift between the two
    would surface here.
    """

    def setup_method(self) -> None:
        reset_pricing_cache()

    def teardown_method(self) -> None:
        reset_pricing_cache()

    # ── Anthropic ──────────────────────────────────────────────────

    def test_claude_3_5_sonnet_latest_1000_in_1000_out(self) -> None:
        # 1000 in @ $0.003/1k = $0.003; 1000 out @ $0.015/1k = $0.015 → $0.018
        assert estimate_cost_usd("claude-3-5-sonnet-latest", 1000, 1000) == pytest.approx(
            0.018, abs=1e-6
        )

    def test_claude_3_5_sonnet_input_only(self) -> None:
        # 1000 input → $0.003
        assert estimate_cost_usd(
            "claude-3-5-sonnet-20241022", 1000, 0
        ) == pytest.approx(0.003, abs=1e-6)

    def test_claude_3_opus_legacy(self) -> None:
        # claude-3-opus-20240229 still the $15/$75-per-1M model
        assert estimate_cost_usd("claude-3-opus-20240229", 1000, 1000) == pytest.approx(
            0.09, abs=1e-6
        )

    def test_claude_3_haiku_legacy(self) -> None:
        # 1000 in @ $0.00025 = $0.00025; 1000 out @ $0.00125 = $0.00125 → $0.0015
        assert estimate_cost_usd("claude-3-haiku-20240307", 1000, 1000) == pytest.approx(
            0.0015, abs=1e-6
        )

    # ── Google Gemini ──────────────────────────────────────────────

    def test_gemini_2_5_flash_1000_in_1000_out(self) -> None:
        # 1000 in @ $0.0003 = $0.0003; 1000 out @ $0.0025 = $0.0025 → $0.0028
        assert estimate_cost_usd("gemini-2.5-flash", 1000, 1000) == pytest.approx(
            0.0028, abs=1e-6
        )

    def test_gemini_1_5_pro_1000_in_1000_out(self) -> None:
        # 1000 in @ $0.00125 = $0.00125; 1000 out @ $0.005 = $0.005 → $0.00625
        assert estimate_cost_usd("gemini-1.5-pro", 1000, 1000) == pytest.approx(
            0.00625, abs=1e-6
        )

    def test_gemini_1_5_flash_8b_sub_cent_precision(self) -> None:
        # 1000 in @ $0.0000375 + 1000 out @ $0.00015 → $0.0001875
        assert estimate_cost_usd("gemini-1.5-flash-8b", 1000, 1000) == pytest.approx(
            0.0001875, abs=1e-6
        )

    # ── Mistral ────────────────────────────────────────────────────

    def test_mistral_large_latest_1000_in_1000_out(self) -> None:
        # 1000 in @ $0.0005 + 1000 out @ $0.0015 → $0.002
        assert estimate_cost_usd("mistral-large-latest", 1000, 1000) == pytest.approx(
            0.002, abs=1e-6
        )

    def test_mistral_small_same_as_gpt_4o_mini(self) -> None:
        # mistral-small-latest and gpt-4o-mini share the same 1.5c/6c per 1M rates.
        assert estimate_cost_usd(
            "mistral-small-latest", 1000, 1000
        ) == estimate_cost_usd("gpt-4o-mini", 1000, 1000)

    def test_codestral_mid_tier(self) -> None:
        # 1000 in @ $0.0003 + 1000 out @ $0.0009 → $0.0012
        assert estimate_cost_usd("codestral-latest", 1000, 1000) == pytest.approx(
            0.0012, abs=1e-6
        )

    # ── Built-in fallback matches bundled JSON ─────────────────────

    def test_builtin_table_matches_bundled_json_for_new_providers(self) -> None:
        """Bundled data/pricing.json and _BUILTIN_PRICING_TABLE must agree
        for every multi-provider key, so falling back to the builtin
        dict on a missing/corrupt file still produces the documented rate.
        """
        from pathlib import Path

        # parents[1] from tests/test_llm_client.py → tests/ → project_root.
        # (distinct from src/llm/client.py which needs parents[2] because the
        # package is one level deeper).
        bundled = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "pricing.json")
            .read_text(encoding="utf-8")
        )
        for model, builtin_rate in _BUILTIN_PRICING_TABLE.items():
            if model == "_default":
                continue
            assert model in bundled, f"bundled JSON missing provider model {model!r}"
            file_rate = tuple(bundled[model])
            assert file_rate == builtin_rate, (
                f"rate drift for {model!r}: builtin={builtin_rate} "
                f"file={file_rate}"
            )

    def test_claude_3_opus_latest_uses_introduced_rate(self) -> None:
        """The ``-latest`` alias maps to the current primary release (4.x
        family), which is significantly cheaper than the legacy 20240229
        release. Pinning both rates here guards against future drift.
        """
        # claude-3-opus-latest: $5/$25 per 1M = $0.005/$0.025 per 1k → $0.030
        assert estimate_cost_usd("claude-3-opus-latest", 1000, 1000) == pytest.approx(
            0.03, abs=1e-6
        )

    def test_mistral_large_versioned_alias_matches_latest(self) -> None:
        """``mistral-large-2407`` (the version-tagged Mistral Large 2 form)
        must NOT fall back to _default — it shares the latest rate.
        """
        assert estimate_cost_usd("mistral-large-2407", 1000, 1000) == pytest.approx(
            0.002, abs=1e-6
        )

    def test_split_cost_usd_returns_three_tuple_for_anthropic(self) -> None:
        """``split_cost_usd`` is what ``_LangfuseTracer.record_llm_response``
        consumes to populate ``cost_usd_input`` / ``cost_usd_output`` /
        ``cost_usd_total``. Confirm the 3-tuple shape for a non-OpenAI model.
        """
        in_c, out_c, total = split_cost_usd(
            "claude-3-5-sonnet-latest",
            {"input_tokens": 1000, "output_tokens": 1000},
        )
        assert in_c == pytest.approx(0.003, abs=1e-6)
        assert out_c == pytest.approx(0.015, abs=1e-6)
        assert total == pytest.approx(0.018, abs=1e-6)

    def test_split_cost_usd_returns_three_tuple_for_gemini(self) -> None:
        """Same shape check for Google Gemini — sub-cent precision must
        round-trip without collapsing to 0.000000. Tolerance covers
        banker's-rounding quirks at 6-decimal granularity (``round(0.0000375, 6)``
        returns ``0.000037`` in CPython because the IEEE 754 float repr
        is slightly less than the decimal value).
        """
        in_c, out_c, total = split_cost_usd(
            "gemini-1.5-flash-8b",
            {"input_tokens": 1000, "output_tokens": 1000},
        )
        # Pre-round through `round()` so the assertion mirrors the
        # contract — ``split_cost_usd`` applies ``round(_, 6)``.
        assert in_c == pytest.approx(round(0.0000375, 6), abs=1e-9)
        assert out_c == pytest.approx(round(0.00015, 6), abs=1e-9)
        assert total == pytest.approx(round(0.0000375, 6) + round(0.00015, 6), abs=1e-9)

    def test_unknown_model_falls_back_to_default(self) -> None:
        # Future / unknown model from any provider → _default (0.001, 0.003).
        assert estimate_cost_usd(
            "claude-unknown-future-2026", 1000, 1000
        ) == pytest.approx(0.004, abs=1e-6)
        assert estimate_cost_usd(
            "gemini-future-v9-pro", 1000, 1000
        ) == pytest.approx(0.004, abs=1e-6)
        assert estimate_cost_usd(
            "mistral-mega-xxl", 1000, 1000
        ) == pytest.approx(0.004, abs=1e-6)
