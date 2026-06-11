from __future__ import annotations

import pytest

from research_swarm.config.settings import Settings, get_settings


class TestSettings:
    """Tests for the Settings pydantic-settings model.

    Uses monkeypatch.setenv to explicitly control all environment variables
    so that tests are isolated from the user's .env file and shell environment.
    """

    def test_default_values(self, monkeypatch) -> None:
        """Test the default field values when env vars match defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("MCP_PORT", "8765")

        settings = Settings()
        assert settings.openai_model == "gpt-4o"
        assert settings.openai_base_url is None
        # Langfuse fields have no empty→None validator, so env "" stays as ""
        assert settings.langfuse_public_key == ""
        assert settings.langfuse_secret_key == ""
        assert settings.langfuse_host == "https://cloud.langfuse.com"
        assert settings.mcp_host == "127.0.0.1"
        assert settings.mcp_port == 8765

    def test_custom_values(self, monkeypatch) -> None:
        """Env vars set to custom values should be reflected."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-custom")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
        monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
        monkeypatch.setenv("MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("MCP_PORT", "9000")

        settings = Settings()
        assert settings.openai_model == "gpt-3.5-turbo"
        assert settings.openai_base_url == "https://api.example.com/v1"
        assert settings.langfuse_public_key == "pk-lf"
        assert settings.langfuse_secret_key == "sk-lf"
        assert settings.langfuse_host == "https://us.cloud.langfuse.com"
        assert settings.mcp_host == "0.0.0.0"
        assert settings.mcp_port == 9000

    def test_llm_api_key_property(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        settings = Settings()
        assert settings.llm_api_key == "sk-test"

    def test_llm_api_key_raises_when_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        settings = Settings()
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            _ = settings.llm_api_key

    def test_llm_base_url_property(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.com")
        settings = Settings()
        assert settings.llm_base_url == "https://api.test.com"

    def test_llm_base_url_returns_none_when_not_set(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        settings = Settings()
        assert settings.llm_base_url is None

    def test_mcp_url_property(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("MCP_PORT", "8765")
        settings = Settings()
        assert settings.mcp_url == "http://127.0.0.1:8765"

    def test_mcp_url_custom(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MCP_HOST", "10.0.0.1")
        monkeypatch.setenv("MCP_PORT", "5555")
        settings = Settings()
        assert settings.mcp_url == "http://10.0.0.1:5555"

    def test_env_var_loading(self, monkeypatch) -> None:
        """Settings should load values from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5")
        monkeypatch.setenv("OPENAI_BASE_URL", "")

        settings = Settings()
        assert settings.openai_api_key == "sk-from-env"
        assert settings.openai_model == "gpt-5"


class TestOpenAIBaseUrlNormalization:
    """Tests for the openai_base_url field validator."""

    def test_none_value_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        settings = Settings()
        assert settings.openai_base_url is None

    def test_empty_string_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        settings = Settings()
        assert settings.openai_base_url is None

    def test_trailing_slash_removed(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
        settings = Settings()
        assert settings.openai_base_url == "https://api.openai.com/v1"

    def test_chat_completions_path_removed(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        settings = Settings()
        assert settings.openai_base_url == "https://api.openai.com/v1"

    def test_chat_completions_with_trailing_slash_removed(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions/")
        settings = Settings()
        assert settings.openai_base_url == "https://api.openai.com/v1"

    def test_normal_url_preserved(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api.com")
        settings = Settings()
        assert settings.openai_base_url == "https://custom.api.com"

    def test_url_with_multiple_slashes(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test.com////")
        settings = Settings()
        assert settings.openai_base_url == "https://api.test.com"

    def test_without_v1_path(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai")
        settings = Settings()
        assert settings.openai_base_url == "https://api.groq.com/openai"

    def test_without_v1_trailing_slash(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/")
        settings = Settings()
        assert settings.openai_base_url == "https://api.groq.com/openai"


class TestSettingsAliases:
    """Verify environment variable aliases work."""

    def test_openai_api_key_alias(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        settings = Settings()
        assert settings.openai_api_key == "sk-from-env"

    def test_openai_model_alias(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        settings = Settings()
        assert settings.openai_model == "gpt-4o-mini"

    def test_openai_base_url_alias(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api.com/v1")
        settings = Settings()
        assert settings.openai_base_url == "https://custom.api.com/v1"

    def test_langfuse_public_key_alias(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-mykey")
        settings = Settings()
        assert settings.langfuse_public_key == "pk-mykey"

    def test_langfuse_secret_key_alias(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-mysecret")
        settings = Settings()
        assert settings.langfuse_secret_key == "sk-mysecret"


class TestGetSettings:
    """Tests for the get_settings factory function."""

    def test_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching_works(self) -> None:
        """lru_cache should return the same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
