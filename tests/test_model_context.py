"""Tests for model_context.py — model metadata lookup."""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

METADATA_SAMPLE = [
    {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Anthropic: Claude 3.5 Haiku",
        "context_length": 200000,
        "max_completion_tokens": 8192,
        "created": 1729500000,
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Google: Gemini 2.0 Flash",
        "context_length": 1048576,
        "max_completion_tokens": 8192,
        "created": 1735689600,
    },
    {
        "id": "openai/gpt-4o",
        "name": "OpenAI: GPT-4o",
        "context_length": 128000,
        "max_completion_tokens": 16384,
        "created": 1715367049,
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "OpenAI: GPT-4o Mini",
        "context_length": 128000,
        "max_completion_tokens": 16384,
        "created": 1720000000,
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek: V3",
        "context_length": 65536,
        "max_completion_tokens": 8192,
        "created": 1735689600,
    },
]


@pytest.fixture(autouse=True)
def _load_sample_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a sample model_metadata.json and patch the module path."""
    metadata_file = tmp_path / "model_metadata.json"
    metadata_file.write_text(json.dumps(METADATA_SAMPLE), encoding="utf-8")

    # Patch the path resolver before importing the module
    import kitty.providers.model_context as mc

    monkeypatch.setattr(mc, "_METADATA_PATH", metadata_file)
    # Force reload of metadata with patched path
    mc._load_metadata.cache_clear()
    yield
    mc._load_metadata.cache_clear()


# ---------------------------------------------------------------------------
# provider_config override
# ---------------------------------------------------------------------------


class TestProviderConfigOverride:
    """Manual context_window in provider_config takes top priority."""

    def test_override_takes_precedence(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openai",
            model="gpt-4o",
            provider_config={"context_window": 50000},
        )
        assert result == 50000

    def test_override_without_metadata_match(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="ollama",
            model="llama3-custom",
            provider_config={"context_window": 32000},
        )
        assert result == 32000


# ---------------------------------------------------------------------------
# Exact model ID lookup
# ---------------------------------------------------------------------------


class TestExactModelLookup:
    """Lookup by exact model ID from the metadata table."""

    def test_openrouter_model_exact_match(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openrouter",
            model="openai/gpt-4o",
        )
        assert result == 128000

    def test_openrouter_model_with_org_prefix(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openrouter",
            model="anthropic/claude-3.5-haiku",
        )
        assert result == 200000

    def test_non_openrouter_provider_strips_prefix(self):
        """For non-OpenRouter providers, model like 'openai/gpt-4o' should
        match the metadata entry by stripping the provider prefix."""
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openai",
            model="gpt-4o",
        )
        assert result == 128000

    def test_deepseek_model(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openrouter",
            model="deepseek/deepseek-chat",
        )
        assert result == 65536

    def test_google_gemini(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
        )
        assert result == 1048576


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


class TestFallback:
    """Unknown models fall back to DEFAULT_CONTEXT_TOKENS."""

    def test_unknown_model_returns_default(self):
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        result = get_model_context_tokens(
            provider="ollama",
            model="llama3-custom",
        )
        assert result == DEFAULT_CONTEXT_TOKENS

    def test_unknown_provider_model_returns_default(self):
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        result = get_model_context_tokens(
            provider="novita",
            model="some-unknown-model",
        )
        assert result == DEFAULT_CONTEXT_TOKENS

    def test_invalid_provider_config_override_falls_back_to_metadata(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openai",
            model="gpt-4o",
            provider_config={"context_window": "not-a-number"},
        )
        assert result == 128000

    def test_bool_provider_config_override_is_ignored(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openai",
            model="gpt-4o",
            provider_config={"context_window": True},
        )
        assert result == 128000

    def test_ambiguous_suffix_match_returns_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import kitty.providers.model_context as mc

        metadata_file = tmp_path / "ambiguous.json"
        metadata_file.write_text(
            json.dumps(
                [
                    {"id": "provider-a/shared-model", "name": "A", "context_length": 10000},
                    {"id": "provider-b/shared-model", "name": "B", "context_length": 20000},
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(mc, "_METADATA_PATH", metadata_file)
        mc._load_metadata.cache_clear()

        result = mc.get_model_context_tokens(provider="custom_openai", model="shared-model")
        assert result == mc.DEFAULT_CONTEXT_TOKENS
        mc._load_metadata.cache_clear()

    def test_invalid_metadata_context_length_returns_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import kitty.providers.model_context as mc

        metadata_file = tmp_path / "invalid-context.json"
        metadata_file.write_text(
            json.dumps([{"id": "openai/gpt-bad", "name": "Bad", "context_length": "unknown"}]),
            encoding="utf-8",
        )
        monkeypatch.setattr(mc, "_METADATA_PATH", metadata_file)
        mc._load_metadata.cache_clear()

        result = mc.get_model_context_tokens(provider="openai", model="gpt-bad")
        assert result == mc.DEFAULT_CONTEXT_TOKENS
        mc._load_metadata.cache_clear()


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    """Model ID matching is case-insensitive."""

    def test_uppercase_model(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openrouter",
            model="OpenAI/GPT-4o",
        )
        assert result == 128000

    def test_mixed_case_model(self):
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(
            provider="openai",
            model="GPT-4o-Mini",
        )
        assert result == 128000


# ---------------------------------------------------------------------------
# Empty / missing metadata file
# ---------------------------------------------------------------------------


class TestMissingMetadata:
    """Graceful handling when metadata file is missing or empty."""

    def test_missing_file_returns_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import kitty.providers.model_context as mc

        monkeypatch.setattr(mc, "_METADATA_PATH", tmp_path / "nonexistent.json")
        mc._load_metadata.cache_clear()

        result = mc.get_model_context_tokens(provider="openai", model="gpt-4o")
        assert result == mc.DEFAULT_CONTEXT_TOKENS
        mc._load_metadata.cache_clear()

    def test_empty_array_returns_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import kitty.providers.model_context as mc

        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]", encoding="utf-8")
        monkeypatch.setattr(mc, "_METADATA_PATH", empty_file)
        mc._load_metadata.cache_clear()

        result = mc.get_model_context_tokens(provider="openai", model="gpt-4o")
        assert result == mc.DEFAULT_CONTEXT_TOKENS
        mc._load_metadata.cache_clear()

    def test_invalid_json_returns_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import kitty.providers.model_context as mc

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(mc, "_METADATA_PATH", bad_file)
        mc._load_metadata.cache_clear()

        result = mc.get_model_context_tokens(provider="openai", model="gpt-4o")
        assert result == mc.DEFAULT_CONTEXT_TOKENS
        mc._load_metadata.cache_clear()
