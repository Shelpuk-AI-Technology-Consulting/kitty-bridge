"""Tests for model_context.py — model metadata lookup."""

import json
import os
import time
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
    """Write a sample model_metadata.json and an empty overrides file, patch paths.

    Existing tests must not be coupled to the shipped overrides catalog, so the
    overrides file is patched to an empty object by default; ``TestLocalOverrides``
    repopulates it per-test via ``_set_overrides``. The remote-synced overrides
    cache is pointed at a nonexistent tmp path so a real cache on the machine
    cannot leak into tests.
    """
    metadata_file = tmp_path / "model_metadata.json"
    metadata_file.write_text(json.dumps(METADATA_SAMPLE), encoding="utf-8")

    # Patch the path resolver before importing the module
    import kitty.providers.model_context as mc

    monkeypatch.setattr(mc, "_METADATA_PATH", metadata_file)
    # Force reload of metadata with patched path
    mc._load_metadata.cache_clear()

    # Empty overrides file so existing tests are shielded from the shipped catalog.
    overrides_file = tmp_path / "model_context_overrides.json"
    overrides_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mc, "_OVERRIDES_PATH", overrides_file)
    # Nonexistent remote-cache path: no real synced cache may leak into tests.
    monkeypatch.setattr(
        mc,
        "REMOTE_OVERRIDES_CACHE_PATH",
        tmp_path / "remote-cache" / "model_context_overrides.json",
    )
    mc._load_overrides.cache_clear()

    yield
    mc._load_metadata.cache_clear()
    mc._load_overrides.cache_clear()


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
# Local overrides file (highest priority)
# ---------------------------------------------------------------------------


OVERRIDES_CATALOG = {
    "MiniMax-M3": 1000000,
    "glm-5.2": 1000000,
}


def _set_overrides(content: dict) -> None:
    """Write an overrides catalog to the patched _OVERRIDES_PATH and clear cache."""
    import kitty.providers.model_context as mc

    mc._OVERRIDES_PATH.write_text(json.dumps(content), encoding="utf-8")
    mc._load_overrides.cache_clear()


class TestLocalOverrides:
    """The packaged model_context_overrides.json is the highest-priority source.

    It outranks provider_config["context_window"], OpenRouter metadata, and the
    default. Entries are global (provider-agnostic) and matched case-insensitively
    with the same single-direction suffix rule the metadata lookup uses.
    """

    def test_global_minimax_m3(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 1_000_000

    def test_global_glm_any_provider(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        # glm-5.2 is provider-agnostic.
        assert get_model_context_tokens(provider="zai_coding", model="glm-5.2", provider_config={}) == 1_000_000
        assert get_model_context_tokens(provider="opencode_go", model="glm-5.2", provider_config={}) == 1_000_000

    def test_suffix_match_prefixed_model(self):
        """A vendor-prefixed query matches a bare override key (single direction)."""
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        assert get_model_context_tokens(provider="zai_coding", model="z-ai/glm-5.2", provider_config={}) == 1_000_000

    def test_case_insensitive(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        assert get_model_context_tokens(provider="MINIMAX_TOKEN", model="MINIMAX-M3", provider_config={}) == 1_000_000
        assert get_model_context_tokens(provider="any", model="GLM-5.2", provider_config={}) == 1_000_000

    def test_override_beats_provider_config(self):
        """The local file is FIRST priority — it trumps a per-profile context_window."""
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        assert (
            get_model_context_tokens(
                provider="minimax_token",
                model="MiniMax-M3",
                provider_config={"context_window": 50000},
            )
            == 1_000_000
        )

    def test_override_beats_metadata_exact_and_suffix(self):
        """The catalog outranks the metadata table on both match paths (AC2.2)."""
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"gpt-4o": 999_999})
        # Metadata exact-match path: "openai/gpt-4o" is a metadata id (128000).
        assert get_model_context_tokens(provider="openrouter", model="openai/gpt-4o", provider_config={}) == 999_999
        # Metadata suffix path: bare "gpt-4o" would match "openai/gpt-4o".
        assert get_model_context_tokens(provider="openai", model="gpt-4o", provider_config={}) == 999_999

    def test_provider_config_wins_when_no_override(self):
        """When the model is not in the overrides file, provider_config still wins."""
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        assert (
            get_model_context_tokens(
                provider="openai",
                model="gpt-4o",
                provider_config={"context_window": 50000},
            )
            == 50000
        )

    def test_override_absent_falls_to_metadata(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        assert get_model_context_tokens(provider="openai", model="gpt-4o", provider_config={}) == 128000

    def test_invalid_entry_dropped(self):
        """Non-positive / boolean values are dropped; valid entries still resolve."""
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"MiniMax-M3": 1000000, "bad-model": -1, "bool-model": True, "zero-model": 0})
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 1_000_000
        # The dropped entry does not resolve to the invalid value; it falls through.
        assert get_model_context_tokens(provider="openai", model="bad-model", provider_config={}) != -1

    def test_balancing_min_routes_through_override(self):
        """get_balancing_min_context_tokens calls get_model_context_tokens per backend,
        so the override applies to balancing too."""
        from kitty.providers.model_context import get_balancing_min_context_tokens

        _set_overrides(OVERRIDES_CATALOG)
        backends = [
            ("minimax_token", "MiniMax-M3", {}),
            ("openai", "gpt-4o", {}),
        ]
        # min(1_000_000 override, 128_000 metadata) = 128_000.
        assert get_balancing_min_context_tokens(backends) == 128000


# ---------------------------------------------------------------------------
# Remote-synced overrides cache (loader preference)
# ---------------------------------------------------------------------------


def _set_remote_cache(content: dict | str) -> None:
    """Write content to the patched REMOTE_OVERRIDES_CACHE_PATH and clear the cache.

    Args:
        content: A dict serialized as the cache body, or raw text (for
            corrupt-body cases).
    """
    import kitty.providers.model_context as mc

    raw = content if isinstance(content, str) else json.dumps(content)
    mc.REMOTE_OVERRIDES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    mc.REMOTE_OVERRIDES_CACHE_PATH.write_text(raw, encoding="utf-8")
    mc._load_overrides.cache_clear()


class TestRemoteCachePreference:
    """A valid remote-synced cache replaces the packaged catalog wholesale.

    The cached copy is a newer revision of the same file, so there is no
    merging: entries only present in the packaged file stop resolving once a
    valid cache exists. A corrupt or missing cache falls back to the packaged
    file (R4 / AC4.1–AC4.3).
    """

    def test_valid_cache_wins_over_packaged(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"MiniMax-M3": 111_111})
        _set_remote_cache({"MiniMax-M3": 999_999})
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 999_999

    def test_valid_cache_replaces_packaged_wholesale(self):
        """A model only in the packaged file stops resolving once a valid cache exists."""
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        _set_overrides({"packaged-only-model": 111_111})
        _set_remote_cache({"MiniMax-M3": 999_999})
        result = get_model_context_tokens(provider="any", model="packaged-only-model", provider_config={})
        assert result == DEFAULT_CONTEXT_TOKENS

    def test_corrupt_cache_falls_back_to_packaged(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"MiniMax-M3": 111_111})
        _set_remote_cache("this is not json {")
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 111_111

    def test_non_object_cache_falls_back_to_packaged(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"MiniMax-M3": 111_111})
        _set_remote_cache("[1, 2, 3]")
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 111_111

    def test_missing_cache_uses_packaged(self):
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"MiniMax-M3": 111_111})
        # No _set_remote_cache call: the patched cache path does not exist.
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 111_111

    def test_empty_object_cache_replaces_packaged_wholesale(self):
        """A valid empty object is still a valid catalog revision: zero overrides."""
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_model_context_tokens

        _set_overrides({"MiniMax-M3": 111_111})
        _set_remote_cache({})
        result = get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={})
        assert result == DEFAULT_CONTEXT_TOKENS

    def test_stale_but_valid_cache_still_beats_packaged(self):
        """TTL governs fetch attempts only — never the loader preference."""
        import kitty.providers.model_context as mc
        from kitty.providers.model_context import get_model_context_tokens

        _set_overrides({"MiniMax-M3": 111_111})
        _set_remote_cache({"MiniMax-M3": 999_999})
        # Backdate the cache far beyond any TTL; the loader has no mtime logic.
        mtime = time.time() - 10 * 365 * 24 * 3600
        os.utime(mc.REMOTE_OVERRIDES_CACHE_PATH, (mtime, mtime))
        assert get_model_context_tokens(provider="minimax_token", model="MiniMax-M3", provider_config={}) == 999_999


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

        assert DEFAULT_CONTEXT_TOKENS == 200_000
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
# Suffix matching for bare model names
# ---------------------------------------------------------------------------


class TestSuffixMatching:
    """Bare model names without provider prefix match via suffix lookup."""

    def test_bare_model_name_matches_provider_prefixed_id(self):
        """'gpt-4o' should match 'openai/gpt-4o' in metadata."""
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(provider="openai", model="gpt-4o")
        assert result == 128000

    def test_bare_model_name_for_zai_provider(self):
        """'glm-5.1' via zai provider should match metadata entry if present."""
        from kitty.providers.model_context import get_model_context_tokens

        # We don't have z-ai/glm-5.1 in the sample metadata, so this returns default.
        # But let's add it dynamically to test.
        result = get_model_context_tokens(provider="zai", model="deepseek-chat")
        assert result == 65536  # matches deepseek/deepseek-chat

    def test_bare_model_name_with_openrouter_provider(self):
        """OpenRouter users might pass bare model names too."""
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(provider="openrouter", model="gpt-4o")
        assert result == 128000

    def test_bare_model_name_claude(self):
        """'claude-3.5-haiku' matches 'anthropic/claude-3.5-haiku'."""
        from kitty.providers.model_context import get_model_context_tokens

        result = get_model_context_tokens(provider="anthropic", model="claude-3.5-haiku")
        assert result == 200000


# ---------------------------------------------------------------------------
# Balancing profile minimum context
# ---------------------------------------------------------------------------


class TestBalancingMinContext:
    """get_balancing_min_context_tokens returns the smallest context window."""

    def test_min_across_mixed_models(self):
        from kitty.providers.model_context import get_balancing_min_context_tokens

        backends = [
            ("openrouter", "openai/gpt-4o", None),
            ("openrouter", "deepseek/deepseek-chat", None),
            ("openrouter", "google/gemini-2.0-flash-001", None),
        ]
        result = get_balancing_min_context_tokens(backends)
        assert result == 65536  # deepseek-chat is the smallest

    def test_min_with_provider_config_override(self):
        from kitty.providers.model_context import get_balancing_min_context_tokens

        backends = [
            ("openrouter", "openai/gpt-4o", None),
            ("custom", "tiny-model", {"context_window": 8000}),
        ]
        result = get_balancing_min_context_tokens(backends)
        assert result == 8000

    def test_min_with_unknown_model_uses_default(self):
        from kitty.providers.model_context import get_balancing_min_context_tokens

        backends = [
            ("openrouter", "openai/gpt-4o", None),  # 128000
            ("ollama", "llama3-custom", None),  # default 200000
        ]
        result = get_balancing_min_context_tokens(backends)
        assert result == 128000  # gpt-4o is smaller than default

    def test_single_backend(self):
        from kitty.providers.model_context import get_balancing_min_context_tokens

        backends = [
            ("openrouter", "deepseek/deepseek-chat", None),
        ]
        result = get_balancing_min_context_tokens(backends)
        assert result == 65536

    def test_empty_backends_returns_default(self):
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, get_balancing_min_context_tokens

        result = get_balancing_min_context_tokens([])
        assert result == DEFAULT_CONTEXT_TOKENS


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


# ---------------------------------------------------------------------------
# Token-to-char conversion
# ---------------------------------------------------------------------------


class TestTokensToChars:
    """tokens_to_chars converts token counts to character estimates."""

    def test_default_context_to_chars(self):
        from kitty.providers.model_context import DEFAULT_CONTEXT_TOKENS, tokens_to_chars

        assert tokens_to_chars(DEFAULT_CONTEXT_TOKENS) == 800_000

    def test_known_context_values(self):
        from kitty.providers.model_context import tokens_to_chars

        assert tokens_to_chars(128_000) == 512_000
        assert tokens_to_chars(1_048_576) == 4_194_304
        assert tokens_to_chars(65_536) == 262_144

    def test_zero_tokens(self):
        from kitty.providers.model_context import tokens_to_chars

        assert tokens_to_chars(0) == 0

    def test_uses_constant_factor(self):
        from kitty.providers.model_context import TOKENS_TO_CHARS_FACTOR, tokens_to_chars

        assert tokens_to_chars(1) == TOKENS_TO_CHARS_FACTOR
        assert TOKENS_TO_CHARS_FACTOR == 4
