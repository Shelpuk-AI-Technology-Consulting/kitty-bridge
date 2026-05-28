"""Tests for providers/registry.py — get_provider lookup."""

from typing import get_args

import pytest

from kitty.profiles.schema import _PROVIDER_TYPES
from kitty.providers.registry import _registry, get_provider


class TestProviderRegistrySync:
    """Cross-module consistency: _PROVIDER_TYPES Literal must match _registry keys."""

    # Registry keys that are internal-only (not exposed in the user-facing TUI).
    _INTERNAL_ONLY_KEYS: frozenset[str] = frozenset({"zai_coding_cc"})

    def test_provider_literal_matches_registry(self):
        """_PROVIDER_TYPES in schema.py must match _registry keys in registry.py."""
        literal_values = set(get_args(_PROVIDER_TYPES))
        registry_keys = set(_registry.keys())
        expected_registry = literal_values | self._INTERNAL_ONLY_KEYS
        assert expected_registry == registry_keys, (
            f"Mismatch: Literal has {literal_values - registry_keys} extra, "
            f"registry has {registry_keys - expected_registry} extra"
        )


class TestGetProvider:
    def test_returns_zai_regular_adapter(self):
        adapter = get_provider("zai_regular")
        assert adapter.provider_type == "zai_regular"

    def test_returns_zai_coding_adapter(self):
        adapter = get_provider("zai_coding")
        assert adapter.provider_type == "zai_coding"

    def test_returns_minimax_adapter(self):
        adapter = get_provider("minimax")
        assert adapter.provider_type == "minimax"

    def test_returns_novita_adapter(self):
        adapter = get_provider("novita")
        assert adapter.provider_type == "novita"

    def test_raises_keyerror_for_unknown_provider(self):
        with pytest.raises(KeyError):
            get_provider("unknown")

    def test_returns_openrouter_adapter(self):
        adapter = get_provider("openrouter")
        assert adapter.provider_type == "openrouter"

    def test_returns_openai_adapter(self):
        adapter = get_provider("openai")
        assert adapter.provider_type == "openai"

    def test_returns_ollama_adapter(self):
        adapter = get_provider("ollama")
        assert adapter.provider_type == "ollama"

    def test_returns_anthropic_adapter(self):
        adapter = get_provider("anthropic")
        assert adapter.provider_type == "anthropic"

    def test_returns_bedrock_adapter(self):
        adapter = get_provider("bedrock")
        assert adapter.provider_type == "bedrock"

    def test_returns_azure_adapter(self):
        adapter = get_provider("azure")
        assert adapter.provider_type == "azure"

    def test_returns_vertex_adapter(self):
        adapter = get_provider("vertex")
        assert adapter.provider_type == "vertex"

    def test_returns_fireworks_adapter(self):
        adapter = get_provider("fireworks")
        assert adapter.provider_type == "fireworks"

    def test_returns_opencode_go_adapter(self):
        adapter = get_provider("opencode_go")
        assert adapter.provider_type == "opencode_go"

    def test_returns_kimi_adapter(self):
        adapter = get_provider("kimi")
        assert adapter.provider_type == "kimi"
