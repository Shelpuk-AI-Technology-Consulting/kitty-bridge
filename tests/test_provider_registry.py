"""Tests for providers/registry.py — get_provider lookup."""

import pytest

from kitty.providers.registry import get_provider


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

    def test_returns_ollama_adapter(self):
        adapter = get_provider("ollama")
        assert adapter.provider_type == "ollama"
