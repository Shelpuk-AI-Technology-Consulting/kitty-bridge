"""Provider registry — lookup provider adapter by type string."""

from __future__ import annotations

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.azure import AzureOpenAIAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.bedrock import BedrockAdapter
from kitty.providers.fireworks import FireworksAdapter
from kitty.providers.minimax import MiniMaxAdapter
from kitty.providers.novita import NovitaAdapter
from kitty.providers.ollama import OllamaAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openrouter import OpenRouterAdapter
from kitty.providers.vertex import VertexAIAdapter
from kitty.providers.zai import ZaiCodingAdapter, ZaiRegularAdapter

_registry: dict[str, type[ProviderAdapter]] = {
    "zai_regular": ZaiRegularAdapter,
    "zai_coding": ZaiCodingAdapter,
    "minimax": MiniMaxAdapter,
    "novita": NovitaAdapter,
    "ollama": OllamaAdapter,
    "openai": OpenAIAdapter,
    "openrouter": OpenRouterAdapter,
    "anthropic": AnthropicAdapter,
    "bedrock": BedrockAdapter,
    "azure": AzureOpenAIAdapter,
    "vertex": VertexAIAdapter,
    "fireworks": FireworksAdapter,
}


def get_provider(provider_type: str) -> ProviderAdapter:
    """Look up a provider adapter by type string.

    Args:
        provider_type: One of the supported provider type strings (see ``_registry`` keys).

    Returns:
        An instantiated ProviderAdapter.

    Raises:
        KeyError: If provider_type is not recognized.
    """
    cls = _registry.get(provider_type)
    if cls is None:
        raise KeyError(f"Unknown provider type: {provider_type!r}. Available: {sorted(_registry)}")
    return cls()


__all__ = ["get_provider"]
