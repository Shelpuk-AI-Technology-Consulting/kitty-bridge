"""Provider adapters — stateless request/response builders for upstream Chat Completions APIs."""

__all__ = [
    "AnthropicAdapter",
    "AzureOpenAIAdapter",
    "BedrockAdapter",
    "FireworksAdapter",
    "MiniMaxAdapter",
    "NovitaAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "OpenRouterAdapter",
    "ProviderAdapter",
    "ProviderError",
    "VertexAIAdapter",
    "ZaiCodingAdapter",
    "ZaiRegularAdapter",
    "get_provider",
]

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.azure import AzureOpenAIAdapter
from kitty.providers.base import ProviderAdapter, ProviderError
from kitty.providers.bedrock import BedrockAdapter
from kitty.providers.fireworks import FireworksAdapter
from kitty.providers.minimax import MiniMaxAdapter
from kitty.providers.novita import NovitaAdapter
from kitty.providers.ollama import OllamaAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openrouter import OpenRouterAdapter
from kitty.providers.registry import get_provider
from kitty.providers.vertex import VertexAIAdapter
from kitty.providers.zai import ZaiCodingAdapter, ZaiRegularAdapter
