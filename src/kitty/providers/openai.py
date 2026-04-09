"""OpenAI provider adapter — canonical Chat Completions API."""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter(ProviderAdapter):
    """OpenAI API adapter.

    OpenAI provides the canonical Chat Completions API.  No request
    normalization or model name stripping is needed.
    """

    @property
    def provider_type(self) -> str:
        return "openai"

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        if "base_url" in kwargs and kwargs["base_url"]:
            request["base_url"] = kwargs["base_url"]
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    def parse_response(self, response_data: dict) -> dict:
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_msg = body.get("error", {}) if isinstance(body.get("error"), dict) else body.get("error", "Unknown error")
        return ProviderError(f"OpenAI error {status_code}: {error_msg}")
