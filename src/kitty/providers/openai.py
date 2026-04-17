"""OpenAI provider adapter — canonical Chat Completions API."""

from __future__ import annotations

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter(ProviderAdapter):
    """OpenAI API adapter.

    OpenAI provides the canonical Chat Completions API.  Request
    normalization is minimal; only model name prefix stripping is needed.
    """

    @property
    def provider_type(self) -> str:
        return "openai"

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. 'openai/gpt-4o')."""
        if "/" in model:
            return model.split("/", 1)[1] or model
        return model

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    def parse_response(self, response_data: dict) -> dict:
        choices = response_data.get("choices") or [{}]
        choice = choices[0] if choices else {}
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
