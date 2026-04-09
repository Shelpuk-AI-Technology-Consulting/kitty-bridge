"""Novita provider adapter."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["NovitaAdapter"]

_NOVITA_PREFIX = re.compile(r"^novita/", re.IGNORECASE)


class NovitaAdapter(ProviderAdapter):
    """Novita AI API adapter."""

    @property
    def provider_type(self) -> str:
        return "novita"

    @property
    def default_base_url(self) -> str:
        return "https://api.novita.ai/openai/v1"

    def normalize_model_name(self, model: str) -> str:
        stripped = _NOVITA_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        if "base_url" in kwargs and kwargs["base_url"]:
            request["base_url"] = kwargs["base_url"]
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
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
        return ProviderError(f"Novita error {status_code}: {error_msg}")
