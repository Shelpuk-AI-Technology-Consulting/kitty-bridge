"""MiniMax provider adapter — global and CN region endpoints."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["MiniMaxAdapter"]

_GLOBAL_URL = "https://api.minimax.io/v1"
_CN_URL = "https://api.minimaxi.com/v1"

_MINIMAX_PREFIX = re.compile(r"^minimax/", re.IGNORECASE)


class MiniMaxAdapter(ProviderAdapter):
    """MiniMax API adapter with global/CN region support."""

    @property
    def provider_type(self) -> str:
        return "minimax"

    @property
    def default_base_url(self) -> str:
        return _GLOBAL_URL

    def normalize_model_name(self, model: str) -> str:
        stripped = _MINIMAX_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

    def normalize_request(self, cc_request: dict) -> None:
        """Add MiniMax-specific parameters for better compatibility.

        Enables ``reasoning_split`` so the model returns thinking content
        in a separate ``reasoning_details`` field instead of embedding
        ``<اخل>`` tags in ``content``.
        """
        cc_request["reasoning_split"] = True

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        # Determine base URL from provider_config region
        provider_config = kwargs.get("provider_config", {})
        if isinstance(provider_config, dict) and provider_config.get("region") == "cn":
            request["base_url"] = _CN_URL
        elif "base_url" in kwargs and kwargs["base_url"]:
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
        return ProviderError(f"MiniMax error {status_code}: {error_msg}")
