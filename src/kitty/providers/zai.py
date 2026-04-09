"""Z.AI provider adapters — regular and coding endpoints."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["ZaiCodingAdapter", "ZaiRegularAdapter"]

_ZAI_PREFIX = re.compile(r"^z-?ai/", re.IGNORECASE)


class _ZaiBase(ProviderAdapter):
    """Shared base for Z.AI regular and coding adapters."""

    _BASE_URL: str
    _PROVIDER_TYPE: str

    @property
    def provider_type(self) -> str:
        return self._PROVIDER_TYPE

    @property
    def default_base_url(self) -> str:
        return self._BASE_URL

    def normalize_model_name(self, model: str) -> str:
        stripped = _ZAI_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

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
        # Pass through extra kwargs (temperature, top_p, etc.)
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
        return ProviderError(f"Z.AI {self._PROVIDER_TYPE} error {status_code}: {error_msg}")


class ZaiRegularAdapter(_ZaiBase):
    """Z.AI regular API adapter."""

    _PROVIDER_TYPE = "zai_regular"
    _BASE_URL = "https://api.z.ai/api/paas/v4"


class ZaiCodingAdapter(_ZaiBase):
    """Z.AI coding API adapter."""

    _PROVIDER_TYPE = "zai_coding"
    _BASE_URL = "https://api.z.ai/api/coding/paas/v4"
