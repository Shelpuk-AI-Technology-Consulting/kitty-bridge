"""BytePlus ModelArk provider adapter."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["BytePlusAdapter"]

_BYTEPLUS_PREFIX = re.compile(r"^byteplus/", re.IGNORECASE)


class BytePlusAdapter(ProviderAdapter):
    """BytePlus ModelArk Coding Plan adapter.

    BytePlus ModelArk provides an OpenAI-compatible Chat Completions API.
    The Coding Plan endpoint (``/api/coding/v3``) should be used to consume
    the Coding Plan quota; the base endpoint (``/api/v3``) incurs separate
    charges.
    """

    @property
    def provider_type(self) -> str:
        return "byteplus"

    @property
    def default_base_url(self) -> str:
        return "https://ark.ap-southeast.bytepluses.com/api/coding/v3"

    def normalize_model_name(self, model: str) -> str:
        stripped = _BYTEPLUS_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

    @property
    def validation_model(self) -> str:
        return "ark-code-latest"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """BytePlus expects a recognized coding agent User-Agent header."""
        headers = super().build_upstream_headers(api_key)
        headers["User-Agent"] = "claude-code/1.0"
        return headers

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
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage") or {},
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_dict = body.get("error", {})
        if isinstance(error_dict, dict):
            error_msg = error_dict.get("message", "Unknown error")
        else:
            error_msg = error_dict or "Unknown error"
        return ProviderError(f"BytePlus error {status_code}: {error_msg}")
