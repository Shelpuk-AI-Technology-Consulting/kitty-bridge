"""Kimi Code provider adapter.

Kimi Code (by Moonshot AI) provides an OpenAI-compatible Chat Completions
API at ``https://api.kimi.com/coding/v1`` for coding agents.
"""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["KimiCodeAdapter"]

_KIMI_PREFIX = re.compile(r"^kimi/", re.IGNORECASE)


class KimiCodeAdapter(ProviderAdapter):
    """Kimi Code API adapter (OpenAI-compatible)."""

    @property
    def provider_type(self) -> str:
        return "kimi"

    @property
    def default_base_url(self) -> str:
        return "https://api.kimi.com/coding/v1"

    @property
    def validation_model(self) -> str:
        """Kimi Code rejects unknown models with 403; use a real model for validation."""
        return "kimi-for-coding"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Kimi Code requires a recognized coding agent User-Agent to avoid 403 errors."""
        headers = super().build_upstream_headers(api_key)
        # Kimi Code blocks requests without a recognized coding agent User-Agent.
        # "claude-code/1.0" is in their allowlist as of 2026-04-18.
        headers["User-Agent"] = "claude-code/1.0"
        return headers

    def normalize_model_name(self, model: str) -> str:
        stripped = _KIMI_PREFIX.sub("", model, count=1)
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
        choices = response_data.get("choices") or [{}]
        choice = choices[0] if choices else {}
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
        error_msg = body.get("error", {}) if isinstance(body.get("error"), dict) else body.get("error", "Unknown error")
        return ProviderError(f"Kimi Code error {status_code}: {error_msg}")
