"""Xiaomi MiMo provider adapter."""

from __future__ import annotations

import re

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["MimoAdapter"]

_MIMO_PREFIX = re.compile(r"^mimo/", re.IGNORECASE)


class MimoAdapter(ProviderAdapter):
    """Xiaomi MiMo adapter (OpenAI-compatible).

    MiMo provides an OpenAI-compatible Chat Completions API at
    ``https://api.xiaomimimo.com/v1``.
    API keys are obtained from https://platform.xiaomimimo.com/#/console/api-keys
    """

    @property
    def provider_type(self) -> str:
        return "mimo"

    @property
    def default_base_url(self) -> str:
        return "https://api.xiaomimimo.com/v1"

    @property
    def validation_model(self) -> str:
        """mimo-v2-pro is the flagship model available on MiMo API."""
        return "mimo-v2-pro"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Build auth headers for Xiaomi MiMo.

        MiMo uses the ``api-key`` header (not ``Authorization: Bearer``) and
        requires a recognized coding-agent User-Agent to avoid 403 errors.
        """
        headers = super().build_upstream_headers(api_key)
        # MiMo does not use Bearer auth — replace with api-key.
        headers.pop("Authorization", None)
        headers["api-key"] = api_key
        # MiMo blocks requests without a recognized coding agent User-Agent.
        # If MiMo updates their allowlist, update this value accordingly.
        headers["User-Agent"] = "claude-code/1.0"
        return headers

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. 'mimo/mimo-v2-pro')."""
        stripped = _MIMO_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

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
        choices = response_data.get("choices")
        if not choices:
            return {"content": None, "finish_reason": None, "usage": {}}
        choice = choices[0]
        message = choice.get("message", {})
        result: dict = {
            "content": message.get("content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": response_data.get("usage") or {},
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> ProviderError:
        error_dict = body.get("error", {})
        if isinstance(error_dict, dict):
            error_msg = error_dict.get("message", "Unknown error")
        else:
            error_msg = error_dict or "Unknown error"
        return ProviderError(f"MiMo error {status_code}: {error_msg}")
