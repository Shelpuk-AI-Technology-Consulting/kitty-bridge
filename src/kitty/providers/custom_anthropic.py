"""Custom Anthropic-compatible provider adapter — user-configurable Messages API endpoint.

Routes requests to any service that exposes an Anthropic Messages API-compatible
endpoint (``POST /v1/messages`` with ``x-api-key`` auth and SSE streaming).

For Claude Code (Messages API path), the bridge detects ``use_native_messages``
and skips the MessagesTranslator, forwarding the raw Messages API body unchanged.
This enables transparent passthrough of all Messages API fields including
``effort``, ``thinking``, and system prompts.

For other clients (Responses API, Chat Completions), the inherited
``AnthropicAdapter`` CC↔Messages translation is used automatically.
"""

from __future__ import annotations

import json

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderError

__all__ = ["CustomAnthropicAdapter"]


class CustomAnthropicAdapter(AnthropicAdapter):
    """Generic Anthropic Messages API adapter.

    Allows connecting to any service that exposes an Anthropic Messages
    API-compatible endpoint (``POST /v1/messages`` with ``x-api-key`` auth
    and SSE streaming).  The base URL is configured via
    ``provider_config["base_url"]`` at profile creation time.

    Since the upstream format is the native Messages API (which Claude Code
    speaks), this adapter does transparent passthrough for Claude Code requests.
    For non-Claude agents, the inherited CC↔Messages translation is used.
    """

    @property
    def provider_type(self) -> str:
        return "custom_anthropic"

    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com"

    @property
    def upstream_path(self) -> str:
        return "/v1/messages"

    @property
    def requires_custom_url(self) -> bool:
        return True

    @property
    def use_native_messages(self) -> bool:
        return True

    def build_base_url(self, provider_config: dict | None) -> str:
        if provider_config and "base_url" in provider_config:
            return provider_config["base_url"]
        return self.default_base_url

    def translate_to_upstream(self, cc_request: dict) -> dict:
        if cc_request.get("_native_messages_request"):
            return {k: v for k, v in cc_request.items() if k not in self._INTERNAL_KEYS}
        return super().translate_to_upstream(cc_request)

    def translate_from_upstream(self, raw_response: dict) -> dict:
        if raw_response.get("object") == "chat.completion":
            return raw_response
        return super().translate_from_upstream(raw_response)

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []
        for line in raw_str.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    return [raw_bytes]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    return [raw_bytes]
                event_type = data.get("type", "")
                if event_type in (
                    "message_start",
                    "message_delta",
                    "message_stop",
                    "content_block_start",
                    "content_block_stop",
                    "content_block_delta",
                    "ping",
                ):
                    return [raw_bytes]
                return super().translate_upstream_stream_event(raw_bytes)
        return [raw_bytes]

    def map_error(self, status_code: int, body: dict) -> Exception:
        if not isinstance(body, dict):
            return ProviderError(f"Custom Anthropic error {status_code}: {body}")
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        return ProviderError(f"Custom Anthropic error {status_code}: {msg}")
