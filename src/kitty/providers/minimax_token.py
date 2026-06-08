"""MiniMax Token Plan Anthropic adapter — native Messages API passthrough.

Routes Claude Code Messages API requests through MiniMax's Anthropic-compatible
endpoint (``https://api.minimax.io/anthropic``), passing the Messages API format
directly without the intermediate Chat Completions translation layer.

For Claude Code (Messages API path), the bridge detects ``use_native_messages``
and skips the MessagesTranslator, forwarding the raw Messages API body.

For other clients (Responses API, Chat Completions), the inherited
``AnthropicAdapter`` CC↔Messages translation is used automatically.

Region support:
  - Global (default): ``https://api.minimax.io/anthropic``
  - China: ``https://api.minimaxi.com/anthropic``

  The region is configured via ``provider_config["region"] == "cn"`` in the
  profile.  When no region is set, the global endpoint is used.
"""

from __future__ import annotations

import json

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderError

__all__ = ["MiniMaxTokenAnthropicAdapter"]

_GLOBAL_URL = "https://api.minimax.io/anthropic"
_CN_URL = "https://api.minimaxi.com/anthropic"


class MiniMaxTokenAnthropicAdapter(AnthropicAdapter):
    """MiniMax Token Plan adapter using the native Anthropic Messages API endpoint.

    Extends ``AnthropicAdapter`` to reuse CC↔Messages translation for
    non-Claude Code clients (Responses API, Chat Completions), while
    enabling direct Messages passthrough when the server signals the
    request is already in Messages format via ``_native_messages_request``.
    """

    @property
    def provider_type(self) -> str:
        return "minimax_token"

    @property
    def default_base_url(self) -> str:
        return _GLOBAL_URL

    @property
    def use_native_messages(self) -> bool:
        return True

    def build_base_url(self, provider_config: dict | None) -> str:
        if provider_config and provider_config.get("region") == "cn":
            return _CN_URL
        return self.default_base_url

    def normalize_model_name(self, model: str) -> str:
        """Pass model name through unchanged — user provides the exact upstream name."""
        return model

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
            return ProviderError(f"MiniMax Token Plan error {status_code}: {body}")
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        return ProviderError(f"MiniMax Token Plan error {status_code}: {msg}")
