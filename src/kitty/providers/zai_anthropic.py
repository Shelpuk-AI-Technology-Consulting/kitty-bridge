"""Z.AI Anthropic endpoint adapter — native Messages API passthrough.

Routes Claude Code Messages API requests through Z.AI's Anthropic-compatible
endpoint (``https://api.z.ai/api/anthropic/v1/messages``), passing the
Messages API format directly without the intermediate Chat Completions
translation layer.  This enables native ``effort`` and ``thinking`` passthrough.

For Claude Code (Messages API path), the bridge detects ``use_native_messages``
and skips the MessagesTranslator, forwarding the raw Messages API body.

For other clients (Responses API, Chat Completions), the inherited
``AnthropicAdapter`` CC↔Messages translation is used automatically.
"""

from __future__ import annotations

import json
import re

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderError

__all__ = ["ZaiAnthropicAdapter"]

_ANTHROPIC_VERSION = "2023-06-01"

_ZAI_PREFIX = re.compile(r"^z-?ai/", re.IGNORECASE)


class ZaiAnthropicAdapter(AnthropicAdapter):
    """Z.AI Coding adapter using the native Anthropic Messages API endpoint.

    Extends ``AnthropicAdapter`` to reuse CC↔Messages translation for
    non-Claude Code clients (Responses API, Chat Completions), while
    enabling direct Messages passthrough when the server signals the
    request is already in Messages format via ``_native_messages_request``.
    """

    @property
    def provider_type(self) -> str:
        return "zai_coding"

    @property
    def default_base_url(self) -> str:
        return "https://api.z.ai/api/anthropic"

    @property
    def use_native_messages(self) -> bool:
        return True

    def normalize_model_name(self, model: str) -> str:
        stripped = _ZAI_PREFIX.sub("", model, count=1)
        return stripped if stripped else model

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

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
                    "error",
                ):
                    return [raw_bytes]
                return super().translate_upstream_stream_event(raw_bytes)
        return [raw_bytes]

    def map_error(self, status_code: int, body: dict) -> Exception:
        if not isinstance(body, dict):
            return ProviderError(f"Z.AI Anthropic error {status_code}: {body}")
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        return ProviderError(f"Z.AI Anthropic error {status_code}: {msg}")
