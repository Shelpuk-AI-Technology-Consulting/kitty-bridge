"""MiniMax Token Plan Anthropic adapter — translated CC→Messages by default.

Routes Claude Code Messages API requests through MiniMax's Anthropic-compatible
endpoint (``https://api.minimax.io/anthropic``).

By default the bridge runs the standard ``MessagesTranslator`` →
``AnthropicAdapter.translate_to_upstream`` path, which produces a clean
Anthropic Messages body that the upstream accepts. The raw-body native
passthrough (``use_native_messages=True``) is still available as an opt-in
for users who have a specific need to forward the untranslated body
(e.g. to exercise fields the translator strips).

For other clients (Responses API, Chat Completions), the inherited
``AnthropicAdapter`` CC↔Messages translation is used automatically.

Region support:
  - Global (default): ``https://api.minimax.io/anthropic``
  - China: ``https://api.minimaxi.com/anthropic``

  The region is configured via ``provider_config["region"] == "cn"`` in the
  profile.  When no region is set, the global endpoint is used.

Native passthrough opt-in:
  - ``MiniMaxTokenAnthropicAdapter(native_messages=True)``
  - ``MiniMaxTokenAnthropicAdapter(provider_config={"native_messages": True})``

  Why opt-in instead of default: MiniMax's Anthropic-compatible endpoint
  rejects the raw Claude Code body with HTTP 400 because it sends fields
  the translator strips (``context_management``, ``output_config``,
  ``thinking: {type: "adaptive"}``, ``cache_control`` on system blocks,
  advanced ``metadata``). The translated path produces a clean body that
  the upstream accepts. Native passthrough also bypasses the
  request-compaction grouping logic which is Chat-Completions-format
  aware; large requests can lose the ``tool_use`` → ``tool_result`` pairing
  and surface as ``invalid params, tool call result does not follow tool
  call (2013)`` from MiniMax.
"""

from __future__ import annotations

import json

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderError

__all__ = ["MiniMaxTokenAnthropicAdapter"]

_GLOBAL_URL = "https://api.minimax.io/anthropic"
_CN_URL = "https://api.minimaxi.com/anthropic"


class MiniMaxTokenAnthropicAdapter(AnthropicAdapter):
    """MiniMax Token Plan adapter.

    Default: translated path (CC → Anthropic Messages via the inherited
    ``AnthropicAdapter``).

    Opt-in: native passthrough (``use_native_messages=True``) forwards the
    raw Claude Code body to the upstream. See module docstring for caveats.
    """

    def __init__(
        self,
        *,
        native_messages: bool = False,
        provider_config: dict | None = None,
    ) -> None:
        self._native_messages_override = self._resolve_native_messages(native_messages, provider_config)

    @staticmethod
    def _resolve_native_messages(native_messages: bool, provider_config: dict | None) -> bool:
        """Resolve the ``use_native_messages`` flag from kwargs and provider_config.

        ``provider_config["native_messages"]`` takes precedence over the
        explicit kwarg so the profile-driven setting wins. Any non-falsy
        value enables the override; explicit ``False`` in provider_config
        forces the translated path.
        """
        if isinstance(provider_config, dict) and "native_messages" in provider_config:
            return bool(provider_config["native_messages"])
        return bool(native_messages)

    @property
    def provider_type(self) -> str:
        return "minimax_token"

    @property
    def default_base_url(self) -> str:
        return _GLOBAL_URL

    @property
    def use_native_messages(self) -> bool:
        return self._native_messages_override

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
        # Native-passthrough opt-in: forward raw Anthropic SSE bytes
        # untouched. We still attempt to detect the event types so non-
        # Anthropic SSE (e.g. accidental CC stream) falls through to the
        # inherited CC→Messages translation.
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
