"""OpenCode Go provider adapter — auto-routing Chat Completions and Anthropic Messages.

OpenCode Go (https://opencode.ai) is a low-cost subscription ($10/month) that
provides reliable access to popular open coding models behind a single API key.
Models are served through two different endpoints depending on the model:

- Chat Completions (``/v1/chat/completions``): glm-5, glm-5.1, kimi-k2.5,
  mimo-v2-pro, mimo-v2-omni
- Anthropic Messages (``/v1/messages``): minimax-m2.5, minimax-m2.7

The adapter auto-detects the correct endpoint from the model name, so the user
only needs to create one profile and pick a model.
"""

from __future__ import annotations

import json
import logging

from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["OpenCodeGoAdapter"]

logger = logging.getLogger(__name__)

# Models served via the Anthropic Messages API endpoint.
_MESSAGES_MODELS: frozenset[str] = frozenset(
    {
        "minimax-m2.5",
        "minimax-m2.7",
    }
)


def _is_messages_model(model: str) -> bool:
    """Return True if *model* should use the Anthropic Messages endpoint."""
    return model in _MESSAGES_MODELS


class OpenCodeGoAdapter(AnthropicAdapter):
    """OpenCode Go adapter with automatic endpoint routing.

    Routes to ``/v1/chat/completions`` (passthrough) or ``/v1/messages``
    (Anthropic Messages API) depending on the model name.

    F16: Anthropic Messages translation is inherited from ``AnthropicAdapter``
    instead of duplicating the translation helpers here.
    """

    @property
    def provider_type(self) -> str:
        return "opencode_go"

    @property
    def default_base_url(self) -> str:
        return "https://opencode.ai/zen/go"

    @property
    def validation_model(self) -> str:
        """Use a known-valid model for key validation.

        OpenCode returns 401 for unsupported models, which would be
        misinterpreted as an auth failure. Use ``glm-5`` which is always
        available on the Chat Completions endpoint.
        """
        return "glm-5"

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. 'opencode/glm-5')."""
        if "/" in model:
            return model.rsplit("/", 1)[-1]
        return model

    # ── Per-model routing ─────────────────────────────────────────────────

    @property
    def upstream_path(self) -> str:  # noqa: D401 — overridden by get_upstream_path
        """Default path (Chat Completions).  ``get_upstream_path`` routes per model."""
        return "/v1/chat/completions"

    def get_upstream_path(self, model: str) -> str:
        return "/v1/messages" if _is_messages_model(model) else "/v1/chat/completions"

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Default headers (Chat Completions — Bearer auth)."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def build_upstream_headers_for_model(self, api_key: str, model: str) -> dict[str, str]:
        """Build auth headers appropriate for the model's endpoint."""
        if _is_messages_model(model):
            return AnthropicAdapter.build_upstream_headers(self, api_key)
        return self.build_upstream_headers(api_key)

    # ── Routed translation ─────────────────────────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        model = cc_request.get("model", "")
        if _is_messages_model(model):
            return AnthropicAdapter.translate_to_upstream(self, cc_request)
        return ProviderAdapter.translate_to_upstream(self, cc_request)

    def translate_from_upstream(self, raw_response: dict) -> dict:
        # Anthropic responses have a "type" field; CC responses have "object"
        if raw_response.get("type") == "message":
            return AnthropicAdapter.translate_from_upstream(self, raw_response)
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Auto-detect SSE format and translate Anthropic events to CC chunks.

        Anthropic events have a ``"type"`` field (message_start,
        content_block_delta, etc.) while Chat Completions events have
        ``"object": "chat.completion.chunk"``.
        """
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        for line in raw_str.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
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
                return AnthropicAdapter.translate_upstream_stream_event(self, raw_bytes)
            return [raw_bytes]

        return [raw_bytes]

    def translate_upstream_stream_event_for_model(self, raw_bytes: bytes, model: str) -> list[bytes]:
        """Translate SSE events, routing based on model."""
        if _is_messages_model(model):
            return AnthropicAdapter.translate_upstream_stream_event(self, raw_bytes)
        return [raw_bytes]

    # ── Standard ProviderAdapter methods ───────────────────────────────────

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
            "usage": response_data.get("usage", {}),
        }
        if "tool_calls" in message:
            result["tool_calls"] = message["tool_calls"]
        return result

    def map_error(self, status_code: int, body: dict) -> Exception:
        error_obj = body.get("error", body)
        msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        return ProviderError(f"OpenCode Go error {status_code}: {msg}")
