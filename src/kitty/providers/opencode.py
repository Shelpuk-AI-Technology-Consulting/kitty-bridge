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
import uuid

from kitty.providers.anthropic import _safe_json_load_args
from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["OpenCodeGoAdapter"]

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# Models served via the Anthropic Messages API endpoint.
_MESSAGES_MODELS: frozenset[str] = frozenset(
    {
        "minimax-m2.5",
        "minimax-m2.7",
    }
)

# stop_reason → finish_reason
_STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    None: "stop",
}


def _is_messages_model(model: str) -> bool:
    """Return True if *model* should use the Anthropic Messages endpoint."""
    return model in _MESSAGES_MODELS


class OpenCodeGoAdapter(ProviderAdapter):
    """OpenCode Go adapter with automatic endpoint routing.

    Routes to ``/v1/chat/completions`` (passthrough) or ``/v1/messages``
    (Anthropic Messages API) depending on the model name.
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
            return {
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            }
        return self.build_upstream_headers(api_key)

    # ── Passthrough (Chat Completions models) ──────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        model = cc_request.get("model", "")
        if _is_messages_model(model):
            return self._translate_to_anthropic(cc_request)
        return {k: v for k, v in cc_request.items() if k not in self._INTERNAL_KEYS}

    def translate_from_upstream(self, raw_response: dict) -> dict:
        # Anthropic responses have a "type" field; CC responses have "object"
        if raw_response.get("type") == "message":
            return self._translate_from_anthropic(raw_response)
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Auto-detect SSE format and translate accordingly.

        Anthropic events have a ``"type"`` field (message_start, content_block_delta, etc.)
        while Chat Completions events have ``"object": "chat.completion.chunk"``.
        We detect which format the event is in and translate Anthropic events to CC format.
        """
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        # Check if this looks like an Anthropic SSE event
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
                # Anthropic events have "type" field with specific values
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
                    return self._translate_anthropic_stream_event(raw_bytes)
                # Otherwise it's a Chat Completions event — passthrough
                return [raw_bytes]

        return [raw_bytes]

    def translate_upstream_stream_event_for_model(self, raw_bytes: bytes, model: str) -> list[bytes]:
        """Translate SSE events, routing based on model."""
        if _is_messages_model(model):
            return self._translate_anthropic_stream_event(raw_bytes)
        return [raw_bytes]

    # ── Anthropic Messages API translation (inline) ───────────────────────

    def _translate_to_anthropic(self, cc_request: dict) -> dict:
        anthropic: dict = {
            "model": cc_request["model"],
            "max_tokens": cc_request.get("max_tokens", _DEFAULT_MAX_TOKENS),
            "messages": [],
        }
        if cc_request.get("stream") is not None:
            anthropic["stream"] = cc_request["stream"]
        if "temperature" in cc_request and cc_request["temperature"] is not None:
            anthropic["temperature"] = cc_request["temperature"]
        if "top_p" in cc_request and cc_request["top_p"] is not None:
            anthropic["top_p"] = cc_request["top_p"]

        # System messages → top-level system field
        system_parts: list[str] = []
        for msg in cc_request.get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block["text"])
                        elif isinstance(block, str):
                            system_parts.append(block)
        if system_parts:
            anthropic["system"] = "\n".join(system_parts)

        # Translate messages
        for msg in cc_request.get("messages", []):
            role = msg.get("role")
            if role == "system":
                continue
            if role == "assistant":
                anthropic["messages"].append(self._translate_assistant_msg(msg))
            elif role == "tool":
                anthropic["messages"].append(self._translate_tool_result_msg(msg))
            else:
                anthropic["messages"].append({"role": role, "content": msg.get("content", "")})

        # Translate tools
        if "tools" in cc_request and cc_request["tools"]:
            anthropic["tools"] = [
                {
                    "name": t.get("function", {}).get("name", ""),
                    "description": t.get("function", {}).get("description", ""),
                    "input_schema": t.get("function", {}).get("parameters", {"type": "object", "properties": {}}),
                }
                for t in cc_request["tools"]
            ]

        return anthropic

    @staticmethod
    def _translate_assistant_msg(msg: dict) -> dict:
        content_blocks: list[dict] = []
        text = msg.get("content")
        if text:
            content_blocks.append({"type": "text", "text": text})
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": func.get("name", ""),
                    "input": _safe_json_load_args(func.get("arguments")),
                }
            )
        return {"role": "assistant", "content": content_blocks or ""}

    @staticmethod
    def _translate_tool_result_msg(msg: dict) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
            ],
        }

    def _translate_from_anthropic(self, raw: dict) -> dict:
        content_blocks = raw.get("content", [])
        text_parts: list[str] = []
        tool_uses: list[dict] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_uses.append(block)

        message: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
        if tool_uses:
            message["tool_calls"] = [
                {
                    "id": tu.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for tu in tool_uses
            ]

        stop_reason = raw.get("stop_reason")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")
        usage = raw.get("usage", {})

        return {
            "id": raw.get("id", ""),
            "object": "chat.completion",
            "created": 0,
            "model": raw.get("model", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        }

    def _translate_anthropic_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        data_str = None
        for line in raw_str.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()

        if not data_str:
            return []

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return [raw_bytes]

        data_type = data.get("type", "")

        if data_type in ("ping", "content_block_start", "content_block_stop"):
            return []

        if data_type == "message_start":
            msg = data.get("message", {})
            return self._make_cc_chunk({"role": "assistant"}, msg.get("model", ""))

        if data_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return self._make_cc_chunk({"content": delta.get("text", "")})
            return []

        if data_type == "message_delta":
            delta = data.get("delta", {})
            stop = delta.get("stop_reason")
            finish = _STOP_REASON_MAP.get(stop, "stop")
            return self._make_cc_chunk({}, finish_reason=finish)

        if data_type == "message_stop":
            return [b"data: [DONE]\n\n"]

        return [raw_bytes]

    @staticmethod
    def _make_cc_chunk(delta: dict, model: str = "", finish_reason: str | None = None) -> list[bytes]:
        chunk: dict = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return [f"data: {json.dumps(chunk)}\n\n".encode()]

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
