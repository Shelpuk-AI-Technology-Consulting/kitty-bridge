"""Anthropic provider adapter — translates between Chat Completions and Anthropic Messages API."""

from __future__ import annotations

import json
import logging
import uuid

from kitty.providers.base import ProviderAdapter, ProviderError

__all__ = ["AnthropicAdapter"]

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# stop_reason → finish_reason
_STOP_REASON_MAP: dict[str | None, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    None: "stop",
}

# CC finish_reason → Anthropic stop_reason (for reverse mapping if needed)
_FINISH_TO_STOP: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class AnthropicAdapter(ProviderAdapter):
    """Anthropic Messages API adapter.

    Translates between Kitty's internal Chat Completions format and
    Anthropic's Messages API (``POST /v1/messages``).  Anthropic uses
    ``x-api-key`` authentication and a content-block based request/response
    format rather than CC's message/content structure.
    """

    @property
    def provider_type(self) -> str:
        return "anthropic"

    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com"

    @property
    def upstream_path(self) -> str:
        return "/v1/messages"

    # ── Auth headers ─────────────────────────────────────────────────────

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    # ── CC → Anthropic request translation ───────────────────────────────

    def translate_to_upstream(self, cc_request: dict) -> dict:
        """Translate a Chat Completions request into an Anthropic Messages request."""
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

        # Extract system messages → top-level system field
        system_parts: list[str] = []
        for msg in cc_request.get("messages", []):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    # System content as list of blocks — extract text
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
                continue  # already handled above

            if role == "assistant":
                anthropic["messages"].append(self._translate_assistant_msg(msg))
            elif role == "tool":
                anthropic["messages"].append(self._translate_tool_result_msg(msg))
            else:
                anthropic["messages"].append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )

        # Translate tools
        if "tools" in cc_request and cc_request["tools"]:
            anthropic["tools"] = self._translate_tools(cc_request["tools"])

        return anthropic

    def _translate_assistant_msg(self, msg: dict) -> dict:
        """Translate an assistant message with optional tool_calls to Anthropic content blocks."""
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
                    "input": json.loads(func.get("arguments", "{}")),
                }
            )

        return {"role": "assistant", "content": content_blocks or ""}

    def _translate_tool_result_msg(self, msg: dict) -> dict:
        """Translate a tool result message to Anthropic user message with tool_result block."""
        content = msg.get("content", "")
        # Anthropic requires tool_result to be inside a user message
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }
            ],
        }

    def _translate_tools(self, cc_tools: list[dict]) -> list[dict]:
        """Translate CC tool definitions to Anthropic format."""
        anthropic_tools = []
        for tool in cc_tools:
            func = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return anthropic_tools

    # ── Anthropic response → CC translation ──────────────────────────────

    def translate_from_upstream(self, raw_response: dict) -> dict:
        """Translate an Anthropic Messages response into Chat Completions format."""
        content_blocks = raw_response.get("content", [])
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

        stop_reason = raw_response.get("stop_reason")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, "stop")

        usage = raw_response.get("usage", {})
        return {
            "id": raw_response.get("id", ""),
            "object": "chat.completion",
            "created": 0,
            "model": raw_response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        }

    # ── SSE stream event translation ─────────────────────────────────────

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        """Translate an Anthropic SSE event into CC-format SSE chunks.

        Anthropic event flow:
        - message_start → emit CC chunk with role
        - content_block_start → ignored
        - content_block_delta (text_delta) → CC chunk with content delta
        - content_block_delta (input_json_delta) → buffered for tool_use
        - content_block_stop → flush tool buffer if tool_use block
        - message_delta → CC chunk with finish_reason
        - message_stop → emit [DONE]
        - ping → ignored
        """
        raw_str = raw_bytes.decode("utf-8", errors="replace").strip()
        if not raw_str:
            return []

        # Parse SSE lines
        data_str = None
        for line in raw_str.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()

        if not data_str:
            return []

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return [raw_bytes]

        data_type = data.get("type", "")

        # Ignore non-content events
        if data_type in ("ping", "content_block_start", "content_block_stop"):
            return []

        if data_type == "message_start":
            msg = data.get("message", {})
            return self._make_cc_chunk({"role": "assistant"}, msg.get("model", ""))

        if data_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                return self._make_cc_chunk({"content": text})
            # input_json_delta — suppress (tool use streaming not needed for CC)
            return []

        if data_type == "message_delta":
            delta = data.get("delta", {})
            stop = delta.get("stop_reason")
            finish = _STOP_REASON_MAP.get(stop, "stop")
            return self._make_cc_chunk({}, finish_reason=finish)

        if data_type == "message_stop":
            return [b"data: [DONE]\n\n"]

        # Unknown event — passthrough
        return [raw_bytes]

    def _make_cc_chunk(
        self,
        delta: dict,
        model: str = "",
        finish_reason: str | None = None,
    ) -> list[bytes]:
        """Build a CC streaming chunk wrapped in SSE data."""
        chunk: dict = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return [f"data: {json.dumps(chunk)}\n\n".encode()]

    # ── Standard ProviderAdapter methods ─────────────────────────────────

    def normalize_model_name(self, model: str) -> str:
        """Strip provider prefix if present (e.g. 'anthropic/claude-sonnet-4-6')."""
        if "/" in model:
            return model.rsplit("/", 1)[-1]
        return model

    def build_request(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        request: dict = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", False),
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        if "tools" in kwargs and kwargs["tools"]:
            request["tools"] = kwargs["tools"]
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
        return ProviderError(f"Anthropic error {status_code}: {msg}")
