"""Anthropic Messages API <-> Chat Completions translation.

Converts between the Anthropic Messages API format (used by Claude Code) and the
Chat Completions format used by upstream providers.
"""

from __future__ import annotations

import json
import uuid

from kitty.bridge.engine import ToolCallBuffer, TranslationEngine
from kitty.bridge.messages.events import (
    format_content_block_delta_event,
    format_content_block_start_event,
    format_content_block_stop_event,
    format_message_delta_event,
    format_message_start_event,
    format_message_stop_event,
)

__all__ = ["MessagesTranslator"]


class MessagesTranslator:
    """Translates between Anthropic Messages API and Chat Completions formats."""

    def __init__(self, thinking_warned: bool = False) -> None:
        self._thinking_warned: bool = thinking_warned
        self._tool_call_buffers: dict[int, ToolCallBuffer] = {}
        self._tool_call_meta: dict[int, dict] = {}  # index -> {id, tool_id, name, block_index}
        self._content_block_index: int = 0
        self._text_block_opened: bool = False

    @property
    def thinking_warned(self) -> bool:
        return self._thinking_warned

    def reset(self) -> None:
        """Clear internal streaming state between requests."""
        self._tool_call_buffers = {}
        self._tool_call_meta = {}
        self._content_block_index = 0
        self._text_block_opened = False

    # ── Request translation ───────────────────────────────────────────────

    def translate_request(self, messages_request: dict) -> dict:
        """Convert a Messages API request to a Chat Completions request."""
        messages = []

        # System prompt -> system message
        system = messages_request.get("system")
        if system:
            messages.append({"role": "system", "content": system})

        # Messages with content block handling
        for msg in messages_request.get("messages", []):
            translated = self._translate_message(msg)
            if translated is None:
                continue
            if isinstance(translated, list):
                messages.extend(translated)
            else:
                messages.append(translated)

        result: dict = {
            "model": messages_request["model"],
            "messages": messages,
            "stream": messages_request.get("stream", False),
        }

        if "max_tokens" in messages_request:
            result["max_tokens"] = messages_request["max_tokens"]

        # Tools: Anthropic format -> Chat Completions format
        if "tools" in messages_request:
            result["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in messages_request["tools"]
            ]

        # Pass through supported kwargs
        for key in ("temperature", "top_p"):
            if key in messages_request:
                result[key] = messages_request[key]

        # Strip thinking config and warn once
        if "thinking" in messages_request and not self._thinking_warned:
            self._thinking_warned = True
            # thinking is intentionally NOT passed to upstream

        return result

    def _translate_message(self, msg: dict) -> dict | None:
        """Translate a single Messages API message to Chat Completions format.

        Returns the translated message dict. For user messages with multiple
        tool_result blocks, the caller should use ``_translate_messages`` instead.
        """
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            return self._translate_user_message(content)
        if role == "assistant":
            return self._translate_assistant_message(content)

        # Fallback for simple string content
        if isinstance(content, str):
            return {"role": role, "content": content}

        return {"role": role, "content": str(content) if content else ""}

    def _translate_user_message(self, content) -> dict | list[dict]:
        """Translate a user message, handling tool_result content blocks.

        Returns a single message dict for simple content, or a list of message
        dicts when multiple ``tool_result`` blocks need separate ``tool`` role
        messages in Chat Completions format.
        """
        if isinstance(content, str):
            return {"role": "user", "content": content}

        if isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                if len(tool_results) == 1:
                    tr = tool_results[0]
                    return {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", ""),
                    }
                # Multiple tool results -> multiple tool role messages
                return [
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", ""),
                    }
                    for tr in tool_results
                ]

            # Regular content blocks -> concatenate text
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return {"role": "user", "content": "\n".join(text_parts) if text_parts else ""}

        return {"role": "user", "content": str(content) if content else ""}

    def _translate_assistant_message(self, content) -> dict:
        """Translate an assistant message, handling tool_use content blocks."""
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        if isinstance(content, list):
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", f"call_{uuid.uuid4().hex}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif block.get("type") == "thinking":
                    # Strip thinking blocks from upstream request
                    pass

            result: dict = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        return {"role": "assistant", "content": str(content) if content else None}

    # ── Response translation (sync) ──────────────────────────────────────

    def translate_response(self, cc_response: dict) -> dict:
        """Convert a Chat Completions response to a Messages API response."""
        choices = cc_response.get("choices") or [{}]
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        content: list[dict] = []

        # Text content -> text block
        text = message.get("content")
        if text:
            content.append({"type": "text", "text": text})

        # Tool calls -> tool_use blocks
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_data = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": func.get("name", ""),
                    "input": input_data,
                }
            )

        stop_reason = TranslationEngine.map_finish_reason(finish_reason)
        usage = cc_response.get("usage") or {}

        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": cc_response.get("model", ""),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    # ── Stream chunk translation ─────────────────────────────────────────

    def translate_stream_chunk(
        self,
        message_id: str,
        model: str,
        chunk: dict,
    ) -> list[str]:
        """Convert a Chat Completions streaming chunk to Messages API SSE event strings."""
        events: list[str] = []
        choice = chunk.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        # Text delta
        text_content = delta.get("content")
        if text_content:
            # Open text block on first text delta
            if not self._text_block_opened:
                # Emit message_start on first content
                message_obj = {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                events.append(format_message_start_event(message_obj))

                # Open text content block
                events.append(
                    format_content_block_start_event(
                        self._content_block_index,
                        {"type": "text", "text": ""},
                    )
                )
                self._text_block_opened = True

            events.append(
                format_content_block_delta_event(
                    self._content_block_index,
                    {"type": "text_delta", "text": text_content},
                )
            )

        # Tool call delta
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc_delta in tool_calls:
                idx = tc_delta.get("index", 0)

                # New tool call: id + name arrive in first chunk
                if "id" in tc_delta:
                    # Close text block if still open
                    if self._text_block_opened:
                        events.append(format_content_block_stop_event(self._content_block_index))
                        self._content_block_index += 1
                        self._text_block_opened = False

                    # Emit message_start if not yet emitted
                    if not events and self._content_block_index == 0 and not self._text_block_opened:
                        message_obj = {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }
                        events.append(format_message_start_event(message_obj))

                    call_id = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"

                    self._tool_call_meta[idx] = {
                        "id": call_id,
                        "tool_id": tool_id,
                        "name": func.get("name", ""),
                        "block_index": self._content_block_index,
                    }
                    self._tool_call_buffers[idx] = ToolCallBuffer()

                    # Open tool_use content block
                    events.append(
                        format_content_block_start_event(
                            self._content_block_index,
                            {"type": "tool_use", "id": tool_id, "name": func.get("name", ""), "input": {}},
                        )
                    )

                # Argument delta
                func = tc_delta.get("function", {})
                arg_delta = func.get("arguments", "")
                if arg_delta and idx in self._tool_call_buffers:
                    self._tool_call_buffers[idx].append(arg_delta)
                    meta = self._tool_call_meta[idx]
                    events.append(
                        format_content_block_delta_event(
                            meta["block_index"],
                            {"type": "input_json_delta", "partial_json": arg_delta},
                        )
                    )

        # Finish
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            # Close text block if still open
            if self._text_block_opened:
                events.append(format_content_block_stop_event(self._content_block_index))
                self._content_block_index += 1
                self._text_block_opened = False

            # Close any tool_use blocks
            for idx, _buf in self._tool_call_buffers.items():
                meta = self._tool_call_meta[idx]
                events.append(format_content_block_stop_event(meta["block_index"]))

            # Map stop reason
            stop_reason = TranslationEngine.map_finish_reason(finish_reason)
            usage = chunk.get("usage") or {}

            # Emit message_delta with stop_reason + usage
            events.append(
                format_message_delta_event(
                    delta={"stop_reason": stop_reason, "stop_sequence": None},
                    usage={"output_tokens": usage.get("completion_tokens", 0)},
                )
            )

            # Emit message_stop
            events.append(format_message_stop_event())

            # Auto-reset
            self.reset()

        return events
