"""Responses API <-> Chat Completions translation.

Converts between the OpenAI Responses API format (used by Codex CLI) and the
Chat Completions format used by upstream providers.

The translator emits the full Responses API streaming lifecycle:
  response.created -> response.in_progress ->
  response.output_item.added -> response.content_part.added ->
  response.output_text.delta x N ->
  response.output_text.done -> response.content_part.done ->
  response.output_item.done ->
  response.completed
"""

from __future__ import annotations

import re
import uuid

from kitty.bridge.engine import ToolCallBuffer, ToolCallBufferError
from kitty.bridge.responses.events import (
    format_content_part_added_event,
    format_content_part_done_event,
    format_function_call_arguments_delta_event,
    format_function_call_arguments_done_event,
    format_output_item_added_event,
    format_output_item_done_event,
    format_output_text_delta_event,
    format_output_text_done_event,
    format_response_completed_event,
    format_response_created_event,
    format_response_in_progress_event,
)

__all__ = ["ResponsesTranslator"]

# MiniMax interleaved thinking tags: <اخل>...</اخل>
_THINKING_TAG_RE = re.compile(r"<\u0627\u062e\u0644>.*?</\u0627\u062e\u0644>", re.DOTALL)


def _strip_thinking_tags(text: str) -> str:
    """Strip MiniMax-style interleaved thinking tags from content."""
    return _THINKING_TAG_RE.sub("", text).strip()


class ResponsesTranslator:
    """Translates between Responses API and Chat Completions formats."""

    def __init__(self) -> None:
        self._tool_call_buffers: dict[int, ToolCallBuffer] = {}
        self._tool_call_meta: dict[int, dict] = {}  # index -> {id, name, call_id, item_id}
        self._accumulated_text: str = ""
        self._accumulated_reasoning: str = ""
        self._seq: int = 0
        self._text_item_id: str | None = None
        self._reasoning_item_id: str | None = None
        self._text_started: bool = False
        self._reasoning_started: bool = False
        self._last_was_empty: bool = False

    @property
    def response_was_empty(self) -> bool:
        """True if the last translated response produced no meaningful content."""
        return self._last_was_empty

    def reset(self) -> None:
        """Clear internal streaming state between requests."""
        self._tool_call_buffers = {}
        self._tool_call_meta = {}
        self._accumulated_text = ""
        self._accumulated_reasoning = ""
        self._seq = 0
        self._text_item_id = None
        self._reasoning_item_id = None
        self._text_started = False
        self._reasoning_started = False
        self._last_was_empty = False

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    # ── Stream lifecycle ───────────────────────────────────────────────────

    def translate_stream_start(self, response_id: str, model: str = "") -> list[str]:
        """Emit response.created and response.in_progress at stream start.

        Called by the server before opening the upstream connection.
        """
        return [
            format_response_created_event(response_id, seq=self._next_seq(), model=model),
            format_response_in_progress_event(response_id, seq=self._next_seq(), model=model),
        ]

    # ── Request translation ───────────────────────────────────────────────

    def translate_request(self, responses_request: dict) -> dict:
        """Convert a Responses API request to a Chat Completions request."""
        messages = []

        # System instructions -> system message
        instructions = responses_request.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Input items -> messages
        # Accumulate reasoning from 'reasoning' items and merge into the next
        # assistant message's reasoning_content field.
        pending_reasoning: list[str] = []
        for item in responses_request.get("input", []):
            # Extract reasoning text from reasoning items
            if item.get("type") == "reasoning":
                for summary in item.get("summary", []):
                    if summary.get("type") == "summary_text" and summary.get("text"):
                        pending_reasoning.append(summary["text"])
                continue

            msg = self._translate_input_item(item)
            if msg is not None:
                # Attach accumulated reasoning to assistant messages
                if msg.get("role") == "assistant" and pending_reasoning:
                    msg["reasoning_content"] = "\n".join(pending_reasoning)
                    pending_reasoning = []
                messages.append(msg)

        # Merge consecutive system messages (some providers reject multiples)
        messages = self._merge_consecutive_system_messages(messages)

        result: dict = {
            "model": responses_request["model"],
            "messages": messages,
            "stream": responses_request.get("stream", False),
        }

        # max_output_tokens -> max_tokens
        if "max_output_tokens" in responses_request:
            result["max_tokens"] = responses_request["max_output_tokens"]

        # Tools
        if "tools" in responses_request:
            result["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                }
                for t in responses_request["tools"]
                if t.get("type") == "function"
            ]

        # Pass through extra kwargs
        for key in ("temperature", "top_p", "presencePenalty", "frequencyPenalty", "seed"):
            if key in responses_request:
                result[key] = responses_request[key]

        return result

    @staticmethod
    def _convert_content(content: object) -> object:
        """Convert Responses API content parts to Chat Completions format.

        Responses API uses: [{"type": "input_text", "text": "..."}]
        Chat Completions uses: "..." (string) or [{"type": "text", "text": "..."}]
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Convert each part
            parts: list[dict | str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    ptype = part.get("type", "")
                    # input_text, output_text → extract plain text
                    if ptype in ("input_text", "output_text"):
                        parts.append(part.get("text", ""))
                    # input_image, input_file etc → skip (not supported by CC)
                    elif ptype in ("text",):
                        parts.append(part)
                    # Other types: skip
            # If all parts are strings, concatenate into single string
            if all(isinstance(p, str) for p in parts):
                return "\n".join(parts) if parts else ""
            return parts
        return content

    @staticmethod
    def _merge_consecutive_system_messages(messages: list[dict]) -> list[dict]:
        """Merge consecutive system messages into a single system message.

        Some providers (e.g. MiniMax) reject requests with multiple system messages.
        """
        result: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system" and result and result[-1].get("role") == "system":
                prev = result[-1].get("content") or ""
                curr = msg.get("content") or ""
                result[-1]["content"] = f"{prev}\n\n{curr}"
            else:
                result.append(msg)
        return result

    def _translate_input_item(self, item: dict) -> dict | None:
        """Translate a single Responses API input item to a Chat Completions message."""
        item_type = item.get("type")

        if item_type == "function_call":
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.get("call_id", f"call_{uuid.uuid4().hex}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                ],
            }

        if item_type == "function_call_output":
            return {
                "role": "tool",
                "tool_call_id": item["call_id"],
                "content": item.get("output", ""),
            }

        # Standard role-based messages (user, assistant, system, developer → system)
        # Also handles "type": "message" items which have role + content
        if "role" in item:
            role = item["role"]
            if role == "developer":
                role = "system"
            return {
                "role": role,
                "content": self._convert_content(item.get("content", "")),
            }

        return None

    # ── Response translation (sync) ──────────────────────────────────────

    def translate_response(self, cc_response: dict) -> dict:
        """Convert a Chat Completions response to a Responses API response."""
        choices = cc_response.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        output: list[dict] = []

        # Reasoning content -> reasoning output item
        reasoning = message.get("reasoning_content")
        if reasoning:
            output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "summary": [{"type": "summary_text", "text": reasoning}],
                }
            )

        # Text content
        content = message.get("content")
        if content:
            content = _strip_thinking_tags(content)
            if content:
                output.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )

        # Tool calls -> function_call items
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": tc.get("id", f"call_{uuid.uuid4().hex}"),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                    "status": "completed",
                }
            )

        status = "completed"
        if finish_reason == "length":
            status = "incomplete"

        usage = cc_response.get("usage", {})
        return {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "model": cc_response.get("model", ""),
            "output": output,
            "status": status,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    # ── Stream chunk translation ─────────────────────────────────────────

    def _ensure_text_item_started(self, response_id: str) -> None:
        """Lazily emit the output_item.added and content_part.added events for text."""
        if self._text_started:
            return
        self._text_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._text_started = True

    def _ensure_reasoning_item_started(self, response_id: str) -> None:
        """Lazily initialize the reasoning item state."""
        if self._reasoning_started:
            return
        self._reasoning_item_id = f"rs_{uuid.uuid4().hex[:24]}"
        self._reasoning_started = True

    def translate_stream_chunk(
        self,
        response_id: str,
        chunk: dict,
    ) -> list[str]:
        """Convert a Chat Completions streaming chunk to Responses SSE event strings."""
        events: list[str] = []
        choices = chunk.get("choices", [])
        if not choices:
            return events
        choice = choices[0]
        delta = choice.get("delta", {})

        # Reasoning delta
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            if not self._reasoning_started:
                self._ensure_reasoning_item_started(response_id)
                # Emit output_item.added for reasoning
                events.append(
                    format_output_item_added_event(
                        seq=self._next_seq(),
                        output_index=0,
                        item={
                            "id": self._reasoning_item_id,
                            "type": "reasoning",
                            "summary": [],
                        },
                    )
                )
            self._accumulated_reasoning += reasoning_content

        # Text delta
        content = delta.get("content")
        if content:
            # Lazily start text item on first content
            if not self._text_started:
                self._ensure_text_item_started(response_id)
                # Emit output_item.added
                events.append(
                    format_output_item_added_event(
                        seq=self._next_seq(),
                        output_index=0,
                        item={
                            "id": self._text_item_id,
                            "type": "message",
                            "status": "in_progress",
                            "content": [],
                            "role": "assistant",
                        },
                    )
                )
                # Emit content_part.added
                events.append(
                    format_content_part_added_event(
                        seq=self._next_seq(),
                        item_id=self._text_item_id,
                        output_index=0,
                        content_index=0,
                        part={"type": "output_text", "text": ""},
                    )
                )

            self._accumulated_text += content
            events.append(
                format_output_text_delta_event(
                    seq=self._next_seq(),
                    response_id=response_id,
                    item_id=self._text_item_id,
                    output_index=0,
                    content_index=0,
                    delta=content,
                )
            )

        # Tool call delta
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc_delta in tool_calls:
                idx = tc_delta.get("index", 0)

                # New tool call: id + name arrive in first chunk
                if "id" in tc_delta:
                    call_id = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    item_id = f"fc_{uuid.uuid4().hex[:24]}"
                    self._tool_call_meta[idx] = {
                        "call_id": call_id,
                        "name": func.get("name", ""),
                        "item_id": item_id,
                    }
                    self._tool_call_buffers[idx] = ToolCallBuffer()
                    # Emit output_item.added for function call
                    fc_item = {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": func.get("name", ""),
                        "arguments": "",
                        "status": "in_progress",
                    }
                    events.append(
                        format_output_item_added_event(
                            seq=self._next_seq(),
                            output_index=idx,
                            item=fc_item,
                        )
                    )

                # Argument delta
                func = tc_delta.get("function", {})
                arg_delta = func.get("arguments", "")
                if arg_delta and idx in self._tool_call_buffers:
                    self._tool_call_buffers[idx].append(arg_delta)
                    meta = self._tool_call_meta[idx]
                    events.append(
                        format_function_call_arguments_delta_event(
                            seq=self._next_seq(),
                            response_id=response_id,
                            item_id=meta["item_id"],
                            call_id=meta["call_id"],
                            delta=arg_delta,
                        )
                    )

        # Finish
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            events.extend(self._build_finish_events(response_id, chunk))

        return events

    def _clean_text(self) -> str:
        """Return accumulated text with thinking tags stripped."""
        return _strip_thinking_tags(self._accumulated_text) if self._accumulated_text else ""

    def _build_finish_events(self, response_id: str, chunk: dict) -> list[str]:
        """Build the trailing lifecycle events: done events → output_item.done → completed."""
        events: list[str] = []
        choices = chunk.get("choices", [])
        choice = choices[0] if choices else {}
        finish_reason = choice.get("finish_reason", "stop")

        status = "completed"
        if finish_reason == "length":
            status = "incomplete"

        # Close reasoning item if started
        if self._reasoning_started and self._reasoning_item_id:
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=0,
                    item={
                        "type": "reasoning",
                        "id": self._reasoning_item_id,
                        "summary": [{"type": "summary_text", "text": self._accumulated_reasoning}],
                    },
                )
            )

        # Close text content if any was accumulated
        clean_text = self._clean_text()
        if self._text_started and self._text_item_id:
            # output_text.done
            events.append(
                format_output_text_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=0,
                    content_index=0,
                    text=clean_text,
                )
            )
            # content_part.done
            events.append(
                format_content_part_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "output_text", "text": clean_text},
                )
            )
            # output_item.done for message
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=0,
                    item={
                        "id": self._text_item_id,
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": clean_text}],
                        "role": "assistant",
                    },
                )
            )

        # Finalize tool calls
        finalized_args: dict[int, str] = {}
        for idx, buf in self._tool_call_buffers.items():
            meta = self._tool_call_meta[idx]
            try:
                final_args = buf.finalize()
            except ToolCallBufferError:
                final_args = "{}"
            finalized_args[idx] = final_args

            events.append(
                format_function_call_arguments_done_event(
                    seq=self._next_seq(),
                    response_id=response_id,
                    item_id=meta["item_id"],
                    call_id=meta["call_id"],
                    arguments=final_args,
                )
            )

            # output_item.done for function call
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=idx,
                    item={
                        "type": "function_call",
                        "id": meta["item_id"],
                        "call_id": meta["call_id"],
                        "name": meta["name"],
                        "arguments": final_args,
                        "status": "completed" if status == "completed" else "incomplete",
                    },
                )
            )

        # Build output items for the completed response
        output_items: list[dict] = []
        if self._reasoning_started and self._reasoning_item_id:
            output_items.append(
                {
                    "type": "reasoning",
                    "id": self._reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": self._accumulated_reasoning}],
                }
            )
        if clean_text:
            output_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": clean_text}],
                }
            )
        for idx in self._tool_call_buffers:
            meta = self._tool_call_meta[idx]
            final_args = finalized_args.get(idx, "{}")
            output_items.append(
                {
                    "type": "function_call",
                    "id": meta["item_id"],
                    "call_id": meta["call_id"],
                    "name": meta["name"],
                    "arguments": final_args,
                    "status": "completed" if status == "completed" else "incomplete",
                }
            )

        # Build completed event
        usage = chunk.get("usage") or {}
        response_data = {
            "object": "response",
            "model": chunk.get("model", ""),
            "status": status,
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        events.append(format_response_completed_event(response_id, seq=self._next_seq(), response_data=response_data))

        # Track emptiness before reset clears the state
        was_empty = not clean_text and not self._tool_call_buffers and not self._accumulated_reasoning
        self.reset()
        self._last_was_empty = was_empty
        return events

    def synthesize_completed_events(
        self,
        response_id: str,
        model: str = "",
        status: str = "completed",
    ) -> list[str]:
        """Build trailing lifecycle events from accumulated streaming state.

        Called when the upstream stream ends without emitting a finish_reason chunk,
        to ensure the client always receives the full lifecycle before EOF.
        Returns empty list only when status is completed and no content/tool calls
        were accumulated, preventing duplicate completion after normal finish chunks.
        """
        clean_text = self._clean_text()
        if status == "completed" and not clean_text and not self._tool_call_buffers and not self._accumulated_reasoning:
            return []

        events: list[str] = []

        # Close text content
        if self._text_started and self._text_item_id and clean_text:
            events.append(
                format_output_text_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=0,
                    content_index=0,
                    text=clean_text,
                )
            )
            events.append(
                format_content_part_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "output_text", "text": clean_text},
                )
            )
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=0,
                    item={
                        "id": self._text_item_id,
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": clean_text}],
                        "role": "assistant",
                    },
                )
            )

        # Finalize tool calls (cache results to avoid double-finalize bug)
        finalized_args: dict[int, str] = {}
        for idx, buf in self._tool_call_buffers.items():
            meta = self._tool_call_meta[idx]
            try:
                final_args = buf.finalize()
            except ToolCallBufferError:
                final_args = "{}"
            finalized_args[idx] = final_args
            events.append(
                format_function_call_arguments_done_event(
                    seq=self._next_seq(),
                    response_id=response_id,
                    item_id=meta["item_id"],
                    call_id=meta["call_id"],
                    arguments=final_args,
                )
            )
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=idx,
                    item={
                        "type": "function_call",
                        "id": meta["item_id"],
                        "call_id": meta["call_id"],
                        "name": meta["name"],
                        "arguments": final_args,
                        "status": "completed" if status == "completed" else "incomplete",
                    },
                )
            )

        # Build response.completed
        output_items: list[dict] = []
        if clean_text:
            output_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": clean_text}],
                }
            )
        for idx in self._tool_call_buffers:
            meta = self._tool_call_meta[idx]
            final_args = finalized_args.get(idx, "{}")
            output_items.append(
                {
                    "type": "function_call",
                    "id": meta["item_id"],
                    "call_id": meta["call_id"],
                    "name": meta["name"],
                    "arguments": final_args,
                    "status": "completed" if status == "completed" else "incomplete",
                }
            )

        response_data = {
            "object": "response",
            "model": model,
            "status": status,
            "output": output_items,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        events.append(format_response_completed_event(response_id, seq=self._next_seq(), response_data=response_data))

        was_empty = not clean_text and not self._tool_call_buffers
        self.reset()
        self._last_was_empty = was_empty
        return events
