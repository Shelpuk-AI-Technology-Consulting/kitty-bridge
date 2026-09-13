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

__all__ = ["MessagesTranslator", "carry_tool_choice_and_metadata"]

_EMPTY_ASSISTANT_FALLBACK_TEXT = (
    "Upstream model returned an empty response. Please retry. "
    "If the context is full, use /clear to reset the conversation."
)

#: Anthropic ``tool_choice.type`` values whose Chat Completions spelling is a
#: plain string.  ``tool`` is absent because it carries a name.
_SIMPLE_TOOL_CHOICES: dict[str, str] = {"auto": "auto", "any": "required", "none": "none"}


def _names_anthropic_defined_tool(tools: list, name: str) -> bool:
    """Report whether ``tools`` declares ``name`` as an Anthropic-defined tool.

    An ordinary client tool carries ``type: "custom"``, ``type: null`` or no
    ``type`` at all (``ToolParam.type`` is ``Optional[Literal["custom"]]``).  Any
    other ``type`` -- ``web_search_20250305``, ``bash_20250124`` and the rest --
    is Anthropic-defined, and this hop flattens it into a plain function without
    its real schema or runtime.  A name nothing declares is *not* reported: that
    body is the agent's mistake, and the provider's error says so better than a
    silent omission would.

    Args:
        tools: The inbound Messages ``tools`` list.
        name: The tool name a ``tool_choice`` of type ``tool`` selects.

    Returns:
        True when a declaration with that name carries an Anthropic-defined type.
    """
    return any(
        isinstance(tool, dict) and tool.get("name") == name and tool.get("type") not in (None, "custom")
        for tool in tools
    )


def carry_tool_choice_and_metadata(messages_request: dict, cc_request: dict) -> None:
    """Carry an Anthropic ``tool_choice`` and ``metadata`` onto a Chat Completions body.

    Both Messages -> Chat Completions converters call this --
    :meth:`MessagesTranslator.translate_request` and the ``tool_use`` fallback's
    ``_convert_native_to_cc_format`` in :mod:`kitty.bridge.server` -- so the value
    table exists once.  A drifted second copy of hop 1 is how KBR-178's field was
    lost on the retry path.

    ``tool_choice`` is a constraint, not a hint, and the two wires spell its
    values differently: ``{"type": "any"}`` is ``"required"``, and
    ``{"type": "tool", "name": x}`` is ``{"type": "function", "function":
    {"name": x}}``.  A value Anthropic does not publish -- a string, an unknown
    ``type``, a ``tool`` with no string name -- is omitted rather than repaired,
    because kitty has no authority to invent a reading of it (KBR-214 D5).

    Two further omissions keep the fix from creating failures of its own:

    * **No tools, no choice** (D9).  Anthropic accepts a ``tool_choice`` beside
      no tools; OpenAI rejects one ("'tool_choice' is only allowed when 'tools'
      are specified"), so carrying it would turn a legal request into a 400.
    * **A forced call to an Anthropic-defined tool is not carried** (D10).  A
      server tool such as Claude Code's ``web_search`` carries a versioned
      ``type`` and is flattened into a schema-less function on this hop, so
      forcing it would force a call nothing on the route can execute.  Left
      unforced, the turn behaves as it did before this mapping existed.

    ``disable_parallel_tool_use`` is carried only when ``True``, as
    ``parallel_tool_calls: False``.  ``False`` is the default on both wires, so
    writing it would add a field to a request whose behaviour it does not change
    (D2).

    ``metadata`` has no Chat Completions counterpart -- CC's own ``metadata`` is a
    stored-completions tag map -- so it rides the internal ``_metadata`` key,
    which only Anthropic-family adapters restore (D1).

    Args:
        messages_request: The inbound Anthropic Messages body.  Not modified.
        cc_request: The Chat Completions body being built, mutated in place.

    Returns:
        None. ``cc_request`` gains ``tool_choice``, ``parallel_tool_calls`` and
        ``_metadata`` only where the inbound body carries a value for them.
    """
    # Metadata is independent of the tool choice, so it is carried first and a
    # malformed choice below cannot cost it.
    metadata = messages_request.get("metadata")
    if metadata is not None:
        cc_request["_metadata"] = metadata

    tool_choice = messages_request.get("tool_choice")
    tools = messages_request.get("tools")
    # A choice over no tools is legal Anthropic and a 400 on Chat Completions.
    if not isinstance(tool_choice, dict) or not isinstance(tools, list) or not tools:
        return

    kind = tool_choice.get("type")
    name = tool_choice.get("name")
    if kind in _SIMPLE_TOOL_CHOICES:
        cc_request["tool_choice"] = _SIMPLE_TOOL_CHOICES[kind]
    elif kind == "tool" and isinstance(name, str) and not _names_anthropic_defined_tool(tools, name):
        cc_request["tool_choice"] = {"type": "function", "function": {"name": name}}
    else:
        return

    # `ToolChoiceNone` declares no parallel knob, so a stray one there is ignored.
    if kind != "none" and tool_choice.get("disable_parallel_tool_use") is True:
        cc_request["parallel_tool_calls"] = False


#: KBR-203: the thinking ``display`` values Anthropic accepts without a beta header.
_GA_THINKING_DISPLAYS: tuple[str, ...] = ("summarized", "omitted")


class MessagesTranslator:
    """Translates between Anthropic Messages API and Chat Completions formats."""

    def __init__(self, thinking_warned: bool = False) -> None:
        self._thinking_warned: bool = thinking_warned
        self._tool_call_buffers: dict[int, ToolCallBuffer] = {}
        self._tool_call_meta: dict[int, dict] = {}  # index -> {id, tool_id, name, block_index}
        self._content_block_index: int = 0
        self._text_block_opened: bool = False
        self._thinking_block_opened: bool = False
        self._message_started: bool = False
        self._finished: bool = False
        self._last_message_id: str | None = None
        self._last_was_empty: bool = False

    @property
    def thinking_warned(self) -> bool:
        return self._thinking_warned

    @property
    def response_was_empty(self) -> bool:
        """True if the last translated streaming chunk produced only fallback text (no real content)."""
        return self._last_was_empty

    def reset(self) -> None:
        """Clear internal streaming state between requests."""
        self._tool_call_buffers = {}
        self._tool_call_meta = {}
        self._content_block_index = 0
        self._text_block_opened = False
        self._thinking_block_opened = False
        self._message_started = False
        self._finished = False
        self._last_was_empty = False

    def finalize_interrupted_stream(self) -> list[str]:
        """Close the current streaming message after an upstream interruption."""
        if not self._message_started:
            return []

        events: list[str] = []
        had_thinking = self._thinking_block_opened
        had_text_or_tools = self._text_block_opened or bool(self._tool_call_buffers)
        had_any_content = had_thinking or had_text_or_tools or self._content_block_index > 0

        if not had_any_content:
            fallback_text = self._fallback_assistant_text()
            events.append(
                format_content_block_start_event(
                    self._content_block_index,
                    {"type": "text", "text": ""},
                )
            )
            events.append(
                format_content_block_delta_event(
                    self._content_block_index,
                    {"type": "text_delta", "text": fallback_text},
                )
            )
            events.append(format_content_block_stop_event(self._content_block_index))
            self._content_block_index += 1

        if self._thinking_block_opened:
            events.append(format_content_block_stop_event(self._content_block_index))
            self._content_block_index += 1
            self._thinking_block_opened = False

        if self._text_block_opened:
            events.append(format_content_block_stop_event(self._content_block_index))
            self._content_block_index += 1
            self._text_block_opened = False

        for block_index in sorted({meta["block_index"] for meta in self._tool_call_meta.values()}):
            events.append(format_content_block_stop_event(block_index))

        # If only thinking was emitted, add fallback text
        if had_thinking and not had_text_or_tools:
            fallback_text = self._fallback_assistant_text()
            events.append(
                format_content_block_start_event(
                    self._content_block_index,
                    {"type": "text", "text": ""},
                )
            )
            events.append(
                format_content_block_delta_event(
                    self._content_block_index,
                    {"type": "text_delta", "text": fallback_text},
                )
            )
            events.append(format_content_block_stop_event(self._content_block_index))
            self._content_block_index += 1

        events.append(
            format_message_delta_event(
                delta={"stop_reason": "end_turn", "stop_sequence": None},
                usage={"output_tokens": 0},
            )
        )
        events.append(format_message_stop_event())

        self.reset()
        self._finished = True
        self._last_was_empty = not had_any_content
        return events

    def close_open_blocks(self) -> list[str]:
        """Close the content blocks the client has open, and nothing else.

        Used when a stream fails after bytes reached the client: per §11
        Q14(a) of ``.system_design/TEST_SUITE.md`` the turn then ends in one
        terminal error, so unlike :meth:`finalize_interrupted_stream` this
        emits no fallback text, ``message_delta`` or ``message_stop``.

        Returns:
            One ``content_block_stop`` event per open block, or an empty list
            when no message has started. The translator is reset afterwards,
            so a repeat call returns an empty list.
        """
        if not self._message_started:
            return []

        events: list[str] = []
        if self._thinking_block_opened or self._text_block_opened:
            events.append(format_content_block_stop_event(self._content_block_index))
        for block_index in sorted({meta["block_index"] for meta in self._tool_call_meta.values()}):
            events.append(format_content_block_stop_event(block_index))

        self.reset()
        return events

    # ── Request translation ───────────────────────────────────────────────

    def translate_request(self, messages_request: dict) -> dict:
        """Convert a Messages API request to a Chat Completions request."""
        messages = []

        # System prompt -> system message
        system = messages_request.get("system")
        if system:
            # Anthropic allows system as a string or array of content blocks.
            # Chat Completions requires a plain string.
            if isinstance(system, list):
                parts = []
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                system = "\n".join(parts)
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

        # KBR-178: Chat Completions calls this `stop`.  Without the rename the
        # user's stop sequences die here and nothing downstream can restore
        # them.  An empty list is omitted deliberately: it asks for no stop
        # behaviour, and `stop: []` violates the published CC schema
        # (`StopConfiguration` declares `minItems: 1`).
        stop_sequences = messages_request.get("stop_sequences")
        if stop_sequences:
            result["stop"] = stop_sequences

        # KBR-178: Chat Completions declares no `top_k` at all, so carrying it
        # as a bare key would be a field no CC provider accepts.  It travels as
        # internal metadata instead and only Anthropic-family adapters restore
        # it; `_INTERNAL_KEYS` keeps it off every other provider's wire.
        if messages_request.get("top_k") is not None:
            result["_top_k"] = messages_request["top_k"]

        # KBR-214: without this the agent's tool constraint dies here -- a forced
        # tool call becomes optional, and a forbidden one becomes possible.
        carry_tool_choice_and_metadata(messages_request, result)

        # Preserve the effort parameter for Anthropic-compatible upstreams.
        # Claude Code sends this to control reasoning depth (e.g. "low",
        # "medium", "high", "xhigh").  The Anthropic Messages API accepts
        # it as a top-level parameter.
        if "effort" in messages_request:
            result["_effort"] = messages_request["effort"]

        # Extract thinking config into normalized effort metadata
        thinking = messages_request.get("thinking")
        if thinking and isinstance(thinking, dict):
            if not self._thinking_warned:
                self._thinking_warned = True
            if thinking.get("type") == "enabled":
                result["_thinking_enabled"] = True
                result["_reasoning_effort"] = "high"
            elif thinking.get("type") == "adaptive":
                # Adaptive thinking — remember the original type so
                # AnthropicAdapter can restore it verbatim.
                result["_thinking_adaptive"] = True
                result["_thinking_enabled"] = True
                result["_reasoning_effort"] = "high"
            elif thinking.get("type") == "disabled":
                result["_thinking_enabled"] = False
            # KBR-203: Chat Completions has no slot for `display`.  Only the two GA
            # values are carried -- beta "updates" needs a header kitty never
            # sends -- and never with `disabled`, which Anthropic rejects.
            display = thinking.get("display")
            if thinking.get("type") in ("enabled", "adaptive") and display in _GA_THINKING_DISPLAYS:
                result["_thinking_display"] = display

        return result

    def _translate_message(self, msg: dict) -> dict | list[dict] | None:
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
        """Translate an assistant message, handling tool_use and thinking content blocks."""
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        if isinstance(content, list):
            text_parts = []
            tool_calls = []
            thinking_parts = []
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
                    thinking_text = block.get("thinking", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)

            result: dict = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if thinking_parts:
                result["reasoning_content"] = "\n".join(thinking_parts)
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result

        return {"role": "assistant", "content": str(content) if content else None}

    @staticmethod
    def _extract_text_content(raw_content: object) -> str:
        """Extract a non-whitespace text string from Chat Completions content."""
        if isinstance(raw_content, str):
            return raw_content if raw_content.strip() else ""
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_val = part.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        text_parts.append(text_val)
            return "\n".join(text_parts)
        return ""

    @staticmethod
    def _fallback_assistant_text(message: dict | None = None, *, context: dict | None = None) -> str:
        """Return an actionable fallback when upstream emits empty assistant output."""
        if isinstance(message, dict):
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                return refusal
        if not context:
            return _EMPTY_ASSISTANT_FALLBACK_TEXT

        upstream_error = context.get("upstream_error")
        if upstream_error:
            parts: list[str] = [str(upstream_error)]
        else:
            parts = [_EMPTY_ASSISTANT_FALLBACK_TEXT]

        provider = context.get("provider")
        model = context.get("model")
        attempts = context.get("attempts")
        retry_after = context.get("retry_after")
        if provider or model or attempts is not None:
            meta = []
            if provider:
                meta.append(str(provider))
            if model:
                meta.append(str(model))
            if attempts is not None:
                meta.append(f"after {attempts} attempts")
            parts.append(f"({', '.join(meta)})" if meta else "")
        if retry_after is not None:
            parts.append(f"Retry in ~{retry_after}s.")
        return " ".join(p for p in parts if p)

    def _emit_message_start_if_needed(self, events: list[str], message_id: str, model: str) -> None:
        """Emit `message_start` once per streaming response."""
        if self._message_started:
            return
        message_obj: dict = {
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
        self._message_started = True

    # ── Response translation (sync) ──────────────────────────────────────

    def translate_response(self, cc_response: dict, *, context: dict | None = None) -> dict:
        """Convert a Chat Completions response to a Messages API response."""
        choices = cc_response.get("choices") or [{}]
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        content: list[dict] = []

        # reasoning_content -> thinking block (must come before text)
        reasoning = message.get("reasoning_content")
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})

        # Text content -> text block
        text = self._extract_text_content(message.get("content"))
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

        # Defensive fallback: never emit thinking-only or empty assistant output.
        has_text = any(b.get("type") == "text" for b in content)
        if not has_text and not tool_calls:
            content.append({"type": "text", "text": self._fallback_assistant_text(message, context=context)})
            self._last_was_empty = True
        else:
            self._last_was_empty = False

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
        # Detect new message stream — reset _finished so the translator
        # can be reused across requests with different message IDs.
        if message_id != self._last_message_id:
            self._finished = False
            self._last_message_id = message_id

        events: list[str] = []
        choices = chunk.get("choices", [])
        if not choices:
            return events
        choice = choices[0]
        delta = choice.get("delta", {})

        # Reasoning delta
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            if not self._thinking_block_opened:
                self._emit_message_start_if_needed(events, message_id, model)

                events.append(
                    format_content_block_start_event(
                        self._content_block_index,
                        {"type": "thinking", "thinking": ""},
                    )
                )
                self._thinking_block_opened = True

            events.append(
                format_content_block_delta_event(
                    self._content_block_index,
                    {"type": "thinking_delta", "thinking": reasoning_content},
                )
            )

        # Text delta
        text_content = delta.get("content")
        if text_content:
            # Close thinking block if still open
            if self._thinking_block_opened:
                events.append(format_content_block_stop_event(self._content_block_index))
                self._content_block_index += 1
                self._thinking_block_opened = False

            # Open text block on first text delta
            if not self._text_block_opened:
                self._emit_message_start_if_needed(events, message_id, model)

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
                    # Close thinking block if still open
                    if self._thinking_block_opened:
                        events.append(format_content_block_stop_event(self._content_block_index))
                        self._content_block_index += 1
                        self._thinking_block_opened = False

                    # Close text block if still open
                    if self._text_block_opened:
                        events.append(format_content_block_stop_event(self._content_block_index))
                        self._content_block_index += 1
                        self._text_block_opened = False

                    self._emit_message_start_if_needed(events, message_id, model)

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
        if finish_reason is not None and not self._finished:
            had_thinking = self._thinking_block_opened
            had_text_or_tools = self._text_block_opened or bool(self._tool_call_buffers)
            had_any_content = had_thinking or had_text_or_tools or self._content_block_index > 0
            if not had_any_content:
                fallback_text = self._fallback_assistant_text()
                self._emit_message_start_if_needed(events, message_id, model)
                events.append(
                    format_content_block_start_event(
                        self._content_block_index,
                        {"type": "text", "text": ""},
                    )
                )
                events.append(
                    format_content_block_delta_event(
                        self._content_block_index,
                        {"type": "text_delta", "text": fallback_text},
                    )
                )
                events.append(format_content_block_stop_event(self._content_block_index))
                self._content_block_index += 1

            # Close thinking block if still open
            if self._thinking_block_opened:
                events.append(format_content_block_stop_event(self._content_block_index))
                self._content_block_index += 1
                self._thinking_block_opened = False

            # Close text block if still open
            if self._text_block_opened:
                events.append(format_content_block_stop_event(self._content_block_index))
                self._content_block_index += 1
                self._text_block_opened = False

            # Close any tool_use blocks
            for idx, _buf in self._tool_call_buffers.items():
                meta = self._tool_call_meta[idx]
                events.append(format_content_block_stop_event(meta["block_index"]))

            # If only thinking was emitted (no text, no tool calls), add fallback text
            if had_thinking and not had_text_or_tools:
                fallback_text = self._fallback_assistant_text()
                events.append(
                    format_content_block_start_event(
                        self._content_block_index,
                        {"type": "text", "text": ""},
                    )
                )
                events.append(
                    format_content_block_delta_event(
                        self._content_block_index,
                        {"type": "text_delta", "text": fallback_text},
                    )
                )
                events.append(format_content_block_stop_event(self._content_block_index))
                self._content_block_index += 1

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

            # Mark as finished to guard against duplicate finish_reason chunks
            # (some models/providers emit a second empty finish chunk after reset).
            # Set AFTER reset() so the flag survives the state cleanup.
            self._finished = True
            self._last_was_empty = not had_any_content

        return events
