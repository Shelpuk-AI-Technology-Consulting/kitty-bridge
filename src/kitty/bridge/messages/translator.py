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

__all__ = ["MessagesTranslator", "build_user_content_message", "carry_tool_choice_and_metadata"]

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


def build_user_content_message(blocks: list, documents_out: list[dict]) -> dict:
    """Build one CC user message from non-``tool_result`` Messages blocks.

    The shared body of the two Messages→CC converters (:meth:`MessagesTranslator.
    translate_request` and ``server._convert_native_to_cc_format``) — a second
    copy of the value table is the drift that lost KBR-178's field on the
    fallback's retry path. Text-only input keeps the pre-KBR-222 output — a
    joined string — so the common turn's CC body does not change shape. Any
    ``image`` block switches the message to a content-parts list (``text``
    plus ``image_url`` parts), because a string would lose the image again one
    hop later. ``document`` blocks go to *documents_out*, never into the CC
    content.

    Args:
        blocks: The user message's non-``tool_result`` content blocks.
        documents_out: Collector for ``document`` blocks; each entry is
            addressed to the message dict this function builds.

    Returns:
        The CC user message dict.
    """
    # Text blocks join as today; images become CC parts; documents ride the
    # internal key. Block types neither branch handles are dropped, as before
    # this fix — nothing claims them.
    parts: list[dict] = []
    documents: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                url = f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
            elif source.get("type") == "url":
                url = source.get("url", "")
            else:
                # KBR-222: a source type hop 1 cannot express (Anthropic's
                # `file`, or a malformed one) keeps today's drop — an
                # empty-URL part would corrupt the reference and buy an
                # opaque upstream 400.
                continue
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "document":
            documents.append(block)

    message: dict = {
        "role": "user",
        "content": "\n".join(p["text"] for p in parts if p["type"] == "text") if parts else "",
    }
    if any(p["type"] != "text" for p in parts):
        message["content"] = parts
    if documents:
        documents_out.append({"message": message, "blocks": documents})
    return message


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

        # System prompt -> system message.  The original value is remembered
        # verbatim — blocks and cache breakpoints included — and rides the
        # internal ``_anthropic_system`` key so the Anthropic adapters can
        # restore what the thinking signatures are bound to instead of this
        # joined string (KBR-228 part B); every other wire strips the key.
        carried_system = messages_request.get("system")
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

        # Messages with content block handling.  KBR-222: ``documents``
        # collects every ``document`` block across the conversation, each
        # addressed to the CC message dict it belongs to; the list is filled
        # by _translate_user_message and travels on the ``_documents``
        # internal key, which only Anthropic-family adapters restore.
        documents: list[dict] = []
        for msg in messages_request.get("messages", []):
            translated = self._translate_message(msg, documents)
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

        # KBR-222: Chat Completions has no slot for an Anthropic ``document``
        # block, and shipping Anthropic part spelling on the CC wire would
        # 400 turns that previously only lost the document.  It travels as
        # internal metadata instead, like ``_top_k``; ``_INTERNAL_KEYS`` keeps
        # it off the wire of every adapter that does not restore it.
        if documents:
            result["_documents"] = documents

        # KBR-228 part B: the verbatim system carriage, set here where the
        # joined form above has already been written into ``messages``.
        if carried_system:
            result["_anthropic_system"] = carried_system

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

        # KBR-224: `output_config` is Anthropic's documented spelling of the
        # effort control (and the home of structured output).  Chat Completions
        # has no slot for it, so — like `_effort` above — it rides an internal
        # key to the Anthropic-family adapters, which restore it where the
        # upstream documents the field.  Carried verbatim: an undocumented
        # member is the agent's mistake, and the upstream that documents the
        # field reports that better than a silent repair would.
        if messages_request.get("output_config") is not None:
            result["_output_config"] = messages_request["output_config"]

        # Extract thinking config into normalized effort metadata
        thinking = messages_request.get("thinking")
        if thinking and isinstance(thinking, dict):
            if not self._thinking_warned:
                self._thinking_warned = True
            if thinking.get("type") == "enabled":
                result["_thinking_enabled"] = True
                result["_reasoning_effort"] = "high"
                # KBR-225: carry the agent's own budget so AnthropicAdapter can
                # ship it verbatim instead of deriving one from max_tokens.
                # Only a valid budget rides the key -- an int, at least 1024,
                # and strictly below max_tokens (Anthropic's constraint) --
                # because the adapter trusts the key and falls back when it is
                # absent.  max_tokens must itself be an int: comparing against
                # a non-int would move the malformed-input TypeError from the
                # Anthropic-family adapter into this shared translator.  A bool
                # budget is excluded by the floor (every bool is 0 or 1).
                budget = thinking.get("budget_tokens")
                max_tokens = messages_request.get("max_tokens")
                if isinstance(budget, int) and isinstance(max_tokens, int) and 1024 <= budget < max_tokens:
                    result["_thinking_budget_tokens"] = budget
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

    def _translate_message(self, msg: dict, documents_out: list[dict]) -> dict | list[dict] | None:
        """Translate a single Messages API message to Chat Completions format.

        Args:
            msg: The inbound Messages API message dict.
            documents_out: Collector for ``document`` blocks found in user
                content (KBR-222); each entry is addressed to the CC message
                dict it belongs to. Not modified for non-user messages.

        Returns:
            The translated message dict. For user messages with multiple
            tool_result blocks, the caller should use ``_translate_messages`` instead.
        """
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            return self._translate_user_message(content, documents_out)
        if role == "assistant":
            return self._translate_assistant_message(content)

        # Fallback for simple string content
        if isinstance(content, str):
            return {"role": role, "content": content}

        return {"role": role, "content": str(content) if content else ""}

    def _translate_user_message(self, content, documents_out: list[dict]) -> dict | list[dict]:
        """Translate a user message, handling tool_result content blocks.

        Non-tool blocks keep more than their text since KBR-222: an ``image``
        block becomes a CC ``image_url`` content part, and a ``document``
        block — for which Chat Completions has no slot — is appended to
        *documents_out* as ``{"message": <the CC user message it belongs
        to>, "blocks": [<the Anthropic document block, verbatim>]}``. The
        address is the message's identity, not its position: compaction drops
        whole messages, so an index would attach a document to the wrong turn,
        while an identity either matches or forfeits the document.

        Args:
            content: The inbound user message content (string or block list).
            documents_out: Collector for ``document`` blocks, shared across
                the whole request so ``translate_request`` can publish them
                on the ``_documents`` internal key.

        Returns:
            A single message dict for simple content, or a list of message
            dicts when ``tool_result`` blocks need separate ``tool`` role
            messages in Chat Completions format. Non-tool blocks beside a
            tool_result become one trailing user message after the tool
            messages, instead of being dropped.
        """
        if isinstance(content, str):
            return {"role": "user", "content": content}

        if isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            others = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]
            if tool_results:
                tool_msgs = [
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr.get("content", ""),
                    }
                    for tr in tool_results
                ]
                if not others:
                    return tool_msgs[0] if len(tool_msgs) == 1 else tool_msgs
                # KBR-222: the sibling blocks used to die here. They become a
                # trailing user message; a text-first turn is reordered after
                # the tool results, which only the register prose claims.
                return [*tool_msgs, self._user_content_message(others, documents_out)]

            return self._user_content_message(others, documents_out)

        return {"role": "user", "content": str(content) if content else ""}

    def _user_content_message(self, blocks: list, documents_out: list[dict]) -> dict:
        """Build one CC user message from non-``tool_result`` blocks.

        Delegates to :func:`build_user_content_message` — the shared body of
        both Messages→CC converters, so the fallback's retry cannot re-drop
        what hop 1 carries (KBR-178's lesson).

        Args:
            blocks: The user message's non-``tool_result`` content blocks.
            documents_out: Collector for ``document`` blocks; each entry is
                addressed to the message dict this call builds.

        Returns:
            The CC user message dict.
        """
        return build_user_content_message(blocks, documents_out)

    def _translate_assistant_message(self, content) -> dict:
        """Translate an assistant message, handling tool_use and thinking content blocks."""
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        if isinstance(content, list):
            text_parts = []
            tool_calls = []
            thinking_parts = []
            # KBR-228 part B: the signed originals ride the message verbatim —
            # signatures and redacted_thinking included, wire order preserved —
            # so the Anthropic adapters can restore what their upstream
            # signature-binds.  Every other wire strips the key.
            thinking_blocks = [
                dict(block)
                for block in content
                if isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking")
            ]
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
            if thinking_blocks:
                result["_thinking_blocks"] = thinking_blocks
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

        # KBR-228 part A: an Anthropic-family upstream's reply carries its
        # thinking blocks verbatim under ``_thinking_blocks``; they are emitted
        # as-is, signatures included, in wire order.  Thinking always precedes
        # text and tool_use on that wire, so prepending them preserves the
        # order the upstream produced.  The carriage wins over
        # ``reasoning_content``, which mirrors the same text and would
        # duplicate it as an unsigned block.
        carried = message.get("_thinking_blocks")
        if isinstance(carried, list) and carried:
            content.extend(dict(block) for block in carried if isinstance(block, dict))
        else:
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

        # KBR-285: the deprecated single-dict ``function_call`` maps to one
        # tool_use block, the same shape the ``tool_calls`` loop emits. The
        # truthy-dict gate mirrors _is_empty_cc_response's clause exactly, so
        # a malformed non-dict value cannot suppress the fallback.
        function_call = message.get("function_call")
        has_function_call = isinstance(function_call, dict) and bool(function_call)
        if has_function_call:
            try:
                fc_input = json.loads(function_call.get("arguments", "{}"))
            except json.JSONDecodeError:
                fc_input = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": function_call.get("name", ""),
                    "input": fc_input,
                }
            )

        # Defensive fallback: never emit thinking-only or empty assistant output.
        has_text = any(b.get("type") == "text" for b in content)
        if not has_text and not tool_calls and not has_function_call:
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

        # Text delta. KBR-285: a raw-CC upstream may deliver content as a list of
        # multimodal parts — coerce it to the string the Messages wire carries —
        # and a refusal-only delta carries the model's user-facing reply on
        # ``refusal`` with ``content`` null, which becomes text too. A parts
        # list with no text element coerces to "" and emits no delta: the
        # Messages wire has no image-delta equivalent (a raw-CC client still
        # receives the parts verbatim).
        text_content = delta.get("content")
        if isinstance(text_content, list):
            text_content = self._extract_text_content(text_content)
        if not text_content:
            refusal = delta.get("refusal")
            if isinstance(refusal, str) and refusal:
                text_content = refusal
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

        # Tool call delta. KBR-285: the deprecated single-dict ``function_call``
        # maps onto the same machinery — the opening delta synthesises the
        # id/index and carries the name, later deltas argument-append. The
        # existing ``tool_calls`` branch handles both shapes unchanged. No
        # real upstream carries both fields; if one did, a ``tool_calls``
        # list wins and a later legacy delta appends to the index-0 buffer
        # that call opened — the least-bad merge, recorded here so the
        # precondition is explicit.
        tool_calls = delta.get("tool_calls")
        if not tool_calls:
            legacy_call = delta.get("function_call")
            if isinstance(legacy_call, dict) and legacy_call:
                if 0 in self._tool_call_meta:
                    tool_calls = [
                        {
                            "index": 0,
                            "function": {"arguments": legacy_call.get("arguments", "")},
                        }
                    ]
                else:
                    tool_calls = [
                        {
                            "index": 0,
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": legacy_call.get("name", ""),
                                "arguments": legacy_call.get("arguments", ""),
                            },
                        }
                    ]
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
                    # Advance past the opened block (KBR-226) so a parallel call
                    # or a following text block opens the next free index.
                    self._content_block_index += 1

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
