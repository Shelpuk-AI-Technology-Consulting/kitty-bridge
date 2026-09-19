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

__all__ = [
    "InvalidResponsesRequest",
    "ResponsesTranslator",
    "carry_responses_tool_choice",
    "normalize_responses_request",
]

# MiniMax interleaved thinking tags: <اخل>...</اخل>
_THINKING_TAG_RE = re.compile(r"<\u0627\u062e\u0644>.*?</\u0627\u062e\u0644>", re.DOTALL)
_EMPTY_ASSISTANT_FALLBACK_TEXT = (
    "Upstream model returned an empty response. Please retry. "
    "If the context is full, use /clear to reset the conversation."
)

#: Chat Completions ``tool_choice`` string values the Responses wire publishes
#: with the same spelling (KBR-221 R1).
_RESPONSES_SIMPLE_TOOL_CHOICES: frozenset[str] = frozenset({"auto", "none", "required"})


def _names_degraded_responses_tool(tools: object, name: str) -> bool:
    """Report whether the inbound Responses ``tools`` list declares ``name`` as a tool the hop degraded.

    A function-typed ``tool_choice`` is only forceable onto a tool the route
    still declares as a function.  ``type: "custom"`` freeform tools and the
    hosted ``ToolChoiceTypes`` built-ins (``web_search_preview`` and the rest,
    enumerated in :mod:`kitty.bridge.responses.translator` prose and in the
    ``TestResponsesToolChoice.test_tool_choice_hosted_is_omitted`` parametrize)
    carry no Chat Completions form on this hop -- forcing a call to one would
    force a call nothing on the route can execute (KBR-221 D10).  This helper
    tests the simple property ``type != "function"`` because it catches every
    non-function declaration -- hosted types, ``custom``, anything malformed
    the wire may yet publish -- without enumerating the closed set in two
    places that would have to stay in sync.

    A name the list does not declare is **not** reported: that body is the
    agent's mistake, and the provider's error says so better than a silent
    omission would (KBR-221 D8 -- the KBR-214 precedent carries undeclared
    names so the provider's 400 names the mistake).

    Args:
        tools: The inbound Responses ``tools`` list.  A non-list value is
            treated as empty.
        name: The tool name a ``tool_choice`` of type ``function`` selects.

    Returns:
        True when a declaration with that name is present and carries a
        non-function ``type``; False otherwise.
    """
    if not isinstance(tools, list):
        return False
    return any(
        isinstance(tool, dict) and tool.get("name") == name and tool.get("type") != "function"
        for tool in tools
    )


def carry_responses_tool_choice(responses_request: dict, cc_request: dict) -> None:
    """Carry a Responses ``tool_choice`` and ``parallel_tool_calls`` onto a Chat Completions body.

    Called from :meth:`ResponsesTranslator.translate_request`, the way KBR-214's
    :func:`kitty.bridge.messages.translator.carry_tool_choice_and_metadata` is
    called from the Messages converter.  ``tool_choice`` is a constraint, not a
    hint: dropping ``"required"`` lets the model answer in prose where the agent
    demanded a tool call, and dropping ``"none"`` lets it call a tool the agent
    forbade (KBR-221).

    Most Responses choices share the Chat Completions spelling.  The named
    function form moves ``name`` under ``function``; ``allowed_tools`` carries
    by its ``mode`` because the mode is the only part the canonical vocabulary
    models.  The omissions mirror the KBR-214 decisions:

    * **No tools, no choice** (D9).  The gate reads the *Chat Completions*
      tool list this body will ship, not the inbound list: a Responses
      ``tools`` list of only hosted entries filters to an empty CC list, and
      OpenAI rejects a choice beside no tools ("'tool_choice' is only allowed
      when 'tools' are specified").  The gate scopes to the ``tool_choice``
      carry alone -- the parallel knob is a standalone wire field and ships
      regardless.
    * **A forced call to a degraded tool is not carried** (D10).  Hosted,
      MCP and ``custom`` tools are flattened away on this hop; forcing one
      would force a call nothing on the route can execute.  The named-tool
      lookup walks the inbound ``tools`` list, the only place the degraded
      entries still live.  A choice naming an **undeclared** tool is carried
      -- the agent's mistake, and the provider's error names it (D8).
    * **Values with no Chat Completions form are omitted, not repaired**
      (D5).  The eleven hosted types, ``mcp``, and any shape that is neither
      a published string nor a published object are left out; kitty has no
      authority to invent a reading.

    ``parallel_tool_calls`` is forwarded only when the wire carries an
    explicit ``False`` (D2).  ``True`` is the documented default on both wires,
    so writing it would add a field whose behaviour it does not change; a
    non-boolean value would residualise upstream and manufacture an unclaimed
    delta (D5).  Unlike the Anthropic Messages knob -- which nests inside
    ``tool_choice.disable_parallel_tool_use`` and so is structurally tied to
    the choice -- Responses carries ``parallel_tool_calls`` as a standalone
    top-level boolean, and the carry is **independent of the D9 gate**: an
    inbound ``false`` is forwarded even when no tools are present, so an agent
    that sends the knob without tools reaches the backbone with the
    instruction intact.  This matches R2's unconditional mapping table.

    Args:
        responses_request: The inbound OpenAI Responses body.  Not modified.
        cc_request: The Chat Completions body being built, mutated in place.

    Returns:
        None.  ``cc_request`` gains ``tool_choice`` and ``parallel_tool_calls``
        only where the inbound body carries a representable value for them.
    """
    # The two carries are independent: D9 gates only ``tool_choice``; the
    # parallel knob ships even when no tools are present.

    # D2 / D5: forward only the explicit non-default ``False``; ``True`` is the
    # default on both wires and a non-bool value would residualise upstream.
    # Done before the D9 gate -- a Responses ``false`` is the wire's
    # standalone instruction, not a child of the choice.
    if responses_request.get("parallel_tool_calls") is False:
        cc_request["parallel_tool_calls"] = False

    # D9: the gate reads the CC list the body will ship.  An inbound list of
    # only hosted entries filters to an empty CC list, and an absent key means
    # no tools at all -- both cases omit the choice.  Applied only to the
    # ``tool_choice`` carry; the parallel knob was already handled above.
    cc_tools = cc_request.get("tools")
    if not isinstance(cc_tools, list) or not cc_tools:
        return

    choice = responses_request.get("tool_choice")
    if isinstance(choice, str):
        if choice in _RESPONSES_SIMPLE_TOOL_CHOICES:
            cc_request["tool_choice"] = choice
    elif isinstance(choice, dict):
        kind = choice.get("type")
        name = choice.get("name")
        if kind == "function":
            # D10: a function-typed choice naming a tool the hop degraded is
            # omitted.  The lookup walks the inbound list, the only place the
            # degraded entry still lives -- the CC list has already filtered
            # it out.  Undeclared names fall through to the carry (D8).
            if isinstance(name, str) and not _names_degraded_responses_tool(
                responses_request.get("tools"), name
            ):
                cc_request["tool_choice"] = {"type": "function", "function": {"name": name}}
        elif kind == "allowed_tools":
            mode = choice.get("mode")
            # Only the mode has a canonical home (the reader projects just the
            # mode); non-published or non-string modes are omitted (D5).
            if isinstance(mode, str) and mode in _RESPONSES_SIMPLE_TOOL_CHOICES:
                cc_request["tool_choice"] = mode
        # Hosted types, ``mcp`` and ``custom`` fall through -- no CC form
        # (D5), and the degraded-tool rule (D10) keeps a named reference to
        # them from riding along either.


def _empty_assistant_fallback_text(context: dict | None = None) -> str:
    if not context:
        return _EMPTY_ASSISTANT_FALLBACK_TEXT
    provider = context.get("provider")
    model = context.get("model")
    attempts = context.get("attempts")
    retry_after = context.get("retry_after")
    parts = [_EMPTY_ASSISTANT_FALLBACK_TEXT]
    meta = []
    if provider:
        meta.append(str(provider))
    if model:
        meta.append(str(model))
    if attempts is not None:
        meta.append(f"after {attempts} attempts")
    if meta:
        parts.append(f"({', '.join(meta)})")
    if retry_after is not None:
        parts.append(f"Retry in ~{retry_after}s.")
    return " ".join(parts)


def _strip_thinking_tags(text: str) -> str:
    """Strip MiniMax-style interleaved thinking tags from content."""
    return _THINKING_TAG_RE.sub("", text).strip()


def _extract_text_parts(content: list) -> str:
    """Extract a joined text string from a multimodal parts list.

    KBR-285: newer multimodal streaming on OpenAI-shaped backends carries
    ``content`` as a list of content parts. Only text parts carry a string
    the Responses wire can hold; image parts have no output equivalent and
    are dropped (the raw-CC route delivers them verbatim).

    Args:
        content: The parts list from ``delta.content`` / ``message.content``.

    Returns:
        The joined text of the list's text parts (empty string when the
        list carries no text part).
    """
    text_parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_val = part.get("text")
            if isinstance(text_val, str) and text_val.strip():
                text_parts.append(text_val)
    return "\n".join(text_parts)


class InvalidResponsesRequest(ValueError):
    """Raised when a Responses request carries a shape the dialect does not permit.

    Carried out of :func:`normalize_responses_request` so
    :meth:`kitty.bridge.server.BridgeServer._handle_responses` can answer with the
    endpoint's 400 envelope.  Without a distinct type the handler's catch-all
    renders every one of these as a 500, which
    ``.system_design/TEST_SUITE.md`` §6.2.1 forbids: the bridge must never
    return a server error for a client's malformed body.
    """


def normalize_responses_request(body: object) -> dict:
    """Return a Responses request with ``input`` in its array form

    OpenAI's ``CreateResponse`` schema defines ``input`` as ``oneOf`` a string
    -- *"a text input to the model, equivalent to a text input with the*
    ``user`` *role"* -- or an array of input items.  Everything downstream reads
    the array, so the string is converted here, once, and the two forms become
    one request.

    **Call this before the inbound body forks.** ``.system_design/TEST_SUITE.md``
    §3.2.3 records that two upstream bodies are built from one Responses
    request: the Chat Completions body, from :meth:`ResponsesTranslator.translate_request`,
    and -- on ``openai_subscription`` -- a Responses body built inside the
    transport from the *raw* inbound dict.  Normalising only in the translator
    leaves the second path iterating the string character by character and
    shipping the user's text as a list of its own letters (KBR-144).

    The rewrite is **register row M15** (§3.2.1).  It takes §3.3.1a's
    ``not projectable`` escape, because both forms project to the same
    wire-independent conversation -- one user turn carrying the text -- which is
    the reasoning §3.3.1a applies to row P16.  The row exists rather than being
    omitted because the rewrite is real bytes on the ``curl_cffi`` boundary, and
    it binds the future OpenAI-Responses reader to read the two forms alike.

    The shape checks below are **scoped to the fields the translator actually
    reads containers and members off**.  Each one exists because a measured
    body reached the translator and crashed inside it (KBR-159): every
    unguarded container here is an unhandled-exception 500 the
    ``.system_design/TEST_SUITE.md`` §6.2.1 ``not_a_server_error`` check
    forbids.  Fields the translator *tolerates* are deliberately **not**
    validated -- ``instructions`` as an object, ``reasoning`` as a string and
    a bare-number message ``content`` all pass through today, and rejecting
    them would refuse bodies real clients legitimately send (KBR-82's
    "why publish a schema" paragraph).

    The ``function_call_output`` shape with missing ``call_id`` is **not**
    validated here -- KBR-169's orphan-drop pass (``_drop_orphan_response_outputs``
    at ``src/kitty/bridge/server.py:388``) handles that shape by silently
    dropping the unpaired item and answering 200.  Validating it here would
    duplicate KBR-169's job and break its contract -- KBR-169 deliberately
    treats a missing ``call_id`` as "undeclared" (its docstring).

    Args:
        body: The decoded inbound request body.  Typed ``object`` rather than
            ``dict`` because this is a trust boundary: the caller has decoded
            arbitrary JSON, and annotating ``dict`` would make the guard below
            look unreachable to a type checker.

    Returns:
        A body whose ``input``, if present, is a list of input-item objects.
        The argument is never modified; an already-normalised body is returned
        unchanged, so calling this twice is safe.

    Raises:
        InvalidResponsesRequest: The body is not a JSON object; or ``input`` is
            neither a string nor an array of objects; or one of the containers
            the translator iterates (``tools``, a reasoning item's ``summary``)
            carries the wrong shape.
    """
    # A field can only be read off an object; valid JSON is a weaker claim.
    if not isinstance(body, dict):
        raise InvalidResponsesRequest(f"Request body must be a JSON object, got {type(body).__name__}")

    # `tools` is iterated and its members have `.get` called on them in the
    # translator; a non-list here iterates its keys (a dict) or its characters
    # (a string) and crashes.  Function tools must carry a name and a
    # parameters object; every other tool kind (`web_search`, `custom`, MCP,
    # ...) is skipped by the translator on purpose and must stay unvalidated.
    if "tools" in body:
        tools = body["tools"]
        if not isinstance(tools, list):
            raise InvalidResponsesRequest(f"'tools' must be an array, got {type(tools).__name__}")
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise InvalidResponsesRequest(f"'tools[{index}]' must be an object, got {type(tool).__name__}")
            if tool.get("type") == "function":
                name = tool.get("name")
                if not isinstance(name, str) or not name:
                    raise InvalidResponsesRequest(f"'tools[{index}].name' must be a non-empty string")
                if not isinstance(tool.get("parameters"), dict):
                    raise InvalidResponsesRequest(f"'tools[{index}].parameters' must be an object")

    if "input" not in body:
        return body

    value = body["input"]
    # The spec's own equivalence, applied verbatim.
    if isinstance(value, str):
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": value}]}
        return {**body, "input": [item]}

    if not isinstance(value, list):
        raise InvalidResponsesRequest(f"'input' must be a string or an array, got {type(value).__name__}")

    _validate_input_items(value)
    return body


def _validate_input_items(value: list) -> None:
    """Validate the shapes inside ``input`` that the translator reads members off.

    A reasoning item's ``summary`` must be a list of objects carrying a string
    ``text``: the translator iterates it and joins the texts, so both a string
    ``summary`` and a non-string ``text`` crash it. The
    ``function_call_output`` shape is **not** validated here -- see the
    docstring of :func:`normalize_responses_request` for the KBR-169 boundary.

    Args:
        value: The ``input`` field's array form, after the string form has
            been rewritten.

    Raises:
        InvalidResponsesRequest: One of the container shapes above is wrong.
            Items are reported by index, because a client sending a long
            transcript needs to know which one.
    """
    for index, element in enumerate(value):
        if not isinstance(element, dict):
            raise InvalidResponsesRequest(f"'input[{index}]' must be an object, got {type(element).__name__}")

        if element.get("type") == "reasoning":
            summary = element.get("summary", [])
            if not isinstance(summary, list):
                raise InvalidResponsesRequest(
                    f"'input[{index}].summary' must be an array, got {type(summary).__name__}"
                )
            for position, entry in enumerate(summary):
                if not isinstance(entry, dict):
                    raise InvalidResponsesRequest(
                        f"'input[{index}].summary[{position}]' must be an object, "
                        f"got {type(entry).__name__}"
                    )
                if not isinstance(entry.get("text"), str):
                    raise InvalidResponsesRequest(
                        f"'input[{index}].summary[{position}].text' must be a string"
                    )


class ResponsesTranslator:
    """Translates between Responses API and Chat Completions formats."""

    def __init__(self) -> None:
        self._tool_call_buffers: dict[int, ToolCallBuffer] = {}
        # Keyed by the CC tool-call index (a routing key); "output_index" is
        # the Responses slot allocated for the item, never the CC index.
        self._tool_call_meta: dict[int, dict] = {}  # index -> {call_id, name, item_id, output_index}
        self._accumulated_text: str = ""
        self._accumulated_reasoning: str = ""
        self._seq: int = 0
        # One counter positions every output item (reasoning, text, calls):
        # the Responses grammar's output_index, allocated when an item opens.
        self._next_output_index: int = 0
        self._text_output_index: int = 0
        self._reasoning_output_index: int = 0
        # Empty string means "not started"; every read is a truthiness test.
        self._text_item_id: str = ""
        self._reasoning_item_id: str = ""
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
        self._next_output_index = 0
        self._text_output_index = 0
        self._reasoning_output_index = 0
        self._text_item_id = ""
        self._reasoning_item_id = ""
        self._text_started = False
        self._reasoning_started = False
        self._last_was_empty = False

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _allocate_output_index(self) -> int:
        """Allocate the next ``output_index`` for a newly opened output item.

        One counter positions every output item the translator opens. The
        upstream Chat Completions tool-call index only routes argument deltas
        to their buffer; it never becomes a Responses ``output_index``.
        """
        index = self._next_output_index
        self._next_output_index += 1
        return index

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

    def walk_input_items(self, responses_request: dict) -> tuple[list[dict], list[int | None]]:
        """Translate input items to CC messages, recording each item's owner.

        This is :meth:`translate_request`'s conversation loop — the single
        implementation of it — extended to also report, for every **input
        item**, the index of the translated message that carries it (KBR-169).
        Riding items — ``reasoning`` items and types the translation skips —
        are owned by the next **assistant** message, exactly where this loop
        attaches accumulated reasoning; a riding item with no following
        assistant message has no owner and stays ``None``. Owner indices are
        stable through the system-message merge: an item whose message was
        merged into the previous system message is owned by the survivor.

        Args:
            responses_request: A normalized Responses API request.

        Returns:
            A ``(messages, item_owners)`` pair. ``messages`` is exactly what
            :meth:`translate_request` builds; ``item_owners`` has one entry
            per input item.
        """
        messages: list[dict] = []
        item_owners: list[int | None] = []
        # System instructions -> system message
        instructions = responses_request.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Input items -> messages
        # Accumulate reasoning from 'reasoning' items and merge into the next
        # assistant message's reasoning_content field.
        pending_reasoning: list[str] = []
        # Input-item indices awaiting the assistant message that will own them.
        riding: list[int] = []
        for item in responses_request.get("input", []):
            item_index = len(item_owners)
            item_owners.append(None)
            # Extract reasoning text from reasoning items
            if item.get("type") == "reasoning":
                for summary in item.get("summary", []):
                    if summary.get("type") == "summary_text" and summary.get("text"):
                        pending_reasoning.append(summary["text"])
                riding.append(item_index)
                continue

            msg = self._translate_input_item(item)
            if msg is None:
                riding.append(item_index)
                continue
            # Attach accumulated reasoning to assistant messages
            if msg.get("role") == "assistant" and pending_reasoning:
                msg["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning = []
            messages.append(msg)
            item_owners[item_index] = len(messages) - 1
            if msg.get("role") == "assistant":
                for rider in riding:
                    item_owners[rider] = len(messages) - 1
                riding = []

        # Merge consecutive system messages (some providers reject multiples),
        # remapping owner indices onto the merged list.
        merged: list[dict] = []
        remap: list[int] = []
        for msg in messages:
            if msg.get("role") == "system" and merged and merged[-1].get("role") == "system":
                prev = merged[-1].get("content") or ""
                curr = msg.get("content") or ""
                merged[-1]["content"] = f"{prev}\n\n{curr}"
                remap.append(len(merged) - 1)
            else:
                merged.append(msg)
                remap.append(len(merged) - 1)
        messages = merged
        item_owners = [remap[o] if o is not None else None for o in item_owners]
        return messages, item_owners

    def translate_request(self, responses_request: dict) -> dict:
        """Convert a Responses API request to a Chat Completions request."""
        # Idempotent, and the live caller has normalised already -- this is here so
        # the translator is correct for any caller, not only the handler.
        responses_request = normalize_responses_request(responses_request)
        messages, _item_owners = self.walk_input_items(responses_request)

        result: dict = {
            # `CreateResponse` declares no required fields, and register row M1
            # replaces this with the profile's model anyway.
            "model": responses_request.get("model", ""),
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

        # Extract reasoning effort
        reasoning = responses_request.get("reasoning")
        if isinstance(reasoning, dict) and "effort" in reasoning:
            effort = reasoning["effort"]
            result["_reasoning_effort"] = effort
            result["_thinking_enabled"] = effort != "none"

        # KBR-221: carry the agent's tool_choice and parallel_tool_calls.
        carry_responses_tool_choice(responses_request, result)

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
            text_parts = [p for p in parts if isinstance(p, str)]
            if len(text_parts) == len(parts):
                return "\n".join(text_parts) if parts else ""
            return parts
        return content

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

    def translate_response(self, cc_response: dict, *, context: dict | None = None) -> dict:
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

        # Text content. KBR-285: a raw-CC upstream may deliver content as a
        # list of multimodal parts — coerce to the string the Responses wire
        # carries — and a refusal-only reply carries the model's reply on
        # ``refusal`` with ``content`` null, which becomes text too. A parts
        # list with no text element coerces to "" and emits no item: the
        # Responses wire has no image-delta equivalent.
        content = message.get("content")
        if isinstance(content, list):
            content = _extract_text_parts(content)
        if not content:
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal:
                content = refusal
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

        # KBR-285: the deprecated single-dict ``function_call`` maps to one
        # function_call item, the same shape the ``tool_calls`` loop emits.
        # No real upstream carries both; if one did, the ``tool_calls`` loop
        # already ran and the legacy path appends a second item — the least-
        # bad merge, recorded for the precondition.
        function_call = message.get("function_call")
        has_function_call = isinstance(function_call, dict) and bool(function_call)
        if has_function_call:
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": f"call_{uuid.uuid4().hex}",
                    "name": function_call.get("name", ""),
                    "arguments": str(function_call.get("arguments", "{}")),
                    "status": "completed",
                }
            )

        status = "completed"
        if finish_reason == "length":
            status = "incomplete"

        if not output:
            self._last_was_empty = True
            fallback = _empty_assistant_fallback_text(context)
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": fallback}],
                }
            )
        else:
            self._last_was_empty = False

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
        self._text_output_index = self._allocate_output_index()
        self._text_started = True

    def _ensure_reasoning_item_started(self, response_id: str) -> None:
        """Lazily initialize the reasoning item state."""
        if self._reasoning_started:
            return
        self._reasoning_item_id = f"rs_{uuid.uuid4().hex[:24]}"
        self._reasoning_output_index = self._allocate_output_index()
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
                        output_index=self._reasoning_output_index,
                        item={
                            "id": self._reasoning_item_id,
                            "type": "reasoning",
                            "summary": [],
                        },
                    )
                )
            self._accumulated_reasoning += reasoning_content

        # Text delta. KBR-285: a raw-CC upstream may deliver content as a list of
        # multimodal parts — coerce to the string the Responses wire carries —
        # and a refusal-only delta carries the model's reply on ``refusal``
        # with ``content`` null, which becomes text too. A parts list with no
        # text element coerces to "" and emits no delta: the Responses wire
        # has no image-delta equivalent.
        content = delta.get("content")
        if isinstance(content, list):
            content = _extract_text_parts(content)
        if not content:
            refusal = delta.get("refusal")
            if isinstance(refusal, str) and refusal:
                content = refusal
        if content:
            # Lazily start text item on first content
            if not self._text_started:
                self._ensure_text_item_started(response_id)
                # Emit output_item.added
                events.append(
                    format_output_item_added_event(
                        seq=self._next_seq(),
                        output_index=self._text_output_index,
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
                        output_index=self._text_output_index,
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
                    output_index=self._text_output_index,
                    content_index=0,
                    delta=content,
                )
            )

        # Tool call delta. KBR-285: the deprecated single-dict ``function_call``
        # maps onto the same machinery — the opening delta synthesises the
        # id/index and carries the name, later deltas argument-append. The
        # existing ``tool_calls`` branch handles both shapes unchanged. No
        # real upstream carries both fields; if one did, a ``tool_calls``
        # list wins and a later legacy delta appends to the index-0 buffer
        # that call opened.
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
                    call_id = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    item_id = f"fc_{uuid.uuid4().hex[:24]}"
                    # The Responses slot is allocated here; the CC index only
                    # routes later argument deltas to this meta entry.
                    fc_index = self._allocate_output_index()
                    self._tool_call_meta[idx] = {
                        "call_id": call_id,
                        "name": func.get("name", ""),
                        "item_id": item_id,
                        "output_index": fc_index,
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
                            output_index=fc_index,
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
                            output_index=meta["output_index"],
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
                    output_index=self._reasoning_output_index,
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
                    output_index=self._text_output_index,
                    content_index=0,
                    text=clean_text,
                )
            )
            # content_part.done
            events.append(
                format_content_part_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=self._text_output_index,
                    content_index=0,
                    part={"type": "output_text", "text": clean_text},
                )
            )
            # output_item.done for message
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=self._text_output_index,
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
                    output_index=meta["output_index"],
                    call_id=meta["call_id"],
                    arguments=final_args,
                )
            )

            # output_item.done for function call
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=meta["output_index"],
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

        # Build output items for the completed response, ordered by the
        # allocated output_index: opening order can differ from slot order.
        # Every opened item appears, so array position equals output_index.
        indexed_output: list[tuple[int, dict]] = []
        if self._reasoning_started and self._reasoning_item_id:
            indexed_output.append(
                (
                    self._reasoning_output_index,
                    {
                        "type": "reasoning",
                        "id": self._reasoning_item_id,
                        "summary": [{"type": "summary_text", "text": self._accumulated_reasoning}],
                    },
                )
            )
        if self._text_started and self._text_item_id:
            indexed_output.append(
                (
                    self._text_output_index,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": clean_text}],
                    },
                )
            )
        for idx in self._tool_call_buffers:
            meta = self._tool_call_meta[idx]
            final_args = finalized_args.get(idx, "{}")
            indexed_output.append(
                (
                    meta["output_index"],
                    {
                        "type": "function_call",
                        "id": meta["item_id"],
                        "call_id": meta["call_id"],
                        "name": meta["name"],
                        "arguments": final_args,
                        "status": "completed" if status == "completed" else "incomplete",
                    },
                )
            )
        output_items = [item for _, item in sorted(indexed_output, key=lambda pair: pair[0])]

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
        Returns an empty list only when no item was opened and nothing was
        accumulated (status completed), preventing duplicate completion after
        normal finish chunks; an item that opened — even one whose text later
        stripped to nothing — is closed here so no added event is left without
        its done.
        """
        clean_text = self._clean_text()
        # An item whose events already reached the client must be closed here,
        # even when its text stripped to nothing: an opened-but-never-closed
        # item is the defect this counter exists to prevent.
        if (
            status == "completed"
            and not clean_text
            and not self._tool_call_buffers
            and not self._accumulated_reasoning
            and not self._text_started
        ):
            return []

        events: list[str] = []

        # Close reasoning item if started
        if self._reasoning_started and self._reasoning_item_id:
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=self._reasoning_output_index,
                    item={
                        "type": "reasoning",
                        "id": self._reasoning_item_id,
                        "summary": [{"type": "summary_text", "text": self._accumulated_reasoning}],
                    },
                )
            )

        # Close text content
        if self._text_started and self._text_item_id:
            events.append(
                format_output_text_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=self._text_output_index,
                    content_index=0,
                    text=clean_text,
                )
            )
            events.append(
                format_content_part_done_event(
                    seq=self._next_seq(),
                    item_id=self._text_item_id,
                    output_index=self._text_output_index,
                    content_index=0,
                    part={"type": "output_text", "text": clean_text},
                )
            )
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=self._text_output_index,
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
                    output_index=meta["output_index"],
                    call_id=meta["call_id"],
                    arguments=final_args,
                )
            )
            events.append(
                format_output_item_done_event(
                    seq=self._next_seq(),
                    output_index=meta["output_index"],
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

        # Build response.completed's output, ordered by the allocated
        # output_index: opening order can differ from slot order. Every opened
        # item appears, so array position equals output_index throughout.
        indexed_output: list[tuple[int, dict]] = []
        if self._reasoning_started and self._reasoning_item_id:
            indexed_output.append(
                (
                    self._reasoning_output_index,
                    {
                        "type": "reasoning",
                        "id": self._reasoning_item_id,
                        "summary": [{"type": "summary_text", "text": self._accumulated_reasoning}],
                    },
                )
            )
        if self._text_started and self._text_item_id:
            indexed_output.append(
                (
                    self._text_output_index,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": clean_text}],
                    },
                )
            )
        for idx in self._tool_call_buffers:
            meta = self._tool_call_meta[idx]
            final_args = finalized_args.get(idx, "{}")
            indexed_output.append(
                (
                    meta["output_index"],
                    {
                        "type": "function_call",
                        "id": meta["item_id"],
                        "call_id": meta["call_id"],
                        "name": meta["name"],
                        "arguments": final_args,
                        "status": "completed" if status == "completed" else "incomplete",
                    },
                )
            )
        output_items = [item for _, item in sorted(indexed_output, key=lambda pair: pair[0])]

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
