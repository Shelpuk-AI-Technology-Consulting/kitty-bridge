"""Downstream SSE grammar state machines — T-G7 (KBR-83).

`.system_design/TEST_SUITE.md` §6.2.2 · plan task **T-G7**, requirements in
`.requirements/20260917T120913Z_sse_grammar_state_machine/REQUIREMENTS.md`.

The Anthropic streaming format is a grammar, not a schema: a malformed sequence
breaks Claude Code in ways a per-event schema check cannot see. This module is
a state machine over the byte stream **the bridge writes to the agent**: every
stream it produces must be a sentence in the receiving protocol's grammar, or
one of the documented bridge-controlled terminal shapes (a JSON error with no
stream; a closed-structure error terminal; the KBR-250 in-stream exhaustion
shapes).

**Import discipline.** This module imports nothing from ``src/kitty`` — the
same independence rule :mod:`harness.failures` observes. A grammar written in
terms of the bridge's own parsers would inherit their bugs and prove
self-consistency rather than conformance. The guard lives in
:mod:`harness.test_sse_grammar_falsification`.

**Independence note.** :class:`StreamProtocol` deliberately duplicates the four
members of ``harness.bridge.InboundProtocol`` rather than importing it:
``harness.bridge`` imports ``kitty`` (it constructs a real ``BridgeServer``),
and this module must stay outside that import graph. The bridge-driven test
module asserts the two enums agree, so a fifth protocol cannot silently miss a
grammar.

**Classification precedence.** Two rules can both apply to one stream, so the
order is fixed and documented in the requirements:

1. ``malformed`` — any structural violation anywhere in the stream;
2. ``truncated`` — at ``finish()``, any block/item still open (a trailing
   ``error`` event does not rescue an unclosed block — the native
   Messages-wire close-out leaves blocks open, and the client's view of that
   stream is genuinely incomplete);
3. ``complete_sentence`` — the grammar's closing event was reached;
4. ``error_terminal`` — the stream ended with the documented error event(s)
   and all opened structure is closed;
5. ``truncated`` — anything else.
"""

from __future__ import annotations

import codecs
import json
from enum import Enum
from typing import Any, NamedTuple

__all__ = [
    "Classification",
    "StreamProtocol",
    "SseGrammar",
    "AnthropicMessagesGrammar",
    "OpenAIResponsesGrammar",
    "ChatCompletionsGrammar",
    "GeminiGrammar",
    "classify_response",
    "grammar_for",
]


class StreamProtocol(str, Enum):
    """The four inbound streaming routes the bridge serves.

    The string values match ``harness.bridge.InboundProtocol`` member-for-member
    (asserted by the bridge-driven test module) so test code can pass either
    enum where one is expected.

    The non-streaming Gemini route (``:generateContent``) has no member: there
    is no stream to validate.
    """

    MESSAGES = "messages"
    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"
    GEMINI = "gemini"


class Classification(str, Enum):
    """What ``finish()`` says about the stream it just saw.

    Attributes:
        COMPLETE_SENTENCE: The grammar's closing event was reached with all
            structure closed. On Responses this includes the KBR-250 D4 shape
            ``[error, response.completed(status=incomplete)]`` — the closing
            event is grammar; the ``status`` field is payload semantics.
        ERROR_TERMINAL: The stream ended with the documented error event(s) and
            every opened structure was closed first.
        JSON_ERROR: The bridge answered with a JSON error body and no stream
            (KBR-155 D1–D7: every pre-release failure is JSON, not SSE).
        TRUNCATED: An opened block/item was never closed, or the stream stopped
            with no terminal event at all.
        MALFORMED: A structural violation — an event kind illegal at its
            position, an unparseable payload, a frame after the terminal.
    """

    COMPLETE_SENTENCE = "complete_sentence"
    ERROR_TERMINAL = "error_terminal"
    JSON_ERROR = "json_error"
    TRUNCATED = "truncated"
    MALFORMED = "malformed"


class _SseFrame(NamedTuple):
    """One parsed SSE frame.

    Attributes:
        event: The ``event:`` line's value, or ``None`` when the frame carried
            none. Chat Completions and Gemini frames never carry one; a
            Messages or Responses frame without one is malformed.
        data: The ``data:`` lines joined with newlines, or ``None`` when the
            frame carried none. ``"[DONE]"`` is Chat Completions' sentinel, not
            JSON — the JSON check belongs to each grammar, not the parser.
    """

    event: str | None
    data: str | None


class SseGrammar:
    """Base class: SSE framing plus the classification contract.

    Subclasses implement :meth:`_consume_event` (one resolved frame) and
    :meth:`_classify_finish` (the end-of-stream verdict). The base owns byte
    accumulation, frame splitting, the sticky-malformed flag and the
    diagnostic.

    An incremental UTF-8 decoder feeds the buffer, because the bridge writes in
    transport-sized chunks and a multi-byte character can straddle a chunk
    boundary.
    """

    #: Every event kind this grammar knows. A frame whose resolved kind is not
    #: here is malformed — the bridge growing a new kind is exactly the drift
    #: this guard exists to make visible.
    KNOWN_EVENTS: frozenset[str] = frozenset()

    def __init__(self) -> None:
        """Start an empty stream."""
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._position = 0
        self._malformed_reason: str | None = None
        self._terminal_event: str | None = None
        self._last_kind: str | None = None
        self._finished: Classification | None = None
        #: Whether :meth:`feed` ran at least once. A zero-byte feed counts:
        #: the RuntimeError below is a caller-bug detector, not a stream
        #: property, and the documented ``feed(b"")`` escape hatch must
        #: actually disarm it (it did not in the first draft — an empty
        #: decode appends nothing, so position and buffer both stayed at
        #: zero and the same error re-fired).
        self._fed = False

    # -- public API --------------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        """Consume the next bytes of the downstream stream.

        Args:
            chunk: Raw response bytes, in any framing — complete frames, partial
                frames, or several frames per chunk are all accepted. A
                zero-byte chunk is legal and marks the stream as fed, so a
                genuinely empty response classifies (``truncated``) instead
                of raising.

        Raises:
            RuntimeError: When called after :meth:`finish`.
        """
        if self._finished is not None:
            raise RuntimeError("feed() after finish(); the stream is already classified")
        self._fed = True
        self._buffer += self._decoder.decode(chunk)
        # Frames terminate at a blank line. Whatever remains after the last
        # blank line is a partial frame and stays buffered until more bytes
        # arrive or finish() discards it (the SSE spec's own EOF behaviour).
        while "\n\n" in self._buffer:
            raw, self._buffer = self._buffer.split("\n\n", 1)
            self._consume_frame(_parse_frame(raw))

    def finish(self) -> Classification:
        """Close the stream and return the classification.

        A trailing partial frame is discarded, not parsed: an incomplete frame
        never completed, so it cannot change the grammar state. This is the SSE
        spec's EOF behaviour, and it is what makes a stream cut off mid-frame
        classify as ``truncated`` rather than ``malformed`` — the bytes were
        cut, not mis-ordered.

        Returns:
            The classification, memoized; calling twice returns the same one.

        Raises:
            RuntimeError: When :meth:`feed` was never called — a caller bug,
                not a stream property. Call ``feed(b"")`` for a genuinely
                empty stream; it disarms this and the stream classifies
                ``truncated`` normally.
        """
        if self._finished is None:
            if not self._fed:
                raise RuntimeError("finish() on a grammar that was never fed; feed(b\"\") for a genuinely empty stream")
            self._finished = self._classify_finish()
        return self._finished

    @property
    def diagnostic(self) -> str:
        """Return why the stream was judged, for the failure message.

        Returns:
            The malformed reason when the stream is malformed, else a summary
            of the last event seen. Empty before any frame arrives.
        """
        if self._malformed_reason is not None:
            return self._malformed_reason
        if self._last_kind is None:
            return "no frames arrived"
        return f"last event {self._last_kind!r} at frame {self._position}"

    # -- subclass contract -------------------------------------------------

    def _consume_event(self, kind: str, data: str | None) -> None:
        """Absorb one frame of a known kind.

        Args:
            kind: The resolved event kind (the ``event:`` line, or the
                protocol's fallback for data-only frames).
            data: The raw ``data:`` payload, or ``None`` when the frame carried
                none.
        """

    def _classify_finish(self) -> Classification:
        """Return the end-of-stream verdict. Called once, from :meth:`finish`."""
        return Classification.TRUNCATED

    # -- shared machinery ---------------------------------------------------

    def _fail(self, reason: str) -> None:
        """Mark the stream malformed, naming the frame and the violation.

        Sticky: once malformed, later frames cannot restore validity.

        Args:
            reason: What was illegal, phrased for the failing test's message.
        """
        if self._malformed_reason is None:
            self._malformed_reason = f"frame {self._position}: {reason}"

    def _consume_frame(self, frame: _SseFrame) -> None:
        """Validate the frame's shape, resolve its kind, dispatch.

        Args:
            frame: The parsed frame.
        """
        self._position += 1
        if self._malformed_reason is not None:
            return
        if self._terminal_event is not None:
            self._fail(
                f"event {frame.event!r} after terminal event {self._terminal_event!r}"
            )
            return

        kind = self._resolve_kind(frame)
        if kind is None or kind not in self.KNOWN_EVENTS:
            shown = frame.event if frame.event is not None else repr(frame.data)
            self._fail(f"unknown event kind {shown!r}")
            return

        self._last_kind = kind
        self._consume_event(kind, frame.data)

    def _resolve_kind(self, frame: _SseFrame) -> str | None:
        """Return the frame's event kind.

        The ``event:`` line is authoritative. Chat Completions and Gemini
        frames carry no ``event:`` line — their grammars key on the data alone,
        which is why the default returns the line as-is and data-only frames
        resolve to the empty-string kind their grammar defines.

        Args:
            frame: The parsed frame.

        Returns:
            The kind, or ``None`` when the grammar cannot name the frame.
        """
        return frame.event

    def _parsed(self, data: str | None) -> dict[str, Any] | None:
        """Parse a frame's JSON payload, marking the stream malformed on failure.

        Args:
            data: The raw ``data:`` payload, or ``None``.

        Returns:
            The parsed dict, or ``None`` when the payload is absent or not a
            JSON object (which is always malformed for these grammars — the
            bridge never writes a bare array or scalar frame).
        """
        if data is None:
            self._fail("frame carries no data line")
            return None
        try:
            parsed = json.loads(data)
        except ValueError:
            self._fail(f"data is not JSON: {data[:80]!r}")
            return None
        if not isinstance(parsed, dict):
            self._fail(f"data is JSON but not an object: {data[:80]!r}")
            return None
        return parsed


def _parse_frame(raw: str) -> _SseFrame:
    """Parse one SSE frame's lines into its event name and data payload.

    Comment lines (``:``-prefixed) are ignored per the SSE spec; ``id:`` and
    ``retry:`` fields are ignored — the bridge writes neither, but a forward
    proxy between the tests and the bridge could.

    Args:
        raw: The frame's text, without its blank-line terminator.

    Returns:
        The frame, with ``None`` for whichever of the two fields it lacks.
    """
    event: str | None = None
    data_lines: list[str] = []
    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
    data = "\n".join(data_lines) if data_lines else None
    return _SseFrame(event=event, data=data)


# ---------------------------------------------------------------------------
# Anthropic Messages (POST /v1/messages)
# ---------------------------------------------------------------------------


class AnthropicMessagesGrammar(SseGrammar):
    """The Anthropic Messages SSE grammar.

    ``message_start`` → (``ping``)* → blocks → (``message_delta``)* →
    ``message_stop``, plus the documented terminal shapes. Index discipline per
    G39 (parallel tool calls): each index opens **once**, closes **once** after
    its start, carries no delta outside its window, and closes may be out of
    order — blocks may overlap.

    **Why a ``message_start`` followed by a bare ``error`` (no content blocks)
    classifies ``error_terminal`` here, even though KBR-241 removed the
    verbatim-forward pre-content error shape.** The two shapes are byte-
    identical at the client's view: when the upstream sends ``message_start``
    and then aborts on the native passthrough, the close-out (``server.py``
    transport-drop branch's fallback) writes exactly one ``messages_format_error``
    event, and the content blocks never opened. The grammar cannot tell this
    bridge-originated shape from the KBR-155 verbatim-forward shape at the
    byte level — both are ``[message_start, error]`` — so it accepts both. The
    guard that catches a KBR-241 regression is the KBR-241 conformance tests
    on the bridge's error path, not the grammar's grammar. A bridge that
    returns a *content-bearing* stream followed by ``error`` continues to
    classify ``error_terminal``; a stream that never wrote ``message_start``
    at all is ``malformed``.
    """

    KNOWN_EVENTS = frozenset(
        {
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
            "error",
        }
    )

    def __init__(self) -> None:
        """Start before the first event."""
        super().__init__()
        self._message_started = False
        self._saw_content = False
        #: Every index ever opened. An index may open exactly once per stream.
        self._opened: set[int] = set()
        #: Indexes currently open. Out-of-order closes are legal (G39).
        self._open: set[int] = set()

    def _consume_event(self, kind: str, data: str | None) -> None:
        """Absorb one Messages frame.

        Args:
            kind: The resolved event kind.
            data: The raw payload.
        """
        # Every Messages frame the bridge writes carries a JSON data line; the
        # parse happens here (not in the base) so a falsification case can feed
        # a data-less frame and see it named as malformed rather than skipped.
        parsed = self._parsed(data)
        if parsed is None:
            return
        if kind == "message_start":
            if self._message_started or self._saw_content:
                self._fail("message_start after the message already began")
                return
            self._message_started = True
            return
        if kind == "ping":
            return
        if kind == "content_block_start":
            index = parsed.get("index")
            if not self._message_started:
                self._fail("content_block_start before message_start")
                return
            if not isinstance(index, int) or isinstance(index, bool):
                self._fail(f"content_block_start with a non-integer index {index!r}")
                return
            if index in self._opened:
                self._fail(f"content_block_start reopens index {index}")
                return
            if index != len(self._opened):
                self._fail(
                    f"content_block_start index {index} skips an index (next is {len(self._opened)})"
                )
                return
            self._opened.add(index)
            self._open.add(index)
            self._saw_content = True
            return
        if kind == "content_block_delta":
            index = parsed.get("index")
            if not isinstance(index, int) or index not in self._open:
                self._fail(f"content_block_delta at index {index!r} with no open block there")
                return
            return
        if kind == "content_block_stop":
            index = parsed.get("index")
            if not isinstance(index, int) or index not in self._open:
                self._fail(f"content_block_stop at index {index!r} with no open block there")
                return
            self._open.discard(index)
            return
        if kind == "message_delta":
            if not self._message_started:
                self._fail("message_delta before message_start")
                return
            if self._open:
                self._fail(f"message_delta with blocks still open: {sorted(self._open)}")
                return
            self._saw_content = True
            return
        if kind == "message_stop":
            if not self._message_started:
                self._fail("message_stop before message_start")
                return
            if self._open:
                self._fail(f"message_stop with blocks still open: {sorted(self._open)}")
                return
            self._terminal_event = kind
            return
        if kind == "error":
            # Legal only as the last event, and only after the message began.
            # Whether the stream then classifies as error_terminal or truncated
            # is finish()'s call: an open block at finish() wins (precedence
            # rule 2), because the native Messages-wire close-out writes
            # exactly this shape — one error event with the forwarded blocks
            # left open. An ``error`` with no ``message_start`` before it is
            # the KBR-155 verbatim-forward shape KBR-241 removed: the
            # bridge's own fallback only ever fires after ``message_start``
            # is on the wire (it requires ``sr is not None``, and the first
            # written byte is always ``message_start``), so a bare leading
            # ``error`` is a shape the bridge does not produce.
            if not self._message_started:
                self._fail("error event before message_start")
                return
            if "error" not in parsed:
                self._fail("error event without an error member")
                return
            self._terminal_event = kind
            return
        raise AssertionError(f"unhandled known kind {kind!r}")  # pragma: no cover

    def _classify_finish(self) -> Classification:
        """Apply the classification precedence to the end state."""
        if self._malformed_reason is not None:
            return Classification.MALFORMED
        if self._terminal_event == "message_stop":
            return Classification.COMPLETE_SENTENCE
        if self._open:
            return Classification.TRUNCATED
        if self._terminal_event == "error":
            return Classification.ERROR_TERMINAL
        return Classification.TRUNCATED


# ---------------------------------------------------------------------------
# OpenAI Responses (POST /v1/responses)
# ---------------------------------------------------------------------------


#: The output-item kinds the bridge's translator emits. Anything else is a new
#: bridge capability the grammar has not been taught — drift the guard exists
#: to surface, not to accept silently.
_ITEM_KINDS = frozenset({"message", "function_call"})


def _hashable_scalar(value: Any) -> bool:
    """Return whether ``value`` is a hashable scalar (int or str, never bool).

    Used to validate the ``output_index`` / ``content_index`` fields before
    using them as a dict key or a set member. Schemathesis fuzzing reaches
    this grammar with arbitrary JSON, and a list-or-dict value raised
    ``TypeError`` from inside the set/dict lookup — a crash, not the
    malformed classification the contract promises.
    """
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


class OpenAIResponsesGrammar(SseGrammar):
    """The OpenAI Responses SSE grammar, as the bridge writes it.

    ``response.created`` → ``response.in_progress`` → items →
    ``response.completed``, with three documented departures from the vendor's
    happy path:

    - **Lazy lifecycle (KBR-242).** ``response.created`` / ``in_progress`` are
      written on the first non-finish write. A purely-empty D4 attempt never
      emits them, so ``response.created`` is optional and ``[error,
      response.completed]`` is a legal stream.
    - **In-stream exhaustion (KBR-250 D4).** The bridge writes one ``error``
      event and then the post-loop synthesizes the item-done events plus
      ``response.completed(status="incomplete")``. ``error`` is therefore not
      a final terminal here: ``response.completed`` after it is the closing
      event.
    - **Overlapping items (G40).** Items may overlap and close out of order;
      each is keyed by its ``output_index`` and must open once, close once.

    The event vocabulary is exactly what ``kitty.bridge.responses.events``
    emits — an unknown ``response.*`` kind is malformed, so the bridge growing
    a new event kind turns this suite red until the grammar learns it.
    """

    KNOWN_EVENTS = frozenset(
        {
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.completed",
            "response.failed",
            "error",
        }
    )

    def __init__(self) -> None:
        """Start before the first event."""
        super().__init__()
        self._created = False
        self._saw_item = False
        #: output_index → {"kind": str, "open": bool, "parts": set[int]}
        self._items: dict[Any, dict[str, Any]] = {}
        #: Whether an ``error`` event was consumed. Distinct from
        #: :attr:`_terminal_event`, which only records the *closing* event
        #: (``response.completed`` / ``response.failed``); a bare ``[error]``
        #: stream with no follow-up completes this flag but leaves the
        #: terminal unset, which ``_classify_finish`` resolves to
        #: ``error_terminal`` once every item is closed.
        self._error_seen: bool = False

    def _consume_event(self, kind: str, data: str | None) -> None:
        """Absorb one Responses frame.

        Args:
            kind: The resolved event kind.
            data: The raw payload.
        """
        parsed = self._parsed(data)
        if parsed is None:
            return
        # The ``event:`` line is authoritative; a data-only frame falls back to
        # the payload's ``type`` member, which the bridge always writes too.
        if kind == "response.created":
            if self._created or self._saw_item:
                self._fail("response.created after the response already began")
                return
            self._created = True
            return
        if kind == "response.in_progress":
            if not self._created:
                self._fail("response.in_progress before response.created")
                return
            return
        if kind == "response.output_item.added":
            output_index = parsed.get("output_index")
            item = parsed.get("item")
            item_kind = item.get("type") if isinstance(item, dict) else None
            if item_kind not in _ITEM_KINDS:
                self._fail(f"output_item.added with unknown item type {item_kind!r}")
                return
            if not _hashable_scalar(output_index):
                self._fail(f"output_item.added with a non-scalar output_index {output_index!r}")
                return
            key = _index_key(output_index)
            if key in self._items:
                self._fail(f"output_item.added reopens output_index {output_index!r}")
                return
            self._items[key] = {"kind": item_kind, "open": True, "parts": set()}
            self._saw_item = True
            return
        if kind == "response.content_part.added":
            item, key = self._open_item(parsed, "content_part.added")
            if item is None:
                return
            content_index = parsed.get("content_index")
            if not _hashable_scalar(content_index):
                self._fail(f"content_part.added with a non-scalar content_index {content_index!r}")
                return
            item["parts"].add(content_index)
            return
        if kind in ("response.output_text.delta", "response.output_text.done"):
            item, key = self._open_item(parsed, kind)
            if item is None:
                return
            content_index = parsed.get("content_index")
            if not _hashable_scalar(content_index):
                self._fail(f"{kind} with a non-scalar content_index {content_index!r}")
                return
            if content_index not in item["parts"]:
                self._fail(f"{kind} outside any open content part (content_index {content_index!r})")
                return
            return
        if kind == "response.content_part.done":
            item, key = self._open_item(parsed, kind)
            if item is None:
                return
            content_index = parsed.get("content_index")
            if not _hashable_scalar(content_index):
                self._fail(f"content_part.done with a non-scalar content_index {content_index!r}")
                return
            if content_index not in item["parts"]:
                self._fail(f"content_part.done outside any open content part (content_index {content_index!r})")
                return
            item["parts"].discard(content_index)
            return
        if kind in ("response.function_call_arguments.delta", "response.function_call_arguments.done"):
            item, key = self._open_item(parsed, kind)
            if item is None:
                return
            if item["kind"] != "function_call":
                self._fail(f"{kind} against an item of kind {item['kind']!r}")
                return
            return
        if kind == "response.output_item.done":
            output_index = parsed.get("output_index")
            if not _hashable_scalar(output_index):
                self._fail(f"output_item.done with a non-scalar output_index {output_index!r}")
                return
            key = _index_key(output_index)
            item = self._items.get(key)
            if item is None:
                self._fail(f"output_item.done for unknown output_index {parsed.get('output_index')!r}")
                return
            if not item["open"]:
                self._fail(f"output_item.done closes output_index {key!r} twice")
                return
            if item["parts"]:
                self._fail(f"output_item.done with open content parts: {sorted(item['parts'])}")
                return
            item["open"] = False
            return
        if kind == "response.completed":
            open_items = [key for key, item in self._items.items() if item["open"]]
            if open_items:
                self._fail(f"response.completed with items still open: {open_items}")
                return
            self._terminal_event = kind
            return
        if kind in ("response.failed", "error"):
            # Terminal events. On the D4 shape an ``error`` is followed by the
            # synthesized ``response.completed``; the base class rejects any
            # frame after a terminal, so ``error`` here must NOT set the
            # terminal when a completed event may still follow. The bridge
            # writes exactly ``[error, response.completed]`` (KBR-250), so
            # ``error`` is recorded but only ``response.completed`` /
            # ``response.failed`` closes the stream. The bare ``[error]`` case
            # (synthesis never ran) resolves to ``error_terminal`` in
            # ``_classify_finish`` via ``_error_seen`` below.
            self._terminal_event = kind if kind == "response.failed" else self._terminal_event
            self._error_seen = True
            return
        raise AssertionError(f"unhandled known kind {kind!r}")  # pragma: no cover

    def _open_item(self, parsed: dict[str, Any], kind: str) -> tuple[dict[str, Any] | None, Any]:
        """Return the open item an event names, or fail the stream.

        Args:
            parsed: The frame's payload.
            kind: The event kind, for the diagnostic.

        Returns:
            ``(item, key)`` when an open item matches, else ``(None, None)``
            with the stream marked malformed.
        """
        key = _index_key(parsed.get("output_index"))
        item = self._items.get(key)
        if item is None:
            self._fail(f"{kind} for unknown output_index {parsed.get('output_index')!r}")
            return None, None
        if not item["open"]:
            self._fail(f"{kind} against closed output_index {key!r}")
            return None, None
        return item, key

    def _resolve_kind(self, frame: _SseFrame) -> str | None:
        """Return the frame's kind: the event line, else the payload's ``type``.

        Args:
            frame: The parsed frame.

        Returns:
            The kind, or ``None`` when neither source names one.
        """
        if frame.event is not None:
            return frame.event
        if frame.data is not None:
            try:
                parsed = json.loads(frame.data)
            except ValueError:
                return None
            if isinstance(parsed, dict):
                kind = parsed.get("type")
                if isinstance(kind, str):
                    return kind
        return None

    def _classify_finish(self) -> Classification:
        """Apply the classification precedence to the end state."""
        if self._malformed_reason is not None:
            return Classification.MALFORMED
        if self._terminal_event == "response.completed":
            return Classification.COMPLETE_SENTENCE
        if any(item["open"] for item in self._items.values()):
            return Classification.TRUNCATED
        if self._terminal_event == "response.failed":
            return Classification.ERROR_TERMINAL
        if self._error_seen:
            # error with all items closed and no completed event: the D4
            # timeout / internal-error shape when the post-loop synthesis never
            # ran. The error event closed nothing; the structure is closed
            # because nothing was open.
            return Classification.ERROR_TERMINAL
        return Classification.TRUNCATED


def _index_key(value: Any) -> Any:
    """Normalise a validated ``output_index`` to a hashable dict key.

    The bridge writes integers; the vendor's own examples sometimes carry
    strings. Coercing through ``str`` would merge ``"0"`` and ``0`` into one
    item, which is precisely the kind of leniency a grammar must not have — so
    the raw value is the key and mixed spellings stay distinct items. Callers
    must have passed the value through :func:`_hashable_scalar` first: this
    function performs no validation, and an unvalidated list-or-dict value
    would raise ``TypeError`` from the set/dict lookup it feeds.

    Args:
        value: The ``output_index`` field as received.

    Returns:
        The value unchanged, suitable as a dict key (``None`` included, so a
        missing index is one distinct "unknown" bucket that fails loudly on
        the first lookup).
    """
    return value


# ---------------------------------------------------------------------------
# Chat Completions (POST /v1/chat/completions)
# ---------------------------------------------------------------------------


#: The literal sentinel that ends a Chat Completions stream. Documented
#: consumer behaviour makes this the terminal signal: EOF without it is a
#: genuine truncation, while a stream that reaches it is complete even without
#: a ``finish_reason`` chunk (many upstreams omit one, and consumers accept it).
CC_DONE = "[DONE]"


class ChatCompletionsGrammar(SseGrammar):
    """The Chat Completions SSE grammar.

    ``data: {chunk}``* → ``data: [DONE]``. The grammar is deliberately light
    beyond the sentinel: chunks must be JSON objects, an ``{"error": …}`` frame
    is an accepted terminal instead of ``[DONE]``, and nothing may follow the
    sentinel. ``finish_reason`` is **not** required — the CC passthrough
    forwards upstream bytes verbatim, and requiring a field the upstream
    controls would make this grammar a de-facto upstream contract rather than
    a bridge one.
    """

    KNOWN_EVENTS = frozenset({"chunk", "done", "error"})

    def __init__(self) -> None:
        """Start before the first frame."""
        super().__init__()
        self._done_seen = False
        self._error_seen = False
        self._saw_chunk = False

    def _resolve_kind(self, frame: _SseFrame) -> str | None:
        """Resolve data-only CC frames by their payload.

        Args:
            frame: The parsed frame.

        Returns:
            ``"done"`` for the ``[DONE]`` sentinel, ``"chunk"`` for a JSON
            frame, ``"error"`` for a JSON object carrying an ``error`` member,
            ``None`` for anything unnameable. An ``event:`` line in a CC frame
            is foreign to the grammar and resolves to ``None`` — malformed.
        """
        if frame.event is not None:
            return None
        if frame.data is None:
            return None
        if frame.data == CC_DONE:
            return "done"
        try:
            parsed = json.loads(frame.data)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        return "error" if "error" in parsed else "chunk"

    def _consume_event(self, kind: str, data: str | None) -> None:
        """Absorb one CC frame.

        Args:
            kind: The resolved kind.
            data: The raw payload.
        """
        if kind == "done":
            self._done_seen = True
            self._terminal_event = "done"
            return
        if kind == "error":
            # The route's terminal error frame. Like the Responses D4 error it
            # must be last; the base class rejects anything after a terminal.
            self._error_seen = True
            self._terminal_event = "error"
            return
        parsed = self._parsed(data)
        if parsed is None:
            return
        self._saw_chunk = True

    def _classify_finish(self) -> Classification:
        """Apply the classification precedence to the end state."""
        if self._malformed_reason is not None:
            return Classification.MALFORMED
        if self._terminal_event == "done":
            return Classification.COMPLETE_SENTENCE
        if self._terminal_event == "error":
            return Classification.ERROR_TERMINAL
        return Classification.TRUNCATED


# ---------------------------------------------------------------------------
# Gemini (POST /v1beta/models/{model}:streamGenerateContent)
# ---------------------------------------------------------------------------


class GeminiGrammar(SseGrammar):
    """The Gemini SSE grammar.

    ``data: {GenerateContentResponse}``* — each frame must carry ``candidates``
    (a list) or ``error`` (a dict); there is no ``[DONE]`` sentinel and a
    candidate stream that simply ends is complete. An ``error`` frame must be
    the final frame; the D4 exhaustion shape is exactly one
    ``{"error": {"code": 502, "message": …, "reason": "empty_response"}}``
    frame followed by EOF.
    """

    KNOWN_EVENTS = frozenset({"chunk", "error"})

    def __init__(self) -> None:
        """Start before the first frame."""
        super().__init__()
        self._saw_chunk = False
        self._error_seen = False

    def _resolve_kind(self, frame: _SseFrame) -> str | None:
        """Resolve data-only Gemini frames by their payload.

        Args:
            frame: The parsed frame.

        Returns:
            ``"error"`` for a JSON object carrying an ``error`` member,
            ``"chunk"`` for one carrying ``candidates``, ``None`` otherwise.
            An ``event:`` line is foreign to the grammar and resolves to
            ``None`` — malformed.
        """
        if frame.event is not None:
            return None
        if frame.data is None:
            return None
        try:
            parsed = json.loads(frame.data)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        if "error" in parsed:
            return "error"
        if "candidates" in parsed:
            return "chunk"
        return None

    def _consume_event(self, kind: str, data: str | None) -> None:
        """Absorb one Gemini frame.

        Args:
            kind: The resolved kind.
            data: The raw payload.
        """
        parsed = self._parsed(data)
        if parsed is None:
            return
        if kind == "error":
            # Must be the final frame; the base class rejects anything after a
            # terminal, so marking the terminal here enforces exactly that.
            if not isinstance(parsed.get("error"), dict):
                self._fail("error frame whose error member is not an object")
                return
            self._error_seen = True
            self._terminal_event = "error"
            return
        candidates = parsed.get("candidates")
        if not isinstance(candidates, list):
            self._fail(f"chunk with a non-list candidates member: {candidates!r}")
            return
        self._saw_chunk = True

    def _classify_finish(self) -> Classification:
        """Apply the classification precedence to the end state."""
        if self._malformed_reason is not None:
            return Classification.MALFORMED
        if self._terminal_event == "error":
            return Classification.ERROR_TERMINAL
        if self._saw_chunk:
            return Classification.COMPLETE_SENTENCE
        return Classification.TRUNCATED


# ---------------------------------------------------------------------------
# Factories and the response-level entry point
# ---------------------------------------------------------------------------


def grammar_for(protocol: StreamProtocol) -> SseGrammar:
    """Return a fresh state machine for ``protocol``.

    Args:
        protocol: The inbound streaming route whose grammar the stream must
            satisfy.

    Returns:
        A new grammar instance; one instance per stream, never shared.

    Raises:
        ValueError: When ``protocol`` is not one of the four streaming
            protocols.
    """
    grammars: dict[StreamProtocol, type[SseGrammar]] = {
        StreamProtocol.MESSAGES: AnthropicMessagesGrammar,
        StreamProtocol.RESPONSES: OpenAIResponsesGrammar,
        StreamProtocol.CHAT_COMPLETIONS: ChatCompletionsGrammar,
        StreamProtocol.GEMINI: GeminiGrammar,
    }
    try:
        return grammars[protocol]()
    except KeyError:
        raise ValueError(f"no grammar for protocol {protocol!r}") from None


def classify_response(protocol: StreamProtocol, status: int, body: str) -> Classification:
    """Classify a whole downstream response, stream or not.

    The entry point the bridge-driven tests use: ``BridgeFixture.post`` returns
    ``(status, text)`` and this decides whether the text is a stream to parse,
    a JSON error body (``json_error``), or something the bridge should never
    have written.

    Args:
        protocol: The inbound streaming route that answered.
        status: The HTTP status the client saw.
        body: The response body.

    Returns:
        The classification. A body that carries SSE frames is parsed as a
        stream. A JSON body with an error envelope — or any JSON body on a
        non-2xx status — is ``json_error``. An **empty** body splits on the
        status, and both halves are deliberate: with no bytes there is
        nothing to parse, so the status is the only evidence — an error
        status means the bridge meant an error envelope it never wrote
        (``json_error``), a success status means a stream cut before its
        first frame (``truncated``). Anything else (non-stream, non-JSON
        text; a JSON body with no error member on a 2xx) is ``malformed``.
    """
    stripped = body.lstrip()
    if stripped.startswith("event:") or stripped.startswith("data:"):
        grammar = grammar_for(protocol)
        grammar.feed(body.encode("utf-8"))
        return grammar.finish()

    # No stream frames. A pre-release failure is a JSON body carrying an error
    # envelope; the empty-ladder 502 on the Messages route is the shape every
    # drop-before-release scenario produces (KBR-155 D1–D7).
    if not stripped:
        return Classification.JSON_ERROR if status >= 400 else Classification.TRUNCATED
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return Classification.MALFORMED
    if isinstance(parsed, dict) and ("error" in parsed or status >= 400):
        return Classification.JSON_ERROR
    return Classification.MALFORMED
