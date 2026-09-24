"""Scripted failure builders for the bridge recorder.

`.system_design/TEST_SUITE.md` §6.3.1, §7.2 · plan task **T-B4** (KBR-43),
widened by **KBR-302** to serve the Gemini and OpenAI Responses wire grammars
as well as the original two (Anthropic Messages, Chat Completions).

T-W4 ships only the minimal success responders. This module builds the
``Responder`` closures for everything that can go wrong: SSE variants, error
statuses, Cloudflare blocks, empty responses, context-too-large rejections, and
mid-stream disconnects at each of §6.3.1's four injection points. Each builder
produces a deterministic byte sequence — no timing, no sleeps — built on the
``Reply`` API the recorder exposes.

The KBR-302 widening added data-only Gemini chunk builders and event-framed
Responses chunk builders alongside the existing Anthropic Messages and Chat
Completions builders. The library still serves only the formats its
:data:`_SERVED_FORMATS` frozenset names — Bedrock Converse and Ollama Chat
remain out of scope — but a wider Gemini/Responses test cell (T-I8 sibling
sites) can now drive both routes end to end through the same builder API.

**Import discipline.** The library imports nothing from ``src/kitty``. F9's
guard in :mod:`tests.harness.test_failures` enforces it structurally, so a
library that asked the bridge how to classify a body would inherit the
bridge's bugs and the §3.3.1 fidelity claim would collapse. Tests import the
bridge's detectors (``kitty.cloudflare``, ``BridgeServer._is_context_too_large_error``)
as oracles; the library itself never does.

**Determinism.** ``Reply.abort()`` flushes queued bytes via ``transport.close()``
before dropping, so a responder that writes a frame and aborts delivers that
frame to the client — on every platform (KBR-189 / §7.2). A responder that
writes nothing and aborts delivers zero bytes plus EOF.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from harness.contract import CapturedRequest, WireFormat
from harness.recorder import Reply, minimal_success_body, minimal_success_stream

__all__ = [
    "InjectionPoint",
    "ScriptExhausted",
    "sse_frame",
    "cc_chunk",
    "CC_DONE",
    "format_gemini_sse_chunk",
    "error_status",
    "cloudflare_block",
    "context_too_large",
    "empty_response",
    "drop_at",
    "success",
    "scripted",
]


# ---------------------------------------------------------------------------
# Frame-level helpers (F6).
# ---------------------------------------------------------------------------

#: A literal the §7.2 documented shape uses to terminate CC streams.
CC_DONE: bytes = b"data: [DONE]\n\n"


def format_gemini_sse_chunk(data: dict[str, Any]) -> bytes:
    """Format a Gemini SSE chunk (``data: <json>\\n\\n``; no ``event:`` line).

    Gemini's wire grammar is data-only — the bridge's Gemini reader keys on the
    payload's ``candidates`` and ``error`` keys rather than on an SSE event
    name. Mirrors the shape ``kitty.bridge.gemini.events.format_gemini_sse``
    emits, kept as a module-private helper here so the library never imports
    ``src/kitty`` (F9).

    Args:
        data: The chunk's JSON-serialisable body.

    Returns:
        The Gemini chunk bytes.
    """
    return f"data: {json.dumps(data)}\n\n".encode()


def sse_frame(event: str, data: dict[str, Any] | str) -> bytes:
    """Build one Anthropic-Messages SSE frame.

    Anthropic's SSE grammar carries an ``event:`` line in addition to the
    ``data:`` payload (the SDK's stream reader keys on the event name — see
    §11 Q14 D2). Chat Completions' grammar is ``data:``-only and is built by
    :func:`cc_chunk`.

    Args:
        event: The SSE event name, e.g. ``"message_start"``.
        data: Either a JSON-serialisable dict or a verbatim string for the
            ``data:`` line.

    Returns:
        The frame bytes, terminated by a blank line. JSON encoding uses the
        library default for consistency with the recorder's :func:`minimal_success_stream`.
    """
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n".encode()


def cc_chunk(data: dict[str, Any] | str) -> bytes:
    """Build one Chat Completions SSE chunk.

    CC frames carry only a ``data:`` line — no ``event:``. End-of-stream is
    :data:`CC_DONE`, not a JSON object.

    Args:
        data: Either a JSON-serialisable dict or a verbatim string for the
            ``data:`` line.

    Returns:
        The chunk bytes, terminated by a blank line.
    """
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"data: {payload}\n\n".encode()


# ---------------------------------------------------------------------------
# Public injection-point enum (F5).
# ---------------------------------------------------------------------------


class InjectionPoint(Enum):
    """The four mid-stream injection points §6.3.1 names.

    Each member maps to a §6.3.1 row. The ordering matches the design's table
    so a reader can find the row by name in one step.
    """

    BEFORE_FIRST_BYTE = "before_first_byte"  # §6.3.1 row 1 (pre-emission)
    AFTER_TEXT = "after_text"  # §6.3.1 row 2 (post-emission)
    MID_TOOL_ARGUMENTS = "mid_tool_arguments"  # §6.3.1 row 3 (post-emission)
    BEFORE_TERMINAL = "before_terminal"  # §6.3.1 row 4 (post-emission)


class ScriptExhausted(RuntimeError):
    """Raised by :func:`scripted` when an attempt exceeds the scripted sequence.

    F7.b: silent fallback would swallow a test's mis-modeling; loud raise lets
    the failure name itself as the cause.
    """


# ---------------------------------------------------------------------------
# Frame sequences per (format, injection point). Used by `drop_at`, exposed
# publicly via `frames_for` so falsification tests stay in lock-step with the
# library without reaching into its internals.
# ---------------------------------------------------------------------------

#: The marker the partial-JSON delta in MID_TOOL_ARGUMENTS accumulates.
#: Byte-exact: an unterminated string inside an unterminated object — any
#: JSON parser rejects it (AC-12).
_PARTIAL_TOOL_JSON = '{"arg1":"v'

#: The visible text AFTER_TEXT emits on the Anthropic Messages wire.
_TEXT = "kbr-tb4"

#: The tool-use id MID_TOOL_ARGUMENTS emits. Distinct from any ID the agent
#: could send so a downstream assertion cannot collide with a real call.
_TOOL_ID = "kbr-tb4-tool"

#: The tool-use name MID_TOOL_ARGUMENTS emits.
_TOOL_NAME = "probe"


def _anthropic_message_start() -> bytes:
    return sse_frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_kbr_tb4",
                "type": "message",
                "role": "assistant",
                "model": "kbr-tb4-model",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )


def _anthropic_text_start(index: int) -> bytes:
    return sse_frame(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
    )


def _anthropic_text_delta(index: int) -> bytes:
    return sse_frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": _TEXT},
        },
    )


def _anthropic_block_stop(index: int) -> bytes:
    return sse_frame("content_block_stop", {"type": "content_block_stop", "index": index})


def _anthropic_message_delta_end() -> bytes:
    return sse_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
    )


def _anthropic_message_stop() -> bytes:
    return sse_frame("message_stop", {"type": "message_stop"})


def _anthropic_tool_start(index: int) -> bytes:
    return sse_frame(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": _TOOL_ID, "name": _TOOL_NAME, "input": {}},
        },
    )


def _anthropic_tool_delta(index: int) -> bytes:
    return sse_frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": _PARTIAL_TOOL_JSON},
        },
    )


def _cc_role_chunk() -> bytes:
    # Role-only: no `content` key beside the role, so a consumer can tell a
    # pure skeleton (F4.d) from a content-bearing chunk without a special case.
    return cc_chunk(
        {
            "id": "chatcmpl-kbr-tb4",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "kbr-tb4-model",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )


def _cc_content_chunk() -> bytes:
    return cc_chunk(
        {
            "id": "chatcmpl-kbr-tb4",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "kbr-tb4-model",
            "choices": [
                {"index": 0, "delta": {"content": _TEXT}, "finish_reason": None}
            ],
        }
    )


def _cc_finish_chunk() -> bytes:
    return cc_chunk(
        {
            "id": "chatcmpl-kbr-tb4",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "kbr-tb4-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )


def _cc_tool_chunk() -> bytes:
    return cc_chunk(
        {
            "id": "chatcmpl-kbr-tb4",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "kbr-tb4-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_kbr_tb4",
                                "type": "function",
                                "function": {"name": _TOOL_NAME, "arguments": _PARTIAL_TOOL_JSON},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )


def _frames_for_anthropic(point: InjectionPoint) -> tuple[bytes, ...]:
    """The bytes :func:`drop_at` writes for an Anthropic-Messages injection point.

    Exposed via :func:`frames_for` so falsification tests can replay the same
    sequence without reaching into the library's responder closure.
    """
    if point is InjectionPoint.BEFORE_FIRST_BYTE:
        return ()
    if point is InjectionPoint.AFTER_TEXT:
        return (
            _anthropic_message_start(),
            _anthropic_text_start(0),
            _anthropic_text_delta(0),
            _anthropic_block_stop(0),
        )
    if point is InjectionPoint.MID_TOOL_ARGUMENTS:
        # F5.a: tool block deliberately left open — no `content_block_stop`.
        return (
            _anthropic_message_start(),
            _anthropic_tool_start(0),
            _anthropic_tool_delta(0),
        )
    if point is InjectionPoint.BEFORE_TERMINAL:
        return (
            _anthropic_message_start(),
            _anthropic_text_start(0),
            _anthropic_text_delta(0),
            _anthropic_block_stop(0),
            _anthropic_message_delta_end(),
        )
    raise ValueError(f"unknown injection point: {point!r}")


def _frames_for_cc(point: InjectionPoint) -> tuple[bytes, ...]:
    """The bytes :func:`drop_at` writes for a Chat Completions injection point."""
    if point is InjectionPoint.BEFORE_FIRST_BYTE:
        return ()
    if point is InjectionPoint.AFTER_TEXT:
        return (_cc_role_chunk(), _cc_content_chunk())
    if point is InjectionPoint.MID_TOOL_ARGUMENTS:
        return (_cc_role_chunk(), _cc_tool_chunk())
    if point is InjectionPoint.BEFORE_TERMINAL:
        return (_cc_role_chunk(), _cc_content_chunk(), _cc_finish_chunk())
    raise ValueError(f"unknown injection point: {point!r}")


# ---------------------------------------------------------------------------
# Gemini (data-only SSE) — KBR-302.
# ---------------------------------------------------------------------------

#: Per-format stable IDs the Gemini and Responses builders stamp into their
#: minimal-success and empty-response envelopes. Split per format so the
#: Responses wire does not carry Gemini-named IDs — the strings are
#: observable on the wire and a provider can fingerprint them (review-bot
#: note, KBR-302 round 2).
_GEMINI_MODEL_ID = "recorder-gemini-model"
_GEMINI_RESPONSE_ID = "recorder-gemini-response"
_RESPONSES_MODEL_ID = "recorder-responses-model"
_RESPONSES_RESPONSE_ID = "recorder-responses-response"


def _gemini_role_chunk() -> bytes:
    """A Gemini role-only contentless chunk (the empty-ladder trigger on Gemini).

    Returns:
        ``data: {"candidates":[{"content":{"role":"model","parts":[]}}]}`` — no
        text in the parts, no finishReason on the candidate. The Gemini reader
        treats this as contentless.
    """
    return format_gemini_sse_chunk(
        {"candidates": [{"content": {"role": "model", "parts": []}}]}
    )


def _gemini_text_chunk(text: str = _TEXT) -> bytes:
    """A Gemini content-bearing chunk with one text part.

    Args:
        text: The visible text content; defaults to :data:`_TEXT`.

    Returns:
        ``data: {"candidates":[{"content":{"role":"model","parts":[{"text":...}]}}]}``
    """
    return format_gemini_sse_chunk(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
    )


def _gemini_finish_chunk(reason: str = "STOP") -> bytes:
    """A Gemini terminal chunk carrying the chosen stop ``reason``.

    Args:
        reason: The ``finishReason`` value — the bridge's truncation reader
            keys on this. Defaults to ``"STOP"``; the KBR-99 R4 widening
            moves the truncation D3 trigger onto a non-default value (e.g.
            ``"MAX_TOKENS"``).

    Returns:
        ``data: {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":reason}]}``
    """
    return format_gemini_sse_chunk(
        {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": reason}]}
    )


def _gemini_in_stream_error_chunk(
    *, code: int = 502, message: str = "upstream blew up", status: str = "UNAVAILABLE"
) -> bytes:
    """A Gemini in-stream error chunk (KBR-99 R2).

    Args:
        code: The numeric error code the bridge's reader classifies.
        message: The human-readable error message.
        status: The string status (e.g. ``"UNAVAILABLE"``).

    Returns:
        ``data: {"error":{"code":<int>,"message":<str>,"status":<str>}}``.
    """
    return format_gemini_sse_chunk(
        {"error": {"code": code, "message": message, "status": status}}
    )


def _frames_for_gemini(point: InjectionPoint) -> tuple[bytes, ...]:
    """The bytes :func:`drop_at` writes for a Gemini injection point."""
    if point is InjectionPoint.BEFORE_FIRST_BYTE:
        return ()
    if point is InjectionPoint.AFTER_TEXT:
        return (_gemini_role_chunk(), _gemini_text_chunk())
    if point is InjectionPoint.MID_TOOL_ARGUMENTS:
        # A functionCall part carries the tool name; the args are partial.
        return (
            format_gemini_sse_chunk(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": _TOOL_NAME,
                                            "args": {"arg1": "v"},  # partial
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            ),
        )
    if point is InjectionPoint.BEFORE_TERMINAL:
        return (_gemini_role_chunk(), _gemini_text_chunk(), _gemini_finish_chunk())
    raise ValueError(f"unknown injection point: {point!r}")


# ---------------------------------------------------------------------------
# OpenAI Responses (event-framed SSE) — KBR-302.
# ---------------------------------------------------------------------------

#: Sequence number the Responses builder emits on the start/delta/completed
#: events. A real Responses stream uses monotonically increasing numbers per
#: event; the library emits 0/1/2 so the byte sequence is deterministic and
#: the reader's sequence assertions don't depend on a hidden counter.
_RESPONSES_SEQ_CREATED = 0
_RESPONSES_SEQ_DELTA = 1
_RESPONSES_SEQ_COMPLETED = 2


def _responses_event(event: str, payload: dict[str, Any]) -> bytes:
    """Build one OpenAI Responses SSE frame (``event:`` line + ``data:`` line).

    Args:
        event: The event name (``"response.created"`` etc).
        payload: The event's JSON body.

    Returns:
        ``event: <name>\\ndata: <json>\\n\\n`` (the bridge's Responses reader
        keys on the ``type`` field inside the payload).
    """
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _responses_created_event() -> bytes:
    """``response.created`` — the lifecycle opening on a Responses stream.

    Returns:
        ``event: response.created\\ndata: {"type":"response.created","sequence_number":0,"response":{...}}``
    """
    return _responses_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": _RESPONSES_SEQ_CREATED,
            "response": {
                "id": _RESPONSES_RESPONSE_ID,
                "object": "response",
                "status": "in_progress",
                "model": _RESPONSES_MODEL_ID,
                "output": [],
                "usage": None,
            },
        },
    )


def _responses_text_delta_event(text: str = _TEXT) -> bytes:
    """``response.output_text.delta`` — the content-bearing chunk.

    Args:
        text: The visible text content; defaults to :data:`_TEXT`.

    Returns:
        ``event: response.output_text.delta\\ndata: {...,"delta":"<text>"}``
    """
    return _responses_event(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": _RESPONSES_SEQ_DELTA,
            "delta": text,
        },
    )


def _responses_completed_event(*, status: str = "completed") -> bytes:
    """``response.completed`` — the terminal chunk on a Responses stream.

    Args:
        status: Either ``"completed"`` (success) or ``"incomplete"``
            (truncation; the KBR-99 R4 widening moves the truncation D3
            trigger onto ``"incomplete"`` with ``incomplete_details.reason``).

    Returns:
        ``event: response.completed\\ndata: {...,"response":{"status":"<status>",...}}``
    """
    response: dict[str, Any] = {
        "id": _RESPONSES_RESPONSE_ID,
        "object": "response",
        "status": status,
        "model": _RESPONSES_MODEL_ID,
        "output": [],
        "usage": None,
    }
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return _responses_event(
        "response.completed",
        {"type": "response.completed", "sequence_number": _RESPONSES_SEQ_COMPLETED, "response": response},
    )


def _responses_error_event(
    *, code: str = "upstream_error", message: str = "upstream blew up"
) -> bytes:
    """``error`` — a Responses in-stream error chunk (KBR-99 R2).

    Args:
        code: The error code string the bridge's reader classifies.
        message: The human-readable message.

    Returns:
        ``event: error\\ndata: {"type":"error","code":"<code>","message":"<msg>"}``
    """
    return _responses_event(
        "error",
        {"type": "error", "sequence_number": _RESPONSES_SEQ_COMPLETED, "code": code, "message": message},
    )


def _frames_for_responses(point: InjectionPoint) -> tuple[bytes, ...]:
    """The bytes :func:`drop_at` writes for an OpenAI Responses injection point.

    KBR-99 R5: Responses has no pre-content ``event: error`` equivalent
    (errors come inline on regular events). The empty/error pre-emission
    shapes for Responses are served through the AFTER_TEXT path's
    `response.created` + `response.output_text.delta` (a contentless
    Responses stream ends without a delta event, while a content-bearing one
    ends with `response.completed` — the §6.3.1 row 4 oracle). The cell
    for the inline `event: error` is `_responses_error_event`, exposed for
    the §6.3.1 grid consumers that need it; this function does not emit
    error events for any of the four injection points because the
    streaming-recovery grid reaches them through `frames_for` rather than
    through the `error` event (review-bot note, KBR-302 round 2).
    """
    if point is InjectionPoint.BEFORE_FIRST_BYTE:
        return ()
    if point is InjectionPoint.AFTER_TEXT:
        return (_responses_created_event(), _responses_text_delta_event())
    if point is InjectionPoint.MID_TOOL_ARGUMENTS:
        # A function_call_arguments delta is partial — no ``response.completed``.
        return (
            _responses_created_event(),
            _responses_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": _RESPONSES_SEQ_DELTA,
                    "delta": _PARTIAL_TOOL_JSON,
                },
            ),
        )
    if point is InjectionPoint.BEFORE_TERMINAL:
        return (
            _responses_created_event(),
            _responses_text_delta_event(),
            _responses_completed_event(),
        )
    raise ValueError(f"unknown injection point: {point!r}")


def frames_for(fmt: WireFormat, point: InjectionPoint) -> tuple[bytes, ...]:
    """The byte sequence ``drop_at(fmt, point)`` writes before aborting.

    Public so falsification tests can replay the same writes through a
    deliberately-broken responder (F10 / AC-18) without introspecting the
    library's closures.

    Args:
        fmt: The wire format the recorder serves.
        point: The injection point.

    Returns:
        The frames in order; empty for ``BEFORE_FIRST_BYTE``.

    Raises:
        ValueError: When ``fmt`` is outside the four formats this library serves
            (Anthropic, Chat Completions, Gemini, OpenAI Responses) or
            ``point`` is unrecognised.
    """
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        return _frames_for_anthropic(point)
    if fmt is WireFormat.CHAT_COMPLETIONS:
        return _frames_for_cc(point)
    if fmt is WireFormat.GEMINI:
        return _frames_for_gemini(point)
    if fmt is WireFormat.OPENAI_RESPONSES:
        return _frames_for_responses(point)
    raise ValueError(
        f"this library serves {{ANTHROPIC_MESSAGES, CHAT_COMPLETIONS, GEMINI, OPENAI_RESPONSES}}, not {fmt.value}"
    )


# ---------------------------------------------------------------------------
# Builders (F1, F2, F3, F4, F5, F8).
# ---------------------------------------------------------------------------


#: The four formats this library serves — the primary recorder's served set,
#: widened in KBR-302 to drive the Gemini and OpenAI Responses stream
#: handlers end to end.
_SERVED_FORMATS: frozenset[WireFormat] = frozenset(
    {
        WireFormat.ANTHROPIC_MESSAGES,
        WireFormat.CHAT_COMPLETIONS,
        WireFormat.GEMINI,
        WireFormat.OPENAI_RESPONSES,
    }
)


def _check_fmt(fmt: WireFormat) -> None:
    """Reject a format outside the library's scope at construction time (R3.8).

    Args:
        fmt: The format the caller asked for.

    Raises:
        ValueError: When ``fmt`` is not one of the four formats this library
            serves, or is not a :class:`WireFormat` at all. The message reads
            ``fmt.value`` only after the type check, so a non-enum caller
            sees this library's ``ValueError`` rather than an
            ``AttributeError`` from the interpolation.
    """
    served = sorted(f.value for f in _SERVED_FORMATS)
    if not isinstance(fmt, WireFormat) or fmt not in _SERVED_FORMATS:
        actual = fmt.value if isinstance(fmt, WireFormat) else repr(fmt)
        raise ValueError(f"this library serves {served}, not {actual}")


def error_status(
    fmt: WireFormat,
    status: int,
    *,
    error_type: str = "api_error",
    message: str = "upstream error",
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Answer with the format-native error envelope.

    F1. Anthropic Messages wraps the error under ``{"type": "error", "error":
    {"type", "message"}}``; Chat Completions uses ``{"error": {"message",
    "type", "code"}}`` where ``code`` carries the numeric status as the
    provider does in practice.

    Args:
        fmt: The wire format to answer in.
        status: The HTTP status code to send.
        error_type: The provider-specific error type (e.g. ``"overloaded_error"``).
        message: A human-readable error message.

    Returns:
        A :class:`~harness.recorder.Reply` coroutine writing one non-streaming
        JSON reply with status and content-type set.

    Raises:
        ValueError: When ``fmt`` is outside the library's served set.
    """
    _check_fmt(fmt)

    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        body: dict[str, Any] = {
            "type": "error",
            "error": {"type": error_type, "message": message},
        }
    elif fmt is WireFormat.GEMINI:
        # The Gemini error envelope carries the numeric code and a string
        # status — the shape `kitty.bridge.gemini`'s reader (and Google's
        # documented errors) use.
        body = {"error": {"code": status, "message": message, "status": error_type}}
    elif fmt is WireFormat.OPENAI_RESPONSES:
        # The Responses API's non-streaming error envelope mirrors Chat
        # Completions' `{"error": {...}}` with the message/type/code triple.
        body = {"error": {"message": message, "type": error_type, "code": str(status)}}
    else:
        # `code` carries the numeric status as a string — the bridge's
        # `_extract_error_fields` coerces with `str(...)` either way, but a
        # string is the documented contract for CC error envelopes and matches
        # what real providers (Z.AI, OpenAI partial) send.
        body = {"error": {"message": message, "type": error_type, "code": str(status)}}

    payload = json.dumps(body).encode()

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        response.content_length = len(payload)
        await response.begin(status, {"Content-Type": "application/json"})
        await response.write(payload)
        await response.write_eof()

    return responder


def cloudflare_block(
    fmt: WireFormat,
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Answer with a 403 carrying a Cloudflare challenge signature.

    F2. The body embeds the ``cf-mitigated`` substring literal — a hardcoded
    constant in this module, not an import from :mod:`kitty.cloudflare`, so
    the §3.3.1 independence rule holds. The test imports the detector.

    Args:
        fmt: The wire format — Cloudflare is transport-level so both formats
            answer identically.

    Returns:
        A responder writing a ``403 text/html`` with ``cf-mitigated`` in the body.

    Raises:
        ValueError: When ``fmt`` is outside the library's served set.
    """
    _check_fmt(fmt)

    # The body is a short HTML snippet — Cloudflare's real challenge pages
    # carry inline scripts with these markers. ``cf-mitigated`` is one of the
    # three signatures ``kitty.cloudflare._CF_SIGNATURES`` keys on.
    payload = (
        b"<!doctype html><html><head><title>cf-mitigated</title></head>"
        b"<body>cf-mitigated: challenge</body></html>"
    )

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        response.content_length = len(payload)
        await response.begin(403, {"Content-Type": "text/html"})
        await response.write(payload)
        await response.write_eof()

    return responder


def context_too_large(
    fmt: WireFormat,
    *,
    status: int = 400,
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Answer with a context-too-large rejection.

    F3. ``status=413`` is the detector's status-only path — any body matches.
    ``status=400`` requires the body to carry one of the six message needles
    the detector keys on; this builder uses ``"maximum context"`` to satisfy
    it for both formats.

    Args:
        fmt: The wire format to answer in.
        status: HTTP status — 413 for the body-agnostic path, 400 for the
            needle-based path (default).

    Returns:
        A responder writing one non-streaming JSON reply carrying the rejection.

    Raises:
        ValueError: When ``fmt`` is outside the library's served set.
    """
    _check_fmt(fmt)

    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        body: dict[str, Any] = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 250000 tokens > 8192 maximum context length",
            },
        }
    elif fmt is WireFormat.GEMINI:
        body = {
            "error": {
                "code": 400,
                "message": "prompt is too long: 250000 tokens > 8192 maximum context length",
                "status": "INVALID_ARGUMENT",
            }
        }
    elif fmt is WireFormat.OPENAI_RESPONSES:
        # The Responses API's rejection envelope mirrors `error_status`'s —
        # the message/type/code triple with a stringified code. Written as
        # its own branch rather than falling through to the CC `else:` so
        # the per-format shape stays explicit (review-bot note, KBR-302
        # round 1): a future divergence between the two envelopes would
        # otherwise be silently shared.
        body = {
            "error": {
                "message": "This model's maximum context length is 8192 tokens. Please reduce the length.",
                "type": "invalid_request_error",
                "code": str(status),
            }
        }
    else:
        # `code` as a string, matching `error_status`'s CC envelope — a
        # consistent contract across both builders, and the shape real
        # providers send.
        body = {
            "error": {
                "message": "This model's maximum context length is 8192 tokens. Please reduce the length.",
                "type": "invalid_request_error",
                "code": str(status),
            }
        }

    payload = json.dumps(body).encode()

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        response.content_length = len(payload)
        await response.begin(status, {"Content-Type": "application/json"})
        await response.write(payload)
        await response.write_eof()

    return responder


def _gemini_empty_success_body() -> dict[str, Any]:
    """The Gemini non-streaming empty body: one candidate, empty parts, STOP.

    Module-private so :mod:`tests.harness.test_failures` can pin the exact
    JSON shape without driving the responder through the recorder (which
    deliberately stays two-format; KBR-302 round 2).

    Returns:
        The JSON-serialisable empty-success body.
    """
    return {
        "candidates": [
            {"content": {"role": "model", "parts": []}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0},
    }


def _responses_empty_success_body() -> dict[str, Any]:
    """The Responses non-streaming empty body: completed status, empty output.

    Returns:
        The JSON-serialisable empty-success body.
    """
    return {
        "id": _RESPONSES_RESPONSE_ID,
        "object": "response",
        "status": "completed",
        "model": _RESPONSES_MODEL_ID,
        "output": [],
        "usage": None,
    }


def _gemini_content_success_body() -> dict[str, Any]:
    """The Gemini non-streaming minimal success: one candidate, one text part, STOP.

    Returns:
        The JSON-serialisable minimal-success body.
    """
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": _TEXT}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }


def _responses_content_success_body() -> dict[str, Any]:
    """The Responses non-streaming minimal success: one message output item.

    Returns:
        The JSON-serialisable minimal-success body.
    """
    return {
        "id": _RESPONSES_RESPONSE_ID,
        "object": "response",
        "status": "completed",
        "model": _RESPONSES_MODEL_ID,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _TEXT}],
            }
        ],
        "usage": None,
    }


def empty_response(
    fmt: WireFormat,
    *,
    stream: bool = False,
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Answer with a contentless success.

    F4. Non-streaming writes the format's empty success body. Streaming writes
    a stream that never releases the preamble hold (Anthropic Messages), and
    on Chat Completions a content-less completion (a role chunk then
    ``[DONE]`` — no content delta, no finish chunk). Since KBR-276 the hold
    withholds non-content lines on the raw CC wire too, so this shape is what
    fires the empty ladder on that route (KBR-232's converter-gated semantics
    are gone).

    Args:
        fmt: The wire format to answer in.
        stream: ``False`` for a JSON body, ``True`` for an SSE stream.

    Returns:
        A responder writing the empty success in the format's wire grammar.

    Raises:
        ValueError: When ``fmt`` is outside the library's served set.
    """
    _check_fmt(fmt)

    if stream and fmt is WireFormat.ANTHROPIC_MESSAGES:
        # F4.c — contentless Messages stream: message_start + an empty text
        # block (no delta releases the hold) + a stop + a message_delta + a
        # message_stop. The reply is complete (``write_eof``) so the recorder
        # finishes its reply rather than dropping.
        chunks: tuple[bytes, ...] = (
            _anthropic_message_start(),
            _anthropic_text_start(0),
            _anthropic_block_stop(0),
            _anthropic_message_delta_end(),
            _anthropic_message_stop(),
        )

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in chunks:
                await response.write(chunk)
            await response.write_eof()

        return responder

    if stream and fmt is WireFormat.CHAT_COMPLETIONS:
        # F4.d — role chunk + [DONE], the content-less CC completion. Since
        # KBR-276 the pre-emission hold withholds non-content lines on the raw
        # CC wire too: the role chunk no longer sets the bridge's `has_content`
        # flag, so THIS shape is what fires the empty ladder on that route
        # (KBR-232's converter-gated semantics are gone).
        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            await response.write(_cc_role_chunk())
            await response.write(CC_DONE)
            await response.write_eof()

        return responder

    if stream and fmt is WireFormat.GEMINI:
        # F4.e (KBR-302) — the content-less Gemini completion: role chunk +
        # STOP finish chunk, both data-only, no text part anywhere.
        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            await response.write(_gemini_role_chunk())
            await response.write(_gemini_finish_chunk())
            await response.write_eof()

        return responder

    if stream and fmt is WireFormat.OPENAI_RESPONSES:
        # F4.f (KBR-302) — the content-less Responses completion:
        # response.created + response.completed(status="completed"), no
        # output_text delta anywhere.
        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            await response.write(_responses_created_event())
            await response.write(_responses_completed_event())
            await response.write_eof()

        return responder

    # Non-streaming: explicit empty shape, NOT `minimal_success_body` which carries
    # non-empty content. The two formats' empty envelopes differ.
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        body = {
            "id": "msg_kbr_tb4",
            "type": "message",
            "role": "assistant",
            "model": "kbr-tb4-model",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    elif fmt is WireFormat.GEMINI:
        body = _gemini_empty_success_body()
    elif fmt is WireFormat.OPENAI_RESPONSES:
        body = _responses_empty_success_body()
    else:
        body = {
            "id": "chatcmpl-kbr-tb4",
            "object": "chat.completion",
            "created": 0,
            "model": "kbr-tb4-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    payload = json.dumps(body).encode()

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        response.content_length = len(payload)
        await response.begin(200, {"Content-Type": "application/json"})
        await response.write(payload)
        await response.write_eof()

    return responder


def drop_at(
    fmt: WireFormat,
    point: InjectionPoint,
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Write the frames for ``point``, then abort — §6.3.1's mid-stream drop.

    F5. The byte sequence is :func:`frames_for`; ordering is the only source of
    determinism. ``Reply.abort()`` (KBR-189 / §7.2) flushes the queue before
    dropping so the client receives the frames it names.

    Args:
        fmt: The wire format to answer in.
        point: The injection point — see :class:`InjectionPoint`.

    Returns:
        A responder opening an SSE stream, writing :func:`frames_for`, then
        aborting.

    Raises:
        ValueError: When ``fmt`` or ``point`` is unrecognised.
    """
    _check_fmt(fmt)
    frames = frames_for(fmt, point)

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        if point is InjectionPoint.BEFORE_FIRST_BYTE:
            # F5.b — abort without begin; the recorder finishes the connection
            # with no response bytes. Reply.abort() flushes the (empty) queue.
            await response.abort()
            return
        await response.begin(200, {"Content-Type": "text/event-stream"})
        for frame in frames:
            await response.write(frame)
        await response.abort()

    return responder


def success(
    fmt: WireFormat,
    *,
    stream: bool = False,
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Answer with T-W4's minimal success.

    F8. Exposed so :func:`scripted` sequences can end in a clean reply — §6.3.1
    row 1's "clean failover" oracle needs exactly that — without importing
    ``recorder.py`` at every call site. The builder delegates to
    :func:`minimal_success_body` / :func:`minimal_success_stream` so the
    "what a minimal success is" knowledge stays in one place.

    Args:
        fmt: The wire format to answer in.
        stream: ``False`` for a JSON body, ``True`` for an SSE stream.

    Returns:
        A responder writing the recorder's minimal success.

    Raises:
        ValueError: When ``fmt`` is outside the library's served set.
    """
    _check_fmt(fmt)

    # The two formats the recorder's `minimal_success_*` owns delegate to it —
    # "what a minimal success is" stays in one place for the shapes §7.2
    # assigns the recorder. The two formats KBR-302 widened (Gemini, OpenAI
    # Responses) are served inline here because the recorder deliberately
    # stays two-format; the library's own minimal shapes are the byte source.
    if stream and fmt not in (WireFormat.GEMINI, WireFormat.OPENAI_RESPONSES):
        chunks = minimal_success_stream(fmt)

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in chunks:
                await response.write(chunk)
            await response.write_eof()

        return responder

    if stream and fmt is WireFormat.GEMINI:
        chunks_gemini = (
            _gemini_role_chunk(),
            _gemini_text_chunk(),
            _gemini_finish_chunk(),
        )

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in chunks_gemini:
                await response.write(chunk)
            await response.write_eof()

        return responder

    if stream and fmt is WireFormat.OPENAI_RESPONSES:
        chunks_responses = (
            _responses_created_event(),
            _responses_text_delta_event(),
            _responses_completed_event(),
        )

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in chunks_responses:
                await response.write(chunk)
            await response.write_eof()

        return responder

    if fmt is WireFormat.GEMINI:
        body_success = _gemini_content_success_body()
        payload = json.dumps(body_success).encode()

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            response.content_length = len(payload)
            await response.begin(200, {"Content-Type": "application/json"})
            await response.write(payload)
            await response.write_eof()

        return responder

    if fmt is WireFormat.OPENAI_RESPONSES:
        body_success = _responses_content_success_body()
        payload = json.dumps(body_success).encode()

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            response.content_length = len(payload)
            await response.begin(200, {"Content-Type": "application/json"})
            await response.write(payload)
            await response.write_eof()

        return responder

    body = minimal_success_body(fmt)
    payload = json.dumps(body).encode()

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        response.content_length = len(payload)
        await response.begin(200, {"Content-Type": "application/json"})
        await response.write(payload)
        await response.write_eof()

    return responder


def scripted(
    *replies: Callable[[CapturedRequest, Reply], Awaitable[None]],
) -> Callable[[CapturedRequest, Reply], Awaitable[None]]:
    """Dispatch attempts in order; raise :class:`ScriptExhausted` on overrun.

    F7. The counter is per-instance and outlives a single client turn — a test
    reusing one fixture across two turns must build a fresh ``scripted(...)``
    for the second (F7.a). On overrun, raises loudly so a mis-modeled test
    fails with a named cause instead of silently retrying into a green.

    Args:
        *replies: The responders to invoke on attempts 1..N.

    Returns:
        A responder that delegates each call to ``replies[attempt - 1]``.
    """
    if not replies:
        raise ValueError("scripted() requires at least one reply")
    state: dict[str, int] = {"attempt": 0}

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        state["attempt"] += 1
        idx = state["attempt"] - 1
        if idx >= len(replies):
            raise ScriptExhausted(
                f"scripted() received attempt {state['attempt']} but only "
                f"{len(replies)} replies were given; extend the script"
            )
        await replies[idx](captured, response)

    return responder
