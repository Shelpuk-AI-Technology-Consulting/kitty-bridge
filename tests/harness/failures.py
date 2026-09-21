"""Scripted failure builders for the bridge recorder.

`.system_design/TEST_SUITE.md` §6.3.1, §7.2 · plan task **T-B4** (KBR-43).

T-W4 ships only the minimal success responders. This module builds the
``Responder`` closures for everything that can go wrong: SSE variants, error
statuses, Cloudflare blocks, empty responses, context-too-large rejections, and
mid-stream disconnects at each of §6.3.1's four injection points. Each builder
produces a deterministic byte sequence — no timing, no sleeps — built on the
``Reply`` API the recorder exposes.

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
        ValueError: When ``fmt`` is outside the two formats this library serves
            or ``point`` is unrecognised.
    """
    if fmt is WireFormat.ANTHROPIC_MESSAGES:
        return _frames_for_anthropic(point)
    if fmt is WireFormat.CHAT_COMPLETIONS:
        return _frames_for_cc(point)
    raise ValueError(
        f"this library serves {{ANTHROPIC_MESSAGES, CHAT_COMPLETIONS}}, not {fmt.value}"
    )


# ---------------------------------------------------------------------------
# Builders (F1, F2, F3, F4, F5, F8).
# ---------------------------------------------------------------------------


#: The two formats this library serves — the primary recorder's served set.
_SERVED_FORMATS: frozenset[WireFormat] = frozenset(
    {WireFormat.ANTHROPIC_MESSAGES, WireFormat.CHAT_COMPLETIONS}
)


def _check_fmt(fmt: WireFormat) -> None:
    """Reject a format outside the library's scope at construction time (R3.8).

    Args:
        fmt: The format the caller asked for.

    Raises:
        ValueError: When ``fmt`` is not one of the two formats §7.2 assigns to
            the primary recorder, or is not a :class:`WireFormat` at all. The
            message reads ``fmt.value`` only after the type check, so a
            non-enum caller sees this library's ``ValueError`` rather than an
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

    if stream:
        chunks = minimal_success_stream(fmt)

        async def responder(captured: CapturedRequest, response: Reply) -> None:
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in chunks:
                await response.write(chunk)
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
