"""The scripted failure library — builders, helpers and their falsifications.

`.system_design/TEST_SUITE.md` §7.2 · plan task **T-B4** (KBR-43).

The library in :mod:`harness.failures` builds :class:`~harness.recorder.Reply`
closures for the failure shapes the bridge recognises, the four §6.3.1 injection
points, and a per-attempt sequencer — each forced deterministically by a pure
scripted event sequence, no timing. These tests drive the responders through the
real :class:`~harness.recorder.RecordingUpstream` and assert what landed on the
wire. The library's bytes-vs-oracle contract is verified against the bridge's
own failure detectors, which the test (not the library) imports as oracles per
F9's independence rule.

**Layer.** No ``pytestmark``: harness tests default to ``l1`` by path. §8.2
governs — no ``l3`` until T-K6 activates the Subsystem job.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path

import pytest

import harness.failures as failures_module
from harness.contract import CapturedRequest, WireFormat
from harness.recorder import RecordingUpstream, Reply
from harness.test_contract import _KITTY_IMPORT

#: Reply timeout per exchange. 2 s is comfortably above the §7.2 measured
#: transport-close propagation (hundreds of ms in the BEFORE_FIRST_BYTE probe)
#: and well below the l1 gate's tolerance for a stuck test.
_REPLY_TIMEOUT = 2.0

#: More body than a loopback socket accepts in one non-blocking write while its
#: reader is held back — measured on Linux, where the kernel took under 4 MiB —
#: so some of it is still queued in the transport when the abort runs, the state
#: Windows' Proactor loop is in after every write. If a platform ever accepts it
#: all, the falsification's vacuous-guard fails loudly rather than passing
#: vacuously. Matches :data:`tests.harness.test_recorder._OVERRUN_BYTES`.
_OVERRUN_BYTES = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Local exchange helpers. Modelled on ``tests.harness.test_recorder``'s private
# ``_exchange_blocking`` and ``_read_to_eof_blocking``. Defined here rather than
# imported across test files because every harness test owns its own helpers —
# a shared ``tests/harness/_http.py`` would be its own module to register, lint
# and reason about, which is more than four test files need.
# ---------------------------------------------------------------------------


async def _exchange(recorder: RecordingUpstream, raw: bytes) -> bytes:
    """Send ``raw`` and read until the reply is complete *or* the socket closes.

    The blocking half runs in a worker thread: the recorder under test runs on
    the **same event loop** as the test that drives it, and an inline blocking
    ``recv`` would starve the loop so the handler could never answer — the
    failure mode `recorder_conformance.send` documents and the reason that
    driver exists.

    An empty-body reply (e.g. ``drop_at(BEFORE_FIRST_BYTE)``) returns ``b""``.
    An unfinished reply (e.g. ``drop_at(AFTER_TEXT)``) returns what arrived
    before the connection closed; the absence of a chunked terminator or
    ``[DONE]`` is what tells the calling test this row's drop fired.
    """
    return await asyncio.to_thread(
        _exchange_blocking, recorder.host, recorder.port, raw
    )


def _exchange_blocking(host: str, port: int, raw: bytes) -> bytes:
    """Blocking half of :func:`_exchange`. Exits on completion or socket EOF."""
    sock = socket.create_connection((host, port), timeout=_REPLY_TIMEOUT)
    try:
        sock.sendall(raw)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = sock.recv(65536)
            except (TimeoutError, OSError):
                break
            if not chunk:
                break
            chunks.append(chunk)
            if _reply_is_complete(b"".join(chunks)):
                break
    finally:
        sock.close()
    return b"".join(chunks)


def _read_to_eof_blocking(
    host: str, port: int, raw: bytes, released: threading.Event
) -> bytes:
    """Send ``raw``, wait for ``released`` to be set, then read to EOF.

    For the overrun falsification: the responder queues more body than the
    socket accepts, sets ``released`` only after the abort has run, and the
    reader must not start draining until then — otherwise the queue's size is a
    race between the writer and the reader, and the test's vacuous-guard
    ("bytes were still queued at abort") is unfalsifiable.
    """
    sock = socket.create_connection((host, port), timeout=_REPLY_TIMEOUT)
    try:
        sock.sendall(raw)
        released.wait(timeout=_REPLY_TIMEOUT)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = sock.recv(65536)
            except (TimeoutError, OSError):
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()
    return b"".join(chunks)


def _reply_is_complete(reply: bytes) -> bool:
    """A reply counts as complete when its headers ended with a chunked EOF.

    Used by :func:`_exchange_blocking` to break early on a finished reply so
    the ``Connection: keep-alive`` default does not stall every test reading
    to EOF.
    """
    head, _, body = reply.partition(b"\r\n\r\n")
    # Heuristic: any chunked EOF sequence in the body.
    if b"\r\n0\r\n\r\n" in body:
        return True
    # A Content-Length body of the declared size is also complete.
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            try:
                declared = int(line.split(b":", 1)[1].strip())
            except ValueError:
                return False
            return len(body) >= declared
    return False


def _status_line(reply: bytes) -> tuple[int, str]:
    """Return ``(status, reason)`` parsed from the first line of ``reply``."""
    line, _, _ = reply.partition(b"\r\n")
    parts = line.split(b" ", 2)
    return int(parts[1]), parts[2].decode("latin-1") if len(parts) > 2 else ""


def _headers(reply: bytes) -> dict[str, str]:
    """Parse the headers from ``reply`` into a case-insensitive dict."""
    head, _, _ = reply.partition(b"\r\n\r\n")
    out: dict[str, str] = {}
    for line in head.split(b"\r\n")[1:]:
        name, sep, value = line.decode("latin-1").partition(":")
        if sep:
            out[name.strip().lower()] = value.strip()
    return out


def _body(reply: bytes) -> bytes:
    """Return the entity body after the header terminator."""
    _, _, body = reply.partition(b"\r\n\r\n")
    return body


def _sse_events(body: bytes) -> list[tuple[str, dict]]:
    """Split an Anthropic-Messages SSE body into ``(event, data)`` pairs.

    A hand-rolled parser rather than a call into a library: this is the
    reference the library's frames are judged against, so it must not share an
    implementation with the thing under test.
    """
    events: list[tuple[str, dict]] = []
    for frame in body.split(b"\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event_name = ""
        data_str = ""
        for line in frame.split(b"\n"):
            if line.startswith(b"event: "):
                event_name = line[len(b"event: "):].decode("latin-1")
            elif line.startswith(b"data: "):
                data_str = line[len(b"data: "):].decode("latin-1")
        if event_name and data_str:
            events.append((event_name, json.loads(data_str)))
    return events


def _cc_chunks(body: bytes) -> list[dict]:
    """Split a Chat Completions SSE body into ``data:`` chunks."""
    chunks: list[dict] = []
    for frame in body.split(b"\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        for line in frame.split(b"\n"):
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                chunks.append(json.loads(line[len(b"data: "):].decode("latin-1")))
    return chunks


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
async def recorder():
    """A fresh ``RecordingUpstream`` per test, defaulting to CHAT_COMPLETIONS.

    Some tests rebind ``recorder.responder`` to the shape under test; the
    default responder is the recorder's own minimal success. The fixture
    calls :meth:`~harness.recorder.RecordingUpstream.assert_all_paths_matched`
    at teardown so a wrong-format reply (a path with no matching suffix)
    fails loudly rather than being read as an empty response and retried —
    test_recorder.py's fixture does the same.
    """
    upstream = RecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()
        upstream.assert_all_paths_matched()


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestSseFrame:
    """AC-16 — `sse_frame` and `cc_chunk` build their wire's SSE frames."""

    def test_anthropic_frame_is_event_then_data(self) -> None:
        """`sse_frame("message_start", {...})` returns bytes in the Anthropic grammar."""
        frame = failures_module.sse_frame("message_start", {"type": "message_start"})
        assert frame == b"event: message_start\ndata: {\"type\": \"message_start\"}\n\n"

    def test_anthropic_frame_string_data_is_used_verbatim(self) -> None:
        """A string ``data`` is emitted without JSON re-encoding."""
        frame = failures_module.sse_frame("content_block_delta", '{"delta": {"text": "x"}}')
        assert b'data: {"delta": {"text": "x"}}' in frame

    def test_cc_chunk_is_data_only(self) -> None:
        """`cc_chunk({...})` carries only a `data:` line — no `event:`."""
        frame = failures_module.cc_chunk({"choices": [{"delta": {"role": "assistant"}}]})
        assert frame.startswith(b"data: ")
        assert b"event:" not in frame

    def test_cc_done_is_the_literal_sentinel(self) -> None:
        """`CC_DONE` is the upstream's literal terminator."""
        assert failures_module.CC_DONE == b"data: [DONE]\n\n"


class TestErrorStatus:
    """AC-2, AC-3, AC-4 — `error_status` writes the format's native error body."""

    def test_an_unsupported_format_raises(self) -> None:
        """The library serves four formats — KBR-302 widened {GEMINI, OPENAI_RESPONSES} in —
        and the remaining two (`BEDROCK_CONVERSE`, `OLLAMA_CHAT`) still raise.

        The served set is not the recorder's two-format set since KBR-302:
        `frames_for` and every builder name the four wire grammars the
        library's own builders spell. The two formats that carry no inbound
        route at all (§`InboundProtocol`'s gap) stay out of scope and the
        served-set `ValueError` is the pin that keeps them out.
        """
        with pytest.raises(ValueError, match="bedrock_converse"):
            failures_module.error_status(
                WireFormat.BEDROCK_CONVERSE, 500, error_type="x", message="y"
            )
        with pytest.raises(ValueError, match="ollama_chat"):
            failures_module.error_status(
                WireFormat.OLLAMA_CHAT, 500, error_type="x", message="y"
            )

    async def test_anthropic_messages_writes_the_native_envelope(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-2 — Anthropic error shape, status, and content-type."""
        recorder.responder = failures_module.error_status(
            WireFormat.ANTHROPIC_MESSAGES,
            529,
            error_type="overloaded_error",
            message="busy",
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        status, _ = _status_line(reply)
        assert status == 529
        assert _headers(reply).get("content-type") == "application/json"
        body = json.loads(_body(reply))
        assert body == {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "busy"},
        }

    async def test_chat_completions_writes_the_native_envelope(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-3 — Chat Completions error shape with the numeric code."""
        recorder.responder = failures_module.error_status(
            WireFormat.CHAT_COMPLETIONS,
            500,
            error_type="server_error",
            message="oops",
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        status, _ = _status_line(reply)
        assert status == 500
        body = json.loads(_body(reply))
        assert body == {
            "error": {"message": "oops", "type": "server_error", "code": "500"},
        }

    async def test_gemini_writes_the_native_envelope(
        self, recorder: RecordingUpstream
    ) -> None:
        """KBR-302 — the Gemini error envelope carries code/message/status.

        Google's documented error shape (and the bridge's Gemini reader)
        carries a numeric ``code`` and a string ``status`` — the enum value
        (``"INVALID_ARGUMENT"`` etc.), not a free-text message, is what
        ``error_type`` carries for this format.
        """
        recorder.responder = failures_module.error_status(
            WireFormat.GEMINI,
            400,
            error_type="INVALID_ARGUMENT",
            message="prompt too long",
        )
        reply = await _exchange(
            recorder,
            b"POST /v1beta/models/m:generateContent HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        status, _ = _status_line(reply)
        assert status == 400
        body = json.loads(_body(reply))
        assert body == {
            "error": {"code": 400, "message": "prompt too long", "status": "INVALID_ARGUMENT"},
        }

    async def test_openai_responses_writes_the_native_envelope(
        self, recorder: RecordingUpstream
    ) -> None:
        """KBR-302 — the Responses API's non-streaming error envelope.

        The Responses API's non-streaming errors mirror Chat Completions'
        ``{"error": {...}}`` triple; ``code`` stays the stringified status.
        """
        recorder.responder = failures_module.error_status(
            WireFormat.OPENAI_RESPONSES,
            429,
            error_type="rate_limit_error",
            message="slow down",
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/responses HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        status, _ = _status_line(reply)
        assert status == 429
        body = json.loads(_body(reply))
        assert body == {
            "error": {"message": "slow down", "type": "rate_limit_error", "code": "429"},
        }


class TestCloudflareBlock:
    """AC-5 — `cloudflare_block` writes a 403 the bridge's detector classifies."""

    @pytest.mark.parametrize("fmt", [WireFormat.ANTHROPIC_MESSAGES, WireFormat.CHAT_COMPLETIONS])
    async def test_the_bridge_detector_classifies_it_as_a_block(
        self, recorder: RecordingUpstream, fmt: WireFormat
    ) -> None:
        """F2.c — the block is identical for both formats; the detector trips on both.

        The bridge's `is_cloudflare_block` is the oracle; the library never
        imports it (F9). The path suffix differs by format — `/v1/messages` for
        Anthropic, `/v1/chat/completions` for CC — so both raw requests are
        sent.
        """
        from kitty.cloudflare import is_cloudflare_block

        path = b"/v1/messages" if fmt is WireFormat.ANTHROPIC_MESSAGES else b"/v1/chat/completions"
        recorder.responder = failures_module.cloudflare_block(fmt)
        reply = await _exchange(
            recorder,
            b"POST " + path + b" HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        status, _ = _status_line(reply)
        body_text = _body(reply).decode("latin-1")
        assert status == 403
        assert _headers(reply).get("content-type") == "text/html"
        assert is_cloudflare_block(403, body_text) is True


class TestContextTooLarge:
    """AC-6, AC-7 — `context_too_large` writes a body the detector classifies."""

    @pytest.mark.parametrize(
        "fmt,path",
        [
            (WireFormat.ANTHROPIC_MESSAGES, b"/v1/messages"),
            (WireFormat.CHAT_COMPLETIONS, b"/v1/chat/completions"),
        ],
    )
    async def test_status_413_is_classified_independent_of_body(
        self, recorder: RecordingUpstream, fmt: WireFormat, path: bytes
    ) -> None:
        """AC-6 — 413 is the detector's status-only path (both formats)."""
        from kitty.bridge.server import BridgeServer

        recorder.responder = failures_module.context_too_large(fmt, status=413)
        reply = await _exchange(
            recorder,
            b"POST " + path + b" HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        status, _ = _status_line(reply)
        body = json.loads(_body(reply))
        assert status == 413
        assert BridgeServer._is_context_too_large_error(413, body) is True

    @pytest.mark.parametrize(
        "fmt,path",
        [
            (WireFormat.ANTHROPIC_MESSAGES, b"/v1/messages"),
            (WireFormat.CHAT_COMPLETIONS, b"/v1/chat/completions"),
        ],
    )
    async def test_status_400_with_a_needle_substring_is_classified(
        self, recorder: RecordingUpstream, fmt: WireFormat, path: bytes
    ) -> None:
        """AC-7 — 400 + the `maximum context` substring trips the detector (both formats)."""
        from kitty.bridge.server import BridgeServer

        recorder.responder = failures_module.context_too_large(fmt)
        reply = await _exchange(
            recorder,
            b"POST " + path + b" HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = json.loads(_body(reply))
        assert BridgeServer._is_context_too_large_error(400, body) is True


class TestEmptyResponse:
    """AC-8, AC-9 — `empty_response` writes a contentless success."""

    async def test_anthropic_messages_non_streaming_has_empty_content(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-8 — non-streaming Messages: `content: []` with `stop_reason: end_turn`."""
        recorder.responder = failures_module.empty_response(WireFormat.ANTHROPIC_MESSAGES)
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = json.loads(_body(reply))
        assert body["content"] == []
        assert body["stop_reason"] == "end_turn"

    async def test_chat_completions_non_streaming_has_empty_content(
        self, recorder: RecordingUpstream
    ) -> None:
        """F4.b — non-streaming CC: empty `message.content` with `finish_reason: stop`."""
        recorder.responder = failures_module.empty_response(WireFormat.CHAT_COMPLETIONS)
        reply = await _exchange(
            recorder,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = json.loads(_body(reply))
        choice = body["choices"][0]
        assert choice["message"]["content"] == ""
        assert choice["finish_reason"] == "stop"

    async def test_chat_completions_streaming_is_a_content_less_completion(
        self, recorder: RecordingUpstream
    ) -> None:
        """F4.d — streaming CC: role chunk + [DONE], the content-less completion.

        Since KBR-276 this shape is what fires the empty ladder on the raw-CC
        route: the pre-emission hold withholds non-content lines on every CC
        upstream, so the role chunk no longer sets the bridge's `has_content`
        flag (KBR-232's converter-gated semantics are gone). The test pins the
        shape the ladder judges — a role-only chunk, no content delta, no
        finish chunk, terminated by [DONE]; the ladder itself is exercised
        against a real bridge in ``tests/bridge/test_raw_cc_empty_hold.py``.
        """
        recorder.responder = failures_module.empty_response(
            WireFormat.CHAT_COMPLETIONS, stream=True
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = _body(reply)
        chunks = _cc_chunks(body)
        assert len(chunks) == 1
        assert "role" in chunks[0]["choices"][0]["delta"]
        # No content delta beside the role, and the reply terminates with [DONE].
        # The body arrives chunked-encoded, so [DONE] is contained rather than
        # terminal — the final chunk is the `0\r\n\r\n` framing itself.
        assert "content" not in chunks[0]["choices"][0]["delta"]
        assert b"data: [DONE]\n\n" in body

    async def test_anthropic_messages_streaming_releases_nothing(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-9 — streaming Messages: no content event releases the preamble hold.

        A contentless Messages stream reaches the bridge's preamble-hold judge
        without releasing: there is no non-empty ``text_delta`` (D1), no
        non-thinking ``content_block_start`` with content, and the stream ends
        via the ladder. The builder writes ``message_start`` then an empty text
        block (no delta releases it), a stop, a ``message_delta`` and a
        ``message_stop`` — the reply terminates, not drops.
        """
        recorder.responder = failures_module.empty_response(
            WireFormat.ANTHROPIC_MESSAGES, stream=True
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [name for name, _ in events]
        # AC-9: the empty Messages stream carries both message_start and
        # message_stop. The four intermediate events are F4.c's documented shape:
        # an empty text block (no delta), a stop, a message_delta, then a stop.
        # Lock the whole sequence — a regression that dropped message_start, or
        # added a content delta, would otherwise pass.
        assert names == [
            "message_start",
            "content_block_start",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ], names
        # No non-empty text_delta. An empty text_delta (or one whose type is not
        # text_delta) is allowed — D1 says a non-empty text_delta is what would
        # release; its absence is the whole point.
        for _, payload in events:
            if payload.get("type") == "content_block_delta":
                is_text = payload["delta"].get("type") == "text_delta"
                has_text = bool(payload["delta"].get("text"))
                assert not (is_text and has_text), (
                    f"content_block_delta released the preamble hold: {payload!r}"
                )

    async def test_gemini_streaming_is_a_content_less_completion(
        self, recorder: RecordingUpstream
    ) -> None:
        """KBR-302 — streaming Gemini: role chunk + STOP finish, no text part.

        The Gemini content-less completion is what fires the empty ladder on
        `_stream_gemini`: the role chunk carries no `text` part, the finish
        chunk carries `finishReason: "STOP"`. The reader's `has_content` keys
        on a non-empty `parts[*].text`; its absence is the trigger.
        """
        recorder.responder = failures_module.empty_response(
            WireFormat.GEMINI, stream=True
        )
        reply = await _exchange(
            recorder,
            b"POST /v1beta/models/m:streamGenerateContent HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = _body(reply)
        # Two data-only lines: the role chunk and the STOP finish chunk.
        assert b"data: " in body
        assert b"event:" not in body  # Gemini is data-only.
        # No text part anywhere — the bridge's Gemini reader treats this as empty.
        assert b'"text"' not in body
        # The finish chunk carries STOP — the terminal reason.
        assert b'"finishReason": "STOP"' in body

    async def test_openai_responses_streaming_is_a_content_less_completion(
        self, recorder: RecordingUpstream
    ) -> None:
        """KBR-302 — streaming Responses: created + completed, no output_text delta.

        The Responses content-less completion is response.created followed by
        response.completed — no response.output_text.delta event. The reader's
        `has_content` keys on the presence of an output_text delta; its absence
        is the empty-ladder trigger on `_stream_responses`.
        """
        recorder.responder = failures_module.empty_response(
            WireFormat.OPENAI_RESPONSES, stream=True
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/responses HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [name for name, _ in events]
        assert names == ["response.created", "response.completed"], names
        # The completion event carries status=completed — the success terminal.
        completed = events[-1][1]["response"]
        assert completed["status"] == "completed"
        # No output_text delta between the two events — the whole shape is two.
        text_events = [
            name for name, _ in events
            if name in ("response.output_text.delta", "response.output_text.done")
        ]
        assert text_events == [], text_events

    def test_gemini_non_streaming_empty_body_is_one_candidate_with_no_parts(self) -> None:
        """KBR-302 — the Gemini non-streaming empty body (recorder path not used)."""
        body = failures_module._gemini_empty_success_body()
        assert body["candidates"][0]["content"]["parts"] == []
        assert body["candidates"][0]["finishReason"] == "STOP"

    def test_openai_responses_non_streaming_empty_body_is_completed_with_empty_output(self) -> None:
        """KBR-302 — the Responses non-streaming empty body (recorder path not used)."""
        body = failures_module._responses_empty_success_body()
        assert body["status"] == "completed"
        assert body["output"] == []

    def test_gemini_non_streaming_success_body_carries_text_and_stop(self) -> None:
        """KBR-302 — the Gemini non-streaming minimal success (recorder path not used)."""
        body = failures_module._gemini_content_success_body()
        parts = body["candidates"][0]["content"]["parts"]
        assert parts == [{"text": "kbr-tb4"}]
        assert body["candidates"][0]["finishReason"] == "STOP"

    def test_openai_responses_non_streaming_success_body_carries_one_message_output(self) -> None:
        """KBR-302 — the Responses non-streaming minimal success (recorder path not used)."""
        body = failures_module._responses_content_success_body()
        assert body["status"] == "completed"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "kbr-tb4"


class TestDropAt:
    """AC-10..14 — `drop_at` writes the frames its injection point names, then aborts."""

    async def test_before_first_byte_writes_nothing(
        self, recorder: RecordingUpstream, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC-10 / F5.b — abort-before-prepare delivers zero bytes + EOF, no handler error.

        The abort runs before ``begin()``/``prepare()``, a shape no earlier
        recorder test exercised; F5.b requires the first drive of it to prove
        aiohttp's ``finish_response`` path does not surface an unhandled error
        for the unprepared response. The hand probe (2026-09-14) found none —
        this assertion pins that, so a future aiohttp pin bump that starts
        erroring here is visible instead of silent.
        """
        import logging

        recorder.responder = failures_module.drop_at(
            WireFormat.ANTHROPIC_MESSAGES, failures_module.InjectionPoint.BEFORE_FIRST_BYTE
        )
        with caplog.at_level(logging.ERROR):
            reply = await _exchange(
                recorder,
                b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
            )
        assert reply == b""
        handler_errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert handler_errors == [], (
            f"aiohttp surfaced an error for the unprepared abort: {handler_errors}"
        )

    async def test_anthropic_messages_after_text_writes_four_events(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-11 — `AFTER_TEXT` writes 4 events then aborts (no message_delta/stop)."""
        recorder.responder = failures_module.drop_at(
            WireFormat.ANTHROPIC_MESSAGES, failures_module.InjectionPoint.AFTER_TEXT
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [name for name, _ in events]
        assert names == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ]

    async def test_anthropic_messages_mid_tool_arguments_leaves_block_open(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-12 — partial `input_json_delta`, no `content_block_stop`, no terminator."""
        recorder.responder = failures_module.drop_at(
            WireFormat.ANTHROPIC_MESSAGES, failures_module.InjectionPoint.MID_TOOL_ARGUMENTS
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [name for name, _ in events]
        assert "content_block_stop" not in names
        assert "message_delta" not in names
        assert "message_stop" not in names
        deltas = [p for _, p in events if p.get("type") == "content_block_delta"]
        assert len(deltas) == 1
        assert deltas[0]["delta"]["type"] == "input_json_delta"
        partial = deltas[0]["delta"]["partial_json"]
        assert partial == '{"arg1":"v'
        with pytest.raises(json.JSONDecodeError):
            json.loads(partial)

    async def test_anthropic_messages_before_terminal_writes_through_message_delta(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-13 — everything through `message_delta`; no `message_stop`."""
        recorder.responder = failures_module.drop_at(
            WireFormat.ANTHROPIC_MESSAGES, failures_module.InjectionPoint.BEFORE_TERMINAL
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [name for name, _ in events]
        assert names == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
        ]
        assert "message_stop" not in names

    @pytest.mark.parametrize("point", list(failures_module.InjectionPoint))
    async def test_chat_completions_frames_match_the_point(
        self, recorder: RecordingUpstream, point: failures_module.InjectionPoint
    ) -> None:
        """AC-14 — `drop_at(CHAT_COMPLETIONS, ...)` writes in the CC grammar."""
        recorder.responder = failures_module.drop_at(WireFormat.CHAT_COMPLETIONS, point)
        reply = await _exchange(
            recorder,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = _body(reply)
        chunks = _cc_chunks(body)
        has_done = b"data: [DONE]\n\n" in body
        assert recorder.unmatched == []  # path matched a known suffix
        # AC-14: no CC frame carries an `event:` line — `_cc_chunks` skips
        # them, so this is asserted on the raw bytes, before any parsing.
        assert b"event:" not in body, f"a spurious event: line reached the CC wire: {body[:200]!r}"
        if point is failures_module.InjectionPoint.BEFORE_FIRST_BYTE:
            assert body == b""
        elif point is failures_module.InjectionPoint.AFTER_TEXT:
            assert len(chunks) == 2
            assert "role" in chunks[0]["choices"][0]["delta"]
            assert "content" in chunks[1]["choices"][0]["delta"]
            assert not has_done
        elif point is failures_module.InjectionPoint.MID_TOOL_ARGUMENTS:
            assert len(chunks) == 2
            tool = chunks[1]["choices"][0]["delta"].get("tool_calls")
            assert tool is not None
            assert tool[0]["function"]["arguments"] == '{"arg1":"v'
            assert not has_done
        elif point is failures_module.InjectionPoint.BEFORE_TERMINAL:
            assert len(chunks) == 3
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            assert not has_done

    @pytest.mark.parametrize("point", list(failures_module.InjectionPoint))
    def test_gemini_frames_match_the_point(
        self, point: failures_module.InjectionPoint
    ) -> None:
        """KBR-302 — `drop_at(GEMINI, ...)` writes in the Gemini grammar.

        The recorder only serves two formats (ANTHROPIC_MESSAGES,
        CHAT_COMPLETIONS); Gemini bytes can be inspected through the public
        :func:`frames_for` only — the F10/AC-18 falsification pattern
        (replay the same writes through a deliberately-broken responder
        without introspecting the library's closures). Data-only frames
        (no ``event:`` line), payload key ``candidates``.
        """
        frames = failures_module.frames_for(WireFormat.GEMINI, point)
        payloads = [
            json.loads(line[len(b"data: "):])
            for frame in frames
            for line in frame.split(b"\n\n")
            if line.startswith(b"data: ")
        ]
        for f_bytes in frames:
            assert b"event:" not in f_bytes, f"an `event:` line reached the Gemini wire: {f_bytes!r}"
        if point is failures_module.InjectionPoint.BEFORE_FIRST_BYTE:
            assert frames == ()
            return
        assert all("candidates" in p or "error" in p for p in payloads)
        if point is failures_module.InjectionPoint.AFTER_TEXT:
            texts = [
                part["text"]
                for p in payloads
                for cand in p.get("candidates", [])
                for part in cand.get("content", {}).get("parts", [])
                if "text" in part
            ]
            assert texts == ["kbr-tb4"], texts
            # No finishReason yet — the drop fired before the terminal.
            assert not any("finishReason" in p["candidates"][0] for p in payloads)
        elif point is failures_module.InjectionPoint.MID_TOOL_ARGUMENTS:
            calls = [
                part["functionCall"]
                for p in payloads
                for cand in p.get("candidates", [])
                for part in cand.get("content", {}).get("parts", [])
                if "functionCall" in part
            ]
            assert calls, "the mid-tool-arguments drop carries no functionCall part"
            assert calls[0]["args"] == {"arg1": "v"}
        elif point is failures_module.InjectionPoint.BEFORE_TERMINAL:
            # The terminal chunk carries STOP — the bridge's truncation reader keys on it.
            assert any(
                "finishReason" in cand
                for p in payloads
                for cand in p.get("candidates", [])
            )

    @pytest.mark.parametrize("point", list(failures_module.InjectionPoint))
    def test_openai_responses_frames_match_the_point(
        self, point: failures_module.InjectionPoint
    ) -> None:
        """KBR-302 — `drop_at(OPENAI_RESPONSES, ...)` writes in the Responses grammar.

        Event-framed SSE with the ``type`` key inside the payload. See the
        Gemini twin's comment for why the recorder path is not used.
        """
        frames = failures_module.frames_for(WireFormat.OPENAI_RESPONSES, point)
        if point is failures_module.InjectionPoint.BEFORE_FIRST_BYTE:
            assert frames == ()
            return
        events = [
            (name, data)
            for frame in frames
            for name, data in _sse_events(frame)
        ]
        names = [name for name, _ in events]
        assert names[0] == "response.created", names
        if point is failures_module.InjectionPoint.AFTER_TEXT:
            assert names == ["response.created", "response.output_text.delta"]
            assert events[1][1]["delta"] == "kbr-tb4"
        elif point is failures_module.InjectionPoint.MID_TOOL_ARGUMENTS:
            assert names == [
                "response.created",
                "response.function_call_arguments.delta",
            ]
            assert events[1][1]["delta"] == '{"arg1":"v'
        elif point is failures_module.InjectionPoint.BEFORE_TERMINAL:
            assert names == [
                "response.created",
                "response.output_text.delta",
                "response.completed",
            ]
            assert events[-1][1]["response"]["status"] == "completed"


class TestScripted:
    """AC-15 — `scripted(*replies)` dispatches by attempt and fails loudly on overrun."""

    async def test_two_attempts_dispatched_in_order(
        self, recorder: RecordingUpstream
    ) -> None:
        """First attempt → first reply; second attempt → second reply."""
        recorder.responder = failures_module.scripted(
            failures_module.drop_at(
                WireFormat.ANTHROPIC_MESSAGES,
                failures_module.InjectionPoint.AFTER_TEXT,
            ),
            failures_module.error_status(WireFormat.ANTHROPIC_MESSAGES, 529),
        )
        first = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        # First reply: drop — frames but no chunked terminator / no message_stop.
        assert b"event: " in first
        assert b"event: message_stop" not in first
        second = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        # Second reply: error_status 529.
        status, _ = _status_line(second)
        assert status == 529

    async def test_third_attempt_raises_script_exhausted(self) -> None:
        """F7.c — drive the responder coroutine directly with a stub Reply."""
        responder = failures_module.scripted(
            failures_module.error_status(WireFormat.ANTHROPIC_MESSAGES, 529),
        )

        class _StubReply:
            """Just enough of `Reply` for the responder to be called once."""

            async def begin(
                self, status: int, headers: dict | None = None
            ) -> None:
                pass

            async def write(self, chunk: bytes) -> None:
                pass

            async def write_eof(self) -> None:
                pass

            async def abort(self) -> None:
                pass

        captured = CapturedRequest(
            method="POST", scheme="http", host="p", path="/v1/messages", query=""
        )

        async def drive_once() -> None:
            await responder(captured, _StubReply())  # type: ignore[arg-type]

        await drive_once()  # first attempt: succeeds
        with pytest.raises(failures_module.ScriptExhausted):
            await drive_once()  # second attempt: raises


class TestSuccess:
    """AC-17 — `success(fmt, *, stream=False)` returns the recorder's minimal shape."""

    async def test_anthropic_messages_success_delegates_to_minimal_body(
        self, recorder: RecordingUpstream
    ) -> None:
        """Non-streaming success is the recorder's minimal Messages body."""
        from harness.recorder import minimal_success_body

        recorder.responder = failures_module.success(WireFormat.ANTHROPIC_MESSAGES)
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = json.loads(_body(reply))
        # The recorder's helper is the source of truth — compare structurally.
        assert body == minimal_success_body(WireFormat.ANTHROPIC_MESSAGES)

    async def test_chat_completions_success_delegates_to_minimal_body(
        self, recorder: RecordingUpstream
    ) -> None:
        """Non-streaming success is the recorder's minimal CC body."""
        from harness.recorder import minimal_success_body

        recorder.responder = failures_module.success(WireFormat.CHAT_COMPLETIONS)
        reply = await _exchange(
            recorder,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = json.loads(_body(reply))
        assert body == minimal_success_body(WireFormat.CHAT_COMPLETIONS)

    async def test_anthropic_messages_streaming_success_releases_the_hold(
        self, recorder: RecordingUpstream
    ) -> None:
        """AC-17 — streaming success carries non-empty text, ending in `message_stop`."""
        recorder.responder = failures_module.success(
            WireFormat.ANTHROPIC_MESSAGES, stream=True
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [n for n, _ in events]
        assert "message_stop" in names
        text_deltas = [
            p["delta"]["text"]
            for _, p in events
            if p.get("type") == "content_block_delta"
            and p["delta"].get("type") == "text_delta"
        ]
        assert any(t for t in text_deltas)

    async def test_gemini_streaming_success_emits_a_text_delta_then_stop(
        self, recorder: RecordingUpstream
    ) -> None:
        """KBR-302 — the minimal Gemini success stream carries one text delta and STOP.

        `success()` for Gemini inlines its own minimal shape — the recorder
        deliberately stays two-format. The shape is role chunk + text-bearing
        chunk + STOP finish chunk, all data-only.
        """
        recorder.responder = failures_module.success(WireFormat.GEMINI, stream=True)
        reply = await _exchange(
            recorder,
            b"POST /v1beta/models/m:streamGenerateContent HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        body = _body(reply)
        # Data-only — no `event:` lines on the Gemini wire.
        assert b"event:" not in body
        # The text-bearing chunk lands.
        assert b'"text":' in body
        # The terminal chunk carries STOP — the success reason.
        assert b'"finishReason": "STOP"' in body

    async def test_openai_responses_streaming_success_emits_text_delta_then_completed(
        self, recorder: RecordingUpstream
    ) -> None:
        """KBR-302 — the minimal Responses success stream carries a text delta and `completed`."""
        recorder.responder = failures_module.success(
            WireFormat.OPENAI_RESPONSES, stream=True
        )
        reply = await _exchange(
            recorder,
            b"POST /v1/responses HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
        )
        events = _sse_events(_body(reply))
        names = [name for name, _ in events]
        assert names == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ], names
        completed = events[-1][1]["response"]
        assert completed["status"] == "completed"


class TestFalsification:
    """AC-18 / F10 — `Reply.abort()` delivers what was written; `transport.abort()` does not."""

    async def test_overrun_with_reply_abort_delivers_every_byte(
        self, recorder: RecordingUpstream
    ) -> None:
        """The correct `drop_at` abort path (Reply.abort) flushes the queued bytes.

        The frames are sourced from the library's public ``frames_for`` so
        the responder and the test do not duplicate the wire grammar — a
        future change to ``drop_at`` that bypasses ``frames_for`` is visible
        here by the byte-count, but a regression that breaks ``frames_for``
        itself would affect both sides equally and pass this test; the test
        therefore catches the abort semantics, not the wire grammar.
        """
        queued: list[int] = []
        released = threading.Event()

        async def overrun(captured: CapturedRequest, response: Reply) -> None:
            """AFTER_TEXT frames + a 16 MiB overrun body, then Reply.abort()."""
            try:
                await response.begin(200, {"Content-Type": "text/event-stream"})
                for frame in failures_module.frames_for(
                    WireFormat.ANTHROPIC_MESSAGES,
                    failures_module.InjectionPoint.AFTER_TEXT,
                ):
                    await response.write(frame)
                # Onto the transport directly: past 64 KiB aiohttp's own write
                # waits for the queue to fall to its low-water mark.
                transport = response.request.transport
                assert transport is not None
                transport.write(b"x" * _OVERRUN_BYTES)
                queued.append(transport.get_write_buffer_size())
                await response.abort()
            finally:
                released.set()

        recorder.responder = overrun
        reply = await asyncio.to_thread(
            _read_to_eof_blocking,
            recorder.host,
            recorder.port,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
            released,
        )
        assert queued and queued[0] > 0, (
            "vacuous: nothing was still queued at abort time, so this run could "
            "not tell a flushing abort from a discarding one"
        )
        body = reply.partition(b"\r\n\r\n")[2]
        assert body.endswith(b"x" * _OVERRUN_BYTES), (
            "Reply.abort() must flush every queued byte before dropping"
        )

    async def test_overrun_with_transport_abort_loses_queued_bytes(
        self, recorder: RecordingUpstream
    ) -> None:
        """The KBR-189 defect — `transport.abort()` discards the queue.

        A ``drop_at``-shaped responder that uses ``transport.abort()`` instead
        of ``Reply.abort()`` (the KBR-189 defect) loses the queued overrun
        body. The library never imports ``transport.abort()``; this responder
        is hand-rolled here so a future regression that reintroduces the bug is
        caught by the byte-count assertion below.

        **The shape of the loss is platform-shaped (§7.2 / KBR-189).** On
        Linux and macOS writes reach the kernel immediately, so the SSE frames
        aiohttp wrote arrive at the client and only the queued overrun is
        lost. On Windows' Proactor loop every write — headers and frames
        included — is still queued at the abort, so the client receives
        nothing at all. Both outcomes prove the defect; what must hold on
        every leg is that the overrun did NOT survive the abort. Skipping the
        platform branches would either let the Windows defect pass (the bug
        is *more* severe there) or require Windows-specific code that adds no
        fidelity the design does not already document.
        """
        import sys

        queued: list[int] = []
        released = threading.Event()

        async def overrun_broken(captured: CapturedRequest, response: Reply) -> None:
            try:
                await response.begin(200, {"Content-Type": "text/event-stream"})
                for frame in failures_module.frames_for(
                    WireFormat.ANTHROPIC_MESSAGES,
                    failures_module.InjectionPoint.AFTER_TEXT,
                ):
                    await response.write(frame)
                transport = response.request.transport
                assert transport is not None
                transport.write(b"x" * _OVERRUN_BYTES)
                queued.append(transport.get_write_buffer_size())
                # The KBR-189 defect — transport.abort() discards the queue.
                transport.abort()
            finally:
                released.set()

        recorder.responder = overrun_broken
        reply = await asyncio.to_thread(
            _read_to_eof_blocking,
            recorder.host,
            recorder.port,
            b"POST /v1/messages HTTP/1.1\r\nHost: p\r\nContent-Length: 2\r\n\r\n{}",
            released,
        )
        assert queued and queued[0] > 0, (
            "vacuous: nothing was still queued, so this run could not tell a "
            "flushing abort from a discarding one"
        )
        body = reply.partition(b"\r\n\r\n")[2]
        # The common assertion (every leg): the overrun did NOT survive.
        assert not body.endswith(b"x" * _OVERRUN_BYTES), (
            "transport.abort() must lose the queued bytes (the KBR-189 defect)"
        )
        if sys.platform == "win32":
            # Windows' Proactor loop held everything; the abort lost it all.
            assert body == b"", (
                "expected the Proactor-queued reply to be discarded in full"
            )
        else:
            # On selector loops the frames aiohttp wrote had already reached
            # the kernel; only the transport-level overrun queue is lost.
            assert b"event: message_start" in body
            assert b"event: content_block_stop" in body


# ---------------------------------------------------------------------------
# Independence guard (F9 / AC-19).
# ---------------------------------------------------------------------------


def test_the_failures_library_imports_nothing_from_kitty() -> None:
    """Structural guarantee: the failure library must not reach into `src/kitty`.

    The test imports the bridge's detectors; the library must not. A library
    that did would prove self-consistency, not fidelity (§3.3.1).
    """
    source = failures_module.__file__
    assert source is not None
    body = Path(source).read_text(encoding="utf-8")
    # The library's own "read no meaningful source" guard: an empty string
    # would pass vacuously (house rule, tests/test_egress_coverage.py).
    assert len(body) > 1000, "read no meaningful source; the guard would pass vacuously"
    offending = [line.strip() for line in body.splitlines() if _KITTY_IMPORT.search(line)]
    assert offending == [], f"failures.py must not import kitty: {offending}"
