"""Streaming recovery — content guarantees, not just grammar (KBR-99 / T-I7).

``.system_design/TEST_SUITE.md`` §6.3.1, ``SYSTEM_DESIGN.md`` §5.3 S10–S13,
and the four product fixes the product owner decided 2026-09-21. The grid
asserts §6.3.1's four content points (no duplicated text, no tool-call id
reused across attempts, no arguments spliced from two attempts, exactly one
terminal outcome) per route. The drivers are the sibling
``test_post_emission_no_failover.py`` harness — a local scripted aiohttp
upstream — because ``tests/harness/failures.py`` only serves the two wire
formats T-B4 covered and does not build the shapes the grid and the four
fixes need (in-stream errors, finish chunks with a chosen stop reason,
pre-content ``error`` events, timeouts). Its ``frames_for`` byte sequences
are reused where the served formats cover the cell, so the drop bytes stay
in lock-step with T-B4's library. Each fix lands as one commit with its
regression test (red at base, green after — ``TEST_SUITE.md`` §16's
atomic-PR rule).

Per the corrected injection-classes table (REQUIREMENTS.md, empirically
verified 2026-09-21): the translated route's D2 residue
(``end_turn`` + ``message_stop``, no error) holds for transport drops that
land before any finish chunk is read — AFTER_TEXT and MID_TOOL_ARGUMENTS;
the BEFORE_TERMINAL cell is T-G7's genuinely two-way shape set
``{complete_sentence, truncated}`` because the finish chunk's streaming
path auto-resets the translator (``messages/translator.py:1037``), so
``finalize_interrupted_stream()`` has nothing to close and the transport
branch falls to its error-event fallback.

Each failure is forced by scripted bytes or the read timeout — no test
sleeps on timing. The full ``fast_stall``/``short_grace`` patch set keeps
every ladder below the client timeout, so tests count attempts, never time
them.
"""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from harness.contract import WireFormat
from harness.failures import InjectionPoint, frames_for

from kitty.bridge import server as server_module
from kitty.types import BridgeProtocol

from .test_client_disconnect_health import (
    _StubLauncher,
    _StubProvider,
    short_grace,  # noqa: F401 — pytest fixture, used by name
)
from .test_post_emission_no_failover import (
    _CC_CONTENT_CHUNK,
    _build,
    _post_stream,
    _Upstream,
)

pytestmark = pytest.mark.l1


@pytest.fixture
def fast_stall(monkeypatch: pytest.MonkeyPatch, short_grace: None) -> None:  # noqa: F811
    """Shrink the read timeout and the backoff base.

    ``short_grace`` already collapses the transport grace and the empty-ladder
    final delays. This fixture adds the two stream-timeout shrinks the
    empty-response rows need so the client never sees a ``TransportTimeout``
    envelope before the bridge's own error body lands.
    """
    monkeypatch.setattr(server_module, "_STREAM_READ_TIMEOUT", 1)
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.001)


def _balancing_protocol_server(base_url: str, launcher, *, native: bool = False):
    """Build a two-backend balancing BridgeServer with a protocol-specific launcher.

    ``_build`` hardcodes the Messages launcher; the Chat Completions and
    Gemini routes need their own protocol so the server registers those
    inbound paths.

    Args:
        base_url: The upstream base URL both backends target.
        launcher: The launcher adapter declaring the inbound protocol.
        native: Whether the providers speak the Messages wire natively.

    Returns:
        The unstarted server.
    """
    import uuid

    from kitty.bridge.server import BridgeServer
    from kitty.profiles.schema import Profile

    backends = []
    for i in range(2):
        provider = _StubProvider(base_url, native=native)
        profile = Profile(
            name=f"profile-{i}",
            provider="openai",
            model="test-model",
            auth_ref=str(uuid.uuid4()),
        )
        backends.append((provider, f"key-{i}", profile))
    return BridgeServer(
        adapter=launcher,
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="test-model",
        backends=backends,
        backend_cooldown=300,
    )


class _CcLauncher(_StubLauncher):
    """Stub launcher whose agent speaks Chat Completions."""

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.CHAT_COMPLETIONS_API


class _GeminiLauncher(_StubLauncher):
    """Stub launcher whose agent speaks Gemini."""

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.GEMINI_API


# Complete Chat Completions success stream (content + finish + DONE) — the
# attempt a failover or retry is allowed to succeed with.
_CC_FULL_STREAM = (
    _CC_CONTENT_CHUNK + b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
    b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
    b"data: [DONE]\n\n"
)

# An in-stream CC error chunk: the shape every route's `_is_upstream_stream_error`
# detects (a bare top-level "error" object).
_CC_IN_STREAM_ERROR = b'data: {"error": {"message": "upstream blew up", "type": "server_error"}}\n\n'


# ── Local responder building blocks ──────────────────────────────────────


async def _sse(request: web.Request, *chunks: bytes) -> web.StreamResponse:
    """Open a 200 SSE response, write ``chunks``, and end it cleanly."""
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for chunk in chunks:
        await resp.write(chunk)
    await resp.write_eof()
    return resp


async def _error_500(request: web.Request, _ordinal: int) -> web.StreamResponse:
    """A pre-emission upstream failure: 500 before any stream byte."""
    resp = web.StreamResponse(status=500)
    await resp.prepare(request)
    await resp.write_eof()
    return resp


# ── §6.3.1 row 1 — pre-emission clean failover ───────────────────────────


class TestPreEmissionFailover:
    """Pre-emission failure → clean failover; one complete stream, two requests."""

    @pytest.mark.asyncio
    async def test_translated_pre_emission_failure_fails_over_to_one_complete_stream(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        """Row 1, translated route: 500 on attempt 1, success on attempt 2.

        The oracle: exactly two upstream requests, and the client transcript
        is one complete translated message — the second attempt's content
        exactly once, no error event, no duplication.
        """
        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, ordinal: _error_500(req, ordinal) if ordinal == 0 else _sse(req, _CC_FULL_STREAM),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 2
        text = client_body.decode()
        assert text.count("event: message_start") == 1
        assert text.count("event: message_stop") == 1
        assert "event: error" not in text
        assert text.count('"text":"Hi"') + text.count('"text": "Hi"') == 1, (
            f"expected the content exactly once, got: {text[:400]!r}"
        )

    @pytest.mark.asyncio
    async def test_native_pre_emission_failure_fails_over_to_one_complete_stream(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        """Row 1, native passthrough: the same oracle with a Messages-wire upstream."""
        messages_success = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
            b'"type":"message","role":"assistant","content":[],"model":"test-model",'
            b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"hello"}}\n\n'
            b'event: message_delta\ndata: {"type":"message_delta",'
            b'"delta":{"stop_reason":"end_turn","stop_sequence":null},'
            b'"usage":{"output_tokens":1}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        upstream = _Upstream(
            "/v1/messages",
            lambda req, ordinal: _error_500(req, ordinal) if ordinal == 0 else _sse(req, messages_success),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url, native=True)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 2
        # The passthrough forwards the upstream's bytes unchanged.
        assert client_body == messages_success


# ── §6.3.1 rows 2–3 — the D2 residue (transport class, no finish chunk) ──


async def _drop_after_frames(request: web.Request, frames: tuple[bytes, ...]) -> web.StreamResponse:
    """Write ``frames`` then close the connection with a graceful FIN.

    The same flush-then-close semantics ``Reply.abort()`` gives the recorder
    (KBR-189): queued bytes reach the client before the drop surfaces.
    """
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for frame in frames:
        await resp.write(frame)
    assert request.transport is not None
    request.transport.close()
    return resp


class TestD2Residue:
    """Transport drops that land before any finish chunk keep the D2 ending.

    Scoped per the corrected injection-classes table: on the translated route
    a drop at AFTER_TEXT or MID_TOOL_ARGUMENTS (no finish chunk read yet)
    closes with `end_turn` + `message_stop` and no error event — live
    translator state, `finalize_interrupted_stream` closes everything.
    Verified 10/10 per cell at the requirements stage; these pins hold that.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "point",
        [InjectionPoint.AFTER_TEXT, InjectionPoint.MID_TOOL_ARGUMENTS],
        ids=["after_text", "mid_tool_arguments"],
    )
    async def test_translated_transport_drop_before_the_finish_chunk_finalizes(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
        point,
    ) -> None:
        payload = b"".join(frames_for(WireFormat.CHAT_COMPLETIONS, point))
        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, _ordinal: _drop_after_frames(req, (payload,)),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 1, "the D2 residue is a no-retry ending"
        text = client_body.decode()
        assert '"stop_reason":"end_turn"' in text or '"stop_reason": "end_turn"' in text
        assert "event: message_stop" in text
        assert "event: error" not in text, "the D2 residue ends without an error event"


# ── §6.3.1 row 4 — the BEFORE_TERMINAL two-way shape set ─────────────────


class TestBeforeTerminalShapeSet:
    """The translated BEFORE_TERMINAL cell under a transport drop is two-way.

    The finish chunk's auto-reset wipes live state before the drop surfaces,
    so the ending is one of T-G7's `{complete_sentence, truncated}` — the
    buffered close-out flushed (complete sentence) or the transport branch's
    error-event fallback with the block left open (truncated, error event).
    The pin asserts membership in the shape set: never both terminal events,
    never malformed, never a second attempt.
    """

    @pytest.mark.asyncio
    async def test_translated_transport_drop_at_before_terminal_is_one_of_the_two_honest_shapes(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        payload = b"".join(frames_for(WireFormat.CHAT_COMPLETIONS, InjectionPoint.BEFORE_TERMINAL))
        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, _ordinal: _drop_after_frames(req, (payload,)),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 1
        text = client_body.decode()
        complete = "event: message_stop" in text
        truncated = "event: error" in text
        assert complete or truncated, f"neither honest shape: {text[:400]!r}"
        assert not (complete and truncated), "both terminal events is corrupt"
        assert text.count("kbr-tb4") <= 1, "no duplicated content across the race"


# ── R6 pins: KBR-183 single-backend mid-answer, KBR-5 streaming context,
#    CC empty ladder with no role chunk ─────────────────────────────────


class TestSingleBackendMidAnswerQ14a:
    """KBR-183 §11 Q14(a): a single-backend mid-answer failure ends in an error.

    A single-backend pool has nowhere to fail over to; on a timeout after
    content the bridge ends the turn with the error event. The sibling suite
    pins the timeout shape generally; this pin locks the single-backend
    shape the wider §6.3.1 row 1 ('two requests on balancing') does not.
    """

    @pytest.mark.asyncio
    async def test_single_backend_translated_timeout_after_content_ends_in_error(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, _ordinal: upstream.send_then_stall(req, _CC_CONTENT_CHUNK),
        )
        async with upstream as base_url:
            server = _build("single", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        # Single backend: exactly one upstream request; the error event is the
        # only terminal; no second attempt, no duplicated text.
        assert upstream.requests == 1
        text = client_body.decode()
        assert text.count("event: error") == 1
        assert text.count("Hi") == 1
        assert "message_stop" not in text


class TestStreamingContextTooLarge:
    """Comment 16740 pin: streaming on upstream context-too-large today fails
    over (balancing) or surfaces the upstream error (single backend).

    The non-streaming route re-compacts at half budget before retry; the
    streaming route has no re-compaction backstop (KBR-5 scope addition).
    """

    @pytest.mark.asyncio
    async def test_balancing_streaming_context_too_large_fails_over_to_a_clean_stream(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        async def _too_large(request: web.Request, _ordinal: int) -> web.StreamResponse:
            resp = web.StreamResponse(status=413)
            await resp.prepare(request)
            await resp.write_eof()
            return resp

        async def _ok(request: web.Request, _ordinal: int) -> web.StreamResponse:
            return await _sse(request, _CC_FULL_STREAM)

        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, ordinal: _too_large(req, ordinal) if ordinal == 0 else _ok(req, ordinal),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 2, "balancing pool must fail over pre-emission"
        text = client_body.decode()
        assert "event: error" not in text, "the client sees the success, not the 413"

    @pytest.mark.asyncio
    async def test_single_backend_streaming_context_too_large_surfaces_the_upstream_error(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        async def _too_large(request: web.Request, _ordinal: int) -> web.StreamResponse:
            resp = web.StreamResponse(status=413)
            await resp.prepare(request)
            await resp.write_eof()
            return resp

        upstream = _Upstream(
            "/v1/chat/completions",
            _too_large,
        )
        async with upstream as base_url:
            server = _build("single", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 1, "single backend has nowhere to fail over"
        # The error reaches the client — the KBR-5 observation that streaming
        # on the single-backend Quick Start surface the upstream error.
        assert "error" in client_body.decode().lower()


async def _drive_chat_completions(port: int) -> bytes:
    """POST a streaming /v1/chat/completions request and return the raw body."""
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp,
    ):
        return await resp.read()


class TestCcEmptyLadderNoRoleChunk:
    """Comment 19145 pin: the CC empty ladder needs a stream that forwards NOTHING.

    On the CC wire the role chunk itself sets the bridge's has_content flag
    (KBR-232's well-formed-skeleton semantics); the empty ladder does not fire
    when the role chunk arrives. A script that writes nothing is the case
    that exercises the ladder on the CC route.
    """

    @pytest.mark.asyncio
    async def test_cc_route_empty_stream_with_no_role_chunk_fires_the_empty_ladder(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        # Attempt 1 forwards nothing (no role chunk), attempt 2 succeeds.
        async def _empty(request: web.Request, _ordinal: int) -> web.StreamResponse:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write_eof()
            return resp

        async def _ok(request: web.Request, _ordinal: int) -> web.StreamResponse:
            return await _sse(request, _CC_FULL_STREAM)

        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, ordinal: _empty(req, ordinal) if ordinal == 0 else _ok(req, ordinal),
        )
        async with upstream as base_url:
            server = _balancing_protocol_server(base_url, _CcLauncher())
            port = await server.start_async()
            try:
                client_body = await _drive_chat_completions(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 2
        # The empty attempt was discarded — the client receives the success bytes.
        assert b'"finish_reason":"stop"' in client_body


# ── §1.4 falsification — the no-duplication oracle must catch a deliberate defect ──


def _assert_no_duplicated_content(text: str, marker: str) -> None:
    """Assert ``text`` carries ``marker`` at most once — the §6.3.1 content guarantee.

    Args:
        text: The wire body the client received (decoded).
        marker: A T-B4-vendored token unique enough that a legitimate
            transcript cannot carry it twice.
    """
    assert text.count(marker) <= 1, (
        f"content guarantee violated: marker {marker!r} appears {text.count(marker)} times in the transcript"
    )


class TestGridOracleFalsification:
    """The no-duplication oracle catches a deliberately duplicated transcript.

    §1.4 requires every harness to ship with a falsification case proving it
    can fail. This test demonstrates the helper the grid uses catches the
    class of defect the row promises to detect — duplicated text from a
    second-attempt replay.
    """

    def test_no_duplication_oracle_fails_on_a_replayed_text_marker(self) -> None:
        """A duplicated marker must raise; a single marker must pass."""
        duplicated = (
            "event: message_start\ndata: {}\n\n"
            'event: content_block_delta\ndata: {"type":"content_block_delta",'
            '"index":0,"delta":{"type":"text_delta","text":"kbr-tb4"}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta",'
            '"index":0,"delta":{"type":"text_delta","text":"kbr-tb4"}}\n\n'
        )
        with pytest.raises(AssertionError, match="content guarantee violated"):
            _assert_no_duplicated_content(duplicated, "kbr-tb4")

        # Single occurrence: the same helper accepts it.
        once = duplicated.replace(
            'event: content_block_delta\ndata: {"type":"content_block_delta",'
            '"index":0,"delta":{"type":"text_delta","text":"kbr-tb4"}}\n\n',
            "",
            1,
        )
        _assert_no_duplicated_content(once, "kbr-tb4")


# ── Fix R2 — Gemini terminal diagnostic on post-content in-stream error ──


async def _drive_gemini(port: int) -> bytes:
    """POST a streaming Gemini request and return the raw client body."""
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}/v1beta/models/test-model:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp,
    ):
        return await resp.read()


class TestGeminiInStreamErrorExhaustionFix:
    """R2: ``_stream_gemini``'s exhaustion arm writes a terminal diagnostic.

    SYSTEM_DESIGN §5.3 S10 closes the only silent arm: when an in-stream
    error surfaces after the request has written, the turn ends in one
    ``data: {"error": {"code": 502, "message": ...}}`` SSE event and EOF —
    the route's own KBR-247 convention. Before the fix the arm logs and
    breaks and the stream simply stops (status="incomplete", no event);
    the red assertion expects the terminal event.
    """

    @pytest.mark.asyncio
    async def test_gemini_in_stream_error_after_content_writes_one_error_event(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        async def _content_then_in_stream_error(request: web.Request, _ordinal: int) -> web.StreamResponse:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(_CC_CONTENT_CHUNK)
            await resp.write(_CC_IN_STREAM_ERROR)
            await resp.write_eof()
            return resp

        upstream = _Upstream(
            "/v1/chat/completions",
            _content_then_in_stream_error,
        )
        async with upstream as base_url:
            server = _balancing_protocol_server(base_url, _GeminiLauncher())
            port = await server.start_async()
            try:
                client_body = await _drive_gemini(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 1
        text = client_body.decode()
        # Exactly one error data line with the route's terminal discriminator.
        error_lines = [
            line for line in text.splitlines()
            if line.startswith("data:") and '"error"' in line
        ]
        assert len(error_lines) == 1, (
            f"expected exactly one Gemini terminal error event, got {len(error_lines)}: {text!r}"
        )
        assert '"code": 502' in error_lines[0]


# ── Fix R3 — D4 unification on the translated Messages empty-stream gate ──


async def _drive_messages_json(port: int) -> tuple[int, bytes]:
    """POST a streaming /v1/messages request and return ``(status, body)``."""
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}/v1/messages",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp,
    ):
        return resp.status, await resp.read()


class TestTranslatedEmptyStreamD4Unification:
    """R3: both empty shapes (no-finish and finish-chunk) exhaust into D4.

    Before the fix the no-finish arm returned the D4 ``502 empty_response``
    while the finish-chunk arm exhausted into the M12 fallback text inside
    a ``200`` (KBR-235 deliberately kept the split). The unification closes
    the split: Q14(a) already says a ``200`` carrying substituted text is
    the one thing the route must never produce.

    Red at base: a finish-chunk-only empty stream (the well-formed-skeleton
    shape — role chunk, empty finish, ``[DONE]``, no content delta) on a
    two-backend balancing pool exhausts the ladder; the client receives
    the M12 fallback inside a ``200``. After the fix the client receives
    the D4 ``502 api_error`` body with ``reason: empty_response``.
    """

    @pytest.mark.asyncio
    async def test_translated_finish_chunk_only_empty_stream_exhausts_into_d4(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        role_chunk = (
            b'data: {"id":"c1","choices":[{"index":0,'
            b'"delta":{"role":"assistant","content":""}}],"model":"test-model"}\n\n'
        )
        finish_chunk = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
        )
        well_formed_skeleton = role_chunk + finish_chunk + b"data: [DONE]\n\n"

        async def _empty_skeleton(request: web.Request, _ordinal: int) -> web.StreamResponse:
            return await _sse(request, well_formed_skeleton)

        upstream = _Upstream("/v1/chat/completions", _empty_skeleton)
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                status, body = await _drive_messages_json(port)
            finally:
                await server.stop_async()

        # Both attempts hit the ladder (2 backends); today the second attempt
        # also exhausts into the same fallback path.
        assert status == 502, (
            f"expected D4 502 empty_response, got {status}; body: {body[:300]!r}"
        )
        body_text = body.decode()
        assert '"reason": "empty_response"' in body_text or '"reason":"empty_response"' in body_text
        # The fallback text must not reach the client (S11: the ladder
        # exhausts into the D4 error, never the M12 substitution). Assert on
        # the constant's VALUE, not its Python name.
        assert "Upstream model returned an empty response" not in body_text
        # Empty replies do not quarantine (the empty ladder's no-quarantine
        # health model), so the ladder walks the full attempt budget on a
        # two-backend balancing pool — `(server_module._MAX_RETRIES + 1) * 2 +
        # len(server_module._EMPTY_FINAL_DELAYS)` attempts — before the
        # final-delay index falls off the back of `_EMPTY_FINAL_DELAYS` and
        # the gate exhausts into D4. Pin the exact count (not just >= 2).
        assert upstream.requests == (
            (server_module._MAX_RETRIES + 1) * 2
            + len(server_module._EMPTY_FINAL_DELAYS)
        ), (
            f"expected the full empty-ladder attempt budget, saw {upstream.requests}"
        )


# ── Fix R4 — streaming D3 on the translated Messages route ───────────────


class TestTranslatedStreamingD3:
    """R4: a truncation before content fails at once with the D3 400.

    SYSTEM_DESIGN §5.3 S12 extends D3 to the translated route: a
    finish-chunk empty stream whose stop reason maps into
    ``_NATIVE_TRUNCATING_STOP_REASONS`` ends the ladder on that attempt with
    the ``400`` ``invalid_request_error`` body and
    ``reason: "<stop_reason>_before_content"`` — the same body the native
    route builds via ``_d3_truncation_error_body``. The scripted upstream
    sends ``finish_reason: "length"`` (mapped to ``max_tokens``) and the
    literal ``"model_context_window_exceeded"`` (a pass-through spelling —
    no standard CC finish reason maps to it, so only an upstream that sends
    that literal reaches the context-window arm).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("finish_reason", "expected_reason"),
        [
            ("length", "max_tokens_before_content"),
            ("model_context_window_exceeded", "model_context_window_exceeded_before_content"),
        ],
        ids=["length_maps_to_max_tokens", "literal_context_window_passthrough"],
    )
    async def test_translated_truncation_finish_chunk_fails_at_once_with_d3(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
        finish_reason,
        expected_reason,
    ) -> None:
        finish_chunk = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"' + finish_reason.encode() + b'"}],"model":"test-model","usage":null}\n\n'
        )
        truncation_stream = finish_chunk + b"data: [DONE]\n\n"

        async def _truncated(request: web.Request, _ordinal: int) -> web.StreamResponse:
            return await _sse(request, truncation_stream)

        upstream = _Upstream("/v1/chat/completions", _truncated)
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                status, body = await _drive_messages_json(port)
            finally:
                await server.stop_async()

        # D3 ends the ladder on the FIRST attempt — no retry, no failover.
        assert upstream.requests == 1, (
            f"a truncation before content must not be retried; saw {upstream.requests} requests"
        )
        assert status == 400, f"expected the D3 400, got {status}; body: {body[:300]!r}"
        body_text = body.decode()
        assert f'"reason":"{expected_reason}"' in body_text or (
            f'"reason": "{expected_reason}"' in body_text
        )


# ── Fix R5 — Messages-wire pre-content error quarantine parity ─────────


class TestMessagesWirePreContentErrorQuarantine:
    """R5: the Messages-wire pre-content error ladder charges the cooldown.

    SYSTEM_DESIGN §5.3 S13 closes the D2-amendment half KBR-241 left open:
    a pre-content ``event: error`` on a Messages-wire stream now charges
    ``_get_stream_error_cooldown`` on the failing backend, matching the
    CC-wire in-stream error cooldown. Without parity, a persistently-
    erroring backend kept drawing ~1/n of the attempts on a balancing pool.

    Red at base: the empty ladder runs the retries without marking any
    backend unhealthy — the assertion that every backend the ladder drew
    ends charged fails today (all stay healthy). After the fix each drawn
    backend carries the stream cooldown (both, on this always-erroring
    two-backend pool).
    """

    @pytest.mark.asyncio
    async def test_messages_wire_pre_content_error_charges_the_backend_cooldown(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        # A pre-content Anthropic-style error event — the shape the
        # preamble hold recognises by name line (server.py:5896).
        pre_content_error = (
            b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",'
            b'"message":"Overloaded"}}\n\n'
        )

        async def _pre_content_error_stream(
            request: web.Request, _ordinal: int
        ) -> web.StreamResponse:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(pre_content_error)
            await resp.write_eof()
            return resp

        upstream = _Upstream("/v1/messages", _pre_content_error_stream)
        async with upstream as base_url:
            server = _build("balanced", base_url, native=True)
            port = await server.start_async()
            try:
                await _post_stream(port)
            finally:
                await server.stop_async()

        # Parity fix: the upstream errors on every attempt, so the ladder
        # charges each backend it draws — on this two-backend pool, both end
        # quarantined (then `_any_healthy_backend()` goes False and the
        # ladder ends). Which backend was drawn first is random; that both
        # were drawn and charged is not. The assertion is the charge itself —
        # healthy False, the stream branch's own counter incremented — not
        # the draw order.
        charged = [h for h in server._backend_health if not h["healthy"]]
        assert len(charged) == 2, (
            "every backend the ladder drew must carry the stream cooldown"
        )
        assert all(h["failure_count"] >= 1 for h in charged)
        assert all(h["stream_error_count"] >= 1 for h in charged)


class TestPostEmissionTimeoutEndings:
    """A read timeout after content ends the turn per Q14(a) — no second attempt.

    These are the grid's post-emission cells driven by the timeout class: the
    deterministic driver whose ending (blocks closed, one terminal error, one
    upstream request) holds on every route.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("frames", "content_marker"),
        [
            (frames_for(WireFormat.CHAT_COMPLETIONS, InjectionPoint.AFTER_TEXT), "kbr-tb4"),
            (
                frames_for(WireFormat.CHAT_COMPLETIONS, InjectionPoint.MID_TOOL_ARGUMENTS),
                "kbr-tb4-tool",
            ),
        ],
        ids=["after_text", "mid_tool_arguments"],
    )
    async def test_translated_timeout_ends_the_turn_without_a_second_attempt(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
        frames,
        content_marker,
    ) -> None:
        """Rows 2–3, translated route: content identity survives the failure.

        The T-B4 frame sequences carry a text marker and a tool id unique to
        this library, so the transcript can be checked for duplication and
        splice byte-exactly: each appears at most once, the upstream saw
        exactly one request, and the turn ends in the Q14(a) error event.
        """
        payload = b"".join(frames)
        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, _ordinal: upstream.send_then_stall(req, payload),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 1, "no second attempt may follow emitted bytes"
        text = client_body.decode()
        # One terminal error, no duplicated content, no spliced arguments.
        assert text.count("event: error") == 1
        _assert_no_duplicated_content(text, content_marker)
        assert "message_stop" not in text, "the error event is the only terminal"

    @pytest.mark.asyncio
    async def test_translated_timeout_before_terminal_closes_the_buffered_block(
        self,
        fast_stall,
        short_grace,  # noqa: F811 — fixture shadowing the module-level import
    ) -> None:
        """Row 4, translated route, timeout class: the Q14(a) shape still holds.

        The finish chunk's auto-reset wiped the translator's live state, but
        the timeout arm reconstructs the block stops from the buffered finish
        events before the error — the content never arrives twice.
        """
        payload = b"".join(frames_for(WireFormat.CHAT_COMPLETIONS, InjectionPoint.BEFORE_TERMINAL))
        upstream = _Upstream(
            "/v1/chat/completions",
            lambda req, _ordinal: upstream.send_then_stall(req, payload),
        )
        async with upstream as base_url:
            server = _build("balanced", base_url)
            port = await server.start_async()
            try:
                client_body = await _post_stream(port)
            finally:
                await server.stop_async()

        assert upstream.requests == 1
        text = client_body.decode()
        assert text.count("event: error") == 1
        assert text.count("kbr-tb4") <= 1
