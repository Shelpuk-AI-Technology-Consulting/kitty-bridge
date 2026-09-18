"""Bridge-driven tests for the SSE grammar state machine — T-G7 (KBR-83).

`.system_design/TEST_SUITE.md` §6.2.2 · plan task **T-G7** · requirements in
`.requirements/20260917T120913Z_sse_grammar_state_machine/REQUIREMENTS.md`.

The companion to ``tests/harness/test_sse_grammar_falsification.py``: that
suite proves the grammar can reject malformed sequences; this one proves the
bridge's real downstream text classifies as documented for every scenario
each inbound route produces. The matrix is the requirements' §4 table —
each row is one parametrised case, each assertion traces to one requirement.

**Layer.** ``l2`` (Contract). The grammar is an L2 guard; driving a real
bridge + recorder is the L2 contract test, per TEST_SUITE.md §6.2.2.

**Why the gap between scenarios and the falsification suite.** The
falsification module feeds hand-built bytes; this one drives a real
``BridgeFixture`` and reads what reached the client. A bridge regression that
changed the close-out sequence (server.py:5909-5975) would only show here;
a regression in the grammar's rules would only show in the falsification
suite. Both must stay green.
"""

from __future__ import annotations

import asyncio

import pytest
from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    TransportFactory,
    inbound_path,
    minimal_inbound_body,
    pin_backend_order,
    transport,
)
from harness.contract import WireFormat
from harness.failures import (
    InjectionPoint,
    drop_at,
    empty_response,
    error_status,
    scripted,
    success,
)
from harness.recorder import CapturedRequest, Reply, Responder
from harness.sse_grammar import Classification, StreamProtocol, classify_response

from kitty.bridge import server as server_module

# ---------------------------------------------------------------------------
# Layer marker.
# ---------------------------------------------------------------------------

#: L2 (Contract). Layer is set on the module so the gate that asserts every
#: test carries exactly one layer marker stays green; the default would be l1,
#: which would put this in the fast gate along with the falsification suite.
pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Fast-stall fixture — patches the bridge's stream ladder to sub-second values.
# Without this, the empty-response ladder's default ``_EMPTY_FINAL_DELAYS``
# (20 s, 40 s) and the default ``_STREAM_READ_TIMEOUT`` (120 s) collide with
# ``BridgeFixture.post``'s 10 s client timeout and the test dies as
# ``TransportTimeout`` instead of asserting on the JSON-error body the bridge
# returns. C5 in the requirements review.
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the bridge's stream ladder so the empty-response rows are testable.

    The grace patches come from ``tests/bridge/test_client_disconnect_health.py``'s
    ``short_grace``: a ``drop_at(BEFORE_FIRST_BYTE)`` abort repeats for the full
    30 s grace window at defaults, which collides with the client timeout.
    """
    monkeypatch.setattr(server_module, "_STREAM_READ_TIMEOUT", 1)
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.001)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", (0.001, 0.001))
    monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_PERIOD", 0.3)
    monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.01, 0.02))


# ---------------------------------------------------------------------------
# InboundProtocol ↔ StreamProtocol agreement — the independence check.
# ---------------------------------------------------------------------------


def test_stream_protocol_members_match_inbound_protocol() -> None:
    """``StreamProtocol`` deliberately duplicates ``InboundProtocol``.

    Importing ``harness.bridge`` for the enum would pull ``kitty`` into the
    grammar's import graph; the duplication is the discipline. This assertion
    makes the duplication load-bearing: a fifth protocol cannot silently miss a
    grammar.
    """
    assert {member.value for member in StreamProtocol} == {
        member.value for member in InboundProtocol
    }


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _transport_with(fmt: WireFormat, responder: Responder | None) -> TransportFactory:
    """Return a transport factory wired with ``responder``.

    Args:
        fmt: The upstream wire format the recorder serves.
        responder: A failure-library responder, or ``None`` for the default.

    Returns:
        A callable the fixture uses to construct the transport.
    """
    return transport("aiohttp", fmt, responder=responder)


async def _post(
    fixture: BridgeFixture,
    protocol: InboundProtocol,
    *,
    timeout: float = 30.0,
) -> tuple[int, str]:
    """POST a streaming minimal body and return the raw response text.

    Args:
        fixture: The started ``BridgeFixture``.
        protocol: The inbound route to hit.
        timeout: Total client timeout; raised above the patched ladder delays.

    Returns:
        ``(status, text)`` — ``text`` is the full response body. For SSE
        streams it is the raw bytes; for the JSON-error rows it is the body
        of the close-out envelope.
    """
    path = inbound_path(protocol, model="m", stream=True)
    body = minimal_inbound_body(protocol, "kbr-tg7", model="m", stream=True)
    return await fixture.post(path, body, timeout=timeout)


def _hang_responder(release: asyncio.Event) -> Responder:
    """Build a responder that accepts the connection then never writes a byte.

    The bridge's preamble hold waits ``_STREAM_READ_TIMEOUT`` for content,
    runs the empty ladder, and answers with a JSON error — KBR-155 D7. The
    test must set the release event **inside** the fixture block, before
    teardown: ``BridgeFixture.stop()`` waits for in-flight upstream handlers,
    so a release that fires after ``__aexit__`` costs aiohttp's full 60 s
    shutdown_timeout per test.

    Args:
        release: An event the responder awaits; the test sets it before
            teardown so the hanging recorder task returns and the fixture's
            stop drains in milliseconds (BridgeFixture docstring: "a test
            that deliberately leaves a request hanging must release it before
            teardown, or it pays the full hold time in the gate").

    Returns:
        A responder closure suitable for the failure library.
    """

    async def responder(captured: CapturedRequest, response: Reply) -> None:
        """Accept the connection, then await the release event."""
        del captured
        await response.begin(200, {"Content-Type": "text/event-stream"})
        await release.wait()

    return responder


# ---------------------------------------------------------------------------
# Messages route — translated path (CC upstream → Messages downstream).
# ---------------------------------------------------------------------------


class TestMessagesTranslatedPath:
    """The path an OpenAI-compatible CC upstream takes through the translator.

    The recorder serves CC; the bridge translates the CC stream into the
    Anthropic Messages SSE the agent expects. ``drop_at`` shapes close out via
    ``finalize_interrupted_stream`` (the transport-drop branch of
    ``BridgeServer._stream_messages``; search ``server.py`` for the call —
    line numbers drift), which produces a complete sentence.
    """

    async def test_success_stream_is_a_complete_sentence(self) -> None:
        async with BridgeFixture(_transport_with(WireFormat.CHAT_COMPLETIONS, None)) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        assert status == 200
        assert classify_response(StreamProtocol.MESSAGES, status, text) is Classification.COMPLETE_SENTENCE

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            (InjectionPoint.AFTER_TEXT, Classification.COMPLETE_SENTENCE),
            (InjectionPoint.MID_TOOL_ARGUMENTS, Classification.COMPLETE_SENTENCE),
            # BEFORE_TERMINAL is genuinely two-way: the injected finish chunk may
            # be processed (complete sentence) or lost to the abort race before
            # the read loop reaches it (truncated — blocks closed by the injected
            # frames, but no terminal event). KBR-189 flushes the abort's queued
            # bytes, but which side of the finish chunk the read exception lands
            # on is not deterministic. The contract is "one of the two honest
            # shapes", never malformed and never a JSON error.
            (InjectionPoint.BEFORE_TERMINAL, None),
        ],
    )
    async def test_drop_at_post_emission_classifies_as_documented(
        self, point: InjectionPoint, expected: Classification | None
    ) -> None:
        """Post-emission drops close out: finalize (or the race window) — never a JSON error."""
        async with BridgeFixture(
            _transport_with(WireFormat.CHAT_COMPLETIONS, drop_at(WireFormat.CHAT_COMPLETIONS, point))
        ) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        assert status == 200
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        if expected is None:
            assert verdict in (Classification.COMPLETE_SENTENCE, Classification.TRUNCATED), (
                f"point={point}; body={text[:200]!r}"
            )
        else:
            assert verdict is expected, f"point={point}; body={text[:200]!r}"

    @pytest.mark.parametrize(
        ("upstream_fmt", "builder"),
        [
            (WireFormat.CHAT_COMPLETIONS, "drop_at"),
            (WireFormat.CHAT_COMPLETIONS, "empty_response"),
        ],
    )
    async def test_pre_release_failure_is_json_error(
        self, fast_stall: None, upstream_fmt: WireFormat, builder: str
    ) -> None:
        """Pre-release ladder exhaustion (KBR-155 D1–D7): no stream, JSON error.

        Two builders reach the same ladder on this path: a transport abort
        before any byte (``drop_at(BEFORE_FIRST_BYTE)``) and a well-formed
        stream that never releases the preamble hold
        (``empty_response(..., stream=True)``). Both must answer JSON, never a
        partial stream.
        """
        if builder == "drop_at":
            responder: Responder = drop_at(upstream_fmt, InjectionPoint.BEFORE_FIRST_BYTE)
        else:
            responder = empty_response(upstream_fmt, stream=True)
        async with BridgeFixture(_transport_with(upstream_fmt, responder)) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.JSON_ERROR, (
            f"builder={builder}; status={status}; body={text[:120]!r}"
        )

    async def test_stall_past_read_timeout_is_json_error(self, fast_stall: None) -> None:
        """A silent upstream that never writes triggers the read timeout → JSON error.

        The §4 "stall past read timeout before any emission" row: the hang
        responder accepts the connection and never writes, so the bridge's
        preamble hold times out (patched to 1 s) and the empty ladder runs to
        exhaustion. The answer must be a JSON error body, never a partial
        stream. The release event is set in teardown so the recorder's
        still-hanging task drains before the fixture stops.
        """
        release = asyncio.Event()
        # The release fires INSIDE the block, before teardown: BridgeFixture
        # docstring — "a test that deliberately leaves a request hanging must
        # release it before teardown, or it pays the full hold time in the
        # gate". A finally after __aexit__ is too late: stop() would drain the
        # hung handler for aiohttp's full 60 s shutdown_timeout first.
        async with BridgeFixture(_transport_with(WireFormat.CHAT_COMPLETIONS, _hang_responder(release))) as fixture:
            try:
                status, text = await _post(fixture, InboundProtocol.MESSAGES)
            finally:
                release.set()
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.JSON_ERROR, f"status={status}; body={text[:120]!r}"

    async def test_pre_stream_http_error_is_json_error(self, fast_stall: None) -> None:
        """A non-stream HTTP error from the upstream is the JSON-error row of §4.

        The 503 lands in the empty-response ladder; without ``fast_stall`` the
        ladder's 20 s and 40 s final delays collide with the client timeout.
        """
        async with BridgeFixture(
            _transport_with(WireFormat.CHAT_COMPLETIONS, error_status(WireFormat.CHAT_COMPLETIONS, 503))
        ) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.JSON_ERROR, f"status={status}; body={text[:120]!r}"

    async def test_failover_after_pre_release_drop_is_a_complete_sentence(
        self, fast_stall: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First backend drops before any byte; second serves the success — the client sees one clean stream."""
        responder = scripted(
            drop_at(WireFormat.CHAT_COMPLETIONS, InjectionPoint.BEFORE_FIRST_BYTE),
            success(WireFormat.CHAT_COMPLETIONS, stream=True),
        )
        # A statement, not a context manager: pin_backend_order patches
        # ``random.choices`` through monkeypatch and returns None.
        pin_backend_order(monkeypatch)
        async with BridgeFixture(
            _transport_with(WireFormat.CHAT_COMPLETIONS, responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.COMPLETE_SENTENCE, f"status={status}; body={text[:120]!r}"


# ---------------------------------------------------------------------------
# Messages route — native path (ANTHROPIC upstream → Messages passthrough).
# ---------------------------------------------------------------------------


#: The KBR-236 empty-verdict-after-content shape is intentionally NOT driven
#: here: its grammar-level contract lives in the falsification suite
#: (``test_empty_after_emission_shape_is_error_terminal``) and its bridge-level
#: ground truth in ``tests/bridge/test_post_emission_no_failover.py``
#: (``TestEmptyVerdictAfterContent``), whose hand-built server exercises the
#: close-out path directly. See that module for why the shape is what it is.


class TestMessagesNativePath:
    """The path the Anthropic-upstream Messages passthrough takes.

    The bridge forwards Anthropic SSE bytes verbatim; ``drop_at`` post-emission
    produces the fallback ``messages_format_error`` event (the native path's
    transport-drop branch in ``BridgeServer._stream_messages``; search
    ``server.py`` for the ``or [messages_format_error(...)]`` fallback —
    line numbers drift), which leaves the client's open content block
    unclosed. Native is the path where ``truncated`` is the honest
    classification; the translated path produces ``complete_sentence``.
    """

    async def test_success_stream_is_a_complete_sentence(self) -> None:
        async with BridgeFixture(_transport_with(WireFormat.ANTHROPIC_MESSAGES, None)) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        assert status == 200
        assert classify_response(StreamProtocol.MESSAGES, status, text) is Classification.COMPLETE_SENTENCE

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            # AFTER_TEXT and BEFORE_TERMINAL: the injected frames close the block
            # (AFTER_TEXT writes content_block_stop; BEFORE_TERMINAL adds
            # message_delta), then the transport-drop fallback appends one error
            # event. Closed structure + error terminal → error_terminal.
            (InjectionPoint.AFTER_TEXT, Classification.ERROR_TERMINAL),
            # MID_TOOL_ARGUMENTS leaves the tool block open — the one honest
            # truncation the harness can build (precedence rule 2).
            (InjectionPoint.MID_TOOL_ARGUMENTS, Classification.TRUNCATED),
            (InjectionPoint.BEFORE_TERMINAL, Classification.ERROR_TERMINAL),
        ],
    )
    async def test_drop_at_post_emission_classifies_as_documented(
        self, point: InjectionPoint, expected: Classification
    ) -> None:
        """Native fallback: one error event; classification per what the injected frames left open."""
        async with BridgeFixture(
            _transport_with(WireFormat.ANTHROPIC_MESSAGES, drop_at(WireFormat.ANTHROPIC_MESSAGES, point))
        ) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        assert status == 200
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is expected, f"point={point}; body={text[:200]!r}"

    @pytest.mark.parametrize(
        ("upstream_fmt", "builder"),
        [
            (WireFormat.ANTHROPIC_MESSAGES, "drop_at"),
            (WireFormat.ANTHROPIC_MESSAGES, "empty_response"),
        ],
    )
    async def test_pre_release_failure_is_json_error(
        self, fast_stall: None, upstream_fmt: WireFormat, builder: str
    ) -> None:
        """Native-path twin of the translated pre-release rows.

        ``drop_at(BEFORE_FIRST_BYTE)`` aborts before any byte;
        ``empty_response(..., stream=True)`` writes a well-formed Messages
        skeleton whose empty text block never releases the preamble hold. Both
        exhaust the ladder to a JSON error (KBR-155 D1–D7).
        """
        if builder == "drop_at":
            responder: Responder = drop_at(upstream_fmt, InjectionPoint.BEFORE_FIRST_BYTE)
        else:
            responder = empty_response(upstream_fmt, stream=True)
        async with BridgeFixture(_transport_with(upstream_fmt, responder)) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.JSON_ERROR, (
            f"builder={builder}; status={status}; body={text[:120]!r}"
        )

    async def test_stall_past_read_timeout_is_json_error(self, fast_stall: None) -> None:
        """Native-path twin: a silent Anthropic upstream exhausts the ladder to JSON."""
        release = asyncio.Event()
        transport_ = _transport_with(WireFormat.ANTHROPIC_MESSAGES, _hang_responder(release))
        # Release INSIDE the block — see the translated twin's note.
        async with BridgeFixture(transport_) as fixture:
            try:
                status, text = await _post(fixture, InboundProtocol.MESSAGES)
            finally:
                release.set()
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.JSON_ERROR, f"status={status}; body={text[:120]!r}"

    async def test_failover_after_pre_release_drop_is_a_complete_sentence(
        self, fast_stall: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Native-path twin: first backend drops, second serves — one clean stream.

        The §4 native failover cell: ``drop_at(BEFORE_FIRST_BYTE)`` on the
        first backend, then a well-formed Anthropic success stream on the
        second. The client sees one complete sentence, never the drop.
        """
        responder = scripted(
            drop_at(WireFormat.ANTHROPIC_MESSAGES, InjectionPoint.BEFORE_FIRST_BYTE),
            success(WireFormat.ANTHROPIC_MESSAGES, stream=True),
        )
        pin_backend_order(monkeypatch)
        async with BridgeFixture(
            _transport_with(WireFormat.ANTHROPIC_MESSAGES, responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            status, text = await _post(fixture, InboundProtocol.MESSAGES)
        verdict = classify_response(StreamProtocol.MESSAGES, status, text)
        assert verdict is Classification.COMPLETE_SENTENCE, f"status={status}; body={text[:120]!r}"


# ---------------------------------------------------------------------------
# Responses route — CC upstream translated to Responses downstream (Codex path).
# ---------------------------------------------------------------------------


class TestResponsesRoute:
    """CC upstream → Responses downstream, the realistic Codex-over-OpenAI-compatible path."""

    async def test_success_stream_is_a_complete_sentence(self) -> None:
        async with BridgeFixture(_transport_with(WireFormat.CHAT_COMPLETIONS, None)) as fixture:
            status, text = await _post(fixture, InboundProtocol.RESPONSES)
        assert status == 200
        verdict = classify_response(StreamProtocol.RESPONSES, status, text)
        assert verdict is Classification.COMPLETE_SENTENCE, f"body={text[:200]!r}"

    async def test_d4_exhaustion_is_a_complete_sentence(self, fast_stall: None) -> None:
        """KBR-250 D4 on the Responses route: ``[error, response.completed(incomplete)]``.

        The bridge's Responses empty-judgement ladder (KBR-250) needs an upstream
        stream with no role chunk and no content — the only CC stream the
        Responses translator judges empty is ``[DONE]`` alone, which the
        adapter treats as contentless and the empty gate fires. The post-loop
        synthesizes ``response.completed(status="incomplete")`` after one
        ``responses_format_error(code="empty_response")`` event — the lazy
        lifecycle (KBR-242) means no ``response.created`` is ever emitted.

        Grammar contract: ``[error, response.completed]`` with no preceding
        ``response.created`` is the canonical D4 terminal shape → ``complete_sentence``.
        """
        async with BridgeFixture(
            _transport_with(WireFormat.CHAT_COMPLETIONS, _serve_d4_empty)
        ) as fixture:
            status, text = await _post(fixture, InboundProtocol.RESPONSES)
        verdict = classify_response(StreamProtocol.RESPONSES, status, text)
        assert verdict is Classification.COMPLETE_SENTENCE, f"body={text[:200]!r}"


async def _serve_d4_empty(captured: CapturedRequest, response: Reply) -> None:
    """Responder: a CC stream of only ``[DONE]`` — contentless, triggers D4.

    Args:
        captured: The captured request, unused.
        response: The recorder's response handle.
    """
    del captured
    await response.begin(200, {"Content-Type": "text/event-stream"})
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()


# ---------------------------------------------------------------------------
# Chat Completions route — direct CC passthrough.
# ---------------------------------------------------------------------------


class TestChatCompletionsRoute:
    """Direct CC passthrough: ``[DONE]`` is the terminal signal."""

    async def test_success_stream_is_a_complete_sentence(self) -> None:
        async with BridgeFixture(_transport_with(WireFormat.CHAT_COMPLETIONS, None)) as fixture:
            status, text = await _post(fixture, InboundProtocol.CHAT_COMPLETIONS)
        assert status == 200
        assert classify_response(StreamProtocol.CHAT_COMPLETIONS, status, text) is Classification.COMPLETE_SENTENCE


# ---------------------------------------------------------------------------
# Gemini route — translated from CC upstream.
# ---------------------------------------------------------------------------


class TestGeminiRoute:
    """Gemini downstream over a CC upstream (the default transport serves this).

    The transport pairing stays inside the primary recorder so the grammar
    suite does not couple to a containment slice's gate verdicts (the
    reviewer's C3): the aiohttp transport's default responder answers with a
    CC stream, the bridge translates CC to Gemini via ``GeminiTranslator``.
    """

    async def test_success_stream_is_a_complete_sentence(self) -> None:
        async with BridgeFixture(_transport_with(WireFormat.CHAT_COMPLETIONS, None)) as fixture:
            status, text = await _post(fixture, InboundProtocol.GEMINI)
        assert status == 200
        verdict = classify_response(StreamProtocol.GEMINI, status, text)
        assert verdict is Classification.COMPLETE_SENTENCE, f"body={text[:200]!r}"
