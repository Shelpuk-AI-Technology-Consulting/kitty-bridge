"""KBR-47 (T-C4) — the M6 recovery path engages on a balancing profile.

``.system_design/TEST_SUITE.md`` §7.1 · plan task **T-C4** (KBR-47), requirements
``.requirements/20260914T184942Z_kbr47_tc4_corpus_entries/REQUIREMENTS.md``.

A balancing ``BridgeServer`` driven by the M6 corpus entry, with a recorder
scripted to answer 413 on the first attempt and 200 on every later attempt.
The test asserts the two observable effects of the recovery path engaging:
the recovery log line is emitted, and the recorder's second capture's body is
strictly smaller than the first's.

Per KBR-47's second comment, **this** test is where trigger M6 is declared
met — the loader refuses the trigger in a manifest (``NOT_CORPUS_DECIDABLE``)
because it is a property of the upstream response, arranged by a scripted
recorder rather than carried by the inbound request.

The L1 wiring half (entry shape, format, and path-of-the-compactor claims)
lives in :mod:`tests.bridge.test_tc4_corpus_wiring`; this module is the
separate L3 file the layer-marker rule requires (per-test markers make the
whole file claim one layer, so the L1 tests and this one cannot share a
file).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from harness import corpus as k
from harness.bridge import BridgeFixture, pin_backend_order, transport
from harness.contract import CapturedRequest, WireFormat
from harness.recorder import Reply

from kitty.bridge.server import BridgeServer, CompactionFailedError

pytestmark = pytest.mark.l3

#: The committed corpus root. Same rule the L1 wiring module applies — this
#: module is deliberately not importing from it (tests never import from tests),
#: so the two definitions must agree and the duplicated constant is the cost.
CORPUS = Path(__file__).resolve().parents[1] / "corpus"


#: A 413 reply body the bridge recognises. ``_is_context_too_large_error`` keys
#: on a status in (400, 413) **and** a body pattern; status 413 alone is
#: sufficient (see :meth:`kitty.bridge.server.BridgeServer._is_context_too_large_error`),
#: so the body just needs to be valid JSON the error-translation path accepts.
_CONTEXT_TOO_LARGE_BODY = (
    b'{"error":{"type":"invalid_request_error","message":"prompt is too long: 700000 tokens"}}'
)


def _make_413_then_success_responder() -> Any:
    """Return a responder that 413s on the first call, 200s afterwards.

    The state is a counter held by closure — the responder has no other way
    to observe what call number it is on, and counting in the closure is
    simpler than reaching into the recorder from the request. State held in
    the responder's closure survives across awaits on the same loop.

    Returns:
        A :class:`~harness.recorder.Responder` closure.
    """
    state = {"calls": 0}

    async def _responder(_captured: CapturedRequest, reply: Reply) -> None:
        """Answer 413 on the first request, 200 on every later request.

        Args:
            _captured: The capture, unused — state is in the closure.
            reply: The unprepared response.
        """
        is_first = state["calls"] == 0
        state["calls"] += 1

        if is_first:
            reply.content_length = len(_CONTEXT_TOO_LARGE_BODY)
            await reply.begin(413, {"content-type": "application/json"})
            await reply.write(_CONTEXT_TOO_LARGE_BODY)
            await reply.write_eof()
            return

        payload = json.dumps(
            {
                "id": "msg_recovered",
                "type": "message",
                "role": "assistant",
                "model": "stub",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            }
        ).encode()
        reply.content_length = len(payload)
        await reply.begin(200, {"content-type": "application/json"})
        await reply.write(payload)
        await reply.write_eof()

    return _responder


#: The streaming Anthropic Messages reply the recorder gives for a recovered
#: turn. Same event grammar ``scripts/capture_corpus_t_c1.py`` spells out; the
#: bridge's ``_stream_messages`` parses SSE lines, so a JSON 200 would fall into
#: the empty-response ladder and surface a 502 instead of the recovery's success.
_STREAMING_SUCCESS_EVENTS: tuple[bytes, ...] = tuple(
    f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
    for name, payload in [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_recovered_stream", "type": "message", "role": "assistant",
                "model": "stub", "content": [], "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }),
        ("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
)


def _make_413_then_streaming_success_responder() -> Any:
    """Return a responder that 413s on the first call, streams 200 afterwards.

    The shape mirrors :func:`_make_413_then_success_responder` — state in the
    closure, 413 on the first request — but the success reply is the Anthropic
    SSE grammar the streaming Messages path parses, not a JSON body.

    Returns:
        A :class:`~harness.recorder.Responder` closure.
    """
    state = {"calls": 0}

    async def _responder(_captured: CapturedRequest, reply: Reply) -> None:
        """Answer 413 on the first request, an SSE 200 on every later request.

        Args:
            _captured: The capture, unused — state is in the closure.
            reply: The unprepared response.
        """
        is_first = state["calls"] == 0
        state["calls"] += 1

        if is_first:
            reply.content_length = len(_CONTEXT_TOO_LARGE_BODY)
            await reply.begin(413, {"content-type": "application/json"})
            await reply.write(_CONTEXT_TOO_LARGE_BODY)
            await reply.write_eof()
            return

        await reply.begin(200, {"content-type": "text/event-stream"})
        for event in _STREAMING_SUCCESS_EVENTS:
            await reply.write(event)
        await reply.write_eof()

    return _responder


#: The streaming Chat Completions reply the recorder gives for a recovered
#: turn — CC SSE chunk grammar (``chat.completion.chunk`` objects, ``[DONE]``
#: sentinel), the shape the CC-wire streaming route forwards.
_CC_STREAMING_SUCCESS_EVENTS: tuple[bytes, ...] = (
    b'data: {"id":"chatcmpl-recovered","object":"chat.completion.chunk","created":1,'
    b'"model":"stub","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},'
    b'"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n",
)


def _make_413_then_cc_streaming_success_responder() -> Any:
    """Return a responder that 413s once, then streams a CC SSE 200.

    Mirrors :func:`_make_413_then_streaming_success_responder` with the CC
    wire's chunk grammar instead of Anthropic Messages events — the shape the
    ``/v1/chat/completions`` streaming route parses.

    Returns:
        A :class:`~harness.recorder.Responder` closure.
    """
    state = {"calls": 0}

    async def _responder(_captured: CapturedRequest, reply: Reply) -> None:
        """Answer 413 on the first request, a CC SSE 200 on later requests.

        Args:
            _captured: The capture, unused — state is in the closure.
            reply: The unprepared response.
        """
        is_first = state["calls"] == 0
        state["calls"] += 1

        if is_first:
            reply.content_length = len(_CONTEXT_TOO_LARGE_BODY)
            await reply.begin(413, {"content-type": "application/json"})
            await reply.write(_CONTEXT_TOO_LARGE_BODY)
            await reply.write_eof()
            return

        await reply.begin(200, {"content-type": "text/event-stream"})
        for event in _CC_STREAMING_SUCCESS_EVENTS:
            await reply.write(event)
        await reply.write_eof()

    return _responder


def _oversized_cc_body() -> dict[str, Any]:
    """Build an oversized CC conversation for the Chat Completions route.

    Same shape as the committed Messages twin (many ~5k-char alternating
    turns, ~605k serialized chars) so pre-flight compaction short-circuits
    under the test's 800k budget while the body still clears the 600k
    oversized gate. Built programmatically rather than committed as a second
    corpus entry: the corpus carries the inbound Anthropic-Messages capture;
    this body exercises the CC-wire ingress, a different projection of the
    same M6 trigger, and committing it would double the owner-review surface
    for no additional trigger coverage.

    Returns:
        The request body.
    """
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "stream": True,
        "messages": [
            message
            for _ in range(60)
            for message in (
                {"role": "user", "content": "u" * 5_000},
                {"role": "assistant", "content": "a" * 5_000},
            )
        ],
    }


def _load_entry(entry_id: str) -> k.CorpusEntry:
    """Load one committed entry by id.

    Args:
        entry_id: The manifest's filename stem.

    Returns:
        The entry, with its body bytes.

    Raises:
        AssertionError: When the entry is not committed under that id.
    """
    path = CORPUS / f"{entry_id}.json"
    assert path.is_file(), f"{entry_id} is not committed under tests/corpus/"
    return k.load_entry(path)


class TestTheM6EntryExercisesTheRecoveryPathOnABalancingProfile:
    """The scripted-recorder half — the test that declares trigger M6 met.

    Round-robin selection (``pin_backend_order``) is needed so the test reaches
    ``_request_with_retry_balancing`` rather than the single-backend helper —
    ``tests/harness/test_bridge.py::TestBothBridgeShapes::test_pinning_makes_selection_round_robin``
    measured that an undisciplined two-backend test can pass with the
    selection code deleted.
    """

    async def test_a_413_with_an_oversized_body_engages_tighter_recompaction(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The recovery path runs on a 413 with an oversized body.

        Asserts AC-7's two observable effects of the path engaging: the
        recovery log line is emitted, and the recorder's second capture's body
        is strictly smaller than the first's (so the recovery produced a
        smaller body to retry with).
        """
        pin_backend_order(monkeypatch)

        # Sized so pre-flight compaction short-circuits (body 605k < budget
        # 800k - overhead) leaving the body over the 600k oversized gate, and
        # the recovery at factor=0.5 reduces from 605k into a 395k tighter
        # budget — so the second upstream request body is strictly smaller
        # than the first's. The harness profile is not in the model context
        # catalog and would default to ``_MAX_REQUEST_CHARS`` = 4 M, which
        # makes the recovery's tighter budget also huge and the assertion
        # vacuous.
        monkeypatch.setattr(BridgeServer, "_get_max_context_chars", lambda self: 800_000)

        responder = _make_413_then_success_responder()

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            # ``UpstreamTransport`` (Protocol) does not declare ``recorder``;
            # the concrete ``AiohttpTransport`` does, and ``transport()``
            # returns that. Runtime is fine; ``type: ignore`` records the gap.
            recorder = fixture.transport.recorder  # type: ignore[attr-defined]
            entry = _load_entry("m6_recovery_oversized_paired")
            assert entry.id == "m6_recovery_oversized_paired", (
                f"the M6 entry id drifted; got {entry.id!r}"
            )
            inbound_body = json.loads(entry.request.body)

            with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
                status, _text = await fixture.post("/v1/messages", inbound_body)

        assert status == 200, "the second attempt succeeded after the recovery"
        assert len(recorder.requests) == 2, (
            "the M6 path retries the same backend once with a tighter-compacted body"
        )
        first_body = recorder.requests[0].body
        second_body = recorder.requests[1].body
        assert len(second_body) < len(first_body), (
            "the second upstream request body must be smaller than the first; "
            "the recovery path's whole purpose is to ship a smaller body upstream"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "compacting tighter" in log_blob.lower(), (
            "the recovery-path log line must fire; an absent log means the path never ran"
        )

    async def test_a_streaming_413_with_an_oversized_body_engages_tighter_recompaction(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The KBR-256 recovery path runs on a streaming 413 with an oversized body.

        Mirrors the non-streaming case above (same M6 predicate, same recovery
        log line) but drives the streaming Messages handler with
        ``stream: true`` — the shape Claude Code ships. The bridge's pre-byte
        gate means the recovery can fire before any byte reaches the client;
        the upstream is retried to the **same** backend (model identity across
        the two request bodies), and the second body is strictly smaller than
        the first. The client receives the SSE event grammar ``_stream_messages``
        parses (``message_start`` through ``message_stop``), not a JSON 200 the
        empty-response ladder would 502 on.
        """
        pin_backend_order(monkeypatch)
        # Same sizing rationale as the non-streaming case — pre-flight short-circuits,
        # the recovery at factor=0.5 reduces the body into a tighter budget, and the
        # second upstream body is strictly smaller.
        monkeypatch.setattr(BridgeServer, "_get_max_context_chars", lambda self: 800_000)

        responder = _make_413_then_streaming_success_responder()

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            recorder = fixture.transport.recorder  # type: ignore[attr-defined]
            entry = _load_entry("m6_recovery_oversized_paired_streaming")
            assert entry.id == "m6_recovery_oversized_paired_streaming", (
                f"the streaming twin id drifted; got {entry.id!r}"
            )
            inbound_body = json.loads(entry.request.body)
            assert inbound_body.get("stream") is True, (
                "the streaming twin must carry stream: true; the bridge routes "
                "on cc_request['stream'] and a false here would land on the "
                "non-streaming path and re-run the existing test instead"
            )

            with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
                status, response_text = await fixture.post("/v1/messages", inbound_body)
            # Read the backend health while the bridge is still up — the fixture
            # nulls ``server`` on exit. ``_backend_health[0]`` is the backend that
            # 413'd (pin_backend_order makes selection start there).
            backend_health = fixture.server._backend_health  # type: ignore[attr-defined]
            first_backend_healthy = backend_health[0]["healthy"]

        assert status == 200, (
            f"the streaming recovery path returned {status}; a 502 here means "
            "the bridge fell into the empty-response ladder because the 200 "
            "responder emitted JSON instead of SSE"
        )
        assert len(recorder.requests) == 2, (
            "the streaming M6 path must retry the same backend once with a "
            "tighter-compacted body"
        )
        # Same backend on both attempts — model identity, since failover re-normalises
        # the model and the recorder shares the transport across backends. If the bridge
        # had failover'd, ``requests[1]``'s model would differ from ``requests[0]``'s.
        first_model = json.loads(recorder.requests[0].body)["model"]
        second_model = json.loads(recorder.requests[1].body)["model"]
        assert first_model == second_model, (
            f"the recovery retried a different backend (m0 -> {second_model}); "
            "the streaming M6 path must retry the same backend, not failover"
        )
        first_body = recorder.requests[0].body
        second_body = recorder.requests[1].body
        assert len(second_body) < len(first_body), (
            "the second upstream request body must be smaller than the first; "
            "the recovery path's whole purpose is to ship a smaller body upstream"
        )
        # AC-3: the backend that 413'd must not have been marked unhealthy for the
        # recovery — the recovery path explicitly does NOT mark.
        assert first_backend_healthy is True, (
            "the backend that triggered recovery must stay healthy=True; "
            "the recovery path's contract is to skip mark-unhealthy on the "
            "oversized 413 (KBR-256 R2)"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "compacting tighter" in log_blob.lower(), (
            "the streaming recovery-path log line must fire; an absent log "
            "means the path never ran on the streaming ladder"
        )
        # AC-1: the client received SSE. The bridge's streaming Messages path
        # parses ``message_start``/``message_stop`` events; their presence in
        # the response text is the observable that an SSE stream reached the
        # client rather than the empty-response 502.
        assert "event: message_start" in response_text, (
            "the streaming success reply must contain an Anthropic SSE "
            "message_start event the bridge's streaming parser can read"
        )
        assert "event: message_stop" in response_text, (
            "the streaming success reply must terminate with the SSE "
            "message_stop event"
        )

    async def test_a_streaming_413_that_cannot_compact_fails_over_without_marking(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The CompactionFailedError fallback fails over without marking (AC-4).

        Isolates the recovery's ``CompactionFailedError`` arm: the M6 predicate
        passes, the recovery fires, but the compactor cannot shrink further —
        so the fallback must select the next backend **without** marking either
        backend unhealthy (the 413 is the conversation's fault, not a backend's;
        the non-streaming fallback at ``_request_with_retry_balancing`` never
        marks on this arm either). The client still receives a terminal
        outcome — the failover's own reply.
        """
        pin_backend_order(monkeypatch)
        monkeypatch.setattr(BridgeServer, "_get_max_context_chars", lambda self: 800_000)

        calls = {"compactions": 0}

        def _raising_compact(self: BridgeServer, cc_request: dict, factor: float = 0.5) -> None:
            """Raise the exhaustion error the fallback arm exists for.

            Args:
                self: The server instance, unused — the raise is unconditional.
                cc_request: The request payload, unused.
                factor: The compaction factor, unused.
            """
            calls["compactions"] += 1
            raise CompactionFailedError("test: cannot shrink further")

        monkeypatch.setattr(BridgeServer, "_compact_with_tighter_budget", _raising_compact)

        responder = _make_413_then_streaming_success_responder()

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            recorder = fixture.transport.recorder  # type: ignore[attr-defined]
            entry = _load_entry("m6_recovery_oversized_paired_streaming")
            inbound_body = json.loads(entry.request.body)

            with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
                status, _text = await fixture.post("/v1/messages", inbound_body)
            backend_health = fixture.server._backend_health  # type: ignore[attr-defined]
            healths = [backend_health[i]["healthy"] for i in range(len(backend_health))]

        assert calls["compactions"] == 1, (
            "the recovery must have fired exactly once before exhausting"
        )
        assert status == 200, "the fallback's failover reply must reach the client"
        assert len(recorder.requests) == 2, (
            "the fallback must have re-issued the request to the failover backend"
        )
        first_model = json.loads(recorder.requests[0].body)["model"]
        second_model = json.loads(recorder.requests[1].body)["model"]
        assert first_model != second_model, (
            f"the fallback must fail over to a different backend; both requests "
            f"went to {first_model}"
        )
        assert all(healths), (
            f"no backend may be marked unhealthy on the CompactionFailedError "
            f"fallback; healths: {healths}"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "cannot be compacted further" in log_blob.lower(), (
            "the fallback's log line must fire; an absent log means the "
            "CompactionFailedError arm never ran"
        )

    async def test_a_streaming_413_on_a_small_body_fails_over_without_recovering(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 413 on a small streaming body takes the standard failover (AC-2).

        Compaction only helps when there is content to shrink: the M6 predicate
        requires ``_is_oversized_request``, so a 413 on a small body is the
        pre-existing generic-failure arm's input — mark unhealthy and fail
        over. This pins the negative half of R1 on the streaming code path the
        same way ``test_non_streaming_small_request_1261_fails_over`` pins it
        on the non-streaming ladder (tests/bridge/test_stage11_oversized.py).
        """
        pin_backend_order(monkeypatch)

        responder = _make_413_then_streaming_success_responder()

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            recorder = fixture.transport.recorder  # type: ignore[attr-defined]
            small_streaming_body = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }

            with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
                status, response_text = await fixture.post("/v1/messages", small_streaming_body)
            backend_health = fixture.server._backend_health  # type: ignore[attr-defined]
            first_backend_healthy = backend_health[0]["healthy"]

        assert status == 200, "the failover's reply must reach the client"
        assert len(recorder.requests) == 2, (
            "the standard failover issues one request per backend"
        )
        first_model = json.loads(recorder.requests[0].body)["model"]
        second_model = json.loads(recorder.requests[1].body)["model"]
        assert first_model != second_model, (
            f"the small-body 413 must fail over to a different backend; both "
            f"requests went to {first_model}"
        )
        assert first_backend_healthy is False, (
            "the small-body 413 must mark the backend unhealthy (standard "
            "failover); the recovery path is the only arm that skips the mark"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "compacting tighter" not in log_blob.lower(), (
            "the recovery must NOT fire on a small body; its log line "
            "appearing means the oversized gate failed to screen this request"
        )
        assert "event: message_start" in response_text, (
            "the client still receives the SSE success stream from the "
            "failover backend"
        )

    async def test_a_single_backend_streaming_413_recovers_without_a_pool(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The streaming recovery is not balancing-only (KBR-256 round-4 review).

        The M6 row's pre-KBR-256 prose scoped the recovery to balancing
        profiles because the non-streaming arm lives in
        ``_request_with_retry_balancing`` alone. The streaming arms gate on
        ``recovery_retries < n_backends`` with ``n_backends = 1`` in
        single-backend mode, so a single-profile streaming 413 recovers the
        same way — the PR's own motivation (``Claude Code`` ships
        ``stream: true``; today the only backend is cooled down for a
        conversation that is merely too big). This test holds the prose claim:
        one backend, oversized streaming 413, compact-retry the same backend,
        no mark.

        Args:
            monkeypatch:pytest.MonkeyPatch
            caplog: The log capture.
        """
        monkeypatch.setattr(BridgeServer, "_get_max_context_chars", lambda self: 800_000)

        responder = _make_413_then_streaming_success_responder()

        async with BridgeFixture(
            transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES, responder=responder),
        ) as fixture:
            recorder = fixture.transport.recorder  # type: ignore[attr-defined]
            entry = _load_entry("m6_recovery_oversized_paired_streaming")
            inbound_body = json.loads(entry.request.body)

            with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
                status, response_text = await fixture.post("/v1/messages", inbound_body)

        assert status == 200, "the single-backend recovery reply must reach the client"
        # No _backend_health assertion here: the single-backend shape has no
        # health list at all (`_mark_backend_unhealthy` returns early without a
        # pool), which is why the pre-KBR-256 single-backend 413 could only
        # surface the error — there was no failover to take the backend's place.
        assert len(recorder.requests) == 2, (
            "the single-backend streaming path must compact and retry the only "
            "backend once, not mark-and-surface"
        )
        first_model = json.loads(recorder.requests[0].body)["model"]
        second_model = json.loads(recorder.requests[1].body)["model"]
        assert first_model == second_model, (
            "both attempts must go to the single backend — there is nowhere "
            "else to fail over to"
        )
        assert len(recorder.requests[1].body) < len(recorder.requests[0].body), (
            "the retried body must be the tighter-compacted one"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "compacting tighter" in log_blob.lower(), (
            "the recovery log line must fire on the single-backend shape"
        )
        assert "event: message_stop" in response_text, (
            "the client receives the SSE success stream from the recovered "
            "attempt"
        )

    async def test_the_cc_streaming_413_with_an_oversized_body_engages_tighter_recompaction(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The KBR-256 recovery fires on the Chat Completions streaming route.

        Mirrors the Messages streaming case — same M6 predicate, same compact-
        and-retry-same-backend semantics — but drives the ``/v1/chat/
        completions`` ingress with a programmatically-built oversized CC body
        and a CC SSE success reply. The CC route is the OpenAI-compatible
        ingress; it is the second of the four streaming routes the recovery
        was added to, and the second covered by the aiohttp harness here.
        Responses and Gemini are deferred on recorder work (curl_cffi /
        botocore respectively).
        """
        pin_backend_order(monkeypatch)
        monkeypatch.setattr(BridgeServer, "_get_max_context_chars", lambda self: 800_000)

        responder = _make_413_then_cc_streaming_success_responder()

        async with BridgeFixture(
            transport("aiohttp", WireFormat.CHAT_COMPLETIONS, responder=responder),
            backend_models=["m0", "m1"],
        ) as fixture:
            recorder = fixture.transport.recorder  # type: ignore[attr-defined]
            inbound_body = _oversized_cc_body()
            assert inbound_body.get("stream") is True, (
                "the test body must carry stream: true; without it the bridge "
                "would not engage the streaming Chat Completions route"
            )

            with caplog.at_level(logging.INFO, logger="kitty.bridge.server"):
                status, response_text = await fixture.post("/v1/chat/completions", inbound_body)
            first_backend_healthy = fixture.server._backend_health[0]["healthy"]  # type: ignore[attr-defined]

        assert status == 200, "the CC streaming recovery reply must reach the client"
        assert len(recorder.requests) == 2, (
            "the CC streaming M6 path must retry the same backend once with a "
            "tighter-compacted body"
        )
        first_model = json.loads(recorder.requests[0].body)["model"]
        second_model = json.loads(recorder.requests[1].body)["model"]
        assert first_model == second_model, (
            f"the recovery retried a different backend (m0 -> {second_model}); "
            "the CC streaming M6 path must retry the same backend, not failover"
        )
        first_body = recorder.requests[0].body
        second_body = recorder.requests[1].body
        assert len(second_body) < len(first_body), (
            "the second upstream request body must be smaller than the first"
        )
        assert first_backend_healthy is True, (
            "the CC streaming recovery path must not mark the backend unhealthy"
        )
        log_blob = "\n".join(record.getMessage() for record in caplog.records)
        assert "compacting tighter" in log_blob.lower(), (
            "the CC streaming recovery log line must fire; an absent log "
            "means the recovery never ran on the CC ladder"
        )
        assert "chatcmpl-recovered" in response_text, (
            "the client must receive a CC SSE success stream "
            "(chat.completion.chunk object) from the recovered attempt"
        )
