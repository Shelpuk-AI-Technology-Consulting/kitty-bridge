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

from kitty.bridge.server import BridgeServer

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
