"""The botocore recorder and its transport, against real sockets.

`.system_design/TEST_SUITE.md` §3.2.3, §5.5, §7.2, §7.2.3, §7.5 · plan task
**T-B3** (KBR-42) ·
`.requirements/20260914T182837Z_botocore_endpoint_recorder/REQUIREMENTS.md`.

Two products are under observation here and they are driven differently, because
they are reached differently. ``bedrock`` is driven through a real
``BridgeServer`` on T-W8's fixture; the EventStream encoder is round-tripped
through the pinned ``botocore.eventstream`` parser only (see below for why no
live-boto3 round-trip is exercised here).

**A live ``boto3.client(...).converse_stream(...)`` round-trip is not exercised
in this module.** Measured: boto3's synchronous ``urllib3.PoolManager``
blocks the asyncio loop on ``socket.recv`` while the recorder's aiohttp
server's response waits to be processed — every such test times out at 60
seconds with ``ReadTimeoutError`` and **never reaches the recorder's capture
method**. The pinned ``botocore.eventstream.EventStreamBuffer`` parser
round-trip in :class:`TestTheRepliesItSends` is what proves the encoder
matches the wire format, and the
:class:`TestThroughARealBridge` class proves the recorder answers a real
``BridgeServer`` driving the product's own adapter. A live-boto3 round-trip
belongs in a thread-based test or a separate worktree-driven harness, not in
the L1 gate.

Every conformance probe writes its request bytes itself, for the reason
``test_recorder.py`` records: a client library reorders, re-cases, adds and
drops headers, so a probe sent through one could not tell a recorder that
*loses* casing from a client that never *sent* mixed casing.

**Layer.** These bind real sockets and read as ``l3``, but they carry the
``l1`` path default deliberately: §8.2 states *"a test may not be moved to
``l3`` before the Subsystem job exists"*, and no job selects ``l3`` today, so
an ``l3`` marker would remove them from every gate. The same reasoning
``test_provider_aiohttp.py`` records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import aiohttp
import pytest

from harness.botocore import BotocoreTransport
from harness.botocore_recorder import (
    BEDROCK_CONVERSE_STREAM_SUFFIX,
    BEDROCK_CONVERSE_SUFFIX,
    BedrockRecordingUpstream,
    bedrock_success_body,
    bedrock_success_stream_events,
    encode_eventstream,
    format_for_bedrock_path,
)
from harness.bridge import (
    BridgeFixture,
    InboundProtocol,
    UpstreamTransport,
    assert_transport_reaches_its_recorder,
    inbound_path,
    marker,
    minimal_inbound_body,
    registered_transports,
    transport,
)
from harness.connect_proxy import unattributable_peer_ports
from harness.contract import WireFormat
from harness.recorder import UnmatchedPathError
from harness.recorder_conformance import (
    PER_EXCHANGE_CHECKS,
    PER_SESSION_CHECKS,
    RICH_PROBE,
    correlate,
    probe,
    recording_of,
    send,
)

#: The format this transport serves, named once so a case differs from its
#: neighbours only in the thing it is about.
FORMAT = WireFormat.BEDROCK_CONVERSE

#: The inbound route that exercises it. ``BEDROCK_CONVERSE`` has no inbound
#: route of its own — the bridge reaches it upstream and never serves it —
#: so this is named, not derived (§7.5.1). The Bedrock adapter is a
#: Chat-Completions-wire adapter, so the bridge's ``/v1/chat/completions``
#: route reaches it.
ROUTE = InboundProtocol.CHAT_COMPLETIONS

#: AWS-published documentation fake credentials. CI has no AWS credentials
#: and no real network; botocore does not validate the auth header against
#: AWS (the recorder receives the request, not AWS). These are the example
#: values from the AWS docs and are not credentials of any kind.
_FAKE_AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
_FAKE_AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_FAKE_AWS_REGION = "us-east-1"

#: T-W4's rich probe, re-aimed at this recorder's endpoint. Derived rather
#: than copied: what makes that constant valuable is its mixed header casing,
#: its duplicated header, its obs-text value and its two differently-cased
#: percent escapes, and a hand-written near-copy is a second source of truth
#: that stays green when T-W4 adds a hostile case to the original. Only the
#: dispatch suffix differs, and §7.2's own note on ``RICH_PROBE`` says why
#: that matters: a probe matching no suffix would silently exercise the
#: wrong-format fallback in every test that used it.
#:
#: The ``count=1`` argument is explicit so a future reader does not have to
#: re-derive it: the first match is the only one — ``RICH_PROBE``'s path
#: carries ``/chat/completions`` once.
RICH_BEDROCK_PROBE = RICH_PROBE.replace(b"/chat/completions", b"/converse", 1)

#: Forbidden-source-tree import (mirrors ``test_provider_aiohttp.py`` and
#: ``test_recorder.py``). The recorder must not import the product it
#: observes — §3.3.1's independent-oracle rule. ``import boto3`` /
#: ``import botocore`` are allowed (they are the provider family, not the
#: bridge); only ``from kitty`` / ``import kitty`` are forbidden.
_KITTY_IMPORT = re.compile(r"^\s*(?:from\s+kitty|import\s+kitty)\b", re.MULTILINE)


@pytest.fixture
async def recorder():
    """Start a botocore recorder on an ephemeral loopback port, and stop it after.

    One instance per test, deliberately: a shared recorder mixes captures across
    tests, and the ordering claims are only meaningful per-instance.

    Yields:
        The running :class:`~harness.botocore_recorder.BedrockRecordingUpstream`.
    """
    upstream = BedrockRecordingUpstream(default_format=FORMAT)
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()
        # The loud fallback, enforced structurally rather than by each test
        # remembering to ask: a wrong-format reply is not a loud failure on
        # its own, it is an apparently empty response and the retry ladder.
        upstream.assert_all_paths_matched()


def test_botocore_recorder_imports_nothing_from_kitty() -> None:
    """The recorder must not inherit the product's bugs.

    §3.3.1's independent-oracle rule governs everything the fidelity claim
    rests on: a recorder that asked kitty how to parse a body would inherit
    kitty's bugs. ``import boto3`` and ``import botocore`` are allowed — they
    are the provider family, not the bridge; only ``from kitty`` / ``import
    kitty`` are forbidden.
    """
    source = Path(__file__).parent.joinpath("botocore_recorder.py").read_text(encoding="utf-8")
    assert not _KITTY_IMPORT.search(source), "the recorder imports from src/kitty, defeating §3.3.1"


def test_the_rich_probe_really_targets_this_recorder() -> None:
    """The derived probe must differ from the original and select this format.

    A ``replace`` that matched nothing would leave the probe aimed at the
    primary recorder's suffix, and every per-exchange check below would then
    be run over a request that took the fallback — the self-check §6.2
    requires of anything that derives its own subject.
    """
    assert RICH_BEDROCK_PROBE != RICH_PROBE, "the suffix rewrite matched nothing"

    target = RICH_BEDROCK_PROBE.split(b" ")[1].split(b"?")[0].decode()
    assert format_for_bedrock_path(target) is FORMAT


class TestTheRecorderPassesEveryConformanceCheck:
    """R3 — this recorder satisfies the contract T-W4 judges all four against."""

    @pytest.mark.parametrize("check", PER_EXCHANGE_CHECKS, ids=lambda c: c.__name__)
    async def test_per_exchange_check(self, recorder: BedrockRecordingUpstream, check) -> None:
        """Assert one capture matches the request that produced it.

        Args:
            recorder: The running recorder.
            check: One of the ten per-exchange checks, ``check_peer_port``
                among them — the field T-E4's tunnel join reads.
        """
        sent = await send(recorder.host, recorder.port, RICH_BEDROCK_PROBE, marker="rich")
        captured = correlate(recording_of(recorder), sent)
        check(captured, sent, scheme=recorder.scheme)

    @pytest.mark.parametrize("check", PER_SESSION_CHECKS, ids=lambda c: c.__name__)
    async def test_per_session_check(self, recorder: BedrockRecordingUpstream, check) -> None:
        """Assert the recording as a whole is faithful.

        Args:
            recorder: The running recorder.
            check: One of the four per-session checks.
        """
        sent = [
            await send(recorder.host, recorder.port, probe(name, path=BEDROCK_CONVERSE_SUFFIX), marker=name)
            for name in ("first", "second")
        ]
        check(recording_of(recorder), sent, [s.source_port for s in sent])


class TestTheRepliesItSends:
    """R1, R2 — the three bodies, judged by what actually has to read them."""

    def test_the_non_streaming_success_reads_as_a_success_to_the_product(self) -> None:
        """The real adapter must get content and a finish reason out of it.

        Asserting the keys instead would assert this module agrees with
        itself. ``translate_from_upstream`` is what stands between this body
        and the bridge's emptiness judgement, so it is what decides whether
        the reply is a success or an 80-second retry ladder.
        """
        from kitty.providers.bedrock import BedrockAdapter

        translated = BedrockAdapter().translate_from_upstream(bedrock_success_body())

        choice = translated["choices"][0]
        assert choice["message"]["content"], "the reply reads as empty, which costs the retry ladder"
        assert choice["finish_reason"] == "stop"

    def test_the_streamed_success_has_a_message_stop(self) -> None:
        """Without a ``messageStop`` the adapter emits no finish reason.

        ``_translate_stream_event`` produces a CC ``finish_reason`` only when
        it sees ``messageStop``; the other three events carry content, not
        a finish. An earlier draft sent ``messageStart`` /
        ``contentBlockDelta`` / ``contentBlockStop`` only, and the bridge
        judged the stream empty — measured.
        """
        events = bedrock_success_stream_events()
        types = [next(iter(event.keys())) for event in events]
        assert "messageStop" in types, "no messageStop means no finish_reason, which costs the retry ladder"

    def test_a_streaming_reply_decodes_via_the_pinned_botocore_parser(self) -> None:
        """The encoder must round-trip through the parser every AWS SDK ships.

        The pinned ``botocore.eventstream.EventStreamBuffer`` is what boto3
        uses in production, so a round-trip here is the cheapest proof the
        encoder matches the wire format. A failure here would silently
        reach boto3 and surface as ``EventStream`` parse errors three
        layers away.
        """
        import botocore.eventstream as ev

        encoded = encode_eventstream(bedrock_success_stream_events())

        buf = ev.EventStreamBuffer()
        buf.add_data(encoded)

        decoded = [m for m in buf]
        assert len(decoded) == 4, "the parser dropped a frame"

        wire_types = [m.headers[":event-type"] for m in decoded]
        assert wire_types == ["messageStart", "contentBlockDelta", "contentBlockStop", "messageStop"]

        # Each payload is the JSON we passed in. One event is enough to prove
        # the body survived framing; the type list proves the others did.
        start = json.loads(decoded[0].payload.decode())
        assert start == {"role": "assistant"}

    def test_a_streaming_reply_with_a_bad_message_crc_fails_to_decode(self) -> None:
        """A wrong message CRC must fail the pinned parser.

        The harness rule (§1.4) requires this case to exist: the conformance
        suite runs unchanged over the subclass and so cannot tell a wrong
        encoder from a working one. A frame with a corrupted message CRC is
        the load-bearing test of the encoder's CRC; the suite passes only
        when this case *fails* the round-trip, demonstrating the recorder
        can fail.
        """
        import botocore.eventstream as ev

        encoded = encode_eventstream(bedrock_success_stream_events())
        # Flip the last byte of the trailing message CRC. `encoded[:-1]`
        # keeps every other byte, so the buffer still holds the full frame
        # and the parser reaches the corrupted CRC — flipping with
        # `encoded[:-4]` would truncate three bytes and the parse would die
        # of "not enough data" instead.
        corrupted = encoded[:-1] + bytes([encoded[-1] ^ 0xFF])

        buf = ev.EventStreamBuffer()
        buf.add_data(corrupted)

        # Three good frames yield first, then the corrupt one raises —
        # iterate to exhaustion, never just once.
        with pytest.raises(ev.ChecksumMismatch):
            for _ in buf:
                pass

    def test_encoding_is_deterministic(self) -> None:
        """Two encodes of the same events must produce identical bytes.

        A second harness-rule falsification (§1.4): the conformance suite
        cannot tell a stateful encoder from a deterministic one. Encoding
        the same events twice and asserting byte equality means a
        regression that introduces e.g. an incrementing counter or a
        time-dependent header fails this case — without it, two captures
        of an identical stream would diff and the suite would chase the
        diff into the recorder instead of into the real defect.
        """
        events = bedrock_success_stream_events()
        a = encode_eventstream(events)
        b = encode_eventstream(events)
        assert a == b, "the encoder is non-deterministic — re-encoding the same events must yield identical bytes"

    def test_a_format_this_recorder_does_not_serve_is_refused_at_construction(self) -> None:
        """Failing here beats replying in a format no adapter asked for."""
        with pytest.raises(ValueError, match="bedrock_converse"):
            BedrockRecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)

    async def test_the_replies_carry_the_headers_a_provider_would_send(
        self, recorder: BedrockRecordingUpstream
    ) -> None:
        """A recorder impersonates a provider, and headers are part of that.

        Neither claim is visible to the adapter — it reads the body and
        ignores both — so nothing else in this module goes red if either is
        wrong. The non-streaming reply is JSON with a length rather than
        chunked; the streaming reply carries
        ``application/vnd.amazon.eventstream``.

        **Path selects the reply format here, not the body.** The other
        recorders read ``"stream": true`` out of the body — the Bedrock
        transport pops that key before the HTTP request is built (register
        row P18). A body-based check on ``/converse`` would always see no
        ``stream`` and the streaming assertion below is exercised against
        the path that actually selects the streamed reply.

        Args:
            recorder: The running recorder.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{recorder.base_url}{BEDROCK_CONVERSE_SUFFIX}", json={"stream": False}) as whole:
                assert whole.headers["Content-Type"] == "application/json"
                assert "Content-Length" in whole.headers
                assert "Transfer-Encoding" not in whole.headers

            async with session.post(
                f"{recorder.base_url}{BEDROCK_CONVERSE_STREAM_SUFFIX}", json={"stream": True}
            ) as streamed:
                assert streamed.headers["Content-Type"] == "application/vnd.amazon.eventstream"
                await streamed.read()

    async def test_a_path_matching_no_suffix_is_recorded(self, recorder: BedrockRecordingUpstream) -> None:
        """The teardown assertion has to have something to report.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, probe("stray", path="/v1/chat/completions"), marker="stray")

        assert recorder.unmatched == ["/v1/chat/completions"]
        with pytest.raises(UnmatchedPathError, match="/v1/chat/completions"):
            recorder.assert_all_paths_matched()

        # Clear it, or the fixture's own teardown check reports this deliberate
        # miss as a failure.
        recorder.unmatched.clear()

    def test_the_two_suffixes_are_distinct_questions(self) -> None:
        """``converse`` and ``converse-stream`` are different endpoints.

        The two endpoints reply in different formats — JSON vs EventStream
        — and a streaming reply to a non-streaming request (or vice versa)
        costs the 80-second retry ladder. Asserting the table does not
        conflate them is what keeps that separation real.
        """
        assert format_for_bedrock_path(f"/v1{BEDROCK_CONVERSE_SUFFIX}") is FORMAT
        assert format_for_bedrock_path(f"/v1{BEDROCK_CONVERSE_STREAM_SUFFIX}") is FORMAT
        assert format_for_bedrock_path("/v1/chat/completions") is None


class TestTheTransport:
    """R4 — what T-W8's interface asks of a transport, on this one."""

    def test_it_satisfies_the_extension_interface(self) -> None:
        """The check is structural: T-W8's protocol is ``runtime_checkable``."""
        assert isinstance(BotocoreTransport(FORMAT), UpstreamTransport)

    def test_importing_the_module_registers_it(self) -> None:
        """An Epic B transport is reachable by name once its module is imported."""
        assert BotocoreTransport.name in registered_transports()

    def test_binding_before_starting_raises(self) -> None:
        """There is no port to name until the recorder has one."""
        with pytest.raises(RuntimeError, match="not running"):
            BotocoreTransport(FORMAT).bind()

    async def test_it_binds_a_bedrock_adapter_pointed_at_its_recorder(self) -> None:
        """``bind()`` is the seam: the fixture never builds this config itself."""
        from kitty.providers.bedrock import BedrockAdapter

        subject = BotocoreTransport(FORMAT)
        await subject.start()
        try:
            adapter, config = subject.bind()

            # The harness adapter is a subclass: ``isinstance`` holds against
            # the base class, and the override's only job is resolving the
            # harness key to the fake pair.
            assert isinstance(adapter, BedrockAdapter)
            assert config == {"endpoint_url": subject.recorder.base_url, "region": _FAKE_AWS_REGION}
        finally:
            await subject.stop()

    async def test_bind_shares_the_one_adapter_across_calls(self) -> None:
        """``bind()`` returns the same adapter on repeated calls.

        **One adapter per transport, reused.** T-B1's docstring states this
        and T-W8's :func:`backend_for` expects it — balancing members share
        one adapter so :meth:`stop` has a single session to close. The
        bedrock adapter is stateless across calls (``_get_boto3_client``
        builds a fresh client per request, KBR-190), so reusing the
        instance carries no risk of state leaks.

        Distinct from ``tests/test_provider_bedrock.py::TestItCachesNoTransport``,
        which asserts the *adapter-level* property (a fresh client per
        ``_get_boto3_client`` call, no cache on the instance) and which
        already passes — this test asserts the *binding-layer* property
        instead.
        """
        subject = BotocoreTransport(FORMAT)
        await subject.start()
        try:
            adapter_one, config_one = subject.bind()
            adapter_two, config_two = subject.bind()

            assert adapter_one is adapter_two, (
                "two binds returned different instances — backend_for shares the binding across members"
            )
            assert config_one == config_two, "two binds returned different configs for the same transport"
        finally:
            await subject.stop()


class TestThroughARealBridge:
    """R1, R4, R5 — the claim the whole delivery exists to support."""

    async def test_a_request_reaches_the_recorder_as_converse(self) -> None:
        """The user's text survives the bridge's CC → Converse translation.

        The inbound CC body is also asserted on, not only on the capture:
        the non-streaming reply is otherwise judged only by the fixture
        ``status == 200``, which cannot tell a correct translation from a
        plausible-but-empty one (the streaming path has this in
        :meth:`test_a_streamed_request_via_the_bridge_yields_finish_reason`
        — this test carries the non-streaming half, so a regression in
        ``translate_from_upstream`` that returned a contentless CC body
        would be caught at the boundary the client sees).
        """
        subject = BotocoreTransport(FORMAT)
        sent = marker()

        async with BridgeFixture(subject) as fixture:
            status, text = await fixture.post(inbound_path(ROUTE), minimal_inbound_body(ROUTE, sent))
            captures = list(subject.captures)

        assert status == 200
        assert len(captures) == 1
        assert captures[0].path.endswith(BEDROCK_CONVERSE_SUFFIX)
        assert sent.encode() in (captures[0].body or b"")

        # The non-streaming CC reply, as the client sees it: non-empty
        # content and a finish reason. Without this the test above proves
        # only that a 200 arrived, which a plausible-but-empty translation
        # also satisfies.
        reply = json.loads(text)
        choice = reply["choices"][0]
        assert choice["message"]["content"], "the reply reads as empty, which costs the retry ladder"
        assert choice["finish_reason"] == "stop"

    async def test_the_capture_lacks_modelid_and_stream(self) -> None:
        """P18's mutation is observable on the wire.

        This is the **characterisation capture** for [KBR-89](https://shelpuk.atlassian.net/browse/KBR-89)
        (T-H2 — extract ``_bedrock_body``): the bedrock transport pops
        ``modelId`` and ``stream`` from the Converse body before posting
        (P18) and lifts them into call arguments or path segments; if the
        capture carries them, P18 has regressed. The capture also asserts
        the body carries the fields the translation **does** keep, so a
        future reader can tell translation regression apart from P18
        regression.
        """
        subject = BotocoreTransport(FORMAT)

        async with BridgeFixture(subject) as fixture:
            await fixture.post(inbound_path(ROUTE), minimal_inbound_body(ROUTE, marker(), stream=False))
            captures = list(subject.captures)

        assert len(captures) == 1
        body = json.loads((captures[0].body or b"").decode())

        # P18's mutation: these must not be in the body. The ``modelId``
        # half is real and load-bearing; the ``stream`` half is a guard
        # against a future ``translate_to_upstream`` that adds the key —
        # today the Converse payload never carries it.
        assert "modelId" not in body, "P18 popped modelId; finding it here means the pop regressed"
        assert "stream" not in body, "P18 popped stream; finding it here means the pop regressed"

        # Translation: these must be in the body.
        assert "messages" in body, "translation must carry messages"
        assert "inferenceConfig" in body, "translation must carry inferenceConfig"
        assert body["messages"], "messages must not be empty for a minimal inbound"

    async def test_the_capture_carries_the_peer_port_containment_joins_on(self) -> None:
        """§5.2.1's join key, required of every recorder and not only the primary."""
        subject = BotocoreTransport(FORMAT)

        async with BridgeFixture(subject) as fixture:
            await fixture.post(inbound_path(ROUTE), minimal_inbound_body(ROUTE, marker()))
            captures = list(subject.captures)
            connections = list(subject.connections)

        assert captures[0].peer_port
        assert [c.peer_port for c in connections] == [captures[0].peer_port]
        assert sum(c.requests for c in connections) == 1

        # The join is over **connections**, never requests (§5.2.1), and
        # `unattributable_peer_ports` raises rather than guess when a port is
        # unset -- so putting this transport's own connection log through it
        # is what proves the field satisfies T-E4's precondition, not merely
        # that something was recorded. With no tunnels to explain them,
        # every port comes back unattributed; that is the shape, and T-E4
        # supplies the proxy that makes the list empty.
        assert unattributable_peer_ports([c.peer_port for c in connections], []) == [captures[0].peer_port]

    async def test_a_streamed_request_via_the_bridge_yields_finish_reason(self) -> None:
        """The converse_stream path end-to-end produces a finish_reason."""
        subject = BotocoreTransport(FORMAT)
        sent = marker()

        async with BridgeFixture(subject) as fixture:
            status, text = await fixture.post(
                inbound_path(ROUTE), minimal_inbound_body(ROUTE, sent, stream=True)
            )
            captures = list(subject.captures)

        assert status == 200
        assert "data: [DONE]" in text
        # The finish reason comes from the messageStop event's stopReason.
        # Without it the stream's last chunk could be malformed and the test
        # would still pass on the content chunk alone -- measured, not
        # supposed.
        assert '"finish_reason": "stop"' in text or '"finish_reason":"stop"' in text
        # **The path suffix is what signals streaming, not the body.** The
        # Bedrock transport pops ``stream`` from the Converse payload
        # before the HTTP request is built (register row **P18**); the
        # capture's path carries ``/converse-stream`` and the body lacks
        # the key the same way the bridge's CC inbound request would be
        # missing the field it just popped.
        body = json.loads(captures[0].body)
        assert captures[0].path.endswith(BEDROCK_CONVERSE_STREAM_SUFFIX)
        assert "stream" not in body, "P18 popped stream; finding it here means the pop regressed"
        assert "modelId" not in body, "P18 popped modelId; finding it here means the pop regressed"

    async def test_it_passes_the_integration_conformance_check(self) -> None:
        """R5 — the check every transport is judged by, on this one by name.

        The meta-test in ``test_bridge.py`` runs the same check from the
        ``CONFORMANCE_CASES`` row. Both are wanted: that one proves the row
        is wired, this one fails in this module when the transport is what
        broke.
        """
        await assert_transport_reaches_its_recorder(transport(BotocoreTransport.name, FORMAT), protocol=ROUTE)
