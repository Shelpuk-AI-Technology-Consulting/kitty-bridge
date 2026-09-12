"""The primary aiohttp recorder, against real sockets.

`.system_design/TEST_SUITE.md` §7.2 · plan task **T-W4** (KBR-27).

Every probe here writes its request bytes itself. A client library is free to
reorder, re-case, add and drop headers — aiohttp's own client does several of
those — so a test that sent through one could not tell a recorder that *loses*
casing from a client that never *sent* mixed casing.

**Layer.** These bind real sockets and read as `l3`, but they carry the `l1`
path default deliberately: §8.2 states *"a test may not be moved to `l3` before
the Subsystem job exists"*, and today no job selects `l3`, so an `l3` marker
would remove them from every gate. `tests/test_egress_https_proxy.py` is `l1` for
the same reason, and T-K6 owns the reclassification together with the job that
runs it.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import socket
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import pytest
from exemptions import ratchet

import harness.recorder as recorder_module
import harness.recorder_conformance as conformance_module
from harness.contract import CapturedRequest, WireFormat
from harness.recorder import (
    RecordingUpstream,
    Reply,
    UnmatchedPathError,
    format_for_path,
    minimal_success_body,
    minimal_success_stream,
)
from harness.recorder_conformance import (
    NO_HOST_PROBE,
    PER_EXCHANGE_CHECKS,
    PER_SESSION_CHECKS,
    RICH_PROBE,
    check_arrival_increases,
    check_connection_logged,
    correlate,
    probe,
    recording_of,
    send,
)
from harness.test_contract import _KITTY_IMPORT


@pytest.fixture
async def recorder():
    """Start a recorder on an ephemeral loopback port, and stop it after.

    One instance per test, deliberately: a shared recorder mixes captures across
    tests, and the ordering claims are only meaningful per-instance.

    Yields:
        The running :class:`RecordingUpstream`.
    """
    upstream = RecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()
        # R2.2's loud fallback, enforced structurally rather than by each test
        # remembering to ask. A wrong-format reply is not a loud failure on its
        # own -- the adapter parses nothing out of it, the bridge reads the
        # response as empty, and the test pays the 80-second retry ladder.
        upstream.assert_all_paths_matched()


#: The condition wait: 400 steps of 5 ms, so a genuinely stuck test fails in
#: about two seconds instead of hanging the gate, while a healthy one returns
#: on the first poll.
_SETTLE_ATTEMPTS = 400
_SETTLE_STEP = 0.005

#: How long to wait for a reply before giving up on it.
_REPLY_TIMEOUT = 5.0


#: A clock step far coarser than a probe pair takes, so two back-to-back probes
#: land in one bucket unless something waits. Windows' real step is ~15.6 ms and
#: a probe pair is a couple of milliseconds, which is the same relationship; this
#: widens the gap so the property is exercised rather than raced for. It must
#: stay inside the driver's poll budget, or it would rightly give up waiting.
_COARSE_CLOCK_STEP = 0.05

#: Windows' real ``time.monotonic`` step, used by the *stepped* clock below
#: rather than by the quantised one -- there the value only has to be visible,
#: here it stands for the platform being modelled.
_STEPPING_CLOCK_STEP = 0.0156


class _ReachedTheSocket(Exception):
    """Stop a probe at the socket boundary, where the placement case reads it."""


class TestTheDriverSeparatesProbesOnACoarseClock:
    """KBR-188 — the Windows failure, reproduced on every platform.

    ``check_arrival_increases`` asserts arrival stamps *strictly* increase. Two
    probes sent back to back are separated by far less than Windows'
    ~15.6 ms clock step, so the recorder stamps both at the same instant there
    and the assertion is false — while on Linux and macOS, which resolve to
    nanoseconds, it is true. KBR-164 met that as a red leg and KBR-188 exempted
    the cells; but an exemption fails when its assertion *passes*, and whether a
    given pair collides is a race, so the leg went red in **both** directions on
    alternate runs (measured on CI, 2026-09-12: one run failed the assertion,
    the next failed the exemption for passing).

    :func:`~harness.recorder_conformance._advance_clock` moves the fix into the
    driver, where §7.2's four recorders share it. These cases pin it against a
    **simulated** coarse clock, so the property is proven on the platform this
    suite actually runs on rather than only by watching a Windows leg go green.
    """

    @staticmethod
    def _coarse(monkeypatch: pytest.MonkeyPatch, step: float) -> None:
        """Quantise ``time.monotonic`` to ``step``, process-wide.

        Patching the stdlib module object is what makes this faithful: the
        recorder stamps arrivals with the same clock the driver waits on, which
        is exactly the situation on Windows.

        Args:
            monkeypatch: Reverts the patch, which is what contains a change this
                broad.
            step: The quantum. Zero freezes the clock outright.
        """
        real = time.monotonic
        origin = real()

        def quantised() -> float:
            """Return the current instant, floored to ``step``.

            Returns:
                A non-decreasing time that changes only once per step.
            """
            if not step:
                return origin
            return (real() // step) * step

        monkeypatch.setattr(time, "monotonic", quantised)

    @staticmethod
    def _stepping(monkeypatch: pytest.MonkeyPatch) -> None:
        """Install a clock that moves only when something sleeps on it.

        :meth:`_coarse` quantises *real* time, which leaves a collision a race:
        a pair that straddles a step boundary does not collide, so a case built
        on it can pass against the very defect it exists to catch. That is not
        hypothetical -- it is how the placement defect below survived its first
        fix. Here nothing but ``time.sleep`` moves the clock, so the claim is
        settled by the code rather than by how fast the runner happened to be.

        Sound only because the case using it never touches the event loop:
        asyncio reads ``time.monotonic`` for its own timers, and a clock real
        time cannot move would stall them.

        Args:
            monkeypatch: Reverts both patches.
        """
        now = 1000.0

        def stepped() -> float:
            """Return the current instant.

            Returns:
                The clock, which only :func:`slept` advances.
            """
            return now

        def slept(_seconds: float) -> None:
            """Advance the clock one step instead of waiting.

            Args:
                _seconds: The requested delay, unused; no real time passes.
            """
            nonlocal now
            now += _STEPPING_CLOCK_STEP

        monkeypatch.setattr(time, "monotonic", stepped)
        monkeypatch.setattr(time, "sleep", slept)

    def test_it_waits_until_the_clock_reports_a_new_instant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guarantee the whole fix rests on, asserted directly.

        Args:
            monkeypatch: Installs the coarse clock.
        """
        self._coarse(monkeypatch, _COARSE_CLOCK_STEP)
        before = time.monotonic()

        conformance_module._advance_clock()

        assert time.monotonic() > before

    def test_a_stopped_clock_bounds_the_wait_instead_of_hanging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A clock that never moves is a defect to report, not a reason to hang.

        ``check_arrival_increases`` is the check that catches a constant clock,
        and it can only do so if the driver returns and lets it run.

        The bound is a poll count, so this counts polls. An earlier draft
        asserted elapsed wall time instead and went red on the macOS leg, where
        ``time.sleep(0.001)`` takes about eleven milliseconds: sleep accuracy is
        the platform's business, and the count is the only part the driver
        promises. Counting also turns the regression it guards against -- a
        deadline computed from a clock that has stopped -- from a hung job into
        a failed assertion.

        Args:
            monkeypatch: Freezes the clock outright and counts the polls.
        """
        self._coarse(monkeypatch, 0)
        polls = 0

        def counted(_seconds: float) -> None:
            """Count one poll, failing rather than hanging if the bound is gone.

            Args:
                _seconds: The requested delay, unused; no real time need pass
                    for a clock that is frozen anyway.
            """
            nonlocal polls
            polls += 1
            assert polls <= conformance_module._CLOCK_TICK_POLLS, "the wait is unbounded"

        monkeypatch.setattr(time, "sleep", counted)

        conformance_module._advance_clock()

        assert polls == conformance_module._CLOCK_TICK_POLLS

    def test_the_clock_is_advanced_before_the_request_is_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wait has to precede the exchange, not follow it.

        Waiting *after* the reply stands a probe apart from the driver's own
        previous probe and from nothing else. A probe that follows a request the
        driver did not send still shares that request's instant -- and that is
        how the Windows leg stayed red after the first fix: §6.3's slow-body
        pair hand-rolls its first request on a raw socket, so nothing separated
        it from the driven probe that follows. Measured before the move, at a
        50 ms quantum: the pair collided in four runs out of eight.

        Read at the socket rather than at the recorder, because the connection
        is opened before any arrival can be stamped -- if the clock has already
        moved by then, no arrival of this probe can reuse an earlier instant.

        Args:
            monkeypatch: Installs the stepped clock and the stub connector.
        """
        reached: list[float] = []

        def connector(*_args: object, **_kwargs: object) -> None:
            """Record the instant the driver reached the socket, then stop it.

            Args:
                *_args: The address and timeout, unused.
                **_kwargs: The same, unused.

            Raises:
                _ReachedTheSocket: Always. The probe has nothing left to prove
                    once the connection would have been opened, and there is no
                    server on the other end to answer it.
            """
            reached.append(time.monotonic())
            raise _ReachedTheSocket

        self._stepping(monkeypatch)
        monkeypatch.setattr(conformance_module.socket, "create_connection", connector)
        began = time.monotonic()

        with pytest.raises(_ReachedTheSocket):
            conformance_module.send_raw("h", 1, probe("placement"), marker="placement")

        assert reached[0] > began, f"connection opened at {reached[0]}, call began at {began}"

    async def test_two_probes_are_stamped_at_different_instants(
        self, recorder: RecordingUpstream, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end-to-end claim, on a clock too coarse to give it away.

        Delete the wait and this fails: a probe pair takes a couple of
        milliseconds against a 50 ms quantum, so the two share a bucket unless
        the driver stands them apart. The two cases above are what pin the wait
        deterministically; this is what proves it reaches the recorder's stamps.

        Args:
            recorder: The running recorder.
            monkeypatch: Installs the coarse clock.
        """
        self._coarse(monkeypatch, _COARSE_CLOCK_STEP)

        sent = [await send(recorder.host, recorder.port, probe(m), marker=m) for m in ("first", "second")]

        check_arrival_increases(recording_of(recorder), sent, [s.source_port for s in sent])


class TestTheRecorderPassesEveryConformanceCheck:
    """R3.3 — the primary recorder satisfies the contract Epic B inherits."""

    @pytest.mark.parametrize("check", PER_EXCHANGE_CHECKS, ids=lambda c: c.__name__)
    async def test_per_exchange_check(self, recorder: RecordingUpstream, check) -> None:
        """Assert one capture matches the request that produced it.

        Args:
            recorder: The running recorder.
            check: One of the ten per-exchange checks.
        """
        sent = await send(recorder.host, recorder.port, RICH_PROBE, marker="rich")
        captured = correlate(recording_of(recorder), sent)
        check(captured, sent, scheme=recorder.scheme)

    @pytest.mark.parametrize("check", PER_SESSION_CHECKS, ids=lambda c: c.__name__)
    async def test_per_session_check(self, recorder: RecordingUpstream, check) -> None:
        """Assert the recording as a whole is faithful.

        Args:
            recorder: The running recorder.
            check: One of the four per-session checks.
        """
        sent = [
            await send(recorder.host, recorder.port, probe(marker), marker=marker)
            for marker in ("first", "second")
        ]
        check(recording_of(recorder), sent, [s.source_port for s in sent])


class TestCaptureFidelity:
    """R1 — the specific things §4.3 C1, §3.3.5 and §5.2.1 read."""

    async def test_it_records_the_percent_encoded_path_undecoded(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the path keeps its escapes, and their casing.

        ``request.path`` decodes and would render this ``/v1/messages/a/b/c``.
        §3.3.5 asserts on routing, and on Azure the deployment id in the path is
        the only thing distinguishing two byte-identical bodies.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, RICH_PROBE, marker="rich")
        assert recorder.requests[0].path == "/v1/mess%61ges/a%2Fb%2fc/chat/completions"

    async def test_it_records_the_raw_query_unreordered(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the query survives undecoded and in the order it was sent.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, RICH_PROBE, marker="rich")
        assert recorder.requests[0].query == "Beta=A%20B&key=k&beta=c"

    async def test_it_records_an_absent_query_as_empty(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert no query is recorded as the empty string, not ``None``.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, probe("q"), marker="q")
        assert recorder.requests[0].query == ""

    async def test_it_records_headers_with_casing_order_and_duplicates(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the exact header evidence §4.3 C1 asserts on survives.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, RICH_PROBE, marker="rich")
        headers = list(recorder.requests[0].headers)

        assert ("ANTHROPIC-BETA", "one") in headers
        assert ("anthropic-beta", "two") in headers
        assert [n for n, _ in headers][:4] == [
            "Host", "X-Api-Key", "anthropic-version", "ANTHROPIC-BETA",
        ]

    async def test_it_decodes_obs_text_header_values_rather_than_failing(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert a `latin-1`-only header value survives.

        ``utf-8`` refuses ``caf\\xe9``, so a recorder decoding with it would lose
        the request entirely rather than record it.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, RICH_PROBE, marker="rich")
        assert ("X-Weird", "café") in list(recorder.requests[0].headers)

    async def test_it_records_the_host_header_verbatim(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the authority keeps its casing and its port.

        ``request.url`` lower-cases the host.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, RICH_PROBE, marker="rich")
        assert recorder.requests[0].host == "Upstream.Example:443"

    async def test_it_records_no_host_rather_than_inventing_one(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert an absent ``Host`` is recorded as absent.

        This is the defect aiohttp's ``request.host`` produces unaided: with no
        ``Host`` header it returns ``socket.getfqdn()`` — the build machine's own
        name — and which value that is differs across the aiohttp versions
        ``pyproject.toml`` allows.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, NO_HOST_PROBE, marker="nohost")
        assert recorder.requests[0].host == ""

    async def test_it_records_the_peer_port_not_its_own(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the recorded port is the client's, §5.2.1's join key.

        Args:
            recorder: The running recorder.
        """
        sent = await send(recorder.host, recorder.port, probe("p"), marker="p")
        assert recorder.requests[0].peer_port == sent.source_port
        assert recorder.requests[0].peer_port != recorder.port

    async def test_it_captures_a_body_above_aiohttps_default_ceiling(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert a 2 MiB body is captured whole.

        aiohttp caps request bodies at 1 MiB by default, above which it raises
        **413** — which is M6's own trigger, so the harness would manufacture the
        compaction the fidelity oracle exists to detect. §7.1's corpus is
        specified to contain transcripts over the compaction budget.

        Args:
            recorder: The running recorder.
        """
        body = b"x" * (2 * 1024 * 1024)
        await send(recorder.host, recorder.port, probe("big", body=body), marker="big")
        assert recorder.requests[0].body == body

    async def test_it_captures_a_chunked_body_dechunked(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert transfer-coding is undone, which is the level a projection reads at.

        Args:
            recorder: The running recorder.
        """
        raw = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\nX-Probe-Marker: ch\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n"
        )
        await send(recorder.host, recorder.port, raw, marker="ch")
        assert recorder.requests[0].body == b"abcde"

    async def test_it_captures_a_gzip_body_still_compressed(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert content-coding is *not* undone.

        aiohttp decompresses by default, which would leave the capture carrying a
        plaintext body beside a ``Content-Encoding: gzip`` header claiming
        otherwise — an internally inconsistent capture, read by the header
        assertion in §4.3 C1.

        Args:
            recorder: The running recorder.
        """
        payload = gzip.compress(b"hello-compressed")
        raw = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\nX-Probe-Marker: gz\r\n"
            b"Content-Encoding: gzip\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        await send(recorder.host, recorder.port, raw, marker="gz")

        captured = recorder.requests[0]
        assert captured.body == payload
        assert ("Content-Encoding", "gzip") in list(captured.headers)

    async def test_arrival_increases_across_requests(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert arrival moves forward from one request to the next.

        This is the weaker half of R1.9. The other half — that the stamp is taken
        at handler entry rather than after the body is read — cannot be shown by
        sequential requests, because they are correctly ordered whichever point
        the stamp is taken at. ``test_recorder_falsification.py``'s slow-body
        probe is what proves it, by making the two orders disagree.

        Args:
            recorder: The running recorder.
        """
        await send(recorder.host, recorder.port, probe("a1"), marker="a1")
        await send(recorder.host, recorder.port, probe("a2"), marker="a2")

        first, second = recorder.requests
        assert first.arrival is not None and second.arrival is not None

        assert second.arrival > first.arrival

    async def test_a_request_that_never_completes_is_not_published(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert a client that disconnects mid-body leaves no phantom capture.

        The slot is reserved at handler entry so that a short request cannot
        overtake a long one; if the body read then fails, that slot must not
        reach :attr:`RecordingUpstream.requests`. §6.3.1 injects a client
        disconnect at four points, so T-B4 and T-W9 meet this immediately — and
        an unfilled slot would surface to them as a capture nobody sent, naming
        the wrong problem entirely.

        Args:
            recorder: The running recorder.
        """
        # Announce a body, send one byte of it, then vanish.
        truncated = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\n"
            b"X-Probe-Marker: gone\r\nContent-Length: 50\r\n\r\nx"
        )
        sock = socket.create_connection((recorder.host, recorder.port))
        aborted_port = sock.getsockname()[1]
        try:
            await asyncio.to_thread(sock.sendall, truncated)
            await _until(lambda: bool(recorder.connections), what="the connection to be accepted")
        finally:
            sock.close()

        sent = await send(recorder.host, recorder.port, probe("real"), marker="real")

        assert [c.path for c in recorder.requests] == ["/v1/chat/completions"]
        assert all(c.method for c in recorder.requests), (
            f"a slot for an incomplete request was published: {recorder.requests}"
        )

        # The connection log has to agree with the request list, not merely be
        # populated. Filtering the unfilled slot out of `requests` while still
        # counting it on its connection would leave the two halves of R1.10
        # contradicting each other — and `check_connection_logged` is the thing
        # that would notice, so it is what this asserts through.
        check_connection_logged(recording_of(recorder), [sent], [aborted_port, sent.source_port])


class TestTheConnectionLog:
    """R1.10 — the evidence §5.2.1 needs and the request list cannot give."""

    async def test_it_logs_a_connection_that_sends_no_request(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert a connection carrying nothing is still recorded.

        §5.2.1: *"An upstream connection with no matching tunnel port is a
        bypass, and it is the only thing this assertion needs to catch."* Such a
        connection is invisible in the request list by construction, so a
        recorder without this log cannot support the containment claim at all.

        Args:
            recorder: The running recorder.
        """
        sock = socket.create_connection((recorder.host, recorder.port))
        source_port = sock.getsockname()[1]
        try:
            await _until(lambda: bool(recorder.connections), what="the connection to be accepted")
        finally:
            sock.close()

        assert recorder.requests == []
        assert source_port in {c.peer_port for c in recorder.connections}
        assert all(c.requests == 0 for c in recorder.connections)

    async def test_one_keep_alive_connection_is_one_record_carrying_both_requests(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert requests sharing a connection share its record.

        This is why §4.3 C5's "count distinct TCP connections" cannot be a
        de-duplication of peer ports, and why the log is keyed on the connection
        rather than on the port.

        Args:
            recorder: The running recorder.
        """
        sock = socket.create_connection((recorder.host, recorder.port))
        source_port = sock.getsockname()[1]
        try:
            for expected, marker in enumerate(("k1", "k2"), start=1):
                sock.sendall(probe(marker))
                await _until(
                    lambda n=expected: len(recorder.requests) >= n,
                    what=f"request {expected} on the keep-alive connection",
                )
        finally:
            sock.close()

        assert len(recorder.requests) == 2
        assert {r.peer_port for r in recorder.requests} == {source_port}

        carrying = [c for c in recorder.connections if c.peer_port == source_port]
        assert len(carrying) == 1, f"expected one connection record, got {carrying}"
        assert carrying[0].requests == 2

    async def test_concurrent_connections_are_logged_separately(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert two connections open at once each get their own record.

        Args:
            recorder: The running recorder.
        """
        socks = []
        try:
            for marker in ("c1", "c2"):
                sock = socket.create_connection((recorder.host, recorder.port))
                sock.sendall(probe(marker))
                socks.append(sock)
            await _until(lambda: len(recorder.requests) >= 2, what="both concurrent requests")
            ports = {s.getsockname()[1] for s in socks}
        finally:
            for sock in socks:
                sock.close()

        assert ports <= {c.peer_port for c in recorder.connections}

        # Each request is attributed to its own connection, not both to one.
        # A recorder that looked the connection up by peer *port* would still
        # pass the line above; this is what distinguishes it.
        by_port = {c.peer_port: c.requests for c in recorder.connections}
        assert [by_port[port] for port in ports] == [1, 1], (
            f"each concurrent connection should carry one request; got {by_port}"
        )


class TestMinimalSuccessResponses:
    """R2 — replies the bridge accepts in one attempt."""

    @pytest.mark.parametrize(
        "fmt", [WireFormat.ANTHROPIC_MESSAGES, WireFormat.CHAT_COMPLETIONS],
        ids=lambda f: f.value,
    )
    def test_the_non_streaming_success_is_not_empty_by_the_bridges_own_judgement(
        self, fmt: WireFormat
    ) -> None:
        """Assert the bridge would not treat the reply as an empty response.

        An empty response costs 80 seconds of retry ladder
        (``_EMPTY_RETRY_DELAYS`` + ``_EMPTY_FINAL_DELAYS``) and, on a balancing
        profile, fails over — and failover *is* body-mutating (§4.3 C3), so the
        harness would manufacture a delta no product code caused.

        This is the one place this suite imports from ``src/kitty``, and
        deliberately: the bridge's own judgement is the only authority on the
        question. The independence rule protects the *oracle*, not a test whose
        whole claim is "the real bridge accepts this".

        Args:
            fmt: The wire format under test.
        """
        from kitty.bridge.server import BridgeServer

        assert BridgeServer._is_empty_cc_response(minimal_success_body(fmt)) is False

    def test_the_emptiness_judgement_still_rejects_a_hollow_body(self) -> None:
        """The negative control for the check above.

        Without it, a judgement that had stopped discriminating would report
        every reply as non-empty and the assertion would pass forever.
        """
        from kitty.bridge.server import BridgeServer

        hollow = {"choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                               "finish_reason": "stop"}]}
        assert BridgeServer._is_empty_cc_response(hollow) is True

    def test_the_chat_completions_stream_is_not_empty_by_the_translators_judgement(self) -> None:
        """Assert the streaming reply survives the streaming emptiness oracle.

        The non-streaming check above does not cover this: a streaming reply is
        judged by ``translator.response_was_empty`` instead
        (``server.py`` 2983/3831/4487).
        """
        from kitty.bridge.messages.translator import MessagesTranslator

        translator = MessagesTranslator()
        for chunk in _sse_payloads(minimal_success_stream(WireFormat.CHAT_COMPLETIONS)):
            translator.translate_stream_chunk("chatcmpl-recorder", "recorder-model", chunk)

        assert translator.response_was_empty is False

    def test_the_streaming_judgement_still_rejects_a_contentless_stream(self) -> None:
        """The negative control for the check above."""
        from kitty.bridge.messages.translator import MessagesTranslator

        translator = MessagesTranslator()
        translator.translate_stream_chunk(
            "chatcmpl-x", "m",
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        )
        assert translator.response_was_empty is True

    def test_the_anthropic_stream_is_a_sentence_in_the_sse_grammar(self) -> None:
        """Assert the Anthropic stream is well-formed, because nothing else checks it.

        Verified at ``server.py:3616``: a native Messages stream is forwarded to
        the client byte-for-byte and never reaches a translator, so **no**
        bridge-side emptiness judgement sees it. §6.2.2's grammar is the only
        guard on this path, which is why it is asserted here rather than assumed.
        """
        events = [
            payload["type"]
            for payload in _sse_payloads(minimal_success_stream(WireFormat.ANTHROPIC_MESSAGES))
        ]
        assert events == [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ]

        deltas = [
            p for p in _sse_payloads(minimal_success_stream(WireFormat.ANTHROPIC_MESSAGES))
            if p["type"] == "content_block_delta"
        ]
        assert deltas and deltas[0]["delta"]["text"], "the stream must carry actual text"

    async def test_it_streams_when_the_request_asks_to_stream(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert ``"stream": true`` is answered with SSE, not JSON.

        A streaming request answered with a JSON body is not a success; it lands
        in the same retry ladder an empty reply would.

        Args:
            recorder: The running recorder.
        """
        reply = await _exchange(recorder, probe("s", body=b'{"stream": true}'))
        assert b"text/event-stream" in reply
        assert b"data: " in reply

    async def test_it_returns_json_when_the_request_does_not_stream(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert a non-streaming request gets a JSON body.

        Args:
            recorder: The running recorder.
        """
        reply = await _exchange(recorder, probe("n", body=b'{"stream": false}'))
        assert b"application/json" in reply


class TestFormatDispatch:
    """R2.2 — the reply's format follows the URL, as a real provider's does."""

    @pytest.mark.parametrize(
        ("path", "expected_marker"),
        [
            ("/v1/messages", b'"type": "message"'),
            ("/api/anthropic/v1/messages", b'"type": "message"'),
            ("/v1/chat/completions", b'"object": "chat.completion"'),
            ("/openai/deployments/d1/chat/completions", b'"object": "chat.completion"'),
            ("/endpoints/openapi/chat/completions", b'"object": "chat.completion"'),
        ],
    )
    async def test_real_adapter_paths_select_the_right_format(
        self, recorder: RecordingUpstream, path: str, expected_marker: bytes
    ) -> None:
        """Assert every real upstream path shape dispatches correctly.

        These are the paths the shipped adapters actually post to. An exact
        match on ``/chat/completions`` would have selected correctly for three
        adapters and wrongly for the rest — Azure bakes the deployment into the
        path, vertex prefixes ``/endpoints/openapi``, and zai_anthropic serves
        Messages under ``/api/anthropic``.

        Args:
            recorder: The running recorder.
            path: The upstream path.
            expected_marker: A byte string only the right format's body carries.
        """
        reply = await _exchange(recorder, probe("d", path=path))
        assert expected_marker in reply
        recorder.assert_all_paths_matched()

    async def test_an_unmatched_path_is_recorded_and_fails_at_teardown(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the fallback is loud rather than silent.

        A wrong-format reply does not fail loudly on its own: the adapter parses
        nothing out of it, the bridge judges the response empty, and the test
        pays the retry ladder. So the fallback is recorded and turned into an
        error at teardown.

        Args:
            recorder: The running recorder.
        """
        await _exchange(recorder, probe("u", path="/not/an/api"))

        assert recorder.unmatched == ["/not/an/api"]
        with pytest.raises(UnmatchedPathError, match="fallback format"):
            recorder.assert_all_paths_matched()

        # Opt out of the fixture's own teardown guard: this test *meant* to take
        # the fallback, and the guard would otherwise fail it for succeeding.
        recorder.unmatched.clear()

    async def test_the_pure_lookup_agrees_with_the_recording_one_and_records_nothing(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert ``format_for_path`` answers the same question without the side effect.

        T-W8's bridge fixture judges its captured paths against the format its
        transport declares, at teardown, on every fixture exit. It must not do
        that with :meth:`RecordingUpstream._format_for`, which *appends to*
        ``unmatched`` on a miss — an assertion that mutates the evidence it is
        judging. Both halves are asserted here because a lookup that silently
        stopped agreeing with the recording one would be a second source of
        truth, which is the thing splitting it was meant to avoid.

        Args:
            recorder: The running recorder.
        """
        assert format_for_path("/v1/messages") is WireFormat.ANTHROPIC_MESSAGES
        assert format_for_path("/openai/deployments/d/chat/completions") is WireFormat.CHAT_COMPLETIONS

        # `None` for a miss, not the fallback: "no rule applies" and "the
        # fallback applies" are different facts, and only a caller knows which.
        assert format_for_path("/not/an/api") is None
        assert recorder.unmatched == []

    async def test_a_malformed_body_still_gets_a_success(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert an unparseable body is answered, not rejected.

        T-C6 contributes a malformed body to the corpus. A harness 5xx there
        reads to the bridge as an upstream error and triggers failover, which
        *is* body-mutating — the harness would fabricate the delta the oracle
        exists to find.

        Args:
            recorder: The running recorder.
        """
        reply = await _exchange(recorder, probe("m", body=b"{not json at all"))
        assert reply.startswith(b"HTTP/1.1 200"), reply[:40]

    def test_it_refuses_a_default_format_it_does_not_serve(self) -> None:
        """Assert construction fails for a format §7.2 assigns elsewhere.

        Failing here beats replying in a format no adapter asked for. Bedrock
        Converse, OpenAI Responses and Ollama ``/api/chat`` belong to T-B1–T-B3.
        """
        with pytest.raises(ValueError, match="primary recorder serves"):
            RecordingUpstream(default_format=WireFormat.BEDROCK_CONVERSE)


class TestTheResponderSeam:
    """R2.4 — what T-B4 builds its failure library on."""

    async def test_a_custom_responder_owns_status_and_headers(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert the hook can produce an error status, not only a body.

        §7.2 requires every recorder to replay "error statuses, Cloudflare
        blocks, empty responses, context-too-large rejections". A hook that
        could only supply a body could express none of them.

        Args:
            recorder: The running recorder.
        """

        async def blocked(captured: CapturedRequest, response: Reply) -> None:
            """Reply as a Cloudflare block would.

            Args:
                captured: The recorded request.
                response: The response to write.
            """
            response.content_length = 7
            await response.begin(403, {"Server": "cloudflare"})
            await response.write(b"blocked")
            await response.write_eof()

        recorder.responder = blocked
        reply = await _exchange(recorder, probe("b"))

        assert reply.startswith(b"HTTP/1.1 403")
        assert b"cloudflare" in reply
        # The request is still recorded: what the recorder replies has no
        # bearing on what it observed.
        assert len(recorder.requests) == 1

    async def test_a_responder_can_abort_mid_stream(
        self, recorder: RecordingUpstream
    ) -> None:
        """Assert a disconnect part-way through a stream is expressible.

        §6.3.1 injects a failure at four points, one of which is *after content
        has been emitted*. A hook returning a finished response could not
        express it, and T-B4 would have to edit the recorder — which is the
        thing this seam exists to prevent.

        Args:
            recorder: The running recorder.
        """

        async def truncate(captured: CapturedRequest, response: Reply) -> None:
            """Write two events, then drop the connection.

            Args:
                captured: The recorded request.
                response: The response to write.
            """
            await response.begin(200, {"Content-Type": "text/event-stream"})
            for chunk in minimal_success_stream(WireFormat.CHAT_COMPLETIONS)[:2]:
                await response.write(chunk)
            await response.abort()

        recorder.responder = truncate
        reply = await _exchange(recorder, probe("t", body=b'{"stream": true}'))

        # KBR-188/189: exempt the WINDOWS CELL only -- this assertion gates
        # normally on the Linux and macOS legs, and fails the job the day
        # Windows starts passing. §8.3's parametrised-cell shape.
        exempt = sys.platform == "win32"
        with ratchet("recorder-responder-abort-emits-before-dropping") if exempt else nullcontext():
            assert b"data: " in reply

        assert b"[DONE]" not in reply, "the stream must have been cut before its terminator"


class TestTheRecorderStaysIndependentOfKitty:
    """D6 — the rule `contract.py` enforces, extended to this module.

    Asserting the property in prose only is what ``contract.py``'s own docstring
    calls a defect: *"a rule that reads like a guarantee and guarantees
    nothing"*.
    """

    @pytest.mark.parametrize(
        "module", [recorder_module, conformance_module], ids=lambda m: m.__name__
    )
    def test_it_imports_nothing_from_kitty(self, module) -> None:
        """Assert a support module is not written in terms of the code under test.

        A recorder that asked kitty how to read a request would inherit kitty's
        bugs, and the fidelity claim would reduce to self-consistency.

        **Both** support modules are guarded, not just the recorder. The
        conformance module happens to import nothing from kitty today, and
        nothing would have noticed if that changed: none of its checks touch a
        translator, so adding one would pass every test here while quietly
        taking on the dependency the recorder forbids. The two *test* modules
        are deliberately not guarded — ``test_recorder.py`` imports kitty on
        purpose, to ask the bridge's own judgement whether a reply reads as
        empty.

        Args:
            module: The support module whose source is read.
        """
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

        offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]
        assert offending == [], f"{module.__name__} must not import kitty: {offending}"


async def _until(predicate: Callable[[], bool], *, what: str) -> None:
    """Yield to the event loop until ``predicate`` holds.

    Waits on a **condition**, never on a fixed span: a `sleep()`-based wait is
    the leading cause of flaky tests, and it is slow by construction because the
    span has to be generous enough for the slowest machine.

    Args:
        predicate: The condition to wait for.
        what: What is being waited for, for the failure message.

    Raises:
        AssertionError: When the condition has not held within the budget. This
            is a real failure, not a slow machine: the work is local and
            in-process.
    """
    for _ in range(_SETTLE_ATTEMPTS):
        if predicate():
            return
        await asyncio.sleep(_SETTLE_STEP)
    raise AssertionError(f"timed out waiting for {what}")


async def _exchange(recorder: RecordingUpstream, raw: bytes) -> bytes:
    """Send ``raw`` and return the reply bytes.

    Stops at the end of the response rather than at end-of-connection. aiohttp
    keeps a connection alive for ``keepalive_timeout`` (3630 seconds by
    default), so reading to EOF would stall every one of these tests until the
    socket timeout expired — minutes across the module, in the gating layer.

    Args:
        recorder: The running recorder.
        raw: The request bytes.

    Returns:
        The reply: headers, and as much body as arrives with them.
    """
    return await asyncio.to_thread(_exchange_blocking, recorder.host, recorder.port, raw)


def _exchange_blocking(host: str, port: int, raw: bytes) -> bytes:
    """Do the blocking half of :func:`_exchange`.

    Args:
        host: The recorder's bind address.
        port: The recorder's listening port.
        raw: The request bytes.

    Returns:
        The reply bytes.
    """
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


def _reply_is_complete(reply: bytes) -> bool:
    """Return whether a full response has arrived.

    Args:
        reply: What has been read so far.

    Returns:
        True once the headers are in and the body they describe is complete —
        by ``Content-Length`` where one is given, by the terminating zero-length
        chunk for a chunked body, and by the headers alone otherwise.
    """
    head, separator, body = reply.partition(b"\r\n\r\n")
    if not separator:
        return False

    lowered = head.lower()
    for line in lowered.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip() == b"content-length":
            return len(body) >= int(value.strip())

    if b"transfer-encoding: chunked" in lowered:
        return body.endswith(b"0\r\n\r\n")

    # No length and no chunking: the body runs to end-of-connection, so the
    # headers are all that can be waited for deterministically.
    return True


def _sse_payloads(chunks: tuple[bytes, ...]) -> list[dict]:
    """Return the JSON payloads of an SSE stream's ``data:`` lines.

    Args:
        chunks: The stream's chunks.

    Returns:
        Each ``data:`` payload parsed, skipping the ``[DONE]`` sentinel.
    """
    payloads = []
    for line in b"".join(chunks).decode().splitlines():
        if not line.startswith("data: "):
            continue
        body = line[len("data: "):].strip()
        if body and body != "[DONE]":
            payloads.append(json.loads(body))
    return payloads


class TestTheFixtureGuardIsActuallyWired:
    """A guard the fixture does not call is a guard that protects nothing.

    :meth:`RecordingUpstream.assert_all_paths_matched` has its own behavioural
    test above, but that proves only that the method works — not that anything
    runs it. A review pass found the method implemented and never called, so
    R2.2's "fails the fixture at teardown" was satisfied by two hand-written
    call sites and by nothing structural.

    Reading the fixture's source is the same technique
    ``tests/harness/test_contract.py`` and ``tests/test_egress_coverage.py``
    already use for guards whose *wiring* is the thing at risk.
    """

    def test_the_recorder_fixture_calls_the_unmatched_path_guard(self) -> None:
        """Assert the fixture runs the fallback guard on teardown.

        Without this, deleting one line from the fixture would silently return
        the suite to the state the review found: every test free to be answered
        in the wrong format, and the resulting empty-looking response paying the
        80-second retry ladder rather than failing.
        """
        source = Path(__file__).read_text(encoding="utf-8")
        fixture = source.split("async def recorder()", 1)[1].split("\nclass ", 1)[0]

        assert "assert_all_paths_matched()" in fixture, (
            "the recorder fixture must call assert_all_paths_matched() on teardown; "
            "R2.2's loud fallback is otherwise enforced by nothing"
        )

    def test_this_guard_reads_the_fixture_and_not_the_whole_file(self) -> None:
        """The positive control: the slice must actually isolate the fixture.

        A split that silently returned the entire file would find the call
        somewhere else and pass forever.
        """
        source = Path(__file__).read_text(encoding="utf-8")
        fixture = source.split("async def recorder()", 1)[1].split("\nclass ", 1)[0]

        assert len(fixture) < len(source) / 4, "the slice is not isolating the fixture"
        assert "class TestCaptureFidelity" not in fixture
