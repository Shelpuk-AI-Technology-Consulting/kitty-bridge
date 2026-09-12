"""Deliberately broken recorders the conformance suite must reject.

`.system_design/TEST_SUITE.md` §7.2 · plan **§1.4** (KBR-27, T-W4).

> *"The first working version of every harness ships with at least one
> falsification case — a deliberate defect it must detect, running in the
> suite."*

The rule exists because four design-review rounds produced four harnesses that
would have passed while proving nothing. A recording upstream is the harness the
whole of I1 reads its evidence from, so a check of it that is wired to nothing
would not be noticed by anything else.

Each recorder below breaks **one** thing, and each is asserted twice:

1. its named check rejects it, and
2. **no other check fails on the same exchange.**

The second half is what catches an over-broad check. Without it, a check that
failed on everything would satisfy every case here and the matrix would be
worthless — which is exactly the shape of defect §1.4 was written about.

Each defect also names the **probe** that exposes it. A defect is only
observable against an input that distinguishes it: a recorder reading
``request.host`` is byte-for-byte correct on any request that carries a ``Host``
header, and one that re-encodes the body through ``str`` is correct on any body
that happens to be valid UTF-8.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
from aiohttp import web

from harness.contract import CapturedRequest, WireFormat
from harness.recorder import ConnectionRecord, RecordingUpstream, _ConnectionLoggingServer
from harness.recorder_conformance import (
    LATIN_PROBE,
    MARKER_HEADER,
    NO_HOST_PROBE,
    PER_EXCHANGE_CHECKS,
    PER_SESSION_CHECKS,
    RICH_PROBE,
    Recording,
    SentRequest,
    _parse_sent,
    check_connection_logged,
    correlate,
    marker_of,
    open_connection_only,
    probe,
    recording_of,
    send,
)

# --------------------------------------------------------------------------
# The defective recorders
# --------------------------------------------------------------------------


class LowerCasingRecorder(RecordingUpstream):
    """Normalises header names, the way any mapping-backed capture would."""

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with every header name lower-cased.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        captured = super().capture(request, body, arrival)
        return _replace_headers(captured, [(n.lower(), v) for n, v in captured.headers])


class SortingRecorder(RecordingUpstream):
    """Emits headers in sorted order, losing the order they arrived in."""

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with its headers sorted.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        captured = super().capture(request, body, arrival)
        return _replace_headers(captured, sorted(captured.headers))


class DeduplicatingRecorder(RecordingUpstream):
    """Keeps one entry per header name, as a ``dict`` would."""

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with repeated header names collapsed.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        captured = super().capture(request, body, arrival)
        seen: set[str] = set()
        kept = []
        for name, value in captured.headers:
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            kept.append((name, value))
        return _replace_headers(captured, kept)


class OwnPortRecorder(RecordingUpstream):
    """Records its own listening port instead of the peer's.

    The failure this produces downstream is not a broken join against §5.2.1's
    tunnel log — it is a join that *succeeds*, against the wrong connection.
    """

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with the recorder's own listening port.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return _replace(super().capture(request, body, arrival), peer_port=self.port)


class DecodedQueryRecorder(RecordingUpstream):
    """Uses ``request.query_string``, which is percent-decoded."""

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with a percent-decoded query.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return _replace(super().capture(request, body, arrival), query=request.query_string)


class DroppedQueryRecorder(RecordingUpstream):
    """Keeps no query at all.

    On Azure the query carries the API version while the path carries the
    deployment, and P6 removes ``model`` from the body — so a lost query can
    make two different routes indistinguishable.
    """

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with no query at all.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return _replace(super().capture(request, body, arrival), query="")


class DecodedPathRecorder(RecordingUpstream):
    """Uses ``request.path``, which is percent-decoded."""

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with a percent-decoded path.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return _replace(super().capture(request, body, arrival), path=request.path)


class HostAttributeRecorder(RecordingUpstream):
    """Uses ``request.host``, which invents a value when none was sent.

    aiohttp returns ``socket.getfqdn()`` — the machine's own name — for a
    request with no ``Host`` header. Only the no-``Host`` probe can see it.
    """

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with aiohttp's own ``request.host``.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return _replace(super().capture(request, body, arrival), host=request.host)


class FrozenClockRecorder(RecordingUpstream):
    """Stamps the same arrival time on everything."""

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with a fixed arrival timestamp.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        return _replace(super().capture(request, body, arrival), arrival=0.0)


class LateClockRecorder(RecordingUpstream):
    """Stamps arrival when the capture is built, not when the request arrived.

    The difference only shows when a body takes time to arrive — which is the
    normal case for the large transcripts §7.1's corpus is made of.
    """

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with arrival stamped now, not at entry.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        # Must be the *same* clock the recorder stamps with (KBR-208).
        # `monotonic()` and `perf_counter()` have unrelated reference points, so
        # mixing them would make this falsification pass or fail by accident
        # rather than because the timestamp was taken late.
        return _replace(super().capture(request, body, arrival), arrival=time.perf_counter())


class TextRoundTripRecorder(RecordingUpstream):
    """Records ``request.text()`` re-encoded, losing the original bytes.

    Invisible on valid UTF-8, which is why its probe declares ``charset=latin-1``
    and sends high bytes: aiohttp then decodes latin-1 and the re-encode to
    UTF-8 returns different bytes.
    """

    def capture(
        self, request: web.BaseRequest, body: bytes, arrival: float
    ) -> CapturedRequest:
        """Return the capture with the body decoded and re-encoded.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The deliberately wrong capture.
        """
        captured = super().capture(request, body, arrival)
        charset = request.charset or "utf-8"
        return _replace(captured, body=body.decode(charset).encode("utf-8"))


class ReorderingRecorder(RecordingUpstream):
    """Publishes captures in a stable order of its own, not arrival order.

    Sorted by the probe marker, descending — not by path, which every request in
    an ordinary probe shares, and which would therefore make this "defect" a
    no-op that proved nothing.

    **The defect is in the published view, not in the slots.** Sorting
    ``_slots`` in place would move entries out from under the indices reserved
    for requests still in flight, so a third concurrent request would overwrite
    someone else's row — a second, accidental defect, and the falsification
    matrix could no longer say which one a failure came from. A recorder
    subclass is a template as much as a test; the shape it teaches has to be the
    safe one.
    """

    @property
    def requests(self) -> list[CapturedRequest]:
        """Return the captures sorted by marker instead of by arrival.

        Returns:
            The completed captures, in the wrong order.
        """
        return sorted(super().requests, key=lambda c: marker_of(c) or "", reverse=True)


class DroppingRecorder(RecordingUpstream):
    """Silently loses every second request.

    Nothing but the coverage check catches this, and §4.3 C6's *"``/healthz``
    never causes an upstream request"* rests on a recorder that cannot lose one.

    Like :class:`ReorderingRecorder`, the loss is in the published view. Popping
    from ``_slots`` would shift every higher index down by one and corrupt the
    rows of requests still in flight.
    """

    @property
    def requests(self) -> list[CapturedRequest]:
        """Return every other capture.

        Returns:
            The completed captures with the odd-numbered ones missing.
        """
        return [c for i, c in enumerate(super().requests) if i % 2 == 0]


class MiscountingConnectionRecorder(RecordingUpstream):
    """Logs every connection, but never records what rode on it.

    The other half of R1.10, and the half the matrix was missing: a log that
    lists the right connections but cannot attribute a capture to one of them
    still cannot support §4.3 C5's distinct-connection count. It passes the
    "every opened port is logged" half of :func:`check_connection_logged`
    outright, so only the carried-count assertion rejects it.
    """

    def count_on_connection(self, connection_id: int) -> None:
        """Do not attribute the request to its connection.

        Args:
            connection_id: The connection's id, ignored.
        """


class LazyConnectionServer(_ConnectionLoggingServer):
    """Registers the connection with aiohttp, but logs nothing at accept time."""

    def connection_made(self, handler: Any, transport: Any) -> None:
        """Accept the connection without recording it.

        `web.Server.connection_made` is still called, so aiohttp keeps working:
        the defect is the missing evidence, not a broken server.

        Args:
            handler: The ``RequestHandler`` for this connection.
            transport: The connection's transport.
        """
        web.Server.connection_made(self, handler, transport)


class LazyConnectionRecorder(RecordingUpstream):
    """Logs a connection only when it produces a request.

    This is R4.1's row 13, and it is the **plausible** version of the defect: a
    recorder that logs connections from inside the request handler looks
    completely correct for every request-bearing connection, and is wrong only
    for the one shape §5.2.1 cares about — a connection that carries nothing.

    An earlier version of this class logged *no* connections at all, which any
    ordinary request would catch. That made the row untestable for its own
    defect and is exactly the §1.4 trap.
    """

    _server_class = LazyConnectionServer

    def capture(self, request, body, arrival):
        """Log the connection now, on the way to recording the request.

        Args:
            request: The inbound aiohttp request.
            body: The entity body, already read.
            arrival: The timestamp taken at handler entry.

        Returns:
            The capture, unmodified.
        """
        peer = request.transport.get_extra_info("peername") if request.transport else None
        self.connections.append(
            ConnectionRecord(len(self.connections), peer[1] if peer else -1, 1)
        )
        return super().capture(request, body, arrival)


def _replace(captured: CapturedRequest, **changes: object) -> CapturedRequest:
    """Return a copy of ``captured`` with ``changes`` applied.

    Args:
        captured: The capture to copy.
        **changes: Fields to override.

    Returns:
        The modified capture.
    """
    return replace(captured, **changes)  # type: ignore[arg-type]


def _replace_headers(
    captured: CapturedRequest, headers: Sequence[tuple[str, str]]
) -> CapturedRequest:
    """Return a copy of ``captured`` carrying ``headers``.

    Args:
        captured: The capture to copy.
        headers: The replacement header pairs.

    Returns:
        The modified capture.
    """
    return _replace(captured, headers=tuple(headers))


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


_LATIN_BODY = b'{"n":"caf\xe9"}'
_LATIN = (
    b"POST /v1/chat/completions HTTP/1.1\r\n"
    b"Host: h\r\n"
    b"X-Probe-Marker: latin\r\n"
    b"Content-Type: application/json; charset=latin-1\r\n"
    b"Content-Length: " + str(len(_LATIN_BODY)).encode() + b"\r\n"
    b"\r\n" + _LATIN_BODY
)


async def _run_single(cls: type[RecordingUpstream], raw: bytes, marker: str):
    """Drive one request at a recorder and return the capture and what was sent.

    Args:
        cls: The recorder class to instantiate.
        raw: The probe request bytes.
        marker: The probe's correlation marker.

    Returns:
        The recorder, the capture and the :class:`SentRequest`.
    """
    upstream = cls(default_format=WireFormat.CHAT_COMPLETIONS)
    await upstream.start()
    try:
        sent = await send(upstream.host, upstream.port, raw, marker=marker)
        captured = correlate(recording_of(upstream), sent)
        return upstream, captured, sent
    finally:
        await upstream.stop()


def _assert_only(
    failing, captured: CapturedRequest, sent: SentRequest, scheme: str, expected: str
) -> None:
    """Assert ``failing`` rejects the capture and every other check accepts it.

    Args:
        failing: The check that must reject.
        captured: The defective capture.
        sent: The request that produced it.
        scheme: The recorder's declared scheme.
        expected: Text the rejection message must contain. Without it, a check
            that blew up on some unrelated condition would satisfy R4.2's
            "rejected by its named check" — failing, but not for its own reason.

    Raises:
        AssertionError: When the named check passes, rejects for the wrong
            reason, or when any other check fails on the same exchange.
    """
    with pytest.raises(AssertionError, match=expected):
        failing(captured, sent, scheme=scheme)

    for check in PER_EXCHANGE_CHECKS:
        if check is failing:
            continue
        try:
            check(captured, sent, scheme=scheme)
        except AssertionError as exc:  # pragma: no cover - only on a real defect
            raise AssertionError(
                f"{failing.__name__} was the named check, but {check.__name__} also "
                f"failed on the same exchange: {exc}. A defect must be attributable "
                "to one check, or the falsification matrix cannot say what it caught."
            ) from exc


class TestCaptureDefectsAreCaughtByExactlyTheirOwnCheck:
    """R4.1 / R4.2 for the ten per-exchange checks."""

    @pytest.mark.parametrize(
        ("cls", "check_name", "raw", "marker", "expected"),
        [
            (LowerCasingRecorder, "check_header_casing", RICH_PROBE, "rich", "header casing"),
            (SortingRecorder, "check_header_order", RICH_PROBE, "rich", "header order"),
            (DeduplicatingRecorder, "check_header_duplicates", RICH_PROBE, "rich",
             "header values per name"),
            (OwnPortRecorder, "check_peer_port", RICH_PROBE, "rich", "peer_port"),
            (DecodedQueryRecorder, "check_query", RICH_PROBE, "rich", "query"),
            (DroppedQueryRecorder, "check_query", RICH_PROBE, "rich", "query"),
            (DecodedPathRecorder, "check_path", RICH_PROBE, "rich", "path"),
            (HostAttributeRecorder, "check_host", NO_HOST_PROBE, "nohost", "invented"),
            (TextRoundTripRecorder, "check_body", LATIN_PROBE, "latin", "body"),
        ],
        ids=lambda v: v.__name__ if isinstance(v, type) else (v if isinstance(v, str) else ""),
    )
    async def test_defect_fails_its_named_check_and_no_other(
        self, cls, check_name: str, raw: bytes, marker: str, expected: str
    ) -> None:
        """Assert one deliberate capture defect is attributable to one check.

        Args:
            cls: The defective recorder.
            check_name: The check that must reject it.
            raw: The probe that exposes the defect.
            marker: The probe's marker.
            expected: Text the rejection message must carry, so the check is
                known to have failed for its own reason.
        """
        failing = next(c for c in PER_EXCHANGE_CHECKS if c.__name__ == check_name)
        upstream, captured, sent = await _run_single(cls, raw, marker)
        _assert_only(failing, captured, sent, upstream.scheme, expected)

    async def test_the_host_defect_is_invisible_when_a_host_header_is_sent(self) -> None:
        """Assert the no-``Host`` probe is load-bearing, not decoration.

        ``request.host`` returns the ``Host`` header verbatim whenever one was
        sent, so on any other probe this recorder is indistinguishable from a
        correct one. Recording that here stops someone "simplifying" the
        falsification matrix by reusing the rich probe for every case.
        """
        failing = next(c for c in PER_EXCHANGE_CHECKS if c.__name__ == "check_host")
        upstream, captured, sent = await _run_single(HostAttributeRecorder, RICH_PROBE, "rich")
        failing(captured, sent, scheme=upstream.scheme)

    async def test_the_text_round_trip_defect_is_invisible_on_utf8(self) -> None:
        """Assert the latin-1 probe is load-bearing.

        A body that is already valid UTF-8 survives ``decode`` + ``encode``
        unchanged, so on the ordinary probe this recorder looks correct.
        """
        failing = next(c for c in PER_EXCHANGE_CHECKS if c.__name__ == "check_body")
        upstream, captured, sent = await _run_single(
            TextRoundTripRecorder, probe("plain", body=b'{"a":1}'), "plain"
        )
        failing(captured, sent, scheme=upstream.scheme)


class TestSessionDefectsAreCaughtByExactlyTheirOwnCheck:
    """R4.1 / R4.2 for the four whole-recording checks."""

    @staticmethod
    async def _run_pair(cls: type[RecordingUpstream]):
        """Drive two ordinary requests at a recorder.

        Args:
            cls: The recorder class to instantiate.

        Returns:
            The recording, the requests sent, and the ports opened.
        """
        upstream = cls(default_format=WireFormat.CHAT_COMPLETIONS)
        await upstream.start()
        try:
            sent = [
                await send(upstream.host, upstream.port, probe(marker), marker=marker)
                for marker in ("aaa", "bbb")
            ]
            return recording_of(upstream), sent, [s.source_port for s in sent]
        finally:
            await upstream.stop()

    @staticmethod
    def _assert_only(failing, recording: Recording, sent, opened, expected: str) -> None:
        """Assert only ``failing`` rejects this recording, and for its own reason.

        Args:
            failing: The check that must reject.
            recording: The defective recording.
            sent: The requests that produced it.
            opened: The ports the driver opened.
            expected: Text the rejection message must contain.

        Raises:
            AssertionError: When the named check passes, rejects for the wrong
                reason, or when another check fails.
        """
        with pytest.raises(AssertionError, match=expected):
            failing(recording, sent, opened)

        for check in PER_SESSION_CHECKS:
            if check is failing:
                continue
            try:
                check(recording, sent, opened)
            except AssertionError as exc:  # pragma: no cover - only on a real defect
                raise AssertionError(
                    f"{failing.__name__} was the named check, but {check.__name__} "
                    f"also failed: {exc}"
                ) from exc

    async def test_a_reordering_recorder_fails_only_the_order_check(self) -> None:
        """Assert misordering is attributable to the ordering check alone."""
        failing = next(c for c in PER_SESSION_CHECKS if c.__name__ == "check_request_order")
        recording, sent, opened = await self._run_pair(ReorderingRecorder)
        self._assert_only(failing, recording, sent, opened, "request order")

    async def test_a_dropping_recorder_fails_only_the_coverage_check(self) -> None:
        """Assert a lost request is attributable to the coverage check alone."""
        failing = next(c for c in PER_SESSION_CHECKS if c.__name__ == "check_request_count")
        recording, sent, opened = await self._run_pair(DroppingRecorder)
        self._assert_only(failing, recording, sent, opened, "no capture for")

    async def test_a_miscounting_connection_log_fails_only_the_connection_check(self) -> None:
        """Assert a log that cannot attribute a capture to a connection is caught.

        Distinct from the lazy log above: every opened port *is* recorded, so
        the first half of the check passes and only the carried-count half
        fires. Both halves need their own defect, or one of them is an assertion
        nothing exercises.
        """
        failing = next(c for c in PER_SESSION_CHECKS if c.__name__ == "check_connection_logged")
        recording, sent, opened = await self._run_pair(MiscountingConnectionRecorder)
        self._assert_only(failing, recording, sent, opened, "accounts for 0 requests")

    async def test_a_frozen_clock_fails_only_the_arrival_check(self) -> None:
        """Assert a constant clock is attributable to the arrival check alone."""
        failing = next(c for c in PER_SESSION_CHECKS if c.__name__ == "check_arrival_increases")
        recording, sent, opened = await self._run_pair(FrozenClockRecorder)
        self._assert_only(failing, recording, sent, opened, "strictly increase")

    async def test_a_late_clock_is_caught_when_a_body_arrives_slowly(self) -> None:
        """Assert stamping arrival after the body read is caught.

        The probe is what makes this observable: the first request's body is
        delivered *after* the second request has completed, so a recorder that
        stamps at capture time reverses the two. A recorder stamping at handler
        entry — which is what the real one does — records them in the order they
        arrived.
        """
        failing = next(c for c in PER_SESSION_CHECKS if c.__name__ == "check_arrival_increases")

        upstream = LateClockRecorder(default_format=WireFormat.CHAT_COMPLETIONS)
        await upstream.start()
        try:
            recording, sent, opened = await _drive_slow_body_pair(upstream)
        finally:
            await upstream.stop()

        with pytest.raises(AssertionError, match="strictly increase"):
            failing(recording, sent, opened)

    async def test_the_same_probe_passes_against_the_real_recorder(self) -> None:
        """The positive control for the slow-body probe.

        Without it, a probe that had stopped producing overlapping requests
        would make the case above pass for the wrong reason.
        """
        upstream = RecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)
        await upstream.start()
        try:
            recording, sent, opened = await _drive_slow_body_pair(upstream)
        finally:
            await upstream.stop()

        for check in PER_SESSION_CHECKS:
            check(recording, sent, opened)

    async def test_a_lazily_logged_connection_log_fails_the_connection_check(self) -> None:
        """Assert a log that only sees request-bearing connections is caught.

        §5.2.1's bypass is a connection that carries **no** HTTP request. A
        recorder that logs from inside the request handler is correct for every
        other connection, so this is the only probe that distinguishes it — and
        the reason :func:`check_connection_logged` is given the opened ports
        rather than inferring them from what was sent.
        """
        upstream = LazyConnectionRecorder(default_format=WireFormat.CHAT_COMPLETIONS)
        await upstream.start()
        try:
            silent_port = await open_connection_only(upstream.host, upstream.port)
            sent = await send(upstream.host, upstream.port, probe("c"), marker="c")
            recording = recording_of(upstream)
        finally:
            await upstream.stop()

        opened = [silent_port, sent.source_port]
        with pytest.raises(AssertionError, match="opened but never logged"):
            check_connection_logged(recording, [sent], opened)

    async def test_the_real_recorder_logs_the_same_silent_connection(self) -> None:
        """The positive control for the probe above.

        Without it, a probe that had stopped opening a silent connection would
        make the case above pass for the wrong reason — and the defect it names
        would go unexercised again.
        """
        upstream = RecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)
        await upstream.start()
        try:
            silent_port = await open_connection_only(upstream.host, upstream.port)
            sent = await send(upstream.host, upstream.port, probe("c"), marker="c")
            recording = recording_of(upstream)
        finally:
            await upstream.stop()

        check_connection_logged(recording, [sent], [silent_port, sent.source_port])


async def _drive_slow_body_pair(upstream: RecordingUpstream):
    """Send two requests whose arrival order and completion order differ.

    The first request's headers arrive first, but its body is withheld until the
    second request has been served in full. A recorder stamping ``arrival`` at
    handler entry therefore records them in one order, and one stamping at
    capture time records them in the other.

    Args:
        upstream: The running recorder.

    Returns:
        The recording, the requests sent in the order they began, and the ports
        opened.
    """
    body = b'{"slow":1}'
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\n"
        f"{MARKER_HEADER}: slow\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode()

    slow_sock = socket.create_connection((upstream.host, upstream.port))
    try:
        slow_port = slow_sock.getsockname()[1]
        await asyncio.to_thread(slow_sock.sendall, head)
        # The handler is now blocked reading the body, so its arrival is stamped.
        await _wait_until(
            lambda: len(upstream.connections) >= 1, "the slow request to be accepted"
        )

        fast = await send(upstream.host, upstream.port, probe("fast"), marker="fast")

        await asyncio.to_thread(slow_sock.sendall, body)
        await _wait_until(lambda: len(upstream.requests) >= 2, "the slow request's body")
    finally:
        slow_sock.close()

    slow = _parse_sent(head + body, slow_port, "slow")
    return recording_of(upstream), [slow, fast], [slow_port, fast.source_port]


async def _wait_until(predicate, what: str) -> None:
    """Yield to the loop until ``predicate`` holds.

    Args:
        predicate: The condition to wait for.
        what: What is being waited for, for the failure message.

    Raises:
        AssertionError: When the condition does not hold within the budget.
    """
    for _ in range(400):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")
