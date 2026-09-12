"""What every recording upstream must do, and the driver that proves it.

`.system_design/TEST_SUITE.md` §7.2 · plan task **T-W4** (KBR-27).

§7.2 assigns one recorder per transport — four of them — and plan §5 says each
"must pass T-W4's conformance test". This module is that test, expressed as
**pure functions over data** so that every one of them can be handed a
deliberately wrong :class:`~harness.contract.CapturedRequest` and asked whether
it notices. Plan §1.4 makes that mandatory, not stylistic.

**Why a raw socket and not a client library.**  A client library reorders,
re-cases, adds and drops headers; aiohttp's client does several of those. A test
that sends through one cannot distinguish a recorder that *loses* casing from a
client that never *sent* mixed casing. The same reasoning is what makes this
driver shared rather than per-recorder: botocore's SigV4 canonicalises the path
and owns the header set, and curl injects ``Host``, ``Accept``, ``User-Agent``
and ``Expect: 100-continue`` — so a conformance check driven through either
library would assert only that the recorder captured whatever the library chose
to send, which is equally true of a recorder that lower-cases everything the
library had already lower-cased. All four recorders in §7.2 are HTTP servers the
harness itself runs; TLS is the only difference, and an
:class:`ssl.SSLContext` covers it.

**Captures are correlated to sends by a marker, never by position.**  Order is
then a separate assertion (:func:`check_request_order`) instead of an assumption
baked into the other thirteen checks. Without that, a recorder that merely
*reorders* its list would fail every per-exchange check as well, and the
falsification suite could not say which defect it had caught.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.contract import CapturedRequest

__all__ = [
    "SentRequest",
    "Recording",
    "send_raw",
    "send",
    "open_only",
    "open_connection_only",
    "correlate",
    "marker_of",
    "MARKER_HEADER",
    "recording_of",
    "probe",
    "RICH_PROBE",
    "RICH_BODY",
    "NO_HOST_PROBE",
    "LATIN_PROBE",
    "LATIN_BODY",
    "PER_EXCHANGE_CHECKS",
    "PER_SESSION_CHECKS",
    "CHECK_NAMES",
    "check_method",
    "check_scheme",
    "check_host",
    "check_path",
    "check_query",
    "check_body",
    "check_header_casing",
    "check_header_order",
    "check_header_duplicates",
    "check_peer_port",
    "check_arrival_increases",
    "check_request_order",
    "check_request_count",
    "check_connection_logged",
]

#: How long the driver waits on a probe before giving up. A hung probe must fail
#: fast and loudly: these tests run in the `l1` gate, where a socket that never
#: answers would otherwise stall the job rather than fail it.
PROBE_TIMEOUT = 5.0

#: How long :func:`_advance_clock` will wait for the clock to report a new
#: instant, expressed as a **number of polls** rather than a deadline. Windows'
#: ``time.monotonic`` advances in ~15.6 ms steps, so 100 polls of a millisecond
#: is several ticks of headroom.
#:
#: Counted rather than timed on purpose: a deadline has to be computed from a
#: clock, and the case this budget exists for is a clock that has **stopped** —
#: under which ``now < deadline`` stays true forever and the bound never fires.
#: That is not hypothetical. The first draft of this function bounded itself with
#: ``time.monotonic()`` and hung the suite on the frozen-clock case
#: :func:`check_arrival_increases` exists to catch.
_CLOCK_TICK_POLLS = 100
_CLOCK_TICK_POLL = 0.001

#: The header every probe request carries its correlation marker in. See
#: :func:`marker_of` for why it must be a header and nothing else.
MARKER_HEADER = "X-Probe-Marker"


@dataclass(frozen=True)
class SentRequest:
    """Exactly what the driver put on the wire, and what it can be read as.

    Every field but :attr:`raw` and :attr:`source_port` is **derived from**
    :attr:`raw` by :func:`_parse_sent`, never declared alongside it. A driver
    that reported what it *intended* to send rather than what it actually wrote
    is the classic harness lie — all fourteen checks compare a capture against
    this object, so a discrepancy between intent and wire would be invisible in
    every one of them.

    Attributes:
        raw: The bytes written to the socket, in full.
        source_port: The driver socket's own port, which a conforming recorder
            reports as :attr:`~harness.contract.CapturedRequest.peer_port`.
        marker: The unique value identifying this request among a probe's
            several, used to correlate captures to sends without relying on
            position.
        method: The request method, parsed from :attr:`raw`.
        path: The target's path component.
        query: The target's query component, ``""`` when absent.
        headers: Header pairs in wire order, casing and duplicates preserved,
            decoded ``latin-1``.
        body: The entity body.
    """

    raw: bytes
    source_port: int
    marker: str
    method: str
    path: str
    query: str
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def header(self, name: str) -> str | None:
        """Return the first value for ``name``, matched case-insensitively.

        Args:
            name: The header name to look for.

        Returns:
            The value, or ``None`` when the header was not sent.
        """
        lowered = name.lower()
        for sent_name, value in self.headers:
            if sent_name.lower() == lowered:
                return value
        return None


@dataclass
class Recording:
    """What a recorder observed during one probe session.

    The two lists are separate because §5.2.1 needs them to be: a bypass that
    opens a TCP connection and sends no HTTP request at all is invisible in
    :attr:`requests`, and that connection is *"the only thing this assertion
    needs to catch"*.

    Attributes:
        requests: Captures, in the order the recorder recorded them.
        connections: One entry per accepted connection, as
            ``(connection_id, peer_port, carried_requests)``.
    """

    requests: list[CapturedRequest] = field(default_factory=list)
    connections: list[tuple[int, int, int]] = field(default_factory=list)


def _parse_sent(raw: bytes, source_port: int, marker: str) -> SentRequest:
    """Read back the bytes the driver wrote, as a recorder should read them.

    Deliberately a hand-written parser rather than a call into any HTTP library:
    this is the reference the recorder is judged against, so it must not share
    an implementation — or a bug — with the thing under test.

    Args:
        raw: The exact bytes written to the socket.
        source_port: The driver socket's own port.
        marker: The correlation marker carried by this request.

    Returns:
        The parsed :class:`SentRequest`.

    Raises:
        ValueError: When ``raw`` is not a well-formed HTTP/1.x request. The
            driver builds these bytes itself, so this is a defect in the probe,
            never in the recorder.
    """
    head, separator, body = raw.partition(b"\r\n\r\n")
    if not separator:
        raise ValueError("probe bytes contain no header terminator")

    lines = head.split(b"\r\n")
    request_line = lines[0].decode("latin-1")
    try:
        method, target, _version = request_line.split(" ", 2)
    except ValueError as exc:
        raise ValueError(f"malformed request line: {request_line!r}") from exc

    # latin-1 throughout: header bytes may carry obs-text that utf-8 refuses,
    # and the contract's `headers` field is typed `str`.
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        name, header_separator, value = line.decode("latin-1").partition(":")
        if not header_separator:
            raise ValueError(f"malformed header line: {line!r}")
        headers.append((name, value.lstrip(" ")))

    path, _, query = target.partition("?")
    return SentRequest(
        raw=raw,
        source_port=source_port,
        marker=marker,
        method=method,
        path=path,
        query=query,
        headers=tuple(headers),
        body=body,
    )


def send_raw(
    host: str,
    port: int,
    raw: bytes,
    *,
    marker: str,
    ssl_context: ssl.SSLContext | None = None,
) -> SentRequest:
    """Write ``raw`` to a recorder and report what was written.

    Args:
        host: The recorder's bind address.
        port: The recorder's listening port.
        raw: The complete request bytes, headers and body.
        marker: The unique value identifying this request within a probe.
        ssl_context: Wraps the socket when given, which is how the same driver
            reaches T-B2's TLS-terminating recorder (§7.2).

    Returns:
        The :class:`SentRequest`, carrying the socket's own source port.
    """
    # Stand this probe apart from whatever arrived before it, and do it *before*
    # the request can reach the recorder: the earlier arrival may have been
    # produced by hand rather than by this driver. See :func:`_advance_clock`.
    _advance_clock()

    sock = socket.create_connection((host, port), timeout=PROBE_TIMEOUT)
    try:
        if ssl_context is not None:
            sock = ssl_context.wrap_socket(sock, server_hostname=host)
        # Read the source port from the socket itself. Under TLS this must come
        # after the wrap, because `wrap_socket` returns a different object.
        source_port = sock.getsockname()[1]
        sock.sendall(raw)
        # Wait for the reply before closing: it is the only signal that the
        # recorder has finished handling the request, so the capture is
        # guaranteed to exist by the time this returns.
        _drain(sock)
    finally:
        sock.close()

    return _parse_sent(raw, source_port, marker)


def _advance_clock() -> None:
    """Block until ``time.monotonic()`` reports an instant later than now.

    **Why the driver owns this.** :func:`check_arrival_increases` asserts that
    arrival stamps *strictly* increase across a recorded session. That is a claim
    about the recorder, but whether it can be true at all depends on the
    platform's clock: ``time.monotonic()`` resolves to ~15.6 ms on Windows and to
    nanoseconds on Linux and macOS, so two probes sent back to back are stamped
    at the same instant on one platform and at different instants on the others.
    Measured on CI: the same assertion failed on Windows in one run and passed in
    the next, and KBR-188's exemptions — which fail when the assertion
    *unexpectedly passes* — made the leg red in both directions.

    Separating the probes is therefore the driver's job, not each test's, and not
    an exemption's. It is done here, once, so every recorder and every check
    inherits it: §7.2's four recorders are judged by one driver.

    **Called before the exchange, not after it.** Waiting at the end of
    :func:`send_raw` would separate a probe only from the driver's own previous
    probe, leaving one that follows a hand-rolled request — §6.3's slow-body
    pair sends its first request on a raw socket — sharing that request's
    instant. Waiting at the start separates every probe from everything before
    it, whoever sent it.

    **A condition, never a fixed sleep.** It returns as soon as the clock has
    moved, which on Linux and macOS is the first look and costs nothing
    measurable. The rule the repo states about waiting is exactly this: wait on a
    condition with a timeout.

    On a clock that never advances it gives up after :data:`_CLOCK_TICK_POLLS`
    looks and returns rather than raising. A stopped clock is the *constant
    clock* defect :func:`check_arrival_increases` exists to catch, so the honest
    outcome is to let that check fail — not to hang the gate, and not to report
    the driver's own impatience as a recorder defect.
    """
    started = time.monotonic()
    for _ in range(_CLOCK_TICK_POLLS):
        if time.monotonic() != started:
            return
        time.sleep(_CLOCK_TICK_POLL)


async def send(
    host: str,
    port: int,
    raw: bytes,
    *,
    marker: str,
    ssl_context: ssl.SSLContext | None = None,
) -> SentRequest:
    """Send a probe from a worker thread, so the recorder can serve it.

    :func:`send_raw` uses blocking sockets, and a recorder under test runs on
    the **same event loop** as the test that drives it. Calling it inline would
    block that loop for the whole exchange: the server could never accept the
    connection, the read would time out, and the recording would come back empty
    — a failure that looks like a broken recorder rather than a broken probe.

    Args:
        host: The recorder's bind address.
        port: The recorder's listening port.
        raw: The complete request bytes.
        marker: The unique value identifying this request within a probe.
        ssl_context: Wraps the socket when given.

    Returns:
        The :class:`SentRequest`, carrying the socket's own source port.
    """
    return await asyncio.to_thread(
        send_raw, host, port, raw, marker=marker, ssl_context=ssl_context,
    )


def open_only(host: str, port: int, *, ssl_context: ssl.SSLContext | None = None) -> int:
    """Open a connection, send nothing, close it, and report its source port.

    This is §5.2.1's bypass shape as a probe. It produces no
    :class:`SentRequest`, because nothing was sent — which is precisely why
    :func:`check_connection_logged` takes the opened ports separately.

    Args:
        host: The recorder's bind address.
        port: The recorder's listening port.
        ssl_context: Wraps the socket when given.

    Returns:
        The source port of the connection that was opened.
    """
    sock = socket.create_connection((host, port), timeout=PROBE_TIMEOUT)
    try:
        if ssl_context is not None:
            sock = ssl_context.wrap_socket(sock, server_hostname=host)
        return int(sock.getsockname()[1])
    finally:
        sock.close()


async def open_connection_only(
    host: str, port: int, *, ssl_context: ssl.SSLContext | None = None
) -> int:
    """Open a silent connection from a worker thread.

    Args:
        host: The recorder's bind address.
        port: The recorder's listening port.
        ssl_context: Wraps the socket when given.

    Returns:
        The source port of the connection that was opened.
    """
    return await asyncio.to_thread(open_only, host, port, ssl_context=ssl_context)


def _drain(sock: socket.socket) -> bytes:
    """Read one chunk of the reply, or give up at the timeout.

    Args:
        sock: The connected socket.

    Deliberately **not** a read to end-of-connection: aiohttp holds a
    connection open for ``keepalive_timeout`` (3630 seconds by default), so
    reading to EOF would stall for the socket timeout on every probe.

    Returns:
        Whatever arrived; the content is not inspected, because what the
        recorder *replies* is the script's business and not a conformance
        property.
    """
    chunks: list[bytes] = []
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            # One read is enough to know the recorder answered. Looping to EOF
            # would block for the whole keep-alive timeout on a connection the
            # recorder is entitled to hold open.
            break
    except (TimeoutError, OSError):
        pass
    return b"".join(chunks)


def recording_of(upstream: Any) -> Recording:
    """Adapt a recorder's own observations to what the checks read.

    Duck-typed on ``requests`` and ``connections`` rather than typed against
    :class:`~harness.recorder.RecordingUpstream`, so this module stays free of
    any particular recorder — T-B1, T-B2 and T-B3 each bring their own and all
    four are judged here.

    Args:
        upstream: Any recorder exposing ``requests`` and ``connections``.

    Returns:
        The :class:`Recording` the checks consume.
    """
    return Recording(
        requests=list(upstream.requests),
        connections=[(c.connection_id, c.peer_port, c.requests) for c in upstream.connections],
    )


def probe(marker: str, *, path: str = "/v1/chat/completions", body: bytes = b"{}") -> bytes:
    """Build a plain, well-formed probe request.

    Args:
        marker: The correlation marker; travels in a header, because every other
            location is destroyed by one of the falsification defects.
        path: The request target.
        body: The entity body.

    Returns:
        The complete request bytes.
    """
    return (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: upstream.test\r\n"
        f"{MARKER_HEADER}: {marker}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
    ).encode() + body


#: The body of :data:`RICH_PROBE`. Non-ASCII so a recorder that round-trips the
#: body through ``str`` has something to lose.
RICH_BODY = b'{"stream": false, "note": "\xc2\xa0"}'

#: The probe every per-exchange check is run against: mixed header casing, a
#: duplicated header, an obs-text value that ``utf-8`` would refuse, a
#: percent-encoded path whose two escapes differ in case, and a query that is
#: neither sorted nor decodable without changing it.
#:
#: The path ends in ``/chat/completions`` deliberately. The encoded segments are
#: what :func:`check_path` reads; the *suffix* is what the recorder dispatches
#: on, and a probe that matched no suffix would silently exercise the
#: wrong-format fallback in every test that used it.
RICH_PROBE = (
    b"POST /v1/mess%61ges/a%2Fb%2fc/chat/completions?Beta=A%20B&key=k&beta=c HTTP/1.1\r\n"
    b"Host: Upstream.Example:443\r\n"
    b"X-Api-Key: secret\r\n"
    b"anthropic-version: 2023-06-01\r\n"
    b"ANTHROPIC-BETA: one\r\n"
    b"anthropic-beta: two\r\n"
    b"X-Weird: caf\xe9\r\n"
    b"X-Probe-Marker: rich\r\n"
    b"Content-Length: " + str(len(RICH_BODY)).encode() + b"\r\n"
    b"\r\n" + RICH_BODY
)

#: HTTP/1.0 with no ``Host`` header. The only probe on which a recorder reading
#: aiohttp's ``request.host`` is distinguishable from a correct one.
NO_HOST_PROBE = (
    b"POST /v1/chat/completions HTTP/1.0\r\n"
    b"X-Probe-Marker: nohost\r\n"
    b"Content-Length: 2\r\n"
    b"\r\n{}"
)

#: The body of :data:`LATIN_PROBE`, high bytes declared ``latin-1``.
LATIN_BODY = b'{"n":"caf\xe9"}'

#: A body that survives ``decode`` but not ``decode`` + ``encode("utf-8")``. The
#: only probe on which a recorder recording ``request.text()`` re-encoded is
#: distinguishable from a correct one — on valid UTF-8 that defect is a no-op,
#: and on undeclared binary it raises instead of mis-capturing.
LATIN_PROBE = (
    b"POST /v1/chat/completions HTTP/1.1\r\n"
    b"Host: h\r\n"
    b"X-Probe-Marker: latin\r\n"
    b"Content-Type: application/json; charset=latin-1\r\n"
    b"Content-Length: " + str(len(LATIN_BODY)).encode() + b"\r\n"
    b"\r\n" + LATIN_BODY
)


def marker_of(captured: CapturedRequest) -> str | None:
    """Return the probe marker a capture carries, or ``None``.

    The marker travels in a **header**, looked up case-insensitively and
    compared on its *value*. Every other location is destroyed by one of the
    defects the falsification suite deliberately introduces: a marker in the
    query dies to the query-dropping and query-decoding defects, one in the path
    dies to the path-decoding defect, and one in the body dies to the
    re-encoding defect. Correlation would then fail with "no capture found"
    instead of the named check failing — a check failing for the wrong reason,
    which R4.2 exists to forbid.

    Case-insensitive on the *name* so the name-lower-casing defect does not
    break correlation either.

    Args:
        captured: A recorded request.

    Returns:
        The marker value, or ``None`` when the header is absent.
    """
    for name, value in captured.headers:
        if name.lower() == MARKER_HEADER.lower():
            return value
    return None


def correlate(recording: Recording, sent: SentRequest) -> CapturedRequest:
    """Return the capture matching ``sent``, found by its marker header.

    Args:
        recording: What the recorder observed.
        sent: The request to find.

    Returns:
        The single capture carrying ``sent``'s marker.

    Raises:
        AssertionError: When no capture, or more than one, carries the marker.
            Matching by marker rather than by position is what keeps a
            *reordering* defect from failing every per-exchange check as well as
            the ordering one. Refusing an ambiguous match rather than taking the
            first is what stops a request-duplicating recorder passing silently.
    """
    matches = [c for c in recording.requests if marker_of(c) == sent.marker]
    assert len(matches) == 1, (
        f"expected exactly one capture carrying marker {sent.marker!r}, found {len(matches)}"
    )
    return matches[0]


# --------------------------------------------------------------------------
# Per-exchange checks — one capture against the request that produced it
# --------------------------------------------------------------------------


def check_method(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the method survived.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: The scheme the recorder declares it serves; unused here, present
            so every per-exchange check shares one signature.

    Raises:
        AssertionError: When the recorded method differs from the one sent.
    """
    assert captured.method == sent.method, (
        f"method: recorded {captured.method!r}, sent {sent.method!r}"
    )


def check_scheme(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the scheme matches the one the recorder declares it serves.

    Compared against a declared value rather than a literal ``"http"``: §7.2's
    curl_cffi recorder terminates TLS, so a literal would fail T-B2 on the day
    it is written.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote; unused.
        scheme: The scheme the recorder declares it serves.

    Raises:
        AssertionError: When the recorded scheme differs.
    """
    assert captured.scheme == scheme, (
        f"scheme: recorded {captured.scheme!r}, recorder declares {scheme!r}"
    )


def check_host(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the authority is the ``Host`` header as sent, or empty when absent.

    A recorder reading aiohttp's ``request.host`` passes this on any probe that
    sends a ``Host`` — and fabricates the machine's own FQDN on one that does
    not, which is why the falsification suite aims this check at the no-``Host``
    probe specifically.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When the recorded host is not what was sent, or when a
            host was invented for a request that carried no ``Host`` header.
    """
    expected = sent.header("Host") or ""
    assert captured.host == expected, (
        f"host: recorded {captured.host!r}, sent {expected!r}"
        + ("; a host was invented for a request that sent none" if not expected else "")
    )


def check_path(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the path survived percent-encoded and un-normalised.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When the recorded path differs from the one sent —
            decoded, case-folded in its escapes, or re-split.
    """
    assert captured.path == sent.path, (
        f"path: recorded {captured.path!r}, sent {sent.path!r}"
    )


def check_query(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the query string survived raw, unordered and unparsed.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When the recorded query differs from the one sent.
    """
    assert captured.query == sent.query, (
        f"query: recorded {captured.query!r}, sent {sent.query!r}"
    )


def check_body(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the entity body survived byte-for-byte.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When the recorded body differs from the one sent.
    """
    assert captured.body == sent.body, (
        f"body: recorded {captured.body!r}, sent {sent.body!r}"
    )


def check_header_casing(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert every recorded header name is spelled as some sent name was.

    §4.3 C1 asserts on the exact header set, so a recorder that normalises
    casing destroys the evidence before the assertion runs.

    **Containment, not equality** — deliberately. The three header checks have to
    be orthogonal, or R4.2's "no other check fails" is unreachable: a recorder
    that *drops* a duplicate would fail an equality-based casing check too, and
    the falsification matrix could no longer say which defect was caught. Here,
    dropping leaves every surviving spelling correct, so only
    :func:`check_header_duplicates` fires.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When a recorded name is spelled in a way nothing sent.
    """
    sent_spellings = {name for name, _ in sent.headers}
    wrong = sorted({name for name, _ in captured.headers if name not in sent_spellings})
    assert not wrong, (
        f"header casing: recorded {wrong}, which no sent header is spelled as; "
        f"sent {sorted(sent_spellings)}"
    )


def check_header_order(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert headers kept their relative order.

    Recorded for the C1b baseline report rather than asserted against a vendor
    baseline (§7.2) — but a recorder that cannot preserve order cannot produce
    that report at all.

    **An order-preserving subsequence of the lower-cased names**, for the same
    orthogonality reason as :func:`check_header_casing`: comparing the sequences
    for equality would make this fire on a *dropped* header and on a *re-cased*
    one as well as on a reordered one. Lower-casing here leaves casing entirely
    to that check; allowing a subsequence leaves drops entirely to
    :func:`check_header_duplicates`.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When the recorded names are not in sent relative order.
    """
    recorded = [name.lower() for name, _ in captured.headers]
    expected = [name.lower() for name, _ in sent.headers]

    remaining = iter(expected)
    in_order = all(any(candidate == name for candidate in remaining) for name in recorded)
    assert in_order, f"header order: recorded {recorded} is not in the sent order {expected}"


def check_header_duplicates(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert each header name kept all its values, in order.

    A mapping-shaped recorder loses exactly this: ``anthropic-beta`` sent twice
    becomes one entry in a dict. Grouped by lower-cased name so that re-casing
    and reordering — which this check must stay silent about — do not disturb
    it.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When any name's sequence of values differs from what was
            sent.
    """
    recorded = _values_by_name(captured.headers)
    expected = _values_by_name(sent.headers)
    assert recorded == expected, (
        f"header values per name: recorded {recorded}, sent {expected}"
    )


def _values_by_name(headers: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """Group header values by lower-cased name, preserving their order.

    Args:
        headers: Header pairs in wire order.

    Returns:
        A mapping from lower-cased name to its values, in the order they
        appeared.
    """
    grouped: dict[str, list[str]] = {}
    for name, value in headers:
        grouped.setdefault(name.lower(), []).append(value)
    return grouped


def check_peer_port(captured: CapturedRequest, sent: SentRequest, *, scheme: str) -> None:
    """Assert the peer port is the client's own, not the recorder's.

    §5.2.1 joins the proxy's tunnel log on this value, so a recorder reporting
    its own listening port produces a join that succeeds against the wrong
    connection.

    Args:
        captured: What the recorder recorded.
        sent: What the driver wrote.
        scheme: Unused.

    Raises:
        AssertionError: When the recorded port is not the driver's source port.
    """
    assert captured.peer_port == sent.source_port, (
        f"peer_port: recorded {captured.peer_port!r}, client sent from {sent.source_port!r}"
    )


# --------------------------------------------------------------------------
# Per-session checks — the recording as a whole
# --------------------------------------------------------------------------


def check_arrival_increases(
    recording: Recording, sent: Sequence[SentRequest], opened_ports: Sequence[int]
) -> None:
    """Assert ``arrival`` moves forward, and is stamped before the reply.

    Driven against a probe whose responder is deliberately slow, so a recorder
    that stamps ``arrival`` *after* producing the response cannot pass by luck.

    **Arrivals are read in sent order, not in recorded order.** Walking the
    recorded list would make this fire on a recorder whose only fault is
    *appending out of order* — the defect :func:`check_request_order` owns — and
    R4.2 would be unreachable for both.

    Args:
        recording: What the recorder observed.
        sent: The requests that produced it, in the order they were sent.
        opened_ports: Unused; present so all four session checks share one
            signature and can be run from a single loop.

    Raises:
        AssertionError: When any timestamp is missing, not a float, or not
            strictly greater than that of the request sent before it.
    """
    by_marker = {marker_of(c): c for c in recording.requests}
    arrivals = [by_marker[s.marker].arrival for s in sent if s.marker in by_marker]

    assert all(isinstance(a, float) for a in arrivals), f"arrival must be float, got {arrivals}"

    for earlier, later in zip(arrivals, arrivals[1:], strict=False):
        assert later is not None and earlier is not None and later > earlier, (
            f"arrival must strictly increase in sent order; got {arrivals}. A constant "
            "clock, or a timestamp taken after the response, produces this."
        )


def check_request_order(
    recording: Recording, sent: Sequence[SentRequest], opened_ports: Sequence[int]
) -> None:
    """Assert the captures that exist are in the order they were sent.

    Phrased over request **identity**, never over ``arrival``, and as an
    order-preserving **subsequence** rather than an equality. Both choices exist
    to keep R4.2 reachable: identity keeps a clock defect out of this check, and
    subsequence keeps a *dropped* request out of it —
    :func:`check_request_count` owns that one.

    Args:
        recording: What the recorder observed.
        sent: The requests that produced it, in the order they were sent.
        opened_ports: Unused; see :func:`check_arrival_increases`.

    Raises:
        AssertionError: When the recorded captures are not in sent relative
            order.
    """
    recorded = [marker_of(c) for c in recording.requests]
    expected = [s.marker for s in sent]

    remaining = iter(expected)
    in_order = all(any(candidate == marker for candidate in remaining) for marker in recorded)
    assert in_order, f"request order: recorded {recorded} is not in the sent order {expected}"


def check_request_count(
    recording: Recording, sent: Sequence[SentRequest], opened_ports: Sequence[int]
) -> None:
    """Assert every request that was sent has a capture.

    Phrased as coverage rather than as a length comparison, so a recorder that
    merely *reorders* passes here and fails only
    :func:`check_request_order`. A recorder that silently drops a request is
    caught here and by nothing else — §4.3 C6's "``/healthz`` never causes an
    upstream request" will lean on exactly this claim.

    Args:
        recording: What the recorder observed.
        sent: The requests that produced it.
        opened_ports: Unused; see :func:`check_arrival_increases`.

    Raises:
        AssertionError: When a sent request has no capture, or when the recorder
            produced captures nobody sent.
    """
    recorded = [marker_of(c) for c in recording.requests]
    missing = [s.marker for s in sent if s.marker not in recorded]
    assert not missing, f"no capture for {missing}; recorded {recorded}"

    unexpected = [m for m in recorded if m not in {s.marker for s in sent}]
    assert not unexpected, f"recorded captures nobody sent: {unexpected}"


def check_connection_logged(
    recording: Recording, sent: Sequence[SentRequest], opened_ports: Sequence[int]
) -> None:
    """Assert every connection the driver opened was logged, requests or not.

    §5.2.1's load-bearing negative is an upstream connection with no matching
    tunnel — *"the only thing this assertion needs to catch"*. Such a connection
    carries no HTTP request, so it is invisible in ``sent`` as well as in the
    request list. **That is why this check needs ``opened_ports``**: judged
    against ``sent`` alone it could only ever ask whether request-bearing
    connections were logged, which is the one thing a lazily-logging recorder
    gets right. An earlier version did exactly that and could not fail for the
    defect it names.

    Args:
        recording: What the recorder observed.
        sent: The requests that produced it.
        opened_ports: Every source port the driver opened, including ones that
            sent nothing.

    Raises:
        AssertionError: When a connection the driver opened has no record, or
            when the records do not account for every request.
    """
    logged_ports = {peer_port for _, peer_port, _ in recording.connections}

    unlogged = [port for port in opened_ports if port not in logged_ports]
    assert not unlogged, (
        f"connections opened but never logged: {unlogged}; logged {sorted(logged_ports)}. "
        "A connection that carries no request is still a connection, and a bypass "
        "looks exactly like one."
    )

    # Per-connection request counts are the other half of R1.10: a capture must
    # be attributable to the connection that carried it.
    carried = sum(count for _, _, count in recording.connections)
    assert carried == len(sent), (
        f"connection log accounts for {carried} requests, {len(sent)} were sent"
    )


#: The ten checks that judge one capture against the request that produced it.
PER_EXCHANGE_CHECKS: tuple[Callable[..., None], ...] = (
    check_method,
    check_scheme,
    check_host,
    check_path,
    check_query,
    check_body,
    check_header_casing,
    check_header_order,
    check_header_duplicates,
    check_peer_port,
)

#: The four checks that judge a recording as a whole.
PER_SESSION_CHECKS: tuple[Callable[..., None], ...] = (
    check_arrival_increases,
    check_request_order,
    check_request_count,
    check_connection_logged,
)

#: Every check by name, so a falsification case can name the one it must trip
#: and the suite can assert the list is the fourteen the design specifies.
CHECK_NAMES: tuple[str, ...] = tuple(
    check.__name__ for check in PER_EXCHANGE_CHECKS + PER_SESSION_CHECKS
)
