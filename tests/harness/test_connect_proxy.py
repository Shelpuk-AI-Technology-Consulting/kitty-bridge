"""L1 tests for the shared CONNECT proxy's containment capabilities.

`.system_design/TEST_SUITE.md` §5.2.1, §5.2.2, §7.3 · plan task **T-W5** (KBR-28).

The extraction itself is regression-tested by
:mod:`tests.test_egress_https_proxy`, which drives all three transport stacks
through the fixture and must pass unchanged.  This module tests only what T-W5
*adds*, and it adds exactly what the containment harness (T-E1/T-E2) cannot be
written without:

* the **tunnel source port**, §5.2.1's join key — an upstream connection that
  joins to no tunnel is a bypass, and that is the only thing the containment
  assertion needs to catch;
* **mid-test stoppability**, so §5.2.2 phase 2 can assert that with the proxy
  down the upstream accepts zero connections.

**Why a plain TCP sink rather than the TLS target.**  The join is between the
proxy's outbound socket and the *connection* the far end accepts, not the
request that rides on it (§5.2.1).  A plain sink records the peer port at
**accept** time, which is the semantics containment needs: a bypass that
connects and then fails TLS is still a bypass.  A TLS server's handler runs only
after a completed handshake and would miss exactly that case.  It also keeps
these tests off the TLS-in-TLS path, which aiohttp cannot take below Python
3.11.

**The falsification case** is
:meth:`TestUnattributableConnections.test_a_direct_connection_is_reported_as_unattributable`
(plan §1.4).  It opens one connection through the proxy and one straight past
it, and requires the join to name the second and clear the first.  Without it a
join that matched everything, or a ``source_port`` that was recorded but never
correlated, would look identical to a working one.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import ssl
from collections.abc import AsyncIterator, Callable

import pytest

from harness.connect_proxy import (
    EXPECTED_PROXY_AUTH,
    HARNESS_UPSTREAM_HOST,
    TARGET_BODY,
    CertFiles,
    ConnectAttempt,
    ConnectProxy,
    TlsTarget,
    server_ssl_context,
    unattributable_peer_ports,
)

#: How long a test waits for an asynchronous side effect before failing. Long
#: enough to absorb a loaded CI runner, short enough that a genuine hang is
#: reported as a failure rather than as a stuck job.
_SETTLE_TIMEOUT = 5.0


@dataclasses.dataclass
class _TcpSink:
    """A plain TCP server that records who connects and answers nothing.

    Attributes:
        port: The 127.0.0.1 port it listens on.
        peer_ports: The source port of every connection accepted, in order.
    """

    port: int
    peer_ports: list[int]


@pytest.fixture
async def tcp_sink() -> AsyncIterator[_TcpSink]:
    """Run a connection-recording TCP sink for one test.

    Yields:
        The running sink, with its port and the peer ports it has accepted.
    """
    peer_ports: list[int] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Record the peer port, then hold the connection until the peer leaves.

        Args:
            reader: The accepted connection's read side.
            writer: The accepted connection's write side.
        """
        # Recorded at accept time and before any read: a bypass that connects
        # and then sends nothing must still be counted (§5.2.1).
        peer_ports.append(writer.get_extra_info("peername")[1])
        try:
            await reader.read()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield _TcpSink(port=server.sockets[0].getsockname()[1], peer_ports=peer_ports)
    finally:
        # Deliberately no `wait_closed()`: on Python 3.12.1+ it blocks until
        # every connection is dropped, and a test that leaves a tunnel open is
        # the normal case here. This is a plain TCP server with no TLS
        # shutdown to negotiate, so closing the listener is the whole teardown;
        # the loop reclaims the handler tasks. `ConnectProxy` and `TlsTarget`
        # need `stop()` precisely because they do not have that luxury.
        server.close()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    """Wait for a condition to hold, rather than sleeping for a guessed interval.

    Args:
        predicate: Called repeatedly; the wait ends when it returns ``True``.

    Raises:
        AssertionError: When the condition has not held within
            :data:`_SETTLE_TIMEOUT`.
    """
    # A `sleep()` long enough to be reliable on a loaded runner is also long
    # enough to make the suite slow, which is the usual road to a flaky test.
    async def poll() -> None:
        """Yield to the loop until the condition holds."""
        while not predicate():
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(poll(), timeout=_SETTLE_TIMEOUT)
    except asyncio.TimeoutError:  # pragma: no cover - only on a real failure
        pytest.fail(f"condition did not hold within {_SETTLE_TIMEOUT}s")


async def _open_tunnel(
    proxy: ConnectProxy, certs: CertFiles, target: str, *, authorization: str = EXPECTED_PROXY_AUTH
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:
    """Open one CONNECT tunnel through the proxy by hand.

    Driving the handshake directly, rather than through a client library, keeps
    these tests on a single TLS layer -- the proxy hop -- so they run on every
    supported Python instead of only on the 3.11+ that aiohttp needs for
    TLS-in-TLS.

    Args:
        proxy: The running proxy.
        certs: The session's throwaway certificates, for the proxy hop's trust.
        target: The ``host:port`` to ask the proxy to tunnel to.
        authorization: The ``Proxy-Authorization`` value to present; pass a
            wrong one to exercise the rejection path.

    Returns:
        The tunnel's streams and the proxy's status line.
    """
    context = ssl.create_default_context(cafile=str(certs.ca))
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port, ssl=context, server_hostname="localhost")
    request = (
        f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\nProxy-Authorization: {authorization}\r\n\r\n"
    )
    writer.write(request.encode())
    await writer.drain()

    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await reader.read(4096)
        if not chunk:
            break
        head += chunk
    status = head.decode("latin-1").split("\r\n")[0]
    return reader, writer, status


class TestTunnelSourcePortRecording:
    """`ConnectAttempt` carries the join key §5.2.1 needs, or it carries nothing."""

    async def test_records_the_outbound_source_port_of_an_established_tunnel(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """The recorded port is the one the far end actually saw connect."""
        _reader, writer, status = await _open_tunnel(connect_proxy, certs, f"127.0.0.1:{tcp_sink.port}")
        try:
            assert "200" in status
            await _wait_until(lambda: len(tcp_sink.peer_ports) == 1)

            assert connect_proxy.attempts[0].source_port == tcp_sink.peer_ports[0]
        finally:
            writer.close()

    async def test_records_no_source_port_when_authentication_is_rejected(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """A rejected CONNECT is still recorded, but it opened no tunnel."""
        wrong = "Basic " + base64.b64encode(b"nobody:nothing").decode()
        _reader, writer, status = await _open_tunnel(
            connect_proxy, certs, f"127.0.0.1:{tcp_sink.port}", authorization=wrong
        )
        try:
            assert "407" in status

            assert len(connect_proxy.attempts) == 1
            assert connect_proxy.attempts[0].authenticated is False
            # The load-bearing assertion. A companion `peer_ports == []` would
            # be decorative: read with no settling wait, it would also pass
            # while a wrongly-opened connection was still waiting to be
            # accepted, and there is no deterministic way to wait for an event
            # that must never happen.
            assert connect_proxy.attempts[0].source_port is None
        finally:
            writer.close()


class TestProxiedLegResolution:
    """The harness owns the resolver because the harness is the proxy (§5.3)."""

    async def test_tunnels_to_a_name_the_public_dns_cannot_resolve(
        self, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """A sealed-network upstream is addressed by an unresolvable name, by design.

        §5.3 requires the upstream to sit outside the ``localhost`` family, or
        ``egress.should_bypass()`` routes it direct and the harness proves the
        opposite of its claim.  The name it picks is a ``.invalid`` one, which
        RFC 2606 guarantees never resolves — so the proxy has to map it, and
        that mapping is the seam T-E1 builds on.
        """
        target = f"{HARNESS_UPSTREAM_HOST}:443"
        proxy = ConnectProxy(resolve={target: ("127.0.0.1", tcp_sink.port)})
        await proxy.start(server_ssl_context(certs.proxy_cert, certs.proxy_key))
        try:
            _reader, writer, status = await _open_tunnel(proxy, certs, target)
            try:
                assert "200" in status
                await _wait_until(lambda: len(tcp_sink.peer_ports) == 1)

                assert proxy.attempts[0].target == target
                assert proxy.attempts[0].source_port == tcp_sink.peer_ports[0]
            finally:
                writer.close()
        finally:
            await proxy.stop()

    async def test_an_unmapped_name_is_answered_502_and_opens_no_tunnel(
        self, connect_proxy: ConnectProxy, certs: CertFiles
    ) -> None:
        """"Tunnel attempted, upstream unreachable" is not "no tunnel attempted".

        §5.2.2 phase 2 has to tell those two apart to diagnose a run, so an
        unreachable upstream gets a status line rather than a silent close.
        """
        _reader, writer, status = await _open_tunnel(connect_proxy, certs, f"{HARNESS_UPSTREAM_HOST}:443")
        try:
            assert "502" in status

            assert connect_proxy.attempts[0].authenticated is True
            assert connect_proxy.attempts[0].source_port is None
        finally:
            writer.close()

    async def test_the_target_certificate_answers_to_the_harness_name(
        self, certs: CertFiles, tls_target: TlsTarget
    ) -> None:
        """A mapped route is worthless if the hop it reaches cannot be verified."""
        context = ssl.create_default_context(cafile=str(certs.ca))
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", tls_target.port, ssl=context, server_hostname=HARNESS_UPSTREAM_HOST
        )
        try:
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            await writer.drain()

            assert TARGET_BODY.encode() in await reader.read(4096)
        finally:
            writer.close()


class TestMidTestStoppability:
    """§5.2.2 phase 2 needs the proxy down while the test is still running."""

    async def test_stopping_the_proxy_refuses_further_connections(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """After `stop()` the port is dead, and the far end saw only the first connection."""
        _reader, writer, _status = await _open_tunnel(connect_proxy, certs, f"127.0.0.1:{tcp_sink.port}")
        await _wait_until(lambda: len(tcp_sink.peer_ports) == 1)
        writer.close()

        await connect_proxy.stop()

        with pytest.raises(ConnectionRefusedError):
            await asyncio.open_connection("127.0.0.1", connect_proxy.port)
        assert len(tcp_sink.peer_ports) == 1

    async def test_stopping_the_target_returns_while_a_connection_is_open(
        self, certs: CertFiles, tls_target: TlsTarget
    ) -> None:
        """The target carries the same teardown hazard the proxy does.

        §5.2.2 phase 2 stops the proxy mid-exchange, which is exactly the state
        that leaves a connection attached at the target end.
        """
        context = ssl.create_default_context(cafile=str(certs.ca))
        _reader, writer = await asyncio.open_connection(
            "127.0.0.1", tls_target.port, ssl=context, server_hostname="localhost"
        )
        try:
            await asyncio.wait_for(tls_target.stop(), timeout=_SETTLE_TIMEOUT)
        finally:
            writer.close()

    async def test_stopping_the_proxy_returns_when_the_client_ignores_close_notify(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """The case the abort design exists for, and the one a polite client hides.

        `asyncio.open_connection`'s SSL protocol answers an inbound
        ``close_notify`` on its own, without the application reading anything,
        so a test using it proves only that `stop()` returns for a
        **cooperative** peer.  A pooled urllib3 or botocore connection sitting
        idle through §5.2.2 phase 2 is the non-cooperative case, and there a
        graceful `close()` waits out `ssl_shutdown_timeout` -- 30 seconds.
        `pause_reading()` reproduces that deterministically, with no thread and
        no sleep.
        """
        _reader, writer, status = await _open_tunnel(connect_proxy, certs, f"127.0.0.1:{tcp_sink.port}")
        try:
            assert "200" in status
            await _wait_until(lambda: len(tcp_sink.peer_ports) == 1)
            writer.transport.pause_reading()

            await asyncio.wait_for(connect_proxy.stop(), timeout=_SETTLE_TIMEOUT)
        finally:
            writer.transport.abort()

    async def test_stopping_the_target_returns_when_the_client_ignores_close_notify(
        self, certs: CertFiles, tls_target: TlsTarget
    ) -> None:
        """The target carries the same hazard, by a different route.

        Its handler is never cancelled, so it reaches its own `finally` and
        closes the connection gracefully.  If `stop()` can no longer reach that
        connection by then, the listener stays attached to it for the full
        shutdown timeout.
        """
        context = ssl.create_default_context(cafile=str(certs.ca))
        _reader, writer = await asyncio.open_connection(
            "127.0.0.1", tls_target.port, ssl=context, server_hostname="localhost"
        )
        try:
            # Paused before the request, and never read: the response is served
            # and the handler runs its teardown while this client answers
            # nothing. `completed` is the only observable for "the handler has
            # finished", and waiting on it is what puts `stop()` on the far side
            # of the teardown -- which is where the connection escapes.
            writer.transport.pause_reading()
            writer.write(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            await writer.drain()
            await _wait_until(lambda: tls_target.completed == 1)

            await asyncio.wait_for(tls_target.stop(), timeout=_SETTLE_TIMEOUT)
        finally:
            writer.transport.abort()

    async def test_stopping_the_proxy_twice_is_harmless(self, connect_proxy: ConnectProxy) -> None:
        """The fixture's teardown calls `stop()` after a test already has."""
        await connect_proxy.stop()

        await connect_proxy.stop()

    async def test_stopping_the_proxy_returns_while_a_tunnel_is_open(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """`stop()` tears down live tunnels rather than waiting on them.

        Python 3.12.1 changed ``asyncio.Server.wait_closed()`` to block until the
        server is closed **and** every connection is dropped; 3.10 and 3.11
        return immediately.  A ``stop()`` that closed the listener and awaited
        ``wait_closed()`` would therefore hang on 3.12 and 3.13 and pass on
        3.10 and 3.11 -- a harness that behaves differently on two
        contributors' machines.
        """
        _reader, writer, status = await _open_tunnel(connect_proxy, certs, f"127.0.0.1:{tcp_sink.port}")
        try:
            assert "200" in status
            await _wait_until(lambda: len(tcp_sink.peer_ports) == 1)

            await asyncio.wait_for(connect_proxy.stop(), timeout=_SETTLE_TIMEOUT)
        finally:
            writer.close()


class TestUnattributableConnections:
    """§5.2.1's join, stated once so T-E2 inherits it rather than re-deriving it."""

    @pytest.mark.parametrize(
        ("peer_ports", "source_ports", "expected"),
        [
            pytest.param([4001], [4001], [], id="a tunnelled connection is attributable"),
            pytest.param([4001, 4002], [4001], [4002], id="an untunnelled connection is not"),
            pytest.param([4001, 4001], [4001], [], id="one tunnel may carry several connections"),
            pytest.param([], [4001], [], id="a tunnel carrying nothing is not a finding"),
            pytest.param([4002], [], [4002], id="no tunnel at all leaves every connection unexplained"),
            pytest.param([4001, 4001], [], [4001, 4001], id="two unexplained connections are two findings"),
        ],
    )
    def test_reports_exactly_the_peer_ports_no_tunnel_explains(
        self, peer_ports: list[int], source_ports: list[int], expected: list[int]
    ) -> None:
        """The join is at the connection level, never request-to-CONNECT (§5.2.1)."""
        attempts = [ConnectAttempt(target="t:443", authenticated=True, source_port=p) for p in source_ports]

        assert unattributable_peer_ports(peer_ports, attempts) == expected

    def test_a_one_shot_iterable_is_judged_rather_than_consumed(self) -> None:
        """A generator argument must not come back clean because it was read twice.

        The annotation invites one: a caller reducing a request log writes
        ``(r.peer_port for r in log)``. An implementation that sweeps for unset
        ports and then filters would exhaust it on the first pass and report no
        bypass for a breached run -- the same unfalsifiability the unset-port
        guard exists to refuse, arriving through the front door.
        """
        ports = (port for port in [4001, 4002])
        attempts = [ConnectAttempt(target="t:443", authenticated=True, source_port=4001)]

        assert unattributable_peer_ports(ports, attempts) == [4002]

    def test_an_unset_peer_port_is_refused_rather_than_judged(self) -> None:
        """A recorder that never filled the field cannot be silently forgiven.

        `CapturedRequest.peer_port` defaults to `None` and is T-W4's to
        populate. Treating an unset port as attributable would make containment
        pass on a recorder that observes nothing; treating it as a bypass would
        fail every run for the wrong reason. Neither is a judgement.
        """
        with pytest.raises(ValueError, match="position"):
            unattributable_peer_ports([4001, None], [])  # type: ignore[list-item]

    def test_a_rejected_tunnels_absent_source_port_matches_nothing(self) -> None:
        """`None` is the absence of a join key, not a key that joins to anything."""
        rejected = ConnectAttempt(target="t:443", authenticated=False, source_port=None)

        assert unattributable_peer_ports([4001], [rejected]) == [4001]

    async def test_the_join_holds_when_many_tunnels_are_open_at_once(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """One connection joining one tunnel proves nothing about ordering.

        The containment slices drive concurrent sessions through one listener,
        and `attempts` has no defined order under concurrency -- so the claim
        that survives is a set equality, not an index-by-index match.
        """
        target = f"127.0.0.1:{tcp_sink.port}"
        tunnels = await asyncio.gather(*(_open_tunnel(connect_proxy, certs, target) for _ in range(8)))
        try:
            await _wait_until(lambda: len(tcp_sink.peer_ports) == 8)

            # Eight distinct ports, asserted separately: a set equality alone
            # would also hold if two tunnels had somehow recorded one port.
            assert len(set(tcp_sink.peer_ports)) == 8
            assert {attempt.source_port for attempt in connect_proxy.attempts} == set(tcp_sink.peer_ports)
            assert unattributable_peer_ports(tcp_sink.peer_ports, connect_proxy.attempts) == []
        finally:
            for _reader, writer, _status in tunnels:
                writer.close()

    async def test_a_direct_connection_is_reported_as_unattributable(
        self, connect_proxy: ConnectProxy, certs: CertFiles, tcp_sink: _TcpSink
    ) -> None:
        """The falsification case: a real bypass, deliberately introduced (plan §1.4).

        A containment harness that has never been shown to fail is
        indistinguishable from one that cannot fail (design §5.2.2 phase 3).

        This does **not** discharge T-E2's own phase-3 obligation. That one
        injects a bypass into the *product* -- a patched `should_bypass`, or a
        session built without the proxy -- and requires the harness to fail.
        This one proves only that the join can tell the two kinds of connection
        apart at all, which is the precondition for T-E2's test meaning
        anything.
        """
        _reader, tunnelled, _status = await _open_tunnel(connect_proxy, certs, f"127.0.0.1:{tcp_sink.port}")
        await _wait_until(lambda: len(tcp_sink.peer_ports) == 1)
        # The bypass: straight to the far end, past the proxy entirely.
        _direct_reader, direct = await asyncio.open_connection("127.0.0.1", tcp_sink.port)
        try:
            await _wait_until(lambda: len(tcp_sink.peer_ports) == 2)
            direct_port = direct.get_extra_info("sockname")[1]

            unattributable = unattributable_peer_ports(tcp_sink.peer_ports, connect_proxy.attempts)

            # Both directions, stated rather than inferred. The equality below
            # already excludes "both were rejected" -- that would return two
            # ports, not one -- but a join is a claim about each connection,
            # and reading only the negative half invites the next author to
            # weaken the positive one without noticing.
            #
            # The join key is the proxy's OUTBOUND port, not the client's:
            # `tunnelled`'s own sockname belongs to the client-to-proxy leg and
            # never reaches the far end.
            assert connect_proxy.attempts[0].source_port in tcp_sink.peer_ports
            assert unattributable == [direct_port]
        finally:
            direct.close()
            tunnelled.close()
