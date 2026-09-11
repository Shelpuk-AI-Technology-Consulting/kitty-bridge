"""Shared recording CONNECT proxy and TLS target — the harness's proxy leg.

`.system_design/TEST_SUITE.md` §7.3 · plan task **T-W5** ([KBR-28]).

This module owns the local TLS CONNECT proxy, the local TLS target and the
throwaway certificate set they present.  All three were private to
:mod:`tests.test_egress_https_proxy` until T-W5; that module still holds the
only tests of ``kitty egress test``'s probe, but the servers now live here
because four other tasks need them:

* **T-E1/T-E2** (containment) need a proxy that records the **source port** of
  each tunnel it opens and that can be **stopped mid-test**;
* **T-G8/T-G10/T-G11** (dependency proxy contracts) need it as a plain fixture.

The alternative to extraction was a second proxy implementation, which is how
two harnesses come to disagree about what "tunnelled" means.

**The fixtures are registered as a pytest plugin** from ``tests/conftest.py``,
so any test asks for ``connect_proxy`` by name without importing this module.
``__all__`` names the *importable* surface; the four fixtures --  ``certs``,
``connect_proxy``, ``tls_target`` and ``aiohttp_trusts_test_ca`` -- are part of
the contract too, and are addressed by name rather than imported.

**A missing ``openssl`` fails, it does not skip.**  ``certs`` is now shared
infrastructure, and §8's rule is that a skip in a gating job is a failure:
a suite that silently stops proving containment because a tool is absent is
indistinguishable from one that proves it.  Do not "fix" :func:`_run_openssl`
into a skip.

**What this module does not decide.**  It provides the *seam* for addressing an
upstream by a name the public DNS cannot resolve — :data:`HARNESS_UPSTREAM_HOST`
in the target certificate, and :attr:`ConnectProxy.resolve` on the tunnel path.
Which transport gets which route, and the **direct**-leg override that §5.2.2
phase 1 needs, are T-E1's (KBR-61).

.. _KBR-28: https://shelpuk.atlassian.net/browse/KBR-28
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import ssl
import subprocess
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path

import aiohttp.connector
import pytest

from kitty.egress import EgressConfig

__all__ = [
    "EXPECTED_PROXY_AUTH",
    "HARNESS_UPSTREAM_HOST",
    "PROXY_PASSWORD",
    "PROXY_USER",
    "TARGET_BODY",
    "CertFiles",
    "ConnectAttempt",
    "ConnectProxy",
    "TlsTarget",
    "proxy_config",
    "server_ssl_context",
    "unattributable_peer_ports",
]

PROXY_USER = "testuser"
PROXY_PASSWORD = "testpass"

#: The name a sealed-network harness addresses its upstream by (§5.3).  It must
#: sit outside the ``localhost`` family or ``egress.should_bypass()`` sends the
#: request direct and the harness passes while proving the opposite of its
#: claim; RFC 2606 guarantees ``.invalid`` never resolves publicly, so the name
#: cannot escape the test environment.  Carried in the target certificate's SAN
#: so a client that reaches it through :attr:`ConnectProxy.resolve` can still
#: verify the hop.
HARNESS_UPSTREAM_HOST = "upstream.kitty-test.invalid"

#: Fixed target response body. Contains no surrounding whitespace so it
#: survives ``_probe``'s ``.strip()`` unchanged (AC3).
TARGET_BODY = "kitty-egress-tls-target"

#: The ``Proxy-Authorization`` value :class:`ConnectProxy` accepts. Exported
#: because botocore-shaped clients pass it as an explicit header rather than
#: deriving it from the URL's userinfo.
EXPECTED_PROXY_AUTH = "Basic " + base64.b64encode(f"{PROXY_USER}:{PROXY_PASSWORD}".encode()).decode()


# ── Certificates (session-scoped, files only) ────────────────────────────


@dataclasses.dataclass(frozen=True)
class CertFiles:
    """Paths of the throwaway certificate set used by the local servers.

    Attributes:
        ca: The throwaway CA certificate to load as client trust.
        proxy_cert: Certificate the TLS CONNECT proxy presents.
        proxy_key: Private key matching ``proxy_cert``.
        target_cert: Certificate the TLS target presents.
        target_key: Private key matching ``target_cert``.
    """

    ca: Path
    proxy_cert: Path
    proxy_key: Path
    target_cert: Path
    target_key: Path


def _run_openssl(*args: str) -> None:
    """Run one openssl command, failing the test loudly on any error.

    Args:
        *args: Arguments following the ``openssl`` executable name.

    Raises:
        pytest.fail: When openssl exits non-zero; the captured stderr is
            included so certificate-generation mistakes are readable (AC1).
    """
    completed = subprocess.run(["openssl", *args], capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(f"openssl {args[0]} failed (exit {completed.returncode}):\n{completed.stderr}")


def _generate_leaf(
    certs_dir: Path, name: str, ca_cert: Path, ca_key: Path, *, extra_sans: Sequence[str] = ()
) -> tuple[Path, Path]:
    """Generate one CA-signed leaf certificate for a local server.

    The extensions satisfy Python 3.13's OpenSSL strict-mode verification and
    the fact that both servers are addressed as ``127.0.0.1`` (AC1).

    Args:
        certs_dir: Directory receiving the key, CSR and certificate files.
        name: Filename stem distinguishing this server's files.
        ca_cert: The CA certificate that signs the leaf.
        ca_key: The CA private key that signs the leaf.
        extra_sans: Further ``subjectAltName`` entries, already qualified
            (``DNS:``/``IP:``). The target takes :data:`HARNESS_UPSTREAM_HOST`
            this way; the proxy is only ever addressed as ``127.0.0.1``, and a
            certificate should not claim names its server does not answer to.

    Returns:
        The (certificate, key) paths for the leaf.
    """
    sans = ",".join(["DNS:localhost", "IP:127.0.0.1", *extra_sans])
    key = certs_dir / f"{name}.key"
    csr = certs_dir / f"{name}.csr"
    pem = certs_dir / f"{name}.pem"
    _run_openssl(
        "req",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key),
        "-out",
        str(csr),
        "-nodes",
        "-subj",
        "/CN=localhost",
        "-addext",
        f"subjectAltName={sans}",
        "-addext",
        "keyUsage=critical,digitalSignature,keyEncipherment",
        "-addext",
        "extendedKeyUsage=serverAuth",
    )
    _run_openssl(
        "x509",
        "-req",
        "-in",
        str(csr),
        "-CA",
        str(ca_cert),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-out",
        str(pem),
        "-days",
        "2",
        "-copy_extensions",
        "copyall",
    )
    return pem, key


@pytest.fixture(scope="session")
def certs(tmp_path_factory: pytest.TempPathFactory) -> CertFiles:
    """Generate a throwaway CA plus proxy and target certificates (AC1).

    Args:
        tmp_path_factory: Pytest factory for a session-wide temp directory.

    Returns:
        The generated certificate and key paths.
    """
    certs_dir = tmp_path_factory.mktemp("egress-tls-certs")

    # The CA needs keyCertSign/keyUsage or Python 3.13 strict-mode verification
    # rejects it ("CA cert does not include key usage extension").
    ca_key = certs_dir / "ca.key"
    ca_cert = certs_dir / "ca.pem"
    _run_openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_cert),
        "-days",
        "2",
        "-nodes",
        "-subj",
        "/CN=kitty-egress-test-CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )

    proxy_cert, proxy_key = _generate_leaf(certs_dir, "proxy", ca_cert, ca_key)
    target_cert, target_key = _generate_leaf(
        certs_dir, "target", ca_cert, ca_key, extra_sans=[f"DNS:{HARNESS_UPSTREAM_HOST}"]
    )
    return CertFiles(
        ca=ca_cert, proxy_cert=proxy_cert, proxy_key=proxy_key, target_cert=target_cert, target_key=target_key
    )


def server_ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    """Build a TLS server context from one of the throwaway leaf certs.

    Args:
        cert: The certificate to present.
        key: The matching private key.

    Returns:
        The loaded server-side SSL context.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert), str(key))
    return context


# ── Local servers (function-scoped, one event loop each) ─────────────────


@dataclasses.dataclass
class ConnectAttempt:
    """One CONNECT request observed by the local proxy.

    Attributes:
        target: The ``host:port`` the client asked to tunnel to. Always that
            shape: a request with a method other than CONNECT is answered 405
            and recorded as nothing, so this never holds a request-target.
        authenticated: Whether the presented Basic credentials matched.
        source_port: Local port of the proxy's outbound socket for this tunnel,
            or ``None`` when no tunnel was opened -- a rejected authentication,
            or an upstream that refused the connection.

    ``source_port`` is §5.2.1's join key. Peer *address* cannot do that job:
    with proxy and upstream on loopback in one process, a tunnelled connection
    and a direct one both present ``127.0.0.1``. Nor can counting requests
    against CONNECTs: one tunnel may carry many requests and a rejected tunnel
    carries none, so the correlation is connection-to-tunnel and the port is
    what identifies a connection at both ends.
    """

    target: str
    authenticated: bool
    source_port: int | None = None


def unattributable_peer_ports(
    peer_ports: Iterable[int], attempts: Iterable[ConnectAttempt]
) -> list[int]:
    """Return the upstream connections no tunnel through the proxy explains.

    This is design §5.2.1's containment assertion, stated once so every
    transport slice inherits it rather than re-deriving it. An upstream
    connection with no matching tunnel port is a bypass, and it is the only
    thing the assertion needs to catch.

    Args:
        peer_ports: The source port of every **connection** the upstream
            accepted. Connections, not requests: one tunnel may carry many
            requests, so a request-level log must be reduced to its distinct
            connections before it is passed here.
        attempts: Every CONNECT the proxy observed, established or not.

    Returns:
        The peer ports with no matching established tunnel, in the order given
        and with duplicates kept -- two unexplained connections are two
        findings, not one.

    Raises:
        ValueError: When a peer port is ``None``. A recorder that never filled
            the field would otherwise make containment unfalsifiable in
            whichever direction the default happened to fall; the honest answer
            is that this run cannot be judged.

    A rejected attempt contributes no port, so ``None`` on an *attempt* is the
    absence of a key rather than a key matching anything.

    **The premise, which is a property of this harness's wiring and not a law.**
    A source port identifies a connection only because every leg here
    terminates on the same destination ``ip:port``, so the kernel will not hand
    the same port to a second connection while the first is in ``TIME_WAIT``.
    Point a future leg at a different destination and a direct connection may
    legitimately draw a live tunnel's source port and be attributed to it. The
    resulting error is a false negative -- a breach unreported -- never a false
    positive, so it degrades safety rather than stability. Re-derive this before
    adding a second upstream port.
    """
    # Materialise first: the body reads `peer_ports` twice, and the annotation
    # invites a generator -- a caller reducing a request log writes
    # `(r.peer_port for r in log)`. Exhausting it on the unset-port sweep would
    # report no bypass for a breached run.
    peer_ports = list(peer_ports)
    missing = [index for index, port in enumerate(peer_ports) if port is None]
    if missing:
        raise ValueError(f"peer port unset at position(s) {missing}: the connection log cannot be judged")

    tunnelled = {attempt.source_port for attempt in attempts if attempt.source_port is not None}

    return [port for port in peer_ports if port not in tunnelled]


#: How long :func:`_drain_server` will keep trying before it gives up and
#: fails the test. A hung teardown must be reported, never waited out.
_SHUTDOWN_DEADLINE = 5.0


async def _drain_server(server: asyncio.Server, writers: Iterable[asyncio.StreamWriter]) -> None:
    """Wait for an already-closed listener to let go of every connection.

    Args:
        server: The listener, **already closed by the caller**. Closing is the
            caller's job so that it happens before any ``await``; a listener
            still open across one accepts connections whose handlers are
            created after the teardown pass and never cancelled.
        writers: The connections this server knows about, aborted each round.

    Raises:
        TimeoutError: When the server has not finished closing within
            :data:`_SHUTDOWN_DEADLINE`.

    Python 3.12.1 changed ``asyncio.Server.wait_closed()`` to block until the
    server is closed **and** every connection is dropped; 3.10 and 3.11 return
    immediately. So a teardown that merely closes the listener hangs on the
    newer interpreters and passes on the older ones -- a harness that behaves
    differently on two contributors' machines.

    Connections are **aborted**, never closed: ``close()`` on a TLS transport
    begins a graceful shutdown bounded by ``ssl_shutdown_timeout``, 30 seconds
    by default, and a peer that never answers ``close_notify`` holds the wait
    for all of it. This module has been bitten by that timeout twice already.

    The retry loop is not belt-and-braces. A connection whose TLS handshake has
    completed on the client side may not yet have been dispatched to a handler,
    so it is attached to the server and absent from ``writers``; asyncio exposes
    no way to see it before 3.13's ``abort_clients()``. Re-aborting each round
    catches it the moment it appears.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SHUTDOWN_DEADLINE
    while True:
        for writer in list(writers):
            writer.transport.abort()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=0.05)
            return
        except (TimeoutError, asyncio.TimeoutError):
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"server still had connections attached {_SHUTDOWN_DEADLINE}s after close()"
                ) from None


async def _read_head(reader: asyncio.StreamReader) -> bytes | None:
    """Read a request head terminated by a blank line.

    Args:
        reader: The stream to read from.

    Returns:
        The head bytes without the terminating blank line, or ``None`` when
        the peer closed the connection before completing a head.
    """
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = await reader.read(4096)
        if not chunk:
            return None
        data += chunk
    return data.split(b"\r\n\r\n", 1)[0]


async def _pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
    """Copy bytes one way until EOF or a broken connection.

    Args:
        source: The side bytes are read from.
        sink: The side bytes are written to.
    """
    try:
        while True:
            chunk = await source.read(65536)
            if not chunk:
                break
            sink.write(chunk)
            await sink.drain()
    except (ConnectionError, OSError):
        pass


class ConnectProxy:
    """Minimal TLS CONNECT proxy enforcing Basic auth and recording attempts.

    Attributes:
        port: The 127.0.0.1 port the proxy listens on, valid once
            :meth:`start` has returned.
        attempts: Every CONNECT observed, including rejected ones (AC2).

    The proxy owns its listener rather than leaving it to the fixture, because
    design §5.2.2 phase 2 -- *proxy down, and the upstream must accept zero
    connections* -- has to take it down **mid-test**. That is the assertion
    separating real containment from a bridge that quietly falls back to a
    direct route when the gateway fails.
    """

    def __init__(self, resolve: Mapping[str, tuple[str, int]] | None = None) -> None:
        """Initialise an empty attempt record.

        Args:
            resolve: CONNECT targets this proxy resolves itself, as
                ``"host:port" -> (host, port)``. Empty by default, which leaves
                every target to the system resolver and is what the transport
                tests want.
        """
        self.port = 0
        self.attempts: list[ConnectAttempt] = []
        self.resolve = dict(resolve or {})
        self._server: asyncio.Server | None = None
        self._tunnels: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self, ssl_context: ssl.SSLContext) -> None:
        """Begin listening on an ephemeral 127.0.0.1 port.

        Args:
            ssl_context: The server-side context the proxy presents on the
                client hop.
        """
        self._server = await asyncio.start_server(self.handle, "127.0.0.1", 0, ssl=ssl_context)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop the proxy: refuse new connections and drop any tunnel still open.

        Idempotent, because a test that stops the proxy mid-run is still
        followed by the fixture's teardown.

        Raises:
            TimeoutError: When the listener will not finish closing; see
                :func:`_drain_server`, which carries the reasoning.
        """
        server = self._server
        if server is None:
            return

        # Closed before any `await`, so the listener cannot accept a connection
        # whose handler is created after the cancel pass below and so never
        # cancelled.
        server.close()
        # Abort BEFORE cancelling. Cancelling a stream handler makes asyncio's
        # own done-callback call `transport.close()` a *second* time, and the
        # second call sets `_ssl_protocol = None`, after which every `abort()`
        # is a silent no-op and the listener stays attached for the full
        # `ssl_shutdown_timeout`. Aborting first gets in ahead of that.
        for writer in list(self._writers):
            writer.transport.abort()
        # Then cancel: a handler blocked in `open_connection` to an upstream
        # that never answers would not notice its client socket being aborted.
        for tunnel in list(self._tunnels):
            tunnel.cancel()
        await asyncio.gather(*self._tunnels, return_exceptions=True)
        await _drain_server(server, self._writers)
        # Cleared last: a `stop()` that raised has not stopped anything, and
        # the fixture's teardown call must chase the connection again rather
        # than return quietly and swallow the diagnostic.
        self._server = None
        self._writers.clear()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one CONNECT exchange: record it, check auth, then pipe.

        Args:
            reader: The client side of the accepted (already TLS) connection.
            writer: The client side of the accepted (already TLS) connection.
        """
        # Registered so `stop()` can drop a tunnel still in flight; asyncio
        # creates one task per accepted connection and exposes no other handle.
        tunnel = asyncio.current_task()
        if tunnel is not None:
            self._tunnels.add(tunnel)

        self._writers.add(writer)
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            head = await _read_head(reader)
            if head is None:
                return
            lines = head.decode("latin-1").split("\r\n")
            method, target, _version = lines[0].split(" ", 2)
            headers = {}
            for line in lines[1:]:
                key, sep, value = line.partition(":")
                if sep:
                    headers[key.strip().lower()] = value.strip()

            if method != "CONNECT":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                await writer.drain()
                return

            authenticated = headers.get("proxy-authorization") == EXPECTED_PROXY_AUTH
            attempt = ConnectAttempt(target=target, authenticated=authenticated)
            self.attempts.append(attempt)
            if not authenticated:
                writer.write(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")
                await writer.drain()
                return

            host, port_text = target.rsplit(":", 1)
            # The harness owns the resolver because the harness is the proxy
            # (§5.3): a client tunnelling to a `.invalid` name never resolves
            # it itself, so this is the only place the name can be mapped.
            address = self.resolve.get(target, (host, int(port_text)))
            try:
                upstream_reader, upstream_writer = await asyncio.open_connection(*address)
            except OSError:
                # Distinguishable from "no tunnel attempted": the attempt is on
                # the record with no source port, and the client gets a status
                # rather than a silent close.
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await writer.drain()
                return
            self._writers.add(upstream_writer)
            # Filled only now, because only now does a tunnel exist: the
            # attempt is recorded before the outbound connection so that a
            # rejected CONNECT is still on the record (§5.2.1).
            attempt.source_port = upstream_writer.get_extra_info("sockname")[1]
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            # Close the tunnel as soon as either direction ends. Waiting for
            # both deadlocks when one side closes without the other noticing,
            # stalling teardown until the TLS close_notify timeout.
            to_upstream = asyncio.create_task(_pipe(reader, upstream_writer))
            to_client = asyncio.create_task(_pipe(upstream_reader, writer))
            _done, pending = await asyncio.wait({to_upstream, to_client}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            if tunnel is not None:
                self._tunnels.discard(tunnel)
            # The writers stay registered until `stop()` clears them. A handler
            # that de-registered here and then called `close()` would leave the
            # connection holding the listener open -- a graceful TLS shutdown
            # against a peer that never answers -- with nothing left for
            # `stop()` to abort. Aborting a transport twice is harmless; losing
            # track of one is not.
            if upstream_writer is not None:
                upstream_writer.close()
            writer.close()


class TlsTarget:
    """A local TLS server answering any request with :data:`TARGET_BODY`.

    Attributes:
        port: The 127.0.0.1 port the target listens on, valid once
            :meth:`start` has returned.
        completed: How many connections the handler has finished serving. A
            test that needs to act *after* a handler has run its teardown has
            no other observable: the response arriving proves only that the
            body was written.

    It owns its listener for the same reason :class:`ConnectProxy` does. A
    fixture that closed the listener and awaited ``wait_closed()`` would hang on
    Python 3.12.1+ whenever a connection was still attached -- and §5.2.2 phase
    2 stops the proxy mid-exchange, which is precisely the state that leaves one
    attached at the target.
    """

    def __init__(self) -> None:
        """Initialise a target that is not yet listening."""
        self.port = 0
        self.completed = 0
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self, ssl_context: ssl.SSLContext) -> None:
        """Begin listening on an ephemeral 127.0.0.1 port.

        Args:
            ssl_context: The server-side context the target presents.
        """
        self._server = await asyncio.start_server(self.handle, "127.0.0.1", 0, ssl=ssl_context)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop the target, aborting any connection still attached.

        Idempotent, for the same reason :meth:`ConnectProxy.stop` is.

        Raises:
            TimeoutError: When the listener will not finish closing; see
                :func:`_drain_server`.
        """
        server = self._server
        if server is None:
            return

        # No pre-abort pass here, unlike :meth:`ConnectProxy.stop`: that one
        # exists only to get ahead of the cancel that disarms `abort()`, and
        # this class cancels nothing. `_drain_server` does the aborting.
        server.close()
        await _drain_server(server, self._writers)
        self._server = None
        self._writers.clear()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answer any single request with the fixed target body.

        Args:
            reader: The client side of the accepted (already TLS) connection.
            writer: The client side of the accepted (already TLS) connection.
        """
        self._writers.add(writer)
        try:
            if await _read_head(reader) is None:
                return
            body = TARGET_BODY.encode()
            # Connection: close makes pooling clients (urllib3) close the tunnel
            # themselves; without it the target's graceful TLS shutdown would
            # wait out a 30-second close_notify timeout during teardown.
            head = (
                f"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            )
            writer.write(head.encode() + body)
            await writer.drain()
        finally:
            # Kept registered for `stop()` to abort; see `ConnectProxy.handle`.
            self.completed += 1
            writer.close()


@pytest.fixture
async def connect_proxy(certs: CertFiles) -> AsyncIterator[ConnectProxy]:
    """Run a local TLS CONNECT proxy for one test (AC2).

    Args:
        certs: The session's throwaway certificates.

    Yields:
        The running proxy, with its port and attempt record.
    """
    proxy = ConnectProxy()
    await proxy.start(server_ssl_context(certs.proxy_cert, certs.proxy_key))
    try:
        yield proxy
    finally:
        await proxy.stop()


@pytest.fixture
async def tls_target(certs: CertFiles) -> AsyncIterator[TlsTarget]:
    """Run a local TLS target serving the fixed body for one test.

    Args:
        certs: The session's throwaway certificates.

    Yields:
        The running target's address.
    """
    target = TlsTarget()
    await target.start(server_ssl_context(certs.target_cert, certs.target_key))
    try:
        yield target
    finally:
        await target.stop()


# ── Test-side seams ──────────────────────────────────────────────────────


@pytest.fixture
def aiohttp_trusts_test_ca(monkeypatch: pytest.MonkeyPatch, certs: CertFiles) -> None:
    """Swap aiohttp's import-time cached verified SSL context (AC4).

    aiohttp builds ``_SSL_CONTEXT_VERIFIED`` at import time and returns it for
    both the proxy hop and the tunneled target hop, so wrapping
    ``ssl.create_default_context`` at test time would be a no-op in a full
    suite run. Replacing the cached object is the order-independent seam.

    Args:
        monkeypatch: Pytest's monkeypatch fixture (auto-restores the cache).
        certs: The session's throwaway certificates.
    """
    context = ssl.create_default_context()
    context.load_verify_locations(str(certs.ca))
    monkeypatch.setattr(aiohttp.connector, "_SSL_CONTEXT_VERIFIED", context)


def proxy_config(port: int, password: str = PROXY_PASSWORD) -> EgressConfig:
    """Build the https-proxy configuration the tests drive.

    Args:
        port: The local proxy port.
        password: Proxy password; pass a wrong one for the negative test.

    Returns:
        The configuration under test.
    """
    return EgressConfig(proxy_url=f"https://127.0.0.1:{port}", username=PROXY_USER, password=password)
