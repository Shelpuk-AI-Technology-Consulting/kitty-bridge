"""Hermetic transport tests proving an ``https://`` egress proxy carries traffic.

Every other egress test in the suite is mocked at the socket layer. This module
performs real TLS handshakes instead: a local TLS CONNECT proxy (enforcing Basic
auth, recording every CONNECT it sees) and a local TLS target, both on ephemeral
127.0.0.1 ports with throwaway certificates, no internet access.

What it proves, per transport stack:

* aiohttp — drives the real :func:`kitty.cli.egress_cmd._probe` (the function
  behind ``kitty egress test``) with an ``https://`` proxy and asserts the
  target's body comes back through the tunnel (TLS-in-TLS). Skipped below
  Python 3.11, where aiohttp treats stdlib asyncio as TLS-in-TLS-incapable.
* curl_cffi — the ``proxies=`` mapping exactly as
  :mod:`kitty.providers.openai_subscription` passes it. This path emits
  curl_cffi's unconditional "https over https proxy" advisory warning for an
  https target behind an https proxy — the exact configuration under test;
  the warning is expected and must not be "fixed" away.
* urllib3 — a proxy manager shaped the way botocore builds one in
  ``botocore.httpsession._get_proxy_manager`` (URL with credentials plus an
  explicit ``Proxy-Authorization`` header: urllib3 itself never converts URL
  userinfo into that header).

See ``.requirements/20260820T122157Z_egress_https_proxy_tests/REQUIREMENTS.md``
for the full specification (R1–R8, AC1–AC10).
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import ssl
import subprocess
import sys
from pathlib import Path

import aiohttp.connector
import curl_cffi.requests
import pytest
import urllib3
from curl_cffi import curl as curl_cffi_curl

from kitty.cli import egress_cmd
from kitty.egress import EgressConfig

PROXY_USER = "testuser"
PROXY_PASSWORD = "testpass"

#: Fixed target response body. Contains no surrounding whitespace so it
#: survives ``_probe``'s ``.strip()`` unchanged (AC3).
TARGET_BODY = "kitty-egress-tls-target"

_EXPECTED_PROXY_AUTH = "Basic " + base64.b64encode(f"{PROXY_USER}:{PROXY_PASSWORD}".encode()).decode()

_AIOHTTP_NEEDS_311 = sys.version_info < (3, 11)
_AIOHTTP_SKIP_REASON = "aiohttp requires Python 3.11 for TLS-in-TLS over stdlib asyncio (bpo-44011)"


# ── Certificates (session-scoped, files only) ────────────────────────────


@dataclasses.dataclass(frozen=True)
class _CertFiles:
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


def _generate_leaf(certs_dir: Path, name: str, ca_cert: Path, ca_key: Path) -> tuple[Path, Path]:
    """Generate one CA-signed leaf certificate for a local server.

    The extensions satisfy Python 3.13's OpenSSL strict-mode verification and
    the fact that both servers are addressed as ``127.0.0.1`` (AC1).

    Args:
        certs_dir: Directory receiving the key, CSR and certificate files.
        name: Filename stem distinguishing this server's files.
        ca_cert: The CA certificate that signs the leaf.
        ca_key: The CA private key that signs the leaf.

    Returns:
        The (certificate, key) paths for the leaf.
    """
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
        "subjectAltName=DNS:localhost,IP:127.0.0.1",
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
def certs(tmp_path_factory: pytest.TempPathFactory) -> _CertFiles:
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
    target_cert, target_key = _generate_leaf(certs_dir, "target", ca_cert, ca_key)
    return _CertFiles(
        ca=ca_cert, proxy_cert=proxy_cert, proxy_key=proxy_key, target_cert=target_cert, target_key=target_key
    )


# ── Local servers (function-scoped, one event loop each) ─────────────────


@dataclasses.dataclass
class ConnectAttempt:
    """One CONNECT request observed by the local proxy.

    Attributes:
        target: The ``host:port`` the client asked to tunnel to.
        authenticated: Whether the presented Basic credentials matched.
    """

    target: str
    authenticated: bool


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


class _ConnectProxy:
    """Minimal TLS CONNECT proxy enforcing Basic auth and recording attempts.

    Attributes:
        port: The 127.0.0.1 port the proxy listens on (set by the fixture).
        attempts: Every CONNECT observed, including rejected ones (AC2).
    """

    def __init__(self) -> None:
        """Initialise an empty attempt record."""
        self.port = 0
        self.attempts: list[ConnectAttempt] = []

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one CONNECT exchange: record it, check auth, then pipe.

        Args:
            reader: The client side of the accepted (already TLS) connection.
            writer: The client side of the accepted (already TLS) connection.
        """
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

            authenticated = headers.get("proxy-authorization") == _EXPECTED_PROXY_AUTH
            self.attempts.append(ConnectAttempt(target=target, authenticated=authenticated))
            if not authenticated:
                writer.write(b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n")
                await writer.drain()
                return

            host, port_text = target.rsplit(":", 1)
            upstream_reader, upstream_writer = await asyncio.open_connection(host, int(port_text))
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
            if upstream_writer is not None:
                upstream_writer.close()
            writer.close()


@dataclasses.dataclass(frozen=True)
class _TlsTarget:
    """The local TLS target's address.

    Attributes:
        port: The 127.0.0.1 port the target listens on.
    """

    port: int


async def _handle_target(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer any single request with the fixed target body.

    Args:
        reader: The client side of the accepted (already TLS) connection.
        writer: The client side of the accepted (already TLS) connection.
    """
    try:
        if await _read_head(reader) is None:
            return
        body = TARGET_BODY.encode()
        # Connection: close makes pooling clients (urllib3) close the tunnel
        # themselves; without it the target's graceful TLS shutdown would wait
        # out a 30-second close_notify timeout during teardown.
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        )
        writer.write(head.encode() + body)
        await writer.drain()
    finally:
        writer.close()


def _server_ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
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


@pytest.fixture
async def connect_proxy(certs: _CertFiles) -> _ConnectProxy:
    """Run a local TLS CONNECT proxy for one test (AC2).

    Args:
        certs: The session's throwaway certificates.

    Yields:
        The running proxy, with its port and attempt record.
    """
    proxy = _ConnectProxy()
    server = await asyncio.start_server(
        proxy.handle, "127.0.0.1", 0, ssl=_server_ssl_context(certs.proxy_cert, certs.proxy_key)
    )
    proxy.port = server.sockets[0].getsockname()[1]
    try:
        yield proxy
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def tls_target(certs: _CertFiles) -> _TlsTarget:
    """Run a local TLS target serving the fixed body for one test.

    Args:
        certs: The session's throwaway certificates.

    Yields:
        The running target's address.
    """
    server = await asyncio.start_server(
        _handle_target, "127.0.0.1", 0, ssl=_server_ssl_context(certs.target_cert, certs.target_key)
    )
    try:
        yield _TlsTarget(port=server.sockets[0].getsockname()[1])
    finally:
        server.close()
        await server.wait_closed()


# ── Test-side seams ──────────────────────────────────────────────────────


@pytest.fixture
def target_url(monkeypatch: pytest.MonkeyPatch, tls_target: _TlsTarget) -> str:
    """Point ``_probe``'s echo URL at the local TLS target (AC4).

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
        tls_target: The running local target.

    Returns:
        The target URL ``_probe`` will fetch.
    """
    url = f"https://127.0.0.1:{tls_target.port}"
    monkeypatch.setattr(egress_cmd, "IP_ECHO_URL", url)
    return url


@pytest.fixture
def aiohttp_trusts_test_ca(monkeypatch: pytest.MonkeyPatch, certs: _CertFiles) -> None:
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


def _proxy_config(port: int, password: str = PROXY_PASSWORD) -> EgressConfig:
    """Build the https-proxy configuration the tests drive.

    Args:
        port: The local proxy port.
        password: Proxy password; pass a wrong one for the negative test.

    Returns:
        The configuration under test.
    """
    return EgressConfig(proxy_url=f"https://127.0.0.1:{port}", username=PROXY_USER, password=password)


# ── aiohttp: the real `_probe` end to end (Python ≥3.11 only) ────────────


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
class TestAiohttpProbeThroughHttpsProxy:
    """``kitty egress test``'s probe carries an authenticated https:// proxy."""

    async def test_probe_succeeds_through_authenticated_tls_proxy(
        self,
        connect_proxy: _ConnectProxy,
        tls_target: _TlsTarget,
        target_url: str,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Correct credentials tunnel to the target and return its body (AC3)."""
        body, _elapsed_ms, error = await egress_cmd._probe(_proxy_config(connect_proxy.port))

        assert error is None
        assert body == TARGET_BODY
        assert len(connect_proxy.attempts) == 1
        assert connect_proxy.attempts[0].authenticated
        assert connect_proxy.attempts[0].target == f"127.0.0.1:{tls_target.port}"

    async def test_probe_reports_407_on_wrong_password(
        self,
        connect_proxy: _ConnectProxy,
        target_url: str,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Bad credentials surface as the proxy's 407, not TLS noise (AC5)."""
        body, _elapsed_ms, error = await egress_cmd._probe(_proxy_config(connect_proxy.port, password="wrong-password"))

        assert body is None
        assert error is not None
        assert "407" in error
        assert len(connect_proxy.attempts) == 1
        assert not connect_proxy.attempts[0].authenticated


# ── curl_cffi: the provider's proxies= form (all supported Pythons) ──────


class TestCurlCffiThroughHttpsProxy:
    """curl_cffi carries the provider's ``proxies=`` mapping over a TLS proxy."""

    async def test_get_succeeds_through_authenticated_tls_proxy(
        self,
        connect_proxy: _ConnectProxy,
        tls_target: _TlsTarget,
        certs: _CertFiles,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``proxies=proxies_dict()`` tunnels to the target (AC6)."""
        # verify=True is silently converted to a path if these are set, and on
        # curl_cffi 0.15.x (the uv.lock pin) a path covers only the target hop.
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
        # Point curl_cffi's default bundle at the test CA so _ensure_cacert
        # applies it to both CAINFO (target hop) and PROXY_CAINFO (proxy hop).
        # Works for the sync path only: each requests.get() builds a fresh Curl
        # that reads the module global. A switch to AsyncSession would also need
        # curl_cffi.aio.DEFAULT_CACERT patched (it is bound at aio.py import).
        monkeypatch.setattr(curl_cffi_curl, "DEFAULT_CACERT", str(certs.ca))

        config = _proxy_config(connect_proxy.port)

        def request() -> curl_cffi.requests.Response:
            """Perform the blocking GET off the event loop."""
            return curl_cffi.requests.get(
                f"https://127.0.0.1:{tls_target.port}",
                proxies=config.proxies_dict(),
                impersonate="chrome136",
                timeout=15,
            )

        response = await asyncio.to_thread(request)

        assert response.status_code == 200
        assert response.text == TARGET_BODY
        assert len(connect_proxy.attempts) == 1
        assert connect_proxy.attempts[0].authenticated


# ── urllib3: the botocore-shaped proxy manager (all supported Pythons) ───


class TestUrllib3ThroughHttpsProxy:
    """urllib3 carries botocore-shaped proxy configuration over a TLS proxy."""

    async def test_get_succeeds_through_authenticated_tls_proxy(
        self,
        connect_proxy: _ConnectProxy,
        tls_target: _TlsTarget,
        certs: _CertFiles,
    ) -> None:
        """A botocore-shaped ProxyManager tunnels to the target (AC7)."""
        config = _proxy_config(connect_proxy.port)
        proxy_context = ssl.create_default_context()
        proxy_context.load_verify_locations(str(certs.ca))

        # Shaped as botocore.httpsession._get_proxy_manager builds one: URL
        # with credentials plus an explicit Proxy-Authorization header —
        # urllib3 itself never converts URL userinfo into that header.
        manager = urllib3.proxy_from_url(
            config.url_with_credentials(),
            proxy_headers={"Proxy-Authorization": _EXPECTED_PROXY_AUTH},
            proxy_ssl_context=proxy_context,
            ca_certs=str(certs.ca),
        )

        def request() -> urllib3.BaseHTTPResponse:
            """Perform the blocking GET off the event loop."""
            return manager.request("GET", f"https://127.0.0.1:{tls_target.port}/", timeout=15.0)

        response = await asyncio.to_thread(request)

        assert response.status == 200
        assert response.data.decode() == TARGET_BODY
        assert len(connect_proxy.attempts) == 1
        assert connect_proxy.attempts[0].authenticated
