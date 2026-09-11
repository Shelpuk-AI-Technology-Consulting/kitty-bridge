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

The servers themselves now live in :mod:`harness.connect_proxy` (plan task
T-W5), because the containment harness and the dependency proxy contracts need
the same proxy. This module keeps the tests and the one seam that is specific to
``kitty egress test`` rather than to the proxy — :func:`target_url`.

See ``.requirements/20260820T122157Z_egress_https_proxy_tests/REQUIREMENTS.md``
for the full specification (R1–R8, AC1–AC10).
"""

from __future__ import annotations

import asyncio
import ssl
import sys

import curl_cffi.requests
import pytest
import urllib3
from curl_cffi import curl as curl_cffi_curl
from harness.connect_proxy import (
    EXPECTED_PROXY_AUTH,
    TARGET_BODY,
    CertFiles,
    ConnectProxy,
    TlsTarget,
    proxy_config,
)

from kitty.cli import egress_cmd

_AIOHTTP_NEEDS_311 = sys.version_info < (3, 11)
_AIOHTTP_SKIP_REASON = "aiohttp requires Python 3.11 for TLS-in-TLS over stdlib asyncio (bpo-44011)"


@pytest.fixture
def target_url(monkeypatch: pytest.MonkeyPatch, tls_target: TlsTarget) -> str:
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


# ── aiohttp: the real `_probe` end to end (Python ≥3.11 only) ────────────


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
class TestConcurrentSessionsThroughOneProxy:
    """Several kitty agents share one authenticated egress gateway.

    Each ``kitty`` process uses ``_probe`` (via ``kitty egress test``) and
    makes upstream calls through the resolved gateway. The egress proxy is the
    shared part across processes — this pins that the transport survives N
    concurrent sessions tunnelling through one listener.
    """

    async def test_concurrent_probes_all_succeed_through_shared_proxy(
        self,
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        target_url: str,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        results = await asyncio.gather(
            *(egress_cmd._probe(proxy_config(connect_proxy.port)) for _ in range(8))
        )

        for body, _elapsed_ms, error in results:
            assert error is None
            assert body == TARGET_BODY
        assert len(connect_proxy.attempts) == 8
        assert all(a.authenticated and a.target == f"127.0.0.1:{tls_target.port}" for a in connect_proxy.attempts)


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
class TestAiohttpProbeThroughHttpsProxy:
    """``kitty egress test``'s probe carries an authenticated https:// proxy."""

    async def test_probe_succeeds_through_authenticated_tls_proxy(
        self,
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        target_url: str,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Correct credentials tunnel to the target and return its body (AC3)."""
        body, _elapsed_ms, error = await egress_cmd._probe(proxy_config(connect_proxy.port))

        assert error is None
        assert body == TARGET_BODY
        assert len(connect_proxy.attempts) == 1
        assert connect_proxy.attempts[0].authenticated
        assert connect_proxy.attempts[0].target == f"127.0.0.1:{tls_target.port}"

    async def test_probe_reports_407_on_wrong_password(
        self,
        connect_proxy: ConnectProxy,
        target_url: str,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Bad credentials surface as the proxy's 407, not TLS noise (AC5)."""
        body, _elapsed_ms, error = await egress_cmd._probe(proxy_config(connect_proxy.port, password="wrong-password"))

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
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        certs: CertFiles,
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

        config = proxy_config(connect_proxy.port)

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
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        certs: CertFiles,
    ) -> None:
        """A botocore-shaped ProxyManager tunnels to the target (AC7)."""
        config = proxy_config(connect_proxy.port)
        proxy_context = ssl.create_default_context()
        proxy_context.load_verify_locations(str(certs.ca))

        # Shaped as botocore.httpsession._get_proxy_manager builds one: URL
        # with credentials plus an explicit Proxy-Authorization header —
        # urllib3 itself never converts URL userinfo into that header.
        manager = urllib3.proxy_from_url(
            config.url_with_credentials(),
            proxy_headers={"Proxy-Authorization": EXPECTED_PROXY_AUTH},
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
