"""Tests for R6: TLS support."""

from __future__ import annotations

from pathlib import Path

import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.zai import ZaiRegularAdapter


def _make_server(**kwargs):
    return BridgeServer(
        adapter=None,
        provider=ZaiRegularAdapter(),
        resolved_key="test-key",
        model="gpt-4o",
        provider_config={},
        **kwargs,
    )


def _generate_self_signed_cert(tmp_path: Path) -> tuple[Path, Path]:
    """Generate a self-signed cert+key for testing using openssl."""
    import subprocess

    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert_path, key_path


class TestTLSSupport:
    """TLS configuration for the bridge server."""

    @pytest.mark.asyncio
    async def test_tls_cert_and_key_enables_https(self, tmp_path: Path):
        """Providing both cert and key enables HTTPS."""
        cert, key = _generate_self_signed_cert(tmp_path)
        server = _make_server(tls_cert=str(cert), tls_key=str(key))
        port = await server.start_async()
        try:
            import aiohttp

            # Connect without TLS verification (self-signed)
            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=conn) as session, session.get(
                f"https://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_only_cert_raises_error(self, tmp_path: Path):
        """Providing only tls_cert without tls_key raises an error."""
        cert = tmp_path / "cert.pem"
        cert.write_text("dummy")
        with pytest.raises(ValueError, match="tls_key"):
            _make_server(tls_cert=str(cert), tls_key=None)

    @pytest.mark.asyncio
    async def test_only_key_raises_error(self, tmp_path: Path):
        """Providing only tls_key without tls_cert raises an error."""
        key = tmp_path / "key.pem"
        key.write_text("dummy")
        with pytest.raises(ValueError, match="tls_cert"):
            _make_server(tls_cert=None, tls_key=str(key))

    @pytest.mark.asyncio
    async def test_no_tls_works_as_http(self):
        """Without TLS, the server serves plain HTTP."""
        server = _make_server()
        port = await server.start_async()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()


class TestTLSWarning:
    """Warning on non-localhost binding without TLS."""

    def test_non_localhost_without_tls_warns(self, capsys):
        """Binding to 0.0.0.0 without TLS should print a warning."""
        server = _make_server(host="0.0.0.0")
        # Check that _check_tls_warning is called during start_async
        # We test the method directly for simplicity
        assert server._should_warn_no_tls() is True

    def test_localhost_without_tls_no_warn(self):
        server = _make_server(host="127.0.0.1")
        assert server._should_warn_no_tls() is False

    def test_non_localhost_with_tls_no_warn(self, tmp_path: Path):
        cert, key = _generate_self_signed_cert(tmp_path)
        server = _make_server(host="0.0.0.0", tls_cert=str(cert), tls_key=str(key))
        assert server._should_warn_no_tls() is False
