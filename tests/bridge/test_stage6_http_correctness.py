"""Stage 6 tests — HTTP correctness fixes (F29, F30, F32, F34).

These tests target:
- F29: aiohttp connection pool configuration
- F30: _EMPTY_FINAL_DELAYS coupling
- F32: /healthz returns 503 when degraded
- F34: malformed JSON body errors are logged
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import aiohttp
import pytest

from kitty.bridge.server import (
    _EMPTY_FINAL_DELAYS,
    BridgeServer,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stubs ────────────────────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API):
        self._protocol = protocol

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return self._protocol

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    def __init__(self, base_url: str = "https://api.example.com/v1"):
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"err {status_code}")


# ── F29: Connection pool size configuration ──────────────────────────────


class TestConnectionPoolConfiguration:
    """F29: aiohttp session uses a configured TCPConnector, not default (100)."""

    @pytest.mark.asyncio
    async def test_session_uses_tcp_connector_with_explicit_limit(self):
        """The aiohttp session must use a TCPConnector with an explicit limit."""
        server = BridgeServer(_StubLauncher(), _StubProvider(), "key")
        session = await server._get_session()
        # Find the connector attached to the session
        connector = session.connector
        assert connector is not None, "Session must have a connector"
        # The connector's limit should be set to something explicit (not 100 default)
        # We accept any limit >= 100, but require it to be set (not None).
        assert connector.limit is not None, "TCPConnector must have an explicit limit set"
        await session.close()

    @pytest.mark.asyncio
    async def test_session_connector_is_force_close_or_keepalive_safe(self):
        """The connector configuration should prevent port exhaustion under load."""
        server = BridgeServer(_StubLauncher(), _StubProvider(), "key")
        session = await server._get_session()
        connector = session.connector
        # Either force_close is True OR keepalive_timeout is bounded (prevents leaks)
        assert connector.force_close is True or connector.keepalive_timeout is not None
        await session.close()

    @pytest.mark.asyncio
    async def test_session_limit_is_above_aiohttp_default(self):
        """The configured limit should be higher than aiohttp's 100 default
        to support load without queueing.  Accept any limit >= 200 to allow
        for environment-specific tuning."""
        server = BridgeServer(_StubLauncher(), _StubProvider(), "key")
        session = await server._get_session()
        connector = session.connector
        assert connector.limit >= 200, (
            f"Connection pool limit too low ({connector.limit}); should be >= 200 to avoid request queueing under load"
        )
        await session.close()


# ── F30: _EMPTY_FINAL_DELAYS length coupling ────────────────────────────


class TestEmptyFinalDelaysCoupling:
    """F30: max_attempts is computed via len(_EMPTY_FINAL_DELAYS), not a hardcoded +2."""

    def test_empty_final_delays_is_a_list(self):
        """_EMPTY_FINAL_DELAYS must be a list (not a tuple or hardcoded number)."""
        assert isinstance(_EMPTY_FINAL_DELAYS, list)
        assert len(_EMPTY_FINAL_DELAYS) >= 1

    def test_empty_final_delays_values_are_positive(self):
        """Each delay value must be a positive number."""
        for v in _EMPTY_FINAL_DELAYS:
            assert isinstance(v, (int, float))
            assert v > 0


# ── F32: /healthz returns 503 when degraded ──────────────────────────────


class TestHealthzDegradedStatusCode:
    """F32: /healthz returns HTTP 503 when all backends are unhealthy."""

    def _make_backends(self, n: int = 2):
        import uuid

        from kitty.profiles.schema import Profile

        backends = []
        for i in range(n):
            provider = _StubProvider(base_url=f"https://api{i}.example.com/v1")
            profile = Profile(
                name=f"profile-{i}",
                provider="openai",
                model=f"model-{i}",
                auth_ref=str(uuid.uuid4()),
            )
            backends.append((provider, f"key-{i}", profile))
        return backends

    @pytest.mark.asyncio
    async def test_healthz_returns_503_when_all_unhealthy(self):
        """All backends unhealthy → HTTP 503, not 200."""
        backends = self._make_backends(2)
        server = BridgeServer(
            adapter=_StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="model-0",
            backends=backends,
        )
        # Mark all backends unhealthy with far-future cooldown
        for idx in range(2):
            server._backend_health[idx]["healthy"] = False
            server._backend_health[idx]["failed_at"] = 1000.0
            server._backend_health[idx]["cooldown"] = 300
        with patch("kitty.bridge.server.time.monotonic", return_value=1010.0):
            response = await server._handle_healthz(None)
        # The response should be a 503 JSON response
        assert response.status == 503, f"Expected 503 when all backends unhealthy, got {response.status}"

    @pytest.mark.asyncio
    async def test_healthz_returns_200_when_any_healthy(self):
        """At least one healthy backend → HTTP 200."""
        backends = self._make_backends(2)
        server = BridgeServer(
            adapter=_StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="model-0",
            backends=backends,
        )
        # Mark one unhealthy, one healthy
        server._backend_health[0]["healthy"] = False
        server._backend_health[0]["failed_at"] = 1000.0
        server._backend_health[0]["cooldown"] = 300
        server._backend_health[1]["healthy"] = True
        with patch("kitty.bridge.server.time.monotonic", return_value=1010.0):
            response = await server._handle_healthz(None)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_healthz_body_indicates_degraded_status(self):
        """The 503 response body should contain 'degraded' status."""
        backends = self._make_backends(2)
        server = BridgeServer(
            adapter=_StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="model-0",
            backends=backends,
        )
        for idx in range(2):
            server._backend_health[idx]["healthy"] = False
            server._backend_health[idx]["failed_at"] = 1000.0
            server._backend_health[idx]["cooldown"] = 300
        with patch("kitty.bridge.server.time.monotonic", return_value=1010.0):
            response = await server._handle_healthz(None)
        assert response.status == 503
        body = json.loads(response.text)
        assert body["status"] == "degraded"


# ── F34: Malformed JSON in request body is logged ────────────────────────


class TestMalformedJsonLogging:
    """F34: When a request body has malformed JSON, a warning must be logged.

    Currently the 4 handlers catch ``(json.JSONDecodeError, Exception)``
    and return a 400 without logging.  This makes it hard to distinguish
    malformed-JSON errors from oversize errors in production logs.
    """

    @pytest.mark.asyncio
    async def test_responses_handler_logs_malformed_json(self, caplog):
        """Malformed JSON to /v1/responses must produce a WARNING log."""

        # Use bridge mode (no adapter) so /v1/responses is registered
        server = BridgeServer(adapter=None, provider=_StubProvider(), resolved_key="k", model="m")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        data=b"this is not json {",
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        assert resp.status == 400
            warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any("malformed" in m.lower() or "json" in m.lower() for m in warning_msgs), (
                f"Expected malformed-JSON warning, got: {warning_msgs}"
            )
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_handler_logs_malformed_json(self, caplog):
        """Malformed JSON to /v1/messages must produce a WARNING log."""

        server = BridgeServer(adapter=None, provider=_StubProvider(), resolved_key="k", model="m")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=b"not json {",
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        assert resp.status == 400
            warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any("malformed" in m.lower() or "json" in m.lower() for m in warning_msgs)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_gemini_handler_logs_malformed_json(self, caplog):
        """Malformed JSON to /v1beta/models/...:generateContent must log."""

        server = BridgeServer(adapter=None, provider=_StubProvider(), resolved_key="k", model="m")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1beta/models/foo:generateContent",
                        data=b"not json",
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        assert resp.status == 400
            warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any("malformed" in m.lower() or "json" in m.lower() for m in warning_msgs)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_handler_logs_malformed_json(self, caplog):
        """Malformed JSON to /v1/chat/completions must log."""

        server = BridgeServer(adapter=None, provider=_StubProvider(), resolved_key="k", model="m")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        data=b"not json",
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        assert resp.status == 400
            warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any("malformed" in m.lower() or "json" in m.lower() for m in warning_msgs)
        finally:
            await server.stop_async()
