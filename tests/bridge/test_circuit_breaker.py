"""Tests for balancing profile circuit breaker — backend health tracking and retry."""

from __future__ import annotations

import time
import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# -- Helpers -----------------------------------------------------------------


class StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.CHAT_COMPLETIONS_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    def __init__(self, provider_type: str = "stub", base_url: str = "https://api.example.com/v1"):
        self._provider_type = provider_type
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        return self._provider_type

    @property
    def default_base_url(self) -> str:
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Error {status_code}")


UPSTREAM_OK = {
    "id": "chatcmpl-1",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_backends(n: int) -> list:
    backends = []
    for i in range(n):
        provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(name=f"profile-{i}", provider="openai", model=f"model-{i}", auth_ref=str(uuid.uuid4()))
        backends.append((provider, key, profile))
    return backends


def _make_server(n_backends: int = 3, cooldown: int = 300) -> BridgeServer:
    backends = _make_backends(n_backends)
    server = BridgeServer(
        adapter=StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=cooldown,
    )
    return server


# -- Step 1: Backend health tracking -----------------------------------------


class TestBackendHealthTracking:
    """Backend health data structure and _mark_backend_unhealthy."""

    def test_new_backend_is_healthy(self):
        server = _make_server(3)
        assert len(server._backend_health) == 3
        for h in server._backend_health:
            assert h["healthy"] is True
            assert h["failed_at"] is None

    def test_mark_backend_unhealthy(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(1)
        assert server._backend_health[0]["healthy"] is True
        assert server._backend_health[1]["healthy"] is False
        assert server._backend_health[1]["failed_at"] is not None
        assert server._backend_health[2]["healthy"] is True

    def test_mark_backend_records_monotonic_time(self):
        server = _make_server(3)
        before = time.monotonic()
        server._mark_backend_unhealthy(0)
        after = time.monotonic()
        assert before <= server._backend_health[0]["failed_at"] <= after

    def test_no_backends_has_empty_health(self):
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=StubProvider(),
            resolved_key="key",
            model="model",
        )
        assert server._backend_health == []


# -- Step 2: _get_next_backend skips unhealthy backends ----------------------


class TestGetNextBackendHealthAware:
    """_get_next_backend skips unhealthy backends and respects cooldown."""

    def test_skips_unhealthy_backend(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(1)

        # Should cycle through 0 and 2 only
        keys = []
        for _ in range(4):
            _, key, _, _ = server._get_next_backend()
            keys.append(key)
        assert keys == ["key-0", "key-2", "key-0", "key-2"]

    def test_cooldown_expired_backend_recovers(self):
        server = _make_server(3, cooldown=0)  # Instant cooldown
        server._mark_backend_unhealthy(1)

        # With cooldown=0, backend should recover immediately
        keys = []
        for _ in range(6):
            _, key, _, _ = server._get_next_backend()
            keys.append(key)
        # All three backends should be used
        assert "key-0" in keys
        assert "key-1" in keys
        assert "key-2" in keys

    def test_cooldown_not_expired_still_skipped(self):
        server = _make_server(3, cooldown=9999)
        server._mark_backend_unhealthy(1)

        keys = []
        for _ in range(4):
            _, key, _, _ = server._get_next_backend()
            keys.append(key)
        assert "key-1" not in keys

    def test_all_unhealthy_still_returns_backend(self):
        """When ALL backends are unhealthy, return next one anyway (let it fail naturally)."""
        server = _make_server(2, cooldown=9999)
        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(1)

        _, key, _, _ = server._get_next_backend()
        assert key in ("key-0", "key-1")

    def test_non_balancing_unaffected(self):
        """Non-balancing mode (no backends) works as before."""
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=StubProvider(),
            resolved_key="single-key",
            model="single-model",
        )
        _, key, model, _ = server._get_next_backend()
        assert key == "single-key"
        assert model == "single-model"


# -- Step 3: Automatic retry integration ------------------------------------


class TestCircuitBreakerRetry:
    """Handlers automatically retry with next healthy backend on failure."""

    @pytest.mark.asyncio
    async def test_first_backend_fails_retries_on_second(self):
        """Backend-0 returns 500, backend-1 succeeds — agent gets success."""
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Backend-0 fails
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=500,
                payload={"error": {"message": "Internal error"}},
            )
            # Backend-1 succeeds
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["choices"][0]["message"]["content"] == "Hello!"

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_first_two_fail_third_succeeds(self):
        """Backend-0 and backend-1 fail, backend-2 succeeds."""
        server = _make_server(3)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api0.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail 0"}})
            m.post("https://api1.example.com/v1/chat/completions", status=502, payload={"error": {"message": "fail 1"}})
            m.post("https://api2.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_all_backends_fail_propagates_error(self):
        """When all backends fail, the error propagates to the agent."""
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api0.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail 0"}})
            m.post("https://api1.example.com/v1/chat/completions", status=502, payload={"error": {"message": "fail 1"}})

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 500

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_balancing_no_retry(self):
        """Non-balancing mode: error propagates directly, no retry."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail"}})

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 500

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_unhealthy_backend_skipped_on_next_request(self):
        """After backend-0 fails, it's skipped for the next request within cooldown."""
        server = _make_server(2, cooldown=9999)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Request 1: backend-0 fails, backend-1 succeeds
            m.post("https://api0.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail"}})
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

            # Request 2: backend-0 should be skipped (in cooldown), backend-1 used directly
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

        await server.stop_async()
