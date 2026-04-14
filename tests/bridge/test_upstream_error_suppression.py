"""Tests for upstream stream error suppression — errors must not leak to the agent."""

from __future__ import annotations

import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Helpers ──────────────────────────────────────────────────────────────────


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


def _make_balancing_server(n_backends: int = 2, backend_cooldown: int = 300) -> BridgeServer:
    backends = []
    for i in range(n_backends):
        provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(
            name=f"profile-{i}",
            provider="openai",
            model=f"model-{i}",
            auth_ref=str(uuid.uuid4()),
        )
        backends.append((provider, key, profile))

    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=backend_cooldown,
    )


def _make_server(backend_cooldown: int = 300) -> BridgeServer:
    provider = StubProvider()
    return BridgeServer(
        adapter=None,
        provider=provider,
        resolved_key="key-0",
        model="model-0",
        backend_cooldown=backend_cooldown,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBalancingStreamErrorSuppress:
    """Upstream stream errors in balancing mode must be invisible to the agent."""

    @pytest.mark.asyncio
    async def test_stream_error_triggers_failover_agent_sees_success(self):
        """An upstream error chunk in backend-0 triggers failover to backend-1.

        The agent must NOT see the raw error — only a clean success response.
        """
        server = _make_balancing_server(n_backends=2)
        port = await server.start_async()
        try:
            # Backend-0 sends an error chunk, then ok chunks
            # Backend-1 sends ok chunks
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    body=(
                        'data: {"id":"c","model":"m",'
                        '"choices":[{"index":0,"delta":{"content":"Hi"}}],'
                        '"error":{"code":502,"message":"Provider error"}}\n\n'
                        'data: [DONE]\n\n'
                    ),
                )
                m.post(
                    "https://api1.example.com/v1/chat/completions",
                    body=(
                        'data: {"id":"c","model":"m",'
                        '"choices":[{"index":0,'
                        '"delta":{"content":"Hello from backend-1!"}}]}\n\n'
                        'data: {"id":"c","model":"m",'
                        '"choices":[{"index":0,"delta":{},'
                        '"finish_reason":"stop"}]}\n\n'
                        'data: [DONE]\n\n'
                    ),
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ) as resp:
                    assert resp.status == 200
                    body = await resp.text()
                    # Agent must NOT see the raw error chunk
                    assert '"error"' not in body
                    assert "Provider error" not in body
                    assert "Hello from backend-1" in body
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_all_backends_fail_agent_sees_clean_error(self):
        """When all backends fail, the agent sees a clean error, NOT raw JSON."""
        server = _make_balancing_server(n_backends=2)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502,"message":"Provider error"}}\n\ndata: [DONE]\n\n',
                )
                m.post(
                    "https://api1.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502,"message":"Provider error"}}\n\ndata: [DONE]\n\n',
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ) as resp:
                    assert resp.status == 200
                    body = await resp.text()
                    # Agent must NOT see raw upstream error content
                    assert "Provider error" not in body
                    assert '"code":502' not in body
                    # The response may contain a clean SSE error message — that's fine
                    # The important thing is the raw upstream JSON structure is suppressed
        finally:
            await server.stop_async()


class TestNonBalancingStreamErrorSuppress:
    """Non-balancing: upstream errors must produce a clean error, NOT raw JSON."""

    @pytest.mark.asyncio
    async def test_non_balancing_error_not_leaked(self):
        """The agent must NOT see the raw error chunk JSON."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502,"message":"Provider error"}}\n\ndata: [DONE]\n\n',
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ) as resp:
                    assert resp.status == 200
                    body = await resp.text()
                    assert "Provider error" not in body
                    assert "Provider returned error" not in body
        finally:
            await server.stop_async()


class TestStreamErrorLogLevel:
    """Stream errors must NOT use logger.error (visible in terminal)."""

    @pytest.mark.asyncio
    async def test_stream_error_uses_warning_not_error(self, caplog):
        """Upstream stream errors must be logged at WARNING level, not ERROR."""
        import logging

        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502,"message":"Provider error"}}\n\ndata: [DONE]\n\n',
                )
                with caplog.at_level(logging.WARNING, logger="kitty.bridge"):
                    async with aiohttp.ClientSession() as session, session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                    ) as resp:
                        await resp.read()

                # The warning should be logged; no ERROR-level stream chunk messages
                warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
                error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
                assert any("Upstream sent error in stream chunk" in m for m in warning_messages)
                assert not any("Upstream sent error in stream chunk" in m for m in error_messages)
        finally:
            await server.stop_async()


class TestStreamErrorCooldownExponentialBackoff:
    """Stream errors must use exponential backoff, not a fixed long cooldown."""

    @pytest.mark.asyncio
    async def test_first_stream_error_30s_cooldown(self):
        """First stream error sets cooldown to 30s."""
        server = _make_balancing_server(n_backends=1, backend_cooldown=300)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502}}\n\ndata: [DONE]\n\n',
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ) as resp:
                    await resp.read()

            # Backend-0 should have cooldown = 30s (first failure)
            assert server._backend_health[0]["cooldown"] == 30
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_second_stream_error_60s_cooldown(self):
        """Second stream error on same backend doubles cooldown to 60s."""
        server = _make_balancing_server(n_backends=1, backend_cooldown=300)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                # First request fails
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502}}\n\ndata: [DONE]\n\n',
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ) as resp:
                    await resp.read()

                # Second request fails
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    body='data: {"error":{"code":502}}\n\ndata: [DONE]\n\n',
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                ) as resp:
                    await resp.read()

            # Backend-0 cooldown should have doubled: 30 -> 60
            assert server._backend_health[0]["cooldown"] == 60
        finally:
            await server.stop_async()
