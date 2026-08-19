"""Tests for egress routing in the bridge — which session reaches which host.

The proxy is configured on the ``ClientSession`` rather than per request, so
these tests assert on the session a destination resolves to. That is the real
decision point: aiohttp cannot opt an individual request out of its session's
proxy, so choosing the session *is* choosing whether traffic is proxied.
"""

from __future__ import annotations

from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.egress import EgressConfig
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

PROXY_URL = "http://proxy.example.com:12323"


class _StubLauncher(LauncherAdapter):
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


class _StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Error {status_code}")


def _server(egress: EgressConfig | None) -> BridgeServer:
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=_StubProvider(),
        resolved_key="key",
        model="model",
        egress=egress,
    )


@pytest.fixture()
def egress() -> EgressConfig:
    return EgressConfig(proxy_url=PROXY_URL, username="myuser", password="s3cr3t")


class TestSessionSelection:
    """R6/R7/R8: destination decides whether the proxy is used."""

    @pytest.mark.asyncio()
    async def test_public_host_uses_the_proxied_session(self, egress: EgressConfig):
        server = _server(egress)
        try:
            session = await server._session_for("https://api.anthropic.com/v1/messages")

            assert session._default_proxy == PROXY_URL
            assert session._default_proxy_auth is not None
            assert session._default_proxy_auth.login == "myuser"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_loopback_host_uses_the_direct_session(self, egress: EgressConfig):
        """A rented proxy cannot reach the caller's own machine."""
        server = _server(egress)
        try:
            session = await server._session_for("http://localhost:11434/v1/chat/completions")

            assert session._default_proxy is None
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_private_lan_host_uses_the_direct_session(self, egress: EgressConfig):
        server = _server(egress)
        try:
            session = await server._session_for("http://192.168.1.50:8000/v1")

            assert session._default_proxy is None
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_without_egress_nothing_is_proxied(self):
        server = _server(None)
        try:
            for url in ("https://api.anthropic.com/v1", "http://localhost:11434/v1"):
                session = await server._session_for(url)
                assert session._default_proxy is None
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_proxied_and_direct_sessions_are_distinct_and_cached(self, egress: EgressConfig):
        server = _server(egress)
        try:
            public_a = await server._session_for("https://api.anthropic.com/v1")
            public_b = await server._session_for("https://api.minimax.io/anthropic")
            local_a = await server._session_for("http://127.0.0.1:11434/v1")
            local_b = await server._session_for("http://localhost:11434/v1")

            assert public_a is public_b, "proxied session should be reused"
            assert local_a is local_b, "direct session should be reused"
            assert public_a is not local_a
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_unauthenticated_proxy_sets_no_auth(self):
        server = _server(EgressConfig(proxy_url=PROXY_URL))
        try:
            session = await server._session_for("https://api.anthropic.com/v1")

            assert session._default_proxy == PROXY_URL
            assert session._default_proxy_auth is None
        finally:
            await server.stop_async()


class TestSessionLifecycle:
    @pytest.mark.asyncio()
    async def test_stop_closes_both_sessions(self, egress: EgressConfig):
        server = _server(egress)
        proxied = await server._session_for("https://api.anthropic.com/v1")
        direct = await server._session_for("http://localhost:11434/v1")

        await server.stop_async()

        assert proxied.closed
        assert direct.closed

    @pytest.mark.asyncio()
    async def test_closed_proxied_session_is_rebuilt(self, egress: EgressConfig):
        server = _server(egress)
        try:
            first = await server._session_for("https://api.anthropic.com/v1")
            await first.close()

            second = await server._session_for("https://api.anthropic.com/v1")

            assert second is not first
            assert not second.closed
            assert second._default_proxy == PROXY_URL
        finally:
            await server.stop_async()


class _LocalProvider(_StubProvider):
    """Provider pointed at a loopback endpoint, like a local Ollama."""

    @property
    def default_base_url(self) -> str:
        return "http://localhost:11434/v1"


UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


async def _drive_request(server: BridgeServer, upstream_url: str) -> list[tuple[str, object]]:
    """Send one request through the bridge and record how each hop was routed.

    Wraps ``ClientSession._request`` so every outbound call reports the session
    it actually went out on. That is the behavioural claim under test: reverting
    the per-destination session selection makes these assertions fail, which
    asserting on ``_session_for`` alone would not.

    Args:
        server: A started-on-demand bridge.
        upstream_url: Upstream endpoint to mock.

    Returns:
        ``(url, session_default_proxy)`` for every non-loopback request.
    """
    seen: list[tuple[str, object]] = []
    port = await server.start_async()
    try:
        with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
            mocked.post(upstream_url, payload=UPSTREAM_RESPONSE, repeat=True)
            mocked_request = aiohttp.ClientSession._request

            async def spy(self, method, url, *args, **kwargs):
                if "127.0.0.1" not in str(url):
                    seen.append((str(url), self._default_proxy))
                return await mocked_request(self, method, url, *args, **kwargs)

            with patch.object(aiohttp.ClientSession, "_request", spy):
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    ) as resp,
                ):
                    assert resp.status == 200
                    await resp.json()
    finally:
        await server.stop_async()
    return seen


class TestRequestsActuallyGoOutProxied:
    """R6/R7/R8 end to end, over the real request path."""

    @pytest.mark.asyncio()
    async def test_every_public_upstream_request_carries_the_proxy(self, egress: EgressConfig):
        server = _server(egress)

        routed = await _drive_request(server, "https://api.example.com/v1/chat/completions")

        assert routed, "no upstream request was observed — the test proves nothing"
        unproxied = [url for url, proxy in routed if proxy != PROXY_URL]
        assert not unproxied, f"these upstream requests bypassed the egress proxy: {unproxied}"

    @pytest.mark.asyncio()
    async def test_loopback_upstream_is_not_proxied(self, egress: EgressConfig):
        """A rented proxy cannot reach the caller's own machine."""
        server = BridgeServer(
            adapter=_StubLauncher(),
            provider=_LocalProvider(),
            resolved_key="key",
            model="model",
            egress=egress,
        )

        routed = await _drive_request(server, "http://localhost:11434/v1/chat/completions")

        assert routed
        assert all(proxy is None for _url, proxy in routed), f"local endpoint was tunnelled: {routed}"

    @pytest.mark.asyncio()
    async def test_without_egress_no_request_is_proxied(self):
        server = _server(None)

        routed = await _drive_request(server, "https://api.example.com/v1/chat/completions")

        assert routed
        assert all(proxy is None for _url, proxy in routed)


class _MessagesLauncher(_StubLauncher):
    """Launcher that makes the bridge serve the Anthropic Messages API."""

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API
