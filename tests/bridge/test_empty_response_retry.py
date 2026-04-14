"""Tests for empty response retry and failover logic in the kitty bridge."""

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


# An "empty" CC response: status 200, but no content or tool calls in the first choice.
EMPTY_CC_RESPONSE = {
    "id": "chatcmpl-empty",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
}

OK_CC_RESPONSE = {
    "id": "chatcmpl-ok",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_server(n_backends: int = 1, cooldown: int = 300) -> BridgeServer:
    if n_backends == 1:
        provider = StubProvider(provider_type="stub", base_url="https://api.example.com/v1")
        server = BridgeServer(
            adapter=None,  # None adapter registers ALL protocol routes
            provider=provider,
            resolved_key="key-0",
            model="model-0",
        )
        return server

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

    server = BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=cooldown,
    )
    return server


def _make_balancing_server(n_backends: int = 2, cooldown: int = 300) -> BridgeServer:
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
        backend_cooldown=cooldown,
    )


# -- Tests ------------------------------------------------------------------


class TestEmptyResponseDetection:
    """Test the internal empty response detection logic."""

    def test_null_content_is_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response(EMPTY_CC_RESPONSE) is True

    def test_empty_string_is_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response({
            "choices": [{"message": {"content": ""}}],
        }) is True

    def test_whitespace_is_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response({
            "choices": [{"message": {"content": "   "}}],
        }) is True

    def test_no_choices_is_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response({"choices": []}) is True

    def test_tool_calls_not_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response({
            "choices": [{"message": {"content": None, "tool_calls": [{"id": "1"}]}}],
        }) is False

    def test_text_content_not_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response(OK_CC_RESPONSE) is False


class TestEmptyResponseNonBalancing:
    """Retry logic for single-profile setup (no balancing)."""

    @pytest.mark.asyncio
    async def test_non_streaming_retries_on_empty_then_succeeds(self):
        """Non-streaming: Empty -> Empty -> Success."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post("https://api.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post("https://api.example.com/v1/chat/completions", payload=OK_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Hello!" in body["content"][0]["text"]

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_streaming_exhausts_retries_emits_fallback(self):
        """Non-streaming: Empty x 4 -> Fallback text emitted."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for _ in range(4):
                m.post("https://api.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Upstream model returned an empty response" in body["content"][0]["text"]

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_streaming_retries_on_empty_then_succeeds(self):
        """Streaming: Empty -> Success."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # First attempt: SSE stream with finish_reason but no content
            m.post(
                "https://api.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )
            # Second attempt: success
            m.post(
                "https://api.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                assert "Hello" in content
                assert content.count("event: message_start") == 1

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_streaming_exhausts_retries_emits_fallback(self):
        """Streaming: Empty x 4 -> Fallback text emitted."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for _ in range(4):
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body='data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                assert "Upstream model returned an empty response" in content

        await server.stop_async()


class TestEmptyResponseBalancing:
    """Failover logic for balancing profiles on empty responses."""

    @pytest.mark.asyncio
    async def test_streaming_failover_on_empty(self):
        """Streaming: first backend empty -> marks unhealthy -> other backend succeeds."""
        server = _make_balancing_server(2)
        # Force deterministic backend selection: backend-0 first, then backend-1
        selection_order = iter([0, 1])

        def _next_backend():
            idx = next(selection_order)
            return (
                server._backends[idx][0],
                server._backends[idx][1],
                server._backends[idx][2].model,
                server._backends[idx][2].provider_config or {},
                idx,
            )

        server._get_next_backend = _next_backend
        # Simpler: just monkey-patch _select_backend to pick backends in order
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            idx = next(_pick)
            provider, key, profile = self._backends[idx]
            self._active_provider = provider
            self._active_key = key
            self._active_model = profile.model
            self._active_provider_config = profile.provider_config or {}
            self._current_backend_idx = idx

        server._select_backend = _fixed_select

        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        empty_stream = 'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        success_stream = (
            'data: {"choices":[{"delta":{"content":"World"}}]}\n\n'
            'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Backend-0 (first): empty, Backend-1 (second): success
            m.post("https://api0.example.com/v1/chat/completions", body=empty_stream)
            m.post("https://api1.example.com/v1/chat/completions", body=success_stream)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                assert "World" in content
                assert content.count("event: message_start") == 1

        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is True
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_streaming_failover_on_empty(self):
        """Non-streaming: first backend empty -> marks unhealthy -> other backend succeeds."""
        server = _make_balancing_server(2)
        # Force deterministic backend selection: backend-0 first, then backend-1
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            idx = next(_pick)
            provider, key, profile = self._backends[idx]
            self._active_provider = provider
            self._active_key = key
            self._active_model = profile.model
            self._active_provider_config = profile.provider_config or {}
            self._current_backend_idx = idx

        server._select_backend = _fixed_select

        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Backend-0 (first): empty, Backend-1 (second): success
            m.post("https://api0.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post("https://api1.example.com/v1/chat/completions", payload=OK_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Hello!" in body["content"][0]["text"]

        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is True
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_all_backends_empty_emits_fallback(self):
        """All backends empty -> fallback text emitted."""
        server = _make_balancing_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Both backends return empty for all attempts
            for url_prefix in ("api0", "api1"):
                for _ in range(2):
                    m.post(f"https://{url_prefix}.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Upstream model returned an empty response" in body["content"][0]["text"]

        await server.stop_async()
