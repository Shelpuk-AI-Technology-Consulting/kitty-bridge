"""Tests for bridge server random balancing — backend selection per request."""

import random
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter
from kitty.providers.bedrock import BedrockAdapter
from kitty.types import BridgeProtocol


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
    """Provider that records which URL/headers were used."""

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


UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_backends(n: int):
    """Create n (provider, key, profile_dict) tuples for balancing tests."""
    import uuid

    from kitty.profiles.schema import Profile

    backends = []
    for i in range(n):
        provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(name=f"profile-{i}", provider="openai", model=f"model-{i}", auth_ref=str(uuid.uuid4()))
        backends.append((provider, key, profile))
    return backends


class TestRandomSelection:
    def test_single_backend_always_returns_same(self):
        """With one backend, _get_next_backend always returns the same."""
        backends = _make_backends(1)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        random.seed(42)
        results = [server._get_next_backend()[1] for _ in range(10)]
        assert all(k == "key-0" for k in results)

    def test_multiple_backends_all_selected(self):
        """With multiple backends, random selection should hit all of them."""
        backends = _make_backends(3)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        random.seed(42)
        results = {server._get_next_backend()[1] for _ in range(100)}
        assert results == {"key-0", "key-1", "key-2"}

    def test_no_backends_uses_single_profile(self):
        """When backends is None, falls back to single profile mode."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
        )
        p, key, model, config, _idx = server._get_next_backend()
        assert key == "single-key"
        assert model == "single-model"

    def test_no_backends_no_model_returns_none(self):
        """When backends is None and no model set, returns None for model."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
        )
        p, key, model, config, _idx = server._get_next_backend()
        assert key == "single-key"
        assert model is None


class TestBalancingIntegration:
    @pytest.mark.asyncio
    async def test_chat_completions_uses_balancing(self):
        """Two chat completion requests should route to backends."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Mock both upstream endpoints
            m.post("https://api0.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    await resp.json()

                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    await resp.json()

        await server.stop_async()
        # Verify at least one backend was called (with 2 backends and 2 requests, both are likely but not guaranteed)
        from yarl import URL

        total_requests = sum(
            len(list(m.requests.get(("POST", URL(f"https://api{i}.example.com/v1/chat/completions")), [])))
            for i in range(2)
        )
        assert total_requests >= 2

    @pytest.mark.asyncio
    async def test_chat_completions_single_backend(self):
        """Without backends, single profile mode works."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="my-key",
            model="my-model",
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

        await server.stop_async()


class TestProviderConfigPropagation:
    def test_provider_config_per_backend(self):
        """Each backend's provider_config is correctly set as _active_provider_config."""
        import uuid

        from kitty.profiles.schema import Profile

        backends = []
        for i in range(3):
            provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
            profile = Profile(
                name=f"profile-{i}",
                provider="openai",
                model=f"model-{i}",
                auth_ref=str(uuid.uuid4()),
                provider_config={"custom_url": f"https://custom{i}.example.com"},
            )
            backends.append((provider, f"key-{i}", profile))

        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )

        # Each _select_backend call picks a random backend;
        # verify that its provider_config is one of the expected values
        valid_configs = {f"https://custom{i}.example.com" for i in range(3)}

        random.seed(42)
        for _ in range(20):
            server._select_backend()
            assert server._active_provider_config["custom_url"] in valid_configs

    def test_single_profile_uses_init_provider_config(self):
        """Without backends, _active_provider_config uses the init parameter."""
        provider = StubProvider()
        config = {"base_url": "https://custom.example.com"}
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
            provider_config=config,
        )

        server._select_backend()
        assert server._active_provider_config == config


class TestBalancingAllCustomTransport:
    """Verify that when ALL balanced backends use custom transport,
    the bridge NEVER falls through to the aiohttp upstream path.

    Regression test for a bug where balanced profiles with only
    openai_subscription (use_custom_transport=True) members still
    produced 'Upstream Cloudflare block' errors from the aiohttp path.
    """

    @pytest.mark.asyncio
    async def test_responses_stream_uses_custom_transport(self):
        """Responses API streaming should call provider.stream_request(), not aiohttp."""
        import uuid
        import json as _json

        from kitty.profiles.schema import Profile

        # Build SSE response that mimics Codex backend output
        sse_events = [
            b'data: {"type":"response.created","response":{"id":"resp_test","status":"in_progress"}}\n\n',
            b'data: {"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"hi"}]}}\n\n',
            b'data: [DONE]\n\n',
        ]

        # Create 2 custom-transport backends
        backends = []
        for i in range(2):
            provider = BedrockAdapter()  # use_custom_transport=True
            # Override stream_request with a mock that yields SSE chunks
            collected_chunks = []

            async def _fake_stream(req, write, *, _chunks=sse_events, _collected=collected_chunks):
                for chunk in _chunks:
                    _collected.append(chunk)
                    await write(chunk)

            provider.stream_request = AsyncMock(side_effect=_fake_stream)
            profile = Profile(
                name=f"test-profile-{i}",
                provider="bedrock",
                model=f"model-{i}",
                auth_ref=str(uuid.uuid4()),
            )
            backends.append((provider, f"key-{i}", profile))

        server = BridgeServer(
            adapter=None,  # bridge mode
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/responses"

        request_body = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    # Read the full SSE stream
                    body = await resp.read()
                    assert b"response.created" in body or b"error" not in body.lower()[:200]
        finally:
            await server.stop_async()

        # Verify provider.stream_request was called (not aiohttp upstream)
        called_count = sum(1 for b in backends if b[0].stream_request.called)
        assert called_count >= 1, "At least one custom-transport provider should have been called"

    @pytest.mark.asyncio
    async def test_messages_stream_uses_custom_transport(self):
        """Messages API streaming should call provider.stream_request(), not aiohttp."""
        import uuid

        from kitty.profiles.schema import Profile

        # Build SSE response that mimics Codex backend output
        sse_events = [
            b'data: {"type":"response.created","response":{"id":"resp_test","status":"in_progress"}}\n\n',
            b'data: {"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"hi"}]}}\n\n',
            b'data: [DONE]\n\n',
        ]

        backends = []
        for i in range(2):
            provider = BedrockAdapter()  # use_custom_transport=True

            async def _fake_stream(req, write, *, _chunks=sse_events):
                for chunk in _chunks:
                    await write(chunk)

            provider.stream_request = AsyncMock(side_effect=_fake_stream)
            profile = Profile(
                name=f"test-profile-{i}",
                provider="bedrock",
                model=f"model-{i}",
                auth_ref=str(uuid.uuid4()),
            )
            backends.append((provider, f"key-{i}", profile))

        server = BridgeServer(
            adapter=None,
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 100,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    body = await resp.read()
                    # Should produce Messages API SSE events
                    assert b"message_start" in body or b"message_stop" in body
        finally:
            await server.stop_async()

        called_count = sum(1 for b in backends if b[0].stream_request.called)
        assert called_count >= 1

    @pytest.mark.asyncio
    async def test_non_streaming_uses_custom_transport(self):
        """Non-streaming requests should call provider.make_request(), not aiohttp."""
        import uuid

        from kitty.profiles.schema import Profile

        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        backends = []
        for i in range(2):
            provider = BedrockAdapter()
            provider.make_request = AsyncMock(return_value=mock_response)
            profile = Profile(
                name=f"test-profile-{i}",
                provider="bedrock",
                model=f"model-{i}",
                auth_ref=str(uuid.uuid4()),
            )
            backends.append((provider, f"key-{i}", profile))

        server = BridgeServer(
            adapter=None,
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
            "max_tokens": 100,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_body) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data.get("role") == "assistant" or "content" in str(data)
        finally:
            await server.stop_async()

        called_count = sum(1 for b in backends if b[0].make_request.called)
        assert called_count >= 1
