"""Tests for bridge server random balancing — backend selection per request."""

import json
import random
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter, ProviderError
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


def _make_backends(n: int, backup_indices: set[int] | None = None):
    """Create n (provider, key, profile) tuples for balancing tests.

    Args:
        n: How many backends to build.
        backup_indices: Indices whose profile is marked as a reserve-tier
            member (``backup=True``). Defaults to none.
    """
    import uuid

    from kitty.profiles.schema import Profile

    backup_indices = backup_indices or set()
    backends = []
    for i in range(n):
        provider = StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(
            name=f"profile-{i}",
            provider="openai",
            model=f"model-{i}",
            auth_ref=str(uuid.uuid4()),
            backup=i in backup_indices,
        )
        backends.append((provider, key, profile))
    return backends


class TestAllBackendsUnhealthyFallback:
    """Tests for all-unhealthy backend fallback behavior."""

    def test_all_unhealthy_fast_fails_when_cooldowns_far_future(self):
        from kitty.bridge.server import AllBackendsUnhealthyError

        backends = _make_backends(3)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        now = 1000.0
        for idx in range(3):
            health = server._backend_health[idx]
            health["healthy"] = False
            health["failed_at"] = now
            health["cooldown"] = 300

        with (
            patch("kitty.bridge.server.time.monotonic", return_value=now + 10),
            pytest.raises(AllBackendsUnhealthyError) as exc_info,
        ):
            server._get_next_backend()

        assert exc_info.value.retry_after == 290
        assert len(exc_info.value.backends) == 3

    def test_all_unhealthy_does_not_fast_fail_when_retry_soon(self):
        backends = _make_backends(3)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        now = 1000.0
        for idx in range(3):
            health = server._backend_health[idx]
            health["healthy"] = False
            health["failed_at"] = now
            health["cooldown"] = 300

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 250):
            provider, key, model, _config, idx = server._get_next_backend()

        assert idx in {0, 1, 2}
        assert key == f"key-{idx}"
        assert model == f"model-{idx}"
        assert provider is backends[idx][0]

    def test_all_unhealthy_near_retry_selection_is_not_deterministic_oldest(self):
        backends = _make_backends(4)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        now = 1000.0
        for idx in range(4):
            health = server._backend_health[idx]
            health["healthy"] = False
            health["failed_at"] = now - idx
            health["cooldown"] = 300

        selected = set()
        with patch("kitty.bridge.server.time.monotonic", return_value=now + 250):
            for _ in range(100):
                *_rest, idx = server._get_next_backend()
                selected.add(idx)

        assert len(selected) > 1

    @pytest.mark.asyncio()
    async def test_healthz_returns_backend_status(self):
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        now = 1000.0
        server._backend_health[1]["healthy"] = False
        server._backend_health[1]["failed_at"] = now
        server._backend_health[1]["cooldown"] = 300

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 10):
            response = await server._handle_healthz(None)

        assert response.status == 200
        assert response.text is not None
        assert '"status": "ok"' in response.text
        assert '"backends"' in response.text
        assert '"name": "profile-1"' in response.text
        assert '"remaining_cooldown": 290' in response.text


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


class _MessagesLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class TestMessagesStreamingRetry:
    @pytest.mark.asyncio
    async def test_instream_error_after_partial_output_does_not_retry(self):
        """Messages streaming should close the lifecycle once client-visible output has started."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=_MessagesLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        try:
            first_url = "https://api0.example.com/v1/chat/completions"
            second_url = "https://api1.example.com/v1/chat/completions"
            hello_chunk = b'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},'
            hello_chunk += b'"finish_reason":null}],"model":"test-model"}\n\n'
            error_chunk = b'data: {"error":{"message":"boom"}}\n\n'
            partial_stream = hello_chunk + error_chunk
            msg_req = {
                "model": "model-0",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            }
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(first_url, body=partial_stream, headers={"Content-Type": "text/event-stream"})
                m.post(second_url, body=partial_stream, headers={"Content-Type": "text/event-stream"})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(f"http://127.0.0.1:{port}/v1/messages", json=msg_req) as resp,
                ):
                    body = await resp.read()
                    assert resp.status == 200
                    text = body.decode("utf-8", errors="replace")
                    assert text.count("event: message_start") == 1
                    assert "Recovered" not in text
                    assert "Hello" in text
                    assert "event: content_block_stop" in text
                    assert "event: message_delta" in text
                    assert '"stop_reason": "end_turn"' in text
                    assert "event: message_stop" in text
                    assert "event: error" not in text
                    assert text.index("Hello") < text.index("event: content_block_stop")
                    assert text.index("event: content_block_stop") < text.index("event: message_delta")
                    assert text.index("event: message_delta") < text.index("event: message_stop")

                recorded_urls = [str(key[1]) for key, calls in m.requests.items() for _request in calls]
                assert len(recorded_urls) == 1
                assert recorded_urls[0] in {first_url, second_url}
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_instream_error_before_output_retries_safely(self):
        """Messages streaming should retry safely if no output has been emitted yet."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=_MessagesLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        random.seed(42)  # deterministic backend selection: api0 first
        port = await server.start_async()
        try:
            first_url = "https://api0.example.com/v1/chat/completions"
            second_url = "https://api1.example.com/v1/chat/completions"
            error_stream = b'data: {"error":{"message":"boom"}}\n\n'
            rec_chunk = b'data: {"id":"2","choices":[{"index":0,"delta":{"content":"Recovered"},'
            rec_chunk += b'"finish_reason":null}],"model":"test-model"}\n\n'
            rec_finish = b'data: {"id":"2","choices":[{"index":0,"delta":{},'
            rec_finish += b'"finish_reason":"stop"}],"model":"test-model"}\n\n'
            recovery_stream = rec_chunk + rec_finish + b"data: [DONE]\n\n"
            msg_req = {
                "model": "model-0",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            }
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(first_url, body=error_stream, headers={"Content-Type": "text/event-stream"})
                m.post(second_url, body=recovery_stream, headers={"Content-Type": "text/event-stream"})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(f"http://127.0.0.1:{port}/v1/messages", json=msg_req) as resp,
                ):
                    body = await resp.read()
                    assert resp.status == 200
                    text = body.decode("utf-8", errors="replace")
                    assert text.count("event: message_start") == 1
                    assert "Recovered" in text
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_http_cloudflare_before_output_retries_safely(self):
        """Messages streaming should fail over on HTTP Cloudflare 403 before output."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=_MessagesLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        try:
            first_url = "https://api0.example.com/v1/chat/completions"
            second_url = "https://api1.example.com/v1/chat/completions"
            cloudflare_html = "<html><head>cf-browser-verification window._cf_chl_opt = {};</head></html>"
            rec_chunk = b'data: {"id":"2","choices":[{"index":0,"delta":{"content":"Recovered"},'
            rec_chunk += b'"finish_reason":null}],"model":"test-model"}\n\n'
            rec_finish = b'data: {"id":"2","choices":[{"index":0,"delta":{},'
            rec_finish += b'"finish_reason":"stop"}],"model":"test-model"}\n\n'
            recovery_stream = rec_chunk + rec_finish + b"data: [DONE]\n\n"
            msg_req = {
                "model": "model-0",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            }
            with (
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
                patch("kitty.bridge.server.random.choices", side_effect=[[0], [1]]),
            ):
                m.post(first_url, status=403, body=cloudflare_html, headers={"Content-Type": "text/html"})
                m.post(second_url, body=recovery_stream, headers={"Content-Type": "text/event-stream"})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(f"http://127.0.0.1:{port}/v1/messages", json=msg_req) as resp,
                ):
                    body = await resp.read()
                    assert resp.status == 200
                    text = body.decode("utf-8", errors="replace")
                    assert "Recovered" in text
                    assert "<html>" not in text
                    assert server._backend_health[0]["healthy"] is False
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_http_cloudflare_single_backend_emits_sanitized_error(self):
        """Messages streaming should sanitize Cloudflare HTML when no failover exists."""
        backends = _make_backends(1)
        server = BridgeServer(
            adapter=_MessagesLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        try:
            url = "https://api0.example.com/v1/chat/completions"
            cloudflare_html = "<html><head>cf-browser-verification window._cf_chl_opt = {};</head></html>"
            msg_req = {
                "model": "model-0",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            }
            with (
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
                patch("kitty.bridge.server.random.choices", return_value=[0]),
            ):
                m.post(url, status=403, body=cloudflare_html, headers={"Content-Type": "text/html"})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(f"http://127.0.0.1:{port}/v1/messages", json=msg_req) as resp,
                ):
                    assert resp.status == 403
                    data = await resp.json()
                    assert data["type"] == "error"
                    assert "cloudflare bot detection" in data["error"]["message"].lower()
                    assert "<html>" not in data["error"]["message"]
                    assert "cf-browser-verification" not in data["error"]["message"]
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_http_cloudflare_all_backends_exhausted_emits_sanitized_error(self):
        """Messages streaming should stop cleanly when all backends hit Cloudflare."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=_MessagesLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        try:
            first_url = "https://api0.example.com/v1/chat/completions"
            second_url = "https://api1.example.com/v1/chat/completions"
            cloudflare_html = "<html><head>cf-browser-verification window._cf_chl_opt = {};</head></html>"
            msg_req = {
                "model": "model-0",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            }
            with (
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
                patch("kitty.bridge.server.random.choices", side_effect=[[0], [1]]),
            ):
                m.post(first_url, status=403, body=cloudflare_html, headers={"Content-Type": "text/html"})
                m.post(second_url, status=403, body=cloudflare_html, headers={"Content-Type": "text/html"})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(f"http://127.0.0.1:{port}/v1/messages", json=msg_req) as resp,
                ):
                    assert resp.status == 403
                    data = await resp.json()
                    assert data["type"] == "error"
                    assert "cloudflare bot detection" in data["error"]["message"].lower()
                    assert "<html>" not in data["error"]["message"]
                    assert server._backend_health[0]["healthy"] is False
                    assert server._backend_health[1]["healthy"] is False
        finally:
            await server.stop_async()


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

        from kitty.profiles.schema import Profile

        # Build SSE response that mimics Codex backend output
        sse_events = [
            b'data: {"type":"response.created","response":{"id":"resp_test","status":"in_progress"}}\n\n',
            (
                b'data: {"type":"response.output_item.done",'
                b'"item":{"type":"message","content":[{"type":"output_text","text":"hi"}]}}\n\n'
            ),
            b"data: [DONE]\n\n",
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
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
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
            (
                b'data: {"type":"response.output_item.done",'
                b'"item":{"type":"message","content":[{"type":"output_text","text":"hi"}]}}\n\n'
            ),
            b"data: [DONE]\n\n",
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
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.read()
                assert body
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
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data.get("role") == "assistant" or "content" in str(data)
        finally:
            await server.stop_async()

        called_count = sum(1 for b in backends if b[0].make_request.called)
        assert called_count >= 1

    @pytest.mark.asyncio
    async def test_streaming_skips_backends_without_stream_request(self):
        import uuid

        from kitty.profiles.schema import Profile

        class NoStreamProvider(ProviderAdapter):
            def __init__(self):
                self._provider_type = "nostream"

            @property
            def provider_type(self) -> str:
                return self._provider_type

            @property
            def default_base_url(self) -> str:
                return "https://api.nostream.example.com/v1"

            def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
                return {"model": model, "messages": messages, **kwargs}

            def translate_to_upstream(self, cc_request: dict) -> dict:
                return {
                    "model": cc_request["model"],
                    "messages": cc_request["messages"],
                    "stream": True,
                }

            def parse_response(self, response_data: dict) -> dict:
                return response_data

            def map_error(self, status_code: int, body: dict) -> Exception:
                return Exception(f"Error {status_code}")

            def make_request(self, cc_request: dict) -> dict:
                return UPSTREAM_RESPONSE

        stream_provider = BedrockAdapter()

        async def _fake_stream(req, write):
            await write(b'data: {"type":"response.created","response":{"id":"resp_test","status":"in_progress"}}\n\n')
            await write(
                b'data: {"type":"response.output_text.delta","delta":"hello","response":{"id":"resp_test"}}\n\n'
            )
            await write(b"data: [DONE]\n\n")

        stream_provider.stream_request = AsyncMock(side_effect=_fake_stream)

        backends = [
            (
                NoStreamProvider(),
                "nostream-key",
                Profile(
                    name="nostream",
                    provider="openai",
                    model="nostream-model",
                    auth_ref=str(uuid.uuid4()),
                ),
            ),
            (
                stream_provider,
                "stream-key",
                Profile(
                    name="stream",
                    provider="bedrock",
                    model="stream-model",
                    auth_ref=str(uuid.uuid4()),
                ),
            ),
        ]

        server = BridgeServer(
            adapter=None,
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="nostream-model",
            backends=backends,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True}

        try:
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                _ = await resp.read()
                assert resp.status == 200
                assert server._active_provider is stream_provider
                assert resp.content_type == "text/event-stream"
        finally:
            await server.stop_async()

        assert server._active_provider is stream_provider


class TestCustomTransportCloudflareClassification:
    """Verify that ProviderError with is_cloudflare=True triggers
    failure_kind='cloudflare' in the balancer, not a generic stream error."""

    @pytest.mark.asyncio
    async def test_cf_provider_error_marks_cloudflare_failure_kind(self):
        """Custom-transport CF ProviderError should mark backend with failure_kind='cloudflare'."""
        import uuid

        from kitty.profiles.schema import Profile
        from kitty.providers.base import ProviderError

        # Create a custom-transport provider that raises a CF ProviderError
        provider = BedrockAdapter()  # use_custom_transport=True

        async def _cf_stream_tagged(req, write):
            exc = ProviderError("Cloudflare bot detection blocked the Codex backend request.")
            exc.is_cloudflare = True
            raise exc

        provider.stream_request = AsyncMock(side_effect=_cf_stream_tagged)

        profile = Profile(
            name="test-cf-profile",
            provider="bedrock",
            model="test-model",
            auth_ref=str(uuid.uuid4()),
        )
        backends = [(provider, "key-0", profile)]
        server = BridgeServer(
            adapter=None,  # bridge mode
            provider=provider,
            resolved_key="key-0",
            model="test-model",
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
            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                body = await resp.read()
                # Should have gotten the CF error message in the SSE response
                assert b"Cloudflare" in body or b"cloudflare" in body.lower()
        finally:
            await server.stop_async()

        # Verify the backend was marked unhealthy after cloudflare error
        health = server._backend_health[0]
        assert health["healthy"] is False


# ── Auth error classification and profile-specific message tests ───────────


class AuthFailingCustomProvider(StubProvider):
    """Custom-transport provider that always raises auth errors."""

    @property
    def use_custom_transport(self) -> bool:
        return True

    async def make_request(self, cc_request: dict) -> dict:
        from kitty.providers.base import ProviderError

        err = ProviderError("Authentication refresh failed: refresh_token_reused")
        err.http_status = 401
        raise err

    async def stream_request(self, cc_request: dict, write) -> None:
        from kitty.providers.base import ProviderError

        err = ProviderError("Authentication refresh failed: refresh_token_reused")
        err.http_status = 401
        raise err


class TestAuthErrorClassification:
    """Tests for auth failure classification and profile-specific messaging."""

    def test_provider_error_401_classifies_as_auth(self):
        """ProviderError with http_status=401 is classified as 'auth'."""
        from kitty.providers.base import ProviderError

        err = ProviderError("auth failure")
        err.http_status = 401
        assert BridgeServer._provider_error_failure_kind(err) == "auth"

    def test_provider_error_429_classifies_as_rate_limit(self):
        """ProviderError with http_status=429 is classified as 'rate_limit'."""
        from kitty.providers.base import ProviderError

        err = ProviderError("rate limited")
        err.http_status = 429
        assert BridgeServer._provider_error_failure_kind(err) == "rate_limit"

    def test_provider_error_0_classifies_as_hard(self):
        """ProviderError with default http_status=0 is classified as 'hard'."""
        from kitty.providers.base import ProviderError

        err = ProviderError("some error")
        assert BridgeServer._provider_error_failure_kind(err) == "hard"

    def test_custom_transport_auth_error_message_includes_backend_profile_name(self):
        """Auth errors surface the backend profile name with re-login guidance."""
        from kitty.providers.base import ProviderError

        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        server._current_backend_idx = 1

        err = ProviderError("Authentication refresh failed: refresh_token_reused")
        err.http_status = 401

        msg = server._custom_transport_error_message(err)

        assert "profile-1" in msg, f"Expected profile name in message, got: {msg}"
        assert "kitty auth openai" in msg, f"Expected re-login command in message, got: {msg}"
        assert "refresh_token_reused" in msg

    def test_non_auth_error_message_not_rewritten(self):
        """Non-auth errors pass through unchanged."""
        from kitty.providers.base import ProviderError

        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        server._current_backend_idx = 1

        err = ProviderError("rate limited")
        err.http_status = 429

        msg = server._custom_transport_error_message(err)
        assert msg == "rate limited"


class TestRateLimitRetryAfter:
    """Tests for ProviderError.retry_after propagation through the bridge."""

    def test_provider_error_has_retry_after_default_none(self):
        from kitty.providers.base import ProviderError

        err = ProviderError("some error")
        assert err.retry_after is None

    def test_provider_error_retry_after_settable(self):
        from kitty.providers.base import ProviderError

        err = ProviderError("rate limited")
        err.retry_after = 42
        assert err.retry_after == 42

    def test_rate_limit_failure_kind_uses_retry_after_cooldown(self):
        """When a ProviderError with http_status=429 carries retry_after, the
        bridge should use that value as the backend cooldown instead of the
        default 300s."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )
        server._current_backend_idx = 0

        err = ProviderError("rate limited")
        err.http_status = 429
        err.retry_after = 17

        kind = server._provider_error_failure_kind(err)
        assert kind == "rate_limit"

        server._mark_backend_unhealthy(
            0,
            failure_kind=kind,
            cooldown=err.retry_after,
        )
        health = server._backend_health[0]
        assert health["healthy"] is False
        assert health["cooldown"] == 17

    def test_rate_limit_without_retry_after_uses_default_cooldown(self):
        """When retry_after is not set, the default backend_cooldown is used."""
        backends = _make_backends(2)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="test-model",
            backends=backends,
        )

        err = ProviderError("rate limited")
        err.http_status = 429

        server._mark_backend_unhealthy(0, failure_kind="rate_limit")
        health = server._backend_health[0]
        assert health["cooldown"] == 300


# ── Cross-mode failover tests (custom-transport → standard backend) ─────────


class _FailingCustomTransportProvider(ProviderAdapter):
    """Custom-transport provider that always raises a ProviderError(400)."""

    @property
    def provider_type(self) -> str:
        return "failing-custom"

    @property
    def default_base_url(self) -> str:
        return "https://failing.example.com/v1"

    @property
    def use_custom_transport(self) -> bool:
        return True

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def translate_to_upstream(self, cc_request: dict) -> dict:
        return {"model": cc_request["model"], "messages": cc_request["messages"]}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Error {status_code}")

    async def stream_request(self, cc_request: dict, write) -> None:
        err = ProviderError("OpenAI subscription error 400: ")
        err.http_status = 400
        raise err


class TestCrossModeFailover:
    """Verify that when a custom-transport backend fails, the bridge falls
    through to a standard streaming backend instead of surfacing the error."""

    # Shared SSE chunks for standard Chat Completions upstream mock.
    _SSE_CHUNKS = (
        b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"Hi"},'
        b'"finish_reason":null}],"model":"test"}\n\n'
        b'data: {"id":"test","choices":[{"index":0,"delta":{},'
        b'"finish_reason":"stop"}],"model":"test","usage":null}\n\n'
        b"data: [DONE]\n\n"
    )

    def _make_cross_mode_server(self):
        """Create a balancing server with a failing custom-transport backend
        and a healthy standard backend, forcing selection order [0, 1]."""
        import uuid

        from kitty.profiles.schema import Profile

        failing_provider = _FailingCustomTransportProvider()
        standard_provider = StubProvider(provider_type="standard", base_url="https://standard.example.com/v1")
        backends = [
            (
                failing_provider,
                "failing-key",
                Profile(
                    name="failing",
                    provider="openai_subscription",
                    model="fail-model",
                    auth_ref=str(uuid.uuid4()),
                ),
            ),
            (
                standard_provider,
                "standard-key",
                Profile(
                    name="standard",
                    provider="openai",
                    model="std-model",
                    auth_ref=str(uuid.uuid4()),
                ),
            ),
        ]
        server = BridgeServer(
            adapter=None,
            provider=backends[0][0],
            resolved_key="failing-key",
            model="fail-model",
            backends=backends,
        )
        return server, backends

    def _patch_selection_order(self, server):
        """Force backend selection order: 0 first, then 1."""
        _pick = iter([0, 1, 0, 1])

        def _fixed_get_next(self=server, **kwargs):
            idx = next(_pick)
            provider, key, profile = self._backends[idx]
            return provider, key, profile.model, {}, idx

        server._get_next_backend = _fixed_get_next

    @pytest.mark.asyncio
    async def test_messages_stream_cross_mode_failover(self):
        """Messages API: custom transport 400 → standard backend succeeds."""
        server, _ = self._make_cross_mode_server()
        self._patch_selection_order(server)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 100,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://standard.example.com/v1/chat/completions", body=self._SSE_CHUNKS)
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(url, json=request_body) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert b"Hi" in body
            finally:
                await server.stop_async()

        # Verify cross-mode failover: failing backend was used first (failure_count increased).
        assert server._backend_health[0]["failure_count"] >= 1

    @pytest.mark.asyncio
    async def test_responses_stream_cross_mode_failover(self):
        """Responses API: custom transport 400 → standard backend succeeds."""
        server, _ = self._make_cross_mode_server()
        self._patch_selection_order(server)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/responses"
        request_body = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://standard.example.com/v1/chat/completions", body=self._SSE_CHUNKS)
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(url, json=request_body) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert b"Hi" in body
            finally:
                await server.stop_async()

        # Cross-mode failover succeeded — failing backend was used first.
        assert server._backend_health[0]["failure_count"] >= 1


class TestGeminiTranslatorResetOnFailover:
    """F22: Gemini in-stream error failover must call translator.reset()."""

    _SSE_CHUNKS = TestCrossModeFailover._SSE_CHUNKS

    @pytest.mark.asyncio
    async def test_gemini_http_failover_resets_translator(self):
        """First Gemini upstream returns 500 → failover → translator.reset() called."""
        from kitty.bridge.gemini.translator import GeminiTranslator

        reset_calls: list[int] = []
        original_reset = GeminiTranslator.reset

        def tracking_reset(self):
            reset_calls.append(1)
            return original_reset(self)

        GeminiTranslator.reset = tracking_reset
        try:
            provider_a = StubProvider(provider_type="openai_a", base_url="https://api-a.example.com/v1")
            provider_b = StubProvider(provider_type="openai_b", base_url="https://api-b.example.com/v1")
            backends = _make_backends(2)
            backends[0] = (provider_a, backends[0][1], backends[0][2])
            backends[1] = (provider_b, backends[1][1], backends[1][2])
            server = BridgeServer(
                adapter=None,  # bridge mode registers Gemini endpoints
                provider=backends[0][0],
                resolved_key=backends[0][1],
                model="model-0",
                backends=backends,
                backend_cooldown=300,
            )

            picks = iter([0, 1])

            def fixed_get_next(self=server, **kwargs):
                idx = next(picks)
                provider, key, profile = self._backends[idx]
                return provider, key, profile.model, {}, idx

            server._get_next_backend = fixed_get_next

            port = await server.start_async()
            url = f"http://127.0.0.1:{port}/v1beta/models/test-model:streamGenerateContent"
            request_body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api-a.example.com/v1/chat/completions", status=500)
                m.post("https://api-b.example.com/v1/chat/completions", body=self._SSE_CHUNKS)
                try:
                    async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                        assert resp.status == 200
                        body = await resp.read()
                        assert b"Hi" in body
                finally:
                    await server.stop_async()

            assert len(reset_calls) >= 1, "GeminiTranslator.reset() was not called during failover"
        finally:
            GeminiTranslator.reset = original_reset


class TestStreamingBackendSelectionRequireStreaming:
    """F28: require_streaming=True must not fall back to non-streaming backends."""

    def test_all_non_streaming_raises_unhealthy(self):
        """When all backends are non-streaming-capable and require_streaming=True,
        _get_next_backend must raise AllBackendsUnhealthyError, not fall through."""
        from kitty.bridge.server import AllBackendsUnhealthyError

        backends = _make_backends(3)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key="key-0",
            model="model-0",
            backends=backends,
        )
        # All 3 backends use StubProvider which doesn't override stream_request
        with pytest.raises(AllBackendsUnhealthyError):
            server._get_next_backend(require_streaming=True)


class TestBackupTierSelection:
    """Backup members form a reserve tier used only when no primary is healthy."""

    @staticmethod
    def _server(backends):
        return BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )

    @staticmethod
    def _mark_unhealthy(server, indices, *, now: float, cooldown: int = 300):
        for idx in indices:
            health = server._backend_health[idx]
            health["healthy"] = False
            health["failed_at"] = now
            health["cooldown"] = cooldown

    def test_backup_never_selected_while_a_primary_is_healthy(self):
        """R3: three healthy plans must absorb all traffic, not the metered key."""
        backends = _make_backends(4, backup_indices={3})
        server = self._server(backends)

        random.seed(42)
        selected = {server._get_next_backend()[1] for _ in range(200)}

        assert selected == {"key-0", "key-1", "key-2"}

    def test_backup_selected_when_all_primaries_unhealthy(self):
        """R4: once every plan is in cooldown, the reserve takes over."""
        backends = _make_backends(4, backup_indices={3})
        server = self._server(backends)
        now = 1000.0
        self._mark_unhealthy(server, range(3), now=now)

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 10):
            selected = {server._get_next_backend()[1] for _ in range(20)}

        assert selected == {"key-3"}

    def test_single_healthy_primary_still_beats_backup(self):
        """R3: the tier boundary holds even when only one primary survives."""
        backends = _make_backends(4, backup_indices={3})
        server = self._server(backends)
        now = 1000.0
        self._mark_unhealthy(server, [0, 1], now=now)

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 10):
            selected = {server._get_next_backend()[1] for _ in range(50)}

        assert selected == {"key-2"}

    def test_traffic_returns_to_primary_when_cooldown_expires(self):
        """R5: recovery is automatic on the next request, without a restart."""
        backends = _make_backends(4, backup_indices={3})
        server = self._server(backends)
        now = 1000.0
        # profile-0 recovers first; the other two plans stay down.
        self._mark_unhealthy(server, [0], now=now, cooldown=60)
        self._mark_unhealthy(server, [1, 2], now=now, cooldown=300)

        # While every primary is still cooling down, the reserve serves.
        with patch("kitty.bridge.server.time.monotonic", return_value=now + 10):
            assert server._get_next_backend()[1] == "key-3"

        # Once profile-0's window elapses it is eligible again and outranks the reserve.
        with patch("kitty.bridge.server.time.monotonic", return_value=now + 61):
            selected = {server._get_next_backend()[1] for _ in range(20)}

        assert selected == {"key-0"}

    def test_all_backup_pool_behaves_as_flat_pool(self):
        """R6: a pool with no primary members must not stall."""
        backends = _make_backends(3, backup_indices={0, 1, 2})
        server = self._server(backends)

        random.seed(42)
        selected = {server._get_next_backend()[1] for _ in range(200)}

        assert selected == {"key-0", "key-1", "key-2"}

    def test_lone_backup_member_is_selectable(self):
        """R6: a single-member pool marked backup still resolves."""
        backends = _make_backends(1, backup_indices={0})
        server = self._server(backends)

        assert server._get_next_backend()[1] == "key-0"

    def test_failure_weighting_still_applies_within_the_primary_tier(self):
        """R7: tiering must not flatten the inverse-failure_count weighting."""
        backends = _make_backends(3, backup_indices={2})
        server = self._server(backends)
        server._backend_health[0]["failure_count"] = 99

        random.seed(42)
        picks = [server._get_next_backend()[1] for _ in range(200)]

        assert picks.count("key-1") > picks.count("key-0")
        assert "key-2" not in picks

    def test_backup_tier_used_when_primary_is_not_stream_capable(self):
        """R8: streaming capability is filtered before the tier split.

        StubProvider inherits ProviderAdapter.stream_request, so it is excluded
        from a streaming request; BedrockAdapter overrides it at class level.
        """
        backends = _make_backends(2, backup_indices={1})
        # Backend 0 is a primary that cannot stream; backend 1 is a streaming reserve.
        backends[1] = (BedrockAdapter(), backends[1][1], backends[1][2])

        server = self._server(backends)

        # Non-streaming requests still prefer the primary tier.
        assert server._get_next_backend()[1] == "key-0"
        # Streaming requests have no eligible primary, so the reserve is used.
        assert server._get_next_backend(require_streaming=True)[1] == "key-1"

    def test_non_balancing_mode_is_unaffected(self):
        """R13: backup only has meaning inside a balancing pool."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="solo-key",
            model="model-0",
        )

        assert server._backend_is_backup == []
        assert server._get_next_backend()[4] == -1


class TestBackupTierHealthz:
    """R12: /healthz exposes tier membership for debugging."""

    @pytest.mark.asyncio()
    async def test_healthz_reports_backup_flag_per_backend(self):
        backends = _make_backends(3, backup_indices={2})
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )

        response = await server._handle_healthz(None)

        payload = json.loads(response.text)
        assert [b["backup"] for b in payload["backends"]] == [False, False, True]


class TestBackupTierDegradedPaths:
    """Edge cases where the reserve tier meets the existing health machinery."""

    @staticmethod
    def _server(backends):
        return BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )

    def test_any_healthy_backend_counts_a_healthy_reserve(self):
        """Deliberate design decision, guarded.

        _any_healthy_backend gates "should I keep retrying?" at ~15 handler
        sites. A healthy reserve *is* a valid next attempt, so this probe must
        stay tier-agnostic — narrowing it to primaries would abort requests
        while the reserve is still available.
        """
        backends = _make_backends(3, backup_indices={2})
        server = self._server(backends)
        now = 1000.0
        for idx in (0, 1):
            server._backend_health[idx].update({"healthy": False, "failed_at": now, "cooldown": 300})

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 10):
            assert server._any_healthy_backend() is True

    def test_any_healthy_backend_false_when_reserve_is_down_too(self):
        backends = _make_backends(3, backup_indices={2})
        server = self._server(backends)
        now = 1000.0
        for idx in range(3):
            server._backend_health[idx].update({"healthy": False, "failed_at": now, "cooldown": 300})

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 10):
            assert server._any_healthy_backend() is False

    def test_all_unhealthy_error_reports_the_reserve_too(self):
        """The 503 payload must account for every member, reserve included."""
        from kitty.bridge.server import AllBackendsUnhealthyError

        backends = _make_backends(3, backup_indices={2})
        server = self._server(backends)
        now = 1000.0
        for idx in range(3):
            server._backend_health[idx].update({"healthy": False, "failed_at": now, "cooldown": 300})

        with (
            patch("kitty.bridge.server.time.monotonic", return_value=now + 10),
            pytest.raises(AllBackendsUnhealthyError) as exc_info,
        ):
            server._get_next_backend()

        assert len(exc_info.value.backends) == 3
        assert {b["name"] for b in exc_info.value.backends} == {"profile-0", "profile-1", "profile-2"}

    def test_near_expiry_gamble_may_use_the_reserve(self):
        """With everything down, soonest-recovery wins over tier preference."""
        backends = _make_backends(2, backup_indices={1})
        server = self._server(backends)
        now = 1000.0
        # The primary is far from recovering; the reserve is nearly back.
        server._backend_health[0].update({"healthy": False, "failed_at": now, "cooldown": 3000})
        server._backend_health[1].update({"healthy": False, "failed_at": now, "cooldown": 30})

        with patch("kitty.bridge.server.time.monotonic", return_value=now + 20):
            assert server._get_next_backend()[1] == "key-1"

    def test_backup_and_default_flags_are_independent(self):
        """is_default governs which backend the CLI picks; backup governs tiering."""
        import uuid as _uuid

        from kitty.profiles.schema import Profile

        backends = _make_backends(2, backup_indices={1})
        reserve = Profile(
            name="reserve",
            provider="openai",
            model="model-1",
            auth_ref=str(_uuid.uuid4()),
            is_default=True,
            backup=True,
        )
        backends[1] = (backends[1][0], backends[1][1], reserve)
        server = self._server(backends)

        random.seed(42)
        assert {server._get_next_backend()[1] for _ in range(50)} == {"key-0"}


class TestBackupTierEndToEndFailover:
    """The user story: plans exhaust, the metered key takes over, no client error."""

    @pytest.mark.asyncio
    async def test_request_fails_over_to_reserve_after_primaries_return_429(self):
        backends = _make_backends(3, backup_indices={2})
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        primary_urls = [f"https://api{i}.example.com/v1/chat/completions" for i in range(2)]
        reserve_url = "https://api2.example.com/v1/chat/completions"

        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for url in primary_urls:
                    m.post(url, status=429, payload={"error": "rate limited"}, repeat=True)
                m.post(reserve_url, payload=UPSTREAM_RESPONSE, repeat=True)

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                    ) as resp,
                ):
                    assert resp.status == 200
                    data = await resp.json()

                assert data["choices"][0]["message"]["content"] == "Hello!"

                from yarl import URL

                assert m.requests.get(("POST", URL(reserve_url))), "reserve backend was never reached"
        finally:
            await server.stop_async()

        # Both plans were demoted; the reserve stayed healthy.
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is False
        assert server._backend_health[2]["healthy"] is True

    @pytest.mark.asyncio
    async def test_reserve_is_untouched_while_a_primary_succeeds(self):
        backends = _make_backends(2, backup_indices={1})
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
        )
        port = await server.start_async()
        reserve_url = "https://api1.example.com/v1/chat/completions"

        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api0.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE, repeat=True)
                m.post(reserve_url, payload=UPSTREAM_RESPONSE, repeat=True)

                async with aiohttp.ClientSession() as session:
                    for _ in range(5):
                        async with session.post(
                            f"http://127.0.0.1:{port}/v1/chat/completions",
                            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                        ) as resp:
                            assert resp.status == 200
                            await resp.json()

                from yarl import URL

                assert not m.requests.get(("POST", URL(reserve_url))), "reserve took traffic while the primary was up"
        finally:
            await server.stop_async()
