"""Tests for empty response retry and failover logic in the kitty bridge."""

from __future__ import annotations

import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

import kitty.bridge.server as _server_module
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
        assert (
            server._is_empty_cc_response(
                {
                    "choices": [{"message": {"content": ""}}],
                }
            )
            is True
        )

    def test_whitespace_is_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "choices": [{"message": {"content": "   "}}],
                }
            )
            is True
        )

    def test_no_choices_is_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response({"choices": []}) is True

    def test_tool_calls_not_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "choices": [{"message": {"content": None, "tool_calls": [{"id": "1"}]}}],
                }
            )
            is False
        )

    def test_text_content_not_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response(OK_CC_RESPONSE) is False

    # KBR-235 AC-6: the Messages-shaped arm agrees with Q14 D1, mirroring
    # PreambleHold._block_start_releases — content is decided by block type.

    def test_server_tool_use_block_is_not_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "server_tool_use", "id": "t1", "name": "web_search", "input": {}}],
                }
            )
            is False
        )

    def test_web_search_result_block_is_not_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "web_search_tool_result", "tool_use_id": "t1", "content": []}],
                }
            )
            is False
        )

    def test_unknown_block_type_is_not_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "some_future_block", "data": "x"}],
                }
            )
            is False
        )

    def test_tool_use_without_id_is_not_empty(self):
        """D1 decides by type: a tool_use block is content whether or not it carries an id."""
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "tool_use", "name": "f", "input": {}}],
                }
            )
            is False
        )

    def test_whitespace_text_block_is_not_empty(self):
        """The Messages arm pins the hold's predicate: any non-empty text, whitespace included."""
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "text", "text": " "}],
                }
            )
            is False
        )

    def test_thinking_only_is_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "thinking", "thinking": "reasoning"}],
                }
            )
            is True
        )

    def test_empty_text_block_is_empty(self):
        server = _make_server(1)
        assert (
            server._is_empty_cc_response(
                {
                    "type": "message",
                    "content": [{"type": "text", "text": ""}],
                }
            )
            is True
        )

    def test_no_content_key_is_empty(self):
        server = _make_server(1)
        assert server._is_empty_cc_response({"type": "message"}) is True


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
            for _ in range(6):
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
            for _ in range(6):
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
            """Select backends in the scripted order, without the weighted draw."""
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

        assert server._backend_health[0]["healthy"] is True
        assert server._backend_health[1]["healthy"] is True
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_streaming_failover_on_empty(self):
        """Non-streaming: first backend empty -> marks unhealthy -> other backend succeeds."""
        server = _make_balancing_server(2)
        # Force deterministic backend selection: backend-0 first, then backend-1
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
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

        assert server._backend_health[0]["healthy"] is True
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
            for _ in range(4):
                m.post("https://api0.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
                m.post("https://api1.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Upstream model returned an empty response" in body["content"][0]["text"]

        await server.stop_async()


class TestEmptyResponseFinalRetries:
    """Final retries before fallback on empty responses."""

    @pytest.mark.asyncio
    async def test_streaming_final_retry_recovers(self, monkeypatch):
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])
        server = _make_balancing_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        empty_stream = 'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        success_stream = (
            'data: {"choices":[{"delta":{"content":"Recovered"}}]}\n\n'
            'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api0.example.com/v1/chat/completions", body=empty_stream)
            m.post("https://api1.example.com/v1/chat/completions", body=empty_stream)
            m.post("https://api0.example.com/v1/chat/completions", body=empty_stream)
            m.post("https://api1.example.com/v1/chat/completions", body=empty_stream)
            m.post("https://api0.example.com/v1/chat/completions", body=success_stream)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                assert "Recovered" in content
                assert "Upstream model returned an empty response" not in content

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_streaming_final_retries_fallback(self, monkeypatch):
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])
        server = _make_balancing_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        empty_stream = 'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # 8 original attempts (4 per backend) + 2 final-retry attempts = 10 total
            for _ in range(10):
                m.post("https://api0.example.com/v1/chat/completions", body=empty_stream)
                m.post("https://api1.example.com/v1/chat/completions", body=empty_stream)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                assert "Upstream model returned an empty response" in content

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_streaming_final_retry_recovers(self, monkeypatch):
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])
        server = _make_balancing_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Normal loop: 2 backends, each returns empty once
            m.post("https://api0.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post("https://api1.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            # Final retries: OK responses
            m.post("https://api0.example.com/v1/chat/completions", payload=OK_CC_RESPONSE)
            m.post("https://api1.example.com/v1/chat/completions", payload=OK_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Hello!" in body["content"][0]["text"]

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_streaming_final_retries_fallback(self, monkeypatch):
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])
        server = _make_balancing_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for _ in range(6):
                m.post("https://api0.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
                m.post("https://api1.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert "Upstream model returned an empty response" in body["content"][0]["text"]

        await server.stop_async()


# -- KBR-235: a stream with no content chunk and no finish_reason chunk ------

_DONE_ONLY_STREAM = "data: [DONE]\n\n"

# One content delta as the final, unterminated line: only the tail-flush path reads it.
_UNTERMINATED_TAIL_STREAM = 'data: {"choices":[{"delta":{"content":"Tail"}}]}'


def _posts(mocked: aioresponses) -> int:
    """Count the upstream POSTs ``aioresponses`` recorded.

    Args:
        mocked: The active ``aioresponses`` context.

    Returns:
        The number of POSTs the bridge made to the mocked upstreams.
    """
    return sum(len(calls) for (method, _), calls in mocked.requests.items() if method == "POST")


class TestNoFinishEmptyStream:
    """A translated stream with no content and no finish_reason takes the empty ladder (KBR-235)."""

    @pytest.fixture(autouse=True)
    def _no_retry_delays(self, monkeypatch):
        """Zero every retry delay; the tests count attempts, never time them."""
        monkeypatch.setattr(_server_module, "_BACKOFF_BASE", 0.0)
        monkeypatch.setattr(_server_module, "_EMPTY_RETRY_DELAYS", [0.0, 0.0])
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    async def test_zero_bytes_are_retried_and_only_the_retry_reaches_the_client(self):
        """A 200 with an empty body is an empty reply: retried, nothing of it streamed."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", body="")
            m.post(
                "https://api.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                posts = _posts(m)

        assert posts == 2, "the zero-byte attempt must be retried"
        assert "Hello" in content
        assert content.count("event: message_start") == 1, "nothing of the empty attempt may reach the client"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_done_only_stream_is_retried_and_only_the_retry_reaches_the_client(self):
        """A 200 whose stream is only ``[DONE]`` is an empty reply: retried, nothing of it streamed."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", body=_DONE_ONLY_STREAM)
            m.post(
                "https://api.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                posts = _posts(m)

        assert posts == 2, "the [DONE]-only attempt must be retried"
        assert "Hello" in content
        assert content.count("event: message_start") == 1, "nothing of the empty attempt may reach the client"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_balancing_failover_delivers_only_the_retry(self):
        """On a balancing pool a no-finish empty attempt fails over and writes nothing."""
        server = _make_balancing_server(2)
        # Force deterministic backend selection: backend-0, then backend-1, then backend-0 again.
        _pick = iter([0, 1, 0])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
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

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Backend-0: [DONE]-only. Backend-1: zero bytes. Backend-0 again: content.
            m.post("https://api0.example.com/v1/chat/completions", body=_DONE_ONLY_STREAM)
            m.post("https://api1.example.com/v1/chat/completions", body="")
            m.post(
                "https://api0.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{"content":"World"}}]}\n\n'
                'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                posts = _posts(m)

        assert posts == 3, "two empty attempts, then the failover that carried content"
        assert "World" in content
        assert content.count("event: message_start") == 1, "nothing of the empty attempts may reach the client"
        assert server._backend_health[0]["healthy"] is True, "the empty ladder rotates, it does not quarantine"
        assert server._backend_health[1]["healthy"] is True
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_exhaustion_returns_the_d4_error_and_marks_nothing(self):
        """Every attempt empty ends in the D4 502; no health transition, no counted completion."""
        server = _make_balancing_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api0.example.com/v1/chat/completions", body=_DONE_ONLY_STREAM, repeat=True)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 502
                body = await resp.json()
                posts = _posts(m)

        assert posts == 6, "the single-backend streaming ladder budget is (_MAX_RETRIES + 1) + final delays"
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"
        assert body["error"]["reason"] == "empty_response"
        assert body["error"]["message"] == _server_module._NATIVE_EMPTY_REPLY_MESSAGE
        assert server._backend_health[0]["healthy"] is True, "exhaustion marks nothing"
        assert server._session_stats()["attempts"] == 6, "each empty attempt is counted; none is a completion"
        assert all(
            record["completions"] == 0 for record in server._session_stats()["models_served"].values()
        ), "a discarded empty attempt is not a completion"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_balancing_all_backends_empty_returns_the_d4_error(self):
        """Every backend empty on every attempt ends in the D4 502 with the full 10-post budget."""
        server = _make_balancing_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(f"https://api{i}.example.com/v1/chat/completions", body=_DONE_ONLY_STREAM, repeat=True)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 502
                body = await resp.json()
                posts = _posts(m)

        assert posts == 10, "(_MAX_RETRIES + 1) * 2 backends + final delays"
        assert body["error"]["reason"] == "empty_response"
        assert server._backend_health[0]["healthy"] is True, "exhaustion marks nothing"
        assert server._backend_health[1]["healthy"] is True, "exhaustion marks nothing"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_content_in_an_unterminated_final_line_is_delivered_and_finalized(self):
        """Tail-flush content counts as a write: delivered once, finalized, never retried."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", body=_UNTERMINATED_TAIL_STREAM)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                posts = _posts(m)

        assert posts == 1, "content reached the client; a retry would duplicate it"
        assert "Tail" in content
        assert "event: message_stop" in content, "the truncated stream must still be finalized"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_an_in_stream_error_still_wins_over_the_empty_gate(self):
        """An error chunk on an otherwise-empty stream takes the error path, never the D4 return."""
        server = _make_balancing_server(2)
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
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

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Backend-0: an in-stream error with no content. Backend-1: content.
            m.post(
                "https://api0.example.com/v1/chat/completions",
                body='data: {"error": {"type": "overloaded_error", "message": "overloaded"}}\n\ndata: [DONE]\n\n',
            )
            m.post(
                "https://api1.example.com/v1/chat/completions",
                body='data: {"choices":[{"delta":{"content":"World"}}]}\n\n'
                'data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                content = await resp.text()
                posts = _posts(m)

        assert posts == 2, "the error failed over; the D4 path must not absorb it"
        assert "World" in content
        assert server._backend_health[0]["healthy"] is False, "an in-stream error quarantines"
        assert server._backend_health[1]["healthy"] is True
        await server.stop_async()


class TestD1ContentDelivery:
    """D1 content on a Messages-shaped reply is delivered on the first attempt (KBR-235)."""

    @pytest.fixture(autouse=True)
    def _no_retry_delays(self, monkeypatch):
        """Zero every retry delay; the tests count attempts, never time them."""
        monkeypatch.setattr(_server_module, "_EMPTY_RETRY_DELAYS", [0.0, 0.0])
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    async def test_server_tool_use_reply_is_delivered_without_retry(self):
        """A reply of only a server_tool_use block is content: one post, passed through verbatim."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        reply = {
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "server_tool_use", "id": "t1", "name": "web_search", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", payload=reply)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                posts = _posts(m)

        assert posts == 1, "D1 content is not retried"
        assert body["content"][0]["type"] == "server_tool_use", "delivered verbatim, no fallback text injected"
        assert "Upstream model returned an empty response" not in str(body)
        await server.stop_async()


class TestD3TruncationBeforeContent:
    """A truncation before content ends the ladder at once with the D3 400 (KBR-235)."""

    @pytest.fixture(autouse=True)
    def _no_retry_delays(self, monkeypatch):
        """Zero every retry delay; the tests count attempts, never time them."""
        monkeypatch.setattr(_server_module, "_EMPTY_RETRY_DELAYS", [0.0, 0.0])
        monkeypatch.setattr(_server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    @staticmethod
    def _truncating_reply(stop_reason: str) -> dict:
        """Build a Messages-shaped reply stopped before any content.

        Args:
            stop_reason: The upstream stop reason to carry.

        Returns:
            A Messages-shaped response with no D1-content blocks.
        """
        return {
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
    @pytest.mark.asyncio
    async def test_truncation_ends_the_ladder_at_once_with_a_400(self, stop_reason):
        """No retry, no delivery of the blank reply: one post, then the D3 400."""
        server = _make_server(1)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", payload=self._truncating_reply(stop_reason))

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                body = await resp.json()
                posts = _posts(m)
                assert resp.status == 400

        assert posts == 1, "no retry can improve a truncation; the ladder ends on that attempt"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["reason"] == f"{stop_reason}_before_content"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_truncation_after_an_empty_attempt_still_ends_the_ladder(self):
        """An earlier retryable empty does not restart the ladder once the truncation arrives."""
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
            m.post(
                "https://api.example.com/v1/chat/completions",
                payload=self._truncating_reply("max_tokens"),
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 400
                body = await resp.json()
                posts = _posts(m)

        assert posts == 2, "the empty attempt was retried; the truncation ended the ladder"
        assert body["error"]["reason"] == "max_tokens_before_content"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_truncation_in_the_balancing_loop_ends_it(self):
        """The balancing failover loop stops at once on a truncating reply."""
        server = _make_balancing_server(2)
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
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
            # Main balancing loop: backend-0 empty, backend-1 truncating.
            m.post("https://api0.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post(
                "https://api1.example.com/v1/chat/completions",
                payload=self._truncating_reply("max_tokens"),
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 400
                body = await resp.json()
                posts = _posts(m)

        assert posts == 2, "the failover loop ended on the truncation with no further post"
        assert body["error"]["reason"] == "max_tokens_before_content"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_truncation_after_the_compaction_retry_ends_the_ladder(self):
        """The inline same-backend retry after a tighter compaction stops on a truncation."""
        server = _make_balancing_server(1)
        # Force the oversized path deterministically: the compact-retry only
        # fires for genuinely large requests (see test_stage11_oversized.py).
        server._is_oversized_request = lambda cc_request: True  # noqa: E731
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # First attempt: context-too-large, triggering the tighter compaction
            # and the inline retry on the same backend. That retry truncates.
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=400,
                payload={"error": {"code": "1261", "message": "prompt exceeds max length"}},
            )
            m.post(
                "https://api0.example.com/v1/chat/completions",
                payload=self._truncating_reply("max_tokens"),
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 400
                body = await resp.json()
                posts = _posts(m)

        assert posts == 2, "the compaction retry ended on the truncation, with no further attempt"
        assert body["error"]["reason"] == "max_tokens_before_content"
        await server.stop_async()

    @pytest.mark.asyncio
    async def test_truncation_in_the_final_retry_loop_ends_it(self):
        """The balancing final-retry loop stops at once on a truncating reply."""
        server = _make_balancing_server(2)
        _pick = iter([0, 1, 1, 1])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
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
            # Main balancing loop: both backends empty. First final retry: truncating.
            m.post("https://api0.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post("https://api1.example.com/v1/chat/completions", payload=EMPTY_CC_RESPONSE)
            m.post(
                "https://api1.example.com/v1/chat/completions",
                payload=self._truncating_reply("max_tokens"),
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 400
                body = await resp.json()
                posts = _posts(m)

        assert posts == 3, "the final-retry loop ended on the truncation, with no further delay or post"
        assert body["error"]["reason"] == "max_tokens_before_content"
        await server.stop_async()
