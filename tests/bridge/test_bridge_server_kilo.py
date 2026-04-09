"""Tests for bridge/server.py — Chat Completions pass-through handler (Kilo Code CLI)."""

import asyncio

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol


class StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol):
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

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
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
        return Exception(f"Upstream error {status_code}: {body}")


UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
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


def _make_server(model: str | None = None):
    adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
    provider = StubProvider()
    server = BridgeServer(adapter, provider, "test-key", model=model)
    return server


class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_registers_correct_endpoint(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                # /v1/chat/completions should exist
                async with session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "x", "messages": []},
                ) as resp:
                    # May get 200 or upstream error, but NOT 404
                    assert resp.status != 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_other_endpoints_return_404(self):
        """Only /v1/chat/completions is registered — not /v1/responses etc."""
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/v1/responses",
                    json={},
                ) as resp:
                    assert resp.status == 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_healthz_still_works(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                    assert resp.status == 200
                    assert await resp.json() == {"status": "ok"}
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=b"not json",
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    assert resp.status == 400
                    data = await resp.json()
                    assert "error" in data
        finally:
            await server.stop_async()


class TestChatCompletionsPassthrough:
    @pytest.mark.asyncio
    async def test_non_streaming_passthrough(self):
        """Non-streaming request forwarded verbatim after normalization."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    payload=UPSTREAM_RESPONSE,
                )
                async with aiohttp.ClientSession() as session:
                    request_body = {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json=request_body,
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data == UPSTREAM_RESPONSE
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_model_normalization_applied(self):
        """Profile model overrides the request model."""
        server = _make_server(model="override-model")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    payload=UPSTREAM_RESPONSE,
                )
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "original-model", "messages": []},
                    ) as resp:
                        assert resp.status == 200
                        # Verify the upstream request used the overridden model
                        url = "https://api.example.com/v1/chat/completions"
                        request_list = m.requests[("POST", aiohttp.client.URL(url))]
                        sent_json = request_list[0].kwargs["json"]
                        assert sent_json["model"] == "override-model"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_upstream_error_returns_error(self):
        """Upstream 500 returns error response to client."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=500,
                    payload={"error": {"message": "Internal error"}},
                )
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "x", "messages": []},
                    ) as resp:
                        assert resp.status == 500
                        data = await resp.json()
                        assert "error" in data
        finally:
            await server.stop_async()


class TestChatCompletionsStreaming:
    @pytest.mark.asyncio
    async def test_streaming_passthrough(self):
        """Streaming SSE chunks are forwarded verbatim to the client."""
        server = _make_server()
        port = await server.start_async()
        try:
            # Build SSE stream from upstream
            sse_chunks = [
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
                b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]

            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=200,
                    body=b"".join(sse_chunks),
                    content_type="text/event-stream",
                )
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "x", "messages": [], "stream": True},
                    ) as resp:
                        assert resp.status == 200
                        raw = await resp.read()
                        # Verify SSE content is forwarded
                        assert b'"content":"Hello"' in raw or b"Hello" in raw
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_stream_upstream_error(self):
        """Upstream error during streaming returns error SSE event."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=500,
                    body=b"Internal Server Error",
                )
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "x", "messages": [], "stream": True},
                    ) as resp:
                        assert resp.status == 200  # StreamResponse is already 200
                        raw = await resp.read()
                        assert b"error" in raw.lower()
        finally:
            await server.stop_async()
