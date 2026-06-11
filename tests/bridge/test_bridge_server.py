"""Tests for bridge/server.py — Bridge HTTP server lifecycle and request forwarding."""

import asyncio
import json

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import _CLIENT_MAX_SIZE, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stub adapters for testing ───────────────────────────────────────────────


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
        request = {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}
        for key in ("tools", "temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Helpers ─────────────────────────────────────────────────────────────────


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


def _make_responses_request():
    return {"model": "test-model", "input": [{"role": "user", "content": "hi"}]}


def _make_messages_request():
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
    }


def _parse_sse_events(body: bytes) -> list[dict]:
    events: list[dict] = []
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


# ── Tests ───────────────────────────────────────────────────────────────────


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_returns_port(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        try:
            port = await server.start_async()
            assert isinstance(port, int)
            assert port > 0
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_start_stop_cycle(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        assert port > 0
        await server.stop_async()
        # Second cycle should work
        port2 = await server.start_async()
        assert port2 > 0
        await server.stop_async()


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthz_returns_ok(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{port}/healthz") as resp,
            ):
                assert resp.status == 200
                data = await resp.json()
                assert data == {"status": "ok"}
        finally:
            await server.stop_async()


class TestEarlyErrorConnectionClose:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("protocol", "path"),
        [
            (BridgeProtocol.RESPONSES_API, "/v1/responses"),
            (BridgeProtocol.MESSAGES_API, "/v1/messages"),
            (BridgeProtocol.CHAT_COMPLETIONS_API, "/v1/chat/completions"),
            (BridgeProtocol.GEMINI_API, "/v1beta/models/test-model:generateContent"),
        ],
    )
    async def test_invalid_json_closes_connection(self, protocol, path):
        adapter = StubLauncher(protocol)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}{path}",
                    data=b'{"model":"test-model","messages":[',
                    headers={"Content-Type": "application/json"},
                ) as resp,
            ):
                assert resp.status == 400
                assert resp.headers.get("Connection") == "close"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_oversized_body_closes_connection(self):
        adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=b'{"payload":"' + (b"x" * (_CLIENT_MAX_SIZE + 1)) + b'"}',
                    headers={"Content-Type": "application/json"},
                ) as resp,
            ):
                assert resp.status == 400
                assert resp.headers.get("Connection") == "close"
        finally:
            await server.stop_async()


class TestProtocolEndpoints:
    @pytest.mark.asyncio
    async def test_responses_api_registers_correct_endpoint(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{port}/v1/messages", json={}) as resp,
            ):
                # /v1/messages should return 404 for responses protocol
                assert resp.status == 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_api_registers_correct_endpoint(self):
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{port}/v1/responses", json={}) as resp,
            ):
                # /v1/responses should return 404 for messages protocol
                assert resp.status == 404
        finally:
            await server.stop_async()


class TestSyncRequestForwarding:
    @pytest.mark.asyncio
    async def test_responses_api_forwarding(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    payload=UPSTREAM_RESPONSE,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_make_responses_request(),
                    ) as resp,
                ):
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["object"] == "response"
                    assert data["status"] == "completed"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_api_forwarding(self):
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    payload=UPSTREAM_RESPONSE,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_make_messages_request(),
                    ) as resp,
                ):
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["type"] == "message"
                    assert data["role"] == "assistant"
        finally:
            await server.stop_async()


class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, status=429)
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_make_responses_request(),
                    ) as resp,
                ):
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "completed"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, status=500)
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_make_responses_request(),
                    ) as resp,
                ):
                    assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, status=401, payload={"error": {"message": "unauthorized"}})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_make_responses_request(),
                    ) as resp,
                ):
                    assert resp.status == 401
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                for _ in range(4):
                    m.post(url, status=500, payload={"error": {"message": "internal error"}})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_make_responses_request(),
                    ) as resp,
                ):
                    assert resp.status == 500
        finally:
            await server.stop_async()


class TestConcurrentRequests:
    @pytest.mark.asyncio
    async def test_two_simultaneous_posts(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    tasks = [
                        session.post(
                            f"http://127.0.0.1:{port}/v1/responses",
                            json=_make_responses_request(),
                        )
                        for _ in range(2)
                    ]
                    responses = await asyncio.gather(*tasks)
                    for resp in responses:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["status"] == "completed"
                        resp.close()
        finally:
            await server.stop_async()


class TestStreamConnectionReset:
    """Server must not crash when client disconnects during/after streaming."""

    @pytest.mark.asyncio
    async def test_responses_stream_handles_client_disconnect_gracefully(self):
        """ClientConnectionResetError during write_eof must not propagate as unhandled error.

        MiniMax streams complete successfully but Codex CLI may close the connection
        before the server calls write_eof(). This race condition must be caught silently.
        """
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            upstream_chunks = [
                (
                    b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"Hi"},'
                    b'"finish_reason":null}],"model":"test"}\n\n'
                ),
                (
                    b'data: {"id":"test","choices":[{"index":0,"delta":{},'
                    b'"finish_reason":"stop"}],"model":"test","usage":null}\n\n'
                ),
                b"data: [DONE]\n\n",
            ]

            with aioresponses(passthrough=["http://127.0.0.1"]) as m:

                async def _streaming_upstream(url, **kwargs):
                    """Mock upstream that returns SSE chunks."""
                    resp = aiohttp.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
                    writer = await resp.prepare(aiohttp.test_utils.make_mocked_request("POST", "/"))
                    for chunk in upstream_chunks:
                        await writer.write(chunk)
                    await writer.write_eof()
                    return resp

                # Use a simpler mock: return a streaming response
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(upstream_chunks),
                    headers={"Content-Type": "text/event-stream"},
                )

                # Read the full response — client will disconnect after reading
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={"model": "test-model", "input": [{"role": "user", "content": "hi"}], "stream": True},
                    ) as resp,
                ):
                    assert resp.status == 200
                    # Read all SSE events
                    body = await resp.read()
                    assert b"response.completed" in body

        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_stream_handles_client_disconnect_gracefully(self):
        """Messages API streaming must also handle client disconnect during cleanup."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            upstream_chunks = [
                (
                    b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"Hi"},'
                    b'"finish_reason":null}],"model":"test"}\n\n'
                ),
                (
                    b'data: {"id":"test","choices":[{"index":0,"delta":{},'
                    b'"finish_reason":"stop"}],"model":"test","usage":null}\n\n'
                ),
                b"data: [DONE]\n\n",
            ]

            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(upstream_chunks),
                    headers={"Content-Type": "text/event-stream"},
                )

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                            "stream": True,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert len(body) > 0

        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_responses_stream_upstream_error_still_emits_completed(self):
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=500,
                    payload={"error": {"code": "1234", "message": "Internal network failure"}},
                )

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={"model": "test-model", "input": [{"role": "user", "content": "hi"}], "stream": True},
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()

                assert b"response.completed" in body
                sse_events = _parse_sse_events(body)
                completed = [e for e in sse_events if e.get("type") == "response.completed"]
                error_events = [e for e in sse_events if e.get("type") == "error"]
                assert len(error_events) == 1
                assert len(completed) == 1
                assert completed[0]["response"]["status"] == "incomplete"
                seqs = [e["sequence_number"] for e in sse_events if "sequence_number" in e]
                assert seqs == sorted(seqs)
        finally:
            await server.stop_async()


class _NormalizingProvider(StubProvider):
    """Provider that strips a 'stub/' prefix for testing model normalization."""

    def normalize_model_name(self, model: str) -> str:
        if model.lower().startswith("stub/"):
            return model[5:]
        return model


class TestModelNormalization:
    @pytest.mark.asyncio
    async def test_model_normalization_responses_api_sync(self):
        """Non-streaming Responses API request with prefixed model name."""
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = _NormalizingProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": "stub/test-model",
                            "input": [{"role": "user", "content": "hi"}],
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    # Check that the upstream received the normalized model name
                    request_list = m.requests[("POST", aiohttp.client.URL(url))]
                    assert len(request_list) == 1
                    sent_json = request_list[0].kwargs.get("json")
                    assert sent_json is not None
                    assert sent_json["model"] == "test-model"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_model_normalization_messages_api(self):
        """Non-streaming Messages API request with prefixed model name."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = _NormalizingProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "stub/test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    request_list = m.requests[("POST", aiohttp.client.URL(url))]
                    assert len(request_list) == 1
                    sent_json = request_list[0].kwargs.get("json")
                    assert sent_json is not None
                    assert sent_json["model"] == "test-model"
        finally:
            await server.stop_async()


class TestModelOverride:
    """BridgeServer must override the agent's model with the profile model when provided."""

    @pytest.mark.asyncio
    async def test_model_override_messages_api(self):
        """Agent sends 'claude-sonnet-4-20250514' but bridge overrides with profile model."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key", model="minimax-m2.7")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    request_list = m.requests[("POST", aiohttp.client.URL(url))]
                    sent_json = request_list[0].kwargs["json"]
                    assert sent_json["model"] == "minimax-m2.7"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_model_override_responses_api(self):
        """Agent sends a model name but bridge overrides with profile model."""
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key", model="gpt-4o")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": "o3",
                            "input": [{"role": "user", "content": "hi"}],
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    request_list = m.requests[("POST", aiohttp.client.URL(url))]
                    sent_json = request_list[0].kwargs["json"]
                    assert sent_json["model"] == "gpt-4o"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_model_override_with_normalization(self):
        """Override applies before normalization — profile model gets normalized."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = _NormalizingProvider()
        server = BridgeServer(adapter, provider, "test-key", model="stub/minimax-m2.7")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    request_list = m.requests[("POST", aiohttp.client.URL(url))]
                    sent_json = request_list[0].kwargs["json"]
                    # Profile model "stub/minimax-m2.7" should be normalized to "minimax-m2.7"
                    assert sent_json["model"] == "minimax-m2.7"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_no_override_when_model_is_none(self):
        """When model is not provided, the agent's model passes through unchanged."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key", model=None)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                url = "https://api.example.com/v1/chat/completions"
                m.post(url, payload=UPSTREAM_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    request_list = m.requests[("POST", aiohttp.client.URL(url))]
                    sent_json = request_list[0].kwargs["json"]
                    # No override — agent model passes through
                    assert sent_json["model"] == "claude-sonnet-4-20250514"
        finally:
            await server.stop_async()


# ── Cloudflare detection ──────────────────────────────────────────────────


class TestCloudflareDetection:
    def test_detects_cloudflare_block(self) -> None:
        assert BridgeServer._is_cloudflare_block(403, "<html>cf-mitigated: challenge</html>") is True

    def test_rejects_non_cloudflare_403(self) -> None:
        assert BridgeServer._is_cloudflare_block(403, "<html>Forbidden</html>") is False

    def test_translate_upstream_error_cloudflare(self) -> None:
        msg = BridgeServer._translate_upstream_error(
            403,
            "<html>cf-browser-verification window._cf_chl_opt = {};</html>",
        )
        lower = msg.lower()
        assert "cloudflare bot detection" in lower
        assert "not an api key problem" in lower

    def test_should_retry_stream_rejects_cloudflare(self) -> None:
        assert BridgeServer._should_retry_stream(403, "<html>cf-mitigated: challenge</html>") is False

    def test_should_retry_stream_keeps_retryable_errors(self) -> None:
        assert BridgeServer._should_retry_stream(503, "service unavailable") is True


class TestCloudflareCooldownSoftening:
    """First-hit Cloudflare blocks should use a short transient cooldown and
    skip family-level escalation; only repeat hits should escalate."""

    def _make_server_with_backend(self) -> BridgeServer:
        """Build a minimal BridgeServer with one backend for unit-testing
        ``_mark_backend_unhealthy`` / ``_mark_backend_healthy``."""
        import uuid

        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        profile = Profile(name="cf-test", provider="openai", model="m", auth_ref=str(uuid.uuid4()))
        return BridgeServer(
            adapter,
            provider,
            "test-key",
            backends=[(provider, "test-key", profile)],
            backend_cooldown=300,
        )

    def test_first_hit_uses_short_cooldown_and_skips_family_abuse(self) -> None:
        from kitty.bridge.server import _CLOUDFLARE_FIRST_HIT_COOLDOWN

        server = self._make_server_with_backend()
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")

        health = server._backend_health[0]
        assert health["healthy"] is False
        assert health["cloudflare_error_count"] == 1
        assert health["cooldown"] == _CLOUDFLARE_FIRST_HIT_COOLDOWN
        # No family-level escalation on the first hit.
        assert server._family_cooldown == {}

    def test_second_hit_escalates_to_family_cooldown(self) -> None:
        server = self._make_server_with_backend()
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")

        health = server._backend_health[0]
        assert health["cloudflare_error_count"] == 2
        # Single-backend cooldown is capped, but family-abuse should now be active.
        family = server._get_backend_family(0)
        assert family in server._family_cooldown
        assert server._family_cooldown[family]["abuse_count"] == 1

    def test_mark_backend_healthy_resets_cloudflare_counter(self) -> None:
        server = self._make_server_with_backend()
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        assert server._backend_health[0]["cloudflare_error_count"] == 1

        server._mark_backend_healthy(0)
        assert server._backend_health[0]["cloudflare_error_count"] == 0
        assert server._backend_health[0]["healthy"] is True


class TestCloudflareDecisionHelper:
    """``_decide_cloudflare_action`` controls the per-attempt action the
    streaming sites take on a Cloudflare 403."""

    def _make_server(self, n_backends: int = 2) -> BridgeServer:
        import uuid

        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        profiles = [
            Profile(name=f"cf-b{i}", provider="openai", model="m", auth_ref=str(uuid.uuid4()))
            for i in range(n_backends)
        ]
        backends = [(provider, f"k{i}", p) for i, p in enumerate(profiles)]
        return BridgeServer(
            adapter,
            provider,
            "k0",
            backends=backends,
            backend_cooldown=300,
        )

    def test_first_attempt_returns_retry_same(self) -> None:
        server = self._make_server(n_backends=1)
        server._current_backend_idx = 0
        cf_retried: set[int] = set()

        action = server._decide_cloudflare_action(
            attempt=0,
            max_attempts=4,
            cf_retried=cf_retried,
        )
        assert action == "retry_same"
        assert 0 in cf_retried
        # No marking happens on retry_same.
        assert server._backend_health[0]["healthy"] is True

    def test_second_attempt_after_retry_marks_unhealthy_and_fails_over(self) -> None:
        server = self._make_server(n_backends=2)
        server._current_backend_idx = 0
        cf_retried: set[int] = {0}  # backend 0 already retried this request

        action = server._decide_cloudflare_action(
            attempt=1,
            max_attempts=8,
            cf_retried=cf_retried,
        )
        assert action == "failover"
        # Marks backend 0 unhealthy with cloudflare failure kind.
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[0]["cloudflare_error_count"] == 1
        # Backend 1 still healthy and reachable.
        assert server._any_healthy_backend() is True

    def test_surface_when_no_alternates_and_already_retried(self) -> None:
        server = self._make_server(n_backends=1)
        server._current_backend_idx = 0
        cf_retried: set[int] = {0}

        action = server._decide_cloudflare_action(
            attempt=1,
            max_attempts=2,
            cf_retried=cf_retried,
        )
        assert action == "surface"
        assert server._backend_health[0]["healthy"] is False


class TestCloudflareRegression:
    @pytest.mark.asyncio
    async def test_non_streaming_cloudflare_403_returns_cloudflare_message(self):
        """Non-streaming upstream Cloudflare HTML must surface the Cloudflare message, not auth error."""
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=403,
                    body="<html>cf-browser-verification window._cf_chl_opt = {};</html>",
                    headers={"Content-Type": "text/html"},
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_make_responses_request(),
                    ) as resp,
                ):
                    assert resp.status == 403
                    data = await resp.json()
                    message = data["error"]["message"].lower()
                    assert "cloudflare bot detection" in message
                    assert "api key is invalid" not in message
        finally:
            await server.stop_async()


class TestCustomTransportStreamDisconnect:
    """Regression tests for custom-transport streaming when sr.write() fails.

    When StreamResponse.write() raises (client disconnected mid-stream),
    the handler must catch it silently — NOT propagate to _access_log_middleware
    as an unhandled "Handler error".
    """

    @pytest.mark.asyncio
    async def test_messages_custom_transport_emit_handles_disconnect(self):
        """sr.write() failures during custom-transport SSE emit must not propagate.

        The _stream_messages custom-transport success path writes parsed SSE
        events via sr.write(). If the client disconnects, these writes raise
        ConnectionResetError which must be caught, not leaked as Handler errors.
        """
        from unittest.mock import patch

        sse_data = (
            b'data: {"type":"response.output_text.delta","delta":"Hello "}\n\n'
            b'data: {"type":"response.completed","response":{"id":"resp_1",'
            b'"object":"response","status":"completed",'
            b'"output":[{"type":"message","role":"assistant","content":'
            b'[{"type":"output_text","text":"Hello "}]}],'
            b'"model":"test","usage":{"input_tokens":5,"output_tokens":2}}}\n\n'
        )

        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = _CustomTransportProvider(raw_chunks=[sse_data])
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()

        async def _failing_write(_self, _data):
            raise ConnectionResetError("Cannot write to closing transport")

        try:
            with patch.object(aiohttp.web.StreamResponse, "write", _failing_write):
                async with aiohttp.ClientSession() as session:
                    resp = await session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                            "stream": True,
                        },
                    )
                    assert resp.status == 200
                    resp.close()
            await asyncio.sleep(0.1)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_custom_transport_error_handles_disconnect(self):
        """Chat Completions custom-transport error writes must not leak exceptions.

        When the custom transport raises and sr.write() also fails during
        error emit, the handler must catch the write failure silently.
        """
        from unittest.mock import patch

        adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
        provider = _CustomTransportProvider(raise_on_stream=RuntimeError("rate limited"))
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()

        async def _failing_write(_self, _data):
            raise ConnectionResetError("Cannot write to closing transport")

        try:
            with patch.object(aiohttp.web.StreamResponse, "write", _failing_write):
                async with aiohttp.ClientSession() as session:
                    resp = await session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                            "stream": True,
                        },
                    )
                    assert resp.status == 200
                    resp.close()
            await asyncio.sleep(0.1)
        finally:
            await server.stop_async()


class _CustomTransportProvider(ProviderAdapter):
    """Provider that simulates custom transport for testing."""

    def __init__(self, raw_chunks: list[bytes] | None = None, raise_on_stream: Exception | None = None):
        self._raw_chunks = raw_chunks or []
        self._raise_on_stream = raise_on_stream

    @property
    def provider_type(self) -> str:
        return "custom_transport_test"

    @property
    def default_base_url(self) -> str:
        return "https://custom.example.com/v1"

    @property
    def use_custom_transport(self) -> bool:
        return True

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")

    async def stream_request(self, cc_request: dict, write) -> None:
        """Stream raw SSE chunks, or raise if configured to fail."""
        if self._raise_on_stream:
            raise self._raise_on_stream
        for chunk in self._raw_chunks:
            await write(chunk)


class TestCustomTransportErrorPropagation:
    """Test that custom-transport errors propagate the correct HTTP status and error type."""

    @pytest.mark.asyncio
    async def test_messages_429_returns_rate_limit_error(self):
        from kitty.providers.base import ProviderError

        err = ProviderError("OpenAI subscription rate limited: quota exceeded")
        err.http_status = 429

        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = _CustomTransportProvider(raise_on_stream=err)
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1024,
                        "stream": True,
                    },
                )
                assert resp.status == 429
                body = await resp.json()
                assert body["type"] == "error"
                assert body["error"]["type"] == "rate_limit_error"
                assert "rate limited" in body["error"]["message"].lower()
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_401_returns_authentication_error(self):
        from kitty.providers.base import ProviderError

        err = ProviderError("OpenAI subscription auth failed. Please re-authenticate.")
        err.http_status = 401

        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = _CustomTransportProvider(raise_on_stream=err)
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1024,
                        "stream": True,
                    },
                )
                assert resp.status == 401
                body = await resp.json()
                assert body["type"] == "error"
                assert body["error"]["type"] == "authentication_error"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_generic_error_returns_502(self):
        """Non-ProviderError exceptions still return 502 with api_error."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = _CustomTransportProvider(raise_on_stream=RuntimeError("unexpected"))
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1024,
                        "stream": True,
                    },
                )
                assert resp.status == 502
                body = await resp.json()
                assert body["error"]["type"] == "api_error"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_map_provider_error_helper(self):
        """Unit test for _map_provider_error static method."""
        from kitty.providers.base import ProviderError

        assert BridgeServer._map_provider_error(RuntimeError("x")) == (502, "api_error")
        assert BridgeServer._map_provider_error(ProviderError("x")) == (502, "api_error")

        err429 = ProviderError("x")
        err429.http_status = 429
        assert BridgeServer._map_provider_error(err429) == (429, "rate_limit_error")

        err401 = ProviderError("x")
        err401.http_status = 401
        assert BridgeServer._map_provider_error(err401) == (401, "authentication_error")

        err403 = ProviderError("x")
        err403.http_status = 403
        assert BridgeServer._map_provider_error(err403) == (403, "authentication_error")

        err500 = ProviderError("x")
        err500.http_status = 500
        assert BridgeServer._map_provider_error(err500) == (500, "api_error")

    @pytest.mark.asyncio
    async def test_provider_error_failure_kind_helper(self):
        """Unit test for _provider_error_failure_kind static method."""
        from kitty.providers.base import ProviderError

        assert BridgeServer._provider_error_failure_kind(RuntimeError("x")) == "hard"

        err429 = ProviderError("x")
        err429.http_status = 429
        assert BridgeServer._provider_error_failure_kind(err429) == "rate_limit"

        err401 = ProviderError("x")
        err401.http_status = 401
        assert BridgeServer._provider_error_failure_kind(err401) == "auth"

        err_cf = ProviderError("x")
        err_cf.is_cloudflare = True
        assert BridgeServer._provider_error_failure_kind(err_cf) == "cloudflare"

        assert BridgeServer._provider_error_failure_kind(ConnectionResetError("x")) == "transport"
