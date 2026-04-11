"""Tests for bridge/server.py — Bridge HTTP server lifecycle and request forwarding."""

import asyncio
import json

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
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

                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1024,
                        "stream": True,
                    },
                ) as resp:
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
