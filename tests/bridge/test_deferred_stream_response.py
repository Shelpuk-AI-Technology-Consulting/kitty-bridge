"""Tests for deferred StreamResponse preparation in _stream_messages.

When all upstream backends fail BEFORE any content is streamed to the client,
the bridge should return a proper HTTP error response (non-200) rather than a
200-status SSE stream containing only an ``event: error`` payload.  This allows
Claude Code's compaction handler to properly detect API errors instead of
misinterpreting them as "response did not contain valid text content".
"""

import json
from unittest.mock import MagicMock

import aiohttp
from aiohttp import web

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stubs ────────────────────────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    """Minimal launcher for testing Messages API routes."""

    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API):
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


class _FakeUpstreamProvider(ProviderAdapter):
    """Provider that points at a fake upstream HTTP server."""

    def __init__(self, base_url: str):
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Fake upstream servers ────────────────────────────────────────────────────


async def _start_fake_upstream(handler) -> tuple[web.AppRunner, int]:
    """Start a minimal aiohttp server with the given handler on a free port."""
    app = web.Application()
    app.router.add_post("/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


async def _error_handler(request: web.Request) -> web.Response:
    """Upstream that always returns 503."""
    return web.json_response(
        {"error": {"message": "service unavailable"}},
        status=503,
    )


async def _rate_limit_handler(request: web.Request) -> web.Response:
    """Upstream that always returns 429."""
    return web.json_response(
        {"error": {"message": "rate limited"}},
        status=429,
    )


async def _success_handler(request: web.Request) -> web.StreamResponse:
    """Upstream that returns a successful streaming CC response."""
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
    )
    await resp.prepare(request)
    chunks = [
        b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"Summary text"},'
        b'"finish_reason":null}],"model":"test"}\n\n',
        b'data: {"id":"test","choices":[{"index":0,"delta":{},'
        b'"finish_reason":"stop"}],"model":"test","usage":null}\n\n',
        b"data: [DONE]\n\n",
    ]
    for chunk in chunks:
        await resp.write(chunk)
    await resp.write_eof()
    return resp


# ── Helpers ──────────────────────────────────────────────────────────────────

_STREAMING_MESSAGES_REQUEST = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "summarize this conversation"}],
    "max_tokens": 1024,
    "stream": True,
}


def _parse_sse_events(body: bytes) -> list[dict]:
    """Parse SSE event stream into a list of JSON data payloads."""
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


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDeferredStreamResponse:
    """Verify that _stream_messages defers StreamResponse preparation."""

    async def test_all_backends_fail_returns_http_error(self):
        """When upstream returns a non-200 error before any content is streamed,
        the bridge must return a proper HTTP error response (not 200 + SSE error).
        """
        upstream_runner, upstream_port = await _start_fake_upstream(_error_handler)
        try:
            provider = _FakeUpstreamProvider(f"http://127.0.0.1:{upstream_port}")
            server = BridgeServer(_StubLauncher(), provider, "test-key")
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_STREAMING_MESSAGES_REQUEST,
                    ) as resp,
                ):
                    # Must be a non-200 HTTP error, not a 200 SSE stream with error event
                    assert resp.status >= 400, (
                        f"Expected HTTP error status but got {resp.status}; "
                        "bridge should return HTTP-level error when no content was streamed"
                    )
                    data = await resp.json()
                    assert data["type"] == "error"
                    assert "error" in data
            finally:
                await server.stop_async()
        finally:
            await upstream_runner.cleanup()

    async def test_successful_stream_still_works(self):
        """A successful streaming request must still return 200 with proper SSE events."""
        upstream_runner, upstream_port = await _start_fake_upstream(_success_handler)
        try:
            provider = _FakeUpstreamProvider(f"http://127.0.0.1:{upstream_port}")
            server = BridgeServer(_StubLauncher(), provider, "test-key")
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_STREAMING_MESSAGES_REQUEST,
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    events = _parse_sse_events(body)
                    event_types = {e.get("type") for e in events}
                    assert "message_start" in event_types
                    assert "message_stop" in event_types
            finally:
                await server.stop_async()
        finally:
            await upstream_runner.cleanup()

    async def test_upstream_429_returns_http_429(self):
        """Rate-limit errors before content must surface as HTTP 429, not 200 + SSE error."""
        upstream_runner, upstream_port = await _start_fake_upstream(_rate_limit_handler)
        try:
            provider = _FakeUpstreamProvider(f"http://127.0.0.1:{upstream_port}")
            server = BridgeServer(_StubLauncher(), provider, "test-key")
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_STREAMING_MESSAGES_REQUEST,
                    ) as resp,
                ):
                    assert resp.status == 429, f"Expected HTTP 429 but got {resp.status}"
                    data = await resp.json()
                    assert data["type"] == "error"
            finally:
                await server.stop_async()
        finally:
            await upstream_runner.cleanup()

    async def test_balancing_all_backends_fail_returns_http_error(self):
        """In balancing mode, when every backend fails before streaming content,
        the bridge must return an HTTP error.
        """
        upstream_runner, upstream_port = await _start_fake_upstream(_error_handler)
        try:
            provider_a = _FakeUpstreamProvider(f"http://127.0.0.1:{upstream_port}")
            provider_b = _FakeUpstreamProvider(f"http://127.0.0.1:{upstream_port}")
            profile_a = MagicMock(spec=Profile)
            profile_a.name = "a"
            profile_a.model = "test-model"
            profile_a.provider_config = {}
            profile_b = MagicMock(spec=Profile)
            profile_b.name = "b"
            profile_b.model = "test-model"
            profile_b.provider_config = {}
            backends = [
                (provider_a, "key-a", profile_a),
                (provider_b, "key-b", profile_b),
            ]
            server = BridgeServer(
                _StubLauncher(),
                _FakeUpstreamProvider(f"http://127.0.0.1:{upstream_port}"),
                "test-key",
                backends=backends,
            )
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_STREAMING_MESSAGES_REQUEST,
                    ) as resp,
                ):
                    assert resp.status >= 400, (
                        f"Expected HTTP error but got {resp.status} in balancing mode"
                    )
                    data = await resp.json()
                    assert data["type"] == "error"
            finally:
                await server.stop_async()
        finally:
            await upstream_runner.cleanup()
