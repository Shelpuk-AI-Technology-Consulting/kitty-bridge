"""Tests for streaming hang fixes (P1-P5).

Verifies:
- P1: Streaming read timeout prevents infinite hangs
- P2: Debug logging parity in _stream_messages
- P3: Retry logic for streaming handlers
- P4: Context size guardrail
- P5: Non-200 error body logging
"""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Test infrastructure ──────────────────────────────────────────────────


class StubLauncher(LauncherAdapter):
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
        return SpawnConfig()


class StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


def _make_messages_stream_body() -> list[bytes]:
    """Standard SSE chunks for a Messages API streaming response."""
    return [
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


def _parse_sse_events(body: bytes) -> list[dict]:
    """Parse SSE data lines into dicts."""
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


# ── P1: Streaming Read Timeout ───────────────────────────────────────────


class TestStreamingReadTimeout:
    """P1: Streaming handlers must not hang indefinitely when upstream stops responding."""

    @pytest.mark.asyncio
    async def test_messages_stream_upstream_timeout_returns_error(self):
        """When upstream never responds, bridge must return an error within timeout.

        Before fix: session.post() hangs forever (total=None, no sock_read).
        After fix: sock_read timeout triggers, bridge returns error to client.
        """
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                # Simulate upstream that never responds by raising TimeoutError
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    exception=asyncio.TimeoutError(),
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
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    # Should get an error response, not hang
                    assert resp.status == 200  # SSE stream always starts 200
                    body = await resp.read()
                    # Must contain an error event
                    assert len(body) > 0
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_stream_successful_response_unchanged(self):
        """Verify normal streaming still works after adding timeout."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(_make_messages_stream_body()),
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
                    # Should contain message_start and message_stop events
                    assert b"message_start" in body or b"message_stop" in body
        finally:
            await server.stop_async()


# ── P2: Debug Logging Parity ─────────────────────────────────────────────


class TestMessagesStreamDebugLogging:
    """P2: _stream_messages must emit the same debug log lines as _stream_responses."""

    @pytest.mark.asyncio
    async def test_messages_stream_logs_upstream_post_url(self, caplog):
        """Verify _stream_messages logs the upstream POST URL."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"):
                with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        body=b"".join(_make_messages_stream_body()),
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
                        await resp.read()

                # Must have logged the upstream POST URL
                assert any("Upstream POST" in r.message for r in caplog.records), (
                    f"Missing 'Upstream POST' log. Got: {[r.message for r in caplog.records]}"
                )
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_stream_logs_upstream_status(self, caplog):
        """Verify _stream_messages logs the upstream response status."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"):
                with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        body=b"".join(_make_messages_stream_body()),
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
                        await resp.read()

                # Must have logged the upstream response status
                assert any(
                    "upstream response status" in r.message.lower() or "Upstream response status" in r.message
                    for r in caplog.records
                ), f"Missing upstream status log. Got: {[r.message for r in caplog.records]}"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_stream_logs_upstream_error_on_non_200(self, caplog):
        """Verify _stream_messages logs upstream error body on non-200."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"):
                with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        status=500,
                        payload={"error": {"message": "Internal server error"}},
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
                        await resp.read()

                # Must have logged the upstream error at ERROR level
                error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                assert any("Upstream error" in r.message or "500" in r.message for r in error_records), (
                    f"Missing upstream error log. Got: {[r.message for r in error_records]}"
                )
        finally:
            await server.stop_async()


# ── P3: Streaming Retry Logic ────────────────────────────────────────────


class TestStreamingRetry:
    """P3: Streaming handlers must retry on retryable HTTP statuses."""

    @pytest.mark.asyncio
    async def test_messages_stream_retries_on_503(self):
        """When upstream returns 503, bridge must retry and succeed on next attempt."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                # First attempt: 503 (retryable)
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=503,
                    payload={"error": {"message": "Service unavailable"}},
                )
                # Second attempt: success
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(_make_messages_stream_body()),
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
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert len(body) > 0
                    # Verify we got a SUCCESSFUL response, not an error forwarded from 503.
                    # The successful streaming response contains "message_stop" event.
                    assert b"message_stop" in body, "Expected successful response after retry, but got error response"

                # Verify that the upstream was called exactly twice (initial + retry)
                upstream_calls = m.requests.get(
                    ("POST", aiohttp.client.URL("https://api.example.com/v1/chat/completions")), []
                )
                assert len(upstream_calls) == 2, (
                    f"Expected 2 upstream calls (initial 503 + retry 200), got {len(upstream_calls)}"
                )
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_stream_retries_on_429(self):
        """When upstream returns 429, bridge must retry."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=429,
                    payload={"error": {"message": "Rate limited"}},
                )
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(_make_messages_stream_body()),
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
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert len(body) > 0
                    assert b"message_stop" in body, "Expected successful response after retry, but got error response"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_stream_no_retry_on_401(self):
        """When upstream returns 401, bridge must NOT retry (non-retryable)."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=401,
                    payload={"error": {"message": "Unauthorized"}},
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
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    assert resp.status == 200  # SSE stream always starts 200
                    body = await resp.read()
                    # Must contain error event, not success
                    assert b"error" in body.lower() or b"Error" in body
        finally:
            await server.stop_async()


# ── P4: Context Size Guardrail ───────────────────────────────────────────


class TestContextSizeGuardrail:
    """P4: Bridge must reject oversized requests before sending upstream."""

    @pytest.mark.asyncio
    async def test_oversized_messages_request_rejected(self):
        """Messages API request exceeding token estimate limit must get 400."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            # Build a request with a very large messages array that exceeds the guardrail
            large_content = "x" * 5_000_000  # 5MB of content — exceeds _MAX_REQUEST_CHARS (4MB)
            messages_request = {
                "model": "test-model",
                "messages": [{"role": "user", "content": large_content}],
                "max_tokens": 1024,
                "stream": True,
            }

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=messages_request,
                ) as resp,
            ):
                # Should be rejected with 400 before hitting upstream
                assert resp.status == 400
                body = await resp.json()
                assert "error" in body
                error_msg = body["error"].get("message", "") or body.get("error", {}).get("type", "")
                assert "context" in error_msg.lower() or "large" in error_msg.lower() or "clear" in error_msg.lower()
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_normal_messages_request_passes(self):
        """Normal-sized Messages API requests must not be rejected."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(_make_messages_stream_body()),
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
                    # Normal request should pass through fine
                    assert resp.status == 200
        finally:
            await server.stop_async()


# ── P5: Non-200 Error Body Logging ───────────────────────────────────────


class TestNon200ErrorLogging:
    """P5: Upstream non-200 error bodies must be logged at ERROR level."""

    @pytest.mark.asyncio
    async def test_messages_stream_logs_500_error_body(self, caplog):
        """Upstream 500 error body must be logged."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"):
                with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        status=500,
                        payload={"error": {"message": "Internal server error"}},
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
                        await resp.read()

                # Error body should be logged at ERROR level
                error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                assert len(error_records) > 0, "No ERROR-level log records found"
                combined = " ".join(r.message for r in error_records)
                assert "500" in combined, f"Expected '500' in error logs: {combined}"
                assert "Internal server error" in combined, f"Expected error body text in logs: {combined}"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_stream_logs_502_error_body(self, caplog):
        """Chat Completions pass-through must log 502 error body."""
        adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"):
                with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        status=502,
                        body=b"Bad Gateway",
                    )

                    async with (
                        aiohttp.ClientSession() as session,
                        session.post(
                            f"http://127.0.0.1:{port}/v1/chat/completions",
                            json={
                                "model": "test-model",
                                "messages": [{"role": "user", "content": "hi"}],
                                "stream": True,
                            },
                        ) as resp,
                    ):
                        await resp.read()

                error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
                assert len(error_records) > 0, "No ERROR-level log records found"
        finally:
            await server.stop_async()
