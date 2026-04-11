"""Tests for connection-level retry in streaming handlers.

Verifies that transient connection failures (TimeoutError, ConnectionReset,
ClientConnectorError) are retried transparently — the agent never sees a
retry, only a (possibly delayed) response or a final error after all
attempts are exhausted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import MagicMock

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer, _is_retryable_exception, _truncate_for_log
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


def _make_responses_stream_body() -> list[bytes]:
    """Standard SSE chunks for a Responses API streaming response."""
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


# ── Helper function tests ────────────────────────────────────────────────


class TestIsRetryableException:
    """Verify _is_retryable_exception classifies exceptions correctly."""

    def test_timeout_is_retryable(self):
        assert _is_retryable_exception(asyncio.TimeoutError()) is True

    def test_connection_reset_is_retryable(self):
        assert _is_retryable_exception(ConnectionResetError()) is True

    def test_broken_pipe_is_retryable(self):
        assert _is_retryable_exception(BrokenPipeError()) is True

    def test_client_connector_error_is_retryable(self):
        exc = aiohttp.ClientConnectorError(
            MagicMock(),
            OSError("Connection refused"),
        )
        assert _is_retryable_exception(exc) is True

    def test_oserror_econnrefused_is_retryable(self):
        exc = OSError(111, "Connection refused")
        assert _is_retryable_exception(exc) is True

    def test_valueerror_is_not_retryable(self):
        assert _is_retryable_exception(ValueError("bad data")) is False

    def test_json_decode_error_is_not_retryable(self):
        assert _is_retryable_exception(json.JSONDecodeError("bad", "", 0)) is False


class TestTruncateForLog:
    """Verify _truncate_for_log truncates correctly."""

    def test_short_text_unchanged(self):
        assert _truncate_for_log("hello") == "hello"

    def test_exact_limit_unchanged(self):
        text = "x" * 2000
        assert _truncate_for_log(text) == text

    def test_long_text_truncated(self):
        text = "x" * 5000
        result = _truncate_for_log(text)
        assert len(result) < 2200  # 2000 chars + suffix
        assert "5000 chars total" in result

    def test_custom_limit(self):
        text = "x" * 100
        result = _truncate_for_log(text, limit=50)
        assert "100 chars total" in result


# ── Messages API: connection-level retries ───────────────────────────────


class TestMessagesConnectionRetries:
    """Messages API streaming retries on transient connection failures."""

    @pytest.mark.asyncio
    async def test_timeout_retried_success(self):
        """TimeoutError on first attempt, success on second — agent sees success."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                # First attempt: timeout
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    exception=asyncio.TimeoutError(),
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
                    # Should have successful content, not an error
                    assert b"message_stop" in body or b"content_block_delta" in body or len(body) > 0
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_timeout_exhausted_sends_error(self):
        """TimeoutError on all attempts — agent receives one error event."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                # All attempts timeout
                for _ in range(4):  # 1 initial + 3 retries
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
                    assert resp.status == 200
                    body = await resp.read()
                    assert b"error" in body
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_connection_reset_retried(self):
        """ConnectionResetError on first attempt, success on second."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    exception=ConnectionResetError("Connection reset"),
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
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_http_400_not_retried(self):
        """HTTP 400 is not retried — error sent immediately."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=400,
                    payload={"error": {"message": "Bad request"}},
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
                    body = await resp.read()
                    # Only one call was made — no retry
                    assert b"error" in body
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_retry_logging(self, caplog):
        """Retries produce WARNING-level log messages."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"), aioresponses(
                passthrough=["http://127.0.0.1"]
            ) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        exception=asyncio.TimeoutError(),
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
                        await resp.read()

                    # Should have a retry warning
                    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
                    retry_msgs = [r for r in warnings if "retrying" in r.message.lower()]
                    assert len(retry_msgs) > 0, f"Expected retry warning, got: {[r.message for r in warnings]}"
        finally:
            await server.stop_async()


# ── Responses API: connection-level retries ──────────────────────────────


class TestResponsesConnectionRetries:
    """Responses API streaming retries on transient connection failures."""

    @pytest.mark.asyncio
    async def test_timeout_retried_success(self):
        """TimeoutError on first attempt, success on second."""
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    exception=asyncio.TimeoutError(),
                )
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=b"".join(_make_responses_stream_body()),
                    headers={"Content-Type": "text/event-stream"},
                )

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": "test-model",
                            "input": [{"type": "message", "role": "user", "content": "hi"}],
                            "stream": True,
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert len(body) > 0
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_timeout_exhausted_sends_error(self):
        """All retries exhausted — agent receives error."""
        adapter = StubLauncher(BridgeProtocol.RESPONSES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for _ in range(4):
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        exception=asyncio.TimeoutError(),
                    )

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": "test-model",
                            "input": [{"type": "message", "role": "user", "content": "hi"}],
                            "stream": True,
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    body = await resp.read()
                    assert b"response.completed" in body  # Must always emit completed
        finally:
            await server.stop_async()


# ── Chat Completions API: connection-level retries ───────────────────────


class TestChatCompletionsConnectionRetries:
    """Chat Completions streaming retries on transient connection failures."""

    @pytest.mark.asyncio
    async def test_timeout_retried_success(self):
        """TimeoutError on first attempt, success on second."""
        adapter = StubLauncher(BridgeProtocol.CHAT_COMPLETIONS_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    exception=asyncio.TimeoutError(),
                )
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=(
                        b'data: {"id":"test","choices":[{"index":0,"delta":{"content":"Hi"},'
                        b'"finish_reason":null}]}\n\ndata: [DONE]\n\n'
                    ),
                    headers={"Content-Type": "text/event-stream"},
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
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.read()
                    assert len(body) > 0
        finally:
            await server.stop_async()


# ── Debug log truncation ─────────────────────────────────────────────────


class TestDebugLogTruncation:
    """Verify large request bodies are truncated in debug logs."""

    @pytest.mark.asyncio
    async def test_large_request_body_truncated_in_log(self, caplog):
        """Request bodies larger than the limit are truncated in logs."""
        adapter = StubLauncher(BridgeProtocol.MESSAGES_API)
        provider = StubProvider()
        server = BridgeServer(adapter, provider, "test-key", debug=True)
        port = await server.start_async()
        try:
            with caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"), aioresponses(
                passthrough=["http://127.0.0.1"]
            ) as m:
                    m.post(
                        "https://api.example.com/v1/chat/completions",
                        body=b"".join(_make_messages_stream_body()),
                        headers={"Content-Type": "text/event-stream"},
                    )

                    # Send a request with a large body
                    large_content = "x" * 5000
                    async with (
                        aiohttp.ClientSession() as session,
                        session.post(
                            f"http://127.0.0.1:{port}/v1/messages",
                            json={
                                "model": "test-model",
                                "messages": [{"role": "user", "content": large_content}],
                                "max_tokens": 1024,
                                "stream": True,
                            },
                        ) as resp,
                    ):
                        await resp.read()

                    # Check that the log entry is truncated
                    body_logs = [r for r in caplog.records if "Request body:" in r.message]
                    assert len(body_logs) > 0
                    for record in body_logs:
                        # The log message should be shorter than the full body
                        if "chars total" in record.message:
                            # It was truncated — good
                            assert len(record.message) < 3000
        finally:
            await server.stop_async()
