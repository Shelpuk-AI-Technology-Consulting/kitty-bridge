"""Tests that AllBackendsUnhealthyError in top-level handlers returns
503 Service Unavailable with a Retry-After header, not 500 Internal Server Error.

When all backends are exhausted, the bridge knows the soonest retry window
(``AllBackendsUnhealthyError.retry_after``).  Claude Code / Responses / Gemini
clients can use the ``Retry-After`` header to back off appropriately.
Without this fix, the bridge returned a generic 500 with no timing info.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import aiohttp
import pytest

from kitty.bridge.server import AllBackendsUnhealthyError, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter

# ── Stubs ──────────────────────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self):
        from kitty.types import BridgeProtocol

        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, *args, **kwargs) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def normalize_model_name(self, model: str) -> str:
        return model

    def translate_to_upstream(self, cc_request: dict) -> dict:
        return {"model": cc_request["model"], "messages": cc_request.get("messages", [])}

    def translate_from_upstream(self, raw_response: dict) -> dict:
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        return [raw_bytes]

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        from kitty.providers.base import ProviderError

        return ProviderError(f"Stub error {status_code}")


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_server() -> BridgeServer:
    return BridgeServer(_StubLauncher(), _StubProvider(), "test-key")


def _make_bridge_mode_server() -> BridgeServer:
    """Bridge mode registers all protocol routes (/v1/messages, /v1/responses,
    /v1/chat/completions, /v1/...)."""
    return BridgeServer(None, _StubProvider(), "test-key")  # type: ignore[arg-type]


def _messages_request() -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }


def _responses_request() -> dict:
    return {
        "model": "test-model",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def _cc_request() -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }


# ── AllBackendsUnhealthyError type tests ──────────────────────────────────


class TestAllBackendsUnhealthyErrorType:
    def test_carries_retry_after(self):
        err = AllBackendsUnhealthyError([{"name": "stub"}], retry_after=264)
        assert err.retry_after == 264

    def test_carries_backend_list(self):
        backends = [{"name": "a"}, {"name": "b"}]
        err = AllBackendsUnhealthyError(backends, retry_after=120)
        assert len(err.backends) == 2

    def test_message_includes_retry_after(self):
        err = AllBackendsUnhealthyError([], retry_after=42)
        assert "42" in str(err)


# ── Messages API handler returns 503 ──────────────────────────────────────


class TestMessagesHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after_header(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=264,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503, f"Expected 503, got {resp.status}"
                    retry_after = resp.headers.get("Retry-After")
                    assert retry_after is not None, "Missing Retry-After header"
                    assert int(retry_after) == 264
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_body_contains_error(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=120,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    data = await resp.json()
                    assert "error" in data
                    msg = json.dumps(data).lower()
                    # Should mention unavailability / retry / backend
                    assert any(
                        token in msg
                        for token in (
                            "unavail",
                            "backend",
                            "retry",
                            "503",
                        )
                    ), f"Error body lacks context: {data}"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_does_not_leak_traceback(self):
        """Python traceback markers must not appear in the error body."""
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=120,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    body_text = await resp.text()
                    assert "Traceback" not in body_text
                    assert "AllBackendsUnhealthyError" not in body_text
        finally:
            await server.stop_async()


# ── Responses API handler returns 503 ─────────────────────────────────────


class TestResponsesHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=180,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_responses_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    retry_after = resp.headers.get("Retry-After")
                    assert retry_after is not None
                    assert int(retry_after) == 180
        finally:
            await server.stop_async()


# ── Chat Completions handler returns 503 ──────────────────────────────────


class TestChatCompletionsHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=90,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json=_cc_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    assert resp.headers.get("Retry-After") is not None
        finally:
            await server.stop_async()


# ── Gemini handler returns 503 ────────────────────────────────────────────


class TestGeminiHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=300,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1beta/models/test:generateContent",
                        json={"contents": [{"parts": [{"text": "hi"}]}]},
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    assert resp.headers.get("Retry-After") is not None
        finally:
            await server.stop_async()


# ── Negative: success path still works ────────────────────────────────────


class TestHappyPathNotAffected:
    """Verify the 503 fallback doesn't interfere with successful requests."""

    @pytest.mark.asyncio
    async def test_successful_request_returns_200(self):
        from aioresponses import aioresponses

        # Non-streaming CC response
        body = json.dumps(
            {
                "id": "test-1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ).encode()

        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", status=200, body=body)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["role"] == "assistant"
        finally:
            await server.stop_async()
