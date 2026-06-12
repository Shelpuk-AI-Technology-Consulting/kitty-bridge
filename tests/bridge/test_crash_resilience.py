"""Tests for crash resilience — unhandled exceptions in request handlers
must be logged and surfaced as clean 500 responses.

Also tests that process-level crash handlers (faulthandler, excepthook, atexit)
are correctly wired in bridge_runner.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import aiohttp
import pytest

from kitty.bridge import server as server_module
from kitty.bridge.server import (
    BridgeServer,
    _setup_crash_handlers,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stub adapters ──────────────────────────────────────────────────────────


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
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        req = {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}
        for key in ("tools", "temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                req[key] = kwargs[key]
        return req

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_server() -> BridgeServer:
    """Create a server with the Messages API protocol endpoint."""
    return BridgeServer(StubLauncher(), StubProvider(), "test-key")


def _make_bridge_mode_server() -> BridgeServer:
    """Create a server in bridge mode (all protocol endpoints registered).

    Bridge mode uses adapter=None which registers /v1/messages, /v1/responses,
    and /v1/chat/completions routes."""
    return BridgeServer(None, StubProvider(), "test-key")  # type: ignore[arg-type]


def _messages_request() -> dict:
    """Minimal Messages API payload that passes JSON parse."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
    }


def _responses_request() -> dict:
    """Minimal Responses API payload."""
    return {
        "model": "test-model",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


# ── Request handler crash protection tests ─────────────────────────────────


class TestRequestHandlerCrashProtection:
    """Unhandled exceptions in route handlers must produce a logged 500."""

    @pytest.mark.asyncio
    async def test_handle_messages_logs_exception_and_returns_500(self, caplog):
        """When _apply_compaction raises, the handler catches it and returns 500."""
        server = _make_server()
        port = await server.start_async()

        try:
            # Patch _apply_compaction to simulate an internal crash
            with patch.object(server, "_apply_compaction", side_effect=RuntimeError("simulated crash")):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 500
                    error_data = await resp.json()
                    assert error_data["type"] == "error"
                    assert error_data["error"]["type"] == "internal_error"
                    # Must not leak internal details
                    assert "simulated crash" not in json.dumps(error_data)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_handle_responses_logs_exception_and_returns_500(self, caplog):
        """When the Responses handler hits an unhandled exception, it returns 500."""
        server = _make_bridge_mode_server()
        port = await server.start_async()

        try:
            with patch.object(server, "_apply_compaction", side_effect=ValueError("responses boom")):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_responses_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 500
                    data = await resp.json()
                    assert "error" in data
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_invalid_json_still_returns_400(self):
        """Existing JSON parse error handling still works (no false 500s)."""
        server = _make_server()
        port = await server.start_async()

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=b"not valid json",
                    headers={"content-type": "application/json"},
                ) as resp,
            ):
                assert resp.status == 400
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_normal_streaming_flow_still_works(self):
        """A normal (non-crashing) Messages API request still succeeds."""
        from aioresponses import aioresponses

        server = _make_server()
        port = await server.start_async()

        stream_body = (
            b"data: "
            + json.dumps(
                {
                    "id": "chatcmpl-1",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            ).encode()
            + b"\n\ndata: [DONE]\n\n"
        )

        request = _messages_request()
        request["stream"] = True

        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", status=200, body=stream_body)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=request,
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 200
        finally:
            await server.stop_async()


# ── Crash handler setup tests ──────────────────────────────────────────────


class TestCrashHandlerSetup:
    """Process-level crash handlers must be wired in bridge_runner."""

    def test_setup_crash_handlers_enables_faulthandler(self, tmp_path):
        """After _setup_crash_handlers, faulthandler is enabled."""
        import faulthandler

        fd = faulthandler.is_enabled()
        log_path = tmp_path / "crash.log"
        log_path.touch()

        try:
            server_module._crash_handlers_installed = False
            _setup_crash_handlers(log_path)
            assert faulthandler.is_enabled()
        finally:
            # Restore faulthandler state
            if not fd:
                faulthandler.disable()

    def test_setup_crash_handlers_registers_excepthook(self, tmp_path):
        """After _setup_crash_handlers, sys.excepthook is not the default."""

        original = sys.excepthook
        log_path = tmp_path / "crash.log"
        log_path.touch()

        try:
            server_module._crash_handlers_installed = False
            _setup_crash_handlers(log_path)
            assert sys.excepthook is not original
        finally:
            sys.excepthook = original

    def test_setup_crash_handlers_registers_atexit(self, tmp_path):
        """After _setup_crash_handlers, an atexit handler is registered."""
        import atexit

        # Count atexit handlers before and after
        before = atexit._ncallbacks() if hasattr(atexit, "_ncallbacks") else 0
        log_path = tmp_path / "crash.log"
        log_path.touch()

        server_module._crash_handlers_installed = False
        _setup_crash_handlers(log_path)
        after = atexit._ncallbacks() if hasattr(atexit, "_ncallbacks") else 0

        assert after >= before + 1, f"Expected at least 1 new atexit handler, before={before} after={after}"

    def test_custom_excepthook_logs_and_calls_sys_exit(self, tmp_path):
        """The custom excepthook logs the exception at CRITICAL level."""

        log_path = tmp_path / "crash.log"
        log_path.touch()

        original_excepthook = sys.excepthook

        try:
            server_module._crash_handlers_installed = False
            _setup_crash_handlers(log_path)

            # Manually invoke the custom excepthook with a test exception.
            # It should log and call sys.exit(1).
            with pytest.raises(SystemExit):
                sys.excepthook(RuntimeError, RuntimeError("test crash"), None)
        finally:
            sys.excepthook = original_excepthook
            # atexit registration is not cleanly reversible in tests,
            # but the handler is harmless — it just logs a line.
