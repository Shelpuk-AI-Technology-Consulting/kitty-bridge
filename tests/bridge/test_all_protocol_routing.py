"""Tests for R2: Bridge registers all protocol endpoints in bridge mode."""

from __future__ import annotations

import aiohttp
import pytest

from kitty.bridge.server import BridgeServer
from kitty.launchers.claude import ClaudeAdapter
from kitty.launchers.codex import CodexAdapter
from kitty.launchers.gemini import GeminiAdapter
from kitty.launchers.kilo import KiloAdapter
from kitty.providers.zai import ZaiRegularAdapter


def _make_server(adapter=None, **kwargs):
    """Create a BridgeServer with sensible defaults for testing."""
    return BridgeServer(
        adapter=adapter,
        provider=ZaiRegularAdapter(),
        resolved_key="test-key",
        model="gpt-4o",
        provider_config={},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Bridge mode (adapter=None): ALL endpoints registered
# ---------------------------------------------------------------------------


class TestBridgeModeAllEndpoints:
    """In bridge mode, ALL protocol endpoints must be registered."""

    @pytest.mark.asyncio
    async def test_healthz_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            # POST with empty body — should get 400 or upstream error, NOT 404
            async with aiohttp.ClientSession() as session, session.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={},
            ) as resp:
                assert resp.status != 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={},
                ) as resp,
            ):
                assert resp.status != 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_responses_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/responses",
                    json={},
                ) as resp,
            ):
                assert resp.status != 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_gemini_generate_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1beta/models/gpt-4o:generateContent",
                    json={},
                ) as resp,
            ):
                assert resp.status != 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_gemini_stream_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1beta/models/gpt-4o:streamGenerateContent",
                    json={},
                ) as resp,
            ):
                assert resp.status != 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_models_list_registered(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f"http://127.0.0.1:{port}/v1/models",
                ) as resp,
            ):
                assert resp.status == 200
                body = await resp.json()
                assert body["object"] == "list"
                assert isinstance(body["data"], list)
                assert len(body["data"]) >= 1
                model = body["data"][0]
                assert model["object"] == "model"
                assert model["id"] == "gpt-4o"
                assert model["owned_by"] == "kitty-bridge"
        finally:
            await server.stop_async()


# ---------------------------------------------------------------------------
# Agent launch mode: only matching protocol registered (regression guard)
# ---------------------------------------------------------------------------


class TestAgentLaunchModeSingleProtocol:
    """With an adapter, only the matching protocol endpoint should be registered."""

    @pytest.mark.asyncio
    async def test_claude_adapter_registers_messages_only(self):
        server = _make_server(adapter=ClaudeAdapter())
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                # Messages endpoint exists
                async with session.post(f"http://127.0.0.1:{port}/v1/messages", json={}) as resp:
                    assert resp.status != 404
                # Chat completions does NOT exist
                async with session.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={}) as resp:
                    assert resp.status == 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_codex_adapter_registers_responses_only(self):
        server = _make_server(adapter=CodexAdapter())
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                # Responses endpoint exists
                async with session.post(f"http://127.0.0.1:{port}/v1/responses", json={}) as resp:
                    assert resp.status != 404
                # Chat completions does NOT exist
                async with session.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={}) as resp:
                    assert resp.status == 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_gemini_adapter_registers_gemini_only(self):
        server = _make_server(adapter=GeminiAdapter())
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                # Gemini endpoint exists
                async with session.post(
                    f"http://127.0.0.1:{port}/v1beta/models/gpt-4o:generateContent",
                    json={},
                ) as resp:
                    assert resp.status != 404
                # Chat completions does NOT exist
                async with session.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={}) as resp:
                    assert resp.status == 404
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_kilo_adapter_registers_chat_completions_only(self):
        server = _make_server(adapter=KiloAdapter())
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                # Chat completions endpoint exists
                async with session.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={}) as resp:
                    assert resp.status != 404
                # Messages does NOT exist
                async with session.post(f"http://127.0.0.1:{port}/v1/messages", json={}) as resp:
                    assert resp.status == 404
        finally:
            await server.stop_async()
