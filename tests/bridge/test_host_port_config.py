"""Tests for R1: Configurable host and port."""

from __future__ import annotations

import socket

import aiohttp
import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.zai import ZaiRegularAdapter


def _make_server(**kwargs):
    return BridgeServer(
        adapter=None,
        provider=ZaiRegularAdapter(),
        resolved_key="test-key",
        model="gpt-4o",
        provider_config={},
        **kwargs,
    )


class TestBridgeServerHostPort:
    """BridgeServer accepts host and port parameters."""

    @pytest.mark.asyncio
    async def test_default_port_is_random(self):
        """Without port arg, OS assigns a random available port."""
        server = _make_server()
        port = await server.start_async()
        try:
            assert port > 0
            assert port != 0
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_specific_port_is_used(self):
        """Passing port= uses that specific port."""
        # Find a free port first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        server = _make_server(port=free_port)
        port = await server.start_async()
        try:
            assert port == free_port
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_port_conflict_raises_error(self):
        """Binding to an already-used port raises an OSError."""
        # Start first server
        server1 = _make_server()
        port1 = await server1.start_async()

        try:
            # Try to start second server on same port
            server2 = _make_server(port=port1)
            with pytest.raises(OSError):
                await server2.start_async()
        finally:
            await server1.stop_async()

    @pytest.mark.asyncio
    async def test_default_host_is_localhost(self):
        """Default host is 127.0.0.1."""
        server = _make_server()
        port = await server.start_async()
        try:
            # Connect to 127.0.0.1 explicitly
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_port_property_after_start(self):
        """server.port returns the actual bound port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        server = _make_server(port=free_port)
        await server.start_async()
        try:
            assert server.port == free_port
        finally:
            await server.stop_async()
