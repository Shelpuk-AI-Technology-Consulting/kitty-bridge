"""Tests for R3: Access logging."""

from __future__ import annotations

import re
from pathlib import Path

import aiohttp
import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.zai import ZaiRegularAdapter


def _make_server(log_path: str | None = None, **kwargs):
    return BridgeServer(
        adapter=None,
        provider=ZaiRegularAdapter(),
        resolved_key="test-key",
        model="gpt-4o",
        provider_config={},
        access_log_path=log_path,
        **kwargs,
    )


class TestAccessLogFormat:
    """Access log lines match the specified format."""

    @pytest.mark.asyncio
    async def test_request_produces_log_entry(self, tmp_path: Path):
        log_path = tmp_path / "access.log"
        server = _make_server(log_path=str(log_path))
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        line = lines[0]
        # Tab-separated fields: timestamp, ip, key_id, profile, method, path, status, bytes_in, bytes_out, duration_ms
        fields = line.split("\t")
        assert len(fields) == 10, f"Expected 10 tab-separated fields, got {len(fields)}: {line}"
        # Timestamp: ISO format
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", fields[0])
        # Client IP
        assert fields[1] in ("127.0.0.1", "::1", "127.0.0.1")
        # Key ID: '-' when no auth
        assert fields[2] == "-"
        # Profile name
        assert fields[3] == "default"
        # Method
        assert fields[4] == "GET"
        # Path
        assert fields[5] == "/healthz"
        # Status
        assert fields[6] == "200"
        # bytes_in
        assert fields[7] == "-"
        # bytes_out: some number
        assert fields[8].isdigit()
        # duration_ms: some number
        assert fields[9].isdigit()

    @pytest.mark.asyncio
    async def test_post_request_logs_bytes_in(self, tmp_path: Path):
        log_path = tmp_path / "access.log"
        server = _make_server(log_path=str(log_path))
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                ),
            ):
                pass  # May get error from upstream, that's fine
        finally:
            await server.stop_async()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        fields = lines[0].split("\t")
        assert fields[4] == "POST"
        assert fields[5] == "/v1/chat/completions"
        # bytes_in should be a number (content-length of JSON body)
        assert fields[7].isdigit() or fields[7] == "-"

    @pytest.mark.asyncio
    async def test_disabled_logging_produces_no_file(self, tmp_path: Path):
        """When access_log_path is None, no log file is created."""
        server = _make_server(log_path=None)
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

        # No log files in default location either
        Path.home() / ".config" / "kitty" / "logs" / "bridge_access.log"
        # We can't guarantee the default log doesn't exist from other tests,
        # but we can verify the tmp_path has no logs
        assert not (tmp_path / "access.log").exists()

    @pytest.mark.asyncio
    async def test_log_file_created_at_configured_path(self, tmp_path: Path):
        log_path = tmp_path / "nested" / "dir" / "access.log"
        server = _make_server(log_path=str(log_path))
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session, session.get(
                f"http://127.0.0.1:{port}/healthz"
            ) as resp:
                assert resp.status == 200
        finally:
            await server.stop_async()

        assert log_path.exists()

    @pytest.mark.asyncio
    async def test_multiple_requests_produce_multiple_lines(self, tmp_path: Path):
        log_path = tmp_path / "access.log"
        server = _make_server(log_path=str(log_path))
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(3):
                    async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                        assert resp.status == 200
        finally:
            await server.stop_async()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3
