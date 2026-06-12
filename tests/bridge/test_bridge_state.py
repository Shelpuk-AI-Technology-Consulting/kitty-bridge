"""Tests for R7: Bridge state file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kitty.bridge.server import BridgeServer
from kitty.bridge.state import BridgeState, load_state, remove_state, write_state
from kitty.providers.zai import ZaiRegularAdapter


def _default_state_path() -> Path:
    return Path.home() / ".config" / "kitty" / "bridge_state.json"


class TestBridgeStateFile:
    """State file is written on start and removed on shutdown."""

    @pytest.mark.asyncio
    async def test_state_file_written_on_start(self, tmp_path: Path):
        state_path = tmp_path / "bridge_state.json"
        server = BridgeServer(
            adapter=None,
            provider=ZaiRegularAdapter(),
            resolved_key="test-key",
            model="gpt-4o",
            state_file=str(state_path),
            profile_name="test-profile",
        )
        await server.start_async()
        try:
            assert state_path.exists()
            data = json.loads(state_path.read_text())
            assert data["pid"] == os.getpid()
            assert data["host"] == "127.0.0.1"
            assert data["port"] > 0
            assert data["profile"] == "test-profile"
            assert "started_at" in data
            assert data["tls"] is False
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_state_file_removed_on_shutdown(self, tmp_path: Path):
        state_path = tmp_path / "bridge_state.json"
        server = BridgeServer(
            adapter=None,
            provider=ZaiRegularAdapter(),
            resolved_key="test-key",
            model="gpt-4o",
            state_file=str(state_path),
        )
        await server.start_async()
        assert state_path.exists()
        await server.stop_async()
        assert not state_path.exists()

    @pytest.mark.asyncio
    async def test_state_file_with_tls_flag(self, tmp_path: Path):
        """State file records tls=True when TLS is configured."""
        state_path = tmp_path / "bridge_state.json"
        # We need a cert/key to start TLS — skip if openssl not available
        import subprocess

        try:
            cert = tmp_path / "cert.pem"
            key = tmp_path / "key.pem"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                    "-days",
                    "1",
                    "-nodes",
                    "-subj",
                    "/CN=localhost",
                ],
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("openssl not available")

        server = BridgeServer(
            adapter=None,
            provider=ZaiRegularAdapter(),
            resolved_key="test-key",
            model="gpt-4o",
            state_file=str(state_path),
            tls_cert=str(cert),
            tls_key=str(key),
        )
        await server.start_async()
        try:
            data = json.loads(state_path.read_text())
            assert data["tls"] is True
        finally:
            await server.stop_async()


class TestBridgeStateHelpers:
    """Unit tests for state file helper functions."""

    def test_write_and_load_state(self, tmp_path: Path):
        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=12345, host="0.0.0.0", port=8080, profile="my-zai", started_at="2026-04-11T10:30:00Z", tls=False
        )
        write_state(state_path, state)

        loaded = load_state(state_path)
        assert loaded is not None
        assert loaded.pid == 12345
        assert loaded.host == "0.0.0.0"
        assert loaded.port == 8080
        assert loaded.profile == "my-zai"
        assert loaded.tls is False

    def test_load_state_missing_file(self, tmp_path: Path):
        result = load_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_remove_state(self, tmp_path: Path):
        state_path = tmp_path / "state.json"
        state_path.write_text("{}")
        remove_state(state_path)
        assert not state_path.exists()

    def test_remove_state_already_gone(self, tmp_path: Path):
        """Removing a nonexistent state file should not raise."""
        remove_state(tmp_path / "nonexistent.json")

    def test_timestamp_is_utc(self, tmp_path: Path):
        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=1, host="127.0.0.1", port=8080, profile="p", started_at="2026-04-11T10:30:00Z", tls=False
        )
        write_state(state_path, state)
        loaded = load_state(state_path)
        assert loaded is not None
        assert loaded.started_at.endswith("Z")


class TestWriteStateAtomicity:
    """F40: write_state must use temp file + os.replace for atomic writes."""

    def test_write_state_uses_os_replace(self, tmp_path, monkeypatch):
        import os

        from kitty.bridge import state as state_mod

        calls = []
        real_replace = os.replace

        def recording_replace(src, dst):
            calls.append((src, dst))
            return real_replace(src, dst)

        monkeypatch.setattr(state_mod.os, "replace", recording_replace)
        state_path = tmp_path / "bridge_state.json"
        write_state(
            state_path,
            BridgeState(pid=123, host="127.0.0.1", port=8080, profile="p", started_at="now", tls=False),
        )
        assert calls, "write_state should use os.replace for atomic commit"
        src, dst = calls[0]
        assert str(src).endswith(".tmp")
        assert dst == state_path
        assert not state_path.with_suffix(".json.tmp").exists()


class TestHealthMonitor:
    """F41: After spawn, a background health monitor polls /healthz and restarts the bridge.

    A standalone ``_health_monitor(state, on_unhealthy, *, interval, timeout)`` helper
    polls ``/healthz`` on the running bridge.  If a health-check fails, it calls
    ``on_unhealthy`` which can restart the bridge.
    """

    def test_health_monitor_detects_unhealthy_and_invokes_callback(self):
        import threading

        from kitty.bridge.manage import _health_monitor

        stop_event = threading.Event()
        invoked = threading.Event()
        calls = []

        def on_unhealthy(state_dict: dict) -> None:
            calls.append(state_dict)
            invoked.set()

        # state dict with a dead host:port
        state = {"host": "127.0.0.1", "port": 1, "pid": 1, "profile": "p", "tls": False, "started_at": "now"}
        # Run the monitor in a background thread with short interval
        thread = threading.Thread(
            target=_health_monitor,
            args=(state, on_unhealthy),
            kwargs={"interval": 0.05, "stop_event": stop_event},
            daemon=True,
        )
        thread.start()
        try:
            # Should detect unhealthy within a few intervals
            assert invoked.wait(timeout=5.0), "Health monitor did not invoke callback for dead port"
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
        assert calls, "on_unhealthy should have been called at least once"
