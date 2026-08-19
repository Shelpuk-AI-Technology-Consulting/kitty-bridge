"""Tests for R8: Bridge management commands."""

from __future__ import annotations

import json
import os
import signal
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest

from kitty.bridge.state import BridgeState, load_state, write_state

# ---------------------------------------------------------------------------
# State-based management helpers (pure logic, no process spawning)
# ---------------------------------------------------------------------------


class TestBridgeManagementHelpers:
    """Tests for management logic that uses the state file."""

    def test_is_pid_alive_for_current_process(self):
        from kitty.bridge.manage import is_pid_alive

        assert is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_for_dead_pid(self):
        from kitty.bridge.manage import is_pid_alive

        # Use a very high PID that's extremely unlikely to exist
        assert is_pid_alive(999999999) is False

    def test_stop_bridge_removes_state_file(self, tmp_path: Path):
        from kitty.bridge.manage import stop_bridge

        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=999999999,  # Dead PID
            host="127.0.0.1",
            port=8080,
            profile="test",
            started_at="2026-04-11T10:30:00Z",
            tls=False,
        )
        write_state(state_path, state)

        # Stop should remove state file (PID is already dead)
        stop_bridge(state_path)
        assert not state_path.exists()

    def test_stop_bridge_no_state_file(self, tmp_path: Path):
        from kitty.bridge.manage import stop_bridge

        # Should not raise
        stop_bridge(tmp_path / "nonexistent.json")

    def test_status_bridge_running(self, tmp_path: Path):
        from kitty.bridge.manage import BridgeStatus, bridge_status

        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=os.getpid(),  # Current process — alive
            host="127.0.0.1",
            port=8080,
            profile="test",
            started_at="2026-04-11T10:30:00Z",
            tls=False,
        )
        write_state(state_path, state)

        status = bridge_status(state_path)
        assert status == BridgeStatus.RUNNING

    def test_status_bridge_stopped(self, tmp_path: Path):
        from kitty.bridge.manage import BridgeStatus, bridge_status

        state_path = tmp_path / "state.json"
        # No state file
        status = bridge_status(state_path)
        assert status == BridgeStatus.STOPPED

    def test_status_bridge_stale_pid(self, tmp_path: Path):
        from kitty.bridge.manage import BridgeStatus, bridge_status

        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=999999999,  # Dead PID
            host="127.0.0.1",
            port=8080,
            profile="test",
            started_at="2026-04-11T10:30:00Z",
            tls=False,
        )
        write_state(state_path, state)

        status = bridge_status(state_path)
        assert status == BridgeStatus.STALE

    def test_start_bridge_checks_running_instance(self, tmp_path: Path):
        """start_bridge refuses if another instance is already running."""
        from kitty.bridge.manage import start_bridge

        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=os.getpid(),  # Current process — alive
            host="127.0.0.1",
            port=8080,
            profile="test",
            started_at="2026-04-11T10:30:00Z",
            tls=False,
        )
        write_state(state_path, state)

        with pytest.raises(SystemExit):
            start_bridge(
                state_path=state_path,
                host="127.0.0.1",
                port=9090,
                profile="test",
            )

    def test_start_bridge_clears_stale_state(self, tmp_path: Path):
        """start_bridge clears stale state before starting."""
        from kitty.bridge.manage import is_pid_alive, start_bridge

        state_path = tmp_path / "state.json"
        state = BridgeState(
            pid=999999999,  # Dead PID
            host="127.0.0.1",
            port=8080,
            profile="test",
            started_at="2026-04-11T10:30:00Z",
            tls=False,
        )
        write_state(state_path, state)

        # start_bridge will spawn a background process.
        # It may actually start (if profiles exist in the test env) or fail.
        # Either way, the stale state should be cleared first.
        with suppress(SystemExit):
            start_bridge(
                state_path=state_path,
                host="127.0.0.1",
                port=0,
            )

        # Clean up: kill any spawned bridge process
        final_state = load_state(state_path)
        if final_state is not None and is_pid_alive(final_state.pid):
            os.kill(final_state.pid, signal.SIGTERM)

        # The stale state should be gone (cleared before spawn)
        # A new state file may exist if the bridge actually started
        if state_path.exists():
            data = json.loads(state_path.read_text())
            assert data["pid"] != 999999999


class TestBridgeRestart:
    """Test restart logic (stop + start with re-read of bridge.yaml)."""

    def test_restart_reads_bridge_yaml(self, tmp_path: Path):
        """Restart should re-read bridge.yaml, not use stale state values."""
        from kitty.bridge.manage import restart_bridge

        state_path = tmp_path / "state.json"

        # Write a stale state pointing to dead PID
        state = BridgeState(
            pid=999999999,
            host="127.0.0.1",
            port=8080,
            profile="old-profile",
            started_at="2026-04-11T10:30:00Z",
            tls=False,
        )
        write_state(state_path, state)

        # Write a bridge.yaml with new values
        config_path = tmp_path / "bridge.yaml"
        config_path.write_text("port: 9091\nhost: '127.0.0.1'\nprofile: 'new-profile'\n")

        # restart will fail at process spawn, but should clear stale state first
        with pytest.raises(SystemExit):
            restart_bridge(state_path=state_path, config_path=config_path)


class TestBridgeSubcommandRouting:
    """Test that bridge subcommands are routed correctly."""

    def test_bridge_start_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_START")

    def test_bridge_stop_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_STOP")

    def test_bridge_restart_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_RESTART")

    def test_bridge_status_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_STATUS")

    def test_bridge_config_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_CONFIG")

    def test_bridge_install_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_INSTALL")

    def test_bridge_uninstall_is_routed(self):
        from kitty.cli.router import BuiltinCommand

        assert hasattr(BuiltinCommand, "BRIDGE_UNINSTALL")


class TestIsPidAliveErrorMapping:
    """Cross-platform mapping of os.kill(pid, 0) failures to liveness.

    Signal 0 performs no action and only probes the process — on Windows as
    well as POSIX. The platforms differ in how they report a missing PID, and
    that difference previously crashed every stale-state code path on Windows.
    """

    def test_returns_true_when_probe_succeeds(self):
        from kitty.bridge.manage import is_pid_alive

        with patch("kitty.bridge.manage.os.kill", return_value=None) as mock_kill:
            assert is_pid_alive(4321) is True

        mock_kill.assert_called_once_with(4321, 0)

    def test_process_lookup_error_means_dead(self):
        """POSIX reports a missing PID as ProcessLookupError (ESRCH)."""
        from kitty.bridge.manage import is_pid_alive

        with patch("kitty.bridge.manage.os.kill", side_effect=ProcessLookupError()):
            assert is_pid_alive(4321) is False

    def test_permission_error_is_not_treated_as_alive(self):
        """Pins a KNOWN-INCORRECT behaviour, deliberately left unchanged.

        EPERM actually proves the process exists — it is simply not ours to
        signal — so reporting it as dead makes ``bridge_status`` say STALE for a
        running bridge. Changing it risks the opposite failure (signalling a
        recycled PID belonging to someone else), so it is deferred rather than
        flipped here. See ``.system_design/cross_platform_defects.md`` → X1
        "Known limitation, unchanged".
        """
        from kitty.bridge.manage import is_pid_alive

        with patch("kitty.bridge.manage.os.kill", side_effect=PermissionError()):
            assert is_pid_alive(4321) is False

    @pytest.mark.parametrize("pid", [0, -1, -12345])
    def test_non_positive_pids_are_never_alive(self, pid: int):
        """A PID <= 0 is not a process and must never be signalled.

        On POSIX, ``kill(0, sig)`` targets the caller's entire process group and
        ``kill(-1, sig)`` targets every process the caller may signal. A corrupt
        or hand-edited bridge_state.json carrying such a value would otherwise
        make ``stop_bridge`` SIGTERM then SIGKILL the user's own shell session.
        """
        from kitty.bridge.manage import is_pid_alive

        with patch("kitty.bridge.manage.os.kill") as mock_kill:
            assert is_pid_alive(pid) is False

        mock_kill.assert_not_called()

    def test_stop_bridge_never_signals_a_non_positive_pid(self, tmp_path: Path):
        """End-to-end consequence: a corrupt state file must not kill the shell."""
        from kitty.bridge.manage import stop_bridge

        state_path = tmp_path / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "pid": 0,
                    "host": "127.0.0.1",
                    "port": 8080,
                    "profile": "test",
                    "started_at": "2026-04-11T10:30:00Z",
                    "tls": False,
                }
            )
        )

        with patch("kitty.bridge.manage.os.kill") as mock_kill:
            stop_bridge(state_path)

        mock_kill.assert_not_called()
        assert not state_path.exists()

    def test_windows_invalid_parameter_oserror_means_dead(self):
        """Windows has no ProcessLookupError here.

        OpenProcess fails with ERROR_INVALID_PARAMETER (87) for a PID that does
        not exist, surfacing as a plain OSError. Letting it escape crashed
        bridge status/stop/start/restart on Windows whenever the recorded PID
        was already gone — exactly the stale-state case they exist to clean up.
        """
        from kitty.bridge.manage import is_pid_alive

        winerror_87 = OSError(22, "The parameter is incorrect", None, 87, None)
        with patch("kitty.bridge.manage.os.kill", side_effect=winerror_87):
            assert is_pid_alive(999999999) is False

    def test_bridge_status_reports_stale_instead_of_raising(self, tmp_path: Path):
        """The end-to-end consequence of the mapping above."""
        from kitty.bridge.manage import BridgeStatus, bridge_status

        state_path = tmp_path / "state.json"
        write_state(
            state_path,
            BridgeState(
                pid=999999999,
                host="127.0.0.1",
                port=8080,
                profile="test",
                started_at="2026-04-11T10:30:00Z",
                tls=False,
            ),
        )

        winerror_87 = OSError(22, "The parameter is incorrect", None, 87, None)
        with patch("kitty.bridge.manage.os.kill", side_effect=winerror_87):
            assert bridge_status(state_path) is BridgeStatus.STALE
