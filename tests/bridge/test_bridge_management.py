"""Tests for R8: Bridge management commands."""

from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from kitty.bridge.state import BridgeState, write_state

# ---------------------------------------------------------------------------
# State-based management helpers (pure logic, no process spawning)
# ---------------------------------------------------------------------------


class TestBridgeManagementHelpers:
    """Tests for management logic that uses the state file."""

    def test_probe_pid_for_current_process(self):
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        assert probe_pid(os.getpid()) is ProcessLiveness.ALIVE

    def test_probe_pid_for_dead_pid(self):
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        # Use a very high PID that's extremely unlikely to exist
        assert probe_pid(999999999) is ProcessLiveness.DEAD

    def test_probe_pid_never_signals_zero_on_windows(self, monkeypatch):
        """KBR-180: ``os.kill(pid, 0)`` is a Ctrl+C broadcast on Windows.

        🔴 ``signal.CTRL_C_EVENT`` **is** ``0``, so on Windows that call does
        not probe -- it raises a console Ctrl+C delivered to every process
        sharing the console window, including the user's own shell. The
        Windows CI leg caught it as a ``KeyboardInterrupt`` that aborted the
        run 227 tests in.

        This runs on **every** platform, deliberately: the defect is invisible
        on the Linux legs that make up four of the six, so a Windows-only
        regression test would be checked by one leg and could rot unnoticed
        in between. Faking the platform is what makes the claim checkable
        everywhere.
        """
        from kitty.bridge import manage

        # Fails loudly rather than silently passing if the dispatch is ever
        # removed: a test that asserts `os.kill` was not called would also
        # pass if `probe_pid` did nothing at all.
        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"probe_pid reached os.kill{args!r} on Windows")

        monkeypatch.setattr(manage.sys, "platform", "win32")
        monkeypatch.setattr(manage.os, "kill", _explode)
        monkeypatch.setattr(
            manage, "_probe_pid_windows", lambda pid: manage.ProcessLiveness.ALIVE
        )

        assert manage.probe_pid(4321) is manage.ProcessLiveness.ALIVE

    def test_probe_pid_screens_non_positive_pids_before_any_dispatch(self):
        """A corrupt state file must not reach either platform path.

        On POSIX a pid of 0 addresses the caller's whole process group; on
        Windows it is CTRL_C_EVENT's own group broadcast. The screen is what
        stops a corrupt ``bridge_state.json`` from reaching either.
        """
        from kitty.bridge import manage

        # Asserts the SCREEN, not just the answer. `probe_pid(0) is DEAD` would
        # also hold if the pid reached a probe that happened to report dead --
        # so the claim 'before any dispatch' would go unchecked. Both platform
        # paths are blocked, because the screen protects both.
        def _explode(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"a non-positive pid reached a probe: {args!r}")

        with (
            patch.object(manage.os, "kill", _explode),
            patch.object(manage, "_probe_pid_windows", _explode),
        ):
            assert manage.probe_pid(0) is manage.ProcessLiveness.DEAD
            assert manage.probe_pid(-1) is manage.ProcessLiveness.DEAD

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
        from kitty.bridge.manage import start_bridge

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

        # Stand in for the spawned child: a real ``python -m kitty.bridge_runner``
        # would refresh the model-context catalog over the network and write the
        # user cache, so the suite must never spawn one. The stand-in exits at
        # once (taking start_bridge's error path) and records whether the stale
        # state was already cleared at spawn time.
        state_at_spawn: list[bool] = []

        def _spawn(*_args, **_kwargs):
            state_at_spawn.append(state_path.exists())
            return SimpleNamespace(poll=lambda: 1, stdout=io.BytesIO(), returncode=1)

        with (
            patch("kitty.bridge.manage.subprocess.Popen", side_effect=_spawn) as mock_popen,
            pytest.raises(SystemExit),
        ):
            start_bridge(
                state_path=state_path,
                host="127.0.0.1",
                port=0,
            )

        mock_popen.assert_called_once()
        assert state_at_spawn == [False], "stale state must be cleared before the spawn"

    def test_start_bridge_tells_the_child_where_to_write_its_state(self, tmp_path: Path):
        """The child is handed the state path the parent will poll (KBR-220).

        Without it the child wrote to its own default while the parent waited on
        ``state_path``, so any caller passing a non-default path -- the CLI on
        macOS, Windows and XDG Linux -- reported a healthy bridge as failed.
        """
        from kitty.bridge.manage import start_bridge

        state_path = tmp_path / "somewhere" / "state.json"
        spawned: list[list[str]] = []

        def _spawn(cmd, *_args, **_kwargs):
            """Record the child command, then exit at once as a failed child."""
            spawned.append(list(cmd))
            return SimpleNamespace(poll=lambda: 1, stdout=io.BytesIO(), returncode=1)

        with patch("kitty.bridge.manage.subprocess.Popen", side_effect=_spawn), pytest.raises(SystemExit):
            start_bridge(state_path=state_path)

        [cmd] = spawned
        assert "--state-file" in cmd, f"child command carries no state path: {cmd}"
        assert cmd[cmd.index("--state-file") + 1] == str(state_path)


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

        # Stand in for the spawned child (see test_start_bridge_clears_stale_state):
        # it exits at once, so restart takes its error path, and the recorded
        # command proves the freshly read bridge.yaml is handed to the child.
        spawned_cmd: list[str] = []

        def _spawn(cmd, *_args, **_kwargs):
            spawned_cmd.extend(cmd)
            return SimpleNamespace(poll=lambda: 1, stdout=io.BytesIO(), returncode=1)

        # restart will fail at process spawn, but should clear stale state first
        with (
            patch("kitty.bridge.manage.subprocess.Popen", side_effect=_spawn) as mock_popen,
            pytest.raises(SystemExit),
        ):
            restart_bridge(state_path=state_path, config_path=config_path)

        mock_popen.assert_called_once()
        assert "--config" in spawned_cmd
        assert str(config_path) in spawned_cmd
        # No stale-state value leaks into the child: the old state's port and
        # profile must not reappear as spawn arguments.
        assert "--port" not in spawned_cmd
        assert "old-profile" not in spawned_cmd


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


class TestBridgeStatusCommandOutput:
    """R7: what the user actually reads when the bridge is not theirs."""

    def test_status_names_the_address_and_says_kitty_cannot_manage_it(self, capsys):
        from kitty.bridge.manage import BridgeStatus
        from kitty.cli.main import main
        from kitty.cli.router import BuiltinCommand, RouteResult

        state = BridgeState(
            pid=4321,
            host="127.0.0.1",
            port=8080,
            profile="work",
            started_at="2026-08-19T10:30:00Z",
            tls=False,
        )

        # Route explicitly: with an empty profile store the router sends every
        # command to the setup wizard, so this test would otherwise pass or fail
        # depending on whether the machine running it has profiles configured.
        route = RouteResult(builtin=BuiltinCommand.BRIDGE_STATUS)

        with (
            patch("sys.argv", ["kitty", "bridge", "status"]),
            patch("kitty.cli.router.CLIRouter.route", return_value=route),
            patch("kitty.bridge.manage.bridge_status", return_value=BridgeStatus.UNMANAGEABLE),
            patch("kitty.bridge.state.load_state", return_value=state),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "http://127.0.0.1:8080" in out
        assert "4321" in out
        assert "another user" in out.lower()
        # `status` is the first command users run, so it carries the same
        # escape hatch as `stop` and `start` rather than only naming the file.
        assert "bridge_state.json" in out
        # The old message for this situation; reporting it would send the user
        # to `bridge stop`, which now (correctly) refuses.
        assert "stale" not in out.lower()


class TestStatusForAProcessWeMayNotSignal:
    """R3: the decision table for a PID owned by another user.

    ``probe_pid`` and ``bridge_reachable`` are patched here because their own
    behaviour is pinned by the two classes below; what is under test is how
    ``bridge_status`` combines them.
    """

    @staticmethod
    def _write_state(state_path: Path) -> None:
        write_state(
            state_path,
            BridgeState(
                pid=4321,
                host="127.0.0.1",
                port=8080,
                profile="test",
                started_at="2026-08-19T10:30:00Z",
                tls=False,
            ),
        )

    def test_unsignallable_and_reachable_is_unmanageable(self, tmp_path: Path):
        """The bug in #3: this used to report STALE for a healthy bridge."""
        from kitty.bridge.manage import BridgeStatus, ProcessLiveness, bridge_status

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=True) as mock_probe,
        ):
            assert bridge_status(state_path) is BridgeStatus.UNMANAGEABLE

        mock_probe.assert_called_once_with("127.0.0.1", 8080)

    def test_unsignallable_and_unreachable_is_stale(self, tmp_path: Path):
        """Nothing serving at the recorded address — the PID was recycled."""
        from kitty.bridge.manage import BridgeStatus, ProcessLiveness, bridge_status

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=False),
        ):
            assert bridge_status(state_path) is BridgeStatus.STALE

    def test_alive_does_not_probe_the_socket(self, tmp_path: Path):
        """A bridge we own is RUNNING without paying for a connect attempt."""
        from kitty.bridge.manage import BridgeStatus, ProcessLiveness, bridge_status

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.ALIVE),
            patch("kitty.bridge.manage.bridge_reachable") as mock_probe,
        ):
            assert bridge_status(state_path) is BridgeStatus.RUNNING

        mock_probe.assert_not_called()

    def test_dead_does_not_probe_the_socket(self, tmp_path: Path):
        """A missing process is stale regardless of who else holds the port."""
        from kitty.bridge.manage import BridgeStatus, ProcessLiveness, bridge_status

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.DEAD),
            patch("kitty.bridge.manage.bridge_reachable") as mock_probe,
        ):
            assert bridge_status(state_path) is BridgeStatus.STALE

        mock_probe.assert_not_called()


class TestStopBridgeWithAProcessWeMayNotSignal:
    """R4: the defect's most damaging branch.

    Before the fix, ``stop`` deleted the state file for a bridge it had not
    stopped, leaving an orphaned process and no record pointing at it.
    """

    @staticmethod
    def _write_state(state_path: Path) -> None:
        write_state(
            state_path,
            BridgeState(
                pid=4321,
                host="127.0.0.1",
                port=8080,
                profile="test",
                started_at="2026-08-19T10:30:00Z",
                tls=False,
            ),
        )

    def test_refuses_and_keeps_the_state_file_when_reachable(self, tmp_path: Path, capsys):
        from kitty.bridge.manage import ProcessLiveness, stop_bridge

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=True),
            patch("kitty.bridge.manage.os.kill") as mock_kill,
            pytest.raises(SystemExit) as exc_info,
        ):
            stop_bridge(state_path)

        assert exc_info.value.code == 1
        mock_kill.assert_not_called()
        assert state_path.exists()

        message = capsys.readouterr().err
        assert "4321" in message
        assert "127.0.0.1:8080" in message
        assert "cannot stop" in message.lower()
        assert str(state_path) in message

    def test_clears_the_state_file_without_signalling_when_unreachable(self, tmp_path: Path):
        """PID recycled onto a stranger's process: clean up, signal nobody.

        This is the same situation ``bridge_status`` reports as STALE, so
        ``stop`` must be able to clear it — otherwise the user is left with a
        state file no command will remove.
        """
        from kitty.bridge.manage import ProcessLiveness, stop_bridge

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=False),
            patch("kitty.bridge.manage.os.kill") as mock_kill,
        ):
            stop_bridge(state_path)

        mock_kill.assert_not_called()
        assert not state_path.exists()

    def test_a_bridge_we_own_is_still_stopped(self, tmp_path: Path):
        """R6: the ALIVE path is untouched by this change."""
        from kitty.bridge.manage import ProcessLiveness, stop_bridge

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        # Alive on the first probe, gone on every probe after the signal.
        probes: list[int] = []

        def _liveness(_pid: int):
            probes.append(_pid)
            return ProcessLiveness.ALIVE if len(probes) == 1 else ProcessLiveness.DEAD

        with (
            patch("kitty.bridge.manage.probe_pid", side_effect=_liveness),
            patch("kitty.bridge.manage.os.kill") as mock_kill,
        ):
            stop_bridge(state_path)

        mock_kill.assert_called_once_with(4321, signal.SIGTERM)
        assert not state_path.exists()


class TestStartBridgeWithAProcessWeMayNotSignal:
    """R5: refuse rather than start a second bridge beside an orphan."""

    @staticmethod
    def _write_state(state_path: Path) -> None:
        write_state(
            state_path,
            BridgeState(
                pid=4321,
                host="127.0.0.1",
                port=8080,
                profile="test",
                started_at="2026-08-19T10:30:00Z",
                tls=False,
            ),
        )

    def test_refuses_and_spawns_nothing_when_reachable(self, tmp_path: Path, capsys):
        from kitty.bridge.manage import ProcessLiveness, start_bridge

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=True),
            patch("kitty.bridge.manage.subprocess.Popen") as mock_popen,
            pytest.raises(SystemExit) as exc_info,
        ):
            start_bridge(state_path=state_path)

        assert exc_info.value.code == 1
        mock_popen.assert_not_called()
        assert state_path.exists(), "the running bridge's state file was cleared"

        message = capsys.readouterr().err
        assert "4321" in message
        assert "127.0.0.1:8080" in message
        assert "cannot manage" in message.lower()
        # The way out must be one the user can act on: `bridge start` takes no
        # --port flag, so the message names the state file instead.
        assert str(state_path) in message

    def test_proceeds_when_the_recycled_pid_serves_nothing(self, tmp_path: Path):
        """Unreachable means the PID was recycled — the state file is stale."""
        from kitty.bridge.manage import ProcessLiveness, start_bridge

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        # The spawned child is what writes the state file; stand in for it so
        # start_bridge takes its success path instead of timing out.
        def _spawn(*_args, **_kwargs):
            self._write_state(state_path)
            return SimpleNamespace(poll=lambda: None, stdout=io.BytesIO(), returncode=None)

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=False),
            patch("kitty.bridge.manage.subprocess.Popen", side_effect=_spawn) as mock_popen,
        ):
            start_bridge(state_path=state_path)

        mock_popen.assert_called_once()


class TestRestartWithAProcessWeMayNotSignal:
    """`bridge status` tells the user kitty cannot restart it — prove that.

    ``restart_bridge`` calls stop then start, and both refuse independently.
    Without this test, a change that made ``stop`` return quietly on this branch
    would let ``restart`` spawn a second bridge with nothing going red.
    """

    def test_restart_aborts_in_the_stop_phase(self, tmp_path: Path):
        from kitty.bridge.manage import ProcessLiveness, restart_bridge

        state_path = tmp_path / "state.json"
        write_state(
            state_path,
            BridgeState(
                pid=4321,
                host="127.0.0.1",
                port=8080,
                profile="test",
                started_at="2026-08-19T10:30:00Z",
                tls=False,
            ),
        )

        with (
            patch("kitty.bridge.manage.probe_pid", return_value=ProcessLiveness.UNKNOWN),
            patch("kitty.bridge.manage.bridge_reachable", return_value=True),
            patch("kitty.bridge.manage.subprocess.Popen") as mock_popen,
            patch("kitty.bridge.manage.os.kill") as mock_kill,
            pytest.raises(SystemExit) as exc_info,
        ):
            restart_bridge(state_path=state_path)

        assert exc_info.value.code == 1
        mock_kill.assert_not_called()
        mock_popen.assert_not_called()
        assert state_path.exists()


class TestBridgeReachable:
    """TCP reachability of the address recorded in the state file.

    Real sockets, not mocks: the behaviour under test *is* the socket
    behaviour, so a mock would only assert that the code calls the function it
    obviously calls.
    """

    def test_listening_socket_is_reachable(self):
        from kitty.bridge.manage import bridge_reachable

        server = socket.socket()
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            _, port = server.getsockname()
            assert bridge_reachable("127.0.0.1", port) is True
        finally:
            server.close()

    # "" is included for completeness, not as coverage: getaddrinfo("") already
    # yields loopback, so that case passes even with the normalisation removed.
    @pytest.mark.parametrize("bind_host", ["0.0.0.0", ""])
    def test_wildcard_bind_is_reachable_over_loopback(self, bind_host: str):
        """A bridge on 0.0.0.0 records 0.0.0.0, which is not a connectable address.

        Windows rejects a connect to the wildcard with WSAEADDRNOTAVAIL (10049)
        even while the socket is listening, so without normalisation kitty would
        report a live foreign bridge as gone — and ``stop`` would then delete its
        state file. A wildcard bind is the natural configuration for exactly the
        shared bridge this feature is about.
        """
        from kitty.bridge.manage import bridge_reachable

        server = socket.socket()
        try:
            server.bind(("0.0.0.0", 0))
            server.listen(1)
            _, port = server.getsockname()
            assert bridge_reachable(bind_host, port) is True
        finally:
            server.close()

    @pytest.mark.parametrize("bind_host", ["::", "::0", "0:0:0:0:0:0:0:0"])
    def test_ipv6_wildcard_bind_is_reachable_over_loopback(self, bind_host: str):
        """`::` is not the only spelling of the IPv6 unspecified address.

        ``bridge.yaml`` passes ``host`` through untouched and getaddrinfo
        accepts every one of these, so each can reach the state file.
        """
        from kitty.bridge.manage import bridge_reachable

        server = socket.socket(socket.AF_INET6)
        try:
            server.bind(("::", 0))
            server.listen(1)
            port = server.getsockname()[1]
            assert bridge_reachable(bind_host, port) is True
        finally:
            server.close()

    def test_closed_port_is_not_reachable(self):
        """Same address after the listener goes away — a stale state file."""
        from kitty.bridge.manage import bridge_reachable

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        _, port = server.getsockname()
        server.close()

        assert bridge_reachable("127.0.0.1", port) is False

    def test_unroutable_address_returns_false_without_raising(self):
        """TEST-NET-1 (RFC 5737) is reserved and never routed.

        The probe runs on user-facing commands, so an address that neither
        accepts nor refuses must resolve to a bounded ``False`` rather than
        hanging or propagating a timeout.
        """
        from kitty.bridge.manage import bridge_reachable

        assert bridge_reachable("192.0.2.1", 9, timeout=0.2) is False

    def test_a_real_address_is_not_rewritten_to_loopback(self):
        """Only the unspecified address is normalised — nothing else.

        A listener runs on loopback while the probe targets a different, real
        address on the same port. Over-eager normalisation would redirect the
        probe to the local listener and claim a bridge that is not there.
        """
        from kitty.bridge.manage import bridge_reachable

        server = socket.socket()
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            _, port = server.getsockname()
            assert bridge_reachable("192.0.2.1", port, timeout=0.2) is False
        finally:
            server.close()

    @pytest.mark.parametrize(
        ("host", "expected"),
        [
            ("0.0.0.0", "127.0.0.1"),
            ("", "127.0.0.1"),
            ("::", "::1"),
            ("::0", "::1"),
            ("0:0:0:0:0:0:0:0", "::1"),
            # The IPv4-mapped wildcard names an IPv4 bind, so it wants an IPv4
            # loopback -- decided from ``ipv4_mapped``, never from the
            # patch-dependent ``is_unspecified``. See KBR-146.
            ("::ffff:0.0.0.0", "127.0.0.1"),
            ("::ffff:0:0", "127.0.0.1"),
            # A mapped address that is not a wildcard is a real bind target and
            # is probed exactly as recorded, like any other real address.
            ("::ffff:127.0.0.1", "::ffff:127.0.0.1"),
            ("::ffff:192.0.2.1", "::ffff:192.0.2.1"),
            ("127.0.0.1", "127.0.0.1"),
            ("192.0.2.1", "192.0.2.1"),
            ("::1", "::1"),
            ("localhost", "localhost"),
            ("bridge.internal", "bridge.internal"),
        ],
    )
    def test_connect_target_table(self, host: str, expected: str):
        """The normalisation itself, with no socket in the way."""
        from kitty.bridge.manage import _connect_target

        assert _connect_target(host) == expected

    @pytest.mark.parametrize("property_reports", [False, True])
    def test_ipv4_mapped_wildcard_resolves_alike_whatever_is_unspecified_reports(
        self, monkeypatch: pytest.MonkeyPatch, property_reports: bool
    ):
        """The mapped wildcard resolves alike on both sides of CPython gh-122792.

        Args:
            monkeypatch: pytest's patching fixture, which reverts the stdlib
                property at teardown.
            property_reports: What ``IPv6Address.is_unspecified`` is forced to
                report -- ``False`` reproduces a pre-backport interpreter,
                ``True`` a patched one.

        ``IPv6Address.is_unspecified`` delegates to the mapped IPv4 address from
        3.10.16, 3.11.11, 3.12.7 and 3.13.1 onwards, and does not below them.
        ``requires-python = ">=3.10"`` admits every one of those releases, so a
        single interpreter can only ever demonstrate its own side of the change.
        Forcing the property is the only way to hold both sides inside one run,
        and it is the exact seam the defect came through: the pre-fix function
        short-circuited on this property and handed back the wildcard unchanged.

        The claim is output invariance, not non-consultation -- an
        implementation that reads the property and discards its answer is legal
        and passes.
        """
        from kitty.bridge.manage import _connect_target

        # Patched on IPv6Address, which owns the property outright on every
        # supported branch -- so this shadows nothing and leaves no residue.
        monkeypatch.setattr(
            ipaddress.IPv6Address,
            "is_unspecified",
            property(lambda self: property_reports),
        )

        # Without this the test has a silent no-op mode: an implementation that
        # stopped routing through ``ipaddress`` would make the patch inert and
        # leave this a duplicate of the table row, still claiming to prove
        # version-independence.
        assert ipaddress.ip_address("::ffff:0.0.0.0").is_unspecified is property_reports

        assert _connect_target("::ffff:0.0.0.0") == "127.0.0.1"

    def test_unresolvable_hostname_returns_false(self):
        """DNS failure is an OSError subclass and must not escape either."""
        from kitty.bridge.manage import bridge_reachable

        assert bridge_reachable("kitty-bridge.invalid", 9, timeout=0.2) is False


class TestTheWindowsLivenessDecisions:
    """The Windows probe's decisions, checked on every platform.

    🔴 Raised in PR review: the ``ctypes`` body of ``_probe_pid_windows`` runs
    on one leg of six, so the mapping it implements was proved only where it
    is hardest to run and impossible to provoke -- nothing can produce an
    ``ERROR_ACCESS_DENIED`` process on demand in CI.

    The decisions are therefore pure functions, the shape TEST_SUITE.md §8.1
    requires of every decision in this codebase, and these cases hand each one
    the values Windows would. What stays Windows-only is the ``ctypes`` call
    itself, which is plumbing rather than a decision.
    """

    def test_access_denied_means_the_process_exists(self):
        """ERROR_ACCESS_DENIED is the opposite conclusion from a missing PID."""
        from kitty.bridge.manage import ProcessLiveness, liveness_from_open_failure

        assert liveness_from_open_failure(5) is ProcessLiveness.UNKNOWN

    def test_any_other_open_failure_means_dead(self):
        """ERROR_INVALID_PARAMETER (87) is what a PID nothing holds produces."""
        from kitty.bridge.manage import ProcessLiveness, liveness_from_open_failure

        assert liveness_from_open_failure(87) is ProcessLiveness.DEAD
        assert liveness_from_open_failure(0) is ProcessLiveness.DEAD

    def test_a_wait_that_times_out_means_alive(self):
        """🔴 The inversion: a process handle is signalled once it EXITS.

        So the wait timing out (``WAIT_TIMEOUT``, 0x102) is the sign of life,
        and the wait succeeding is the death certificate. Reading this the
        natural way round is the mistake this case exists to catch.
        """
        from kitty.bridge.manage import ProcessLiveness, liveness_from_wait

        assert liveness_from_wait(0x102) is ProcessLiveness.ALIVE

    def test_a_wait_that_completes_means_dead(self):
        """WAIT_OBJECT_0 (0) means the handle is signalled: the process exited."""
        from kitty.bridge.manage import ProcessLiveness, liveness_from_wait

        assert liveness_from_wait(0) is ProcessLiveness.DEAD


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the POSIX errno mapping of os.kill(pid, 0); Windows never calls it (KBR-180)",
)
class TestProbePidErrorMapping:
    """Mapping of ``os.kill(pid, 0)`` errno outcomes to liveness, on POSIX.

    🔴 This class used to call itself *cross-platform* and open with "Signal 0
    performs no action and only probes the process — on Windows as well as
    POSIX". That claim was false, and it is the one KBR-180 was filed against:
    ``signal.CTRL_C_EVENT`` **is** ``0`` on Windows, so the call broadcasts a
    Ctrl+C to the console instead of probing. The tests below patch
    ``manage.os.kill`` and assert the errno mapping around it, which is a POSIX
    claim about a POSIX call.

    Skipped on Windows because the behaviour under test **does not exist**
    there — §8's one permitted kind of skip — not because the product is
    broken there. :func:`probe_pid` dispatches to ``_probe_pid_windows``
    before reaching ``os.kill``, and that path is covered by
    :meth:`TestBridgeManagementHelpers.test_probe_pid_for_current_process` and
    ``test_probe_pid_for_dead_pid``, which exercise the real implementation on
    the Windows leg, plus
    ``test_probe_pid_never_signals_zero_on_windows``, which holds the dispatch
    itself on every platform.
    """

    def test_signallable_process_is_alive(self):
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        with patch("kitty.bridge.manage.os.kill", return_value=None) as mock_kill:
            assert probe_pid(4321) is ProcessLiveness.ALIVE

        mock_kill.assert_called_once_with(4321, 0)

    def test_process_lookup_error_means_dead(self):
        """POSIX reports a missing PID as ProcessLookupError (ESRCH)."""
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        with patch("kitty.bridge.manage.os.kill", side_effect=ProcessLookupError()):
            assert probe_pid(4321) is ProcessLiveness.DEAD

    def test_permission_denied_means_unknown_not_dead(self):
        """EPERM proves the process exists — it is simply not ours to signal.

        Reporting it as dead made ``bridge status`` claim STALE for a running
        bridge, ``start`` launch a second one, and ``stop`` delete the state
        file without stopping anything. Reporting it as alive would have been
        worse: ``stop`` would then signal whatever inherited a recycled PID.
        """
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        with patch("kitty.bridge.manage.os.kill", side_effect=PermissionError()):
            assert probe_pid(4321) is ProcessLiveness.UNKNOWN

    @pytest.mark.parametrize("pid", [0, -1, -12345])
    def test_non_positive_pids_are_never_alive(self, pid: int):
        """A PID <= 0 is not a process and must never be signalled.

        On POSIX, ``kill(0, sig)`` targets the caller's entire process group and
        ``kill(-1, sig)`` targets every process the caller may signal. A corrupt
        or hand-edited bridge_state.json carrying such a value would otherwise
        make ``stop_bridge`` SIGTERM then SIGKILL the user's own shell session.
        """
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        with patch("kitty.bridge.manage.os.kill") as mock_kill:
            assert probe_pid(pid) is ProcessLiveness.DEAD

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
        from kitty.bridge.manage import ProcessLiveness, probe_pid

        winerror_87 = OSError(22, "The parameter is incorrect", None, 87, None)
        with patch("kitty.bridge.manage.os.kill", side_effect=winerror_87):
            assert probe_pid(999999999) is ProcessLiveness.DEAD

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


class TestStopBridgeForceKillIsCrossPlatform:
    """`kitty bridge stop` must work on Windows, where SIGKILL does not exist.

    The force-kill branch runs when a bridge has not exited ~10s after SIGTERM —
    i.e. exactly when the user is trying to clear a wedged process. Raising there
    also skipped `remove_state`, leaving behind the stale state file the command
    exists to clean up.
    """

    @staticmethod
    def _write_state(state_path: Path, pid: int = 4321) -> None:
        write_state(
            state_path,
            BridgeState(
                pid=pid,
                host="127.0.0.1",
                port=8080,
                profile="test",
                started_at="2026-04-11T10:30:00Z",
                tls=False,
            ),
        )

    def test_falls_back_to_sigterm_when_sigkill_is_unavailable(self, tmp_path: Path):
        """Simulates win32, where signal.SIGKILL is absent."""
        from kitty.bridge import manage

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        windows_signal = SimpleNamespace(SIGTERM=signal.SIGTERM)
        assert not hasattr(windows_signal, "SIGKILL")

        with (
            patch.object(manage, "signal", windows_signal),
            patch.object(manage, "probe_pid", return_value=manage.ProcessLiveness.ALIVE),
            patch.object(manage, "time"),
            patch.object(manage.os, "kill") as mock_kill,
        ):
            manage.stop_bridge(state_path)

        signals_sent = [call.args[1] for call in mock_kill.call_args_list]
        assert signals_sent, "no signal was sent to the wedged process"
        assert all(sig == signal.SIGTERM for sig in signals_sent)
        assert not state_path.exists(), "the stale state file was left behind"

    @pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX-only behaviour")
    def test_uses_sigkill_where_available(self, tmp_path: Path):
        """POSIX behaviour must be unchanged — SIGTERM first, then SIGKILL."""
        from kitty.bridge import manage

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch.object(manage, "probe_pid", return_value=manage.ProcessLiveness.ALIVE),
            patch.object(manage, "time"),
            patch.object(manage.os, "kill") as mock_kill,
        ):
            manage.stop_bridge(state_path)

        signals_sent = [call.args[1] for call in mock_kill.call_args_list]
        assert signals_sent[0] == signal.SIGTERM
        assert signals_sent[-1] == signal.SIGKILL

    def test_state_file_is_removed_even_if_the_signal_fails(self, tmp_path: Path):
        from kitty.bridge import manage

        state_path = tmp_path / "state.json"
        self._write_state(state_path)

        with (
            patch.object(manage, "probe_pid", return_value=manage.ProcessLiveness.ALIVE),
            patch.object(manage, "time"),
            patch.object(manage.os, "kill", side_effect=ProcessLookupError),
        ):
            manage.stop_bridge(state_path)

        assert not state_path.exists()


class TestStartBridgeReportingAChildThatDiedAtImport:
    """Reporting a bridge child that exited before it wrote its state file.

    The parent explains the failure by reading the dead child's output back. A child
    that never reached :func:`kitty.bridge_runner.main` never ran
    :func:`kitty.io_encoding.harden_output_streams`, so those bytes carry the machine's
    locale codepage rather than UTF-8, and a strict decode of them killed the parent
    mid-diagnostic (KBR-154). Every case here supplies the child's bytes as a literal,
    never from the host locale, so the claim is proven on any runner.
    """

    # cp1251 for 'File "C:/Users/Пётр/boot.py", line 1', in the form CPython's default
    # stderr emits it: `backslashreplace` escapes only what the codepage cannot encode,
    # so an encodable non-ASCII name goes out as raw single bytes.
    _CP1251_TRACEBACK = b'File "C:/Users/\xcf\xb8\xf2\xf0/boot.py", line 1'

    @staticmethod
    def _dead_child(stderr_bytes: bytes) -> SimpleNamespace:
        """Build a stand-in for a child that exited before writing its state file.

        Args:
            stderr_bytes: The bytes the child is to have written to its stderr.

        Returns:
            An object exposing the three attributes ``start_bridge`` reads from a
            :class:`subprocess.Popen`: ``poll``, ``returncode`` and ``stdout``, which
            carries the child's stderr since KBR-176 merged the two streams. It hands
            over one byte per read, so a decode of each read rather than of the whole
            output would split every multi-byte character and fail the verbatim case.
        """
        return SimpleNamespace(poll=lambda: 1, returncode=1, stdout=_OneBytePerRead(stderr_bytes))

    def _report(self, tmp_path: Path, stderr_bytes: bytes) -> SystemExit:
        """Run ``start_bridge`` against a dead child and return the exit it raised.

        Args:
            tmp_path: Directory for the state file, which is never written, so
                ``start_bridge`` takes its failure branch.
            stderr_bytes: The bytes the stand-in child is to have written to stderr.

        Returns:
            The :exc:`SystemExit` raised by :func:`kitty.bridge.manage.start_bridge`.
        """
        from kitty.bridge.manage import start_bridge

        with (
            patch(
                "kitty.bridge.manage.subprocess.Popen",
                return_value=self._dead_child(stderr_bytes),
            ),
            pytest.raises(SystemExit) as excinfo,
        ):
            start_bridge(state_path=tmp_path / "state.json", host="127.0.0.1", port=0)

        return excinfo.value

    def test_a_child_whose_stderr_is_not_utf8_does_not_kill_the_parent(self, tmp_path: Path, capsys):
        """Exit 1 rather than dying on the decode of the child's stderr.

        That the child's own text survives is
        :meth:`test_the_undecodable_bytes_are_rendered_beside_the_ascii`'s claim; this one
        pins only that the parent reaches its exit instead of raising.
        """
        exit_exc = self._report(tmp_path, self._CP1251_TRACEBACK)

        assert exit_exc.code == 1
        assert "Bridge failed to start (exit code 1)" in capsys.readouterr().err

    def test_the_undecodable_bytes_are_rendered_beside_the_ascii(self, tmp_path: Path, capsys):
        """Render the bytes UTF-8 cannot decode instead of dropping them."""
        self._report(tmp_path, self._CP1251_TRACEBACK)
        err = capsys.readouterr().err

        # The ASCII either side of the operator's account name is what makes the
        # child's traceback readable at all, so it must survive untouched.
        assert 'File "C:/Users/' in err
        assert '/boot.py", line 1' in err

        # Escapes, not U+FFFD: the byte values stay on screen, which is how a support
        # engineer works out which codepage — and so which install — broke.
        assert r"\xf2" in err
        assert r"\xf0" in err
        assert "\ufffd" not in err, "errors='replace' would satisfy neither KBR-154 AC2 nor this"

    def test_a_child_whose_stderr_is_valid_utf8_is_reported_verbatim(self, tmp_path: Path, capsys):
        """Leave a decodable diagnostic exactly as the child wrote it."""
        self._report(tmp_path, "Ошибка импорта".encode())

        assert "Ошибка импорта" in capsys.readouterr().err


class TestStartBridgeWithALoudChild:
    """A bridge child that writes more than one pipe buffer before its state file (KBR-176).

    A pipe holds a bounded amount before its writer blocks — about 64 KiB on Linux, less on
    Windows. A parent that does not read while it waits leaves such a child blocked in
    ``write()``: it never writes its state and never exits, so the operator was told
    "Bridge started but state file not found" while the child's diagnostic sat unread in the
    buffer that wedged it. The wedge needs a real pipe and a writer that really blocks, so
    these cases spawn real children. They are ``python -c`` scripts rather than
    ``kitty.bridge_runner``, which would refresh the catalog over the network; only the
    command is swapped, and ``start_bridge``'s own pipe wiring reaches the real
    :class:`subprocess.Popen` unchanged.
    """

    # More than the default pipe buffer on every CI platform, by a wide margin.
    _LOUD_BYTES = 256 * 1024

    # Captured before any patch: ``kitty.bridge.manage.subprocess`` is this same module.
    _REAL_POPEN = subprocess.Popen

    @pytest.fixture
    def children(self):
        """Collect the children a case spawns, then kill each one and close its pipe.

        The order matters: closing a pipe while ``start_bridge``'s reader is blocked on it
        waits for the child to exit, and closing it between two reads raises inside that
        reader. So the child is killed, reaped, and its reader allowed to finish first; the
        reader is found by the name production gives it.

        Yields:
            list[subprocess.Popen]: Filled by :meth:`_spawning` as children start.
        """
        spawned: list[subprocess.Popen] = []
        from kitty.bridge.manage import OUTPUT_READER_NAME

        yield spawned
        for child in spawned:
            child.kill()
            child.wait(timeout=30)
        for reader in threading.enumerate():
            if reader.name == OUTPUT_READER_NAME:
                reader.join(timeout=30)
        for child in spawned:
            if child.stdout is not None:
                child.stdout.close()

    def _spawning(self, script: str, children: list[subprocess.Popen]):
        """Build a ``Popen`` replacement that runs ``script`` with the caller's own wiring.

        Args:
            script: Python source for the child, run as ``python -c``.
            children: Receives every child started, so the fixture can kill it.

        Returns:
            A callable with :class:`subprocess.Popen`'s signature.
        """

        def _spawn(_cmd, **kwargs):
            """Start ``script`` in place of ``_cmd``, keeping every keyword argument.

            Args:
                _cmd: The command ``start_bridge`` built, which is not run.
                **kwargs: ``start_bridge``'s own arguments to :class:`subprocess.Popen`.

            Returns:
                subprocess.Popen: The started child.
            """
            child = self._REAL_POPEN([sys.executable, "-c", script], **kwargs)
            children.append(child)
            return child

        return _spawn

    @pytest.mark.parametrize("stream", ["stdout", "stderr"])
    def test_a_child_louder_than_a_pipe_buffer_still_comes_up(self, stream: str, tmp_path: Path, children, capsys):
        """Report the URL for a child that is loud before it writes its state.

        Both streams are cases: before KBR-176 the parent read neither while it waited, and
        a stdout that is never read wedges a child exactly as stderr does.
        """
        from kitty.bridge.manage import start_bridge

        state_path = tmp_path / "state.json"
        state = {
            "host": "127.0.0.1",
            "port": 54321,
            "profile": "loud",
            "started_at": "2026-09-12T20:00:00Z",
            "tls": False,
        }
        # Renamed into place: start_bridge loads the file the instant it exists
        script = (
            "import json, os, sys, time\n"
            f"getattr(sys, {stream!r}).write('w' * {self._LOUD_BYTES})\n"
            f"getattr(sys, {stream!r}).flush()\n"
            f"state = dict({state!r}, pid=os.getpid())\n"
            f"with open({str(state_path) + '.tmp'!r}, 'w') as f:\n"
            "    json.dump(state, f)\n"
            f"os.replace({str(state_path) + '.tmp'!r}, {str(state_path)!r})\n"
            "time.sleep(120)\n"
        )

        with patch("kitty.bridge.manage.subprocess.Popen", side_effect=self._spawning(script, children)):
            start_bridge(state_path=state_path, host="127.0.0.1", port=0)

        assert "http://127.0.0.1:54321" in capsys.readouterr().out

    def test_the_cli_exits_cleanly_while_its_reader_still_waits_on_the_bridge(self, tmp_path: Path):
        """Let the parent process end normally with the output reader still blocked.

        After every successful start, and every start that outlasts the window, ``kitty``
        exits while its daemon reader is still inside a read on the live bridge's pipe. How
        an interpreter shuts down around such a thread is platform behaviour, so it is run
        rather than reasoned about: a separate interpreter plays ``kitty``, and must exit 0,
        promptly, with no fatal error. The bridge is loud first, so this also fails before
        KBR-176.
        """
        state_path = tmp_path / "state.json"
        bridge = (
            "import json, os, sys, time\n"
            f"sys.stderr.write('w' * {self._LOUD_BYTES})\n"
            "sys.stderr.flush()\n"
            "state = dict(host='127.0.0.1', port=54321, profile='loud', started_at='now', tls=False, pid=os.getpid())\n"
            f"with open({str(state_path) + '.tmp'!r}, 'w') as f:\n"
            "    json.dump(state, f)\n"
            f"os.replace({str(state_path) + '.tmp'!r}, {str(state_path)!r})\n"
            "time.sleep(120)\n"
        )
        cli = (
            "import subprocess, sys\n"
            "from unittest.mock import patch\n"
            "from kitty.bridge.manage import start_bridge\n"
            "real = subprocess.Popen\n"
            "def spawn(_cmd, **kwargs):\n"
            f"    return real([sys.executable, '-c', {bridge!r}], **kwargs)\n"
            "with patch('kitty.bridge.manage.subprocess.Popen', side_effect=spawn):\n"
            f"    start_bridge(state_path={str(state_path)!r}, host='127.0.0.1', port=0)\n"
        )

        try:
            finished = subprocess.run(
                [sys.executable, "-c", cli], capture_output=True, text=True, errors="replace", timeout=60
            )
        finally:
            # The bridge outlives the CLI by design; its state file is how it is found
            with contextlib.suppress(OSError, ValueError, KeyError):
                os.kill(json.loads(state_path.read_text())["pid"], getattr(signal, "SIGKILL", signal.SIGTERM))

        assert finished.returncode == 0, finished.stderr
        assert "Fatal Python error" not in finished.stderr
        assert "http://127.0.0.1:54321" in finished.stdout

    def test_a_loud_child_that_dies_is_reported_with_everything_it_wrote(self, tmp_path: Path, children, capsys):
        """Show a dead child's diagnostic from both streams, past a buffer of noise.

        The stdout marker is what separates the chosen design, both streams merged into the
        one the parent reads, from discarding stdout: a ``DEVNULL`` stdout would still pass
        every other assertion here.
        """
        from kitty.bridge.manage import start_bridge

        script = (
            "import sys\n"
            f"sys.stdout.write('w' * {self._LOUD_BYTES})\n"
            "sys.stdout.write('\\nkbr176 marker written to stdout\\n')\n"
            "sys.stdout.flush()\n"
            "sys.stderr.write(\"ModuleNotFoundError: No module named 'kbr176_absent'\\n\")\n"
            "sys.stderr.flush()\n"
            "sys.exit(1)\n"
        )

        with (
            patch("kitty.bridge.manage.subprocess.Popen", side_effect=self._spawning(script, children)),
            pytest.raises(SystemExit) as excinfo,
        ):
            start_bridge(state_path=tmp_path / "state.json", host="127.0.0.1", port=0)

        err = capsys.readouterr().err
        assert excinfo.value.code == 1
        assert "Bridge failed to start (exit code 1)" in err
        assert "kbr176 marker written to stdout" in err
        assert "ModuleNotFoundError: No module named 'kbr176_absent'" in err
        assert "Bridge started" not in err

    def test_a_child_still_starting_is_left_running_and_not_called_started(self, tmp_path: Path, capsys):
        """Report only facts about a child still running without a state file, and leave it be.

        A healthy bridge can take longer than the window, and on some platforms its state file
        is not where the parent looks (KBR-220), so the report neither calls it started nor
        promises it will come up nor advises killing it. The stand-in behaves like the pipe of
        a live child: it never reaches its end, and a read for more than it holds waits. The
        window's sleeps each wait only until the reader has taken what is there, so the report
        is written with that output collected and no real time passes.
        """
        from kitty.bridge.manage import start_bridge

        output = _LivePipe(b"Refreshing the model-context catalog\n")
        child = SimpleNamespace(
            pid=48213,
            poll=lambda: None,
            returncode=None,
            stdout=output,
            terminate=Mock(),
            kill=Mock(),
        )
        clock = SimpleNamespace(sleep=lambda _s: output.caught_up.wait(0.1))

        try:
            with (
                patch("kitty.bridge.manage.subprocess.Popen", return_value=child),
                patch("kitty.bridge.manage.time", clock),
                patch("kitty.bridge.manage.os.kill") as os_kill,
                pytest.raises(SystemExit) as excinfo,
            ):
                start_bridge(state_path=tmp_path / "state.json", host="127.0.0.1", port=0)
        finally:
            output.released.set()

        err = capsys.readouterr().err
        assert excinfo.value.code == 1
        assert "did not report ready within 5 seconds and is still running (PID 48213)" in err
        assert "Refreshing the model-context catalog" in err
        assert "Bridge started" not in err
        assert "end process" not in err
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        os_kill.assert_not_called()

    def test_output_still_arriving_as_the_child_exits_is_waited_for(self, tmp_path: Path, capsys):
        """Report a dead child's last output even when it reaches the parent after the exit.

        The parent sees the exit through ``poll()``, which says nothing about whether the
        reader has drained the pipe yet. Here the output is held back until the exit has been
        seen, so a report written from whatever had arrived by then would be empty.
        """
        from kitty.bridge.manage import start_bridge

        output = _OutputAfterExit(b"ModuleNotFoundError: the line that lands last\n")
        child = SimpleNamespace(pid=48213, poll=output.see_exit, returncode=1, stdout=output)

        with (
            patch("kitty.bridge.manage.subprocess.Popen", return_value=child),
            pytest.raises(SystemExit) as excinfo,
        ):
            start_bridge(state_path=tmp_path / "state.json", host="127.0.0.1", port=0)

        assert excinfo.value.code == 1
        assert "ModuleNotFoundError: the line that lands last" in capsys.readouterr().err


class _LivePipe(io.BytesIO):
    """The output pipe of a child that is still running.

    It never reaches its end while the test holds it: a read that would find the end, or a
    ``read(n)`` asking for more than is there, waits until :attr:`released` is set, as a
    real pipe waits for its writer. That is what separates ``read1``, which hands over what
    has arrived, from ``read(n)``, which would sit on it.

    Attributes:
        caught_up: Set when a reader has taken everything written so far.
        released: Set by the test to let a waiting reader see the end.
    """

    def __init__(self, data: bytes) -> None:
        """Hold ``data`` as everything the child has written so far.

        Args:
            data: The bytes a reader is to find.
        """
        super().__init__(data)
        self.caught_up = threading.Event()
        self.released = threading.Event()

    def read1(self, size: int | None = -1) -> bytes:
        """Return what has arrived, waiting for release only once nothing is left.

        Args:
            size: The maximum number of bytes to return; all of them when negative.

        Returns:
            The bytes read; empty only after release.
        """
        chunk = super().read1(size)
        if not chunk:
            self._wait_for_release()
        return chunk

    def read(self, size: int | None = -1) -> bytes:
        """Wait for release first whenever the request cannot be met from what has arrived.

        Args:
            size: The number of bytes wanted; everything to the end when negative.

        Returns:
            The bytes read.
        """
        if size is None or size < 0 or size > len(self.getbuffer()) - self.tell():
            self._wait_for_release()
        return super().read(size)

    def _wait_for_release(self) -> None:
        """Signal that the reader has caught up, then wait, bounded, for the test's release."""
        self.caught_up.set()
        self.released.wait(10)


class _OneBytePerRead(io.BytesIO):
    """A dead child's output that hands over a single byte per ``read1``.

    Real reads end wherever the child's writes did, so a multi-byte character can straddle
    two of them; one byte per read makes that certain instead of occasional.
    """

    def read1(self, size: int | None = -1) -> bytes:
        """Read at most one byte.

        Args:
            size: Zero returns nothing; any other value returns at most one byte.

        Returns:
            The next byte, or empty at the end.
        """
        return super().read1(1 if size is None or size != 0 else 0)


class _OutputAfterExit(io.BytesIO):
    """A child's output stream whose data arrives only after the parent has seen the exit.

    Stands in for the last bytes still in flight when a child exits.
    """

    # Long enough that a report which does not wait for the reader is written first.
    _IN_FLIGHT_SECONDS = 0.2

    def __init__(self, data: bytes) -> None:
        """Hold ``data`` back until :meth:`see_exit` has been called.

        Args:
            data: The bytes the reader is to find once the exit has been seen.
        """
        super().__init__(data)
        self._exit_seen = threading.Event()

    def see_exit(self) -> int:
        """Report the child as exited, as ``Popen.poll`` would, releasing the data.

        Returns:
            The exit code, ``1``.
        """
        self._exit_seen.set()
        return 1

    def read1(self, size: int | None = -1) -> bytes:
        """Wait until the exit has been seen and the data is in flight, then read.

        Args:
            size: The maximum number of bytes to return; all of them when negative.

        Returns:
            The bytes read; empty at the end.
        """
        if self._exit_seen.wait(timeout=10) and self.tell() == 0:
            time.sleep(self._IN_FLIGHT_SECONDS)
        return super().read1(size)


class TestTheWindowsConsoleDetachment:
    """A background bridge must not share the console that started it (KBR-231).

    ``start_new_session=True`` is POSIX-only: on Windows the child keeps the
    launcher's console and its process group, so a Ctrl+C — or a closed window,
    which delivers ``CTRL_CLOSE_EVENT`` to every attached process — ends a
    bridge the user was told runs in the background. These cases are the
    ticket's observation-first probe and, once fixed, its guard: the bridge
    must be absent from the launching console, and must survive a console-wide
    ``CTRL_BREAK_EVENT``. The break, not a Ctrl+C, is the broadcast: the vendor
    docs state "CTRL+BREAK is always treated as a signal", so no ignore
    attribute can mask it, and it travels the same per-console-attachment
    channel a console close travels — ``CTRL_CLOSE_EVENT`` itself has no
    event-generating API, so the channel is what the probe exercises.

    They need a real console and real processes, so they run only on the
    Windows leg. The skip is the kind ``TEST_SUITE.md`` §8.4 permits — the
    console APIs do not exist on POSIX — and the gate's ``-rsfE`` keeps every
    one of these skips named on the POSIX legs. No layer marker is carried:
    the file's ``l1`` path default is what puts the probe on the Fast gate's
    Windows leg, which is where the observation-first acceptance criterion
    needs it (an ``l3`` marker would deselect it from every job, since no job
    runs ``l3`` yet).
    """

    # How long the launcher waits: for the control child to report readiness
    # (its PID file), for the go-file (the test reading the console snapshot
    # and proving the blast radius), and for the control child to die of the
    # broadcast before giving up and letting the test's own poll name the
    # broadcast as not lethal. Whole seconds: they are stringified into the
    # launcher script, where a rendered float would break range().
    _GO_FILE_SECONDS = 30
    _CONTROL_START_SECONDS = 30
    _CONTROL_EXIT_SECONDS = 15

    @staticmethod
    def _bridge_script(state_path: str) -> str:
        """Build the bridge stand-in: record state the way the runner does, then idle.

        A ``python -c`` stand-in rather than ``kitty.bridge_runner``, which
        would refresh the model-context catalog over the network — the same
        rule ``TestStartBridgeWithALoudChild`` states.

        Args:
            state_path: Where the stand-in writes its ``bridge_state.json``.

        Returns:
            Python source for ``python -c``.
        """
        return (
            "import json, os, time\n"
            "state = dict(host='127.0.0.1', port=54321, profile='console',"
            " started_at='now', tls=False, pid=os.getpid())\n"
            f"with open({state_path + '.tmp'!r}, 'w') as f:\n"
            "    json.dump(state, f)\n"
            f"os.replace({state_path + '.tmp'!r}, {state_path!r})\n"
            "time.sleep(120)\n"
        )

    @staticmethod
    def _start_bridge_child(
        script: str, state_path: Path, spawned: list[subprocess.Popen]
    ) -> None:
        """Run ``start_bridge`` with its real wiring, swapping only the child command.

        Args:
            script: Python source the child runs instead of ``bridge_runner``.
            state_path: The state path handed to ``start_bridge``.
            spawned: Receives the child, so the caller can kill and reap it.
        """
        from kitty.bridge.manage import start_bridge

        real_popen = subprocess.Popen

        def _spawn(_cmd, **kwargs):
            """Start ``script`` in place of ``_cmd``, keeping every keyword argument.

            Args:
                _cmd: The command ``start_bridge`` built, which is not run.
                **kwargs: ``start_bridge``'s own arguments to ``Popen``.

            Returns:
                subprocess.Popen: The started child.
            """
            child = real_popen([sys.executable, "-c", script], **kwargs)
            spawned.append(child)
            return child

        with patch("kitty.bridge.manage.subprocess.Popen", side_effect=_spawn):
            start_bridge(state_path=state_path, host="127.0.0.1", port=0)

    @staticmethod
    def _kill_pid(pid: int) -> None:
        """Terminate a child the test holds no handle for, by the PID it recorded.

        Args:
            pid: The PID to end.

        On Windows ``os.kill`` terminates unconditionally and needs no console,
        which is what makes it usable on a detached child — and on a PID that
        is already gone, which this suppresses rather than raising on.
        """
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))

    @classmethod
    def _cleanup(cls, state_path: Path, spawned: list[subprocess.Popen]) -> None:
        """Kill and reap every child a case started, then any known only by PID.

        The order matters: a handle-held child is killed and reaped first — a
        terminated process stays alive to a PID probe until every handle is
        gone — and only then is the PID recorded in the state file signalled,
        so the next case's probe can never mistake a corpse for a bridge.

        Args:
            state_path: The state file a stand-in may have written its PID to.
            spawned: The children this test still holds a ``Popen`` for.
        """
        recorded: int | None = cls._read_bridge_pid(state_path, deadline_seconds=5.0)
        for child in spawned:
            child.kill()
            child.wait(timeout=30)
            if child.stdout is not None:
                child.stdout.close()
        if recorded is not None:
            cls._kill_pid(recorded)

    @staticmethod
    def _read_bridge_pid(state_path: Path, *, deadline_seconds: float = 10.0) -> int | None:
        """Read the bridge PID from ``state_path``, polling until readable.

        ``start_bridge`` opens the state file for writing as part of its own
        bookkeeping; under CI load the bridge stand-in's write and that open
        can race, and ``read_text()`` on a freshly-truncated empty file returns
        ``""`` — which :func:`json.loads` turns into ``JSONDecodeError`` on
        line 1 col 1. The class docstring already states "an empty or
        truncated answer must fail loudly rather than read as 'the bridge is
        not attached'": this helper is the loud failure, with a deadline so
        a genuinely missing state file still surfaces fast.

        Args:
            state_path: The state file the bridge stand-in / ``start_bridge``
                wrote to.
            deadline_seconds: How long to keep polling before giving up.

        Returns:
            The recorded PID, or ``None`` if the file never became readable.
            Callers that need the PID must handle ``None`` themselves — the
            helper does not raise so the ``finally`` cleanup can stay
            exception-safe.
        """
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            try:
                text = state_path.read_text()
            except OSError:
                text = ""
            if text:
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = None
                pid = payload["pid"] if isinstance(payload, Mapping) else None
                if isinstance(pid, int):
                    return pid
            time.sleep(0.1)
        return None

    @staticmethod
    def _console_process_pids() -> set[int]:
        """Return the PIDs of every process attached to this process's console.

        Windows-only; every caller is skipped to ``win32``.

        Returns:
            The PIDs ``GetConsoleProcessList`` reports, in no guaranteed order.

        Raises:
            RuntimeError: When the calling process has no console, or the
                buffer was too small to hold every PID — the API stores
                *nothing* in that case, and an empty or truncated answer must
                fail loudly rather than read as "the bridge is not attached".
                A probe that observed nothing is TEST_SUITE.md §8's
                green-because-it-stopped-looking.
        """
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetConsoleProcessList.argtypes = (
            ctypes.POINTER(wintypes.DWORD),
            wintypes.DWORD,
        )
        kernel32.GetConsoleProcessList.restype = wintypes.DWORD

        # Generous fixed buffer, one call: a console has a handful of members,
        # never thousands, and the count return is what proves the read whole.
        buffer = (wintypes.DWORD * 1024)()
        count = kernel32.GetConsoleProcessList(buffer, len(buffer))
        if count == 0:
            raise RuntimeError("GetConsoleProcessList reports no console for this process")
        if count > len(buffer):
            raise RuntimeError("GetConsoleProcessList needs more room than the probe gave it")
        return set(buffer[:count])

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
    def test_a_background_bridge_is_absent_from_the_launching_console(
        self, tmp_path: Path
    ):
        """The bridge child is not a member of the console that started it.

        The structural half of the claim, and the load-bearing one: console
        control events are delivered per console attachment, so non-attachment
        is what makes a closed window survivable. The positive control comes
        first — the probe must see the very process asking, or an empty answer
        would be indistinguishable from a detached bridge.
        """
        state_path = tmp_path / "state.json"
        spawned: list[subprocess.Popen] = []
        try:
            self._start_bridge_child(self._bridge_script(str(state_path)), state_path, spawned)
            [child] = spawned

            pids = self._console_process_pids()
            assert os.getpid() in pids, "the probe must see its own console to mean anything"
            assert child.pid not in pids, (
                "the bridge child shares the launcher's console, so a Ctrl+C "
                "or a closed window in it would end the bridge (KBR-231)"
            )
        finally:
            self._cleanup(state_path, spawned)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
    def test_a_background_bridge_survives_a_console_break_in_its_launching_console(
        self, tmp_path: Path
    ):
        """A console-wide break in the launching console leaves the bridge running.

        The causal half, and the closer analogue of a closed window. The
        broadcast is issued by a launcher child with its **own** console, never
        by this test process: a group-0 control event reaches every process on
        the console, including the CI step's own shell, which no test can
        immunise. The launcher embeds the ``Popen``-swap — the
        ``TestStartBridgeWithALoudChild`` patch, replicated inside its own
        interpreter — so ``start_bridge``'s real wiring starts the stand-in,
        spawns a plainly-attached control child, and only then installs a
        handler that returns ``TRUE`` so it survives its own broadcast. A
        handler, unlike the ``NULL/TRUE`` ignore attribute, is not inherited by
        child processes; the ordering matters even so, because the attribute is
        not the only per-process state a child could arrive holding. The
        control child's death is confirmed before the bridge's survival is
        asserted: a broadcast that killed nothing would also "prove" the
        bridge's survival, for the wrong reason.
        """
        control_path = tmp_path / "control.pid"
        control_script = (
            "import os, time\n"
            f"open({str(control_path)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(120)\n"
        )
        state_path = tmp_path / "state.json"
        snapshot_path = tmp_path / "launcher_console.json"
        go_path = tmp_path / "go"
        launcher_script = (
            "import ctypes, json, os, subprocess, sys, time\n"
            "from unittest.mock import patch\n"
            "from kitty.bridge.manage import start_bridge\n"
            "real_popen = subprocess.Popen\n"
            "def spawn(_cmd, **kwargs):\n"
            f"    return real_popen([sys.executable, '-c', {self._bridge_script(str(state_path))!r}], **kwargs)\n"
            "with patch('kitty.bridge.manage.subprocess.Popen', side_effect=spawn):\n"
            f"    start_bridge(state_path={str(state_path)!r}, host='127.0.0.1', port=0)\n"
            f"control = subprocess.Popen([sys.executable, '-c', {control_script!r}],\n"
            "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "# The control child writes its PID as its first statement: waiting\n"
            "# for that file proves the child is fully started -- its console\n"
            "# control machinery included -- before anything is broadcast to\n"
            "# it. The first probe broadcast ~instantly after the spawn, and\n"
            "# the Windows leg showed the control child alive 12 s later: an\n"
            "# event delivered before the child could handle it is lost, not\n"
            "# survived, and would void the whole proof.\n"
            f"for _ in range({self._CONTROL_START_SECONDS * 20}):\n"
            f"    if os.path.exists({str(control_path)!r}):\n"
            "        break\n"
            "    time.sleep(0.05)\n"
            "else:\n"
            "    print('control child never reported its PID', file=sys.stderr)\n"
            "    sys.exit(5)\n"
            "# Publish this console's members and hold the broadcast until the\n"
            "# test has proven the blast radius and written the go-file: the\n"
            "# break must never fire before its isolation is a checked fact.\n"
            "buf = (ctypes.c_uint * 1024)()\n"
            "n = ctypes.windll.kernel32.GetConsoleProcessList(buf, len(buf))\n"
            f"json.dump(dict(launcher=os.getpid(), console=list(buf[:n])),\n"
            f"    open({str(snapshot_path)!r}, 'w'))\n"
            f"for _ in range({self._GO_FILE_SECONDS * 5}):\n"
            f"    if os.path.exists({str(go_path)!r}):\n"
            "        break\n"
            "    time.sleep(0.2)\n"
            "else:\n"
            "    print('go file never arrived', file=sys.stderr)\n"
            "    sys.exit(3)\n"
            "# A handler that returns TRUE stops the dispatch: the launcher\n"
            "# survives the console-wide break it is about to send. Installed\n"
            "# only after both children exist.\n"
            "HANDLER = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(lambda event: 1)\n"
            "ctypes.windll.kernel32.SetConsoleCtrlHandler(HANDLER, True)\n"
            "# A failed broadcast must fail loudly: the first probe ignored\n"
            "# this call's return value, so a broadcast that was never sent\n"
            "# and a child that survived a sent one were indistinguishable\n"
            "# in the recorded red run.\n"
            "if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, 0):\n"
            "    print('GenerateConsoleCtrlEvent failed', file=sys.stderr)\n"
            "    sys.exit(4)\n"
            "# Waiting on the control child, not on a clock: its exit is the\n"
            "# observable that the broadcast reached the console's members. A\n"
            "# child that outlives the wait is left running and reported: the\n"
            "# launcher still exits 0, and the test's own poll names the\n"
            "# broadcast as not lethal -- the failure path this test promises.\n"
            "timed_out = False\n"
            "try:\n"
            f"    control.wait({self._CONTROL_EXIT_SECONDS})\n"
            "except subprocess.TimeoutExpired:\n"
            "    timed_out = True\n"
            "if timed_out:\n"
            f"    print('control child still alive after "
            f"{self._CONTROL_EXIT_SECONDS}s', file=sys.stderr)\n"
            "else:\n"
            "    print('control child exit code', control.returncode,\n"
            "        file=sys.stderr)\n"
            "time.sleep(0.5)\n"
        )
        launcher = subprocess.Popen(
            [sys.executable, "-c", launcher_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # A private console for the launcher and everything it starts —
            # the broadcast's whole blast radius. CREATE_NEW_CONSOLE is the
            # one flag the vendor doc describes outright: "a new console that
            # is accessible to the child process but not to the parent
            # process", with no allocation subtleties to argue about.
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        try:
            # The snapshot must be read — and the blast radius proven — before
            # the go-file authorises the broadcast: had the launcher landed on
            # this process's console, the group-0 break would reach the CI
            # step's shell, and this test would be manufacturing that event.
            deadline = time.monotonic() + self._GO_FILE_SECONDS
            while not snapshot_path.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            assert snapshot_path.exists(), "the launcher never published its console snapshot"
            snapshot = json.loads(snapshot_path.read_text())
            own_console = self._console_process_pids()
            assert own_console, "this test has no console to be disjoint from"
            assert snapshot["launcher"] not in own_console, (
                "the launcher was not given its own console; the broadcast's "
                "blast radius is not what this test assumes"
            )
            assert own_console.isdisjoint(snapshot["console"]), (
                "the launcher's console and this test's console share members; "
                "the broadcast would not be contained"
            )
            go_path.write_text("go")

            stdout, stderr = launcher.communicate(timeout=90)
            assert launcher.returncode == 0, (
                f"launcher failed ({launcher.returncode}): {stderr or stdout}"
            )

            from kitty.bridge.manage import ProcessLiveness, probe_pid

            control_pid = int(control_path.read_text())
            bridge_pid = self._read_bridge_pid(state_path, deadline_seconds=10.0)
            assert bridge_pid is not None, (
                f"the bridge never wrote {state_path} within the deadline; "
                "the console-break probe cannot prove survival"
            )

            deadline = time.monotonic() + 10
            while probe_pid(control_pid) is ProcessLiveness.ALIVE and time.monotonic() < deadline:
                time.sleep(0.2)
            assert probe_pid(control_pid) is not ProcessLiveness.ALIVE, (
                "the broadcast was not lethal to a console member, so it "
                "proves nothing about the bridge"
            )
            assert probe_pid(bridge_pid) is ProcessLiveness.ALIVE, (
                "the bridge died with its launching console (KBR-231)"
            )
        finally:
            self._cleanup(state_path, [])
            with contextlib.suppress(OSError, ValueError):
                self._kill_pid(int(control_path.read_text()))
            # A failed assertion above leaves the launcher parked in its
            # go-file wait -- it would exit 3 by itself, but only after the
            # whole wait budget. On the happy path this is a no-op.
            with contextlib.suppress(OSError):
                launcher.kill()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows console behaviour")
    def test_bridge_stop_still_ends_a_detached_bridge(self, tmp_path: Path):
        """``stop_bridge`` ends the bridge and clears its state, console or no console.

        The ticket's AC2 tail: detachment must not outlive the user's ability
        to stop the thing. ``os.kill`` on Windows terminates unconditionally
        and needs no console, so this holds before the fix as well — the case
        pins it through the change.
        """
        from kitty.bridge.manage import ProcessLiveness, probe_pid, stop_bridge

        state_path = tmp_path / "state.json"
        spawned: list[subprocess.Popen] = []
        try:
            self._start_bridge_child(self._bridge_script(str(state_path)), state_path, spawned)
            bridge_pid = self._read_bridge_pid(state_path, deadline_seconds=10.0)
            assert bridge_pid is not None, (
                f"the bridge never wrote {state_path} within the deadline; "
                "the stop probe cannot prove anything"
            )
            assert probe_pid(bridge_pid) is ProcessLiveness.ALIVE

            stop_bridge(state_path)

            assert not state_path.exists()
            assert probe_pid(bridge_pid) is not ProcessLiveness.ALIVE
        finally:
            self._cleanup(state_path, spawned)


class TestTheBackgroundSpawnDecision:
    """The per-platform detachment arguments ``start_bridge`` spawns the child with (KBR-231).

    A pure decision is tested on every leg by handing it the platform — the
    pattern ``test_probe_pid_never_signals_zero_on_windows`` established. That
    the defect is Windows-only is exactly why the POSIX legs must still check
    the Windows answer: they make up four of the six legs, and a Windows-only
    regression test could rot unnoticed in between.
    """

    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            ("win32", {"creationflags": 0x00000200 | 0x00000008}),
            ("linux", {"start_new_session": True}),
            ("darwin", {"start_new_session": True}),
        ],
    )
    def test_the_detachment_arguments_per_platform(self, platform: str, expected: dict):
        """Give the decision each platform and expect exactly its detachment keys.

        Args:
            platform: A ``sys.platform`` value to hand the decision.
            expected: The keyword arguments the decision must return for it.
        """
        from kitty.bridge.manage import background_spawn_kwargs

        assert background_spawn_kwargs(platform) == expected

    def test_the_windows_flags_are_the_documented_win32_values(self):
        """Pin the two constants to WinBase.h's literal values."""
        from kitty.bridge import manage

        # WinBase.h's values, pinned as literals: a near miss (0x2000 for 0x200,
        # say) would still type-check and still detach nothing.
        assert manage._CREATE_NEW_PROCESS_GROUP == 0x00000200
        assert manage._DETACHED_PROCESS == 0x00000008

    def test_the_posix_arguments_carry_no_windows_key(self):
        """Refuse the POSIX form any ``creationflags`` key, even a falsy one."""
        from kitty.bridge.manage import background_spawn_kwargs

        # POSIX Popen raises ValueError for a nonzero creationflags, so the
        # decision must not hand POSIX a Windows key even with a falsy value.
        assert "creationflags" not in background_spawn_kwargs("linux")

    def test_start_bridge_passes_the_decision_to_popen(self, tmp_path: Path):
        """The spawn site runs the child under the decision's own arguments.

        Captures the real ``Popen`` call and compares its detachment keys with
        the decision for the running platform: ``start_new_session=True`` kept
        on POSIX, ``creationflags`` on Windows, and never a key from the other.
        """
        from kitty.bridge import manage
        from kitty.bridge.manage import start_bridge

        captured: dict = {}

        def _spawn(_cmd, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(poll=lambda: 1, stdout=io.BytesIO(), returncode=1)

        with (
            patch("kitty.bridge.manage.subprocess.Popen", side_effect=_spawn),
            pytest.raises(SystemExit),
        ):
            start_bridge(state_path=tmp_path / "state.json", host="127.0.0.1", port=0)

        detachment = {
            key: captured[key]
            for key in ("start_new_session", "creationflags")
            if key in captured
        }
        assert detachment == manage.background_spawn_kwargs(sys.platform)
        # And nothing else drifted at the spawn site: the pipe wiring and the
        # merged stderr are the KBR-176 contract, the environment the egress
        # one.
        assert set(captured) == set(detachment) | {"stdout", "stderr", "env"}
        assert captured["stdout"] is subprocess.PIPE
        assert captured["stderr"] is subprocess.STDOUT
