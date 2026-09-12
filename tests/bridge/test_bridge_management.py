"""Tests for R8: Bridge management commands."""

from __future__ import annotations

import io
import ipaddress
import json
import os
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
            return SimpleNamespace(poll=lambda: 1, stderr=None, returncode=1)

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
            return SimpleNamespace(poll=lambda: 1, stderr=None, returncode=1)

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
            return SimpleNamespace(poll=lambda: None, stderr=None, returncode=None)

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

    The parent explains the failure by reading the dead child's stderr back. A child
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
            :class:`subprocess.Popen`: ``poll``, ``returncode`` and ``stderr``.
        """
        return SimpleNamespace(poll=lambda: 1, returncode=1, stderr=io.BytesIO(stderr_bytes))

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
