"""Tests for R8: Bridge management commands."""

from __future__ import annotations

import json
import os
import signal
import socket
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
            # IPv4-mapped: unspecified, but wants an IPv4 loopback.
            ("::ffff:0.0.0.0", "127.0.0.1"),
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

    def test_unresolvable_hostname_returns_false(self):
        """DNS failure is an OSError subclass and must not escape either."""
        from kitty.bridge.manage import bridge_reachable

        assert bridge_reachable("kitty-bridge.invalid", 9, timeout=0.2) is False


class TestProbePidErrorMapping:
    """Cross-platform mapping of os.kill(pid, 0) outcomes to liveness.

    Signal 0 performs no action and only probes the process — on Windows as
    well as POSIX. The platforms differ in how they report a missing PID, and
    that difference previously crashed every stale-state code path on Windows.
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
