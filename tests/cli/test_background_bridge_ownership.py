"""A background bridge under another user's account is never managed by ours.

KBR-96 / plan task T-I4; ``TEST_SUITE.md`` §6.3.2, row "Background bridge
owned by another user". The kernel refuses to signal another account's
process (POSIX ``EPERM``), and signalling a PID the caller does not own is
unsafe: the operating system may have recycled that PID onto something
unrelated. The product already refuses — ``stop_bridge`` exits without
signalling and leaves the state file, ``start_bridge`` refuses to spawn a
second bridge beside the foreign one, and ``restart_bridge`` aborts in its
stop phase. Nothing pinned any of it; these tests do.

**In-process management calls.** ``probe_pid``'s ``UNKNOWN`` outcome cannot be
reproduced without a second account, and a ``monkeypatch`` cannot cross the
CLI-subprocess boundary ``tests/cli/test_bridge_state_location.py`` (KBR-220)
uses — there ``sys.exit(1)`` becomes a returncode. So the tests import
``kitty.bridge.manage`` and call ``stop_bridge``/``start_bridge``/
``restart_bridge``/``bridge_status`` directly, patching exactly one seam:
``manage.probe_pid``. Everything else is real — the ``kitty.bridge_runner``
child process, its socket, the state file, the management code path. The
patched classification is stated per test; the state file records the
child's *own* PID — the seam is the classification, not the PID number.

**Isolation** follows the KBR-220 pattern (temporary home, ``XDG_*`` /
``WIN_PD_OVERRIDE_LOCAL_APPDATA`` redirects, a pre-write refusal if
platformdirs did not move, a fresh catalog cache so the child's model-context
refresh never touches the network inside ``start``'s 5-second window). This
module does not share that file's fixture: its needs are a strict subset (no
keys-file variants, no ``kitty`` CLI subprocesses), so a shared fixture would
couple two modules with different surfaces; the KBR-220 module remains the
reference for the pattern.

**Layer.** ``l3``: one subsystem (the management commands) plus its real
infrastructure (real child processes, real sockets, real state file).
Acknowledged debt in ``PENDING_ACTIVATION_LAYERS`` until T-K6 activates the
Subsystem job; the fast gate's ``-m "l1 or l2"`` excludes this file today.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kitty.bridge.manage import BridgeStatus, ProcessLiveness
from kitty.bridge.state import BridgeState, load_state, write_state

pytestmark = pytest.mark.l3

ROOT = Path(__file__).resolve().parent.parent.parent

# Any well-formed UUID: the profile references its credential by one.
_AUTH_REF = "0b6f3a52-7f2e-4f3b-9d6e-2a1c5e8b4d10"

# Generous against a cold runner, and only ever reached on failure.
_COMMAND_TIMEOUT_SECONDS = 60
_PROCESS_EXIT_TIMEOUT_SECONDS = 20

# Both scripts run inside the isolated environment: the first reports where
# kitty will look (the fixture refuses before writing anything unless that is
# the temp directory), the second installs a profile, a credential, the
# bridge.yaml and a fresh catalog cache there.
_WHERE = """\
import json
from pathlib import Path

import platformdirs

print(json.dumps({"config_dir": platformdirs.user_config_dir("kitty"),
                  "cache_dir": platformdirs.user_cache_dir("kitty"),
                  "home": str(Path.home())}))
"""

_INSTALL = f'''\
import shutil
from pathlib import Path

import platformdirs

from kitty.credentials.file_backend import FileBackend
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore
from kitty.providers import model_context

profile = Profile(name="ownership", provider="zai_coding", model="glm-4.6", auth_ref="{_AUTH_REF}", is_default=True)
ProfileStore().save(profile)
FileBackend().set("{_AUTH_REF}", "sk-not-a-real-key")

cache = model_context.REMOTE_OVERRIDES_CACHE_PATH
cache.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(model_context.__file__).with_name("model_context_overrides.json"), cache)

config_dir = Path(platformdirs.user_config_dir("kitty"))
(config_dir / "bridge.yaml").write_text("host: 127.0.0.1\\nport: 0\\n", encoding="utf-8")
'''


@dataclass
class IsolatedKitty:
    """An isolated kitty installation that real child processes run against.

    Attributes:
        root: Temporary directory holding everything below.
        home: The children's home directory.
        config_dir: The children's ``user_config_dir("kitty")``, not under
            ``home``.
        env: Environment for every child process.
        spawned: Processes a test started directly, stopped at teardown.
        outputs: Everything each child wrote to its log, read at teardown.
    """

    root: Path
    home: Path
    config_dir: Path
    env: dict[str, str]
    spawned: list[subprocess.Popen] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def child_log(self, name: str) -> Path:
        """Return a log path under the root for one child's combined output.

        Args:
            name: Distinguishing stem for the log file.

        Returns:
            A path that does not exist yet; the caller opens it for writing.
        """
        return self.root / f"{name}.log"


def _isolated_kitty(tmp_path: Path) -> Iterator[IsolatedKitty]:
    """Build the isolated kitty the ownership tests run against.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        The isolated installation, with every spawned child stopped at
        teardown.
    """
    home = tmp_path / "home"
    (home / ".config" / "kitty").mkdir(parents=True)
    # The default keys file: the tests never exercise the auth decision, but a
    # real install has one, and auth on keeps the child's startup identical to
    # the KBR-220 harness's.
    (home / ".config" / "kitty" / "bridge_keys.txt").write_text("ownership-client-key\n", encoding="utf-8")

    # KITTY_* settings in the developer's environment must not steer the child.
    env = {name: value for name, value in os.environ.items() if not name.startswith("KITTY_")}
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        WIN_PD_OVERRIDE_LOCAL_APPDATA=str(tmp_path / "localappdata"),
        PYTHONIOENCODING="utf-8",
        # The working tree first: from a git worktree an editable install can
        # resolve another checkout.
        PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])),
    )

    def _run(script: str) -> str:
        """Run ``script`` in the isolated environment and return what it printed."""
        done = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout

    # Refuse before writing anything: an unredirected platformdirs would mean
    # the developer's real profiles (the KBR-220 lesson).
    where = json.loads(_run(_WHERE))
    config_dir, cache_dir = Path(where["config_dir"]), Path(where["cache_dir"])
    if not (config_dir.is_relative_to(tmp_path) and cache_dir.is_relative_to(tmp_path) and Path(where["home"]) == home):
        pytest.fail(f"kitty's directories were not isolated (platformdirs older than 4.8?): {where}")
    # Negative control: with the config dir under the home, unfixed code
    # passes on stock Linux and the harness proves nothing.
    assert config_dir != home / ".config" / "kitty", where

    _run(_INSTALL)

    install = IsolatedKitty(root=tmp_path, home=home, config_dir=config_dir, env=env)
    try:
        yield install
    finally:
        # A red test must not leave a bridge serving.
        for proc in install.spawned:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)


@pytest.fixture
def isolated_kitty(tmp_path: Path) -> Iterator[IsolatedKitty]:
    """Build an isolated kitty installation and stop every child at teardown.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        The isolated installation.
    """
    yield from _isolated_kitty(tmp_path)


def _spawn_bridge(
    install: IsolatedKitty,
    state_file: Path,
    *extra: str,
    stdout: int | None = None,
) -> subprocess.Popen:
    """Start ``python -m kitty.bridge_runner`` directly, as a service would.

    Output goes to a file under the root, never a pipe, so a bridge that
    outlives the test cannot wedge on a full pipe buffer — except where a
    test deliberately asks for a pipe (the KBR-219 regression).

    Args:
        install: The isolated installation to run against.
        state_file: Where the child must record its state.
        *extra: Arguments after ``--state-file``.
        stdout: Override for the child's stdout. ``None`` keeps the default
            log file; ``subprocess.PIPE`` hands the test the read end, which
            is the parent-exit simulation of the KBR-219 regression.

    Returns:
        The started process, registered for teardown.
    """
    log = install.child_log(f"bridge-{state_file.stem}-{len(install.spawned)}")
    out = log.open("wb") if stdout is None else stdout
    proc = subprocess.Popen(
        [sys.executable, "-m", "kitty.bridge_runner", "--state-file", str(state_file), *extra],
        env=install.env,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    install.spawned.append(proc)
    # Reap the moment it exits, as a service manager would: an unreaped child
    # stays a zombie of this test process and still answers a liveness probe,
    # which would make every "did it die?" assertion wait out its timeout.
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def _bridge_output(install: IsolatedKitty) -> str:
    """Return every spawned child's combined output, for failure messages."""
    parts = []
    for log in sorted(install.root.glob("bridge-*.log")):
        parts.append(log.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(parts)
    install.outputs.append(text)
    return text


def _wait_for_state(state_file: Path, proc: subprocess.Popen, install: IsolatedKitty) -> BridgeState:
    """Wait until the bridge records its state, and fail with its output if it never does.

    Args:
        state_file: Where its state is expected.
        proc: The bridge process.
        install: The installation the bridge runs against, for the log.

    Returns:
        The recorded state.
    """
    deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
    while not state_file.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    output = _bridge_output(install)
    assert state_file.exists(), f"the bridge recorded no state at {state_file}:\n{output}"
    state = load_state(state_file)
    assert state is not None, f"the state file at {state_file} does not parse:\n{output}"
    return state


def _really_alive(pid: int) -> bool:
    """Report whether ``pid`` names a live process under the real kernel probe.

    Bypasses :func:`kitty.bridge.manage.probe_pid` on purpose: the tests
    monkey-patch that classifier to simulate foreign ownership, and the
    assertion helpers here need the *true* answer — "did the child's
    process go away?" — not the simulated classification.

    Args:
        pid: The process to check.

    Returns:
        True when the kernel reports the process exists. A ``PermissionError``
        also returns True (the process exists; we just may not signal it),
        matching ``probe_pid``'s ALIVE + UNKNOWN union for the purposes of
        these tests.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_until_gone(pid: int) -> bool:
    """Wait for a process to stop being alive.

    Args:
        pid: The process to watch.

    Returns:
        ``True`` if it ended within the timeout.
    """
    deadline = time.monotonic() + _PROCESS_EXIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _really_alive(pid):
            return True
        time.sleep(0.1)
    return False


def _still_serving(state: BridgeState) -> bool:
    """Report whether the recorded address accepts a TCP connection.

    The probe connects and immediately closes — whether the bridge still
    serves, not what it says.

    Args:
        state: The state whose host and port are probed.

    Returns:
        True when the connection was accepted.
    """
    import socket

    try:
        with socket.create_connection((state.host, state.port), timeout=2.0):
            return True
    except OSError:
        return False


def _assert_still_running(state: BridgeState, install: IsolatedKitty) -> None:
    """Assert the child process is alive and its address still accepts connections.

    Args:
        state: The bridge state to check.
        install: The installation, for the failure message's log.
    """
    output = _bridge_output(install)
    assert _really_alive(state.pid), f"the bridge (PID {state.pid}) is no longer alive:\n{output}"
    assert _still_serving(state), f"the bridge (PID {state.pid}) no longer accepts connections:\n{output}"


class TestAForeignBridgeIsNotStopped:
    """``stop`` refuses to signal a bridge it does not own."""

    def test_stop_leaves_a_foreign_bridge_running_and_its_state_in_place(
        self, isolated_kitty: IsolatedKitty, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """A foreign-owned, serving bridge survives ``stop_bridge`` untouched.

        The kernel answers a probe of another account's PID with EPERM —
        ``UNKNOWN``. Signalling that PID is unsafe: after recycling it would
        belong to an unrelated process. The product must print the error and
        exit non-zero without signalling, and must keep the state file (the
        only pointer at the running bridge).
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        proc = _spawn_bridge(isolated_kitty, state_file)
        state = _wait_for_state(state_file, proc, isolated_kitty)

        # The seam: the child's own PID, classified as another account's —
        # what the kernel reports for a bridge started by a different user.
        monkeypatch.setattr("kitty.bridge.manage.probe_pid", lambda pid: ProcessLiveness.UNKNOWN)

        from kitty.bridge.manage import stop_bridge

        with pytest.raises(SystemExit) as excinfo:
            stop_bridge(state_file)
        assert excinfo.value.code == 1, f"stop_bridge exited {excinfo.value.code}"
        assert "another user account" in capsys.readouterr().err, capsys.readouterr().err

        # Not stopped: the same process, still serving.
        _assert_still_running(state, isolated_kitty)
        # The state file was left in place: it is the only pointer at the
        # bridge, and deleting it would strand the process.
        after = load_state(state_file)
        assert after == state, f"the state file changed during stop: {after}"

    def test_status_reports_a_foreign_serving_bridge_as_unmanageable(
        self, isolated_kitty: IsolatedKitty, monkeypatch: pytest.MonkeyPatch
    ):
        """``bridge_status`` answers UNMANAGEABLE for a foreign, serving bridge.

        The distinction matters to the user: STALE would invite deleting a
        state file that still points at a running bridge.
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        proc = _spawn_bridge(isolated_kitty, state_file)
        _wait_for_state(state_file, proc, isolated_kitty)

        monkeypatch.setattr("kitty.bridge.manage.probe_pid", lambda pid: ProcessLiveness.UNKNOWN)

        from kitty.bridge.manage import bridge_status

        assert bridge_status(state_file) is BridgeStatus.UNMANAGEABLE


class TestNoSecondBridgeStartsBesideAForeignOne:
    """``start`` and ``restart`` refuse to spawn beside a foreign bridge."""

    def test_start_refuses_to_spawn_a_second_bridge_beside_a_foreign_one(
        self, isolated_kitty: IsolatedKitty, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """``start_bridge`` exits non-zero without spawning beside a foreign bridge.

        A second bridge would leave the first orphaned but holding its port.
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        proc = _spawn_bridge(isolated_kitty, state_file)
        state = _wait_for_state(state_file, proc, isolated_kitty)

        monkeypatch.setattr("kitty.bridge.manage.probe_pid", lambda pid: ProcessLiveness.UNKNOWN)

        from kitty.bridge.manage import start_bridge

        with pytest.raises(SystemExit) as excinfo:
            start_bridge(state_path=state_file, config_path=isolated_kitty.config_dir / "bridge.yaml")
        assert excinfo.value.code == 1, f"start_bridge exited {excinfo.value.code}"
        assert "another user account" in capsys.readouterr().err, capsys.readouterr().err

        # The first bridge is untouched, and the state still names it — no
        # second bridge registered itself.
        _assert_still_running(state, isolated_kitty)
        assert load_state(state_file) == state, "the state file changed during the refused start"

    def test_restart_aborts_in_its_stop_phase_beside_a_foreign_bridge(
        self, isolated_kitty: IsolatedKitty, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        """``restart_bridge`` stops at its stop phase; no second bridge starts.

        The restart's stop leg hits the same refusal as ``stop_bridge``, and
        the start leg must never run.
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        proc = _spawn_bridge(isolated_kitty, state_file)
        state = _wait_for_state(state_file, proc, isolated_kitty)

        monkeypatch.setattr("kitty.bridge.manage.probe_pid", lambda pid: ProcessLiveness.UNKNOWN)

        from kitty.bridge.manage import restart_bridge

        with pytest.raises(SystemExit):
            restart_bridge(state_path=state_file, config_path=isolated_kitty.config_dir / "bridge.yaml")
        assert "another user account" in capsys.readouterr().err, capsys.readouterr().err

        _assert_still_running(state, isolated_kitty)
        assert load_state(state_file) == state, "the state file changed during the aborted restart"


class TestTheOwnershipGuardIsReal:
    """The falsification control: a guard that cannot tell foreign from ours fails the harness."""

    def test_the_harness_detects_a_guard_that_reads_a_foreign_pid_as_ours(
        self, isolated_kitty: IsolatedKitty, monkeypatch: pytest.MonkeyPatch
    ):
        """With the PID read as ALIVE — the misclassification a broken guard makes — stop must proceed.

        Plan §1.4's rule: the first version of a harness ships with a
        deliberate defect it must detect. Collapsing ``UNKNOWN`` into
        ``ALIVE`` is that defect; the same setup that the refusals run on
        must, under it, actually stop the bridge.
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        proc = _spawn_bridge(isolated_kitty, state_file)
        state = _wait_for_state(state_file, proc, isolated_kitty)

        # The defect being falsified is the *ownership* collapse (UNKNOWN
        # treated as ours), not the death-detection one. Probe correctly
        # answers DEAD for a reaped process; the bug is the ALIVE answer it
        # gives for one that exists but is someone else's. That keeps the
        # stop_bridge wait loop short and the test honest.
        monkeypatch.setattr(
            "kitty.bridge.manage.probe_pid",
            lambda pid: ProcessLiveness.ALIVE if _really_alive(pid) else ProcessLiveness.DEAD,
        )

        from kitty.bridge.manage import stop_bridge

        stop_bridge(state_file)

        assert _wait_until_gone(state.pid), f"stop_bridge left the bridge (PID {state.pid}) running"
        assert not state_file.exists(), "stop_bridge left the state file behind"


class TestAnUnreachableForeignPidIsStale:
    """The complement: UNKNOWN plus an address nothing answers is stale, not foreign."""

    def _stale_foreign_state(self, isolated_kitty: IsolatedKitty, unused_tcp_port: int) -> BridgeState:
        """Write a state file whose address nothing listens on.

        Args:
            isolated_kitty: The isolated installation.
            unused_tcp_port: A port reserve-then-closed by the fixture — no
                listener, no TIME_WAIT, so ``connect()`` fails immediately on
                every platform.

        Returns:
            The state that was written.
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        state = BridgeState(
            pid=os.getpid(),
            host="127.0.0.1",
            port=unused_tcp_port,
            profile="ownership",
            started_at="2026-09-21T12:00:00Z",
            tls=False,
        )
        write_state(state_file, state)
        return state

    def test_stop_clears_the_state_of_an_unreachable_foreign_pid(
        self, isolated_kitty: IsolatedKitty, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
    ):
        """``stop_bridge`` removes a stale foreign record instead of exiting.

        A PID the caller may not signal, at an address nothing answers, is
        the PID-recycling case: the state file is merely stale, which is
        what the command exists to clear.
        """
        self._stale_foreign_state(isolated_kitty, unused_tcp_port)
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"

        monkeypatch.setattr("kitty.bridge.manage.probe_pid", lambda pid: ProcessLiveness.UNKNOWN)

        from kitty.bridge.manage import bridge_status, stop_bridge

        assert bridge_status(state_file) is BridgeStatus.STALE
        stop_bridge(state_file)
        assert not state_file.exists(), "stop_bridge left a stale state file behind"

    def test_start_proceeds_when_the_foreign_pid_is_unreachable(
        self,
        isolated_kitty: IsolatedKitty,
        unused_tcp_port: int,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """``start_bridge`` spawns a fresh bridge over a stale foreign record.

        The refusal exists for a bridge that is *serving*; an unreachable
        address means nothing holds the port, so the start must proceed.
        """
        stale = self._stale_foreign_state(isolated_kitty, unused_tcp_port)
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"

        monkeypatch.setattr("kitty.bridge.manage.probe_pid", lambda pid: ProcessLiveness.UNKNOWN)

        # The child inherits os.environ (start_bridge builds child_env from
        # it), so the isolated environment reaches the spawned interpreter.
        for name, value in isolated_kitty.env.items():
            monkeypatch.setenv(name, value)

        from kitty.bridge.manage import start_bridge

        start_bridge(state_path=state_file, config_path=isolated_kitty.config_dir / "bridge.yaml")

        fresh = _wait_for_state(state_file, None, isolated_kitty)  # type: ignore[arg-type]
        assert fresh.pid != stale.pid, "the start did not record a new bridge"


class TestABridgeSurvivesItsParent:
    """KBR-219: a child printing after its parent's pipe closed must not die."""

    def test_a_child_printing_after_its_parent_pipe_closed_still_comes_up(
        self, isolated_kitty: IsolatedKitty
    ):
        """A child whose parent's read end closed before its first stderr write still reaches ready.

        KBR-219's documented kill: ``kitty bridge start`` gives up after five
        seconds and exits, the pipe's read end closes, and the child's
        no-TLS warning — printed inside ``start_async`` *before* the state
        file is written — raises ``BrokenPipeError``. A healthy bridge dies
        silently. This test spawns the real runner with a non-loopback
        plaintext bind (the warning's trigger), closes the parent's read end
        immediately — the child still owes an interpreter boot, three orders
        of magnitude more time than the close takes — and requires the
        bridge to come up anyway.
        """
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        # stdout=PIPE: the exact shape `start_bridge` spawns. Closing the
        # read end below is the parent-has-exited condition for every later
        # child write.
        proc = _spawn_bridge(
            isolated_kitty,
            state_file,
            "--host",
            "0.0.0.0",
            stdout=subprocess.PIPE,
        )
        # The parent-exit simulation: close the read end at once. The child
        # has not even finished importing — measured in the KBR-219 ticket —
        # so nothing has been written yet.
        assert proc.stdout is not None
        proc.stdout.close()

        deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
        while not state_file.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)

        output = _bridge_output(install=isolated_kitty)
        assert proc.poll() is None, (
            f"the bridge child died before reporting ready (exit {proc.returncode}):\n{output}"
        )
        state = load_state(state_file)
        assert state is not None, f"the bridge recorded no state after its parent's pipe closed:\n{output}"
        assert "Traceback" not in output, f"the child died on a traceback:\n{output}"

    def test_a_pipe_spawned_childs_streams_point_to_devnull_after_ready(
        self, isolated_kitty: IsolatedKitty
    ):
        """After ready, a pipe-spawned child's fds 1 and 2 resolve to os.devnull.

        Behaviour cannot prove this: every post-ready stderr writer routes
        through ``logging`` or ``warnings``, both of which swallow
        ``OSError`` (CPython issue 5971), so the bridge keeps serving on
        unfixed code too. The fd state is what bites.
        """
        if not os.path.exists("/proc"):
            pytest.skip("the /proc fd oracle is Linux-only")
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        proc = _spawn_bridge(isolated_kitty, state_file, stdout=subprocess.PIPE)
        state = _wait_for_state(state_file, proc, isolated_kitty)

        for fd in (1, 2):
            target = os.readlink(f"/proc/{state.pid}/fd/{fd}")
            assert target == os.devnull, f"fd {fd} points at {target}, not {os.devnull}"

    def test_a_file_spawned_childs_streams_are_never_redirected(
        self, isolated_kitty: IsolatedKitty
    ):
        """A child whose stdout is a regular file keeps that file after ready.

        Service managers hand the bridge files, sockets or /dev/null — never
        a pipe. The redirect must be scoped to pipes, or it would swallow a
        deployment's log stream. Guard-regression shape: pre-fix this passes
        trivially (no redirect exists); it bites when someone widens the
        redirect beyond pipes.
        """
        if not os.path.exists("/proc"):
            pytest.skip("the /proc fd oracle is Linux-only")
        state_file = isolated_kitty.home / ".config" / "kitty" / "bridge_state.json"
        # --host 0.0.0.0 makes the child actually write something (the
        # no-TLS warning) to the file before ready.
        proc = _spawn_bridge(
            isolated_kitty,
            state_file,
            "--host",
            "0.0.0.0",
            stdout=None,
        )
        state = _wait_for_state(state_file, proc, isolated_kitty)

        # The child's stdout is the log file this harness opened for it,
        # and the redirect's pipes-only guard leaves it alone.
        for fd in (1, 2):
            target = os.readlink(f"/proc/{state.pid}/fd/{fd}")
            assert "bridge-" in target, f"fd {fd} was redirected away from its file to {target}"
            assert target != os.devnull, f"fd {fd} was redirected to os.devnull despite a non-pipe"
