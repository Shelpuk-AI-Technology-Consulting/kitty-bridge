"""End-to-end tests that ``kitty bridge`` and the bridge it starts agree on one state file.

KBR-220: ``kitty bridge start|stop|restart|status`` looked for ``bridge_state.json``
in ``platformdirs.user_config_dir("kitty")`` while the bridge wrote it under
``~/.config/kitty``. The two coincide only on Linux without ``XDG_CONFIG_HOME``,
so on macOS, Windows and XDG Linux every healthy start was reported as a failure,
``status`` never saw a bridge, and ``stop`` could not stop one.

These tests run the real commands and the real bridge, because the defect lives
*between* two processes. Each side read correctly by its own logic, and every
in-process test handed both sides the same path.

KBR-230 adds the background bridge's keys-file auth policy to the same harness:
the fresh-install shape (no keys file anywhere), a named-but-missing file, and
the ``bridge config`` display.

**Isolation.** The children get a temporary home (``HOME``/``USERPROFILE``) and a
temporary ``platformdirs`` config and cache directory: ``XDG_CONFIG_HOME`` and
``XDG_CACHE_HOME`` on Linux and macOS, ``WIN_PD_OVERRIDE_LOCAL_APPDATA`` on Windows.
Plain ``LOCALAPPDATA`` does **not** redirect Windows, because platformdirs asks
``SHGetKnownFolderPath``, which ignores it. The overrides need platformdirs 4.8
(4.6 for XDG on macOS), and ``pyproject.toml`` allows 4.0. So before anything is
written, a child reports where it resolves, and the fixture refuses to go on
unless that is inside the temporary directory. An old platformdirs then fails
loudly instead of editing the developer's real profiles.

The config directory is also deliberately *not* ``home/.config/kitty``. On stock
Linux the two coincide, and there the regression test would pass on unfixed code.

**No network inside the start window.** A real bridge refreshes the model-context
catalog before it reports ready, with a 10 s fetch timeout, while ``kitty bridge
start`` waits 5 s. The catalog cache is seeded fresh in the isolated cache
directory, so no fetch runs and the user's own cache is never written.

**Layer.** ``l1`` by path default although it spawns processes:
``TEST_SUITE.md`` §8.2 forbids ``l3`` before the Subsystem job exists, and lists
this module among the ``l1`` modules that start real processes.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kitty.bridge.manage import ProcessLiveness, probe_pid
from kitty.bridge.state import load_state

ROOT = Path(__file__).resolve().parent.parent.parent

# Any well-formed UUID: profiles reference their credential by one.
_AUTH_REF = "0b6f3a52-7f2e-4f3b-9d6e-2a1c5e8b4d10"

# Generous against a cold Windows runner, and only ever reached on failure.
_COMMAND_TIMEOUT_SECONDS = 60
_PROCESS_EXIT_TIMEOUT_SECONDS = 20

# Both run inside the isolated environment: the first reports where kitty will look,
# the second stores a profile, a credential, bridge.yaml and a fresh catalog cache there.
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

profile = Profile(name="e2e", provider="zai_coding", model="glm-4.6", auth_ref="{_AUTH_REF}", is_default=True)
ProfileStore().save(profile)
FileBackend().set("{_AUTH_REF}", "sk-not-a-real-key")

cache = model_context.REMOTE_OVERRIDES_CACHE_PATH
cache.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path(model_context.__file__).with_name("model_context_overrides.json"), cache)

config_dir = Path(platformdirs.user_config_dir("kitty"))
(config_dir / "bridge.yaml").write_text("host: 127.0.0.1\\nport: 0\\n", encoding="utf-8")
'''


@dataclass
class KittyInstall:
    """An isolated kitty installation that real child processes run against.

    Attributes:
        root: Temporary directory holding everything below.
        home: The children's home directory.
        config_dir: The children's ``user_config_dir("kitty")``, not under ``home``.
        env: Environment for every child process.
        spawned: Processes a test started directly, stopped at teardown.
        outputs: Everything each ``kitty`` command printed, read at teardown.
    """

    root: Path
    home: Path
    config_dir: Path
    env: dict[str, str]
    spawned: list[subprocess.Popen] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def kitty(self, *args: str) -> tuple[int, str]:
        """Run ``python -m kitty`` with ``args`` and return its exit code and output.

        Output goes to a file, not a pipe. A bridge that outlives the command
        might otherwise hold the pipe open and make this call wait for it.

        Args:
            *args: Command-line arguments for ``kitty``.

        Returns:
            The exit code, and stdout and stderr combined.
        """
        log = self.root / "command.log"
        with log.open("wb") as out:
            proc = subprocess.run(
                [sys.executable, "-m", "kitty", *args],
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
        output = log.read_text(encoding="utf-8", errors="replace")
        self.outputs.append(output)
        return proc.returncode, output


def _wait_until_gone(pid: int) -> bool:
    """Wait for a process to stop being alive.

    Args:
        pid: The process to watch.

    Returns:
        ``True`` if it ended within the timeout.
    """
    deadline = time.monotonic() + _PROCESS_EXIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if probe_pid(pid) is not ProcessLiveness.ALIVE:
            return True
        time.sleep(0.1)
    return False


def _isolated_install(tmp_path: Path, *, with_keys: bool) -> Iterator[KittyInstall]:
    """Build the isolated kitty the fixtures hand out, with or without a keys file.

    Args:
        tmp_path: pytest's per-test temporary directory.
        with_keys: Write the default keys file. The fresh-install tests (KBR-230)
            need the install *without* it: nothing in kitty creates that file.

    Yields:
        The isolated installation.
    """
    home = tmp_path / "home"
    (home / ".config" / "kitty").mkdir(parents=True)
    if with_keys:
        # The default keys file: until KBR-230 a background bridge required it.
        (home / ".config" / "kitty" / "bridge_keys.txt").write_text("e2e-client-key\n", encoding="utf-8")

    # KITTY_* settings in the developer's environment (a gateway, a session summary path) must not steer the bridge.
    env = {name: value for name, value in os.environ.items() if not name.startswith("KITTY_")}
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        WIN_PD_OVERRIDE_LOCAL_APPDATA=str(tmp_path / "localappdata"),
        PYTHONIOENCODING="utf-8",
        # The working tree first: from a git worktree an editable install can resolve another checkout.
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

    # Refuse before writing anything: an unredirected platformdirs would mean the developer's real profiles.
    where = json.loads(_run(_WHERE))
    config_dir, cache_dir = Path(where["config_dir"]), Path(where["cache_dir"])
    if not (config_dir.is_relative_to(tmp_path) and cache_dir.is_relative_to(tmp_path) and Path(where["home"]) == home):
        pytest.fail(f"kitty's directories were not isolated (platformdirs older than 4.8?): {where}")
    # Negative control: with the config dir under the home, unfixed code passes on stock Linux.
    assert config_dir != home / ".config" / "kitty", where

    _run(_INSTALL)

    install = KittyInstall(root=tmp_path, home=home, config_dir=config_dir, env=env)
    try:
        yield install
    finally:
        # A red test must not leave a bridge serving: stop anything recorded anywhere under the root.
        for proc in install.spawned:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
        pids = {state.pid for f in tmp_path.rglob("bridge_state.json") if (state := load_state(f)) is not None}
        # "did not report ready ... still running (PID n)" leaves a bridge with no state file to find it by.
        pids.update(
            int(pid) for output in install.outputs for pid in re.findall(r"still running \(PID (\d+)\)", output)
        )
        for pid in pids:
            if probe_pid(pid) is ProcessLiveness.ALIVE:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                _wait_until_gone(pid)


@pytest.fixture
def kitty_install(tmp_path: Path) -> Iterator[KittyInstall]:
    """Build an isolated kitty with one profile and the default keys file, and stop every bridge it ran at teardown.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        The isolated installation.
    """
    yield from _isolated_install(tmp_path, with_keys=True)


@pytest.fixture
def kitty_install_without_keys(tmp_path: Path) -> Iterator[KittyInstall]:
    """Build an isolated kitty with no keys file anywhere: the fresh-install shape (KBR-230).

    Args:
        tmp_path: pytest's per-test temporary directory.

    Yields:
        The isolated installation.
    """
    yield from _isolated_install(tmp_path, with_keys=False)


class TestKittyBridgeFindsTheBridgeItStarted:
    """``start``, ``status`` and ``stop`` all reach the bridge ``start`` launched."""

    def test_start_status_restart_and_stop_all_manage_the_one_running_bridge(self, kitty_install: KittyInstall):
        """A real ``kitty bridge start`` is seen by ``status``, replaced by ``restart`` and ended by ``stop``.

        Before KBR-220 ``start`` waited out its window on a file the bridge
        never wrote and exited 1. ``status`` then said "not running", ``stop``
        did nothing, and ``restart`` started a second bridge beside the first.
        """
        state_file = kitty_install.home / ".config" / "kitty" / "bridge_state.json"

        code, output = kitty_install.kitty("bridge", "start")
        assert code == 0, f"kitty bridge start failed:\n{output}"
        assert "http://127.0.0.1:" in output, output

        code, output = kitty_install.kitty("bridge", "status")
        assert code == 0, f"kitty bridge status did not see the bridge:\n{output}"
        assert "Running:" in output, output
        first = load_state(state_file)
        assert first is not None, "the bridge recorded no state where status reads it"

        code, output = kitty_install.kitty("bridge", "restart")
        assert code == 0, f"kitty bridge restart failed:\n{output}"
        assert _wait_until_gone(first.pid), f"kitty bridge restart left the first bridge (PID {first.pid}) running"
        second = load_state(state_file)
        # The whole record, not the PID alone: a PID can be reused once its process is gone.
        assert second is not None and second != first, f"restart recorded no new bridge: {second}"

        code, output = kitty_install.kitty("bridge", "stop")
        assert code == 0, f"kitty bridge stop failed:\n{output}"
        assert _wait_until_gone(second.pid), f"kitty bridge stop left the bridge (PID {second.pid}) running"

        code, output = kitty_install.kitty("bridge", "status")
        assert (code, "Bridge is not running." in output) == (1, True), output


def _spawn_bridge(install: KittyInstall, *extra: str) -> subprocess.Popen:
    """Start ``python -m kitty.bridge_runner`` directly, as a service would.

    The process is reaped the moment it exits, as a service manager would do. An
    unreaped child still answers a liveness probe, so ``stop`` would wait out
    its full 10 s for it.

    Args:
        install: The isolated installation to run against.
        *extra: Arguments after ``--config``.

    Returns:
        The started process, also registered for teardown.
    """
    with (install.root / "bridge.log").open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "kitty.bridge_runner", "--config", str(install.config_dir / "bridge.yaml"), *extra],
            env=install.env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    install.spawned.append(proc)
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def _wait_for_state(install: KittyInstall, proc: subprocess.Popen, state_file: Path) -> None:
    """Wait until the bridge records its state, and fail with its output if it never does.

    Args:
        install: The installation the bridge runs against.
        proc: The bridge process.
        state_file: Where its state is expected.
    """
    deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
    while not state_file.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    output = (install.root / "bridge.log").read_text(encoding="utf-8", errors="replace")
    assert state_file.exists(), f"the bridge recorded no state at {state_file}:\n{output}"


class TestTheBridgeRecordsItsStateWhereItIsTold:
    """``--state-file`` decides where the bridge writes, which is what ``start`` polls."""

    def test_a_bridge_given_a_state_file_writes_there_and_not_at_the_default(self, kitty_install: KittyInstall):
        """The bridge honours the path ``kitty bridge start`` hands it.

        ``start_bridge`` passing the flag proves nothing if the bridge ignores
        it. With the CLI and the default agreeing, only a path that differs from
        the default can show that.
        """
        told = kitty_install.root / "elsewhere" / "bridge_state.json"
        proc = _spawn_bridge(kitty_install, "--state-file", str(told))

        _wait_for_state(kitty_install, proc, told)
        assert not (kitty_install.home / ".config" / "kitty" / "bridge_state.json").exists()


class TestKittyBridgeFindsAServiceStartedBridge:
    """A bridge started as a system service would be is managed by the same commands."""

    def test_status_and_stop_reach_a_bridge_started_without_kitty_bridge_start(self, kitty_install: KittyInstall):
        """``status`` and ``stop`` reach a bridge that was never told a state path.

        systemd, launchd and the Windows service script run the bridge with only
        ``--config``, so it records its state at the default. That default must
        be the file ``kitty bridge status`` reads.
        """
        proc = _spawn_bridge(kitty_install)
        _wait_for_state(kitty_install, proc, kitty_install.home / ".config" / "kitty" / "bridge_state.json")

        code, output = kitty_install.kitty("bridge", "status")
        assert code == 0, f"kitty bridge status did not see the service-started bridge:\n{output}"
        assert "Running:" in output, output

        code, output = kitty_install.kitty("bridge", "stop")
        assert code == 0, f"kitty bridge stop failed:\n{output}"
        assert _wait_until_gone(proc.pid), f"kitty bridge stop left the bridge (PID {proc.pid}) running"
        if os.name != "nt":
            # SIGTERM reached the stop handler and the bridge shut down cleanly. On Windows
            # `stop` ends the process outright, so there is no handler to have run.
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            assert proc.returncode == 0, f"the bridge did not shut down cleanly: exit {proc.returncode}"


class TestABridgeWithNoKeysFileStartsWithAuthOff:
    """The fresh-install case (KBR-230): nothing creates a keys file, so auth is off, not a crash."""

    def test_start_serves_healthz_without_credentials_and_stop_ends_it(
        self, kitty_install_without_keys: KittyInstall
    ):
        """``kitty bridge start`` succeeds with no keys file and ``/healthz`` answers without credentials.

        Before KBR-230 the start died in ``parse_keys_file`` with a
        ``FileNotFoundError`` traceback: the config default named a file nothing
        creates.
        """
        install = kitty_install_without_keys
        state_file = install.home / ".config" / "kitty" / "bridge_state.json"

        code, output = install.kitty("bridge", "start")
        assert code == 0, f"kitty bridge start failed without a keys file:\n{output}"
        assert "http://127.0.0.1:" in output, output
        assert "Traceback" not in output, output
        state = load_state(state_file)
        assert state is not None, f"the bridge recorded no state:\n{output}"

        # Auth off is a claim about the wire, not the exit code: the middleware
        # would 401 this request if any keys file had been loaded (the control
        # for that is TestAKeysFilePresentKeepsAuthOn).
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{state.host}:{state.port}/healthz", timeout=10) as response:
            assert response.status == 200, f"/healthz answered {response.status}"

        code, output = install.kitty("bridge", "stop")
        assert code == 0, f"kitty bridge stop failed:\n{output}"

    def test_an_explicitly_named_missing_keys_file_fails_with_a_clear_error(
        self, kitty_install_without_keys: KittyInstall
    ):
        """A ``keys_file:`` naming a missing file stops the start with one clear line, not a traceback.

        An explicit configuration is never silently ignored (KBR-230 AC-2).
        """
        install = kitty_install_without_keys
        named = install.root / "absent" / "keys.txt"
        (install.config_dir / "bridge.yaml").write_text(
            f"host: 127.0.0.1\nport: 0\nkeys_file: {named}\n", encoding="utf-8"
        )

        code, output = install.kitty("bridge", "start")
        assert code == 1, f"the start ignored a named missing keys file:\n{output}"
        assert f"Keys file not found: {named}" in output, output
        assert "Traceback" not in output, output


class TestAKeysFilePresentKeepsAuthOn:
    """The control for the auth-off test: with the default keys file present, the bridge 401s."""

    def test_healthz_rejects_a_request_without_credentials(self, kitty_install: KittyInstall):
        """``/healthz`` without a Bearer key returns 401 while a keys file is in play.

        Proves the auth-off test's 200 means the middleware is disabled, not
        that `/healthz` is open by design.
        """
        install = kitty_install
        state_file = install.home / ".config" / "kitty" / "bridge_state.json"

        code, output = install.kitty("bridge", "start")
        assert code == 0, f"kitty bridge start failed:\n{output}"
        state = load_state(state_file)
        assert state is not None, f"the bridge recorded no state:\n{output}"

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        rejected: urllib.error.HTTPError | None = None
        try:
            opener.open(f"http://{state.host}:{state.port}/healthz", timeout=10).read()
        except urllib.error.HTTPError as exc:
            rejected = exc
        assert rejected is not None and rejected.code == 401, (
            f"the bridge answered an unauthenticated /healthz with a keys file in play:\n{output}"
        )

        code, output = install.kitty("bridge", "stop")
        assert code == 0, f"kitty bridge stop failed:\n{output}"


class TestBridgeConfigReportsTheEffectiveKeysFile:
    """``kitty bridge config`` shows the auth a start would really apply (KBR-230)."""

    def test_a_fresh_install_reports_auth_disabled(self, kitty_install_without_keys: KittyInstall):
        """With nothing named and no default file, the display says auth is off."""
        code, output = kitty_install_without_keys.kitty("bridge", "config")
        assert code == 0, f"kitty bridge config failed:\n{output}"
        assert "Keys file: (none — auth disabled)" in output, output

    def test_a_default_keys_file_is_reported_by_path(self, kitty_install: KittyInstall):
        """With the default keys file present, the display shows its path."""
        install = kitty_install
        default = install.home / ".config" / "kitty" / "bridge_keys.txt"

        code, output = install.kitty("bridge", "config")
        assert code == 0, f"kitty bridge config failed:\n{output}"
        assert f"Keys file: {default}" in output, output

    def test_a_named_missing_keys_file_is_flagged_in_the_display(self, kitty_install_without_keys: KittyInstall):
        """A named-but-missing keys file does not display like a working one (KBR-230).

        §1.6's row 2 and row 1 must be distinguishable on screen, or the user
        reads auth-on from a `bridge start` that will be refused.
        """
        install = kitty_install_without_keys
        named = install.root / "absent" / "keys.txt"
        (install.config_dir / "bridge.yaml").write_text(
            f"host: 127.0.0.1\nport: 0\nkeys_file: {named}\n", encoding="utf-8"
        )

        code, output = install.kitty("bridge", "config")
        assert code == 0, f"kitty bridge config failed:\n{output}"
        assert f"Keys file: {named} (not found" in output, output
