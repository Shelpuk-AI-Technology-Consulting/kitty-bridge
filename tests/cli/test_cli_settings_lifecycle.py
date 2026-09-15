"""L3 subsystem tests for the settings lifecycle (KBR-93, plan task T-I1).

Each scenario spawns *real* child processes against a *real* — but fully
redirected — filesystem and proves the promise issue #22 bought: whatever way
a ``kitty claude`` session ends, the user's ``~/.claude/settings.json`` is
byte-identical to before, and the session-scoped settings file kitty wrote is
either cleaned up by kitty's own paths (``finally`` on normal exit,
``atexit`` on interpreter shutdown) or orphaned by SIGKILL for the user to
repair with ``kitty cleanup`` — which must repair the global file and never
touch anything it does not own.

Layer: L3 (subsystem — real processes, real filesystem). No CI job selects
``l3`` yet; ``tests/layers.py::PENDING_ACTIVATION_LAYERS`` parks the layer on
plan task T-K6, and the Fast gate's ``-m "l1 or l2"`` never runs these. A
developer's bare ``pytest`` does.

Hermeticity: the child environment redirects every path the product resolves
— ``HOME`` (settings.json, the hardcoded ``~/.config/kitty`` crash backup),
the platformdirs trees (profiles, credentials, egress, model-context cache)
and the temp dir the session file is written into by ``mkstemp``. The model-context
cache is primed fresh so ``launch_async``'s catalog refresh short-circuits
without a network call, and ``validate=False`` skips the pre-flight. The
harness asserts the primed cache is untouched after every scenario — the
no-outbound-HTTP oracle (any fetch would ``os.replace`` the file).

Falsification (plan §1.4): ``test_cleanup_does_not_fire_on_a_users_own_remote_proxy``
is the in-suite falsification case — a deliberately non-kitty state that a
cleanup without the ``_kitty_values_present`` heuristic would wrongly strip.
The stub sentinel falsifies binary shadowing: if the stub were not
executable, ``discover_binary`` would fall through to a real
``/usr/local/bin/claude`` and every launch scenario would fail on the
missing sentinel instead of silently testing the wrong binary.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# TEST_SUITE.md §6.3.2 — "CLI with real filesystem and processes". tests/cli/
# defaults to l1 by path, so the layer is named here explicitly.
pytestmark = pytest.mark.l3

# The interpreter the children run under — sys.executable verbatim. It must
# NOT be Path.resolve()d: the venv's python is a symlink into the base
# interpreter, and resolving it points the child at the base python, whose
# site-packages lack kitty (and every other dev dependency).
_VENV_PYTHON = sys.executable

# Poll cadence and ceiling for the readiness barriers (R7): the parent never
# sleeps a fixed interval before signalling a child; it waits for the files
# that prove the child reached the state the signal assumes.
_POLL_INTERVAL_SECONDS = 0.05
_READY_TIMEOUT_SECONDS = 30.0

# Subprocess ceiling. The primed cache keeps every scenario well under a
# second of work; a hang here is a product defect the timeout turns visible.
_CHILD_TIMEOUT_SECONDS = 120.0

# The pristine user-global content, as bytes: what the "user" had before any
# kitty session. Byte identity, not JSON semantics, is the assertion.
_PRISTINE_GLOBAL = b'{\n  "model": "opus",\n  "env": {"API_TIMEOUT_MS": "3000000"}\n}\n'

# Primed model-context cache body: an empty override map is valid catalog
# content (_body_is_valid accepts any JSON object) and short-circuits refresh.
_PRIMED_CACHE_BODY = b"{}"

# The child driver, written into the sandbox per test. It seeds the profile
# and credential stores through the real store classes, runs the real
# ``launch_async`` against the real ``ClaudeAdapter`` (binary discovered via
# the sandboxed PATH), records the session-file path for the parent's
# readiness barrier, and exits with launch_async's mapped code. In the
# ``probe_atexit`` mode it injects a failure at the seam between atexit
# registration (the ``_register_atexit_cleanup`` call in ``launch_async``)
# and the try block — the call site of ``build_child_env`` — so the
# registered atexit handler is the only cleanup that can run. The driver
# also configures a sandbox log file via logging.basicConfig so that the
# atexit path's ``atexit cleanup: restored`` INFO line is observable by the
# parent (the bridge's crash excepthook swallows the traceback itself —
# stderr stays empty — so the logger is the only oracle).
_CHILD_DRIVER_SOURCE = '''
"""Child driver for the L3 settings-lifecycle scenarios (sandbox-only)."""

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from kitty.providers.base import ProviderAdapter


class _StubProvider(ProviderAdapter):
    @property
    def provider_type(self):
        return "stub"

    @property
    def default_base_url(self):
        return "https://stub.example.com/v1"

    def build_request(self, model, messages, **kwargs):
        return {"model": model, "messages": messages}

    def parse_response(self, response_data):
        return response_data

    def map_error(self, status_code, body):
        return RuntimeError(f"upstream error {status_code}: {body}")


class _RecordingClaudeAdapter:
    """Records the session-file path returned by ``prepare_launch``.

    A thin wrapper that holds the real ``ClaudeAdapter`` instance and
    delegates every attribute access to it, with ``prepare_launch``
    overridden to persist the returned session-file path before handing it
    back. ``__getattr__``-style delegation is enough here because
    ``launch_async`` only reads attributes (it never ``isinstance``-checks the
    adapter — there is no class-identity check anywhere in the launch
    path). The persistence is best-effort: a write failure must not poison
    the cleanup path under test.
    """

    def __init__(self, record_path):
        from kitty.launchers.claude import ClaudeAdapter as _ClaudeAdapter

        self._record_path = record_path
        self._inner = _ClaudeAdapter()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def prepare_launch(self, env_overrides, settings_path=None):
        prepared = self._inner.prepare_launch(env_overrides, settings_path=settings_path)
        try:
            self._record_path.write_text(str(prepared), encoding="utf-8")
        except Exception:
            pass
        return prepared


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    record_path = Path(spec["record_session_path"])
    probe_atexit = bool(spec.get("probe_atexit", False))
    atexit_log_path = Path(spec["atexit_log"])

    # Route the kitty loggers to a sandbox file BEFORE any kitty import can
    # log. The bridge's crash excepthook writes its critical record through
    # the ``kitty.bridge`` logger (root handler), and the atexit path's
    # cleanup logs ``atexit cleanup: restored`` through
    # ``kitty.cli.launcher`` — both propagate to the root handler this
    # installs. force=True is required: kitty's imports may configure
    # handlers first, and without it basicConfig is a silent no-op.
    logging.basicConfig(
        filename=str(atexit_log_path),
        level=logging.INFO,
        force=True,
    )

    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialStore
    from kitty.profiles.schema import Profile
    from kitty.profiles.store import ProfileStore

    store = ProfileStore()
    cred_store = CredentialStore(backends=[FileBackend()])

    profile = Profile(
        name="l3-lifecycle",
        provider="zai_regular",
        model="stub-model",
        auth_ref=str(uuid.uuid4()),
        is_default=True,
    )
    store.save(profile)
    cred_store.set(profile.auth_ref, "sk-l3-stub-key")

    adapter = _RecordingClaudeAdapter(record_path=record_path)
    provider = _StubProvider()

    if probe_atexit:
        # Inject at the seam between the ``_register_atexit_cleanup`` call
        # in ``launch_async`` and its try block. The build_child_env call
        # is the seam: it runs after registration and before the try.
        import kitty.cli.launcher as launcher_mod

        def _raise(_cfg):
            raise RuntimeError(
                "atexit-path probe: failure between atexit registration and the try block"
            )

        launcher_mod.build_child_env = _raise

    from kitty.cli.launcher import launch_async

    code = asyncio.run(
        launch_async(
            adapter=adapter,
            provider=provider,
            profile=profile,
            cred_store=cred_store,
            validate=False,
        )
    )
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
'''


def _sandbox_env(tmp_root: Path, home: Path, bin_dir: Path) -> dict[str, str]:
    """Build the fully redirected environment a child process runs under.

    Every path the product resolves from the environment is pointed inside
    the sandbox, on every platform: ``HOME``/``USERPROFILE`` for the
    ``Path.home()``-derived paths, the XDG trees for platformdirs on Linux,
    ``APPDATA``/``LOCALAPPDATA`` for platformdirs on Windows, and all three
    temp vars because ``tempfile.gettempdir()`` consults ``TMPDIR`` then
    ``TEMP`` then ``TMP``.

    Args:
        tmp_root: Sandbox directory the session file and child logs go to.
        home: Sandbox home directory.
        bin_dir: Directory holding the stub ``claude`` binary; prepended to
            ``PATH`` so ``shutil.which`` finds it first.

    Returns:
        The child environment dictionary.
    """
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
            "APPDATA": str(home / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(home / "AppData" / "Local"),
            "TMPDIR": str(tmp_root),
            "TEMP": str(tmp_root),
            "TMP": str(tmp_root),
        }
    )
    # A developer shell may carry a kitty egress gateway; the sandbox has no
    # such config and must not inherit one.
    env.pop("KITTY_EGRESS_PROXY", None)
    # PYTHONPATH inherited from the developer's shell would let children
    # ``import kitty`` from a different checkout and run tests against a
    # different code tree without anyone noticing. Stripping it pins the
    # child to the venv's editable install.
    env.pop("PYTHONPATH", None)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    return env


def _platformdirs_paths_under(redirect: dict[str, str]) -> tuple[Path, Path]:
    """Resolve platformdirs' kitty config and cache dirs under ``redirect``.

    platformdirs reads the environment at call time, so the redirect is
    applied to this process's ``os.environ`` for the duration of the call and
    restored afterwards. Deriving the paths through platformdirs itself —
    rather than hardcoding ``~/.cache/kitty`` — keeps the prime correct on
    macOS (``~/Library/Caches/kitty``) and Windows (``%LOCALAPPDATA%``) too.

    Args:
        redirect: The environment overrides, as built by :func:`_sandbox_env`.

    Returns:
        The ``(cache_dir, config_dir)`` platformdirs would resolve for a
        child launched with that environment.
    """
    keys = list(redirect)
    saved = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(redirect)
        from platformdirs import user_cache_dir, user_config_dir

        return Path(user_cache_dir("kitty")), Path(user_config_dir("kitty"))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _Sandbox:
    """A redirected world one scenario runs in.

    Attributes:
        root: The sandbox root (pytest's ``tmp_path``).
        home: The child's ``$HOME``.
        global_settings: The user-global ``~/.claude/settings.json``.
        crash_backup: The hardcoded crash-recovery backup path
            (``~/.config/kitty/claude-settings-backup.json``).
        cache_file: The model-context override cache platformdirs resolves.
        env: The child environment.
        stub_sentinel: File the stub ``claude`` writes as its first action.
        session_record: File the child driver writes the session-file path to
            right after ``prepare_launch``.
        child_log: The launch child's combined stdout+stderr.
    """

    def __init__(self, root: Path) -> None:
        """Create the directory tree and the redirected environment.

        Args:
            root: The sandbox root directory.
        """
        self.root = root
        self.home = root / "home"
        self.bin_dir = root / "bin"
        self.tmp = root / "tmp"
        for directory in (self.home, self.bin_dir, self.tmp, self.home / ".claude"):
            directory.mkdir(parents=True, exist_ok=True)

        redirect = {
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "XDG_CACHE_HOME": str(self.home / ".cache"),
            "XDG_DATA_HOME": str(self.home / ".local" / "share"),
            "XDG_STATE_HOME": str(self.home / ".local" / "state"),
            "APPDATA": str(self.home / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(self.home / "AppData" / "Local"),
            "TMPDIR": str(self.tmp),
            "TEMP": str(self.tmp),
            "TMP": str(self.tmp),
        }
        self.env = _sandbox_env(self.tmp, self.home, self.bin_dir)
        self.global_settings = self.home / ".claude" / "settings.json"
        self.crash_backup = self.home / ".config" / "kitty" / "claude-settings-backup.json"
        self.cache_dir, self.config_dir = _platformdirs_paths_under(redirect)
        self.cache_file = self.cache_dir / "model_context_overrides.json"
        # The child driver routes every log record — including the atexit
        # path's lone ``atexit cleanup: restored`` INFO line — into this file
        # via logging.basicConfig(filename=...). AC-5 reads it as the
        # discriminator that proves the atexit handler ran, not the finally.
        self.atexit_log = self.root / "atexit.log"

        # The sandbox must be disjoint from the developer's real world: if
        # any of these equalities ever hold, the tests below would assert on
        # (and mutate) live user state. The config tree matters as much as
        # home and cache: ProfileStore and FileBackend default to
        # user_config_dir("kitty"), so a broken XDG_CONFIG_HOME redirect
        # would otherwise silently write profiles.json and a credential into
        # the developer's real config dir.
        assert self.home != Path.home(), "sandbox home must not be the real home"
        assert self.cache_dir != Path(user_cache_dir_probe()), "sandbox cache must not be the real cache"
        assert self.config_dir != Path(user_config_dir_probe()), (
            "sandbox config dir must not be the real config dir "
            "(ProfileStore and FileBackend would otherwise seed the developer's real profile store)"
        )

        self.stub_sentinel = self.root / "stub-ran.txt"
        self.session_record = self.root / "session-path.txt"
        self.child_log = self.root / "child.log"

    def write_global(self, body: bytes) -> None:
        """Write the user-global settings file with exact bytes.

        Args:
            body: The file content.
        """
        self.global_settings.write_bytes(body)

    def prime_cache(self) -> None:
        """Write the model-context cache fresh, so no refresh fetch happens.

        ``refresh_model_context_overrides`` calls ``_cache_is_fresh`` first,
        which is mtime-based and short-circuits when the cache file is
        younger than the TTL — the validation that the body is a real
        catalog only runs *inside* the fetch path, which the fresh mtime
        bypasses. The primed body itself (``{}``) is never validated; any
        existing bytes would do. The parent records the mtime for the
        no-fetch oracle in :meth:`assert_cache_untouched`.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(_PRIMED_CACHE_BODY)
        self._cache_mtime_before = self.cache_file.stat().st_mtime_ns

    def assert_cache_untouched(self) -> None:
        """Assert no child fetched the catalog during the scenario.

        Raises:
            AssertionError: When the primed cache changed bytes or mtime —
                the only way that happens is an outbound fetch (which
                ``os.replace``s the file).
        """
        assert self.cache_file.read_bytes() == _PRIMED_CACHE_BODY, "model-context cache content changed"
        assert self.cache_file.stat().st_mtime_ns == self._cache_mtime_before, (
            "model-context cache was rewritten during the scenario — a refresh fetch ran"
        )

    def assert_stub_ran(self) -> None:
        """Assert the stub ``claude`` binary executed.

        Raises:
            AssertionError: When the sentinel is missing. The sentinel is
                written by the stub itself, so its absence means the launch
                resolved to some other binary (or none).
        """
        if not self.stub_sentinel.exists():
            exists = self.child_log.exists()
            log = self.child_log.read_text(encoding="utf-8", errors="replace") if exists else "<none>"
            raise AssertionError(
                f"the stub claude binary never ran — discovery resolved elsewhere "
                f"(child log: {log})"
            )

    def write_stub(self, mode: str) -> None:
        """Write the stub ``claude`` binary onto the sandboxed PATH.

        POSIX gets an ``sh`` script; Windows a ``.cmd`` file (the extensions
        ``discover_binary``'s ``shutil.which`` consults via ``PATHEXT``).

        Args:
            mode: ``"exit0"`` — write the sentinel and exit 0 (normal-exit
                scenarios); ``"sleep"`` — write the sentinel and block until
                signalled, so the parent's signal has something to land on.
        """
        if sys.platform == "win32":
            stub_path = self.bin_dir / "claude.cmd"
            # ping with a count is the most portable blocking primitive on
            # Windows; both sleep and signal scenarios use it. SIGKILL /
            # SIGTERM via proc.kill() land as TerminateProcess, which kills
            # either flavour identically on Windows.
            sleep_line = (
                'ping -n 999 127.0.0.1 >nul\r\n'
                if mode == "sleep"
                else "exit /b 0\r\n"
            )
            script = (
                "@echo off\r\n"
                f'type nul > "{self.stub_sentinel}"\r\n'
                f"{sleep_line}"
            )
            stub_path.write_text(script, encoding="utf-8", newline="")
            return

        stub_path = self.bin_dir / "claude"
        script = f'#!/bin/sh\nprintf ran > "{self.stub_sentinel}"\n'
        if mode == "exit0":
            script += "exit 0\n"
        else:
            # exec keeps one process to signal; the shell's default SIGTERM
            # disposition is what the forwarded signal finally delivers.
            script += 'exec sleep 300\n'
        stub_path.write_text(script, encoding="utf-8", newline="\n")
        stub_path.chmod(0o755)

    def wait_for_ready(self, proc: subprocess.Popen[str], *, probe_atexit: bool = False) -> None:
        """Block until the child proves it is inside ``proc.wait()``.

        Two barriers (R7): the session-file path recorded by the driver
        (``prepare_launch`` done, atexit registered) and the stub sentinel
        (the child process exists, signal handlers installed). Signalling
        before either would race the very code under test. The atexit-probe
        mode raises before the subprocess spawn, so the stub sentinel is
        intentionally skipped there — only the prepare-launch barrier
        proves the registration point was reached.

        Args:
            proc: The launch child.
            probe_atexit: When True, drop the stub-sentinel barrier.

        Raises:
            AssertionError: When the remaining barriers do not appear inside
                the timeout; the child log is attached for diagnosis.
        """
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            ready = self.session_record.exists() and (
                self.stub_sentinel.exists() if not probe_atexit else True
            )
            if ready:
                return
            if proc.poll() is not None:
                break
            time.sleep(_POLL_INTERVAL_SECONDS)
        self._fail_with_log(proc, "the launch child never reached its readiness barriers")

    def read_session_path(self) -> Path:
        """Return the session-file path the driver recorded.

        Returns:
            The path ``prepare_launch`` returned inside the child.
        """
        return Path(self.session_record.read_text(encoding="utf-8").strip())

    def _fail_with_log(self, proc: subprocess.Popen[str], message: str) -> None:
        """Kill the child and raise with its captured output attached.

        Args:
            proc: The child to kill.
            message: The failure headline.

        Raises:
            AssertionError: Always — this helper exists to build one.
        """
        proc.kill()
        proc.wait()
        log = self.child_log.read_text(encoding="utf-8", errors="replace") if self.child_log.exists() else "<none>"
        raise AssertionError(f"{message}\n--- child log ---\n{log}")


def user_cache_dir_probe() -> str:
    """Return the real (unredirected) platformdirs cache dir for kitty.

    Exists only so :class:`_Sandbox` can assert the sandbox is disjoint from
    the developer's real cache. Deliberately a module function, not a method,
    so the call reads unambiguously as "the real environment's value".

    Returns:
        The real user cache directory for the ``kitty`` app.
    """
    from platformdirs import user_cache_dir

    return user_cache_dir("kitty")


def user_config_dir_probe() -> str:
    """Return the real (unredirected) platformdirs config dir for kitty.

    Sibling of :func:`user_cache_dir_probe`; covers ProfileStore and
    FileBackend which resolve under ``user_config_dir(\"kitty\")``.

    Returns:
        The real user config directory for the ``kitty`` app.
    """
    from platformdirs import user_config_dir

    return user_config_dir("kitty")


def _write_child_driver(sandbox: _Sandbox, *, probe_atexit: bool) -> Path:
    """Write the child driver script and its task spec into the sandbox.

    Args:
        sandbox: The sandbox to write into.
        probe_atexit: When True the driver injects a failure at the
            ``build_child_env`` call site so atexit is the only cleanup path.

    Returns:
        The driver script's path.
    """
    driver_path = sandbox.root / "child_driver.py"
    spec_path = sandbox.root / "child_spec.json"
    spec = {
        "record_session_path": str(sandbox.session_record),
        "atexit_log": str(sandbox.atexit_log),
        "probe_atexit": probe_atexit,
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    driver_path.write_text(_CHILD_DRIVER_SOURCE, encoding="utf-8")
    return driver_path


def _spawn_launch_child(sandbox: _Sandbox, *, probe_atexit: bool) -> subprocess.Popen[str]:
    """Spawn the launch child under the sandboxed environment.

    Args:
        sandbox: The sandbox providing env and paths.
        probe_atexit: Passed through to the driver spec.

    Returns:
        The running child process; its output streams into
        ``sandbox.child_log``.
    """
    driver_path = _write_child_driver(sandbox, probe_atexit=probe_atexit)
    # stderr merged into stdout so a single sink captures both — passing the
    # same file object as both stdout and stderr hits a Popen dup2 ordering
    # quirk where the second dup wins and the first stream ends up wherever
    # fd 1 started, which loses half the child's output.
    log_handle = sandbox.child_log.open("wb")
    return subprocess.Popen(  # noqa: S603 — fixed argv, sandboxed env
        [str(_VENV_PYTHON), str(driver_path), str(sandbox.root / "child_spec.json")],
        env=sandbox.env,
        cwd=str(sandbox.tmp),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )


def _run_cleanup_child(sandbox: _Sandbox) -> subprocess.CompletedProcess[str]:
    """Run the real ``python -m kitty cleanup`` under the sandboxed env.

    The full CLI entry runs: banner, parser, router (with the recovery
    exemption), egress resolution, and ``_run_cleanup`` — whose default
    paths resolve from the redirected ``HOME`` at import time. This is the
    product command a user would type, not a shortcut around it.

    Args:
        sandbox: The sandbox providing env and paths.

    Returns:
        The completed process.
    """
    return subprocess.run(  # noqa: S603 — fixed argv, sandboxed env
        [str(_VENV_PYTHON), "-m", "kitty", "cleanup"],
        env=sandbox.env,
        cwd=str(sandbox.tmp),
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_SECONDS,
        check=False,
    )


def _wait_for_exit(proc: subprocess.Popen[str], sandbox: _Sandbox) -> int:
    """Wait for the launch child to exit, failing with its log on hang.

    Args:
        proc: The launch child.
        sandbox: Its sandbox (for the log path).

    Returns:
        The exit code.

    Raises:
        AssertionError: When the child does not exit inside the timeout.
    """
    try:
        return proc.wait(timeout=_CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        sandbox._fail_with_log(proc, f"the launch child did not exit within {_CHILD_TIMEOUT_SECONDS}s")
        raise  # unreachable; _fail_with_log always raises


@pytest.fixture()
def sandbox(tmp_path: Path) -> _Sandbox:
    """Provide a fully redirected sandbox with a pristine user-global.

    Args:
        tmp_path: pytest's per-test temp directory.

    Returns:
        The ready sandbox: global settings written, model-context cache
        primed, stub ``claude`` (exit0 flavour) on PATH.
    """
    box = _Sandbox(tmp_path)
    box.write_global(_PRISTINE_GLOBAL)
    box.prime_cache()
    box.write_stub("exit0")
    return box


# ── AC-1: normal exit — finally block cleans up ─────────────────────────────


def test_normal_exit_removes_session_file_and_leaves_global_byte_identical(sandbox: _Sandbox) -> None:
    """After a clean child exit the global is untouched and the session file is gone.

    Args:
        sandbox: The redirected sandbox.
    """
    proc = _spawn_launch_child(sandbox, probe_atexit=False)
    sandbox.wait_for_ready(proc)
    exit_code = _wait_for_exit(proc, sandbox)

    assert exit_code == 0, sandbox.child_log.read_text(encoding="utf-8", errors="replace")
    sandbox.assert_stub_ran()
    sandbox.assert_cache_untouched()
    assert not sandbox.read_session_path().exists(), "kitty left the per-session settings file behind"
    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, "the user-global settings file was modified"


# ── AC-2: SIGTERM — forwarded to the child, finally block cleans up ─────────


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGTERM) is TerminateProcess on Windows — indistinguishable from SIGKILL",
)
def test_sigterm_is_forwarded_and_finally_restores_the_global(sandbox: _Sandbox) -> None:
    """SIGTERM to kitty is forwarded; the child dies; the finally path restores.

    Args:
        sandbox: The redirected sandbox.
    """
    sandbox.write_stub("sleep")
    proc = _spawn_launch_child(sandbox, probe_atexit=False)
    sandbox.wait_for_ready(proc)

    proc.send_signal(signal.SIGTERM)
    exit_code = _wait_for_exit(proc, sandbox)

    # launch_async maps the child's signal death to 128 + signal number:
    # the forwarded SIGTERM killed the stub, and kitty surfaced that.
    assert exit_code == 128 + int(signal.SIGTERM), sandbox.child_log.read_text(encoding="utf-8", errors="replace")
    sandbox.assert_stub_ran()
    sandbox.assert_cache_untouched()
    assert not sandbox.read_session_path().exists(), "the finally path left the session file behind"
    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, "the user-global settings file was modified"


# ── AC-3: SIGKILL — orphaned session file, cleanup is a no-op for the global ─


def test_sigkill_orphans_the_session_file_and_cleanup_is_a_noop(sandbox: _Sandbox) -> None:
    """SIGKILL skips every in-process cleanup; ``kitty cleanup`` repairs nothing.

    With the per-session design the SIGKILLed session orphaned its own file
    but never wrote the global — so the repair command must say "already
    clean", leave the global byte-identical, and never touch the orphan (it
    only owns ``~/.claude/settings.json`` and the crash backup).

    Args:
        sandbox: The redirected sandbox.
    """
    sandbox.write_stub("sleep")
    proc = _spawn_launch_child(sandbox, probe_atexit=False)
    sandbox.wait_for_ready(proc)

    # The readiness barriers guarantee the session file exists on disk before
    # the kill, so the orphan assertions cannot pass vacuously.
    orphan = sandbox.read_session_path()
    assert orphan.exists(), "readiness barrier passed but the session file is missing"

    proc.kill()
    _wait_for_exit(proc, sandbox)

    assert orphan.exists(), "SIGKILL did not orphan the session file — something cleaned up in-flight"

    cleaned = _run_cleanup_child(sandbox)
    joined = cleaned.stdout + cleaned.stderr
    assert cleaned.returncode == 0, joined
    assert "already clean" in joined, f"cleanup did not take the no-op branch: {joined}"
    assert orphan.exists(), "kitty cleanup deleted a session file it does not own"
    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, "the user-global settings file was modified"
    sandbox.assert_cache_untouched()


# ── AC-4: staged pre-fix-style kitty state — cleanup restores byte-exactly ───


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "_load_backup reads with universal newlines and _atomic_write_text writes with "
        "default newline=None, so a LF backup round-trips to CRLF on Windows — the "
        "byte-exact assertion needs a newline-aware comparison there"
    ),
)
def test_staged_kitty_state_is_restored_byte_exactly_from_backup(sandbox: _Sandbox) -> None:
    """A SIGKILLed pre-fix-style session is repaired exactly from the backup.

    The staged state is what a pre-#22 kitty left behind: the global carries
    the bridge values and the crash backup carries the true original.

    Args:
        sandbox: The redirected sandbox.
    """
    kitty_patched = (
        b'{\n  "env": {"ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",'
        b' "ANTHROPIC_BASE_URL": "http://127.0.0.1:45678"}\n}\n'
    )
    original = b'{\n  "model": "sonnet",\n  "env": {"API_TIMEOUT_MS": "999"}\n}\n'
    sandbox.write_global(kitty_patched)
    sandbox.crash_backup.parent.mkdir(parents=True, exist_ok=True)
    sandbox.crash_backup.write_bytes(original)

    cleaned = _run_cleanup_child(sandbox)
    joined = cleaned.stdout + cleaned.stderr

    assert cleaned.returncode == 0, joined
    assert "Restored" in joined, f"cleanup did not take the exact-restore branch: {joined}"
    assert sandbox.global_settings.read_bytes() == original, "the global was not restored byte-exactly"
    assert not sandbox.crash_backup.exists(), "cleanup left the backup behind after restoring"
    sandbox.assert_cache_untouched()


# ── AC-5: atexit — the only cleanup when the try block is never entered ──────


def test_atexit_path_cleans_up_when_the_try_block_is_never_entered(sandbox: _Sandbox) -> None:
    """A failure between atexit registration and the try block still cleans up.

    ``launch_async`` calls ``_register_atexit_cleanup`` and then enters its
    ``try`` block; the ``build_child_env`` call sits in between. The probe
    makes that call raise, so the interpreter unwinds through an unhandled
    exception and the registered atexit handler is the only cleanup that
    can run. The atexit discriminator reads the ``atexit cleanup: restored``
    INFO line the handler emits — observable because the child driver
    routes the kitty loggers to a sandbox file before any library import
    can swallow them.

    Args:
        sandbox: The redirected sandbox.
    """
    proc = _spawn_launch_child(sandbox, probe_atexit=True)
    sandbox.wait_for_ready(proc, probe_atexit=True)
    exit_code = _wait_for_exit(proc, sandbox)

    assert exit_code != 0, "the probe must fail the launch for atexit to be the only cleanup path"
    # Primary discriminator: the atexit handler is the *only* code path that
    # emits ``atexit cleanup: restored``. If a refactor moved build_child_env
    # inside the try block, the except branch would run cleanup_launch
    # directly and _clear_atexit_cleanup would empty the state before atexit
    # fired — the marker would not appear, the assertion would fail, and
    # the test would stop vacuously green.
    atexit_log_text = ""
    if sandbox.atexit_log.exists():
        atexit_log_text = sandbox.atexit_log.read_text(encoding="utf-8", errors="replace")
    assert "atexit cleanup: restored" in atexit_log_text, (
        f"the atexit path did not run — cleanup_launch was reached only via "
        f"the finally branch. atexit's log was:\n{atexit_log_text}"
    )
    # Secondary discriminator: the except branch prints "Failed to launch
    # '...'" to stderr when it catches an exception during spawn. The
    # genuine atexit path's traceback goes through the logger (caught
    # above) and not stderr, so stderr stays clean of this string.
    assert "Failed to launch" not in sandbox.child_log.read_text(
        encoding="utf-8", errors="replace"
    ), "the atexit path did not run — the build_child_env probe likely fired inside the try block"
    # The stub never spawns in this mode — build_child_env raises before
    # subprocess.start — so the stub-sentinel oracle is intentionally
    # skipped here.
    sandbox.assert_cache_untouched()
    assert not sandbox.read_session_path().exists(), "the atexit path left the session file behind"
    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, "the user-global settings file was modified"


# ── AC-6: negative control — the heuristic must not fire on non-kitty state ──


def test_cleanup_does_not_fire_on_a_users_own_remote_proxy(sandbox: _Sandbox) -> None:
    """A user's own remote proxy in the global survives ``kitty cleanup``.

    The control state is deliberately *not* kitty's: a non-loopback base URL
    (a remote proxy is someone's real setup) and no kitty auth token. A
    loopback URL here would be a false control — ``_kitty_values_present``
    treats every loopback URL as kitty-written by design, so the test would
    exercise the restore branch instead of the guard. The stdout pin ("Removed
    stale backup") catches a future broadening of the heuristic that flips
    the branch while the file assertions still hold.

    Args:
        sandbox: The redirected sandbox.
    """
    users_own = (
        b'{\n  "env": {"ANTHROPIC_BASE_URL": "https://api.myproxy.example.com",'
        b' "API_TIMEOUT_MS": "3000000"}\n}\n'
    )
    backup_content = b'{\n  "model": "haiku"\n}\n'
    sandbox.write_global(users_own)
    sandbox.crash_backup.parent.mkdir(parents=True, exist_ok=True)
    sandbox.crash_backup.write_bytes(backup_content)

    cleaned = _run_cleanup_child(sandbox)
    joined = cleaned.stdout + cleaned.stderr

    assert cleaned.returncode == 0, joined
    assert "Removed stale backup" in joined, f"cleanup did not take the stale-backup branch: {joined}"
    assert sandbox.global_settings.read_bytes() == users_own, "kitty cleanup stripped the user's own proxy values"
    assert not sandbox.crash_backup.exists(), "the stale backup was not removed"
    sandbox.assert_cache_untouched()
