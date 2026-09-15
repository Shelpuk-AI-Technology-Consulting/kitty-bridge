"""L3 subsystem tests for concurrent ``kitty claude`` sessions (KBR-94, T-I2).

Two *real* launch children — each running the real ``launch_async`` against
the real ``ClaudeAdapter``, its own bridge server and a stub ``claude``
binary — run side by side in a fully redirected sandbox, and the scenario
proves the promise issue #22 bought the per-session design for: each session
gets its own ``--settings`` temp file, neither session touches the user's
``~/.claude/settings.json``, and the second session's start does not disturb
the first.

The unit-level counterparts (``tests/test_claude_settings_multi_session.py``)
prove the same invariants on the adapter in-process; this file proves them at
the process boundary, where two bridge servers, two child processes and two
``launch_async`` lifecycles actually coexist. The lifecycle counterpart
(``tests/cli/test_cli_settings_lifecycle.py``, KBR-93, merged in PR #162)
proves one session at a time; concurrency is the row it does not cover.

This file is self-contained — no import from the sibling L3 file — so the
two harnesses evolve independently. That trades ~250 lines of duplicated
sandbox / driver / barrier code for two independently mergeable files; a
future change to one copy must be hand-mirrored to the other until a
consolidation task lifts the shared harness into ``tests/harness/``.

Layer: L3 (subsystem — real processes, real filesystem). No CI job selects
``l3`` yet; ``tests/layers.py::PENDING_ACTIVATION_LAYERS`` parks the layer on
plan task T-K6, and the Fast gate's ``-m "l1 or l2"`` never runs these. A
developer's bare ``pytest`` does. The three scenario tests are POSIX-only:
ending a session is signalled with SIGTERM, which ``subprocess.Popen``
delivers as TerminateProcess on Windows — indistinguishable from SIGKILL, so
the cleanup assertions cannot pass there (the falsification control is
pure-Python and runs everywhere).

Hermeticity: the child environment redirects every path the product resolves
— ``HOME`` (settings.json, the crash backup), the platformdirs trees, and the
temp dir ``mkstemp`` writes the session file into. The model-context cache is
primed fresh so ``launch_async``'s catalog refresh short-circuits without a
network call, and ``validate=False`` skips the pre-flight. The harness
asserts the primed cache is untouched after every scenario — the
no-outbound-HTTP oracle (any fetch would ``os.replace`` the file).

Falsification (plan §1.4): ``test_shared_settings_path_clobber_is_visible_to_the_probe``
is the in-suite falsification case — it reproduces the pre-#22 failure shape
(both sessions' routing written onto one file) and proves the same
``env.ANTHROPIC_BASE_URL`` read the concurrent assertions perform actually
reports the clobber. A probe that could not distinguish a clobber from
isolation would let the concurrent tests pass vacuously.
"""

from __future__ import annotations

import contextlib
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

# Poll cadence and ceiling for the readiness barriers: the parent never
# sleeps a fixed interval before asserting on a child; it waits for the
# files that prove the child reached the state the assertion assumes.
_POLL_INTERVAL_SECONDS = 0.05
_READY_TIMEOUT_SECONDS = 30.0

# Subprocess ceiling. The primed cache keeps every scenario well under a
# second of work; a hang here is a product defect the timeout turns visible.
_CHILD_TIMEOUT_SECONDS = 120.0

# Teardown ceiling per child. Teardown is supposed to be near-instant
# (SIGTERM-forward-and-exit runs in well under a second on a healthy child);
# 30 s is generous headroom for a loaded CI runner while bounding the worst
# case if a hang happens to be on the teardown path.
_TEARDOWN_TIMEOUT_SECONDS = 30.0

# The pristine user-global content, as bytes: what the "user" had before any
# kitty session. Byte identity, not JSON semantics, is the assertion.
_PRISTINE_GLOBAL = b'{\n  "model": "opus",\n  "env": {"API_TIMEOUT_MS": "3000000"}\n}\n'

# Primed model-context cache body: an empty override map is valid catalog
# content (_body_is_valid accepts any JSON object) and short-circuits refresh.
_PRIMED_CACHE_BODY = b"{}"


def _redirect_paths(tmp_root: Path, home: Path) -> dict[str, str]:
    """Build the path-redirect mapping every sandboxed process shares.

    The single source of the 11 path overrides: ``_sandbox_env`` applies it
    to the child environment and :class:`_Sandbox.__init__` feeds it to
    ``_platformdirs_paths_under`` so the primed cache resolves to the same
    file the children see. One mapping, two consumers — a new redirect key
    is added here once, never twice.

    Args:
        tmp_root: Sandbox directory the session files and child logs go to.
        home: Sandbox home directory.

    Returns:
        The redirect mapping, from environment variable to sandbox path.
    """
    return {
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


def _sandbox_env(tmp_root: Path, home: Path, bin_dir: Path) -> dict[str, str]:
    """Build the fully redirected environment a child process runs under.

    Every path the product resolves from the environment is pointed inside
    the sandbox, on every platform: ``HOME``/``USERPROFILE`` for the
    ``Path.home()``-derived paths, the XDG trees for platformdirs on Linux,
    ``APPDATA``/``LOCALAPPDATA`` for platformdirs on Windows, and all three
    temp vars because ``tempfile.gettempdir()`` consults ``TMPDIR`` then
    ``TEMP`` then ``TMP``.

    Args:
        tmp_root: Sandbox directory the session files and child logs go to.
        home: Sandbox home directory.
        bin_dir: Directory holding the stub ``claude`` binary; prepended to
            ``PATH`` so ``shutil.which`` finds it first.

    Returns:
        The child environment dictionary.
    """
    env = os.environ.copy()
    env.update(_redirect_paths(tmp_root, home))
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


def user_cache_dir_probe() -> str:
    """Return the real (unredirected) platformdirs cache dir for kitty.

    Exists only so the sandbox can assert it is disjoint from the
    developer's real cache. Deliberately a module function, not a method,
    so the call reads unambiguously as "the real environment's value".

    Returns:
        The real user cache directory for the ``kitty`` app.
    """
    from platformdirs import user_cache_dir

    return user_cache_dir("kitty")


def user_config_dir_probe() -> str:
    """Return the real (unredirected) platformdirs config dir for kitty.

    Sibling of :func:`user_cache_dir_probe`; covers ProfileStore and
    FileBackend which resolve under ``user_config_dir("kitty")``.

    Returns:
        The real user config directory for the ``kitty`` app.
    """
    from platformdirs import user_config_dir

    return user_config_dir("kitty")


class _Sandbox:
    """A redirected world the two-session scenario runs in.

    Attributes:
        root: The sandbox root (pytest's ``tmp_path``).
        home: The children's ``$HOME``.
        global_settings: The user-global ``~/.claude/settings.json``.
        cache_file: The model-context override cache platformdirs resolves.
        env: The child environment (identical for both children — they are
            two invocations of the same user, not two users).
        stub_sentinel: File the stub ``claude`` writes as its first action;
            one file for both children, proving at least the spawn happened.
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

        redirect = _redirect_paths(self.tmp, self.home)
        self.env = _sandbox_env(self.tmp, self.home, self.bin_dir)
        self.global_settings = self.home / ".claude" / "settings.json"
        self.cache_dir, self.config_dir = _platformdirs_paths_under(redirect)
        self.cache_file = self.cache_dir / "model_context_overrides.json"

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

    def write_stub(self) -> None:
        """Write the stub ``claude`` binary onto the sandboxed PATH.

        The stub writes its sentinel and then blocks (``sleep``), so both
        launch children hold a live child process the parent can observe
        and later signal. POSIX gets an ``sh`` script; Windows a ``.cmd``
        file (the extensions ``discover_binary``'s ``shutil.which``
        consults via ``PATHEXT``) whose blocking primitive is ``ping`` —
        SIGKILL/SIGTERM via ``proc.kill()``/``send_signal`` land as
        TerminateProcess, which kills either flavour identically.
        """
        if sys.platform == "win32":
            stub_path = self.bin_dir / "claude.cmd"
            stub_path.write_text(
                "@echo off\r\n"
                f'type nul > "{self.stub_sentinel}"\r\n'
                "ping -n 999 127.0.0.1 >nul\r\n",
                encoding="utf-8",
                newline="",
            )
            return

        stub_path = self.bin_dir / "claude"
        stub_path.write_text(
            f'#!/bin/sh\nprintf ran > "{self.stub_sentinel}"\nexec sleep 300\n',
            encoding="utf-8",
            newline="\n",
        )
        stub_path.chmod(0o755)


# The child driver, written into the sandbox per test. It seeds the profile
# and credential stores through the real store classes, runs the real
# ``launch_async`` against the real ``ClaudeAdapter`` (binary discovered via
# the sandboxed PATH), records the session-file path for the parent's
# readiness barrier, and exits with launch_async's mapped code. Each of the
# two concurrent children runs its own copy of this driver in its own
# process, with its own spec — so its own session record, bridge server and
# atexit state.
_CHILD_DRIVER_SOURCE = '''
"""Child driver for the concurrent-sessions L3 scenario (sandbox-only)."""

import asyncio
import json
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

    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialStore
    from kitty.profiles.schema import Profile
    from kitty.profiles.store import ProfileStore

    store = ProfileStore()
    cred_store = CredentialStore(backends=[FileBackend()])

    profile = Profile(
        name="l3-concurrent",
        provider="zai_regular",
        model="stub-model",
        auth_ref=str(uuid.uuid4()),
        is_default=True,
    )
    store.save(profile)
    cred_store.set(profile.auth_ref, "sk-l3-stub-key")

    from kitty.cli.launcher import launch_async

    code = asyncio.run(
        launch_async(
            adapter=_RecordingClaudeAdapter(record_path=record_path),
            provider=_StubProvider(),
            profile=profile,
            cred_store=cred_store,
            validate=False,
        )
    )
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
'''



def _spawn_launch_child(sandbox: _Sandbox, name: str) -> subprocess.Popen[str]:
    """Spawn one launch child under the sandboxed environment.

    Each child gets its own driver copy, spec and session record — two
    invocations of the same launcher, not one launcher serving two
    sessions, which is precisely the topology issue #22 was about.

    Args:
        sandbox: The sandbox providing the redirected environment.
        name: ``"a"`` or ``"b"`` — suffixes the child's scratch files.

    Returns:
        The running child process; its output streams into a per-child log.
    """
    driver_path = sandbox.root / f"child_driver_{name}.py"
    spec_path = sandbox.root / f"child_spec_{name}.json"
    session_record = sandbox.root / f"session-path-{name}.txt"
    spec = {"record_session_path": str(session_record)}
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    driver_path.write_text(_CHILD_DRIVER_SOURCE, encoding="utf-8")
    log_path = sandbox.root / f"child-{name}.log"
    # stderr merged into stdout so a single sink captures both — passing the
    # same file object as both stdout and stderr hits a Popen dup2 ordering
    # quirk where the second dup wins and the first stream ends up wherever
    # fd 1 started, which loses half the child's output.
    log_handle = log_path.open("wb")
    return subprocess.Popen(  # noqa: S603 — fixed argv, sandboxed env
        [str(_VENV_PYTHON), str(driver_path), str(spec_path)],
        env=sandbox.env,
        cwd=str(sandbox.tmp),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )


def _child_log_text(sandbox: _Sandbox, name: str) -> str:
    """Return a child's combined output, or a placeholder when it wrote none.

    Args:
        sandbox: The sandbox holding the per-child logs.
        name: The child's letter.

    Returns:
        The decoded log text.
    """
    log_path = sandbox.root / f"child-{name}.log"
    if not log_path.exists():
        return "<none>"
    return log_path.read_text(encoding="utf-8", errors="replace")


def _wait_for_session_record(sandbox: _Sandbox, name: str, proc: subprocess.Popen[str]) -> Path:
    """Block until one child has recorded its session-file path.

    The record appears inside ``prepare_launch`` — after the per-session
    file exists and atexit is registered, before the stub spawns. Asserting
    on the session files before this barrier would race the very code under
    test: the file the assertion reads may not exist yet.

    Args:
        sandbox: The sandbox holding the per-child records.
        name: The child's letter.
        proc: That child's process, polled so a crash surfaces as a clear
            failure instead of a timeout.

    Returns:
        The recorded session-file path.

    Raises:
        AssertionError: When the record does not appear inside the timeout
            or the child exits first; the child log is attached.
    """
    record_path = sandbox.root / f"session-path-{name}.txt"
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        # Non-empty read — Path.write_text opens the file before the write
        # closes it; an empty read in that microsecond window would parse as
        # ``Path(".")`` and produce a confusing "both sessions resolved to
        # the same settings file" failure. Require content.
        text = record_path.read_text(encoding="utf-8").strip() if record_path.exists() else ""
        if text:
            return Path(text)
        if proc.poll() is not None:
            break
        time.sleep(_POLL_INTERVAL_SECONDS)
    proc.kill()
    proc.wait()
    raise AssertionError(
        f"launch child {name!r} never recorded its session file "
        f"(exit code {proc.returncode})\n--- child {name} log ---\n{_child_log_text(sandbox, name)}"
    )


def _wait_for_stub(sandbox: _Sandbox, procs: list[subprocess.Popen[str]]) -> None:
    """Block until the stub ``claude`` has spawned in at least one child.

    The sentinel is written by the stub itself, so its absence means no
    launch resolved to the stubbed binary. The stub runs in ``sleep`` mode,
    so a child that reached it stays alive until the parent signals — the
    state the not-disturbed assertion needs.

    Args:
        sandbox: The sandbox holding the sentinel.
        procs: Both children, polled so a crash surfaces as a clear failure.

    Raises:
        AssertionError: When the sentinel does not appear inside the timeout
            or both children exit first; the children's logs are attached.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if sandbox.stub_sentinel.exists():
            return
        if all(proc.poll() is not None for proc in procs):
            break
        time.sleep(_POLL_INTERVAL_SECONDS)
    for proc in procs:
        proc.kill()
    for proc in procs:
        proc.wait()
    raise AssertionError(
        "the stub claude binary never ran in either child"
        f"\n--- child a log ---\n{_child_log_text(sandbox, 'a')}"
        f"\n--- child b log ---\n{_child_log_text(sandbox, 'b')}"
    )


def _terminate_children(procs: list[subprocess.Popen[str]]) -> None:
    """End any still-running launch children; never mask a test failure.

    A mid-flight assertion failure would otherwise leak two live launch
    children — each holding a bridge server and a sleeping stub ``claude``
    (``exec sleep 300``) — into the rest of the suite. Every scenario calls
    this from a ``finally``. SIGTERM first (launch_async forwards it to the
    stub, so the whole process tree exits through the product's own path);
    escalate to kill for a child that ignores it. The wait is bounded by
    ``_TEARDOWN_TIMEOUT_SECONDS`` per child — teardown must not turn a
    hung-child defect into a four-minute stall before the original
    assertion surfaces. Failures here are suppressed so the original
    assertion error survives teardown.

    Args:
        procs: The launch children to end.
    """
    for proc in procs:
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.send_signal(signal.SIGTERM)
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=_TEARDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait()


def _session_port(session_path: Path) -> int:
    """Extract the bridge port from a session file's base URL.

    Args:
        session_path: The per-session settings file.

    Returns:
        The port embedded in ``env.ANTHROPIC_BASE_URL``.

    Raises:
        AssertionError: When the file is missing or carries no parseable
            loopback base URL.
    """
    assert session_path.exists(), f"session file {session_path} is missing"
    body = json.loads(session_path.read_text(encoding="utf-8"))
    base_url = body["env"]["ANTHROPIC_BASE_URL"]
    assert base_url.startswith("http://127.0.0.1:"), f"unexpected base URL {base_url!r}"
    return int(base_url.rsplit(":", 1)[1])


@pytest.fixture()
def sandbox(tmp_path: Path) -> _Sandbox:
    """Provide a fully redirected sandbox with a pristine user-global.

    Args:
        tmp_path: pytest's per-test temp directory.

    Returns:
        The ready sandbox: global settings written, model-context cache
        primed, blocking stub ``claude`` on PATH.
    """
    box = _Sandbox(tmp_path)
    box.write_global(_PRISTINE_GLOBAL)
    box.prime_cache()
    box.write_stub()
    return box


# ── AC-1: each concurrent session gets its own --settings temp file ─────────


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGTERM) is TerminateProcess on Windows — indistinguishable from SIGKILL",
)
def test_concurrent_sessions_each_get_their_own_settings_file(sandbox: _Sandbox) -> None:
    """Two simultaneous sessions produce two distinct per-session files.

    Both children are spawned before either is waited on, so their
    ``prepare_launch`` calls genuinely overlap in wall-clock time. Each must
    record a session-file path of its own; the two paths must differ; both
    files must exist while both children still run; and each must route to
    its own bridge — the two embedded ports differ.

    Args:
        sandbox: The redirected sandbox.
    """
    proc_a = _spawn_launch_child(sandbox, "a")
    proc_b = _spawn_launch_child(sandbox, "b")
    try:
        session_a = _wait_for_session_record(sandbox, "a", proc_a)
        session_b = _wait_for_session_record(sandbox, "b", proc_b)
        _wait_for_stub(sandbox, [proc_a, proc_b])

        # Both children are alive past their barriers: the files on disk are
        # a snapshot of two sessions mid-flight, not one session's
        # before/after.
        assert proc_a.poll() is None, "child a exited before the assertion"
        assert proc_b.poll() is None, "child b exited before the assertion"

        assert session_a != session_b, "both sessions resolved to the same settings file"
        assert session_a.exists(), "session a's settings file vanished while it runs"
        assert session_b.exists(), "session b's settings file vanished while it runs"

        port_a = _session_port(session_a)
        port_b = _session_port(session_b)
        assert port_a != port_b, "both sessions route to the same bridge port"
    finally:
        # Leak containment: a mid-flight assertion failure must not leave two
        # live launch children (bridge servers + sleeping stubs) running.
        _terminate_children([proc_a, proc_b])

    sandbox.assert_cache_untouched()


# ── AC-2: neither session touches ~/.claude/settings.json ────────────────────


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGTERM) is TerminateProcess on Windows — indistinguishable from SIGKILL",
)
def test_concurrent_sessions_do_not_touch_the_global_settings(sandbox: _Sandbox) -> None:
    """The user-global file is byte-identical before, during and after.

    Byte identity — not JSON-round-trip equality — at all three points: the
    per-session design promises the user's file is never rewritten, and a
    semantically-equal-but-byte-different file would still be a rewrite.

    Args:
        sandbox: The redirected sandbox.
    """
    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, "the fixture global is not pristine"

    proc_a = _spawn_launch_child(sandbox, "a")
    proc_b = _spawn_launch_child(sandbox, "b")
    try:
        # The two calls are the readiness barrier; the recorded paths
        # matter only to AC-1/AC-3, so they are not bound here.
        _wait_for_session_record(sandbox, "a", proc_a)
        _wait_for_session_record(sandbox, "b", proc_b)
        _wait_for_stub(sandbox, [proc_a, proc_b])

        # Mid-flight: both sessions fully prepared, neither has exited —
        # the point where the pre-#22 design had already rewritten the
        # global twice.
        assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, (
            "the user-global settings file was modified while both sessions run"
        )
    finally:
        # Leak containment: see AC-1's ``finally`` above.
        _terminate_children([proc_a, proc_b])

    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, (
        "the user-global settings file was modified by the session lifecycles"
    )
    sandbox.assert_cache_untouched()


# ── AC-3: the second session's start does not disturb the first (issue #22) ──


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGTERM) is TerminateProcess on Windows — indistinguishable from SIGKILL",
)
def test_second_session_start_does_not_disturb_the_first(sandbox: _Sandbox) -> None:
    """Session A's routing survives session B's start, end to end.

    Issue #22's failure mode: the second launch rewrote the one shared
    settings file, silently rerouting the first session to the second
    bridge. Under the per-session design the second launch cannot reach the
    first session's file at all — this scenario pins that at the process
    boundary: after B is fully started, A is still alive, A's file still
    carries A's port, and after both sessions end, A's cleanup removed A's
    file without ever having needed to repair the global.

    Args:
        sandbox: The redirected sandbox.
    """
    proc_a = _spawn_launch_child(sandbox, "a")
    proc_b: subprocess.Popen[str] | None = None
    try:
        session_a = _wait_for_session_record(sandbox, "a", proc_a)
        port_a = _session_port(session_a)

        # B starts while A is already mid-flight — the exact ordering issue
        # #22 broke under, not a simultaneous start.
        proc_b = _spawn_launch_child(sandbox, "b")
        session_b = _wait_for_session_record(sandbox, "b", proc_b)
        _wait_for_stub(sandbox, [proc_a, proc_b])
        port_b = _session_port(session_b)

        assert proc_a.poll() is None, "session a died when session b started"
        assert port_a != port_b, "the two sessions resolved to one bridge port"

        # A's file is A's: same path, same routing, while B runs.
        assert session_a.exists(), "session b's start removed session a's settings file"
        assert _session_port(session_a) == port_a, (
            "session b's start rerouted session a's settings file (issue #22 regression)"
        )
    finally:
        # Leak containment: proc_b may not exist yet if an earlier line
        # raised; ``_terminate_children`` skips processes that already
        # exited, and the SIGTERM path is the same one the happy flow uses,
        # so teardown and success end the sessions identically.
        to_end = [proc_a] + ([proc_b] if proc_b is not None else [])
        _terminate_children(to_end)

    # Both children were ended via SIGTERM (the Ctrl-C path): the mapped
    # exit code is not the assertion, the file-cleanup outcome is.
    assert not session_a.exists(), "session a's settings file survived cleanup"
    assert session_b is not None and not session_b.exists(), "session b's settings file survived cleanup"
    assert sandbox.global_settings.read_bytes() == _PRISTINE_GLOBAL, (
        "the user-global settings file was modified by the session lifecycles"
    )
    sandbox.assert_cache_untouched()


# ── Falsification (plan §1.4): the probe sees the pre-#22 clobber ───────────


def test_shared_settings_path_clobber_is_visible_to_the_probe(tmp_path: Path) -> None:
    """The oracle the concurrent tests use reports a shared-path clobber.

    Reproduces the pre-issue-#22 failure shape — both sessions' routing
    written onto ONE settings file, the second write winning — and asserts
    the ``env.ANTHROPIC_BASE_URL`` read the concurrent assertions perform
    reports the first session's routing as lost. ``prepare_launch`` cannot
    produce this state any more (mkstemp hands every call a fresh path —
    that IS the fix), so the control writes both env blocks the way the
    shared-file design did. If this control ever failed, the concurrent
    tests could pass vacuously: their probe could not tell isolation from a
    clobber.

    Args:
        tmp_path: pytest's per-test temp directory.
    """
    shared = tmp_path / "shared-session-settings.json"

    def _legacy_write(port: int) -> None:
        """Write the per-session env block the shared-file design produced."""
        body = {
            "env": {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                "ANTHROPIC_API_KEY": f"sk-{port}",
                "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
            }
        }
        shared.write_text(json.dumps(body, indent=2), encoding="utf-8")

    _legacy_write(10001)
    _legacy_write(10002)

    # The probe the concurrent tests use — ``_session_port`` — is the same
    # read-and-compare oracle the AC-1/AC-3 assertions perform. Calling it
    # here ties the control to that oracle: a future bug in ``_session_port``
    # (e.g. reading a stale mtime sibling) would not survive the control.
    # One assertion suffices: a probe bug that reported the first write
    # (or any value but the second writer's) fails here — the integer
    # complement (``!= 10001``) is implied and would be a tautology.
    assert _session_port(shared) == 10002, (
        "the shared-file write did not clobber — the probe's premise is broken"
    )
