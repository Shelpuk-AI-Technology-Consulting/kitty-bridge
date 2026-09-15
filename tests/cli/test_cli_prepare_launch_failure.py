"""T-I3: a ``prepare_launch`` write failure must abort the launch (KBR-95).

**The product claim.** kitty routes a session through the local bridge by handing
the agent a per-session credentials file. Without that file the child would fall
back to ``~/.claude/settings.json``, whose ``env`` block outranks the process env —
bypassing the bridge, the egress guard, and usage logging, and billing the user's
own account. So if the per-session file cannot be written the launch must fail: it
must abort with a clear error, leave the user's ``~/.claude/settings.json``
byte-identical, release the bridge, and never spawn the agent.

**Why this is L3 and not the existing L1 test.** The L1 slice
(``tests/test_launch_orchestrator.py::test_launch_fails_closed_when_the_session_file_cannot_be_written``)
proves the orchestrator *component* honours an exception — with the credential
store mocked, binary discovery patched, and the child-spawn call patched. It would
not notice a refactor that resolves the child binary through a different path,
wires the credential store differently, or leaks the bound bridge port on the
abort. This module drives the same fail-closed branch through the real slice: a
real ``FileBackend`` credential file, real binary discovery finding a real stub
executable on a real ``PATH``, a real ``BridgeServer`` that binds an ephemeral
port and must release it, and a real stub binary that — if the branch were ever
removed — would actually run and leave a marker behind.

**How the failure is produced.** §6.3.2's "why a real child rather than a mock"
rationale concerns Claude Code's own precedence behaviour, which T-I3 does not
touch; what "real" must mean here is that the OS, not a stub, produces the failure.
``prepare_launch`` writes its session file into the OS temp directory via
``tempfile.mkstemp``. The sandbox redirects ``tempfile.gettempdir`` to a path that
is a **regular file**: the kernel answers ``mkstemp`` with ``ENOTDIR``
(``NotADirectoryError``, an ``OSError``) on every supported platform. We patch
``tempfile.gettempdir`` rather than ``TMPDIR`` because CPython's
``_get_default_tempdir`` falls through to the next candidate directory on any
OSError — a ``TMPDIR`` pointing at a regular file would silently land the launch
on ``/tmp`` and the branch would never be exercised (the test would pass
vacuously).

**Why a pure reporter and an in-suite falsification suite.** Per plan §1.4 the
first working version of every harness ships with a falsification case *running in
the suite*; a demonstrated-then-reverted mutation is the supplement, not the
substitute. The T-I3 contract is encoded as :func:`problems`, a pure reporter over
:class:`LaunchOutcome`; :class:`TestTheReporterCatchesDefects` feeds it
deliberately-wrong outcomes and asserts it names each defect;
:class:`TestPrepareLaunchFailureFailsClosed` drives the real launch and asserts
the contract is empty. The pattern is the same one
``tests/harness/cache_breakpoints.py`` and ``transcripts.py`` use.

**Layer.** L3 (§6.3.2), pending activation: ``tests/layers.py::PENDING_ACTIVATION_LAYERS``
acknowledges that no CI job selects ``l3`` until plan task T-K6 turns the
Subsystem job on. The tests run locally on a bare ``pytest -m l3``; when T-K6
lands they gate.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import sys
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

# L3 — TEST_SUITE.md §6.3.2. Pending activation until plan task T-K6.
pytestmark = pytest.mark.l3

from kitty.bridge.server import BridgeServer  # noqa: E402
from kitty.cli import launcher as launcher_mod  # noqa: E402
from kitty.cli.launcher import launch_async  # noqa: E402
from kitty.credentials.file_backend import FileBackend  # noqa: E402
from kitty.credentials.store import CredentialStore  # noqa: E402
from kitty.egress import set_egress  # noqa: E402
from kitty.launchers.claude import ClaudeAdapter  # noqa: E402
from kitty.profiles.schema import Profile  # noqa: E402
from kitty.providers.registry import get_provider  # noqa: E402

# ── Test fixtures ─────────────────────────────────────────────────────────────


# The pristine user-global the sandbox seeds — real bytes, later asserted
# byte-identical. The shape is deliberate: carries a non-kitty env var so any
# refactor that copies user values into the session file is caught by the
# adapter-level tests (KBR-245), not here.
_USER_SETTINGS: dict[str, object] = {
    "env": {"ANTHROPIC_AUTH_TOKEN": "user-token", "CUSTOM_KEY": "keep-me"},
}

# TCP-connect timeout for the port-release probe — a readiness check, not a wait.
_TCP_PROBE_TIMEOUT_SECONDS = 0.2


def _write_user_settings(settings_path: Path) -> bytes:
    """Seed the sandbox's user-global ``settings.json`` and return its exact bytes.

    Args:
        settings_path: Where the sandbox's ``~/.claude/settings.json`` should live.

    Returns:
        The exact bytes written, for the byte-identity assertion.
    """
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(_USER_SETTINGS, indent=2).encode("utf-8")
    settings_path.write_bytes(content)
    return content


def _write_claude_stub(bin_dir: Path) -> None:
    """Write a real ``claude`` executable whose first action leaves an execution marker.

    POSIX: an ``sh`` script at ``bin_dir/claude`` (mode 0755); ``discover_binary``
    finds it via ``shutil.which`` over the sandboxed ``PATH``.

    Windows: ``bin_dir/claude.cmd``; ``discover_binary``'s Windows branch iterates
    ``"", ".exe", ".cmd", ".bat"`` suffixes and ``shutil.which`` honours
    ``PATHEXT``, so ``.cmd`` is discoverable.

    In both flavours the stub reads ``KITTY_CHILD_MARKER`` and touches it before
    exiting 0 — the marker file's existence after the launch is the proof that the
    stub was invoked.

    Args:
        bin_dir: The sandbox bin directory placed at the front of ``PATH``.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        stub = bin_dir / "claude.cmd"
        stub.write_text(
            "@echo off\r\n"
            'if defined KITTY_CHILD_MARKER type nul > "%KITTY_CHILD_MARKER%"\r\n'
            "exit /b 0\r\n",
            encoding="utf-8",
        )
    else:
        stub = bin_dir / "claude"
        stub.write_text(
            "#!/bin/sh\n"
            ': > "$KITTY_CHILD_MARKER"\n'
            "exit 0\n",
            encoding="utf-8",
        )
        os.chmod(stub, 0o755)


def _tcp_connectable(
    host: str, port: int, *, timeout: float = _TCP_PROBE_TIMEOUT_SECONDS
) -> bool:
    """Report whether ``host:port`` accepts a TCP connection.

    Args:
        host: The bind host. Tests use ``127.0.0.1`` — the bridge binds loopback.
        port: The port to probe.
        timeout: Connect timeout in seconds; the probe is a readiness check.

    Returns:
        ``True`` when a connection was established, ``False`` on any connect error.
        Refusal is the observable of a released port; a leaked listener keeps
        accepting.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── Outcome record and pure reporter ──────────────────────────────────────────


@dataclass(frozen=True)
class LaunchOutcome:
    """The observable state of one ``launch_async`` invocation.

    Each field is one half of the T-I3 contract; :func:`problems` reads them in
    order and returns one named entry per violation. Pure data — no methods that
    touch the filesystem, the network, or the orchestrator. The same record can
    be constructed by a real launch and by a falsification test.

    Attributes:
        exit_code: Return value of ``launch_async``.
        stderr: Combined stderr captured during the launch.
        bound_port: Port the bridge bound (recorded by the sandbox's
            ``start_async`` wrapper), or ``None`` if the bridge never started.
        port_still_connectable: Whether a TCP connect to ``bound_port`` succeeds
            *after* the launch returns. The release assertion is the negation.
        child_marker_exists: Whether the stub ``claude``'s execution-marker
            file exists after the launch.
        settings_bytes: The user-global ``~/.claude/settings.json`` bytes after
            the launch.
        original_settings_bytes: The bytes the test wrote to the user-global
            file before the launch.
        blocked_tmp_still_a_file: Whether the regular file masquerading as the
            OS temp directory is still a file after the launch (mkstemp must not
            promote it to a directory).
        orphan_session_files: Any ``kitty-claude-settings-*.json`` files under
            the sandbox root — mkstemp must not leave partial writes.
        atexit_state_empty: Whether the orchestrator's atexit ledger is empty
            after the launch (a session whose prepare failed has nothing to
            clean up; a non-empty ledger would double-restore later).
    """

    exit_code: int
    stderr: str
    bound_port: int | None
    port_still_connectable: bool
    child_marker_exists: bool
    settings_bytes: bytes
    original_settings_bytes: bytes
    blocked_tmp_still_a_file: bool
    orphan_session_files: list[str] = field(default_factory=list)
    atexit_state_empty: bool = True


def problems(outcome: LaunchOutcome) -> list[str]:
    """Return the T-I3 contract violations in *outcome*.

    Empty list = the launch failed closed. Each entry names one specific
    violation; the falsification suite asserts the matching entry is present
    when the corresponding defect is fed (and *only* that entry — a defect-
    specific assertion, not a count).

    Args:
        outcome: The launch's observable state.

    Returns:
        A list of problem messages, one per detected violation. Empty when the
        launch honoured every guarantee the contract names.
    """
    found: list[str] = []
    if outcome.exit_code != 1:
        found.append(f"exit code is {outcome.exit_code}, expected 1")
    if "could not write" not in outcome.stderr or "session settings file" not in outcome.stderr:
        found.append("stderr does not name the write failure to the operator")
    if outcome.bound_port is None:
        found.append("bridge never started — the launch never reached prepare_launch")
    elif outcome.port_still_connectable:
        found.append(
            f"bridge port {outcome.bound_port} still accepts connections; it was not released"
        )
    if outcome.child_marker_exists:
        found.append(
            "child stub was invoked; the launch proceeded past the write failure"
        )
    if outcome.settings_bytes != outcome.original_settings_bytes:
        found.append("user-global settings file was modified")
    if not outcome.blocked_tmp_still_a_file:
        found.append("the blocked-tmpdir sentinel was deleted or promoted to a directory")
    if outcome.orphan_session_files:
        found.append(f"orphan session-file writes: {outcome.orphan_session_files}")
    if not outcome.atexit_state_empty:
        found.append("atexit ledger was not cleared after the failed launch")
    return found


# ── Sandbox fixture ───────────────────────────────────────────────────────────


@dataclass
class _Sandbox:
    """A redirected world one ``launch_async`` invocation runs in.

    Attributes:
        settings_path: Sandbox path for ``~/.claude/settings.json``.
        settings_bytes: The exact bytes the test wrote to ``settings_path``
            before the launch, for the byte-identity assertion.
        cred_store: A real ``CredentialStore`` over a real ``FileBackend`` at
            a sandbox path — every read/write the launch performs is a real
            file operation.
        auth_ref: The credential reference carried by :attr:`profile`.
        bound_ports: Ports the bridge bound during the launch (typically one).
        marker_path: Where the stub ``claude`` would leave a marker if invoked.
        blocked_tmp: The regular file masquerading as the OS temp directory.
        provider: The provider the launch receives — real ``zai_regular``
            (the construction path the real CLI takes).
        profile: A ``Profile`` whose ``auth_ref`` matches the seeded credential.
    """

    settings_path: Path
    settings_bytes: bytes
    cred_store: CredentialStore
    auth_ref: str
    bound_ports: list[int]
    marker_path: Path
    blocked_tmp: Path
    provider: object
    profile: Profile


@pytest.fixture
def launch_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Sandbox]:
    """Build and yield a :class:`_Sandbox` whose launch reaches ``prepare_launch``.

    Three kinds of redirect happen here, each for a different reason:

    * **Module globals bound at import time** — ``_DEFAULT_SETTINGS_PATH`` and
      ``_DEBUG_LOG_PATH`` are ``Path.home()``-derived constants, so ``HOME``
      alone cannot move them. ``_DEBUG_LOG_PATH`` matters even with debug off:
      the crash-handler installer opens that file on every bridge start (T-I1's
      W3 hermeticity lesson — no test I/O in the developer's or runner's real
      tree). ``BridgeServer._state_file`` needs no redirect: ``launch_async``
      constructs the server without it (the parameter defaults to ``None`` and
      only ``bridge_runner.py`` — the background bridge — passes one), so no
      state file is written on this path.
    * **The OS failure (R1)** — ``tempfile.gettempdir`` is redirected to a
      path that is a regular *file*, so ``prepare_launch``'s real ``mkstemp``
      fails with a kernel-produced ``NotADirectoryError``. The only patched
      behaviour is the location the temp directory resolves to; nothing
      fabricates an exception.
    * **Network hermeticity** — the catalog refresh is a best-effort HTTP
      fetch that would otherwise block the test on a connect timeout; the
      same offline patch is the standing posture of the L1 orchestrator
      fixture. ``validate=False`` (in the test body) is *required*, not
      optional: with the real ``zai_regular`` provider and a stub credential,
      ``validate_api_key`` would make a real outbound HTTP call — and worse,
      its 401 would return 1 *before* step 6, masking the OSError the branch
      exists for.
    """
    auth_ref = str(uuid.uuid4())

    home = tmp_path / "home"
    cache_dir = tmp_path / "cache"
    bin_dir = tmp_path / "bin"
    creds_file = tmp_path / "credentials.json"
    settings_path = home / ".claude" / "settings.json"
    marker_path = tmp_path / "child-ran"

    settings_bytes = _write_user_settings(settings_path)
    _write_claude_stub(bin_dir)
    creds_file.write_text(
        json.dumps({auth_ref: base64.b64encode(b"sk-test").decode("ascii")}),
        encoding="utf-8",
    )
    os.chmod(creds_file, 0o600)

    monkeypatch.setattr(
        "kitty.launchers.claude._DEFAULT_SETTINGS_PATH", settings_path
    )
    monkeypatch.setattr(
        "kitty.bridge.server._DEBUG_LOG_PATH", cache_dir / "kitty" / "bridge.log"
    )
    # Defense in depth for any late ``Path.home()`` read.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(
        "PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    )

    # Record the port the real bridge binds (thin wrapper: bind/release semantics
    # untouched) so the TCP probe can check the port after the abort.
    bound_ports: list[int] = []
    real_start = BridgeServer.start_async

    async def _record_start(self: BridgeServer) -> int:
        port = await real_start(self)
        bound_ports.append(port)
        return port

    monkeypatch.setattr(BridgeServer, "start_async", _record_start)

    async def _skip_refresh(**_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(
        launcher_mod.model_context_sync,
        "refresh_model_context_overrides",
        _skip_refresh,
    )
    # The process-wide egress default is None; state it rather than inherit
    # whatever a prior test in the session left installed.
    set_egress(None)

    # R1: the OS answers mkstemp with ENOTDIR against this regular file.
    blocked_tmp = tmp_path / "blocked-tmp"
    blocked_tmp.write_text("a regular file, not a directory", encoding="utf-8")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(blocked_tmp))

    sandbox = _Sandbox(
        settings_path=settings_path,
        settings_bytes=settings_bytes,
        cred_store=CredentialStore(backends=[FileBackend(path=creds_file)]),
        auth_ref=auth_ref,
        bound_ports=bound_ports,
        marker_path=marker_path,
        blocked_tmp=blocked_tmp,
        provider=get_provider("zai_regular"),
        profile=Profile(
            name="t-i3", provider="zai_regular", model="m", auth_ref=auth_ref
        ),
    )
    yield sandbox
    # ``monkeypatch`` restores attrs and env vars itself; ``set_egress(None)``
    # above is the global default.


# ── Reporter falsification suite (in-suite per plan §1.4) ─────────────────────


# A healthy outcome every falsification case perturbs by exactly one field.
# Each perturbation produces a single-element problem list — the test asserts
# the matching message and nothing else (no defensive count).
_BASE_OUTCOME = LaunchOutcome(
    exit_code=1,
    stderr="Error: could not write the claude session settings file: [Errno 20] Not a directory",
    bound_port=54321,
    port_still_connectable=False,
    child_marker_exists=False,
    settings_bytes=b'{"env": {"ANTHROPIC_AUTH_TOKEN": "user-token"}}',
    original_settings_bytes=b'{"env": {"ANTHROPIC_AUTH_TOKEN": "user-token"}}',
    blocked_tmp_still_a_file=True,
)


class TestTheReporterCatchesDefects:
    """``problems(outcome)`` returns an empty list for a healthy outcome and a
    named entry for every defect the T-I3 contract protects against.

    Each per-defect case feeds a single-field perturbation to the healthy
    outcome and asserts the matching problem is named — a defect-specific
    assertion, not a count. The negative control first; the eight cases cover
    every branch of :func:`problems`.
    """

    def test_healthy_outcome_reports_no_problems(self) -> None:
        """Negative control: the healthy outcome the real launch must produce."""
        assert problems(_BASE_OUTCOME) == []

    def test_exit_zero_is_caught(self) -> None:
        """A launch that returns 0 instead of 1 is a fatal product regression."""
        assert problems(replace(_BASE_OUTCOME, exit_code=0)) == [
            "exit code is 0, expected 1"
        ]

    def test_stderr_missing_the_failure_line_is_caught(self) -> None:
        """The user-facing error must name both halves of the message."""
        assert problems(
            replace(_BASE_OUTCOME, stderr="Error: something else happened")
        ) == ["stderr does not name the write failure to the operator"]

    def test_port_still_connectable_is_caught(self) -> None:
        """A bridge that holds its listening socket leaks the session."""
        assert problems(replace(_BASE_OUTCOME, port_still_connectable=True)) == [
            "bridge port 54321 still accepts connections; it was not released"
        ]

    def test_child_marker_existing_is_caught(self) -> None:
        """A child that ran against the user's credentials is the billing
        failure the branch exists to prevent.
        """
        assert problems(replace(_BASE_OUTCOME, child_marker_exists=True)) == [
            "child stub was invoked; the launch proceeded past the write failure"
        ]

    def test_user_file_modified_is_caught(self) -> None:
        """The user-global ``settings.json`` must be untouched."""
        mutated = b'{"env": {"ANTHROPIC_AUTH_TOKEN": "different"}}'
        assert problems(replace(_BASE_OUTCOME, settings_bytes=mutated)) == [
            "user-global settings file was modified"
        ]

    def test_blocked_tmp_promoted_to_directory_is_caught(self) -> None:
        """A future ``mkstemp`` variant that on ENOTDIR promotes the path
        would silently land the launch on a sibling dir — caught here.
        """
        assert problems(replace(_BASE_OUTCOME, blocked_tmp_still_a_file=False)) == [
            "the blocked-tmpdir sentinel was deleted or promoted to a directory"
        ]

    def test_orphan_session_file_is_caught(self) -> None:
        """``mkstemp`` that fails atomically leaves nothing; a partial write
        would be caught here.
        """
        orphans = ["/sandbox/kitty-claude-settings-abcde.json"]
        assert problems(
            replace(_BASE_OUTCOME, orphan_session_files=orphans)
        ) == [f"orphan session-file writes: {orphans}"]

    def test_atexit_ledger_not_cleared_is_caught(self) -> None:
        """A non-empty atexit ledger would double-restore later (the finally
        block and the registered handler both running).
        """
        assert problems(replace(_BASE_OUTCOME, atexit_state_empty=False)) == [
            "atexit ledger was not cleared after the failed launch"
        ]


# ── TCP probe controls ────────────────────────────────────────────────────────


class TestTcpProbe:
    """The TCP probe's discriminating power is proven, not assumed.

    A probe that always returns ``False`` (or always ``True``) would make AC-3
    vacuous; these two tests prove the probe can detect a live listener
    (positive) and reports a closed port (negative).
    """

    def test_probe_detects_a_live_loopback_listener(self) -> None:
        """Positive control: a fresh ``socket.bind`` is connectable."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            assert _tcp_connectable("127.0.0.1", port)
        finally:
            listener.close()

    def test_probe_reports_a_closed_loopback_port(self) -> None:
        """Negative control: nothing is listening on this port."""
        # Bind and immediately close to get an unlikely-to-collide port.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        finally:
            sock.close()
        assert not _tcp_connectable("127.0.0.1", port)


# ── Positive L3 drive ─────────────────────────────────────────────────────────


class TestPrepareLaunchFailureFailsClosed:
    """§6.3.2: ``prepare_launch`` cannot write the file — the launch must fail.

    Drives the real ``launch_async`` once with the real ClaudeAdapter, real
    FileBackend credential, real binary discovery, real BridgeServer, and the
    OS failure injected through ``tempfile.gettempdir``. The full contract is
    asserted as ``problems(outcome) == []``; the in-suite falsification suite
    above proves each problem the contract names is actually discriminable.
    """

    @pytest.mark.asyncio
    async def test_launch_fails_closed_when_session_file_cannot_be_written(
        self,
        launch_sandbox: _Sandbox,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The positive case: the launch returns 1, names the failure,
        releases the bridge, never spawns the child, leaves the user file
        intact, and clears the atexit ledger.
        """
        monkeypatch.setenv("KITTY_CHILD_MARKER", str(launch_sandbox.marker_path))

        exit_code = await launch_async(
            adapter=ClaudeAdapter(),
            provider=launch_sandbox.provider,
            profile=launch_sandbox.profile,
            cred_store=launch_sandbox.cred_store,
            extra_args=[],
            validate=False,
        )

        bound_port = launch_sandbox.bound_ports[0] if launch_sandbox.bound_ports else None
        outcome = LaunchOutcome(
            exit_code=exit_code,
            stderr=capsys.readouterr().err,
            bound_port=bound_port,
            port_still_connectable=(
                _tcp_connectable("127.0.0.1", bound_port)
                if bound_port is not None
                else False
            ),
            child_marker_exists=launch_sandbox.marker_path.exists(),
            settings_bytes=launch_sandbox.settings_path.read_bytes(),
            original_settings_bytes=launch_sandbox.settings_bytes,
            blocked_tmp_still_a_file=launch_sandbox.blocked_tmp.is_file(),
            orphan_session_files=sorted(
                str(p)
                for p in launch_sandbox.blocked_tmp.parent.glob(
                    "kitty-claude-settings-*.json"
                )
            ),
            atexit_state_empty=(launcher_mod._atexit_cleanup_state == []),
        )

        # The full contract, as a single assertion against the pure reporter —
        # each branch of which is independently proven discriminable by the
        # ``TestTheReporterCatchesDefects`` suite above.
        assert problems(outcome) == []
