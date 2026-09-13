"""Live proof that ``kitty claude -w <name> --tmux`` survives a terminal hang-up.

Traces to ``.system_design/SYSTEM_DESIGN.md`` §3.4 and AC15 of
``.requirements/20260913T165119Z_tmux_survives_disconnect/REQUIREMENTS.md``.

The test runs the real ``kitty`` and ``claude`` against the default kitty profile
of the current user, under a pseudo-terminal made by ``script``, on a private
tmux server. It hangs that terminal up the way a dropped SSH connection does and
then checks that kitty, its bridge and Claude are still working. A negative
control repeats the hang-up without ``--tmux`` and must see kitty die, which
proves the hang-up is real.

With ``KITTY_TMUX_E2E=1`` (set by ``.github/workflows/tmux-disconnect.yml``) a
missing prerequisite fails the test; otherwise it skips. Failure messages name
the check that failed and never include pane text or process environments,
because both can carry secrets into a public CI log.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kitty.cli import tmux_wrap

_STRICT = os.environ.get("KITTY_TMUX_E2E") == "1"
_READY_TIMEOUT = 90.0
_ANSWER_TIMEOUT = 180.0


def _require(condition: bool, what: str) -> None:
    """Fail in CI, skip elsewhere, when a prerequisite is missing.

    Args:
        condition: Whether the prerequisite holds.
        what: A description of the prerequisite, safe to print.
    """
    if condition:
        return
    if _STRICT:
        pytest.fail(f"prerequisite missing: {what}")
    pytest.skip(f"prerequisite missing: {what}")


def _wait_for(check: Callable[[], object], timeout: float, what: str) -> object:
    """Poll ``check`` until it returns a truthy value.

    Args:
        check: The condition to poll.
        timeout: Seconds before giving up.
        what: The condition's description for the failure message.

    Returns:
        The first truthy value ``check`` returned.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.5)
    pytest.fail(f"timed out after {timeout:.0f}s waiting for: {what}")


def _resolved_key() -> str:
    """Return the API key kitty will hand Claude for ``kitty claude`` (the default backend).

    A balancing profile launches with its first member's key, as
    ``kitty.cli.main._launch_target_balancing`` does.
    """
    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialStore
    from kitty.profiles.resolver import ProfileResolver
    from kitty.profiles.schema import BalancingProfile
    from kitty.profiles.store import ProfileStore

    resolver = ProfileResolver(ProfileStore())
    try:
        backend = resolver.resolve_default_backend()
    except Exception:
        _require(False, "a default kitty profile")
    if isinstance(backend, BalancingProfile):
        backend = resolver.resolve_balancing(backend.name)[0]
    key = CredentialStore(backends=[FileBackend()]).get(backend.auth_ref)  # type: ignore[union-attr]
    _require(bool(key), "a credential for the default kitty profile")
    return key or ""


@pytest.fixture
def world(tmp_path: Path) -> Iterator[dict[str, object]]:
    """Provide a git repo, a private tmux server, a seeded Claude config and the child environment.

    Yields:
        ``repo`` (Path), ``env`` (dict) and ``tmux`` (a function running tmux on
        the private server and returning ``(exit_code, stdout)``).
    """
    for program in ("kitty", "claude", "tmux", "script", "git"):
        _require(shutil.which(program) is not None, f"{program} on PATH")

    # A short socket directory: tmux socket paths under pytest's tmp_path are too long.
    socket_dir = tempfile.mkdtemp(prefix="kt-", dir="/tmp")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE_CODE_", "TMUX")) and k != "CLAUDECODE"}
    env.update(TMUX_TMPDIR=socket_dir, DISABLE_AUTOUPDATER="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "init"]):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
            cwd=repo,
            check=True,
        )

    # Interactive Claude in a fresh home stops at trust and at the custom-key prompt (§3.4).
    config = Path.home() / ".claude.json"
    previous = config.read_bytes() if config.exists() else None
    seeded = json.loads(previous) if previous else {}
    seeded["hasCompletedOnboarding"] = True
    seeded.setdefault("projects", {})[str(repo)] = {"hasTrustDialogAccepted": True}
    responses = seeded.setdefault("customApiKeyResponses", {"approved": [], "rejected": []})
    responses.setdefault("approved", []).append(_resolved_key()[-20:])
    config.write_text(json.dumps(seeded), encoding="utf-8")

    def tmux(*args: str) -> tuple[int, str]:
        """Run tmux against the private server."""
        done = subprocess.run(["tmux", *args], env=env, capture_output=True, text=True, check=False)
        return done.returncode, done.stdout

    try:
        yield {"repo": repo, "env": env, "tmux": tmux}
    finally:
        tmux("kill-server")
        shutil.rmtree(socket_dir, ignore_errors=True)
        if previous is None:
            config.unlink(missing_ok=True)
        else:
            config.write_bytes(previous)


def _start_under_a_terminal(command: list[str], repo: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start ``command`` on a fresh pseudo-terminal, as an SSH login would.

    ``script``'s stdin is a pipe kept open until the hang-up: when it ends,
    ``script`` sends end-of-input into the terminal and the program would exit
    for the wrong reason.
    """
    shell = f"stty cols 160 rows 45; exec {shlex.join(command)}"
    return subprocess.Popen(
        ["script", "-qfec", shell, "/dev/null"],
        cwd=repo,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _hang_up(terminal: subprocess.Popen[bytes]) -> None:
    """Hang the terminal up: SIGHUP its session's process group, then end ``script``."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(terminal.pid, signal.SIGHUP)
    terminal.kill()
    terminal.wait(timeout=10)
    if terminal.stdin:
        terminal.stdin.close()


def _children(pid: int) -> list[int]:
    """Return the direct children of ``pid``."""
    path = Path(f"/proc/{pid}/task/{pid}/children")
    return [int(p) for p in path.read_text().split()] if path.exists() else []


def _descendants(pid: int) -> list[int]:
    """Return every descendant of ``pid``, parents before children."""
    found: list[int] = []
    for child in _children(pid):
        found += [child, *_descendants(child)]
    return found


def _bridge_port(pids: list[int]) -> int | None:
    """Find Claude's bridge port by reading only ``ANTHROPIC_BASE_URL`` from each process environment."""
    for pid in pids:
        with contextlib.suppress(OSError):
            for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                if entry.startswith(b"ANTHROPIC_BASE_URL=http://127.0.0.1:"):
                    return int(entry.rsplit(b":", 1)[1])
    return None


def _healthy(port: int) -> bool:
    """Report whether the bridge on ``port`` answers ``/healthz`` with 200."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as response:
            return bool(response.status == 200)
    except OSError:
        return False


def _cmdline(pid: int) -> bytes:
    """Return the command line of ``pid``, or nothing if it is gone."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return b""


def _alive(pid: int) -> bool:
    """Report whether ``pid`` is a live, non-zombie process."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_a_tmux_launch_survives_the_terminal_hanging_up(world: dict[str, object]) -> None:
    """After the hang-up the session, the inner kitty and its bridge live, and Claude still answers."""
    repo, env, tmux = world["repo"], world["env"], world["tmux"]
    assert isinstance(repo, Path) and isinstance(env, dict) and callable(tmux)
    session = tmux_wrap.session_name(str(repo), "e2e-tmux")

    terminal = _start_under_a_terminal(
        ["kitty", "--no-validate", "claude", "-w", "e2e-tmux", "--tmux=classic"], repo, env
    )
    try:
        pane_pid = int(
            str(
                _wait_for(
                    lambda: tmux("list-panes", "-t", f"={session}:", "-F", "#{pane_pid}")[1].strip(),
                    _READY_TIMEOUT,
                    "the tmux session",
                )
            )
        )
        port = int(
            str(
                _wait_for(
                    lambda: _bridge_port([pane_pid, *_descendants(pane_pid)]),
                    _READY_TIMEOUT,
                    "Claude under the inner kitty",
                )
            )
        )
        _wait_for(lambda: _healthy(port), _READY_TIMEOUT, "the bridge to answer /healthz before the hang-up")
    finally:
        _hang_up(terminal)

    time.sleep(3)
    # Booleans only: pytest's assertion detail would otherwise print process data into the job log.
    session_alive = tmux("has-session", "-t", f"={session}")[0] == 0
    assert session_alive, "the tmux session did not survive the hang-up"
    inner_alive = _alive(pane_pid) and b"kitty.cli.tmux_inner" in _cmdline(pane_pid)
    assert inner_alive, "no inner kitty survived the hang-up"
    bridge_alive = _healthy(port)
    assert bridge_alive, "the bridge stopped answering /healthz after the hang-up"

    left, right = random.randint(1000, 4999), random.randint(1000, 4999)
    tmux("send-keys", "-t", f"={session}:", "-l", f"What is {left} plus {right}? Reply with only the number.")
    tmux("send-keys", "-t", f"={session}:", "Enter")
    _wait_for(
        lambda: str(left + right) in tmux("capture-pane", "-p", "-t", f"={session}:", "-S", "-200")[1],
        _ANSWER_TIMEOUT,
        "Claude to answer the prompt after the hang-up",
    )


def test_without_tmux_the_same_hang_up_kills_kitty(world: dict[str, object]) -> None:
    """Negative control: a plain launch reaches the same readiness, and the hang-up ends it."""
    repo, env = world["repo"], world["env"]
    assert isinstance(repo, Path) and isinstance(env, dict)

    terminal = _start_under_a_terminal(["kitty", "--no-validate", "claude", "-w", "e2e-plain"], repo, env)
    try:
        port = int(
            str(_wait_for(lambda: _bridge_port(_descendants(terminal.pid)), _READY_TIMEOUT, "Claude under kitty"))
        )
        _wait_for(lambda: _healthy(port), _READY_TIMEOUT, "the bridge to answer /healthz before the hang-up")
        launched = _descendants(terminal.pid)
    finally:
        _hang_up(terminal)

    _wait_for(lambda: not _healthy(port), 30, "the bridge to stop after the hang-up")
    _wait_for(lambda: not any(_alive(pid) for pid in launched), 30, "every process of the plain launch to exit")
