"""Re-run kitty inside tmux when Claude Code is launched with ``--tmux``.

Claude Code's own ``--tmux`` moves only Claude into tmux. Kitty hosts its bridge
in-process and stays on the user's terminal, so an SSH disconnect kills kitty and
the bridge while Claude keeps running against a dead port. This module decides
whether kitty should instead start a tmux session running *itself*, and builds
everything that session needs.

The pure functions mirror how Claude Code 2.1.269 reads its worktree name and
names its tmux session, so a kitty session has the name users already see. The
design, and the reason for every rule here, is in
``.system_design/SYSTEM_DESIGN.md`` §3.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO

TMUX_FLAGS = frozenset({"--tmux", "--tmux=classic"})

_WORKTREE_FLAGS = frozenset({"-w", "--worktree"})
_WORKTREE_PREFIX = "--worktree="

# Claude Code's random-name vocabulary, copied so a generated name looks like one of Claude's.
_ADJECTIVES = ("swift", "bright", "calm", "keen", "bold")
_NOUNS = ("fox", "owl", "elm", "oak", "ray")
_SUFFIX_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

_PR_PATTERNS = (
    re.compile(r"^https?://[^/]+/[^/]+/[^/]+/pull/(\d+)/?(?:[?#].*)?$", re.IGNORECASE),
    re.compile(r"^https?://[^/]+/[^?#]+/-/merge_requests/(\d+)/?(?:[?#].*)?$", re.IGNORECASE),
    re.compile(r"^#(\d+)$"),
)

_VERSION = re.compile(r"(\d+)\.(\d+)")
_MINIMUM_TMUX = (3, 2)

MARKER = "KITTY_TMUX_WRAPPED"
ENV_FILE_VARIABLE = "KITTY_TMUX_ENV_FILE"
INNER_MODULE = "kitty.cli.tmux_inner"

# Python reads these before any code runs, so the environment file arrives too late for them (T6a).
_INTERPRETER_START_VARIABLES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)


@dataclass(frozen=True)
class Host:
    """Everything :func:`handle` needs from the outside world, injectable for tests.

    Attributes:
        environ: The process environment.
        cwd: The working directory the launch runs from.
        platform: ``sys.platform``.
        is_tty: Whether both stdin and stdout are terminals.
        python: The interpreter that runs kitty.
        rng: Random source for a generated worktree name.
        run: Run a command with captured output; returns ``(exit_code, stdout)``.
        run_attached: Run a command on the user's terminal; returns its exit code.
        which: Locate an executable, or return ``None``.
        write_env_file: Write an environment privately; returns the file's path.
        remove_file: Delete a file if it exists.
        err: Stream for messages to the user.
    """

    environ: Mapping[str, str]
    cwd: str
    platform: str
    is_tty: bool
    python: str
    rng: random.Random
    run: Callable[[list[str]], tuple[int, str]]
    run_attached: Callable[[list[str]], int]
    which: Callable[[str], str | None]
    write_env_file: Callable[[Mapping[str, str]], str]
    remove_file: Callable[[str], None]
    err: TextIO


@dataclass(frozen=True)
class WrapOutcome:
    """What the caller does next.

    Attributes:
        exit_code: When not ``None``, kitty has finished (tmux ran, or an error
            was reported) and should exit with this code.
        agent_args: When ``exit_code`` is ``None``, the arguments to launch the
            agent with in process.
    """

    exit_code: int | None
    agent_args: list[str]


def wants_tmux(adapter_name: str, agent_args: Sequence[str]) -> bool:
    """Report whether a launch asks for Claude Code's tmux mode.

    Args:
        adapter_name: The launcher adapter's name, e.g. ``"claude"``.
        agent_args: The arguments kitty will pass to the agent.

    Returns:
        ``True`` only for the Claude Code adapter with an exact ``--tmux`` or
        ``--tmux=classic`` token among its arguments.
    """
    return adapter_name == "claude" and any(arg in TMUX_FLAGS for arg in agent_args)


def strip_tmux_flags(args: Sequence[str]) -> list[str]:
    """Remove every ``--tmux`` and ``--tmux=classic`` token.

    Args:
        args: An argument list.

    Returns:
        A new list with the tmux tokens removed and every other token in order.
    """
    return [arg for arg in args if arg not in TMUX_FLAGS]


def has_worktree_flag(agent_args: Sequence[str]) -> bool:
    """Report whether the agent arguments ask Claude Code for a worktree.

    Args:
        agent_args: The arguments kitty will pass to the agent.

    Returns:
        ``True`` for ``-w``, ``--worktree`` or ``--worktree=...``, with or without a name.
    """
    return any(arg in _WORKTREE_FLAGS or arg.startswith(_WORKTREE_PREFIX) for arg in agent_args)


def worktree_name(agent_args: Sequence[str]) -> str | None:
    """Read the worktree name the way Claude Code reads it.

    After ``-w`` or ``--worktree`` the next token is the name unless it starts
    with ``-``; ``--worktree=NAME`` names it inline; the last one wins. A pull
    request reference (``#N``, a GitHub PR URL, a GitLab MR URL) becomes
    ``pr-N``, as Claude Code names such a worktree.

    Args:
        agent_args: The arguments kitty will pass to the agent.

    Returns:
        The worktree name, or ``None`` when no flag carries one.
    """
    name: str | None = None
    for index, arg in enumerate(agent_args):
        if arg in _WORKTREE_FLAGS:
            following = agent_args[index + 1] if index + 1 < len(agent_args) else ""
            if following and not following.startswith("-"):
                name = following
        elif arg.startswith(_WORKTREE_PREFIX):
            name = arg[len(_WORKTREE_PREFIX) :]
    if not name:
        return None
    # Claude resolves the pull request itself; only the name it will use matters here.
    for pattern in _PR_PATTERNS:
        match = pattern.match(name)
        if match:
            return f"pr-{match.group(1)}"
    return name


def session_name(repo_root: str, name: str) -> str:
    """Build the tmux session name Claude Code would use.

    Args:
        repo_root: The main checkout's directory (not a linked worktree's).
        name: The worktree name.

    Returns:
        ``<repo folder>_worktree-<name>`` with ``/`` in the name turned into
        ``+`` and then every ``/`` and ``.`` turned into ``_``.
    """
    folder = os.path.basename(repo_root.rstrip("/\\"))
    return re.sub(r"[/.]", "_", f"{folder}_worktree-{name.replace('/', '+')}")


def generate_worktree_name(rng: random.Random) -> str:
    """Invent a worktree name in Claude Code's style.

    Args:
        rng: The random source, injected so a caller can pin the result.

    Returns:
        A name such as ``calm-owl-k3x9``.
    """
    suffix = "".join(rng.choice(_SUFFIX_ALPHABET) for _ in range(4))
    return f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}-{suffix}"


def build_inner_argv(argv: Sequence[str], generated_name: str | None) -> list[str]:
    """Build the argument list for the kitty that runs inside tmux.

    Args:
        argv: The outer kitty's arguments, without the program name.
        generated_name: A name kitty invented because none was given, or
            ``None`` when the arguments already carry one.

    Returns:
        ``argv`` without its tmux tokens and, when a name was generated, with
        that name attached to the last worktree flag, which is the one Claude
        Code reads.
    """
    inner = strip_tmux_flags(argv)
    if generated_name is None:
        return inner
    for index in range(len(inner) - 1, -1, -1):
        if inner[index] in _WORKTREE_FLAGS:
            return [*inner[: index + 1], generated_name, *inner[index + 1 :]]
        if inner[index].startswith(_WORKTREE_PREFIX):
            return [*inner[:index], f"{_WORKTREE_PREFIX}{generated_name}", *inner[index + 1 :]]
    return inner


def parse_tmux_version(text: str) -> tuple[int, int] | None:
    """Read the major and minor version from ``tmux -V`` output.

    Args:
        text: The output, e.g. ``"tmux 3.4"`` or ``"tmux next-3.6"``.

    Returns:
        ``(major, minor)``, or ``None`` when no version number is present.
    """
    match = _VERSION.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def handle(adapter_name: str, kitty_argv: Sequence[str], agent_args: Sequence[str], host: Host) -> WrapOutcome:
    """Decide how a Claude Code launch with ``--tmux`` proceeds, and run tmux if needed.

    Callers check :func:`wants_tmux` first. The rules and their order are
    ``.system_design/SYSTEM_DESIGN.md`` §3.2.

    Args:
        adapter_name: The launcher adapter's name.
        kitty_argv: The outer kitty's arguments, without the program name.
        agent_args: The arguments kitty would pass to the agent.
        host: The injected outside world.

    Returns:
        Either an exit code, or the agent arguments for an in-process launch.
    """
    unchanged = WrapOutcome(exit_code=None, agent_args=list(agent_args))

    # Claude refuses, or a tmux session cannot attach: behave exactly as before this feature.
    if not wants_tmux(adapter_name, agent_args) or not has_worktree_flag(agent_args):
        return unchanged
    if host.platform == "win32" or not host.is_tty:
        return unchanged
    # --git-common-dir names the main checkout's .git even from a linked worktree; it may be relative.
    root_code, git_common_dir = host.run(["git", "rev-parse", "--git-common-dir"])
    if root_code != 0 or not git_common_dir.strip():
        return unchanged
    repo_root = os.path.dirname(os.path.normpath(os.path.join(host.cwd, git_common_dir.strip())))

    # Already inside tmux, the pane survives a disconnect; a nested --tmux would return at once (T2).
    if host.environ.get("TMUX"):
        return WrapOutcome(exit_code=None, agent_args=strip_tmux_flags(agent_args))

    if host.which("tmux") is None:
        _say(host, "Error: tmux is not installed. Install it, or run without --tmux.")
        return WrapOutcome(exit_code=1, agent_args=[])
    version = parse_tmux_version(host.run(["tmux", "-V"])[1])
    if version is not None and version < _MINIMUM_TMUX:
        _say(host, "Error: kitty's --tmux support needs tmux 3.2 or newer.")
        return WrapOutcome(exit_code=1, agent_args=[])

    name = worktree_name(agent_args)
    generated = None if name else generate_worktree_name(host.rng)
    session = session_name(repo_root, name or generated or "")
    target = f"={session}"

    # A same-named session is only safe to join when kitty created it (T8).
    if host.run(["tmux", "has-session", "-t", target])[0] == 0:
        marker_code, marker = host.run(["tmux", "show-environment", "-t", target, MARKER])
        if marker_code != 0 or marker.strip() != f"{MARKER}=1":
            _say(
                host,
                f"Error: a tmux session named {session} already exists and kitty did not start it.\n"
                f"Attach to it with: tmux attach -t {session}\n"
                f"Or remove it with: tmux kill-session -t {session}",
            )
            return WrapOutcome(exit_code=1, agent_args=[])
        _say(host, f"Reattaching to {session}; this command's profile and flags are ignored.")
        return WrapOutcome(exit_code=host.run_attached(["tmux", "attach-session", "-t", target]), agent_args=[])

    _say(host, f"Session: {session} (reattach: tmux attach -t {session})")
    env_file = host.write_env_file(host.environ)
    command = [
        "tmux",
        "new-session",
        "-s",
        session,
        "-c",
        host.cwd,
        "-e",
        f"{MARKER}=1",
        "-e",
        f"{ENV_FILE_VARIABLE}={env_file}",
    ]
    for variable in _INTERPRETER_START_VARIABLES:
        if variable in host.environ:
            command += ["-e", f"{variable}={host.environ[variable]}"]
    command += ["--", host.python, "-m", INNER_MODULE, *build_inner_argv(kitty_argv, generated)]
    client_exit = host.run_attached(command)

    # Only a live session started with this very file may still need it; otherwise it is orphaned.
    _, owner = host.run(["tmux", "show-environment", "-t", target, ENV_FILE_VARIABLE])
    if owner.strip() == f"{ENV_FILE_VARIABLE}={env_file}":
        _say(host, f"Reattach with: tmux attach -t {session}")
    else:
        host.remove_file(env_file)
    return WrapOutcome(exit_code=client_exit, agent_args=[])


def _say(host: Host, message: str) -> None:
    """Write one message line to the user.

    Args:
        host: The injected outside world, whose ``err`` stream receives the line.
        message: The text, without a trailing newline.
    """
    host.err.write(f"{message}\n")
    host.err.flush()


def write_env_file(environ: Mapping[str, str]) -> str:
    """Write an environment to a new file only the current user can read.

    Args:
        environ: The variables to write.

    Returns:
        The file's path. ``mkstemp`` creates it with mode ``0600``.
    """
    descriptor, path = tempfile.mkstemp(prefix="kitty-tmux-env-", suffix=".json")
    # ensure_ascii keeps undecodable bytes (surrogate escapes) intact through the round trip.
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        json.dump(dict(environ), handle, ensure_ascii=True)
    return path


def _remove_if_present(path: str) -> None:
    """Delete a file, treating an already-missing file as success.

    Args:
        path: The file to delete.
    """
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def run_captured(argv: list[str], *, cwd: str) -> tuple[int, str]:
    """Run a command and capture its output.

    The captured output is decoded as UTF-8 with ``errors="replace"``, never
    with the locale codepage ``text=True`` alone selects: one child byte the
    codepage cannot represent used to raise :exc:`UnicodeDecodeError` into
    every caller (KBR-265).

    Args:
        argv: The command.
        cwd: The directory to run it in.

    Returns:
        ``(exit_code, stdout)``. A program that cannot be started gives ``(127, "")``.
    """
    try:
        completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    except OSError:
        return 127, ""
    return completed.returncode, completed.stdout


def default_host() -> Host:
    """Build the :class:`Host` for a real launch from the current process.

    Returns:
        A host bound to this process's environment, terminal and file system.
    """
    cwd = os.getcwd()
    return Host(
        environ=os.environ,
        cwd=cwd,
        platform=sys.platform,
        is_tty=sys.stdin.isatty() and sys.stdout.isatty(),
        python=sys.executable,
        rng=random.Random(),
        run=lambda argv: run_captured(argv, cwd=cwd),
        run_attached=lambda argv: subprocess.run(argv, cwd=cwd, check=False).returncode,
        which=shutil.which,
        write_env_file=write_env_file,
        remove_file=_remove_if_present,
        err=sys.stderr,
    )
