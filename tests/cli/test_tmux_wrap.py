"""Tests for :mod:`kitty.cli.tmux_wrap`, the decision to re-run kitty inside tmux.

Traces to ``.system_design/SYSTEM_DESIGN.md`` §3 and
``.requirements/20260913T165119Z_tmux_survives_disconnect/REQUIREMENTS.md``.
Every side effect (environment, working directory, terminal, platform, child
processes, the environment file) is injected through
:class:`kitty.cli.tmux_wrap.Host`, so these are layer-1 tests with no tmux, git
or filesystem dependency except where a test says otherwise.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from kitty.cli import tmux_wrap
from kitty.cli.main import _build_parser

_GENERATED = re.compile(r"^(swift|bright|calm|keen|bold)-(fox|owl|elm|oak|ray)-[0-9a-z]{4}$")


# --- R1: trigger -------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "args", "expected"),
    [
        ("claude", ["-w", "mike", "--tmux"], True),
        ("claude", ["-w", "mike", "--tmux=classic"], True),
        ("codex", ["-w", "mike", "--tmux"], False),
        ("claude", ["-w", "mike", "--tmuxx"], False),
        ("claude", ["-w", "mike", "--tmux=cc"], False),
        ("claude", ["-w", "mike"], False),
    ],
)
def test_wants_tmux_only_for_claude_with_an_exact_tmux_token(adapter: str, args: list[str], expected: bool) -> None:
    """The wrap is considered for Claude Code's two exact ``--tmux`` spellings and nothing else."""
    assert tmux_wrap.wants_tmux(adapter, args) is expected


def test_strip_tmux_flags_removes_every_tmux_token_and_keeps_order() -> None:
    """Both spellings are removed wherever they appear; other tokens keep their order."""
    args = ["--tmux", "-w", "mike", "--tmux=classic", "--effort=high"]
    assert tmux_wrap.strip_tmux_flags(args) == ["-w", "mike", "--effort=high"]


# --- R4: worktree name -------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-w", "mike"], "mike"),
        (["--worktree", "mike"], "mike"),
        (["--worktree=mike"], "mike"),
        (["-w", "--effort=high"], None),
        (["-w"], None),
        (["--worktree="], None),
        (["-w", "#42"], "pr-42"),
        (["-w", "https://github.com/o/r/pull/42/"], "pr-42"),
        (["-w", "HTTPS://gitlab.com/g/r/-/merge_requests/7?x=1"], "pr-7"),
        (["-w", "https://github.com/o/r/issues/4"], "https://github.com/o/r/issues/4"),
        (["--effort=high"], None),
    ],
)
def test_worktree_name_follows_claudes_scan(args: list[str], expected: str | None) -> None:
    """The name is read the way Claude Code reads it, including the PR forms."""
    assert tmux_wrap.worktree_name(args) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-w"], True),
        (["--worktree"], True),
        (["--worktree="], True),
        (["--worktree=mike"], True),
        (["--tmux", "--effort=high"], False),
        (["-wmike"], False),
    ],
)
def test_has_worktree_flag_recognises_the_three_spellings(args: list[str], expected: bool) -> None:
    """A worktree flag without a name still counts: kitty will generate the name."""
    assert tmux_wrap.has_worktree_flag(args) is expected


# --- R5: session name --------------------------------------------------------


@pytest.mark.parametrize(
    ("root", "name", "expected"),
    [
        ("/home/u/projects/kitty-bridge", "mike", "kitty-bridge_worktree-mike"),
        ("/r/my.app", "feat/x.y", "my_app_worktree-feat+x_y"),
        ("/r/app/", "pr-42", "app_worktree-pr-42"),
    ],
)
def test_session_name_matches_claude_codes_rule(root: str, name: str, expected: str) -> None:
    """``/`` in the name becomes ``+``, then every ``/`` and ``.`` becomes ``_``."""
    assert tmux_wrap.session_name(root, name) == expected


# --- R6: generated name ------------------------------------------------------


def test_generated_name_has_claudes_shape() -> None:
    """Many draws all fit the adjective-noun-suffix shape Claude Code uses."""
    rng = random.Random(0)
    names = {tmux_wrap.generate_worktree_name(rng) for _ in range(200)}
    assert all(_GENERATED.match(n) for n in names), sorted(names)[:5]
    assert len(names) > 100


def test_generated_name_is_determined_by_the_rng() -> None:
    """The same seed gives the same name, so tests of the orchestration can pin it."""
    assert tmux_wrap.generate_worktree_name(random.Random(7)) == tmux_wrap.generate_worktree_name(random.Random(7))


# --- R7: inner argv ----------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "generated", "expected"),
    [
        (
            ["--debug", "glm", "claude", "-w", "mike", "--tmux=classic", "--effort=high"],
            None,
            ["--debug", "glm", "claude", "-w", "mike", "--effort=high"],
        ),
        (
            ["glm", "claude", "-w", "--debug", "--tmux"],
            "calm-owl-ab12",
            ["glm", "claude", "-w", "calm-owl-ab12", "--debug"],
        ),
        (
            ["claude", "--worktree=", "--tmux"],
            "calm-owl-ab12",
            ["claude", "--worktree=calm-owl-ab12"],
        ),
        (
            ["claude", "--tmux", "--worktree"],
            "calm-owl-ab12",
            ["claude", "--worktree", "calm-owl-ab12"],
        ),
        (
            ["claude", "-w", "--worktree=", "--tmux"],
            "calm-owl-ab12",
            ["claude", "-w", "--worktree=calm-owl-ab12"],
        ),
    ],
)
def test_build_inner_argv(argv: list[str], generated: str | None, expected: list[str]) -> None:
    """``--tmux`` is removed and a generated name goes where Claude Code will read it."""
    inner = tmux_wrap.build_inner_argv(argv, generated)
    assert inner == expected
    if generated is not None:
        assert tmux_wrap.worktree_name(inner) == generated


@pytest.mark.parametrize(
    "argv",
    [
        ["--debug", "glm", "claude", "-w", "mike", "--tmux=classic", "--effort=high"],
        ["glm", "claude", "-w", "--debug", "--tmux"],
        ["--logging", "claude", "--worktree=", "--tmux", "--resume", "x"],
    ],
)
def test_inner_argv_parses_to_the_same_kitty_flags_and_agent_args(argv: list[str]) -> None:
    """Kitty's own parser reads the inner argv as the outer one, minus ``--tmux``, plus the name."""
    parser = _build_parser()
    outer, outer_unknown = parser.parse_known_args(argv)
    agent_args = tmux_wrap.strip_tmux_flags(outer_unknown)
    generated = None if tmux_wrap.worktree_name(agent_args) else "calm-owl-ab12"

    inner, inner_unknown = parser.parse_known_args(tmux_wrap.build_inner_argv(argv, generated))

    expected_agent = tmux_wrap.build_inner_argv(agent_args, generated)
    assert vars(inner) == vars(outer)
    assert inner_unknown == expected_agent
    assert tmux_wrap.worktree_name(inner_unknown) == (generated or tmux_wrap.worktree_name(agent_args))


def test_parse_tmux_version_reads_major_minor_or_nothing() -> None:
    """Version strings tmux really prints; an unparseable one yields ``None``."""
    assert tmux_wrap.parse_tmux_version("tmux 3.4\n") == (3, 4)
    assert tmux_wrap.parse_tmux_version("tmux 3.2a") == (3, 2)
    assert tmux_wrap.parse_tmux_version("tmux 3.1c") == (3, 1)
    assert tmux_wrap.parse_tmux_version("tmux next-3.6") == (3, 6)
    assert tmux_wrap.parse_tmux_version("tmux openbsd-7.4") == (7, 4)
    assert tmux_wrap.parse_tmux_version("tmux master") is None


# --- Orchestration: R2, R3, R8-R11 --------------------------------------------

_ROOT_COMMAND = ["git", "rev-parse", "--git-common-dir"]


class FakeHost:
    """Build a :class:`tmux_wrap.Host` whose processes and files are scripted and recorded.

    ``replies`` maps a command's argv (as a tuple) to ``(exit_code, stdout)``;
    an unscripted command returns ``(1, "")``. Every captured and attached
    command is appended to ``calls`` so a test can assert on order and shape.
    """

    def __init__(self, **overrides: object) -> None:
        """Create a wrap-eligible host: a TTY on Linux, outside tmux, in a git repo with tmux 3.4.

        Args:
            **overrides: Replacement values for any :class:`tmux_wrap.Host` field,
                plus ``replies`` (merged over the defaults) and ``attached_exit``.
        """
        self.calls: list[list[str]] = []
        self.files: dict[str, str] = {}
        self.written: list[tuple[str, dict[str, str]]] = []
        self.errors: list[str] = []
        self.replies: dict[tuple[str, ...], tuple[int, str]] = {
            tuple(_ROOT_COMMAND): (0, "/home/u/projects/kitty-bridge/.git\n"),
            ("tmux", "-V"): (0, "tmux 3.4\n"),
        }
        self.replies.update(overrides.pop("replies", {}))  # type: ignore[arg-type]
        self.attached_exit = overrides.pop("attached_exit", 0)
        fields: dict[str, object] = {
            "environ": {"PATH": "/usr/bin", "HOME": "/home/u"},
            "cwd": "/home/u/projects/kitty-bridge",
            "platform": "linux",
            "is_tty": True,
            "python": "/venv/bin/python",
            "rng": random.Random(3),
            "run": self._run,
            "run_attached": self._run_attached,
            "which": lambda name: f"/usr/bin/{name}",
            "write_env_file": self._write_env_file,
            "remove_file": lambda path: self.files.pop(path, None),
            "err": self,
        }
        fields.update(overrides)
        self.host = tmux_wrap.Host(**fields)  # type: ignore[arg-type]

    def write(self, text: str) -> int:
        """Collect text the code under test writes to its error stream.

        Args:
            text: The text written.

        Returns:
            The number of characters written.
        """
        self.errors.append(text)
        return len(text)

    def flush(self) -> None:
        """Accept a flush of the error stream."""

    @property
    def stderr(self) -> str:
        """Return everything written to the error stream so far."""
        return "".join(self.errors)

    def _run(self, argv: list[str]) -> tuple[int, str]:
        """Record a captured command and return its scripted reply."""
        self.calls.append(argv)
        return self.replies.get(tuple(argv), (1, ""))

    def _run_attached(self, argv: list[str]) -> int:
        """Record an attached command and return the scripted client exit code."""
        self.calls.append(argv)
        self.errors.append("<attached>")
        return self.attached_exit  # type: ignore[return-value]

    def _write_env_file(self, environ: object) -> str:
        """Record the environment the code under test asked to write."""
        path = f"/tmp/kitty-env-{len(self.written)}.json"
        self.files[path] = repr(dict(environ))  # type: ignore[call-overload]
        self.written.append((path, dict(environ)))  # type: ignore[call-overload]
        return path

    def tmux_commands(self, subcommand: str) -> list[list[str]]:
        """Return every recorded ``tmux <subcommand>`` call."""
        return [c for c in self.calls if c[:2] == ["tmux", subcommand]]


_ARGV = ["glm", "claude", "-w", "mike", "--tmux=classic", "--effort=high"]
_AGENT = ["-w", "mike", "--tmux=classic", "--effort=high"]
_SESSION = "kitty-bridge_worktree-mike"


def _handle(fake: FakeHost, argv: list[str] = _ARGV, agent: list[str] = _AGENT) -> tmux_wrap.WrapOutcome:
    """Run the orchestration for the Claude adapter against a fake host.

    Args:
        fake: The scripted host.
        argv: The outer kitty's arguments.
        agent: The routed agent arguments.

    Returns:
        What :func:`tmux_wrap.handle` decided.
    """
    return tmux_wrap.handle("claude", argv, agent, fake.host)


@pytest.mark.parametrize(
    ("overrides", "agent"),
    [
        ({}, ["--tmux", "--effort=high"]),
        ({"platform": "win32"}, _AGENT),
        ({"is_tty": False}, _AGENT),
        ({"replies": {tuple(_ROOT_COMMAND): (128, "")}}, _AGENT),
    ],
    ids=["no-worktree-flag", "windows", "not-a-tty", "not-a-git-repo"],
)
def test_pass_through_leaves_the_launch_untouched(overrides: dict[str, object], agent: list[str]) -> None:
    """Each pass-through condition alone launches in process with the arguments unchanged."""
    fake = FakeHost(**overrides)
    outcome = _handle(fake, agent=agent)
    assert outcome == tmux_wrap.WrapOutcome(exit_code=None, agent_args=agent)
    assert not [c for c in fake.calls if c[0] == "tmux"]


@pytest.mark.parametrize("overrides", [{"platform": "win32"}, {"is_tty": False}], ids=["windows", "not-a-tty"])
def test_pass_through_decides_before_asking_git(overrides: dict[str, object]) -> None:
    """Windows and a missing terminal pass through without running any command at all."""
    fake = FakeHost(**overrides)
    _handle(fake)
    assert fake.calls == []


def test_a_relative_git_directory_is_resolved_against_the_working_directory() -> None:
    """``git rev-parse --git-common-dir`` prints ``.git`` at the top of a checkout."""
    fake = FakeHost(cwd="/srv/my.repo", replies={tuple(_ROOT_COMMAND): (0, ".git\n")})
    _handle(fake)
    (command,) = fake.tmux_commands("new-session")
    assert command[3] == "my_repo_worktree-mike"


def test_pass_through_is_checked_before_the_inside_tmux_rule() -> None:
    """Inside tmux with no worktree flag, ``--tmux`` still reaches Claude untouched."""
    fake = FakeHost(environ={"TMUX": "/tmp/tmux-1/default,1,0"})
    outcome = _handle(fake, agent=["--tmux"])
    assert outcome.agent_args == ["--tmux"]


def test_inside_tmux_strips_the_flag_and_launches_in_place() -> None:
    """With ``TMUX`` set the pane already survives a disconnect, so only the flag goes."""
    fake = FakeHost(environ={"TMUX": "/tmp/tmux-1/default,1,0"})
    outcome = _handle(fake)
    assert outcome == tmux_wrap.WrapOutcome(exit_code=None, agent_args=["-w", "mike", "--effort=high"])
    assert not [c for c in fake.calls if c[0] == "tmux"]


def test_an_empty_tmux_variable_counts_as_outside() -> None:
    """``TMUX=""`` is not a tmux session, so the wrap proceeds."""
    fake = FakeHost(environ={"TMUX": ""})
    assert _handle(fake).exit_code is not None
    assert fake.tmux_commands("new-session")


def test_missing_tmux_is_an_error() -> None:
    """No tmux binary: a plain message and exit 1, before any tmux command."""
    fake = FakeHost(which=lambda name: None)
    outcome = _handle(fake)
    assert outcome.exit_code == 1
    assert "tmux is not installed" in fake.stderr
    assert not [c for c in fake.calls if c[0] == "tmux"]


@pytest.mark.parametrize(
    ("version", "refused"),
    [("tmux 3.1c", True), ("tmux 2.9", True), ("tmux 3.2a", False), ("tmux next-3.6", False), ("tmux master", False)],
)
def test_tmux_older_than_3_2_is_refused(version: str, refused: bool) -> None:
    """``new-session -e`` needs 3.2; an unparseable version is given the benefit of the doubt."""
    fake = FakeHost(replies={("tmux", "-V"): (0, version)})
    outcome = _handle(fake)
    assert (outcome.exit_code == 1 and "3.2" in fake.stderr) is refused
    assert bool(fake.tmux_commands("new-session")) is not refused


def test_a_session_kitty_created_is_reattached() -> None:
    """A marked session is attached, not recreated, and no environment file is written."""
    fake = FakeHost(
        replies={
            ("tmux", "has-session", "-t", f"={_SESSION}"): (0, ""),
            ("tmux", "show-environment", "-t", f"={_SESSION}", "KITTY_TMUX_WRAPPED"): (0, "KITTY_TMUX_WRAPPED=1\n"),
        },
        attached_exit=5,
    )
    outcome = _handle(fake)
    assert outcome.exit_code == 5
    assert fake.calls[-1] == ["tmux", "attach-session", "-t", f"={_SESSION}"]
    assert "ignored" in fake.stderr.split("<attached>")[0]
    assert fake.written == []
    assert not fake.tmux_commands("new-session")


@pytest.mark.parametrize(
    "marker_reply",
    [(1, ""), (0, "-KITTY_TMUX_WRAPPED\n"), (0, "KITTY_TMUX_WRAPPED=0\n")],
    ids=["unset", "removed", "other-value"],
)
def test_a_session_kitty_did_not_create_is_refused(marker_reply: tuple[int, str]) -> None:
    """An unmarked session of the same name may be a Claude that bypasses kitty, so kitty stops."""
    fake = FakeHost(
        replies={
            ("tmux", "has-session", "-t", f"={_SESSION}"): (0, ""),
            ("tmux", "show-environment", "-t", f"={_SESSION}", "KITTY_TMUX_WRAPPED"): marker_reply,
        }
    )
    outcome = _handle(fake)
    assert outcome.exit_code == 1
    assert f"tmux attach -t {_SESSION}" in fake.stderr
    assert f"tmux kill-session -t {_SESSION}" in fake.stderr
    assert fake.written == []
    assert not fake.tmux_commands("attach-session")
    assert not fake.tmux_commands("new-session")


def test_new_session_command_has_the_exact_shape() -> None:
    """The marker, the file path, then interpreter-start variables that are set, then the inner kitty."""
    environ = {
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": "/extra",
        "ANTHROPIC_API_KEY": "sk-secret",
    }
    fake = FakeHost(environ=environ)
    _handle(fake)
    ((path, _),) = fake.written
    assert fake.tmux_commands("new-session") == [
        [
            "tmux", "new-session", "-s", _SESSION, "-c", "/home/u/projects/kitty-bridge",
            "-e", "KITTY_TMUX_WRAPPED=1",
            "-e", f"KITTY_TMUX_ENV_FILE={path}",
            "-e", "PYTHONPATH=/extra",
            "-e", "LANG=C.UTF-8",
            "--", "/venv/bin/python", "-m", "kitty.cli.tmux_inner",
            "glm", "claude", "-w", "mike", "--effort=high",
        ]
    ]  # fmt: skip


def test_no_other_environment_value_reaches_the_tmux_command() -> None:
    """Secrets travel only in the private file, never on the tmux command line."""
    fake = FakeHost(
        environ={"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-secret", "KITTY_EGRESS_PROXY": "http://u:pw@p:1"}
    )
    _handle(fake)
    (command,) = fake.tmux_commands("new-session")
    assert not [arg for arg in command if "sk-secret" in arg or "pw@" in arg]
    assert fake.written[0][1] == fake.host.environ


def test_session_line_is_printed_before_tmux_starts() -> None:
    """After a disconnect the client never returns, so the name must already be on screen."""
    fake = FakeHost()
    _handle(fake)
    before_attach = fake.stderr.split("<attached>")[0]
    assert f"Session: {_SESSION} (reattach: tmux attach -t {_SESSION})" in before_attach


def test_a_generated_name_is_used_for_the_session_and_the_inner_argv() -> None:
    """With no name, the invented one names the session and is handed to Claude."""
    fake = FakeHost()
    _handle(fake, argv=["claude", "-w", "--tmux"], agent=["-w", "--tmux"])
    generated = tmux_wrap.generate_worktree_name(random.Random(3))
    (command,) = fake.tmux_commands("new-session")
    assert command[3] == f"kitty-bridge_worktree-{generated}"
    assert command[-3:] == ["claude", "-w", generated]


@pytest.mark.parametrize("client_exit", [0, 3])
def test_client_exit_code_is_passed_through(client_exit: int) -> None:
    """Kitty exits with the tmux client's code (which is tmux's, not the inner kitty's)."""
    fake = FakeHost(attached_exit=client_exit)
    assert _handle(fake).exit_code == client_exit


@pytest.mark.parametrize("client_exit", [0, 3])
def test_leftover_env_file_is_deleted_once_the_session_is_gone(client_exit: int) -> None:
    """A pane that died before reading the file must not leave the environment on disk."""
    fake = FakeHost(attached_exit=client_exit)
    _handle(fake)
    assert len(fake.written) == 1
    assert fake.files == {}


def _detach_into_session_holding(fake: FakeHost, env_file_value: Callable[[str], str]) -> None:
    """Make the tmux client return into a live session whose environment names some file.

    Args:
        fake: The scripted host to modify.
        env_file_value: Given this launch's file path, the path the live session reports.
    """
    original_attached = fake.host.run_attached

    def attach(argv: list[str]) -> int:
        """Record the attach, then make ``show-environment`` report the chosen path."""
        path = fake.written[-1][0]
        reply = (0, f"{tmux_wrap.ENV_FILE_VARIABLE}={env_file_value(path)}\n")
        fake.replies[("tmux", "show-environment", "-t", f"={_SESSION}", tmux_wrap.ENV_FILE_VARIABLE)] = reply
        return original_attached(argv)

    fake.host = dataclasses.replace(fake.host, run_attached=attach)


def test_env_file_is_left_for_a_live_session_and_the_hint_repeated() -> None:
    """While this launch's session lives, the inner kitty owns the file; kitty only repeats the hint."""
    fake = FakeHost()
    _detach_into_session_holding(fake, lambda path: path)
    _handle(fake)
    assert len(fake.files) == 1
    assert fake.stderr.split("<attached>")[1].count(f"tmux attach -t {_SESSION}") == 1


def test_env_file_is_deleted_when_another_launch_owns_the_session() -> None:
    """Two launches racing for one name: the loser's file must not stay on disk, and no hint misleads."""
    fake = FakeHost()
    _detach_into_session_holding(fake, lambda path: "/tmp/someone-elses.json")
    _handle(fake)
    assert fake.files == {}
    assert "tmux attach" not in fake.stderr.split("<attached>")[1]


# --- The real host: R5 root discovery and R10 environment file ---------------


def test_write_env_file_is_private_json_that_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The file is 0600 JSON and gives back every value, including an undecodable byte."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    environ = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-secret", "ODD": "caf\udce9"}

    path = tmux_wrap.write_env_file(environ)

    assert Path(path).parent == tmp_path
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = Path(path).read_text(encoding="ascii")
    assert json.loads(raw) == environ


def _git(cwd: Path, *args: str) -> None:
    """Run a git command quietly with a throwaway identity and no commit signing.

    Args:
        cwd: The directory to run in.
        *args: The git arguments.
    """
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_session_root_is_the_main_checkout_even_from_a_linked_worktree(tmp_path: Path) -> None:
    """From a linked worktree the session is named after the main repository, as Claude Code does."""
    main = tmp_path / "my.repo"
    main.mkdir()
    _git(main, "init", "-q")
    _git(main, "commit", "-q", "--allow-empty", "-m", "init")
    _git(main, "worktree", "add", "-q", str(tmp_path / "linked"))

    for cwd in (main, tmp_path / "linked"):
        fake = FakeHost(cwd=str(cwd))
        fake.host = dataclasses.replace(fake.host, run=_recording_real_run(fake, cwd))
        _handle(fake)
        (command,) = fake.tmux_commands("new-session")
        assert command[3] == "my_repo_worktree-mike", cwd


def _recording_real_run(fake: FakeHost, cwd: Path) -> Callable[[list[str]], tuple[int, str]]:
    """Run git for real in ``cwd``; answer tmux commands from the fake's script.

    Args:
        fake: The scripted host whose tmux replies are reused.
        cwd: The directory git runs in.

    Returns:
        A replacement for :attr:`tmux_wrap.Host.run`.
    """

    def run(argv: list[str]) -> tuple[int, str]:
        """Dispatch one captured command."""
        if argv[0] == "git":
            return tmux_wrap.run_captured(argv, cwd=str(cwd))
        return fake._run(argv)

    return run


def test_run_captured_reports_a_missing_program_as_a_failure(tmp_path: Path) -> None:
    """A missing binary is an ordinary non-zero result, so ``git`` absent means "not a repo"."""
    assert tmux_wrap.run_captured(["kitty-no-such-program-xyz"], cwd=str(tmp_path))[0] != 0


# --- R14: wiring into kitty.cli.main.main ------------------------------------


def _run_main(argv: list[str], agent_args: list[str], fake: FakeHost, *, balancing: bool = False) -> tuple[int, object]:
    """Drive ``main`` with a routed Claude target, a fake host and a spy launch.

    Args:
        argv: The kitty arguments ``sys.argv`` carries.
        agent_args: The agent arguments the router returns.
        fake: The scripted host ``default_host`` returns.
        balancing: Route to a balancing profile instead of a plain one.

    Returns:
        The exit code and the ``_launch_target`` mock.
    """
    from unittest.mock import MagicMock, patch

    from kitty.cli.router import RouteResult
    from kitty.launchers.base import LauncherAdapter
    from kitty.profiles.schema import BalancingProfile, Profile

    adapter = MagicMock(spec=LauncherAdapter)
    adapter.name = "claude"
    backend: object
    if balancing:
        backend = BalancingProfile(name="pool", members=["a", "b"])
    else:
        backend = Profile(name="glm", provider="openrouter", model="m", auth_ref=str(uuid.uuid4()))
    routed = RouteResult(adapter=adapter, backend=backend, extra_args=agent_args)  # type: ignore[arg-type]

    with (
        patch("sys.argv", ["kitty", *argv]),
        patch("kitty.egress_store.resolve_egress", return_value=None),
        patch("kitty.cli.router.CLIRouter.route", return_value=routed),
        patch("kitty.cli.tmux_wrap.default_host", return_value=fake.host),
        patch("kitty.cli.main._launch_target", return_value=0) as launch,
        pytest.raises(SystemExit) as exited,
    ):
        from kitty.cli.main import main

        main()
    return exited.value.code, launch  # type: ignore[return-value]


@pytest.mark.parametrize("balancing", [False, True], ids=["profile", "balancing"])
def test_main_hands_a_tmux_launch_to_tmux_instead_of_launching(balancing: bool) -> None:
    """Outside tmux, ``main`` exits with the tmux client's code and never starts a bridge itself."""
    fake = FakeHost(attached_exit=4)
    code, launch = _run_main(_ARGV, _AGENT, fake, balancing=balancing)
    assert code == 4
    assert fake.tmux_commands("new-session")
    assert not launch.called  # type: ignore[attr-defined]


def test_main_inside_tmux_launches_with_the_flag_removed() -> None:
    """Inside tmux, ``main`` launches in process with ``--tmux`` stripped from the agent args."""
    fake = FakeHost(environ={"TMUX": "/tmp/tmux-1/default,1,0"})
    code, launch = _run_main(_ARGV, _AGENT, fake)
    assert code == 0
    assert launch.call_args.args[3] == ["-w", "mike", "--effort=high"]  # type: ignore[attr-defined]


def test_main_without_tmux_flag_does_not_build_a_host() -> None:
    """A launch without ``--tmux`` never touches the wrap, so its behaviour is exactly as before."""
    from unittest.mock import patch

    fake = FakeHost()
    with patch("kitty.cli.tmux_wrap.handle") as handle:
        code, launch = _run_main(["glm", "claude", "-w", "mike"], ["-w", "mike"], fake)
    assert code == 0
    assert not handle.called
    assert launch.call_args.args[3] == ["-w", "mike"]  # type: ignore[attr-defined]
