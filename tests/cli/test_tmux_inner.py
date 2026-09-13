"""Tests for :mod:`kitty.cli.tmux_inner`, the kitty entry point that runs inside tmux.

Traces to ``.system_design/SYSTEM_DESIGN.md`` §3.2 (T6, T7, T9, T12) and
requirements R12-R13 / AC12-AC13 of
``.requirements/20260913T165119Z_tmux_survives_disconnect/REQUIREMENTS.md``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kitty.cli import tmux_inner

_TMUX_OWNED = {
    "TMUX": "/tmp/tmux-1/default,7,0",
    "TMUX_PANE": "%3",
    "TERM": "tmux-256color",
    "TERM_PROGRAM": "tmux",
    "TERM_PROGRAM_VERSION": "3.4",
    "KITTY_TMUX_WRAPPED": "1",
}


@pytest.fixture(autouse=True)
def _restore_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the ``sys.argv`` assignment :func:`tmux_inner.run` makes for ``main``."""
    monkeypatch.setattr(sys, "argv", list(sys.argv))


def _env_file(tmp_path: Path, content: object) -> Path:
    """Write an environment file the way the outer kitty would.

    Args:
        tmp_path: The directory to write into.
        content: A mapping to serialise, or raw text written as is.

    Returns:
        The file's path.
    """
    path = tmp_path / "env.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="ascii")
    return path


class _Recorder:
    """A stand-in for ``kitty.cli.main.main`` that records what it saw and then behaves as told."""

    def __init__(self, environ: dict[str, str], outcome: object = None) -> None:
        """Remember the environment to observe and what to do when called.

        Args:
            environ: The mapping the code under test replaces, observed at call time.
            outcome: ``None`` to return normally, an ``int`` to ``sys.exit`` with,
                or an exception instance to raise.
        """
        self.environ = environ
        self.outcome = outcome
        self.file = environ.get("KITTY_TMUX_ENV_FILE", "")
        self.seen_environ: dict[str, str] | None = None
        self.seen_argv: list[str] | None = None
        self.file_existed: bool | None = None

    def __call__(self) -> None:
        """Record the environment and argv, then return, exit or raise."""
        self.seen_environ = dict(self.environ)
        self.seen_argv = list(sys.argv)
        self.file_existed = Path(self.file).exists()
        if isinstance(self.outcome, int):
            sys.exit(self.outcome)
        if isinstance(self.outcome, BaseException):
            raise self.outcome


def _run(
    tmp_path: Path,
    *,
    file_content: object = None,
    outcome: object = None,
    stdin: str = "\n",
    environ: dict[str, str] | None = None,
) -> tuple[int, _Recorder, dict[str, str], str, io.StringIO]:
    """Run the inner entry against a recorder, returning everything a test asserts on.

    Args:
        tmp_path: Where the environment file is written.
        file_content: The file's content; ``None`` writes a typical outer environment.
        outcome: What the stand-in ``main`` does (see :class:`_Recorder`).
        stdin: Text the hold prompt can read.
        environ: The pane's environment before loading; defaults to the tmux-owned set.

    Returns:
        The exit code, the recorder, the environment after the run, the error
        output, and the stdin stream (to check whether the hold read from it).
    """
    current = dict(_TMUX_OWNED) if environ is None else environ
    content = {"HOME": "/home/u", "PATH": "/usr/bin", "TMUX": "", "TERM": "xterm-256color"}
    path = _env_file(tmp_path, content if file_content is None else file_content)
    current["KITTY_TMUX_ENV_FILE"] = str(path)
    recorder = _Recorder(current, outcome)
    stdin_stream = io.StringIO(stdin)
    stderr = io.StringIO()
    code = tmux_inner.run(
        ["glm", "claude", "-w", "mike"],
        environ=current,
        main=recorder,
        stdin=stdin_stream,
        stderr=stderr,
    )
    return code, recorder, current, stderr.getvalue(), stdin_stream


def test_loaded_environment_replaces_the_current_one(tmp_path: Path) -> None:
    """A variable only the tmux server had is gone; the file's variables are present."""
    current = dict(_TMUX_OWNED, SERVER_ONLY="stale")
    code, recorder, _, _, _ = _run(tmp_path, environ=current)
    assert code == 0
    assert recorder.seen_environ is not None
    assert "SERVER_ONLY" not in recorder.seen_environ
    assert recorder.seen_environ["HOME"] == "/home/u"


def test_tmux_owned_variables_keep_the_panes_values(tmp_path: Path) -> None:
    """The file's outer-terminal TMUX and TERM must not overwrite the pane's, or kitty wraps again."""
    _, recorder, _, _, _ = _run(tmp_path)
    assert recorder.seen_environ is not None
    for name, value in _TMUX_OWNED.items():
        assert recorder.seen_environ[name] == value, name


def test_a_tmux_owned_variable_unset_in_the_pane_stays_unset(tmp_path: Path) -> None:
    """``TERM_PROGRAM`` present only in the file is not imported from the outer terminal."""
    current = {k: v for k, v in _TMUX_OWNED.items() if k != "TERM_PROGRAM"}
    content = {"HOME": "/home/u", "TERM_PROGRAM": "iTerm.app"}
    _, recorder, _, _, _ = _run(tmp_path, environ=current, file_content=content)
    assert recorder.seen_environ is not None
    assert "TERM_PROGRAM" not in recorder.seen_environ


def test_the_env_file_is_deleted_before_kitty_runs(tmp_path: Path) -> None:
    """The environment may hold API keys; it is gone by the time kitty starts, not when it ends."""
    _, recorder, _, _, _ = _run(tmp_path)
    assert recorder.file_existed is False


def test_argv_is_handed_to_main_under_the_kitty_program_name(tmp_path: Path) -> None:
    """``main`` parses ``sys.argv`` itself, so the inner arguments go there."""
    _, recorder, _, _, _ = _run(tmp_path)
    assert recorder.seen_argv == ["kitty", "glm", "claude", "-w", "mike"]


def test_a_successful_run_does_not_hold_the_pane(tmp_path: Path) -> None:
    """Exit 0 closes the pane at once, without reading stdin."""
    code, _, _, stderr, stdin = _run(tmp_path, outcome=0)
    assert code == 0
    assert "press Enter" not in stderr
    assert stdin.tell() == 0


def test_a_failed_run_holds_the_pane_until_enter(tmp_path: Path) -> None:
    """A non-zero exit shows its code and waits for a line, then exits with that code."""
    code, _, _, stderr, stdin = _run(tmp_path, outcome=2)
    assert code == 2
    assert "kitty exited with code 2 - press Enter to close" in stderr
    assert stdin.tell() == 1


def test_a_crash_prints_the_traceback_then_holds(tmp_path: Path) -> None:
    """An exception is shown in full before the prompt and becomes exit code 1."""
    code, _, _, stderr, _ = _run(tmp_path, outcome=RuntimeError("boom"))
    assert code == 1
    assert stderr.index("RuntimeError: boom") < stderr.index("kitty exited with code 1")


def test_ctrl_c_is_not_held(tmp_path: Path) -> None:
    """A user who presses Ctrl-C asked to stop, so the pane closes without a prompt."""
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, outcome=KeyboardInterrupt())


def test_end_of_input_releases_the_hold(tmp_path: Path) -> None:
    """With stdin already at EOF the prompt returns instead of hanging."""
    code, _, _, stderr, _ = _run(tmp_path, outcome=3, stdin="")
    assert code == 3
    assert "press Enter" in stderr


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_an_unreadable_env_file_is_deleted_and_the_pane_held(tmp_path: Path, content: str) -> None:
    """A bad file never reaches ``main``, is still removed, and the error stays visible."""
    code, recorder, _, stderr, _ = _run(tmp_path, file_content=content)
    assert code == 1
    assert recorder.seen_argv is None
    assert not (tmp_path / "env.json").exists()
    assert "press Enter" in stderr


def test_a_missing_env_file_holds_the_pane(tmp_path: Path) -> None:
    """No variable naming the file is a launch error, not a silent fall-back to the server's environment."""
    recorder = _Recorder({})
    stderr = io.StringIO()
    code = tmux_inner.run([], environ=dict(_TMUX_OWNED), main=recorder, stdin=io.StringIO(""), stderr=stderr)
    assert code == 1
    assert recorder.seen_argv is None
    assert "press Enter" in stderr.getvalue()


def test_importing_the_inner_entry_does_not_import_kitty_modules_that_read_the_environment() -> None:
    """``Path.home()`` is read at import time elsewhere, so nothing may be imported before the swap."""
    code = (
        "import sys, kitty.cli.tmux_inner; "
        "loaded = [m for m in sys.modules if m.startswith('kitty.') and m not in "
        "('kitty.cli', 'kitty.cli.tmux_inner')]; "
        "print(loaded)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "[]"


def test_kitty_is_imported_only_after_the_environment_is_replaced(tmp_path: Path) -> None:
    """The real path (no injected ``main``) swaps the environment before ``kitty.cli.main`` is imported.

    Modules under ``kitty`` read ``HOME`` when imported, so an earlier import
    would bake in the tmux server's home directory.
    """
    env_file = _env_file(tmp_path, {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")})
    probe = (
        "import os, sys\n"
        "from kitty.cli import tmux_inner\n"
        "original = tmux_inner.load_environment\n"
        "seen = []\n"
        "def spy(environ):\n"
        "    seen.append('kitty.cli.main' in sys.modules)\n"
        "    original(environ)\n"
        "tmux_inner.load_environment = spy\n"
        "code = tmux_inner.run(['--version'], environ=os.environ, stdin=sys.stdin, stderr=sys.stderr)\n"
        "print(code, seen, 'kitty.cli.main' in sys.modules)\n"
    )
    child_env = {**os.environ, "KITTY_TMUX_ENV_FILE": str(env_file)}
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=child_env,
        stdin=subprocess.DEVNULL,
    )
    assert completed.stdout.strip().splitlines()[-1] == "0 [False] True"
