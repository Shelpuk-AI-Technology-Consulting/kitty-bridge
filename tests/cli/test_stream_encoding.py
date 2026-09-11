"""Tests for output-stream encoding hardening at the kitty entry points.

KBR-10: on Windows, Python encodes ``stdout`` with the machine's *locale*
codepage whenever the stream is not a console — a pipe, a redirect to a file, a
parent process capturing output. Any character that codepage cannot represent
killed the process with :exc:`UnicodeEncodeError`, so ``kitty doctor >
diagnostics.txt`` — the command the README tells a user to run when something is
wrong — was the invocation that crashed.

The module has two halves. :class:`TestHardenOutputStreams` drives
:func:`kitty.io_encoding.harden_output_streams` against real
:class:`io.TextIOWrapper` objects and proves what it does to them. The
subprocess tests spawn a real child, because the claim is about the *interpreter
start-up encoding* and no in-process test can create one: ``PYTHONIOENCODING``
is read before the test exists.

**Layer.** ``l1`` by path default, and it stays there even though it spawns
processes: ``TEST_SUITE.md`` §8.2 forbids moving a test to ``l3`` before the
Subsystem job exists, and ``l3`` is selected by no job today. §8.2 lists this
file among the ``l1`` modules that spawn processes, for T-K6 and T-H1.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Locale codepages a Windows machine really uses, none of which is a superset of
# the other: cp1252 holds the em dash but not the status glyphs, cp437 holds
# neither, and ascii is the floor. Testing only cp1252 would miss that
# `kitty --version` crashes, because cp1252 happens to contain the one character
# the banner needs.
HOSTILE_ENCODINGS = ("cp1252", "cp437", "ascii")

# Every command a non-interactive caller can reach. `--version` and `--help` are
# here because the ticket wrongly listed them as safe: argparse emits ASCII, but
# the banner printed before it does not.
COMMANDS = (
    ("--version",),
    ("--help",),
    ("doctor",),
    ("egress", "show"),
    ("profile",),
    ("setup",),
)

# A directory name outside cp1252. This is the realistic Windows shape, not a
# contrivance: on a Russian, Japanese or Chinese install `C:\Users\<name>` is
# non-ASCII by default and `%LOCALAPPDATA%` derives from it.
NON_ASCII_DIRECTORY = "дом"

# Substrings that mean the child died rather than reported. Both are needed:
# `kitty egress show` exits 1 with and without the bug, so an exit-code
# assertion alone passes against the defect — the ticket names this trap
# explicitly.
_CRASH_MARKERS = ("UnicodeEncodeError", "Traceback (most recent call last)")


def _isolated_environment(home: Path, encoding: str) -> dict[str, str]:
    """Build a scrubbed environment for a child kitty process.

    The child must not read or write the developer's real ``~/.config/kitty``
    — ``rules/python-tests.md`` makes that a critical finding — and its exit
    code must reflect the code path rather than whichever profiles the runner
    happens to have.

    Args:
        home: Directory to use as the child's home. Created if absent.
        encoding: Value for ``PYTHONIOENCODING``, which fixes the child's
            start-up stream encoding and so simulates a Windows locale on a
            Linux runner.

    Returns:
        A complete environment mapping, with nothing inherited but the few
        variables an interpreter needs to start.
    """
    home.mkdir(parents=True, exist_ok=True)

    # Built from empty with an allowlist, never scrubbed from os.environ: a
    # denylist would have to enumerate every KITTY_* variable `main()` reads,
    # and would silently admit the next one somebody adds.
    environment = {
        "PYTHONIOENCODING": encoding,
        # POSIX reads HOME; Windows `ntpath.expanduser` never does -- it reads
        # USERPROFILE, then HOMEDRIVE + HOMEPATH. Without all of them `Path.home()`
        # raises, and a dozen kitty modules call it, so the child would die of a
        # missing home and be reported as an encoding crash.
        "HOME": str(home),
        "USERPROFILE": str(home),
        "HOMEDRIVE": "",
        "HOMEPATH": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        # Read by platformdirs on Windows, not by kitty directly.
        "LOCALAPPDATA": str(home / "AppData" / "Local"),
        "PATH": "",
    }

    # Windows cannot start an interpreter without this; everything else is
    # deliberately dropped.
    for inherited in ("SYSTEMROOT", "SystemRoot"):
        if inherited in os.environ:
            environment[inherited] = os.environ[inherited]

    return environment


def _run_child(argv: list[str], *, home: Path, encoding: str) -> subprocess.CompletedProcess[bytes]:
    """Run a child interpreter with a fixed stream encoding and capture raw bytes.

    Args:
        argv: Arguments after the interpreter, e.g. ``["-m", "kitty", "doctor"]``.
        home: Directory to isolate the child into.
        encoding: Value for ``PYTHONIOENCODING``.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` as :class:`bytes`.

    Raises:
        subprocess.TimeoutExpired: If the child has not exited within 60
            seconds, so a hang is reported here rather than as an undiagnosed
            30-minute CI timeout.

    🔴 Bytes, never ``text=True``. A ``cp1252`` child writes the em dash as the
    single byte ``0x97``; decoding that as UTF-8 — which ``text=True`` does on a
    UTF-8 runner — raises :exc:`UnicodeDecodeError` *in this process*. The
    harness would then go red in its own plumbing at the base revision, and the
    "fails before the fix" evidence would be a decode bug in disguise.
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, *argv],
        env=_isolated_environment(home, encoding),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _run_kitty(command: tuple[str, ...], *, home: Path, encoding: str) -> subprocess.CompletedProcess[bytes]:
    """Run one kitty command in an isolated child.

    Args:
        command: The command words, e.g. ``("egress", "show")``.
        home: Directory to isolate the child into.
        encoding: Value for ``PYTHONIOENCODING``.

    Returns:
        The completed process, with output as :class:`bytes`.
    """
    return _run_child(["-m", "kitty", *command], home=home, encoding=encoding)


def _crashed(completed: subprocess.CompletedProcess[bytes]) -> bool:
    """Report whether a child died of an unhandled exception.

    Args:
        completed: A finished child process.

    Returns:
        True when the child's stderr carries a traceback or an encoding error.

    ``errors="replace"`` because the bytes under test are, by construction, not
    valid UTF-8 — decoding them strictly would raise here instead of answering
    the question.
    """
    stderr = completed.stderr.decode("utf-8", errors="replace")

    return any(marker in stderr for marker in _CRASH_MARKERS)


@pytest.fixture(scope="module")
def utf8_baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, ...], int]:
    """Record each command's exit code under a UTF-8 start-up encoding.

    Args:
        tmp_path_factory: Module-scoped temporary directory factory.

    Returns:
        Exit code per command, measured once and reused by every encoding.

    This is the oracle for "the command's own exit code". Hard-coding the
    numbers instead would encode today's behaviour into the test and break on
    any unrelated change to what ``kitty doctor`` returns. The baseline is
    asserted crash-free before it is trusted: a baseline that itself died would
    otherwise let the comparison pass by matching one crash against another.
    """
    baseline: dict[tuple[str, ...], int] = {}

    for command in COMMANDS:
        # A fresh home per command, because every cell this is the oracle for
        # gets one. Sharing would compare a virgin run against one that had
        # already left lock files behind.
        home = tmp_path_factory.mktemp("utf8-baseline-home")
        completed = _run_kitty(command, home=home, encoding="utf-8")
        assert not _crashed(completed), (
            f"the UTF-8 baseline for {command} crashed, so it cannot serve as the "
            f"oracle for the hostile-codepage runs:\n"
            f"{completed.stderr.decode('utf-8', errors='replace')}"
        )
        baseline[command] = completed.returncode

    return baseline


@pytest.fixture(scope="module")
def non_ascii_home_baseline(tmp_path_factory: pytest.TempPathFactory) -> int:
    """Record ``kitty cleanup``'s exit code when its home path is not ASCII.

    Args:
        tmp_path_factory: Module-scoped temporary directory factory.

    Returns:
        The exit code under a UTF-8 start-up encoding.
    """
    home = tmp_path_factory.mktemp("non-ascii-baseline") / NON_ASCII_DIRECTORY
    completed = _run_kitty(("cleanup",), home=home, encoding="utf-8")

    assert not _crashed(completed), (
        "the UTF-8 baseline for `kitty cleanup` under a non-ASCII home crashed:\n"
        f"{completed.stderr.decode('utf-8', errors='replace')}"
    )

    return completed.returncode


def _harden() -> None:
    """Call the entry point's stream hardening.

    The import is here rather than at module scope so that a missing
    ``harden_output_streams`` fails only the unit tests. A module-level import
    would fail *collection*, taking the subprocess regression tests down with
    it — and the §16 "red at the base revision" evidence would then show an
    ``ImportError`` where it has to show the defect itself.
    """
    from kitty.io_encoding import harden_output_streams

    harden_output_streams()


def _locale_stream(encoding: str = "cp1252") -> io.TextIOWrapper:
    """Build a strict, locale-encoded text stream over an in-memory buffer.

    Args:
        encoding: The codepage to start the stream in.

    Returns:
        A stream shaped like the ``sys.stdout`` a Windows machine hands a piped
        process: a real :class:`io.TextIOWrapper`, not a double, so the tests
        assert on what ``reconfigure`` actually does rather than on a recorded
        call.
    """
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")


class TestHardenOutputStreams:
    """Unit tests for :func:`kitty.io_encoding.harden_output_streams`.

    Every test replaces **both** streams. Leaving ``sys.stderr`` alone would let
    the call reconfigure pytest's own capture object as a side effect, which is
    harmless but would make one test's behaviour depend on another's.

    The three cases that break ``sys.stdout`` also assert that ``sys.stderr``
    *was* hardened. Without that second assertion, turning the loop's
    ``continue`` into a ``break`` — or wrapping the whole loop in one ``try`` —
    would silently leave stderr on the locale codepage and every test here would
    still pass. That matters: ``print_error`` and ``print_warning`` write to
    stderr, and the bridge child's stderr is the stream ``bridge/manage.py``
    reads back.
    """

    def test_it_switches_a_locale_encoded_stream_to_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cp1252 stream ends up UTF-8, and the bytes it writes prove it."""
        stream = _locale_stream()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", _locale_stream())

        _harden()
        stream.write("✓ — ℹ")
        stream.flush()

        assert (stream.encoding, stream.errors) == ("utf-8", "backslashreplace")
        assert stream.buffer.getvalue() == "✓ — ℹ".encode()

    def test_it_hardens_stderr_as_well_as_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both streams are reconfigured, not just the one the ticket named."""
        err = _locale_stream("cp437")
        monkeypatch.setattr(sys, "stdout", _locale_stream("cp437"))
        monkeypatch.setattr(sys, "stderr", err)

        _harden()

        assert (err.encoding, err.errors) == ("utf-8", "backslashreplace")

    def test_a_lone_surrogate_no_longer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UTF-8 alone is not enough: ``errors`` has to be non-strict too.

        A lone surrogate is what :func:`os.fsdecode` produces from an
        undecodable filename, and UTF-8 cannot represent it. Reconfiguring the
        encoding without the error handler would leave this raising, and
        reconfiguring with ``encoding`` alone silently resets ``errors`` to
        ``strict``.
        """
        stream = _locale_stream()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", _locale_stream())

        _harden()
        stream.write("name-\udcff")
        stream.flush()

        assert stream.buffer.getvalue() == rb"name-\udcff"

    def test_a_stream_that_cannot_be_reconfigured_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``io.StringIO`` has no ``reconfigure``, and must not become a crash."""
        stream = io.StringIO()
        err = _locale_stream()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", err)

        _harden()

        assert sys.stdout is stream
        assert (err.encoding, err.errors) == ("utf-8", "backslashreplace")

    def test_a_stream_whose_reconfigure_raises_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A closed stream raises :exc:`ValueError`; hardening must survive it.

        Reachable in production: ``kitty >&-`` closes stdout before the process
        starts. Verified that both a closed and a detached stream raise
        ``ValueError`` here rather than something broader.
        """
        stream = _locale_stream()
        stream.close()
        err = _locale_stream()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", err)

        _harden()

        assert sys.stdout is stream
        assert (err.encoding, err.errors) == ("utf-8", "backslashreplace")

    def test_a_missing_stream_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under ``pythonw.exe`` there is no stdout at all, and it is ``None``."""
        err = _locale_stream()
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", err)

        _harden()

        assert sys.stdout is None
        assert (err.encoding, err.errors) == ("utf-8", "backslashreplace")


@pytest.mark.parametrize("encoding", HOSTILE_ENCODINGS)
@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: "-".join(c).lstrip("-"))
def test_a_command_is_unaffected_by_a_hostile_codepage(
    command: tuple[str, ...],
    encoding: str,
    tmp_path: Path,
    utf8_baseline: dict[tuple[str, ...], int],
) -> None:
    """Every kitty command survives a locale codepage it cannot encode into.

    Both assertions carry one logical claim — this invocation behaves as it does
    under UTF-8 — so they belong in one test. The exit-code half alone would
    pass against the defect for ``egress show``, which exits 1 either way; the
    traceback half is what discriminates.

    Args:
        command: The kitty command words under test.
        encoding: The start-up encoding forced on the child.
        tmp_path: Per-test isolated home.
        utf8_baseline: Exit code per command under UTF-8.

    This sweep is **breadth, not depth**. On Linux ``doctor``, ``profile`` and
    ``setup`` exit at the TTY check having printed only the banner, and
    ``--help`` dies on the banner before argparse's help path is reached, so all
    six commands exercise the same two string literals. The depth — text kitty
    does not control — is
    :func:`test_a_non_ascii_home_path_survives_a_hostile_codepage`.
    """
    completed = _run_kitty(command, home=tmp_path / "home", encoding=encoding)

    assert not _crashed(completed), (
        f"`kitty {' '.join(command)}` died under {encoding}:\n"
        f"{completed.stderr.decode('utf-8', errors='replace')}"
    )
    assert completed.returncode == utf8_baseline[command], (
        f"`kitty {' '.join(command)}` returned {completed.returncode} under {encoding} "
        f"but {utf8_baseline[command]} under utf-8; the codepage changed the outcome."
    )


@pytest.mark.parametrize("encoding", HOSTILE_ENCODINGS)
def test_a_non_ascii_home_path_survives_a_hostile_codepage(
    encoding: str,
    tmp_path: Path,
    non_ascii_home_baseline: int,
) -> None:
    """Text kitty does not control also survives — the discriminating case.

    ``kitty cleanup`` prints the settings path it inspected, so a home directory
    named ``дом`` puts a string kitty never chose onto **stdout**. This is what
    separates the fix that was built from the one that was rejected: a patch
    that only swapped the five decorative glyphs for ASCII would turn every cell
    of the breadth sweep green and still crash here, at the path.

    Args:
        encoding: The start-up encoding forced on the child.
        tmp_path: Per-test isolated home root.
        non_ascii_home_baseline: The same command's exit code under UTF-8.

    The stdout assertion is the positive half: without it the test would pass
    against a kitty that had stopped printing the path altogether.
    """
    completed = _run_kitty(("cleanup",), home=tmp_path / NON_ASCII_DIRECTORY, encoding=encoding)

    assert not _crashed(completed), (
        f"`kitty cleanup` died under {encoding} with a non-ASCII home path:\n"
        f"{completed.stderr.decode('utf-8', errors='replace')}"
    )
    assert completed.returncode == non_ascii_home_baseline
    assert NON_ASCII_DIRECTORY in completed.stdout.decode("utf-8", errors="replace"), (
        "the path kitty inspected did not reach stdout, so this test would have "
        "passed without proving anything about encoding it."
    )


@pytest.mark.parametrize("encoding", HOSTILE_ENCODINGS)
def test_the_harness_detects_an_unhardened_child(encoding: str, tmp_path: Path) -> None:
    """Falsification: the harness must fail when handed a known defect.

    ``TEST_SUITE_IMPLEMENTATION_PLAN.md`` §1.4 — a harness never shown to fail
    is indistinguishable from one that cannot. The deliberate defect is real
    product code (:func:`kitty.tui.display.print_info`) called without the entry
    point that hardens the streams, which is exactly the state ``kitty egress
    show`` was in before this change.

    ``print_info`` rather than :func:`~kitty.tui.display.print_banner`: the
    banner's em dash **is** in cp1252 (0x97), so it does not crash there — which
    is precisely why the ticket wrongly recorded ``kitty --version`` as safe.
    ``ℹ`` (U+2139) is in none of the three, and is the character the reported
    ``egress show`` traceback names.

    Parametrised over all three encodings so the case cannot later be narrowed
    to one that no longer bites.

    Routed through :func:`_run_child`, not around it, so the whole runner is
    under test: an environment builder that silently stopped passing
    ``PYTHONIOENCODING`` would leave the tests above permanently green, and
    turns this one red.

    Args:
        encoding: The start-up encoding forced on the child.
        tmp_path: Per-test isolated home.
    """
    completed = _run_child(
        ["-c", "from kitty.tui.display import print_info; print_info('x')"],
        home=tmp_path / "home",
        encoding=encoding,
    )

    assert _crashed(completed), (
        f"an unhardened child printing an unencodable glyph under {encoding} did "
        "not register as a crash, so the harness cannot detect the defect it "
        "exists to catch."
    )


@pytest.mark.parametrize("encoding", HOSTILE_ENCODINGS)
def test_the_real_process_streams_end_up_utf8(encoding: str, tmp_path: Path) -> None:
    """The hardening reaches the genuine ``sys.stdout``, not just a test double.

    The unit tests substitute a stream; the subprocess matrix only observes the
    *absence* of a crash, which ``errors="replace"`` or ``"ignore"`` would also
    produce while silently discarding the handler the design chose. This case
    reads the two attributes back out of a real interpreter and pins both.

    Args:
        encoding: The start-up encoding forced on the child.
        tmp_path: Per-test isolated home.
    """
    completed = _run_child(
        [
            "-c",
            "from kitty.io_encoding import harden_output_streams; harden_output_streams(); "
            "import sys; print(sys.stdout.encoding, sys.stdout.errors, "
            "sys.stderr.encoding, sys.stderr.errors)",
        ],
        home=tmp_path / "home",
        encoding=encoding,
    )

    assert completed.stdout.decode().split() == [
        "utf-8",
        "backslashreplace",
        "utf-8",
        "backslashreplace",
    ]


def test_the_glyphs_still_reach_stdout_on_a_utf8_stream(tmp_path: Path) -> None:
    """Kitty's own non-ASCII output survives — the fix did not degrade it.

    Every other test here proves the *absence* of a crash, which an
    implementation that stripped the em dash and ``ℹ`` from ``display.py`` would
    also achieve. That is the ASCII-glyph fix §5 D1 rejects, arriving by the
    back door, and without this assertion the suite could not see it.

    Args:
        tmp_path: Per-test isolated home.
    """
    completed = _run_kitty(("--version",), home=tmp_path / "home", encoding="utf-8")

    assert "—" in completed.stdout.decode("utf-8")


class _Sentinel(Exception):
    """Raised by a stub to stop an entry point at the moment under test."""


class TestEntryPointsHardenFirst:
    """Both entry points must harden before they write or parse anything.

    Without these, a ``harden_output_streams()`` call dropped from
    ``bridge_runner.main`` — or moved below the first write in either — is green
    across every other test in this module, because the subprocess matrix only
    ever spawns ``python -m kitty``. Follows the recorder-plus-sentinel pattern
    of ``tests/test_entry_point_refresh.py``: the sentinel guarantees the
    assertion runs at the exact moment the next step would have happened.
    """

    def test_the_cli_hardens_before_printing_the_banner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``kitty.cli.main`` hardens, then prints the banner — in that order."""
        import kitty.cli.main as cli_main
        import kitty.tui.display as display

        order: list[str] = []

        def fake_banner(_version: str) -> None:
            order.append("write")
            raise _Sentinel

        monkeypatch.setattr(cli_main, "harden_output_streams", lambda: order.append("harden"))
        monkeypatch.setattr(display, "print_banner", fake_banner)

        with pytest.raises(_Sentinel):
            cli_main.main()

        assert order == ["harden", "write"]

    def test_the_bridge_runner_hardens_before_parsing_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``kitty.bridge_runner`` hardens before argparse can write usage text.

        The bridge child's streams are pipes by construction, and argparse
        prints to them on a bad flag, so the ordering matters here for the same
        reason it does in the CLI.
        """
        import argparse

        import kitty.bridge_runner as bridge_runner

        order: list[str] = []

        def fake_parse_args(_self: argparse.ArgumentParser, *_args: object, **_kwargs: object) -> None:
            order.append("parse")
            raise _Sentinel

        monkeypatch.setattr(bridge_runner, "harden_output_streams", lambda: order.append("harden"))
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)

        with pytest.raises(_Sentinel):
            bridge_runner.main()

        assert order == ["harden", "parse"]
