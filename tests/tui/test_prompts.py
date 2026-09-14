"""Tests for TUI prompt utilities (questionary-based)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from itertools import cycle
from unittest.mock import MagicMock, patch

import pytest

from kitty.tui.menu import CheckboxMenu, SelectionMenu
from kitty.tui.prompts import NonTTYError, check_tty, prompt_confirm, prompt_secret, prompt_text


def _mock_tty(is_tty: bool = True, *, stdout_is_tty: bool | None = None):
    """Present both standard streams as terminals, or not.

    🔴 Both, since KBR-204. Patching stdin alone made this helper agree with the
    defect: ``check_tty`` could not see a redirected stdout, which is how every
    interactive command other than the menus stayed exposed on Windows after
    KBR-187. Same shape as the helper in ``tests/tui/test_menu.py``.

    Args:
        is_tty: Whether standard input is a terminal.
        stdout_is_tty: Whether standard output is one. Defaults to ``is_tty``;
            pass it explicitly to build the asymmetric case.

    Returns:
        A context manager patching both ``isatty`` calls and the KBR-218 console
        probe interpretation.
    """
    out = is_tty if stdout_is_tty is None else stdout_is_tty

    @contextmanager
    def _both():
        """Hold both ``isatty`` patches and the probe patch for the block.

        Yields:
            None. The patched state applies inside the ``with`` block.
        """
        with (
            patch("sys.stdin.isatty", return_value=is_tty),
            patch("sys.stdout.isatty", return_value=out),
            # KBR-218: on Windows ``can_interact`` consults ``_handle_attached``
            # rather than ``isatty``. Patching it keeps the helper's "simulated
            # interactivity" semantics intact on the Windows CI leg. The patch
            # is inert on POSIX (can_interact takes the ``isatty`` branch);
            # ``create=True`` because the attribute does not exist at the test
            # commit — the fix commit adds it. ``cycle`` because the patched
            # block may invoke ``can_interact`` more than once (a wizard /
            # OAuth flow calls ``check_tty`` at the top level and again inside
            # each prompt), so the values must repeat rather than exhaust on
            # the third call.
            patch(
                "kitty.tui.prompts._handle_attached",
                side_effect=cycle([is_tty, out]),
                create=True,
            ),
        ):
            yield

    return _both()


class TestCheckTty:
    def test_raises_on_non_tty(self) -> None:
        with _mock_tty(False), pytest.raises(NonTTYError, match="interactive"):
            check_tty()

    def test_passes_on_tty(self) -> None:
        with _mock_tty(True):
            check_tty()  # should not raise

    def test_raises_when_stdout_is_redirected_even_though_stdin_looks_interactive(self) -> None:
        """KBR-204: the asymmetric case KBR-187 fixed for menus, here for prompts.

        On Windows ``isatty()`` is true for any character device, ``NUL``
        included, so a child with stdin redirected to ``NUL`` reports an
        interactive stdin. A prompt must read keys *and* draw, and drawing over a
        pipe is where ``prompt_toolkit`` raises ``NoConsoleScreenBufferError``.

        Simulated rather than waited for, so every CI leg checks it rather than
        only the Windows one.
        """
        with _mock_tty(True, stdout_is_tty=False), pytest.raises(NonTTYError, match="interactive"):
            check_tty()

    def test_raises_when_stdin_is_redirected_even_though_stdout_is_a_terminal(self) -> None:
        """The pre-existing half of the rule survives the change: stdin alone still decides a refusal."""
        with _mock_tty(False, stdout_is_tty=True), pytest.raises(NonTTYError, match="interactive"):
            check_tty()


class TestPromptText:
    def test_returns_questionary_answer(self) -> None:
        mock_q = MagicMock()
        mock_q.ask.return_value = "hello"
        with _mock_tty(True), patch("kitty.tui.prompts.questionary") as mock_module:
            mock_module.text.return_value = mock_q
            result = prompt_text("Enter name: ")
        assert result == "hello"

    def test_raises_non_tty_error_on_non_tty(self) -> None:
        with _mock_tty(False), pytest.raises(NonTTYError):
            prompt_text("Enter name: ")

    def test_questionary_not_called_on_non_tty(self) -> None:
        with _mock_tty(False), patch("kitty.tui.prompts.questionary") as mock_module, pytest.raises(NonTTYError):
            prompt_text("Enter name: ")
        mock_module.text.assert_not_called()

    def test_passes_label_to_questionary(self) -> None:
        mock_q = MagicMock()
        mock_q.ask.return_value = "val"
        with _mock_tty(True), patch("kitty.tui.prompts.questionary") as mock_module:
            mock_module.text.return_value = mock_q
            prompt_text("My Label: ")
        args, _ = mock_module.text.call_args
        assert "My Label" in args[0]


class TestPromptSecret:
    def test_returns_prompt_toolkit_answer(self) -> None:
        with _mock_tty(True), patch("kitty.tui.prompts.pt_prompt", return_value="s3cr3t") as mock_prompt:
            result = prompt_secret("API key: ")
        assert result == "s3cr3t"
        mock_prompt.assert_called_once_with("API key: ", is_password=True)

    def test_raises_non_tty_error_on_non_tty(self) -> None:
        with _mock_tty(False), pytest.raises(NonTTYError):
            prompt_secret("API key: ")

    def test_prompt_toolkit_not_called_on_non_tty(self) -> None:
        with _mock_tty(False), patch("kitty.tui.prompts.pt_prompt") as mock_prompt, pytest.raises(NonTTYError):
            prompt_secret("API key: ")
        mock_prompt.assert_not_called()


class TestPromptConfirm:
    def test_returns_true_on_yes(self) -> None:
        mock_q = MagicMock()
        mock_q.ask.return_value = True
        with _mock_tty(True), patch("kitty.tui.prompts.questionary") as mock_module:
            mock_module.confirm.return_value = mock_q
            result = prompt_confirm("Continue?")
        assert result is True

    def test_returns_false_on_no(self) -> None:
        mock_q = MagicMock()
        mock_q.ask.return_value = False
        with _mock_tty(True), patch("kitty.tui.prompts.questionary") as mock_module:
            mock_module.confirm.return_value = mock_q
            result = prompt_confirm("Continue?")
        assert result is False

    def test_default_forwarded_to_questionary(self) -> None:
        mock_q = MagicMock()
        mock_q.ask.return_value = False
        with _mock_tty(True), patch("kitty.tui.prompts.questionary") as mock_module:
            mock_module.confirm.return_value = mock_q
            prompt_confirm("Continue?", default=False)
        _, kwargs = mock_module.confirm.call_args
        assert kwargs.get("default") is False

    def test_raises_non_tty_error_on_non_tty(self) -> None:
        with _mock_tty(False), pytest.raises(NonTTYError):
            prompt_confirm("Continue?")

    def test_questionary_not_called_on_non_tty(self) -> None:
        with _mock_tty(False), patch("kitty.tui.prompts.questionary") as mock_module, pytest.raises(NonTTYError):
            prompt_confirm("Continue?")
        mock_module.confirm.assert_not_called()

    def test_cancelled_returns_default(self) -> None:
        """When questionary returns None (cancelled), fall back to the default value."""
        mock_q = MagicMock()
        mock_q.ask.return_value = None
        with _mock_tty(True), patch("kitty.tui.prompts.questionary") as mock_module:
            mock_module.confirm.return_value = mock_q
            assert prompt_confirm("Continue?", default=True) is True
            assert prompt_confirm("Continue?", default=False) is False


# Each prompt paired with the widget it would draw with, as a patch target and the
# attribute on that target whose call means "drawing started".
_PROMPTS: list[tuple[str, Callable[[], object], str, str | None]] = [
    ("prompt_text", lambda: prompt_text("Name: "), "kitty.tui.prompts.questionary", "text"),
    ("prompt_secret", lambda: prompt_secret("API key: "), "kitty.tui.prompts.pt_prompt", None),
    ("prompt_confirm", lambda: prompt_confirm("Continue?"), "kitty.tui.prompts.questionary", "confirm"),
]


@pytest.mark.parametrize(("name", "call", "target", "widget"), _PROMPTS, ids=[p[0] for p in _PROMPTS])
def test_each_prompt_refuses_a_redirected_stdout_without_drawing(
    name: str, call: Callable[[], object], target: str, widget: str | None
) -> None:
    """KBR-204: no prompt starts drawing when its output is not a terminal.

    Raising is not enough on its own: a prompt that raised *after* building the
    widget would already have hit ``NoConsoleScreenBufferError`` on Windows. So the
    widget must never be touched.

    Args:
        name: The prompt under test, for the failure message.
        call: Invokes the prompt with a representative label.
        target: Patch target for the drawing dependency.
        widget: Attribute on ``target`` that builds the widget, or ``None`` when
            ``target`` is itself the callable.
    """
    with _mock_tty(True, stdout_is_tty=False), patch(target) as drawing, pytest.raises(NonTTYError):
        call()

    started = drawing if widget is None else getattr(drawing, widget)
    assert started.call_count == 0, f"{name} began drawing over a redirected stdout"


def _check_tty_refuses() -> bool:
    """Report whether :func:`kitty.tui.prompts.check_tty` refuses in the current patched state.

    Returns:
        ``True`` when ``check_tty`` raises :class:`~kitty.tui.prompts.NonTTYError`.
    """
    try:
        check_tty()
    except NonTTYError:
        return True
    return False


@pytest.mark.parametrize(
    ("stdin_is_tty", "stdout_is_tty"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_prompts_and_menus_give_one_answer_to_can_this_process_interact(
    stdin_is_tty: bool, stdout_is_tty: bool
) -> None:
    """KBR-204: the prompt guard and the menu guard cannot disagree.

    Until KBR-204 they were two predicates, and KBR-187 fixed only one — so
    ``kitty egress`` passed ``check_tty`` and then met a menu that silently
    declined, exiting 0 with no diagnosis. The claim is behavioural: across every
    combination of stream states, ``check_tty`` refuses exactly when both menus
    decline without drawing.

    Args:
        stdin_is_tty: Whether standard input is presented as a terminal.
        stdout_is_tty: Whether standard output is presented as a terminal.
    """
    with _mock_tty(stdin_is_tty, stdout_is_tty=stdout_is_tty), patch("kitty.tui.menu.questionary") as menus:
        refuses = _check_tty_refuses()
        select_declined = SelectionMenu("Pick", ["a"]).show() is None and menus.select.call_count == 0
        checkbox_declined = CheckboxMenu("Pick", ["a"]).show() is None and menus.checkbox.call_count == 0

    assert (refuses, select_declined, checkbox_declined) in {(True, True, True), (False, False, False)}, (
        f"stdin_is_tty={stdin_is_tty}, stdout_is_tty={stdout_is_tty}: check_tty refuses={refuses}, "
        f"SelectionMenu declined={select_declined}, CheckboxMenu declined={checkbox_declined}"
    )


# KBR-218 — the Windows console probe. The ctypes plumbing is Windows-only
# plumbing; the interpretation (`GetConsoleMode` returned nonzero ⇒ the handle
# names a real console) is the decision, extracted as the pure function
# `_handle_attached` and pinned here on every leg. The truth table pins the
# AND-composition `can_interact()` performs over the two handle probes.


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [(0, False), (1, True), (0x1F, True)],
    ids=["zero-refuses", "one-attaches", "nonzero-sentinel-attaches"],
)
def test_handle_attached_interprets_a_console_mode_return(raw_mode: int, expected: bool) -> None:
    """KBR-218: ``GetConsoleMode``'s return value, not its mode, decides.

    ``GetConsoleMode`` writes the console's mode flags into its out-parameter
    and returns a BOOL — nonzero when the handle names a real console, zero on
    any failure (not a console, no handle, a process started without one). The
    interpretation is ``bool(raw)`` and it is a pure function so it can be
    pinned on every leg with raw values; the ctypes call that produces the raw
    value is Windows-only plumbing (see ``.system_design/SYSTEM_DESIGN.md`` §7.4,
    the same shape ``tests/bridge/test_bridge_management.py`` uses for
    ``_probe_pid_windows``).

    Args:
        raw_mode: The raw DWORD ``GetConsoleMode`` returned.
        expected: What ``_handle_attached`` must report for it.
    """
    from kitty.tui import prompts

    assert prompts._handle_attached(raw_mode) is expected


@pytest.mark.parametrize(
    ("stdin_attached", "stdout_attached", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_can_interact_and_composes_the_console_probe(
    monkeypatch: pytest.MonkeyPatch,
    stdin_attached: bool,
    stdout_attached: bool,
    expected: bool,
) -> None:
    """KBR-218: on Windows the guard ANDs the probe over both standard handles.

    The ctypes call is Windows-only plumbing; the AND-composition is a
    decision, and it is pinned here on every leg by patching the plumbing
    seam. The first plumbing call returns the stdin raw and the second the
    stdout raw; ``_handle_attached`` stays real and interprets each one
    before the AND composes them. ``sys.platform`` is forced to ``win32`` so
    the Windows branch is reachable on every leg.

    This is the positive direction of the fix: every subprocess case exercises
    the refusal branch, so without this table a regression that makes
    ``can_interact()`` always False on ``win32`` (silently refusing every real
    console) ships green against the whole suite. The interpretation itself
    (``bool(raw_mode)``) is pinned separately by
    :func:`test_handle_attached_interprets_a_console_mode_return`.

    Args:
        monkeypatch: Pytest patcher, restored after the test.
        stdin_attached: What the probe reports for standard input.
        stdout_attached: What the probe reports for standard output.
        expected: What ``can_interact()`` must answer for that combination.
    """
    import sys as _sys

    from kitty.tui import prompts

    # 1 = console attached (raw GetConsoleMode return), 0 = not. The patched
    # plumbing skips ctypes entirely, which is the only way the Windows branch
    # of ``can_interact`` is reachable on a POSIX test runner.
    side_effect_iter = iter([1 if stdin_attached else 0, 1 if stdout_attached else 0])
    monkeypatch.setattr(prompts, "_query_console_mode", lambda _hv: next(side_effect_iter))
    monkeypatch.setattr(_sys, "platform", "win32")

    assert prompts.can_interact() is expected
