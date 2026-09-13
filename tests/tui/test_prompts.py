"""Tests for TUI prompt utilities (questionary-based)."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
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
        A context manager patching both ``isatty`` calls.
    """
    out = is_tty if stdout_is_tty is None else stdout_is_tty

    @contextmanager
    def _both():
        """Hold both ``isatty`` patches for the duration of the block.

        Yields:
            None. The patched state applies inside the ``with`` block.
        """
        with patch("sys.stdin.isatty", return_value=is_tty), patch("sys.stdout.isatty", return_value=out):
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
