"""Tests for TUI selection menu."""

from __future__ import annotations

from unittest.mock import patch

from kitty.tui.menu import SelectionMenu


def _mock_tty(is_tty: bool = True):
    return patch("sys.stdin.isatty", return_value=is_tty)


class TestSelectionMenu:
    """Tests for SelectionMenu.show()."""

    def test_cancel_returns_none(self) -> None:
        """Pressing Enter/empty input returns None (cancel)."""
        with _mock_tty(True), patch("builtins.input", return_value=""):
            menu = SelectionMenu("Pick", ["a", "b"])
            result = menu.show()
        assert result is None

    def test_q_returns_none(self) -> None:
        """Pressing 'q' returns None (cancel)."""
        with _mock_tty(True), patch("builtins.input", return_value="q"):
            menu = SelectionMenu("Pick", ["a", "b"])
            result = menu.show()
        assert result is None

    def test_select_first_option(self) -> None:
        """Selecting '1' returns the first option."""
        with _mock_tty(True), patch("builtins.input", return_value="1"):
            menu = SelectionMenu("Pick", ["alpha", "beta"])
            result = menu.show()
        assert result == "alpha"

    def test_select_second_option(self) -> None:
        """Selecting '2' returns the second option."""
        with _mock_tty(True), patch("builtins.input", return_value="2"):
            menu = SelectionMenu("Pick", ["alpha", "beta"])
            result = menu.show()
        assert result == "beta"

    def test_non_tty_returns_none(self) -> None:
        """Non-interactive terminal returns None immediately."""
        with _mock_tty(False):
            menu = SelectionMenu("Pick", ["a", "b"])
            result = menu.show()
        assert result is None

    def test_empty_options_returns_none(self) -> None:
        """Empty option list returns None."""
        with _mock_tty(True):
            menu = SelectionMenu("Pick", [])
            result = menu.show()
        assert result is None

    def test_eof_returns_none(self) -> None:
        """EOFError during input returns None."""
        with _mock_tty(True), patch("builtins.input", side_effect=EOFError):
            menu = SelectionMenu("Pick", ["a", "b"])
            result = menu.show()
        assert result is None

    def test_keyboard_interrupt_returns_none(self) -> None:
        """KeyboardInterrupt during input returns None."""
        with _mock_tty(True), patch("builtins.input", side_effect=KeyboardInterrupt):
            menu = SelectionMenu("Pick", ["a", "b"])
            result = menu.show()
        assert result is None

    def test_invalid_number_retries(self, capsys: object) -> None:
        """Invalid numeric input retries until a valid choice is made."""
        with _mock_tty(True), patch("builtins.input", side_effect=["99", "0", "-1", "1"]):
            menu = SelectionMenu("Pick", ["a", "b"])
            result = menu.show()
        assert result == "a"

    def test_output_contains_title(self, capsys: object) -> None:
        """Menu output includes the title text."""
        with _mock_tty(True), patch("builtins.input", return_value="q"):
            menu = SelectionMenu("Pick a provider", ["novita"])
            menu.show()
        captured = capsys.readouterr()
        assert "Pick a provider" in captured.out

    def test_output_contains_option_text(self, capsys: object) -> None:
        """Menu output includes the option text."""
        with _mock_tty(True), patch("builtins.input", return_value="q"):
            menu = SelectionMenu("Pick", ["novita", "minimax"])
            menu.show()
        captured = capsys.readouterr()
        assert "novita" in captured.out
        assert "minimax" in captured.out
