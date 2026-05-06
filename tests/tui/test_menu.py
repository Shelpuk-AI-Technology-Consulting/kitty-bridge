"""Tests for TUI selection and checkbox menus (questionary-based)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kitty.tui.menu import CheckboxMenu, SelectionMenu


def _mock_tty(is_tty: bool = True):
    return patch("sys.stdin.isatty", return_value=is_tty)


class TestSelectionMenu:
    """Tests for SelectionMenu.show() — questionary.select wrapper."""

    def test_returns_selected_value(self) -> None:
        """Returns the string answer from questionary.select().ask()."""
        mock_q = MagicMock()
        mock_q.ask.return_value = "alpha"
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.select.return_value = mock_q
            result = SelectionMenu("Pick", ["alpha", "beta"]).show()
        assert result == "alpha"

    def test_cancel_returns_none(self) -> None:
        """Returns None when ask() returns None (user pressed Ctrl+C or Escape)."""
        mock_q = MagicMock()
        mock_q.ask.return_value = None
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.select.return_value = mock_q
            result = SelectionMenu("Pick", ["a", "b"]).show()
        assert result is None

    def test_non_tty_returns_none_without_questionary_call(self) -> None:
        """Returns None immediately on non-TTY; questionary.select is never called."""
        with _mock_tty(False), patch("kitty.tui.menu.questionary") as mock_module:
            result = SelectionMenu("Pick", ["a", "b"]).show()
        assert result is None
        mock_module.select.assert_not_called()

    def test_empty_options_returns_none(self) -> None:
        """Returns None for empty option list without calling questionary."""
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            result = SelectionMenu("Pick", []).show()
        assert result is None
        mock_module.select.assert_not_called()

    def test_questionary_called_with_title_and_choices(self) -> None:
        """questionary.select is called with the menu title and options."""
        mock_q = MagicMock()
        mock_q.ask.return_value = "beta"
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.select.return_value = mock_q
            SelectionMenu("My Title", ["alpha", "beta"]).show()
        call_kwargs = mock_module.select.call_args
        assert call_kwargs is not None
        # message (title) passed as first positional or keyword arg
        args, kwargs = call_kwargs
        message = args[0] if args else kwargs.get("message", "")
        assert "My Title" in message
        # choices list present
        choices = kwargs.get("choices") or (args[1] if len(args) > 1 else None)
        assert choices is not None
        assert "alpha" in choices
        assert "beta" in choices


class TestCheckboxMenu:
    """Tests for CheckboxMenu.show() — questionary.checkbox wrapper."""

    def test_returns_selected_list(self) -> None:
        """Returns the list of checked values from questionary.checkbox().ask()."""
        mock_q = MagicMock()
        mock_q.ask.return_value = ["a", "c"]
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.checkbox.return_value = mock_q
            result = CheckboxMenu("Pick many", ["a", "b", "c"]).show()
        assert result == ["a", "c"]

    def test_cancel_returns_none(self) -> None:
        """Returns None when ask() returns None."""
        mock_q = MagicMock()
        mock_q.ask.return_value = None
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.checkbox.return_value = mock_q
            result = CheckboxMenu("Pick many", ["a", "b"]).show()
        assert result is None

    def test_non_tty_returns_none_without_questionary_call(self) -> None:
        """Returns None immediately on non-TTY; questionary.checkbox never called."""
        with _mock_tty(False), patch("kitty.tui.menu.questionary") as mock_module:
            result = CheckboxMenu("Pick many", ["a", "b"]).show()
        assert result is None
        mock_module.checkbox.assert_not_called()

    def test_empty_selection_returns_empty_list(self) -> None:
        """Returns [] when user confirms with nothing checked."""
        mock_q = MagicMock()
        mock_q.ask.return_value = []
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.checkbox.return_value = mock_q
            result = CheckboxMenu("Pick many", ["a", "b"]).show()
        assert result == []

    def test_preselected_choices_forwarded(self) -> None:
        """default_checked items are passed as pre-selected to questionary."""
        import questionary as real_questionary

        mock_q = MagicMock()
        mock_q.ask.return_value = ["b"]
        # Patch only questionary.checkbox (not the whole module) so Choice stays real.
        with _mock_tty(True), patch("kitty.tui.menu.questionary.checkbox", return_value=mock_q) as mock_cb:
            CheckboxMenu("Pick", ["a", "b", "c"], default_checked=["b"]).show()
        call_args = mock_cb.call_args
        assert call_args is not None
        _args, kwargs = call_args
        choices = kwargs.get("choices") or (_args[1] if len(_args) > 1 else None)
        assert choices is not None
        # "b" should be a real questionary.Choice with checked=True
        assert all(isinstance(c, real_questionary.Choice) for c in choices)
        checked_values = [c.value for c in choices if c.checked]
        assert "b" in checked_values

    def test_validate_forwarded_to_questionary(self) -> None:
        """validate kwarg is forwarded to questionary.checkbox."""
        mock_q = MagicMock()
        mock_q.ask.return_value = ["a", "b"]
        validator = lambda x: True  # noqa: E731
        with _mock_tty(True), patch("kitty.tui.menu.questionary") as mock_module:
            mock_module.checkbox.return_value = mock_q
            CheckboxMenu("Pick", ["a", "b"], validate=validator).show()
        _args, kwargs = mock_module.checkbox.call_args
        assert kwargs.get("validate") is validator
