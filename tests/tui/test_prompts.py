"""Tests for TUI prompt utilities (questionary-based)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kitty.tui.prompts import NonTTYError, check_tty, prompt_confirm, prompt_secret, prompt_text


def _mock_tty(is_tty: bool = True):
    return patch("sys.stdin.isatty", return_value=is_tty)


class TestCheckTty:
    def test_raises_on_non_tty(self) -> None:
        with _mock_tty(False), pytest.raises(NonTTYError, match="interactive"):
            check_tty()

    def test_passes_on_tty(self) -> None:
        with _mock_tty(True):
            check_tty()  # should not raise


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
