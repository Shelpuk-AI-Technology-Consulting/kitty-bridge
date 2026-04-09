"""Tests for TUI prompts — text input, secret input, confirmation."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from kitty.tui.prompts import NonTTYError, prompt_confirm, prompt_secret, prompt_text


@contextmanager
def _mock_tty(is_tty: bool = True):
    """Context manager to mock TTY state."""
    with patch("sys.stdin.isatty", return_value=is_tty):
        yield


class TestPromptText:
    def test_returns_user_input(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="hello"):
            result = prompt_text("Enter name")
        assert result == "hello"

    def test_returns_empty_string(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value=""):
            result = prompt_text("Enter name")
        assert result == ""

    def test_passes_label_to_input(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="x") as mock_input:
            prompt_text("Your label: ")
        mock_input.assert_called_once_with("Your label: ")

    def test_non_tty_raises(self) -> None:
        with _mock_tty(is_tty=False), pytest.raises(NonTTYError):
            prompt_text("Enter name")


class TestPromptSecret:
    def test_returns_user_input(self) -> None:
        with _mock_tty(), patch("getpass.getpass", return_value="secret123"):
            result = prompt_secret("API key")
        assert result == "secret123"

    def test_passes_label_to_getpass(self) -> None:
        with _mock_tty(), patch("getpass.getpass", return_value="x") as mock_getpass:
            prompt_secret("Enter key: ")
        mock_getpass.assert_called_once_with("Enter key: ")

    def test_non_tty_raises(self) -> None:
        with _mock_tty(is_tty=False), pytest.raises(NonTTYError):
            prompt_secret("API key")


class TestPromptConfirm:
    def test_default_true_returns_true_on_empty(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value=""):
            result = prompt_confirm("Continue?", default=True)
        assert result is True

    def test_default_false_returns_false_on_empty(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value=""):
            result = prompt_confirm("Continue?", default=False)
        assert result is False

    def test_y_returns_true(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="y"):
            result = prompt_confirm("Continue?")
        assert result is True

    def test_yes_returns_true(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="yes"):
            result = prompt_confirm("Continue?")
        assert result is True

    def test_n_returns_false(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="n"):
            result = prompt_confirm("Continue?")
        assert result is False

    def test_no_returns_false(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="no"):
            result = prompt_confirm("Continue?")
        assert result is False

    def test_case_insensitive(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="Y"):
            result = prompt_confirm("Continue?")
        assert result is True

    def test_invalid_input_retries(self) -> None:
        with _mock_tty(), patch("builtins.input", side_effect=["maybe", "y"]):
            result = prompt_confirm("Continue?")
        assert result is True

    def test_prompt_includes_default_hint(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="") as mock_input:
            prompt_confirm("Continue?", default=True)
        call_args = mock_input.call_args[0][0]
        assert "Y/n" in call_args or "[Y]" in call_args

    def test_prompt_includes_default_hint_false(self) -> None:
        with _mock_tty(), patch("builtins.input", return_value="") as mock_input:
            prompt_confirm("Continue?", default=False)
        call_args = mock_input.call_args[0][0]
        assert "y/N" in call_args or "[n]" in call_args

    def test_non_tty_raises(self) -> None:
        with _mock_tty(is_tty=False), pytest.raises(NonTTYError):
            prompt_confirm("Continue?")
