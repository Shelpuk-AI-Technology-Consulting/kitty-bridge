"""TUI prompt utilities — text input, secret input, confirmation."""

from __future__ import annotations

import getpass
import sys

__all__ = ["check_tty", "prompt_confirm", "prompt_secret", "prompt_text"]


class NonTTYError(Exception):
    """Raised when a prompt is attempted in a non-TTY environment."""


def check_tty() -> None:
    """Raise NonTTYError if stdin is not a TTY."""
    if not sys.stdin.isatty():
        raise NonTTYError("This command requires an interactive terminal (TTY)")


def prompt_text(label: str) -> str:
    """Prompt the user for text input.

    Args:
        label: The prompt label to display.

    Returns:
        The user's input string.
    """
    check_tty()
    return input(label)


def prompt_secret(label: str) -> str:
    """Prompt the user for secret input (masked).

    Args:
        label: The prompt label to display.

    Returns:
        The user's secret input string.
    """
    check_tty()
    return getpass.getpass(label)


def prompt_confirm(label: str, default: bool = True) -> bool:
    """Prompt the user for yes/no confirmation.

    Args:
        label: The question to display.
        default: Default value when user presses Enter without input.

    Returns:
        True for yes, False for no.
    """
    check_tty()
    hint = "[Y/n]" if default else "[y/N]"

    while True:
        response = input(f"{label} {hint} ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        # Invalid input — retry
