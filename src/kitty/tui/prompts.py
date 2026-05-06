"""TUI prompt utilities — text input, secret input, confirmation via questionary."""

from __future__ import annotations

import sys

import questionary
from prompt_toolkit import prompt as pt_prompt

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
        The user's input string (may be empty if cancelled; callers should validate).

    Raises:
        NonTTYError: If stdin is not a TTY.
    """
    check_tty()
    result = questionary.text(label).ask()
    return result if result is not None else ""


def prompt_secret(label: str) -> str:
    """Prompt the user for secret input (masked with asterisks).

    Shows one ``*`` per character so the user can see that input was accepted.

    Args:
        label: The prompt label to display.

    Returns:
        The user's secret input string.

    Raises:
        NonTTYError: If stdin is not a TTY.
    """
    check_tty()
    try:
        return pt_prompt(label, is_password=True)
    except (KeyboardInterrupt, EOFError):
        return ""


def prompt_confirm(label: str, default: bool = True) -> bool:
    """Prompt the user for yes/no confirmation.

    Args:
        label: The question to display.
        default: Default value when user presses Enter without input or cancels.

    Returns:
        True for yes, False for no.

    Raises:
        NonTTYError: If stdin is not a TTY.
    """
    check_tty()
    result = questionary.confirm(label, default=default).ask()
    # questionary returns None on Ctrl+C — fall back to the default
    return result if result is not None else default
