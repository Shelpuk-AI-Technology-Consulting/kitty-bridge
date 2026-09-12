"""TUI prompt utilities — text input, secret input, confirmation via questionary."""

from __future__ import annotations

import sys

import questionary
from prompt_toolkit import prompt as pt_prompt

__all__ = ["can_interact", "check_tty", "prompt_confirm", "prompt_secret", "prompt_text"]


class NonTTYError(Exception):
    """Raised when a prompt is attempted in a non-TTY environment."""


def can_interact() -> bool:
    """Report whether this process can host an interactive prompt or menu.

    Both streams are checked, not just stdin, and the second one is the point:
    an interactive UI must **read keys and draw**, so a console on one side is
    not enough.

    🔴 KBR-187, then KBR-204. Reading ``sys.stdin.isatty()`` alone is a POSIX
    reading of the question. On Windows ``isatty()`` is true for any *character
    device*, and that includes ``NUL`` -- so a child spawned with stdin
    redirected to ``NUL`` reported an interactive stdin, and died inside
    ``prompt_toolkit`` with ``NoConsoleScreenBufferError`` when it tried to build
    a Win32 screen buffer over a **piped stdout**. KBR-187 fixed the menus'
    copy of this guard; KBR-204 found :func:`check_tty` still held the old one,
    so ``kitty egress`` passed it and exited 0 when its menu declined. On POSIX
    the same child reads ``/dev/null`` as not-a-tty, which is why no Linux run
    ever saw either -- and both are KBR-10's symptom, kitty misbehaving because
    its output is redirected, reached by a different route.

    ⚠️ This is the **only** answer to the question. :func:`check_tty` and
    :mod:`kitty.tui.menu` both consult it, so the two paths cannot drift again.

    Returns:
        ``True`` when both standard input and standard output are terminals.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def check_tty() -> None:
    """Refuse to continue unless this process can interact.

    Raises:
        NonTTYError: If :func:`can_interact` is false -- stdin or stdout is not
            a terminal.
    """
    if not can_interact():
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
