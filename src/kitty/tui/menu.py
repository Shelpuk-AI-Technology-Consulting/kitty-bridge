"""TUI selection menus — arrow-key navigation and checkbox selection via questionary."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, cast

import questionary

__all__ = ["CheckboxMenu", "SelectionMenu"]


def _can_host_a_menu() -> bool:
    """Report whether this process can host an interactive menu.

    Both streams are checked, not just stdin, and the second one is the point:
    a menu must **read keys and draw**, so a console on one side is not enough.

    🔴 KBR-187. This guard read ``sys.stdin.isatty()`` alone, which is a POSIX
    reading of the question. On Windows ``isatty()`` is true for any *character
    device*, and that includes ``NUL`` -- so a child spawned with stdin
    redirected to ``NUL`` reported an interactive stdin, passed this guard, and
    died inside ``prompt_toolkit`` with ``NoConsoleScreenBufferError`` when it
    tried to build a Win32 screen buffer over a **piped stdout**. On POSIX the
    same child reads ``/dev/null`` as not-a-tty and returns here, which is why
    no Linux run ever saw it.

    That is the same user-facing failure as KBR-10 -- kitty crashing because its
    output is redirected -- reached by a different route.

    Returns:
        ``True`` when both standard input and standard output are terminals.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


class SelectionMenu:
    """Single-item arrow-key menu.

    Uses questionary.select() for inline rendering with arrow-key navigation.
    Returns None in non-interactive (non-TTY) environments or when cancelled.
    """

    def __init__(self, title: str, options: list[Any]) -> None:
        self._title = title
        self._options = options

    def show(self) -> str | None:
        """Display the menu and return the selected option.

        Returns:
            The selected option string, or None if cancelled or non-interactive.
        """
        if not _can_host_a_menu():
            return None
        if not self._options:
            return None
        # questionary ships no stubs; ask() returns the choice or None.
        return cast("str | None", questionary.select(
            self._title,
            choices=self._options,
        ).ask())


class CheckboxMenu:
    """Multi-item checkbox menu.

    Uses questionary.checkbox() for inline rendering with Space-to-toggle navigation.
    Returns None in non-interactive (non-TTY) environments or when cancelled.
    Supports pre-checked items and optional validation.
    """

    def __init__(
        self,
        title: str,
        options: list[str],
        default_checked: list[str] | None = None,
        validate: Callable[[list[str]], bool | str] | None = None,
    ) -> None:
        self._title = title
        self._options = options
        self._default_checked = set(default_checked or [])
        self._validate = validate

    def show(self) -> list[str] | None:
        """Display the checkbox menu and return the selected options.

        Returns:
            List of selected option strings (may be empty), or None if cancelled
            or non-interactive.
        """
        if not _can_host_a_menu():
            return None

        choices: list[Any] = [
            questionary.Choice(title=opt, value=opt, checked=opt in self._default_checked) for opt in self._options
        ]

        kwargs: dict[str, Any] = {"choices": choices}
        if self._validate is not None:
            kwargs["validate"] = self._validate

        return cast("list[str] | None", questionary.checkbox(self._title, **kwargs).ask())
