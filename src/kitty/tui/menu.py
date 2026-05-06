"""TUI selection menus — arrow-key navigation and checkbox selection via questionary."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import questionary

__all__ = ["CheckboxMenu", "SelectionMenu"]


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
        if not sys.stdin.isatty():
            return None
        if not self._options:
            return None
        return questionary.select(
            self._title,
            choices=self._options,
        ).ask()


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
        if not sys.stdin.isatty():
            return None

        choices: list[Any] = [
            questionary.Choice(title=opt, value=opt, checked=opt in self._default_checked) for opt in self._options
        ]

        kwargs: dict[str, Any] = {"choices": choices}
        if self._validate is not None:
            kwargs["validate"] = self._validate

        return questionary.checkbox(self._title, **kwargs).ask()
