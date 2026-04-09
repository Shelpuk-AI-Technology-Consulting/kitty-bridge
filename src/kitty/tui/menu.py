"""TUI selection menu — numbered list with keyboard input."""

from __future__ import annotations

import sys

from rich.console import Console

__all__ = ["SelectionMenu"]

_console = Console()


class SelectionMenu:
    """Numbered selection menu for interactive CLI.

    Displays options as a numbered list and reads the user's choice.
    Returns None in non-interactive (non-TTY) environments.
    """

    def __init__(self, title: str, options: list[str]) -> None:
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

        while True:
            _console.print(f"\n[bold]{self._title}[/bold]")
            for i, option in enumerate(self._options, 1):
                _console.print(f"  [cyan]{i}.[/cyan] {option}")
            _console.print("  [dim]q. Cancel[/dim]")

            try:
                choice = input(f"Select [1-{len(self._options)}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None

            if choice.lower() == "q" or choice == "":
                return None

            try:
                index = int(choice)
                if 1 <= index <= len(self._options):
                    return self._options[index - 1]
            except ValueError:
                pass

            # Invalid choice — loop and retry
