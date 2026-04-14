"""TUI display utilities — status, error, warning, table output."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

__all__ = [
    "CheckItem",
    "LiveChecklist",
    "clear_screen",
    "print_banner",
    "print_error",
    "print_info",
    "print_panel",
    "print_section",
    "print_status",
    "print_step",
    "print_table",
    "print_warning",
    "status_spinner",
]

KITTY_THEME = Theme({
    "kitty.ok": "green",
    "kitty.err": "red",
    "kitty.warn": "yellow",
    "kitty.info": "blue",
    "kitty.title": "bold",
    "kitty.accent": "cyan bold",
    "kitty.muted": "dim",
    "kitty.border": "blue",
})


def _should_enable_color() -> bool:
    """Determine whether color output should be enabled.

    Per the NO_COLOR spec (no-color.org), FORCE_COLOR takes precedence over NO_COLOR.
    """
    if os.environ.get("FORCE_COLOR"):
        return True
    return os.environ.get("NO_COLOR") is None


_console = Console(theme=KITTY_THEME, no_color=not _should_enable_color())
_stderr_console = Console(stderr=True, theme=KITTY_THEME, no_color=not _should_enable_color())


def clear_screen() -> None:
    """Clear the terminal screen."""
    _console.clear()


def print_banner(version: str) -> None:
    """Print the kitty startup banner with version and tagline."""
    _console.print(
        f"[kitty.accent]kitty[/kitty.accent] [kitty.muted]v{version}[/kitty.muted]"
        " — launch coding agents through a local API bridge"
    )


def print_section(title: str) -> None:
    """Print a visual section divider with a styled title."""
    _console.rule(f"[kitty.title]{title}[/kitty.title]", style="kitty.border")


def print_step(step: int, total: int, label: str) -> None:
    """Print a wizard step indicator."""
    _console.rule(f"[kitty.title]Step {step} of {total}:[/kitty.title] {label}", style="kitty.accent")


def print_info(message: str) -> None:
    """Print a neutral informational message to stdout."""
    _console.print(f"[kitty.info]ℹ[/kitty.info] {message}")


def print_status(message: str) -> None:
    """Print a formatted status message to stdout."""
    _console.print(f"[kitty.ok]✓[/kitty.ok] {message}")


def print_error(message: str) -> None:
    """Print a formatted error message to stderr."""
    _stderr_console.print(f"[kitty.err]✗[/kitty.err] {message}")


def print_warning(message: str) -> None:
    """Print a formatted warning message to stderr."""
    _stderr_console.print(f"[kitty.warn]![/kitty.warn] {message}")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted table to stdout.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.
    """
    table = Table()
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    _console.print(table)


def print_panel(title: str, content: str, border_style: str = "kitty.ok") -> None:
    """Print a bordered panel with title and content.

    Args:
        title: Panel title.
        content: Panel body text (may contain Rich markup).
        border_style: Rich style name for the border.
    """
    _console.print(Panel.fit(content, title=title, border_style=border_style))


@contextmanager
def status_spinner(message: str) -> Generator[None]:
    """Context manager showing a spinner while work is in progress.

    Args:
        message: Text displayed next to the spinner.
    """
    with _console.status(f"[kitty.accent]{message}"):
        yield


class CheckItem:
    """A single check row for LiveChecklist."""

    __slots__ = ("label", "ok", "detail")

    def __init__(self, label: str) -> None:
        self.label = label
        self.ok: bool | None = None
        self.detail: str = ""

    @property
    def status_text(self) -> str:
        if self.ok is True:
            return "[kitty.ok]✓[/kitty.ok]"
        if self.ok is False:
            return "[kitty.err]✗[/kitty.err]"
        return "[kitty.muted]⠿[/kitty.muted]"


class LiveChecklist:
    """Live-updating checklist display for doctor-style diagnostics.

    In a TTY, renders an in-place table that updates as checks complete.
    In a non-TTY, falls back to sequential printing.
    """

    def __init__(self, title: str) -> None:
        self._title = title
        self._items: list[CheckItem] = []
        self._is_tty = sys.stdout.isatty()

    def add(self, label: str) -> CheckItem:
        """Add a pending check item and return it for later resolution."""
        item = CheckItem(label)
        self._items.append(item)
        if not self._is_tty:
            _console.print(f"[kitty.muted]⠿[/kitty.muted] {label}")
        return item

    def resolve(self, item: CheckItem, ok: bool, detail: str = "") -> None:
        """Mark a check item as resolved."""
        item.ok = ok
        item.detail = detail
        if not self._is_tty:
            if ok:
                print_status(f"{item.label}: {detail}" if detail else item.label)
            else:
                print_error(f"{item.label}: {detail}" if detail else item.label)

    def run(self) -> int:
        """Run all checks and return failure count.

        Each check should be added via ``add()`` before calling this,
        or use the higher-level ``run_checks`` helper.
        """
        failures = sum(1 for item in self._items if item.ok is False)
        return failures

    def run_checks(
        self,
        checks: list[tuple[str, Callable[[], tuple[bool, str]]]],
    ) -> int:
        """Run a list of checks, updating the display as each completes.

        Args:
            checks: List of (label, check_fn) tuples. Each check_fn returns
                    (ok: bool, detail: str).

        Returns:
            Number of failed checks.
        """
        print_section(self._title)

        if self._is_tty:
            table = Table.grid(padding=(0, 2))
            table.add_column()
            table.add_column()
            table.add_column()

            items: list[CheckItem] = []
            for label, _ in checks:
                item = CheckItem(label)
                items.append(item)
                table.add_row(item.status_text, label, "")

            with Live(table, console=_console, refresh_per_second=4):
                for item, (_, check_fn) in zip(items, checks, strict=True):
                    try:
                        ok, detail = check_fn()
                    except Exception as exc:
                        ok, detail = False, str(exc)
                    item.ok = ok
                    item.detail = detail
                    table.rows[items.index(item)].cells = (
                        item.status_text,
                        item.label,
                        detail,
                    )
        else:
            items: list[CheckItem] = []
            for label, check_fn in checks:
                item = CheckItem(label)
                items.append(item)
                _console.print(f"[kitty.muted]⠿[/kitty.muted] {label}")
                try:
                    ok, detail = check_fn()
                except Exception as exc:
                    ok, detail = False, str(exc)
                item.ok = ok
                item.detail = detail
                if ok:
                    print_status(f"{label}: {detail}" if detail else label)
                else:
                    print_error(f"{label}: {detail}" if detail else label)

        return sum(1 for item in items if item.ok is False)
