"""TUI display utilities — status, error, warning, table output."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

__all__ = [
    "clear_screen",
    "print_banner",
    "print_error",
    "print_info",
    "print_section",
    "print_status",
    "print_step",
    "print_table",
    "print_warning",
]

_console = Console()
_stderr_console = Console(stderr=True)


def clear_screen() -> None:
    """Clear the terminal screen."""
    _console.clear()


def print_banner(version: str) -> None:
    """Print the kitty startup banner with version and tagline."""
    _console.print(
        f"[bold blue]kitty[/bold blue] [dim]v{version}[/dim] — launch coding agents through a local API bridge"
    )


def print_section(title: str) -> None:
    """Print a visual section divider with a styled title."""
    _console.rule(f"[bold]{title}[/bold]", style="blue")


def print_step(step: int, total: int, label: str) -> None:
    """Print a wizard step indicator."""
    _console.rule(f"[bold cyan]Step {step} of {total}:[/bold cyan] {label}", style="cyan")


def print_info(message: str) -> None:
    """Print a neutral informational message to stdout."""
    _console.print(f"[blue]ℹ[/blue] {message}")


def print_status(message: str) -> None:
    """Print a formatted status message to stdout."""
    _console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print a formatted error message to stderr."""
    _stderr_console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a formatted warning message to stderr."""
    _stderr_console.print(f"[yellow]![/yellow] {message}")


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
