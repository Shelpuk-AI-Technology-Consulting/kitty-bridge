"""Tests for TUI display — status, error, warning, table output."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from kitty.tui.display import (
    clear_screen,
    print_banner,
    print_error,
    print_info,
    print_section,
    print_status,
    print_step,
    print_table,
    print_warning,
)


class TestPrintStatus:
    def test_prints_status_message(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_status("All good")
        output = buf.getvalue()
        assert "All good" in output

    def test_includes_indicator(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_status("Done")
        output = buf.getvalue()
        # Should have some kind of visual indicator
        assert len(output.strip()) > len("Done")


class TestPrintError:
    def test_prints_error_message(self) -> None:
        buf = StringIO()
        with patch("sys.stderr", buf):
            print_error("Something went wrong")
        output = buf.getvalue()
        assert "Something went wrong" in output


class TestPrintWarning:
    def test_prints_warning_message(self) -> None:
        buf = StringIO()
        with patch("sys.stderr", buf):
            print_warning("Be careful")
        output = buf.getvalue()
        assert "Be careful" in output


class TestPrintTable:
    def test_prints_headers_and_rows(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_table(
                headers=["Name", "Provider"],
                rows=[["dev", "zai_regular"], ["prod", "minimax"]],
            )
        output = buf.getvalue()
        assert "Name" in output
        assert "Provider" in output
        assert "dev" in output
        assert "prod" in output

    def test_empty_rows(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_table(headers=["Name"], rows=[])
        output = buf.getvalue()
        assert "Name" in output

    def test_handles_wide_columns(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_table(
                headers=["Short", "Very Long Column Name"],
                rows=[["a", "b"]],
            )
        output = buf.getvalue()
        assert "Very Long Column Name" in output


class TestPrintBanner:
    def test_prints_version(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_banner("0.1.1")
        output = buf.getvalue()
        assert "0.1.1" in output

    def test_prints_kitty_name(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_banner("0.1.1")
        output = buf.getvalue()
        assert "kitty" in output

    def test_prints_tagline(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_banner("0.1.1")
        output = buf.getvalue()
        assert "launch coding agents" in output


class TestPrintSection:
    def test_prints_title(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_section("Profile Management")
        output = buf.getvalue()
        assert "Profile Management" in output

    def test_contains_horizontal_rule(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_section("Test")
        output = buf.getvalue()
        assert "──" in output or "─" in output


class TestPrintStep:
    def test_prints_step_number(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_step(1, 6, "Provider selection")
        output = buf.getvalue()
        assert "Step 1" in output

    def test_prints_total(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_step(1, 6, "Provider selection")
        output = buf.getvalue()
        assert "of 6" in output

    def test_prints_label(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_step(1, 6, "Provider selection")
        output = buf.getvalue()
        assert "Provider selection" in output


class TestPrintInfo:
    def test_prints_message(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_info("Bridging to provider")
        output = buf.getvalue()
        assert "Bridging to provider" in output

    def test_includes_indicator(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_info("Test")
        output = buf.getvalue()
        assert len(output.strip()) > len("Test")


class TestClearScreen:
    def test_calls_console_clear(self) -> None:
        with patch("kitty.tui.display._console") as mock_console:
            clear_screen()
        mock_console.clear.assert_called_once()


class TestKittyTheme:
    def test_theme_defines_all_semantic_styles(self) -> None:
        from kitty.tui.display import KITTY_THEME
        for name in ("kitty.ok", "kitty.err", "kitty.warn", "kitty.info",
                      "kitty.title", "kitty.accent", "kitty.muted", "kitty.border"):
            assert name in KITTY_THEME.styles

    def test_consoles_use_theme(self) -> None:
        from kitty.tui.display import KITTY_THEME, _console, _stderr_console
        # Consoles constructed with theme= should resolve custom style names
        # Print via the console and verify the style is in the stack
        assert _console._theme_stack is not None
        assert _stderr_console._theme_stack is not None
        # The theme was passed at construction, so the stack should contain it
        assert KITTY_THEME.styles.get("kitty.ok") is not None

    def test_no_color_env_strips_ansi(self) -> None:
        import os
        buf = StringIO()
        with patch("sys.stdout", buf), patch.dict(os.environ, {"NO_COLOR": "1"}):
            from rich.console import Console

            from kitty.tui.display import KITTY_THEME
            c = Console(theme=KITTY_THEME, file=buf, no_color=True)
            c.print("[kitty.ok]✓[/kitty.ok] test")
        output = buf.getvalue()
        assert "\x1b[" not in output
        assert "✓" in output
        assert "test" in output

    def test_force_color_overrides_no_color(self) -> None:
        import os

        with patch.dict(os.environ, {"NO_COLOR": "1", "FORCE_COLOR": "1"}):
            # We need to re-import or re-create console because it's a singleton
            # For testing we can create a new one with the same logic we intend to use
            from kitty.tui.display import _should_enable_color
            color_enabled = _should_enable_color()
            assert color_enabled is True, "FORCE_COLOR should override NO_COLOR"


    def test_display_uses_no_inline_color_strings(self) -> None:
        import re
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent.parent / "src" / "kitty" / "tui" / "display.py"
        content = src.read_text()
        # Look for [red], [green], [blue], [yellow], [bold], [dim], [cyan]
        # but NOT inside theme definitions or kitty.* style references
        # Match inline color strings like [green], [red], [bold blue] etc.
        # Exclude lines that are part of the KITTY_THEME definition
        lines = content.split("\n")
        in_theme = False
        for line in lines:
            stripped = line.strip()
            if "KITTY_THEME" in stripped and "=" in stripped:
                in_theme = True
                continue
            if in_theme and stripped == "})":
                in_theme = False
                continue
            if in_theme:
                continue
            # Check for non-kitty color/style references
            assert not re.search(r'\[(red|green|blue|yellow|bold\s+cyan|bold\s+blue|bold|dim|cyan)\b', line), \
                f"Inline color found in display.py: {line.strip()}"


class TestPrintPanel:
    def test_prints_panel_with_title(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            from kitty.tui.display import print_panel
            print_panel("My Title", "Hello world")
        output = buf.getvalue()
        assert "My Title" in output
        assert "Hello world" in output

    def test_prints_panel_with_border(self) -> None:
        buf = StringIO()
        with patch("sys.stdout", buf):
            from kitty.tui.display import print_panel
            print_panel("Title", "Content")
        output = buf.getvalue()
        # Panel renders box-drawing characters for borders
        assert "│" in output or "─" in output


class TestStatusSpinner:
    def test_context_manager_calls_console_status(self) -> None:
        with patch("kitty.tui.display._console") as mock_console:
            mock_console.status.return_value.__enter__ = lambda s: None
            mock_console.status.return_value.__exit__ = lambda s, *a: None
            from kitty.tui.display import status_spinner
            with status_spinner("Loading..."):
                pass
        mock_console.status.assert_called_once()

    def test_status_message_contains_text(self) -> None:
        with patch("kitty.tui.display._console") as mock_console:
            mock_console.status.return_value.__enter__ = lambda s: None
            mock_console.status.return_value.__exit__ = lambda s, *a: None
            from kitty.tui.display import status_spinner
            with status_spinner("Validating key"):
                pass
        call_args = mock_console.status.call_args[0][0]
        assert "Validating key" in call_args
