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
