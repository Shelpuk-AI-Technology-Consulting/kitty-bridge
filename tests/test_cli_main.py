"""Tests for CLI main entry point — argument passthrough."""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid

import pytest


class TestCLIParseKnownArgs:
    """Verify that argparse uses parse_known_args so unknown flags pass through."""

    @staticmethod
    def _make_parser() -> argparse.ArgumentParser:
        """Recreate the parser from kitty.cli.main."""
        parser = argparse.ArgumentParser(
            prog="kitty",
            description="Kitty Code — launch coding agents through a local API bridge.",
        )
        parser.add_argument("--version", "-v", action="version", version="kitty test")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("command", nargs="*", help="Command to run")
        return parser

    def test_parse_args_rejects_unknown_flags(self) -> None:
        """parse_args() rejects unknown flags — this is the bug we're fixing."""
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["glm", "claude", "--dangerously-skip-permissions", "--resume", "foo"]
            )

    def test_parse_known_args_passes_unknown_flags_through(self) -> None:
        """parse_known_args() should leave unknown flags in the remainder."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args(
            ["glm", "claude", "--dangerously-skip-permissions", "--resume", "foo"]
        )
        assert args.command == ["glm", "claude"]
        assert unknown == ["--dangerously-skip-permissions", "--resume", "foo"]

    def test_parse_known_args_debug_flag_still_works(self) -> None:
        """Known flags like --debug should still be parsed correctly."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args(
            ["--debug", "claude", "--resume", "foo"]
        )
        assert args.debug is True
        assert args.command == ["claude"]
        assert unknown == ["--resume", "foo"]

    def test_parse_known_args_no_unknown(self) -> None:
        """When there are no unknown flags, remainder should be empty."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args(["claude"])
        assert args.command == ["claude"]
        assert unknown == []

    def test_parse_known_args_empty(self) -> None:
        """Empty args should work."""
        parser = self._make_parser()
        args, unknown = parser.parse_known_args([])
        assert args.command == []
        assert unknown == []


class TestCLIIntegrationPassthrough:
    """Integration test: verify kitty passes agent flags through without error."""

    def test_unknown_flags_not_rejected(self) -> None:
        """Running kitty with agent-specific flags should not produce an argparse error."""
        result = subprocess.run(
            [sys.executable, "-m", "kitty", "--version"],
            capture_output=True,
            text=True,
        )
        # --version exits with 0 and prints version
        assert result.returncode == 0
        assert "kitty" in result.stdout

    def test_main_with_agent_flags_does_not_crash(self) -> None:
        """Running kitty main() with agent CLI flags should not raise SystemExit for argparse errors.

        We patch the router and launch_target to avoid needing a real profile/credential setup.
        """
        from unittest.mock import MagicMock, patch

        from kitty.cli.router import BuiltinCommand, RouteResult
        from kitty.launchers.base import LauncherAdapter
        from kitty.profiles.schema import Profile

        profile = Profile(
            name="glm",
            provider="openrouter",
            model="claude-3",
            auth_ref=str(uuid.uuid4()),
            is_default=True,
        )
        mock_adapter = MagicMock(spec=LauncherAdapter)
        mock_result = RouteResult(
            adapter=mock_adapter,
            profile=profile,
            extra_args=["--dangerously-skip-permissions", "--resume", "tui-display-polish"],
        )

        with (
            patch("sys.argv", ["kitty", "glm", "claude", "--dangerously-skip-permissions", "--resume", "tui-display-polish"]),
            patch("kitty.cli.router.CLIRouter.route", return_value=mock_result),
            patch("kitty.cli.main._launch_target", return_value=0) as mock_launch,
        ):
            with pytest.raises(SystemExit) as exc_info:
                from kitty.cli.main import main

                main()

            # Should exit 0 from successful launch, NOT exit 2 from argparse error
            assert exc_info.value.code == 0
            # Verify extra_args were passed to launch_target
            launch_call = mock_launch.call_args
            assert launch_call[0][3] == ["--dangerously-skip-permissions", "--resume", "tui-display-polish"]
