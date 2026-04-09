"""Tests for launchers/discovery.py — Binary discovery via PATH and platform-specific fallbacks."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kitty.launchers.discovery import (
    _expand_nvm_dirs,
    _is_executable,
    discover_binary,
)


class TestDiscoverBinary:
    def test_returns_path_when_on_path(self):
        with patch("kitty.launchers.discovery.shutil.which", return_value="/usr/bin/codex"):
            result = discover_binary("codex")
            assert result == Path("/usr/bin/codex")

    def test_returns_none_when_not_found(self):
        with (
            patch("kitty.launchers.discovery.shutil.which", return_value=None),
            patch("kitty.launchers.discovery._COMMON_PATHS", {"codex": {"linux": []}}),
            patch("kitty.launchers.discovery.sys.platform", "linux"),
        ):
            result = discover_binary("codex")
            assert result is None

    def test_checks_platform_fallback_dirs(self, tmp_path):
        binary_dir = tmp_path / "bin"
        binary_dir.mkdir()
        binary_file = binary_dir / "claude"
        binary_file.write_text("#!/bin/sh\necho claude")
        binary_file.chmod(0o755)

        with (
            patch("kitty.launchers.discovery.shutil.which", return_value=None),
            patch(
                "kitty.launchers.discovery._COMMON_PATHS",
                {"claude": {"linux": [str(binary_dir)]}},
            ),
            patch("kitty.launchers.discovery.sys.platform", "linux"),
        ):
            result = discover_binary("claude")
            assert result == binary_file

    def test_skips_non_executable_files(self, tmp_path):
        binary_dir = tmp_path / "bin"
        binary_dir.mkdir()
        binary_file = binary_dir / "codex"
        binary_file.write_text("not executable")

        with (
            patch("kitty.launchers.discovery.shutil.which", return_value=None),
            patch(
                "kitty.launchers.discovery._COMMON_PATHS",
                {"codex": {"linux": [str(binary_dir)]}},
            ),
            patch("kitty.launchers.discovery.sys.platform", "linux"),
            patch("kitty.launchers.discovery._is_executable", return_value=False),
        ):
            result = discover_binary("codex")
            assert result is None

    def test_case_sensitive_name_lookup(self, tmp_path):
        """Verify that discover_binary uses the exact name passed for candidate path."""
        binary_dir = tmp_path / "bin"
        binary_dir.mkdir()
        # Create a file named "codex" (lowercase)
        (binary_dir / "codex").write_text("#!/bin/sh")

        # On Windows, filesystem is case-insensitive so we mock _is_executable
        # to simulate case-sensitive behavior (candidate "Codex" != file "codex")
        with (
            patch("kitty.launchers.discovery.shutil.which", return_value=None),
            patch(
                "kitty.launchers.discovery._COMMON_PATHS",
                {"Codex": {"linux": [str(binary_dir)]}},
            ),
            patch("kitty.launchers.discovery.sys.platform", "linux"),
            # Mock _is_executable to only return True for exact "codex" match
            patch(
                "kitty.launchers.discovery._is_executable",
                side_effect=lambda p: p.name == "codex",
            ),
        ):
            result = discover_binary("Codex")
            assert result is None

    def test_unsupported_platform_returns_none(self):
        with (
            patch("kitty.launchers.discovery.shutil.which", return_value=None),
            patch("kitty.launchers.discovery.sys.platform", "freebsd"),
        ):
            result = discover_binary("codex")
            assert result is None


class TestExpandNvmDirs:
    def test_returns_sorted_existing_dirs(self, tmp_path):
        nvm_base = tmp_path / ".nvm" / "versions" / "node"
        v18_bin = nvm_base / "v18.20.0" / "bin"
        v20_bin = nvm_base / "v20.0.0" / "bin"
        v18_bin.mkdir(parents=True)
        v20_bin.mkdir(parents=True)

        result = _expand_nvm_dirs(nvm_base)
        assert result == sorted([v18_bin, v20_bin])

    def test_returns_empty_when_no_dirs_exist(self, tmp_path):
        nvm_base = tmp_path / ".nvm" / "versions" / "node"
        result = _expand_nvm_dirs(nvm_base)
        assert result == []


class TestIsExecutable:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_executable_file_posix(self, tmp_path):
        f = tmp_path / "script"
        f.write_text("#!/bin/sh")
        f.chmod(0o755)
        assert _is_executable(f) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_non_executable_file_posix(self, tmp_path):
        f = tmp_path / "script"
        f.write_text("data")
        f.chmod(0o644)
        assert _is_executable(f) is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_executable_file_windows(self, tmp_path):
        f = tmp_path / "codex.exe"
        f.write_text("binary")
        assert _is_executable(f) is True

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_non_executable_ext_windows(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("text")
        assert _is_executable(f) is False

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "nonexistent"
        assert _is_executable(f) is False

    def test_directory_not_executable(self, tmp_path):
        d = tmp_path / "somedir"
        d.mkdir()
        assert _is_executable(d) is False
