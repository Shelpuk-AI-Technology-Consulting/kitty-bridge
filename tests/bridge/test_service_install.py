"""Tests for R9: System service installation."""

from __future__ import annotations

import sys

from kitty.bridge.service import generate_launchd_plist, generate_systemd_unit, generate_windows_script


class TestSystemdUnitGeneration:
    """Generate valid systemd user unit file."""

    def test_unit_contains_restart_on_failure(self):
        unit = generate_systemd_unit(
            executable=sys.executable,
            config_path="/home/user/.config/kitty/bridge.yaml",
        )
        assert "Restart=on-failure" in unit

    def test_unit_contains_executable(self):
        unit = generate_systemd_unit(
            executable="/usr/bin/kitty",
            config_path="/home/user/.config/kitty/bridge.yaml",
        )
        assert "ExecStart=/usr/bin/kitty" in unit
        assert "bridge" in unit

    def test_unit_contains_description(self):
        unit = generate_systemd_unit(
            executable=sys.executable,
            config_path="/home/user/.config/kitty/bridge.yaml",
        )
        assert "[Unit]" in unit
        assert "Description=" in unit

    def test_unit_is_valid_systemd(self):
        unit = generate_systemd_unit(
            executable="/usr/bin/kitty",
            config_path="/home/user/.config/kitty/bridge.yaml",
        )
        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "[Install]" in unit
        assert "WantedBy=default.target" in unit


class TestLaunchdPlistGeneration:
    """Generate valid launchd plist."""

    def test_plist_is_valid_xml(self):
        plist = generate_launchd_plist(
            executable="/usr/local/bin/kitty",
            config_path="/Users/user/.config/kitty/bridge.yaml",
        )
        assert "<?xml version" in plist
        assert "<plist" in plist
        assert "</plist>" in plist

    def test_plist_contains_label(self):
        plist = generate_launchd_plist(
            executable="/usr/local/bin/kitty",
            config_path="/Users/user/.config/kitty/bridge.yaml",
        )
        assert "com.kitty.bridge" in plist

    def test_plist_contains_program_arguments(self):
        plist = generate_launchd_plist(
            executable="/usr/local/bin/kitty",
            config_path="/Users/user/.config/kitty/bridge.yaml",
        )
        assert "kitty" in plist
        assert "bridge" in plist

    def test_plist_contains_run_at_load(self):
        plist = generate_launchd_plist(
            executable="/usr/local/bin/kitty",
            config_path="/Users/user/.config/kitty/bridge.yaml",
        )
        assert "RunAtLoad" in plist


class TestWindowsScriptGeneration:
    """Generate NSSM install script for Windows."""

    def test_script_contains_nssm_install(self):
        script = generate_windows_script(
            executable="C:\\Python\\kitty.exe",
            config_path="C:\\Users\\user\\.config\\kitty\\bridge.yaml",
        )
        assert "nssm" in script.lower()
        assert "install" in script.lower()

    def test_script_contains_executable_path(self):
        script = generate_windows_script(
            executable="C:\\Python\\kitty.exe",
            config_path="C:\\Users\\user\\.config\\kitty\\bridge.yaml",
        )
        assert "C:\\Python\\kitty.exe" in script

    def test_script_is_powershell(self):
        script = generate_windows_script(
            executable="C:\\Python\\kitty.exe",
            config_path="C:\\Users\\user\\.config\\kitty\\bridge.yaml",
        )
        assert script.startswith("#")  # Script comment header


class TestDryRun:
    """--dry-run prints to stdout without installing."""

    def test_systemd_dry_run_returns_content(self):
        unit = generate_systemd_unit(
            executable="/usr/bin/kitty",
            config_path="/home/user/.config/kitty/bridge.yaml",
        )
        # Dry run just returns the content, installation is separate
        assert isinstance(unit, str)
        assert len(unit) > 0
