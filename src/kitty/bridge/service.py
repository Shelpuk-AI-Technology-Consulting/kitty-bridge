"""System service file generation for kitty bridge."""

from __future__ import annotations

import sys
from pathlib import Path

_DEFAULT_CONFIG = str(Path.home() / ".config" / "kitty" / "bridge.yaml")


def _resolve_executable() -> str:
    """Resolve the kitty executable path."""
    return sys.executable


def generate_systemd_unit(
    executable: str | None = None,
    config_path: str | None = None,
) -> str:
    """Generate a systemd user unit file for kitty bridge.

    Args:
        executable: Path to kitty executable. Defaults to current Python.
        config_path: Path to bridge.yaml.

    Returns:
        systemd unit file content as string.
    """
    executable = executable or _resolve_executable()
    config_path = config_path or _DEFAULT_CONFIG
    return f"""[Unit]
Description=Kitty Bridge API Gateway
After=network.target

[Service]
Type=simple
ExecStart={executable} -m kitty.bridge_runner --config {config_path}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def generate_launchd_plist(
    executable: str | None = None,
    config_path: str | None = None,
) -> str:
    """Generate a launchd plist for kitty bridge on macOS.

    Args:
        executable: Path to kitty executable.
        config_path: Path to bridge.yaml.

    Returns:
        plist XML content as string.
    """
    executable = executable or _resolve_executable()
    config_path = config_path or _DEFAULT_CONFIG
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.kitty.bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>{executable}</string>
        <string>-m</string>
        <string>kitty.bridge_runner</string>
        <string>--config</string>
        <string>{config_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/kitty-bridge.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/kitty-bridge.err</string>
</dict>
</plist>
"""


def generate_windows_script(
    executable: str | None = None,
    config_path: str | None = None,
) -> str:
    """Generate a PowerShell script for installing kitty bridge as a Windows service via NSSM.

    Args:
        executable: Path to kitty executable.
        config_path: Path to bridge.yaml.

    Returns:
        PowerShell script content as string.
    """
    executable = executable or _resolve_executable()
    config_path = config_path or _DEFAULT_CONFIG
    return f"""# Kitty Bridge Windows Service Installation Script
# This script uses NSSM (Non-Sucking Service Manager) to install kitty bridge as a Windows service.
# Install NSSM first: https://nssm.cc/download
#
# Review this script before running it.

$serviceName = "KittyBridge"
$executable = "{executable}"
$config = "{config_path}"

# Check if NSSM is available
nssm version 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {{
    Write-Error "NSSM not found. Install it from https://nssm.cc/download"
    exit 1
}}

# Install the service
nssm install $serviceName $executable "-m" "kitty.bridge_runner" "--config" $config

# Set description
nssm set $serviceName Description "Kitty Bridge API Gateway"

# Configure auto-restart
nssm set $serviceName AppRestartDelay 5000

# Start the service
nssm start $serviceName

Write-Host "Kitty Bridge service installed and started."
Write-Host "Manage with: nssm start/stop/restart $serviceName"
Write-Host "Remove with: nssm remove $serviceName confirm"
"""
