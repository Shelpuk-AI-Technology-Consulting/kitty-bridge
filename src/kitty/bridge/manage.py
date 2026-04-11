"""Bridge management — start, stop, restart, status."""

from __future__ import annotations

import enum
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from kitty.bridge.state import BridgeState, load_state, remove_state, write_state


class BridgeStatus(enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"  # State file exists but PID is dead


_DEFAULT_STATE_PATH = Path.home() / ".config" / "kitty" / "bridge_state.json"


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _get_state_path() -> Path:
    return _DEFAULT_STATE_PATH


def bridge_status(state_path: Path | str | None = None) -> BridgeStatus:
    """Check the status of the bridge."""
    state_path = Path(state_path) if state_path else _get_state_path()
    state = load_state(state_path)
    if state is None:
        return BridgeStatus.STOPPED
    if is_pid_alive(state.pid):
        return BridgeStatus.RUNNING
    return BridgeStatus.STALE


def stop_bridge(state_path: Path | str | None = None) -> None:
    """Stop a running bridge instance.

    Sends SIGTERM, waits up to 10 seconds, then SIGKILL if needed.
    Always removes the state file.
    """
    state_path = Path(state_path) if state_path else _get_state_path()
    state = load_state(state_path)

    if state is None:
        return

    if is_pid_alive(state.pid):
        try:
            os.kill(state.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        # Wait up to 10 seconds for process to exit
        for _ in range(100):
            if not is_pid_alive(state.pid):
                break
            time.sleep(0.1)

        # Force kill if still alive
        if is_pid_alive(state.pid):
            try:
                os.kill(state.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    remove_state(state_path)


def start_bridge(
    *,
    state_path: Path | str | None = None,
    config_path: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    profile: str | None = None,
    log_access: bool | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> None:
    """Start the bridge in the background.

    Checks for running instances, clears stale state, spawns background process.
    """
    state_path = Path(state_path) if state_path else _get_state_path()

    # Check for running instance
    state = load_state(state_path)
    if state is not None and is_pid_alive(state.pid):
        print(
            f"Error: Bridge is already running (PID {state.pid}, "
            f"{state.host}:{state.port}, profile={state.profile})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Clear stale state
    remove_state(state_path)

    # Build command to spawn
    cmd = [sys.executable, "-m", "kitty.bridge_runner"]
    if host:
        cmd.extend(["--host", host])
    if port is not None:
        cmd.extend(["--port", str(port)])
    if profile:
        cmd.extend(["--profile", profile])
    if config_path:
        cmd.extend(["--config", str(config_path)])
    if log_access is True:
        cmd.append("--log")
    elif log_access is False:
        cmd.append("--no-log")
    if tls_cert:
        cmd.extend(["--tls-cert", tls_cert])
    if tls_key:
        cmd.extend(["--tls-key", tls_key])

    # Spawn background process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Wait briefly for the process to start and write state
    for _ in range(50):
        if state_path.exists():
            break
        # Check if process already exited (startup error)
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    state = load_state(state_path)
    if state is not None:
        scheme = "https" if state.tls else "http"
        print(f"{scheme}://{state.host}:{state.port}")
    else:
        # Process may have failed to start
        if proc.poll() is not None:
            # Process already exited — read error
            print(f"Error: Bridge failed to start (exit code {proc.returncode})", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr.read().decode(), file=sys.stderr)
        else:
            # Process is running but state file never appeared
            print("Error: Bridge started but state file not found", file=sys.stderr)
        sys.exit(1)


def restart_bridge(
    *,
    state_path: Path | str | None = None,
    config_path: Path | str | None = None,
    **kwargs,
) -> None:
    """Restart the bridge. Re-reads bridge.yaml for new start."""
    state_path = Path(state_path) if state_path else _get_state_path()

    # Stop the old instance
    stop_bridge(state_path)

    # Start new instance (re-reads config from bridge.yaml)
    start_bridge(
        state_path=state_path,
        config_path=config_path,
        **kwargs,
    )
