"""Bridge management — start, stop, restart, status."""

from __future__ import annotations

import contextlib
import enum
import ipaddress
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

from kitty.bridge.state import load_state, remove_state
from kitty.egress import ENV_PROXY, get_egress


class BridgeStatus(enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"  # State file exists but PID is dead
    UNMANAGEABLE = "unmanageable"  # Bridge is serving, but under another user


_DEFAULT_STATE_PATH = Path.home() / ".config" / "kitty" / "bridge_state.json"

PROBE_TIMEOUT_SECONDS = 0.5


class ProcessLiveness(enum.Enum):
    """Outcome of probing a PID.

    Attributes:
        ALIVE: The process exists and this user may signal it.
        DEAD: No process holds that PID.
        UNKNOWN: A process holds that PID but it is not ours to signal.
    """

    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


def probe_pid(pid: int) -> ProcessLiveness:
    """Probe whether a process exists and whether this user may signal it.

    Signal ``0`` performs no action and only probes for the process, on Windows
    as well as POSIX — it does not terminate the target. The three outcomes are
    kept apart because they call for different handling: a permission-denied
    probe proves the process is *running* under another account, which is the
    opposite conclusion from a missing process, yet both raise from
    :func:`os.kill`.

    Args:
        pid: Process ID to probe.

    Returns:
        :attr:`ProcessLiveness.ALIVE`, :attr:`ProcessLiveness.DEAD`, or
        :attr:`ProcessLiveness.UNKNOWN`.
    """
    # Reject non-positive PIDs before signalling. On POSIX a pid of 0 addresses
    # the caller's whole process group and -1 addresses every process the caller
    # may signal, so a corrupt bridge_state.json would turn stop_bridge() into a
    # SIGTERM/SIGKILL against the user's own shell session.
    if pid <= 0:
        return ProcessLiveness.DEAD
    try:
        os.kill(pid, 0)
        return ProcessLiveness.ALIVE
    except ProcessLookupError:
        return ProcessLiveness.DEAD
    except PermissionError:
        # EPERM (POSIX) / ERROR_ACCESS_DENIED (Windows) proves the process
        # exists — we simply may not signal it. Reporting it as dead is what
        # issue #3 is about; reporting it as alive would let stop_bridge()
        # signal a stranger's process after PID recycling.
        return ProcessLiveness.UNKNOWN
    except OSError:
        # Windows has no ProcessLookupError for this: OpenProcess fails with
        # ERROR_INVALID_PARAMETER (87) for a PID that does not exist, which
        # surfaces as a plain OSError. Without this branch every stale-state
        # code path (bridge status/stop/start/restart) crashes on Windows.
        return ProcessLiveness.DEAD


def _connect_target(host: str) -> str:
    """Map a bind address to an address this machine can connect to.

    A bridge bound to a wildcard records that address verbatim in
    ``bridge_state.json``, but a wildcard is a bind target, not a connect
    target: Windows rejects a connect to it with ``WSAEADDRNOTAVAIL`` (10049)
    even while the socket is listening. Such a bridge is always reachable from
    this machine over loopback.

    The check is by value rather than by spelling, because ``bridge.yaml``
    passes ``host`` through untouched and ``::``, ``::0`` and
    ``0:0:0:0:0:0:0:0`` all name the same unspecified address.

    Args:
        host: Host as recorded in the state file.

    Returns:
        The loopback address of the matching family for a wildcard host, and
        ``host`` unchanged for anything else — including hostnames.
    """
    if host == "":
        return "127.0.0.1"
    # A hostname is not an address literal and is probed exactly as recorded.
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if not address.is_unspecified:
        return host
    # ``is_unspecified`` is also true for the IPv4-mapped form ``::ffff:0.0.0.0``,
    # which wants an IPv4 loopback despite reporting version 6.
    if address.version == 6 and address.ipv4_mapped is None:
        return "::1"
    return "127.0.0.1"


def bridge_reachable(host: str, port: int, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    """Report whether something accepts TCP connections at ``host:port``.

    Used to settle :attr:`ProcessLiveness.UNKNOWN`: a process kitty may not
    signal is only *this* bridge if it is serving at the address the state file
    recorded. The probe connects and immediately closes; it does not speak HTTP,
    because ``/healthz`` sits behind the auth middleware whenever ``keys_file``
    is set and would additionally need TLS handling for a bridge started with
    ``--tls-cert``. Whether the connection is accepted answers the only question
    being asked.

    Args:
        host: Host recorded in ``bridge_state.json``.
        port: Port recorded in ``bridge_state.json``.
        timeout: Connect timeout in seconds. The probe runs on user-facing
            commands and never retries, so it stays short.

    Note:
        A wildcard bind address is probed over loopback instead — see
        ``_connect_target``.

    Returns:
        True if the connection was accepted, False for any failure — refused,
        timed out, or unresolvable.
    """
    target = _connect_target(host)
    # Every failure mode here (refused, timeout, DNS, unreachable network) is an
    # OSError subclass, and all of them mean the same thing to the caller.
    try:
        with socket.create_connection((target, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get_state_path() -> Path:
    return _DEFAULT_STATE_PATH


def bridge_status(state_path: Path | str | None = None) -> BridgeStatus:
    """Check the status of the bridge.

    Args:
        state_path: Path to ``bridge_state.json``; the default location is used
            when omitted.

    Returns:
        :attr:`BridgeStatus.STOPPED` when no state file exists,
        :attr:`BridgeStatus.RUNNING` for a bridge this user owns,
        :attr:`BridgeStatus.UNMANAGEABLE` for one running under another account,
        and :attr:`BridgeStatus.STALE` when the recorded process is gone.
    """
    state_path = Path(state_path) if state_path else _get_state_path()
    state = load_state(state_path)
    if state is None:
        return BridgeStatus.STOPPED
    liveness = probe_pid(state.pid)
    if liveness is ProcessLiveness.ALIVE:
        return BridgeStatus.RUNNING
    # A PID we may not signal is only *this* bridge if it is still serving at the
    # recorded address; otherwise the PID was recycled and the file is stale.
    if liveness is ProcessLiveness.UNKNOWN and bridge_reachable(state.host, state.port):
        return BridgeStatus.UNMANAGEABLE
    return BridgeStatus.STALE


def _health_monitor(
    state: dict,
    on_unhealthy: typing.Callable[[dict], typing.Any],
    *,
    interval: float = 30.0,
    timeout: float = 5.0,
    stop_event: threading.Event | None = None,
) -> None:
    """F41: Background health monitor that polls ``/healthz`` and invokes callback on failure.

    Runs in a thread (or thread-like context).  Polls the bridge's ``/healthz``
    endpoint at the given interval.  On a failed health-check (connection error,
    timeout, or non-2xx status), invokes ``on_unhealthy(state_dict)``.

    Args:
        state: dict with ``host``, ``port``, and optionally ``tls``.  Used to
            build the healthcheck URL.
        on_unhealthy: Callable ``(state) -> None`` invoked when health-checks fail.
        interval: Seconds between health-checks (default 30).
        timeout: HTTP request timeout (default 5).
        stop_event: Optional ``threading.Event`` to stop the monitor.
    """
    import threading
    import urllib.error
    import urllib.request

    event = stop_event or threading.Event()
    scheme = "https" if state.get("tls") else "http"
    url = f"{scheme}://{state['host']}:{state['port']}/healthz"
    consecutive_failures = 0
    while not event.is_set():
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ok = 200 <= resp.status < 300
            if ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            consecutive_failures += 1

        # Only invoke on_unhealthy after 2 consecutive failures to avoid
        # noise from a single transient network blip.
        if consecutive_failures >= 2:
            try:
                on_unhealthy(state)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Health monitor: on_unhealthy raised {exc}", file=sys.stderr)
            consecutive_failures = 0
        event.wait(timeout=interval)


def stop_bridge(state_path: Path | str | None = None) -> None:
    """Stop a running bridge instance.

    Sends SIGTERM, waits up to 10 seconds, then force-kills if needed (SIGKILL
    where available, SIGTERM on Windows where it does not exist), and removes
    the state file.

    Args:
        state_path: Path to ``bridge_state.json``; the default location is used
            when omitted.

    Raises:
        SystemExit: When the recorded process is running under another user
            account and is still serving, so kitty can neither stop it nor
            safely forget it.
    """
    state_path = Path(state_path) if state_path else _get_state_path()
    state = load_state(state_path)

    if state is None:
        return

    liveness = probe_pid(state.pid)

    # A PID we may not signal is never signalled: after PID recycling it would
    # belong to an unrelated process. If the bridge is still serving, say so and
    # keep the state file — deleting it would strand the process with nothing
    # pointing at it. If nothing answers, the PID was recycled and the file is
    # merely stale, which is exactly what this command exists to clear.
    if liveness is ProcessLiveness.UNKNOWN:
        if bridge_reachable(state.host, state.port):
            print(
                f"Error: Bridge (PID {state.pid}, {state.host}:{state.port}, "
                f"profile={state.profile}) is running under another user account. "
                f"kitty cannot stop it — stop it from that account, or as an "
                f"administrator. The state file was left in place; delete "
                f"{state_path} if you are sure that process is not a kitty bridge.",
                file=sys.stderr,
            )
            sys.exit(1)
        remove_state(state_path)
        return

    if liveness is ProcessLiveness.ALIVE:
        with contextlib.suppress(ProcessLookupError):
            os.kill(state.pid, signal.SIGTERM)

        # Wait up to 10 seconds for process to exit
        for _ in range(100):
            if probe_pid(state.pid) is not ProcessLiveness.ALIVE:
                break
            time.sleep(0.1)

        # Force kill if still alive. SIGKILL is POSIX-only; on Windows os.kill
        # terminates unconditionally for any signal other than the console
        # control events, so SIGTERM is the right fallback there. Without this,
        # the branch raised AttributeError and skipped remove_state() below,
        # leaving the stale state file this command exists to clear.
        if probe_pid(state.pid) is ProcessLiveness.ALIVE:
            force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            with contextlib.suppress(ProcessLookupError):
                os.kill(state.pid, force_signal)

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

    Args:
        state_path: Path to ``bridge_state.json``; the default location is used
            when omitted.
        config_path: Path to ``bridge.yaml`` for the spawned child.
        host: Bind address override for the child.
        port: Bind port override for the child.
        profile: Profile name override for the child.
        log_access: Whether the child writes an access log.
        tls_cert: Path to a TLS certificate for the child.
        tls_key: Path to the matching TLS private key.

    Raises:
        SystemExit: When a bridge is already running — including one owned by
            another user account that is still serving at the recorded address —
            or when the spawned child fails to come up.
    """
    state_path = Path(state_path) if state_path else _get_state_path()

    # F42: Acquire a lock on the state file to prevent concurrent start_bridge races.
    import filelock

    lock_path = str(state_path) + ".start.lock"
    try:
        start_lock = filelock.FileLock(lock_path, timeout=5)
        start_lock.acquire()
    except filelock.Timeout:
        print("Error: Another start_bridge is in progress. Try again in a few seconds.", file=sys.stderr)
        sys.exit(1)
    try:
        # Check for running instance
        state = load_state(state_path)
        if state is not None:
            liveness = probe_pid(state.pid)
            if liveness is ProcessLiveness.ALIVE:
                print(
                    f"Error: Bridge is already running (PID {state.pid}, "
                    f"{state.host}:{state.port}, profile={state.profile})",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Another user's bridge still serving at the recorded address:
            # starting a second one would leave the first orphaned but holding
            # its port. An unreachable address means the PID was recycled, so
            # the state file really is stale and the start proceeds below.
            if liveness is ProcessLiveness.UNKNOWN and bridge_reachable(state.host, state.port):
                print(
                    f"Error: Bridge is already running under another user account "
                    f"(PID {state.pid}, {state.host}:{state.port}, profile={state.profile}). "
                    f"kitty cannot manage it — stop it from that account, or delete "
                    f"{state_path} if you are sure that process is not a kitty bridge.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Clear stale state
        remove_state(state_path)

        # Build command to spawn
        # The egress gateway is handed over through the environment, never
        # argv: a proxy password on the command line is visible to `ps`.
        child_env = dict(os.environ)
        egress = get_egress()
        if egress is not None:
            child_env[ENV_PROXY] = egress.url_with_credentials()

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
            env=child_env,
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
    finally:
        start_lock.release()


def restart_bridge(
    *,
    state_path: Path | str | None = None,
    config_path: Path | str | None = None,
    **kwargs,
) -> None:
    """Restart the bridge. Re-reads bridge.yaml for new start.

    Args:
        state_path: Path to ``bridge_state.json``; the default location is used
            when omitted.
        config_path: Path to ``bridge.yaml``, re-read for the new instance.
        **kwargs: Forwarded to :func:`start_bridge`.

    Raises:
        SystemExit: Propagated from :func:`stop_bridge` or :func:`start_bridge`.
            A bridge running under another user account aborts the restart in
            the stop phase, so no second instance is started beside it.
    """
    state_path = Path(state_path) if state_path else _get_state_path()

    # Stop the old instance
    stop_bridge(state_path)

    # Start new instance (re-reads config from bridge.yaml)
    start_bridge(
        state_path=state_path,
        config_path=config_path,
        **kwargs,
    )
