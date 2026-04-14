"""Launch orchestrator — wires bridge server + adapter + child process."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

from kitty.bridge.server import BridgeServer
from kitty.credentials.store import CredentialNotFoundError, CredentialStore
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.launchers.discovery import discover_binary
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.validation import validate_api_key

__all__ = ["launch", "launch_async", "map_child_exit_code", "build_child_env", "resolve_binary"]

logger = logging.getLogger(__name__)

# Module-level state for atexit cleanup.
# Stores (adapter, original_settings_content, settings_path) after prepare_launch.
# Cleared after successful cleanup to prevent double-restore.
_atexit_cleanup_state: list[tuple[LauncherAdapter, str, Path]] = []
_atexit_registered = False


def _atexit_cleanup() -> None:
    """Restore agent settings files that were patched by prepare_launch.

    Registered via atexit so cleanup runs even on unhandled exceptions,
    sys.exit(), or SIGTERM (which triggers normal Python shutdown).
    SIGKILL cannot be caught — use `kitty cleanup` for that.
    """
    for adapter, original, settings_path in _atexit_cleanup_state:
        try:
            adapter.cleanup_launch(original, settings_path=settings_path)
            logger.info("atexit cleanup: restored %s", settings_path)
        except Exception as exc:
            msg = f"kitty: failed to restore {settings_path}: {exc}"
            logger.warning("atexit cleanup: %s", msg)
            print(msg, file=sys.stderr)
    _atexit_cleanup_state.clear()


def _register_atexit_cleanup(
    adapter: LauncherAdapter,
    original: str | None,
    settings_path: Path,
) -> None:
    """Register atexit cleanup if not already registered and store state."""
    global _atexit_registered
    if original is None:
        return
    _atexit_cleanup_state.append((adapter, original, settings_path))
    if not _atexit_registered:
        atexit.register(_atexit_cleanup)
        _atexit_registered = True


def _clear_atexit_cleanup() -> None:
    """Clear atexit state after successful cleanup in the finally block."""
    _atexit_cleanup_state.clear()


def map_child_exit_code(code: int) -> int:
    """Map a child process exit code to the kitty process exit code.

    Rules:
    - Positive exit codes (0-255): pass through unchanged
    - Negative exit codes (signal death on CPython): map to 128 + signal_number, capped at 255
    """
    if code < 0:
        return min(128 + abs(code), 255)
    return code


def build_child_env(spawn_config: SpawnConfig) -> dict[str, str]:
    """Build the child process environment from SpawnConfig semantics.

    Order: copy parent → clear → override.
    """
    env = os.environ.copy()
    for key in spawn_config.env_clear:
        env.pop(key, None)
    env.update(spawn_config.env_overrides)
    return env


def resolve_binary(name: str) -> Path:
    """Resolve a binary path, exiting with a user-friendly error if not found."""
    path = discover_binary(name)
    if path is None:
        logger.error("Binary %r not found on PATH or common install directories", name)
        print(
            f"Error: '{name}' not found. Install it first or check your PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return path


async def launch_async(
    adapter: LauncherAdapter,
    provider: ProviderAdapter,
    profile: Profile,
    cred_store: CredentialStore,
    extra_args: list[str] | None = None,
    *,
    debug: bool = False,
    validate: bool = True,
    backends: list[tuple[ProviderAdapter, str, Profile]] | None = None,
    logging_enabled: bool = False,
) -> int:
    """Launch the full bridge + child process lifecycle.

    Steps:
    1. Resolve API key from credential store
    2. Validate API key (pre-flight check)
    3. Start the bridge server
    4. Build spawn config from adapter
    5. Discover the child binary
    6. Patch agent-specific external config
    7. Spawn child process with signal forwarding
    8. Wait for child to exit
    9. Stop the bridge server
    10. Return mapped exit code
    """
    extra_args = extra_args or []

    # 1. Resolve credential
    try:
        resolved_key = cred_store.resolve(profile)
    except CredentialNotFoundError as exc:
        logger.error("Credential resolution failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # 2. Pre-flight API key validation
    if validate:
        result = await validate_api_key(provider, resolved_key, profile.provider_config)
        if not result.valid:
            print(f"Error: {result.reason}", file=sys.stderr)
            print("Hint: Update your API key with 'kitty setup' or check the provider dashboard.", file=sys.stderr)
            return 1
        if result.warning:
            print(f"Warning: {result.warning}", file=sys.stderr)
    elif debug:
        print("[kitty debug] Skipping API key validation (--no-validate)", file=sys.stderr)

    # 3. Start bridge server
    server = BridgeServer(
        adapter,
        provider,
        resolved_key,
        model=profile.model,
        debug=debug,
        provider_config=profile.provider_config,
        backends=backends,
        logging_enabled=logging_enabled,
    )
    port = await server.start_async()
    logger.info("Bridge started on port %d", port)
    if server.log_path:
        print(f"Bridge debug log: {server.log_path}", file=sys.stderr)

    # 4. Build spawn config
    spawn_config = adapter.build_spawn_config(profile, port, resolved_key)

    # 5. Discover binary
    binary_path = resolve_binary(adapter.binary_name)

    # 6. Build full command and environment
    cmd = [str(binary_path)] + spawn_config.cli_args + extra_args
    env = build_child_env(spawn_config)

    # Debug: log key env vars for diagnosing connectivity issues
    logger.info(
        "Child env: ANTHROPIC_BASE_URL=%s ANTHROPIC_MODEL=%s ANTHROPIC_API_KEY=%s...(%d chars)",
        env.get("ANTHROPIC_BASE_URL", "<not set>"),
        env.get("ANTHROPIC_MODEL", "<not set>"),
        (env.get("ANTHROPIC_API_KEY") or "")[:4],
        len(env.get("ANTHROPIC_API_KEY") or ""),
    )
    if debug:
        print(
            f"[kitty debug] ANTHROPIC_BASE_URL={env.get('ANTHROPIC_BASE_URL', '<not set>')}",
            file=sys.stderr,
        )
        print(
            f"[kitty debug] ANTHROPIC_MODEL={env.get('ANTHROPIC_MODEL', '<not set>')}",
            file=sys.stderr,
        )
        print(
            f"[kitty debug] ANTHROPIC_API_KEY={env.get('ANTHROPIC_API_KEY', '<not set>')[:8]}...",
            file=sys.stderr,
        )

    # 7. Patch agent-specific external config (e.g. Claude Code settings.json)
    original_settings: str | None = None
    settings_path: Path | None = None
    if hasattr(adapter, "prepare_launch"):
        # Resolve the settings path before calling prepare_launch so we can
        # pass it explicitly for both patching and cleanup.
        if hasattr(adapter, "_DEFAULT_SETTINGS_PATH"):
            settings_path = adapter._DEFAULT_SETTINGS_PATH
        else:
            settings_path = Path.home() / ".claude" / "settings.json"
        original_settings = adapter.prepare_launch(
            spawn_config.env_overrides,
            settings_path=settings_path,
        )
        if original_settings is not None:
            _register_atexit_cleanup(adapter, original_settings, settings_path)
        if debug:
            print(
                f"[kitty debug] settings.json patched (original={'saved' if original_settings else 'none'})",
                file=sys.stderr,
            )
    else:
        if debug:
            print("[kitty debug] adapter has no prepare_launch method", file=sys.stderr)

    # 8. Spawn child process with signal forwarding
    logger.info("Launching child: %s", " ".join(cmd))
    child_exit_code = 0
    proc: asyncio.subprocess.Process | None = None
    try:
        stdin_arg = sys.stdin if sys.stdin.isatty() else None
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdin=stdin_arg,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Forward SIGINT/SIGTERM to the child process
        child_pid = proc.pid

        def _forward_signal(signum: int, _frame: object) -> None:
            if child_pid is not None:
                logger.info("Forwarding signal %d to child pid %d", signum, child_pid)
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(child_pid, signum)

        old_sigint = signal.signal(signal.SIGINT, _forward_signal)
        old_sigterm = signal.signal(signal.SIGTERM, _forward_signal)
        try:
            child_exit_code = await proc.wait()
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

    except Exception as exc:
        logger.error("Failed to launch child process: %s", exc)
        print(f"Error: Failed to launch {adapter.binary_name!r}: {exc}", file=sys.stderr)
        child_exit_code = 1
        # Ensure child is terminated on error
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (TimeoutError, ProcessLookupError):
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
    finally:
        # 9. Restore external config (must not prevent server shutdown on failure)
        if hasattr(adapter, "cleanup_launch"):
            with contextlib.suppress(Exception):
                adapter.cleanup_launch(
                    original_settings,
                    settings_path=settings_path,
                )
        # Clear atexit state so it doesn't double-restore
        _clear_atexit_cleanup()

        # 10. Stop bridge server
        await server.stop_async()

    # Return mapped exit code
    return map_child_exit_code(child_exit_code)


def launch(
    adapter: LauncherAdapter,
    provider: ProviderAdapter,
    profile: Profile,
    cred_store: CredentialStore,
    extra_args: list[str] | None = None,
    *,
    debug: bool = False,
    validate: bool = True,
    backends: list[tuple[ProviderAdapter, str, Profile]] | None = None,
    logging_enabled: bool = False,
) -> int:
    """Synchronous wrapper around launch_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                coro = launch_async(
                    adapter,
                    provider,
                    profile,
                    cred_store,
                    extra_args,
                    debug=debug,
                    validate=validate,
                    backends=backends,
                    logging_enabled=logging_enabled,
                )
                future = pool.submit(asyncio.run, coro)
                return future.result()
        coro = launch_async(
            adapter,
            provider,
            profile,
            cred_store,
            extra_args,
            debug=debug,
            validate=validate,
            backends=backends,
            logging_enabled=logging_enabled,
        )
        return loop.run_until_complete(coro)
    except RuntimeError:
        coro = launch_async(
            adapter,
            provider,
            profile,
            cred_store,
            extra_args,
            debug=debug,
            validate=validate,
            backends=backends,
            logging_enabled=logging_enabled,
        )
        return asyncio.run(coro)
