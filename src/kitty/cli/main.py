"""CLI entry point for kitty."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kitty import __version__

__all__ = ["main", "map_child_exit_code"]

if TYPE_CHECKING:
    from kitty.profiles.schema import BalancingProfile


def map_child_exit_code(code: int) -> int:
    """Map a child process exit code to the kitty exit code. Re-exported from launcher."""
    from kitty.cli.launcher import map_child_exit_code as _map

    return _map(code)


def _build_parser():
    """Build the argument parser. Exposed for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kitty",
        description="Kitty Bridge — launch coding agents through a local API bridge.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"kitty {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging to ~/.cache/kitty/bridge.log",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip pre-flight API key validation",
    )
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Log LLM call token usage to ~/.cache/kitty/usage.log",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write usage logs to PATH instead of ~/.cache/kitty/usage.log (implies --logging)",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Command to run: setup, profile, doctor, codex, claude, or a profile name.",
    )
    return parser


def main() -> None:
    """Main entry point for the kitty CLI."""
    from kitty.tui.display import print_banner

    print_banner(__version__)

    parser = _build_parser()
    args, unknown_args = parser.parse_known_args()

    # Merge unknown flags back into the command list so they pass through to the agent.
    # parse_known_args() stops consuming unknowns at the first positional token (the
    # command name), so unknown flags always appear after the command in the merged list.
    all_command_args = args.command + unknown_args

    if args.debug and unknown_args:
        print(f"[kitty debug] forwarding unknown flags to agent: {unknown_args}", file=sys.stderr)
    # Lazy imports to avoid heavy dependency loading for --version/--help
    from kitty.cli.router import BuiltinCommand, CLIRouter
    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialStore
    from kitty.launchers.claude import ClaudeAdapter
    from kitty.launchers.codex import CodexAdapter
    from kitty.launchers.gemini import GeminiAdapter
    from kitty.launchers.kilo import KiloAdapter
    from kitty.profiles.store import ProfileStore

    adapters = {"codex": CodexAdapter(), "claude": ClaudeAdapter(), "gemini": GeminiAdapter(), "kilo": KiloAdapter()}
    profile_store = ProfileStore()
    cred_store = CredentialStore(backends=[FileBackend()])
    router = CLIRouter(profile_store, adapters)

    result = router.route(all_command_args)

    if result.builtin == BuiltinCommand.SETUP:
        _run_setup(profile_store, cred_store)
    elif result.builtin == BuiltinCommand.PROFILE:
        _run_profile_menu(profile_store, cred_store)
    elif result.builtin == BuiltinCommand.DOCTOR:
        _run_doctor(profile_store)
    elif result.builtin == BuiltinCommand.CLEANUP:
        _run_cleanup()
    elif result.builtin == BuiltinCommand.BRIDGE_START:
        from pathlib import Path as _Path

        from platformdirs import user_config_dir as _ucd

        _config_dir = _Path(_ucd("kitty"))
        from kitty.bridge.manage import start_bridge

        start_bridge(
            state_path=_config_dir / "bridge_state.json",
            config_path=_config_dir / "bridge.yaml",
        )
    elif result.builtin == BuiltinCommand.BRIDGE_STOP:
        from pathlib import Path as _Path

        from platformdirs import user_config_dir as _ucd

        _config_dir = _Path(_ucd("kitty"))
        from kitty.bridge.manage import stop_bridge

        stop_bridge(_config_dir / "bridge_state.json")
    elif result.builtin == BuiltinCommand.BRIDGE_RESTART:
        from pathlib import Path as _Path

        from platformdirs import user_config_dir as _ucd

        _config_dir = _Path(_ucd("kitty"))
        from kitty.bridge.manage import restart_bridge

        restart_bridge(
            state_path=_config_dir / "bridge_state.json",
            config_path=_config_dir / "bridge.yaml",
        )
    elif result.builtin == BuiltinCommand.BRIDGE_STATUS:
        from pathlib import Path as _Path

        from platformdirs import user_config_dir as _ucd

        _config_dir = _Path(_ucd("kitty"))
        from kitty.bridge.manage import BridgeStatus, bridge_status
        from kitty.bridge.state import load_state

        _state_path = _config_dir / "bridge_state.json"
        status = bridge_status(_state_path)
        if status == BridgeStatus.RUNNING:
            state = load_state(_state_path)
            scheme = "https" if state.tls else "http"
            print(f"Running: {scheme}://{state.host}:{state.port} (profile={state.profile}, PID {state.pid})")
            sys.exit(0)
        elif status == BridgeStatus.STALE:
            print("Stale state file found (process not running). Run 'kitty bridge stop' to clean up.")
            sys.exit(1)
        else:
            print("Bridge is not running.")
            sys.exit(1)
    elif result.builtin == BuiltinCommand.BRIDGE_CONFIG:
        from pathlib import Path as _Path

        from platformdirs import user_config_dir as _ucd

        _config_path = _Path(_ucd("kitty")) / "bridge.yaml"
        from kitty.bridge.config import load_bridge_config

        config = load_bridge_config(_config_path)
        print(f"Host: {config.host}")
        print(f"Port: {config.port}")
        print(f"Profile: {config.profile or '(default)'}")
        print(f"Keys file: {config.keys_file}")
        print(f"Log access: {config.log_access}")
        print(f"Log dir: {config.log_dir}")
        print(f"TLS cert: {config.tls_cert or '(none)'}")
        print(f"TLS key: {config.tls_key or '(none)'}")
    elif result.builtin == BuiltinCommand.BRIDGE_INSTALL:
        from pathlib import Path as _Path

        from platformdirs import user_config_dir as _ucd

        _config_path = str(_Path(_ucd("kitty")) / "bridge.yaml")
        dry_run = "--dry-run" in result.extra_args
        from kitty.bridge.service import generate_launchd_plist, generate_systemd_unit, generate_windows_script

        if sys.platform == "linux":
            content = generate_systemd_unit(executable=sys.executable, config_path=_config_path)
            if dry_run:
                print(content)
            else:
                unit_path = _Path.home() / ".config" / "systemd" / "user" / "kitty-bridge.service"
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_path.write_text(content)
                import subprocess

                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                print(f"Installed: {unit_path}")
                print("Enable: systemctl --user enable --now kitty-bridge")
        elif sys.platform == "darwin":
            content = generate_launchd_plist(executable=sys.executable, config_path=_config_path)
            if dry_run:
                print(content)
            else:
                plist_path = _Path.home() / "Library" / "LaunchAgents" / "com.kitty.bridge.plist"
                plist_path.parent.mkdir(parents=True, exist_ok=True)
                plist_path.write_text(content)
                print(f"Installed: {plist_path}")
                print("Load: launchctl load " + str(plist_path))
        else:
            content = generate_windows_script(executable=sys.executable, config_path=_config_path)
            if dry_run:
                print(content)
            else:
                script_path = _Path("install-bridge-service.ps1")
                script_path.write_text(content)
                print(f"Generated {script_path}. Review and run it to install.")
    elif result.builtin == BuiltinCommand.BRIDGE_UNINSTALL:
        import subprocess
        from pathlib import Path as _Path

        if sys.platform == "linux":
            unit_path = _Path.home() / ".config" / "systemd" / "user" / "kitty-bridge.service"
            subprocess.run(["systemctl", "--user", "stop", "kitty-bridge"], capture_output=True)
            subprocess.run(["systemctl", "--user", "disable", "kitty-bridge"], capture_output=True)
            if unit_path.exists():
                unit_path.unlink()
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                print("Uninstalled kitty-bridge service.")
        elif sys.platform == "darwin":
            plist_path = _Path.home() / "Library" / "LaunchAgents" / "com.kitty.bridge.plist"
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            if plist_path.exists():
                plist_path.unlink()
                print("Uninstalled kitty-bridge service.")
        else:
            print("Run: nssm remove KittyBridge confirm")
    elif result.builtin == BuiltinCommand.BRIDGE:
        backend = result.backend or result.profile
        if backend is None:
            parser.error("No default profile configured. Create one with 'kitty setup' first.")
            sys.exit(1)
        _run_bridge(
            backend, cred_store,
            debug=args.debug, validate=not args.no_validate,
            logging_enabled=args.logging or args.log_file is not None,
            usage_log_path=args.log_file,
        )
    elif result.adapter is not None and (result.backend is not None or result.profile is not None):
        backend = result.backend or result.profile
        exit_code = _launch_target(
            result.adapter, backend, cred_store, result.extra_args,
            debug=args.debug, validate=not args.no_validate,
            logging_enabled=args.logging or args.log_file is not None,
            usage_log_path=args.log_file,
        )
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


def _run_setup(profile_store: object, cred_store: object) -> None:
    from kitty.cli.setup_cmd import run_setup_wizard

    run_setup_wizard(profile_store, cred_store)  # type: ignore[arg-type]


def _run_profile_menu(profile_store: object, cred_store: object) -> None:
    from kitty.cli.profile_cmd import run_profile_menu

    run_profile_menu(profile_store)  # type: ignore[arg-type]


def _run_doctor(profile_store: object) -> None:
    from kitty.cli.doctor_cmd import run_doctor

    exit_code = run_doctor(profile_store)  # type: ignore[arg-type]
    sys.exit(exit_code)


def _run_cleanup() -> None:
    from kitty.cli.cleanup_cmd import run_cleanup

    exit_code = run_cleanup()
    sys.exit(exit_code)


def _run_bridge(
    backend: object,
    cred_store: object,
    *,
    debug: bool = False,
    validate: bool = True,
    logging_enabled: bool = False,
    usage_log_path: Path | None = None,
) -> None:
    """Run bridge mode — start OpenAI-compatible API server without launching agent."""
    import asyncio
    import signal
    import sys
    from contextlib import suppress

    from kitty.bridge.server import BridgeServer
    from kitty.profiles.schema import BalancingProfile
    from kitty.providers.registry import get_provider
    from kitty.tui.display import print_error, print_panel, print_status, print_warning, status_spinner

    if isinstance(backend, BalancingProfile):
        _run_bridge_balancing(
            backend, cred_store,
            debug=debug, validate=validate,
            logging_enabled=logging_enabled,
            usage_log_path=usage_log_path,
        )
        return

    profile = backend  # type: ignore[assignment]

    # Resolve API key from credential store
    auth_ref = profile.auth_ref  # type: ignore[union-attr]
    resolved_key = cred_store.get(auth_ref)  # type: ignore[union-attr]
    if not resolved_key:
        print_error(f"No API key found for profile {profile.name!r}")
        sys.exit(1)

    # Get provider adapter
    provider = get_provider(profile.provider)  # type: ignore[union-attr]

    # Validate API key if requested
    if validate:
        with status_spinner("Validating API key..."):
            # TODO: Implement actual validation
            pass

    # Create bridge server (no adapter for bridge mode — direct Chat Completions API)
    server = BridgeServer(
        adapter=None,  # type: ignore[arg-type]
        provider=provider,
        resolved_key=resolved_key,
        model=profile.model,  # type: ignore[union-attr]
        debug=debug,
        provider_config=getattr(profile, "provider_config", {}),
        logging_enabled=logging_enabled,
        _usage_log_path=usage_log_path,
    )

    async def run_server() -> None:
        port = await server.start_async()

        print_panel(
            "Kitty Bridge Mode",
            f"[kitty.ok]Bridge server running on http://127.0.0.1:{port}[/kitty.ok]\n\n"
            f"Profile: [kitty.accent]{profile.name}[/kitty.accent]\n"  # type: ignore[union-attr]
            f"Provider: [kitty.accent]{profile.provider}[/kitty.accent]\n"  # type: ignore[union-attr]
            f"Model: [kitty.accent]{profile.model}[/kitty.accent]\n\n"  # type: ignore[union-attr]
            f"Endpoints:\n"
            f"  • POST /v1/chat/completions\n"
            f"  • POST /v1/messages\n"
            f"  • POST /v1/responses\n"
            f"  • POST /v1beta/models/{{model}}:generateContent\n"
            f"  • GET  /v1/models\n"
            f"  • GET  /healthz\n\n"
            f"Press Ctrl+C to stop",
        )

        # Set up graceful shutdown
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)

        try:
            await stop_event.wait()
        finally:
            print_warning("Shutting down bridge server...")
            await server.stop_async()
            print_status("Bridge server stopped.")

    with suppress(KeyboardInterrupt):
        asyncio.run(run_server())
    sys.exit(0)


def _run_bridge_balancing(
    balancing: BalancingProfile,
    cred_store: object,
    *,
    debug: bool = False,
    validate: bool = True,
    logging_enabled: bool = False,
    usage_log_path: Path | None = None,
) -> None:
    """Run bridge mode with a balancing profile — random selection across healthy members."""
    import asyncio
    import signal
    import sys

    from kitty.bridge.server import BridgeServer
    from kitty.profiles.resolver import ProfileResolver

    # Resolve all member profiles
    from kitty.profiles.store import ProfileStore as _PS
    from kitty.providers.registry import get_provider
    from kitty.tui.display import print_error, print_panel, print_status, print_warning

    profile_store = _PS()
    resolver = ProfileResolver(profile_store)
    member_profiles = resolver.resolve_balancing(balancing.name)

    # Build backends list: (provider, resolved_key, profile)
    backends = []
    for mp in member_profiles:
        key = cred_store.get(mp.auth_ref)  # type: ignore[union-attr]
        if not key:
            print_error(f"No API key found for member profile {mp.name!r}")
            sys.exit(1)
        provider = get_provider(mp.provider)
        backends.append((provider, key, mp))

    # Create bridge server with balancing backends
    first_provider = backends[0][0]
    first_key = backends[0][1]
    server = BridgeServer(
        adapter=None,  # type: ignore[arg-type]
        provider=first_provider,
        resolved_key=first_key,
        model=member_profiles[0].model,
        debug=debug,
        provider_config=member_profiles[0].provider_config,
        backends=backends,
        logging_enabled=logging_enabled,
        _usage_log_path=usage_log_path,
    )

    async def run_server() -> None:
        port = await server.start_async()

        members_info = "\n".join(
            f"  • [kitty.accent]{mp.name}[/kitty.accent] ({mp.provider}/{mp.model})"
            for mp in member_profiles
        )
        print_panel(
            "Kitty Bridge Mode (Balancing)",
            f"[kitty.ok]Bridge server running on http://127.0.0.1:{port}[/kitty.ok]\n\n"
            f"Profile: [kitty.accent]{balancing.name}[/kitty.accent] (balancing)\n"
            f"Members:\n{members_info}\n\n"
            f"Endpoints:\n"
            f"  • POST /v1/chat/completions\n"
            f"  • POST /v1/messages\n"
            f"  • POST /v1/responses\n"
            f"  • POST /v1beta/models/{{model}}:generateContent\n"
            f"  • GET  /v1/models\n"
            f"  • GET  /healthz\n\n"
            f"Press Ctrl+C to stop",
        )

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)

        try:
            await stop_event.wait()
        finally:
            print_warning("Shutting down bridge server...")
            await server.stop_async()
            print_status("Bridge server stopped.")

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server())
    sys.exit(0)


def _launch_target(
    adapter: object,
    backend: object,
    cred_store: object,
    extra_args: list[str],
    *,
    debug: bool = False,
    validate: bool = True,
    logging_enabled: bool = False,
    usage_log_path: Path | None = None,
) -> int:
    from kitty.cli.launcher import launch
    from kitty.profiles.schema import BalancingProfile
    from kitty.providers.registry import get_provider

    if isinstance(backend, BalancingProfile):
        return _launch_target_balancing(
            adapter, backend, cred_store, extra_args,
            debug=debug, validate=validate, logging_enabled=logging_enabled,
            usage_log_path=usage_log_path,
        )

    profile = backend
    return launch(
        adapter=adapter,  # type: ignore[arg-type]
        provider=get_provider(profile.provider),  # type: ignore[union-attr]
        profile=profile,  # type: ignore[arg-type]
        cred_store=cred_store,  # type: ignore[arg-type]
        extra_args=extra_args,
        debug=debug,
        validate=validate,
        logging_enabled=logging_enabled,
        usage_log_path=usage_log_path,
    )


def _launch_target_balancing(
    adapter: object,
    balancing: BalancingProfile,
    cred_store: object,
    extra_args: list[str],
    *,
    debug: bool = False,
    validate: bool = True,
    logging_enabled: bool = False,
    usage_log_path: Path | None = None,
) -> int:
    """Launch a coding agent with a balancing profile (random healthy member selection)."""
    from kitty.cli.launcher import launch
    from kitty.profiles.resolver import ProfileResolver
    from kitty.profiles.store import ProfileStore
    from kitty.providers.registry import get_provider

    profile_store = ProfileStore()
    resolver = ProfileResolver(profile_store)
    member_profiles = resolver.resolve_balancing(balancing.name)

    # Build backends list
    backends = []
    for mp in member_profiles:
        key = cred_store.get(mp.auth_ref)  # type: ignore[union-attr]
        if not key:
            from kitty.tui.display import print_error

            print_error(f"No API key found for member profile {mp.name!r}")
            return 1
        provider = get_provider(mp.provider)
        backends.append((provider, key, mp))

    # Use first member's provider/model for adapter spawn config
    first_profile = member_profiles[0]
    first_provider = backends[0][0]

    return launch(
        adapter=adapter,  # type: ignore[arg-type]
        provider=first_provider,
        profile=first_profile,
        cred_store=cred_store,  # type: ignore[arg-type]
        extra_args=extra_args,
        debug=debug,
        validate=validate,
        backends=backends,
        logging_enabled=logging_enabled,
        usage_log_path=usage_log_path,
    )
