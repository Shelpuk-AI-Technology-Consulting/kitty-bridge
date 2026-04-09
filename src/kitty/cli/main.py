"""CLI entry point for kitty."""

from __future__ import annotations

import sys

from kitty import __version__

__all__ = ["main", "map_child_exit_code"]


def map_child_exit_code(code: int) -> int:
    """Map a child process exit code to the kitty exit code. Re-exported from launcher."""
    from kitty.cli.launcher import map_child_exit_code as _map

    return _map(code)


def main() -> None:
    """Main entry point for the kitty CLI."""
    import argparse

    from kitty.tui.display import print_banner

    print_banner(__version__)

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
        "command",
        nargs="*",
        help="Command to run: setup, profile, doctor, codex, claude, or a profile name.",
    )
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
    elif result.adapter is not None and result.profile is not None:
        exit_code = _launch_target(
            result.adapter, result.profile, cred_store,
            result.extra_args, debug=args.debug,
            validate=not args.no_validate,
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


def _launch_target(
    adapter: object,
    profile: object,
    cred_store: object,
    extra_args: list[str],
    *,
    debug: bool = False,
    validate: bool = True,
) -> int:
    from kitty.cli.launcher import launch
    from kitty.providers.registry import get_provider

    return launch(
        adapter=adapter,  # type: ignore[arg-type]
        provider=get_provider(profile.provider),  # type: ignore[union-attr]
        profile=profile,  # type: ignore[arg-type]
        cred_store=cred_store,  # type: ignore[arg-type]
        extra_args=extra_args,
        debug=debug,
        validate=validate,
    )
