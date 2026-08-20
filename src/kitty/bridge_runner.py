"""Bridge runner module for background bridge mode.

This module is invoked by `kitty bridge start` via `python -m kitty.bridge_runner`.
It starts the bridge server in the foreground (the background daemon manager
in `manage.py` handles the process spawning).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from kitty.bridge.server import BridgeServer


def main() -> None:
    parser = argparse.ArgumentParser(prog="kitty.bridge_runner")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--no-log", action="store_true", default=False)
    parser.add_argument("--log-file", default=None, metavar="PATH")
    parser.add_argument("--tls-cert", default=None)
    parser.add_argument("--tls-key", default=None)
    args = parser.parse_args()

    # Load config if specified
    host = args.host or "127.0.0.1"
    port = args.port or 0
    access_log_path = None
    keys_file = None
    tls_cert = args.tls_cert
    tls_key = args.tls_key

    if args.config:
        from kitty.bridge.config import load_bridge_config

        config = load_bridge_config(
            args.config,
            cli_host=args.host,
            cli_port=args.port,
            cli_tls_cert=args.tls_cert,
            cli_tls_key=args.tls_key,
        )
        host = config.host
        port = config.port
        tls_cert = config.tls_cert
        tls_key = config.tls_key
        keys_file = config.keys_file

        if config.resolved_log_access(background=True) and not args.no_log:
            access_log_path = str(Path(config.log_dir) / "bridge_access.log")

    if args.log and not args.no_log and not access_log_path:
        access_log_path = str(Path.home() / ".config" / "kitty" / "logs" / "bridge_access.log")

    # Resolve provider and key from profile
    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialStore
    from kitty.profiles.schema import BalancingProfile
    from kitty.profiles.store import ProfileStore
    from kitty.providers.registry import get_provider

    usage_log_path = Path(args.log_file) if args.log_file else None
    logging_enabled = usage_log_path is not None

    profile_store = ProfileStore()
    cred_store = CredentialStore(backends=[FileBackend()])

    # Resolve egress before anything can open a socket. The background bridge
    # gets it from the environment or the stored gateway; the proxy password is
    # never passed on the command line, where `ps` would expose it.
    from kitty.egress import set_egress
    from kitty.egress_guard import egress_block_reason
    from kitty.egress_store import resolve_egress

    try:
        egress = resolve_egress(cred_store=cred_store)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    set_egress(egress)

    # State file — always use default path
    state_path = Path.home() / ".config" / "kitty" / "bridge_state.json"

    profile_name = args.profile
    backend = profile_store.get_backend(profile_name) if profile_name else None
    if backend is None:
        from kitty.profiles.resolver import ProfileResolver

        resolver = ProfileResolver(profile_store)
        backend = resolver.resolve_default_backend()

    if backend is None:
        print("No profile configured. Run 'kitty setup' first.", file=sys.stderr)
        sys.exit(1)

    if isinstance(backend, BalancingProfile):
        from kitty.profiles.resolver import ProfileResolver

        resolver = ProfileResolver(profile_store)
        members = resolver.resolve_balancing(backend.name)
        backends = []
        for mp in members:
            key = cred_store.get(mp.auth_ref)
            if not key:
                print(f"No API key for profile {mp.name!r}", file=sys.stderr)
                sys.exit(1)
            backends.append((get_provider(mp.provider, mp.provider_config), key, mp))

        _block = egress_block_reason(backends[0][0], members[0], backends[0][1], backends)
        if _block:
            print(f"Error: {_block}", file=sys.stderr)
            sys.exit(1)

        server = BridgeServer(
            adapter=None,
            provider=backends[0][0],
            resolved_key=backends[0][1],
            host=host,
            port=port,
            model=members[0].model,
            provider_config=members[0].provider_config,
            backends=backends,
            access_log_path=access_log_path,
            profile_name=backend.name,
            keys_file=keys_file,
            tls_cert=tls_cert,
            tls_key=tls_key,
            state_file=str(state_path),
            logging_enabled=logging_enabled,
            egress=egress,
            _usage_log_path=usage_log_path,
        )
    else:
        profile = backend
        resolved_key = cred_store.get(profile.auth_ref)
        if not resolved_key:
            print(f"No API key for profile {profile.name!r}", file=sys.stderr)
            sys.exit(1)

        _guard_provider = get_provider(profile.provider, profile.provider_config)
        _block = egress_block_reason(_guard_provider, profile, resolved_key)
        if _block:
            print(f"Error: {_block}", file=sys.stderr)
            sys.exit(1)

        server = BridgeServer(
            adapter=None,
            provider=_guard_provider,
            resolved_key=resolved_key,
            host=host,
            port=port,
            model=profile.model,
            provider_config=profile.provider_config,
            access_log_path=access_log_path,
            profile_name=profile.name,
            keys_file=keys_file,
            tls_cert=tls_cert,
            tls_key=tls_key,
            state_file=str(state_path),
            logging_enabled=logging_enabled,
            egress=egress,
            _usage_log_path=usage_log_path,
        )

    async def run() -> None:
        # Refresh the model-context overrides catalog before the bridge
        # starts. Egress is already installed above, best-effort, never raises.
        from kitty.providers import model_context_sync

        await model_context_sync.refresh_model_context_overrides()
        await server.start_async()
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)

        try:
            await stop_event.wait()
        finally:
            await server.stop_async()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
