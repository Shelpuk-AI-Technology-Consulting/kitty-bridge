"""Bridge runner module for background bridge mode.

This module is invoked by `kitty bridge start` via `python -m kitty.bridge_runner`.
It starts the bridge server in the foreground (the background daemon manager
in `manage.py` handles the process spawning).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

from kitty.bridge.server import BridgeServer
from kitty.bridge.state import default_state_path
from kitty.bridge.stop_signals import install_stop_handlers
from kitty.io_encoding import harden_output_streams, relinquish_output_streams


def main() -> None:
    """Run the bridge server in the foreground until it is signalled to stop.

    Invoked as ``python -m kitty.bridge_runner`` by
    :func:`kitty.bridge.manage.start_bridge`, which owns the daemonisation. This
    is the second of kitty's two process entry points, so anything that
    configures the *process* rather than the server belongs here as well as in
    :func:`kitty.cli.main.main`.

    Reads its configuration from the command line; see ``--help``.
    """
    # `kitty bridge start` spawns this process with stderr merged into a stdout
    # pipe, so the locale-codepage path is not an edge case here -- it is the only
    # path. The parent reads that output back and decodes it as UTF-8
    # (`bridge/manage.py`), which is correct only because of this call.
    harden_output_streams()

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
    parser.add_argument("--state-file", default=None, metavar="PATH")
    args = parser.parse_args()

    # Load config if specified
    host = args.host or "127.0.0.1"
    port = args.port or 0
    access_log_path = None
    keys_file = None
    tls_cert = args.tls_cert
    tls_key = args.tls_key

    if args.config:
        from kitty.bridge.config import load_bridge_config, resolve_keys_file

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
        # Auth keys file (KBR-230). A named-but-missing file stops the start
        # with one clear line; with nothing named, the default file is used
        # when it exists and auth stays off when it does not.
        if config.keys_file is not None and not Path(config.keys_file).exists():
            print(
                f"Error: Keys file not found: {config.keys_file} (keys_file in {args.config}) "
                f"— create that file, or remove the keys_file line",
                file=sys.stderr,
            )
            sys.exit(1)
        keys_file = resolve_keys_file(config)

        if config.resolved_log_access(background=True) and not args.no_log:
            access_log_path = str(Path(config.log_dir) / "bridge_access.log")

    if args.log and not args.no_log and not access_log_path:
        access_log_path = str(Path.home() / ".config" / "kitty" / "logs" / "bridge_access.log")

    # Resolve provider and key from profile
    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialError, CredentialStore
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

    # `kitty bridge start` names the file it polls; a service-started bridge
    # names none and takes the default `kitty bridge status` also reads.
    state_path = Path(args.state_file) if args.state_file else default_state_path()

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
            # KBR-87: a corrupt stored value reports cleanly and exits, where
            # today's missing-key path does.
            try:
                key = cred_store.get(mp.auth_ref)
            except CredentialError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
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
        # KBR-87: a corrupt stored value reports cleanly and exits, where
        # today's missing-key path does.
        try:
            resolved_key = cred_store.get(profile.auth_ref)
        except CredentialError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
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
        # Ready is reported: the state file exists, the parent has printed the
        # URL or given up. From here the parent's pipe serves nothing — a
        # write to it after the parent exits would raise BrokenPipeError and
        # kill a serving bridge (KBR-219) — so pipe-shaped streams go to
        # os.devnull. Service-manager-provided streams (journal socket, log
        # file) are not pipes and stay.
        relinquish_output_streams()
        stop_event = asyncio.Event()
        install_stop_handlers(asyncio.get_running_loop(), stop_event.set)

        try:
            await stop_event.wait()
        finally:
            await server.stop_async()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
