"""CLI router -- maps positional args to builtins, adapters, and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kitty.launchers.base import LauncherAdapter
from kitty.profiles.resolver import ProfileResolver
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore

__all__ = ["BuiltinCommand", "CLIRouter", "RouteResult"]


class BuiltinCommand(str, Enum):
    """Built-in CLI commands."""

    SETUP = "setup"
    PROFILE = "profile"
    DOCTOR = "doctor"
    CLEANUP = "cleanup"
    BRIDGE = "bridge"


class RoutingError(Exception):
    """Raised when the router cannot resolve the given arguments."""


@dataclass
class RouteResult:
    """Result of routing CLI arguments."""

    adapter: LauncherAdapter | None = None
    profile: Profile | None = None
    extra_args: list[str] = field(default_factory=list)
    builtin: BuiltinCommand | None = None
    needs_setup: bool = False


# Maps lowercase CLI words to BuiltinCommand values.
_BUILTIN_MAP: dict[str, BuiltinCommand] = {bc.value: bc for bc in BuiltinCommand}

# Default launcher adapter key used when routing by profile name alone.
_DEFAULT_ADAPTER_KEY = "codex"


class CLIRouter:
    """Routes CLI positional arguments to built-in commands, adapters, or profiles.

    Routing priority:
    1. Built-in command (setup, profile, doctor)
    2. Launcher target (codex, claude)
    3. Profile name (case-insensitive)
    4. No match -- raise RoutingError

    When no profiles exist in the store, every route resolves to SETUP.
    """

    def __init__(self, store: ProfileStore, adapters: dict[str, LauncherAdapter]) -> None:
        self._store = store
        self._adapters = adapters
        self._resolver = ProfileResolver(store)

    def route(self, args: list[str]) -> RouteResult:
        """Route the given positional arguments.

        Args:
            args: Positional CLI arguments (e.g. ``["codex", "--verbose"]``).

        Returns:
            A RouteResult describing what to run.

        Raises:
            RoutingError: If the first argument matches nothing.
            NoDefaultProfileError: If a launcher target is given but no default
                profile is configured.
        """
        # When the profile store is empty, always direct to setup.
        if not self._store.load_all():
            return RouteResult(builtin=BuiltinCommand.SETUP, needs_setup=True)

        if not args:
            return RouteResult(builtin=BuiltinCommand.SETUP, needs_setup=True)

        head = args[0]
        rest = args[1:]
        head_lower = head.lower()

        # 1. Built-in command match (bridge can also come after profile name)
        builtin = _BUILTIN_MAP.get(head_lower)
        if builtin is not None and builtin != BuiltinCommand.BRIDGE:
            return RouteResult(builtin=builtin, extra_args=rest)

        # 2. Bridge command (standalone or with profile)
        if builtin == BuiltinCommand.BRIDGE:
            profile = self._resolver.resolve_default()
            return RouteResult(builtin=builtin, profile=profile, extra_args=rest)

        # 3. Launcher target match
        adapter = self._adapters.get(head_lower)
        if adapter is not None:
            profile = self._resolver.resolve_default()
            return RouteResult(adapter=adapter, profile=profile, extra_args=rest)

        # 4. Profile name match (profile may be followed by bridge or agent)
        profile = self._store.get(head_lower)
        if profile is not None:
            # Second arg may be bridge or a launcher target
            if rest:
                second = rest[0].lower()
                if second == "bridge":
                    return RouteResult(builtin=BuiltinCommand.BRIDGE, profile=profile, extra_args=rest[1:])
                if second in self._adapters:
                    adapter = self._adapters[second]
                    return RouteResult(adapter=adapter, profile=profile, extra_args=rest[1:])
            # Default to codex if no target specified
            adapter = self._adapters[_DEFAULT_ADAPTER_KEY]
            return RouteResult(adapter=adapter, profile=profile, extra_args=rest)

        # 5. No match
        raise RoutingError(f"Unknown command or profile: {head!r}")
