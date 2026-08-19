"""Launcher adapter interface, spawn configuration, and BridgeProtocol re-export."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol

__all__ = ["BridgeProtocol", "LauncherAdapter", "SpawnConfig"]


@dataclass
class SpawnConfig:
    """Configuration for spawning a child coding-agent process.

    Semantics (in order):
    1. Copy parent environment.
    2. Unset all keys listed in ``env_clear``.
    3. Apply all ``env_overrides``.
    4. Append ``cli_args`` after the binary name.
    """

    env_overrides: dict[str, str] = field(default_factory=dict)
    env_clear: list[str] = field(default_factory=list)
    cli_args: list[str] = field(default_factory=list)


class LauncherAdapter(ABC):
    """Interface for launcher target adapters (Codex, Claude Code, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""

    @property
    @abstractmethod
    def binary_name(self) -> str:
        """Name of the executable binary to discover and launch."""

    @property
    @abstractmethod
    def bridge_protocol(self) -> BridgeProtocol:
        """Wire protocol this adapter expects from the local bridge."""

    @abstractmethod
    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
    ) -> SpawnConfig:
        """Build the spawn configuration for the child process.

        Args:
            profile: Resolved profile with provider, model, and base_url.
            bridge_port: Port the local bridge is listening on.
            resolved_key: Raw API key string resolved from the credential store.
        """

    def prepare_launch(
        self,
        env_overrides: dict[str, str],
        settings_path: Path | None = None,
    ) -> str | None:
        """Patch the agent's own config file before spawning it, if needed.

        Some agents read a settings file whose ``env`` block overrides
        process-level environment variables, so pointing them at the local
        bridge requires editing that file for the duration of the session.
        Adapters whose agent needs no such patching inherit this default.

        Args:
            env_overrides: Environment values the bridge needs the agent to use.
            settings_path: Location of the agent's settings file. Defaults to
                the adapter's own location.

        Returns:
            The original file content, for :meth:`cleanup_launch` to restore, or
            ``None`` when nothing was patched. Returning ``None`` is what tells
            the launcher there is nothing to clean up.
        """
        return None

    def cleanup_launch(
        self,
        original: str | None,
        settings_path: Path | None = None,
    ) -> None:
        """Restore whatever :meth:`prepare_launch` patched.

        Called on the normal path and again from an atexit handler, so it must
        be safe to invoke with nothing to restore.

        Args:
            original: Content returned by :meth:`prepare_launch`, or ``None``.
            settings_path: Location of the agent's settings file.
        """
        return None
