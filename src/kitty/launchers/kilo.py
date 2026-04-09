"""Launcher adapter for Kilo Code CLI (https://github.com/Kilo-Org/kilocode).

Configures Kilo CLI to route requests through the local bridge by writing
a temporary ``kilo.json`` config file that defines a ``kitty`` provider
pointing at the bridge.  MCP servers are configured in a separate
``opencode.json`` so the two don't conflict.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.types import BridgeProtocol

__all__ = ["KiloAdapter"]

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "kilo"
# Use kilo.json (recommended by Kilo docs) for provider config.
# MCP servers go in opencode.json — keeping them separate avoids conflicts.
_DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_DIR / "kilo.json"
_PROVIDER_ID = "kitty"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path_str, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise


class KiloAdapter(LauncherAdapter):
    """Launcher adapter for Kilo Code CLI.

    Configures Kilo CLI to route requests through the local bridge using
    a temporary ``kilo.json`` config file that defines a ``kitty``
    provider pointing at the bridge.

    Uses ``prepare_launch`` / ``cleanup_launch`` to save and restore the
    original config file around the session, following the same pattern as
    ClaudeAdapter's settings.json patching.

    Note: This adapter is not safe for concurrent use. The orchestrator
    guarantees ``build_spawn_config`` is called before ``prepare_launch``.
    """

    def __init__(self) -> None:
        self._bridge_port: int = 0
        self._resolved_key: str = ""
        self._model: str = ""

    @property
    def name(self) -> str:
        return "kilo"

    @property
    def binary_name(self) -> str:
        return "kilo"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.CHAT_COMPLETIONS_API

    def build_spawn_config(
        self, profile: Profile, bridge_port: int, resolved_key: str,
    ) -> SpawnConfig:
        # Stash values for prepare_launch (called later by the orchestrator
        # with only env_overrides as argument).
        self._bridge_port = bridge_port
        self._resolved_key = resolved_key
        self._model = profile.model

        return SpawnConfig(
            cli_args=[],
            env_overrides={},
            env_clear=[],
        )

    def prepare_launch(
        self,
        env_overrides: dict[str, str],
        config_path: Path = _DEFAULT_CONFIG_PATH,
    ) -> str | None:
        """Write a temporary Kilo CLI config pointing at the bridge.

        Args:
            env_overrides: Env vars from ``build_spawn_config`` (unused; values
                are read from instance attributes stashed during
                ``build_spawn_config``).
            config_path: Path to the Kilo CLI config file (for testing).

        Returns:
            The original file content for ``cleanup_launch``, or ``None``.

        Raises:
            RuntimeError: If ``build_spawn_config`` was not called first.
        """
        if not self._bridge_port:
            raise RuntimeError("build_spawn_config must be called before prepare_launch")

        original: str | None = None
        config: dict = {}

        if config_path.exists():
            original = config_path.read_text(encoding="utf-8")
            try:
                config = json.loads(original)
            except json.JSONDecodeError:
                logger.warning("Kilo config is malformed JSON, will overwrite")
            if not isinstance(config, dict):
                config = {}

        providers = config.setdefault("provider", {})
        providers[_PROVIDER_ID] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Kitty Bridge",
            "options": {
                "baseURL": f"http://127.0.0.1:{self._bridge_port}/v1",
                "apiKey": self._resolved_key,
            },
            "models": {
                self._model: {
                    "id": self._model,
                    "name": self._model,
                },
            },
        }

        # Set the active model (Kilo auto-prefixes the provider ID)
        config["model"] = f"kitty/{self._model}"

        _atomic_write_json(config_path, config)
        return original

    def cleanup_launch(
        self,
        original: str | None,
        config_path: Path = _DEFAULT_CONFIG_PATH,
    ) -> None:
        """Restore the original Kilo CLI config file.

        Args:
            original: The content returned by ``prepare_launch``.
            config_path: Path to the Kilo CLI config file (for testing).
        """
        if original is None:
            # We created the file from scratch; remove it
            try:
                config_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove temporary Kilo config")
            return
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(original, encoding="utf-8")
        except Exception:
            logger.warning("Failed to restore Kilo config")
            raise
