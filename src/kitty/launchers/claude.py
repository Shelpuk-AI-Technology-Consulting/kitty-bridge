"""ClaudeAdapter — configures Anthropic Claude Code CLI to talk to the local bridge."""

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

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using a temp file + rename.

    On POSIX, rename(2) is atomic within the same filesystem.
    This prevents corruption if the process is killed mid-write.
    """
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path_str, path)
    except Exception:
        # Clean up the temp file on failure so we don't leave orphans.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path_str)
        raise


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text content atomically using a temp file + rename."""
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path_str, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path_str)
        raise

__all__ = ["ClaudeAdapter"]

_CONFLICTING_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
)

# Env vars that must be injected into Claude Code's settings.json env block
# because settings.json env overrides process-level env vars.
_SETTINGS_ENV_OVERRIDE_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)

# Keys to remove from settings.json env block — none currently.
# Previously removed ANTHROPIC_AUTH_TOKEN but Claude Code needs it for
# /login checks even though the bridge uses Bearer auth.
_SETTINGS_ENV_REMOVE_KEYS: tuple[str, ...] = ()

_DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


class ClaudeAdapter(LauncherAdapter):
    """Launcher adapter for Anthropic Claude Code CLI.

    Configures Claude Code to route requests through the local bridge using
    environment variables (``ANTHROPIC_BASE_URL``, ``ANTHROPIC_API_KEY``).
    The model is set via ``ANTHROPIC_MODEL`` (startup selection) and all three
    alias overrides (``ANTHROPIC_DEFAULT_*_MODEL``) so Claude Code uses the
    profile's model regardless of which alias it picks.

    Because Claude Code's ``settings.json`` ``env`` block overrides
    process-level env vars, :meth:`prepare_launch` temporarily patches the
    file to inject bridge-specific values, and :meth:`cleanup_launch` restores
    the original content when the session ends.
    """

    @property
    def name(self) -> str:
        return "claude"

    @property
    def binary_name(self) -> str:
        return "claude"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(
            cli_args=[],
            env_overrides={
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{bridge_port}",
                "ANTHROPIC_API_KEY": resolved_key,
                "ANTHROPIC_MODEL": profile.model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": profile.model,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": profile.model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": profile.model,
            },
            env_clear=list(_CONFLICTING_ENV_VARS),
        )

    def prepare_launch(
        self,
        env_overrides: dict[str, str],
        settings_path: Path = _DEFAULT_SETTINGS_PATH,
    ) -> str | None:
        """Temporarily patch Claude Code's settings.json to inject bridge env vars.

        Claude Code's ``settings.json`` ``env`` block takes priority over
        process-level environment variables.  This method injects our bridge
        URL and model into that block so they are guaranteed to take effect.

        Args:
            env_overrides: The env vars from :meth:`build_spawn_config`.
            settings_path: Path to the Claude Code settings file (for testing).

        Returns:
            The original file content for :meth:`cleanup_launch`, or ``None``
            if there is no settings file to patch, the file is missing, or
            the JSON is malformed.
        """
        logger.info("prepare_launch: settings_path=%s exists=%s", settings_path, settings_path.exists())
        if not settings_path.exists():
            logger.warning("prepare_launch: settings.json not found at %s — skipping patch", settings_path)
            return None

        original = settings_path.read_text(encoding="utf-8")
        try:
            settings = json.loads(original)
        except json.JSONDecodeError as exc:
            logger.error("prepare_launch: settings.json is malformed JSON: %s — skipping patch", exc)
            return None

        if not isinstance(settings, dict):
            logger.error("prepare_launch: settings.json root is not an object — skipping patch")
            return None

        env = settings.setdefault("env", {})

        logger.info(
            "prepare_launch: settings.json env before patch: %s",
            {k: (v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v)
             for k, v in env.items()},
        )

        # Clean up stale localhost ANTHROPIC_BASE_URL from previous crashed sessions
        existing_base_url = env.get("ANTHROPIC_BASE_URL", "")
        if existing_base_url and ("127.0.0.1" in existing_base_url or "localhost" in existing_base_url):
            logger.info("prepare_launch: removing stale ANTHROPIC_BASE_URL=%s from previous session", existing_base_url)
            env.pop("ANTHROPIC_BASE_URL", None)
            # Also remove other kitty-injected keys from the stale session
            for key in _SETTINGS_ENV_OVERRIDE_KEYS:
                if key in env and key != "ANTHROPIC_BASE_URL":
                    logger.debug("prepare_launch: removing stale %s from previous session", key)
                    env.pop(key, None)

        for key in _SETTINGS_ENV_OVERRIDE_KEYS:
            if key in env_overrides:
                env[key] = env_overrides[key]

        for key in _SETTINGS_ENV_REMOVE_KEYS:
            env.pop(key, None)

        logger.info(
            "prepare_launch: settings.json env after patch: %s",
            {k: (v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v)
             for k, v in env.items()},
        )

        _atomic_write_json(settings_path, settings)
        return original

    def cleanup_launch(
        self,
        original: str | None,
        settings_path: Path = _DEFAULT_SETTINGS_PATH,
    ) -> None:
        """Restore Claude Code's settings.json to its original state.

        Args:
            original: The content returned by :meth:`prepare_launch`.
            settings_path: Path to the Claude Code settings file (for testing).
        """
        if original is None:
            return
        logger.info("cleanup_launch: restoring %s", settings_path)
        try:
            _atomic_write_text(settings_path, original)
        except Exception:
            logger.warning("cleanup_launch: failed to restore %s — user may need to fix manually", settings_path)
            raise
