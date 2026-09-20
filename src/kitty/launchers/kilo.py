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

# ``_atomic_write_text`` and ``_is_local_kitty_url`` are single-sourced from
# the Claude launcher so the byte-identity contract (``newline=""``, KBR-262)
# and the loopback-host detector cannot drift across launchers.
from kitty.launchers.claude import _atomic_write_text, _is_local_kitty_url
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


_DEFAULT_BACKUP_PATH = Path.home() / ".config" / "kitty" / "kilo-config-backup.json"


def save_kilo_config_backup(original: str, backup_path: Path | None = None) -> None:
    """Save the original kilo.json content to a backup file for crash recovery.

    Args:
        original: Original kilo.json content.
        backup_path: Backup file location. Defaults to the module-level
            ``_DEFAULT_BACKUP_PATH``, resolved at call time (not import time)
            so tests can redirect it by patching the module attribute.
    """
    if backup_path is None:
        backup_path = _DEFAULT_BACKUP_PATH
    try:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(backup_path, original)
        logger.debug("save_kilo_config_backup: wrote backup to %s", backup_path)
    except OSError as exc:
        logger.warning("save_kilo_config_backup: failed to write backup: %s", exc)


def load_kilo_config_backup(backup_path: Path | None = None) -> str | None:
    """Load the kilo config backup, returning None if it doesn't exist.

    Args:
        backup_path: Backup file location. Defaults to the module-level
            ``_DEFAULT_BACKUP_PATH``, resolved at call time.

    Returns:
        The backed-up content, or ``None`` when no backup exists.
    """
    if backup_path is None:
        backup_path = _DEFAULT_BACKUP_PATH
    if not backup_path.exists():
        return None
    with backup_path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def delete_kilo_config_backup(backup_path: Path | None = None) -> None:
    """Delete the kilo config backup file (idempotent).

    Args:
        backup_path: Backup file location. Defaults to the module-level
            ``_DEFAULT_BACKUP_PATH``, resolved at call time.
    """
    if backup_path is None:
        backup_path = _DEFAULT_BACKUP_PATH
    backup_path.unlink(missing_ok=True)


def _kilo_kitty_values_present(config: object) -> bool:
    """Return ``True`` when ``config`` looks like a kitty session wrote it.

    Used by :meth:`KiloAdapter.prepare_launch` to skip the backup write when
    the captured original already carries kitty markers (the clean-capture
    rule that prevents the crash-then-relaunch clobber), and by
    :func:`kitty.cli.cleanup_cmd.run_kilo_cleanup` to gate the restore on
    whether the current config still owns the file.

    A loopback ``baseURL`` is the unambiguous kitty signal (Claude parity, the
    single-sourced :func:`kitty.launchers.claude._is_local_kitty_url` owns the
    host list). The ``kitty/`` model prefix catches hand-trimmed crash damage
    (kitty always writes the ``provider.kitty`` block and the ``model`` key
    as a pair). A remote-URL ``provider.kitty`` alone does **not** count: a
    user who named their own provider ``kitty`` must not have their config
    auto-restored by ``kitty cleanup``.

    Args:
        config: The parsed kilo.json (or anything else).

    Returns:
        ``True`` when the config carries a loopback ``provider.kitty`` base
        URL or a top-level ``model`` prefixed ``kitty/``; ``False`` otherwise
        (including non-dict input).
    """
    if not isinstance(config, dict):
        return False
    providers = config.get("provider")
    if isinstance(providers, dict):
        kitty = providers.get(_PROVIDER_ID)
        if isinstance(kitty, dict):
            options = kitty.get("options")
            if isinstance(options, dict):
                base_url = options.get("baseURL")
                if isinstance(base_url, str) and _is_local_kitty_url(base_url):
                    return True
    model = config.get("model")
    return isinstance(model, str) and model.startswith(f"{_PROVIDER_ID}/")


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
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        *,
        context_tokens: int | None = None,
    ) -> SpawnConfig:
        """Build the spawn configuration for the Kilo CLI child process.

        Kilo's provider block is written later by :meth:`prepare_launch`, so
        this method only stashes the launch values. It has no use for
        ``context_tokens``, which is accepted and ignored so the orchestrator
        call site stays uniform across adapters.
        """
        del context_tokens  # Kilo CLI has no use for the model context window
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

    @property
    def default_settings_path(self) -> Path | None:
        """Location of the Kilo CLI config file.

        Returns:
            Path to ``~/.config/kilo/kilo.json``.
        """
        return _DEFAULT_CONFIG_PATH

    def prepare_launch(
        self,
        env_overrides: dict[str, str],
        settings_path: Path | None = None,
    ) -> str | None:
        """Write a temporary Kilo CLI config pointing at the bridge.

        Args:
            env_overrides: Env vars from ``build_spawn_config`` (unused; values
                are read from instance attributes stashed during
                ``build_spawn_config``).
            settings_path: Path to the Kilo CLI config file (for testing).

        Returns:
            The original file content for ``cleanup_launch``, or ``None``.

        Raises:
            RuntimeError: If ``build_spawn_config`` was not called first.

        Notes:
            ``newline=""`` disables universal newlines (``\r\n`` -> ``\n`` on
            every platform); without it, a CRLF config on disk is silently read
            as LF, breaking the byte-identity contract that ``cleanup_launch``'s
            restore asserts against the user's original.

            The returned ``original`` (when not ``None``) is also persisted to
            ``~/.config/kitty/kilo-config-backup.json`` so a SIGKILL mid-session
            no longer leaves the kitty provider block in the user's global
            config (KBR-268). The backup is **only** written when the captured
            original is clean — see the clean-capture rule comment in the
            body. On the normal exit path :meth:`cleanup_launch` removes the
            backup after a successful restore; on a crash, ``kitty cleanup``
            reads it.
        """
        config_path = settings_path or _DEFAULT_CONFIG_PATH
        if not self._bridge_port:
            raise RuntimeError("build_spawn_config must be called before prepare_launch")

        original: str | None = None
        config: dict = {}

        if config_path.exists():
            with config_path.open("r", encoding="utf-8", newline="") as f:
                original = f.read()
            try:
                config = json.loads(original)
            except json.JSONDecodeError:
                logger.warning("Kilo config is malformed JSON, will overwrite")
            if not isinstance(config, dict):
                config = {}

            # Clean-capture rule (KBR-268): a captured original that already
            # carries kitty markers is a crashed earlier session's patch.
            # Never back it up — the crash-then-relaunch sequence would
            # otherwise overwrite the true backup with the very patch that
            # needs recovering. A malformed original parses as {} and counts
            # as clean: the user's bytes are still the honest original.
            if _kilo_kitty_values_present(config):
                logger.warning(
                    "kilo.json still carries values from a crashed kitty "
                    "session; run `kitty cleanup` to attempt recovery "
                    "(requires the backup from before that session)."
                )
            else:
                save_kilo_config_backup(original)

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
        settings_path: Path | None = None,
    ) -> None:
        """Restore the original Kilo CLI config file, ownership-aware.

        Restores (and removes the crash backup) only when this session still
        owns the file: the captured original is clean and the current
        ``kilo.json`` still carries kitty markers. Any other combination — a
        polluted captured original, or a current file that is clean, missing,
        or unreadable — leaves the file and the backup untouched (KBR-268,
        Claude ``_restore_owned_settings`` parity): the last writer owns the
        file, and the backup must survive for ``kitty cleanup``.

        Args:
            original: The content returned by ``prepare_launch``.
            settings_path: Path to the Kilo CLI config file (for testing).

        Notes:
            ``newline=""`` disables CPython's ``\n`` -> ``os.linesep``
            translation on write; without it, on Windows a ``\n`` in
            ``original`` becomes ``\r\n`` on disk and breaks the byte-identity
            contract that ``prepare_launch``'s capture establishes.

            The backup is deleted **after** a successful restore write, so a
            failure anywhere leaves the backup in place for ``kitty cleanup``.
        """
        config_path = settings_path or _DEFAULT_CONFIG_PATH
        if original is None:
            # We created the file from scratch; remove it
            try:
                config_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove temporary Kilo config")
            return

        # Polluted snapshot: the captured original carries kitty markers, so
        # this session captured another session's patch as its "original"
        # (concurrent-session interleave). Restoring it would overwrite the
        # current file with equally-dead content AND delete the only backup
        # holding the true original — leave both alone instead.
        try:
            original_parsed = json.loads(original)
        except json.JSONDecodeError:
            original_parsed = None
        if _kilo_kitty_values_present(original_parsed):
            logger.warning(
                "cleanup_launch: captured original carries kitty markers "
                "(likely another session's patch); leaving file and backup "
                "alone. Run `kitty cleanup` after all sessions end."
            )
            return

        # Ownership check: only restore when the current file is readable and
        # still carries kitty markers. Clean, missing, or unreadable current
        # file means the user or another session owns it now — never guess by
        # writing (Claude parity, claude.py `_restore_owned_settings`).
        try:
            current = config_path.read_text(encoding="utf-8")
            current_parsed = json.loads(current)
        except (OSError, ValueError):
            logger.info(
                "cleanup_launch: %s missing or unreadable — leaving it alone",
                config_path,
            )
            return
        if not _kilo_kitty_values_present(current_parsed):
            logger.info(
                "cleanup_launch: %s no longer carries this session's values — "
                "another session or the user owns it; leaving file and backup "
                "untouched",
                config_path,
            )
            return

        # Restore then delete the backup inside one guard (Claude parity): a
        # failed delete (Windows AV lock, permissions) must not escape the
        # orchestrator's finally as an uncaught traceback.
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(original, encoding="utf-8", newline="")
            delete_kilo_config_backup()
        except Exception:
            logger.warning("Failed to restore Kilo config")
            raise
