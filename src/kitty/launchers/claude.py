"""ClaudeAdapter — configures Anthropic Claude Code CLI to talk to the local bridge."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

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

_DEFAULT_BACKUP_PATH = Path.home() / ".config" / "kitty" / "claude-settings-backup.json"


def save_settings_backup(original: str, backup_path: Path | None = None) -> None:
    """Save the original settings.json content to a backup file for crash recovery.

    Args:
        original: Original settings.json content.
        backup_path: Backup file location. Defaults to the module-level
            ``_DEFAULT_BACKUP_PATH``, resolved at call time (not import time)
            so tests can redirect it by patching the module attribute.
    """
    if backup_path is None:
        backup_path = _DEFAULT_BACKUP_PATH
    try:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(backup_path, original)
        logger.debug("save_settings_backup: wrote backup to %s", backup_path)
    except OSError as exc:
        logger.warning("save_settings_backup: failed to write backup: %s", exc)


def load_settings_backup(backup_path: Path | None = None) -> str | None:
    """Load the settings backup, returning None if it doesn't exist.

    Args:
        backup_path: Backup file location. Defaults to the module-level
            ``_DEFAULT_BACKUP_PATH``, resolved at call time.

    Returns:
        The backed-up settings content, or ``None`` when no backup exists.
    """
    if backup_path is None:
        backup_path = _DEFAULT_BACKUP_PATH
    if not backup_path.exists():
        return None
    return backup_path.read_text(encoding="utf-8")


def delete_settings_backup(backup_path: Path | None = None) -> None:
    """Delete the settings backup file.

    Args:
        backup_path: Backup file location. Defaults to the module-level
            ``_DEFAULT_BACKUP_PATH``, resolved at call time.
    """
    if backup_path is None:
        backup_path = _DEFAULT_BACKUP_PATH
    backup_path.unlink(missing_ok=True)


def _is_local_kitty_url(value: str) -> bool:
    """Return ``True`` when ``value`` looks like a kitty bridge URL.

    All kitty-injected ``ANTHROPIC_BASE_URL`` values point at a loopback
    interface (the local bridge), so a localhost URL in the file is a strong
    signal that a live kitty session last wrote it.

    Args:
        value: Candidate URL string.

    Returns:
        ``True`` when the URL's hostname is loopback (v4 or v6).
    """
    try:
        hostname = (urlparse(value).hostname or "").lower()
    except Exception:
        return False
    return hostname in ("127.0.0.1", "localhost", "::1")


def _kitty_values_present(env: object) -> bool:
    """Return ``True`` when ``env`` looks like a live kitty session wrote it.

    Used by :meth:`ClaudeAdapter.prepare_launch` to decide whether an existing
    backup file is still trustworthy (the first writer of an overlap chain
    owns it; later writers must not clobber) and by `kitty cleanup` to decide
    whether phase-1 crash recovery should restore from it.

    Args:
        env: A parsed ``env`` block from settings.json, or anything else.

    Returns:
        ``True`` when the block carries a localhost base URL or the
        kitty-injected auth token; ``False`` otherwise.
    """
    if not isinstance(env, dict):
        return False
    base_url = env.get("ANTHROPIC_BASE_URL")
    if isinstance(base_url, str) and _is_local_kitty_url(base_url):
        return True
    return env.get("ANTHROPIC_AUTH_TOKEN") == "kitty-bridge-token"


class _SessionSnapshot(str):
    """Pre-launch settings snapshot annotated with this session's injections.

    A ``str`` subclass so it flows unchanged through every call site that
    already passes the snapshot text around (orchestrator, atexit state,
    tests), while :meth:`ClaudeAdapter.cleanup_launch` can distinguish a
    snapshot taken by :meth:`ClaudeAdapter.prepare_launch` from a bare string
    and apply overlap-aware restore semantics.

    Attributes:
        injected: The ``env`` values this session wrote into settings.json
            (only keys from ``_SETTINGS_ENV_OVERRIDE_KEYS``), used to decide
            whether this session still owns the file at cleanup time.
    """

    __slots__ = ("injected",)

    injected: dict[str, str]

    def __new__(cls, text: str, injected: dict[str, str]) -> _SessionSnapshot:
        """Create a snapshot of ``text`` carrying the injected env values.

        Args:
            text: The original settings.json content.
            injected: The ``env`` values this session injected.

        Returns:
            The annotated snapshot (compares equal to ``text``).
        """
        snapshot = super().__new__(cls, text)
        snapshot.injected = dict(injected)
        return snapshot


_CONFLICTING_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
)

# Env vars that must be injected into Claude Code's settings.json env block
# because settings.json env overrides process-level env vars.
_SETTINGS_ENV_OVERRIDE_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
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

    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        *,
        context_tokens: int | None = None,
    ) -> SpawnConfig:
        """Build the spawn configuration for the Claude Code child process.

        Sets the bridge URL, auth, and model env vars. When ``context_tokens``
        is a positive number it is also exported as
        ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` so Claude Code uses the model's
        real context window instead of its 200K fallback for non-claude
        models. Setting it for ``claude-*`` models is harmless — Claude Code
        ignores the variable there.

        Args:
            profile: Resolved profile with provider, model, and base_url.
            bridge_port: Port the local bridge is listening on.
            resolved_key: Raw API key string resolved from the credential store.
            context_tokens: Resolved model context window in tokens, or
                ``None`` when unknown (the key is then omitted entirely).

        Returns:
            The spawn configuration for the child process.
        """
        env_overrides = {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{bridge_port}",
            "ANTHROPIC_API_KEY": resolved_key,
            "ANTHROPIC_AUTH_TOKEN": "kitty-bridge-token",
            "ANTHROPIC_MODEL": profile.model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": profile.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": profile.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": profile.model,
        }
        if context_tokens is not None and context_tokens > 0:
            env_overrides["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context_tokens)
        return SpawnConfig(
            cli_args=[],
            env_overrides=env_overrides,
            env_clear=list(_CONFLICTING_ENV_VARS),
        )

    @property
    def default_settings_path(self) -> Path | None:
        """Location of Claude Code's settings file.

        Returns:
            Path to ``~/.claude/settings.json``.
        """
        return _DEFAULT_SETTINGS_PATH

    def prepare_launch(
        self,
        env_overrides: dict[str, str],
        settings_path: Path | None = None,
    ) -> str | None:
        """Temporarily patch Claude Code's settings.json to inject bridge env vars.

        Claude Code's ``settings.json`` ``env`` block takes priority over
        process-level environment variables.  This method injects our bridge
        URL and model into that block so they are guaranteed to take effect.

        Args:
            env_overrides: The env vars from :meth:`build_spawn_config`.
            settings_path: Path to the Claude Code settings file (for testing).

        Returns:
            A :class:`_SessionSnapshot` of the original file content for
            :meth:`cleanup_launch` (compares equal to the original text), or
            ``None`` if there is no settings file to patch, the file is
            missing, or the JSON is malformed.
        """
        settings_path = settings_path or _DEFAULT_SETTINGS_PATH
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

        # Save the crash backup before patching. First writer of an overlap
        # chain owns it: later sessions must not overwrite it with a file that
        # already contains another session's patch (issue #21). Exception: a
        # backup left behind by an ended chain (file carries no live kitty
        # values) is stale and is refreshed, so a later session can never
        # "restore" an ancient pre-kitty file over the user's current one.
        if not _DEFAULT_BACKUP_PATH.exists() or not _kitty_values_present(settings.get("env")):
            save_settings_backup(original)

        env = settings.setdefault("env", {})

        logger.info(
            "prepare_launch: settings.json env before patch: %s",
            {k: (v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v) for k, v in env.items()},
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
            {k: (v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v) for k, v in env.items()},
        )

        _atomic_write_json(settings_path, settings)
        injected = {key: env_overrides[key] for key in _SETTINGS_ENV_OVERRIDE_KEYS if key in env_overrides}
        return _SessionSnapshot(original, injected)

    def cleanup_launch(
        self,
        original: str | None,
        settings_path: Path | None = None,
    ) -> None:
        """Restore Claude Code's settings.json to the user's pre-kitty state.

        Overlap-aware: when the snapshot came from :meth:`prepare_launch` (a
        :class:`_SessionSnapshot`), the file is restored only if this session
        still owns it — every value this session injected is still present.
        A later session (or a user edit) overwrites ownership, and its state
        must not be disturbed. A bare string restores verbatim (legacy
        callers/tests).

        Args:
            original: The content returned by :meth:`prepare_launch`.
            settings_path: Path to the Claude Code settings file (for testing).
        """
        settings_path = settings_path or _DEFAULT_SETTINGS_PATH
        if original is None:
            return
        logger.info("cleanup_launch: restoring %s", settings_path)
        try:
            if isinstance(original, _SessionSnapshot):
                self._restore_owned_settings(original, settings_path)
            else:
                # Legacy path: caller passed plain snapshot text.
                _atomic_write_text(settings_path, original)
                delete_settings_backup()
        except Exception:
            logger.warning("cleanup_launch: failed to restore %s — user may need to fix manually", settings_path)
            raise

    def _restore_owned_settings(self, snapshot: _SessionSnapshot, settings_path: Path) -> None:
        """Restore settings.json if this session still owns it.

        Ownership rule: this session owns the file when every env value it
        injected is still present unchanged. Only the owner may restore. The
        restore source is the crash backup — the first session's snapshot of
        the user's true pre-kitty content — because a later session's own
        snapshot is polluted with an earlier session's patch (issue #19).

        Args:
            snapshot: Pre-launch snapshot carrying this session's injected env.
            settings_path: Path to the Claude Code settings file.
        """
        # Read the current file; a missing/unreadable file means someone else
        # changed the landscape — never guess by writing.
        try:
            current = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("cleanup_launch: %s missing or unreadable — leaving it alone", settings_path)
            return

        # Without an env block or injected values ownership is unknowable;
        # fall back to the pre-overlap verbatim restore.
        current_env = current.get("env") if isinstance(current, dict) else None
        if not isinstance(current_env, dict) or not snapshot.injected:
            _atomic_write_text(settings_path, snapshot)
            delete_settings_backup()
            return

        # Ownership check: another session or a user hand-edit means leave the
        # file AND the backup alone (issue #19, #20).
        owns = all(current_env.get(key) == value for key, value in snapshot.injected.items())
        if not owns:
            logger.info(
                "cleanup_launch: %s no longer carries this session's values — "
                "another session or the user owns it; leaving file and backup untouched",
                settings_path,
            )
            return

        # Last writer: restore the unpolluted user original from the backup,
        # falling back to this session's snapshot if the backup is gone.
        restore_text = load_settings_backup()
        if restore_text is None:
            restore_text = snapshot
        _atomic_write_text(settings_path, restore_text)
        delete_settings_backup()
