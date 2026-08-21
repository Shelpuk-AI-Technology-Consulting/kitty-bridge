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

    Used only by `kitty cleanup`, to decide whether phase-1 crash recovery
    should restore from the backup.

    Deliberately broader than the launch-time check in
    :meth:`ClaudeAdapter._warn_if_global_carries_kitty_values`, which keys on
    the auth token alone: `kitty cleanup` is an explicit repair request, so a
    loopback URL is signal enough, whereas warning on every launch must not
    fire for someone running their own local proxy. Do not merge the two.

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


class _SessionSettingsFile(str):
    """Path to this session's own Claude Code settings file.

    A ``str`` subclass so it flows unchanged through the orchestrator's atexit
    state and the adapter contract, while :meth:`ClaudeAdapter.cleanup_launch`
    can tell it apart from the legacy bare-``str`` snapshot. The distinction is
    load-bearing: the legacy branch writes its argument as the *entire content*
    of the settings file, so a plain path string would overwrite the user's
    settings.json with a path.
    """

    __slots__ = ()


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

_DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


class ClaudeAdapter(LauncherAdapter):
    """Launcher adapter for Anthropic Claude Code CLI.

    Configures Claude Code to route requests through the local bridge using
    environment variables (``ANTHROPIC_BASE_URL``, ``ANTHROPIC_API_KEY``).
    The model is set via ``ANTHROPIC_MODEL`` (startup selection) and all three
    alias overrides (``ANTHROPIC_DEFAULT_*_MODEL``) so Claude Code uses the
    profile's model regardless of which alias it picks.

    Because a settings ``env`` block overrides process-level env vars,
    :meth:`prepare_launch` writes a **per-session** settings file carrying
    kitty's values and :meth:`settings_cli_args` points Claude Code at it with
    ``--settings``. The user-global ``~/.claude/settings.json`` is only read
    (for a stale-value warning), never written, so concurrent sessions cannot
    disturb each other's configuration.
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

    def settings_cli_args(self, prepared: str | None) -> list[str]:
        """Return the CLI args that point Claude Code at the session file.

        Args:
            prepared: The value returned by :meth:`prepare_launch`.

        Returns:
            ``["--settings", <path>]``, or ``[]`` when nothing was prepared
            (no file to point at, so no dangling flag).
        """
        if prepared is None:
            return []
        return ["--settings", str(prepared)]

    def prepare_launch(
        self,
        env_overrides: dict[str, str],
        settings_path: Path | None = None,
    ) -> str | None:
        """Write this session's own Claude Code settings file.

        Claude Code's settings ``env`` block overrides process-level
        environment variables, so pointing it at the bridge requires a settings
        source rather than the child's env alone. kitty writes a **per-session**
        file and passes it as ``claude --settings <path>``: that scope outranks
        the user, project, and local settings files, and — unlike the previous
        design — leaves the user-global ``~/.claude/settings.json`` untouched,
        so a second session's start cannot disturb a running session (issue
        #22).

        Only kitty's own keys are written. Claude Code merges a ``--settings``
        ``env`` block per variable, so the user's other entries and every other
        top-level key keep their user-scope values; copying them in would
        freeze a snapshot of the user's configuration at launch time.

        Args:
            env_overrides: The env vars from :meth:`build_spawn_config`.
            settings_path: Location of the user-global settings file. Read
                only, for the stale-value warning below; never written.

        Returns:
            A :class:`_SessionSettingsFile` holding the path of the file to
            pass to ``--settings``.

        Raises:
            OSError: When the session file cannot be written. The caller must
                fail closed — without the file the session would silently run
                on the user's own credentials, bypassing the bridge.
        """
        settings_path = settings_path or _DEFAULT_SETTINGS_PATH
        self._warn_if_global_carries_kitty_values(settings_path)

        session_env = {key: env_overrides[key] for key in _SETTINGS_ENV_OVERRIDE_KEYS if key in env_overrides}

        # mkstemp (0600 on POSIX) in the OS temp dir: the OS reaps orphans left
        # by a SIGKILL, which a kitty-owned directory would accumulate forever.
        fd, path_str = tempfile.mkstemp(prefix="kitty-claude-settings-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"env": session_env}, handle, indent=2)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path_str)
            raise

        logger.info(
            "prepare_launch: wrote session settings %s with env %s",
            path_str,
            {k: (v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v) for k, v in session_env.items()},
        )
        return _SessionSettingsFile(path_str)

    def _warn_if_global_carries_kitty_values(self, settings_path: Path) -> None:
        """Warn when the user-global settings file still holds kitty values.

        kitty no longer rewrites that file, so a session killed by a *pre-fix*
        version would leave a dead ``127.0.0.1`` base URL there indefinitely:
        kitty sessions would keep working (they use ``--settings``), while every
        plain ``claude`` run failed with no hint of the cause.

        Best-effort by design — a missing, unreadable, or malformed file is not
        an error here, and must never block a launch.

        Args:
            settings_path: Location of the user-global settings file.
        """
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # ValueError covers both JSONDecodeError and UnicodeDecodeError —
            # a settings file saved as UTF-16 (Notepad's "Unicode") must warn
            # at most, never abort a launch.
            return
        if not isinstance(settings, dict):
            return

        # The auth token is kitty's unambiguous signature. A loopback base URL
        # alone would also match a user's own local proxy (LiteLLM, Ollama),
        # and `kitty cleanup` would strip their keys.
        env = settings.get("env")
        if isinstance(env, dict) and env.get("ANTHROPIC_AUTH_TOKEN") == "kitty-bridge-token":
            logger.warning(
                "%s still carries values from a crashed kitty session; plain `claude` runs may fail. "
                "Run `kitty cleanup` to restore it.",
                settings_path,
            )

    def cleanup_launch(
        self,
        original: str | None,
        settings_path: Path | None = None,
    ) -> None:
        """Undo whatever :meth:`prepare_launch` set up for this session.

        The normal path deletes this session's own settings file and touches
        nothing else — it is idempotent, because the orchestrator calls it from
        both a ``finally`` block and an ``atexit`` handler.

        Two legacy paths remain for callers predating per-session isolation: a
        :class:`_SessionSnapshot` restores the user-global file only if this
        session still owns it (every value it injected is still present), and a
        bare string restores that file verbatim.

        Args:
            original: The value returned by :meth:`prepare_launch`.
            settings_path: Path to the user-global settings file, used only by
                the legacy paths.
        """
        settings_path = settings_path or _DEFAULT_SETTINGS_PATH
        if original is None:
            return
        # Per-session file: delete this session's file and nothing else. Checked
        # FIRST — the legacy branch below writes its argument as the entire file
        # content, so a path string reaching it would clobber the user's settings.
        if isinstance(original, _SessionSettingsFile):
            try:
                os.unlink(original)
            except FileNotFoundError:
                pass  # Already cleaned up: finally + atexit both call this.
            except OSError as exc:
                # Never fatal — the session is over either way — but do not
                # claim success; on Windows the child may still hold the file.
                logger.warning("cleanup_launch: could not remove %s: %s", original, exc)
            else:
                logger.info("cleanup_launch: removed session settings %s", original)
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
        except (OSError, ValueError):
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
