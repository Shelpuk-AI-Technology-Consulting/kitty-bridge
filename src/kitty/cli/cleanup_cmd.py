"""Cleanup command — remove stale bridge-injected values from agent settings files."""

from __future__ import annotations

import json
from pathlib import Path

_DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
_DEFAULT_KILO_CONFIG_PATH = Path.home() / ".config" / "kilo" / "kilo.json"

# Keys that kitty injects into Claude Code's settings.json env block.
_KITTY_INJECTED_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "ENABLE_CLAUDEAI_MCP_SERVERS",
)


def _get_backup_path() -> Path:
    """Return the Claude settings backup path used for exact restore.

    Single-sourced from :mod:`kitty.launchers.claude` so the writer
    (``prepare_launch``) and the crash-recovery reader can never drift apart.

    Returns:
        Path of the crash-recovery backup file.
    """
    # Lazy so patching the source module attribute redirects this too.
    from kitty.launchers.claude import _DEFAULT_BACKUP_PATH

    return _DEFAULT_BACKUP_PATH


def _get_kilo_backup_path() -> Path:
    """Return the Kilo config backup path used for exact restore.

    Single-sourced from :mod:`kitty.launchers.kilo` so the writer
    (``prepare_launch``) and the crash-recovery reader can never drift apart.

    Returns:
        Path of the Kilo crash-recovery backup file.
    """
    # Lazy so patching the source module attribute redirects this too.
    from kitty.launchers.kilo import _DEFAULT_BACKUP_PATH

    return _DEFAULT_BACKUP_PATH


def _is_stale_base_url(value: str) -> bool:
    """Check if an ANTHROPIC_BASE_URL points to a local kitty bridge.

    Delegates to the single shared detector in
    :func:`kitty.launchers.claude._is_local_kitty_url` so the launcher and the
    crash-recovery cleaner can never disagree about what counts as a kitty URL.

    Args:
        value: Candidate ``ANTHROPIC_BASE_URL`` value.

    Returns:
        ``True`` when the URL points at a loopback host.
    """
    from kitty.launchers.claude import _is_local_kitty_url

    return _is_local_kitty_url(value)


def _detect_stale_env(env: dict) -> list[str]:
    """Return list of keys in env that are stale kitty-injected values."""
    stale: list[str] = []

    base_url = env.get("ANTHROPIC_BASE_URL")
    if base_url is not None and isinstance(base_url, str) and _is_stale_base_url(base_url):
        stale.append("ANTHROPIC_BASE_URL")
        for key in _KITTY_INJECTED_KEYS:
            if key != "ANTHROPIC_BASE_URL" and key in env:
                stale.append(key)
        return stale

    auth_token = env.get("ANTHROPIC_AUTH_TOKEN")
    if auth_token == "kitty-bridge-token":
        for key in _KITTY_INJECTED_KEYS:
            if key in env:
                stale.append(key)

    return stale


def _load_backup(backup_path: Path) -> str | None:
    """Load a settings backup if one exists.

    ``newline=""`` disables universal newlines (``\r\n`` -> ``\n`` on
    every platform); without it, a CRLF backup on disk is silently read as
    LF, breaking the byte-identity contract that ``kitty cleanup``'s
    restore asserts against the user's original.
    """
    if not backup_path.exists():
        return None
    with backup_path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _restore_from_backup(settings_path: Path, backup_path: Path) -> bool:
    """Restore Claude settings from an exact backup."""
    original = _load_backup(backup_path)
    if original is None:
        return False

    try:
        from kitty.launchers.claude import _atomic_write_text

        _atomic_write_text(settings_path, original)
        backup_path.unlink(missing_ok=True)
        print(f"Restored {settings_path} from backup {backup_path}")
        return True
    except OSError as exc:
        print(f"Error: Failed to restore {settings_path} from backup: {exc}")
        return False


def _display_value(value: object) -> str:
    """Format a value for display, truncating long strings."""
    if isinstance(value, str):
        return value[:37] + "..." if len(value) > 40 else value
    return repr(value)


def run_cleanup(settings_path: Path = _DEFAULT_SETTINGS_PATH) -> int:
    """Remove stale bridge-injected values from Claude Code's settings.json.

    Uses a two-phase strategy:
    1. If a backup file exists AND the settings file still carries live kitty
       values, restore the exact original content (crash recovery). A backup
       without live kitty values is a stale leftover from an ended session
       chain and is deleted instead of restored — restoring it would silently
       revert the user's current settings. A missing settings file is
       resurrected from the backup: it is the only surviving copy of the
       user's original.
    2. Otherwise, fall back to heuristic detection of stale kitty values.

    Args:
        settings_path: Path to Claude Code's settings.json.

    Returns:
        0 on success, 1 on error.
    """
    backup_path = _get_backup_path()

    # Phase 1: Exact restore from backup (crash recovery) — only while a live
    # kitty session (or a crashed one) still owns the settings file.
    if backup_path.exists():
        from kitty.launchers.claude import _kitty_values_present

        if not settings_path.exists():
            # The file vanished mid-session; the backup is the only surviving
            # original — resurrect the file rather than destroy the backup.
            if _restore_from_backup(settings_path, backup_path):
                return 0
        else:
            try:
                current = json.loads(settings_path.read_text(encoding="utf-8"))
                live_kitty = isinstance(current, dict) and _kitty_values_present(current.get("env"))
            except (OSError, ValueError):
                # Unreadable: assume a crashed patch and let the exact backup win.
                # ValueError covers JSONDecodeError and UnicodeDecodeError alike —
                # a settings file saved as UTF-16 (Notepad's "Unicode") must not
                # crash the one command that repairs it.
                live_kitty = True
            if live_kitty:
                if _restore_from_backup(settings_path, backup_path):
                    return 0
            else:
                backup_path.unlink(missing_ok=True)
                print(f"Removed stale backup {backup_path}")

    if not settings_path.exists():
        print(f"No settings file at {settings_path} — nothing to clean up.")
        return 0

    try:
        settings_text = settings_path.read_text(encoding="utf-8")
        settings = json.loads(settings_text)
    except (OSError, ValueError) as exc:
        # ValueError subsumes JSONDecodeError and UnicodeDecodeError, so a
        # damaged or UTF-16 settings file is reported, never raised.
        print(f"Error: Cannot read {settings_path}: {exc}")
        return 1

    if not isinstance(settings, dict):
        print(f"Error: {settings_path} is not a JSON object — skipping")
        return 1

    env = settings.get("env")
    if not isinstance(env, dict) or not env:
        print("No env block in settings.json — already clean.")
        return 0

    stale_keys = _detect_stale_env(env)
    if not stale_keys:
        print("No stale kitty bridge values found — already clean.")
        return 0

    for key in stale_keys:
        value = env.pop(key)
        print(f"  Removed {key} = {_display_value(value)}")

    # Write back atomically
    try:
        from kitty.launchers.claude import _atomic_write_json

        _atomic_write_json(settings_path, settings)
    except OSError as exc:
        print(f"Error: Failed to write {settings_path}: {exc}")
        return 1

    print(f"Cleaned {len(stale_keys)} stale value(s) from {settings_path}")
    return 0

    if not settings_path.exists():
        print(f"No settings file at {settings_path} — nothing to clean up.")
        # Clean up orphaned backup if settings.json doesn't exist.
        if backup_path.exists():
            backup_path.unlink(missing_ok=True)
            print(f"Removed orphaned backup {backup_path}")
        return 0

    try:
        settings_text = settings_path.read_text(encoding="utf-8")
        settings = json.loads(settings_text)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: Cannot read {settings_path}: {exc}")
        return 1

    if not isinstance(settings, dict):
        print(f"Error: {settings_path} is not a JSON object — skipping")
        return 1

    env = settings.get("env")
    if not isinstance(env, dict) or not env:
        print("No env block in settings.json — already clean.")
        return 0

    stale_keys = _detect_stale_env(env)
    if not stale_keys:
        print("No stale kitty bridge values found — already clean.")
        return 0

    for key in stale_keys:
        value = env.pop(key)
        print(f"  Removed {key} = {_display_value(value)}")

    # Write back atomically
    try:
        from kitty.launchers.claude import _atomic_write_json

        _atomic_write_json(settings_path, settings)
    except OSError as exc:
        print(f"Error: Failed to write {settings_path}: {exc}")
        return 1

    print(f"Cleaned {len(stale_keys)} stale value(s) from {settings_path}")
    return 0


def run_kilo_cleanup(settings_path: Path = _DEFAULT_KILO_CONFIG_PATH) -> int:
    """Remove a crashed Kilo session's bridge values from ``kilo.json``.

    Backup-only exact restore (KBR-268): the crash-recovery counterpart of
    :meth:`kitty.launchers.kilo.KiloAdapter.prepare_launch`'s clean-capture
    backup. There is deliberately no heuristic strip arm — kitty *overwrites*
    kilo.json's ``model`` key, so without the exact backup the user's
    pre-session model is unknowable and no heuristic can fully restore it.

    A backup whose current config no longer carries kitty markers is a stale
    leftover from an ended session chain and is deleted instead of restored —
    restoring it would silently revert the user's current config. A missing
    or unreadable ``kilo.json`` counts as a crashed patch: the exact backup
    is the only surviving original, so it wins. (The deliberate opposite of
    the live session's ``cleanup_launch``, which never guesses on an
    unreadable file.)

    Args:
        settings_path: Path to ``kilo.json``.

    Returns:
        0 on success — including the no-backup no-op and the stale-backup
        removal — or 1 when reading the backup or writing the restore fails
        (there is no heuristic phase to fall through to).
    """
    backup_path = _get_kilo_backup_path()
    if not backup_path.exists():
        return 0

    # Read the backup first: if it cannot be read, the config must stay as
    # staged — a half-restored kilo.json would be worse than the crash damage.
    try:
        original = _load_backup(backup_path)
    except OSError as exc:
        print(f"Error: Failed to read Kilo backup {backup_path}: {exc}")
        return 1
    if original is None:
        # Raced away between exists() and the read; nothing to restore.
        return 0

    # Decide whether the current config still carries kitty markers. A missing
    # or unreadable file counts as crashed (the backup is the only surviving
    # original); ValueError covers JSONDecodeError and UnicodeDecodeError alike.
    from kitty.launchers.kilo import _kilo_kitty_values_present

    try:
        current = settings_path.read_text(encoding="utf-8")
        live_kitty = _kilo_kitty_values_present(json.loads(current))
    except (OSError, ValueError):
        live_kitty = True

    if not live_kitty:
        backup_path.unlink(missing_ok=True)
        print(f"Removed stale backup {backup_path}")
        return 0

    try:
        from kitty.launchers.claude import _atomic_write_text

        _atomic_write_text(settings_path, original)
    except OSError as exc:
        print(f"Error: Failed to restore {settings_path} from backup: {exc}")
        return 1
    backup_path.unlink(missing_ok=True)
    print(f"Restored {settings_path} from backup {backup_path}")
    return 0
