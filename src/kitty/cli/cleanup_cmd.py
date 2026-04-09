"""Cleanup command — remove stale bridge-injected values from agent settings files."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

_DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Keys that kitty injects into Claude Code's settings.json env block.
_KITTY_INJECTED_KEYS: tuple[str, ...] = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)


def _is_stale_base_url(value: str) -> bool:
    """Check if an ANTHROPIC_BASE_URL points to a local kitty bridge."""
    try:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()
        return hostname in ("127.0.0.1", "localhost", "::1")
    except Exception:
        return False


def _detect_stale_env(env: dict) -> list[str]:
    """Return list of keys in env that are stale kitty-injected values."""
    stale: list[str] = []

    base_url = env.get("ANTHROPIC_BASE_URL")
    if base_url is not None and isinstance(base_url, str) and _is_stale_base_url(base_url):
        stale.append("ANTHROPIC_BASE_URL")
        # If BASE_URL is stale, all other kitty keys are stale too
        for key in _KITTY_INJECTED_KEYS:
            if key != "ANTHROPIC_BASE_URL" and key in env:
                stale.append(key)

    return stale


def _display_value(value: object) -> str:
    """Format a value for display, truncating long strings."""
    if isinstance(value, str):
        return value[:37] + "..." if len(value) > 40 else value
    return repr(value)


def run_cleanup(settings_path: Path = _DEFAULT_SETTINGS_PATH) -> int:
    """Remove stale bridge-injected values from Claude Code's settings.json.

    Returns:
        0 on success, 1 on error.
    """
    if not settings_path.exists():
        print(f"No settings file at {settings_path} — nothing to clean up.")
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
