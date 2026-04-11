"""API key file parsing for bridge authentication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KeyEntry:
    """A parsed key entry from the keys file."""

    key: str
    profile: str | None = None


def parse_keys_file(path: Path | str) -> list[KeyEntry]:
    """Parse a bridge keys file.

    Format:
        # Comment lines start with #
        # Blank lines are ignored
        <key> [: <profile_name>]

    The first colon on a line separates the key from the optional profile name.
    Keys are trimmed and compared case-sensitively. Duplicate keys raise ValueError.

    Args:
        path: Path to the keys file.

    Returns:
        List of KeyEntry objects.

    Raises:
        ValueError: If duplicate keys are found or file is malformed.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Keys file not found: {path}")

    entries: list[KeyEntry] = []
    seen_keys: dict[str, int] = {}

    for line_num, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue

        # Split on first colon
        if ":" in line:
            key_part, profile_part = line.split(":", 1)
            key = key_part.strip()
            profile = profile_part.strip() or None
        else:
            key = line.strip()
            profile = None

        if not key:
            raise ValueError(f"Keys file {path}:{line_num}: empty key")

        # Check for duplicates
        if key in seen_keys:
            raise ValueError(f"Duplicate key '{key}' found in {path} (lines {seen_keys[key]} and {line_num})")
        seen_keys[key] = line_num

        entries.append(KeyEntry(key=key, profile=profile))

    return entries
