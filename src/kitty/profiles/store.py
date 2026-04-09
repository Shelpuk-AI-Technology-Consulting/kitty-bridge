"""Profile CRUD, versioned JSON storage with filelock and atomic writes."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import filelock
from platformdirs import user_config_dir

from kitty.profiles.schema import Profile

logger = logging.getLogger(__name__)

STORE_VERSION = 1


class ProfileStore:
    """Persistent store for Profile objects backed by a JSON file.

    Uses filelock for concurrent access safety and atomic writes
    (write to temp file + os.replace) for crash resilience.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            config_dir = Path(user_config_dir("kitty"))
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / "profiles.json"
        self._path = path
        self._lock = filelock.FileLock(str(path) + ".lock")

    def load_all(self) -> list[Profile]:
        """Read all profiles from the store. Returns empty list on missing/corrupt file."""
        try:
            with self._lock:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            logger.warning("Store version mismatch or corrupt data in %s", self._path)
            return []

        profiles: list[Profile] = []
        for item in data.get("profiles", []):
            try:
                profiles.append(Profile.model_validate(item))
            except Exception:
                logger.warning("Skipping invalid profile entry in store")
        return profiles

    def save(self, profile: Profile) -> None:
        """Upsert a profile. Enforces single-default invariant atomically."""
        with self._lock:
            profiles = self._load_raw()
            # Remove existing entry with same name (case-insensitive)
            name_lower = profile.name.lower()
            profiles = [p for p in profiles if p["name"].lower() != name_lower]
            # If this profile is default, clear previous default
            if profile.is_default:
                for p in profiles:
                    p["is_default"] = False
            profiles.append(profile.model_dump(mode="json"))
            self._write_raw(profiles)

    def delete(self, name: str) -> None:
        """Delete a profile by name (case-insensitive). No-op if not found."""
        with self._lock:
            profiles = self._load_raw()
            name_lower = name.lower()
            original_len = len(profiles)
            profiles = [p for p in profiles if p["name"].lower() != name_lower]
            if len(profiles) != original_len:
                self._write_raw(profiles)

    def get(self, name: str) -> Profile | None:
        """Get a profile by name (case-insensitive). Returns None if not found."""
        name_lower = name.lower()
        for profile in self.load_all():
            if profile.name.lower() == name_lower:
                return profile
        return None

    def _load_raw(self) -> list[dict[str, Any]]:
        """Load raw profile dicts from JSON file (caller must hold lock)."""
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

        if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
            return []
        return list(data.get("profiles", []))

    def _write_raw(self, profiles: list[dict[str, Any]]) -> None:
        """Write profiles to JSON file atomically (caller must hold lock)."""
        data = {"version": STORE_VERSION, "profiles": profiles}
        content = json.dumps(data, indent=2, ensure_ascii=False)
        # Atomic write: temp file in same dir, then os.replace
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=self._path.stem + ".",
            dir=self._path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, self._path)
        except BaseException:
            # Clean up temp file on any error
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


__all__ = ["STORE_VERSION", "ProfileStore"]
