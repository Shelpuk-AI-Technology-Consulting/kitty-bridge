"""Fallback file-based credential backend with restrictive permissions."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import filelock
from platformdirs import user_config_dir

from kitty.credentials.store import CredentialBackend

logger = logging.getLogger(__name__)


class FileBackend(CredentialBackend):
    """File-based credential backend with base64 encoding.

    Storage format: JSON ``{ref: base64_value}``.
    File permissions: POSIX 0600 (file), 0700 (directory).
    Uses filelock + atomic write-via-tempfile-rename for safety.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            config_dir = Path(user_config_dir("kitty"))
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / "credentials.json"
        self._path = path
        self._lock = filelock.FileLock(str(path) + ".lock")

    def get(self, ref: str) -> str | None:
        try:
            with self._lock:
                data = self._read_raw()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        encoded = data.get(ref)
        if encoded is None:
            return None
        try:
            return base64.b64decode(encoded).decode("utf-8")
        except Exception:
            return None

    def set(self, ref: str, value: str) -> None:
        with self._lock:
            data = self._read_raw()
            data[ref] = base64.b64encode(value.encode("utf-8")).decode("ascii")
            self._write_raw(data)
            self._set_permissions()

    def delete(self, ref: str) -> None:
        with self._lock:
            data = self._read_raw()
            data.pop(ref, None)
            self._write_raw(data)

    def _read_raw(self) -> dict[str, str]:
        try:
            raw = self._path.read_text(encoding="utf-8")
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
            return {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write_raw(self, data: dict[str, Any]) -> None:
        content = json.dumps(data, indent=2, ensure_ascii=False)
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
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _set_permissions(self) -> None:
        """Set restrictive permissions on the credentials file and its directory."""
        if os.name == "posix":
            try:
                os.chmod(self._path, 0o600)
                os.chmod(self._path.parent, 0o700)
            except OSError:
                pass


__all__ = ["FileBackend"]
