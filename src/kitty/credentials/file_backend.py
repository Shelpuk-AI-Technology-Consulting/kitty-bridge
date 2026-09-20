"""Fallback file-based credential backend with restrictive permissions."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import filelock
from platformdirs import user_config_dir

from kitty.credentials.store import CredentialBackend, CredentialError

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
        # F38: Explicit timeout prevents indefinite hang on stale lock from crashed process.
        self._lock = filelock.FileLock(str(path) + ".lock", timeout=5)

    def get(self, ref: str) -> str | None:
        """Retrieve a credential by reference.

        Returns:
            The credential value, or ``None`` when the reference is absent.
            ``None`` is reserved for "no such credential": a stored value that no
            longer decodes raises :class:`CredentialError` (KBR-87), so store
            damage is distinguishable from a missing key at this boundary.

        Raises:
            CredentialError: (a) When the reference exists but its stored value
                is undecodable — not valid base64, not decodable as UTF-8, or
                not a string (KBR-87). (b) When the credentials file itself is
                damaged — not valid UTF-8 bytes, or its JSON parses to a
                non-dict (KBR-291). The message names the file path and the
                ``*.corrupt.<ts>.<pid>`` backup holding the original bytes.
                ``set``/``delete`` swallow the file-level raise via
                :meth:`_read_raw_for_write` so the recovery command
                (``kitty setup``) does not crash on its own write.
        """
        try:
            with self._lock:
                data = self._read_raw()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        encoded = data.get(ref)
        if encoded is None:
            return None
        # Corruption is store damage, not absence (KBR-87). The handler covers
        # every measured shape: binascii.Error (invalid base64 alphabet and
        # invalid base64 length) and UnicodeDecodeError both subclass ValueError,
        # so `except (ValueError, TypeError)` is exactly as narrow as naming the
        # subtypes and complete over every shape. The non-ASCII ValueError that
        # `"mötley-key"` produces is raised by `b64decode`'s ASCII-encode step,
        # not by the validate= regex — validate=False raises the same error there.
        # `validate=True` is the load-bearing choice for inputs that happen to
        # alphabet-strip to a valid length (e.g. `"ab@=="` silently decodes to
        # `"i"` under validate=False and rejects under validate=True); the pin is
        # in `tests/test_credential_store.py`.
        try:
            return base64.b64decode(encoded, validate=True).decode("utf-8")
        except (ValueError, TypeError) as exc:
            raise CredentialError(
                f"Credential for ref {ref!r} is corrupt ({type(exc).__name__}): "
                "restore it from a backup or re-enter the credential."
            ) from exc

    def set(self, ref: str, value: str) -> None:
        with self._lock:
            data = self._read_raw_for_write()
            data[ref] = base64.b64encode(value.encode("utf-8")).decode("ascii")
            self._write_raw(data)
            self._set_permissions()

    def delete(self, ref: str) -> None:
        with self._lock:
            data = self._read_raw_for_write()
            data.pop(ref, None)
            self._write_raw(data)

    def _read_raw(self) -> dict[str, str]:
        """Read and parse the credentials file.

        Returns:
            The parsed ``{ref: base64}`` dict, or an empty dict when the
            file is absent or its JSON is invalid (the F37 path: invalid
            JSON is backed up to ``*.corrupt.<ts>.<pid>``, the store
            resets to ``{}``, and ``get`` returns ``None`` — a
            deliberately different signal from the file-level raises
            below).

        Raises:
            CredentialError: When the file exists but is damaged — its
                bytes are not valid UTF-8 (shape b, KBR-291) or its JSON
                parses to a non-dict (shape a, KBR-291). The corrupt
                file is backed up to ``*.corrupt.<ts>.<pid>`` before
                raising, so the next write cannot silently destroy it.
                ``FileBackend.set`` / ``delete`` swallow this via
                :meth:`_read_raw_for_write`; ``get`` lets it propagate
                so the KBR-87 receiver map produces the clean message.
        """
        # Read the file as UTF-8 text. A UnicodeDecodeError means the bytes
        # themselves are damaged — F37's gap for shape (b). os.replace is
        # bytes-level (no decode needed), so the backup preserves the
        # original bytes verbatim; recreating an empty file adds nothing.
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return {}
        except UnicodeDecodeError as exc:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = self._path.with_suffix(f".json.corrupt.{ts}.{os.getpid()}")
            logger.critical(
                "Credentials file %s is corrupt (not valid UTF-8). "
                "Backing up to %s. All previously stored API keys may be lost!",
                self._path,
                backup,
            )
            with contextlib.suppress(OSError):
                os.replace(self._path, backup)
            raise CredentialError(
                f"Credentials file {self._path} is corrupt "
                f"(not valid UTF-8: {exc}). "
                "Run 'kitty setup' to reconfigure stored credentials; "
                f"the original bytes are preserved at {backup}."
            ) from exc

        # Parse JSON. F37: back up the corrupt file before returning empty
        # so the next write cannot silently overwrite it.
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = self._path.with_suffix(f".json.corrupt.{ts}.{os.getpid()}")
            logger.critical(
                "Credentials file %s is corrupt (not valid JSON). "
                "Backing up to %s and starting fresh. "
                "All previously stored API keys may be lost!",
                self._path,
                backup,
            )
            with contextlib.suppress(OSError):
                os.replace(self._path, backup)
            # Create a fresh empty file so future writes never overwrite the
            # corrupt original (which now lives at backup path).
            with contextlib.suppress(OSError):
                self._write_raw({})
            return {}

        # Validate shape — valid JSON that is not an object is file damage
        # (shape a, KBR-291). Without the backup the next write destroys
        # the original silently; with it, the backup is the recovered
        # record. No chaining: the JSON parsed cleanly, so there is no
        # underlying exception to preserve.
        #
        # We do not write `{}` after the backup — the file at `self._path`
        # is left absent. That way, when `os.replace` succeeds the path is
        # gone and `_read_raw_for_write` can swallow via its
        # `self._path.exists()` guard; when `os.replace` fails (read-only
        # mount, etc.) the path still holds the damaged original and the
        # guard propagates instead — the recovery command sees the honest
        # "backup could not be made" message rather than silently
        # overwriting the user's data with no backup anywhere.
        if not isinstance(result, dict):
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = self._path.with_suffix(f".json.corrupt.{ts}.{os.getpid()}")
            logger.critical(
                "Credentials file %s is corrupt (top-level %s, expected object). "
                "Backing up to %s. All previously stored API keys may be lost!",
                self._path,
                type(result).__name__,
                backup,
            )
            with contextlib.suppress(OSError):
                os.replace(self._path, backup)
            raise CredentialError(
                f"Credentials file {self._path} is corrupt "
                f"(top-level {type(result).__name__}, expected object). "
                "Run 'kitty setup' to reconfigure stored credentials; "
                f"the original is preserved at {backup}."
            )

        return result

    def _read_raw_for_write(self) -> dict[str, str]:
        """Read for the write path: a damaged file reads as empty.

        ``get`` raises on file-level damage (the read signal the user
        needs); the write path forgives it — the user is overwriting the
        file anyway, and ``kitty setup`` (the recovery command the error
        message names) reaches a ``set`` call, so raising here would
        crash the very command the message names (KBR-291 D6). The
        CRITICAL log + backup still fire from ``_read_raw``; nothing is
        silent.

        The one exception: if the file is still at ``self._path`` after
        the raise, the backup failed (``os.replace`` is wrapped in
        ``contextlib.suppress(OSError)`` for the read-only-mount case).
        Letting the write proceed would overwrite the still-damaged
        original with no backup anywhere — the user was promised the
        original is preserved, and would silently lose it. Surface the
        exception instead so the recovery command reports the failure.

        Returns:
            The parsed data, or an empty dict when the file is absent
            or its backup succeeded.
        """
        try:
            return self._read_raw()
        except CredentialError:
            if self._path.exists():
                # Backup failed; the original is still at self._path.
                # Don't let the write silently overwrite it.
                raise
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
