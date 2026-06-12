"""OS-native keyring credential backend."""

from __future__ import annotations

import contextlib
import logging

import keyring

from kitty.credentials.store import CredentialBackend, CredentialError

logger = logging.getLogger(__name__)


class KeyringBackend(CredentialBackend):
    """Credential backend that delegates to the OS-native keyring.

    Uses ``keyring`` package which abstracts Windows Credential Manager,
    macOS Keychain, and Linux Secret Service.
    """

    _SERVICE = "kitty"

    def get(self, ref: str) -> str | None:
        try:
            value = keyring.get_password(self._SERVICE, ref)
            return value
        except Exception:
            return None

    def set(self, ref: str, value: str) -> None:
        # F39: Wrap in try/except — keyring crashes on headless Linux without D-Bus.
        try:
            keyring.set_password(self._SERVICE, ref, value)
        except Exception as exc:
            logger.error("keyring set_password failed for ref=%r: %s", ref, exc)
            raise CredentialError(
                f"Failed to store credential in OS keyring: {exc}. "
                "Consider using file-based credential storage as a fallback."
            ) from exc

    def delete(self, ref: str) -> None:
        with contextlib.suppress(keyring.errors.PasswordDeleteError):
            keyring.delete_password(self._SERVICE, ref)


__all__ = ["KeyringBackend"]
