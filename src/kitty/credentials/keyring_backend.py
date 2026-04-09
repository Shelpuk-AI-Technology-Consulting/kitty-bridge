"""OS-native keyring credential backend."""

from __future__ import annotations

import contextlib

import keyring

from kitty.credentials.store import CredentialBackend


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
        keyring.set_password(self._SERVICE, ref, value)

    def delete(self, ref: str) -> None:
        with contextlib.suppress(keyring.errors.PasswordDeleteError):
            keyring.delete_password(self._SERVICE, ref)


__all__ = ["KeyringBackend"]
