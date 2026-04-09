"""Credential backend interface and credential store with fallback chain."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

from kitty.profiles.schema import Profile


class CredentialBackend(ABC):
    """Interface for credential storage backends."""

    @abstractmethod
    def get(self, ref: str) -> str | None:
        """Retrieve a credential by reference. Returns None if not found."""

    @abstractmethod
    def set(self, ref: str, value: str) -> None:
        """Store a credential by reference."""

    @abstractmethod
    def delete(self, ref: str) -> None:
        """Delete a credential by reference."""


class CredentialNotFoundError(Exception):
    """Raised when a credential cannot be resolved from any backend."""


class CredentialStore:
    """Credential store that tries backends in order (fallback chain)."""

    def __init__(self, backends: list[CredentialBackend]) -> None:
        self._backends = backends

    def get(self, ref: str) -> str | None:
        """Try each backend in order. Returns the first non-None result."""
        for backend in self._backends:
            value = backend.get(ref)
            if value is not None:
                return value
        return None

    def set(self, ref: str, value: str, backend_index: int = 0) -> None:
        """Write a credential to the specified backend."""
        self._backends[backend_index].set(ref, value)

    def delete(self, ref: str) -> None:
        """Delete a credential from all backends."""
        for backend in self._backends:
            with contextlib.suppress(Exception):
                backend.delete(ref)

    def resolve(self, profile: Profile) -> str:
        """Resolve a profile's auth_ref to an API key.

        Raises:
            CredentialNotFoundError: If no backend contains the credential.
        """
        value = self.get(profile.auth_ref)
        if value is None:
            raise CredentialNotFoundError(f"Credential for auth_ref {profile.auth_ref!r} not found in any backend")
        return value


__all__ = ["CredentialBackend", "CredentialNotFoundError", "CredentialStore"]
