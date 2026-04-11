"""Profile lookup — explicit name, default, case-insensitive matching."""

from __future__ import annotations

from kitty.profiles.schema import BackendConfig, BalancingProfile, Profile
from kitty.profiles.store import ProfileStore


class ProfileNotFoundError(Exception):
    """Raised when a requested profile name does not exist."""


class NoDefaultProfileError(Exception):
    """Raised when no default profile is set."""


class ProfileResolver:
    """Resolves profiles by explicit name or default selection."""

    def __init__(self, store: ProfileStore) -> None:
        self._store = store

    def resolve(self, name: str | None) -> Profile:
        """Resolve a regular profile by explicit name or default.

        Args:
            name: Profile name (case-insensitive) or None for default.

        Returns:
            The resolved Profile.

        Raises:
            ProfileNotFoundError: If name is given but no matching profile exists.
            NoDefaultProfileError: If name is None but no default profile is set.
        """
        if name is not None:
            profile = self._store.get(name)
            if profile is None:
                raise ProfileNotFoundError(f"Profile {name!r} not found")
            return profile
        return self.resolve_default()

    def resolve_default(self) -> Profile:
        """Resolve the default profile. Raises NoDefaultProfileError if none set."""
        for profile in self._store.load_all():
            if profile.is_default:
                return profile
        raise NoDefaultProfileError("No default profile set")

    def resolve_backend(self, name: str | None) -> BackendConfig:
        """Resolve any backend (Profile or BalancingProfile) by name or default.

        Raises:
            ProfileNotFoundError: If name is given but no matching backend exists.
            NoDefaultProfileError: If name is None but no default is set.
        """
        if name is not None:
            backend = self._store.get_backend(name)
            if backend is None:
                raise ProfileNotFoundError(f"Profile {name!r} not found")
            return backend
        return self.resolve_default_backend()

    def resolve_default_backend(self) -> BackendConfig:
        """Resolve the default backend (any type). Raises NoDefaultProfileError if none set."""
        for backend in self._store.get_all_backends():
            if backend.is_default:
                return backend
        raise NoDefaultProfileError("No default profile set")

    def resolve_balancing(self, name: str) -> list[Profile]:
        """Resolve a balancing profile to its ordered list of member Profiles.

        Raises:
            ProfileNotFoundError: If the balancing profile or any member is not found.
        """
        backend = self._store.get_backend(name)
        if not isinstance(backend, BalancingProfile):
            raise ProfileNotFoundError(f"Balancing profile {name!r} not found")
        members: list[Profile] = []
        for member_name in backend.members:
            profile = self._store.get(member_name)
            if profile is None:
                raise ProfileNotFoundError(f"Member profile {member_name!r} not found")
            members.append(profile)
        return members

    def list_profiles(self) -> list[Profile]:
        """Return all regular profiles in the store."""
        return self._store.load_all()


__all__ = ["NoDefaultProfileError", "ProfileNotFoundError", "ProfileResolver"]
