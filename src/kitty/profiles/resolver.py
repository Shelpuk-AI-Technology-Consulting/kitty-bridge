"""Profile lookup — explicit name, default, case-insensitive matching."""

from __future__ import annotations

from kitty.profiles.schema import Profile
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
        """Resolve a profile by explicit name or default.

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

    def list_profiles(self) -> list[Profile]:
        """Return all profiles in the store."""
        return self._store.load_all()


__all__ = ["NoDefaultProfileError", "ProfileNotFoundError", "ProfileResolver"]
