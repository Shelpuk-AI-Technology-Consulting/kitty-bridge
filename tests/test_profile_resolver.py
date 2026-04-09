"""Tests for profiles/resolver.py — Profile lookup (explicit, default, case-insensitive)."""

import uuid

import pytest

from kitty.profiles.resolver import NoDefaultProfileError, ProfileNotFoundError, ProfileResolver
from kitty.profiles.store import ProfileStore

VALID_UUID = str(uuid.uuid4())


def _make_store(tmp_path, profiles_data: list[dict] | None = None) -> ProfileStore:
    from kitty.profiles.schema import Profile

    store = ProfileStore(path=tmp_path / "profiles.json")
    if profiles_data:
        for data in profiles_data:
            store.save(Profile(**data))
    return store


class TestResolveExplicit:
    def test_resolve_explicit_profile_by_name(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {"name": "alpha", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
                {"name": "beta", "provider": "novita", "model": "gpt-4o-mini", "auth_ref": VALID_UUID},
            ],
        )
        resolver = ProfileResolver(store)
        profile = resolver.resolve("alpha")
        assert profile.name == "alpha"
        assert profile.provider == "zai_regular"

    def test_resolve_case_insensitive(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {"name": "my-profile", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
            ],
        )
        resolver = ProfileResolver(store)
        assert resolver.resolve("MY-PROFILE").name == "my-profile"
        assert resolver.resolve("My-Profile").name == "my-profile"

    def test_resolve_explicit_not_found_raises(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {"name": "alpha", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
            ],
        )
        resolver = ProfileResolver(store)
        with pytest.raises(ProfileNotFoundError, match="ghost"):
            resolver.resolve("ghost")


class TestResolveDefault:
    def test_resolve_default(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {
                    "name": "alpha",
                    "provider": "zai_regular",
                    "model": "gpt-4o",
                    "auth_ref": VALID_UUID,
                    "is_default": True,
                },
                {"name": "beta", "provider": "novita", "model": "gpt-4o-mini", "auth_ref": VALID_UUID},
            ],
        )
        resolver = ProfileResolver(store)
        profile = resolver.resolve(None)
        assert profile.name == "alpha"
        assert profile.is_default is True

    def test_resolve_default_raises_when_none_set(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {"name": "alpha", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
            ],
        )
        resolver = ProfileResolver(store)
        with pytest.raises(NoDefaultProfileError):
            resolver.resolve(None)

    def test_resolve_default_raises_when_empty_store(self, tmp_path):
        store = _make_store(tmp_path)
        resolver = ProfileResolver(store)
        with pytest.raises(NoDefaultProfileError):
            resolver.resolve(None)


class TestListProfiles:
    def test_list_returns_all_profiles(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {"name": "alpha", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
                {"name": "beta", "provider": "novita", "model": "gpt-4o-mini", "auth_ref": VALID_UUID},
            ],
        )
        resolver = ProfileResolver(store)
        profiles = resolver.list_profiles()
        assert len(profiles) == 2
        assert {p.name for p in profiles} == {"alpha", "beta"}

    def test_list_returns_empty_on_empty_store(self, tmp_path):
        store = _make_store(tmp_path)
        resolver = ProfileResolver(store)
        assert resolver.list_profiles() == []
