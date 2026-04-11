"""Tests for ProfileResolver with balancing profiles — resolve_backend, resolve_balancing, defaults."""

import uuid

import pytest

from kitty.profiles.resolver import (
    NoDefaultProfileError,
    ProfileNotFoundError,
    ProfileResolver,
)
from kitty.profiles.schema import BalancingProfile, Profile
from kitty.profiles.store import ProfileStore

VALID_UUID = str(uuid.uuid4())


def _make_store(
    tmp_path, profiles_data: list[dict] | None = None, balancing_data: list[dict] | None = None
) -> ProfileStore:
    store = ProfileStore(path=tmp_path / "profiles.json")
    if profiles_data:
        for data in profiles_data:
            store.save(Profile(**data))
    if balancing_data:
        for data in balancing_data:
            store.save(BalancingProfile(**data))
    return store


BASE_PROFILES = [
    {"name": "alpha", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
    {"name": "beta", "provider": "novita", "model": "gpt-4o-mini", "auth_ref": VALID_UUID},
    {"name": "gamma", "provider": "openrouter", "model": "claude-3.5", "auth_ref": VALID_UUID},
]


class TestResolveBackend:
    def test_resolve_backend_returns_regular_profile(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES[:1])
        resolver = ProfileResolver(store)
        result = resolver.resolve_backend("alpha")
        assert isinstance(result, Profile)
        assert result.name == "alpha"

    def test_resolve_backend_returns_balancing_profile(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES, [{"name": "bal", "members": ["alpha", "beta"]}])
        resolver = ProfileResolver(store)
        result = resolver.resolve_backend("bal")
        assert isinstance(result, BalancingProfile)
        assert result.members == ["alpha", "beta"]

    def test_resolve_backend_not_found_raises(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES[:1])
        resolver = ProfileResolver(store)
        with pytest.raises(ProfileNotFoundError, match="ghost"):
            resolver.resolve_backend("ghost")

    def test_resolve_backend_case_insensitive(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES, [{"name": "bal", "members": ["alpha", "beta"]}])
        resolver = ProfileResolver(store)
        assert resolver.resolve_backend("BAL").name == "bal"


class TestResolveBalancing:
    def test_resolve_balancing_returns_member_profiles(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES, [{"name": "bal", "members": ["alpha", "beta"]}])
        resolver = ProfileResolver(store)
        members = resolver.resolve_balancing("bal")
        assert len(members) == 2
        assert members[0].name == "alpha"
        assert members[1].name == "beta"

    def test_resolve_balancing_with_three_members(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES, [{"name": "bal", "members": ["alpha", "beta", "gamma"]}])
        resolver = ProfileResolver(store)
        members = resolver.resolve_balancing("bal")
        assert [m.name for m in members] == ["alpha", "beta", "gamma"]

    def test_resolve_balancing_raises_for_regular_profile(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES[:1])
        resolver = ProfileResolver(store)
        with pytest.raises(ProfileNotFoundError, match="alpha"):
            resolver.resolve_balancing("alpha")

    def test_resolve_balancing_raises_for_missing_member(self, tmp_path):
        """If a member profile was deleted, resolve should raise."""
        store = _make_store(tmp_path, BASE_PROFILES[:1], [{"name": "bal", "members": ["alpha", "deleted"]}])
        resolver = ProfileResolver(store)
        with pytest.raises(ProfileNotFoundError, match="deleted"):
            resolver.resolve_balancing("bal")

    def test_resolve_balancing_not_found_raises(self, tmp_path):
        store = _make_store(tmp_path)
        resolver = ProfileResolver(store)
        with pytest.raises(ProfileNotFoundError, match="ghost"):
            resolver.resolve_balancing("ghost")


class TestResolveDefaultBackend:
    def test_resolve_default_backend_regular(self, tmp_path):
        store = _make_store(
            tmp_path,
            [
                {
                    "name": "alpha",
                    "provider": "zai_regular",
                    "model": "gpt-4o",
                    "auth_ref": VALID_UUID,
                    "is_default": True,
                }
            ],
        )
        resolver = ProfileResolver(store)
        result = resolver.resolve_default_backend()
        assert isinstance(result, Profile) and result.name == "alpha"

    def test_resolve_default_backend_balancing(self, tmp_path):
        store = _make_store(
            tmp_path,
            BASE_PROFILES,
            [{"name": "bal", "members": ["alpha", "beta"], "is_default": True}],
        )
        resolver = ProfileResolver(store)
        result = resolver.resolve_default_backend()
        assert isinstance(result, BalancingProfile) and result.name == "bal"

    def test_resolve_default_backend_raises_when_none_set(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES)
        resolver = ProfileResolver(store)
        with pytest.raises(NoDefaultProfileError):
            resolver.resolve_default_backend()

    def test_resolve_default_backend_raises_when_empty(self, tmp_path):
        store = _make_store(tmp_path)
        resolver = ProfileResolver(store)
        with pytest.raises(NoDefaultProfileError):
            resolver.resolve_default_backend()


class TestListProfilesBackwardCompat:
    def test_list_profiles_returns_only_regular(self, tmp_path):
        store = _make_store(tmp_path, BASE_PROFILES[:2], [{"name": "bal", "members": ["alpha", "beta"]}])
        resolver = ProfileResolver(store)
        profiles = resolver.list_profiles()
        assert all(isinstance(p, Profile) for p in profiles)
        assert len(profiles) == 2
