"""Tests for ProfileStore with balancing profiles — save/load, type discriminator, member validation."""

import json
import uuid

from kitty.profiles.schema import BalancingProfile, Profile
from kitty.profiles.store import ProfileStore

VALID_UUID = str(uuid.uuid4())


def _make_profile(name: str = "test", **overrides: object) -> Profile:
    defaults: dict = {"name": name, "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID}
    defaults.update(overrides)
    return Profile(**defaults)


def _make_balancing(name: str = "bal", members: list[str] | None = None, **overrides: object) -> BalancingProfile:
    defaults: dict = {"name": name, "members": members or ["alpha", "beta"], "is_default": False}
    defaults.update(overrides)
    return BalancingProfile(**defaults)


class TestBalancingStoreSaveAndGet:
    def test_save_and_get_balancing_profile(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        # Save regular profiles first
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        # Save balancing profile
        bp = _make_balancing("my-bal", members=["alpha", "beta"])
        store.save(bp)
        retrieved = store.get_backend("my-bal")
        assert retrieved is not None
        assert isinstance(retrieved, BalancingProfile)
        assert retrieved.name == "my-bal"
        assert retrieved.members == ["alpha", "beta"]

    def test_get_backend_returns_profile_for_regular(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        retrieved = store.get_backend("alpha")
        assert isinstance(retrieved, Profile)
        assert retrieved.name == "alpha"

    def test_get_backend_returns_none_for_missing(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        assert store.get_backend("ghost") is None

    def test_get_regular_still_returns_none_for_balancing(self, tmp_path):
        """Backward compat: get() only returns Profile, not BalancingProfile."""
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("bal", members=["alpha", "beta"]))
        assert store.get("bal") is None  # get() is Profile-only
        assert store.get_backend("bal") is not None  # get_backend() works


class TestBalancingStoreLoadAll:
    def test_load_all_returns_only_regular_profiles(self, tmp_path):
        """Backward compat: load_all() returns only Profile instances."""
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("bal", members=["alpha", "beta"]))
        profiles = store.load_all()
        assert len(profiles) == 2
        assert all(isinstance(p, Profile) for p in profiles)

    def test_get_all_backends_returns_both_types(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("bal", members=["alpha", "beta"]))
        backends = store.get_all_backends()
        assert len(backends) == 3
        types = {(type(b).__name__) for b in backends}
        assert types == {"Profile", "BalancingProfile"}


class TestBalancingStoreUpsert:
    def test_upsert_replaces_balancing_profile(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_profile("gamma"))
        store.save(_make_balancing("bal", members=["alpha", "beta"]))
        store.save(_make_balancing("bal", members=["alpha", "gamma"]))
        retrieved = store.get_backend("bal")
        assert isinstance(retrieved, BalancingProfile)
        assert retrieved.members == ["alpha", "gamma"]


class TestBalancingStoreDelete:
    def test_delete_balancing_profile(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("bal", members=["alpha", "beta"]))
        store.delete("bal")
        assert store.get_backend("bal") is None
        # Regular profiles still there
        assert store.get("alpha") is not None


class TestBalancingStoreDefaultInvariant:
    def test_set_balancing_as_default_clears_previous(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha", is_default=True))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("bal", members=["alpha", "beta"], is_default=True))
        # alpha should no longer be default
        alpha = store.get("alpha")
        assert alpha is not None and not alpha.is_default
        bal = store.get_backend("bal")
        assert isinstance(bal, BalancingProfile) and bal.is_default

    def test_set_regular_as_default_clears_balancing_default(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("bal", members=["alpha", "beta"], is_default=True))
        store.save(_make_profile("gamma", is_default=True))
        bal = store.get_backend("bal")
        assert isinstance(bal, BalancingProfile) and not bal.is_default


class TestBalancingStoreBackwardCompat:
    def test_existing_json_without_type_field_loads_as_regular(self, tmp_path):
        """Simulate an old-format profiles.json with no type discriminator."""
        path = tmp_path / "profiles.json"
        old_data = {
            "version": 1,
            "profiles": [
                {"name": "legacy", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
            ],
        }
        path.write_text(json.dumps(old_data))
        store = ProfileStore(path=path)
        profiles = store.load_all()
        assert len(profiles) == 1
        assert profiles[0].name == "legacy"
        assert isinstance(profiles[0], Profile)

    def test_get_backend_loads_old_format_as_regular(self, tmp_path):
        path = tmp_path / "profiles.json"
        old_data = {
            "version": 1,
            "profiles": [
                {"name": "legacy", "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID},
            ],
        }
        path.write_text(json.dumps(old_data))
        store = ProfileStore(path=path)
        backend = store.get_backend("legacy")
        assert isinstance(backend, Profile)


class TestBalancingStoreCaseInsensitive:
    def test_get_backend_is_case_insensitive(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.save(_make_balancing("my-bal", members=["alpha", "beta"]))
        assert store.get_backend("MY-BAL") is not None
        assert store.get_backend("My-Bal") is not None
