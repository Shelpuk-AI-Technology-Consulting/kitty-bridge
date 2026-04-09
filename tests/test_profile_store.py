"""Tests for profiles/store.py — Profile CRUD, atomic storage, concurrent access."""

import json
import uuid

from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore

VALID_UUID = str(uuid.uuid4())


def _make_profile(name: str = "test", **overrides: object) -> Profile:
    defaults: dict = {"name": name, "provider": "zai_regular", "model": "gpt-4o", "auth_ref": VALID_UUID}
    defaults.update(overrides)
    return Profile(**defaults)


class TestProfileStoreSaveAndGet:
    def test_save_and_retrieve(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        profile = _make_profile("my-profile")
        store.save(profile)
        retrieved = store.get("my-profile")
        assert retrieved is not None
        assert retrieved.name == "my-profile"

    def test_save_multiple_profiles(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        all_profiles = store.load_all()
        names = {p.name for p in all_profiles}
        assert names == {"alpha", "beta"}

    def test_upsert_replaces_existing(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha", model="gpt-4o"))
        store.save(_make_profile("alpha", model="gpt-4o-mini"))
        retrieved = store.get("alpha")
        assert retrieved is not None
        assert retrieved.model == "gpt-4o-mini"

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        assert store.get("ghost") is None


class TestProfileStoreDefault:
    def test_save_default_replaces_previous(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha", is_default=True))
        store.save(_make_profile("beta", is_default=True))
        all_profiles = store.load_all()
        defaults = [p for p in all_profiles if p.is_default]
        assert len(defaults) == 1
        assert defaults[0].name == "beta"
        # alpha should no longer be default
        alpha = next(p for p in all_profiles if p.name == "alpha")
        assert alpha.is_default is False


class TestProfileStoreDelete:
    def test_delete_profile(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.save(_make_profile("beta"))
        store.delete("alpha")
        assert store.get("alpha") is None
        assert store.get("beta") is not None

    def test_delete_nonexistent_is_noop(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.delete("ghost")  # should not raise


class TestProfileStoreCaseInsensitive:
    def test_get_is_case_insensitive(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("my-profile"))
        assert store.get("MY-PROFILE") is not None
        assert store.get("My-Profile") is not None

    def test_delete_is_case_insensitive(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        store.delete("ALPHA")
        assert store.get("alpha") is None


class TestProfileStoreMissingFile:
    def test_load_all_returns_empty_on_missing_file(self, tmp_path):
        store = ProfileStore(path=tmp_path / "nonexistent.json")
        assert store.load_all() == []

    def test_get_returns_none_on_missing_file(self, tmp_path):
        store = ProfileStore(path=tmp_path / "nonexistent.json")
        assert store.get("any") is None


class TestProfileStoreCorruptData:
    def test_load_all_returns_empty_on_corrupt_json(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text("{invalid json")
        store = ProfileStore(path=path)
        assert store.load_all() == []

    def test_load_all_returns_empty_on_version_mismatch(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps({"version": 999, "profiles": []}))
        store = ProfileStore(path=path)
        assert store.load_all() == []

    def test_atomic_write_survives_partial_write(self, tmp_path):
        """Verify that a partial temp file does not corrupt the store."""
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("alpha"))
        # Simulate a partial temp file leftover
        (tmp_path / "profiles.json.tmp").write_text("{partial")
        # Should still read the last good state
        profiles = store.load_all()
        assert len(profiles) == 1
        assert profiles[0].name == "alpha"


class TestProfileStoreConcurrentAccess:
    def test_concurrent_saves_do_not_corrupt(self, tmp_path):
        store = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("initial"))
        # Simulate concurrent save by creating a second store instance
        store2 = ProfileStore(path=tmp_path / "profiles.json")
        store.save(_make_profile("from-store1"))
        store2.save(_make_profile("from-store2"))
        all_profiles = store.load_all()
        names = {p.name for p in all_profiles}
        assert "from-store1" in names
        assert "from-store2" in names


class TestProfileStoreXDGPath:
    def test_default_path_uses_config_dir(self, tmp_path, monkeypatch):
        """Verify the store uses platformdirs config dir (patched)."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        monkeypatch.setattr("kitty.profiles.store.user_config_dir", lambda app: str(config_dir / app))
        store = ProfileStore()  # uses default path
        store.save(_make_profile("xdg-test"))
        expected_file = config_dir / "kitty" / "profiles.json"
        assert expected_file.exists()
