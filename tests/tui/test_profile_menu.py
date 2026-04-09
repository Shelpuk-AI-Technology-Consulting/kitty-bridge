"""Tests for profile menu command -- interactive profile management."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from kitty.cli.profile_cmd import _create_profile_flow, run_profile_menu
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore


def _make_profile(
    name: str = "test-profile",
    provider: str = "zai_regular",
    model: str = "gpt-4o",
    is_default: bool = False,
) -> Profile:
    return Profile(
        name=name,
        provider=provider,
        model=model,
        auth_ref=str(uuid.uuid4()),
        is_default=is_default,
    )


@pytest.fixture()
def store(tmp_path: object) -> ProfileStore:
    return ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]


class TestRunProfileMenu:
    def test_non_tty_raises(self, store: ProfileStore) -> None:
        """Profile menu rejects non-TTY with deterministic error."""
        with patch("sys.stdin.isatty", return_value=False), \
             pytest.raises(Exception, match="interactive"):
            run_profile_menu(store)

    def test_list_profiles(self, store: ProfileStore) -> None:
        """Listing profiles shows all stored profiles."""
        store.save(_make_profile("dev", is_default=True))
        store.save(_make_profile("staging"))
        assert len(store.load_all()) == 2

    def test_create_profile_interactive(self, store: ProfileStore) -> None:
        """Create a new profile through the menu flow."""
        cred_store = CredentialStore(
            backends=[FileBackend(path=store._path.parent / "creds.json")]
        )

        with patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-4o", "myprofile"]), \
             patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="zai_regular"), \
             patch("kitty.cli.profile_cmd.prompt_secret", return_value="sk-test-key-123"), \
             patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True):
            profile = _create_profile_flow(store, cred_store)

        assert profile is not None
        assert profile.name == "myprofile"
        assert profile.provider == "zai_regular"
        assert profile.model == "gpt-4o"
        assert profile.is_default is True
        # Verify saved
        loaded = store.get("myprofile")
        assert loaded is not None

    def test_delete_profile(self, store: ProfileStore) -> None:
        """Delete a profile from the store."""
        store.save(_make_profile("to-delete"))
        assert store.get("to-delete") is not None
        store.delete("to-delete")
        assert store.get("to-delete") is None

    def test_set_default_profile(self, store: ProfileStore) -> None:
        """Setting a profile as default clears previous default."""
        store.save(_make_profile("first", is_default=True))
        store.save(_make_profile("second", is_default=False))

        # Manually set second as default
        second = store.get("second")
        assert second is not None
        updated = second.model_copy(update={"is_default": True})
        store.save(updated)

        first = store.get("first")
        assert first is not None
        assert first.is_default is False
        second_check = store.get("second")
        assert second_check is not None
        assert second_check.is_default is True
