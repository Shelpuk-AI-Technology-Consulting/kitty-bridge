"""Tests for profile menu command -- interactive profile management (questionary-based)."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import BalancingProfile, Profile
from kitty.profiles.store import ProfileStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_balancing(name: str, members: list[str], is_default: bool = False) -> BalancingProfile:
    return BalancingProfile(name=name, members=members, is_default=is_default)


@pytest.fixture()
def store(tmp_path):
    return ProfileStore(path=tmp_path / "profiles.json")


@pytest.fixture()
def cred_store(store):
    return CredentialStore(backends=[FileBackend(path=store._path.parent / "creds.json")])


# ---------------------------------------------------------------------------
# Lazy imports of functions under test — imported after fixtures so that
# mocking targets are resolved at test time, not import time.
# ---------------------------------------------------------------------------

def _import_create_profile_flow():
    from kitty.cli.profile_cmd import _create_profile_flow
    return _create_profile_flow


def _import_create_balancing_flow():
    from kitty.cli.profile_cmd import _create_balancing_flow
    return _create_balancing_flow


def _import_edit_profile_flow():
    from kitty.cli.profile_cmd import _edit_profile_flow
    return _edit_profile_flow


def _import_edit_balancing_flow():
    from kitty.cli.profile_cmd import _edit_balancing_flow
    return _edit_balancing_flow


def _import_find_reusable_auth_ref():
    from kitty.cli.profile_cmd import _find_reusable_auth_ref
    return _find_reusable_auth_ref


def _import_run_profile_menu():
    from kitty.cli.profile_cmd import run_profile_menu
    return run_profile_menu


# ---------------------------------------------------------------------------
# run_profile_menu guard
# ---------------------------------------------------------------------------

class TestRunProfileMenuGuard:
    def test_non_tty_raises(self, store: ProfileStore) -> None:
        """Profile menu rejects non-TTY with deterministic error."""
        run_profile_menu = _import_run_profile_menu()
        with patch("sys.stdin.isatty", return_value=False), pytest.raises(Exception, match="interactive"):
            run_profile_menu(store)


# ---------------------------------------------------------------------------
# _find_reusable_auth_ref
# ---------------------------------------------------------------------------

class TestFindReusableAuthRef:
    def test_returns_none_when_no_profiles(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        find = _import_find_reusable_auth_ref()
        assert find(store, cred_store, "zai_regular") is None

    def test_returns_none_when_no_same_provider(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        find = _import_find_reusable_auth_ref()
        p = _make_profile("p1", provider="minimax")
        store.save(p)
        cred_store.set(p.auth_ref, "key-123")
        assert find(store, cred_store, "zai_regular") is None

    def test_returns_auth_ref_when_same_provider_exists(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        find = _import_find_reusable_auth_ref()
        p = _make_profile("p1", provider="zai_regular")
        store.save(p)
        cred_store.set(p.auth_ref, "key-abc")
        result = find(store, cred_store, "zai_regular")
        assert result == p.auth_ref

    def test_returns_none_when_provider_key_missing(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        find = _import_find_reusable_auth_ref()
        p = _make_profile("p1", provider="zai_regular")
        store.save(p)
        # no credential stored
        assert find(store, cred_store, "zai_regular") is None


# ---------------------------------------------------------------------------
# _create_profile_flow (T12–T14)
# ---------------------------------------------------------------------------

class TestCreateProfileFlow:
    def test_creates_profile_with_new_key(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T12: No same-provider profile → fresh key prompt, profile saved."""
        create = _import_create_profile_flow()
        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="zai_regular"),
            patch("kitty.cli.profile_cmd._find_reusable_auth_ref", return_value=None),
            patch("kitty.cli.profile_cmd.prompt_secret", return_value="sk-test-key"),
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-4o", "myprofile"]),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            profile = create(store, cred_store)

        assert profile.name == "myprofile"
        assert profile.provider == "zai_regular"
        assert profile.model == "gpt-4o"
        assert profile.is_default is True
        # Key stored under fresh auth_ref
        assert cred_store.get(profile.auth_ref) == "sk-test-key"

    def test_reuses_key_when_user_accepts(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T13: Same-provider profile exists, user accepts reuse → shared auth_ref, no new credential."""
        create = _import_create_profile_flow()
        existing = _make_profile("existing", provider="zai_regular")
        store.save(existing)
        cred_store.set(existing.auth_ref, "existing-key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="zai_regular"),
            patch("kitty.cli.profile_cmd._find_reusable_auth_ref", return_value=existing.auth_ref),
            patch("kitty.cli.profile_cmd.prompt_confirm", side_effect=[True, True]),  # reuse=True, default=True
            patch("kitty.cli.profile_cmd.prompt_secret") as mock_secret,
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-4o", "newprofile"]),
        ):
            profile = create(store, cred_store)

        assert profile.auth_ref == existing.auth_ref
        mock_secret.assert_not_called()  # key was reused, not re-entered

    def test_prompts_new_key_when_user_declines_reuse(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T14: Same-provider profile exists, user declines reuse → fresh key prompt."""
        create = _import_create_profile_flow()
        existing = _make_profile("existing", provider="zai_regular")
        store.save(existing)
        cred_store.set(existing.auth_ref, "old-key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="zai_regular"),
            patch("kitty.cli.profile_cmd._find_reusable_auth_ref", return_value=existing.auth_ref),
            patch("kitty.cli.profile_cmd.prompt_confirm", side_effect=[False, True]),  # reuse=False, default=True
            patch("kitty.cli.profile_cmd.prompt_secret", return_value="new-key"),
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-4o", "newprofile"]),
        ):
            profile = create(store, cred_store)

        assert profile.auth_ref != existing.auth_ref
        assert cred_store.get(profile.auth_ref) == "new-key"

    def test_cancelled_provider_raises(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Cancelling provider selection raises NonTTYError."""
        from kitty.tui.prompts import NonTTYError
        create = _import_create_profile_flow()
        with patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value=None), pytest.raises(NonTTYError):
            create(store, cred_store)


# ---------------------------------------------------------------------------
# _create_balancing_flow (T15–T16)
# ---------------------------------------------------------------------------

class TestCreateBalancingFlow:
    def test_creates_balancing_profile_via_checkbox(self, store: ProfileStore) -> None:
        """T15: Checkbox returns ≥2 profiles → BalancingProfile saved."""
        create = _import_create_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        store.save(_make_profile("p3"))

        with (
            patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=["p1", "p2"]),
            patch("kitty.cli.profile_cmd.prompt_text", return_value="my-balancer"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            bp = create(store)

        assert bp.name == "my-balancer"
        assert set(bp.members) == {"p1", "p2"}
        assert bp.is_default is True

    def test_raises_value_error_on_fewer_than_two_selections(self, store: ProfileStore) -> None:
        """T16: Checkbox returns <2 → ValueError raised, nothing saved."""
        create = _import_create_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))

        cb_patch = patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=["p1"])
        with cb_patch, pytest.raises(ValueError, match="2"):
            create(store)

    def test_raises_value_error_on_empty_selection(self, store: ProfileStore) -> None:
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        create = _import_create_balancing_flow()
        with patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=[]), pytest.raises(ValueError):
            create(store)

    def test_raises_value_error_on_cancel(self, store: ProfileStore) -> None:
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        create = _import_create_balancing_flow()
        with patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=None), pytest.raises(ValueError):
            create(store)

    def test_raises_when_fewer_than_two_regular_profiles(self, store: ProfileStore) -> None:
        store.save(_make_profile("only-one"))
        create = _import_create_balancing_flow()
        with pytest.raises(ValueError, match="at least 2"):
            create(store)


# ---------------------------------------------------------------------------
# _edit_profile_flow (T17–T18)
# ---------------------------------------------------------------------------

class TestEditProfileFlow:
    def test_updates_model(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T17: User picks Model field → new model saved, auth_ref unchanged."""
        edit = _import_edit_profile_flow()
        p = _make_profile("myprofile", model="old-model")
        store.save(p)
        old_auth_ref = p.auth_ref

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="Model"),
            patch("kitty.cli.profile_cmd.prompt_text", return_value="new-model"),
        ):
            edit(store, cred_store, "myprofile")

        saved = store.get("myprofile")
        assert saved is not None
        assert saved.model == "new-model"
        assert saved.auth_ref == old_auth_ref  # auth_ref unchanged

    def test_updates_api_key_copy_on_write(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T18: User picks API Key field → new auth_ref created, old credential untouched."""
        edit = _import_edit_profile_flow()
        p = _make_profile("myprofile")
        store.save(p)
        cred_store.set(p.auth_ref, "original-key")
        old_auth_ref = p.auth_ref

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="API Key"),
            patch("kitty.cli.profile_cmd.prompt_secret", return_value="brand-new-key"),
        ):
            edit(store, cred_store, "myprofile")

        saved = store.get("myprofile")
        assert saved is not None
        assert saved.auth_ref != old_auth_ref  # new auth_ref (copy-on-write)
        assert cred_store.get(saved.auth_ref) == "brand-new-key"
        assert cred_store.get(old_auth_ref) == "original-key"  # original untouched

    def test_updates_both_fields(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        edit = _import_edit_profile_flow()
        p = _make_profile("myprofile", model="old-model")
        store.save(p)
        cred_store.set(p.auth_ref, "old-key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="Both"),
            patch("kitty.cli.profile_cmd.prompt_text", return_value="new-model"),
            patch("kitty.cli.profile_cmd.prompt_secret", return_value="new-key"),
        ):
            edit(store, cred_store, "myprofile")

        saved = store.get("myprofile")
        assert saved is not None
        assert saved.model == "new-model"
        assert cred_store.get(saved.auth_ref) == "new-key"

    def test_cancelled_edit_makes_no_changes(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        edit = _import_edit_profile_flow()
        p = _make_profile("myprofile", model="original-model")
        store.save(p)

        with patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value=None):
            edit(store, cred_store, "myprofile")

        saved = store.get("myprofile")
        assert saved is not None
        assert saved.model == "original-model"


# ---------------------------------------------------------------------------
# _edit_balancing_flow (T19–T21)
# ---------------------------------------------------------------------------

class TestEditBalancingFlow:
    def test_adds_member(self, store: ProfileStore) -> None:
        """T19: Add a member to balancing profile."""
        edit = _import_edit_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        store.save(_make_profile("p3"))
        bp = _make_balancing("bal", ["p1", "p2"])
        store.save(bp)

        with patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=["p1", "p2", "p3"]):
            edit(store, "bal")

        saved = store.get_backend("bal")
        assert isinstance(saved, BalancingProfile)
        assert set(saved.members) == {"p1", "p2", "p3"}

    def test_removes_member(self, store: ProfileStore) -> None:
        """T20: Remove a member → saves updated balancing profile."""
        edit = _import_edit_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        store.save(_make_profile("p3"))
        bp = _make_balancing("bal", ["p1", "p2", "p3"])
        store.save(bp)

        with patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=["p1", "p2"]):
            edit(store, "bal")

        saved = store.get_backend("bal")
        assert isinstance(saved, BalancingProfile)
        assert set(saved.members) == {"p1", "p2"}

    def test_rejects_fewer_than_two_members(self, store: ProfileStore) -> None:
        """T21: Confirming with <2 members shows error, does not save."""
        edit = _import_edit_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        bp = _make_balancing("bal", ["p1", "p2"])
        store.save(bp)

        with patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=["p1"]):
            edit(store, "bal")  # should print error, not raise

        saved = store.get_backend("bal")
        assert isinstance(saved, BalancingProfile)
        assert set(saved.members) == {"p1", "p2"}  # unchanged

    def test_cancelled_makes_no_changes(self, store: ProfileStore) -> None:
        edit = _import_edit_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        bp = _make_balancing("bal", ["p1", "p2"])
        store.save(bp)

        with patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=None):
            edit(store, "bal")  # cancelled → no changes

        saved = store.get_backend("bal")
        assert isinstance(saved, BalancingProfile)
        assert set(saved.members) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Duplicate name guard
# ---------------------------------------------------------------------------

class TestDuplicateNameGuard:
    def test_create_profile_rejects_duplicate_name_then_accepts_new(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """If a profile name already exists, show error and re-prompt until unique."""
        create = _import_create_profile_flow()
        store.save(_make_profile("taken"))

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="zai_regular"),
            patch("kitty.cli.profile_cmd._find_reusable_auth_ref", return_value=None),
            patch("kitty.cli.profile_cmd.prompt_secret", return_value="sk-key"),
            # first name attempt "taken" → duplicate; second attempt "fresh" → ok
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-4o", "taken", "fresh"]),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
            patch("kitty.cli.profile_cmd.print_error") as mock_error,
        ):
            profile = create(store, cred_store)

        assert profile.name == "fresh"
        mock_error.assert_called()  # at least one error printed for the duplicate

    def test_create_balancing_rejects_duplicate_name_then_accepts_new(self, store: ProfileStore) -> None:
        """Balancing profile creation also rejects duplicate names."""
        create = _import_create_balancing_flow()
        store.save(_make_profile("p1"))
        store.save(_make_profile("p2"))
        store.save(_make_balancing("taken", ["p1", "p2"]))

        with (
            patch("kitty.cli.profile_cmd.CheckboxMenu.show", return_value=["p1", "p2"]),
            # first name "taken" → duplicate; second "fresh-bal" → ok
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["taken", "fresh-bal"]),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=False),
            patch("kitty.cli.profile_cmd.print_error") as mock_error,
        ):
            bp = create(store)

        assert bp.name == "fresh-bal"
        mock_error.assert_called()


# ---------------------------------------------------------------------------
# Credential cleanup on profile delete
# ---------------------------------------------------------------------------

def _import_delete_profile_flow():
    from kitty.cli.profile_cmd import _delete_profile_flow
    return _delete_profile_flow


class TestDeleteCredentialCleanup:
    def test_deletes_orphaned_credential_when_last_user(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """When the last profile using an auth_ref is deleted, the credential is cleaned up."""
        delete = _import_delete_profile_flow()
        p = _make_profile("todelete")
        store.save(p)
        cred_store.set(p.auth_ref, "the-key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="todelete"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)

        assert store.get("todelete") is None
        assert cred_store.get(p.auth_ref) is None  # credential cleaned up

    def test_preserves_shared_credential_when_other_profile_uses_same_auth_ref(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """If another profile shares the same auth_ref, credential is NOT deleted."""
        delete = _import_delete_profile_flow()
        shared_ref = str(uuid.uuid4())
        p1 = Profile(name="p1", provider="zai_regular", model="m", auth_ref=shared_ref, is_default=False)
        p2 = Profile(name="p2", provider="minimax", model="m2", auth_ref=shared_ref, is_default=False)
        store.save(p1)
        store.save(p2)
        cred_store.set(shared_ref, "shared-key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="p1"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)

        assert store.get("p1") is None
        assert cred_store.get(shared_ref) == "shared-key"  # preserved — p2 still needs it


# ---------------------------------------------------------------------------
# Delete cascade: removing from balancing profiles
# ---------------------------------------------------------------------------

class TestDeleteWithBalancingCascade:
    def test_deleting_profile_removes_from_balancing(self, store, cred_store):
        """Deleting a profile auto-removes it from a balancing profile's members."""
        delete = _import_delete_profile_flow()
        p = _make_profile("target")
        store.save(p)
        store.save(_make_profile("other1"))
        store.save(_make_profile("other2"))
        store.save(_make_balancing("bal", ["target", "other1", "other2"]))
        cred_store.set(p.auth_ref, "key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="target"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)

        # Regular profile deleted
        assert store.get("target") is None
        # Balancing profile updated: target removed
        bal = store.get_backend("bal")
        assert isinstance(bal, BalancingProfile)
        assert "target" not in bal.members
        assert bal.members == ["other1", "other2"]

    def test_deleting_profile_cascades_balancing_below_minimum(self, store, cred_store):
        """When removal leaves a balancing profile with <2 members, it is deleted too."""
        delete = _import_delete_profile_flow()
        p = _make_profile("target")
        store.save(p)
        store.save(_make_profile("other"))
        store.save(_make_balancing("bal", ["target", "other"]))
        cred_store.set(p.auth_ref, "key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="target"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)

        # Both deleted
        assert store.get("target") is None
        assert store.get_backend("bal") is None

    def test_deleting_profile_removes_from_multiple_balancing_profiles(self, store, cred_store):
        """Profile in two balancing profiles → both updated."""
        delete = _import_delete_profile_flow()
        p = _make_profile("target")
        store.save(p)
        store.save(_make_profile("a"))
        store.save(_make_profile("b"))
        store.save(_make_profile("c"))
        store.save(_make_balancing("bal1", ["target", "a", "b"]))
        store.save(_make_balancing("bal2", ["target", "a", "c"]))

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="target"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)

        assert store.get("target") is None
        bal1 = store.get_backend("bal1")
        bal2 = store.get_backend("bal2")
        assert isinstance(bal1, BalancingProfile) and bal1.members == ["a", "b"]
        assert isinstance(bal2, BalancingProfile) and bal2.members == ["a", "c"]

    def test_deleting_balancing_profile_directly(self, store, cred_store):
        """Deleting a balancing profile (not a regular one) still works."""
        delete = _import_delete_profile_flow()
        store.save(_make_profile("a"))
        store.save(_make_profile("b"))
        store.save(_make_balancing("bal", ["a", "b"]))

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="bal"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)

        assert store.get_backend("bal") is None
        assert store.get("a") is not None
        assert store.get("b") is not None

    def test_cascaded_balancing_deletion_no_credential_error(self, store, cred_store):
        """Deleting a profile that cascades a balancing delete does not try to clean balancing's auth_ref."""
        delete = _import_delete_profile_flow()
        p = _make_profile("target")
        store.save(p)
        store.save(_make_profile("other"))
        store.save(_make_balancing("bal", ["target", "other"]))
        cred_store.set(p.auth_ref, "key")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="target"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            delete(store, cred_store)  # should not raise

        # Credential cleaned up (target was the only user)
        assert cred_store.get(p.auth_ref) is None

    def test_cancel_does_not_modify_balancing_profiles(self, store, cred_store):
        """Cancelling at the confirmation prompt leaves balancing profiles untouched."""
        delete = _import_delete_profile_flow()
        p = _make_profile("target")
        store.save(p)
        store.save(_make_profile("other1"))
        store.save(_make_profile("other2"))
        store.save(_make_balancing("bal", ["target", "other1", "other2"]))

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="target"),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=False),
        ):
            delete(store, cred_store)

        # Nothing changed
        assert store.get("target") is not None
        bal = store.get_backend("bal")
        assert isinstance(bal, BalancingProfile)
        assert bal.members == ["target", "other1", "other2"]


# ---------------------------------------------------------------------------
# OAuth provider create flow
# ---------------------------------------------------------------------------

def _make_oauth_session() -> OAuthSession:
    return OAuthSession(
        client_id="app_test",
        access_token="at_test",
        refresh_token="rt_test",
        id_token="id_test",
        api_key="sk-test-oauth-key",
        access_token_expires_at=9999999999.0,
        api_key_expires_at=9999999999.0,
        _file_path=None,
    )


def _mock_run_oauth_for_provider(session: OAuthSession):
    """Create an async mock for run_oauth_for_provider."""
    async def _fake(profile_store, cred_store, provider):
        auth_ref = str(uuid.uuid4())
        config_dir = profile_store._path.parent
        persisted = OAuthSession.create_session_file(session, auth_ref, config_dir)
        cred_store.set(auth_ref, str(persisted._file_path))
        return auth_ref, str(persisted._file_path)
    return AsyncMock(side_effect=_fake)


class TestCreateProfileFlowOAuth:
    def test_oauth_provider_skips_api_key_prompt(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """When provider requires OAuth, no API key prompt appears; OAuth flow runs instead."""
        create = _import_create_profile_flow()
        session = _make_oauth_session()

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="OpenAI ChatGPT (subscription)"),
            patch("kitty.cli.auth_cmd.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-5.3-codex", "my-oauth-profile"]),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
            patch("kitty.cli.profile_cmd.prompt_secret") as mock_secret,
        ):
            profile = create(store, cred_store)

        assert profile.provider == "openai_subscription"
        assert profile.model == "gpt-5.3-codex"
        assert profile.name == "my-oauth-profile"
        mock_secret.assert_not_called()  # no API key prompt

    def test_oauth_provider_uses_default_model(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """OAuth provider uses default model when user enters empty string."""
        create = _import_create_profile_flow()
        session = _make_oauth_session()

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="OpenAI ChatGPT (subscription)"),
            patch("kitty.cli.auth_cmd.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["", "oauth-profile"]),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            profile = create(store, cred_store)

        assert profile.model == "gpt-5.3-codex"

    def test_oauth_provider_session_file_is_accessible(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Credential store holds a path to an existing session file for OAuth profiles."""
        create = _import_create_profile_flow()
        session = _make_oauth_session()

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="OpenAI ChatGPT (subscription)"),
            patch("kitty.cli.auth_cmd.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch("kitty.cli.profile_cmd.prompt_text", side_effect=["gpt-5.3-codex", "oauth-prof"]),
            patch("kitty.cli.profile_cmd.prompt_confirm", return_value=True),
        ):
            profile = create(store, cred_store)

        session_path = cred_store.get(profile.auth_ref)
        assert session_path is not None
        assert Path(session_path).exists()


# ---------------------------------------------------------------------------
# OAuth provider edit flow
# ---------------------------------------------------------------------------

class TestEditProfileFlowOAuth:
    def test_shows_reauthenticate_for_oauth_provider(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Edit menu shows 'Re-authenticate' instead of 'API Key' for OAuth providers."""
        edit = _import_edit_profile_flow()
        session = _make_oauth_session()

        p = _make_profile("oauth-prof", provider="openai_subscription", model="gpt-5.3-codex")
        store.save(p)
        cred_store.set(p.auth_ref, "/fake/path.json")

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", side_effect=[
                "Re-authenticate",  # field selection
            ]),
            patch("kitty.cli.auth_cmd.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch("kitty.cli.profile_cmd.prompt_secret") as mock_secret,
        ):
            edit(store, cred_store, "oauth-prof")

        saved = store.get("oauth-prof")
        assert saved is not None
        assert saved.auth_ref != p.auth_ref  # new auth_ref from OAuth
        mock_secret.assert_not_called()  # no API key prompt for OAuth providers

    def test_edit_model_for_oauth_provider(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Editing just the model works for OAuth providers."""
        edit = _import_edit_profile_flow()
        p = _make_profile("oauth-prof", provider="openai_subscription", model="old-model")
        store.save(p)
        old_auth_ref = p.auth_ref

        with (
            patch("kitty.cli.profile_cmd.SelectionMenu.show", return_value="Model"),
            patch("kitty.cli.profile_cmd.prompt_text", return_value="new-model"),
        ):
            edit(store, cred_store, "oauth-prof")

        saved = store.get("oauth-prof")
        assert saved is not None
        assert saved.model == "new-model"
        assert saved.auth_ref == old_auth_ref
