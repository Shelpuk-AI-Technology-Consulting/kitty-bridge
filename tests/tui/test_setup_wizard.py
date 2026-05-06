"""Tests for setup wizard — first-run profile creation."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.cli.setup_cmd import run_setup_wizard
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.profiles.schema import PROVIDER_LABELS
from kitty.profiles.store import ProfileStore

# Patch targets for setup_cmd (it imports directly)
_MOD = "kitty.cli.setup_cmd"


@contextmanager
def _mock_tty():
    """Context manager to mock TTY state as True."""
    with patch("sys.stdin.isatty", return_value=True):
        yield


@pytest.fixture()
def store(tmp_path: object) -> ProfileStore:
    return ProfileStore(path=tmp_path / "profiles.json")  # type: ignore[arg-type]


@pytest.fixture()
def cred_store(tmp_path: object) -> CredentialStore:
    return CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])  # type: ignore[arg-type]


class TestRunSetupWizard:
    def test_non_tty_raises(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Setup wizard rejects non-TTY with deterministic error."""
        with patch("sys.stdin.isatty", return_value=False), pytest.raises(Exception, match="interactive"):
            run_setup_wizard(store, cred_store)

    def test_wizard_completes_all_steps(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Wizard completes all steps and returns a saved Profile."""
        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["zai_regular"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=None),
            patch(f"{_MOD}.prompt_secret", return_value="sk-test-api-key-12345"),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-4o", "myprofile"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, False]),
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        assert profile.name == "myprofile"
        assert profile.provider == "zai_regular"
        assert profile.model == "gpt-4o"
        assert profile.is_default is True

        # Verify saved to store
        loaded = store.get("myprofile")
        assert loaded is not None
        assert loaded.name == "myprofile"

        # Verify credential saved
        key = cred_store.get(profile.auth_ref)
        assert key == "sk-test-api-key-12345"

    def test_default_profile_name_suggestion(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """When user enters empty profile name, defaults to provider name."""
        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["novita"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=None),
            patch(f"{_MOD}.prompt_secret", return_value="sk-key"),
            patch(f"{_MOD}.prompt_text", side_effect=["model-x", ""]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[False, False]),
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        assert profile.name == "novita"
        assert profile.provider == "novita"

    def test_profile_name_rejects_reserved(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Wizard rejects reserved names and retries."""
        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["zai_regular"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=None),
            patch(f"{_MOD}.prompt_secret", return_value="sk-key"),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-4o", "setup", "goodname"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, False]),
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        assert profile.name == "goodname"

    def test_profile_name_rejects_invalid_format(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Wizard rejects invalid name format and retries."""
        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["zai_regular"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=None),
            patch(f"{_MOD}.prompt_secret", return_value="sk-key"),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-4o", "BAD NAME!", "valid-name"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, False]),
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        assert profile.name == "valid-name"

    def test_connectivity_check_failure_warns(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Connectivity check failure warns but allows proceeding."""
        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["zai_regular"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=None),
            patch(f"{_MOD}.prompt_secret", return_value="sk-key"),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-4o", "test"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, True]),
            patch(f"{_MOD}._check_connectivity", return_value=False),
            patch(f"{_MOD}.print_warning") as mock_warn,
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        mock_warn.assert_called()

    def test_connectivity_check_success(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Connectivity check success prints status."""
        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["zai_regular"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=None),
            patch(f"{_MOD}.prompt_secret", return_value="sk-key"),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-4o", "test"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, True]),
            patch(f"{_MOD}._check_connectivity", return_value=True),
            patch(f"{_MOD}.print_status") as mock_status,
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        mock_status.assert_called()

    def test_wizard_reuses_existing_key(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T22: wizard offers key reuse when same-provider profile exists; accepted → shared auth_ref."""
        import uuid as _uuid

        existing_ref = str(_uuid.uuid4())

        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["minimax"]),
            patch(f"{_MOD}._find_reusable_auth_ref", return_value=existing_ref),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, True, False]),  # reuse=True, default=True, no conn.check
            patch(f"{_MOD}.prompt_secret") as mock_secret,
            patch(f"{_MOD}.prompt_text", side_effect=["moonshot-v1", "myprofile"]),
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile.auth_ref == existing_ref
        mock_secret.assert_not_called()

    def test_wizard_cancelled_at_provider_step_raises(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """T23: Cancelling provider selection raises NonTTYError."""
        from kitty.tui.prompts import NonTTYError

        with _mock_tty(), patch(f"{_MOD}.SelectionMenu.show", return_value=None), pytest.raises(NonTTYError):
            run_setup_wizard(store, cred_store)


# ---------------------------------------------------------------------------
# OAuth provider path in setup wizard
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


class TestSetupWizardOAuth:
    def test_oauth_provider_launches_browser_not_key_prompt(
        self,
        store: ProfileStore,
        cred_store: CredentialStore,
    ) -> None:
        """When openai_subscription is selected, OAuth flow runs instead of API key prompt."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["openai_subscription"]),
            patch("kitty.cli.auth_cmd.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-5.3-codex", "my-openai"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[False]),  # set_default, no conn check
            patch(f"{_MOD}.prompt_secret") as mock_secret,
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile.provider == "openai_subscription"
        assert profile.model == "gpt-5.3-codex"
        assert profile.name == "my-openai"
        mock_secret.assert_not_called()  # no API key prompt

    def test_oauth_provider_uses_default_model(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """OAuth provider uses default model when user enters empty string."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.SelectionMenu.show", return_value=PROVIDER_LABELS["openai_subscription"]),
            patch("kitty.cli.auth_cmd.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["", "openai-prof"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[False]),
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile.model == "gpt-5.3-codex"
