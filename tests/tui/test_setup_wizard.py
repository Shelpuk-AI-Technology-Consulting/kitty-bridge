"""Tests for setup wizard — first-run profile creation."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from kitty.cli.setup_cmd import run_setup_wizard
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
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
            patch(f"{_MOD}.SelectionMenu.show", return_value="zai_regular"),
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
            patch(f"{_MOD}.SelectionMenu.show", return_value="novita"),
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
            patch(f"{_MOD}.SelectionMenu.show", return_value="zai_regular"),
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
            patch(f"{_MOD}.SelectionMenu.show", return_value="zai_regular"),
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
            patch(f"{_MOD}.SelectionMenu.show", return_value="zai_regular"),
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
            patch(f"{_MOD}.SelectionMenu.show", return_value="zai_regular"),
            patch(f"{_MOD}.prompt_secret", return_value="sk-key"),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-4o", "test"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, True]),
            patch(f"{_MOD}._check_connectivity", return_value=True),
            patch(f"{_MOD}.print_status") as mock_status,
        ):
            profile = run_setup_wizard(store, cred_store)

        assert profile is not None
        mock_status.assert_called()
