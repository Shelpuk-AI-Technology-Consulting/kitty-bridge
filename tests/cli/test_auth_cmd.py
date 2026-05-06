"""Tests for kitty auth openai command."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.profiles.store import ProfileStore

# Module paths for patching
_MOD = "kitty.cli.auth_cmd"
_HELPER_MOD = "kitty.cli.auth_cmd.run_oauth_for_provider"


@contextmanager
def _mock_tty():
    """Context manager to mock TTY state as True."""
    with patch("sys.stdin.isatty", return_value=True):
        yield


@pytest.fixture()
def store(tmp_path: Path) -> ProfileStore:
    return ProfileStore(path=tmp_path / "profiles.json")


@pytest.fixture()
def cred_store(tmp_path: Path) -> CredentialStore:
    return CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])


def _make_oauth_session() -> OAuthSession:
    """Return a valid-looking OAuthSession for testing."""
    return OAuthSession(
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        access_token="at_test",
        refresh_token="rt_test",
        id_token="id_test",
        api_key="sk-test-openai-key",
        access_token_expires_at=9999999999.0,
        api_key_expires_at=9999999999.0,
        _file_path=None,
    )


def _mock_run_oauth_for_provider(session: OAuthSession):
    """Create an async mock for run_oauth_for_provider that returns (auth_ref, path)."""

    async def _fake_run(profile_store, cred_store, provider):
        import uuid

        auth_ref = str(uuid.uuid4())
        config_dir = profile_store._path.parent
        persisted = OAuthSession.create_session_file(session, auth_ref, config_dir)
        session_path = str(persisted._file_path)
        cred_store.set(auth_ref, session_path)
        return auth_ref, session_path

    return AsyncMock(side_effect=_fake_run)


class TestRunOauthForProvider:
    def test_creates_session_file_and_stores_path(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """run_oauth_for_provider creates a session file and stores the path."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(return_value=session)),
        ):
            from kitty.cli.auth_cmd import run_oauth_for_provider

            auth_ref, session_path = asyncio.run(run_oauth_for_provider(store, cred_store, "openai_subscription"))

        assert auth_ref is not None
        stored_path = cred_store.get(auth_ref)
        assert stored_path == session_path
        assert Path(session_path).exists()

    def test_raises_on_oauth_failure(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """OAuth flow failure is printed and re-raised."""

        class OAuthFailure(Exception):
            pass

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_flow", new=AsyncMock(side_effect=OAuthFailure("Browser not available"))),
            patch(f"{_MOD}.print_error") as mock_error,
        ):
            from kitty.cli.auth_cmd import run_oauth_for_provider

            with pytest.raises(OAuthFailure):
                asyncio.run(run_oauth_for_provider(store, cred_store, "openai_subscription"))

        mock_error.assert_called()
        assert "OAuth flow failed" in mock_error.call_args[0][0]

    def test_non_tty_raises(self) -> None:
        """Non-TTY environment raises NonTTYError."""
        from kitty.tui.prompts import NonTTYError

        with patch("sys.stdin.isatty", return_value=False):
            from kitty.cli.auth_cmd import run_oauth_for_provider

            with pytest.raises(NonTTYError):
                asyncio.run(run_oauth_for_provider(store, cred_store, "openai_subscription"))


class TestAuthOpenaiProfileCreation:
    def test_auth_cmd_creates_profile_with_openai_subscription_provider(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Profile is saved with provider=openai_subscription after successful OAuth."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-5.3-codex", "my-openai-profile"]),
            patch(f"{_MOD}.prompt_confirm", return_value=True),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile is not None
        assert profile.provider == "openai_subscription"
        assert profile.model == "gpt-5.3-codex"
        assert profile.name == "my-openai-profile"

        # Verify saved to store
        loaded = store.get("my-openai-profile")
        assert loaded is not None
        assert loaded.provider == "openai_subscription"

    def test_auth_cmd_stores_oauth_session_in_file(
        self, store: ProfileStore, cred_store: CredentialStore, tmp_path: Path
    ) -> None:
        """OAuth session file is created and credential store holds the path."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-5.3-codex", "openai-sub"]),
            patch(f"{_MOD}.prompt_confirm", return_value=True),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        # Session file path should be stored in credential store
        stored_path = cred_store.get(profile.auth_ref)
        assert stored_path is not None

        session_file = Path(stored_path)
        assert session_file.exists(), f"Session file not found at {session_file}"

        # Session file should contain valid JSON
        with open(session_file) as f:
            data = json.load(f)
        assert data["client_id"] == session.client_id
        assert data["api_key"] == session.api_key

    def test_auth_cmd_asks_for_model_and_profile_name(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Command prompts for model name and profile name via prompt_text."""
        session = _make_oauth_session()

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["custom-model", "my-profile"]),
            patch(f"{_MOD}.prompt_confirm", return_value=False),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile.model == "custom-model"
        assert profile.name == "my-profile"
        assert profile.is_default is False

    def test_auth_cmd_uses_default_model_and_profile_name(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """When user accepts defaults (presses Enter), gpt-5.3-codex and openai-sub are used."""
        session = _make_oauth_session()

        # Simulate empty inputs → defaults used
        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["", ""]),
            patch(f"{_MOD}.prompt_confirm", return_value=False),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=MagicMock(valid=True))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile.model == "gpt-5.3-codex"
        assert profile.name == "openai-sub"

    def test_auth_cmd_raises_if_oauth_flow_fails(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """OAuth flow failure propagates to caller."""

        class OAuthFailure(Exception):
            pass

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_for_provider", new=AsyncMock(side_effect=OAuthFailure("Browser not available"))),
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            with pytest.raises(OAuthFailure):
                asyncio.run(run_auth_openai(store, cred_store))

    def test_auth_cmd_validation_failure_warns_but_creates_profile(
        self, store: ProfileStore, cred_store: CredentialStore
    ) -> None:
        """Validation failure warns the user but the profile is still created."""
        session = _make_oauth_session()

        validation_result = MagicMock()
        validation_result.valid = False
        validation_result.reason = "Invalid API key"

        with (
            _mock_tty(),
            patch(f"{_MOD}.run_oauth_for_provider", _mock_run_oauth_for_provider(session)),
            patch(f"{_MOD}.prompt_text", side_effect=["gpt-5.3-codex", "test-profile"]),
            patch(f"{_MOD}.prompt_confirm", return_value=True),
            patch("kitty.validation.validate_api_key", new=AsyncMock(return_value=validation_result)),
            patch(f"{_MOD}.print_warning") as mock_warn,
        ):
            from kitty.cli.auth_cmd import run_auth_openai

            profile = asyncio.run(run_auth_openai(store, cred_store))

        assert profile is not None
        assert store.get("test-profile") is not None
        mock_warn.assert_called()
        warn_call = mock_warn.call_args[0][0]
        assert "validation failed" in warn_call

    def test_auth_cmd_non_tty_raises(self, store: ProfileStore, cred_store: CredentialStore) -> None:
        """Non-TTY environment raises NonTTYError."""
        from kitty.tui.prompts import NonTTYError

        with patch("sys.stdin.isatty", return_value=False):
            from kitty.cli.auth_cmd import run_auth_openai

            with pytest.raises(NonTTYError):
                asyncio.run(run_auth_openai(store, cred_store))
