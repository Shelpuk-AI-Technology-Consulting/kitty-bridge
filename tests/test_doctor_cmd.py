"""Tests for doctor command — diagnostics and health checks."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from kitty.cli.doctor_cmd import run_doctor
from kitty.profiles.schema import Profile
from kitty.profiles.store import ProfileStore


def _make_profile(
    name: str = "test-profile",
    provider: str = "zai_regular",
    model: str = "gpt-4o",
    is_default: bool = True,
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


class TestDoctorAllChecks:
    def test_all_pass_returns_zero(self, store: ProfileStore) -> None:
        """Doctor returns 0 when all checks pass."""
        store.save(_make_profile(is_default=True))

        with (
            patch("kitty.cli.doctor_cmd.discover_binary", return_value=MagicMock()),
            patch("kitty.credentials.store.CredentialStore.get", return_value="sk-key"),
        ):
            exit_code = run_doctor(store)

        assert exit_code == 0

    def test_missing_binary_returns_nonzero(self, store: ProfileStore) -> None:
        """Doctor returns non-zero when a binary is missing."""
        store.save(_make_profile(is_default=True))

        with (
            patch("kitty.cli.doctor_cmd.discover_binary", return_value=None),
            patch("kitty.credentials.store.CredentialStore.get", return_value="sk-key"),
        ):
            exit_code = run_doctor(store)

        assert exit_code != 0


class TestDoctorTargetFlag:
    def test_target_codex_validates_binary(self, store: ProfileStore) -> None:
        """--target codex validates only the codex binary."""
        with patch("kitty.cli.doctor_cmd.discover_binary") as mock_disc:
            mock_disc.return_value = MagicMock()
            exit_code = run_doctor(store, target_name="codex")

        assert exit_code == 0
        mock_disc.assert_called_with("codex")

    def test_target_claude_validates_binary(self, store: ProfileStore) -> None:
        """--target claude validates only the claude binary."""
        with patch("kitty.cli.doctor_cmd.discover_binary") as mock_disc:
            mock_disc.return_value = MagicMock()
            exit_code = run_doctor(store, target_name="claude")

        assert exit_code == 0
        mock_disc.assert_called_with("claude")

    def test_target_missing_binary_fails(self, store: ProfileStore) -> None:
        """--target with missing binary returns non-zero."""
        with patch("kitty.cli.doctor_cmd.discover_binary", return_value=None):
            exit_code = run_doctor(store, target_name="codex")

        assert exit_code != 0

    def test_unknown_target_fails(self, store: ProfileStore) -> None:
        """--target with unknown name returns non-zero."""
        with patch("kitty.cli.doctor_cmd.discover_binary", return_value=None):
            exit_code = run_doctor(store, target_name="unknown-target")

        assert exit_code != 0


class TestDoctorProfileFlag:
    def test_profile_validates_provider(self, store: ProfileStore) -> None:
        """--profile validates the profile's provider is resolvable."""
        profile = _make_profile()
        store.save(profile)

        with patch("kitty.credentials.store.CredentialStore.get", return_value="sk-key"):
            exit_code = run_doctor(store, profile_name="test-profile")

        assert exit_code == 0

    def test_profile_missing_credential_fails(self, store: ProfileStore) -> None:
        """--profile fails when auth_ref doesn't resolve."""
        profile = _make_profile()
        store.save(profile)

        with patch("kitty.credentials.store.CredentialStore.get", return_value=None):
            exit_code = run_doctor(store, profile_name="test-profile")

        assert exit_code != 0

    def test_profile_not_found_fails(self, store: ProfileStore) -> None:
        """--profile with non-existent name returns non-zero."""
        with patch("kitty.credentials.store.CredentialStore.get", return_value=None):
            exit_code = run_doctor(store, profile_name="nonexistent")

        assert exit_code != 0


class TestDoctorNoDefault:
    def test_no_default_profile_warns(self, store: ProfileStore) -> None:
        """Doctor warns when no default profile is set."""
        store.save(_make_profile(is_default=False))

        with patch("kitty.cli.doctor_cmd.discover_binary", return_value=MagicMock()):
            exit_code = run_doctor(store)

        # Should still return non-zero because no default
        assert exit_code != 0
