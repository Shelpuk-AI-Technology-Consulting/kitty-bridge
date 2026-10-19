"""Tests for doctor command — diagnostics and health checks."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from kitty.auth.claude_subscription import ClaudeSubscriptionStatus
from kitty.cli.doctor_cmd import _check_profile, _make_credential_check, _make_subscription_check, run_doctor
from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
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


class TestDoctorCorruptCredentialReporting:
    """KBR-87: the doctor surfaces a corrupt stored value as a failed check, never a crash.

    The diagnostic tool exists to report the condition that corruption causes;
    crashing on it would defeat the tool. The raise comes from a real FileBackend
    so the test exercises the contract end to end (the unit-level raise is pinned
    in ``tests/test_credential_store.py``).
    """

    def _corrupt_cred_store(self, tmp_path: object, ref: str) -> CredentialStore:
        (tmp_path / "credentials.json").write_text(  # type: ignore[union-attr]
            json.dumps({ref: "@@@ not base64 @@@"}), encoding="utf-8"
        )
        return CredentialStore(
            backends=[FileBackend(path=tmp_path / "credentials.json")]  # type: ignore[arg-type]
        )

    def test_make_credential_check_reports_corruption(self, tmp_path: object) -> None:
        profile = _make_profile()
        cred_store = self._corrupt_cred_store(tmp_path, profile.auth_ref)

        passed, message = _make_credential_check(cred_store, profile)()

        assert passed is False
        assert profile.auth_ref in message
        assert "corrupt" in message

    def test_check_profile_reports_corruption_as_a_failed_check(
        self, tmp_path: object, store: ProfileStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        profile = _make_profile()
        store.save(profile)
        cred_store = self._corrupt_cred_store(tmp_path, profile.auth_ref)

        failures = _check_profile(store, cred_store, profile.name)

        # The provider check passes (zai_regular resolves), so exactly the
        # credential check fails — a weaker bound would pass even if a regression
        # broke the provider check too.
        assert failures == 1
        captured = capsys.readouterr()
        assert "corrupt" in captured.out + captured.err


class TestDoctorSubscriptionProfile:
    """A subscription profile holds no credential; the doctor checks the session.

    ``_make_credential_check`` resolves every ``auth_ref``, which a subscription
    profile intentionally lacks — checking it there would report a permanent
    false failure. Both doctor paths (the checklist loop and ``_check_profile``)
    must use the Claude Code session reader instead and never touch the store.
    """

    def _subscription_profile(self, name: str = "sub") -> Profile:
        return _make_profile(name=name, provider="anthropic_subscription", model="claude-sonnet-4-6")

    def test_make_subscription_check_reports_an_active_session(self) -> None:
        status = ClaudeSubscriptionStatus(available=True, email="a@b.c", subscription_type="max", reason=None)

        with patch(
            "kitty.auth.claude_subscription.read_claude_subscription_status",
            return_value=status,
        ):
            passed, message = _make_subscription_check()()

        assert passed is True
        assert "claude.ai subscription active" in message
        assert "a@b.c" in message

    def test_make_subscription_check_reports_the_remediation_when_unavailable(self) -> None:
        status = ClaudeSubscriptionStatus(
            available=False, email=None, subscription_type=None, reason="Claude Code is not signed in."
        )

        with patch(
            "kitty.auth.claude_subscription.read_claude_subscription_status",
            return_value=status,
        ):
            passed, message = _make_subscription_check()()

        assert passed is False
        assert message == "Claude Code is not signed in."

    def test_check_profile_uses_the_session_check_and_never_the_credential_store(
        self, store: ProfileStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        profile = self._subscription_profile()
        store.save(profile)

        def _no_credentials(*_args: object, **_kwargs: object) -> str:
            """Fail the test if the doctor resolves a subscription auth_ref."""
            raise AssertionError("credential store must not be consulted for a subscription profile")

        status = ClaudeSubscriptionStatus(available=True, email=None, subscription_type="pro", reason=None)
        with (
            patch("kitty.credentials.store.CredentialStore.get", side_effect=_no_credentials),
            patch("kitty.auth.claude_subscription.read_claude_subscription_status", return_value=status),
        ):
            failures = _check_profile(store, CredentialStore(backends=[FileBackend()]), profile.name)

        assert failures == 0
        captured = capsys.readouterr()
        assert "claude.ai subscription active" in captured.out + captured.err

    def test_run_doctor_checks_the_session_only_for_subscription_members(self, store: ProfileStore) -> None:
        """A mixed pool resolves credentials once — for the API-key member."""
        store.save(self._subscription_profile(name="sub"))
        store.save(_make_profile(name="metered"))

        status = ClaudeSubscriptionStatus(available=True, email=None, subscription_type="pro", reason=None)
        with (
            patch("kitty.cli.doctor_cmd.discover_binary", return_value=MagicMock()),
            # Hermetic against the box's real stored egress gateway, whose
            # password resolution also reads the credential store.
            patch("kitty.egress_store.resolve_egress", return_value=None),
            patch("kitty.credentials.store.CredentialStore.get", return_value="sk-key") as get_mock,
            patch("kitty.auth.claude_subscription.read_claude_subscription_status", return_value=status),
        ):
            exit_code = run_doctor(store)

        assert exit_code == 0
        assert get_mock.call_count == 1


class TestDoctorNoDefault:
    def test_no_default_profile_warns(self, store: ProfileStore) -> None:
        """Doctor warns when no default profile is set."""
        store.save(_make_profile(is_default=False))

        with patch("kitty.cli.doctor_cmd.discover_binary", return_value=MagicMock()):
            exit_code = run_doctor(store)

        # Should still return non-zero because no default
        assert exit_code != 0


class TestDoctorEgressCheck:
    """R13: the gateway is machine-level, so it is reported even with no profiles."""

    def test_egress_check_runs_when_no_profiles_exist(self, tmp_path):
        """A fresh install is exactly when someone is setting a gateway up."""
        from kitty.cli.doctor_cmd import run_doctor
        from kitty.profiles.store import ProfileStore

        store = ProfileStore(path=tmp_path / "profiles.json")
        executed: list[str] = []

        def _fake_check(_cred_store):
            def check() -> tuple[bool, str]:
                executed.append("ran")
                return True, "not configured"

            return check

        with patch("kitty.cli.doctor_cmd._make_egress_check", side_effect=_fake_check):
            run_doctor(store)

        assert executed == ["ran"], "the egress check never ran on a profile-less install"

    def test_egress_check_reports_the_configured_gateway_masked(self, tmp_path):
        from kitty.cli.doctor_cmd import _make_egress_check
        from kitty.credentials.file_backend import FileBackend
        from kitty.credentials.store import CredentialStore
        from kitty.egress import EgressConfig
        from kitty.egress_store import EgressRecord, EgressStore

        cred_store = CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])
        cred_store.set("ref", "s3cr3tpw")
        egress_store = EgressStore(path=tmp_path / "egress.json")
        egress_store.save(EgressRecord(proxy_url="http://proxy.example.com:3128", username="u", auth_ref="ref"))

        with patch("kitty.egress_store.EgressStore", return_value=egress_store):
            ok, detail = _make_egress_check(cred_store)()

        expected = EgressConfig(proxy_url="http://proxy.example.com:3128", username="u", password="s3cr3tpw")
        assert ok is True
        assert detail.endswith(expected.masked())
        assert "s3cr3tpw" not in detail

    def test_egress_check_reports_when_unconfigured(self, tmp_path):
        from kitty.cli.doctor_cmd import _make_egress_check
        from kitty.credentials.file_backend import FileBackend
        from kitty.credentials.store import CredentialStore
        from kitty.egress_store import EgressStore

        cred_store = CredentialStore(backends=[FileBackend(path=tmp_path / "creds.json")])

        with patch("kitty.egress_store.EgressStore", return_value=EgressStore(path=tmp_path / "egress.json")):
            ok, detail = _make_egress_check(cred_store)()

        assert ok is True
        assert "not configured" in detail
