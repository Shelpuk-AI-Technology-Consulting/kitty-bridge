"""Test launcher eligibility for Anthropic subscription profiles."""

from __future__ import annotations

import uuid

import pytest

from kitty.profiles.eligibility import (
    ProfileEligibilityError,
    filter_launcher_eligible_profiles,
    require_launcher_eligible_profile,
)
from kitty.profiles.schema import Profile


def _profile(name: str, provider: str, *, backup: bool = False) -> Profile:
    """Build a profile for launcher-eligibility tests.

    Args:
        name: Profile name.
        provider: Provider type.
        backup: Whether the profile is a balancing reserve member.

    Returns:
        A valid profile instance.
    """
    return Profile(
        name=name,
        provider=provider,  # type: ignore[arg-type]
        model="model",
        auth_ref=str(uuid.uuid4()),
        backup=backup,
    )


@pytest.mark.parametrize("launcher_name", [None, "codex", "gemini", "kilo"])
def test_require_launcher_eligible_profile_rejects_subscription_outside_claude(
    launcher_name: str | None,
) -> None:
    """Reject direct subscription use outside the Claude launcher.

    Args:
        launcher_name: Launcher name, or None for generic bridge mode.
    """
    subscription = _profile("subscription", "anthropic_subscription")

    with pytest.raises(ProfileEligibilityError, match="Claude Code"):
        require_launcher_eligible_profile(subscription, launcher_name)


def test_require_launcher_eligible_profile_allows_subscription_for_claude() -> None:
    """Permit direct subscription use through the Claude launcher."""
    subscription = _profile("subscription", "anthropic_subscription")

    require_launcher_eligible_profile(subscription, "claude")


def test_launch_async_rejects_subscription_before_credential_resolution() -> None:
    """Reject a non-Claude direct launch before profile credential lookup."""
    import asyncio
    from unittest.mock import MagicMock

    from kitty.cli.launcher import launch_async
    from kitty.credentials.file_backend import FileBackend
    from kitty.credentials.store import CredentialStore
    from kitty.launchers.codex import CodexAdapter
    from kitty.providers.anthropic_subscription import AnthropicSubscriptionAdapter

    subscription = _profile("subscription", "anthropic_subscription")
    backend = MagicMock(spec=FileBackend)
    credential_store = CredentialStore(backends=[backend])

    exit_code = asyncio.run(
        launch_async(
            adapter=CodexAdapter(),
            provider=AnthropicSubscriptionAdapter(),
            profile=subscription,
            cred_store=credential_store,
            validate=False,
        )
    )

    assert exit_code == 1
    backend.get.assert_not_called()


def test_filter_launcher_eligible_profiles_excludes_subscription_and_preserves_members() -> None:
    """Keep the eligible profile order and reserve flags intact."""
    primary = _profile("primary", "openai")
    subscription = _profile("subscription", "anthropic_subscription", backup=True)
    backup = _profile("backup", "anthropic", backup=True)

    eligible = filter_launcher_eligible_profiles([primary, subscription, backup], "codex")

    assert eligible == [primary, backup]
    assert eligible[-1].backup is True


def test_filter_launcher_eligible_profiles_keeps_subscription_for_claude() -> None:
    """Keep a subscription profile in a Claude balancing pool."""
    subscription = _profile("subscription", "anthropic_subscription")
    api_key = _profile("api-key", "anthropic")

    assert filter_launcher_eligible_profiles([subscription, api_key], "claude") == [subscription, api_key]
