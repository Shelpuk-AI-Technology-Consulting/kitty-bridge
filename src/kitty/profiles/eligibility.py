"""Apply launcher-specific profile eligibility rules."""

from __future__ import annotations

from collections.abc import Sequence

from kitty.profiles.schema import Profile

__all__ = [
    "ProfileEligibilityError",
    "filter_launcher_eligible_profiles",
    "require_launcher_eligible_profile",
]

_SUBSCRIPTION_PROVIDER = "anthropic_subscription"
_CLAUDE_LAUNCHER = "claude"


class ProfileEligibilityError(ValueError):
    """Signal that a profile is incompatible with the selected launcher."""


def require_launcher_eligible_profile(profile: Profile, launcher_name: str | None) -> None:
    """Reject a direct profile incompatible with the selected launcher.

    Args:
        profile: Direct profile selected for launch or bridge mode.
        launcher_name: Launcher adapter name, or None for generic bridge mode.

    Raises:
        ProfileEligibilityError: If a subscription profile is not launched by Claude Code.
    """
    # Subscription credentials are formed only by Claude Code's OAuth request.
    if profile.provider == _SUBSCRIPTION_PROVIDER and launcher_name != _CLAUDE_LAUNCHER:
        raise ProfileEligibilityError(
            f"Profile {profile.name!r} uses an Anthropic Claude.ai subscription and requires the Claude Code launcher"
        )


def filter_launcher_eligible_profiles(profiles: Sequence[Profile], launcher_name: str | None) -> list[Profile]:
    """Return balancing members eligible for the selected launcher.

    Args:
        profiles: Ordered balancing-profile members.
        launcher_name: Launcher adapter name, or None for generic bridge mode.

    Returns:
        Ordered eligible members. Non-Claude launchers omit subscription members.
    """
    # Claude Code can relay its own subscription request; other launchers cannot form one.
    if launcher_name == _CLAUDE_LAUNCHER:
        return list(profiles)
    return [profile for profile in profiles if profile.provider != _SUBSCRIPTION_PROVIDER]
