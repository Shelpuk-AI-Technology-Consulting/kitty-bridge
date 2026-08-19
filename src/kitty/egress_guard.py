"""Fail-closed check applied before any bridge is started.

Egress is only worth having if it cannot be bypassed. A provider whose
transport cannot honour the proxy would connect from the machine's own address
while the user believes every request shares one egress IP — silently, and
precisely when it matters. So kitty refuses to start instead.

This lives in its own top-level module rather than in ``kitty.cli.launcher``
because all four start paths need it: the two agent-launch paths, foreground
``kitty bridge``, and the background bridge runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kitty.egress import get_egress

if TYPE_CHECKING:
    from kitty.profiles.schema import Profile
    from kitty.providers.base import ProviderAdapter

__all__ = ["egress_block_reason"]


def egress_block_reason(
    provider: ProviderAdapter,
    profile: Profile,
    resolved_key: str,
    backends: list[tuple[ProviderAdapter, str, Profile]] | None = None,
) -> str | None:
    """Return why the configured egress gateway forbids this launch.

    Every backend that will actually serve traffic is checked, not just the
    first, so one unproxyable member of a balancing pool cannot slip past.

    Args:
        provider: Adapter for the single-profile case.
        profile: Profile the adapter was built from; named in the message.
        resolved_key: Credential the single-profile adapter will use. Some
            adapters can only answer for a specific authentication mode.
        backends: ``(provider, key, profile)`` tuples when running a balancing
            profile. When given, these are checked instead of the single pair.

    Returns:
        A user-facing explanation, or ``None`` when the launch may proceed —
        including whenever no egress gateway is configured.
    """
    egress = get_egress()
    if egress is None:
        return None

    candidates = list(backends) if backends else [(provider, resolved_key, profile)]
    for candidate_provider, candidate_key, candidate_profile in candidates:
        config = getattr(candidate_profile, "provider_config", {}) or {}
        if candidate_provider.supports_egress(candidate_key, config):
            continue
        name = getattr(candidate_profile, "name", "?")
        return (
            f"Profile {name!r} uses a transport that cannot route through the egress proxy "
            f"{egress.masked()}. Egress is configured, so kitty will not start — that traffic "
            f"would leave from this machine's own IP instead. Use a different profile, or "
            f"remove the gateway with 'kitty egress'."
        )
    return None
