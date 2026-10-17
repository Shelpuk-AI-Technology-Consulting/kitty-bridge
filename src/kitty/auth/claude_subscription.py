"""Inspect the local Claude Code subscription session without reading credentials."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["ClaudeSubscriptionStatus", "read_claude_subscription_status"]


@dataclass(frozen=True)
class ClaudeSubscriptionStatus:
    """Describe the non-secret Claude Code subscription state.

    Args:
        available: Whether Claude Code reports an active Claude.ai subscription.
        email: Account email reported by Claude Code, when available.
        subscription_type: Subscription name reported by Claude Code, when available.
        reason: User-safe remediation message when the session is unavailable.
    """

    available: bool
    email: str | None
    subscription_type: str | None
    reason: str | None


def read_claude_subscription_status(
    run_status: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ClaudeSubscriptionStatus:
    """Read Claude Code's non-secret subscription status.

    Delegates storage and token interpretation to the installed Claude Code
    executable, avoiding platform-specific credential-file and Keychain access.

    Args:
        run_status: Process runner used to invoke the Claude Code status command.

    Returns:
        The available subscription details or a user-safe remediation message.
    """
    # Claude Code owns token storage and refresh, so Kitty only asks for status.
    try:
        result = run_status(
            ["claude", "auth", "status", "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ClaudeSubscriptionStatus(
            available=False,
            email=None,
            subscription_type=None,
            reason="Claude Code is not installed or is not on PATH. Install Claude Code, sign in, then try again.",
        )

    # A failed status command does not expose trustworthy session data.
    if result.returncode != 0:
        return ClaudeSubscriptionStatus(
            available=False,
            email=None,
            subscription_type=None,
            reason="Claude Code is not signed in. Run `claude auth login`, then try again.",
        )

    # Parse only public status fields; the command must never supply credentials here.
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ClaudeSubscriptionStatus(
            available=False,
            email=None,
            subscription_type=None,
            reason="Claude Code returned invalid authentication status. Run `claude auth login`, then try again.",
        )

    if not isinstance(payload, dict):
        return ClaudeSubscriptionStatus(
            available=False,
            email=None,
            subscription_type=None,
            reason="Claude Code returned invalid authentication status. Run `claude auth login`, then try again.",
        )

    # A subscription session needs both the Claude.ai auth method and a subscription type.
    logged_in = payload.get("loggedIn") is True
    auth_method = payload.get("authMethod")
    subscription_type = payload.get("subscriptionType")
    if not logged_in:
        return ClaudeSubscriptionStatus(
            available=False,
            email=None,
            subscription_type=None,
            reason="Claude Code is not signed in. Run `claude auth login`, then try again.",
        )
    if auth_method != "claude.ai" or not isinstance(subscription_type, str) or not subscription_type.strip():
        return ClaudeSubscriptionStatus(
            available=False,
            email=None,
            subscription_type=None,
            reason="Claude Code is not using a Claude.ai subscription. Run `claude auth login`, then try again.",
        )

    email = payload.get("email")
    return ClaudeSubscriptionStatus(
        available=True,
        email=email if isinstance(email, str) and email.strip() else None,
        subscription_type=subscription_type.strip(),
        reason=None,
    )
