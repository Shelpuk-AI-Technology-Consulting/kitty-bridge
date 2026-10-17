"""Test local Claude Code subscription session inspection."""

from __future__ import annotations

import json
from subprocess import CompletedProcess

import pytest

from kitty.auth.claude_subscription import ClaudeSubscriptionStatus, read_claude_subscription_status


def _status_payload(**overrides: object) -> str:
    """Build a valid non-secret Claude Code status response.

    Args:
        **overrides: Values that replace the default status fields.

    Returns:
        JSON output from the Claude Code status command.
    """
    payload: dict[str, object] = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "email": "pilot@example.test",
        "subscriptionType": "max",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_read_claude_subscription_status_valid_subscriber_returns_safe_details() -> None:
    """Return subscriber details without retaining a credential value."""
    calls: list[tuple[object, ...]] = []

    def run_status(*args: object, **_kwargs: object) -> CompletedProcess[str]:
        """Return a successful Claude Code status process result.

        Args:
            *args: Process arguments passed by the session reader.
            **_kwargs: Ignored process keyword arguments.

        Returns:
            A successful process result with subscriber details.
        """
        calls.append(args)
        return CompletedProcess(
            args=["claude", "auth", "status", "--json"],
            returncode=0,
            stdout=_status_payload(),
            stderr="",
        )

    result = read_claude_subscription_status(run_status)

    assert calls == [(["claude", "auth", "status", "--json"],)]
    assert result == ClaudeSubscriptionStatus(
        available=True,
        email="pilot@example.test",
        subscription_type="max",
        reason=None,
    )


def test_read_claude_subscription_status_ignores_unknown_sensitive_fields() -> None:
    """Keep unrecognized status fields out of the safe result."""
    payload = _status_payload(accessToken="must-not-surface", refreshToken="must-not-surface")

    def run_status(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        """Return a status result that carries unrecognized fields.

        Args:
            *_args: Ignored process arguments.
            **_kwargs: Ignored process keyword arguments.

        Returns:
            A successful process result with extra sensitive-shaped fields.
        """
        return CompletedProcess(args=["claude"], returncode=0, stdout=payload, stderr="")

    status = read_claude_subscription_status(run_status)

    assert status == ClaudeSubscriptionStatus(
        available=True,
        email="pilot@example.test",
        subscription_type="max",
        reason=None,
    )
    assert "must-not-surface" not in repr(status)


@pytest.mark.parametrize(
    ("result", "expected_reason"),
    [
        (
            CompletedProcess(args=["claude"], returncode=1, stdout="", stderr="Not logged in"),
            "Claude Code is not signed in. Run `claude auth login`, then try again.",
        ),
        (
            CompletedProcess(args=["claude"], returncode=0, stdout="not-json", stderr=""),
            "Claude Code returned invalid authentication status. Run `claude auth login`, then try again.",
        ),
        (
            CompletedProcess(args=["claude"], returncode=0, stdout=_status_payload(loggedIn=False), stderr=""),
            "Claude Code is not signed in. Run `claude auth login`, then try again.",
        ),
        (
            CompletedProcess(args=["claude"], returncode=0, stdout=_status_payload(authMethod="api_key"), stderr=""),
            "Claude Code is not using a Claude.ai subscription. Run `claude auth login`, then try again.",
        ),
    ],
)
def test_read_claude_subscription_status_unusable_session_returns_actionable_reason(
    result: CompletedProcess[str], expected_reason: str
) -> None:
    """Explain why Claude Code cannot supply a subscription session.

    Args:
        result: Stubbed process result from the Claude Code status command.
        expected_reason: User-safe explanation for the unusable result.
    """

    def run_status(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        """Return the predefined Claude Code status process result.

        Args:
            *_args: Ignored process arguments.
            **_kwargs: Ignored process keyword arguments.

        Returns:
            The test-defined process result.
        """
        return result

    status = read_claude_subscription_status(run_status)

    assert status.available is False
    assert status.reason == expected_reason
    assert status.email is None
    assert status.subscription_type is None


def test_read_claude_subscription_status_missing_binary_returns_actionable_reason() -> None:
    """Explain how to install Claude Code when its status command is unavailable."""

    def run_status(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        """Raise the process error emitted for a missing executable.

        Args:
            *_args: Ignored process arguments.
            **_kwargs: Ignored process keyword arguments.

        Raises:
            FileNotFoundError: Always, to simulate an absent Claude binary.
        """
        raise FileNotFoundError

    status = read_claude_subscription_status(run_status)

    assert status == ClaudeSubscriptionStatus(
        available=False,
        email=None,
        subscription_type=None,
        reason="Claude Code is not installed or is not on PATH. Install Claude Code, sign in, then try again.",
    )
