"""Bridge-level tests: a hinted rate-limit park does not feed the stream-error ladder.

Review round 3 (2026-10-29): ``_mark_backend_unhealthy``'s counter block
treated any short-cooldown mark as a stream error, so a member flapping
hinted 429s accumulated ``stream_error_count`` and its next transient stream
failure parked for a doubled cooldown. The ladder escalation itself is
covered by the circuit-breaker suite; these tests pin the boundary.

**Layer.** Collects as ``l1`` alongside the other loopback tests.
"""

from __future__ import annotations

import uuid

from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile


class _StubLauncher:
    """Stand-in launcher; the counter tests never spawn anything."""


def _server() -> BridgeServer:
    """Build a minimal balancing server for counter assertions.

    Returns:
        A server with two stub backends, not started.
    """
    from kitty.providers.custom_anthropic import CustomAnthropicAdapter

    profiles = [
        Profile(name=f"cb-{i}", provider="custom_anthropic", model="m", auth_ref=str(uuid.uuid4()))
        for i in range(2)
    ]
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=CustomAnthropicAdapter(),
        resolved_key="k",
        model="m",
        backends=[(CustomAnthropicAdapter(), "k", profile) for profile in profiles],
    )


class TestRateLimitMarkLeavesStreamCounterAlone:
    """A rate_limit mark resets the stream counter instead of incrementing it."""

    def test_hinted_rate_limit_marks_do_not_accumulate_stream_errors(self) -> None:
        """Two hinted 429 parks leave ``stream_error_count`` at zero."""
        server = _server()
        for _round in range(2):
            server._mark_backend_unhealthy(0, failure_kind="rate_limit", cooldown=42)
        assert server._backend_health[0]["stream_error_count"] == 0
        assert server._backend_health[0]["last_failure_kind"] == "rate_limit"

    def test_stream_failure_after_rate_limit_flapping_parks_at_ladder_base(self) -> None:
        """The first transient stream failure after 429 flapping escalates once, not twice.

        The ladder parks ``30 * 2**count``: with the counter left at one by
        the exempted 429 parks, the next stream failure is 60 s — without the
        exemption the flapped member would carry count 2 and park 120 s.
        """
        server = _server()
        server._mark_backend_unhealthy(0, failure_kind="rate_limit", cooldown=42)
        server._mark_backend_unhealthy(0, failure_kind="stream", cooldown=30)
        assert server._backend_health[0]["stream_error_count"] == 1
        assert server._get_stream_error_cooldown(0) == 60

    def test_repeated_stream_failures_still_escalate(self) -> None:
        """Genuine stream failures keep escalating (the ladder itself is intact)."""
        server = _server()
        server._mark_backend_unhealthy(0, failure_kind="stream", cooldown=30)
        server._mark_backend_unhealthy(0, failure_kind="stream", cooldown=30)
        assert server._backend_health[0]["stream_error_count"] == 2
        assert server._get_stream_error_cooldown(0) == 120
