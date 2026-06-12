"""F21: Cloudflare thrash loop — alternating success/failure never escalates.

When ``_mark_backend_healthy`` resets ``cloudflare_error_count`` to 0,
the pattern Cloudflare-block → success → Cloudflare-block never increases
the cooldown, so the backend thrashes indefinitely.

The fix: decay ``cloudflare_error_count`` by 1 on success instead of
resetting to 0, allowing consecutive blocks to escalate the cooldown.
"""

from __future__ import annotations

import uuid

import pytest

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter

# -- Helpers ----------------------------------------------------------------


class _StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self):
        from kitty.types import BridgeProtocol

        return BridgeProtocol.CHAT_COMPLETIONS_API

    def build_spawn_config(self, *args, **kwargs) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": False}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"err {status_code}")


def _make_backends(n: int) -> list:
    backends = []
    for i in range(n):
        provider = _StubProvider()
        key = f"key-{i}"
        profile = Profile(
            name=f"profile-{i}",
            provider="openai",
            model=f"model-{i}",
            auth_ref=str(uuid.uuid4()),
        )
        backends.append((provider, key, profile))
    return backends


def _make_server(n_backends: int = 3) -> BridgeServer:
    backends = _make_backends(n_backends)
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=300,
    )


# -- Tests ------------------------------------------------------------------


class TestCloudflareThrashLoop:
    """F21: cloudflare_error_count must decay by 1 on success, not reset to 0."""

    def test_cloudflare_count_decays_not_resets_on_success(self):
        """After 2 Cloudflare blocks → success, count decays to 1, not 0.

        Before the fix, _mark_backend_healthy unconditionally sets
        ``cloudflare_error_count = 0``.  A backend that gets Cloudflare-blocked,
        has a single success, then gets blocked again is treated as a first
        hit (15s cooldown) instead of escalating.

        With decay (``max(0, count - 1)``), the count goes from 2 → 1 after
        one success, so the next block becomes the 2nd hit (escalated), not
        the 1st.
        """
        server = _make_server(n_backends=3)

        # Two Cloudflare blocks without intervening success → count=2
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        assert server._backend_health[0]["cloudflare_error_count"] == 2

        # One success → old code resets to 0 here (bug)
        server._mark_backend_healthy(0)
        # F21 fix: cloudflare_error_count must decay (2→1), not reset (2→0).
        assert server._backend_health[0]["cloudflare_error_count"] == 1, (
            f"cloudflare_error_count should decay to 1, got {server._backend_health[0]['cloudflare_error_count']}"
        )

        # Next Cloudflare block → should be 3rd hit (count=2), not 1st hit
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        actual = server._backend_health[0]["cloudflare_error_count"]
        assert actual >= 2, f"Expected cloudflare_error_count >= 2 after 3 blocks, got {actual}"

    def test_cloudflare_escalates_without_intervening_success(self):
        """Repeated Cloudflare blocks without success should escalate cooldown."""
        server = _make_server(n_backends=3)

        # First hit — 15s
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        first_cooldown = server._backend_health[0]["cooldown"]

        # Second hit without success — should be longer
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        second_cooldown = server._backend_health[0]["cooldown"]

        assert second_cooldown >= first_cooldown, (
            f"Cloudflare cooldown should NOT decrease across consecutive blocks "
            f"(got {second_cooldown}s after {first_cooldown}s)"
        )

    @pytest.mark.parametrize(
        "initial_count,expected_after_success",
        [(2, 1), (3, 2), (5, 4), (1, 0)],
    )
    def test_cloudflare_decay_formula(self, initial_count: int, expected_after_success: int):
        """Decay: max(0, count - 1) — count=1 decays to 0, count=5 to 4, etc."""
        server = _make_server(n_backends=3)
        for _ in range(initial_count):
            server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        assert server._backend_health[0]["cloudflare_error_count"] == initial_count

        server._mark_backend_healthy(0)
        assert server._backend_health[0]["cloudflare_error_count"] == expected_after_success

    def test_failure_count_not_affected_by_decay(self):
        """The cumulative failure_count is never decremented."""
        server = _make_server(n_backends=3)

        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        server._mark_backend_unhealthy(0, failure_kind="cloudflare")
        assert server._backend_health[0]["failure_count"] == 2

        server._mark_backend_healthy(0)
        # failure_count must not be decremented — it's a cumulative lifetime stat
        assert server._backend_health[0]["failure_count"] == 2
