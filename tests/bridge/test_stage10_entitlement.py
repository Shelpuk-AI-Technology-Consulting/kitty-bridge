"""Stage 10 — Entitlement / plan-mismatch 403 misclassification.

The bridge used to treat every HTTP 403 as `auth` → 900 s cooldown. For
403s that signal "your plan does not include this model" (Ollama
subscription, OpenAI tier mismatch, etc.), a 900 s retry wastes a
failover slot every cycle, so the bridge fast-fails with a 503 once
all 6 backends line up in cooldown.

Stage 10 introduces a new `entitlement` failure kind backed by a
24 h cooldown, with a pure detector `is_entitlement_error()` that
matches a small set of substrings in the upstream body. These tests
are the unit-level guard for that detector.
"""

from __future__ import annotations

import logging
import time
import uuid

import pytest

from kitty.bridge.server import (
    _ENTITLEMENT_PATTERNS,
    BridgeServer,
    is_entitlement_error,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter, ProviderError
from kitty.types import BridgeProtocol

# -- Test fixtures ------------------------------------------------------------


class StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.CHAT_COMPLETIONS_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    def __init__(self, provider_type: str = "stub", base_url: str = "https://api.example.com/v1"):
        self._provider_type = provider_type
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        return self._provider_type

    @property
    def default_base_url(self) -> str:
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Error {status_code}")


def _make_backends(n: int, provider_types: list[str] | None = None) -> list:
    backends = []
    for i in range(n):
        provider_type = provider_types[i] if provider_types is not None else f"stub-{i}"
        provider = StubProvider(provider_type=provider_type, base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(name=f"profile-{i}", provider="openai", model=f"model-{i}", auth_ref=str(uuid.uuid4()))
        backends.append((provider, key, profile))
    return backends


def _make_server(n_backends: int = 3, cooldown: int = 300) -> BridgeServer:
    backends = _make_backends(n_backends)
    return BridgeServer(
        adapter=StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=cooldown,
    )


# -- Step 10.1: is_entitlement_error ----------------------------------------


class TestIsEntitlementError:
    """Pure detector for plan / subscription 403 bodies."""

    def test_403_with_subscription_message_returns_true(self):
        msg = "Ollama Cloud error 403: this model requires a subscription, upgrade for access"
        assert is_entitlement_error(403, msg) is True

    def test_403_with_plain_forbidden_returns_false(self):
        assert is_entitlement_error(403, "Forbidden") is False

    def test_401_always_returns_false(self):
        assert is_entitlement_error(401, "this model requires a subscription") is False

    def test_500_always_returns_false(self):
        assert is_entitlement_error(500, "this model requires a subscription") is False

    def test_non_string_body_returns_false(self):
        assert is_entitlement_error(403, {"error": "requires a subscription"}) is False
        assert is_entitlement_error(403, None) is False
        assert is_entitlement_error(403, 42) is False
        assert is_entitlement_error(403, ["requires a subscription"]) is False

    @pytest.mark.parametrize("pattern", list(_ENTITLEMENT_PATTERNS))
    def test_every_documented_pattern_matches_403(self, pattern: str):
        assert is_entitlement_error(403, pattern) is True

    def test_case_insensitive(self):
        assert is_entitlement_error(403, "REQUIRES A SUBSCRIPTION") is True
        assert is_entitlement_error(403, "Upgrade For Access") is True

    def test_substring_embedded_in_larger_body(self):
        body = "Provider: openai. Detail: this model requires a subscription to access. Action: visit /upgrade"
        assert is_entitlement_error(403, body) is True


# -- Step 10.2: _provider_error_failure_kind --------------------------------


def _provider_error(message: str, *, http_status: int, is_cloudflare: bool = False) -> ProviderError:
    """Build a ProviderError mirroring how provider.map_error() constructs them."""
    err = ProviderError(message)
    err.http_status = http_status
    err.is_cloudflare = is_cloudflare
    return err


class TestProviderErrorFailureKind:
    """Entitlement 403s classify as a distinct 'entitlement' failure kind."""

    def test_cloudflare_beats_entitlement(self):
        err = _provider_error(
            "this model requires a subscription", http_status=403, is_cloudflare=True
        )
        assert BridgeServer._provider_error_failure_kind(err) == "cloudflare"

    def test_entitlement_beats_auth(self):
        err = _provider_error(
            "Ollama Cloud error 403: this model requires a subscription, upgrade for access",
            http_status=403,
        )
        assert BridgeServer._provider_error_failure_kind(err) == "entitlement"

    def test_plain_403_still_classified_as_auth(self):
        """A 403 without an entitlement message is a credential error (regression guard)."""
        err = _provider_error("Forbidden: invalid api key", http_status=403)
        assert BridgeServer._provider_error_failure_kind(err) == "auth"

    def test_401_classified_as_auth(self):
        err = _provider_error("Unauthorized", http_status=401)
        assert BridgeServer._provider_error_failure_kind(err) == "auth"

    def test_429_classified_as_rate_limit(self):
        err = _provider_error("Too many requests", http_status=429)
        assert BridgeServer._provider_error_failure_kind(err) == "rate_limit"

    def test_non_provider_error_falls_through(self):
        """Non-ProviderError exceptions keep transport/hard classification."""
        assert BridgeServer._provider_error_failure_kind(ConnectionResetError()) == "transport"
        assert BridgeServer._provider_error_failure_kind(RuntimeError("boom")) == "hard"


# -- Step 10.3: _mark_backend_unhealthy for 'entitlement' -------------------


class TestMarkBackendUnhealthyEntitlement:
    """An entitlement failure sets a 24 h cooldown like auth sets 900 s."""

    def test_entitlement_sets_24h_cooldown(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(0, failure_kind="entitlement")
        assert server._backend_health[0]["cooldown"] == 86400

    def test_entitlement_increments_failure_count(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(0, failure_kind="entitlement")
        assert server._backend_health[0]["failure_count"] == 1

    def test_entitlement_resets_stream_and_transport_counts(self):
        server = _make_server(3, cooldown=300)
        # Establish prior transient error state.
        server._mark_backend_unhealthy(0, cooldown=30, failure_kind="stream")
        assert server._backend_health[0]["stream_error_count"] == 1
        server._mark_backend_unhealthy(0, failure_kind="entitlement")
        assert server._backend_health[0]["stream_error_count"] == 0
        assert server._backend_health[0]["transport_error_count"] == 0

    def test_entitlement_backend_stays_unhealthy_after_one_hour(self):
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, failure_kind="entitlement")
        # Simulate one hour passing — well within the 24 h cooldown.
        server._backend_health[0]["failed_at"] = time.monotonic() - 3600
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[0]["cooldown"] >= 86400

    def test_auth_still_uses_900s_cooldown(self):
        """Regression guard: auth (401 / plain 403) must stay at 900 s."""
        server = _make_server(3)
        server._mark_backend_unhealthy(0, failure_kind="auth")
        assert server._backend_health[0]["cooldown"] == 900


# -- Step 10.4: _map_provider_error ----------------------------------------


class TestMapProviderError:
    """The Anthropic-compatible error.type distinguishes entitlement from auth."""

    def test_entitlement_403_maps_to_entitlement_error(self):
        err = _provider_error(
            "Ollama Cloud error 403: this model requires a subscription, upgrade for access",
            http_status=403,
        )
        assert BridgeServer._map_provider_error(err) == (403, "entitlement_error")

    def test_plain_403_still_maps_to_authentication_error(self):
        """Regression guard: a non-entitlement 403 stays authentication_error."""
        err = _provider_error("Forbidden: invalid api key", http_status=403)
        assert BridgeServer._map_provider_error(err) == (403, "authentication_error")

    def test_401_maps_to_authentication_error(self):
        err = _provider_error("Unauthorized", http_status=401)
        assert BridgeServer._map_provider_error(err) == (401, "authentication_error")

    def test_429_maps_to_rate_limit_error(self):
        err = _provider_error("Too many requests", http_status=429)
        assert BridgeServer._map_provider_error(err) == (429, "rate_limit_error")

    def test_non_provider_error_maps_to_502_api_error(self):
        assert BridgeServer._map_provider_error(RuntimeError("boom")) == (502, "api_error")


# -- Step 10.5: End-to-end custom-transport failover -----------------------


class _EntitlementCustomTransportProvider(StubProvider):
    """Custom-transport provider that mimics ollama_cloud's subscription 403.

    ``stream_request`` raises the exact ProviderError observed in
    ``debug/bridge.log`` (HTTP 403, "this model requires a subscription").
    """

    @property
    def use_custom_transport(self) -> bool:
        return True

    async def stream_request(self, cc_request: dict, write) -> None:  # type: ignore[override]
        err = ProviderError(
            "Ollama Cloud error 403: this model requires a subscription, upgrade for access"
        )
        err.http_status = 403
        raise err


class _SuccessCustomTransportProvider(StubProvider):
    """Custom-transport provider that streams a valid Chat Completions chunk."""

    @property
    def use_custom_transport(self) -> bool:
        return True

    async def stream_request(self, cc_request: dict, write) -> None:  # type: ignore[override]
        await write(b"data: {\"id\":\"chatcmpl-ok\"}\n\n")

    @staticmethod
    def parse_stream_to_cc_response(raw_bytes: bytes) -> dict:
        return {
            "id": "chatcmpl-ok",
            "created": 0,
            "model": "model-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }


def _make_custom_transport_server() -> BridgeServer:
    """Backend 0 = entitlement-403, backend 1 = success (both custom transport)."""
    provider0 = _EntitlementCustomTransportProvider(provider_type="ollama_cloud", base_url="https://ollama.example/v1")
    provider1 = _SuccessCustomTransportProvider(provider_type="stub", base_url="https://api1.example.com/v1")
    profile0 = Profile(name="profile-0", provider="ollama_cloud", model="glm-5.2", auth_ref=str(uuid.uuid4()))
    profile1 = Profile(name="profile-1", provider="openai", model="model-1", auth_ref=str(uuid.uuid4()))
    backends = [
        (provider0, "key-0", profile0),
        (provider1, "key-1", profile1),
    ]
    return BridgeServer(
        adapter=StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=300,
    )


class TestEntitlementFailover:
    """Reproduces debug/bridge.log L45084+: ollama 403 → 24h cooldown + failover."""

    @pytest.mark.asyncio
    async def test_subscription_403_gets_24h_cooldown_and_failover(self, caplog):
        import aiohttp

        server = _make_custom_transport_server()
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        caplog.set_level(logging.INFO, logger="kitty.bridge.server")

        # Backend selection is random-weighted, so loop until the entitlement
        # backend (index 0) is exercised. With probability 1 it is picked
        # within a few requests; the request always succeeds via failover.
        exercised = False
        async with aiohttp.ClientSession() as session:
            for _ in range(20):
                async with session.post(url, json=body) as resp:
                    assert resp.status == 200
                    raw = await resp.read()
                if server._backend_health[0]["cooldown"] == 86400:
                    exercised = True
                    break
            assert exercised, "entitlement backend was never selected"
            assert b"Hello!" in raw

        # Core regression: 24 h cooldown, NOT the old 900 s auth cooldown.
        assert server._backend_health[0]["cooldown"] == 86400
        assert server._backend_health[0]["healthy"] is False
        assert any(
            "marked unhealthy for 86400s after entitlement error" in rec.getMessage()
            for rec in caplog.records
        ), [rec.getMessage() for rec in caplog.records]

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_entitlement_backend_not_retried_within_session(self):
        """A follow-up request must keep selecting the healthy backend.

        The entitlement backend is parked for 24 h; it must not be retried at
        the old 900 s cadence (which caused the 503 storm in the wild).
        """
        import aiohttp

        server = _make_custom_transport_server()
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            # Exercise the entitlement backend first.
            for _ in range(20):
                async with session.post(url, json=body) as resp:
                    assert resp.status == 200
                    await resp.read()
                if server._backend_health[0]["cooldown"] == 86400:
                    break
            assert server._backend_health[0]["cooldown"] == 86400

            # Additional requests: entitlement backend stays parked.
            for _ in range(10):
                async with session.post(url, json=body) as resp:
                    assert resp.status == 200
                    await resp.read()
                # Cooldown must remain 86400 — never downgraded to 900 s or reset.
                assert server._backend_health[0]["cooldown"] == 86400
                assert server._backend_health[0]["healthy"] is False

        await server.stop_async()
