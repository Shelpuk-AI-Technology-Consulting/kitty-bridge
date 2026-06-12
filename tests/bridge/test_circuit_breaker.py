"""Tests for balancing profile circuit breaker — backend health tracking and retry."""

from __future__ import annotations

import random
import time
import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# -- Helpers -----------------------------------------------------------------


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


UPSTREAM_OK = {
    "id": "chatcmpl-1",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_backends(n: int, provider_types: list[str] | None = None) -> list:
    backends = []
    for i in range(n):
        provider_type = provider_types[i] if provider_types is not None else f"stub-{i}"
        provider = StubProvider(provider_type=provider_type, base_url=f"https://api{i}.example.com/v1")
        key = f"key-{i}"
        profile = Profile(name=f"profile-{i}", provider="openai", model=f"model-{i}", auth_ref=str(uuid.uuid4()))
        backends.append((provider, key, profile))
    return backends


def _make_server_with_families(provider_types: list[str], cooldown: int = 300) -> BridgeServer:
    backends = _make_backends(len(provider_types), provider_types=provider_types)
    server = BridgeServer(
        adapter=StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=cooldown,
    )
    return server


def _make_server(n_backends: int = 3, cooldown: int = 300) -> BridgeServer:
    backends = _make_backends(n_backends)
    server = BridgeServer(
        adapter=StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="model-0",
        backends=backends,
        backend_cooldown=cooldown,
    )
    return server


# -- Step 1: Backend health tracking -----------------------------------------


class TestBackendHealthTracking:
    """Backend health data structure and _mark_backend_unhealthy."""

    def test_new_backend_is_healthy(self):
        server = _make_server(3)
        assert len(server._backend_health) == 3
        for h in server._backend_health:
            assert h["healthy"] is True
            assert h["failed_at"] is None

    def test_mark_backend_unhealthy(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(1)
        assert server._backend_health[0]["healthy"] is True
        assert server._backend_health[1]["healthy"] is False
        assert server._backend_health[1]["failed_at"] is not None
        assert server._backend_health[2]["healthy"] is True

    def test_mark_backend_records_monotonic_time(self):
        server = _make_server(3)
        before = time.monotonic()
        server._mark_backend_unhealthy(0)
        after = time.monotonic()
        assert before <= server._backend_health[0]["failed_at"] <= after

    def test_no_backends_has_empty_health(self):
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=StubProvider(),
            resolved_key="key",
            model="model",
        )
        assert server._backend_health == []


# -- Step 2: _get_next_backend skips unhealthy backends ----------------------


class TestGetNextBackendHealthAware:
    """_get_next_backend skips unhealthy backends and respects cooldown."""

    def test_skips_unhealthy_backend(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(1)

        # Backend 1 must never be returned while unhealthy
        for _ in range(20):
            _, key, _, _, _ = server._get_next_backend()
            assert key in ("key-0", "key-2"), f"Got {key}, expected only key-0 or key-2"


class TestFailureWeightedSelection:
    """Backends with more failures should be selected less often among healthy peers."""

    def test_repeatedly_failing_backend_deprioritized(self):
        """A backend that has failed 5 times should be selected much less often than one that never failed."""
        server = _make_server(3, cooldown=0)

        # Simulate backend-0 failing 5 times and recovering each time
        for _ in range(5):
            server._mark_backend_unhealthy(0)

        # All backends are healthy (cooldown=0), but backend-0 has 5 failures
        for h in server._backend_health:
            h["healthy"] = True
            h["failed_at"] = None

        random.seed(42)
        counts = {"key-0": 0, "key-1": 0, "key-2": 0}
        for _ in range(300):
            _, key, _, _, _ = server._get_next_backend()
            counts[key] += 1

        # Backend-0 should be selected less often than either of the healthy peers
        assert counts["key-0"] < counts["key-1"], f"key-0 ({counts['key-0']}) should be < key-1 ({counts['key-1']})"
        assert counts["key-0"] < counts["key-2"], f"key-0 ({counts['key-0']}) should be < key-2 ({counts['key-2']}"

    def test_failure_count_increments_on_mark_unhealthy(self):
        """_mark_backend_unhealthy should track cumulative failure count."""
        server = _make_server(3)
        assert server._backend_health[0].get("failure_count", 0) == 0

        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["failure_count"] == 1

        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["failure_count"] == 2

    def test_failure_count_persists_across_recovery(self):
        """Failure count should persist even after cooldown expires and backend recovers."""
        server = _make_server(3, cooldown=0)

        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["failure_count"] == 2

        # Backend recovers (cooldown=0)
        server._backend_health[0]["healthy"] = True
        server._backend_health[0]["failed_at"] = None

        # failure_count should still be 2, not reset
        assert server._backend_health[0]["failure_count"] == 2

    def test_equal_failure_counts_give_equal_selection(self):
        """Backends with equal failure counts should be selected equally."""
        server = _make_server(2, cooldown=0)

        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(1)

        # Both recovered
        for h in server._backend_health:
            h["healthy"] = True
            h["failed_at"] = None

        random.seed(42)
        counts = {"key-0": 0, "key-1": 0}
        for _ in range(200):
            _, key, _, _, _ = server._get_next_backend()
            counts[key] += 1

        # Should be roughly equal (allow 30% tolerance)
        ratio = counts["key-0"] / counts["key-1"]
        assert 0.7 <= ratio <= 1.3, f"Selection ratio {ratio} too skewed: {counts}"

    def test_cooldown_expired_backend_recovers(self):
        import random as _random

        server = _make_server(3, cooldown=0)  # Instant cooldown
        server._mark_backend_unhealthy(1)

        # With cooldown=0, backend should recover immediately
        _random.seed(42)
        keys = []
        for _ in range(30):
            _, key, _, _, _ = server._get_next_backend()
            keys.append(key)
        # All three backends should be used
        assert "key-0" in keys
        assert "key-1" in keys
        assert "key-2" in keys

    def test_cooldown_not_expired_still_skipped(self):
        server = _make_server(3, cooldown=9999)
        server._mark_backend_unhealthy(1)

        keys = []
        for _ in range(4):
            _, key, _, _, _ = server._get_next_backend()
            keys.append(key)
        assert "key-1" not in keys

    def test_all_unhealthy_still_returns_backend(self):
        """When ALL backends are unhealthy but cooldown is near-expiry, return one anyway."""
        server = _make_server(2, cooldown=30)
        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(1)

        _, key, _, _, _ = server._get_next_backend()
        assert key in ("key-0", "key-1")

    def test_all_unhealthy_fast_fails_when_cooldown_far_future(self):
        """When ALL backends are unhealthy with long cooldowns, raise fast-fail error."""
        from kitty.bridge.server import AllBackendsUnhealthyError

        server = _make_server(2, cooldown=9999)
        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(1)

        with pytest.raises(AllBackendsUnhealthyError):
            server._get_next_backend()

    def test_non_balancing_unaffected(self):
        """Non-balancing mode (no backends) works as before."""
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=StubProvider(),
            resolved_key="single-key",
            model="single-model",
        )
        _, key, model, _, _ = server._get_next_backend()
        assert key == "single-key"
        assert model == "single-model"


# -- Step 3: Automatic retry integration ------------------------------------


class TestCircuitBreakerRetry:
    """Handlers automatically retry with next healthy backend on failure."""

    @pytest.mark.asyncio
    async def test_first_backend_fails_retries_on_second(self):
        """Backend-0 returns 500, backend-1 succeeds — agent gets success."""
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Backend-0 fails
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=500,
                payload={"error": {"message": "Internal error"}},
            )
            # Backend-1 succeeds
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["choices"][0]["message"]["content"] == "Hello!"

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_first_backend_429_fails_over_to_second(self):
        """Backend-0 returns 429, backend-1 succeeds — balancing mode fails over."""
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        sse_body = (
            b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello!"},"finish_reason":null}],'
            b'"model":"test-model"}\n\n'
            b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            b'"model":"test-model"}\n\n'
            b"data: [DONE]\n\n"
        )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=429,
                payload={"error": {"message": "Rate limited"}},
            )
            m.post(
                "https://api1.example.com/v1/chat/completions",
                body=sse_body,
                headers={"Content-Type": "text/event-stream"},
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.read()
                assert b"Hello!" in body
                assert b'"finish_reason":"stop"' in body
                assert b"[DONE]" in body

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_first_backend_429_non_streaming_fails_over_to_second(self):
        """Backend-0 returns 429, backend-1 succeeds — non-streaming balancing fails over immediately.

        Only ONE 429 mock is registered for backend-0. If the bridge retries 429 on the
        same backend, aioresponses will raise and the test fails — proving the bridge
        must fail over to backend-1 after a single 429.
        """
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Only ONE 429 response for backend-0 — must fail over immediately
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=429,
                payload={"error": {"message": "Rate limited"}},
            )
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["choices"][0]["message"]["content"] == "Hello!"

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_balancing_429_still_retries(self):
        """Single-backend mode preserves 429 retry/backoff behavior."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for _ in range(4):
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    status=429,
                    payload={"error": {"message": "Rate limited"}},
                )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 429

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_all_backends_429_propagates_last_error(self):
        """When all balancing backends return 429, the bridge surfaces an error response."""
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=429,
                payload={"error": {"message": "Rate limited 0"}},
            )
            m.post(
                "https://api1.example.com/v1/chat/completions",
                status=429,
                payload={"error": {"message": "Rate limited 1"}},
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status in (429, 500)

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_family_429_then_403_avoids_same_family(self):
        server = _make_server_with_families(["family-a", "family-a", "family-b"])
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(
                "https://api0.example.com/v1/chat/completions",
                status=429,
                payload={"error": {"message": "Rate limited"}},
            )
            m.post(
                "https://api2.example.com/v1/chat/completions",
                payload=UPSTREAM_OK,
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200
                body = await resp.json()
                assert body["choices"][0]["message"]["content"] == "Hello!"

        await server.stop_async()

    def test_rate_limited_backend_recovers_after_cooldown(self):
        server = _make_server_with_families(["family-a", "family-b"], cooldown=0)

        server._mark_backend_unhealthy(0, cooldown=0, failure_kind="rate_limit")

        _, key, _, _, _ = server._get_next_backend()
        assert key in ("key-0", "key-1")
        assert server._backend_health[0]["healthy"] is True
        assert server._backend_health[0]["failed_at"] is None
        assert server._backend_health[0]["stream_error_count"] == 0

        seen = set()
        for _ in range(20):
            _, key, _, _, _ = server._get_next_backend()
            seen.add(key)
        assert "key-0" in seen
        assert "key-1" in seen

    def test_429_and_cloudflare_only_affect_failed_backend(self):
        server = _make_server_with_families(["family-a", "family-a", "family-b"])

        server._mark_backend_unhealthy(0, cooldown=300, failure_kind="rate_limit")
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is True
        assert server._backend_health[2]["healthy"] is True

        server._mark_backend_unhealthy(1, cooldown=300, failure_kind="cloudflare")
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is False
        assert server._backend_health[2]["healthy"] is True

    @pytest.mark.asyncio
    async def test_first_two_fail_third_succeeds(self):
        """Backend-0 and backend-1 fail, backend-2 succeeds."""
        server = _make_server(3)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api0.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail 0"}})
            m.post("https://api1.example.com/v1/chat/completions", status=502, payload={"error": {"message": "fail 1"}})
            m.post("https://api2.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_all_backends_fail_propagates_error(self):
        """When all backends fail, the error propagates to the agent."""
        server = _make_server(2)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api0.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail 0"}})
            m.post("https://api1.example.com/v1/chat/completions", status=502, payload={"error": {"message": "fail 1"}})

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 500

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_non_balancing_no_retry(self):
        """Non-balancing mode: error propagates directly, no retry."""
        provider = StubProvider()
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=provider,
            resolved_key="single-key",
            model="single-model",
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail"}})

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 500

        await server.stop_async()

    @pytest.mark.asyncio
    async def test_unhealthy_backend_skipped_on_next_request(self):
        """After backend-0 fails, it's skipped for the next request within cooldown."""
        server = _make_server(2, cooldown=9999)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # Request 1: backend-0 fails, backend-1 succeeds
            m.post("https://api0.example.com/v1/chat/completions", status=500, payload={"error": {"message": "fail"}})
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

            # Request 2: backend-0 should be skipped (in cooldown), backend-1 used directly
            m.post("https://api1.example.com/v1/chat/completions", payload=UPSTREAM_OK)

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

        await server.stop_async()


# -- Step 4: Stream-specific cooldown and pool collapse prevention ------------


class TestStreamErrorCooldown:
    """_get_stream_error_cooldown returns escalating cooldowns for repeated failures."""

    def test_first_stream_error_returns_30s(self):
        server = _make_server(3)
        assert server._get_stream_error_cooldown(0) == 30

    def test_second_stream_error_returns_60s(self):
        server = _make_server(3)
        # First failure: mark with a short cooldown to increment stream_error_count
        server._mark_backend_unhealthy(0, cooldown=30)
        assert server._get_stream_error_cooldown(0) == 60

    def test_third_stream_error_returns_120s(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(0, cooldown=30)
        server._mark_backend_unhealthy(0, cooldown=60)
        assert server._get_stream_error_cooldown(0) == 120

    def test_fourth_stream_error_returns_240s(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(0, cooldown=30)
        server._mark_backend_unhealthy(0, cooldown=60)
        server._mark_backend_unhealthy(0, cooldown=120)
        assert server._get_stream_error_cooldown(0) == 240

    def test_capped_at_backend_cooldown(self):
        server = _make_server(3, cooldown=100)
        server._mark_backend_unhealthy(0, cooldown=30)
        server._mark_backend_unhealthy(0, cooldown=60)
        server._mark_backend_unhealthy(0, cooldown=100)  # reached cap
        assert server._get_stream_error_cooldown(0) == 100

    def test_default_failure_resets_stream_error_count(self):
        """A non-stream failure (no custom cooldown) resets the error count."""
        server = _make_server(3)
        server._mark_backend_unhealthy(0, cooldown=30)
        assert server._backend_health[0]["stream_error_count"] == 1
        # Hard failure resets
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["stream_error_count"] == 0
        assert server._get_stream_error_cooldown(0) == 30

    def test_negative_index_returns_default(self):
        server = _make_server(3)
        assert server._get_stream_error_cooldown(-1) == 30

    def test_no_backends_returns_default(self):
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=StubProvider(),
            resolved_key="key",
            model="model",
        )
        assert server._get_stream_error_cooldown(-1) == 30


class TestMarkUnhealthyWithStreamCooldown:
    """_mark_backend_unhealthy with stream-specific cooldown stores the right values."""

    def test_stream_cooldown_stored(self):
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, cooldown=30)
        assert server._backend_health[0]["cooldown"] == 30
        assert server._backend_health[0]["stream_error_count"] == 1

    def test_stream_error_count_increments(self):
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, cooldown=30)
        server._mark_backend_unhealthy(0, cooldown=60)
        assert server._backend_health[0]["stream_error_count"] == 2
        assert server._backend_health[0]["cooldown"] == 60

    def test_default_cooldown_resets_count(self):
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, cooldown=30)
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["cooldown"] == 300
        assert server._backend_health[0]["stream_error_count"] == 0


class TestTransportErrorCooldown:
    """Transport / connection-reset failures escalate cooldown on repeated errors."""

    def test_first_transport_error_uses_default_cooldown(self):
        server = _make_server(3, cooldown=300)
        assert server._get_transport_error_cooldown(0) == 300

    def test_second_transport_error_escalates(self):
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, cooldown=server._get_transport_error_cooldown(0), failure_kind="transport")
        assert server._get_transport_error_cooldown(0) > 300

    def test_third_transport_error_escalates_further(self):
        server = _make_server(3, cooldown=300)
        first = server._get_transport_error_cooldown(0)
        server._mark_backend_unhealthy(0, cooldown=first, failure_kind="transport")
        second = server._get_transport_error_cooldown(0)
        server._mark_backend_unhealthy(0, cooldown=second, failure_kind="transport")
        assert server._get_transport_error_cooldown(0) > second

    def test_capped_at_double_backend_cooldown(self):
        server = _make_server(3, cooldown=300)
        server._backend_health[0]["transport_error_count"] = 10
        assert server._get_transport_error_cooldown(0) == 600

    def test_hard_failure_resets_transport_error_count(self):
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, cooldown=300, failure_kind="transport")
        assert server._backend_health[0]["transport_error_count"] == 1
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["transport_error_count"] == 0
        assert server._get_transport_error_cooldown(0) == 300


class TestLeastRecentlyFailedBackend:
    """When all backends are unhealthy, prefer the one that failed longest ago."""

    def test_prefers_near_expiry_backends_are_randomized(self):
        """When all backends unhealthy but within fast-fail threshold, selection is randomized."""
        server = _make_server(3, cooldown=30)
        server._mark_backend_unhealthy(0)
        server._backend_health[0]["failed_at"] = time.monotonic() - 10
        server._mark_backend_unhealthy(1)
        server._backend_health[1]["failed_at"] = time.monotonic() - 5
        server._mark_backend_unhealthy(2)
        server._backend_health[2]["failed_at"] = time.monotonic() - 1

        selected = set()
        for _ in range(50):
            _, key, _, _, _ = server._get_next_backend()
            selected.add(key)

        assert len(selected) > 1

    def test_two_backends_picks_older(self):
        """With cooldown=0 and failed_at in the past, both are eligible."""
        server = _make_server(2, cooldown=0)
        server._mark_backend_unhealthy(0)
        server._backend_health[0]["failed_at"] = time.monotonic() - 20
        server._mark_backend_unhealthy(1)
        server._backend_health[1]["failed_at"] = time.monotonic() - 1

        _, key, _, _, _ = server._get_next_backend()
        assert key in ("key-0", "key-1")


class TestFamilyCooldown:
    """Family-level anti-abuse cooldown after 429/403."""

    def test_429_only_marks_failed_backend(self):
        """After a 429, only the failed backend is unhealthy — siblings unaffected."""
        server = _make_server_with_families(["family-a", "family-a", "family-b"])

        server._mark_backend_unhealthy(0, cooldown=300, failure_kind="rate_limit")
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is True
        assert server._backend_health[2]["healthy"] is True

    def test_429_sibling_backend_still_selectable(self):
        """After backend-0 gets 429, backend-1 (same family) should still be selectable."""
        server = _make_server_with_families(["family-a", "family-a", "family-b"])
        server._mark_backend_unhealthy(0, cooldown=300, failure_kind="rate_limit")

        # backend-1 is healthy and NOT deprioritized by family cooldown
        keys_seen = set()
        random.seed(42)
        for _ in range(100):
            _, key, _, _, _ = server._get_next_backend()
            keys_seen.add(key)

        # backend-1 and backend-2 should both appear
        assert "key-1" in keys_seen, "Sibling in same family should still be selected"
        assert "key-2" in keys_seen, "Backend in different family should be selected"

    def test_different_family_not_affected(self):
        """A 429 on family-a should not affect family-b backends."""
        server = _make_server_with_families(["family-a", "family-b"])
        server._mark_backend_unhealthy(0, cooldown=300, failure_kind="rate_limit")

        # Backend-1 (family-b) should be selected
        _, key, _, _, _ = server._get_next_backend()
        assert key == "key-1"

    def test_cloudflare_403_only_marks_failed_backend(self):
        """A Cloudflare 403 should only mark the failed backend unhealthy."""
        server = _make_server_with_families(["family-a", "family-b"], cooldown=300)

        server._mark_backend_unhealthy(0, cooldown=300, failure_kind="cloudflare")
        assert server._backend_health[0]["healthy"] is False
        assert server._backend_health[1]["healthy"] is True


# -- Phase 1: Backend exhaustion fixes ------------------------------------------


class TestMarkBackendHealthy:
    """_mark_backend_healthy resets a backend to healthy state."""

    def test_resets_healthy_flag(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(1)
        assert server._backend_health[1]["healthy"] is False

        server._mark_backend_healthy(1)
        assert server._backend_health[1]["healthy"] is True

    def test_clears_failed_at(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(1)
        assert server._backend_health[1]["failed_at"] is not None

        server._mark_backend_healthy(1)
        assert server._backend_health[1]["failed_at"] is None

    def test_resets_stream_and_transport_error_counts(self):
        server = _make_server(3)
        server._mark_backend_unhealthy(0, cooldown=30)
        server._mark_backend_unhealthy(0, cooldown=60)
        assert server._backend_health[0]["stream_error_count"] == 2

        server._mark_backend_healthy(0)
        assert server._backend_health[0]["stream_error_count"] == 0
        assert server._backend_health[0]["transport_error_count"] == 0

    def test_does_not_reset_failure_count(self):
        """Failure count is cumulative and should NOT be reset by health recovery."""
        server = _make_server(3)
        server._mark_backend_unhealthy(0)
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["failure_count"] == 2

        server._mark_backend_healthy(0)
        assert server._backend_health[0]["failure_count"] == 2

    def test_out_of_range_is_noop(self):
        """Calling with an out-of-range index should not raise."""
        server = _make_server(2)
        server._mark_backend_healthy(99)  # should not raise


class TestSingleBackendCooldownCap:
    """Single-backend configurations should use shorter cooldowns."""

    def test_single_backend_cooldown_capped(self):
        """A single backend should get a capped cooldown, not the full 300s."""
        backends = _make_backends(1)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
            backend_cooldown=300,
        )
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["cooldown"] <= 30

    def test_multi_backend_keeps_full_cooldown(self):
        """Multi-backend should keep the full cooldown."""
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0)
        assert server._backend_health[0]["cooldown"] == 300

    def test_single_backend_with_explicit_short_cooldown(self):
        """Explicit short cooldown should not be increased by the cap."""
        backends = _make_backends(1)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
            backend_cooldown=300,
        )
        server._mark_backend_unhealthy(0, cooldown=10)
        assert server._backend_health[0]["cooldown"] == 10

    @pytest.mark.asyncio
    async def test_single_backend_recovers_after_successful_request(self):
        """After a successful CC stream, a single backend should be marked healthy again."""
        backends = _make_backends(1)
        server = BridgeServer(
            adapter=StubLauncher(),
            provider=backends[0][0],
            resolved_key=backends[0][1],
            model="model-0",
            backends=backends,
            backend_cooldown=300,
        )
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        sse_body = (
            b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello!"},"finish_reason":null}],'
            b'"model":"test-model"}\n\n'
            b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            b'"model":"test-model"}\n\n'
            b"data: [DONE]\n\n"
        )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            # First request succeeds
            m.post(
                "https://api0.example.com/v1/chat/completions",
                body=sse_body,
                headers={"Content-Type": "text/event-stream"},
            )

            async with aiohttp.ClientSession() as session, session.post(url, json=request_body) as resp:
                assert resp.status == 200

            # Backend should be marked healthy after success
            assert server._backend_health[0]["healthy"] is True

        await server.stop_async()


# -- Phase 2: 401 auth failure blacklisting --------------------------------------


class TestMarkBackendUnhealthyAuth:
    """_mark_backend_unhealthy with failure_kind='auth' blacklists the backend."""

    def test_auth_failure_sets_max_cooldown(self):
        """Auth failure should set a very long (session-persistent) cooldown."""
        server = _make_server(3)
        server._mark_backend_unhealthy(0, failure_kind="auth")
        assert server._backend_health[0]["cooldown"] >= 900  # 15 min

    def test_auth_failure_resets_error_counts(self):
        """Auth failure resets stream/transport error counts like 'hard'."""
        server = _make_server(3)
        server._mark_backend_unhealthy(0, cooldown=30)
        assert server._backend_health[0]["stream_error_count"] == 1

        server._mark_backend_unhealthy(0, failure_kind="auth")
        assert server._backend_health[0]["stream_error_count"] == 0
        assert server._backend_health[0]["transport_error_count"] == 0

    def test_auth_failure_backend_stays_unhealthy(self):
        """After auth failure, backend should not recover within a normal session."""
        server = _make_server(3, cooldown=300)
        server._mark_backend_unhealthy(0, failure_kind="auth")
        # Simulate a long time passing — should still be unhealthy
        server._backend_health[0]["failed_at"] = time.monotonic() - 3600  # 1 hour ago
        assert not server._backend_health[0]["healthy"]
        # Cooldown hasn't expired (900s)
        assert server._backend_health[0]["cooldown"] >= 900
