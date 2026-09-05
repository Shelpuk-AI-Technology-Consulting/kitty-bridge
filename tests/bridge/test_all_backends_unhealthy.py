"""Tests that AllBackendsUnhealthyError in top-level handlers returns
503 Service Unavailable with a Retry-After header, not 500 Internal Server Error.

When all backends are exhausted, the bridge knows the soonest retry window
(``AllBackendsUnhealthyError.retry_after``).  Claude Code / Responses / Gemini
clients can use the ``Retry-After`` header to back off appropriately.
Without this fix, the bridge returned a generic 500 with no timing info.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import aiohttp
import pytest

from kitty.bridge.server import AllBackendsUnhealthyError, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter

# ── Stubs ──────────────────────────────────────────────────────────────────


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

        return BridgeProtocol.MESSAGES_API

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
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def normalize_model_name(self, model: str) -> str:
        return model

    def translate_to_upstream(self, cc_request: dict) -> dict:
        return {"model": cc_request["model"], "messages": cc_request.get("messages", [])}

    def translate_from_upstream(self, raw_response: dict) -> dict:
        return raw_response

    def translate_upstream_stream_event(self, raw_bytes: bytes) -> list[bytes]:
        return [raw_bytes]

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        from kitty.providers.base import ProviderError

        return ProviderError(f"Stub error {status_code}")


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_server() -> BridgeServer:
    return BridgeServer(_StubLauncher(), _StubProvider(), "test-key")


def _make_bridge_mode_server() -> BridgeServer:
    """Bridge mode registers all protocol routes (/v1/messages, /v1/responses,
    /v1/chat/completions, /v1/...)."""
    return BridgeServer(None, _StubProvider(), "test-key")  # type: ignore[arg-type]


def _messages_request() -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }


def _responses_request() -> dict:
    return {
        "model": "test-model",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def _cc_request() -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }


# ── AllBackendsUnhealthyError type tests ──────────────────────────────────


class TestAllBackendsUnhealthyErrorType:
    def test_carries_retry_after(self):
        err = AllBackendsUnhealthyError([{"name": "stub"}], retry_after=264)
        assert err.retry_after == 264

    def test_carries_backend_list(self):
        backends = [{"name": "a"}, {"name": "b"}]
        err = AllBackendsUnhealthyError(backends, retry_after=120)
        assert len(err.backends) == 2

    def test_message_includes_retry_after(self):
        err = AllBackendsUnhealthyError([], retry_after=42)
        assert "42" in str(err)


# ── Messages API handler returns 503 ──────────────────────────────────────


class TestMessagesHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after_header(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=264,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503, f"Expected 503, got {resp.status}"
                    retry_after = resp.headers.get("Retry-After")
                    assert retry_after is not None, "Missing Retry-After header"
                    assert int(retry_after) == 264
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_body_contains_error(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=120,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    data = await resp.json()
                    assert "error" in data
                    msg = json.dumps(data).lower()
                    # Should mention unavailability / retry / backend
                    assert any(
                        token in msg
                        for token in (
                            "unavail",
                            "backend",
                            "retry",
                            "503",
                        )
                    ), f"Error body lacks context: {data}"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_does_not_leak_traceback(self):
        """Python traceback markers must not appear in the error body."""
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=120,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    body_text = await resp.text()
                    assert "Traceback" not in body_text
                    assert "AllBackendsUnhealthyError" not in body_text
        finally:
            await server.stop_async()


# ── Responses API handler returns 503 ─────────────────────────────────────


class TestResponsesHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=180,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_responses_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    retry_after = resp.headers.get("Retry-After")
                    assert retry_after is not None
                    assert int(retry_after) == 180
        finally:
            await server.stop_async()


# ── Chat Completions handler returns 503 ──────────────────────────────────


class TestChatCompletionsHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=90,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json=_cc_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    assert resp.headers.get("Retry-After") is not None
        finally:
            await server.stop_async()


# ── Gemini handler returns 503 ────────────────────────────────────────────


class TestGeminiHandlerReturns503:
    @pytest.mark.asyncio
    async def test_returns_503_with_retry_after(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub"}],
                    retry_after=300,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1beta/models/test:generateContent",
                        json={"contents": [{"parts": [{"text": "hi"}]}]},
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    assert resp.headers.get("Retry-After") is not None
        finally:
            await server.stop_async()


# ── last_failure_kind persistence ─────────────────────────────────────────


class TestLastFailureKindPersistence:
    """The health record remembers the failure kind that quarantined a backend,
    so the all-unhealthy 503 can name the cause.  The memory is cleared
    wherever the backend returns to health."""

    def _make_server_with_backends(self) -> BridgeServer:
        """Build a BridgeServer with a real backends list — health tracking
        (and therefore ``last_failure_kind``) only exists in that mode."""
        import uuid

        from kitty.profiles.schema import Profile

        provider = _StubProvider()
        profile = Profile(name="kind-test", provider="openai", model="m", auth_ref=str(uuid.uuid4()))
        return BridgeServer(
            _StubLauncher(),
            provider,
            "test-key",
            backends=[(provider, "test-key", profile)],
            backend_cooldown=300,
        )

    def test_initial_value_is_none(self):
        server = self._make_server_with_backends()
        assert server._backend_health[0]["last_failure_kind"] is None

    def test_set_by_mark_unhealthy(self):
        server = self._make_server_with_backends()
        server._mark_backend_unhealthy(0, failure_kind="rate_limit")
        assert server._backend_health[0]["last_failure_kind"] == "rate_limit"

    def test_cleared_by_mark_healthy(self):
        server = self._make_server_with_backends()
        server._mark_backend_unhealthy(0, failure_kind="rate_limit")
        server._mark_backend_healthy(0)
        assert server._backend_health[0]["last_failure_kind"] is None

    def test_cleared_by_cooldown_expiry(self):
        import time

        server = self._make_server_with_backends()
        server._mark_backend_unhealthy(0, failure_kind="rate_limit")
        # Rewind failed_at past the cooldown so _select_backend sees expiry.
        server._backend_health[0]["failed_at"] = time.monotonic() - 100_000
        server._select_backend()
        assert server._backend_health[0]["last_failure_kind"] is None


# ── Enriched 503 body names the cause ─────────────────────────────────────


def _parse(resp):
    return json.loads(resp.text)


class TestAllUnhealthyResponseNamesCause:
    """The all-unhealthy 503 body must tell the user *why*: which failure kind
    each backend hit and when it becomes retryable.  The error ``type`` stays
    in the retryable ``api_error`` class for every cause — Anthropic clients
    treat ``authentication_error`` / ``permission_error`` /
    ``invalid_request_error`` as non-retryable, and a cause-specific type on a
    503 could make Claude Code abort a transient outage.  The cause rides in
    the message and a structured ``error.backends`` field instead."""

    def _exc(self, backends, retry_after=196):
        return AllBackendsUnhealthyError(backends, retry_after=retry_after)

    def test_unanimous_rate_limit(self):
        resp = BridgeServer._all_unhealthy_response(
            self._exc(
                [
                    {"name": "secondary", "reason": "rate_limit", "remaining_cooldown": 196},
                    {"name": "zai_coding", "reason": "rate_limit", "remaining_cooldown": 180},
                ]
            )
        )
        assert resp.status == 503
        assert resp.headers["Retry-After"] == "196"
        body = _parse(resp)
        assert body["error"]["type"] == "api_error"
        assert "rate-limited by the upstream provider" in body["error"]["message"]
        assert "secondary" in body["error"]["message"]
        assert "zai_coding" in body["error"]["message"]
        assert body["error"]["backends"] == [
            {"name": "secondary", "reason": "rate_limit", "remaining_cooldown": 196},
            {"name": "zai_coding", "reason": "rate_limit", "remaining_cooldown": 180},
        ]

    def test_mixed_reasons_neutral_headline(self):
        resp = BridgeServer._all_unhealthy_response(
            self._exc(
                [
                    {"name": "a", "reason": "rate_limit", "remaining_cooldown": 100},
                    {"name": "b", "reason": "transport", "remaining_cooldown": 50},
                ]
            )
        )
        body = _parse(resp)
        assert body["error"]["type"] == "api_error"
        assert "All 2 backends are currently unavailable" in body["error"]["message"]
        # Both causes still visible in the per-backend list.
        assert "a (rate_limit, ready in 100s)" in body["error"]["message"]
        assert "b (transport, ready in 50s)" in body["error"]["message"]

    def test_minimal_dict_still_503(self):
        """Existing callers pass only ``name`` — no KeyError, no 500, no leak."""
        resp = BridgeServer._all_unhealthy_response(self._exc([{"name": "stub"}]))
        body = _parse(resp)
        assert resp.status == 503
        assert body["error"]["type"] == "api_error"
        assert "All 1 backends are currently unavailable" in body["error"]["message"]
        assert "stub (unknown)" in body["error"]["message"]
        assert body["error"]["backends"] == [
            {"name": "stub", "reason": "unknown", "remaining_cooldown": None}
        ]
        assert "Traceback" not in resp.text
        assert "AllBackendsUnhealthyError" not in resp.text

    def test_not_stream_capable_headline(self):
        resp = BridgeServer._all_unhealthy_response(
            self._exc(
                [
                    {"name": "a", "reason": "not_stream_capable", "remaining_cooldown": 0},
                    {"name": "b", "reason": "not_stream_capable", "remaining_cooldown": 0},
                ]
            )
        )
        body = _parse(resp)
        assert "No stream-capable backend is available" in body["error"]["message"]

    @pytest.mark.parametrize(
        ("reason", "headline"),
        [
            ("transport", "unreachable (connection errors)"),
            ("auth", "rejecting the credentials"),
            ("cloudflare", "blocked by the provider firewall"),
            ("entitlement", "requires a plan/subscription upgrade"),
            ("oversized", "too large for the provider context window"),
            ("hard", "returning provider errors"),
            ("stream", "returning provider errors"),
        ],
    )
    def test_cause_headlines(self, reason, headline):
        resp = BridgeServer._all_unhealthy_response(
            self._exc([{"name": "a", "reason": reason, "remaining_cooldown": 10}])
        )
        assert headline in _parse(resp)["error"]["message"]

    @pytest.mark.parametrize(
        "style",
        ["anthropic", "openai_chat", "openai_responses", "google"],
    )
    def test_per_protocol_envelopes(self, style):
        """Each protocol gets its native error envelope (the bridge renders
        per-protocol shapes for all its other errors) with the shared cause
        message and backends payload inside."""
        resp = BridgeServer._all_unhealthy_response(
            self._exc([{"name": "a", "reason": "rate_limit", "remaining_cooldown": 100}]),
            style=style,
        )
        body = _parse(resp)
        assert resp.status == 503
        err = body["error"]
        if style == "anthropic":
            assert body["type"] == "error"
            assert err["type"] == "api_error"
        elif style == "google":
            assert err["code"] == 503
            assert err["status"] == "UNAVAILABLE"
        elif style == "openai_responses":
            assert err["code"] == "upstream_error"
        else:
            assert err["type"] == "upstream_error"
        assert "rate-limited by the upstream provider" in err["message"]
        assert err["backends"] == [
            {"name": "a", "reason": "rate_limit", "remaining_cooldown": 100}
        ]

    @pytest.mark.asyncio
    async def test_gemini_503_native_envelope_over_http(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub", "reason": "rate_limit", "remaining_cooldown": 300}],
                    retry_after=300,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1beta/models/test:generateContent",
                        json={"contents": [{"parts": [{"text": "hi"}]}]},
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    assert resp.headers.get("Retry-After") is not None
                    body = await resp.json()
                    assert body["error"]["code"] == 503
                    assert body["error"]["status"] == "UNAVAILABLE"
                    assert "rate-limited by the upstream provider" in body["error"]["message"]
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_messages_503_names_cause_over_http(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [
                        {"name": "secondary", "reason": "rate_limit", "remaining_cooldown": 196},
                        {"name": "zai_coding", "reason": "rate_limit", "remaining_cooldown": 180},
                    ],
                    retry_after=196,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    assert resp.headers["Retry-After"] == "196"
                    body = await resp.json()
                    assert body["error"]["type"] == "api_error"
                    assert "rate-limited by the upstream provider" in body["error"]["message"]
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_chat_completions_503_names_cause_over_http(self):
        server = _make_bridge_mode_server()
        port = await server.start_async()
        try:
            with patch.object(
                server,
                "_select_backend",
                side_effect=AllBackendsUnhealthyError(
                    [{"name": "stub", "reason": "transport", "remaining_cooldown": 90}],
                    retry_after=90,
                ),
            ):
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json=_cc_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 503
                    body = await resp.json()
                    # OpenAI envelope — no Anthropic "type": "error" wrapper.
                    assert "type" not in body
                    assert body["error"]["type"] == "upstream_error"
                    assert "unreachable (connection errors)" in body["error"]["message"]
        finally:
            await server.stop_async()


# ── Negative: success path still works ────────────────────────────────────


class TestHappyPathNotAffected:
    """Verify the 503 fallback doesn't interfere with successful requests."""

    @pytest.mark.asyncio
    async def test_successful_request_returns_200(self):
        from aioresponses import aioresponses

        # Non-streaming CC response
        body = json.dumps(
            {
                "id": "test-1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ).encode()

        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", status=200, body=body)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_messages_request(),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["role"] == "assistant"
        finally:
            await server.stop_async()
