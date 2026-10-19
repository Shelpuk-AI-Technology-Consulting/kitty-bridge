"""Bridge-level tests for subscription relay backend health (R9).

These cover the one claim no unit test can make: that a raw-relay failure
leaves the shared balancing health in the right state — the failed member
parks under the existing failure kind (rate limits honor the upstream
``Retry-After`` hint), the request fails over only to another subscription
member, and a pool with no eligible member left returns the existing
all-unhealthy response. Driving the real
:class:`~kitty.bridge.server.BridgeServer` matters because the wiring under
test is selection, marking, and the relay loop together; stubbing any of
them would stub out the defect.

The fake upstream is scripted per call (via an ``aioresponses`` callback),
because two registrations for one URL overwrite each other — only a
callback can answer 429-then-200 on the same endpoint.

**Layer.** ``tests/layers.py`` maps only ``tests/integration/`` away from the
``l1`` fallback, so this file collects as ``l1`` alongside every other
loopback test in this directory.
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.anthropic_subscription import AnthropicSubscriptionAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter

#: Upstream endpoint the subscription relay posts to.
_UPSTREAM_URL = "https://api.anthropic.com/v1/messages"

#: Endpoint standing in for an ordinary API-key member. Never mocked with a
#: success: any traffic here means the relay crossed the R4 boundary.
_APIKEY_URL = "https://api-key.example.com/v1/messages"

#: Intentionally non-canonical JSON: odd spacing and key order.
_RAW_BODY = (
    b'{\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
    b'  "max_tokens" : 8,\n  "model" : "claude-sonnet-4-6"\n}'
)

#: Streaming variant of the non-canonical body.
_RAW_STREAM_BODY = (
    b'{\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
    b'  "max_tokens" : 8,\n  "stream" : true,\n  "model" : "claude-sonnet-4-6"\n}'
)

#: Minimal Anthropic Messages reply the fake upstream answers with.
_UPSTREAM_REPLY = {
    "id": "msg_relay_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _make_server(backends: list) -> BridgeServer:
    """Build a balancing server fronting the given members.

    Args:
        backends: ``(provider, key, profile)`` balancing tuples.

    Returns:
        A server bound to an ephemeral port, not yet started, with the
        relay opted in as a Claude Code launch would.
    """
    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key="",
        backends=backends,
        allow_subscription_relay=True,
    )


def _subscription_member(name: str, *, backup: bool = False) -> tuple:
    """Build a tokenless subscription balancing member.

    Args:
        name: Profile name for health attribution.
        backup: Whether the member is a reserve-tier pool member.

    Returns:
        A ``(provider, key, profile)`` balancing tuple with no credential.
    """
    return (
        AnthropicSubscriptionAdapter(),
        "",
        Profile(
            name=name,
            provider="anthropic_subscription",
            model="claude-sonnet-4-6",
            auth_ref=str(uuid.uuid4()),
            backup=backup,
        ),
    )


def _api_key_member(name: str) -> tuple:
    """Build an ordinary API-key member pointed at its own endpoint.

    Args:
        name: Profile name for health attribution.

    Returns:
        A ``(provider, key, profile)`` balancing tuple whose upstream URL is
        distinct, so any traffic to it is observable.
    """
    return (
        CustomAnthropicAdapter(),
        "api-key-value",
        Profile(
            name=name,
            provider="custom_anthropic",
            model="claude-sonnet-4-6",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://api-key.example.com"},
            backup=True,
        ),
    )


def _scripted_responder(replies: list[tuple[int, object, dict[str, str]]], calls: list):
    """Answer successive upstream POSTs from a script, recording each call.

    Args:
        replies: Per-call ``(status, body, headers)`` triples; the last one
            repeats once the script is exhausted.
        calls: List receiving one ``(data_bytes, headers)`` entry per call.

    Returns:
        An ``aioresponses`` callback serving the script in order.
    """

    def respond(url, **kwargs):
        """Serve the next scripted reply and record the shipped bytes.

        Args:
            url: The matched upstream URL.
            **kwargs: The request's keyword arguments.

        Returns:
            The scripted response for this call number.
        """
        calls.append((kwargs.get("data"), dict(kwargs.get("headers") or {})))
        status, body, headers = replies[min(len(calls), len(replies)) - 1]
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
        return CallbackResult(status=status, body=payload, headers=headers)

    return respond


def _rate_limit_reply(retry_after: str) -> tuple[int, object, dict[str, str]]:
    """Build a 429 scripted reply carrying a Retry-After hint.

    Args:
        retry_after: The Retry-After header value to send.

    Returns:
        A scripted-reply triple for :func:`_scripted_responder`.
    """
    return (
        429,
        {"error": {"type": "rate_limit_error", "message": "slow down"}},
        {"Content-Type": "application/json", "Retry-After": retry_after},
    )


def _parked(server: BridgeServer) -> list[tuple[int, str]]:
    """List parked backends as ``(index, failure_kind)`` pairs.

    Args:
        server: The bridge server whose health records to inspect.

    Returns:
        One entry per backend currently marked unhealthy.
    """
    return [
        (idx, health.get("last_failure_kind"))
        for idx, health in enumerate(server._backend_health)
        if not health.get("healthy", True)
    ]


class TestSubscriptionRelayHealth:
    """Raw-relay failures park through the existing health model (R9)."""

    @pytest.mark.asyncio()
    async def test_429_parks_member_for_retry_after_and_fails_over_to_sibling(self) -> None:
        """Honor the upstream retry hint and serve from the sibling."""
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1")]
        server = _make_server(backends)
        calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([_rate_limit_reply("60"), (200, _UPSTREAM_REPLY, {})], calls),
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        assert len(calls) == 2
        assert all(data == _RAW_BODY for data, _headers in calls)
        parked = _parked(server)
        assert len(parked) == 1
        idx, kind = parked[0]
        assert kind == "rate_limit"
        assert server._backend_health[idx]["cooldown"] == 60

    @pytest.mark.asyncio()
    async def test_401_parks_member_as_auth_and_fails_over(self) -> None:
        """Park an OAuth failure as auth, not as a generic error."""
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1")]
        server = _make_server(backends)
        calls: list = []
        unauthorized = (401, {"error": {"type": "authentication_error", "message": "bad token"}}, {})
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([unauthorized, (200, _UPSTREAM_REPLY, {})], calls),
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer stale-token"},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        assert len(calls) == 2
        parked = _parked(server)
        assert len(parked) == 1
        assert parked[0][1] == "auth"

    @pytest.mark.asyncio()
    async def test_403_plan_denial_parks_member_as_entitlement(self) -> None:
        """Distinguish a plan denial from a credential failure."""
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1")]
        server = _make_server(backends)
        calls: list = []
        denied = (403, {"error": {"message": "This model requires a subscription"}}, {})
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([denied, (200, _UPSTREAM_REPLY, {})], calls),
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        assert len(calls) == 2
        parked = _parked(server)
        assert len(parked) == 1
        assert parked[0][1] == "entitlement"

    @pytest.mark.asyncio()
    async def test_failover_never_crosses_to_api_key_member(self) -> None:
        """With no subscription sibling left, answer 503 — never translate."""
        backends = [_subscription_member("sub-only"), _api_key_member("metered-key")]
        server = _make_server(backends)
        sub_calls: list = []
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([_rate_limit_reply("30")], sub_calls),
                    repeat=True,
                )
                m.post(
                    _APIKEY_URL,
                    callback=_scripted_responder([(200, _UPSTREAM_REPLY, {})], key_calls),
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 503, await resp.text()
        finally:
            await server.stop_async()

        assert len(sub_calls) == 1
        assert key_calls == []
        assert _parked(server) == [(0, "rate_limit")]

    @pytest.mark.asyncio()
    async def test_streaming_pre_emission_429_parks_and_surfaces_all_unhealthy(self) -> None:
        """Classify a pre-output streaming failure without emitting bytes."""
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1")]
        server = _make_server(backends)
        calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([_rate_limit_reply("30")], calls),
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_STREAM_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 503, await resp.text()
                    assert resp.content_type == "application/json"
        finally:
            await server.stop_async()

        assert len(calls) == 2
        parked = _parked(server)
        assert len(parked) == 2
        assert sorted(kind for _idx, kind in parked) == ["rate_limit", "rate_limit"]
        assert sorted(server._backend_health[idx]["cooldown"] for idx, _kind in parked) == [30, 30]

    @pytest.mark.asyncio()
    async def test_deterministic_400_stops_burning_members_and_surfaces_the_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deterministically-bad body surfaces Anthropic's 400, not a 503."""
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_DELAYS", (0.0,))
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_PERIOD", 1.0)
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1"), _subscription_member("sub-2")]
        server = _make_server(backends)
        calls: list = []
        rejected = (400, {"error": {"type": "invalid_request_error", "message": "model is withdrawn"}}, {})
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([rejected, rejected], calls),
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 400, await resp.text()
                    payload = await resp.json()
        finally:
            await server.stop_async()

        # Two members saw the same 400; the third was not burned.
        assert len(calls) == 2
        assert payload["error"]["type"] == "api_error"
        assert "model is withdrawn" in payload["error"]["message"]

    @pytest.mark.asyncio()
    async def test_success_after_cooldown_restores_health(self) -> None:
        """A recovered member serves again once its cooldown expires."""
        backends = [_subscription_member("sub-only")]
        server = _make_server(backends)
        calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_scripted_responder([_rate_limit_reply("60"), (200, _UPSTREAM_REPLY, {})], calls),
                    repeat=True,
                )
                async with aiohttp.ClientSession() as session:
                    first = session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    )
                    async with first as resp:
                        assert resp.status == 503, await resp.text()
                assert _parked(server) == [(0, "rate_limit")]
                server._backend_health[0]["cooldown"] = 0
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        assert len(calls) == 2
        assert _parked(server) == []

    @pytest.mark.asyncio()
    async def test_upstream_read_timeout_answers_504_envelope_without_parking(self) -> None:
        """A stalled upstream answers the agent's envelope, not a raw 500."""
        backends = [_subscription_member("sub-only")]
        server = _make_server(backends)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, exception=aiohttp.ServerTimeoutError("read timed out"))
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 504, await resp.text()
                    payload = await resp.json()
        finally:
            await server.stop_async()

        assert payload["error"]["type"] == "api_error"
        assert payload["error"]["message"] == ("Upstream provider timed out (120s). Try /clear to reduce context size.")
        assert _parked(server) == []

    @pytest.mark.asyncio()
    async def test_upstream_read_timeout_parks_no_member_and_fails_over_to_none(self) -> None:
        """A timeout is a slow request, not a sick backend: nothing parks."""
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1")]
        server = _make_server(backends)
        calls: list = []
        port = await server.start_async()

        def _timeout_responder(url: str, **kwargs: object) -> object:
            """Record the attempt, then stall the upstream."""
            calls.append((kwargs.get("data"), dict(kwargs.get("headers") or {})))
            raise aiohttp.ServerTimeoutError("read timed out")

        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    callback=_timeout_responder,
                    repeat=True,
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 504, await resp.text()
        finally:
            await server.stop_async()

        # Exactly one upstream attempt: a timeout triggers no failover hop.
        assert len(calls) == 1
        assert _parked(server) == []

    @pytest.mark.asyncio()
    async def test_transport_blip_rides_out_grace_before_parking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Brief connection trouble retries the same member, as everywhere else."""
        # Keep the grace window instant: the contract under test is the retry
        # decision, not the 30-second wait.
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_DELAYS", (0.0, 0.0))
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_PERIOD", 1.0)
        backends = [_subscription_member("sub-0"), _subscription_member("sub-1")]
        server = _make_server(backends)
        calls: list = []
        port = await server.start_async()

        def _blip_then_reply(url: str, **kwargs: object) -> object:
            """Drop the wire twice, then answer."""
            calls.append((kwargs.get("data"), dict(kwargs.get("headers") or {})))
            if len(calls) <= 2:
                raise aiohttp.ClientConnectionError("connection reset")
            return CallbackResult(status=200, body=json.dumps(_UPSTREAM_REPLY))

        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, callback=_blip_then_reply, repeat=True)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", "Authorization": "Bearer sub-token"},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        # Both blips were absorbed on the same member: no parking, no hop.
        assert len(calls) == 3
        assert _parked(server) == []
        assert server._stats_failovers == 0
