"""Bridge-level tests: mixed-pool subscription members never ride API-key paths.

Live diagnosis 2026-10-29 (pool ``my``): the translated Messages stream-failover
walk selected the ``anthropic_subscription`` member, dispatched it through the
ordinary API-key transport with its by-construction empty stored key, collected
a guaranteed 401, and parked it 900 s — the member had never served a request.
These tests pin the repair: translated selections skip subscription members by
default, the arrival pick still admits them (the relay engages), and the
Messages routes cross into the raw relay as a pre-emission last resort.
Design of record: ``SYSTEM_DESIGN.md`` §16 (2026-10-29 extension).

**Layer.** Collects as ``l1`` alongside every other loopback test in this
directory (``tests/layers.py`` maps only ``tests/integration/`` away).
"""

from __future__ import annotations

import ast
import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import AllBackendsUnhealthyError, BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.anthropic_subscription import AnthropicSubscriptionAdapter
from kitty.providers.base import ProviderAdapter, ProviderError
from kitty.providers.custom_anthropic import CustomAnthropicAdapter

#: Upstream endpoint the subscription relay posts to.
_UPSTREAM_URL = "https://api.anthropic.com/v1/messages"

#: Endpoint standing in for the ordinary API-key member. Any relay traffic
#: here means the crossing engaged on the wrong path.
_APIKEY_URL = "https://api-key.example.com/v1/messages"

#: Intentionally non-canonical JSON bytes: odd spacing and key order. The
#: crossing must ship THESE bytes, not a re-serialization.
_RAW_STREAM_BODY = (
    b'{\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
    b'  "max_tokens" : 8,\n  "stream" : true,\n  "model" : "claude-sonnet-4-6"\n}'
)

#: Non-streaming variant.
_RAW_BODY = (
    b'{\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
    b'  "max_tokens" : 8,\n  "model" : "claude-sonnet-4-6"\n}'
)

#: Anthropic SSE the fake relay upstream answers a streaming relay with.
_SSE_REPLY = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant",'
    b'"content":[],"model":"claude-sonnet-4-6","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
    b"event: message_stop\n\n"
    b"data: [DONE]\n\n"
)

#: Minimal Anthropic Messages reply for a non-streaming relay.
_UPSTREAM_REPLY = {
    "id": "msg_mixed_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _plain_member(name: str) -> tuple:
    """Build an ordinary Anthropic-wire member on its own host.

    Args:
        name: Profile name for health attribution.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
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
        ),
    )


def _subscription_member(name: str) -> tuple:
    """Build a tokenless subscription balancing member.

    Args:
        name: Profile name for health attribution.

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
        ),
    )


class _FailingCustomProvider(ProviderAdapter):
    """Custom-transport provider whose every request raises a hard error."""

    use_custom_transport = True

    @property
    def provider_type(self) -> str:
        """Name the provider class in health attribution.

        Returns:
            The provider type string.
        """
        return "stub_custom"

    @property
    def default_base_url(self) -> str:
        """Give the never-contacted base URL.

        Returns:
            A placeholder URL; the provider fails before any I/O.
        """
        return "https://custom-stub.example.com"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Reflect the request through untouched.

        Args:
            model: The model name.
            messages: The conversation.
            **kwargs: Remaining request fields.

        Returns:
            The request fields as given.
        """
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        """Return the payload unchanged.

        Args:
            response_data: The upstream payload.

        Returns:
            The same payload.
        """
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Wrap a status into an exception.

        Args:
            status_code: The upstream status.
            body: The upstream body.

        Returns:
            A provider error carrying both.
        """
        return ProviderError(status_code, body)

    async def make_request(self, request: dict) -> dict:
        """Fail every request with a hard provider error.

        Args:
            request: The upstream request.

        Returns:
            Never; raises instead.

        Raises:
            ProviderError: Always, with status 500.
        """
        raise ProviderError(500, {"error": "stub custom transport down"})


def _mixed_server(*backends: tuple) -> BridgeServer:
    """Build a server fronting the given members with the relay opted in.

    Args:
        backends: ``(provider, key, profile)`` balancing tuples.

    Returns:
        A server bound to an ephemeral port, not yet started.
    """
    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key="",
        backends=list(backends),
        allow_subscription_relay=True,
    )


def _park(server: BridgeServer, idx: int, cooldown: int) -> None:
    """Park one backend as if it had just failed.

    Args:
        server: The server whose health records to mutate.
        idx: The backend index to park.
        cooldown: The cooldown seconds to install.
    """
    health = server._backend_health[idx]
    health["healthy"] = False
    health["failed_at"] = time.monotonic()
    health["cooldown"] = cooldown


def _relay_calls(m: aioresponses) -> list:
    """Collect the relay upstream's recorded calls from an aioresponses mock.

    Args:
        m: The mock whose request log to scan.

    Returns:
        One entry per POST the bridge made to the relay's endpoint.
    """
    return [call for (method, url), calls in m.requests.items() for call in calls if str(url) == _UPSTREAM_URL]


@pytest.fixture
def fast_ladders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse every retry ladder's sleeps to nothing.

    The non-streaming walk backs off in real time (500s retry internally
    at 1/2/4s, then the empty ladder at 20/40s); the crossing under test
    fires only after exhaustion, so the waits buy the test nothing.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])


class TestSelectionVisibility:
    """Translated selections skip subscription members unless asked (R1/R2)."""

    def test_default_selection_never_returns_subscription_member(self) -> None:
        """A mixed pool's default selection ignores the healthy member."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        for _attempt in range(50):
            picked = server._select_backend()
            assert not isinstance(picked["provider"], AnthropicSubscriptionAdapter)

    def test_default_selection_with_only_subscription_healthy_raises(self) -> None:
        """No visible healthy member means the normal all-unhealthy raise (R10)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 400)
        with pytest.raises(AllBackendsUnhealthyError) as exc:
            server._select_backend()
        # The 503's cause list names only the visible members (R15).
        assert [b["name"] for b in exc.value.backends] == ["plain-0"]

    def test_default_near_expiry_gamble_ignores_subscription_member(self) -> None:
        """The near-expiry gamble picks the parked plain member, never the member (R11)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 10)
        picked = server._select_backend()
        assert not isinstance(picked["provider"], AnthropicSubscriptionAdapter)

    def test_default_fully_cooled_pool_does_not_gamble_into_member(self) -> None:
        """Everything parked far: the raise wins over gambling into the member (AC10)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 400)
        _park(server, 1, 900)
        with pytest.raises(AllBackendsUnhealthyError) as exc:
            server._select_backend()
        assert [b["name"] for b in exc.value.backends] == ["plain-0"]

    def test_require_subscription_selects_subscription_only(self) -> None:
        """The relay's own selection semantics are unchanged (R2)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        for _attempt in range(50):
            picked = server._select_backend(require_subscription=True)
            assert isinstance(picked["provider"], AnthropicSubscriptionAdapter)

    def test_allow_subscription_admits_the_member(self) -> None:
        """The arrival pick can still land on the member (R3)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        picked_adapters = {
            type(server._select_backend(allow_subscription=True)["provider"]).__name__ for _attempt in range(50)
        }
        assert picked_adapters == {"CustomAnthropicAdapter", "AnthropicSubscriptionAdapter"}

    def test_any_healthy_backend_matches_visibility(self) -> None:
        """Guards and selections see the same pool (R10)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 400)
        assert server._any_healthy_backend() is False
        assert server._any_healthy_backend(allow_subscription=True) is True
        assert server._any_healthy_backend(require_subscription=True) is True


class TestLastResortCrossing:
    """An exhausted translated Messages walk crosses into the raw relay (R4-R7)."""

    @pytest.mark.asyncio()
    async def test_streaming_exhausted_plain_walk_crosses_to_relay(self) -> None:
        """Plain member 500s; the crossing ships the original bytes to Anthropic (AC3)."""
        random.seed(1)  # deterministic backend selection: plain-0 (the 500ing member) first
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "translated-walk-failure"}), repeat=True)
                m.post(_UPSTREAM_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"})
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer cc-token"},
                ) as resp:
                    assert resp.status == 200, await resp.text()
                    assert await resp.read() == _SSE_REPLY
                relay_calls = _relay_calls(m)
                assert len(relay_calls) == 1
                shipped = relay_calls[0].kwargs
                # Byte-preservation and end-to-end headers: the crossing is the relay.
                assert shipped.get("data") == _RAW_STREAM_BODY
                assert shipped.get("headers", {}).get("Authorization") == "Bearer cc-token"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_non_streaming_exhausted_walk_crosses_to_relay(self, fast_ladders: None) -> None:
        """The non-streaming twin serves the relayed JSON verbatim (AC5)."""
        random.seed(1)  # deterministic backend selection: plain-0 (the 500ing member) first
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "translated-walk-failure"}), repeat=True)
                m.post(_UPSTREAM_URL, status=200, body=json.dumps(_UPSTREAM_REPLY))
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_BODY,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer cc-token"},
                ) as resp:
                    assert resp.status == 200, await resp.text()
                    assert (await resp.json())["content"][0]["text"] == "Hi"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_parked_member_yields_the_unchanged_terminal(self) -> None:
        """Member also parked: the plain walk's envelope stands, no relay call (AC6)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 1, 900)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "translated-walk-failure"}), repeat=True)
                m.post(_UPSTREAM_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"})
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 500, body_text
                    assert "translated-walk-failure" in body_text
                assert not _relay_calls(m)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_post_emission_failure_never_crosses(self) -> None:
        """A failure after the first client byte closes the stream; no second upstream call (AC7).

        Both members script the same partial-then-error stream, so whichever
        the arrival pick lands on: the client keeps the partial bytes, and
        neither a plain failover nor the crossing may fire a second upstream
        request.
        """
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        hello = (
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"id":"msg_p","type":"message","role":"assistant",'
            b'"content":[],"model":"claude-sonnet-4-6","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
            b"event: content_block_delta\n"
            b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        )
        error = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"boom\"}}\n\n"
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, body=hello + error, headers={"Content-Type": "text/event-stream"})
                m.post(_UPSTREAM_URL, body=hello + error, headers={"Content-Type": "text/event-stream"})
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    text = await resp.read()
                    assert b"Hello" in text
                total_upstream_calls = sum(len(calls) for calls in m.requests.values())
                assert total_upstream_calls == 1
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_crossed_relay_failure_yields_the_relay_envelope(self) -> None:
        """Relay attempt fails: the client sees the relay's envelope, once (AC11/R12)."""
        random.seed(1)  # deterministic backend selection: plain-0 first, so the walk exhausts
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "translated-walk-failure"}), repeat=True)
                m.post(
                    _UPSTREAM_URL,
                    callback=lambda url, **kwargs: CallbackResult(
                        status=429,
                        body=json.dumps({"error": {"type": "rate_limit_error", "message": "slow down"}}),
                        headers={"Retry-After": "60"},
                    ),
                    repeat=True,
                )
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 503, body_text
                    # The relay's own envelope: no translated-walk "down" text.
                    assert "translated-walk-failure" not in body_text
                    payload = json.loads(body_text)
                    assert payload["type"] == "error"
                assert len(_relay_calls(m)) == 1
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_only_subscription_healthy_reaches_the_crossing(self) -> None:
        """Guard parity: one plain (parked) + member healthy serves, never 500s (AC9)."""
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 400)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"})
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_custom_transport_exhaustion_crosses_on_non_streaming(self, fast_ladders: None) -> None:
        """A walk ending in a custom-transport ProviderError crosses too (AC12/R13)."""
        random.seed(1)  # deterministic backend selection: custom-0 (the failing member) first
        custom = (
            _FailingCustomProvider(),
            "",
            Profile(name="custom-0", provider="openai_subscription", model="m", auth_ref=str(uuid.uuid4())),
        )
        server = _mixed_server(custom, _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, status=200, body=json.dumps(_UPSTREAM_REPLY))
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    assert resp.status == 200, await resp.text()
                    assert (await resp.json())["content"][0]["text"] == "Hi"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_custom_transport_error_terminal_crosses_on_streaming(self, fast_ladders: None) -> None:
        """The streaming custom arm's error terminal crosses before its envelope (review round 2)."""
        random.seed(1)
        custom = (
            _FailingCustomProvider(),
            "",
            Profile(name="custom-0", provider="openai_subscription", model="m", auth_ref=str(uuid.uuid4())),
        )
        server = _mixed_server(custom, _subscription_member("sub-1"))
        cc_stream_body = (
            b'{\n  "model" : "test-model",\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
            b'  "max_tokens" : 8,\n  "stream" : true\n}'
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"})
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=cc_stream_body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 200, body_text
                    assert "Hi" in body_text
                assert len(_relay_calls(m)) == 1
        finally:
            await server.stop_async()


class TestCrossingFailureEnvelopes:
    """A relay attempt that fails inside the crossing renders the relay's envelope."""

    @staticmethod
    def _stall_relay(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the relay attempt itself stall.

        Args:
            monkeypatch: pytest monkeypatch fixture.
        """

        async def _raise_timeout(self: BridgeServer, request, body, raw):  # noqa: ANN001
            raise asyncio.TimeoutError()

        monkeypatch.setattr(BridgeServer, "_relay_subscription_messages", _raise_timeout)

    @pytest.mark.asyncio()
    async def test_stalled_crossing_answers_504_on_non_streaming(
        self, monkeypatch: pytest.MonkeyPatch, fast_ladders: None
    ) -> None:
        """The non-streaming crossing's own stall 504s — not 500 internal_error (review 2026-10-29)."""
        random.seed(1)  # deterministic backend selection: plain-0 (the 500ing member) first
        self._stall_relay(monkeypatch)
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "translated-walk-failure"}), repeat=True)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 504, body_text
                    assert "timed out" in body_text
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_stalled_crossing_answers_504_on_streaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The streaming arm keeps its 504 for a stalled crossing (parity pin)."""
        random.seed(1)
        self._stall_relay(monkeypatch)
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "translated-walk-failure"}), repeat=True)
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 504, body_text
                    assert "timed out" in body_text
        finally:
            await server.stop_async()


class TestEmptyTerminalCrossing:
    """The Messages empty terminals are pre-emission exhaustion points: they cross (§16)."""

    @pytest.mark.asyncio()
    async def test_non_streaming_empty_terminal_crosses_to_relay(self, fast_ladders: None) -> None:
        """Every translated member answers empty; the relay serves the turn (review 2026-10-29)."""
        random.seed(1)  # deterministic backend selection: plain-0 (the emptying member) first
        empty_reply = {
            "id": "msg_empty",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=200, body=json.dumps(empty_reply), repeat=True)
                m.post(_UPSTREAM_URL, status=200, body=json.dumps(_UPSTREAM_REPLY))
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_BODY,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 200, body_text
                    assert (await resp.json())["content"][0]["text"] == "Hi"
                assert len(_relay_calls(m)) == 1
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_streaming_empty_ladder_crosses_to_relay(self, fast_ladders: None) -> None:
        """A CC-shaped inbound whose streams never carry content crosses at the S11 terminal."""
        random.seed(1)
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        cc_stream_body = (
            b'{\n  "model" : "test-model",\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
            b'  "max_tokens" : 8,\n  "stream" : true\n}'
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=200, body=b"data: [DONE]\n\n", repeat=True)
                m.post(_UPSTREAM_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"})
                async with aiohttp.ClientSession() as session, session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=cc_stream_body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    body_text = await resp.text()
                    assert resp.status == 200, body_text
                    assert "Hi" in body_text
                assert len(_relay_calls(m)) == 1
        finally:
            await server.stop_async()

    def test_every_messages_empty_terminal_precedes_a_crossing(self) -> None:
        """Drift guard: each Messages handler offers the relay before its exhaustion terminals.

        Counting the ``_relay_last_resort`` calls per handler pins the wiring:
        ``_handle_messages`` carries its non-stream error terminal plus two
        empty gates; ``_stream_messages`` carries the plain error terminal,
        the plain catch-all terminal, the custom arm's error and empty
        terminals, and three empty gates. A new exhaustion terminal without a
        crossing shifts the count and fails here.
        """
        tree = ast.parse(Path(server_module.__file__).read_text(encoding="utf-8"))
        counts = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
                "_handle_messages",
                "_stream_messages",
            ):
                counts[node.name] = sum(
                    1
                    for sub in ast.walk(node)
                    if isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "_relay_last_resort"
                )
        assert counts == {"_handle_messages": 3, "_stream_messages": 6}


class TestRecoveryHoldVisibility:
    """The recovery hold's window sees the subscription member (review 2026-10-29)."""

    @pytest.mark.asyncio()
    async def test_hold_waits_for_the_subscription_member_recovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Translated members park 900 s, the member 120 s: the hold waits for the member.

        Falsifies the review's cell: the all-unhealthy raise's
        ``retry_after`` is the minimum over the candidates pass, and that
        pass admits the member when ``allow_subscription`` was requested —
        so ``_select_backend_or_hold`` holds 120 s and recovers into the
        member instead of answering 503 immediately.
        """
        clock = {"now": 1000.0}
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock["now"] += seconds

        monkeypatch.setattr(server_module.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(server_module.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(server_module, "_recovery_hold_jitter", lambda: 0.0)
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 900)
        _park(server, 1, 120)
        result = await server._select_backend_or_hold(_connected_request())
        assert result is None, "the hold must wait for the member's 120 s recovery, not 503"
        assert sleeps == [120.0]

    @pytest.mark.asyncio()
    async def test_hold_skips_when_only_beyond_window_recoveries_remain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every recovery beyond 300 s (member included) still answers at once."""
        monkeypatch.setattr(server_module.time, "monotonic", lambda: 1000.0)
        server = _mixed_server(_plain_member("plain-0"), _subscription_member("sub-1"))
        _park(server, 0, 900)
        _park(server, 1, 400)
        result = await server._select_backend_or_hold(_connected_request())
        assert isinstance(result, AllBackendsUnhealthyError)
        assert result.retry_after == 400


def _connected_request() -> object:
    """Return a request stub whose transport reports a connected client.

    Returns:
        A stand-in with the ``transport.is_closing`` shape the hold polls.
    """
    return SimpleNamespace(transport=SimpleNamespace(is_closing=lambda: False))
