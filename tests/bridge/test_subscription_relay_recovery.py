"""Bridge-level tests: a rejected relay attempt never stalls a mixed pool.

Reproduced live on 2026-11-04 with one interactive Claude Code session on a
``minimax-mid`` + ``anthropic`` pool: MiniMax's turns left 64-hex-signed
``thinking`` blocks in the history, a later turn reached the raw relay, Anthropic
answered ``messages.3.content.0: Invalid `signature` in `thinking` block``, and
the member was parked 300 s behind a 503 that named one backend while MiniMax sat
healthy. A 429 on a background request produced the same 503. These tests pin
the repair: the relay recovers a signature rejection itself, a failed relay falls
over to the translated members, and a request-shaped rejection parks nobody.
Design of record: ``SYSTEM_DESIGN.md`` §16.

**Layer.** Collects as ``l1`` alongside every other loopback test in this
directory (``tests/layers.py`` maps only ``tests/integration/`` away).
"""

from __future__ import annotations

import json
import logging
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.anthropic_subscription import AnthropicSubscriptionAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter

#: An unbounded relay loop must fail here, not spin for the client's five-minute timeout.
pytestmark = pytest.mark.timeout(120)

#: Endpoint of the raw subscription relay.
_RELAY_URL = "https://api.anthropic.com/v1/messages"

#: Endpoint of the ordinary API-key member.
_APIKEY_URL = "https://api-key.example.com/v1/messages"

#: The OAuth token Claude Code presents; it must never reach an API-key member.
_OAUTH_TOKEN = "cc-oauth-token"

#: Anthropic's rejection text, as returned to the live reproduction.
_SIGNATURE_REJECTION = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "messages.1.content.0: Invalid `signature` in `thinking` block",
    },
}

#: A history whose assistant turn carries reasoning another pool member signed.
_FOREIGN_THINKING_BODY = (
    b'{ "model" : "claude-sonnet-4-6", "max_tokens" : 8, "messages" : ['
    b'{"role":"user","content":"Hello"},'
    b'{"role":"assistant","content":['
    b'{"type":"thinking","thinking":"from minimax","signature":"3b7b0000"},'
    b'{"type":"text","text":"Hi"}]},'
    b'{"role":"user","content":"Again"}] }'
)

#: Non-canonical bytes with no reasoning in the history.
_PLAIN_BODY = b'{ "model" : "claude-sonnet-4-6", "max_tokens" : 8, "messages" : [{"role":"user","content":"Hello"}] }'

#: Streaming variant of the plain bytes.
_PLAIN_STREAM_BODY = (
    b'{ "model" : "claude-sonnet-4-6", "max_tokens" : 8, "stream" : true, '
    b'"messages" : [{"role":"user","content":"Hello"}] }'
)

#: A non-streaming Messages reply.
_REPLY = {
    "id": "msg_recovery",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Served"}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

#: A streaming Messages reply.
_SSE_REPLY = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant",'
    b'"content":[],"model":"claude-sonnet-4-6","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Served"}}\n\n'
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":0}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


#: A signature only Anthropic's replies in these tests ever carry.
_ISSUED_SIGNATURE = "CAQSanthropicissuedsignature"


def _history(signatures: list[str]) -> bytes:
    """Build a request whose assistant turns each carry one signed thinking block.

    Args:
        signatures: One signature per assistant turn, oldest first.

    Returns:
        The request bytes.
    """
    messages: list[dict] = [{"role": "user", "content": "Hello"}]
    for signature in signatures:
        block = {"type": "thinking", "thinking": "reasoning", "signature": signature}
        messages.append({"role": "assistant", "content": [block, {"type": "text", "text": "Hi"}]})
        messages.append({"role": "user", "content": "Again"})
    return json.dumps({"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": messages}).encode()


def _thinking_reply(signature: str) -> dict:
    """Build a non-streaming reply carrying reasoning Anthropic signed.

    Args:
        signature: The signature to carry.

    Returns:
        The reply body.
    """
    block = {"type": "thinking", "thinking": "", "signature": signature}
    return {**_REPLY, "content": [block, {"type": "text", "text": "Served"}]}


def _thinking_sse_reply(signature: str) -> bytes:
    """Build a streaming reply carrying reasoning Anthropic signed.

    Args:
        signature: The signature the ``signature_delta`` event carries.

    Returns:
        The SSE bytes.
    """
    delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": signature}}
    return b"event: content_block_delta\ndata: " + json.dumps(delta).encode() + b"\n\n" + _SSE_REPLY


def _api_key_member(name: str = "metered") -> tuple:
    """Build an ordinary API-key member.

    Args:
        name: Profile name for health attribution.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
    """
    profile = Profile(
        name=name,
        provider="custom_anthropic",
        model="claude-sonnet-4-6",
        auth_ref=str(uuid.uuid4()),
        provider_config={"base_url": "https://api-key.example.com"},
    )
    return (CustomAnthropicAdapter(), "member-key", profile)


def _subscription_member(name: str = "sub") -> tuple:
    """Build a tokenless subscription member.

    Args:
        name: Profile name for health attribution.

    Returns:
        A ``(provider, key, profile)`` balancing tuple.
    """
    profile = Profile(
        name=name,
        provider="anthropic_subscription",
        model="claude-sonnet-4-6",
        auth_ref=str(uuid.uuid4()),
    )
    return (AnthropicSubscriptionAdapter(), "", profile)


def _pool(*backends: tuple) -> BridgeServer:
    """Build a Claude-launched server over the given members.

    Args:
        backends: Balancing tuples.

    Returns:
        A server on an ephemeral port, not yet started.
    """
    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key="",
        backends=list(backends),
        allow_subscription_relay=True,
    )


@pytest.fixture(autouse=True)
def _arrival_lands_on_the_last_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make selection deterministic: the last visible member is always picked.

    Every pool below lists the subscription member last, so the arrival pick
    takes the relay while a translated selection, which cannot see that member,
    takes the API-key member.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setattr(server_module.random, "choices", lambda tier, weights=None, k=1: [tier[-1]])


def _scripted(replies: list[tuple[int, object]], calls: list) -> object:
    """Build a callback serving replies in order, repeating the last.

    Args:
        replies: ``(status, body)`` pairs.
        calls: Receives one ``(data_bytes, headers, json_body)`` entry per call.

    Returns:
        An ``aioresponses`` callback.
    """

    def respond(url, **kwargs):
        """Serve the next scripted reply and record what was shipped.

        Args:
            url: The matched upstream URL.
            **kwargs: The request's keyword arguments.

        Returns:
            The scripted response.
        """
        calls.append((kwargs.get("data"), dict(kwargs.get("headers") or {}), kwargs.get("json")))
        status, body = replies[min(len(calls), len(replies)) - 1]
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
        return CallbackResult(status=status, body=payload)

    return respond


async def _post(port: int, data: bytes) -> tuple[int, bytes]:
    """Post one Claude Code request to the bridge.

    Args:
        port: The bridge port.
        data: The request bytes.

    Returns:
        The client-visible status and body.
    """
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}/v1/messages",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {_OAUTH_TOKEN}"},
        ) as resp,
    ):
        return resp.status, await resp.read()


def _parked_kinds(server: BridgeServer) -> list[str | None]:
    """List the failure kinds of every parked member.

    Args:
        server: The server to inspect.

    Returns:
        One failure kind per unhealthy member.
    """
    return [health["last_failure_kind"] for health in server._backend_health if not health["healthy"]]


class TestThinkingSignatureRecovery:
    """The relay strips reasoning Anthropic refuses and serves the turn itself."""

    @pytest.mark.asyncio()
    async def test_signature_rejection_retries_same_member_without_the_thinking(self) -> None:
        """Serve the turn from the subscription after removing the refused block."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], relay_calls),
                    repeat=True,
                )
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                status, _body = await _post(port, _FOREIGN_THINKING_BODY)
        finally:
            await server.stop_async()

        assert status == 200
        assert relay_calls[0][0] == _FOREIGN_THINKING_BODY, "the first attempt must stay byte-exact"
        retried = json.loads(relay_calls[1][0])
        assert retried["messages"][1]["content"] == [{"type": "text", "text": "Hi"}]
        assert key_calls == []
        assert _parked_kinds(server) == []

    @pytest.mark.asyncio()
    async def test_later_turns_strip_before_relaying_without_a_second_rejection(self) -> None:
        """Spend the rejected round trip once per session, not once per turn."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], relay_calls),
                    repeat=True,
                )
                await _post(port, _FOREIGN_THINKING_BODY)
                relay_calls_after_first_turn = len(relay_calls)
                status, _body = await _post(port, _FOREIGN_THINKING_BODY)
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == relay_calls_after_first_turn + 1
        assert "thinking" not in relay_calls[-1][0].decode()

    @pytest.mark.asyncio()
    async def test_request_without_thinking_stays_byte_exact_after_a_rejection(self) -> None:
        """Re-serialize only a body the strip actually changed."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], relay_calls),
                    repeat=True,
                )
                await _post(port, _FOREIGN_THINKING_BODY)
                await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert relay_calls[-1][0] == _PLAIN_BODY

    @pytest.mark.asyncio()
    async def test_single_profile_session_recovers_too(self) -> None:
        """A resumed session on the direct profile gets the same recovery."""
        server = BridgeServer(
            adapter=None,
            provider=AnthropicSubscriptionAdapter(),
            resolved_key="",
            allow_subscription_relay=True,
        )
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], relay_calls),
                    repeat=True,
                )
                status, _body = await _post(port, _FOREIGN_THINKING_BODY)
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == 2
        assert "from minimax" not in relay_calls[1][0].decode()

    @pytest.mark.asyncio()
    async def test_streaming_request_recovers_with_one_stream_reaching_the_client(self) -> None:
        """Recover before any byte is written, on the path Claude Code normally uses."""
        body = _FOREIGN_THINKING_BODY.replace(b'"max_tokens" : 8,', b'"max_tokens" : 8, "stream" : true,')
        server = _pool(_api_key_member(), _subscription_member())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _SSE_REPLY)], []),
                    repeat=True,
                )
                status, client_bytes = await _post(port, body)
        finally:
            await server.stop_async()

        assert status == 200
        assert client_bytes.count(b"event: message_start") == 1

    @pytest.mark.asyncio()
    async def test_many_foreign_blocks_cost_one_rejected_attempt(self) -> None:
        """Remove every foreign block at once rather than one refused turn at a time."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], relay_calls),
                    repeat=True,
                )
                status, _body = await _post(port, _history(["a0", "a1", "a2", "a3"]))
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == 2
        retried = relay_calls[1][0].decode()
        assert not any(signature in retried for signature in ("a0", "a1", "a2", "a3"))

    @pytest.mark.asyncio()
    async def test_strips_are_counted_for_operators(self) -> None:
        """Show relay strips on the same counter as every other M17 site, before and after the flag."""
        server = _pool(_api_key_member(), _subscription_member())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], []), repeat=True)
                await _post(port, _FOREIGN_THINKING_BODY)
                await _post(port, _FOREIGN_THINKING_BODY)
        finally:
            await server.stop_async()

        assert server._stats_thinking_stripped == {1: 2}

    @pytest.mark.asyncio()
    async def test_malformed_signature_reaches_anthropic_not_a_500(self) -> None:
        """Treat an unhashable signature as unrecognised rather than crashing the strip."""
        body = _FOREIGN_THINKING_BODY.replace(b'"signature":"3b7b0000"', b'"signature":["not","a","string"]')
        server = _pool(_api_key_member(), _subscription_member())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, callback=_scripted([(400, _SIGNATURE_REJECTION), (200, _REPLY)], []), repeat=True)
                status, _body = await _post(port, body)
                second_status, _second_body = await _post(port, body)
        finally:
            await server.stop_async()

        assert (status, second_status) == (200, 200)

    @pytest.mark.asyncio()
    async def test_refusal_of_issued_reasoning_escalates_and_then_stops(self) -> None:
        """Escalate past the filter when Anthropic refuses its own blocks too."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted(
                        [
                            (200, _thinking_reply(_ISSUED_SIGNATURE)),
                            (400, _SIGNATURE_REJECTION),
                            (400, _SIGNATURE_REJECTION),
                            (200, _REPLY),
                        ],
                        relay_calls,
                    ),
                    repeat=True,
                )
                await _post(port, _PLAIN_BODY)
                status, _body = await _post(port, _history(["foreign-sig", _ISSUED_SIGNATURE]))
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == 4
        assert _ISSUED_SIGNATURE in relay_calls[2][0].decode(), "the filter keeps it on the first retry"
        assert _ISSUED_SIGNATURE not in relay_calls[3][0].decode(), "the escalation removes it"

    @pytest.mark.asyncio()
    async def test_endless_refusal_ends_in_fall_over_with_the_unedited_body(self) -> None:
        """Bound the recovery: nothing left to strip means the pool serves the original."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        key_calls: list = []
        history = _history(["foreign-sig", _ISSUED_SIGNATURE])
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted(
                        [(200, _thinking_reply(_ISSUED_SIGNATURE)), (400, _SIGNATURE_REJECTION)], relay_calls
                    ),
                    repeat=True,
                )
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                await _post(port, _PLAIN_BODY)
                status, _body = await _post(port, history)
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == 4, "one first reply, then raw + filtered + escalated"
        data, _headers, sent_json = key_calls[0]
        shipped = json.dumps(sent_json) if sent_json is not None else data.decode()
        assert "foreign-sig" in shipped
        assert _parked_kinds(server) == []

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_reasoning_anthropic_signed_survives_the_strip(self, streaming: bool) -> None:
        """Keep the model's own reasoning on later turns while foreign blocks go."""
        issued = _ISSUED_SIGNATURE
        first_reply = _thinking_sse_reply(issued) if streaming else _thinking_reply(issued)
        first_body = _PLAIN_STREAM_BODY if streaming else _PLAIN_BODY
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted(
                        [(200, first_reply), (400, _SIGNATURE_REJECTION), (200, _REPLY)],
                        relay_calls,
                    ),
                    repeat=True,
                )
                await _post(port, first_body)
                status, _body = await _post(port, _history(["foreign-sig", issued]))
        finally:
            await server.stop_async()

        assert status == 200
        retried = relay_calls[-1][0].decode()
        assert issued in retried
        assert "foreign-sig" not in retried

    @pytest.mark.asyncio()
    async def test_refused_issued_signature_is_not_trusted_again(self) -> None:
        """Stop re-sending a block Anthropic refused despite having signed it."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted(
                        [
                            (200, _thinking_reply(_ISSUED_SIGNATURE)),
                            (400, _SIGNATURE_REJECTION),
                            (200, _REPLY),
                            (200, _REPLY),
                        ],
                        relay_calls,
                    ),
                    repeat=True,
                )
                await _post(port, _PLAIN_BODY)
                await _post(port, _history([_ISSUED_SIGNATURE]))
                relay_calls_before = len(relay_calls)
                status, _body = await _post(port, _history([_ISSUED_SIGNATURE]))
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == relay_calls_before + 1
        assert _ISSUED_SIGNATURE not in relay_calls[-1][0].decode()

    @pytest.mark.asyncio()
    async def test_unrecoverable_signature_rejection_falls_over_without_parking(self) -> None:
        """With nothing left to strip, another member serves and nobody is parked."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, callback=_scripted([(400, _SIGNATURE_REJECTION)], relay_calls), repeat=True)
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                status, _body = await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert status == 200
        assert len(relay_calls) == 1
        assert len(key_calls) == 1
        assert _parked_kinds(server) == []


class TestFalloverToTranslatedMembers:
    """A failed relay with no subscription sibling is served by the rest of the pool."""

    @pytest.mark.asyncio()
    @pytest.mark.parametrize(
        ("status", "kind"),
        [(429, "rate_limit"), (500, "hard"), (401, "auth")],
    )
    async def test_member_shaped_failure_parks_and_another_member_serves(self, status: int, kind: str) -> None:
        """Park the failing member, answer from the pool, and report no 503."""
        server = _pool(_api_key_member(), _subscription_member())
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, status=status, body=json.dumps({"error": {"message": "no"}}), repeat=True)
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                client_status, body = await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert client_status == 200, body
        assert b"Served" in body
        assert len(key_calls) == 1
        assert _parked_kinds(server) == [kind]

    @pytest.mark.asyncio()
    async def test_streaming_failure_before_any_byte_falls_over(self) -> None:
        """Serve a streaming request from the pool when the relay fails pre-emission."""
        server = _pool(_api_key_member(), _subscription_member())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, status=429, body=json.dumps({"error": {"message": "slow"}}), repeat=True)
                m.post(_APIKEY_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"}, repeat=True)
                status, body = await _post(port, _PLAIN_STREAM_BODY)
        finally:
            await server.stop_async()

        assert status == 200, body
        assert b"Served" in body

    @pytest.mark.asyncio()
    async def test_translated_member_never_receives_the_oauth_token(self) -> None:
        """The fall-over authenticates with the member's own key, not Claude Code's."""
        server = _pool(_api_key_member(), _subscription_member())
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, status=429, body=json.dumps({"error": {"message": "slow"}}), repeat=True)
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        _data, headers, _json = key_calls[0]
        assert _OAUTH_TOKEN not in json.dumps(headers)
        assert "member-key" in json.dumps(headers)

    @pytest.mark.asyncio()
    async def test_whole_pool_down_still_answers_503(self) -> None:
        """Report unavailability only when no member at all can take the request."""
        server = _pool(_api_key_member(), _subscription_member())
        server._mark_backend_unhealthy(0, failure_kind="rate_limit", cooldown=600)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, status=429, body=json.dumps({"error": {"message": "slow"}}), repeat=True)
                status, body = await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert status == 503
        assert b"All 2 backends" in body, "the 503 must not name one backend while the whole pool is out"

    @pytest.mark.asyncio()
    async def test_recovery_edits_do_not_reach_the_translated_member(self) -> None:
        """Serve the fall-over from the body Claude Code sent, not the relay's stripped copy."""
        server = _pool(_api_key_member(), _subscription_member())
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _RELAY_URL,
                    callback=_scripted([(400, _SIGNATURE_REJECTION), (500, {"error": "down"})], []),
                    repeat=True,
                )
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                status, _body = await _post(port, _FOREIGN_THINKING_BODY)
        finally:
            await server.stop_async()

        assert status == 200
        data, _headers, sent_json = key_calls[0]
        shipped = json.dumps(sent_json) if sent_json is not None else data.decode()
        assert "from minimax" in shipped

    @pytest.mark.asyncio()
    async def test_transport_failure_falls_over_after_its_grace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve from the pool when the wire to Anthropic stays down."""
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.0,))
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_PERIOD", 0.0)
        server = _pool(_api_key_member(), _subscription_member())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, exception=aiohttp.ClientConnectionError("down"), repeat=True)
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], []), repeat=True)
                status, body = await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert status == 200, body
        assert _parked_kinds(server) == ["transport"]


class TestRequestShapedRejection:
    """A rejection of the request says nothing about the member's health."""

    #: A 400 Anthropic would repeat for this body from any session.
    _REJECTION = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}

    @pytest.mark.asyncio()
    async def test_400_falls_over_without_parking_the_member(self) -> None:
        """Let a member that compacts serve what the byte-exact relay could not."""
        server = _pool(_api_key_member(), _subscription_member())
        key_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, status=400, body=json.dumps(self._REJECTION), repeat=True)
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], key_calls), repeat=True)
                status, _body = await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert status == 200
        assert len(key_calls) == 1
        assert _parked_kinds(server) == []

    @pytest.mark.asyncio()
    async def test_failed_translated_walk_does_not_relay_the_same_body_again(self) -> None:
        """Keep a refused request off the last-resort crossing, and surface the walk's error."""
        server = _pool(_api_key_member(), _subscription_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, callback=_scripted([(400, self._REJECTION)], relay_calls), repeat=True)
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "down"}), repeat=True)
                status, _body = await _post(port, _PLAIN_STREAM_BODY)
        finally:
            await server.stop_async()

        assert len(relay_calls) == 1
        assert status == 500, "the translated walk's own terminal answers; the relay's reason is in the log"

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("rejected", [400, 413, 422])
    async def test_subscription_only_pool_surfaces_the_rejection_once(self, rejected: int) -> None:
        """Show the real rejection after one post when no other member can serve it."""
        server = _pool(_subscription_member("sub-a"))
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_RELAY_URL, callback=_scripted([(rejected, self._REJECTION)], relay_calls), repeat=True)
                status, body = await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert status == rejected
        assert b"unhealthy" not in body
        assert len(relay_calls) == 1
        assert _parked_kinds(server) == []

    @pytest.mark.asyncio()
    async def test_last_resort_crossing_surfaces_a_rejection_once(self) -> None:
        """End the crossing on its first refusal: its translated walk is already exhausted."""
        server = _pool(_subscription_member(), _api_key_member())
        relay_calls: list = []
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_APIKEY_URL, status=500, body=json.dumps({"error": "down"}), repeat=True)
                m.post(_RELAY_URL, callback=_scripted([(422, self._REJECTION)], relay_calls), repeat=True)
                status, _body = await _post(port, _PLAIN_STREAM_BODY)
        finally:
            await server.stop_async()

        assert status == 422
        assert len(relay_calls) == 1


class TestRelayFailureDiagnostics:
    """A relay rejection leaves the upstream's own words in the log."""

    @pytest.mark.asyncio()
    async def test_rejection_body_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Record why the member was rejected, since the park only says its kind."""
        server = _pool(_api_key_member(), _subscription_member())
        port = await server.start_async()
        try:
            with (
                caplog.at_level(logging.ERROR, logger="kitty.bridge.server"),
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
            ):
                m.post(_RELAY_URL, status=500, body=json.dumps({"error": {"message": "overloaded-now"}}), repeat=True)
                m.post(_APIKEY_URL, callback=_scripted([(200, _REPLY)], []), repeat=True)
                await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert "overloaded-now" in caplog.text

    @pytest.mark.asyncio()
    async def test_single_profile_rejection_body_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Record the upstream's words on the direct profile too, which parks nobody."""
        server = BridgeServer(
            adapter=None,
            provider=AnthropicSubscriptionAdapter(),
            resolved_key="",
            allow_subscription_relay=True,
        )
        port = await server.start_async()
        try:
            with (
                caplog.at_level(logging.ERROR, logger="kitty.bridge.server"),
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
            ):
                m.post(_RELAY_URL, status=429, body=json.dumps({"error": {"message": "slow-down-now"}}), repeat=True)
                await _post(port, _PLAIN_BODY)
        finally:
            await server.stop_async()

        assert "slow-down-now" in caplog.text
