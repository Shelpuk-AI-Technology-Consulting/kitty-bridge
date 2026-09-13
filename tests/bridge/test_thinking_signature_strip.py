"""Anthropic's thinking-signature 400 is recovered by stripping thinking and retrying once (KBR-238).

api.anthropic.com rejects an assistant turn whose ``thinking`` block has no
signature, carries an altered one, or — under the prefix check — follows an
edited earlier message.  Kitty produces exactly those histories: the translator
drops signatures, P5e and the M8 carrier inject unsigned blocks, and compaction
edits the signed prefix.  Anthropic's documented recovery, confirmed live on
2026-09-13 against ``claude-sonnet-5``, ``claude-opus-4-6`` and
``claude-fable-5-1``, is to strip every ``thinking`` and ``redacted_thinking``
block and retry once.  The error texts below are the ones those probes returned.

Unit tests pin the detector and the strip; bridge tests drive a real in-process
:class:`~kitty.bridge.server.BridgeServer` against ``aioresponses``.
"""

from __future__ import annotations

import copy
import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer, _is_thinking_signature_error, _strip_thinking_blocks
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.types import BridgeProtocol

#: Verbatim error messages returned by api.anthropic.com in the 2026-09-13 probes.
_MISSING_SIGNATURE = "messages.1.content.0.thinking.signature: Field required"
_INVALID_SIGNATURE = "messages.1.content.0: Invalid `signature` in `thinking` block"
_PREFIX_MISMATCH = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different "
    "conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to "
    '"drop_block". Content before this block differs from when it was created, first at `messages.0.content.0`.'
)

_NATIVE_BASE = "https://api.native.test/anthropic"
_NATIVE_URL = f"{_NATIVE_BASE}/v1/messages"

#: A turn-2 transcript whose assistant turn carries thinking, redacted thinking, text and a tool call.
_HISTORY = [
    {"role": "user", "content": "Check the weather in Paris."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "I should call the tool.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
]

_OK_REPLY = {
    "id": "msg_ok",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-6",
    "content": [{"type": "text", "text": "It is 18C."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _envelope(message: str) -> dict:
    """Wrap a message in Anthropic's error envelope.

    Args:
        message: The ``error.message`` text.

    Returns:
        The error body Anthropic returns with a 400.
    """
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


# ── R1: the detector ────────────────────────────────────────────────────────


@pytest.mark.parametrize("message", [_MISSING_SIGNATURE, _INVALID_SIGNATURE, _PREFIX_MISMATCH])
@pytest.mark.parametrize("as_text", [False, True], ids=["dict", "text"])
def test_the_live_signature_errors_are_recognised(message, as_text):
    """R1 — each rejection Anthropic returned is recognised, parsed or raw.

    Args:
        message: A verbatim live error message.
        as_text: Whether the body arrives as JSON text (the streaming path) rather than a dict.
    """
    body = _envelope(message)

    assert _is_thinking_signature_error(400, json.dumps(body) if as_text else body)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (500, _MISSING_SIGNATURE),
        (400, "messages.0.content: text content blocks must be non-empty"),
        (400, "The `content[].thinking` in the thinking mode must be passed back to the API."),
        (400, "Invalid API key signature"),
    ],
    ids=["5xx-same-text", "unrelated-400", "issue-32-roundtrip", "signature-without-thinking"],
)
def test_other_errors_are_not_recognised(status, message):
    """R1 — a backend fault, another validation error or the issue-#32 wording is not this rejection.

    Args:
        status: The upstream status.
        message: The error message.
    """
    assert not _is_thinking_signature_error(status, _envelope(message))


# ── R2: the strip ───────────────────────────────────────────────────────────


def test_strip_removes_thinking_from_assistant_turns_only_and_keeps_order():
    """R2 — thinking and redacted thinking go; text and tool calls stay in their order; user turns are untouched."""
    body = {"model": "m", "messages": copy.deepcopy(_HISTORY)}

    assert _strip_thinking_blocks(body) is True
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    assert body["messages"][0] == _HISTORY[0]
    assert body["messages"][2] == _HISTORY[2]


def test_strip_leaves_an_empty_content_list_for_a_thinking_only_turn():
    """R2 — Anthropic accepts ``content: []`` (live-probed), so the turn is kept rather than dropped."""
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}]},
        ]
    }

    assert _strip_thinking_blocks(body) is True
    assert body["messages"][1] == {"role": "assistant", "content": []}


def test_strip_never_mutates_the_messages_it_was_given():
    """R2 — the retry must not reach back into the request the next attempt re-serializes."""
    messages = copy.deepcopy(_HISTORY)
    body = {"messages": messages}

    _strip_thinking_blocks(body)

    assert messages == _HISTORY
    assert body["messages"] is not messages


def test_strip_reports_no_change_when_there_is_no_thinking():
    """R2 — a False result is what stops a retry that would re-send identical bytes."""
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
        ]
    }

    assert _strip_thinking_blocks(body) is False


# ── R3–R7: the bridge ───────────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    """Minimal launcher that selects the Messages API bridge protocol."""

    @property
    def name(self) -> str:
        """Return the launcher name.

        Returns:
            A fixed placeholder.
        """
        return "stub"

    @property
    def binary_name(self) -> str:
        """Return the launcher binary name.

        Returns:
            A fixed placeholder.
        """
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol the bridge serves.

        Returns:
            :attr:`BridgeProtocol.MESSAGES_API`.
        """
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        """Return an empty spawn configuration.

        Args:
            profile: Unused.
            bridge_port: Unused.
            resolved_key: Unused.

        Returns:
            A spawn configuration that changes nothing.
        """
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


def _native_server(*, pool: bool = False) -> BridgeServer:
    """Build a native ``custom_anthropic`` bridge, optionally as a two-member pool.

    Args:
        pool: Whether to build a two-member balancing pool on the same endpoint.

    Returns:
        An unstarted bridge.
    """
    config = {"base_url": _NATIVE_BASE}
    if not pool:
        return BridgeServer(
            adapter=_StubLauncher(),
            provider=CustomAnthropicAdapter(),
            resolved_key="key-solo",
            model="claude-opus-4-6",
            provider_config=config,
            host="127.0.0.1",
            port=0,
        )
    backends = []
    for name in ("member-1", "member-2"):
        profile = Profile(
            name=name,
            provider="custom_anthropic",
            model="claude-opus-4-6",
            auth_ref=str(uuid.uuid4()),
            provider_config=config,
        )
        backends.append((CustomAnthropicAdapter(), f"key-{name}", profile))
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="claude-opus-4-6",
        provider_config=config,
        backends=backends,
        host="127.0.0.1",
        port=0,
    )


def _sse_reply() -> str:
    """Render ``_OK_REPLY`` as an Anthropic SSE stream.

    Returns:
        The SSE body.
    """
    start = {**_OK_REPLY, "content": []}
    events = [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "It is 18C."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


async def _drive(server: BridgeServer, url: str, replies: list[tuple[int, object]], *, stream: bool = False):
    """Serve scripted upstream replies in order and return what the client and upstream saw.

    Args:
        server: An unstarted bridge.
        url: The upstream URL to mock.
        replies: ``(status, body)`` per upstream call; a dict body is JSON, a str body is SSE.
        stream: Whether the client request streams.

    Returns:
        ``(status, client_body_text, upstream_calls)`` where each call is ``(json_body, headers)``.
    """
    calls: list[tuple[dict, dict]] = []

    def respond(u, **kwargs):
        """Record the call and answer with the next scripted reply.

        Args:
            u: The matched URL.
            **kwargs: The request's keyword arguments.

        Returns:
            The scripted response.
        """
        calls.append((copy.deepcopy(kwargs.get("json")), dict(kwargs.get("headers") or {})))
        status, body = replies[min(len(calls), len(replies)) - 1]
        if isinstance(body, str):
            return CallbackResult(status=status, headers={"Content-Type": "text/event-stream"}, body=body)
        return CallbackResult(status=status, content_type="application/json", body=json.dumps(body))

    request = {"model": "claude-opus-4-6", "max_tokens": 1024, "stream": stream, "messages": copy.deepcopy(_HISTORY)}
    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        mocked.post(url, callback=respond, repeat=True)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=request) as resp,
            ):
                return resp.status, await resp.text(), calls
        finally:
            await server.stop_async()


def _thinking_types(call_body: dict) -> list[str]:
    """Return the thinking block types an upstream body's assistant turns carry.

    Args:
        call_body: A captured upstream JSON body.

    Returns:
        The ``thinking``/``redacted_thinking`` types found, in order.
    """
    return [
        block["type"]
        for msg in call_body["messages"]
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
        for block in msg["content"]
        if block.get("type") in ("thinking", "redacted_thinking")
    ]


@pytest.mark.asyncio
async def test_non_streaming_signature_rejection_is_stripped_and_retried_once():
    """R3 — the client gets the answer; the retry is the same request without thinking."""
    status, text, calls = await _drive(
        _native_server(), _NATIVE_URL, [(400, _envelope(_INVALID_SIGNATURE)), (200, _OK_REPLY)]
    )

    assert status == 200, text
    assert json.loads(text)["content"][0]["text"] == "It is 18C."
    assert len(calls) == 2
    assert _thinking_types(calls[0][0]) == ["thinking", "redacted_thinking"]
    assert _thinking_types(calls[1][0]) == []


@pytest.mark.asyncio
async def test_a_pool_retries_the_same_backend_and_keeps_it_healthy():
    """R4 — the rejection is the bridge's transcript, not a sick backend: no failover, no quarantine."""
    server = _native_server(pool=True)

    status, text, calls = await _drive(server, _NATIVE_URL, [(400, _envelope(_PREFIX_MISMATCH)), (200, _OK_REPLY)])

    assert status == 200, text
    assert len(calls) == 2
    assert calls[0][1].get("x-api-key") == calls[1][1].get("x-api-key")
    assert all(health["healthy"] and not health.get("failure_count") for health in server._backend_health)


@pytest.mark.asyncio
async def test_streaming_signature_rejection_is_stripped_and_retried_once():
    """R5 — the streaming Messages handler recovers the same way."""
    status, text, calls = await _drive(
        _native_server(), _NATIVE_URL, [(400, _envelope(_MISSING_SIGNATURE)), (200, _sse_reply())], stream=True
    )

    assert status == 200, text
    assert "It is 18C." in text
    assert len(calls) == 2
    assert _thinking_types(calls[1][0]) == []


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_a_second_rejection_after_the_strip_is_surfaced_not_looped(stream):
    """R6 — one strip per request; if the stripped body is rejected too, the client sees an error.

    Args:
        stream: Whether the client request streams.
    """
    status, text, calls = await _drive(
        _native_server(), _NATIVE_URL, [(400, _envelope(_INVALID_SIGNATURE))], stream=stream
    )

    assert status == 400, text
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_chat_completions_upstream_is_not_stripped():
    """R7 — a Chat Completions body carries no thinking blocks, so there is nothing to strip and no retry."""
    server = BridgeServer(
        adapter=_StubLauncher(),
        provider=CustomOpenAIAdapter(),
        resolved_key="key-cc",
        model="gpt-x",
        provider_config={"base_url": "https://api.cc.test/v1"},
        host="127.0.0.1",
        port=0,
    )

    status, _text, calls = await _drive(
        server, "https://api.cc.test/v1/chat/completions", [(400, _envelope(_INVALID_SIGNATURE)), (200, {})]
    )

    assert status == 400
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_stripped_stream_is_not_repaired_back_into_an_unsigned_carrier():
    """R6b — after the strip, the issue-#32 carrier repair does not fire, so the two cannot ping-pong.

    The carrier is an unsigned thinking block, exactly what the signature check
    rejects.  Were the repair allowed after a strip, it would re-add one, the
    strip would remove it again, and the pair would spend every attempt.
    """
    roundtrip = _envelope("The `content[].thinking` in the thinking mode must be passed back to the API.")

    status, _text, calls = await _drive(
        _native_server(), _NATIVE_URL, [(400, _envelope(_INVALID_SIGNATURE)), (400, roundtrip)], stream=True
    )

    assert status == 400
    assert len(calls) == 2
