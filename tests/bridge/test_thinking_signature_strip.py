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

from kitty.bridge import server as server_module
from kitty.bridge.server import (
    BridgeServer,
    UpstreamError,
    _is_thinking_signature_error,
    _recover_rejected_thinking,
    _strip_thinking_blocks,
)
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

#: A longer transcript: a signed turn Anthropic can no longer verify at index 1, and a later signed turn at index 3.
_LONG_HISTORY = [
    *_HISTORY,
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Now Rome.", "signature": "sig-3"},
            {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "Rome"}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "22C"}]},
]

#: Four signed assistant turns (indices 1, 3, 5, 7): enough for two targeted strips to leave thinking for the third.
_FOUR_TURN_HISTORY = [
    *_LONG_HISTORY,
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Done.", "signature": "sig-5"},
            {"type": "text", "text": "Both checked."},
        ],
    },
    {"role": "user", "content": "And Oslo?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Oslo next.", "signature": "sig-7"},
            {"type": "text", "text": "Checking Oslo."},
        ],
    },
    {"role": "user", "content": "Thanks."},
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


def _rejection_at(message_index: int) -> dict:
    """Return Anthropic's invalid-signature rejection naming a given message.

    Args:
        message_index: The index the error path names.

    Returns:
        The error body.
    """
    return _envelope(f"messages.{message_index}.content.0: Invalid `signature` in `thinking` block")


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


def test_strip_removes_every_thinking_block_and_keeps_the_rest_in_order():
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


def test_a_targeted_strip_keeps_thinking_after_the_named_message():
    """R2b — only thinking at or before the rejected message goes; the later, still-valid turn keeps its reasoning.

    Removing thinking from the front of the history leaves later blocks valid
    (live-probed on ``claude-fable-5-1``), so the model's newest reasoning survives.
    """
    body = {"messages": copy.deepcopy(_LONG_HISTORY)}

    assert _strip_thinking_blocks(body, through_message=1) is True
    assert [b["type"] for b in body["messages"][1]["content"]] == ["text", "tool_use"]
    assert body["messages"][3]["content"][0] == {"type": "thinking", "thinking": "Now Rome.", "signature": "sig-3"}


def test_a_targeted_strip_also_removes_unsigned_thinking_anywhere():
    """R2b — an unsigned block can never verify, so it goes in the same pass instead of costing its own rejection."""
    history = copy.deepcopy(_LONG_HISTORY)
    history[3]["content"][0] = {"type": "thinking", "thinking": ""}
    body = {"messages": history}

    assert _strip_thinking_blocks(body, through_message=1) is True
    assert [b["type"] for b in body["messages"][3]["content"]] == ["tool_use"]


@pytest.mark.parametrize(
    ("error", "strips_done", "later_turn_keeps_thinking"),
    [
        (_envelope(_INVALID_SIGNATURE), 0, True),
        (_envelope(_INVALID_SIGNATURE), 1, True),
        (_envelope(_INVALID_SIGNATURE), 2, False),
        (_envelope("thinking.signature: Field required somewhere"), 0, False),
    ],
    ids=["first-strip-targeted", "second-strip-targeted", "third-strip-everything", "no-path-everything"],
)
def test_recovery_targets_the_named_message_then_escalates(error, strips_done, later_turn_keeps_thinking):
    """R2c — two targeted strips, then everything; a rejection naming no message strips everything at once.

    Args:
        error: The upstream rejection body.
        strips_done: Strips already made against the body.
        later_turn_keeps_thinking: Whether the still-valid turn at index 3 should keep its thinking.
    """
    body = {"messages": copy.deepcopy(_LONG_HISTORY)}

    assert _recover_rejected_thinking(body, error, strips_done) is True
    assert (body["messages"][3]["content"][0]["type"] == "thinking") is later_turn_keeps_thinking


def test_recovery_stops_after_three_strips():
    """R2c — the cap bounds the loop however the upstream answers."""
    body = {"messages": copy.deepcopy(_LONG_HISTORY)}

    assert _recover_rejected_thinking(body, _envelope(_INVALID_SIGNATURE), 3) is False
    assert body["messages"] == _LONG_HISTORY


def test_a_targeted_strip_that_finds_nothing_falls_back_to_everything():
    """R2c — a path naming a message with no thinking must not end the recovery while broken thinking remains."""
    body = {"messages": copy.deepcopy(_LONG_HISTORY)}
    error = _envelope("messages.0.content.0: Invalid `signature` in `thinking` block")

    assert _recover_rejected_thinking(body, error, 0) is True
    assert all(
        b["type"] not in ("thinking", "redacted_thinking")
        for m in body["messages"]
        if isinstance(m["content"], list)
        for b in m["content"]
    )


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


async def _drive(
    server: BridgeServer,
    url: str,
    replies: list[tuple[int, object]],
    *,
    stream: bool = False,
    history: list[dict] | None = None,
):
    """Serve scripted upstream replies in order and return what the client and upstream saw.

    Args:
        server: An unstarted bridge.
        url: The upstream URL to mock.
        replies: ``(status, body)`` per upstream call; a dict body is JSON, a str body is SSE.
        stream: Whether the client request streams.
        history: The agent's transcript; ``_HISTORY`` when omitted.

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

    request = {
        "model": "claude-opus-4-6",
        "max_tokens": 1024,
        "stream": stream,
        "messages": copy.deepcopy(history or _HISTORY),
    }
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
async def test_every_strip_is_counted_per_backend_for_the_operator():
    """R6 (KBR-228 comment 4) — each strip lands in ``/stats``' per-backend counter.

    After KBR-228 a strip means the history was edited and valid reasoning was
    lost; the counter is how a compaction-heavy session shows that in
    ``/stats`` instead of only a WARNING log line.  Three rejections give each
    of the three strips something to remove; the fourth call ships the fully
    stripped body and succeeds.
    """
    server = _native_server()

    rejections = [(400, _rejection_at(index)) for index in (1, 3, 5)]
    status, text, calls = await _drive(
        server, _NATIVE_URL, [*rejections, (200, _OK_REPLY)], history=_FOUR_TURN_HISTORY
    )

    assert status == 200, text
    assert len(calls) == 4
    stats = server._session_stats()
    assert stats["thinking_stripped"] == 3
    # All three strips hit the same backend: the retry never failed over.
    assert stats["backends"][0]["thinking_stripped"] == 3


@pytest.mark.asyncio
async def test_streaming_signature_rejection_is_stripped_and_retried_once():
    """R5 — the streaming Messages handler recovers the same way, and counts the strip."""
    server = _native_server()
    status, text, calls = await _drive(
        server, _NATIVE_URL, [(400, _envelope(_MISSING_SIGNATURE)), (200, _sse_reply())], stream=True
    )

    assert status == 200, text
    assert "It is 18C." in text
    assert len(calls) == 2
    assert _thinking_types(calls[1][0]) == []
    stats = server._session_stats()
    assert stats["thinking_stripped"] == 1
    assert stats["backends"][0]["thinking_stripped"] == 1


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_rejections_beyond_the_strip_cap_are_surfaced_not_looped(stream):
    """R6 — three strips at most: two targeted, the third everything; a fourth rejection reaches the client.

    Each rejection names a later turn, as a history broken in several places would,
    so every strip has something to remove.  The fourth signed turn is what makes
    the count matter: only the third strip's escalation removes it, so a count that
    never advanced would keep targeting and send a fifth request.

    Args:
        stream: Whether the client request streams.
    """
    rejections = [(400, _rejection_at(index)) for index in (1, 3, 5, 5)]

    status, text, calls = await _drive(
        _native_server(), _NATIVE_URL, rejections, stream=stream, history=_FOUR_TURN_HISTORY
    )

    assert status == 400, text
    assert len(calls) == 4
    assert _thinking_types(calls[1][0]) == ["thinking", "thinking", "thinking"]
    assert _thinking_types(calls[2][0]) == ["thinking", "thinking"]
    assert _thinking_types(calls[3][0]) == []


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_a_rejection_that_outlives_recovery_does_not_quarantine_a_pool_member(stream):
    """R4b — once the strips run out, the error reaches the client and every backend stays healthy.

    The history is the bridge's fault on every backend alike, so cooling one down
    would only take a healthy member out of the pool (M17, PR #113 review).

    Args:
        stream: Whether the client request streams.
    """
    server = _native_server(pool=True)
    rejections = [(400, _rejection_at(index)) for index in (1, 3, 5, 5)]

    status, text, _calls = await _drive(server, _NATIVE_URL, rejections, stream=stream, history=_FOUR_TURN_HISTORY)

    assert status == 400, text
    assert all(health["healthy"] and not health.get("failure_count") for health in server._backend_health)


@pytest.mark.asyncio
async def test_the_final_retry_ladder_does_not_quarantine_a_member_either(monkeypatch):
    """R4b — the non-streaming pool's last retries apply the same rule as its failover loop.

    The pool's first attempt exhausts its strips; its second attempt rate-limits,
    so the loop ends without the two-400 stop.  Selection is random between
    healthy members, so the rate-limited one may be either; the ladder can only
    reach the other, which rejects again and must stay healthy.

    Args:
        monkeypatch: Removes the ladder's 20 s and 40 s sleeps.
    """
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])
    server = _native_server(pool=True)
    rate_limited = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    exhausted = [(400, _rejection_at(index)) for index in (1, 3, 5, 5)]
    replies = [*exhausted, (429, rate_limited), *exhausted, *exhausted]

    status, text, calls = await _drive(server, _NATIVE_URL, replies, history=_FOUR_TURN_HISTORY)

    assert status == 400, text
    assert len(calls) == 13
    ladder_key = calls[-1][1]["x-api-key"]
    assert ladder_key != calls[4][1]["x-api-key"]
    ladder_member = int(ladder_key.rsplit("-", 1)[1]) - 1
    assert server._backend_health[ladder_member]["healthy"]
    assert not server._backend_health[ladder_member].get("failure_count")


@pytest.mark.asyncio
async def test_a_rejection_after_tighter_compaction_does_not_quarantine_either(monkeypatch):
    """R4b — the verdict follows the compaction retry's error, not the context-too-large that started it.

    Tighter compaction edits the signed history, so its retry is where an
    unrecoverable signature rejection is most likely to appear.

    Args:
        monkeypatch: Stubs the upstream call and compaction, and records cooling.
    """
    server = _native_server(pool=True)
    calls: list[int] = []
    cooled: list[int] = []

    async def upstream(cc_request, *args, **kwargs):
        """Reject as too large on each backend's first call, then as a bad signature.

        Args:
            cc_request: The request; unused.
            *args: Unused.
            **kwargs: Unused.

        Raises:
            UpstreamError: Always.
        """
        calls.append(server._current_backend_idx)
        if len(calls) % 2:
            raise UpstreamError(400, _envelope("prompt is too long: context length exceeded"))
        raise UpstreamError(400, _envelope(_INVALID_SIGNATURE))

    monkeypatch.setattr(server, "_make_upstream_request", upstream)
    monkeypatch.setattr(server, "_is_oversized_request", lambda _request: True)
    monkeypatch.setattr(server, "_compact_with_tighter_budget", lambda cc_request, factor=0.5: None)
    monkeypatch.setattr(server, "_mark_backend_unhealthy", lambda idx, **kwargs: cooled.append(idx))

    status, text, _calls = await _drive(server, _NATIVE_URL, [], history=_FOUR_TURN_HISTORY)

    assert status == 400, text
    assert len(calls) == 4
    assert cooled == []


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_the_retry_keeps_the_reasoning_after_the_rejected_turn(stream):
    """R3b — through the bridge, the named turn loses its thinking and the later valid turn keeps it.

    Args:
        stream: Whether the client request streams.
    """
    ok = _sse_reply() if stream else _OK_REPLY
    status, text, calls = await _drive(
        _native_server(),
        _NATIVE_URL,
        [(400, _envelope(_INVALID_SIGNATURE)), (200, ok)],
        stream=stream,
        history=_LONG_HISTORY,
    )

    assert status == 200, text
    assert len(calls) == 2
    assert _thinking_types(calls[0][0]) == ["thinking", "redacted_thinking", "thinking"]
    assert _thinking_types(calls[1][0]) == ["thinking"]


@pytest.mark.asyncio
async def test_a_failover_after_a_strip_lets_the_next_backend_recover_too():
    """R4b — strips are counted per serialized body: the next backend's rebuilt body gets its own recovery.

    Member A uses all three strips, then rate-limits; the pool fails over and
    rebuilds the body with its thinking restored.  Member B rejects it too.  Had
    the count been per request, B would have had no strips left and been blamed
    and quarantined for kitty's history.
    """
    server = _native_server(pool=True)
    rate_limited = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    replies = [
        (400, _rejection_at(1)),
        (400, _rejection_at(3)),
        (400, _rejection_at(5)),
        (429, rate_limited),
        (400, _rejection_at(1)),
        (200, _sse_reply()),
    ]

    status, text, calls = await _drive(server, _NATIVE_URL, replies, stream=True, history=_FOUR_TURN_HISTORY)

    assert status == 200, text
    assert len(calls) == 6
    first, served_by = calls[0][1].get("x-api-key"), calls[-1][1].get("x-api-key")
    assert calls[4][1].get("x-api-key") == served_by != first
    assert _thinking_types(calls[4][0]) == ["thinking", "redacted_thinking", "thinking", "thinking", "thinking"]
    member = int(served_by.rsplit("-", 1)[1]) - 1
    assert server._backend_health[member]["healthy"]


@pytest.mark.asyncio
async def test_strips_do_not_spend_the_streams_retry_budget(monkeypatch):
    """R6b — three strips followed by an ordinary provider hiccup still end in the answer.

    A strip re-sends kitty's own repaired history, like a transport-grace retry,
    so it must give its attempt back.  Otherwise three strips leave a single
    backend one normal attempt, the empty-response final delays fire for a 503,
    and a turn the unstripped path would have answered fails.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])
    unavailable = {"type": "error", "error": {"type": "overloaded_error", "message": "try again"}}
    replies = [
        (400, _rejection_at(1)),
        (400, _rejection_at(3)),
        (400, _rejection_at(5)),
        (503, unavailable),
        (503, unavailable),
        (503, unavailable),
        (200, _sse_reply()),
    ]

    status, text, calls = await _drive(_native_server(), _NATIVE_URL, replies, stream=True, history=_FOUR_TURN_HISTORY)

    assert status == 200, text
    assert len(calls) == 7


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
