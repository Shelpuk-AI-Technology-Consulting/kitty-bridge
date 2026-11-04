"""An upstream that rejects ``redacted_thinking`` gets it stripped and the request re-sent.

Some Anthropic-compatible endpoints implement ``thinking`` but not
``redacted_thinking``, and their parser refuses the whole request. Claude Code
re-sends its history every turn, so without recovery the session stays blocked.
See ``SYSTEM_DESIGN.md`` section 18.

Unit tests pin the detector and the strip; bridge tests drive a real in-process
:class:`~kitty.bridge.server.BridgeServer` against a strict fake upstream.
"""

from __future__ import annotations

import copy
import json
import logging
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import (
    BridgeServer,
    _is_redacted_thinking_unsupported_error,
    _strip_redacted_thinking_blocks,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.types import BridgeProtocol

_BASE = "https://api.strict.test/anthropic"
_URL = f"{_BASE}/v1/messages"

#: The reported rejection, verbatim apart from its index and position.
_REJECTION = {
    "error": {
        "message": (
            "Failed to deserialize the JSON body into the target type: messages[1].content: unknown variant "
            "`redacted_thinking`, expected one of `text`, `image`, `document`, `thinking`, `server_tool_use`, "
            "`tool_use`, `tool_result`, `web_search_tool_result`, `tool_addition` at line 1 column 1878501"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_request_error",
    }
}

#: A different unknown variant, on an upstream whose accepted list names the block.
_OTHER_VARIANT = {
    "error": {"message": "unknown variant `tool_addition`, expected one of `text`, `thinking`, `redacted_thinking`"}
}

_ROUNDTRIP_REJECTION = {
    "error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API."}
}

_TOOL_HISTORY = [
    {"role": "user", "content": "Check the weather."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Call the tool.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
]

_TEXT_HISTORY = [
    {"role": "user", "content": "Hello."},
    {
        "role": "assistant",
        "content": [{"type": "redacted_thinking", "data": "opaque"}, {"type": "text", "text": "Hi."}],
    },
    {"role": "user", "content": "Again."},
]

_REPLY = {
    "id": "msg_ok",
    "type": "message",
    "role": "assistant",
    "model": "strict-model",
    "content": [{"type": "text", "text": "Answered."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _sse_reply() -> str:
    """Render ``_REPLY`` as an Anthropic SSE stream.

    Returns:
        The SSE body.
    """
    events = [
        {"type": "message_start", "message": {**_REPLY, "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Answered."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _block_types(body: dict) -> list[str]:
    """Return every assistant content block type in an upstream body, in order.

    Args:
        body: A captured upstream JSON body.

    Returns:
        The block types.
    """
    return [
        block["type"]
        for msg in body["messages"]
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
        for block in msg["content"]
    ]


def _without_redacted(body: dict) -> dict:
    """Return a copy of an upstream body with its ``redacted_thinking`` blocks removed.

    Args:
        body: A captured upstream JSON body.

    Returns:
        The expected retry body.
    """
    expected = copy.deepcopy(body)
    for msg in expected["messages"]:
        if isinstance(msg.get("content"), list):
            msg["content"] = [b for b in msg["content"] if b.get("type") != "redacted_thinking"]
    return expected


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


def _server(*, pool: bool = False) -> BridgeServer:
    """Build a native ``custom_anthropic`` bridge, optionally as a two-member pool.

    Args:
        pool: Whether to build a two-member balancing pool on the same endpoint.

    Returns:
        An unstarted bridge.
    """
    config = {"base_url": _BASE}
    if not pool:
        return BridgeServer(
            adapter=_StubLauncher(),
            provider=CustomAnthropicAdapter(),
            resolved_key="key-solo",
            model="strict-model",
            provider_config=config,
            host="127.0.0.1",
            port=0,
        )
    backends = []
    for name in ("member-1", "member-2"):
        profile = Profile(
            name=name,
            provider="custom_anthropic",
            model="strict-model",
            auth_ref=str(uuid.uuid4()),
            provider_config=config,
        )
        backends.append((CustomAnthropicAdapter(), f"key-{name}", profile))
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="strict-model",
        provider_config=config,
        backends=backends,
        host="127.0.0.1",
        port=0,
    )


async def _drive(
    server: BridgeServer,
    respond_with,
    *,
    stream: bool,
    history: list[dict],
    turns: int = 1,
) -> tuple[int, list[tuple[dict, dict]]]:
    """Send client requests through a bridge and capture what the upstream received.

    Args:
        server: An unstarted bridge.
        respond_with: Maps a captured upstream body and its headers to ``(status, body)``.
        stream: Whether the client request streams.
        history: The agent's transcript.
        turns: How many identical client requests to send.

    Returns:
        ``(last_client_status, upstream_calls)`` where each call is ``(json_body, headers)``.
    """
    calls: list[tuple[dict, dict]] = []

    def respond(_url, **kwargs):
        """Record the call and answer it.

        Args:
            _url: The matched URL.
            **kwargs: The request's keyword arguments.

        Returns:
            The scripted response.
        """
        body = copy.deepcopy(kwargs.get("json"))
        headers = dict(kwargs.get("headers") or {})
        calls.append((body, headers))
        status, reply = respond_with(body, headers)
        if isinstance(reply, str):
            return CallbackResult(status=status, headers={"Content-Type": "text/event-stream"}, body=reply)
        return CallbackResult(status=status, content_type="application/json", body=json.dumps(reply))

    request = {"model": "strict-model", "max_tokens": 64, "stream": stream, "messages": copy.deepcopy(history)}
    status = 0
    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        mocked.post(_URL, callback=respond, repeat=True)
        await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(turns):
                    async with session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=request) as resp:
                        status = resp.status
                        await resp.read()
        finally:
            await server.stop_async()
    return status, calls


def _strict_upstream(*, stream: bool):
    """Build a responder that rejects any body carrying ``redacted_thinking``.

    Args:
        stream: Whether to answer an accepted body as an SSE stream.

    Returns:
        A responder for :func:`_drive`.
    """

    def answer(body: dict, _headers: dict) -> tuple[int, object]:
        """Answer one upstream call.

        Args:
            body: The captured upstream body.
            _headers: The captured upstream headers.

        Returns:
            ``(status, reply)``.
        """
        if "redacted_thinking" in _block_types(body):
            return 422, _REJECTION
        return 200, _sse_reply() if stream else _REPLY

    return answer


# ── R1: the detector ────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize("as_text", [False, True], ids=["dict", "text"])
def test_the_reported_rejection_is_recognised(status, as_text):
    """R1 — the wording is recognised at both observed statuses, parsed or raw.

    Args:
        status: The upstream status.
        as_text: Whether the body arrives as JSON text (the streaming path) rather than a dict.
    """
    assert _is_redacted_thinking_unsupported_error(status, json.dumps(_REJECTION) if as_text else _REJECTION)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, _REJECTION),
        (200, _REJECTION),
        (422, _OTHER_VARIANT),
        (400, {"error": {"message": "messages.1.content.0: Invalid `signature` in `thinking` block"}}),
        (400, {"error": {"message": "redacted_thinking blocks are not allowed in this position"}}),
        (400, None),
    ],
    ids=["5xx", "2xx", "only-in-expected-list", "signature", "no-unknown-variant", "no-body"],
)
def test_other_errors_are_not_recognised(status, body):
    """R1 — a block merely named as accepted, or a backend fault, is not this rejection.

    Args:
        status: The upstream status.
        body: The upstream error body.
    """
    assert not _is_redacted_thinking_unsupported_error(status, body)


# ── R2: the strip ───────────────────────────────────────────────────────────


def test_strip_removes_only_redacted_thinking_and_keeps_the_order():
    """R2 — thinking and tool calls stay, in order; other turns are untouched."""
    body = {"model": "m", "messages": copy.deepcopy(_TOOL_HISTORY)}

    assert _strip_redacted_thinking_blocks(body) is True
    assert _block_types(body) == ["thinking", "tool_use"]
    assert body["messages"][0] == _TOOL_HISTORY[0]
    assert body["messages"][2] == _TOOL_HISTORY[2]


def test_strip_leaves_an_empty_list_so_indices_do_not_shift():
    """R2 — a turn holding only the block keeps its place with ``content: []``."""
    history = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]},
        {"role": "user", "content": "Again."},
    ]
    body = {"model": "m", "messages": history}

    assert _strip_redacted_thinking_blocks(body) is True
    assert body["messages"][1] == {"role": "assistant", "content": []}
    assert len(body["messages"]) == 3


def test_strip_never_mutates_the_messages_it_was_given():
    """R2 — the request a later attempt re-serializes must keep its blocks."""
    original = copy.deepcopy(_TOOL_HISTORY)
    messages = copy.deepcopy(_TOOL_HISTORY)

    _strip_redacted_thinking_blocks({"model": "m", "messages": messages})

    assert messages == original


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}]}]},
        {"messages": [{"role": "assistant", "content": "Plain text."}]},
        {"messages": "not-a-list"},
        {},
    ],
    ids=["thinking-only", "string-content", "malformed-messages", "no-messages"],
)
def test_strip_reports_no_change_when_there_is_nothing_to_remove(body):
    """R2 — ``False`` is what stops the retry, so it must be exact.

    Args:
        body: An upstream body with no ``redacted_thinking`` block.
    """
    assert _strip_redacted_thinking_blocks(body) is False


# ── R3: recovery through the bridge ─────────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.parametrize("history", [_TOOL_HISTORY, _TEXT_HISTORY], ids=["tool-use", "text-only"])
@pytest.mark.asyncio
async def test_a_rejection_is_recovered_on_the_same_backend(stream, history):
    """R3 — the client gets the answer; the retry is the first body minus the block.

    Comparing whole bodies is what proves M9 did not run: its Chat Completions
    rebuild would drop fields a native body keeps, and resend the block.

    Args:
        stream: Whether the client request streams.
        history: The agent's transcript.
    """
    status, calls = await _drive(_server(), _strict_upstream(stream=stream), stream=stream, history=history)

    assert status == 200
    assert len(calls) == 2
    assert "redacted_thinking" in _block_types(calls[0][0])
    assert calls[1][0] == _without_redacted(calls[0][0])


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_a_pool_retries_the_same_member_and_keeps_it_healthy(stream):
    """R3 — this is the history's shape, not a sick backend: no failover, no quarantine.

    Args:
        stream: Whether the client request streams.
    """
    server = _server(pool=True)

    status, calls = await _drive(server, _strict_upstream(stream=stream), stream=stream, history=_TOOL_HISTORY)

    assert status == 200
    assert len(calls) == 2
    assert calls[0][1].get("x-api-key") == calls[1][1].get("x-api-key")
    assert all(health["healthy"] and not health.get("failure_count") for health in server._backend_health)


@pytest.mark.asyncio
async def test_the_retry_does_not_spend_an_attempt(monkeypatch):
    """R3 — after the strip, the stream still has its whole budget for real failures.

    The last attempt succeeds only if the strip gave its attempt back. If the strip
    spent one, the final 503 below would reach the client instead of the 200 behind it.

    Args:
        monkeypatch: Removes the retry delays so the test does not sleep.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", tuple(0 for _ in server_module._EMPTY_FINAL_DELAYS))
    budget = server_module._MAX_RETRIES + 1 + len(server_module._EMPTY_FINAL_DELAYS)
    outages_left = [budget - 1]

    def answer(body: dict, _headers: dict) -> tuple[int, object]:
        """Reject the block, then fail with retryable statuses, then accept.

        Args:
            body: The captured upstream body.
            _headers: The captured upstream headers.

        Returns:
            ``(status, reply)``.
        """
        if "redacted_thinking" in _block_types(body):
            return 422, _REJECTION
        if outages_left[0]:
            outages_left[0] -= 1
            return 503, {"error": {"message": "overloaded"}}
        return 200, _sse_reply()

    status, calls = await _drive(_server(), answer, stream=True, history=_TOOL_HISTORY)

    assert status == 200
    assert len(calls) == 1 + budget


# ── R4: the backend is remembered ───────────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_a_later_turn_does_not_pay_another_rejection(stream):
    """R4 — the history returns every turn, so the second turn goes out already stripped.

    Args:
        stream: Whether the client request streams.
    """
    status, calls = await _drive(
        _server(), _strict_upstream(stream=stream), stream=stream, history=_TOOL_HISTORY, turns=2
    )

    assert status == 200
    assert len(calls) == 3
    assert _block_types(calls[2][0]) == ["thinking", "tool_use"]


@pytest.mark.asyncio
async def test_the_memory_belongs_to_the_backend_that_rejected():
    """R4 — a sibling that never rejected keeps receiving the block."""
    strict_key = "key-member-1"
    seen_by_sibling: list[list[str]] = []

    def answer(body: dict, headers: dict) -> tuple[int, object]:
        """Reject on one pool member only.

        Args:
            body: The captured upstream body.
            headers: The captured upstream headers.

        Returns:
            ``(status, reply)``.
        """
        if headers.get("x-api-key") != strict_key:
            seen_by_sibling.append(_block_types(body))
            return 200, _REPLY
        if "redacted_thinking" in _block_types(body):
            return 422, _REJECTION
        return 200, _REPLY

    status, _calls = await _drive(_server(pool=True), answer, stream=False, history=_TOOL_HISTORY, turns=8)

    assert status == 200
    assert seen_by_sibling, "the pool never selected the second member"
    assert all("redacted_thinking" in types for types in seen_by_sibling)


# ── R5: the #32 repair stays available ──────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stripped_body_can_still_receive_the_thinking_carrier():
    """R5 — an upstream may also demand thinking back, so the strip must not bar the repair."""

    def answer(body: dict, _headers: dict) -> tuple[int, object]:
        """Reject the block, then reject a turn with no thinking, then accept.

        Args:
            body: The captured upstream body.
            _headers: The captured upstream headers.

        Returns:
            ``(status, reply)``.
        """
        types = _block_types(body)
        if "redacted_thinking" in types:
            return 422, _REJECTION
        if "thinking" not in types:
            return 400, _ROUNDTRIP_REJECTION
        return 200, _sse_reply()

    status, calls = await _drive(_server(), answer, stream=True, history=_TEXT_HISTORY)

    assert status == 200
    assert len(calls) == 3
    assert _block_types(calls[2][0]) == ["thinking", "text"]


# ── R6: nothing changes for anyone else ─────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_an_upstream_that_accepts_the_block_receives_it_unchanged(stream):
    """R6 — Anthropic requires these blocks back, so nothing may strip them unprompted.

    Args:
        stream: Whether the client request streams.
    """

    def answer(_body: dict, _headers: dict) -> tuple[int, object]:
        """Accept every body.

        Args:
            _body: The captured upstream body.
            _headers: The captured upstream headers.

        Returns:
            ``(status, reply)``.
        """
        return 200, _sse_reply() if stream else _REPLY

    status, calls = await _drive(_server(), answer, stream=stream, history=_TOOL_HISTORY, turns=2)

    assert status == 200
    assert len(calls) == 2
    assert all(_block_types(body) == ["thinking", "redacted_thinking", "tool_use"] for body, _ in calls)


@pytest.mark.asyncio
async def test_a_different_unknown_variant_strips_nothing():
    """R6 — a rejection that names the block only as accepted leaves it in place."""
    status, calls = await _drive(_server(), lambda _b, _h: (422, _OTHER_VARIANT), stream=False, history=_TEXT_HISTORY)

    assert status == 422
    assert all("redacted_thinking" in _block_types(body) for body, _ in calls)


# ── R7: the recovery is visible ─────────────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_a_recovery_logs_a_warning_naming_the_block(stream, caplog):
    """R7 — the log line is what separates this strip from a signature strip.

    Args:
        stream: Whether the client request streams.
        caplog: Captures the bridge's log records.
    """
    with caplog.at_level(logging.WARNING, logger=server_module.logger.name):
        await _drive(_server(), _strict_upstream(stream=stream), stream=stream, history=_TOOL_HISTORY)

    assert any(
        record.levelno == logging.WARNING and "redacted_thinking" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.asyncio
async def test_a_recovery_is_counted_for_the_operator(stream):
    """R7 — one strip lands in ``/stats``, and the remembered turn adds no second count.

    Args:
        stream: Whether the client request streams.
    """
    server = _server()

    await _drive(server, _strict_upstream(stream=stream), stream=stream, history=_TOOL_HISTORY, turns=2)

    stats = server._session_stats()
    assert stats["thinking_stripped"] == 1
    assert stats["backends"][0]["thinking_stripped"] == 1
