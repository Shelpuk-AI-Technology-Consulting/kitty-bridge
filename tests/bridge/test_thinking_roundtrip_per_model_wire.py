"""Subsystem tests for the thinking round-trip repair on a per-model-routed provider.

Covers KBR-7.  ``OpenCodeGoAdapter`` routes by model — an Anthropic Messages
body for ``_MESSAGES_MODELS`` and a Chat Completions body for everything else —
while it inherited a single ``upstream_wire_is_messages_api == True`` from
:class:`~kitty.providers.anthropic.AnthropicAdapter`.  The bridge's thinking
round-trip repair branches on that declaration to choose a carrier, so on a
Chat-Completions-routed model it wrote an Anthropic ``thinking`` content block
into a Chat Completions body.

The repair reaches the wire through two independent call sites, and both are
tested here.  A fix applied to only one of them is worse than no fix: once the
adapter's bare property reports Chat Completions, the site left behind would
write ``reasoning_content`` into an Anthropic transcript instead — the same
defect inverted, on the error-recovery path.

Covers ``.requirements/20260907T142619Z_wire_shape_honesty`` R4a, R4b and R5.
"""

from __future__ import annotations

import json

import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.providers.opencode import OpenCodeGoAdapter
from tests.bridge.test_thinking_roundtrip_failover import (
    _anthropic_sse,
    _post,
    _rejection,
    _ScriptedUpstream,
    _StubLauncher,
)

# The adapter ignores ``provider_config`` for its base URL, so these are the
# real routed endpoints.  ``aioresponses`` intercepts both; nothing leaves the
# process.
CC_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"

# Served on the Chat Completions endpoint and the Messages endpoint respectively
# (https://opencode.ai/docs/go/, verified 2026-09-07).
CC_MODEL = "glm-5.2"
MESSAGES_MODEL = "minimax-m2.5"


def _chat_completions_sse() -> bytes:
    """A minimal, well-formed Chat Completions SSE stream.

    The Anthropic stream in the sibling module cannot stand in for this one:
    the Chat-Completions-routed model returns CC chunks, and the bridge
    translates them back to Messages events for the client.
    """
    chunks = [
        {
            "id": "chatcmpl-ok",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": CC_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Recovered"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-ok",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": CC_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    body = b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
    return body + b"data: [DONE]\n\n"


def _stream_ok(sse: bytes) -> CallbackResult:
    return CallbackResult(status=200, headers={"Content-Type": "text/event-stream"}, body=sse)


def _transcript_without_reasoning() -> dict:
    """A client request whose assistant turn carries no reasoning of any kind.

    The sibling module's fixture cannot be reused here.  It asks for thinking,
    and this provider is not a native-passthrough one, so the Messages route
    translates Anthropic → Chat Completions → Anthropic and
    :meth:`AnthropicAdapter.translate_to_upstream` injects the empty thinking
    block on the way back.  The transcript would then already satisfy the
    round-trip contract, ``_repair_thinking_roundtrip`` would report no change,
    and the retry branch under test would never run — the test would fail on a
    plain 400 rather than on the carrier.

    Omitting ``thinking`` reproduces the incident this repair exists for: a
    transcript authored by a different provider, replayed against a backend
    that enforces the round-trip.
    """
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "stream": True,
        "system": [{"type": "text", "text": "You are a reviewer."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Review this diff"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Reading the file."}]},
            {"role": "user", "content": [{"type": "text", "text": "Carry on"}]},
        ],
    }


def _make_server(model: str) -> BridgeServer:
    """Build a single-backend OpenCode Go bridge pinned to *model*.

    Single-backend mode keys the repair set by index -1, which is the shape a
    user running one profile actually gets.
    """
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=OpenCodeGoAdapter(),
        resolved_key="key-opencode",
        model=model,
        provider_config={},
        host="127.0.0.1",
        port=0,
    )


def _assistants(body: dict) -> list[dict]:
    """Return the assistant turns of an upstream body."""
    return [m for m in body["messages"] if m.get("role") == "assistant"]


class TestProactiveRepairBranchesPerModel:
    """R4a — ``_upstream_body_for`` picks the carrier the emitted body takes."""

    def test_uses_the_chat_completions_carrier_for_a_cc_routed_model(self):
        """The carrier is a field, and the turn's string content survives.

        Before KBR-7 this produced ``[{"type": "thinking", ...}, {"type":
        "text", ...}]`` — Anthropic content blocks inside a Chat Completions
        body, which the endpoint cannot read.
        """
        server = _make_server(CC_MODEL)
        server._thinking_repair_backends.add(server._current_backend_idx)

        body = server._upstream_body_for(
            {
                "model": CC_MODEL,
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": "Review this diff"},
                    {"role": "assistant", "content": "Reading the file."},
                ],
            }
        )

        assistants = _assistants(body)
        assert assistants, body
        for msg in assistants:
            assert msg["reasoning_content"] == ""
            assert msg["content"] == "Reading the file.", "the Chat Completions carrier must not rewrite content"

    def test_uses_the_thinking_block_for_a_messages_routed_model(self):
        """The complement — the Messages route still gets the Anthropic carrier.

        Without this, a fix that simply hard-coded Chat Completions for the
        whole adapter would pass the test above.
        """
        server = _make_server(MESSAGES_MODEL)
        server._thinking_repair_backends.add(server._current_backend_idx)

        body = server._upstream_body_for(
            {
                "model": MESSAGES_MODEL,
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": "Review this diff"},
                    {"role": "assistant", "content": "Reading the file."},
                ],
            }
        )

        assistants = _assistants(body)
        assert assistants, body
        for msg in assistants:
            assert "reasoning_content" not in msg
            assert msg["content"][0] == {"type": "thinking", "thinking": ""}


class TestReactiveRepairBranchesPerModel:
    """R4b — the streaming retry branch picks the same carrier.

    The repair set starts empty, so the first attempt goes out unrepaired and
    only the retry branch can have written the carrier.  That isolates this
    call site from the proactive one above.
    """

    @pytest.mark.asyncio
    async def test_uses_the_chat_completions_carrier_for_a_cc_routed_model(self):
        server = _make_server(CC_MODEL)
        upstream = _ScriptedUpstream(_rejection(), _stream_ok(_chat_completions_sse()))

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(CC_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                status, body = await _post(server, _transcript_without_reasoning())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert len(upstream.bodies) == 2, "expected one rejected attempt and one repaired retry"
        retried = _assistants(upstream.bodies[1])
        assert retried, upstream.bodies[1]
        for msg in retried:
            assert msg.get("reasoning_content") == ""
            assert not isinstance(msg["content"], list), "an Anthropic carrier must not reach a CC body"

    @pytest.mark.asyncio
    async def test_uses_the_thinking_block_for_a_messages_routed_model(self):
        """The complement, and the case a one-site fix would break.

        If only ``_upstream_body_for`` were changed, this branch would read the
        adapter's newly-honest ``False`` property and write the Chat
        Completions carrier into an Anthropic body.
        """
        server = _make_server(MESSAGES_MODEL)
        upstream = _ScriptedUpstream(_rejection(), _stream_ok(_anthropic_sse()))

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(MESSAGES_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                status, body = await _post(server, _transcript_without_reasoning())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert len(upstream.bodies) == 2, "expected one rejected attempt and one repaired retry"
        retried = _assistants(upstream.bodies[1])
        assert retried, upstream.bodies[1]
        for msg in retried:
            assert "reasoning_content" not in msg, "a CC carrier must not reach an Anthropic body"
            assert msg["content"][0] == {"type": "thinking", "thinking": ""}
