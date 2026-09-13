"""Streaming ``/v1/messages`` through an Anthropic-wire translated adapter reaches the client (KBR-227).

Claude Code always streams.  When the profile's provider speaks Anthropic
Messages upstream but is not a native-passthrough adapter — ``anthropic``,
``minimax_token`` in its default mode, ``opencode_go`` for its Messages-routed
models — the bridge used to parse the upstream's Anthropic SSE as Chat
Completions chunks, find no ``choices`` in any of them, and send Claude Code a
``200`` with an empty body.

The fix forwards the stream unchanged whenever the upstream wire is Messages for
the request's routed model, exactly as the native branch already does: the client
and the upstream speak the same protocol, and forwarding keeps what a conversion
through Chat Completions would lose — thinking signatures above all (KBR-228).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
against an ``aioresponses`` upstream, because the defect lives in the handler's
choice of branch, not in any adapter hook.
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.opencode import OpenCodeGoAdapter
from kitty.types import BridgeProtocol

#: A tool schema the auditor can hold a returned ``tool_use`` against.
_READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

#: The signature a thinking block carries; forwarding must keep it byte-exact.
_SIGNATURE = "EqQBCgIYAhIM1gbcDa9GJwZA2b"


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
            :attr:`BridgeProtocol.MESSAGES_API`, so ``/v1/messages`` is served.
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


def _anthropic_events(tool_input: dict) -> list[dict]:
    """Return an Anthropic Messages stream: thinking with a signature, text, then one tool call.

    Args:
        tool_input: The ``Read`` tool's input, streamed as one ``input_json_delta``.

    Returns:
        The stream's event payloads, in order.
    """
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_upstream",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "look first"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": _SIGNATURE}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]


def _render_sse(events: list[dict]) -> str:
    """Render Anthropic event payloads as an SSE body.

    Args:
        events: Event payloads, each carrying its ``type``.

    Returns:
        The ``event:``/``data:`` SSE text.
    """
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _parse_data_lines(sse_text: str) -> list[dict]:
    """Parse every ``data:`` line of an SSE body as JSON.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The decoded payloads, in order.
    """
    return [json.loads(line[5:].strip()) for line in sse_text.splitlines() if line.startswith("data:")]


def _client_request(model: str) -> dict:
    """Return a streaming Claude Code request that declares the ``Read`` tool.

    Args:
        model: The model the agent asks for.

    Returns:
        A Messages API request body.
    """
    return {
        "model": model,
        "max_tokens": 4096,
        "stream": True,
        "messages": [{"role": "user", "content": "read a"}],
        "tools": [{"name": "Read", "description": "Read a file", "input_schema": _READ_SCHEMA}],
    }


async def _stream(provider: ProviderAdapter, model: str, upstream_sse: str) -> tuple[BridgeServer, int, str]:
    """POST a streaming request through a real bridge and return what the client received.

    Args:
        provider: The adapter the bridge serves with.
        model: The model the agent asks for; it also selects ``opencode_go``'s route.
        upstream_sse: The SSE body the mocked upstream returns.

    Returns:
        The server (for its counters), the HTTP status and the client's body text.
    """
    server = BridgeServer(_StubLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
    upstream_url = server._build_upstream_url({"model": model})
    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        mocked.post(upstream_url, status=200, body=upstream_sse, content_type="text/event-stream")
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=_client_request(model)) as resp,
            ):
                return server, resp.status, await resp.text()
        finally:
            await server.stop_async()


_MESSAGES_WIRE_TRANSLATED = [
    pytest.param(AnthropicAdapter, "claude-opus-4-6", id="anthropic"),
    pytest.param(MiniMaxTokenAnthropicAdapter, "MiniMax-M3", id="minimax_token-default"),
    pytest.param(OpenCodeGoAdapter, "minimax-m2.7", id="opencode_go-messages-model"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _MESSAGES_WIRE_TRANSLATED)
async def test_the_upstream_stream_reaches_the_client_unchanged(provider_factory, model):
    """R1/R2 — every upstream event arrives, in order, as the upstream sent it.

    Equality of the whole event list is the claim: text, the tool call's
    streamed arguments, the thinking signature and the usage all ride on it.

    Args:
        provider_factory: Builds an Anthropic-wire adapter that is not native passthrough.
        model: A model that adapter serves on its Messages route.
    """
    provider = provider_factory()
    assert not provider.use_native_messages, "precondition: this test is about the translated route"
    events = _anthropic_events({"path": "a"})

    _, status, body = await _stream(provider, model, _render_sse(events))

    assert status == 200
    assert _parse_data_lines(body) == events


@pytest.mark.asyncio
async def test_a_chat_completions_routed_model_is_still_translated():
    """R3 — the choice is per model: ``opencode_go``'s Chat Completions route keeps translation."""
    chunks = [
        {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hello"}}]},
        {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    upstream_sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    _, status, body = await _stream(OpenCodeGoAdapter(), "glm-5.1", upstream_sse)

    events = _parse_data_lines(body)
    assert status == 200
    assert events[0]["type"] == "message_start"
    assert {"type": "text_delta", "text": "hello"} in [e.get("delta") for e in events]


@pytest.mark.asyncio
async def test_the_tool_use_auditor_reads_the_forwarded_stream():
    """R4 — a malformed ``tool_use`` on this route is still counted, so auditing did not go blind.

    The input wraps the declared fields in an envelope, which the auditor flags
    against the client's own schema (issue #33).  A clean input would prove
    nothing: an auditor that never saw the bytes also reports no anomaly.
    """
    events = _anthropic_events({"result": {"path": "a"}})

    server, status, _ = await _stream(AnthropicAdapter(), "claude-opus-4-6", _render_sse(events))

    assert status == 200
    assert sum(server._stats_malformed_tool_use.values()) == 1
