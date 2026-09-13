"""The server's translated Messages stream gives parallel tool calls their own blocks.

KBR-226: on a translated ``/v1/messages`` route (Chat Completions upstream), every
``tool_use`` was opened at Anthropic block index 0, so a stream with two parallel
tool calls was not a sentence in the SSE grammar (§6.2.2) even though each event
was valid on its own. This proves the fix on the path Claude Code actually uses:
an in-process server, a scripted Chat Completions upstream, and the client-visible
byte stream walked end to end.

Decided stream shape (REQUIREMENTS.md, KBR-226): blocks opened by parallel tool
calls may overlap in time and need not close in order — clients key blocks by
index — but each index opens once, closes once, closes only after it opened, and
carries no delta outside its own start..stop window.
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.types import BridgeProtocol

CC_URL = "https://api.cc.test/v1/chat/completions"


class _StubLauncher(LauncherAdapter):
    """Minimal launcher that selects the Messages API bridge protocol."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


def _cc_chunk(delta: dict, finish: str | None = None) -> dict:
    """Build one Chat Completions streaming chunk.

    Args:
        delta: The ``choices[0].delta`` object of the chunk.
        finish: The chunk's ``finish_reason``, if it is the final chunk.

    Returns:
        A Chat Completions chunk dict.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "MiniMax-M3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _two_tool_calls_then_text_sse() -> bytes:
    """A Chat Completions stream that calls two tools and then writes text.

    The trailing text is what exercises the decided overlap: the tool blocks are
    still open when the text block opens, so the client sees three open blocks.
    """
    chunks = [
        _cc_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"path": "a"}'},
                    }
                ]
            }
        ),
        _cc_chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "Grep", "arguments": '{"pattern": "x"}'},
                    }
                ]
            }
        ),
        _cc_chunk({"content": "Both calls are in."}),
        _cc_chunk({}, finish="tool_calls"),
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _parse_sse(body: bytes) -> list[tuple[str, dict]]:
    """Parse an Anthropic SSE byte stream into ``(event_name, data)`` pairs.

    Args:
        body: The raw response body the bridge wrote to the client.

    Returns:
        One pair per event, in wire order. Blocks with no ``data:`` line
        (there are none on this path) would be skipped.
    """
    parsed: list[tuple[str, dict]] = []
    for block in body.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                parsed.append((data["type"], data))
    return parsed


def _make_server() -> BridgeServer:
    """Build a single-backend Messages server on the translated (CC) wire."""
    profile = Profile(
        name="pool-member-1",
        provider="custom_openai",
        model="MiniMax-M3",
        auth_ref=str(uuid.uuid4()),
        provider_config={"base_url": "https://api.cc.test/v1"},
    )
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=CustomOpenAIAdapter(),
        resolved_key="key-1",
        model="MiniMax-M3",
        provider_config=profile.provider_config,
        host="127.0.0.1",
        port=0,
    )


def _client_request() -> dict:
    """A Claude Code request declaring the two tools the upstream calls."""
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "review"}]}],
        "tools": [
            {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}},
            {"name": "Grep", "description": "Grep a tree", "input_schema": {"type": "object"}},
        ],
    }


@pytest.mark.asyncio
async def test_parallel_tool_use_stream_opens_each_block_at_its_own_index():
    """The client-visible stream opens each tool_use once, at its own index.

    The whole SSE stream is walked against the decided grammar: distinct
    increasing block indices, exactly one start and one stop per index, stops
    only after their own start, and no delta outside its own block's window.
    Overlap is permitted and stop order is free, which the trailing-text chunk
    exercises: its block opens while both tool blocks are still open.
    """
    server = _make_server()
    with aioresponses(passthrough=["http://127.0.0.1"]) as m:
        m.post(
            CC_URL,
            callback=lambda u, **kw: CallbackResult(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                body=_two_tool_calls_then_text_sse(),
            ),
            repeat=True,
        )
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=_client_request()) as resp,
            ):
                assert resp.status == 200
                body = await resp.read()
        finally:
            await server.stop_async()

    events = _parse_sse(body)
    names = [name for name, _ in events]

    # The stream is a complete sentence: it ends with the message_delta /
    # message_stop ending and carries the message_start opening.
    assert "message_start" in names
    assert names[-2:] == ["message_delta", "message_stop"]

    # Each block opens exactly once, at a distinct increasing index, with the
    # tool calls in upstream order followed by the trailing text block.
    starts = [d for name, d in events if name == "content_block_start"]
    assert [(d["index"], d["content_block"]["type"]) for d in starts] == [
        (0, "tool_use"),
        (1, "tool_use"),
        (2, "text"),
    ]
    assert [d["content_block"]["name"] for d in starts[:2]] == ["Read", "Grep"]

    # Per-index window state: overlap permitted, so each index is tracked on
    # its own rather than through one "currently open" cursor.
    opened: set[int] = set()
    closed: set[int] = set()
    for name, d in events:
        if name == "content_block_start":
            assert d["index"] not in opened, f"block {d['index']} opened twice"
            assert d["index"] not in closed, f"block {d['index']} opened after its stop"
            opened.add(d["index"])
        elif name == "content_block_delta":
            assert d["index"] in opened, f"delta for never-opened block {d['index']}"
            assert d["index"] not in closed, f"delta for closed block {d['index']}"
        elif name == "content_block_stop":
            assert d["index"] in opened, f"stop for never-opened block {d['index']}"
            assert d["index"] not in closed, f"block {d['index']} stopped twice"
            closed.add(d["index"])

    # One stop per opened block, none left unclosed; the order is free by
    # decision, so it is asserted as a set.
    stopped = [d["index"] for name, d in events if name == "content_block_stop"]
    assert len(stopped) == 3
    assert set(stopped) == {0, 1, 2}

    # The argument deltas of each call landed under that call's own block.
    args_by_index = {
        d["index"]: json.loads(d["delta"]["partial_json"])
        for name, d in events
        if name == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    }
    assert args_by_index == {0: {"path": "a"}, 1: {"pattern": "x"}}
