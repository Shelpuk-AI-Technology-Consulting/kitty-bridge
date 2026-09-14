"""The server's translated Responses stream gives each output item its own slot.

KBR-240: on a translated ``/v1/responses`` route (Chat Completions upstream),
``ResponsesTranslator`` pinned text to ``output_index: 0`` and took the first
function call's index from the Chat Completions tool-call index — also 0 — so
a stream with prose and then a tool call, the common Codex shape, announced two
items at slot 0 and closed slot 0 twice. This proves the fix on the path Codex
actually uses: an in-process server, a scripted Chat Completions upstream, and
the client-visible byte stream walked end to end.

Decided stream shape (REQUIREMENTS.md, KBR-240; same decision as KBR-226's G39):
items may overlap in time and need not close in order — clients key items by
``item_id`` and position by ``output_index`` — but each index opens once, closes
once, closes only after it opened, and carries no delta outside its own item.
The ``function_call_arguments`` events carry ``output_index`` too: the vendor
grammar defines it there as a required field.
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
    """Minimal launcher that selects the Responses API bridge protocol."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.RESPONSES_API

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


def _prose_then_tool_call_sse() -> bytes:
    """A Chat Completions stream that writes a sentence and then calls a tool.

    Prose-then-call is the common Codex shape, and the one that collided at
    slot 0 before the fix.
    """
    chunks = [
        _cc_chunk({"content": "Let me check."}),
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
        _cc_chunk({}, finish="tool_calls"),
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _parse_sse(body: bytes) -> list[tuple[str, dict]]:
    """Parse a Responses SSE byte stream into ``(event_name, data)`` pairs.

    Args:
        body: The raw response body the bridge wrote to the client.

    Returns:
        One pair per event, in wire order.
    """
    parsed: list[tuple[str, dict]] = []
    for block in body.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                parsed.append((data["type"], data))
    return parsed


def _make_server() -> BridgeServer:
    """Build a single-backend Responses server on the translated (CC) wire."""
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
    """A Codex-shaped Responses request declaring the tool the upstream calls."""
    return {
        "model": "MiniMax-M3",
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "review"}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "Read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            }
        ],
    }


@pytest.mark.asyncio
async def test_prose_then_tool_call_gets_distinct_output_indices():
    """The client-visible stream opens each output item at its own slot.

    The whole SSE stream is walked against the decided grammar: distinct
    increasing ``output_index`` values, exactly one added and one done per
    ``item_id``, done only after its own added, no delta outside its item's
    window, and ``output_index`` present on the ``function_call_arguments``
    events. The completed response's ``output`` array follows slot order.
    """
    server = _make_server()
    with aioresponses(passthrough=["http://127.0.0.1"]) as m:
        m.post(
            CC_URL,
            callback=lambda u, **kw: CallbackResult(
                status=200,
                headers={"Content-Type": "text/event-stream"},
                body=_prose_then_tool_call_sse(),
            ),
            repeat=True,
        )
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}/v1/responses", json=_client_request()) as resp,
            ):
                assert resp.status == 200
                body = await resp.read()
        finally:
            await server.stop_async()

    events = _parse_sse(body)
    names = [name for name, _ in events]

    # The stream ends with the completed event. (It does not *open* with
    # response.created: the server buffers the start events for the
    # empty-response failover and never writes them — a separate, pre-existing
    # finding recorded on the ticket, not part of KBR-240.)
    assert names[-1] == "response.completed"

    # Each output item opens exactly once, at a distinct increasing index,
    # with the prose message first and the tool call second.
    added = [(d["output_index"], d["item"]["type"]) for name, d in events if name == "response.output_item.added"]
    assert added == [(0, "message"), (1, "function_call")]
    fc_item = next(d["item"] for name, d in events if name == "response.output_item.added" and d["item"]["type"] == "function_call")
    assert fc_item["name"] == "Read"
    assert fc_item["call_id"] == "call_1"

    # Per-item window state keyed by item_id: overlap is permitted, so items
    # are tracked on their own rather than through one "currently open" item.
    opened: dict[str, int] = {}
    closed: set[str] = set()
    for name, d in events:
        item_id = d.get("item_id")
        if name == "response.output_item.added":
            opened[d["item"]["id"]] = d["output_index"]
        elif item_id is not None and name not in ("response.output_item.added",):
            # Any item-addressed event (deltas, content parts, done) belongs
            # to an item that has opened and not yet closed, and carries that
            # item's own slot — deltas included, not just added/done.
            assert item_id in opened, f"event {name} for never-opened item {item_id}"
            assert item_id not in closed, f"event {name} for closed item {item_id}"
            assert d["output_index"] == opened[item_id], (
                f"event {name} for item {item_id} at slot {d['output_index']}, "
                f"expected {opened[item_id]}"
            )
        if name == "response.output_item.done":
            assert d["output_index"] == opened[d["item"]["id"]]
            closed.add(d["item"]["id"])

    # One done per added, none left unclosed; the order is free by decision,
    # so it is asserted as a set.
    done = [d["item"]["id"] for name, d in events if name == "response.output_item.done"]
    assert len(done) == 2
    assert set(done) == set(opened)

    # The arguments events carry the vendor-required output_index, and it is
    # the call's own slot.
    args_indices = {
        d["output_index"] for name, d in events if name.startswith("response.function_call_arguments.")
    }
    assert args_indices == {1}

    # The completed response's output array follows slot order: the message
    # (slot 0) before the function call (slot 1).
    completed = events[-1][1]["response"]["output"]
    assert [item["type"] for item in completed] == ["message", "function_call"]
    assert completed[1]["arguments"] == '{"path": "a"}'
