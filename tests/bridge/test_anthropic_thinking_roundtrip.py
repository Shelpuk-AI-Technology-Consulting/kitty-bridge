"""Thinking round-trips through the anthropic translated route (KBR-228).

The default ``anthropic`` route translates Claude Code's Messages request to
Chat Completions, sends an Anthropic Messages body upstream, and translates the
reply back.  Before KBR-228 the reply's ``thinking`` and ``redacted_thinking``
blocks — signatures included — were destroyed on that path, and the follow-up
turn shipped an unsigned rebuild that api.anthropic.com rejects under its
signature binding.  These tests drive a real in-process
:class:`~kitty.bridge.server.BridgeServer` against ``aioresponses`` and pin the
whole round trip: thinking reaches the client with its signature, and the
client's signed blocks go back upstream byte-identical together with the
system prompt they are bound to.
"""

from __future__ import annotations

import copy
import json

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.anthropic import AnthropicAdapter
from kitty.types import BridgeProtocol

_UPSTREAM_URL = "https://api.anthropic.com/v1/messages"

#: An upstream reply whose thinking precedes its text, signatures included.
_SIGNED_REPLY = {
    "id": "msg_up1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {"type": "thinking", "thinking": "I recall the weather.", "signature": "sig-turn-1"},
        {"type": "redacted_thinking", "data": "opaque-bytes"},
        {"type": "text", "text": "It is 18C in Paris."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

#: A system prompt as Claude Code sends it: block list, breakpoints included.
_SYSTEM_BLOCKS = [
    {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "Use the tools."},
]


class _StubLauncher(LauncherAdapter):
    """Minimal launcher so the bridge can be constructed without a real agent CLI."""

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


class _StubCCLauncher(_StubLauncher):
    """Launcher variant whose protocol mounts the Chat Completions endpoint."""

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol the bridge serves.

        Returns:
            :attr:`BridgeProtocol.CHAT_COMPLETIONS_API`.
        """
        return BridgeProtocol.CHAT_COMPLETIONS_API


def _translated_server(protocol_launcher: LauncherAdapter | None = None) -> BridgeServer:
    """Build the default ``anthropic`` bridge: the translated Messages route.

    Args:
        protocol_launcher: The launcher whose protocol picks the mounted
            routes; a Messages-protocol stub when omitted.

    Returns:
        An unstarted bridge.
    """
    return BridgeServer(
        adapter=protocol_launcher or _StubLauncher(),
        provider=AnthropicAdapter(),
        resolved_key="key-anthropic",
        model="claude-sonnet-4-6",
        host="127.0.0.1",
        port=0,
    )


async def _drive(
    server: BridgeServer,
    replies: list[tuple[int, object]],
    *,
    path: str = "/v1/messages",
    stream: bool = False,
    history: list[dict] | None = None,
    system: object = None,
):
    """Serve scripted upstream replies in order and return what the client and upstream saw.

    Args:
        server: An unstarted bridge.
        replies: ``(status, body)`` per upstream call; a dict body is JSON, a str body is SSE.
        path: The client-side endpoint to post to.
        stream: Whether the client request streams.
        history: The agent's transcript; one user turn when omitted.
        system: The agent's ``system`` value; omitted when ``None``.

    Returns:
        ``(status, client_body_text, upstream_calls)`` where each call is
        ``(json_body, headers)``.
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

    request: dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "stream": stream,
        "messages": copy.deepcopy(history) if history is not None else [{"role": "user", "content": "hi"}],
    }
    if system is not None:
        request["system"] = copy.deepcopy(system)
    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        mocked.post(_UPSTREAM_URL, callback=respond, repeat=True)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}{path}", json=request) as resp,
            ):
                return resp.status, await resp.text(), calls
        finally:
            await server.stop_async()


class TestNonStreamingResponseCarriage:
    """The reply's thinking reaches the Messages client with its signature (AC-1, non-streaming)."""

    @pytest.mark.asyncio
    async def test_signed_thinking_reaches_the_messages_client_in_wire_order(self):
        status, text, _calls = await _drive(_translated_server(), [(200, _SIGNED_REPLY)])
        assert status == 200
        body = json.loads(text)
        assert [block["type"] for block in body["content"]] == ["thinking", "redacted_thinking", "text"]
        assert body["content"][0]["thinking"] == "I recall the weather."
        assert body["content"][0]["signature"] == "sig-turn-1"
        assert body["content"][1]["data"] == "opaque-bytes"
        assert body["content"][2]["text"] == "It is 18C in Paris."

    @pytest.mark.asyncio
    async def test_the_carriage_key_never_reaches_the_messages_client(self):
        status, text, _calls = await _drive(_translated_server(), [(200, _SIGNED_REPLY)])
        assert status == 200
        assert "_thinking_blocks" not in json.loads(text)


class TestFollowUpTurnRestoresTheSignedHistory:
    """The client's signed blocks and system go back upstream byte-identical (AC-2).

    Anthropic signature-binds every thinking block to the conversation that
    produced it: the block, and the ``system`` prompt and messages sent before
    it.  Turn 2 therefore re-sends what turn 1 produced — the signed blocks
    the client received, and the system prompt as the client addressed it —
    and kitty must ship both verbatim instead of rebuilding them unsigned and
    joined.
    """

    @pytest.mark.asyncio
    async def test_turn_two_ships_signed_blocks_and_system_byte_identical(self):
        system = list(_SYSTEM_BLOCKS)
        turn1_history = [{"role": "user", "content": "What's the weather in Paris?"}]
        status1, text1, calls = await _drive(
            _translated_server(),
            [(200, _SIGNED_REPLY)],
            history=turn1_history,
            system=system,
        )
        assert status1 == 200
        client_blocks = json.loads(text1)["content"]

        # The client replays exactly what it received, as Claude Code does.
        turn2_history = [
            *turn1_history,
            {"role": "assistant", "content": client_blocks},
            {"role": "user", "content": "And in London?"},
        ]
        status2, _text2, calls2 = await _drive(
            _translated_server(),
            [(200, _SIGNED_REPLY)],
            history=turn2_history,
            system=system,
        )
        assert status2 == 200
        assert len(calls2) == 1
        upstream_body = calls2[0][0]

        assistant_on_wire = [m for m in upstream_body["messages"] if m.get("role") == "assistant"]
        assert assistant_on_wire[0]["content"] == _SIGNED_REPLY["content"]
        assert "_thinking_blocks" not in assistant_on_wire[0]
        assert upstream_body["system"] == system


class TestChatCompletionsContainment:
    """No downstream client except Messages sees the internal carriage (AC-3, response half)."""

    @pytest.mark.asyncio
    async def test_chat_completions_client_never_sees_the_carriage_key(self):
        status, text, _calls = await _drive(
            _translated_server(_StubCCLauncher()), [(200, _SIGNED_REPLY)], path="/v1/chat/completions"
        )
        assert status == 200
        body = json.loads(text)
        assert "_thinking_blocks" not in body["choices"][0]["message"]
