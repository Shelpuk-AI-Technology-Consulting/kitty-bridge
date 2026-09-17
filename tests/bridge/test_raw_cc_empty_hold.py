"""A raw Chat Completions upstream's empty completion fires the empty ladder (KBR-276).

KBR-248 gave ``/v1/chat/completions`` a pre-emission hold for the converted
route (a Messages-wire upstream behind the ``AnthropicCCStreamConverter``) but
deliberately left the hold converter-gated: a raw Chat Completions-wire
upstream — OpenAI, OpenRouter, DeepSeek, any plain-POST CC backend — still
answered a content-less completion with a well-formed skeleton (role chunk,
finish chunk, ``[DONE]``), because its first chunk set ``has_content`` and
permanently disarmed the empty-response ladder. KBR-276 removes the gate: the
hold now withholds every upstream's non-content lines, so an empty raw-CC
attempt stays pre-emission and the existing ladder fires, ending in the same
route-wide D4 terminal KBR-248 shipped.

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
against an ``aioresponses`` upstream, the harness KBR-227/248 proved for this
class of defect. The providers are raw-CC adapters (``_stream_converter_for``
returns ``None`` for them) and the scripted bodies are OpenAI-shaped ``data:``
lines — the wire shape the ticket names.
"""

from __future__ import annotations

import copy
import json

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.types import BridgeProtocol

#: A tool schema the client declares and the auditor can hold a call against.
_READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

#: The raw Chat Completions-wire adapters, each with a model it serves. Both
#: resolve to a plain-POST ``/chat/completions`` upstream with no stream
#: converter — the path KBR-276 widens the hold to.
_RAW_CC = [
    pytest.param(OpenAIAdapter, "gpt-5.2", id="openai"),
    pytest.param(CustomOpenAIAdapter, "gpt-5.2", id="custom_openai"),
]


class _ProtocolLauncher(LauncherAdapter):
    """Minimal launcher that selects the Chat Completions bridge protocol."""

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
            The Chat Completions protocol — every test here is a CC client.
        """
        return BridgeProtocol.CHAT_COMPLETIONS_API

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


def _cc_chunk(delta: dict, finish: str | None = None) -> dict:
    """Build one OpenAI-shaped Chat Completions streaming chunk.

    Args:
        delta: The ``choices[0].delta`` object of the chunk.
        finish: The chunk's ``finish_reason``, if it is the final chunk.

    Returns:
        A Chat Completions chunk dict.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gpt-5.2",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _render_cc_sse(chunks: list[dict]) -> str:
    """Render Chat Completions chunk payloads as an SSE body.

    Args:
        chunks: Chunk payloads, in wire order.

    Returns:
        The ``data:``-prefixed SSE text, terminated by ``[DONE]``.
    """
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def _empty_stream() -> str:
    """Return the content-less completion the ladder tests script.

    The OpenAI shape of an empty completion: a role-only opening chunk, a
    final chunk that carries only ``finish_reason``, and ``[DONE]``. Neither
    chunk satisfies ``_cc_chunk_carries_content`` (D6: a blank text reply is
    still empty).

    Returns:
        The SSE body.
    """
    return _render_cc_sse([_cc_chunk({"role": "assistant", "content": ""}), _cc_chunk({}, finish="stop")])


def _hello_stream() -> str:
    """Return the content-bearing completion the recovery tests script.

    Returns:
        The SSE body: role chunk, one text delta, finish chunk, ``[DONE]``.
    """
    return _render_cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
            _cc_chunk({"content": "hello"}),
            _cc_chunk({}, finish="stop"),
        ]
    )


def _client_request(model: str) -> dict:
    """Return a minimal streaming Chat Completions request.

    Args:
        model: The model the agent asks for.

    Returns:
        The request body.
    """
    return {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": "read a"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "Read", "description": "Read a file", "parameters": _READ_SCHEMA},
            }
        ],
    }


def _parse_data_lines(sse_text: str) -> list[dict]:
    """Parse every JSON ``data:`` line of an SSE body.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The decoded payloads, in order. The ``[DONE]`` sentinel is skipped —
        tests that care about it assert on the raw body instead.
    """
    return [
        json.loads(line[5:].strip())
        for line in sse_text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


def _cc_tool_call_fragments(events: list[dict]) -> list[dict]:
    """Collect every ``tool_calls`` entry from Chat Completions chunks.

    Args:
        events: Decoded Chat Completions chunk payloads.

    Returns:
        The tool-call deltas, in arrival order.
    """
    return [
        tc
        for event in events
        if event.get("choices")
        for tc in event["choices"][0].get("delta", {}).get("tool_calls", [])
    ]


async def _stream(
    provider,
    model: str,
    upstream_bodies: list[tuple[int, str]],
    monkeypatch=None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a streaming request through a real bridge against scripted upstream responses.

    Args:
        provider: The raw-CC adapter the bridge serves with.
        model: The model the agent asks for.
        upstream_bodies: ``(status, body)`` pairs served in order.
        monkeypatch: When given, the retry backoff is collapsed to near zero so
            ladder tests stay fast.

    Returns:
        The server, the HTTP status, the client's body text, how many
        requests reached the upstream, and the upstream request bodies in
        arrival order.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
        # The final empty-response and transport-blip delays are read straight
        # from their tuples and run in real seconds; a test that reaches them
        # has already failed its assertions and should do so quickly.
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.01, 0.01))
    server = BridgeServer(
        _ProtocolLauncher(), provider(), "sk-test", host="127.0.0.1", port=0
    )
    upstream_url = server._build_upstream_url({"model": model})
    calls = {"n": 0}
    bodies: list[dict] = []
    scripted = list(upstream_bodies)

    def _respond(url, **kwargs):
        """Serve the next scripted response and record the hit.

        The last response repeats once the script runs dry, so a defect that
        keeps the ladder walking fails on its assertions quickly instead of
        erroring the upstream transport into the long final-delay schedule.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters; the JSON body is recorded.

        Returns:
            The next scripted response as an ``aioresponses`` result.
        """
        calls["n"] += 1
        # Deep-copied: the strip is copy-on-write on the same dict a later
        # attempt re-serializes, so a shared reference would show the retry's
        # state instead of what this request carried.
        bodies.append(copy.deepcopy(kwargs.get("json") or {}))
        status, body = scripted.pop(0) if len(scripted) > 1 else scripted[0]
        return CallbackResult(status=status, body=body or "", content_type="text/event-stream")

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        # One registration per scripted response plus spares: an aioresponses
        # registration is consumed by a single request, and the retry ladder
        # POSTs again — the hold-last rule in _respond keeps those defined.
        for _registration in range(len(upstream_bodies) + 2):
            mocked.post(upstream_url, callback=_respond)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    json=_client_request(model),
                ) as resp,
            ):
                return server, resp.status, await resp.text(), calls["n"], bodies
        finally:
            await server.stop_async()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _RAW_CC)
async def test_an_empty_raw_cc_stream_fires_the_empty_ladder(provider_factory, model, monkeypatch):
    """KBR-276 AC-1 — an OpenAI-shaped empty completion retries instead of ending the turn.

    Before the fix the role chunk set ``has_content``, so this route answered a
    content-less completion with a well-formed skeleton and the ladder could
    not fire (``calls == 1``, no ``hello`` on the wire). After it, the held
    first attempt writes nothing, the ladder retries, and the second
    response's content reaches the client.

    Args:
        provider_factory: Builds a raw Chat Completions-wire adapter.
        model: A model that adapter serves.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    good = _hello_stream()

    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, model, [(200, _empty_stream()), (200, good)], monkeypatch
    )

    assert status == 200
    assert calls == 2
    # The discarded first attempt wrote nothing — not even its held role
    # chunk — so the client body is exactly the second attempt's stream.
    assert client_body == good


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _RAW_CC)
async def test_a_content_bearing_raw_cc_stream_is_byte_identical(provider_factory, model, monkeypatch):
    """KBR-276 AC-2 — a content-bearing stream's wire output is unchanged.

    The hold releases on the first content-bearing chunk and flushes the held
    role chunk ahead of it, so the client receives exactly the bytes the
    upstream sent, in the upstream's order — the raw path's
    ``translate_upstream_stream_event`` is identity, so byte equality is the
    strong form of the claim.

    Args:
        provider_factory: Builds a raw Chat Completions-wire adapter.
        model: A model that adapter serves.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    body = _hello_stream()

    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, model, [(200, body)], monkeypatch
    )

    assert status == 200
    assert calls == 1
    assert client_body == body


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _RAW_CC)
async def test_a_parallel_tool_call_raw_cc_stream_releases_the_hold(provider_factory, model, monkeypatch):
    """KBR-276 AC-3a — two parallel tool calls release the hold on the first delta.

    ``_cc_chunk_carries_content`` counts a non-empty ``tool_calls`` list as
    content, so the hold releases on the first tool-call chunk: one upstream
    call, both calls on the wire with their arguments whole.

    Args:
        provider_factory: Builds a raw Chat Completions-wire adapter.
        model: A model that adapter serves.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    body = _render_cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
            _cc_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"path": '},
                        }
                    ]
                }
            ),
            _cc_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a"}'}}]}),
            _cc_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "Write", "arguments": '{"path": "b"}'},
                        }
                    ]
                }
            ),
            _cc_chunk({}, finish="tool_calls"),
        ]
    )

    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, model, [(200, body)], monkeypatch
    )

    assert status == 200
    assert calls == 1
    events = _parse_data_lines(client_body)
    tool_deltas = _cc_tool_call_fragments(events)
    opened = [tc for tc in tool_deltas if "id" in tc]
    assert [tc["id"] for tc in opened] == ["call_1", "call_2"]
    arguments: dict[int, str] = {}
    for tc in tool_deltas:
        arguments[tc["index"]] = arguments.get(tc["index"], "") + tc["function"]["arguments"]
    assert json.loads(arguments[0]) == {"path": "a"}
    assert json.loads(arguments[1]) == {"path": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _RAW_CC)
async def test_a_reasoning_only_raw_cc_prefix_releases_the_hold(provider_factory, model, monkeypatch):
    """KBR-276 AC-3b — a stream whose only content is reasoning is not treated as empty.

    ``reasoning_content`` is client-visible content: the hold releases on the
    first thinking delta, the upstream is hit once, and the reasoning reaches
    the client.

    Args:
        provider_factory: Builds a raw Chat Completions-wire adapter.
        model: A model that adapter serves.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    body = _render_cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
            _cc_chunk({"reasoning_content": "only reasoning"}),
            _cc_chunk({}, finish="stop"),
        ]
    )

    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, model, [(200, body)], monkeypatch
    )

    assert status == 200
    assert calls == 1
    events = _parse_data_lines(client_body)
    deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "only reasoning"
    assert "".join(d.get("content", "") for d in deltas) == ""


@pytest.mark.asyncio
async def test_an_exhausted_raw_cc_empty_ladder_ends_in_the_d4_terminal(monkeypatch):
    """KBR-276 AC-4 — every attempt empty ends in the D4 error, not the empty stream.

    The terminal itself shipped route-wide with KBR-248; what KBR-276 changes
    is that a *raw* CC upstream can now reach it, because the hold keeps the
    role chunk off the wire and ``has_content`` stays False. The body carries
    ``type: "empty_response"`` — this route's D4 discriminator is ``type``
    alone — followed by ``[DONE]``, and the held preamble never reaches it.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and pins the
            ladder to exactly two attempts so the harness's three mock
            callbacks suffice.
    """
    # The harness registers ``len(upstream_bodies) + 2`` mock callbacks —
    # three, for a single scripted body. Trim the ladder so the second (final)
    # attempt is the one that exhausts.
    monkeypatch.setattr(server_module, "_MAX_RETRIES", 0)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01])
    _server, status, client_body, calls, _bodies = await _stream(
        OpenAIAdapter, "gpt-5.2", [(200, _empty_stream())], monkeypatch
    )

    assert status == 200
    assert calls > 1
    assert "empty_response" in client_body
    assert "Kitty Bridge received an empty reply from the upstream provider on every attempt" in client_body
    assert client_body.rstrip().endswith("data: [DONE]")
    # The held preamble was discarded, not flushed: no role chunk on the wire.
    assert '"role": "assistant"' not in client_body
