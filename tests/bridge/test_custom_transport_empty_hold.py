"""A custom-transport upstream's empty completion fires the empty ladder (KBR-287).

KBR-248 gave ``/v1/chat/completions`` a pre-emission hold for the converted
route and KBR-276 widened it to every plain-POST Chat Completions upstream,
but the ``use_custom_transport`` segment the KBR-254 cross-class re-dispatch
routes into still synthesised its own CC stream and wrote it straight to the
client: a content-less completion from Bedrock, Ollama Cloud, or the Codex
subscription reached the agent as a well-formed skeleton (role chunk, finish
chunk, ``[DONE]``) and the empty-response ladder could not fire. KBR-287
judges the synthesised chunk list through the shared
``_cc_chunk_carries_content`` predicate before any write — content-bearing
streams are byte-identical to the pre-change synthesis, content-free ones
stay pre-emission and take the ladder, ending in the route's ``empty_response``
D4 terminal.

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
whose custom-transport provider has its ``stream_request`` monkeypatched to
feed canned bytes — the branch consumes collected bytes, not an aiohttp
response, so no HTTP mocking is needed on the custom side. The canned shapes
are what the branch's parse step reads: Chat Completions SSE for the
adapters with their own ``parse_stream_to_cc_response`` (Ollama Cloud, and
Bedrock — added by the review round in this change, closing the gap where
Bedrock's CC-SSE bytes met the Responses-SSE fallback) and Responses-API
SSE for the OpenAI subscription (the ``_parse_sse_to_response`` fallback).
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import _NATIVE_EMPTY_REPLY_MESSAGE, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.bedrock import BedrockAdapter
from kitty.providers.ollama_cloud import OllamaCloudAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter
from kitty.types import BridgeProtocol

#: A tool schema the client declares and the auditor can hold a call against.
_READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


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


# ── Canned upstream shapes ────────────────────────────────────────────────
#
# Bedrock and Ollama Cloud parse Chat Completions SSE with their own
# ``parse_stream_to_cc_response``; the OpenAI subscription has no parser of
# its own, so the branch parses its bytes with the Responses-API SSE
# fallback. Each adapter gets the empty and content-bearing shape its
# parser reads.


def _responses_completed() -> dict:
    """Return the ``response.completed`` event body the fallback parser reads.

    Returns:
        The event dict with the model and usage the parser projects into the
        synthesised stream.
    """
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "model": "test-model",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "output": [],
            "status": "completed",
        },
    }


def _responses_sse(events: list[dict]) -> bytes:
    """Render Responses-API SSE events as the branch's fallback parser reads them.

    Args:
        events: Event payloads, in wire order.

    Returns:
        The ``data:``-prefixed SSE body, terminated by ``[DONE]``.
    """
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def _responses_empty() -> bytes:
    """Return the content-less Responses-SSE completion for the fallback parser.

    Returns:
        A body whose parsed message carries no content and no tool calls.
    """
    return _responses_sse([_responses_completed()])


def _responses_hello() -> bytes:
    """Return the content-bearing Responses-SSE completion for the fallback parser.

    Returns:
        A body whose parsed message carries ``content="hello"``.
    """
    return _responses_sse(
        [
            {"type": "response.output_text.delta", "delta": "hello"},
            _responses_completed(),
        ]
    )


def _responses_tool_calls() -> bytes:
    """Return a tool-call-only Responses-SSE completion for the fallback parser.

    Returns:
        A body whose parsed message carries two complete tool calls and no
        text content.
    """
    return _responses_sse(
        [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read"},
            },
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "call_id": "call_2", "name": "Write"},
            },
            {"type": "response.function_call_arguments.delta", "call_id": "call_1", "delta": '{"path": "a"}'},
            {"type": "response.function_call_arguments.delta", "call_id": "call_2", "delta": '{"path": "b"}'},
            _responses_completed(),
        ]
    )


def _cc_sse(chunks: list[dict]) -> bytes:
    """Render Chat Completions chunk payloads as Ollama Cloud's parser reads them.

    Args:
        chunks: Chunk payloads, in wire order.

    Returns:
        The ``data:``-prefixed SSE body, terminated by ``[DONE]``.
    """
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"


def _cc_chunk(delta: dict, finish: str | None = None) -> dict:
    """Build one Chat Completions streaming chunk payload.

    Args:
        delta: The ``choices[0].delta`` object of the chunk.
        finish: The chunk's ``finish_reason``, if it is the final chunk.

    Returns:
        A Chat Completions chunk dict.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _cc_empty() -> bytes:
    """Return the content-less CC-SSE completion for Ollama Cloud's parser.

    Returns:
        A role-only opening chunk, a finish-only final chunk, and ``[DONE]`` —
        neither satisfies ``_cc_chunk_carries_content`` (D6: a blank text
        reply is still empty).
    """
    return _cc_sse([_cc_chunk({"role": "assistant", "content": ""}), _cc_chunk({}, finish="stop")])


def _cc_hello() -> bytes:
    """Return the content-bearing CC-SSE completion for Ollama Cloud's parser.

    Returns:
        Role chunk, one text delta, finish chunk, ``[DONE]``.
    """
    return _cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
            _cc_chunk({"content": "hello"}),
            _cc_chunk({}, finish="stop"),
        ]
    )


def _cc_tool_calls() -> bytes:
    """Return a tool-call-only CC-SSE completion for Ollama Cloud's parser.

    Returns:
        Role chunk, two complete tool-call deltas (the synthesis reads
        ``function.name`` on every accumulated entry), finish chunk, ``[DONE]``.
    """
    return _cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
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
                            "function": {"name": "Write", "arguments": '{"path": "b"}'},
                        }
                    ]
                }
            ),
            _cc_chunk({}, finish="tool_calls"),
        ]
    )


def _cc_reasoning_only() -> bytes:
    """Return a reasoning-only CC-SSE completion for Ollama Cloud's parser.

    Returns:
        Role chunk, one reasoning delta, finish chunk, ``[DONE]``. The parser
        projects no ``reasoning_content`` and the synthesis reads only
        ``content``/``tool_calls``, so this synthesises the empty shape — the
        projection gap KBR-287 pins as ladder-taking, not skeleton.
    """
    return _cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
            _cc_chunk({"reasoning_content": "only reasoning"}),
            _cc_chunk({}, finish="stop"),
        ]
    )


#: The real custom-transport adapters on this tree, each with the empty and
#: content-bearing canned bytes its parse step reads. (The ticket named
#: ``vertex``; ``VertexAIAdapter`` is a plain-POST OpenAI-compatible
#: passthrough on this tree — ``use_custom_transport`` is False — and is
#: already covered by KBR-276's hold on the plain-POST branch.)
_CUSTOM = [
    pytest.param(BedrockAdapter, _cc_empty, _cc_hello, id="bedrock"),
    pytest.param(OllamaCloudAdapter, _cc_empty, _cc_hello, id="ollama_cloud"),
    pytest.param(OpenAISubscriptionAdapter, _responses_empty, _responses_hello, id="openai_subscription"),
]


# ── Harness ───────────────────────────────────────────────────────────────


def _client_request() -> dict:
    """Return a minimal streaming Chat Completions request.

    Returns:
        The request body, carrying the tool schema the auditor holds calls
        against.
    """
    return {
        "model": "test-model",
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
    upstream_bodies: list[bytes],
    monkeypatch: pytest.MonkeyPatch,
    on_build=None,
) -> tuple[BridgeServer, int, str, int, list[dict]]:
    """POST a streaming request through a real bridge against scripted custom-transport bytes.

    The provider's ``stream_request`` is replaced at instance level (the
    class attribute stays, so ``_any_healthy_backend(require_streaming)``
    still classifies it custom-transport) with a fake that feeds the canned
    bodies in order, repeating the last one — a defect that keeps the ladder
    walking fails on its assertions instead of starving the script. The retry
    backoff and the empty-ladder final delays are collapsed so ladder tests
    stay fast.

    Args:
        provider: A custom-transport adapter class (resolves
            ``use_custom_transport = True``).
        upstream_bodies: The canned bytes each ``stream_request`` call
            yields, in order.
        monkeypatch: Pytest fixture, used for the delay collapse and the
            ``stream_request`` replacement.
        on_build: Optional callable invoked with the constructed server
            before it starts — the seam for tests that patch instance
            methods (e.g. the usage recorder).

    Returns:
        The server, the HTTP status, the client's body text, how many times
        the upstream was entered, and the request bodies the bridge built, in
        arrival order.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(_ProtocolLauncher(), provider(), "sk-test", host="127.0.0.1", port=0)
    calls = {"n": 0}
    bodies: list[dict] = []
    # Parametrised entries are builder callables; direct entries are bytes.
    scripted = [body() if callable(body) else body for body in upstream_bodies]

    async def _fake_stream_request(cc_request, write, _calls=calls, _bodies=bodies, _scripted=scripted):
        """Serve the next canned body and record the hit.

        Args:
            cc_request: The request the bridge built; recorded.
            write: The collector callback the branch assembled.
            _calls: The hit counter, bound at definition time.
            _bodies: The request recorder, bound at definition time.
            _scripted: The canned bodies, bound at definition time.
        """
        _calls["n"] += 1
        _bodies.append(dict(cc_request))
        body = _scripted.pop(0) if len(_scripted) > 1 else _scripted[0]
        await write(body)

    monkeypatch.setattr(server._active_provider, "stream_request", _fake_stream_request)
    if on_build is not None:
        on_build(server)
    await server.start_async()
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"http://127.0.0.1:{server.port}/v1/chat/completions",
                json=_client_request(),
            ) as resp,
        ):
            return server, resp.status, await resp.text(), calls["n"], bodies
    finally:
        await server.stop_async()


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body"), _CUSTOM)
async def test_an_empty_custom_transport_stream_fires_the_empty_ladder(
    provider_factory, empty_body, hello_body, monkeypatch
):
    """KBR-287 AC-1 — a content-less completion retries instead of ending the turn.

    Before the fix the synthesised skeleton was written unconditionally, so
    this route answered a content-less completion with role chunk → finish →
    ``[DONE]`` and the ladder could not fire. After it, the judged-empty
    attempt writes nothing, the ladder retries, and the second response's
    content reaches the client.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: The canned bytes of a content-less completion.
        hello_body: The canned bytes of a content-bearing completion.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, [empty_body, hello_body], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in client_body
    # The discarded first attempt wrote nothing — not even its synthesised
    # role chunk, whose ``content`` is ``null`` on this branch's synthesis.
    assert '"content": null' not in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body"), _CUSTOM)
async def test_a_content_bearing_custom_transport_stream_is_byte_identical(
    provider_factory, empty_body, hello_body, monkeypatch
):
    """KBR-287 AC-2 — a content-bearing stream's wire output is the unchanged synthesis.

    The judge-first verdict releases a content-bearing completion whole: the
    client receives exactly the chunk sequence the branch synthesised before
    the change, in synthesis order. The parser-generated ``id`` and
    ``created`` vary per parse, so the expected body is rebuilt around the
    values the client's first chunk carries and compared byte-for-byte.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: Unused; part of the shared parametrisation.
        hello_body: The canned bytes of a content-bearing completion.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(provider_factory, [hello_body], monkeypatch)

    assert status == 200
    assert calls == 1
    events = _parse_data_lines(client_body)
    first = events[0]
    response_id = first["id"]
    created = first["created"]
    # Usage sits on the finish chunk, not the role/opener; capture from the
    # last event so ollama's parser-hardcoded zeros and the fallback's
    # canned usage are both picked up.
    usage = events[-1].get("usage") if len(events) > 1 else first.get("usage")

    def _expected_chunk(delta: dict, finish: str | None = None, chunk_usage: dict | None = None) -> str:
        """Build one expected synthesis line with the branch's key order.

        Args:
            delta: The chunk's ``choices[0].delta`` object.
            finish: The chunk's ``finish_reason``.
            chunk_usage: The chunk's ``usage`` block, when it carries one.

        Returns:
            The encoded ``data:`` line.
        """
        payload: dict = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if chunk_usage is not None:
            payload["usage"] = chunk_usage
        return f"data: {json.dumps(payload)}\n\n"

    expected = (
        _expected_chunk({"role": "assistant", "content": ""})
        + _expected_chunk({"content": "hello"})
        + _expected_chunk({}, finish="stop", chunk_usage=usage)
        + "data: [DONE]\n\n"
    )
    assert client_body == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body"), _CUSTOM)
async def test_a_tool_call_only_custom_transport_stream_releases_the_verdict(
    provider_factory, empty_body, hello_body, monkeypatch
):
    """KBR-287 AC-3a — a tool-call-only completion is not treated as empty.

    ``_cc_chunk_carries_content`` counts a non-empty ``tool_calls`` list as
    content, so the synthesised openers release the verdict: one upstream
    call, both calls on the wire with their arguments whole.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: Unused; part of the shared parametrisation.
        hello_body: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    # Bedrock and Ollama parse CC-SSE (their own parsers); the OpenAI
    # subscription takes the Responses-SSE fallback.
    tool_body = _cc_tool_calls() if provider_factory is not OpenAISubscriptionAdapter else _responses_tool_calls()

    _server, status, client_body, calls, _bodies = await _stream(provider_factory, [tool_body], monkeypatch)

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
async def test_a_reasoning_only_custom_transport_completion_takes_the_ladder(monkeypatch):
    """KBR-287 AC-3b — a reasoning-only completion is judged empty, not delivered.

    The synthesis projects only ``content`` and ``tool_calls`` — neither
    parser surfaces ``reasoning_content`` — so a reasoning-only completion
    synthesises the empty shape and the ladder runs, where pre-fix the
    skeleton (with the reasoning already dropped) reached the client.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(
        OllamaCloudAdapter, [_cc_reasoning_only(), _cc_hello()], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in client_body
    assert "only reasoning" not in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body"), _CUSTOM)
async def test_an_exhausted_custom_transport_empty_ladder_ends_in_the_d4_terminal(
    provider_factory, empty_body, hello_body, monkeypatch
):
    """KBR-287 AC-4/AC-6 — every attempt empty ends in the route's D4 terminal.

    The ladder runs ``n_backends + len(_EMPTY_FINAL_DELAYS)`` attempts —
    three on a single-backend pool — then ends in the same terminal the
    plain-POST twin uses: ``_NATIVE_EMPTY_REPLY_MESSAGE`` with
    ``type: "empty_response"``, followed by ``[DONE]``. A discarded attempt
    is not a completion: no usage is logged, and no synthesised role chunk
    ever reaches the wire.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: The canned bytes of a content-less completion.
        hello_body: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` calls.
    """
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, [empty_body], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 3  # n_backends (1) + len(_EMPTY_FINAL_DELAYS)
    assert "empty_response" in client_body
    assert _NATIVE_EMPTY_REPLY_MESSAGE in client_body
    assert client_body.rstrip().endswith("data: [DONE]")
    # The held synthesis was discarded, not flushed: no role chunk on the wire.
    assert '"role": "assistant"' not in client_body
    assert usage_log == []


@pytest.mark.asyncio
async def test_an_empty_custom_transport_attempt_crosses_to_a_healthy_plain_backend(monkeypatch):
    """KBR-287 AC-7 — on a mixed pool the empty ladder crosses to the plain peer.

    The empty arm's backend selection is class-agnostic, mirroring the
    plain-POST ladder: the empty custom attempt selects the healthy plain
    backend, the branch's fall-through hands the request to the plain path,
    and that backend's content reaches the client.

    Args:
        monkeypatch: Pytest fixture, pins the backend draw and collapses the
            retry backoff.
    """
    import uuid

    from kitty.profiles.schema import Profile

    custom = OllamaCloudAdapter()
    plain = OpenAIAdapter()
    custom_profile = Profile(
        name="p-custom",
        provider="ollama_cloud",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    plain_profile = Profile(
        name="p-plain",
        provider="openai",
        model="test-model",
        auth_ref=str(uuid.uuid4()),
    )
    backends = [
        (custom, "key-custom", custom_profile),
        (plain, "key-plain", plain_profile),
    ]
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(
        _ProtocolLauncher(),
        custom,
        "key-custom",
        host="127.0.0.1",
        port=0,
        backends=backends,
    )
    calls = {"n": 0}

    async def _fake_stream_request(cc_request, write):
        """Serve the empty completion from the custom backend.

        Args:
            cc_request: Unused.
            write: The collector callback the branch assembled.
        """
        calls["n"] += 1
        await write(_cc_empty())

    monkeypatch.setattr(custom, "stream_request", _fake_stream_request)

    # The plain path rebuilds its URL from the provider that the empty
    # ladder's class-agnostic ``_select_backend`` lands on — the OpenAI
    # adapter in this mixed pool — so register its default endpoint
    # directly. Calling ``server._build_upstream_url`` here would read the
    # custom provider selected at construction time (the ContextVar is not
    # yet populated), giving a different URL than the plain path uses.
    upstream_url = "https://api.openai.com/v1/chat/completions"
    plain_calls = {"n": 0}

    def _respond(url, **kwargs):
        """Serve the plain backend's content-bearing stream.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The scripted SSE response.
        """
        plain_calls["n"] += 1
        return CallbackResult(status=200, body=_cc_hello(), content_type="text/event-stream")

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        for _registration in range(3):
            mocked.post(upstream_url, callback=_respond)
        # Pin the weighted draws: the request starts on the custom backend
        # (index 0) and the empty arm's class-agnostic select must land on
        # the plain peer (index 1), not re-draw the sick custom backend.
        draws = iter([[0], [1]])
        monkeypatch.setattr(server_module.random, "choices", lambda tier, weights=None, k=None: next(draws))
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    json=_client_request(),
                ) as resp,
            ):
                status = resp.status
                client_body = await resp.text()
        finally:
            await server.stop_async()

    assert status == 200
    assert calls["n"] == 1  # the custom attempt; the ladder then crossed
    assert plain_calls["n"] == 1
    assert "hello" in client_body
    assert '"content": null' not in client_body
