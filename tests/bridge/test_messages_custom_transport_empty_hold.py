"""A custom-transport upstream's empty completion fires the empty ladder on ``/v1/messages`` (KBR-297).

KBR-287 judged the ``use_custom_transport`` segment of
``_stream_chat_completions`` and KBR-293 its two siblings
(``_stream_responses`` / ``_stream_gemini``); ``_stream_messages`` — the
Claude Code client's route — was left with no emptiness gate: the branch
parses the provider's bytes, translates them with
``MessagesTranslator.translate_response``, and emits. The translator's
defensive fallback ("never emit thinking-only or empty assistant output")
then fabricates a text block carrying ``_EMPTY_ASSISTANT_FALLBACK_TEXT`` —
the bridge puts words in the model's mouth and ``_log_usage`` bills the
fabricated turn — the empty-response ladder cannot fire, and a balancing
pool keeps routing to the broken upstream.

KBR-297 ports the judge-first shape: collect the provider's bytes, parse
them (the adapter's ``parse_stream_to_cc_response`` or the Responses-SSE
fallback), judge the parsed response through the shared whole-response
predicate ``_is_empty_cc_response`` — no chunk synthesis, the branch is
atomic parse-then-emit — then either translate and emit (content-bearing)
or discard the attempt and walk the empty ladder, ending in the route's own
D4 terminal: a bare JSON 502 with ``error.reason == "empty_response"``
(legal because ``sr`` is deferred on this route, §5.3 S8 of
``SYSTEM_DESIGN.md`` — not the in-stream SSE error terminal the SSE
siblings write).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
whose custom-transport provider has its ``stream_request`` monkeypatched to
feed canned bytes — the branch consumes collected bytes, not an aiohttp
response, so no HTTP mocking is needed on the custom side. The canned shapes
are what each adapter's real ``stream_request`` emits: Chat Completions SSE
for Bedrock and Ollama Cloud, Responses-API SSE for the subscription.
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.messages.translator import _EMPTY_ASSISTANT_FALLBACK_TEXT
from kitty.bridge.server import _NATIVE_EMPTY_REPLY_MESSAGE, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.bedrock import BedrockAdapter
from kitty.providers.ollama_cloud import OllamaCloudAdapter
from kitty.providers.openai import OpenAIAdapter
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter
from kitty.types import BridgeProtocol


class _MessagesLauncher(LauncherAdapter):
    """Minimal launcher that selects the Messages bridge protocol."""

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
            The Messages protocol — every test here is a Claude Code client.
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


# ── Canned upstream shapes ────────────────────────────────────────────────
#
# Bedrock and Ollama Cloud emit Chat Completions SSE on every route (parsed
# by their own ``parse_stream_to_cc_response``); the OpenAI subscription
# emits Responses-API SSE (parsed by the ``_parse_sse_to_response``
# fallback). Each adapter gets the empty and content-bearing shape its real
# ``stream_request`` writes. Ported physically from the KBR-293 sibling
# files (the mirror-as-divergence-guard convention — no shared helper).


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
    """Render Responses-API SSE events as the subscription emits them.

    Args:
        events: Event payloads, in wire order.

    Returns:
        The ``data:``-prefixed SSE body, terminated by ``[DONE]``.
    """
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def _responses_empty() -> bytes:
    """Return the content-less Responses-SSE completion.

    Returns:
        A body whose parsed message carries no content and no tool calls.
    """
    return _responses_sse([_responses_completed()])


def _responses_hello() -> bytes:
    """Return the content-bearing Responses-SSE completion.

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
    """Return a tool-call-only Responses-SSE completion.

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
    """Render Chat Completions chunk payloads as Bedrock/Ollama Cloud emit them.

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
    """Return the content-less CC-SSE completion.

    Returns:
        A role-only opening chunk, a finish-only final chunk, and ``[DONE]`` —
        the parsed message carries nothing ``_is_empty_cc_response`` counts.
    """
    return _cc_sse([_cc_chunk({"role": "assistant", "content": ""}), _cc_chunk({}, finish="stop")])


def _cc_hello() -> bytes:
    """Return the content-bearing CC-SSE completion.

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
    """Return a tool-call-only CC-SSE completion.

    Returns:
        Role chunk, two complete tool-call deltas, finish chunk, ``[DONE]``.
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
    """Return a reasoning-only CC-SSE completion.

    Returns:
        Role chunk, one reasoning delta, finish chunk, ``[DONE]``. The parser
        projects no ``reasoning_content``, so this parses to the empty
        message shape — the projection gap KBR-287/293 pin as
        ladder-taking, not skeleton.
    """
    return _cc_sse(
        [
            _cc_chunk({"role": "assistant", "content": ""}),
            _cc_chunk({"reasoning_content": "only reasoning"}),
            _cc_chunk({}, finish="stop"),
        ]
    )


def _responses_reasoning_only() -> bytes:
    """Return a reasoning-only Responses-SSE completion for the fallback parser.

    Returns:
        A reasoning-summary delta followed by a content-less completion. The
        parser projects no reasoning, so this parses to the empty message
        shape and the ladder runs.
    """
    return _responses_sse(
        [
            {"type": "response.reasoning_summary_text.delta", "delta": "only reasoning"},
            _responses_completed(),
        ]
    )


#: The real custom-transport adapters on this tree, each with the empty and
#: content-bearing canned bytes its real ``stream_request`` emits.
_CUSTOM = [
    pytest.param(BedrockAdapter, _cc_empty, _cc_hello, _cc_tool_calls, id="bedrock"),
    pytest.param(OllamaCloudAdapter, _cc_empty, _cc_hello, _cc_tool_calls, id="ollama_cloud"),
    pytest.param(
        OpenAISubscriptionAdapter,
        _responses_empty,
        _responses_hello,
        _responses_tool_calls,
        id="openai_subscription",
    ),
]


# ── Harness ───────────────────────────────────────────────────────────────


def _client_request() -> dict:
    """Return a minimal streaming Messages API request.

    Returns:
        The request body a Claude Code client sends.
    """
    return {
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def _parse_data_lines(sse_text: str) -> list[dict]:
    """Parse every JSON ``data:`` line of an SSE body.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The decoded payloads, in order. Non-JSON payloads (none on this
        route's wire) and lines without JSON are skipped, so the helper is
        safe to run over a mixed body.
    """
    parsed: list[dict] = []
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return parsed


def _messages_text_deltas(sse_text: str) -> list[str]:
    """Collect every non-empty ``text_delta`` text from Messages events.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The delta texts, in arrival order — the route-protocol content
        oracle (never a raw-substring check).
    """
    return [
        event["delta"]["text"]
        for event in _parse_data_lines(sse_text)
        if event.get("type") == "content_block_delta"
        and event.get("delta", {}).get("type") == "text_delta"
        and event.get("delta", {}).get("text")
    ]


def _messages_tool_calls_received(sse_text: str) -> tuple[list[str], dict[str, str]]:
    """Collect tool-use names and accumulated arguments from Messages events.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The call names in block-opening order, and block index → the
        accumulated ``input_json_delta`` payload for that block.
    """
    names: list[str] = []
    arguments: dict[str, str] = {}
    for event in _parse_data_lines(sse_text):
        if (
            event.get("type") == "content_block_start"
            and event.get("content_block", {}).get("type") == "tool_use"
        ):
            names.append(event["content_block"].get("name", ""))
        elif (
            event.get("type") == "content_block_delta"
            and event.get("delta", {}).get("type") == "input_json_delta"
        ):
            index = str(event.get("index", ""))
            arguments[index] = arguments.get(index, "") + event.get("delta", {}).get("partial_json", "")
    return names, arguments


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
    server = BridgeServer(_MessagesLauncher(), provider(), "sk-test", host="127.0.0.1", port=0)
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
            session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=_client_request()) as resp,
        ):
            return server, resp.status, await resp.text(), calls["n"], bodies
    finally:
        await server.stop_async()


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_an_empty_custom_transport_stream_fires_the_empty_ladder(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-297 AC-FR-2 — a content-less completion retries instead of ending the turn.

    Before the fix the branch translated and emitted unconditionally, so the
    translator's defensive fallback fabricated the empty-assistant text, the
    turn shipped, and the ladder could not fire (``calls == 1``). After it,
    the judged-empty attempt writes nothing — the fallback text never
    reaches the client — the ladder retries, and the second response's
    content reaches the client as a ``text_delta``.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: The canned bytes of a content-less completion.
        hello_body: The canned bytes of a content-bearing completion.
        tool_body: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, [empty_body, hello_body], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in _messages_text_deltas(client_body)
    # The fabricated fallback the translator used to ship for an empty
    # completion never reaches the client on any attempt.
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_a_content_bearing_custom_transport_stream_reaches_the_client(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-297 AC-FR-1 — a content-bearing stream's text reaches the client as Messages events.

    The judge-first verdict releases a content-bearing completion through
    ``translate_response`` exactly as today: the client receives a
    ``content_block_delta`` carrying the text. One upstream call; usage is
    logged exactly once (log-on-release).

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: Unused; part of the shared parametrisation.
        hello_body: The canned bytes of a content-bearing completion.
        tool_body: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, [hello_body], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _messages_text_deltas(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_a_tool_call_only_custom_transport_stream_releases_the_verdict(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-297 AC-FR-4 — a tool-call-only completion is not treated as empty.

    ``_is_empty_cc_response`` counts a non-empty ``tool_calls`` list as
    content, so the verdict releases: one upstream call, both calls on the
    wire as Messages ``tool_use`` content blocks with their arguments whole.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: Unused; part of the shared parametrisation.
        hello_body: Unused; part of the shared parametrisation.
        tool_body: The canned bytes of a tool-call-only completion.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(provider_factory, [tool_body], monkeypatch)

    assert status == 200
    assert calls == 1
    names, arguments = _messages_tool_calls_received(client_body)
    assert names == ["Read", "Write"]
    assert json.loads(arguments["0"]) == {"path": "a"}
    assert json.loads(arguments["1"]) == {"path": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_factory", "reasoning_body", "hello_body"),
    [
        pytest.param(OllamaCloudAdapter, _cc_reasoning_only, _cc_hello, id="ollama_cloud"),
        pytest.param(
            OpenAISubscriptionAdapter,
            _responses_reasoning_only,
            _responses_hello,
            id="openai_subscription",
        ),
    ],
)
async def test_a_reasoning_only_custom_transport_completion_takes_the_ladder(
    provider_factory, reasoning_body, hello_body, monkeypatch
):
    """KBR-297 AC-FR-6 — a reasoning-only completion is judged empty, not delivered.

    Neither parser surfaces reasoning, so a reasoning-only completion parses
    to the empty message shape and the ladder runs — the accepted trade-off
    KBR-287/293 pin. Pre-fix the translator's fallback fabricated the
    empty-assistant text for it; post-fix the fallback text never ships and
    the recovered attempt's content reaches the client.

    Args:
        provider_factory: Builds a custom-transport adapter.
        reasoning_body: The canned bytes of a reasoning-only completion in
            the adapter's native wire.
        hello_body: The canned bytes of a content-bearing completion in the
            same adapter's native wire (the ladder's recovery target).
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(
        provider_factory, [reasoning_body(), hello_body()], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in _messages_text_deltas(client_body)
    assert "only reasoning" not in client_body
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_an_exhausted_custom_transport_empty_ladder_ends_in_the_d4_terminal(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-297 AC-FR-5 — every attempt empty ends in the route's D4 terminal.

    The ladder runs ``n_backends + len(_EMPTY_FINAL_DELAYS)`` attempts —
    three on a single-backend pool — then ends in the same terminal the
    plain path's ``empty_no_finish`` arm uses: a bare JSON 502 with
    ``_NATIVE_EMPTY_REPLY_MESSAGE`` and ``reason: "empty_response"``
    (§5.3 S8 — this route defers ``sr.prepare()``, so a pre-emission
    terminal is a JSON response, not the SSE siblings' in-stream error
    event). A discarded attempt is not a completion: no usage is billed, the
    fallback text is never fabricated, and no content reaches the wire.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_body: The canned bytes of a content-less completion.
        hello_body: Unused; part of the shared parametrisation.
        tool_body: Unused; part of the shared parametrisation.
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

    assert status == 502
    assert calls == 3  # n_backends (1) + len(_EMPTY_FINAL_DELAYS)
    error_body = json.loads(client_body)
    assert error_body["error"]["reason"] == "empty_response"
    assert error_body["error"]["type"] == "api_error"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    # The held syntheses were discarded, not flushed: no content, no
    # fabricated fallback text, no billed usage.
    assert _messages_text_deltas(client_body) == []
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body
    assert usage_log == []


@pytest.mark.asyncio
async def test_an_empty_custom_transport_attempt_crosses_to_a_healthy_plain_backend(monkeypatch):
    """KBR-297 AC-FR-7 — on a mixed pool the empty ladder crosses to the plain peer.

    The empty arm's backend selection is class-agnostic, mirroring the
    plain-POST ladder: the empty custom attempt selects the healthy plain
    backend, the branch's existing fall-through hands the request to the
    plain streaming path, and that backend's content reaches the client as
    Messages text deltas.

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
        _MessagesLauncher(),
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
                session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=_client_request()) as resp,
            ):
                status = resp.status
                client_body = await resp.text()
        finally:
            await server.stop_async()

    assert status == 200
    assert calls["n"] == 1  # the custom attempt; the ladder then crossed
    assert plain_calls["n"] == 1
    assert "hello" in _messages_text_deltas(client_body)
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body
