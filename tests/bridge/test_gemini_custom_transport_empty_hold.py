"""A custom-transport upstream's empty completion fires the empty ladder on ``/v1/gemini`` (KBR-293).

The Gemini sibling of KBR-287: the ``use_custom_transport`` segment of
``_stream_gemini`` pipes provider bytes straight to the client
(``_tracked_write``) — and on this route the wire is wrong for **all three**
custom-transport providers, not just the empty case. Bedrock and Ollama Cloud
emit Chat Completions SSE on every route; the Codex subscription emits
Responses-API SSE (no ``_original_body`` on the Gemini route, so
``stream_request`` falls into ``_cc_to_responses``). A Gemini CLI client can
read neither. A content-bearing completion from any of them never reaches the
client as a Gemini event, and a content-less one delivers a well-formed but
empty turn with the empty-response ladder unable to fire and the backend
staying healthy.

KBR-293 ports the judge-first shape: collect the provider's bytes, parse them
(the adapter's ``parse_stream_to_cc_response`` or the Responses-SSE fallback),
synthesise the CC chunk list, judge it through the shared
``_cc_chunk_carries_content`` predicate, then translate the synthesis through
``GeminiTranslator`` so the wire is Gemini events either way. Content-bearing
streams reach the Gemini client as proper ``candidates`` events; content-free
ones stay pre-emission and take the ladder, ending in the route's
``reason: "empty_response"`` D4 terminal.

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
whose custom-transport provider has its ``stream_request`` monkeypatched to
feed canned bytes. The canned shapes are what each adapter's real
``stream_request`` emits: Chat Completions SSE for Bedrock and Ollama Cloud,
Responses-API SSE for the subscription.
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


class _GeminiLauncher(LauncherAdapter):
    """Minimal launcher that selects the Gemini bridge protocol."""

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
            The Gemini protocol — every test here is a Gemini CLI client.
        """
        return BridgeProtocol.GEMINI_API

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
# ``stream_request`` writes. The shapes mirror
# ``tests/bridge/test_custom_transport_empty_hold.py`` and
# ``tests/bridge/test_responses_custom_transport_empty_hold.py`` byte for
# byte — the physical-mirror-as-divergence-guard convention (KBR-277/KBR-285).


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
        neither satisfies ``_cc_chunk_carries_content`` (D6: a blank text
        reply is still empty).
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
    """Return a minimal streaming Gemini generateContent request.

    Returns:
        The request body a Gemini CLI client sends.
    """
    return {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


def _parse_data_lines(sse_text: str) -> list[dict]:
    """Parse every JSON ``data:`` line of an SSE body.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The decoded payloads, in order. The ``[DONE]`` sentinel is skipped.
    """
    return [
        json.loads(line[5:].strip())
        for line in sse_text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


def _gemini_texts(sse_text: str) -> list[str]:
    """Collect every ``text`` part from Gemini ``candidates`` payloads.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The text parts, in arrival order — the route-protocol content
        oracle (never a raw-substring check: raw CC-SSE and raw
        Responses-SSE both contain ``hello`` as a substring, but neither is
        a Gemini ``candidates`` payload).
    """
    texts: list[str] = []
    for payload in _parse_data_lines(sse_text):
        for candidate in payload.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if "text" in part:
                    texts.append(part["text"])
    return texts


def _gemini_function_calls(sse_text: str) -> list[dict]:
    """Collect every ``functionCall`` part from Gemini ``candidates`` payloads.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The function-call parts, in arrival order.
    """
    calls: list[dict] = []
    for payload in _parse_data_lines(sse_text):
        for candidate in payload.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if "functionCall" in part:
                    calls.append(part["functionCall"])
    return calls


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
    server = BridgeServer(_GeminiLauncher(), provider(), "sk-test", host="127.0.0.1", port=0)
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
                f"http://127.0.0.1:{server.port}/v1beta/models/test-model:streamGenerateContent",
                json=_client_request(),
            ) as resp,
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
    """KBR-293 AC-FR-4 — a content-less completion retries instead of ending the turn.

    Before the fix the provider's bytes were written unconditionally, so this
    route answered a content-less completion with a single skeleton-ish
    write and the ladder could not fire. After it, the judged-empty attempt
    writes nothing, the ladder retries, and the second response's content
    reaches the client as a Gemini text part.

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
    assert "hello" in _gemini_texts(client_body)
    # The discarded first attempt wrote nothing — no raw Chat Completions
    # chunk (the provider's native wire) ever reached the Gemini client.
    assert all(event.get("object") != "chat.completion.chunk" for event in _parse_data_lines(client_body))


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_a_content_bearing_custom_transport_stream_reaches_the_client(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-293 AC-FR-5 — a content-bearing stream's text reaches the client as Gemini events.

    The judge-first verdict releases a content-bearing completion through the
    route's translator: the client receives a Gemini ``candidates`` payload
    carrying the text — not raw provider bytes in whatever wire the adapter
    speaks (on this route that is the wrong wire for every adapter). One
    upstream call; usage is logged exactly once (log-on-release).

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
    assert "hello" in _gemini_texts(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_a_tool_call_only_custom_transport_stream_releases_the_verdict(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-293 AC-FR-9 — a tool-call-only completion is not treated as empty.

    ``_cc_chunk_carries_content`` counts a non-empty ``tool_calls`` list as
    content, so the synthesised openers release the verdict: one upstream
    call, both calls on the wire as Gemini ``functionCall`` parts with their
    arguments whole.

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
    calls_received = _gemini_function_calls(client_body)
    assert [call.get("name") for call in calls_received] == ["Read", "Write"]
    assert calls_received[0].get("args") == {"path": "a"}
    assert calls_received[1].get("args") == {"path": "b"}


@pytest.mark.asyncio
async def test_a_reasoning_only_custom_transport_completion_takes_the_ladder(monkeypatch):
    """KBR-293 AC-FR-8 — a reasoning-only completion is judged empty, not delivered.

    The synthesis projects only ``content`` and ``tool_calls`` — neither
    parser surfaces ``reasoning_content`` — so a reasoning-only completion
    synthesises the empty shape and the ladder runs, where pre-fix the raw
    bytes (with the reasoning already unreadable to the route) reached the
    client.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls, _bodies = await _stream(
        OllamaCloudAdapter, [_cc_reasoning_only(), _cc_hello()], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in _gemini_texts(client_body)
    assert "only reasoning" not in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_body", "hello_body", "tool_body"), _CUSTOM)
async def test_an_exhausted_custom_transport_empty_ladder_ends_in_the_d4_terminal(
    provider_factory, empty_body, hello_body, tool_body, monkeypatch
):
    """KBR-293 AC-FR-6 — every attempt empty ends in the route's D4 terminal.

    The ladder runs ``n_backends + len(_EMPTY_FINAL_DELAYS)`` attempts —
    three on a single-backend pool — then ends in the same terminal the
    plain-POST twin uses: ``_NATIVE_EMPTY_REPLY_MESSAGE`` with
    ``reason: "empty_response"``. A discarded attempt is not a completion:
    no usage is logged, and no content ever reaches the wire.

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

    assert status == 200
    assert calls == 3  # n_backends (1) + len(_EMPTY_FINAL_DELAYS)
    # Parsed D4 discriminator (the KBR-249 vacuous-oracle trap — a raw
    # substring on ``"reason": "empty_response"`` would pass on any payload
    # that happens to contain that string in some other context). The
    # Gemini D4 event is a ``data:`` payload with shape
    # ``{"error":{"code":502,"message":...,"reason":"empty_response"}}``.
    error_payloads = [
        payload for payload in _parse_data_lines(client_body) if isinstance(payload.get("error"), dict)
    ]
    assert any(payload["error"].get("reason") == "empty_response" for payload in error_payloads)
    assert _NATIVE_EMPTY_REPLY_MESSAGE in client_body
    # The held synthesis was discarded, not flushed: no content on the wire,
    # and no raw Chat Completions chunk either.
    assert _gemini_texts(client_body) == []
    assert all(event.get("object") != "chat.completion.chunk" for event in _parse_data_lines(client_body))
    assert usage_log == []


@pytest.mark.asyncio
async def test_an_empty_custom_transport_attempt_crosses_to_a_healthy_plain_backend(monkeypatch):
    """KBR-293 AC-FR-7 — on a mixed pool the empty ladder crosses to the plain peer.

    The empty arm's backend selection is class-agnostic, mirroring the
    plain-POST ladder: the empty custom attempt selects the healthy plain
    backend, the branch's fall-through hands the request to the plain path,
    and that backend's content reaches the client as Gemini events.

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
        _GeminiLauncher(),
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
                    f"http://127.0.0.1:{server.port}/v1beta/models/test-model:streamGenerateContent",
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
    assert "hello" in _gemini_texts(client_body)
    assert all(event.get("object") != "chat.completion.chunk" for event in _parse_data_lines(client_body))
