"""A custom-transport or raw-CC empty non-streaming completion ends in the D4 terminal on ``/v1/gemini`` (KBR-300).

KBR-293 closed the streaming empty-hold on ``/v1/gemini``'s custom-transport
segment: the empty attempt was judged, the ladder fired, and the exhausted
stream ended in the route's ``reason: "empty_response"`` D4 terminal. The
non-streaming arm of ``_handle_gemini`` was left with no gate: ``_request_with_retry``'s
built-in empty ladder walked a judged-empty completion, returned it to the
handler, ``GeminiTranslator.translate_response`` shipped it as a well-formed
empty turn (``candidates[0].content.parts[0].text == ""``) — billed via
``_log_usage`` and the broken backend kept healthy via ``_mark_backend_healthy``
— for both transport classes (plain + custom).

KBR-300 ports the KBR-298 shape to the Gemini non-streaming cell: judge the
parsed ``cc_response`` through ``_is_empty_cc_response`` before
``translate_response``. Content-bearing translates and responds unchanged;
judged-empty ends in the route's D4 terminal as a non-streaming JSON error
(``502`` + ``code: 502`` + ``reason: "empty_response"`` — byte-mirror of
KBR-293's Gemini streaming SSE error payload), with no usage billed and no
healthy-mark. The gate uses the ticket's literal predicate
(``not use_native_messages and _is_empty_cc_response``).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
whose upstream is scripted: custom-transport adapters' ``make_request`` is
monkeypatched at instance level to feed canned raw shapes through the
adapters' real parsers, the plain cell runs against an ``aioresponses``
upstream. Content oracles are parsed from the JSON response body, never raw
substrings.
"""

from __future__ import annotations

import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import _NATIVE_EMPTY_REPLY_MESSAGE, BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
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


class _NativeOpenAIAdapter(OpenAIAdapter):
    """An OpenAI-compatible adapter whose ``use_native_messages`` is True.

    The four KBR-300 gates' literal predicate excluded native providers via
    a ``not use_native_messages and _is_empty_cc_response(cc_response)``
    conjunct. KBR-304 drops that conjunct; this stub exists only to drive
    the post-widening gate's native-provider arm without introducing a new
    real adapter dependency.

    The stub deliberately leaves :attr:`upstream_wire_shape` at the inherited
    :attr:`WireShape.CHAT_COMPLETIONS` — the base-class invariant
    ``use_native_messages ⇒ WireShape.MESSAGES`` is a production-adapter rule
    (see :class:`~kitty.providers.base.ProviderAdapter.use_native_messages`),
    and this test only exercises the gate's conjunct dimension, not the wire
    shape.
    """

    @property
    def use_native_messages(self) -> bool:
        """Return True — the dimension under test.

        Returns:
            Always ``True`` to exercise the widened gate's native arm.
        """
        return True


# ── Canned custom-transport raw upstream shapes ───────────────────────────
#
# Mirrored from KBR-298's file (physical-mirror-as-divergence-guard
# convention, KBR-277/KBR-285). Each adapter's real non-streaming parser
# reads its native shape; the harness feeds the canned shape through the real
# parser so the route judges what the real parse step produces.


def _bedrock_hello() -> dict:
    """Return a content-bearing Bedrock Converse response.

    Returns:
        A raw Converse body whose parsed CC message carries ``content="hello"``.
    """
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 2},
    }


def _bedrock_empty() -> dict:
    """Return a content-less Bedrock Converse response.

    Returns:
        A raw Converse body whose parsed CC message carries no content and no
        tool calls.
    """
    return {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 0},
    }


def _bedrock_tool_calls() -> dict:
    """Return a tool-call-only Bedrock Converse response.

    Returns:
        A raw Converse body whose parsed CC message carries two complete
        tool calls and no text content.
    """
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "call_1", "name": "Read", "input": {"path": "a"}}},
                    {"toolUse": {"toolUseId": "call_2", "name": "Write", "input": {"path": "b"}}},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 1, "outputTokens": 2},
    }


def _bedrock_reasoning_only() -> dict:
    """Return a reasoning-only Bedrock Converse response.

    Returns:
        A raw Converse body whose only content block is a ``reasoningContent``
        block. ``BedrockAdapter.translate_from_upstream`` reads only
        ``text``/``toolUse`` blocks, so this parses to the empty CC message
        shape — the projection gap KBR-287/293/297/300 pin as
        ladder-taking.
    """
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"reasoningContent": {"reasoningText": {"text": "only reasoning"}}}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 0},
    }


def _ollama_hello() -> dict:
    """Return a content-bearing Ollama ``/api/chat`` response.

    Returns:
        A raw chat body whose parsed CC message carries ``content="hello"``.
    """
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": "hello"},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 2,
    }


def _ollama_empty() -> dict:
    """Return a content-less Ollama ``/api/chat`` response.

    Returns:
        A raw chat body whose parsed CC message carries no content and no
        tool calls.
    """
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": ""},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 0,
    }


def _ollama_tool_calls() -> dict:
    """Return a tool-call-only Ollama ``/api/chat`` response.

    Returns:
        A raw chat body whose parsed CC message carries two complete tool
        calls and no text content.
    """
    return {
        "model": "test-model",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "Read", "arguments": {"path": "a"}}},
                {"function": {"name": "Write", "arguments": {"path": "b"}}},
            ],
        },
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 2,
    }


def _ollama_reasoning_only() -> dict:
    """Return a reasoning-only Ollama ``/api/chat`` response.

    Returns:
        A raw chat body whose only extra field is ``message.thinking``.
        ``OllamaCloudAdapter.translate_from_upstream`` reads only
        ``message.content`` and ``message.tool_calls``, so this parses to the
        empty CC message shape.
    """
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": "", "thinking": "only reasoning"},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 0,
    }


def _responses_completed() -> dict:
    """Return the ``response.completed`` event body the subscription parser reads.

    Returns:
        The event dict the parser projects into a response body.
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


def _responses_empty_sse() -> bytes:
    """Return the content-less Codex SSE completion."""
    return _responses_sse([_responses_completed()])


def _responses_hello_sse() -> bytes:
    """Return the content-bearing Codex SSE completion."""
    return _responses_sse(
        [
            {"type": "response.output_text.delta", "delta": "hello"},
            _responses_completed(),
        ]
    )


def _responses_tool_calls_sse() -> bytes:
    """Return a tool-call-only Codex SSE completion."""
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


def _responses_reasoning_only_sse() -> bytes:
    """Return a reasoning-only Codex SSE completion."""
    return _responses_sse(
        [
            {"type": "response.reasoning_summary_text.delta", "delta": "only reasoning"},
            _responses_completed(),
        ]
    )


# ── Canned plain (raw-CC) upstream shapes ─────────────────────────────────


def _cc_hello() -> dict:
    """Return a content-bearing CC response."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _cc_empty() -> dict:
    """Return a content-less CC response."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }


def _cc_tool_calls() -> dict:
    """Return a tool-call-only CC response."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"path": "a"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "Write", "arguments": '{"path": "b"}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _cc_reasoning_only() -> dict:
    """Return a reasoning-content-only CC response.

    KBR-277 counts ``reasoning_content`` non-empty as content, so the gate
    does NOT fire on the raw-CC cell — the reply is released on the first
    attempt and ``GeminiTranslator`` carries the reasoning through the
    Gemini parts (the route's accepted reasoning-carriage behaviour on
    raw-CC).
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "only reasoning",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }


#: The real custom-transport adapters on this tree, each with the empty and
#: content-bearing canned raw shapes its real non-streaming parser reads.
_CUSTOM = [
    pytest.param(BedrockAdapter, _bedrock_empty, _bedrock_hello, _bedrock_tool_calls, id="bedrock"),
    pytest.param(OllamaCloudAdapter, _ollama_empty, _ollama_hello, _ollama_tool_calls, id="ollama_cloud"),
    pytest.param(
        OpenAISubscriptionAdapter,
        _responses_empty_sse,
        _responses_hello_sse,
        _responses_tool_calls_sse,
        id="openai_subscription",
    ),
]


#: The reasoning-only cells (custom transports).
_CUSTOM_REASONING = [
    pytest.param(BedrockAdapter, _bedrock_reasoning_only, _bedrock_hello, id="bedrock"),
    pytest.param(OllamaCloudAdapter, _ollama_reasoning_only, _ollama_hello, id="ollama_cloud"),
    pytest.param(
        OpenAISubscriptionAdapter,
        _responses_reasoning_only_sse,
        _responses_hello_sse,
        id="openai_subscription",
    ),
]


#: The plain raw-CC upstream URL — the ``OpenAIAdapter`` default endpoint.
_UPSTREAM_URL = "https://api.openai.com/v1/chat/completions"


# ── Harness ───────────────────────────────────────────────────────────────


def _client_request() -> dict:
    """Return a minimal non-streaming Gemini generateContent request body.

    Returns:
        The body a Gemini CLI client sends. The route differentiates
        non-streaming (``generateContent``) from streaming
        (``streamGenerateContent``) by the URL path.
    """
    return {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


def _parse_response(body_text: str) -> dict:
    """Parse the non-streaming JSON response body."""
    return json.loads(body_text)


def _gemini_texts(body_text: str) -> list[str]:
    """Collect every ``text`` part from a Gemini non-streaming body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The text parts, in arrival order — the route-protocol content
        oracle.
    """
    parsed = _parse_response(body_text)
    texts: list[str] = []
    for candidate in parsed.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and "text" in part:
                texts.append(part.get("text", ""))
    return texts


def _gemini_function_calls(body_text: str) -> list[dict]:
    """Collect every ``functionCall`` part from a Gemini non-streaming body."""
    parsed = _parse_response(body_text)
    calls: list[dict] = []
    for candidate in parsed.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and "functionCall" in part:
                calls.append(part["functionCall"])
    return calls


def _respond(body: dict) -> CallbackResult:
    """Build a CallbackResult for the canned CC JSON body."""
    return CallbackResult(
        status=200,
        body=json.dumps(body),
        content_type="application/json",
    )


async def _post_custom(
    provider,
    raw_factories: list,
    monkeypatch: pytest.MonkeyPatch,
    on_build=None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming request through a real bridge against scripted custom-transport responses.

    Args:
        provider: A custom-transport adapter instance.
        raw_factories: Callables returning the raw upstream shapes each
            ``make_request`` call yields, in order.
        monkeypatch: Pytest fixture, used for the delay collapse and the
            ``make_request`` replacement.
        on_build: Optional callable invoked with the constructed server before
            it starts.

    Returns:
        The server, the HTTP status, the client's body text, and how many
        times the upstream was entered.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(_GeminiLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
    calls = {"n": 0}

    async def _fake_make_request(cc_request):
        """Serve the next scripted raw shape through the real parser."""
        idx = min(calls["n"], len(raw_factories) - 1)
        calls["n"] += 1
        raw = raw_factories[idx]()
        if isinstance(raw, bytes):
            return provider._parse_sse_to_response(raw)
        return provider.translate_from_upstream(raw)

    monkeypatch.setattr(server._active_provider, "make_request", _fake_make_request)
    if on_build is not None:
        on_build(server)
    await server.start_async()
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"http://127.0.0.1:{server.port}/v1beta/models/test-model:generateContent",
                json=_client_request(),
            ) as resp,
        ):
            return server, resp.status, await resp.text(), calls["n"]
    finally:
        await server.stop_async()


async def _post_plain(
    upstream_bodies: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    on_build=None,
    *,
    provider_factory=OpenAIAdapter,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming request through a real bridge against a scripted raw-CC upstream.

    KBR-304's native-provider cells pass the ``_NativeOpenAIAdapter`` stub
    via ``provider_factory=``; existing plain-CC cells use the default.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(_GeminiLauncher(), provider_factory(), "sk-test", host="127.0.0.1", port=0)
    calls = {"n": 0}

    def _callback(url, **kwargs):
        """Serve the next scripted CC body and record the hit."""
        calls["n"] += 1
        body = upstream_bodies[min(calls["n"] - 1, len(upstream_bodies) - 1)]
        return _respond(body)

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        for _registration in range(len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 4):
            mocked.post(_UPSTREAM_URL, callback=_callback)
        if on_build is not None:
            on_build(server)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{server.port}/v1beta/models/test-model:generateContent",
                    json=_client_request(),
                ) as resp,
            ):
                return server, resp.status, await resp.text(), calls["n"]
        finally:
            await server.stop_async()


# ── Tests — custom-transport cells ────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_a_content_bearing_custom_transport_completion_reaches_the_client(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-6.1 (custom) — a content-bearing completion translates and responds unchanged.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_raw: Unused; part of the shared parametrisation.
        hello_raw: The canned raw shape of a content-bearing completion.
        tool_raw: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call."""
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post_custom(
        provider_factory(), [hello_raw], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _gemini_texts(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_an_empty_custom_transport_completion_ends_in_the_d4_terminal(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-6.2 (custom) — a content-free completion ends in the D4 terminal.

    Pre-fix the handler shipped an empty text part (``text == ""``) as a
    ``200``, billed it, and marked the backend healthy; post-fix no empty
    text part reaches the client, no usage is billed, and no healthy-mark
    fires. The D4 body byte-mirrors KBR-293's Gemini streaming SSE error
    payload: ``error.code == 502``, ``error.message ==
    _NATIVE_EMPTY_REPLY_MESSAGE``, ``error.reason == "empty_response"``.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_raw: The canned raw shape of a content-less completion.
        hello_raw: Unused; part of the shared parametrisation.
        tool_raw: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []
    healthy_log: list[int] = []

    def _record(server):
        """Patch the usage and healthy-mark recorders onto the server."""
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    _server, status, client_body, calls = await _post_custom(
        provider_factory(), [empty_raw], monkeypatch, on_build=_record
    )

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["code"] == 502
    assert error_body["error"]["reason"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_a_tool_call_only_custom_transport_completion_releases_the_verdict(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-6.3 (custom) — a tool-call-only completion is not judged empty.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_raw: Unused; part of the shared parametrisation.
        hello_raw: Unused; part of the shared parametrisation.
        tool_raw: The canned raw shape of a tool-call-only completion.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_custom(provider_factory(), [tool_raw], monkeypatch)

    assert status == 200
    assert calls == 1
    function_calls = _gemini_function_calls(client_body)
    assert [call["name"] for call in function_calls] == ["Read", "Write"]
    assert function_calls[0]["args"] == {"path": "a"}
    assert function_calls[1]["args"] == {"path": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "reasoning_raw", "hello_raw"), _CUSTOM_REASONING)
async def test_a_reasoning_only_custom_transport_completion_takes_the_ladder(
    provider_factory, reasoning_raw, hello_raw, monkeypatch
):
    """KBR-300 AC-FR-6.4 (custom) — a reasoning-only completion is judged empty, not delivered.

    None of the three custom-transport non-streaming parsers surfaces
    reasoning, so a reasoning-only completion parses to the empty message
    shape and the ladder runs — the accepted KBR-287/293/297/300 trade-off.

    Args:
        provider_factory: Builds a custom-transport adapter.
        reasoning_raw: The canned raw shape of a reasoning-only completion.
        hello_raw: The canned raw shape of a content-bearing completion (the
            ladder's recovery target).
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_custom(
        provider_factory(), [reasoning_raw, hello_raw], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in _gemini_texts(client_body)


@pytest.mark.asyncio
async def test_an_empty_custom_transport_attempt_crosses_to_a_healthy_plain_peer(monkeypatch):
    """KBR-300 AC-FR-6.5 — on a mixed pool the empty ladder crosses to the plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next backend
    class-agnostically. On a custom+plain pool the empty custom attempt
    crosses to the plain peer; the peer's content-bearing completion reaches
    the client as a Gemini JSON body. The custom backend was entered exactly
    once.

    Args:
        monkeypatch: Pytest fixture, pins the weighted draws and collapses
            the retry backoff.
    """
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
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(
        _GeminiLauncher(),
        custom,
        "key-custom",
        host="127.0.0.1",
        port=0,
        backends=backends,
    )
    custom_calls = {"n": 0}

    async def _fake_make_request(cc_request):
        """Serve the empty completion from the custom backend."""
        custom_calls["n"] += 1
        return custom.translate_from_upstream(_ollama_empty())

    monkeypatch.setattr(custom, "make_request", _fake_make_request)

    plain_calls = {"n": 0}

    def _respond_plain(url, **kwargs):
        """Serve the plain backend's content-bearing completion."""
        plain_calls["n"] += 1
        return _respond(_cc_hello())

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        for _registration in range(3):
            mocked.post(_UPSTREAM_URL, callback=_respond_plain)
        draws = iter([[0], [1]])
        monkeypatch.setattr(server_module.random, "choices", lambda tier, weights=None, k=None: next(draws))
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{server.port}/v1beta/models/test-model:generateContent",
                    json=_client_request(),
                ) as resp,
            ):
                status = resp.status
                client_body = await resp.text()
        finally:
            await server.stop_async()

    assert status == 200
    assert custom_calls["n"] == 1
    assert plain_calls["n"] == 1
    assert "hello" in _gemini_texts(client_body)


# ── Tests — plain (raw-CC) cells ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_content_bearing_raw_cc_completion_reaches_the_client(monkeypatch):
    """KBR-300 AC-FR-6.1 (plain) — a content-bearing raw-CC completion translates and responds unchanged."""
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call."""
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post_plain(
        [_cc_hello()], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _gemini_texts(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
async def test_an_empty_raw_cc_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-300 AC-FR-6.2 (plain) — a content-free raw-CC completion ends in the D4 terminal.

    Pre-fix ``GeminiTranslator`` shipped an empty text part (``text == ""``)
    as a ``200``, billed it, and marked the backend healthy; post-fix the D4
    terminal replaces the empty part, no usage is billed, and no
    healthy-mark fires.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []
    healthy_log: list[int] = []

    def _record(server):
        """Patch the usage and healthy-mark recorders onto the server."""
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    _server, status, client_body, calls = await _post_plain([_cc_empty()], monkeypatch, on_build=_record)

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["code"] == 502
    assert error_body["error"]["reason"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_tool_call_only_raw_cc_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-6.3 (plain) — a tool-call-only raw-CC completion is not judged empty."""
    _server, status, client_body, calls = await _post_plain([_cc_tool_calls()], monkeypatch)

    assert status == 200
    assert calls == 1
    function_calls = _gemini_function_calls(client_body)
    assert [call["name"] for call in function_calls] == ["Read", "Write"]
    assert function_calls[0]["args"] == {"path": "a"}
    assert function_calls[1]["args"] == {"path": "b"}


@pytest.mark.asyncio
async def test_a_reasoning_only_raw_cc_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-6.4 (plain) — a reasoning-only raw-CC completion is released on first attempt.

    The KBR-277 predicate counts ``reasoning_content`` non-empty as content;
    on a raw-CC upstream the wire shape is CC, so the reply is judged
    non-empty and released on the first attempt.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_plain([_cc_reasoning_only()], monkeypatch)

    assert status == 200
    assert calls == 1
    assert "only reasoning" in client_body


# ── Tests — native-provider cells (KBR-304) ───────────────────────────────
#
# These exercise the post-widening gate's native-provider arm. On the
# sibling routes a native provider's request translates to CC upstream and
# its reply returns through ``translate_from_upstream`` (passthrough for the
# stub), so what the gate judges is CC-shaped regardless of provider class.


@pytest.mark.asyncio
async def test_a_content_bearing_native_completion_reaches_the_client(monkeypatch):
    """KBR-304 AC-FR-3.1 — a content-bearing native completion translates and responds unchanged.

    Args:
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

    _server, status, client_body, calls = await _post_plain(
        [_cc_hello()], monkeypatch, on_build=_record_usage,
        provider_factory=_NativeOpenAIAdapter,
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _gemini_texts(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
async def test_an_empty_native_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-304 AC-FR-3.2 — a content-free native completion ends in the D4 terminal.

    Pre-fix the gate's ``not use_native_messages`` conjunct excluded native
    providers, so the empty CC-shaped reply fell past the gate into
    ``GeminiTranslator.translate_response``'s empty-text-part shipping
    (``candidates[0].content.parts[0].text == ""``) — billed once. KBR-304
    drops the conjunct; the gate now fires and the route's D4 terminal
    replaces the empty text part.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []
    healthy_log: list[int] = []

    def _record(server):
        """Patch the usage and healthy-mark recorders onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    _server, status, client_body, calls = await _post_plain(
        [_cc_empty()], monkeypatch, on_build=_record,
        provider_factory=_NativeOpenAIAdapter,
    )

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["code"] == 502
    assert error_body["error"]["reason"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_tool_call_only_native_completion_releases_the_verdict(monkeypatch):
    """KBR-304 AC-FR-3.3 — a tool-call-only native completion is not judged empty.

    ``_is_empty_cc_response`` counts a non-empty ``tool_calls`` list as
    content (KBR-285 lockstep), so the verdict releases the reply on the
    first attempt: the client receives a Gemini JSON body whose parts carry
    the two function-call entries with their arguments whole.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_plain(
        [_cc_tool_calls()], monkeypatch, provider_factory=_NativeOpenAIAdapter
    )

    assert status == 200
    assert calls == 1
    function_calls = _gemini_function_calls(client_body)
    assert [call["name"] for call in function_calls] == ["Read", "Write"]
    assert function_calls[0]["args"] == {"path": "a"}
    assert function_calls[1]["args"] == {"path": "b"}


@pytest.mark.asyncio
async def test_an_empty_native_attempt_crosses_to_a_healthy_plain_peer(monkeypatch):
    """KBR-304 AC-FR-3.5 — on a [native, plain] pool the empty walk crosses to the plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next backend
    class-agnostically. On a [native, plain] pool the native backend's empty
    walk crosses to the plain peer; the peer's content-bearing completion
    reaches the client as a Gemini JSON body. Total upstream calls == 2.

    Args:
        monkeypatch: Pytest fixture, pins the weighted draws and collapses
            the retry backoff.
    """
    native = _NativeOpenAIAdapter()
    plain = OpenAIAdapter()
    native_profile = Profile(
        name="p-native",
        provider="openai",
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
        (native, "key-native", native_profile),
        (plain, "key-plain", plain_profile),
    ]
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(
        _GeminiLauncher(),
        native,
        "key-native",
        host="127.0.0.1",
        port=0,
        backends=backends,
    )
    calls = {"n": 0}

    def _callback(url, **kwargs):
        """Serve the scripted bodies in order: backend 0 empty, backend 1 content.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The CallbackResult carrying the next scripted CC body.
        """
        calls["n"] += 1
        body = [_cc_empty(), _cc_hello()][min(calls["n"] - 1, 1)]
        return _respond(body)

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        for _registration in range(4):
            mocked.post(_UPSTREAM_URL, callback=_callback)
        draws = iter([[0], [1]])
        monkeypatch.setattr(server_module.random, "choices", lambda tier, weights=None, k=None: next(draws))
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{server.port}/v1beta/models/test-model:generateContent",
                    json=_client_request(),
                ) as resp,
            ):
                status = resp.status
                client_body = await resp.text()
        finally:
            await server.stop_async()

    assert status == 200
    assert calls["n"] == 2
    assert "hello" in _gemini_texts(client_body)
