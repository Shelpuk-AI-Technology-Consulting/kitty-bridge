"""A custom-transport or raw-CC empty non-streaming completion ends in the D4 terminal on the CC route (KBR-300).

KBR-276 closed the streaming empty-hold on the raw-CC segment of
``_stream_chat_completions``: the empty attempt was withheld pre-emission
and the ladder fired. KBR-287 widened the same judge to the custom-transport
segment. KBR-293 left the Gemini route's sister cell closed on the streaming
side. The non-streaming arm of ``_handle_chat_completions`` was left with no
gate: ``_request_with_retry``'s built-in empty ladder walked a judged-empty
completion, returned it to the handler, and the handler shipped the empty CC
body verbatim as a well-formed ``200`` skeleton — billed via ``_log_usage`` and
the broken backend kept healthy via ``_mark_backend_healthy`` — for both
transport classes (plain + custom).

KBR-300 ports the KBR-298 shape to the Chat Completions non-streaming cell.
There is no ``translate_response`` (the route is verbatim passthrough), so the
gate sits immediately before ``_log_usage`` / the verbatim
``web.json_response(cc_response)`` of the body. Content-bearing reaches the
client verbatim, billed once; judged-empty ends in the route's D4 terminal —
a non-streaming JSON error (``502`` + ``type: "empty_response"``, byte-mirror
of KBR-287's Chat Completions streaming D4) — with no usage billed and no
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


class _ChatCompletionsLauncher(LauncherAdapter):
    """Minimal launcher that selects the Chat Completions bridge protocol."""

    @property
    def name(self) -> str:
        """Return the launcher name."""
        return "stub"

    @property
    def binary_name(self) -> str:
        """Return the launcher binary name."""
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol the bridge serves.

        Returns:
            The Chat Completions protocol — every test here is a raw-CC client
            (e.g. Claude Code's configurable-HTTP-OpenAI-compatible path).
        """
        return BridgeProtocol.CHAT_COMPLETIONS_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        """Return an empty spawn configuration."""
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


# ── Canned custom-transport raw upstream shapes ───────────────────────────
#
# Mirrored from KBR-298's file (physical-mirror-as-divergence-guard
# convention, KBR-277/KBR-285). Each adapter's real non-streaming parser
# reads its native shape; the harness feeds the canned shape through the real
# parser so the route judges what the real parse step produces.


def _bedrock_hello() -> dict:
    """Return a content-bearing Bedrock Converse response."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 2},
    }


def _bedrock_empty() -> dict:
    """Return a content-less Bedrock Converse response."""
    return {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 0},
    }


def _bedrock_tool_calls() -> dict:
    """Return a tool-call-only Bedrock Converse response."""
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

    ``BedrockAdapter.translate_from_upstream`` reads only ``text``/``toolUse``
    blocks, so this parses to the empty CC message shape — the projection
    gap KBR-287/293/297/300 pin as ladder-taking.
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
    """Return a content-bearing Ollama ``/api/chat`` response."""
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": "hello"},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 2,
    }


def _ollama_empty() -> dict:
    """Return a content-less Ollama ``/api/chat`` response."""
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": ""},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 0,
    }


def _ollama_tool_calls() -> dict:
    """Return a tool-call-only Ollama ``/api/chat`` response."""
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
    """Return a reasoning-only Ollama ``/api/chat`` response."""
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": "", "thinking": "only reasoning"},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 0,
    }


def _responses_completed() -> dict:
    """Return the ``response.completed`` event body the subscription parser reads."""
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
    """Render Responses-API SSE events as the subscription emits them."""
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
    does NOT fire on the raw-CC cell — the verbatim CC body reaches the
    client with reasoning carriage preserved (the route's accepted
    reasoning-carriage behaviour on raw-CC).
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
    """Return a minimal non-streaming Chat Completions request body."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }


def _parse_response(body_text: str) -> dict:
    """Parse the verbatim CC response body."""
    return json.loads(body_text)


def _cc_message_contents(body_text: str) -> list[str]:
    """Collect every ``choices[0].message.content`` from the verbatim CC body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The message contents, in choice order — the route-protocol content
        oracle.
    """
    parsed = _parse_response(body_text)
    contents: list[str] = []
    for choice in parsed.get("choices") or []:
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            if isinstance(msg, dict) and "content" in msg:
                contents.append(msg.get("content", ""))
    return contents


def _cc_message_tool_calls(body_text: str) -> list[dict]:
    """Collect every ``choices[].message.tool_calls`` from the verbatim CC body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The tool-call dicts, in choice order — the route-protocol tool
        oracle (kept distinct from the ``_cc_tool_calls`` canned-shape
        factory above).
    """
    parsed = _parse_response(body_text)
    calls: list[dict] = []
    for choice in parsed.get("choices") or []:
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            if isinstance(msg, dict):
                calls.extend(msg.get("tool_calls") or [])
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
    """POST a non-streaming CC request through a real bridge against scripted custom-transport responses.

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
    server = BridgeServer(_ChatCompletionsLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
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
            session.post(f"http://127.0.0.1:{server.port}/v1/chat/completions", json=_client_request()) as resp,
        ):
            return server, resp.status, await resp.text(), calls["n"]
    finally:
        await server.stop_async()


async def _post_plain(
    upstream_bodies: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    on_build=None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming CC request through a real bridge against a scripted raw-CC upstream."""
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(_ChatCompletionsLauncher(), OpenAIAdapter(), "sk-test", host="127.0.0.1", port=0)
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
                session.post(f"http://127.0.0.1:{server.port}/v1/chat/completions", json=_client_request()) as resp,
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
    """KBR-300 AC-FR-8.1 (custom) — a content-bearing completion reaches the client verbatim.

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
    assert "hello" in _cc_message_contents(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_an_empty_custom_transport_completion_ends_in_the_d4_terminal(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-8.2 (custom) — a content-free completion ends in the D4 terminal.

    Pre-fix the handler shipped the empty CC body verbatim as a well-formed
    ``200`` skeleton, billed it, and marked the backend healthy; post-fix
    the D4 terminal replaces the empty skeleton, no usage is billed, and no
    healthy-mark fires. The D4 body byte-mirrors KBR-287's Chat Completions
    streaming D4: ``error.message == _NATIVE_EMPTY_REPLY_MESSAGE``,
    ``error.type == "empty_response"`` — verbatim CC carries ``type``, not
    ``reason``.

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
    assert error_body["error"]["type"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_a_tool_call_only_custom_transport_completion_releases_the_verdict(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-8.3 (custom) — a tool-call-only completion is not judged empty.

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
    tool_calls = _cc_message_tool_calls(client_body)
    assert [call["function"]["name"] for call in tool_calls] == ["Read", "Write"]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "a"}
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"path": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "reasoning_raw", "hello_raw"), _CUSTOM_REASONING)
async def test_a_reasoning_only_custom_transport_completion_takes_the_ladder(
    provider_factory, reasoning_raw, hello_raw, monkeypatch
):
    """KBR-300 AC-FR-8.4 (custom) — a reasoning-only completion is judged empty, not delivered.

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
    assert "hello" in _cc_message_contents(client_body)


@pytest.mark.asyncio
async def test_an_empty_custom_transport_attempt_crosses_to_a_healthy_plain_peer(monkeypatch):
    """KBR-300 AC-FR-8.5 — on a mixed pool the empty ladder crosses to the plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next backend
    class-agnostically. On a custom+plain pool the empty custom attempt
    crosses to the plain peer; the peer's content-bearing completion reaches
    the client verbatim. The custom backend was entered exactly once.

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
        _ChatCompletionsLauncher(),
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
                session.post(f"http://127.0.0.1:{server.port}/v1/chat/completions", json=_client_request()) as resp,
            ):
                status = resp.status
                client_body = await resp.text()
        finally:
            await server.stop_async()

    assert status == 200
    assert custom_calls["n"] == 1
    assert plain_calls["n"] == 1
    assert "hello" in _cc_message_contents(client_body)


# ── Tests — plain (raw-CC) cells ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_content_bearing_raw_cc_completion_reaches_the_client(monkeypatch):
    """KBR-300 AC-FR-8.1 (plain) — a content-bearing raw-CC completion reaches the client verbatim."""
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call."""
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post_plain(
        [_cc_hello()], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _cc_message_contents(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
async def test_an_empty_raw_cc_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-300 AC-FR-8.2 (plain) — a content-free raw-CC completion ends in the D4 terminal.

    Pre-fix the handler shipped the empty CC body verbatim as a well-formed
    ``200`` skeleton, billed it, and marked the backend healthy; post-fix
    the D4 terminal replaces the empty skeleton.

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
    assert error_body["error"]["type"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_tool_call_only_raw_cc_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-8.3 (plain) — a tool-call-only raw-CC completion is not judged empty."""
    _server, status, client_body, calls = await _post_plain([_cc_tool_calls()], monkeypatch)

    assert status == 200
    assert calls == 1
    tool_calls = _cc_message_tool_calls(client_body)
    assert [call["function"]["name"] for call in tool_calls] == ["Read", "Write"]
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "a"}
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"path": "b"}


@pytest.mark.asyncio
async def test_a_reasoning_only_raw_cc_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-8.4 (plain) — a reasoning-only raw-CC completion is released verbatim.

    The KBR-277 predicate counts ``reasoning_content`` non-empty as content;
    on a raw-CC upstream the wire shape is CC, so the reply is judged
    non-empty and reaches the client verbatim with reasoning carriage
    preserved.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_plain([_cc_reasoning_only()], monkeypatch)

    assert status == 200
    assert calls == 1
    assert "only reasoning" in client_body
