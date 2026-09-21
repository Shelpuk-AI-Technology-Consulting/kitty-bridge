"""A custom-transport or raw-CC empty non-streaming completion ends in the D4 terminal on ``/v1/responses`` (KBR-300).

KBR-293 closed the streaming empty-hold on ``/v1/responses``: a content-less
upstream stream from Bedrock, Ollama Cloud, or the Codex subscription was
judged empty, the ladder fired, and the exhausted stream ended in the route's
``code: "empty_response"`` discriminator. The non-streaming arm of
``_handle_responses`` was left with no gate: ``_request_with_retry``'s built-in
empty ladder walked a judged-empty completion, returned it to the handler,
``ResponsesTranslator.translate_response`` dressed it in fabricated
``_EMPTY_ASSISTANT_FALLBACK_TEXT``, ``_log_usage`` billed the fabricated turn,
and ``_mark_backend_healthy`` kept the broken upstream in rotation — for **both**
transport classes (plain + custom).

KBR-300 ports the KBR-298 shape to the Responses non-streaming cell: judge the
parsed ``cc_response`` through ``_is_empty_cc_response`` before
``translate_response``. Content-bearing translates and responds unchanged;
judged-empty ends in the route's D4 terminal as a non-streaming JSON error
(``502`` + ``code: "empty_response"`` + ``reason: "empty_response"``), with no
usage billed and no healthy-mark. The gate uses the ticket's literal predicate
(``not use_native_messages and _is_empty_cc_response``); ``use_native_messages``
is a one-conjunct drop away from a wider sweep (recorded in
``REQUIREMENTS.md`` D1, filed for PO review).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
whose upstream is scripted: custom-transport adapters' ``make_request`` is
monkeypatched at instance level to feed canned raw shapes through the
adapters' real parsers (``BedrockAdapter.translate_from_upstream``,
``OllamaCloudAdapter.translate_from_upstream``,
``OpenAISubscriptionAdapter._parse_sse_to_response``), the plain cell runs
against an ``aioresponses`` upstream. Content oracles are parsed from the
JSON response body, never raw substrings (KBR-249 vacuous-oracle rule).
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


class _ResponsesLauncher(LauncherAdapter):
    """Minimal launcher that selects the Responses bridge protocol."""

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
            The Responses protocol — every test here is a Codex CLI client.
        """
        return BridgeProtocol.RESPONSES_API

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


# ── Canned custom-transport raw upstream shapes ───────────────────────────
#
# Each adapter's real non-streaming parser reads its native shape; the harness
# feeds the canned shape through the real parser, so what the route judges is
# what the real parse step produces. Mirrored from
# ``tests/bridge/test_messages_custom_transport_non_streaming_empty_hold.py``
# (the physical-mirror-as-divergence-guard convention, KBR-277/KBR-285).


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
    """Return the content-less Codex SSE completion.

    Returns:
        A body whose parsed message carries no content and no tool calls.
    """
    return _responses_sse([_responses_completed()])


def _responses_hello_sse() -> bytes:
    """Return the content-bearing Codex SSE completion.

    Returns:
        A body whose parsed message carries ``content="hello"``.
    """
    return _responses_sse(
        [
            {"type": "response.output_text.delta", "delta": "hello"},
            _responses_completed(),
        ]
    )


def _responses_tool_calls_sse() -> bytes:
    """Return a tool-call-only Codex SSE completion.

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


def _responses_reasoning_only_sse() -> bytes:
    """Return a reasoning-only Codex SSE completion.

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


# ── Canned plain (raw-CC) upstream shapes ─────────────────────────────────


def _cc_hello() -> dict:
    """Return a content-bearing CC response.

    Returns:
        A CC body whose ``choices[0].message.content`` carries ``"hello"``.
    """
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
    """Return a content-less CC response.

    Returns:
        A CC body whose ``choices[0].message.content`` is empty.
    """
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
    """Return a tool-call-only CC response.

    Returns:
        A CC body whose ``choices[0].message.tool_calls`` carries two complete
        calls.
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

    Returns:
        A CC body whose ``choices[0].message`` carries ``reasoning_content``
        and an empty ``content``. The KBR-277 predicate counts
        ``reasoning_content`` non-empty as content, so the gate does NOT fire
        on this raw-CC cell — the reply is released on the first attempt and
        ``ResponsesTranslator`` carries the reasoning into the Responses
        output (the route's accepted reasoning-carriage behaviour on raw-CC).
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


#: The reasoning-only cells (custom transports). Each adapter's reasoning
#: carriage drops in its non-streaming parser, so the reply parses to the
#: empty message shape and the ladder runs (KBR-287/293/297/300 trade-off).
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
    """Return a minimal non-streaming Responses API request.

    Returns:
        The request body a Codex CLI client sends with ``stream: False``.
    """
    return {
        "model": "test-model",
        "input": [{"type": "message", "role": "user", "content": "hi"}],
        "stream": False,
    }


def _parse_response(body_text: str) -> dict:
    """Parse the non-streaming JSON response body.

    Args:
        body_text: The body the client received.

    Returns:
        The decoded JSON document.
    """
    return json.loads(body_text)


def _responses_output_texts(body_text: str) -> list[str]:
    """Collect every ``output_text`` text from a Responses non-streaming body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The output_text texts, in content order — the route-protocol content
        oracle (never a raw-substring check).
    """
    parsed = _parse_response(body_text)
    texts: list[str] = []
    for item in parsed.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return texts


def _responses_function_calls(body_text: str) -> list[dict]:
    """Collect every ``function_call`` output item from a Responses body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The function_call output items, in order.
    """
    parsed = _parse_response(body_text)
    return [
        item
        for item in parsed.get("output", []) or []
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def _respond(body: dict) -> CallbackResult:
    """Build a CallbackResult for the canned CC JSON body.

    Args:
        body: The CC response dict to return.

    Returns:
        The aioresponses CallbackResult carrying the body as JSON.
    """
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
            it starts — the seam for tests that patch instance methods (e.g.
            the usage and health recorders).

    Returns:
        The server, the HTTP status, the client's body text, and how many
        times the upstream was entered.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(_ResponsesLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
    calls = {"n": 0}

    async def _fake_make_request(cc_request):
        """Serve the next scripted raw shape through the real parser.

        Args:
            cc_request: The request the bridge built; unused beyond the
                contract (the fake replaces only the transport).

        Returns:
            The adapter's real parser applied to the scripted raw shape.
        """
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
            session.post(f"http://127.0.0.1:{server.port}/v1/responses", json=_client_request()) as resp,
        ):
            return server, resp.status, await resp.text(), calls["n"]
    finally:
        await server.stop_async()


async def _post_plain(
    upstream_bodies: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    on_build=None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming request through a real bridge against a scripted raw-CC upstream.

    Args:
        upstream_bodies: The CC bodies each ``_make_upstream_request`` call
            yields, in order.
        monkeypatch: Pytest fixture, used for the delay collapse.
        on_build: Optional callable invoked with the constructed server before
            it starts — the seam for tests that patch instance methods.

    Returns:
        The server, the HTTP status, the client's body text, and how many
        times the upstream was entered.
    """
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])
    server = BridgeServer(_ResponsesLauncher(), OpenAIAdapter(), "sk-test", host="127.0.0.1", port=0)
    calls = {"n": 0}

    def _callback(url, **kwargs):
        """Serve the next scripted CC body and record the hit.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The CallbackResult carrying the next scripted CC body.
        """
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
                session.post(f"http://127.0.0.1:{server.port}/v1/responses", json=_client_request()) as resp,
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
    """KBR-300 AC-FR-4.1 (custom) — a content-bearing completion translates and responds unchanged.

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
        """Patch the server's ``_log_usage`` to capture every call.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post_custom(
        provider_factory(), [hello_raw], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _responses_output_texts(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_an_empty_custom_transport_completion_ends_in_the_d4_terminal(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-4.2 (custom) — a content-free completion ends in the D4 terminal.

    The route's built-in ladder exhausts at
    ``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`` attempts — the
    bound is unchanged by this fix — and the gate then ends the request in
    the route's D4 terminal: a JSON ``502`` with
    ``_NATIVE_EMPTY_REPLY_MESSAGE``, ``code: "empty_response"``, and
    ``reason: "empty_response"``. Before the fix the handler shipped the
    translator's fabricated fallback text as a ``200``, billed it, and
    marked the backend healthy; after it no fallback text reaches the
    client, no usage is billed, and no healthy-mark fires.

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
        """Patch the usage and healthy-mark recorders onto the server.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))
        monkeypatch.setattr(server, "_mark_backend_healthy", lambda idx: healthy_log.append(idx))

    _server, status, client_body, calls = await _post_custom(
        provider_factory(), [empty_raw], monkeypatch, on_build=_record
    )

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["type"] == "error"
    assert error_body["error"]["code"] == "empty_response"
    assert error_body["error"]["reason"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_a_tool_call_only_custom_transport_completion_releases_the_verdict(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-300 AC-FR-4.3 (custom) — a tool-call-only completion is not judged empty.

    ``_is_empty_cc_response`` counts a non-empty ``tool_calls`` list as content
    (KBR-285 lockstep), so the verdict releases the reply on the first
    attempt: the client receives a Responses JSON body whose ``output``
    carries the two function-call items with their arguments whole.

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
    function_calls = _responses_function_calls(client_body)
    assert [call["name"] for call in function_calls] == ["Read", "Write"]
    assert json.loads(function_calls[0]["arguments"]) == {"path": "a"}
    assert json.loads(function_calls[1]["arguments"]) == {"path": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "reasoning_raw", "hello_raw"), _CUSTOM_REASONING)
async def test_a_reasoning_only_custom_transport_completion_takes_the_ladder(
    provider_factory, reasoning_raw, hello_raw, monkeypatch
):
    """KBR-300 AC-FR-4.4 (custom) — a reasoning-only completion is judged empty, not delivered.

    None of the three custom-transport non-streaming parsers surfaces
    reasoning, so a reasoning-only completion parses to the empty message
    shape and the ladder runs — the accepted KBR-287/293/297/300 trade-off.
    Pre-fix the translator's fallback fabricated the empty-assistant text for
    it; post-fix the fallback text never ships and the recovered attempt's
    content reaches the client as Responses output text.

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
    assert "hello" in _responses_output_texts(client_body)


@pytest.mark.asyncio
async def test_an_empty_custom_transport_attempt_crosses_to_a_healthy_plain_peer(monkeypatch):
    """KBR-300 AC-FR-4.5 — on a mixed pool the empty ladder crosses to the plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next backend
    class-agnostically. On a custom+plain pool the empty custom attempt
    crosses to the plain peer; the peer's content-bearing completion reaches
    the client as a Responses JSON body. The custom backend was entered
    exactly once.

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
        _ResponsesLauncher(),
        custom,
        "key-custom",
        host="127.0.0.1",
        port=0,
        backends=backends,
    )
    custom_calls = {"n": 0}

    async def _fake_make_request(cc_request):
        """Serve the empty completion from the custom backend.

        Args:
            cc_request: Unused.

        Returns:
            The parsed CC body of a content-less completion.
        """
        custom_calls["n"] += 1
        return custom.translate_from_upstream(_ollama_empty())

    monkeypatch.setattr(custom, "make_request", _fake_make_request)

    plain_calls = {"n": 0}

    def _respond_plain(url, **kwargs):
        """Serve the plain backend's content-bearing completion.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The scripted CC JSON response.
        """
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
                session.post(f"http://127.0.0.1:{server.port}/v1/responses", json=_client_request()) as resp,
            ):
                status = resp.status
                client_body = await resp.text()
        finally:
            await server.stop_async()

    assert status == 200
    assert custom_calls["n"] == 1
    assert plain_calls["n"] == 1
    assert "hello" in _responses_output_texts(client_body)


# ── Tests — plain (raw-CC) cells ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_content_bearing_raw_cc_completion_reaches_the_client(monkeypatch):
    """KBR-300 AC-FR-4.1 (plain) — a content-bearing raw-CC completion translates and responds unchanged.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff and records
            ``_log_usage`` / ``_mark_backend_healthy`` calls.
    """
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post_plain(
        [_cc_hello()], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _responses_output_texts(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
async def test_an_empty_raw_cc_completion_ends_in_the_d4_terminal(monkeypatch):
    """KBR-300 AC-FR-4.2 (plain) — a content-free raw-CC completion ends in the D4 terminal.

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

    _server, status, client_body, calls = await _post_plain([_cc_empty()], monkeypatch, on_build=_record)

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["type"] == "error"
    assert error_body["error"]["code"] == "empty_response"
    assert error_body["error"]["reason"] == "empty_response"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
async def test_a_tool_call_only_raw_cc_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-4.3 (plain) — a tool-call-only raw-CC completion is not judged empty.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_plain([_cc_tool_calls()], monkeypatch)

    assert status == 200
    assert calls == 1
    function_calls = _responses_function_calls(client_body)
    assert [call["name"] for call in function_calls] == ["Read", "Write"]
    assert json.loads(function_calls[0]["arguments"]) == {"path": "a"}
    assert json.loads(function_calls[1]["arguments"]) == {"path": "b"}


@pytest.mark.asyncio
async def test_a_reasoning_only_raw_cc_completion_releases_the_verdict(monkeypatch):
    """KBR-300 AC-FR-4.4 (plain) — a reasoning-only raw-CC completion is released on first attempt.

    The KBR-277 predicate counts ``reasoning_content`` non-empty as content;
    on a raw-CC upstream the wire shape is CC, so the reply is judged
    non-empty and released on the first attempt. ``ResponsesTranslator``
    carries the reasoning through the Responses output.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post_plain([_cc_reasoning_only()], monkeypatch)

    assert status == 200
    assert calls == 1
    # The reasoning carriage reaches the client somewhere in the Responses
    # output (exact projection is the translator's design, pinned by the
    # existing Responses-translator tests).
    assert "only reasoning" in client_body
