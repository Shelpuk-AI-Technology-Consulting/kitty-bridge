"""A custom-transport upstream's empty non-streaming completion ends in the D4 terminal on ``/v1/messages`` (KBR-298).

KBR-297 judged the streaming custom-transport segment of ``_stream_messages``;
the non-streaming branch of ``_handle_messages`` was left with no emptiness
gate. ``_request_with_retry`` walks its built-in empty ladder (single-backend:
``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`` attempts;
balancing: ``n_backends``, empties never marking a backend unhealthy) and its
exhaustion arm returns the empty response by design — which
``MessagesTranslator.translate_response`` dresses in fabricated fallback text
(``_EMPTY_ASSISTANT_FALLBACK_TEXT``), ``_log_usage`` bills as a completion,
and ``_mark_backend_healthy`` keeps in a balancing pool's rotation.

KBR-298 ports the judge-first shape to the non-streaming cell: the translated
arm judges the returned ``cc_response`` through ``_is_empty_cc_response`` (the
KBR-285 lockstep whole-response judge the streaming twin uses) before
``translate_response`` — content-bearing translates and responds unchanged;
judged-empty ends in the route's D4 terminal as a non-streaming JSON error
(``502`` + ``reason: "empty_response"``), with no usage billed and no
healthy-mark. The judge sits after the ladder — the non-streaming walk is
atomic per attempt, so it can only see a content-bearing reply or the
exhausted empty one — and is gated on ``use_custom_transport`` (the ticket's
cell; the raw-CC sibling is a separate ticket).

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
whose custom-transport provider has its ``make_request`` replaced at instance
level (the class attribute stays, so pool classification still sees a
custom-transport provider) with a fake that feeds the canned **raw upstream
shape** through the adapter's real parser — Bedrock Converse dicts and Ollama
``/api/chat`` dicts through ``translate_from_upstream``, Codex SSE bytes
through ``_parse_sse_to_response`` — so the canned bytes pass through the real
parse step and only the transport is replaced. Content oracles are parsed from
the JSON response body, never raw substrings (the KBR-249 vacuous-oracle rule).
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
# Each factory returns the adapter's **raw** upstream shape: a Bedrock
# Converse response dict, an Ollama ``/api/chat`` response dict, or Codex
# Responses-API SSE bytes. The harness fake feeds it through the adapter's
# real parser (``translate_from_upstream`` / ``_parse_sse_to_response``), so
# what the route judges is what the real parse step produces. Ported
# physically from the KBR-297 sibling file (the mirror-as-divergence-guard
# convention — no shared helper).


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
        tool calls — nothing ``_is_empty_cc_response`` counts.
    """
    return {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 0},
    }


def _bedrock_tool_calls() -> dict:
    """Return a tool-call-only Bedrock Converse response.

    Returns:
        A raw Converse body whose parsed CC message carries two complete tool
        calls and no text content.
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
        shape — the projection gap the KBR-287/293/297 tickets pin as
        ladder-taking, not skeleton.
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
    """Return the ``response.completed`` event body the fallback parser reads.

    Returns:
        The event dict with the model and usage the parser projects into the
        synthesised response.
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
    """Return the content-less Codex SSE completion.

    Returns:
        A body whose parsed message carries no content and no tool calls.
    """
    return _responses_sse([_responses_completed()])


def _responses_hello() -> bytes:
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


def _responses_tool_calls() -> bytes:
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


def _responses_reasoning_only() -> bytes:
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


#: The real custom-transport adapters on this tree, each with the empty and
#: content-bearing canned raw shapes its real non-streaming parser reads.
_CUSTOM = [
    pytest.param(BedrockAdapter, _bedrock_empty, _bedrock_hello, _bedrock_tool_calls, id="bedrock"),
    pytest.param(OllamaCloudAdapter, _ollama_empty, _ollama_hello, _ollama_tool_calls, id="ollama_cloud"),
    pytest.param(
        OpenAISubscriptionAdapter,
        _responses_empty,
        _responses_hello,
        _responses_tool_calls,
        id="openai_subscription",
    ),
]

#: The reasoning-only cells: each adapter's native reasoning carriage (Converse
#: ``reasoningContent`` blocks, Ollama ``message.thinking``, Codex
#: reasoning-summary deltas) beside its content-bearing shape. No custom
#: adapter's non-streaming parser surfaces reasoning, so every cell parses to
#: the empty message shape — the accepted KBR-287/293/297 trade-off.
_CUSTOM_REASONING = [
    pytest.param(BedrockAdapter, _bedrock_reasoning_only, _bedrock_hello, id="bedrock"),
    pytest.param(OllamaCloudAdapter, _ollama_reasoning_only, _ollama_hello, id="ollama_cloud"),
    pytest.param(
        OpenAISubscriptionAdapter,
        _responses_reasoning_only,
        _responses_hello,
        id="openai_subscription",
    ),
]


# ── Harness ───────────────────────────────────────────────────────────────


def _client_request() -> dict:
    """Return a minimal non-streaming Messages API request.

    Returns:
        The request body a Claude Code client sends with ``stream: false``.
    """
    return {
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }


def _parse_response(body_text: str) -> dict:
    """Parse the non-streaming JSON response body.

    Args:
        body_text: The body the client received.

    Returns:
        The decoded JSON document. The route answers JSON in every outcome
        under test (a translated message or the D4 error), so a decode
        failure is itself a defect this helper surfaces loudly.
    """
    return json.loads(body_text)


def _message_text_blocks(body_text: str) -> list[str]:
    """Collect every ``text`` block's text from a translated Messages body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The text block texts, in content order — the route-protocol content
        oracle (never a raw-substring check).
    """
    parsed = _parse_response(body_text)
    return [
        block.get("text", "")
        for block in parsed.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _message_tool_blocks(body_text: str) -> list[dict]:
    """Collect every ``tool_use`` block from a translated Messages body.

    Args:
        body_text: The JSON body the client received.

    Returns:
        The tool_use blocks, in content order.
    """
    parsed = _parse_response(body_text)
    return [
        block for block in parsed.get("content", []) if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


async def _post(
    provider,
    raw_factories: list,
    monkeypatch: pytest.MonkeyPatch,
    on_build=None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a non-streaming request through a real bridge against scripted custom-transport responses.

    The provider's ``make_request`` is replaced at instance level (the class
    attribute stays, so pool classification still sees a custom-transport
    provider) with a fake that feeds the scripted **raw upstream shapes**
    through the adapter's real parser in order, repeating the last one — a
    defect that keeps the ladder walking fails on its call-count assertions
    instead of starving the script. The retry backoff and the empty-ladder
    delays are collapsed (lengths untouched — the F30 coupling derives bounds
    from ``len()``) so ladder tests stay fast.

    Args:
        provider: A custom-transport adapter instance.
        raw_factories: Callables returning the raw upstream shapes (Converse
            dicts, Ollama chat dicts, or Codex SSE bytes) each ``make_request``
            call yields, in order.
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
    server = BridgeServer(_MessagesLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
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
        # The subscription's non-streaming parser is a staticmethod over the
        # collected SSE bytes; Bedrock and Ollama Cloud translate a raw dict.
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
            session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=_client_request()) as resp,
        ):
            return server, resp.status, await resp.text(), calls["n"]
    finally:
        await server.stop_async()


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_a_content_bearing_non_streaming_completion_reaches_the_client(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-298 AC-FR-1 — a content-bearing completion translates and responds unchanged.

    The judge releases a content-bearing completion through
    ``translate_response`` exactly as today: the client receives a Messages
    body whose parsed ``content`` carries the upstream text. One upstream
    call; usage is logged exactly once.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_raw: Unused; part of the shared parametrisation.
        hello_raw: The canned raw shape of a content-bearing completion.
        tool_raw: Unused; part of the shared parametrisation.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    usage_log: list[dict | None] = []

    def _record_usage(server):
        """Patch the server's ``_log_usage`` to capture every call.

        Args:
            server: The bridge instance, attached just before ``start_async``.
        """
        monkeypatch.setattr(server, "_log_usage", lambda usage: usage_log.append(usage))

    _server, status, client_body, calls = await _post(
        provider_factory(), [hello_raw], monkeypatch, on_build=_record_usage
    )

    assert status == 200
    assert calls == 1
    assert "hello" in _message_text_blocks(client_body)
    assert len(usage_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_an_empty_non_streaming_completion_ends_in_the_d4_terminal(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-298 AC-FR-2/FR-5 — a content-free completion ends in the D4 terminal, not fabricated text.

    ``_request_with_retry``'s walk exhausts on a single-backend pool at
    ``len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`` attempts — the
    route's existing bound, unchanged by this fix — and the judge then ends
    the request in the D4 terminal: a JSON ``502`` with
    ``_NATIVE_EMPTY_REPLY_MESSAGE`` and ``reason: "empty_response"``. Before
    the fix the handler shipped the translator's fabricated fallback text as
    a ``200``, billed it, and marked the backend healthy; after it no
    fallback text reaches the client, no usage is billed, and no healthy-mark
    fires.

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

    _server, status, client_body, calls = await _post(
        provider_factory(), [empty_raw], monkeypatch, on_build=_record
    )

    assert status == 502
    assert calls == len(server_module._EMPTY_RETRY_DELAYS) + len(server_module._EMPTY_FINAL_DELAYS) + 1
    error_body = _parse_response(client_body)
    assert error_body["error"]["reason"] == "empty_response"
    assert error_body["error"]["type"] == "api_error"
    assert _NATIVE_EMPTY_REPLY_MESSAGE in error_body["error"]["message"]
    # The exhausted completion was judged, not dressed: no fabricated
    # fallback text, no billed usage, no healthy-mark.
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body
    assert usage_log == []
    assert healthy_log == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "empty_raw", "hello_raw", "tool_raw"), _CUSTOM)
async def test_a_tool_call_only_non_streaming_completion_is_not_judged_empty(
    provider_factory, empty_raw, hello_raw, tool_raw, monkeypatch
):
    """KBR-298 AC-FR-3 — a tool-call-only completion is not treated as empty.

    ``_is_empty_cc_response`` counts a non-empty ``tool_calls`` list as
    content, so the verdict releases: one upstream call, both calls on the
    client body as Messages ``tool_use`` blocks with their arguments whole.

    Args:
        provider_factory: Builds a custom-transport adapter.
        empty_raw: Unused; part of the shared parametrisation.
        hello_raw: Unused; part of the shared parametrisation.
        tool_raw: The canned raw shape of a tool-call-only completion.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post(provider_factory(), [tool_raw], monkeypatch)

    assert status == 200
    assert calls == 1
    tool_blocks = _message_tool_blocks(client_body)
    assert [block["name"] for block in tool_blocks] == ["Read", "Write"]
    assert tool_blocks[0]["input"] == {"path": "a"}
    assert tool_blocks[1]["input"] == {"path": "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "reasoning_raw", "hello_raw"), _CUSTOM_REASONING)
async def test_a_reasoning_only_non_streaming_completion_takes_the_ladder(
    provider_factory, reasoning_raw, hello_raw, monkeypatch
):
    """KBR-298 AC-FR-4 — a reasoning-only completion is judged empty, not delivered.

    No custom adapter's non-streaming parser surfaces reasoning, so a
    reasoning-only completion parses to the empty message shape and the
    ladder runs — the accepted trade-off KBR-287/293/297 pin. Pre-fix the
    translator's fallback fabricated the empty-assistant text for it;
    post-fix the fallback text never ships and the recovered attempt's
    content reaches the client.

    Args:
        provider_factory: Builds a custom-transport adapter.
        reasoning_raw: The canned raw shape of a reasoning-only completion in
            the adapter's native upstream dialect.
        hello_raw: The canned raw shape of a content-bearing completion (the
            ladder's recovery target).
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    _server, status, client_body, calls = await _post(provider_factory(), [reasoning_raw, hello_raw], monkeypatch)

    assert status == 200
    assert calls == 2
    assert "hello" in _message_text_blocks(client_body)
    assert "only reasoning" not in client_body
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body


@pytest.mark.asyncio
async def test_an_empty_custom_attempt_crosses_to_a_healthy_plain_backend(monkeypatch):
    """KBR-298 AC-FR-6 — on a mixed pool the empty walk crosses to the plain peer.

    ``_request_with_retry_balancing``'s empty arm selects the next backend
    class-agnostically; the crossed attempt lands on the plain provider, and
    the plain backend's content-bearing completion reaches the client — the
    transport-gated judge must not fire on the crossed plain response. The
    custom backend was entered exactly once.

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
    monkeypatch.setattr(server_module, "_EMPTY_RETRY_DELAYS", [0.01, 0.01])
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

    async def _fake_make_request(cc_request):
        """Serve the empty completion from the custom backend.

        Args:
            cc_request: Unused.

        Returns:
            The parsed CC body of a content-less completion.
        """
        calls["n"] += 1
        return custom.translate_from_upstream(_ollama_empty())

    monkeypatch.setattr(custom, "make_request", _fake_make_request)

    # The plain path rebuilds its URL from the provider the empty walk's
    # class-agnostic ``_select_backend`` lands on — the OpenAI adapter in
    # this mixed pool — so register its default endpoint directly. Calling
    # ``server._build_upstream_url`` here would read the custom provider
    # selected at construction time (the ContextVar is not yet populated),
    # giving a different URL than the plain path uses.
    upstream_url = "https://api.openai.com/v1/chat/completions"
    plain_calls = {"n": 0}

    def _respond(url, **kwargs):
        """Serve the plain backend's content-bearing completion.

        Args:
            url: The request URL, unused.
            **kwargs: The request parameters, unused.

        Returns:
            The scripted CC JSON response.
        """
        plain_calls["n"] += 1
        return CallbackResult(
            status=200,
            body=json.dumps(
                {
                    "id": "chatcmpl-plain",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }
            ),
            content_type="application/json",
        )

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        for _registration in range(3):
            mocked.post(upstream_url, callback=_respond)
        # Pin the weighted draws: the request starts on the custom backend
        # (index 0) and the empty walk's class-agnostic select must land on
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
    assert calls["n"] == 1  # the custom attempt; the walk then crossed
    assert plain_calls["n"] == 1
    assert "hello" in _message_text_blocks(client_body)
    assert _EMPTY_ASSISTANT_FALLBACK_TEXT not in client_body
