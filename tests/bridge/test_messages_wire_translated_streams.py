"""The three non-Messages streams work over an Anthropic-wire translated adapter (KBR-232).

Codex CLI (``/v1/responses``) and Gemini CLI (``:streamGenerateContent``) got an
empty 200 — their handlers parsed the upstream's Anthropic SSE as Chat
Completions chunks and discarded every event — and Chat Completions clients
received text but lost every tool call and all thinking, because the per-event
translator had no state to stream a ``tool_use`` block's arguments under.

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
against an ``aioresponses`` upstream, the harness KBR-227 proved for this class
of defect: the choice under test is the handler's, not an adapter hook's.  The
upstream stream is the same in every test — thinking, text, then one tool call —
so each protocol's assertions are about what reaches *its* client.
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
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.opencode import OpenCodeGoAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter
from kitty.types import BridgeProtocol

#: A tool schema the client declares and the auditor can hold a call against.
_READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}

#: Verbatim shape of api.anthropic.com's unsigned-signature rejection (KBR-238 probes).
_SIGNATURE_REJECTION = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "messages.1.content.0.thinking.signature: Field required"},
}


class _ProtocolLauncher(LauncherAdapter):
    """Minimal launcher that selects one inbound bridge protocol."""

    def __init__(self, protocol: BridgeProtocol) -> None:
        """Record the protocol the stub launcher advertises.

        Args:
            protocol: The protocol whose route the server registers.
        """
        self._protocol = protocol

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
            The protocol this instance was built with.
        """
        return self._protocol

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


#: The Messages-wire adapters, each with a model that serves on that wire: the
#: translated ones the ticket names, and the native ones (`custom_anthropic`,
#: `zai_coding`) whose upstream streams are the same Anthropic SSE.
_MESSAGES_WIRE = [
    pytest.param(AnthropicAdapter, "claude-opus-4-6", id="anthropic"),
    pytest.param(MiniMaxTokenAnthropicAdapter, "MiniMax-M3", id="minimax_token-default"),
    pytest.param(OpenCodeGoAdapter, "minimax-m2.7", id="opencode_go-messages-model"),
    pytest.param(CustomAnthropicAdapter, "claude-opus-4-6", id="custom_anthropic-native"),
    pytest.param(ZaiAnthropicAdapter, "claude-opus-4-6", id="zai_coding-native"),
]


def _anthropic_events() -> list[dict]:
    """Return the upstream stream every test serves: thinking, text, then one tool call.

    Returns:
        The Anthropic event payloads, in order.
    """
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_upstream",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "look first"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": '},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '"a"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]


def _render_sse(events: list[dict], *, terminate: bool = True) -> str:
    """Render Anthropic event payloads as an SSE body.

    Args:
        events: Event payloads, each carrying its ``type``.
        terminate: When ``False``, the final ``data:`` line is left without its
            trailing newline, so the handler's flush path must process it.

    Returns:
        The ``event:``/``data:`` SSE text.
    """
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return body if terminate else body[:-2]


def _parse_data_lines(sse_text: str) -> list[dict]:
    """Parse every JSON ``data:`` line of an SSE body.

    Args:
        sse_text: The SSE body the client received.

    Returns:
        The decoded payloads, in order.  The ``[DONE]`` sentinel is skipped —
        tests that care about it assert on the raw body instead.
    """
    return [
        json.loads(line[5:].strip())
        for line in sse_text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


def _client_request(protocol: BridgeProtocol, model: str) -> dict:
    """Return a streaming request whose transcript carries a prior thinking turn.

    The prior thinking is what a ``thinking.signature`` rejection bites on
    (AC-3): each protocol carries it in its own shape — ``reasoning_content``
    for Chat Completions, a ``reasoning`` item for Responses, a ``thought``
    part for Gemini — and the translated upstream body holds the unsigned
    block the strip removes.

    Args:
        protocol: The inbound protocol to speak.
        model: The model the agent asks for.

    Returns:
        The request body for that protocol.
    """
    if protocol is BridgeProtocol.CHAT_COMPLETIONS_API:
        return {
            "model": model,
            "stream": True,
            "messages": [
                {"role": "user", "content": "read a"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "reasoning_content": "I should read the file.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"path": "a"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                {"role": "user", "content": "thanks"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "Read", "description": "Read a file", "parameters": _READ_SCHEMA},
                }
            ],
        }
    if protocol is BridgeProtocol.RESPONSES_API:
        return {
            "model": model,
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "read a"}],
                },
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I should read the file."}]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "thanks"}],
                },
            ],
            "tools": [{"type": "function", "name": "Read", "description": "Read a file", "parameters": _READ_SCHEMA}],
        }
    return {
        "contents": [
            {"role": "user", "parts": [{"text": "read a"}]},
            {
                "role": "model",
                "parts": [{"text": "I should read the file.", "thought": True}, {"text": "Checking."}],
            },
            {"role": "user", "parts": [{"text": "thanks"}]},
        ],
        "tools": [
            {"functionDeclarations": [{"name": "Read", "description": "Read a file", "parameters": _READ_SCHEMA}]}
        ],
    }


def _client_path(protocol: BridgeProtocol, model: str) -> str:
    """Return the URL path an agent of this protocol posts to.

    Args:
        protocol: The inbound protocol.
        model: The model, which Gemini carries in the path.

    Returns:
        The path with any query string.
    """
    if protocol is BridgeProtocol.CHAT_COMPLETIONS_API:
        return "/v1/chat/completions"
    if protocol is BridgeProtocol.RESPONSES_API:
        return "/v1/responses"
    return f"/v1beta/models/{model}:streamGenerateContent?alt=sse"


async def _stream(
    protocol: BridgeProtocol,
    provider,
    model: str,
    upstream_bodies: list[tuple[int, str]],
    monkeypatch=None,
) -> tuple[BridgeServer, int, str, int]:
    """POST a streaming request through a real bridge against scripted upstream responses.

    Args:
        protocol: The inbound protocol to speak.
        provider: The adapter the bridge serves with.
        model: The model the agent asks for.
        upstream_bodies: ``(status, body)`` pairs served in order; a body of
            ``None`` means an empty 200.
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
    server = BridgeServer(_ProtocolLauncher(protocol), provider, "sk-test", host="127.0.0.1", port=0)
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
                    f"http://127.0.0.1:{server.port}{_client_path(protocol, model)}",
                    json=_client_request(protocol, model),
                ) as resp,
            ):
                return server, resp.status, await resp.text(), calls["n"], bodies
        finally:
            await server.stop_async()


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


# ── AC-1: text and a complete tool call reach each client ──────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _MESSAGES_WIRE)
async def test_the_chat_completions_stream_carries_text_thinking_and_the_whole_tool_call(
    provider_factory, model, monkeypatch
):
    """R2 — a CC client receives reasoning, text, the tool call and the finish reason.

    The upstream body's last event is left unterminated so the flush path —
    where an Anthropic stream's terminal events land when the connection
    closes without a final newline — is exercised, not just the main loop.

    Args:
        provider_factory: Builds a Messages-wire adapter.
        model: A model that adapter serves on its Messages wire.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    body = _render_sse(_anthropic_events(), terminate=False)

    server, status, client_body, _calls, _bodies = await _stream(
        BridgeProtocol.CHAT_COMPLETIONS_API, provider_factory(), model, [(200, body)], monkeypatch
    )

    assert status == 200
    events = _parse_data_lines(client_body)
    deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
    assert "".join(d.get("content", "") for d in deltas) == "hello"
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "look first"

    # The tool call: one opened entry, arguments accumulated under one index.
    tool_deltas = _cc_tool_call_fragments(events)
    opened = [tc for tc in tool_deltas if "id" in tc]
    assert opened == [
        {"index": 0, "id": "toolu_1", "type": "function", "function": {"name": "Read", "arguments": ""}}
    ]
    arguments = "".join(tc["function"]["arguments"] for tc in tool_deltas)
    assert json.loads(arguments) == {"path": "a"}

    finishes = [
        e["choices"][0]["finish_reason"] for e in events if e.get("choices") and e["choices"][0]["finish_reason"]
    ]
    assert finishes == ["tool_calls"]
    assert client_body.rstrip().endswith("data: [DONE]")

    # AC-6: the bridge's own attribution reads Chat Completions usage, not Anthropic's.
    served = server._session_stats()["models_served"]
    record = next(iter(served.values()))
    assert record["completions"] == 1
    assert record["input_tokens"] == 5
    assert record["output_tokens"] == 9


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _MESSAGES_WIRE)
async def test_the_responses_stream_carries_text_and_the_whole_function_call(provider_factory, model):
    """R3 — Codex receives its text delta, the function call, and a clean completion last.

    Args:
        provider_factory: Builds a Messages-wire adapter.
        model: A model that adapter serves on its Messages wire.
    """
    _server, status, client_body, _calls, _bodies = await _stream(
        BridgeProtocol.RESPONSES_API, provider_factory(), model, [(200, _render_sse(_anthropic_events()))]
    )

    assert status == 200
    # No Chat Completions artefact may reach a Responses client (review finding 1).
    assert "data: [DONE]" not in client_body
    events = _parse_data_lines(client_body)
    types = [e.get("type") for e in events]

    text = "".join(e.get("delta", "") for e in events if e.get("type") == "response.output_text.delta")
    assert "hello" in text

    added = [
        e
        for e in events
        if e.get("type") == "response.output_item.added" and e.get("item", {}).get("type") == "function_call"
    ]
    assert len(added) == 1
    assert added[0]["item"]["call_id"] == "toolu_1"
    assert added[0]["item"]["name"] == "Read"

    arguments = "".join(
        e.get("delta", "") for e in events if e.get("type") == "response.function_call_arguments.delta"
    )
    assert json.loads(arguments) == {"path": "a"}

    assert types[-1] == "response.completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_factory", "model"), _MESSAGES_WIRE)
async def test_the_gemini_stream_carries_text_and_the_whole_function_call(provider_factory, model):
    """R4 — Gemini CLI receives its text part and the functionCall with parsed args.

    Args:
        provider_factory: Builds a Messages-wire adapter.
        model: A model that adapter serves on its Messages wire.
    """
    _server, status, client_body, _calls, _bodies = await _stream(
        BridgeProtocol.GEMINI_API, provider_factory(), model, [(200, _render_sse(_anthropic_events()))]
    )

    assert status == 200
    assert "data: [DONE]" not in client_body
    events = _parse_data_lines(client_body)
    parts = [p for e in events for c in e.get("candidates", []) for p in c.get("content", {}).get("parts", [])]

    assert "hello" in [p.get("text") for p in parts if "text" in p]
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
    assert calls == [{"name": "Read", "args": {"path": "a"}}]

    # The finish event closes the stream: a finishReason, after everything else.
    assert "finishReason" in events[-1]["candidates"][0]


# ── AC-2: failover and empty detection keep working ────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol",),
    [
        pytest.param(BridgeProtocol.CHAT_COMPLETIONS_API, id="cc"),
        pytest.param(BridgeProtocol.RESPONSES_API, id="responses"),
        pytest.param(BridgeProtocol.GEMINI_API, id="gemini"),
    ],
)
async def test_a_retryable_upstream_failure_still_ends_in_content(protocol, monkeypatch):
    """AC-2 — a pre-emission 500 is retried and the client receives the eventual content.

    Args:
        protocol: The inbound protocol under test.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    provider, model = AnthropicAdapter(), "claude-opus-4-6"
    good = _render_sse(_anthropic_events())

    _server, status, client_body, calls, _bodies = await _stream(
        protocol, provider, model, [(500, "boom"), (200, good)], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("protocol",), [
    pytest.param(BridgeProtocol.RESPONSES_API, id="responses"),
    pytest.param(BridgeProtocol.GEMINI_API, id="gemini"),
])
async def test_an_empty_converted_stream_fires_the_empty_ladder(protocol, monkeypatch):
    """AC-2 — a content-less Anthropic stream retries instead of ending the turn.

    Before the converter, such a stream produced no finish events at all, so
    ``response_was_empty`` was never even evaluated on these routes.

    Args:
        protocol: The inbound protocol under test.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    provider, model = AnthropicAdapter(), "claude-opus-4-6"
    empty = _render_sse([
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 0}},
        {"type": "message_stop"},
    ])
    good = _render_sse(_anthropic_events())

    _server, status, client_body, calls, _bodies = await _stream(
        protocol, provider, model, [(200, empty), (200, good)], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in client_body


@pytest.mark.asyncio
@pytest.mark.parametrize(("protocol",), [
    pytest.param(BridgeProtocol.RESPONSES_API, id="responses"),
    pytest.param(BridgeProtocol.GEMINI_API, id="gemini"),
])
async def test_an_in_stream_error_event_before_emission_still_fails_over(protocol, monkeypatch):
    """AC-2 — an ``error`` event over a 200 no longer ends the turn silently.

    The per-event translator swallowed ``error`` events, so this route's
    in-stream failover had never fired; the converter passes them through for
    exactly the handlers' detection to see.

    Args:
        protocol: The inbound protocol under test.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    provider, model = AnthropicAdapter(), "claude-opus-4-6"
    failure = _render_sse([{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}])
    good = _render_sse(_anthropic_events())

    _server, status, client_body, calls, _bodies = await _stream(
        protocol, provider, model, [(200, failure), (200, good)], monkeypatch
    )

    assert status == 200
    assert calls == 2
    assert "hello" in client_body


@pytest.mark.asyncio
async def test_a_chat_completions_in_stream_error_surfaces_instead_of_a_silent_ok(monkeypatch):
    """AC-2 (CC leg) — a mid-stream ``error`` reaches the client as an error.

    The CC handler's in-stream path cannot retry without a backend pool, so
    the claim here is the one that changed: the error is surfaced, where the
    swallowed-event behaviour used to deliver a truncated success.

    Args:
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    failure = _render_sse([{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}])

    _server, status, client_body, _calls, _bodies = await _stream(
        BridgeProtocol.CHAT_COMPLETIONS_API,
        AnthropicAdapter(),
        "claude-opus-4-6",
        [(200, failure)],
        monkeypatch,
    )

    assert status == 200
    # Pool-less, the handler surfaces its own upstream-error event rather than
    # the upstream's message — either way the turn ends visibly in error,
    # where the swallowed-event behaviour used to deliver a truncated success.
    assert "upstream_error" in client_body
    assert "hello" not in client_body


# ── AC-3: a thinking-signature rejection strips and retries ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol",),
    [
        pytest.param(BridgeProtocol.CHAT_COMPLETIONS_API, id="cc"),
        pytest.param(BridgeProtocol.RESPONSES_API, id="responses"),
        pytest.param(BridgeProtocol.GEMINI_API, id="gemini"),
    ],
)
async def test_a_thinking_signature_rejection_strips_and_retries_the_same_backend(protocol, monkeypatch):
    """AC-3 — the 400 the client's unsigned thinking earns is recovered, not surfaced.

    The transcript each protocol sends carries a prior thinking turn; the
    translated upstream body holds it as an unsigned block, the rejection
    names it, and the strip-and-retry of ``_stream_messages`` now happens here
    too.  The backend is retried, not failed over: the second scripted
    response is served by the same upstream.

    Args:
        protocol: The inbound protocol under test.
        monkeypatch: Pytest fixture, collapses the retry backoff.
    """
    provider, model = AnthropicAdapter(), "claude-opus-4-6"
    good = _render_sse(_anthropic_events())

    _server, status, client_body, calls, bodies = await _stream(
        protocol,
        provider,
        model,
        [(400, json.dumps(_SIGNATURE_REJECTION)), (200, good)],
        monkeypatch,
    )

    assert status == 200
    assert calls == 2
    assert "hello" in client_body

    # The retry carried the strip: the first request's upstream body holds the
    # unsigned thinking block the rejection bites on, the second holds none.
    assert len(bodies) == 2
    assert "thinking" in json.dumps(bodies[0])
    assert "thinking" not in json.dumps(bodies[1])
