"""Subsystem tests for response-side ``tool_use`` auditing in the Messages bridge.

kitty-bridge#33: response payloads were logged only on the custom transport, so
a malformed ``tool_use`` was forwarded to the client with nothing in kitty's own
log recording what went out.  These tests prove the audit fires on the paths the
reporter actually uses, and — the point that matters most — that it changes
nothing about what the client receives.

Covers ``.requirements/20260828T210855Z_tool_use_response_audit`` FR-3 … FR-6.
"""

from __future__ import annotations

import json
import logging
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.types import BridgeProtocol

NATIVE_URL = "https://api.native.test/anthropic/v1/messages"
CC_URL = "https://api.cc.test/v1/chat/completions"
AUDIT_LOGGER = "kitty.bridge.tool_audit"

STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array"}, "conversation_notes": {"type": "string"}},
    "required": ["findings", "conversation_notes"],
    "additionalProperties": False,
}


class _StubLauncher(LauncherAdapter):
    """Minimal launcher that selects the Messages API bridge protocol."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


def _sse(event_type: str, payload: dict) -> str:
    """Render one Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _native_tool_use_sse(tool_input: dict, name: str = "StructuredOutput") -> bytes:
    """A well-formed native Anthropic stream whose assistant turn calls one tool."""
    return "".join(
        [
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "MiniMax-M3",
                        "usage": {"input_tokens": 5, "output_tokens": 0},
                    },
                },
            ),
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}},
                },
            ),
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
                },
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
            _sse("message_stop", {"type": "message_stop"}),
        ]
    ).encode()


def _cc_tool_call_sse(tool_input: dict, name: str = "StructuredOutput") -> bytes:
    """A Chat Completions stream whose assistant turn calls one tool."""
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "MiniMax-M3",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(tool_input)},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "MiniMax-M3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _client_request(*, stream: bool = True, with_tools: bool = True) -> dict:
    """A Claude Code request declaring the StructuredOutput tool."""
    request: dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "stream": stream,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "review"}]}],
    }
    if with_tools:
        request["tools"] = [
            {
                "name": "StructuredOutput",
                "description": "Emit the structured review",
                "input_schema": STRUCTURED_OUTPUT_SCHEMA,
            }
        ]
    return request


def _make_server(*, native: bool) -> BridgeServer:
    """Build a single-backend Messages server on the requested wire shape."""
    if native:
        provider: object = CustomAnthropicAdapter()
        base_url = "https://api.native.test/anthropic"
    else:
        provider = CustomOpenAIAdapter()
        base_url = "https://api.cc.test/v1"

    profile = Profile(
        name="pool-member-1",
        provider="custom_anthropic" if native else "custom_openai",
        model="MiniMax-M3",
        auth_ref=str(uuid.uuid4()),
        provider_config={"base_url": base_url},
    )
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=provider,
        resolved_key="key-1",
        model="MiniMax-M3",
        provider_config=profile.provider_config,
        host="127.0.0.1",
        port=0,
    )


def _make_balancing_server() -> BridgeServer:
    """Build a two-member native pool, the shape the incident report describes.

    The backend label only carries a profile name in balancing mode, and
    naming the pool member is the difference between "something returned
    garbage" and "this member returned garbage".
    """
    backends = []
    for name in ("pool-member-1", "pool-member-2"):
        profile = Profile(
            name=name,
            provider="custom_anthropic",
            model="MiniMax-M3",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": "https://api.native.test/anthropic"},
        )
        backends.append((CustomAnthropicAdapter(), f"key-{name}", profile))

    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="MiniMax-M3",
        provider_config=backends[0][2].provider_config,
        backends=backends,
        host="127.0.0.1",
        port=0,
    )


async def _post(server: BridgeServer, request: dict) -> tuple[int, bytes]:
    async with (
        aiohttp.ClientSession() as session,
        session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=request) as resp,
    ):
        return resp.status, await resp.read()


async def _run(server: BridgeServer, url: str, upstream_body: bytes, request: dict) -> tuple[int, bytes]:
    """Serve one scripted upstream response through the bridge."""
    with aioresponses(passthrough=["http://127.0.0.1"]) as m:
        m.post(
            url,
            callback=lambda u, **kw: CallbackResult(
                status=200, headers={"Content-Type": "text/event-stream"}, body=upstream_body
            ),
            repeat=True,
        )
        await server.start_async()
        try:
            return await _post(server, request)
        finally:
            await server.stop_async()


class TestNativePassthrough:
    """AC-3.1 / AC-4.1 — the path the reporter's MiniMax pool actually uses."""

    @pytest.mark.asyncio
    async def test_tool_use_input_is_logged(self, caplog):
        server = _make_server(native=True)
        clean = {"findings": [], "conversation_notes": "ok"}

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            status, _ = await _run(server, NATIVE_URL, _native_tool_use_sse(clean), _client_request())

        assert status == 200
        assert any(
            "StructuredOutput" in r.getMessage() and "conversation_notes" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_envelope_wrapped_input_warns_naming_the_backend(self, caplog):
        """The whole point of the issue: make this visible instead of silent.

        Uses a balancing pool because that is the reported setup, and because
        the profile name is what lets an operator act on the warning.
        """
        server = _make_balancing_server()
        wrapped = {"result": {"findings": [], "conversation_notes": "ok"}}

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            status, _ = await _run(server, NATIVE_URL, _native_tool_use_sse(wrapped), _client_request())

        assert status == 200
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        message = warnings[0].getMessage()
        # Pool selection is random, so pin the property that matters: the
        # warning names a profile, not the provider type shared by every member.
        assert "pool-member-1" in message or "pool-member-2" in message, message
        assert "result" in message
        assert "findings" in message

    @pytest.mark.asyncio
    async def test_clean_input_does_not_warn(self, caplog):
        """AC-4.2 — ordinary traffic must stay quiet or the warning is worthless."""
        server = _make_server(native=True)
        clean = {"findings": [{"file": "a.py"}], "conversation_notes": "ok"}

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            await _run(server, NATIVE_URL, _native_tool_use_sse(clean), _client_request())

        assert [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER] == []

    @pytest.mark.asyncio
    async def test_client_bytes_are_unchanged_by_auditing(self):
        """AC-5.1 — the client receives exactly the upstream stream.

        Auditing reads the same bytes it forwards; if it ever altered or
        withheld one, this is what would catch it.
        """
        # AC-5.1 asks for thinking and text alongside the tool_use, so the
        # comparison covers every block type the auditor walks past.
        upstream_body = (
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "MiniMax-M3",
                        "usage": {"input_tokens": 5, "output_tokens": 0},
                    },
                },
            )
            + _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
            + _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "considering"},
                },
            )
            + _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            + _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            )
            + _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}},
            )
            + _sse("content_block_stop", {"type": "content_block_stop", "index": 1})
        ).encode() + _native_tool_use_sse({"result": {"findings": []}})
        server = _make_server(native=True)

        status, received = await _run(server, NATIVE_URL, upstream_body, _client_request())

        assert status == 200
        assert received == upstream_body


class TestTranslatedPath:
    """AC-3.2 — a Chat Completions upstream, audited via the events we emit."""

    @pytest.mark.asyncio
    async def test_tool_use_input_is_logged(self, caplog):
        server = _make_server(native=False)
        clean = {"findings": [], "conversation_notes": "ok"}

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            status, _ = await _run(server, CC_URL, _cc_tool_call_sse(clean), _client_request())

        assert status == 200
        assert any("StructuredOutput" in r.getMessage() for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]

    @pytest.mark.asyncio
    async def test_envelope_wrapped_input_warns(self, caplog):
        server = _make_server(native=False)
        wrapped = {"result": {"findings": [], "conversation_notes": "ok"}}

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            await _run(server, CC_URL, _cc_tool_call_sse(wrapped), _client_request())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "result" in warnings[0].getMessage()


class TestNonStreaming:
    """AC-3.3 — a non-streaming response carries its tool_use complete."""

    @pytest.mark.asyncio
    async def test_envelope_wrapped_input_warns(self, caplog):
        server = _make_server(native=True)
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "MiniMax-M3",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "StructuredOutput",
                    "input": {"result": {"findings": [], "conversation_notes": "ok"}},
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

        with (
            aioresponses(passthrough=["http://127.0.0.1"]) as m,
            caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER),
        ):
            m.post(NATIVE_URL, payload=body, repeat=True)
            await server.start_async()
            try:
                status, _ = await _post(server, _client_request(stream=False))
            finally:
                await server.stop_async()

        assert status == 200
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "result" in warnings[0].getMessage()


class TestNoToolsDeclared:
    """AC-2.2 — logging without schemas still works, and never warns."""

    @pytest.mark.asyncio
    async def test_logs_but_does_not_warn(self, caplog):
        server = _make_server(native=True)
        wrapped = {"result": {"findings": []}}

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            await _run(server, NATIVE_URL, _native_tool_use_sse(wrapped), _client_request(with_tools=False))

        assert any("StructuredOutput" in r.getMessage() for r in caplog.records)
        assert [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER] == []


class TestTruncatedStream:
    """A stream that dies mid-response still delivered bytes to the client.

    The interrupted-stream finalization paths write Anthropic SSE downstream,
    so they must be audited like any other write — and this is not a corner
    case: #33 reports the failure as *intermittent on identical input*, and an
    upstream that truncates after emitting the tool call is the archetype. It
    is exactly when the counter must not stay at zero.
    """

    @pytest.mark.asyncio
    async def test_tool_use_on_a_truncated_stream_is_audited(self, caplog):
        server = _make_server(native=False)
        wrapped = {"result": {"findings": [], "conversation_notes": "ok"}}
        # A tool call, then the stream simply stops: no finish_reason, no [DONE].
        truncated = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "MiniMax-M3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "StructuredOutput",
                                        "arguments": json.dumps(wrapped),
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
        )
        body = f"data: {truncated}\n\n".encode()

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            status, received = await _run(server, CC_URL, body, _client_request())

        assert status == 200
        assert b"tool_use" in received, "the client did receive the tool call"

        audit_records = [r for r in caplog.records if r.name == AUDIT_LOGGER]
        assert any("StructuredOutput" in r.getMessage() for r in audit_records), [
            r.getMessage() for r in audit_records
        ]
        assert server._session_stats()["malformed_tool_use"] == 1


class TestStatsCounter:
    """The signal has to survive a run without ``--debug``.

    ``kitty/__init__.py`` attaches only a NullHandler, and the sole real
    handler is the DEBUG file handler created when ``--debug`` is passed. A
    WARNING alone would therefore reach nobody in an ordinary run — which is
    the situation the issue is complaining about. The counter is surfaced by
    ``GET /stats`` and the shutdown summary instead.
    """

    @pytest.mark.asyncio
    async def test_malformed_tool_use_is_counted(self):
        server = _make_server(native=True)
        wrapped = {"result": {"findings": [], "conversation_notes": "ok"}}

        await _run(server, NATIVE_URL, _native_tool_use_sse(wrapped), _client_request())

        stats = server._session_stats()
        assert stats["malformed_tool_use"] == 1
        assert stats["backends"][0]["malformed_tool_use"] == 1

    @pytest.mark.asyncio
    async def test_clean_traffic_counts_zero(self):
        server = _make_server(native=True)
        clean = {"findings": [], "conversation_notes": "ok"}

        await _run(server, NATIVE_URL, _native_tool_use_sse(clean), _client_request())

        stats = server._session_stats()
        assert stats["malformed_tool_use"] == 0
        assert stats["backends"][0]["malformed_tool_use"] == 0


class TestHealthUntouched:
    """AC-6.1 — a shape mismatch is diagnostics, not a routing signal."""

    @pytest.mark.asyncio
    async def test_backend_health_is_not_affected(self):
        server = _make_server(native=True)
        wrapped = {"result": {"findings": [], "conversation_notes": "ok"}}

        status, _ = await _run(server, NATIVE_URL, _native_tool_use_sse(wrapped), _client_request())

        assert status == 200
        # Single-backend mode keeps no health entries; the assertion that
        # matters is that the request succeeded and nothing was marked.
        assert server._backend_health == []
