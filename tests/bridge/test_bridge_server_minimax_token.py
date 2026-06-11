"""Bridge-level tests for MiniMax Token Plan provider integration.

These tests exercise the full request/response path through ``BridgeServer``
with ``MiniMaxTokenAnthropicAdapter`` to verify the default translated path
(CC→Messages), the opt-in native passthrough, region routing, balancing
failover, and the compaction regression guard for the
``tool call result does not follow tool call (2013)`` error.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import (
    _COMPACTION_CHAR_THRESHOLD,
    BridgeServer,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.types import BridgeProtocol

_GLOBAL_URL = "https://api.minimax.io/anthropic"
_CN_URL = "https://api.minimaxi.com/anthropic"
_MESSAGES_PATH = "/v1/messages"


# ── Stub launcher ──────────────────────────────────────────────────────────


class _FakeLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def binary_name(self) -> str:
        return "fake"

    @property
    def agent_name(self) -> str:
        return "fake"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, model=None):
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])

    def prepare_launch(self, spawn_config):
        pass

    def cleanup_launch(self, spawn_config):
        pass


# ── Server fixtures ─────────────────────────────────────────────────────────


def _make_server(
    *,
    provider_config: dict | None = None,
    native_messages: bool = False,
    backends: list | None = None,
) -> BridgeServer:
    provider = MiniMaxTokenAnthropicAdapter(
        native_messages=native_messages,
        provider_config=provider_config,
    )
    adapter = _FakeLauncher()
    return BridgeServer(
        adapter,
        provider,
        "sk-minimax-test-key",
        host="127.0.0.1",
        port=0,
        provider_config=provider_config,
        backends=backends,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_profile(
    name: str = "minimax",
    provider: str = "minimax_token",
    model: str = "MiniMax-M3",
    provider_config: dict | None = None,
) -> Profile:
    import uuid

    return Profile(
        name=name,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        auth_ref=str(uuid.uuid4()),
        provider_config=provider_config or {},
    )


def _anthropic_message_response(text: str = "Hi there!") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "MiniMax-M3",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


# ── Translated path (default) ──────────────────────────────────────────────


class TestMiniMaxTokenTranslatedPath:
    """Default behavior: the bridge translates CC→Messages via the inherited
    AnthropicAdapter. This is the path that successfully reaches MiniMax in
    the user's logs; the native passthrough path was failing 96% of the time.
    """

    @pytest.mark.asyncio
    async def test_default_adapter_uses_translated_path(self):
        provider = MiniMaxTokenAnthropicAdapter()
        assert provider.use_native_messages is False

    @pytest.mark.asyncio
    async def test_non_streaming_uses_translated_upstream_body(self):
        """The upstream body is the CC→Messages rewrite, not the raw
        Claude Code body. The provider is identified by ``x-api-key``.
        """
        sent_body: dict[str, Any] = {}
        sent_headers: dict[str, str] = {}

        def capture(url, **kwargs):
            sent_body.update(kwargs.get("json", {}))
            sent_headers.update({k.lower(): v for k, v in (kwargs.get("headers") or {}).items()})
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("Hello!")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=capture)
            server = _make_server()
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    ) as resp,
                ):
                    body = await resp.json()
                    assert resp.status == 200, body
                    assert body["content"][0]["text"] == "Hello!"
            finally:
                await server.stop_async()

        # x-api-key auth, anthropic-version, content-type all present
        assert sent_headers.get("x-api-key") == "sk-minimax-test-key"
        assert sent_headers.get("anthropic-version") == "2023-06-01"
        # The translated body is in Anthropic Messages format
        assert sent_body.get("model") == "MiniMax-M3"
        assert sent_body.get("max_tokens") == 4096
        assert sent_body.get("messages") == [{"role": "user", "content": "hi"}]
        # No CC artifacts leaked
        assert "choices" not in sent_body
        assert "stream" not in sent_body or sent_body["stream"] is False

    @pytest.mark.asyncio
    async def test_tool_use_id_round_trip_through_translated_path(self):
        """The default path must preserve ``tool_use_id`` ↔ ``tool_call_id``
        pairing so MiniMax's
        ``tool call result does not follow tool call (2013)`` validation
        does not fire on a broken pairing.
        """
        sent_body: dict[str, Any] = {}

        def capture(url, **kwargs):
            sent_body.update(kwargs.get("json", {}))
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("Read done")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=capture)
            server = _make_server()
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [
                                {"role": "user", "content": "read /etc/hostname"},
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "tool_use",
                                            "id": "toolu_abc",
                                            "name": "read",
                                            "input": {"path": "/etc/hostname"},
                                        }
                                    ],
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": "toolu_abc",
                                            "content": "kitty-host",
                                        }
                                    ],
                                },
                            ],
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
            finally:
                await server.stop_async()

        # In the Anthropic Messages wire format the assistant message has
        # a ``tool_use`` block whose id matches the following user
        # ``tool_result`` block's ``tool_use_id``.
        assert sent_body["messages"][1]["role"] == "assistant"
        tu = [b for b in sent_body["messages"][1]["content"] if b.get("type") == "tool_use"][0]
        assert tu["id"] == "toolu_abc"
        tr = [b for b in sent_body["messages"][2]["content"] if b.get("type") == "tool_result"][0]
        assert tr["tool_use_id"] == "toolu_abc"

    @pytest.mark.asyncio
    async def test_effort_parameter_preserved_through_translated_path(self):
        """Claude Code sends ``effort`` (e.g. "low", "medium", "high",
        "xhigh") alongside ``thinking``. The translated path must pass
        the effort value through to the Anthropic-compatible upstream so
        the provider can honour the user's reasoning-effort configuration.
        """
        sent_body: dict[str, Any] = {}

        def capture(url, **kwargs):
            sent_body.update(kwargs.get("json", {}))
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("done")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=capture)
            server = _make_server()
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [{"role": "user", "content": "think hard"}],
                            "effort": "xhigh",
                            "thinking": {"type": "adaptive"},
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
            finally:
                await server.stop_async()

        # effort must reach the upstream unchanged
        assert sent_body.get("effort") == "xhigh", (
            f"Expected effort='xhigh' in upstream body, got {sent_body.get('effort')!r}"
        )

    @pytest.mark.asyncio
    async def test_thinking_adaptive_preserved_through_translated_path(self):
        """Claude Code sends ``thinking: {type: "adaptive"}``. The
        translated path must forward this to the Anthropic-compatible
        upstream. This is distinct from the explicit budget_tokens
        thinking mode.
        """
        sent_body: dict[str, Any] = {}

        def capture(url, **kwargs):
            sent_body.update(kwargs.get("json", {}))
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("done")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=capture)
            server = _make_server()
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [{"role": "user", "content": "think"}],
                            "thinking": {"type": "adaptive"},
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
            finally:
                await server.stop_async()

        # thinking: {type: "adaptive"} must reach the upstream
        assert sent_body.get("thinking") == {"type": "adaptive"}, (
            f"Expected thinking={{type: 'adaptive'}}, got {sent_body.get('thinking')!r}"
        )

    """Region selection via ``provider_config["region"]`` routes to the
    correct upstream base URL.
    """

    def test_global_default(self):
        server = _make_server()
        assert server._build_upstream_url() == f"{_GLOBAL_URL}{_MESSAGES_PATH}"

    def test_cn_region_routes_to_cn_host(self):
        server = _make_server(provider_config={"region": "cn"})
        assert server._build_upstream_url() == f"{_CN_URL}{_MESSAGES_PATH}"

    def test_explicit_global_region_uses_global_host(self):
        server = _make_server(provider_config={"region": "global"})
        assert server._build_upstream_url() == f"{_GLOBAL_URL}{_MESSAGES_PATH}"


# ── Opt-in native passthrough ──────────────────────────────────────────────


class TestMiniMaxTokenNativeOptIn:
    """The ``native_messages`` opt-in flag forwards the raw Claude Code
    body to the upstream without CC→Messages translation. This is the
    escape hatch for users on endpoints where the native passthrough
    works.
    """

    @pytest.mark.asyncio
    async def test_native_passthrough_forwards_raw_body(self):
        sent_body: dict[str, Any] = {}

        def capture(url, **kwargs):
            sent_body.update(kwargs.get("json", {}))
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("native ok")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=capture)
            server = _make_server(native_messages=True)
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [{"role": "user", "content": "hi"}],
                            # Fields the translator would strip — preserved
                            # under native passthrough
                            "context_management": {"edits": []},
                            "output_config": {"format": {"type": "text"}},
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
            finally:
                await server.stop_async()

        # The body is forwarded verbatim — no CC→Messages rewrite
        assert sent_body.get("context_management") == {"edits": []}
        assert sent_body.get("output_config") == {"format": {"type": "text"}}

    def test_opt_in_via_provider_config(self):
        server = _make_server(provider_config={"native_messages": True})
        assert server._active_provider.use_native_messages is True

    def test_no_opt_in_uses_translated(self):
        server = _make_server(provider_config={"region": "cn"})
        assert server._active_provider.use_native_messages is False


# ── Compaction regression guard ─────────────────────────────────────────────


class TestMiniMaxTokenCompactionPreservesToolPairing:
    """The compaction grouping logic in ``_compact_messages`` is
    Chat-Completions-format aware. The translated path produces a CC
    ``messages`` list whose ``assistant(tool_calls) + tool(result)`` is
    grouped as an atomic block, so the pruner cannot orphan a
    ``tool_result`` from its ``tool_use``. The native passthrough path
    produces an Anthropic-native ``messages`` list where the pairing is
    missed and a 2013 tool-call validation error surfaces. This test
    exercises the translated path's behavior under compaction.
    """

    @pytest.mark.asyncio
    async def test_compaction_preserves_tool_use_result_pairing(self, caplog):
        """Force compaction to run by sending a request over the
        compaction threshold. After compaction, the assistant(tool_use)
        and following tool(result) must still be adjacent in the
        translated body that reaches the upstream.
        """
        import logging

        sent_body: dict[str, Any] = {}

        def capture(url, **kwargs):
            sent_body.update(kwargs.get("json", {}))
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("compacted ok")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=capture)
            # Use a model whose context-derived max_chars is large enough
            # that the request itself triggers compaction via the
            # absolute cap.
            server = _make_server()
            await server.start_async()
            try:
                import aiohttp

                # Build a payload large enough to trigger compaction.
                # The request must be > _COMPACTION_CHAR_THRESHOLD (2.8M)
                # but < _MAX_REQUEST_CHARS (4M) so compaction (not size
                # check) kicks in.
                large_user_content = "x" * (_COMPACTION_CHAR_THRESHOLD + 100_000)
                messages = [
                    {"role": "user", "content": large_user_content},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_pair",
                                "name": "read",
                                "input": {"path": "/x"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_pair",
                                "content": "result",
                            }
                        ],
                    },
                ]
                body = {
                    "model": "MiniMax-M3",
                    "max_tokens": 4096,
                    "messages": messages,
                }
                # Sanity: payload is over the compaction threshold
                raw_size = len(json.dumps(body, ensure_ascii=False))
                assert raw_size > _COMPACTION_CHAR_THRESHOLD, (
                    f"Payload {raw_size} must exceed compaction threshold "
                    f"{_COMPACTION_CHAR_THRESHOLD} to exercise compaction"
                )

                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://127.0.0.1:{server.port}/v1/messages",
                            json=body,
                        ) as resp:
                            assert resp.status == 200
            finally:
                await server.stop_async()

        # The translated body received by the upstream must not have
        # orphaned the tool_result. If the assistant(tool_use) and
        # following user(tool_result) are present, their pairing is
        # intact. We check by looking for the tool_use_id on both ends.
        upstream_messages = sent_body.get("messages", [])
        # Find any tool_use blocks and the tool_result blocks
        tool_use_ids: set[str] = set()
        tool_result_ids: set[str] = set()
        for m in upstream_messages:
            if m.get("role") == "assistant":
                for b in m.get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tool_use_ids.add(b["id"])
            if m.get("role") == "user":
                for b in m.get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tool_result_ids.add(b["tool_use_id"])
        # All tool_result ids must have a matching tool_use in the body
        assert tool_result_ids.issubset(tool_use_ids), (
            f"Compaction orphaned tool_result(s) {tool_result_ids - tool_use_ids}"
        )


# ── Failover in balancing profile ──────────────────────────────────────────


class TestMiniMaxTokenFailover:
    """When the MiniMax backend returns a 400-class error, the balancing
    profile must try the next healthy backend.
    """

    @pytest.mark.asyncio
    async def test_failover_on_400_to_next_backend(self):
        """MiniMax returns 400, the second backend returns 200. The
        bridge surfaces the second backend's response.
        """
        import random as _random

        from kitty.providers.zai_anthropic import ZaiAnthropicAdapter

        minimax = MiniMaxTokenAnthropicAdapter()
        zai = ZaiAnthropicAdapter()
        zai_profile = _make_profile(name="zai", provider="zai_coding", model="glm-4.6")
        backends = [
            (minimax, "sk-minimax", _make_profile()),
            (zai, "sk-zai", zai_profile),
        ]
        adapter = _FakeLauncher()
        server = BridgeServer(
            adapter,
            minimax,
            "sk-minimax",
            host="127.0.0.1",
            port=0,
            provider_config={},
            backends=backends,
        )
        _random.seed(1)  # deterministic: select first backend (minimax) first

        minimax_calls = 0
        zai_calls = 0

        def minimax_400(url, **kwargs):
            nonlocal minimax_calls
            minimax_calls += 1
            return CallbackResult(
                status=400,
                content_type="application/json",
                body=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "invalid params, context window exceeds limit (2013)",
                        },
                    }
                ),
            )

        def zai_200(url, **kwargs):
            nonlocal zai_calls
            zai_calls += 1
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("from zai")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=minimax_400)
            m.post("https://api.z.ai/api/anthropic/v1/messages", callback=zai_200)
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    ) as resp,
                ):
                    body = await resp.json()
                    assert resp.status == 200, body
                    assert body["content"][0]["text"] == "from zai"
            finally:
                await server.stop_async()

        assert minimax_calls == 1
        assert zai_calls == 1

    @pytest.mark.asyncio
    async def test_failover_on_2013_tool_call_validation(self):
        """MiniMax returns the new 2013 "tool call result does not follow
        tool call" error. The bridge must mark MiniMax unhealthy and
        try the next backend in a balancing profile.
        """
        import random as _random

        from kitty.providers.zai_anthropic import ZaiAnthropicAdapter

        minimax = MiniMaxTokenAnthropicAdapter()
        zai = ZaiAnthropicAdapter()
        zai_profile = _make_profile(name="zai", provider="zai_coding", model="glm-4.6")
        backends = [
            (minimax, "sk-minimax", _make_profile()),
            (zai, "sk-zai", zai_profile),
        ]
        adapter = _FakeLauncher()
        server = BridgeServer(
            adapter,
            minimax,
            "sk-minimax",
            host="127.0.0.1",
            port=0,
            provider_config={},
            backends=backends,
        )
        _random.seed(1)  # deterministic: select minimax first

        minimax_calls = 0
        zai_calls = 0

        def minimax_400(url, **kwargs):
            nonlocal minimax_calls
            minimax_calls += 1
            return CallbackResult(
                status=400,
                content_type="application/json",
                body=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "invalid params, tool call result does not follow tool call (2013)",
                        },
                    }
                ),
            )

        def zai_200(url, **kwargs):
            nonlocal zai_calls
            zai_calls += 1
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(_anthropic_message_response("recovered via zai")),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(f"{_GLOBAL_URL}{_MESSAGES_PATH}", callback=minimax_400)
            m.post("https://api.z.ai/api/anthropic/v1/messages", callback=zai_200)
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "MiniMax-M3",
                            "max_tokens": 4096,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    ) as resp,
                ):
                    body = await resp.json()
                    assert resp.status == 200, body
                    assert body["content"][0]["text"] == "recovered via zai"
            finally:
                await server.stop_async()

        assert minimax_calls == 1
        assert zai_calls == 1
