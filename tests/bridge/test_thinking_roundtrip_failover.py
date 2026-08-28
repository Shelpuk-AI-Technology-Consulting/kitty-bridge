"""Subsystem tests for the thinking round-trip repair inside ``_stream_messages``.

Reproduces kitty-bridge#32: a mid-conversation failover lands on a backend
whose thinking mode requires every assistant turn of the replayed transcript to
carry its reasoning.  The transcript was authored by the previous provider and
carries none, so the receiving backend answers 400.  The bridge must repair and
retry that backend instead of surfacing the error and taking the backend out of
rotation.

Covers ``.requirements/20260828T153428Z_thinking_roundtrip_failover_repair``
FR-3 … FR-7.
"""

from __future__ import annotations

import copy
import json
import uuid

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.types import BridgeProtocol

PRIMARY_URL = "https://api.primary.test/anthropic/v1/messages"
SIBLING_URL = "https://api.sibling.test/anthropic/v1/messages"

THINKING_ROUNDTRIP_400 = (
    '{"error":{"message":"The `content[].thinking` in the thinking mode must be '
    'passed back to the API.","type":"invalid_request_error","param":null,'
    '"code":"invalid_request_error"}}'
)


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


def _anthropic_sse() -> bytes:
    """A minimal, well-formed Anthropic Messages SSE stream."""
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_ok",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "deepseek-v4-flash",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Recovered"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events)


def _client_request() -> dict:
    """A Claude Code transcript whose assistant turns carry no thinking block."""
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "system": [{"type": "text", "text": "You are a reviewer."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Review this diff"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading the file."},
                    {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "a.py"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "print(1)"}]},
        ],
        "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}],
    }


def _make_server() -> BridgeServer:
    """Build a two-backend balancing server of native-passthrough providers.

    Both members speak the Anthropic Messages API, which is the reported
    pool's shape: the family boundary the incident crosses is invisible to
    ``provider_type``.
    """
    backends = []
    for name, base_url in (
        ("primary", "https://api.primary.test/anthropic"),
        ("sibling", "https://api.sibling.test/anthropic"),
    ):
        provider = CustomAnthropicAdapter()
        profile = Profile(
            name=name,
            provider="custom_anthropic",
            model="deepseek-v4-flash",
            auth_ref=str(uuid.uuid4()),
            provider_config={"base_url": base_url},
        )
        backends.append((provider, f"key-{name}", profile))

    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="deepseek-v4-flash",
        provider_config=backends[0][2].provider_config,
        backends=backends,
        host="127.0.0.1",
        port=0,
    )


@pytest.fixture
def prefer_first_backend(monkeypatch: pytest.MonkeyPatch):
    """Make backend selection deterministic: always take the first candidate.

    Selection is uniform-random over the healthy tier, so a test could not
    otherwise tell a same-backend retry from a failover.  Taking the first
    candidate means the sibling is chosen if and only if the primary was
    actually marked unhealthy.
    """
    import kitty.bridge.server as server_module

    monkeypatch.setattr(server_module.random, "choices", lambda population, weights=None, k=1: [population[0]])
    monkeypatch.setattr(server_module.random, "choice", lambda population: population[0])


class _ScriptedUpstream:
    """Records every request body and replies from a scripted sequence.

    The body is deep-copied on arrival.  The repair rewrites the outgoing body
    in place after it has been sent, so recording the reference would let a
    later attempt retroactively rewrite what an earlier attempt is recorded as
    having sent — and every "attempt 1 was unrepaired" assertion would pass or
    fail for the wrong reason.
    """

    def __init__(self, *responses: CallbackResult) -> None:
        self._responses = list(responses)
        self.bodies: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, url, **kwargs) -> CallbackResult:
        self.bodies.append(copy.deepcopy(kwargs.get("json", {})))
        self.urls.append(str(url))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _rejection() -> CallbackResult:
    return CallbackResult(status=400, content_type="application/json", body=THINKING_ROUNDTRIP_400)


def _stream_ok() -> CallbackResult:
    return CallbackResult(status=200, headers={"Content-Type": "text/event-stream"}, body=_anthropic_sse())


async def _post(server: BridgeServer, request: dict) -> tuple[int, str]:
    async with (
        aiohttp.ClientSession() as session,
        session.post(f"http://127.0.0.1:{server.port}/v1/messages", json=request) as resp,
    ):
        return resp.status, (await resp.read()).decode("utf-8")


class TestRepairAndRetrySameBackend:
    """FR-3 / FR-4 — the session survives and the backend keeps its health."""

    @pytest.mark.asyncio
    async def test_client_receives_the_stream_not_the_400(self, prefer_first_backend):
        """AC-3.1 — one rejection no longer kills the session.

        The sibling is scripted to reject too, so a 200 can only come from the
        repaired retry on the primary.  Sharing one scripted upstream between
        both backends would let a plain failover satisfy this test.
        """
        server = _make_server()
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())
        sibling = _ScriptedUpstream(_rejection())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)
            m.post(SIBLING_URL, callback=sibling, repeat=True)

            await server.start_async()
            try:
                status, body = await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert "Recovered" in body
        assert "invalid_request_error" not in body

    @pytest.mark.asyncio
    async def test_retry_body_carries_the_thinking_carrier(self, prefer_first_backend):
        """AC-3.2 — the retried transcript satisfies the contract."""
        server = _make_server()
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)
            m.post(SIBLING_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert len(upstream.bodies) == 2
        first_assistants = [m for m in upstream.bodies[0]["messages"] if m["role"] == "assistant"]
        assert all(
            not any(b.get("type") == "thinking" for b in msg["content"]) for msg in first_assistants
        ), "the first attempt must reproduce the unrepaired transcript"

        retry_assistants = [m for m in upstream.bodies[1]["messages"] if m["role"] == "assistant"]
        assert retry_assistants
        for msg in retry_assistants:
            assert msg["content"][0] == {"type": "thinking", "thinking": ""}

    @pytest.mark.asyncio
    async def test_retry_goes_to_the_same_backend(self, prefer_first_backend):
        """AC-3.3 / AC-4.2 — a repair is not a failover."""
        server = _make_server()
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)
            m.post(SIBLING_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert upstream.urls == [PRIMARY_URL, PRIMARY_URL]

    @pytest.mark.asyncio
    async def test_backend_health_is_untouched(self, prefer_first_backend):
        """AC-4.1 — a request the bridge malformed must not cost the backend 300s."""
        server = _make_server()
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)
            m.post(SIBLING_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                await _post(server, _client_request())
            finally:
                await server.stop_async()

        health = server._backend_health[0]
        assert health["healthy"] is True
        assert health["failed_at"] is None
        assert health["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_repair_is_logged_once(self, prefer_first_backend, caplog):
        """AC-3.4 — the compatibility miss is visible to an operator."""
        import logging

        server = _make_server()
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)
            m.post(SIBLING_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                with caplog.at_level(logging.WARNING, logger="kitty.bridge.server"):
                    await _post(server, _client_request())
            finally:
                await server.stop_async()

        matches = [r for r in caplog.records if "thinking round-trip" in r.getMessage()]
        assert len(matches) == 1, [r.getMessage() for r in caplog.records]


class TestStickyRepair:
    """FR-5 — later turns do not pay for the same discovery again."""

    @pytest.mark.asyncio
    async def test_second_request_is_repaired_before_the_first_attempt(self, prefer_first_backend):
        """AC-5.1 — the backend is remembered for the session."""
        server = _make_server()
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)
            m.post(SIBLING_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                await _post(server, _client_request())
                before = len(upstream.bodies)
                status, body = await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert len(upstream.bodies) == before + 1, "the second turn must not need a rejected attempt"

        assistants = [m for m in upstream.bodies[-1]["messages"] if m["role"] == "assistant"]
        for msg in assistants:
            assert msg["content"][0] == {"type": "thinking", "thinking": ""}


class TestRepairIsPerBackend:
    """The carrier binds to the backend that asked for it, and nothing else."""

    @pytest.mark.asyncio
    async def test_repair_does_not_travel_to_the_sibling_on_failover(self, prefer_first_backend):
        """A fabricated carrier must not be handed to a backend that never asked.

        The repair is written to the outgoing body, not to ``cc_request``, so a
        later failover re-serializes a clean transcript.  Anthropic's contract
        is that a passed-back ``thinking`` block is complete, unmodified and
        signature-verified, so shipping an empty one to a sibling risks turning
        a recoverable failover into a second 400 — charged to that sibling's
        health.
        """
        server = _make_server()
        # Primary: thinking-400, then a hard 500 on the repaired retry, which
        # is a genuine backend fault and does fail over.
        primary = _ScriptedUpstream(
            _rejection(),
            CallbackResult(status=500, content_type="application/json", body='{"error":"boom"}'),
        )
        sibling = _ScriptedUpstream(_stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=primary, repeat=True)
            m.post(SIBLING_URL, callback=sibling, repeat=True)

            await server.start_async()
            try:
                status, body = await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert len(primary.bodies) == 2, "primary should see the original and the repaired transcript"
        assert primary.bodies[1]["messages"][1]["content"][0] == {"type": "thinking", "thinking": ""}

        assert len(sibling.bodies) == 1
        sibling_assistants = [m for m in sibling.bodies[0]["messages"] if m["role"] == "assistant"]
        assert sibling_assistants
        for msg in sibling_assistants:
            assert not any(b.get("type") == "thinking" for b in msg["content"]), (
                "the sibling must receive the client's own transcript, not the carrier "
                "fabricated for the primary"
            )

    @pytest.mark.asyncio
    async def test_sticky_repair_does_not_leak_to_the_sibling(self, prefer_first_backend):
        """AC-5.2 — the sticky set is keyed by backend, not by server.

        The primary rejects and is repaired; the sibling, which never rejected,
        must still receive the client's own transcript on a later turn.
        """
        server = _make_server()
        primary = _ScriptedUpstream(_rejection(), _stream_ok())
        sibling = _ScriptedUpstream(_stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=primary, repeat=True)
            m.post(SIBLING_URL, callback=sibling, repeat=True)

            await server.start_async()
            try:
                await _post(server, _client_request())
                # Take the primary out of rotation so the next turn lands on
                # the sibling, which is not in the sticky set.
                server._mark_backend_unhealthy(0, failure_kind="hard")
                status, body = await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert len(sibling.bodies) == 1
        sibling_assistants = [m for m in sibling.bodies[0]["messages"] if m["role"] == "assistant"]
        assert sibling_assistants
        for msg in sibling_assistants:
            assert not any(b.get("type") == "thinking" for b in msg["content"]), (
                "a backend that never rejected must not inherit the primary's carrier"
            )

    def test_repair_uses_the_wire_dialect_not_the_request_flag(self):
        """The carrier must match the body the adapter actually emits.

        ``CustomAnthropicAdapter`` emits an Anthropic Messages body on **both**
        branches of ``translate_to_upstream`` — natively when
        ``_native_messages_request`` is set, and via the inherited
        CC → Messages translation when the ``tool_use`` format fallback has
        cleared it (``server.py`` sets the flag to False and re-serializes).
        Deriving the dialect from the request flag would write the Chat
        Completions carrier onto an Anthropic-shaped message, so the sticky
        repair would silently malform every later turn on that path.
        """
        server = _make_server()
        server._thinking_repair_backends.add(server._current_backend_idx)

        # Exactly the state the tool_use fallback leaves behind: an
        # Anthropic-wire adapter with the native flag cleared, carrying a
        # Chat-Completions-shaped transcript.
        cc_request = {
            "model": "deepseek-v4-flash",
            "max_tokens": 4096,
            "_native_messages_request": False,
            "messages": [
                {"role": "user", "content": "Review this diff"},
                {"role": "assistant", "content": "Reading the file."},
            ],
        }
        upstream_body = server._upstream_body_for(cc_request)

        assistants = [m for m in upstream_body["messages"] if m["role"] == "assistant"]
        assert assistants, upstream_body
        for msg in assistants:
            assert "reasoning_content" not in msg, (
                "an Anthropic-shaped body must never carry the Chat Completions carrier"
            )
            assert any(b.get("type") == "thinking" for b in msg["content"])

    @pytest.mark.asyncio
    async def test_single_backend_mode_is_covered(self):
        """FR-5 — non-balancing mode is keyed by index -1, not skipped."""
        provider = CustomAnthropicAdapter()
        server = BridgeServer(
            adapter=_StubLauncher(),
            provider=provider,
            resolved_key="key-solo",
            model="deepseek-v4-flash",
            provider_config={"base_url": "https://api.primary.test/anthropic"},
            host="127.0.0.1",
            port=0,
        )
        upstream = _ScriptedUpstream(_rejection(), _stream_ok())

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                await _post(server, _client_request())
                before = len(upstream.bodies)
                status, body = await _post(server, _client_request())
            finally:
                await server.stop_async()

        assert status == 200, body
        assert len(upstream.bodies) == before + 1, "the second turn must not need a rejected attempt"
        assistants = [m for m in upstream.bodies[-1]["messages"] if m["role"] == "assistant"]
        assert assistants
        for msg in assistants:
            assert msg["content"][0] == {"type": "thinking", "thinking": ""}


class TestUnaffectedPaths:
    """FR-6 / FR-7 — everything else behaves exactly as before."""

    @pytest.mark.asyncio
    async def test_success_path_body_is_unchanged(self, prefer_first_backend):
        """AC-7.2 — a backend that never rejects receives the client's own body."""
        server = _make_server()
        upstream = _ScriptedUpstream(_stream_ok())
        request = _client_request()

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=upstream, repeat=True)

            await server.start_async()
            try:
                status, _ = await _post(server, request)
            finally:
                await server.stop_async()

        assert status == 200
        assert len(upstream.bodies) == 1
        assert upstream.bodies[0]["messages"] == request["messages"]

    @pytest.mark.asyncio
    async def test_already_repaired_rejection_falls_back_to_failover(self, prefer_first_backend):
        """AC-6.1 — when the repair cannot help, today's behaviour is unchanged."""
        server = _make_server()
        primary = _ScriptedUpstream(_rejection())
        sibling = _ScriptedUpstream(_stream_ok())

        request = _client_request()
        for msg in request["messages"]:
            if msg["role"] == "assistant":
                msg["content"].insert(0, {"type": "thinking", "thinking": "already reasoned"})

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(PRIMARY_URL, callback=primary, repeat=True)
            m.post(SIBLING_URL, callback=sibling, repeat=True)

            await server.start_async()
            try:
                status, body = await _post(server, request)
            finally:
                await server.stop_async()

        assert status == 200, body
        assert "Recovered" in body
        assert server._backend_health[0]["healthy"] is False
        assert len(sibling.bodies) == 1
