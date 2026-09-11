"""KBR-5 — an unsendable conversation fails downstream, never upstream.

When compaction leaves no non-system message the bridge used to substitute a
synthetic user turn naming the product and send it to the provider. That message
was written for the user and delivered to the one audience it was never meant
for, letting any provider that parses request bodies fingerprint the bridge.

These tests drive the real handlers over real sockets and assert the replacement
behaviour: a protocol-native HTTP 400, no upstream request, and a body carrying
``error.reason == "compaction_failed"``.

**The marker is load-bearing.** ``_check_request_size`` returns a byte-identical
400 envelope from the line immediately after ``_apply_compaction`` in every
handler, so an assertion on status plus envelope alone cannot tell the two
rejections apart and would pass against the unfixed code.
``test_size_rejection_is_not_reported_as_compaction_failure`` is the negative
control that pins the difference.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from harness.contract import WireFormat
from harness.recorder import RecordingUpstream

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer, UpstreamError
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Stubs ──────────────────────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    def __init__(self, protocol: BridgeProtocol = BridgeProtocol.MESSAGES_API):
        self._protocol = protocol

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return self._protocol

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    """Provider that records every upstream body it is asked to build."""

    def __init__(self, provider_type: str = "stub", base_url: str = "https://api.example.com/v1"):
        self._provider_type = provider_type
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        return self._provider_type

    @property
    def default_base_url(self) -> str:
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Helpers ────────────────────────────────────────────────────────────────


def _bridge_mode_server() -> BridgeServer:
    """Server with every protocol endpoint registered (adapter=None)."""
    return BridgeServer(None, _StubProvider(), "test-key")  # type: ignore[arg-type]


def _balancing_server(n_backends: int = 2) -> BridgeServer:
    """Bridge-mode server with a balancing pool."""
    backends = [
        (
            _StubProvider(provider_type=f"stub-{i}", base_url=f"https://api{i}.example.com/v1"),
            f"key-{i}",
            Profile(name=f"profile-{i}", provider="openai", model=f"model-{i}", auth_ref=str(uuid.uuid4())),
        )
        for i in range(n_backends)
    ]
    return BridgeServer(None, _StubProvider(), "test-key", backends=backends)  # type: ignore[arg-type]


class _UpstreamSpy:
    """Counts upstream requests so 'nothing was sent' is a real assertion."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def __call__(self, cc_request: dict, *args, **kwargs) -> dict:
        self.bodies.append(json.loads(json.dumps(cc_request)))
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


#: The header the falsification control injects, and the reason that control is
#: a defect at all. Named here so the control can check its own premise: if this
#: pair ever stopped naming the product, the control would go red saying the
#: guard is broken when what actually changed is the bait (KBR-147).
_BAIT_HEADER = ("X-Kitty-Bridge", "1.9.2")


class _ProductNamedHeaderProvider(_StubProvider):
    """Provider that leaks the product's name into an upstream header.

    The bait for the falsification control below.

    Overriding :meth:`~kitty.providers.base.ProviderAdapter.build_upstream_headers`
    is enough because the base
    :meth:`~kitty.providers.base.ProviderAdapter.build_upstream_headers_for_model`
    — which is what :meth:`~kitty.bridge.server.BridgeServer._build_upstream_headers`
    actually calls — delegates to it.
    """

    def build_upstream_headers(self, api_key: str) -> dict[str, str]:
        """Return the base header set with a product-named header added.

        Args:
            api_key: Resolved API key for the upstream provider.

        Returns:
            The base headers plus :data:`_BAIT_HEADER`.
        """
        return {**super().build_upstream_headers(api_key), _BAIT_HEADER[0]: _BAIT_HEADER[1]}


def _names_the_product(observed: object) -> bool:
    """Return whether a captured header set or request body names the product.

    One scan, called by both the guard and its falsification control. A control
    that reimplemented the scan would prove the copy works, not the guard.

    Args:
        observed: Anything JSON-serializable — a header pair sequence as
            :class:`~harness.contract.CapturedRequest` carries it, or a decoded
            request body.

    Returns:
        True when the product's name appears anywhere in ``observed``.
    """
    return "kitty" in json.dumps(observed, ensure_ascii=False).lower()


@asynccontextmanager
async def _recording_upstream() -> AsyncIterator[RecordingUpstream]:
    """Yield a running recording upstream, checking for fallback replies on exit.

    The §7.2 primary recorder (``tests/harness/recorder.py``, KBR-27) rather than
    a stand-in: it observes what aiohttp actually sent, with original casing and
    duplicates preserved, which is what §4.3 C1 asserts on. A stand-in patched
    over a bridge method observes only what the bridge passed it — the defect
    KBR-147 records.

    Yields:
        The running :class:`~harness.recorder.RecordingUpstream`.
    """
    upstream = RecordingUpstream(default_format=WireFormat.CHAT_COMPLETIONS)
    await upstream.start()
    try:
        yield upstream
    finally:
        await upstream.stop()
        # Checked in `finally`, as the reference fixture does: a fallback-format
        # reply is not loud on its own -- the adapter parses nothing, the bridge
        # reads the reply as empty, and the test pays the 80-second retry ladder
        # *inside* this block, which is when it most needs to say why.
        upstream.assert_all_paths_matched()


#: A conversation whose only non-system turn is an unpaired tool result.
#:
#: Deliberately tiny. Pairing validation drops the orphan and leaves nothing
#: sendable, which is the real trigger — an oversized system prompt is not, and
#: a multi-megabyte fixture would only make these tests slow.
_ORPHAN_CC_MESSAGES = [
    {"role": "system", "content": "you are helpful"},
    {"role": "tool", "content": "result", "tool_call_id": "call_missing"},
]


async def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    """POST ``payload`` and return ``(status, decoded body)``."""
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}{path}",
            json=payload,
            headers={"content-type": "application/json"},
        ) as resp,
    ):
        return resp.status, await resp.json()


# ── Pre-flight: every protocol renders its own dialect ─────────────────────


class TestPreflightCompactionFailure:
    """Each handler turns an unsendable conversation into its own 400."""

    @pytest.mark.asyncio
    async def test_chat_completions_returns_dialect_400_and_sends_nothing(self, monkeypatch):
        """Chat Completions: OpenAI-shaped 400, and the provider hears nothing."""
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1/chat/completions",
                {"model": "m", "messages": _ORPHAN_CC_MESSAGES, "stream": False},
            )
        finally:
            await server.stop_async()

        assert status == 400
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["reason"] == "compaction_failed"
        assert spy.bodies == [], "no upstream request may be made"

    @pytest.mark.asyncio
    async def test_messages_returns_dialect_400_and_sends_nothing(self, monkeypatch):
        """Anthropic Messages: ``{"type": "error", ...}`` envelope."""
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1/messages",
                {
                    "model": "m",
                    "system": "you are helpful",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "toolu_missing", "content": "r"}
                            ],
                        }
                    ],
                    "stream": False,
                },
            )
        finally:
            await server.stop_async()

        assert status == 400
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["reason"] == "compaction_failed"
        assert spy.bodies == [], "no upstream request may be made"

    @pytest.mark.asyncio
    async def test_responses_returns_dialect_400_and_sends_nothing(self, monkeypatch):
        """Responses API: ``error.code`` envelope, kitty's own convention."""
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1/responses",
                {
                    "model": "m",
                    "input": [
                        {"type": "function_call_output", "call_id": "call_missing", "output": "r"}
                    ],
                    "instructions": "you are helpful",
                    "stream": False,
                },
            )
        finally:
            await server.stop_async()

        assert status == 400
        assert body["error"]["code"] == "invalid_request"
        assert body["error"]["reason"] == "compaction_failed"
        assert spy.bodies == [], "no upstream request may be made"

    @pytest.mark.asyncio
    async def test_gemini_returns_dialect_400_and_sends_nothing(self, monkeypatch):
        """Gemini: Google-shaped 400 with ``INVALID_ARGUMENT``.

        A Gemini CLI cannot parse an Anthropic error envelope, which is why the
        response is rendered per dialect rather than reusing one shape.
        """
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1beta/models/gemini-pro:generateContent",
                {
                    "systemInstruction": {"parts": [{"text": "you are helpful"}]},
                    # role "function" is Gemini's tool-response role; "user"
                    # would make the translator drop the part before compaction
                    # ever sees it, and the test would prove nothing.
                    "contents": [
                        {
                            "role": "function",
                            "parts": [
                                {"functionResponse": {"name": "missing", "response": {"result": "r"}}}
                            ],
                        }
                    ],
                },
            )
        finally:
            await server.stop_async()

        assert status == 400
        assert body["error"]["code"] == 400
        assert body["error"]["status"] == "INVALID_ARGUMENT"
        assert body["error"]["reason"] == "compaction_failed"
        assert spy.bodies == [], "no upstream request may be made"


# ── Streaming ──────────────────────────────────────────────────────────────


class TestStreamingCompactionFailure:
    """A streaming request still gets a plain HTTP 400, not an SSE error event.

    Compaction runs *before* the stream branch in every handler, so no bytes
    have been written and the response is still unprepared. Anthropic's own API
    behaves the same way: pre-stream failures are HTTP errors and ``event:
    error`` is reserved for failures after a 200.

    Two variants cover the two branch mechanisms — a body flag, and Gemini's
    route-based selection. The raise is strictly before both, so the remaining
    handlers add nothing.
    """

    @pytest.mark.asyncio
    async def test_messages_stream_true_gets_plain_400(self, monkeypatch):
        """``stream: true`` on Messages is rejected before the stream opens."""
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "m",
                        "system": "you are helpful",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "tool_result", "tool_use_id": "toolu_missing", "content": "r"}
                                ],
                            }
                        ],
                        "stream": True,
                    },
                    headers={"content-type": "application/json"},
                ) as resp,
            ):
                assert resp.status == 400
                assert "text/event-stream" not in resp.headers.get("content-type", "")
                body = await resp.json()
        finally:
            await server.stop_async()

        assert body["error"]["reason"] == "compaction_failed"
        assert spy.bodies == []

    @pytest.mark.asyncio
    async def test_gemini_stream_route_gets_plain_400(self, monkeypatch):
        """Gemini streams by route, not by body flag — cover it explicitly."""
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1beta/models/gemini-pro:streamGenerateContent",
                    json={
                        "systemInstruction": {"parts": [{"text": "you are helpful"}]},
                        # See the non-streaming Gemini test: role must be
                        # "function", or the translator drops the part first.
                        "contents": [
                            {
                                "role": "function",
                                "parts": [
                                    {"functionResponse": {"name": "missing", "response": {"result": "r"}}}
                                ],
                            }
                        ],
                    },
                    headers={"content-type": "application/json"},
                ) as resp,
            ):
                assert resp.status == 400
                assert "text/event-stream" not in resp.headers.get("content-type", "")
                body = await resp.json()
        finally:
            await server.stop_async()

        assert body["error"]["status"] == "INVALID_ARGUMENT"
        assert body["error"]["reason"] == "compaction_failed"
        assert spy.bodies == []


# ── The negative control ───────────────────────────────────────────────────


class TestSizeRejectionIsDistinguishable:
    """``_check_request_size`` must not be mistaken for a compaction failure.

    It returns status 400 with the same Anthropic envelope from the very next
    line of the handler. Without this test, every assertion above would pass
    against the unfixed code.
    """

    @pytest.mark.asyncio
    async def test_size_rejection_is_not_reported_as_compaction_failure(self, monkeypatch):
        """One huge user turn and no system message: rejected, but not by us.

        ``min_tail`` forces the single block into the tail and the guaranteed-fit
        loop stops at one block, so a non-system message survives and nothing
        raises — then the size check rejects the request.
        """
        server = _bridge_mode_server()
        spy = _UpstreamSpy()
        monkeypatch.setattr(server, "_make_upstream_request", spy)
        monkeypatch.setattr(server, "_get_max_context_chars", lambda: 20_000)
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": "x" * 200_000}], "stream": False},
            )
        finally:
            await server.stop_async()

        assert status == 400
        assert "reason" not in body["error"], "a size rejection must not claim to be a compaction failure"
        assert spy.bodies == []


# ── Recovery path ──────────────────────────────────────────────────────────


class TestRecoveryPathCompactionFailure:
    """The on-failure re-compaction path must fail over, not cool the pool down.

    ``_compact_with_tighter_budget`` runs after an upstream context-too-large
    rejection, at half the budget. If that leaves nothing sendable, the bridge
    tries the next backend — the budget is the pool's *minimum* context, so a
    larger-context sibling may still accept the same body — and reports the 400
    only once failover stops.
    """

    @pytest.mark.asyncio
    async def test_irreducible_conversation_fails_over_without_cooling_backends(self, monkeypatch):
        """Every backend is tried, none is marked unhealthy, and the 400 wins.

        Marking unhealthy here would take the whole pool offline for the
        cooldown because one conversation was corrupt, and every other
        concurrent session would start getting 503s.
        """
        server = _balancing_server(n_backends=2)
        attempts: list[int] = []
        cooled: list[int] = []

        async def _always_too_large(cc_request, *args, **kwargs):
            attempts.append(server._current_backend_idx)
            raise UpstreamError(400, "context length exceeded: too many tokens in the request")

        monkeypatch.setattr(server, "_make_upstream_request", _always_too_large)
        monkeypatch.setattr(server, "_mark_backend_unhealthy", lambda idx, **kw: cooled.append(idx))
        # Force the recovery gate: it also requires an oversized message payload.
        monkeypatch.setattr(server, "_is_oversized_request", lambda _req: True)
        monkeypatch.setattr(
            server,
            "_compact_with_tighter_budget",
            lambda cc_request, factor=0.5: server._compact_messages(list(_ORPHAN_CC_MESSAGES), 10),
        )
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
        finally:
            await server.stop_async()

        assert len(attempts) == 2, "both backends must be tried before giving up"
        assert cooled == [], "an irreducible conversation is no evidence against a backend"
        assert status == 400
        assert body["error"]["reason"] == "compaction_failed"

    @pytest.mark.asyncio
    async def test_a_later_unrelated_failure_is_reported_as_itself(self, monkeypatch):
        """Backend A irreducible, backend B rate-limited → the user sees the 429.

        Regression guard for a sticky flag. Reporting "this conversation cannot
        be reduced, start a new one" for a transient rate limit tells the user
        not to retry the one thing that would have worked.
        """
        server = _balancing_server(n_backends=2)
        calls: list[int] = []

        async def _fail(cc_request, *args, **kwargs):
            calls.append(len(calls))
            if len(calls) == 1:
                raise UpstreamError(400, "context length exceeded: too many tokens")
            raise UpstreamError(429, "rate limited")

        monkeypatch.setattr(server, "_make_upstream_request", _fail)
        monkeypatch.setattr(server, "_is_oversized_request", lambda _req: True)

        def _tighter(cc_request, factor=0.5):
            if len(calls) == 1:
                server._compact_messages(list(_ORPHAN_CC_MESSAGES), 10)

        monkeypatch.setattr(server, "_compact_with_tighter_budget", _tighter)
        # The assertion below does not depend on the empty-response ladder, and
        # this test falls into it: backend 0 stays healthy (recovery deliberately
        # does not cool it down), the flag resets on attempt 2, so the loop ends
        # with nothing to raise and drops through to `_EMPTY_FINAL_DELAYS` —
        # 20s + 40s of real sleep before the 429 surfaces, on every Python
        # version in CI. Stubbed for the same reason as in
        # `test_a_genuinely_oversized_payload_reaches_the_tighter_recompaction`.
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [])
        port = await server.start_async()
        try:
            status, body = await _post(
                port,
                "/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
        finally:
            await server.stop_async()

        assert status == 429, "an unrelated later failure must surface as itself"
        assert body["error"].get("reason") != "compaction_failed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,payload,read_reason",
        [
            pytest.param(
                "/v1/messages",
                {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                lambda b: b["error"]["reason"],
                id="messages",
            ),
            pytest.param(
                "/v1/responses",
                {"model": "m", "input": [{"role": "user", "content": "hi"}], "stream": False},
                lambda b: b["error"]["reason"],
                id="responses",
            ),
            pytest.param(
                "/v1beta/models/gemini-pro:generateContent",
                {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                lambda b: b["error"]["reason"],
                id="gemini",
            ),
        ],
    )
    async def test_every_handler_renders_its_dialect_on_the_recovery_path(
        self, monkeypatch, path, payload, read_reason
    ):
        """The recovery raise must not degrade to a 500 in any handler.

        Each handler catches this at a *different* site from the pre-flight one,
        and all four currently degrade differently without it: Messages and
        Responses through their inner ``except Exception`` around
        ``_request_with_retry``, Gemini and Chat Completions through theirs, and
        the two without an outer ``try`` through the access-log middleware. The
        clause is per handler, so the test has to be too.
        """
        server = _balancing_server(n_backends=2)

        async def _always_too_large(cc_request, *args, **kwargs):
            raise UpstreamError(400, "context length exceeded: too many tokens in the request")

        monkeypatch.setattr(server, "_make_upstream_request", _always_too_large)
        monkeypatch.setattr(server, "_is_oversized_request", lambda _req: True)
        monkeypatch.setattr(
            server,
            "_compact_with_tighter_budget",
            # Substitutes the real method deliberately: this test is about the
            # handler's exception rendering, not about budget arithmetic. The
            # real method is covered by the oversized-payload test below.
            lambda cc_request, factor=0.5: server._compact_messages(list(_ORPHAN_CC_MESSAGES), 10),
        )
        port = await server.start_async()
        try:
            status, body = await _post(port, path, payload)
        finally:
            await server.stop_async()

        assert status == 400, f"{path} degraded instead of rendering its dialect"
        assert read_reason(body) == "compaction_failed"

    @pytest.mark.asyncio
    async def test_a_genuinely_oversized_payload_reaches_the_tighter_recompaction(self, monkeypatch):
        """Drive the real 600 KB gate instead of mocking it open.

        The recovery path has three conditions, and the third —
        ``_is_oversized_request``, i.e. serialized messages over
        ``_OVERSIZED_INPUT_THRESHOLD`` — is the one a small fixture silently
        fails, leaving a test that proves nothing. The sibling tests above mock
        it to stay fast; this one sends a real 700 KB history and asserts the
        real ``_compact_with_tighter_budget`` runs.

        **It deliberately does not assert a 400.** Measured against the code,
        the recovery re-compaction cannot reach the post-condition with
        realistic input: pre-flight ``_apply_compaction`` strips every orphan
        tool result (raising if that empties the conversation), so anything
        arriving here is already well-paired, and ``_compact_messages`` groups
        pairs atomically and always keeps one non-system block. The guard at
        that site is therefore defence in depth — kept because
        ``_compact_with_tighter_budget`` does **not** re-run pairing validation,
        so a future change could make it reachable, and because the invariant
        should hold wherever compaction runs. The sibling tests inject the
        exception precisely because no fixture can provoke it here.
        """
        server = _balancing_server(n_backends=1)
        reached: list[float] = []
        real_tighter = server._compact_with_tighter_budget

        def _spy_tighter(cc_request, factor=0.5):
            reached.append(factor)
            return real_tighter(cc_request, factor=factor)

        async def _always_too_large(cc_request, *args, **kwargs):
            raise UpstreamError(400, "context length exceeded: too many tokens in the request")

        monkeypatch.setattr(server, "_make_upstream_request", _always_too_large)
        monkeypatch.setattr(server, "_compact_with_tighter_budget", _spy_tighter)
        # The empty-response fallback sleeps 20s then 40s; nothing here depends
        # on it and the test should not either.
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [])

        # Over _OVERSIZED_INPUT_THRESHOLD (600_000 serialized chars) so the real
        # gate opens. Split across turns to stay a plausible history.
        big_turns = [{"role": "user", "content": "x" * 70_000} for _ in range(10)]
        port = await server.start_async()
        try:
            status, _ = await _post(
                port,
                "/v1/chat/completions",
                {"model": "m", "messages": big_turns, "stream": False},
            )
        finally:
            await server.stop_async()

        assert reached == [0.5], "a real 700KB payload must reach the tighter re-compaction"
        assert status == 400, "the upstream's own context-too-large error still surfaces"


# ── The wire itself ────────────────────────────────────────────────────────


class TestNothingOnTheWireNamesTheProduct:
    """The end-to-end half of the guard: what actually leaves the process.

    Every other test here asserts that *nothing* is sent. These two let a normal
    request through into the §7.2 recording upstream and inspect the bytes and
    headers the provider really received.

    **The header channel used to prove nothing (KBR-147).** The capture was a
    stand-in monkeypatched over ``_build_upstream_headers``, which is called
    *inside* the ``_make_upstream_request`` the same test replaced — so it never
    ran, and the assertion scanned an empty dict. Both tests below therefore
    assert their capture is non-empty before scanning it, and
    ``test_the_guard_detects_a_product_named_header`` is the falsification
    control that shows the scan is wired to something.
    """

    @pytest.mark.asyncio
    async def test_body_and_headers_carry_no_vendor_token(self):
        """A below-threshold request reaches upstream naming nothing.

        The inbound turn deliberately contains ``kitty-bridge``: it must arrive
        **unchanged**, proving the guard is not satisfied by stripping the
        user's words. Satisfying indistinguishability that way would breach
        message fidelity, which is the trap ``TEST_SUITE.md`` §6.2.3 warns about.
        """
        turn = {"role": "user", "content": "Please explain how kitty-bridge works"}
        async with _recording_upstream() as upstream:
            server = BridgeServer(None, _StubProvider(base_url=upstream.base_url), "test-key")  # type: ignore[arg-type]
            port = await server.start_async()
            try:
                status, _ = await _post(
                    port,
                    "/v1/chat/completions",
                    {"model": "m", "messages": [dict(turn)], "stream": False},
                )
            finally:
                await server.stop_async()

        assert status == 200
        # One attempt, so a silent retry ladder is red rather than slow-green.
        assert len(upstream.requests) == 1, f"expected one upstream request, recorded {len(upstream.requests)}"
        captured = upstream.requests[0]

        # The non-vacuity guard KBR-147 was missing. Without it the scan below
        # passes over an empty capture and can never fail.
        assert captured.headers, "the headers were never captured, so this proves nothing"
        assert ("authorization", "Bearer test-key") in [(name.lower(), value) for name, value in captured.headers], (
            f"the real header builder never reached the wire: {captured.headers}"
        )
        assert not _names_the_product(captured.headers), f"a header names the product: {captured.headers}"

        sent = json.loads(captured.body)
        assert turn in sent["messages"], "the agent's own text must reach the provider unchanged"

        # Everything the bridge added, with the agent's own turn removed. The
        # exclusion is `messages` because _StubProvider emits a Chat Completions
        # body; an adapter on another wire shape carries agent text in `system`
        # or `input` too, and the general answer there is the oracle's
        # projection diff (TEST_SUITE.md §3.3.3, KBR-81).
        introduced = {k: v for k, v in sent.items() if k != "messages"}
        assert not _names_the_product(introduced), f"the bridge introduced a vendor token: {introduced}"

    @pytest.mark.asyncio
    async def test_the_guard_detects_a_product_named_header(self):
        """The falsification control: an injected product-named header is caught.

        The guard above passes today, so on its own it cannot distinguish "no
        header names the product" from "the capture is broken again". This drives
        the same path with an adapter that leaks, through the same capture and
        the same scan, and requires the leak to be seen.
        """
        # Check the bait before trusting the conclusion, so a bait that stopped
        # naming the product fails here rather than as "the guard is broken".
        assert "kitty" in "".join(_BAIT_HEADER).lower(), f"the bait no longer names the product: {_BAIT_HEADER}"

        async with _recording_upstream() as upstream:
            provider = _ProductNamedHeaderProvider(base_url=upstream.base_url)
            server = BridgeServer(None, provider, "test-key")  # type: ignore[arg-type]
            port = await server.start_async()
            try:
                await _post(
                    port,
                    "/v1/chat/completions",
                    {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": False},
                )
            finally:
                await server.stop_async()

        assert upstream.requests, "the request never reached upstream, so this proves nothing"
        captured = upstream.requests[0]
        assert captured.headers, "the headers were never captured, so this proves nothing"
        assert _names_the_product(captured.headers), (
            f"the guard cannot see an injected product-named header: {captured.headers}"
        )


# ── The message the user actually reads ────────────────────────────────────


class TestErrorMessageIsActionable:
    """The message is for the user, so it has to be true and useful.

    The text it replaced blamed the system prompt and told the user to run
    ``/clear``. Neither is reliably right: a large system prompt cannot cause
    this, and the real cause is a lost tool-call pairing.
    """

    def test_message_names_the_product_and_the_real_cause(self):
        """It names the product, the true cause, and both remedies."""
        message = BridgeServer._COMPACTION_FAILED_MESSAGE
        assert "Kitty Bridge" in message
        assert "tool" in message.lower(), "the real cause is a lost tool-call pairing"
        assert "/clear" in message
        assert "context window" in message

    def test_message_does_not_blame_the_system_prompt(self):
        """The old, falsified explanation must not come back."""
        assert "system prompt" not in BridgeServer._COMPACTION_FAILED_MESSAGE.lower()

    def test_the_readme_documents_this_error(self):
        """A user who hits this will search the README for the words they saw.

        README is this product's specification, so a user-visible error that is
        not in Troubleshooting is drift — the repo's own documentation rule says
        as much. Anchored on the message's first clause rather than the whole
        string, so rewording the tail does not force a README edit while
        changing what the user actually searches for does.
        """
        readme = (Path(__file__).resolve().parent.parent.parent / "README.md").read_text(encoding="utf-8")
        opening = BridgeServer._COMPACTION_FAILED_MESSAGE.split(":")[0]
        assert opening in readme, (
            f"README has no Troubleshooting entry for {opening!r}. A user who sees this error "
            "will search for it; add the entry rather than deleting this test."
        )

    @pytest.mark.parametrize(
        "style", ["anthropic", "openai_chat", "openai_responses", "google"]
    )
    def test_every_dialect_carries_the_marker(self, style):
        """No dialect may omit the marker that makes this failure identifiable."""
        response = BridgeServer._compaction_failed_response(style=style)
        body = json.loads(response.body)
        assert body["error"]["reason"] == "compaction_failed"
        assert response.status == 400
