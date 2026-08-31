"""Tolerance for network interruptions on both sides of the bridge (issue #38).

Two distinct failures used to be conflated, because aiohttp reports a dead
*client* with the same ``ConnectionResetError`` family it uses for a dead
*upstream*:

* An agent that goes away mid-stream was charged to the provider. The Messages
  streaming loop marked the backend unhealthy for 300s, failed over, wrote to
  the same dead client again, and quarantined every remaining backend — so live
  clients got ``503 All backends unhealthy`` for five minutes while both
  providers were fine.
* A momentary blip on the wire to the provider was treated as an outage on the
  first occurrence, with no tolerance window at all.

The client disconnect is forced by patching
:meth:`aiohttp.web.StreamResponse.write` rather than by racing a real socket
close, so the failure lands at exactly the boundary that broke and the test is
deterministic. The upstream mid-stream abort needs a real socket, so those
tests run against a local aiohttp server instead of ``aioresponses``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from aioresponses import aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import (
    _TRANSPORT_GRACE_DELAYS,
    BridgeServer,
    ClientDisconnectedError,
    TransportGrace,
    _is_retryable_exception,
    _is_transport_error,
)
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# ── Test infrastructure ──────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    """Minimal launcher exposing the Messages API bridge protocol."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig()


class _StubProvider(ProviderAdapter):
    """Chat Completions upstream stub pinned to a per-backend base URL."""

    def __init__(self, base_url: str, native: bool = False) -> None:
        self._base_url = base_url
        self._native = native

    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return self._base_url

    @property
    def use_native_messages(self) -> bool:
        return self._native

    @property
    def upstream_wire_is_messages_api(self) -> bool:
        return self._native

    @property
    def upstream_path(self) -> str:
        return "/messages" if self._native else "/chat/completions"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# Chat Completions SSE: one content delta, one finish chunk, then [DONE].
_CC_CONTENT_CHUNK = (
    b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hi"},'
    b'"finish_reason":null}],"model":"test-model"}\n\n'
)
_CC_STREAM = (
    _CC_CONTENT_CHUNK + b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
    b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
    b"data: [DONE]\n\n"
)

# Non-streaming Chat Completions response.
_CC_JSON_RESPONSE = {
    "id": "c1",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

# Anthropic Messages SSE, forwarded verbatim by the native-passthrough path.
_NATIVE_STREAM = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
    b'"type":"message","role":"assistant","content":[],"model":"test-model",'
    b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


@pytest.fixture
def short_grace(monkeypatch):
    """Shrink every retry delay so these tests don't sleep for a minute."""
    monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_PERIOD", 0.3)
    monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.01, 0.02))
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.01, 0.01])


def _make_server(n_backends: int = 2, native: bool = False, base_urls: list[str] | None = None) -> BridgeServer:
    """Build a balancing BridgeServer with ``n_backends`` distinct upstreams."""
    backends = []
    for i in range(n_backends):
        base_url = base_urls[i] if base_urls else f"https://api{i}.example.com/v1"
        provider = _StubProvider(base_url, native=native)
        profile = Profile(name=f"profile-{i}", provider="openai", model="test-model", auth_ref=str(uuid.uuid4()))
        backends.append((provider, f"key-{i}", profile))
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="test-model",
        backends=backends,
        backend_cooldown=300,
    )


class _WriteFailer:
    """Replacement for ``StreamResponse.write`` that dies like a gone client.

    Args:
        fail_after: Number of writes to let through before failing. 0 kills the
            very first write; a larger value exercises a disconnect that only
            shows up on the buffered finish events.
    """

    def __init__(self, fail_after: int = 0) -> None:
        self.fail_after = fail_after
        self.calls = 0

    async def __call__(self, *args, **kwargs) -> None:
        # Patched onto the class as a plain instance, which is not a
        # descriptor, so no `self` is bound and the arguments are whatever the
        # caller passed. They are not needed: this only ever fails.
        self.calls += 1
        if self.calls > self.fail_after:
            raise aiohttp.ClientConnectionResetError("Cannot write to closing transport")


async def _post_stream(port: int) -> bytes:
    """Issue a streaming /v1/messages request and return whatever comes back."""
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{port}/v1/messages",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1024,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp,
    ):
        return await resp.read()


def _count_posts(mocked: aioresponses) -> int:
    """Count upstream POSTs recorded by an ``aioresponses`` context."""
    return sum(len(calls) for key, calls in mocked.requests.items() if key[0] == "POST")


def _assert_all_backends_untouched(server: BridgeServer) -> None:
    """Assert no backend was marked unhealthy or charged a failure."""
    for idx, health in enumerate(server._backend_health):
        assert health["healthy"] is True, f"backend {idx} was quarantined"
        assert health["failure_count"] == 0, f"backend {idx} was charged a failure"


# ── Exception classification ─────────────────────────────────────────────


class TestClientDisconnectedErrorClassification:
    """A client disconnect is never an upstream transport failure."""

    def test_not_retryable(self):
        assert _is_retryable_exception(ClientDisconnectedError("Cannot write to closing transport")) is False

    def test_not_a_transport_error(self):
        assert _is_transport_error(ClientDisconnectedError("Cannot write to closing transport")) is False

    def test_not_a_transport_error_for_connection_reset_wording(self):
        """The message is the client's, so its wording must not leak through."""
        assert _is_transport_error(ClientDisconnectedError("Connection reset by peer")) is False

    def test_upstream_connection_reset_is_still_a_transport_error(self):
        assert _is_transport_error(aiohttp.ClientConnectionResetError("boom")) is True

    def test_truncated_upstream_body_is_a_transport_error(self):
        assert _is_transport_error(aiohttp.ClientPayloadError("short body")) is True
        assert _is_retryable_exception(aiohttp.ClientPayloadError("short body")) is True

    @pytest.mark.parametrize(
        "exc_type",
        [aiohttp.ServerTimeoutError, aiohttp.SocketTimeoutError, aiohttp.ConnectionTimeoutError],
    )
    def test_timeouts_are_not_transport_errors(self, exc_type):
        """aiohttp files its timeouts under ClientConnectionError, but a
        provider that answered too slowly is not a connection blip and must
        not be handed the grace window on top of its own timeout."""
        assert issubclass(exc_type, aiohttp.ClientConnectionError)
        assert _is_transport_error(exc_type("slow")) is False
        assert _is_retryable_exception(exc_type("slow")) is True


# ── Agent → kitty: a dead client must not cost a backend ──────────────────


class TestClientDisconnectLeavesBackendsHealthy:
    """A dead client must not put healthy providers into cooldown."""

    @pytest.mark.asyncio
    async def test_disconnect_mid_stream_keeps_backends_healthy(self):
        """Every write fails; both backends stay healthy."""
        server = _make_server(2)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(2):
                    m.post(
                        f"https://api{i}.example.com/v1/chat/completions",
                        body=_CC_STREAM,
                        headers={"Content-Type": "text/event-stream"},
                        repeat=True,
                    )
                with patch.object(web.StreamResponse, "write", _WriteFailer(fail_after=0)):
                    await _post_stream(port)
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_disconnect_mid_stream_does_not_retry_upstream(self):
        """There is nobody left to serve, so no failover POST is made."""
        server = _make_server(2)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(2):
                    m.post(
                        f"https://api{i}.example.com/v1/chat/completions",
                        body=_CC_STREAM,
                        headers={"Content-Type": "text/event-stream"},
                        repeat=True,
                    )
                with patch.object(web.StreamResponse, "write", _WriteFailer(fail_after=0)):
                    await _post_stream(port)
                posts = _count_posts(m)
            assert posts == 1
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_disconnect_on_native_passthrough_keeps_backends_healthy(self):
        """The raw-chunk forwarding path is a separate write site."""
        server = _make_server(2, native=True)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(2):
                    m.post(
                        f"https://api{i}.example.com/v1/messages",
                        body=_NATIVE_STREAM,
                        headers={"Content-Type": "text/event-stream"},
                        repeat=True,
                    )
                with patch.object(web.StreamResponse, "write", _WriteFailer(fail_after=0)):
                    await _post_stream(port)
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_disconnect_on_finish_events_keeps_backends_healthy(self):
        """The upstream completed cleanly; only the last write fails."""
        server = _make_server(2)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(2):
                    m.post(
                        f"https://api{i}.example.com/v1/chat/completions",
                        body=_CC_STREAM,
                        headers={"Content-Type": "text/event-stream"},
                        repeat=True,
                    )
                # Let the message_start / content_block deltas through, then die
                # on the buffered finish events written after the upstream EOF.
                failer = _WriteFailer(fail_after=4)
                with patch.object(web.StreamResponse, "write", failer):
                    await _post_stream(port)
            assert failer.calls > failer.fail_after, (
                "the stream produced too few events to reach the buffered finish events — "
                "this test would otherwise silently degrade into the fail-on-first-write case"
            )
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()


# ── kitty → provider: the grace window ───────────────────────────────────


class TestTransportGrace:
    """The per-request budget for riding out upstream connection trouble."""

    def test_first_call_returns_the_first_backoff(self):
        grace = TransportGrace(budget=30.0)
        assert grace.next_delay() == _TRANSPORT_GRACE_DELAYS[0]
        assert grace.retries == 1

    def test_backoff_escalates_then_stops(self):
        """Capped by count too, so instant failures cannot eat every attempt."""
        grace = TransportGrace(budget=1000.0)
        delays = [grace.next_delay() for _ in range(len(_TRANSPORT_GRACE_DELAYS) + 1)]
        assert delays[: len(_TRANSPORT_GRACE_DELAYS)] == list(_TRANSPORT_GRACE_DELAYS)
        assert delays[-1] is None

    def test_delay_never_runs_past_the_end_of_the_window(self):
        grace = TransportGrace(budget=0.5)
        assert grace.next_delay() == pytest.approx(0.5, abs=0.05)

    def test_exhausted_budget_returns_none(self):
        assert TransportGrace(budget=0.0).next_delay() is None

    def test_budget_is_spent_by_elapsed_time_not_retry_count(self):
        """The window is wall-clock, so a slow retry can end it on its own."""
        grace = TransportGrace(budget=0.05)
        assert grace.next_delay() is not None
        time.sleep(0.06)
        assert grace.next_delay() is None

    def test_default_budget_follows_the_module_setting(self, monkeypatch):
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_PERIOD", 0.0)
        assert TransportGrace().next_delay() is None


class TestUpstreamConnectionTolerance:
    """A connection blip to the provider is retried, not charged to health."""

    @pytest.mark.asyncio
    async def test_upstream_blip_is_retried_on_the_same_backend(self, short_grace):
        """The reset is invisible to the agent and costs no backend health."""
        server = _make_server(1)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    exception=aiohttp.ClientConnectionResetError("blip"),
                )
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    body=_CC_STREAM,
                    headers={"Content-Type": "text/event-stream"},
                )
                body = await _post_stream(port)
                posts = _count_posts(m)
            assert posts == 2, "the blip should have been retried"
            assert b"message_stop" in body, "the agent should still get a complete stream"
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_sustained_upstream_failure_still_quarantines(self, short_grace):
        """Past the window it is an outage, not a blip: mark it unhealthy."""
        server = _make_server(1)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    exception=aiohttp.ClientConnectionResetError("upstream went away"),
                    repeat=True,
                )
                await _post_stream(port)
            assert server._backend_health[0]["healthy"] is False
            assert server._backend_health[0]["transport_error_count"] >= 1
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_grace_does_not_delay_a_non_connection_failure(self, short_grace):
        """An HTTP 500 is an answer, not a blip — fail over without waiting."""
        server = _make_server(2)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api0.example.com/v1/chat/completions", status=500, body="boom", repeat=True)
                m.post("https://api1.example.com/v1/chat/completions", status=500, body="boom", repeat=True)
                await _post_stream(port)
            for health in server._backend_health:
                assert health["transport_error_count"] == 0
                assert health["healthy"] is False
        finally:
            await server.stop_async()


class TestUpstreamToleranceAcrossProtocols:
    """Connecting to the provider is retried on every protocol, streamed or
    not — that is the one point in a request where a retry cannot duplicate
    output, because nothing has been read from the upstream yet."""

    def _bridge_server(self) -> BridgeServer:
        """A bridge-mode server, which registers every protocol route."""
        provider = _StubProvider("https://api0.example.com/v1")
        profile = Profile(name="profile-0", provider="openai", model="test-model", auth_ref=str(uuid.uuid4()))
        return BridgeServer(
            adapter=None,  # type: ignore[arg-type]
            provider=provider,
            resolved_key="key-0",
            model="test-model",
            backends=[(provider, "key-0", profile)],
            backend_cooldown=300,
        )

    @pytest.mark.parametrize(
        ("path", "payload", "stream"),
        [
            ("/v1/responses", {"model": "test-model", "input": [{"role": "user", "content": "hi"}]}, True),
            (
                "/v1/chat/completions",
                {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                True,
            ),
            (
                "/v1/chat/completions",
                {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
                False,
            ),
            (
                "/v1beta/models/test-model:streamGenerateContent",
                {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                True,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_connection_blip_is_retried(self, short_grace, path, payload, stream):
        server = self._bridge_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    exception=aiohttp.ClientConnectionResetError("blip"),
                )
                if stream:
                    m.post(
                        "https://api0.example.com/v1/chat/completions",
                        body=_CC_STREAM,
                        headers={"Content-Type": "text/event-stream"},
                    )
                else:
                    m.post("https://api0.example.com/v1/chat/completions", payload=_CC_JSON_RESPONSE)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}{path}",
                        json={**payload, "stream": stream},
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp,
                ):
                    assert resp.status == 200
                    await resp.read()
                assert _count_posts(m) == 2, "the blip should have been retried"
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()


class TestUpstreamAbortAfterBytesReachedTheClient:
    """Once bytes are on the wire a restart would duplicate them, so the
    partial message is closed off rather than retried or failed over — and the
    client is always left with a terminated stream, never one that just stops."""

    @staticmethod
    async def _serve_abort(path: str, chunk: bytes, counter: list[int]) -> tuple[web.AppRunner, str]:
        """Start a local upstream that sends ``chunk`` then aborts the socket."""

        async def _handler(request: web.Request) -> web.StreamResponse:
            counter.append(1)
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(chunk)
            await asyncio.sleep(0)
            request.transport.abort()  # type: ignore[union-attr]
            return resp

        app = web.Application()
        app.router.add_post(path, _handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner, f"http://127.0.0.1:{runner.addresses[0][1]}/v1"

    @pytest.mark.asyncio
    async def test_mid_stream_abort_finalizes_instead_of_erroring(self, short_grace):
        calls: list[int] = []
        runner, base = await self._serve_abort("/v1/chat/completions", _CC_CONTENT_CHUNK, calls)
        server = _make_server(2, base_urls=[base, base])
        port = await server.start_async()
        try:
            body = await _post_stream(port)
            assert len(calls) == 1, "a stream with bytes already delivered must not be restarted"
            assert b"message_stop" in body, "the partial message should be closed off cleanly"
            assert b"internal_error" not in body and b"api_error" not in body
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_native_passthrough_abort_still_terminates_the_stream(self, short_grace):
        """The native path never drives the translator, so it has no half-open
        message to close — it must still get a terminal event rather than an
        SSE stream that simply stops."""
        calls: list[int] = []
        native_chunk = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
            b'"type":"message","role":"assistant","content":[],"model":"test-model",'
            b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
        )
        runner, base = await self._serve_abort("/v1/messages", native_chunk, calls)
        server = _make_server(2, native=True, base_urls=[base, base])
        port = await server.start_async()
        try:
            body = await _post_stream(port)
            assert len(calls) == 1, "a stream with bytes already delivered must not be restarted"
            assert b"message_start" in body
            assert b'"type": "error"' in body or b'"type":"error"' in body, (
                "the client must be told the stream ended, not left hanging"
            )
            _assert_all_backends_untouched(server)
        finally:
            await server.stop_async()
            await runner.cleanup()


class TestRetryLoopStaysBounded:
    """Extending the Messages loop to hand grace retries back must not let it
    run past its last real attempt — the empty-response schedule is indexed by
    `attempt`, so an over-running loop indexes off the end of it."""

    @pytest.mark.asyncio
    async def test_upstream_errors_never_exceed_the_attempt_budget(self, short_grace):
        server = _make_server(3)
        port = await server.start_async()
        max_attempts = (server_module._MAX_RETRIES + 1) * 3 + len(server_module._EMPTY_FINAL_DELAYS)
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(3):
                    m.post(f"https://api{i}.example.com/v1/chat/completions", status=500, body="boom", repeat=True)
                body = await _post_stream(port)
                posts = _count_posts(m)
            assert posts <= max_attempts, f"{posts} upstream calls exceeds the {max_attempts}-attempt budget"
            assert b"error" in body, "the agent should be told the request failed"
        finally:
            await server.stop_async()


class TestGraceIsPerRequestNotPerBackend:
    """The window covers a request, however many backends it touches."""

    @pytest.mark.asyncio
    async def test_non_streaming_shares_one_window_across_backends(self, short_grace, monkeypatch):
        """`_request_with_retry` calls `_make_upstream_request` once per
        backend; a fresh window per call would let one request spend the whole
        tolerance three times over."""
        windows: list[server_module.TransportGrace] = []
        real_init = server_module.TransportGrace.__init__

        def _track(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            windows.append(self)

        monkeypatch.setattr(server_module.TransportGrace, "__init__", _track)

        server = _make_server(3)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                for i in range(3):
                    m.post(
                        f"https://api{i}.example.com/v1/chat/completions",
                        exception=aiohttp.ClientConnectionResetError("blip"),
                        repeat=True,
                    )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1024,
                            "stream": False,
                        },
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp,
                ):
                    await resp.read()
            assert len(windows) == 1, f"expected one grace window for the request, got {len(windows)}"
            assert windows[0].retries <= len(server_module._TRANSPORT_GRACE_DELAYS)
        finally:
            await server.stop_async()


class TestGraceDoesNotSpendTheFailoverBudget:
    """A grace retry re-sends to the same backend, so it must not pull the
    empty-response schedule forward or consume a failover attempt."""

    @pytest.mark.asyncio
    async def test_persistent_blip_does_not_reach_the_empty_response_retries(self, monkeypatch, caplog):
        """A single-backend profile hitting a dead upstream should spend its
        grace and then fail — not also sit through the empty-response delays,
        which are for a backend that answered with nothing, not one that never
        answered at all.

        Asserted on the log rather than on elapsed time: the delays are
        shortened so the test stays fast, which would hide the bug from a
        timing assertion.
        """
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_PERIOD", 0.05)
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.001, 0.001, 0.001, 0.001))
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.001, 0.001])

        server = _make_server(1)
        port = await server.start_async()
        try:
            with (
                caplog.at_level(logging.WARNING, logger="kitty.bridge.server"),
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
            ):
                m.post(
                    "https://api0.example.com/v1/chat/completions",
                    exception=aiohttp.ClientConnectionResetError("blip"),
                    repeat=True,
                )
                await _post_stream(port)
            empty_retries = [r.getMessage() for r in caplog.records if "Empty upstream response" in r.getMessage()]
            assert empty_retries == [], f"grace retries pulled the empty-response schedule forward: {empty_retries}"
            assert server._backend_health[0]["healthy"] is False
        finally:
            await server.stop_async()
