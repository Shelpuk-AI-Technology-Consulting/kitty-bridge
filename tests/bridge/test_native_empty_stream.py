"""The native Messages passthrough judges an empty reply before the client sees it (KBR-155).

Before the preamble hold, ``_stream_messages`` forwarded every native upstream
chunk the instant it arrived and recorded the attempt as a success whatever it
carried, so a contentless Anthropic stream reached Claude Code as a blank turn
with no retry and no failover. These tests drive the whole handler against a
mocked or local upstream and pin what the client receives once the hold is in
place: ``.system_design/TEST_SUITE.md`` §11 Q14(b) and amendments D1–D7.

Covers ``.requirements/20260913T134250Z_native_preamble_hold`` R1 and R3–R12.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import aiohttp
import pytest
from aiohttp import web
from aioresponses import CallbackResult, aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol


class _StubLauncher(LauncherAdapter):
    """Minimal launcher exposing the Messages API bridge protocol."""

    @property
    def name(self) -> str:
        """str: Launcher name."""
        return "stub"

    @property
    def binary_name(self) -> str:
        """str: Launcher binary."""
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """BridgeProtocol: The inbound protocol, Anthropic Messages."""
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        """Return an empty spawn config; nothing is spawned here.

        Args:
            profile: Unused.
            bridge_port: Unused.
            resolved_key: Unused.

        Returns:
            An empty :class:`SpawnConfig`.
        """
        return SpawnConfig()


class _NativeProvider(ProviderAdapter):
    """Upstream stub that declares the native Messages passthrough."""

    def __init__(self, base_url: str) -> None:
        """Pin the stub to one base URL.

        Args:
            base_url: The upstream base URL, distinct per backend.
        """
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        """str: Provider type."""
        return "stub"

    @property
    def default_base_url(self) -> str:
        """str: The pinned base URL."""
        return self._base_url

    @property
    def use_native_messages(self) -> bool:
        """bool: Always ``True`` — the path under test."""
        return True

    @property
    def upstream_wire_is_messages_api(self) -> bool:
        """bool: Always ``True``."""
        return True

    @property
    def upstream_path(self) -> str:
        """str: The Messages endpoint path."""
        return "/messages"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Build a trivial request body.

        Args:
            model: The model name.
            messages: The conversation.
            **kwargs: Ignored.

        Returns:
            The request body.
        """
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        """Return the response unchanged.

        Args:
            response_data: The upstream response.

        Returns:
            ``response_data``.
        """
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Wrap an upstream error.

        Args:
            status_code: The HTTP status.
            body: The error body.

        Returns:
            A plain exception naming both.
        """
        return Exception(f"Upstream error {status_code}: {body}")


def _sse(event_type: str, payload: dict) -> bytes:
    """Render one Anthropic SSE event.

    Args:
        event_type: The SSE ``event:`` name.
        payload: The JSON object for the ``data:`` line.

    Returns:
        The encoded event.
    """
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


_MESSAGE_START = _sse(
    "message_start",
    {
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "test-model"},
    },
)
_EMPTY_TEXT_START = _sse(
    "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
)
_BLOCK_STOP = _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
_MESSAGE_STOP = _sse("message_stop", {"type": "message_stop"})


def _stop(reason: str) -> bytes:
    """Render a ``message_delta`` carrying ``reason``.

    Args:
        reason: The stop reason.

    Returns:
        The encoded event.
    """
    return _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason}})


_CONTENTLESS = _MESSAGE_START + _EMPTY_TEXT_START + _BLOCK_STOP + _stop("end_turn") + _MESSAGE_STOP
_THINKING_ONLY = (
    _MESSAGE_START
    + _sse(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    )
    + _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    )
    + _BLOCK_STOP
    + _stop("end_turn")
    + _MESSAGE_STOP
)
# Deliberately non-canonical key order and spacing: a re-serialising hold would change these bytes.
_CONTENT = (
    b'event: message_start\ndata: {"message":{"role":"assistant","id":"msg_2","content":[]},'
    b'  "type":"message_start"}\n\n'
    + _EMPTY_TEXT_START
    + b'event: content_block_delta\ndata: {"index":0,"delta":{"text":"Hello","type":"text_delta"},'
    b'"type":"content_block_delta"}\n\n' + _BLOCK_STOP + _stop("end_turn") + _MESSAGE_STOP
)
_SSE_HEADERS = {"Content-Type": "text/event-stream"}


@pytest.fixture(autouse=True)
def _no_retry_delays(monkeypatch):
    """Zero every retry delay; the tests count attempts, never time them."""
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])


def _single_backend_server() -> BridgeServer:
    """Build a non-balancing bridge on one native upstream.

    Returns:
        The unstarted server.
    """
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=_NativeProvider("https://api0.example.com/v1"),
        resolved_key="key-0",
        model="test-model",
    )


def _balancing_server(base_urls: list[str] | None = None) -> BridgeServer:
    """Build a balancing bridge over two native upstreams.

    Args:
        base_urls: One base URL per backend; defaults to two distinct mocked hosts.

    Returns:
        The unstarted server.
    """
    backends = []
    for i, base_url in enumerate(base_urls or [f"https://api{i}.example.com/v1" for i in range(2)]):
        profile = Profile(name=f"profile-{i}", provider="openai", model="test-model", auth_ref=str(uuid.uuid4()))
        backends.append((_NativeProvider(base_url), f"key-{i}", profile))
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="test-model",
        backends=backends,
        backend_cooldown=300,
    )


def _upstream(i: int) -> str:
    """Return backend ``i``'s Messages URL.

    Args:
        i: Backend index.

    Returns:
        The URL the bridge posts to.
    """
    return f"https://api{i}.example.com/v1/messages"


def _posts(mocked: aioresponses) -> int:
    """Count upstream POSTs.

    Args:
        mocked: The active ``aioresponses`` context.

    Returns:
        The number of POSTs recorded.
    """
    return sum(len(calls) for (method, _), calls in mocked.requests.items() if method == "POST")


async def _stream(server: BridgeServer) -> tuple[int, bytes]:
    """Start ``server``, send one streaming Messages request, and stop it.

    Args:
        server: The bridge under test.

    Returns:
        The client-visible status and body.
    """
    port = await server.start_async()
    try:
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
            return resp.status, await resp.read()
    finally:
        await server.stop_async()


class TestEmptyReplyIsRetriedBeforeTheClientSeesIt:
    """R4 — nothing of an empty attempt reaches the client, and the ladder runs."""

    @pytest.mark.parametrize(
        "empty",
        [_CONTENTLESS, b"", _THINKING_ONLY],
        ids=["contentless_stream", "zero_bytes", "thinking_only"],
    )
    async def test_single_backend_retries_and_delivers_only_the_retry(self, empty):
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(_upstream(0), body=empty, headers=_SSE_HEADERS)
            m.post(_upstream(0), body=_CONTENT, headers=_SSE_HEADERS)
            status, body = await _stream(_single_backend_server())
            posts = _posts(m)
        assert posts == 2, "the empty attempt must be retried"
        assert status == 200
        assert body == _CONTENT, "the client must receive the retry's bytes, and nothing of the empty attempt"

    async def test_balancing_profile_retries_and_delivers_only_the_retry(self):
        """Selection among healthy backends is random, so the script follows arrival order, not URL."""
        replies = iter([_CONTENTLESS, _CONTENT])

        def _next_reply(url, **kwargs):
            """Answer each POST with the next scripted reply, whichever backend it reached."""
            return CallbackResult(body=next(replies), headers=_SSE_HEADERS)

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(_upstream(i), callback=_next_reply, repeat=True)
            server = _balancing_server()
            status, body = await _stream(server)
            posts = _posts(m)
        assert posts == 2, "the empty attempt must be retried on a balancing profile too"
        assert status == 200
        assert body == _CONTENT
        assert server._session_stats()["attempts"] == 2, "each retry must draw a backend again, as a failover can"
        assert all(h["healthy"] for h in server._backend_health), "an empty reply is retried, not quarantined"

    async def test_balancing_with_no_healthy_backend_gives_up_at_once(self):
        """The translated ladder's quirk, mirrored: nothing healthy and no final delay due means no retry."""
        server = _balancing_server()
        for health in server._backend_health:
            # Unhealthy but inside the fast-fail threshold, so selection gambles on one anyway.
            health.update(healthy=False, failed_at=time.monotonic(), cooldown=30)
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(_upstream(i), body=_CONTENTLESS, headers=_SSE_HEADERS, repeat=True)
            status, body = await _stream(server)
            posts = _posts(m)
        assert posts == 1
        assert status == 502
        assert json.loads(body)["error"]["reason"] == "empty_response"


class TestExhaustion:
    """R5 — every attempt empty ends in a pre-emission 502, never an empty stream."""

    @staticmethod
    def _assert_empty_response_error(status: int, body: bytes) -> None:
        """Assert the client received the D4 exhaustion error.

        Args:
            status: The client-visible HTTP status.
            body: The client-visible body.
        """
        assert status == 502
        error = json.loads(body)
        assert error["type"] == "error"
        assert error["error"]["type"] == "api_error"
        assert error["error"]["reason"] == "empty_response"
        assert "Kitty Bridge" in error["error"]["message"], "Q9: a downstream-only message names the product"

    async def test_single_backend_runs_the_whole_ladder_then_errors(self):
        budget = (server_module._MAX_RETRIES + 1) + len(server_module._EMPTY_FINAL_DELAYS)
        assert budget == 6, "the literal pins the ladder; the derivation above follows the constants"
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(_upstream(0), body=_CONTENTLESS, headers=_SSE_HEADERS, repeat=True)
            status, body = await _stream(_single_backend_server())
            posts = _posts(m)
        assert posts == budget, f"expected the whole {budget}-attempt ladder, saw {posts}"
        self._assert_empty_response_error(status, body)

    async def test_balancing_runs_the_whole_ladder_and_marks_no_backend_healthy(self):
        """Before the hold an empty native reply reset its backend's failure state as a success."""
        budget = (server_module._MAX_RETRIES + 1) * 2 + len(server_module._EMPTY_FINAL_DELAYS)
        server = _balancing_server()
        for health in server._backend_health:
            health["transport_error_count"] = 1  # a success would reset this to 0
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(_upstream(i), body=_CONTENTLESS, headers=_SSE_HEADERS, repeat=True)
            status, body = await _stream(server)
            posts = _posts(m)
        assert posts == budget == 10, f"expected the whole {budget}-attempt ladder, saw {posts}"
        self._assert_empty_response_error(status, body)
        assert [h["transport_error_count"] for h in server._backend_health] == [1, 1], "an empty reply is not a success"


class TestDiscardedAttemptsAreDiagnosable:
    """R12 — nothing of a discarded attempt is written, so the log is the only record of it."""

    async def test_discarded_attempt_logs_its_size_and_stop_reason(self, caplog):
        with (
            caplog.at_level(logging.DEBUG, logger="kitty.bridge.server"),
            aioresponses(passthrough=["http://127.0.0.1"]) as m,
        ):
            m.post(_upstream(0), body=_CONTENTLESS, headers=_SSE_HEADERS)
            m.post(_upstream(0), body=_CONTENT, headers=_SSE_HEADERS)
            await _stream(_single_backend_server())
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(f"{len(_CONTENTLESS)} bytes held, stop_reason=end_turn" in w for w in warnings), warnings
        debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("Discarded native reply head" in d and "message_start" in d for d in debug)


class TestTruncationBeforeContent:
    """R6 / D3 — a truncation no retry can improve fails at once with a 400."""

    @pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
    @pytest.mark.parametrize(
        "prefix",
        [_MESSAGE_START, _THINKING_ONLY.removesuffix(_stop("end_turn") + _MESSAGE_STOP)],
        ids=["nothing", "thinking"],
    )
    async def test_is_not_retried_or_failed_over(self, prefix, stop_reason):
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(
                    _upstream(i), body=prefix + _stop(stop_reason) + _MESSAGE_STOP, headers=_SSE_HEADERS, repeat=True
                )
            status, body = await _stream(_balancing_server())
            posts = _posts(m)
        assert posts == 1, f"no retry and no failover can improve a {stop_reason} truncation"
        assert status == 400
        error = json.loads(body)["error"]
        assert error["type"] == "invalid_request_error"
        assert error["reason"] == f"{stop_reason}_before_content"
        assert "Kitty Bridge" in error["message"]
        assert stop_reason in error["message"]

    async def test_ends_the_ladder_even_after_an_empty_attempt(self):
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(_upstream(0), body=_CONTENTLESS, headers=_SSE_HEADERS)
            m.post(_upstream(0), body=_MESSAGE_START + _stop("max_tokens") + _MESSAGE_STOP, headers=_SSE_HEADERS)
            m.post(_upstream(0), body=_CONTENT, headers=_SSE_HEADERS)
            status, body = await _stream(_single_backend_server())
            posts = _posts(m)
        assert posts == 2, "the truncated attempt must be the last one"
        assert status == 400
        assert json.loads(body)["error"]["reason"] == "max_tokens_before_content"


class TestPassThrough:
    """R3, R7, D2 — what the hold releases reaches the client exactly as sent."""

    async def test_content_stream_is_byte_identical_and_not_retried(self):
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(_upstream(0), body=_CONTENT, headers=_SSE_HEADERS, repeat=True)
            status, body = await _stream(_single_backend_server())
            posts = _posts(m)
        assert (status, posts) == (200, 1)
        assert body == _CONTENT

    async def test_released_stream_is_a_success_that_marks_its_backend_healthy(self):
        server = _balancing_server()
        for health in server._backend_health:
            health["transport_error_count"] = 1  # a success resets this to 0
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(_upstream(i), body=_CONTENT, headers=_SSE_HEADERS, repeat=True)
            status, body = await _stream(server)
        assert (status, body) == (200, _CONTENT)
        assert sorted(h["transport_error_count"] for h in server._backend_health) == [0, 1]

    async def test_upstream_error_event_before_content_is_forwarded_verbatim(self):
        upstream = _MESSAGE_START + _sse(
            "error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            for i in range(2):
                m.post(_upstream(i), body=upstream, headers=_SSE_HEADERS, repeat=True)
            status, body = await _stream(_balancing_server())
            posts = _posts(m)
        assert posts == 1, "the provider's own error is delivered, not retried"
        assert status == 200
        assert body == upstream


class TestFailuresWhileHeld:
    """A failure before release is a pre-emission failure, against a real upstream socket."""

    @staticmethod
    async def _serve(handler) -> tuple[web.AppRunner, str]:
        """Start a local upstream serving ``handler`` on the Messages path.

        Args:
            handler: The aiohttp request handler.

        Returns:
            The runner, to clean up, and the base URL to give a provider.
        """
        app = web.Application()
        app.router.add_post("/v1/messages", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner, f"http://127.0.0.1:{runner.addresses[0][1]}/v1"

    async def test_upstream_drop_while_held_is_retried_on_the_same_backend(self, monkeypatch):
        """Nothing reached the client, so the transport grace retry is still available."""
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.0, 0.0))
        calls: list[int] = []

        async def _handler(request: web.Request) -> web.StreamResponse:
            """Drop the connection after the preamble on the first call, succeed on the next."""
            calls.append(1)
            resp = web.StreamResponse(headers=_SSE_HEADERS)
            await resp.prepare(request)
            if len(calls) == 1:
                await resp.write(_MESSAGE_START + _EMPTY_TEXT_START)
                # close(), not abort(): on Windows abort() discards the queued preamble (KBR-189).
                request.transport.close()  # type: ignore[union-attr]
                return resp
            await resp.write(_CONTENT)
            return resp

        runner, base = await self._serve(_handler)
        try:
            server = BridgeServer(
                adapter=_StubLauncher(), provider=_NativeProvider(base), resolved_key="key-0", model="test-model"
            )
            status, body = await _stream(server)
        finally:
            await runner.cleanup()
        assert len(calls) == 2, "a drop before release must be retried, not closed off"
        assert status == 200
        assert body == _CONTENT, "the client must see only the retry, never the dropped preamble"

    async def test_client_disconnect_while_held_stops_reading_upstream(self):
        """Before the hold a failed write revealed a gone client; while held nothing is written."""
        calls: list[int] = []
        preamble_sent = asyncio.Event()
        upstream_released = asyncio.Event()

        async def _handler(request: web.Request) -> web.StreamResponse:
            """Stream thinking until the bridge lets go of the connection, bounded at 20 s."""
            calls.append(1)
            resp = web.StreamResponse(headers=_SSE_HEADERS)
            await resp.prepare(request)
            try:
                await resp.write(_THINKING_ONLY.removesuffix(_BLOCK_STOP + _stop("end_turn") + _MESSAGE_STOP))
                preamble_sent.set()
                for _ in range(2000):
                    await resp.write(
                        _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "thinking_delta", "thinking": "."},
                            },
                        )
                    )
                    await asyncio.sleep(0.01)
            except (ConnectionError, aiohttp.ClientConnectionResetError):
                upstream_released.set()
            return resp

        runner, base = await self._serve(_handler)
        server = _balancing_server(base_urls=[base, base])
        port = await server.start_async()
        try:
            session = aiohttp.ClientSession()
            request = asyncio.ensure_future(
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                )
            )
            await asyncio.wait_for(preamble_sent.wait(), timeout=10)
            request.cancel()
            await session.close()
            await asyncio.wait_for(upstream_released.wait(), timeout=10)
        finally:
            await server.stop_async()
            await runner.cleanup()
        assert len(calls) == 1, "a gone client must not be served a retry"
        for health in server._backend_health:
            assert (health["healthy"], health["failure_count"]) == (True, 0), "a client fault is not a backend's"

    @pytest.mark.parametrize("first_reply", ["zero_bytes", "dropped_before_body"])
    async def test_client_disconnect_before_a_retry_stops_the_ladder(self, first_reply, monkeypatch):
        """Neither reply offers a held chunk to check on, so the next attempt itself must check the client."""
        monkeypatch.setattr(server_module, "_TRANSPORT_GRACE_DELAYS", (0.0, 0.0))
        calls: list[int] = []
        request_arrived = asyncio.Event()
        client_gone = asyncio.Event()

        async def _handler(request: web.Request) -> web.StreamResponse:
            """Fail the first attempt only after the client has gone; answer empty afterwards."""
            calls.append(1)
            request_arrived.set()
            if len(calls) == 1:
                await asyncio.wait_for(client_gone.wait(), timeout=10)
            resp = web.StreamResponse(headers=_SSE_HEADERS)
            await resp.prepare(request)
            if len(calls) == 1 and first_reply == "dropped_before_body":
                # No held chunk, so only the pre-attempt check can stop the grace retry.
                request.transport.close()  # type: ignore[union-attr]
            return resp

        runner, base = await self._serve(_handler)
        server = BridgeServer(
            adapter=_StubLauncher(), provider=_NativeProvider(base), resolved_key="key-0", model="test-model"
        )
        port = await server.start_async()
        try:
            session = aiohttp.ClientSession()
            request = asyncio.ensure_future(
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                )
            )
            await asyncio.wait_for(request_arrived.wait(), timeout=10)
            request.cancel()
            await session.close()
            # Release the upstream only once the bridge itself has seen the connection close. aiohttp
            # keeps a handler listed until its task ends, so the signal is the lost transport.
            assert server._runner is not None and server._runner.server is not None
            deadline = time.monotonic() + 10
            while (
                any(h.transport is not None for h in server._runner.server.connections) and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.01)
            client_gone.set()
        finally:
            # Stopping waits for the in-flight handler, so every retry it would make lands first.
            await server.stop_async()
            await runner.cleanup()
        assert len(calls) == 1, "a gone client must not be served a retry"

    async def test_empty_retry_after_bytes_reached_the_client_closes_the_stream(self, monkeypatch):
        """KBR-183 still lets a timeout after release retry; an empty retry must then end the open stream."""
        monkeypatch.setattr(server_module, "_STREAM_READ_TIMEOUT", 1.0)
        calls: list[int] = []
        stop = asyncio.Event()

        async def _handler(request: web.Request) -> web.StreamResponse:
            """Send content then stall on the first call; answer empty on the next."""
            calls.append(1)
            resp = web.StreamResponse(headers=_SSE_HEADERS)
            await resp.prepare(request)
            if len(calls) == 1:
                await resp.write(_MESSAGE_START + _EMPTY_TEXT_START)
                await resp.write(
                    _sse(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
                    )
                )
                await stop.wait()
            return resp

        runner, base = await self._serve(_handler)
        try:
            server = BridgeServer(
                adapter=_StubLauncher(), provider=_NativeProvider(base), resolved_key="key-0", model="test-model"
            )
            status, body = await asyncio.wait_for(_stream(server), timeout=15)
        finally:
            stop.set()
            await runner.cleanup()
        assert len(calls) == 2
        assert status == 200
        assert body.startswith(_MESSAGE_START), "the released content reached the client"
        assert b"event: error" in body, "the open stream must end in a terminal error, not hang"
