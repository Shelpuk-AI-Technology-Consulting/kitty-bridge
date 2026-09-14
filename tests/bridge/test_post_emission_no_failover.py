"""Once bytes reach the client, a failure ends the turn instead of retrying it (KBR-183).

§11 Q14(a) of ``.system_design/TEST_SUITE.md``: after the first byte is on
the wire the bridge neither retries nor fails over. It closes whatever block
is open, sends one terminal error, and lets the agent retry the turn. A second
attempt on the same response would repeat text the client already has, or
splice tool arguments from two attempts, into a stream where every event is
valid and the conversation is corrupt.

This is the reduced form of §6.3.1's streaming-recovery rows that the ticket
allows until T-I7 (KBR-99) lands. The upstream is a real local aiohttp server
so the failure is a genuine ``sock_read`` timeout. The stall waits on an event
that is never set, so no test depends on how long anything takes: the oracle
is how many requests the upstream received and which events the client got.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable

import aiohttp
import pytest
from aiohttp import web
from aioresponses import aioresponses

from kitty.bridge import server as server_module
from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter

from .test_client_disconnect_health import (
    _CC_CONTENT_CHUNK,
    _assert_all_backends_untouched,
    _make_server,
    _post_stream,
    _StubLauncher,
    _StubProvider,
    short_grace,  # noqa: F401 — pytest fixture, used by name
)

# Anthropic Messages SSE up to and including the first text delta.
_NATIVE_CONTENT_CHUNK = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
    b'"type":"message","role":"assistant","content":[],"model":"test-model",'
    b'"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"Hi"}}\n\n'
)

# Chat Completions SSE: some text, then a tool call whose arguments stop half-way.
_CC_TOOL_CHUNK = (
    _CC_CONTENT_CHUNK + b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
    b'"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"q\\":"}}]},'
    b'"finish_reason":null}],"model":"test-model"}\n\n'
)

# Content and its finish reason in one chunk: translated, buffered, and not yet written.
_CC_CONTENT_WITH_FINISH_CHUNK = (
    b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hi"},'
    b'"finish_reason":"stop"}],"model":"test-model"}\n\n'
)

# Text, then a tool call that arrives in the same chunk as the finish reason: its start is buffered, never sent.
_CC_TOOL_WITH_FINISH_CHUNK = (
    b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
    b'"type":"function","function":{"name":"lookup","arguments":"{}"}}]},'
    b'"finish_reason":"tool_calls"}],"model":"test-model"}\n\n'
)

# The error text the retry loop sends for a timeout; the outer handler would send something else.
_TIMEOUT_MESSAGE_PREFIX = "Upstream provider timed out"

# A complete Chat Completions stream, for the attempt that is allowed to succeed.
_CC_FULL_STREAM = (
    _CC_CONTENT_CHUNK + b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
    b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
    b"data: [DONE]\n\n"
)

# KBR-236: an empty finish chunk first — the translator judges the reply empty, buffers
# its fallback events and resets — then content the client is written live.
_CC_EMPTY_FINISH_THEN_CONTENT = (
    b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
    b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
    + _CC_CONTENT_CHUNK
    + b"data: [DONE]\n\n"
)

# The same shape with a tool call as the late content: the empty verdict precedes a live
# tool_calls delta, so half-delivered arguments are on the wire when the turn ends.
_CC_EMPTY_FINISH_THEN_TOOL = (
    b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
    b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
    + b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
    b'"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"q\\":"}}]},'
    b'"finish_reason":null}],"model":"test-model"}\n\n'
    + b"data: [DONE]\n\n"
)

Responder = Callable[[web.Request, int], Awaitable[web.StreamResponse]]


@pytest.fixture
def fast_stall(monkeypatch, short_grace):  # noqa: F811 — the imported fixture, requested by name
    """Make a silent upstream time out quickly and every retry delay negligible."""
    monkeypatch.setattr(server_module, "_STREAM_READ_TIMEOUT", 1)
    monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.001)


class _Upstream:
    """A local upstream that counts requests and can stall on demand.

    Args:
        path: The route to serve, e.g. ``/v1/chat/completions``.
        respond: Called with the request and its 0-based ordinal; returns the response.
    """

    def __init__(self, path: str, respond: Responder) -> None:
        self.path = path
        self.respond = respond
        self.requests = 0
        self.stall = asyncio.Event()
        self._runner: web.AppRunner | None = None

    async def __aenter__(self) -> str:
        """Start serving and return the base URL a provider should use.

        Returns:
            The upstream's ``http://127.0.0.1:<port>/v1`` base URL.
        """

        async def _handler(request: web.Request) -> web.StreamResponse:
            """Count the request and hand it to the responder.

            Args:
                request: The incoming upstream request.

            Returns:
                Whatever the responder returns.
            """
            ordinal = self.requests
            self.requests += 1
            return await self.respond(request, ordinal)

        app = web.Application()
        app.router.add_post(self.path, _handler)
        self._runner = web.AppRunner(app, shutdown_timeout=0.1)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        return f"http://127.0.0.1:{self._runner.addresses[0][1]}/v1"

    async def __aexit__(self, *exc_info: object) -> None:
        """Release every stalled handler, then stop serving.

        Args:
            *exc_info: The exception triple, unused.
        """
        # Stalled handlers hold their connections open; let them return first.
        self.stall.set()
        assert self._runner is not None
        await self._runner.cleanup()

    async def send_then_stall(self, request: web.Request, chunk: bytes) -> web.StreamResponse:
        """Send ``chunk`` as a 200 SSE response, then go silent.

        Args:
            request: The incoming upstream request.
            chunk: The bytes to deliver before stalling.

        Returns:
            The response, once the test releases the stall.
        """
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(chunk)
        await self.stall.wait()
        return resp


def _single_server(base_url: str, native: bool = False) -> BridgeServer:
    """Build a BridgeServer with one provider and no balancing.

    Args:
        base_url: The upstream base URL.
        native: Whether the provider speaks the Messages API natively.

    Returns:
        The unstarted server.
    """
    return BridgeServer(
        adapter=_StubLauncher(),
        provider=_StubProvider(base_url, native=native),
        resolved_key="key-0",
        model="test-model",
    )


def _build(mode: str, base_url: str, native: bool = False) -> BridgeServer:
    """Build the server shape a parametrised case names.

    Args:
        mode: ``"balanced"`` for two backends on one upstream, ``"single"`` for no balancing.
        base_url: The upstream base URL.
        native: Whether the provider speaks the Messages API natively.

    Returns:
        The unstarted server.
    """
    if mode == "balanced":
        return _make_server(2, native=native, base_urls=[base_url, base_url])
    return _single_server(base_url, native=native)


async def _run(server: BridgeServer) -> bytes:
    """Start ``server``, send one streaming Messages request, and stop it.

    Args:
        server: The bridge to drive.

    Returns:
        The raw body the client received.
    """
    port = await server.start_async()
    try:
        return await _post_stream(port)
    finally:
        await server.stop_async()


def _events(body: bytes) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``(event name, data)`` pairs.

    Args:
        body: The raw bytes the client received.

    Returns:
        Every named event with its decoded JSON payload, in order.
    """
    parsed = []
    for block in body.decode().split("\n\n"):
        lines = block.strip().splitlines()
        name = next((line[len("event: ") :] for line in lines if line.startswith("event: ")), None)
        data = next((line[len("data: ") :] for line in lines if line.startswith("data: ")), None)
        if name is not None and data is not None:
            parsed.append((name, json.loads(data)))
    return parsed


# ── /v1/messages, aiohttp upstream ───────────────────────────────────────


class TestMessagesTimeoutAfterContent:
    """A ``sock_read`` stall after content reached the client ends the turn."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["balanced", "single"])
    async def test_translated_timeout_after_content_sends_one_upstream_request(self, fast_stall, mode):
        """No second attempt exists, so none can repeat what the client saw."""
        upstream = _Upstream("/v1/chat/completions", lambda req, _: upstream.send_then_stall(req, _CC_CONTENT_CHUNK))
        async with upstream as base:
            body = await _run(_build(mode, base))

        assert upstream.requests == 1, f"the upstream was asked {upstream.requests} times after text was emitted"
        assert body.count(b'"text": "Hi"') == 1
        assert _events(body)[-1][1]["error"]["message"].startswith(_TIMEOUT_MESSAGE_PREFIX)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["balanced", "single"])
    async def test_native_timeout_after_content_sends_one_upstream_request(self, fast_stall, mode):
        """The native passthrough gets the same guarantee as the translated path."""
        upstream = _Upstream("/v1/messages", lambda req, _: upstream.send_then_stall(req, _NATIVE_CONTENT_CHUNK))
        async with upstream as base:
            body = await _run(_build(mode, base, native=True))

        assert upstream.requests == 1, f"the upstream was asked {upstream.requests} times after text was emitted"
        assert [name for name, _ in _events(body)] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "error",
        ]
        assert _events(body)[-1][1]["error"]["message"].startswith(_TIMEOUT_MESSAGE_PREFIX)

    @pytest.mark.asyncio
    async def test_translated_timeout_after_content_closes_the_block_then_errors(self, fast_stall):
        """The open text block is closed, then one error ends the turn — not a synthesised finish."""
        upstream = _Upstream("/v1/chat/completions", lambda req, _: upstream.send_then_stall(req, _CC_CONTENT_CHUNK))
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert [(name, data.get("index")) for name, data in _events(body)] == [
            ("message_start", None),
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("error", None),
        ]

    @pytest.mark.asyncio
    async def test_translated_timeout_mid_tool_arguments_closes_the_tool_block_then_errors(self, fast_stall):
        """Half-sent tool arguments are never continued by another attempt; the tool block is closed."""
        upstream = _Upstream("/v1/chat/completions", lambda req, _: upstream.send_then_stall(req, _CC_TOOL_CHUNK))
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert [(name, data.get("index")) for name, data in _events(body)][-2:] == [
            ("content_block_stop", 1),
            ("error", None),
        ]
        assert body.count(b"input_json_delta") == 1

    @pytest.mark.asyncio
    async def test_translated_timeout_after_the_finish_chunk_closes_the_block_then_errors(self, fast_stall):
        """A stall between the finish chunk and ``[DONE]`` still closes the block the client has open."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Send text, then its finish chunk, then stall.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position, unused.

            Returns:
                The response, once the test releases the stall.
            """
            return await upstream.send_then_stall(request, _CC_CONTENT_CHUNK + _CC_CONTENT_WITH_FINISH_CHUNK)

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert [(name, data.get("index")) for name, data in _events(body)][-2:] == [
            ("content_block_stop", 0),
            ("error", None),
        ]

    @pytest.mark.asyncio
    async def test_timeout_after_a_buffered_block_start_closes_only_what_the_client_saw(self, fast_stall):
        """A block opened inside the unsent finish chunk was never started on the wire, so it is not stopped."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Send text, then a finish chunk that also opens a tool call, then stall.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position, unused.

            Returns:
                The response, once the test releases the stall.
            """
            return await upstream.send_then_stall(request, _CC_CONTENT_CHUNK + _CC_TOOL_WITH_FINISH_CHUNK)

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert [(name, data.get("index")) for name, data in _events(body)] == [
            ("message_start", None),
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("error", None),
        ]

    @pytest.mark.asyncio
    async def test_timeout_after_content_charges_the_stalled_backend(self, fast_stall):
        """The backend that stalled is quarantined, and no other backend is touched."""
        upstream = _Upstream("/v1/chat/completions", lambda req, _: upstream.send_then_stall(req, _CC_CONTENT_CHUNK))
        async with upstream as base:
            server = _build("balanced", base)
            await _run(server)

        quarantined = [health for health in server._backend_health if not health["healthy"]]
        assert [(h["last_failure_kind"], h["cooldown"]) for h in quarantined] == [("hard", 300)]

    @pytest.mark.asyncio
    async def test_timeout_before_content_still_fails_over(self, fast_stall):
        """A stall before any byte reached the client is still recovered silently on another backend."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Stall the first request before any byte; serve every later one in full.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position.

            Returns:
                The response.
            """
            if ordinal == 0:
                await upstream.stall.wait()
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(_CC_FULL_STREAM)
            return resp

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert upstream.requests == 2
        assert [name for name, _ in _events(body)][-1] == "message_stop"
        assert b"event: error" not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["balanced", "single"])
    async def test_failover_after_a_buffered_finish_still_ends_the_stream(self, fast_stall, mode):
        """An attempt whose finish was buffered but never sent must not stop the next attempt from finishing."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Buffer a finish on the first request and stall; serve every later one in full.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position.

            Returns:
                The response.
            """
            if ordinal == 0:
                return await upstream.send_then_stall(request, _CC_CONTENT_WITH_FINISH_CHUNK)
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(_CC_FULL_STREAM)
            return resp

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            body = await _run(_build(mode, base))

        assert [name for name, _ in _events(body)][-1] == "message_stop"


class TestEmptyVerdictAfterContent:
    """An empty-response verdict after content reached the client ends the turn (KBR-236).

    An empty finish chunk sets ``response_was_empty``, buffers its fallback events
    and resets the translator; content arriving after it is written to the client
    live. The empty-response ladder must then read ``sr`` like every other
    post-emission branch (§11 Q14(a)): one upstream request, one terminal error.
    """

    @staticmethod
    async def _serve_stream(request: web.Request, stream: bytes) -> web.StreamResponse:
        """Serve ``stream`` as one complete 200 SSE response.

        Args:
            request: The incoming upstream request.
            stream: The SSE bytes to deliver.

        Returns:
            The completed response.
        """
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(stream)
        return resp

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["balanced", "single"])
    async def test_empty_verdict_after_content_sends_one_upstream_request(self, fast_stall, mode):
        """No second attempt follows content the client already saw."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Serve the empty-verdict stream in full.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position, unused.

            Returns:
                The completed response.
            """
            return await self._serve_stream(request, _CC_EMPTY_FINISH_THEN_CONTENT)

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            server = _build(mode, base)
            body = await _run(server)

        assert upstream.requests == 1, f"the upstream was asked {upstream.requests} times after text was emitted"
        assert body.count(b'"text": "Hi"') == 1
        assert server._model_stats("test-model")["completions"] == 0, "the errored turn was counted as a completion"
        names = [name for name, _ in _events(body)]
        assert names.count("message_start") == 1
        assert names[-1] == "error"
        assert "message_stop" not in names
        assert _events(body)[-1][1]["error"]["message"].startswith("Kitty Bridge received an empty response")

    @pytest.mark.asyncio
    async def test_empty_verdict_after_content_closes_the_block_then_errors(self, fast_stall):
        """The open text block is closed, then one error ends the turn — the buffered fallback never runs."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Serve the empty-verdict stream in full.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position, unused.

            Returns:
                The completed response.
            """
            return await self._serve_stream(request, _CC_EMPTY_FINISH_THEN_CONTENT)

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert [(name, data.get("index")) for name, data in _events(body)] == [
            ("message_start", None),
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("error", None),
        ]

    @pytest.mark.asyncio
    async def test_empty_verdict_after_content_quarantines_no_backend(self, fast_stall):
        """An empty reply is a polite completion: the pool stays healthy and uncharged."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Serve the empty-verdict stream in full.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position, unused.

            Returns:
                The completed response.
            """
            return await self._serve_stream(request, _CC_EMPTY_FINISH_THEN_CONTENT)

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            server = _make_server(2, base_urls=[base, base])
            body = await _run(server)

        assert upstream.requests == 1, f"the pool was asked {upstream.requests} times after text was emitted"
        assert [name for name, _ in _events(body)][-1] == "error"
        _assert_all_backends_untouched(server)

    @pytest.mark.asyncio
    async def test_empty_verdict_after_content_closes_the_tool_block_then_errors(self, fast_stall):
        """A half-delivered tool call is closed, not spliced or continued, then one error ends the turn."""

        async def _respond(request: web.Request, ordinal: int) -> web.StreamResponse:
            """Serve the empty-verdict stream whose late content is a tool call.

            Args:
                request: The incoming upstream request.
                ordinal: The request's 0-based position, unused.

            Returns:
                The completed response.
            """
            return await self._serve_stream(request, _CC_EMPTY_FINISH_THEN_TOOL)

        upstream = _Upstream("/v1/chat/completions", _respond)
        async with upstream as base:
            body = await _run(_build("balanced", base))

        assert [(name, data.get("index")) for name, data in _events(body)][-2:] == [
            ("content_block_stop", 0),
            ("error", None),
        ]
        assert body.count(b"input_json_delta") == 1


# ── /v1/responses and Gemini, custom-transport providers ─────────────────


class _WritesThenFails(ProviderAdapter):
    """Custom-transport provider that may stream one delta, then raises a timeout.

    Args:
        calls: Shared list every instance appends its call to, across backends.
        write_first: Whether to write a delta to the client before failing.
    """

    def __init__(self, calls: list[int], write_first: bool) -> None:
        self._calls = calls
        self._write_first = write_first

    @property
    def provider_type(self) -> str:
        """Return the provider identifier."""
        return "writes-then-fails"

    @property
    def default_base_url(self) -> str:
        """Return a base URL that is never contacted."""
        return "https://unused.example.com/v1"

    @property
    def use_custom_transport(self) -> bool:
        """Route through :meth:`stream_request` instead of aiohttp."""
        return True

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Return a minimal request body.

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
            The same response.
        """
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Return a generic error.

        Args:
            status_code: The upstream status.
            body: The upstream error body.

        Returns:
            The exception to raise.
        """
        return Exception(f"Error {status_code}")

    async def stream_request(self, cc_request: dict, write) -> None:
        """Record the call, optionally write a delta, then time out.

        Args:
            cc_request: The Chat Completions request.
            write: The callback that writes bytes to the client.

        Raises:
            asyncio.TimeoutError: Always, as a mid-stream stall would.
        """
        self._calls.append(1)
        if self._write_first:
            await write(b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n')
        raise asyncio.TimeoutError()


_CUSTOM_TRANSPORT_ROUTES = [
    pytest.param(
        "/v1/responses",
        {"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}], "stream": True},
        id="responses",
    ),
    pytest.param(
        "/v1beta/models/m:streamGenerateContent?alt=sse",
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        id="gemini",
    ),
]


async def _run_custom_transport(path: str, payload: dict, first_writes: bool, calls: list[int]) -> bytes:
    """Drive two custom-transport backends through one streaming request.

    Only the first backend selected fails; which one that is does not matter,
    because both share ``calls`` and the second one reached always succeeds.

    Args:
        path: The bridge route, including any query string.
        payload: The request body.
        first_writes: Whether the failing attempt writes a delta before raising.
        calls: Receives one entry per ``stream_request`` call.

    Returns:
        The raw body the client received.
    """

    class _FirstFailsThenServes(_WritesThenFails):
        """Fails on the first call made to any backend, serves every later one."""

        async def stream_request(self, cc_request: dict, write) -> None:
            """Fail like the parent on the first call overall, then stream a complete delta.

            Args:
                cc_request: The Chat Completions request.
                write: The callback that writes bytes to the client.
            """
            if not calls:
                await super().stream_request(cc_request, write)
            calls.append(1)
            await write(b'data: {"type":"response.output_text.delta","delta":"Served"}\n\n')

    backends = [
        (
            _FirstFailsThenServes(calls, write_first=first_writes),
            f"key-{i}",
            Profile(name=f"profile-{i}", provider="openai_subscription", model="m", auth_ref=str(uuid.uuid4())),
        )
        for i in range(2)
    ]
    server = BridgeServer(adapter=None, provider=backends[0][0], resolved_key="key-0", model="m", backends=backends)
    port = await server.start_async()
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"http://127.0.0.1:{port}{path}", json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp,
        ):
            return await resp.read()
    finally:
        await server.stop_async()


class TestCustomTransportFailureAfterBytes:
    """The Responses and Gemini custom-transport branches honour the same rule."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("path", "payload"), _CUSTOM_TRANSPORT_ROUTES)
    async def test_custom_transport_failure_after_bytes_does_not_fail_over(self, path, payload):
        """A backend that already wrote to the client is not followed by another backend's attempt."""
        calls: list[int] = []
        body = await _run_custom_transport(path, payload, first_writes=True, calls=calls)

        assert len(calls) == 1, f"stream_request ran {len(calls)} times after bytes reached the client"
        assert body.count(b'"delta":"Hi"') == 1
        assert b"Served" not in body
        assert b'"error"' in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("path", "payload"), _CUSTOM_TRANSPORT_ROUTES)
    async def test_custom_transport_failure_before_bytes_still_fails_over(self, path, payload):
        """A backend that failed before writing anything is still replaced by the next one."""
        calls: list[int] = []
        body = await _run_custom_transport(path, payload, first_writes=False, calls=calls)

        assert len(calls) == 2
        assert b"Served" in body


# ── /v1/responses and /v1/gemini, standard (aiohttp) providers ────────────
#
# KBR-247: the same empty-finish-then-content shape that KBR-236 closed on the
# translated /v1/messages branch reaches the standard /v1/responses and
# /v1/gemini streaming paths too. The bridge translates both inbound protocols
# to Chat Completions on the wire, so the upstream we mock is the same
# `/v1/chat/completions` endpoint the Messages tests use — only the inbound
# request shape differs.


_RESPONSES_REQUEST = {
    "model": "test-model",
    "input": [{"type": "message", "role": "user", "content": "hi"}],
    "stream": True,
}

_GEMINI_REQUEST = {
    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
}


def _make_responses_server(n: int, base_urls: list[str] | None = None) -> BridgeServer:
    """Build a BridgeServer with ``n`` openai backends, every protocol route registered."""
    if n == 1:
        provider = _StubProvider("https://api0.example.com/v1")
        return BridgeServer(
            adapter=None,
            provider=provider,
            resolved_key="key-0",
            model="test-model",
        )
    backends = []
    for i in range(n):
        base = base_urls[i] if base_urls else f"https://api{i}.example.com/v1"
        provider = _StubProvider(base)
        profile = Profile(
            name=f"profile-{i}",
            provider="openai",
            model="test-model",
            auth_ref=str(uuid.uuid4()),
        )
        backends.append((provider, f"key-{i}", profile))
    return BridgeServer(
        adapter=None,
        provider=backends[0][0],
        resolved_key=backends[0][1],
        model="test-model",
        backends=backends,
    )


async def _drive_stream(
    server: BridgeServer,
    path: str,
    payload: dict,
    upstream_body: bytes,
) -> tuple[bytes, int]:
    """Drive one inbound stream through ``server`` against ``upstream_body``.

    Args:
        server: The bridge to start.
        path: The inbound route, e.g. ``"/v1/responses"``.
        payload: The inbound JSON body.
        upstream_body: The full Chat Completions SSE body the upstream returns.

    Returns:
        The raw body the client received and the number of upstream POSTs
        recorded by ``aioresponses``.
    """
    port = await server.start_async()
    try:
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(
                "https://api0.example.com/v1/chat/completions",
                body=upstream_body,
                headers={"Content-Type": "text/event-stream"},
                repeat=True,
            )
            m.post(
                "https://api1.example.com/v1/chat/completions",
                body=upstream_body,
                headers={"Content-Type": "text/event-stream"},
                repeat=True,
            )
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}{path}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp,
            ):
                body = await resp.read()
        posts = sum(
            len(calls) for (method, _), calls in m.requests.items() if method == "POST"
        )
        return body, posts
    finally:
        await server.stop_async()


class TestResponsesEmptyVerdictAfterContent:
    """An empty-response verdict on /v1/responses after content reached the client ends the turn (KBR-247)."""

    @pytest.fixture(autouse=True)
    def _no_retry_delays(self, monkeypatch):
        """Zero every retry delay; the tests count attempts, never time them."""
        monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.0)
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["balanced", "single"])
    async def test_empty_verdict_after_content_sends_one_upstream_request(self, mode):
        """No second attempt follows content the client already saw on /v1/responses."""
        n_backends = 2 if mode == "balanced" else 1
        server = _make_responses_server(n_backends)
        body, posts = await _drive_stream(
            server, "/v1/responses", _RESPONSES_REQUEST, _CC_EMPTY_FINISH_THEN_CONTENT
        )

        assert posts == 1, f"the upstream was asked {posts} times after text was emitted"
        assert body.count(b'"delta": "Hi"') == 1
        names = [name for name, _ in _events(body)]
        assert names[-1] == "response.completed"
        assert body.count(b'"status": "incomplete"') == 1
        # Exactly one error event, and it sits between the late content and the
        # closing response.completed — never a second attempt's delta after it.
        error_events = [data for name, data in _events(body) if name == "error"]
        assert len(error_events) == 1
        assert error_events[0]["code"] == "upstream_error"
        last_delta = max(
            (idx for idx, (name, _) in enumerate(_events(body)) if name == "response.output_text.delta"),
            default=-1,
        )
        first_error = next(idx for idx, (name, _) in enumerate(_events(body)) if name == "error")
        assert last_delta < first_error, "the error event must follow the late content"
        # AC-1: the errored turn is not counted as a completion. The guard
        # `break`s out of the retry loop before reaching `_log_usage`, the only
        # call path that increments `_model_stats["completions"]` on this
        # route — pin the invariant here so a future refactor cannot move the
        # log call into the guard path and silently regress AC-1.
        assert server._model_stats("test-model")["completions"] == 0

    @pytest.mark.asyncio
    async def test_empty_verdict_after_content_quarantines_no_backend(self):
        """An empty reply is a polite completion: the pool stays healthy and uncharged."""
        server = _make_responses_server(2)
        body, posts = await _drive_stream(
            server, "/v1/responses", _RESPONSES_REQUEST, _CC_EMPTY_FINISH_THEN_CONTENT
        )

        assert posts == 1, f"the pool was asked {posts} times after text was emitted"
        _assert_all_backends_untouched(server)
        names = [name for name, _ in _events(body)]
        assert names[-1] == "response.completed"

    @pytest.mark.asyncio
    async def test_empty_verdict_after_content_closes_the_tool_call(self):
        """A late tool-call delta after the empty verdict must not splice arguments from a second attempt."""
        late_tool = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
            + b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
            b'"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"q\\":"}}]},'
            b'"finish_reason":null}],"model":"test-model"}\n\n'
            + b"data: [DONE]\n\n"
        )
        server = _make_responses_server(2)
        body, posts = await _drive_stream(
            server, "/v1/responses", _RESPONSES_REQUEST, late_tool
        )

        assert posts == 1, f"the upstream was asked {posts} times after a tool-call delta was emitted"
        # Exactly one arguments-delta event — the half-sent argument string was
        # never completed by a second attempt, and the bridge closed the item
        # rather than continuing it.
        names = [name for name, _ in _events(body)]
        assert names.count("response.function_call_arguments.delta") == 1
        # The synthesized completion closed the tool-call item and the response
        # itself with status="incomplete" — every closed item carries that
        # marker, so we assert at least one and the closing response.completed.
        assert body.count(b'"status": "incomplete"') >= 1
        assert names[-1] == "response.completed"
        # No second attempt's function_call_arguments — only one delta and one
        # `output_item.done` for the tool call.
        assert names.count("response.output_item.done") == 1
        assert body.count(b'"{\\"q\\":') == 1

    @pytest.mark.asyncio
    async def test_pre_emission_empty_response_still_fails_over(self):
        """A pre-emission empty verdict still retries today, unchanged by this guard."""
        pure_empty = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
            b"data: [DONE]\n\n"
        )
        server = _make_responses_server(2)
        # Pin the draw: backend-0 first (empty), then backend-1 (full). The
        # random draw could otherwise open on the healthy backend and pass
        # without ever exercising the ladder.
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
            idx = next(_pick)
            provider, key, profile = server._backends[idx]
            server._active_provider = provider
            server._active_key = key
            server._active_model = profile.model
            server._active_provider_config = profile.provider_config or {}
            server._current_backend_idx = idx

        server._select_backend = _fixed_select

        # Front up empty on backend-0 and the success stream on backend-1. The
        # pre-emission verdict runs the ladder, so the pool is asked at least
        # twice — assert the lower bound, not the upper, because the ladder
        # could keep retrying past the second attempt and still complete cleanly.
        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(
                "https://api0.example.com/v1/chat/completions",
                body=pure_empty,
                headers={"Content-Type": "text/event-stream"},
                repeat=True,
            )
            m.post(
                "https://api1.example.com/v1/chat/completions",
                body=_CC_FULL_STREAM,
                headers={"Content-Type": "text/event-stream"},
                repeat=True,
            )
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json=_RESPONSES_REQUEST,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    body = await resp.read()
            finally:
                await server.stop_async()
        posts = sum(
            len(calls) for (method, _), calls in m.requests.items() if method == "POST"
        )
        assert posts >= 2, "the pre-emission ladder must still run"
        names = [name for name, _ in _events(body)]
        assert names[-1] == "response.completed"
        assert b'"status": "incomplete"' not in body, "the successful retry must complete, not error"

    @pytest.mark.asyncio
    async def test_empty_verdict_then_in_stream_error_emits_one_error(self):
        """The empty-then-content + post-content in-stream error shape writes exactly one error event.

        Q14(a) (KBR-247 design-review): on `_stream_responses` the in-stream
        error exhaustion arm used to fall through to the empty-verdict check,
        so with KBR-247 in place it would have produced two terminal error
        events on the same stream. The fix is a `break` inside the
        `events_emitted` branch of the exhaustion arm. This test pins the
        invariant: one error event, then `response.completed status:
        "incomplete"` from the post-loop synthesize.
        """
        # empty finish chunk → content chunk → in-stream error chunk → [DONE].
        # The in-stream error chunk is `_is_upstream_stream_error` matching,
        # so `_stream_responses` will mark `stream_error=True` while
        # `events_emitted` is also True from the late content.
        empty_then_content_then_error = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
            + _CC_CONTENT_CHUNK
            + b'data: {"error":{"code":"upstream_error","message":"upstream hiccup"}}\n\n'
            b"data: [DONE]\n\n"
        )
        server = _make_responses_server(2)
        body, posts = await _drive_stream(
            server,
            "/v1/responses",
            _RESPONSES_REQUEST,
            empty_then_content_then_error,
        )

        assert posts == 1, f"the upstream was asked {posts} times after content was emitted"
        # Exactly one error event: the exhaustion arm's terminal write.
        error_events = [data for name, data in _events(body) if name == "error"]
        assert len(error_events) == 1, f"the body should carry one error event, found {len(error_events)}"
        # The synthesized completion closed the turn under status="incomplete".
        names = [name for name, _ in _events(body)]
        assert names[-1] == "response.completed"
        assert body.count(b'"status": "incomplete"') >= 1
        # The late text reached the client once and was never spliced with a
        # second attempt's bytes.
        assert body.count(b'"delta": "Hi"') == 1


class TestGeminiEmptyVerdictAfterContent:
    """An empty-response verdict on /v1/gemini after content reached the client ends the turn (KBR-247)."""

    @pytest.fixture(autouse=True)
    def _no_retry_delays(self, monkeypatch):
        """Zero every retry delay; the tests count attempts, never time them."""
        monkeypatch.setattr(server_module, "_BACKOFF_BASE", 0.0)
        monkeypatch.setattr(server_module, "_EMPTY_FINAL_DELAYS", [0.0, 0.0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["balanced", "single"])
    async def test_empty_verdict_after_content_sends_one_upstream_request(self, mode):
        """No second attempt follows content the client already saw on /v1/gemini."""
        n_backends = 2 if mode == "balanced" else 1
        server = _make_responses_server(n_backends)
        body, posts = await _drive_stream(
            server,
            "/v1beta/models/test-model:streamGenerateContent",
            _GEMINI_REQUEST,
            _CC_EMPTY_FINISH_THEN_CONTENT,
        )

        assert posts == 1, f"the upstream was asked {posts} times after text was emitted"
        # The Gemini wire carries the text delta verbatim through the Chat
        # Completions-to-Gemini converter; "Hi" appears once.
        assert body.count(b'"Hi"') == 1
        # The terminal shape is one SSE `data: {"error":...}` event and EOF —
        # nothing after it.
        error_blocks = body.split(b"\n\n")
        error_payloads = [
            json.loads(block.split(b"data: ", 1)[1].decode())
            for block in error_blocks
            if block.startswith(b"data: {") and b'"error":' in block
        ]
        assert len(error_payloads) == 1, f"the body should carry one error payload, found {len(error_payloads)}"
        assert error_payloads[0]["error"]["code"] == 502
        assert error_payloads[0]["error"]["message"].startswith("Kitty Bridge received an empty response")
        # AC-1: the errored turn is not counted as a completion.
        assert server._model_stats("test-model")["completions"] == 0

    @pytest.mark.asyncio
    async def test_empty_verdict_after_content_quarantines_no_backend(self):
        """An empty reply is a polite completion: the pool stays healthy and uncharged."""
        server = _make_responses_server(2)
        body, _posts = await _drive_stream(
            server,
            "/v1beta/models/test-model:streamGenerateContent",
            _GEMINI_REQUEST,
            _CC_EMPTY_FINISH_THEN_CONTENT,
        )

        _assert_all_backends_untouched(server)
        error_blocks = body.split(b"\n\n")
        error_payloads = [
            json.loads(block.split(b"data: ", 1)[1].decode())
            for block in error_blocks
            if block.startswith(b"data: {") and b'"error":' in block
        ]
        assert len(error_payloads) == 1

    @pytest.mark.asyncio
    async def test_empty_verdict_after_text_does_not_deliver_a_late_tool_call(self):
        """A late tool-call delta after the empty verdict is never spliced onto the wire.

        Gemini's translator buffers tool-call arguments into `_tool_call_buffers`
        and only emits a `functionCall` part in the finish chunk. With the
        empty finish chunk first, the post-reset translator starts a new
        buffer for the late tool-call delta; that buffer never crosses the
        wire because no subsequent finish chunk arrives. The guard's terminal
        shape is the same one the empty-text shape produces, and the late
        tool-call bytes are lost rather than spliced from a second attempt.
        """
        # empty finish chunk → live text delta (sets `_request_emitted=True`)
        # → late tool-call delta (buffered, no live SSE event) → [DONE].
        empty_then_text_then_tool = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
            + _CC_CONTENT_CHUNK
            + b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
            b'"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\\"q\\":"}}]},'
            b'"finish_reason":null}],"model":"test-model"}\n\n'
            b"data: [DONE]\n\n"
        )
        server = _make_responses_server(2)
        body, posts = await _drive_stream(
            server,
            "/v1beta/models/test-model:streamGenerateContent",
            _GEMINI_REQUEST,
            empty_then_text_then_tool,
        )

        assert posts == 1, f"the upstream was asked {posts} times after text was emitted"
        # The text crossed the wire once. The tool-call bytes did not.
        assert body.count(b'"Hi"') == 1
        assert b"functionCall" not in body
        # Exactly one terminal error, in the route's own convention.
        error_blocks = body.split(b"\n\n")
        error_payloads = [
            json.loads(block.split(b"data: ", 1)[1].decode())
            for block in error_blocks
            if block.startswith(b"data: {") and b'"error":' in block
        ]
        assert len(error_payloads) == 1
        assert error_payloads[0]["error"]["code"] == 502
        # AC-1: the errored turn is not counted as a completion.
        assert server._model_stats("test-model")["completions"] == 0

    @pytest.mark.asyncio
    async def test_pre_emission_empty_response_still_fails_over(self):
        """A pre-emission empty verdict on /v1/gemini still walks the ladder, unchanged by this guard.

        The Gemini route's empty-verdict check is structurally separate from
        the Responses route's (`_stream_gemini` lines 5776-5809 versus
        `_stream_responses` lines 3837-3871), and `test_empty_response_retry.py`
        only exercises `/v1/messages`. A regression that broke only the
        Gemini pre-emission ladder would pass the existing suite, so this
        class carries the same negative control the Responses class carries:
        an empty verdict before any byte reached the client must still walk
        the failover / retry ladder to completion on a healthy backend.
        """
        pure_empty = (
            b'data: {"id":"c1","choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}],"model":"test-model","usage":null}\n\n'
            b"data: [DONE]\n\n"
        )
        server = _make_responses_server(2)
        # Pin the draw: backend-0 first (empty), then backend-1 (full).
        _pick = iter([0, 1])

        def _fixed_select(self=server):
            """Select backends in the scripted order, without the weighted draw."""
            idx = next(_pick)
            provider, key, profile = server._backends[idx]
            server._active_provider = provider
            server._active_key = key
            server._active_model = profile.model
            server._active_provider_config = profile.provider_config or {}
            server._current_backend_idx = idx

        server._select_backend = _fixed_select

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post(
                "https://api0.example.com/v1/chat/completions",
                body=pure_empty,
                headers={"Content-Type": "text/event-stream"},
                repeat=True,
            )
            m.post(
                "https://api1.example.com/v1/chat/completions",
                body=_CC_FULL_STREAM,
                headers={"Content-Type": "text/event-stream"},
                repeat=True,
            )
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1beta/models/test-model:streamGenerateContent",
                        json=_GEMINI_REQUEST,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp,
                ):
                    body = await resp.read()
            finally:
                await server.stop_async()
        posts = sum(
            len(calls) for (method, _), calls in m.requests.items() if method == "POST"
        )
        assert posts >= 2, "the pre-emission ladder must still walk on Gemini"
        # The successful retry completed normally — no terminal error event
        # crossed the wire (the empty verdict was caught before any byte).
        error_blocks = body.split(b"\n\n")
        error_payloads = [
            json.loads(block.split(b"data: ", 1)[1].decode())
            for block in error_blocks
            if block.startswith(b"data: {") and b'"error":' in block
        ]
        assert error_payloads == [], "the successful retry must complete, not error"
