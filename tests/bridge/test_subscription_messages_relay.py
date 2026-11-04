"""Bridge-level tests for the Anthropic subscription exact Messages relay.

These cover the one claim no unit test can make: that the bytes Claude Code
posts to ``POST /v1/messages`` reach the Anthropic upstream byte-for-byte,
with Claude Code's end-to-end headers intact and nothing synthetic added.
Driving the real :class:`~kitty.bridge.server.BridgeServer` matters because
the defect this guards against lives in the wiring — a ``request.json()``
plus ``json=`` round-trip re-serializes the body, and the standard header
builder injects ``x-api-key``. Stubbing either seam would stub out the defect,
so these tests mock one layer lower, at the ``aiohttp`` transport the relay
posts through (via ``aioresponses``), and read the shipped bytes off the
recorded call.

**Layer.** ``tests/layers.py`` maps only ``tests/integration/`` away from the
``l1`` fallback, so this file collects as ``l1`` alongside every other
loopback test in this directory.
"""

from __future__ import annotations

from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.providers.anthropic_subscription import AnthropicSubscriptionAdapter

#: Upstream endpoint the subscription relay posts to.
_UPSTREAM_URL = "https://api.anthropic.com/v1/messages"

#: Intentionally non-canonical JSON: odd spacing and key order. A relay that
#: decodes and re-serializes cannot reproduce these bytes.
_RAW_BODY = (
    b'{\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
    b'  "max_tokens" : 8,\n  "model" : "claude-sonnet-4-6"\n}'
)

#: Streaming variant of the non-canonical body.
_RAW_STREAM_BODY = (
    b'{\n  "messages" : [ { "role" : "user", "content" : "Hello" } ],\n'
    b'  "max_tokens" : 8,\n  "stream" : true,\n  "model" : "claude-sonnet-4-6"\n}'
)

#: Upstream SSE bytes the fake Anthropic endpoint streams.
_SSE_REPLY = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant",'
    b'"model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,'
    b'"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)

#: Exact upstream reply bytes for the response-fidelity test.
_EXACT_REPLY_BYTES = (
    b'{"type" : "message" , "id" : "msg_exact" , "role" : "assistant" , '
    b'"content" : [] , "model" : "claude-sonnet-4-6" , "stop_reason" : "end_turn" , '
    b'"usage" : {"input_tokens" : 3 , "output_tokens" : 0}}'
)

#: ``_EXACT_REPLY_BYTES`` compressed with brotli (quality 11). Anthropic
#: answers with ``Content-Encoding: br`` whenever the request advertises
#: brotli, which Claude Code always does -- so the relay must be able to
#: decode it. The compressed form is committed as bytes rather than
#: compressed at test time: the test then needs no encoder to run, and
#: reproduces the production failure verbatim when no decoder is present.
_BROTLI_EXACT_REPLY_BYTES = bytes.fromhex(
    "1bc400088cd315f723b4cd27735ba95f76202886a52acf787ee379786698263140e0df763060fbef768345c3b23ca98c4"
    "24d3786510ce5623953ac8df90b0ced46c2a7d28857be01693c1468cd13c5d52f813204fecf7f2f358ae16e932ab5a0b036d9"
    "10f73457418a09540778a347cae15b800b3fc69a6acfe700"
)

#: ``_SSE_REPLY`` compressed with brotli (quality 11), for the streaming
#: counterpart of the decode guard.
_BROTLI_SSE_REPLY = bytes.fromhex(
    "1b9801001c076eacd8932875d3311544702d5fc1e1ab063af0ceaf2c169d00193cb7121a983da2ed5ce101daa2073ef0cdafd"
    "b84083c7bd6449dca718308d30aba2aff79f445963b2184c03afc32b5efd175285e7061407a3956a90b8c71597e25cd5addd"
    "b4ebd1efbdedd5447f5c00abc4c3285ebed3815227b9c3622ce9068e63d1dae60ad188d9f579a068efbed94ffff5fa468219e56c8"
    "a615abbc58fa1a2d031d286828a0561f565f91ca0113124f049e2bbdf1385dfb38f9b922"
)

#: Every header the relay must strip, plus the end-to-end set it must keep.
_ALL_STRIPPED = [
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
]

#: Minimal Anthropic Messages reply the fake upstream answers with.
_UPSTREAM_REPLY = {
    "id": "msg_relay_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

#: End-to-end headers Claude Code sends that the relay must preserve.
_INBOUND_HEADERS = {
    "Authorization": "Bearer sk-ant-oat-test",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "anthropic-beta": "test-beta",
    "anthropic-version": "2023-06-01",
}


def _make_server(provider: AnthropicSubscriptionAdapter | None = None) -> BridgeServer:
    """Build a bridge-mode server fronting the subscription provider.

    Args:
        provider: The subscription adapter to front. Defaults to the real
            Anthropic upstream; the brotli tests pass one pointed at a
            local upstream.

    Returns:
        A server bound to an ephemeral port, not yet started. The resolved
        key is empty: subscription launches carry no Kitty credential. The
        relay is opted in, as a Claude Code launch would.
    """
    return BridgeServer(
        adapter=None,
        provider=provider if provider is not None else AnthropicSubscriptionAdapter(),
        resolved_key="",
        host="127.0.0.1",
        port=0,
        allow_subscription_relay=True,
    )


def _shipped_kwargs(m: aioresponses) -> dict:
    """Return the recorded upstream POST kwargs.

    Args:
        m: The active ``aioresponses`` mock.

    Returns:
        The keyword arguments of the single recorded upstream POST.

    Raises:
        AssertionError: If upstream saw anything but exactly one POST, which
            would make byte-exactness assertions vacuous.
    """
    request_list = m.requests.get(("POST", aiohttp.client.URL(_UPSTREAM_URL)), [])
    assert len(request_list) == 1, f"expected exactly one upstream POST, got {len(request_list)}"
    return request_list[0].kwargs


class _LocalUpstreamProvider(AnthropicSubscriptionAdapter):
    """Subscription adapter whose upstream is a local test server.

    The relay's upstream address comes from the adapter's base URL, so
    pointing it at a loopback server lets a test serve a real HTTP
    response -- brotli-compressed bytes with a ``Content-Encoding``
    header -- instead of a mock that never exercises the decoder.

    Args:
        base_url: The base URL of the local upstream.
    """

    def __init__(self, base_url: str) -> None:
        """Point the adapter at a local upstream.

        Args:
            base_url: The base URL of the local upstream.
        """
        self._base_url = base_url

    @property
    def default_base_url(self) -> str:
        """Return the local upstream's base URL.

        Returns:
            The base URL the local test server was started on.
        """
        return self._base_url


async def _start_brotli_upstream(body: bytes, content_type: str) -> tuple[web.AppRunner, str]:
    """Serve one brotli-compressed body from a local upstream.

    Args:
        body: The already-compressed response bytes.
        content_type: The response content type.

    Returns:
        The runner and the base URL the upstream is reachable at. The
        caller closes the runner.
    """
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        """Answer one relay attempt with the compressed body.

        Args:
            request: The relay's upstream request.

        Returns:
            The compressed reply, labelled with its content encoding.
        """
        await request.read()
        return web.Response(body=body, headers={"Content-Type": content_type, "Content-Encoding": "br"})

    app.router.add_post("/v1/messages", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}"


class TestSubscriptionExactRelay:
    """The raw relay preserves Claude Code's bytes and headers (R7)."""

    @pytest.mark.asyncio()
    async def test_non_streaming_relay_ships_inbound_bytes_verbatim(self) -> None:
        """Post the inbound JSON bytes upstream without re-serialization."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        shipped = _shipped_kwargs(m)
        assert shipped.get("json") is None
        assert shipped.get("data") == _RAW_BODY

    @pytest.mark.asyncio()
    async def test_non_streaming_relay_preserves_end_to_end_headers(self) -> None:
        """Forward OAuth and Anthropic headers; strip routing and framing only."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={
                            "Content-Type": "application/json",
                            **_INBOUND_HEADERS,
                            "x-api-key": "inbound-key-kept",
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        sent_headers = {k.lower(): v for k, v in _shipped_kwargs(m).get("headers", {}).items()}
        assert sent_headers.get("authorization") == "Bearer sk-ant-oat-test"
        assert sent_headers.get("accept-encoding") == "gzip, deflate, br, zstd", (
            "Accept-Encoding never reached Anthropic: the upstream picks its "
            "response encoding from this header, so losing it here would retire "
            "the Brotli decoder the relay needs"
        )
        assert sent_headers.get("anthropic-beta") == "test-beta"
        assert sent_headers.get("anthropic-version") == "2023-06-01"
        assert sent_headers.get("x-api-key") == "inbound-key-kept"
        for stripped in ("host", "content-length", "transfer-encoding", "connection"):
            assert stripped not in sent_headers

    @pytest.mark.asyncio()
    async def test_non_streaming_relay_never_injects_api_key(self) -> None:
        """Send no x-api-key upstream when Claude Code did not send one."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

        sent_headers = {k.lower(): v for k, v in _shipped_kwargs(m).get("headers", {}).items()}
        assert "x-api-key" not in sent_headers

    @pytest.mark.asyncio()
    async def test_streaming_relay_forwards_upstream_sse_verbatim(self) -> None:
        """Pump upstream SSE bytes to the client without translation."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, body=_SSE_REPLY, headers={"Content-Type": "text/event-stream"})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_STREAM_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    client_bytes = await resp.content.read()
        finally:
            await server.stop_async()

        assert client_bytes == _SSE_REPLY
        shipped = _shipped_kwargs(m)
        assert shipped.get("json") is None
        assert shipped.get("data") == _RAW_STREAM_BODY

    @pytest.mark.asyncio()
    async def test_non_streaming_relay_decodes_brotli_encoded_upstream_reply(self) -> None:
        """Decode a brotli-compressed upstream reply and ship the plain bytes.

        Anthropic compresses its reply whenever the request advertises
        brotli, and Claude Code always advertises it. The relay forwards
        that header unchanged -- pinned by
        ``test_relay_headers_strips_full_hop_set`` -- so it must be able to
        decode the reply it asked for.

        The upstream here serves brotli unconditionally. It must not be
        made conditional on the request advertising it: a conditional
        upstream would answer with plain bytes when the relay stopped
        forwarding the header, and the assertion below would then pass
        without the decoder ever running.
        """
        runner, base_url = await _start_brotli_upstream(_BROTLI_EXACT_REPLY_BYTES, "application/json")
        server = _make_server(_LocalUpstreamProvider(base_url))
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_BODY,
                    headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                ) as resp,
            ):
                assert resp.status == 200, await resp.text()
                client_bytes = await resp.read()
        finally:
            await server.stop_async()
            await runner.cleanup()

        assert client_bytes == _EXACT_REPLY_BYTES

    @pytest.mark.asyncio()
    async def test_streaming_relay_decodes_brotli_encoded_upstream_sse(self) -> None:
        """Decode a brotli-compressed upstream SSE reply and forward the events.

        The streaming arm pumps the upstream body chunk by chunk, so a
        compressed reply reaches the client only if the decoder runs on
        that path too. The upstream serves brotli unconditionally for the
        reason the non-streaming test gives.
        """
        runner, base_url = await _start_brotli_upstream(_BROTLI_SSE_REPLY, "text/event-stream")
        server = _make_server(_LocalUpstreamProvider(base_url))
        port = await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=_RAW_STREAM_BODY,
                    headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                ) as resp,
            ):
                assert resp.status == 200, await resp.text()
                client_bytes = await resp.content.read()
        finally:
            await server.stop_async()
            await runner.cleanup()

        assert client_bytes == _SSE_REPLY


class TestSubscriptionRouteBoundary:
    """Subscription backends serve only POST /v1/messages (R7, R11)."""

    @pytest.mark.asyncio()
    async def test_chat_completions_with_subscription_backend_is_rejected_locally(self) -> None:
        """Reject a non-Messages route without upstream traffic."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
                    ) as resp,
                ):
                    assert resp.status == 400, await resp.text()
                    assert "subscription" in (await resp.text()).lower()
                assert ("POST", aiohttp.client.URL(_UPSTREAM_URL)) not in m.requests
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_responses_with_subscription_backend_is_rejected_locally(self) -> None:
        """Reject the Responses route without upstream traffic."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={"model": "claude-sonnet-4-6", "input": "Hi"},
                    ) as resp,
                ):
                    assert resp.status == 400, await resp.text()
                    assert "subscription" in (await resp.text()).lower()
                assert ("POST", aiohttp.client.URL(_UPSTREAM_URL)) not in m.requests
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_gemini_with_subscription_backend_is_rejected_locally(self) -> None:
        """Reject the Gemini route without upstream traffic."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1beta/models/gemini-2.0:generateContent",
                        json={"contents": [{"parts": [{"text": "Hi"}]}]},
                    ) as resp,
                ):
                    assert resp.status == 400, await resp.text()
                    assert "subscription" in (await resp.text()).lower()
                assert ("POST", aiohttp.client.URL(_UPSTREAM_URL)) not in m.requests
        finally:
            await server.stop_async()

    def test_relay_headers_strips_full_hop_set(self) -> None:
        """Strip every routing/framing header and keep every end-to-end one.

        ``Accept-Encoding`` is the one kept header that carries a product
        dependency: the upstream picks its response encoding from what this
        list advertises, so stripping it would silently retire the Brotli
        decoder the relay now needs. Pinning it here makes that retirement a
        failing test rather than a quiet drift.
        """
        server = _make_server()
        inbound = {
            "Authorization": "Bearer sk-ant-oat-test",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "anthropic-beta": "test-beta",
            "anthropic-version": "2023-06-01",
            "x-api-key": "inbound-key-kept",
            "Content-Type": "application/json",
            "X-Custom-End-To-End": "kept",
        }
        inbound.update({name: "strip-me" for name in _ALL_STRIPPED})

        headers = server._subscription_relay_headers(SimpleNamespace(headers=inbound))

        lowered = {k.lower(): v for k, v in headers.items()}
        for name in _ALL_STRIPPED:
            assert name not in lowered
        assert lowered["authorization"] == "Bearer sk-ant-oat-test"
        assert lowered.get("accept-encoding") == "gzip, deflate, br, zstd", (
            "the relay must forward Accept-Encoding unchanged: the upstream chooses "
            "its response encoding from this list, and dropping it would silently "
            "retire the Brotli decoder the relay needs"
        )
        assert lowered["anthropic-beta"] == "test-beta"
        assert lowered["anthropic-version"] == "2023-06-01"
        assert lowered["x-api-key"] == "inbound-key-kept"
        assert lowered["content-type"] == "application/json"
        assert lowered["x-custom-end-to-end"] == "kept"

    @pytest.mark.asyncio()
    async def test_non_streaming_upstream_failure_surfaces_envelope_with_upstream_status(self) -> None:
        """Translate a routine upstream failure, not a 500."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    status=429,
                    payload={"error": {"type": "rate_limit_error", "message": "slow down"}},
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 429, await resp.text()
                    envelope = await resp.json()
                    assert envelope["type"] == "error"
        finally:
            await server.stop_async()

        request_list = m.requests.get(("POST", aiohttp.client.URL(_UPSTREAM_URL)), [])
        assert len(request_list) == 1

    @pytest.mark.asyncio()
    async def test_non_streaming_success_returns_upstream_bytes_verbatim(self) -> None:
        """Pass the upstream body and request id through untouched."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    body=_EXACT_REPLY_BYTES,
                    headers={"Content-Type": "application/json", "request-id": "req_exact_123"},
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    assert await resp.read() == _EXACT_REPLY_BYTES
                    assert resp.headers.get("request-id") == "req_exact_123"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("raw", [b"[1, 2]", b'"just-a-string"', b"{oops", b""])
    async def test_malformed_and_non_object_bodies_rejected_without_upstream(self, raw: bytes) -> None:
        """Return 400 for bodies that are not JSON objects, touching nothing."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=raw,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 400, await resp.text()
                assert ("POST", aiohttp.client.URL(_UPSTREAM_URL)) not in m.requests
        finally:
            await server.stop_async()


class TestSubscriptionRelayContext:
    """The relay is a Claude Code launch privilege (R11)."""

    @pytest.mark.asyncio()
    async def test_messages_relay_without_opt_in_is_rejected_locally(self) -> None:
        """Refuse a subscription pool on a server no Claude launch opted in."""
        server = BridgeServer(
            adapter=None,
            provider=AnthropicSubscriptionAdapter(),
            resolved_key="",
            host="127.0.0.1",
            port=0,
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 400, await resp.text()
                    assert "subscription" in (await resp.text()).lower()
                assert ("POST", aiohttp.client.URL(_UPSTREAM_URL)) not in m.requests
        finally:
            await server.stop_async()


#: Upstream SSE bytes carrying usage in both ``message_start`` and the
#: cumulative ``message_delta`` — the shape the pump must read while
#: forwarding unchanged.
_SSE_REPLY_WITH_USAGE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_u","type":"message","role":"assistant",'
    b'"model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,'
    b'"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    b'"usage":{"output_tokens":5}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


def _make_server_with_usage_log(log_path: object) -> BridgeServer:
    """Build a relay server with usage logging pointed at ``log_path``.

    Args:
        log_path: File path the JSONL usage log is written to.

    Returns:
        A server bound to an ephemeral port, not yet started.
    """
    return BridgeServer(
        adapter=None,
        provider=AnthropicSubscriptionAdapter(),
        resolved_key="",
        host="127.0.0.1",
        port=0,
        allow_subscription_relay=True,
        logging_enabled=True,
        _usage_log_path=log_path,
    )


def _read_usage_entries(log_path: object) -> list[dict]:
    """Parse the JSONL usage log into entries.

    Args:
        log_path: File path the JSONL usage log is written to.

    Returns:
        One dict per completed entry, in write order.
    """
    import json
    from pathlib import Path

    text = Path(log_path).read_text(encoding="utf-8")  # type: ignore[arg-type]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestSubscriptionUsageAttribution:
    """Relay turns record their tokens like every other backend's (R8)."""

    @pytest.mark.asyncio()
    async def test_non_streaming_relay_records_anthropic_tokens(self, tmp_path: object) -> None:
        """An Anthropic usage block maps onto the attribution counters."""
        import asyncio
        from pathlib import Path

        log_path = Path(tmp_path) / "usage.jsonl"
        server = _make_server_with_usage_log(log_path)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, payload=_UPSTREAM_REPLY)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    await resp.read()
            await asyncio.sleep(0)
        finally:
            await server.stop_async()

        entries = _read_usage_entries(log_path)
        assert len(entries) == 1
        assert entries[0]["input_tokens"] == 10
        assert entries[0]["output_tokens"] == 5

    @pytest.mark.asyncio()
    async def test_streaming_relay_records_usage_from_sse_events(self, tmp_path: object) -> None:
        """The pump reads usage counters while forwarding bytes unchanged."""
        import asyncio
        from pathlib import Path

        log_path = Path(tmp_path) / "usage.jsonl"
        server = _make_server_with_usage_log(log_path)
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    _UPSTREAM_URL,
                    body=_SSE_REPLY_WITH_USAGE,
                    headers={"Content-Type": "text/event-stream"},
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_STREAM_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    client_bytes = await resp.content.read()
                    await asyncio.sleep(0)
        finally:
            await server.stop_async()

        # The wire contract is untouched: the client got exactly what Anthropic sent.
        assert client_bytes == _SSE_REPLY_WITH_USAGE

        entries = _read_usage_entries(log_path)
        assert len(entries) == 1
        # input from message_start, cumulative output from message_delta.
        assert entries[0]["input_tokens"] == 10
        assert entries[0]["output_tokens"] == 5


class TestSubscriptionAuthEnvelope:
    """Auth-class relay rejections advise the session, never an API key (R7)."""

    async def _envelope_for(self, status: int, body: object) -> tuple[int, dict]:
        """Drive one relay turn answering ``status``/``body``; return the reply.

        Args:
            status: The upstream failure status.
            body: The upstream failure body.

        Returns:
            The client-facing status and JSON envelope.
        """
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, status=status, payload=body if isinstance(body, dict) else {})
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    return resp.status, await resp.json()
        finally:
            await server.stop_async()

    @pytest.mark.asyncio()
    async def test_401_envelope_advises_claude_login_not_an_api_key(self) -> None:
        """An expired session reads as a session problem, not a key problem."""
        status, envelope = await self._envelope_for(
            401, {"error": {"type": "authentication_error", "message": "bad token"}}
        )

        assert status == 401
        message = envelope["error"]["message"]
        assert "claude auth login" in message
        assert "kitty setup" not in message
        assert "API key" not in message

    @pytest.mark.asyncio()
    async def test_403_plan_denial_envelope_reads_as_a_plan_limit(self) -> None:
        """An entitlement rejection names the plan, not a key rotation."""
        status, envelope = await self._envelope_for(
            403, {"error": {"message": "This model requires a subscription"}}
        )

        assert status == 403
        message = envelope["error"]["message"]
        assert "subscription" in message.lower()
        assert "kitty setup" not in message
        assert "API key" not in message


class TestSubscriptionTransportFailure:
    """A dropped wire to Anthropic still answers the Messages envelope (R3)."""

    @pytest.mark.asyncio()
    async def test_single_backend_transport_failure_answers_envelope_not_middleware_500(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connection failure is an api_error envelope, not the bridge's 500."""
        # The envelope is the contract here; keep the grace window instant.
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_DELAYS", (0.0,))
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_PERIOD", 1.0)
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, exception=aiohttp.ClientConnectionError("connection refused"))
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 500, await resp.text()
                    payload = await resp.json()
        finally:
            await server.stop_async()

        assert payload["type"] == "error"
        assert payload["error"]["type"] == "api_error"
        assert "failed" in payload["error"]["message"].lower()

    @pytest.mark.asyncio()
    async def test_single_backend_transport_failure_rides_out_grace_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single attempt is retried inside the grace window before the 500."""
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_DELAYS", (0.0, 0.0))
        monkeypatch.setattr("kitty.bridge.server._TRANSPORT_GRACE_PERIOD", 1.0)
        server = _make_server()
        port = await server.start_async()
        seen: list = []
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(_UPSTREAM_URL, exception=aiohttp.ClientConnectionError("connection refused"), repeat=True)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 500, await resp.text()
                    payload = await resp.json()
            seen = m.requests.get(("POST", aiohttp.client.URL(_UPSTREAM_URL)), [])
        finally:
            await server.stop_async()

        assert payload["error"]["type"] == "api_error"
        # Initial attempt plus one retry per grace delay: the single backend
        # was retried, not surfaced on the first blip.
        assert len(seen) == 3


class TestSubscriptionStreamLineCap:
    """The streaming usage pump's line buffer is bounded like the translators'."""

    @pytest.mark.asyncio()
    async def test_newline_less_line_beyond_the_cap_does_not_grow_unbounded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A huge newline-less line is dropped, the stream still completes."""
        monkeypatch.setattr("kitty.bridge.server._SSE_MAX_LINE_BYTES", 64)
        oversized_then_usage = (
            b"data: " + b"X" * 200 + b"\n"
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"id":"msg_c","type":"message","role":"assistant",'
            b'"model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,'
            b'"usage":{"input_tokens":7,"output_tokens":1}}}\n\n'
            b"event: message_stop\n"
            b'data: {"type":"message_stop"}\n\n'
        )
        server = _make_server()
        port = await server.start_async()
        try:
            with (
                caplog.at_level("WARNING", logger="kitty.bridge.server"),
                aioresponses(passthrough=["http://127.0.0.1"]) as m,
            ):
                m.post(
                    _UPSTREAM_URL,
                    body=oversized_then_usage,
                    headers={"Content-Type": "text/event-stream"},
                )
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        data=_RAW_STREAM_BODY,
                        headers={"Content-Type": "application/json", **_INBOUND_HEADERS},
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    client_bytes = await resp.content.read()
        finally:
            await server.stop_async()

        assert client_bytes == oversized_then_usage
        # The distinguishing signal the reply promised: the drop is logged, so
        # a regression deleting the cap goes red here rather than passing.
        assert any("exceeded the cap" in record.message for record in caplog.records)
