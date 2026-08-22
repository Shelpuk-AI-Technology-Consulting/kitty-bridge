"""Tests for the ``X-Kitty-*`` per-response backend attribution headers.

Issue #26: a consumer that captures response headers can attribute an
individual call to the backend and real model that served it, rather than to
the alias it asked for.  The headers name the backend selected when the
response began; ``GET /stats`` remains the authoritative session record when a
stream fails over mid-flight.
"""

import uuid

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter

UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "whatever-the-upstream-says",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

UPSTREAM_SSE = (
    'data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


class StubProvider(ProviderAdapter):
    """Provider adapter with a fixed upstream base URL.

    Args:
        provider_type: Value reported by the :attr:`provider_type` property.
        base_url: Upstream base URL the bridge will call.
    """

    def __init__(self, provider_type: str = "stub", base_url: str = "https://api.example.com/v1") -> None:
        self._provider_type = provider_type
        self._base_url = base_url

    @property
    def provider_type(self) -> str:
        """Return the provider identifier."""
        return self._provider_type

    @property
    def default_base_url(self) -> str:
        """Return the upstream base URL."""
        return self._base_url

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Return a Chat Completions payload for the given model and messages."""
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        """Return the upstream response unchanged."""
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        """Return a generic exception for an upstream error status."""
        return Exception(f"Error {status_code}")


def _one_backend(name: str = "minimax-small", model: str = "MiniMax-M3", *, backup: bool = False):
    """Build a single-member balancing pool with a known name, model and tier.

    A one-member pool makes the selected backend deterministic, so the test can
    assert exact header values instead of "one of these".

    Args:
        name: Profile name the header should report.
        model: Real model the header should report.
        backup: Whether the member is a reserve-tier profile.

    Returns:
        A one-element list of ``(provider, resolved_key, profile)``.
    """
    profile = Profile(
        name=name,
        provider="openai",
        model=model,
        auth_ref=str(uuid.uuid4()),
        backup=backup,
    )
    return [(StubProvider(), "key-0", profile)]


def _make_server(backends=None, **kwargs) -> BridgeServer:
    """Build a bridge server over a stub provider.

    Args:
        backends: Balancing backends, or ``None`` for single-backend mode.
        **kwargs: Extra keyword arguments forwarded to ``BridgeServer``.

    Returns:
        An unstarted server.
    """
    provider = backends[0][0] if backends else StubProvider()
    return BridgeServer(
        adapter=None,
        provider=provider,
        resolved_key="key-0",
        model=backends[0][2].model if backends else "real-model",
        backends=backends,
        profile_name=kwargs.pop("profile_name", "solo"),
        **kwargs,
    )


async def _post(port: int, body: dict) -> dict[str, str]:
    """POST a chat completion through the bridge and return its response headers.

    Args:
        port: Port the bridge is listening on.
        body: Request body to send.

    Returns:
        The response headers as a plain dict.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    async with aiohttp.ClientSession() as session, session.post(url, json=body) as resp:
        assert resp.status == 200
        await resp.read()
        return dict(resp.headers)


class TestNonStreamingAttribution:
    """A plain JSON response names what served it."""

    @pytest.mark.asyncio
    async def test_headers_name_the_backend_model_and_tier(self) -> None:
        """A5.1: the three fields a consumer needs to attribute one call."""
        server = _make_server(_one_backend())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                request = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
                headers = await _post(port, request)
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "minimax-small"
        assert headers["X-Kitty-Model"] == "MiniMax-M3"
        assert headers["X-Kitty-Tier"] == "primary"

    @pytest.mark.asyncio
    async def test_the_header_model_is_not_the_requested_alias(self) -> None:
        """A5.1: the alias is exactly what the consumer must stop trusting."""
        server = _make_server(_one_backend())
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                request = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
                headers = await _post(port, request)
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Model"] != "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_reserve_tier_members_report_the_backup_tier(self) -> None:
        """A5.1: tier distinguishes "the plan served this" from "the metered key did"."""
        server = _make_server(_one_backend(backup=True))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                headers = await _post(port, {"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Tier"] == "backup"

    @pytest.mark.asyncio
    async def test_single_backend_mode_reports_the_profile_and_its_model(self) -> None:
        """A5.5: attribution does not depend on running a balancing pool."""
        server = _make_server(profile_name="solo-profile")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                headers = await _post(port, {"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "solo-profile"
        assert headers["X-Kitty-Model"] == "real-model"
        assert headers["X-Kitty-Tier"] == "primary"

    @pytest.mark.asyncio
    async def test_a_non_ascii_model_name_does_not_break_the_response(self) -> None:
        """A5.4: model ids are only checked for non-emptiness, so unicode is reachable.

        Header values are framed as latin-1; an unsanitised name would fail the
        whole response, costing the caller their answer for the sake of a
        diagnostic header.
        """
        server = _make_server(_one_backend(model="Ми́ниМакс"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                headers = await _post(port, {"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Model"].isascii()

    @pytest.mark.asyncio
    async def test_a_control_character_in_a_model_name_does_not_break_the_response(self) -> None:
        """A5.6: aiohttp rejects CR/LF/NUL in header values as header injection.

        Model ids are validated only for non-emptiness, so a newline is
        reachable — and unsanitised it would fail *every* response on that
        backend, with the observability feature as the sole cause.
        """
        server = _make_server(_one_backend(model="MiniMax\r\nX-Injected: yes"))
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                headers = await _post(port, {"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
        finally:
            await server.stop_async()

        assert "X-Injected" not in headers
        assert "\n" not in headers["X-Kitty-Model"]
        assert "\r" not in headers["X-Kitty-Model"]

    @pytest.mark.asyncio
    async def test_no_model_header_is_sent_when_no_override_is_configured(self) -> None:
        """A5.7: an empty header would assert an attribution the bridge cannot make."""
        server = BridgeServer(
            adapter=None,
            provider=StubProvider(),
            resolved_key="key-0",
            model=None,
            profile_name="passthrough",
        )
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                headers = await _post(port, {"model": "alias", "messages": [{"role": "user", "content": "hi"}]})
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "passthrough"
        assert "X-Kitty-Model" not in headers


class TestStreamingAttribution:
    """An SSE response carries the headers too, flushed before the first event."""

    @pytest.mark.asyncio
    async def test_streaming_response_carries_attribution_headers(self) -> None:
        """A5.2: the streaming path is the one Claude Code actually uses."""
        server = _make_server(_one_backend())
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        body = {"model": "alias", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=UPSTREAM_SSE,
                    headers={"Content-Type": "text/event-stream"},
                )
                async with aiohttp.ClientSession() as session, session.post(url, json=body) as resp:
                    assert resp.status == 200
                    assert resp.content_type == "text/event-stream"
                    headers = dict(resp.headers)
                    await resp.read()
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "minimax-small"
        assert headers["X-Kitty-Model"] == "MiniMax-M3"
        assert headers["X-Kitty-Tier"] == "primary"


class TestErrorResponseAttribution:
    """A failed response is the one a consumer most needs attributed."""

    @pytest.mark.asyncio
    async def test_an_internal_error_still_names_the_backend(self, monkeypatch) -> None:
        """A5.8: "why can't this model produce valid JSON?" starts here.

        The access-log middleware synthesises the 500 for an unhandled handler
        error. Attribution must sit outside it, or the one response that
        represents a model-capability failure is the one that names no model.
        """
        server = _make_server(_one_backend())

        async def _boom(self, request):
            """Fail after backend selection, as a handler bug would."""
            self._select_backend()
            raise RuntimeError("handler exploded")

        monkeypatch.setattr(BridgeServer, "_handle_chat_completions", _boom)
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        try:
            async with aiohttp.ClientSession() as session, session.post(url, json={"model": "alias"}) as resp:
                assert resp.status == 500
                headers = dict(resp.headers)
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "minimax-small"
        assert headers["X-Kitty-Model"] == "MiniMax-M3"

    @pytest.mark.asyncio
    async def test_an_unauthenticated_caller_learns_nothing_about_backends(self, tmp_path) -> None:
        """A5.9: attribution sits inside auth, so a 401 discloses no backend."""
        keys_file = tmp_path / "keys.txt"
        keys_file.write_text("sk-test-key\n", encoding="utf-8")
        server = _make_server(_one_backend(), keys_file=str(keys_file))
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        try:
            async with aiohttp.ClientSession() as session, session.post(url, json={"model": "alias"}) as resp:
                assert resp.status == 401
                assert "X-Kitty-Backend" not in resp.headers
        finally:
            await server.stop_async()


class TestMessagesPathAttribution:
    """The Messages path defers preparation, so its header means something subtler."""

    @pytest.mark.asyncio
    async def test_messages_stream_carries_attribution_headers(self) -> None:
        """A5.10: the Messages path is the one Claude Code actually uses.

        Its ``StreamResponse`` is prepared only once content is ready, so the
        header names the backend that produced the first byte rather than the
        one selected at request entry.
        """
        server = _make_server(_one_backend())
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        body = {"model": "alias", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    "https://api.example.com/v1/chat/completions",
                    body=UPSTREAM_SSE,
                    headers={"Content-Type": "text/event-stream"},
                )
                async with aiohttp.ClientSession() as session, session.post(url, json=body) as resp:
                    assert resp.status == 200
                    headers = dict(resp.headers)
                    await resp.read()
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "minimax-small"
        assert headers["X-Kitty-Model"] == "MiniMax-M3"
        assert headers["X-Kitty-Tier"] == "primary"

    @pytest.mark.asyncio
    async def test_messages_non_streaming_carries_attribution_headers(self) -> None:
        """A5.10: the non-streaming Messages response is stamped by the middleware."""
        server = _make_server(_one_backend())
        port = await server.start_async()
        url = f"http://127.0.0.1:{port}/v1/messages"
        body = {"model": "alias", "messages": [{"role": "user", "content": "hi"}]}
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post("https://api.example.com/v1/chat/completions", payload=UPSTREAM_RESPONSE)
                async with aiohttp.ClientSession() as session, session.post(url, json=body) as resp:
                    assert resp.status == 200
                    headers = dict(resp.headers)
                    await resp.read()
        finally:
            await server.stop_async()

        assert headers["X-Kitty-Backend"] == "minimax-small"
        assert headers["X-Kitty-Model"] == "MiniMax-M3"
