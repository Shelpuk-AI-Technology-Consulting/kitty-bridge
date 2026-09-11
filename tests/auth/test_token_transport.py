"""Tests for the OAuth token transport seam.

KBR-161.  The recurring OAuth leg moved from ``aiohttp`` onto a dedicated
impersonating ``curl_cffi`` session, so that a provider sees one client for one
account instead of two.  :class:`~kitty.auth.token_transport.CurlTokenTransport`
is the seam that made the move possible without ``kitty.auth`` growing a
dependency on ``kitty.providers``.

**Why these run against a real local server rather than a fake.**  The rest of
the OAuth tests drive a fake transport, which is correct for *their* claim --
token state-machine logic.  That leaves nobody asserting that the seam actually
speaks the wire.  What belongs here is only what **kitty** owns: that the seam
extracts the status and body its callers branch on, that it forwards the headers
it is handed, and that its timeout is explicit rather than inherited.

The facts about ``curl_cffi`` itself -- form encoding, an explicit
``User-Agent`` beating impersonation, and ``proxies=`` versus ambient
``NO_PROXY`` -- are claims about a dependency, not about kitty, so
``TEST_SUITE.md`` §2.2 allocates them to a dependency contract at L2. They live in
``tests/test_curl_cffi_transport_contract.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import pytest
from aiohttp import web

from kitty.auth.token_transport import CurlTokenTransport

#: Generous enough that a healthy localhost round trip never trips it.
_TIMEOUT = 10.0


class _Recorder:
    """Captures what a stand-in token endpoint received."""

    def __init__(self) -> None:
        self.content_type: str | None = None
        self.user_agent: str | None = None
        self.body: str = ""
        self.hits: int = 0


async def _serve(handler: Callable, port: int) -> web.AppRunner:
    """Start a local aiohttp app on *port* and return its runner."""
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


@pytest.fixture()
async def token_endpoint() -> AsyncIterator[tuple[str, _Recorder]]:
    """A stand-in ``/oauth/token`` that records the request it was sent."""
    recorder = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        recorder.hits += 1
        recorder.content_type = request.headers.get("Content-Type")
        recorder.user_agent = request.headers.get("User-Agent")
        recorder.body = await request.text()
        status = int(request.query.get("status", "200"))
        return web.json_response({"ok": True}, status=status)

    runner = await _serve(handler, 8761)
    try:
        yield "http://127.0.0.1:8761/oauth/token", recorder
    finally:
        await runner.cleanup()


@pytest.fixture()
def transport() -> CurlTokenTransport:
    """A transport over a session impersonating the same target as the API leg."""
    import curl_cffi.requests

    from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

    return CurlTokenTransport(curl_cffi.requests.AsyncSession(impersonate=_CODEX_IMPERSONATE))


class TestItReturnsWhatTheCallersBranchOn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [200, 400])
    async def test_returns_the_status_and_the_body_text(self, transport, token_endpoint, status: int) -> None:
        """Both halves of the tuple ``OAuthSession`` branches on.

        ``_refresh`` decides between raising and parsing on the status, and
        builds its error from the body, so a transport that dropped either
        would turn a clean ``invalid_grant`` into an opaque failure.
        """
        url, _ = token_endpoint

        got_status, body = await transport.post_form(f"{url}?status={status}", {"a": "b"}, timeout=_TIMEOUT)

        assert got_status == status
        assert json.loads(body) == {"ok": True}

    @pytest.mark.asyncio
    async def test_the_headers_it_is_given_reach_the_request(self, transport, token_endpoint) -> None:
        """That an explicit UA *beats impersonation* is curl_cffi's promise and
        is pinned in ``tests/test_curl_cffi_transport_contract.py``; this asserts
        only that the seam forwards what it is handed."""
        url, recorder = token_endpoint

        await transport.post_form(url, {"a": "b"}, headers={"User-Agent": "sentinel/1.0"}, timeout=_TIMEOUT)

        assert recorder.user_agent == "sentinel/1.0"


class TestTheTimeoutIsExplicit:
    @pytest.mark.asyncio
    async def test_timeout_is_required_not_inherited(self, transport, token_endpoint) -> None:
        """Omitting the timeout is an error, not a fallback.

        ``curl_cffi`` happens to default to 30 s today, which is what the
        ``aiohttp`` leg used, but that is an undocumented dependency default
        under an unbounded version range.  ``get_valid_api_key`` holds its
        refresh lock across both POSTs, so an unbounded timeout would stall
        every concurrent request on the session, not just one.
        """
        url, _ = token_endpoint

        with pytest.raises(TypeError):
            await transport.post_form(url, {"a": "b"})  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_a_stalled_endpoint_raises_rather_than_hanging(self, transport) -> None:
        """The timeout is handed to curl, not merely accepted and dropped."""

        async def stall(request: web.Request) -> web.Response:
            await asyncio.sleep(30)
            return web.json_response({})

        runner = await _serve(stall, 8762)
        try:
            with pytest.raises(Exception) as exc_info:
                await transport.post_form("http://127.0.0.1:8762/oauth/token", {"a": "b"}, timeout=1.0)
            assert "timed out" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()
        finally:
            await runner.cleanup()
