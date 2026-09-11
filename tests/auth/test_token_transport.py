"""Tests for the OAuth token transport seam.

KBR-161.  The recurring OAuth leg moved from ``aiohttp`` to the adapter's
impersonating ``curl_cffi`` session, so that a provider sees one client for one
account instead of two.  :class:`~kitty.auth.token_transport.CurlTokenTransport`
is the seam that made the move possible without ``kitty.auth`` growing a
dependency on ``kitty.providers``.

**Why these run against a real local server rather than a fake.**  The rest of
the OAuth tests drive a fake transport, which is correct for *their* claim --
token state-machine logic.  It leaves nobody asserting that the seam actually
speaks the wire.  Every fact below is a property of ``curl_cffi``, not of our
code, and each is load-bearing:

* ``data=dict`` must form-encode exactly as ``aiohttp`` did, or the token
  endpoint rejects the grant;
* an explicit ``User-Agent`` must **survive** curl-impersonate's own header
  injection, or the fix silently does nothing;
* ``NO_PROXY`` must be pinned, because it **defeats** an explicit ``proxies=``
  mapping -- measured, not assumed (§2.4 of the task requirements).

``pyproject.toml`` declares ``curl_cffi>=0.7`` with no upper bound, so these are
the tests that turn red if an upgrade changes any of it.
"""

from __future__ import annotations

import asyncio
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


class TestItSpeaksTheWire:
    @pytest.mark.asyncio
    async def test_posts_form_encoded_with_percent_escaping(self, transport, token_endpoint) -> None:
        """``data=dict`` leaves as ``application/x-www-form-urlencoded``.

        The token endpoint takes form parameters, not JSON.  A ``curl_cffi``
        release that changed this default would break every grant.
        """
        url, recorder = token_endpoint

        await transport.post_form(url, {"grant_type": "refresh_token", "refresh_token": "a&b"}, timeout=_TIMEOUT)

        assert recorder.content_type == "application/x-www-form-urlencoded"
        assert recorder.body == "grant_type=refresh_token&refresh_token=a%26b"

    @pytest.mark.asyncio
    async def test_an_explicit_user_agent_survives_impersonation(self, transport, token_endpoint) -> None:
        """The Codex UA must beat curl-impersonate's own ``User-Agent``.

        This is the single fact the whole of KBR-161's R2 rests on: the session
        is built with ``impersonate=chrome136``, which supplies a Chrome
        user-agent of its own.  If that won, the fix would be invisible.
        """
        url, recorder = token_endpoint

        await transport.post_form(
            url, {"grant_type": "refresh_token"}, headers={"User-Agent": "codex_cli_rs/9.9.9 (Test 1; x86_64)"}, timeout=_TIMEOUT
        )

        assert recorder.user_agent == "codex_cli_rs/9.9.9 (Test 1; x86_64)"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [200, 400])
    async def test_returns_the_status_and_the_body_text(self, transport, token_endpoint, status: int) -> None:
        """Both halves of the tuple the callers branch on.

        ``_refresh`` decides between raising and parsing on the status, and
        builds its error from the body, so a transport that dropped either
        would turn a clean ``invalid_grant`` into an opaque failure.
        """
        url, _ = token_endpoint

        got_status, text = await transport.post_form(f"{url}?status={status}", {"a": "b"}, timeout=_TIMEOUT)

        assert got_status == status
        assert '"ok": true' in text.replace(" ", "").replace('"ok":true', '"ok": true')


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


class TestAmbientProxyEnvironment:
    """Pins the measured ``NO_PROXY`` behaviour that moving transport imports.

    ``aiohttp`` ignores proxy environment variables unless ``trust_env=True``;
    ``curl_cffi`` honours them.  So the OAuth leg, which was immune, is not any
    more.  The exposure is **pre-existing and adapter-wide** -- the API leg
    passes ``proxies=`` identically -- so this is not a regression this change
    invents, but it must not be a surprise either.  Recorded in
    ``TEST_SUITE.md`` §4.5 and filed separately.
    """

    @pytest.fixture()
    async def proxy_and_target(self) -> AsyncIterator[tuple[str, str, _Recorder, _Recorder]]:
        direct, through_proxy = _Recorder(), _Recorder()

        async def direct_handler(request: web.Request) -> web.Response:
            direct.hits += 1
            return web.json_response({})

        async def proxy_handler(request: web.Request) -> web.Response:
            through_proxy.hits += 1
            return web.json_response({})

        d = await _serve(direct_handler, 8763)
        p = await _serve(proxy_handler, 8764)
        try:
            yield "http://127.0.0.1:8763/oauth/token", "http://127.0.0.1:8764", direct, through_proxy
        finally:
            await d.cleanup()
            await p.cleanup()

    @pytest.fixture(autouse=True)
    def _no_ambient_proxy(self, monkeypatch) -> None:
        """The developer's own shell must not decide this test's result."""
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            monkeypatch.delenv(var, raising=False)

    @pytest.mark.asyncio
    async def test_proxies_mapping_is_honoured(self, transport, proxy_and_target) -> None:
        """The baseline: an explicit mapping routes through the gateway."""
        target, proxy, direct, through_proxy = proxy_and_target

        await transport.post_form(target, {"a": "b"}, timeout=_TIMEOUT, proxies={"http": proxy, "https": proxy})

        assert (through_proxy.hits, direct.hits) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_matching_no_proxy_defeats_the_explicit_mapping(
        self, transport, proxy_and_target, monkeypatch
    ) -> None:
        """**The exposure.**  ``NO_PROXY`` wins over ``proxies=``, silently.

        A user with ``NO_PROXY`` covering the auth host bypasses a configured
        egress gateway with no error and no log line.  Pinned so that a
        ``curl_cffi`` upgrade which changes the precedence -- in either
        direction -- turns this red rather than quietly altering containment.
        """
        target, proxy, direct, through_proxy = proxy_and_target
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")

        await transport.post_form(target, {"a": "b"}, timeout=_TIMEOUT, proxies={"http": proxy, "https": proxy})

        assert (through_proxy.hits, direct.hits) == (0, 1)

    @pytest.mark.asyncio
    async def test_a_non_matching_no_proxy_leaves_the_mapping_alone(
        self, transport, proxy_and_target, monkeypatch
    ) -> None:
        """The complement, so the test above is about *matching*, not presence."""
        target, proxy, direct, through_proxy = proxy_and_target
        monkeypatch.setenv("NO_PROXY", "example.invalid")

        await transport.post_form(target, {"a": "b"}, timeout=_TIMEOUT, proxies={"http": proxy, "https": proxy})

        assert (through_proxy.hits, direct.hits) == (1, 0)
