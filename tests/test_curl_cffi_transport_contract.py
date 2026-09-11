"""Dependency contract: what ``curl_cffi`` promises the OpenAI subscription leg.

``TEST_SUITE.md`` §6.2.4 owes a contract for ``curl_cffi``, whose declared range
is ``>=0.7`` with **no upper bound** -- the weakest pin in the repo. §2.2's
allocation table puts these claims here rather than beside the code that relies
on them, for one reason: *they are claims about a dependency, not about kitty*.
Nothing in ``src/kitty`` can be changed to make any of them true.

Four facts are load-bearing, and every one of them is invisible in the code that
depends on it:

1. ``data=dict`` form-encodes. The token endpoint takes form parameters; a
   release that switched the default to JSON would break every OAuth grant.
2. An explicit ``User-Agent`` **beats** the one ``impersonate=`` injects. KBR-161
   consists of setting that header; if impersonation won, the fix would be
   invisible and the guard in ``tests/test_oauth_leg_identity.py`` would pass
   while the wire carried Chrome's user-agent.
3. ``proxies=`` is honoured -- invariant I3's whole basis for the curl transport.
4. **Ambient ``NO_PROXY`` defeats ``proxies=``, and ``CURLOPT_NOPROXY`` defeats
   ``NO_PROXY``.** This is the sharp one. ``aiohttp`` ignores the proxy
   environment unless ``trust_env=True``; ``curl_cffi`` reads it. So a user's
   ``NO_PROXY`` matching the upstream host sends provider traffic straight past
   a configured egress gateway, with no error and no log line -- prompts on the
   API leg, and, since KBR-161 moved it, refresh tokens and API keys on the
   OAuth leg. ``kitty`` closes that by setting ``CURLOPT_NOPROXY`` explicitly at
   session construction (``openai_subscription._new_curl_session``). These tests
   are what make that remedy a checked fact rather than a hopeful comment, and
   they close the ambient-proxy half of gap **G10**.

Every assertion runs against real local sockets, because a mock of a dependency
proves nothing about the dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import curl_cffi.requests
import pytest
from aiohttp import web
from curl_cffi import CurlOpt

from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

pytestmark = pytest.mark.l2

_TIMEOUT = 10.0


class _Recorder:
    """Counts requests and remembers the last one's shape."""

    def __init__(self) -> None:
        self.hits = 0
        self.content_type: str | None = None
        self.user_agent: str | None = None
        self.body = ""


async def _serve(handler: Callable, port: int) -> web.AppRunner:
    """Start a local aiohttp app on *port* and return its runner."""
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


@pytest.fixture()
async def target() -> AsyncIterator[tuple[str, _Recorder]]:
    """A local stand-in for the token endpoint."""
    rec = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        rec.hits += 1
        rec.content_type = request.headers.get("Content-Type")
        rec.user_agent = request.headers.get("User-Agent")
        rec.body = await request.text()
        return web.json_response({"ok": True})

    runner = await _serve(handler, 8781)
    try:
        yield "http://127.0.0.1:8781/oauth/token", rec
    finally:
        await runner.cleanup()


@pytest.fixture()
async def proxy() -> AsyncIterator[tuple[str, _Recorder]]:
    """A local stand-in for the egress gateway."""
    rec = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        rec.hits += 1
        return web.json_response({"ok": True})

    runner = await _serve(handler, 8782)
    try:
        yield "http://127.0.0.1:8782", rec
    finally:
        await runner.cleanup()


@pytest.fixture(autouse=True)
def _no_ambient_proxy(monkeypatch) -> None:
    """The developer's own shell must not decide these results."""
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


def _session(**kwargs) -> curl_cffi.requests.AsyncSession:
    """Build a session impersonating the same target the adapter does."""
    return curl_cffi.requests.AsyncSession(impersonate=_CODEX_IMPERSONATE, **kwargs)


class TestTheRequestShape:
    @pytest.mark.asyncio
    async def test_a_dict_body_is_form_encoded(self, target) -> None:
        """``data=dict`` must remain ``application/x-www-form-urlencoded``."""
        url, rec = target

        await _session().post(url, data={"grant_type": "refresh_token", "refresh_token": "a&b"}, timeout=_TIMEOUT)

        assert rec.content_type == "application/x-www-form-urlencoded"
        assert rec.body == "grant_type=refresh_token&refresh_token=a%26b"

    @pytest.mark.asyncio
    async def test_an_explicit_user_agent_beats_impersonation(self, target) -> None:
        """The header kitty sets must reach the wire, not Chrome's."""
        url, rec = target

        await _session().post(
            url, data={"a": "b"}, headers={"User-Agent": "codex_cli_rs/9.9.9 (Test 1; x86_64)"}, timeout=_TIMEOUT
        )

        assert rec.user_agent == "codex_cli_rs/9.9.9 (Test 1; x86_64)"

    @pytest.mark.asyncio
    async def test_the_impersonation_target_still_exists(self, target) -> None:
        """A removed ``chrome136`` would otherwise surface as a runtime error."""
        url, rec = target

        await _session().post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert rec.hits == 1


class TestEgressContainment:
    """I3: the ``proxies=`` mapping must mean what the invariant says."""

    @pytest.mark.asyncio
    async def test_the_proxies_mapping_is_honoured(self, target, proxy) -> None:
        url, direct = target
        proxy_url, gateway = proxy

        await _session(proxies={"http": proxy_url, "https": proxy_url}).post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (gateway.hits, direct.hits) == (1, 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["NO_PROXY", "no_proxy"])
    async def test_a_matching_no_proxy_defeats_the_mapping(self, target, proxy, monkeypatch, var: str) -> None:
        """**The exposure**, pinned in the direction it actually behaves.

        If a future ``curl_cffi`` reverses this, the remedy below becomes
        unnecessary and this test says so loudly rather than leaving a
        superstition in the session constructor.
        """
        url, direct = target
        proxy_url, gateway = proxy
        monkeypatch.setenv(var, "127.0.0.1")

        await _session(proxies={"http": proxy_url, "https": proxy_url}).post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (gateway.hits, direct.hits) == (0, 1)

    @pytest.mark.asyncio
    async def test_a_non_matching_no_proxy_leaves_the_mapping_alone(self, target, proxy, monkeypatch) -> None:
        """So the case above is about *matching*, not about the variable existing."""
        url, direct = target
        proxy_url, gateway = proxy
        monkeypatch.setenv("NO_PROXY", "example.invalid")

        await _session(proxies={"http": proxy_url, "https": proxy_url}).post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (gateway.hits, direct.hits) == (1, 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["NO_PROXY", "no_proxy"])
    async def test_curlopt_noproxy_overrides_the_environment(self, target, proxy, monkeypatch, var: str) -> None:
        """**The remedy** kitty relies on, for both legs of the provider.

        libcurl consults ``NO_PROXY`` only when ``CURLOPT_NOPROXY`` is unset, so
        setting it empty at construction restores ``proxies=`` as the last word.
        """
        url, direct = target
        proxy_url, gateway = proxy
        monkeypatch.setenv(var, "127.0.0.1")

        await _session(
            proxies={"http": proxy_url, "https": proxy_url},
            curl_options={CurlOpt.NOPROXY: ""},
        ).post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (gateway.hits, direct.hits) == (1, 0)


class TestTheAdapterAppliesTheRemedy:
    """The contract above is only worth having if the code actually uses it."""

    def test_the_session_builder_sets_noproxy_when_egress_is_configured(self, monkeypatch) -> None:
        """Pins the remedy to the construction site, not to a comment."""
        import kitty.providers.openai_subscription as subscription

        captured: dict = {}

        class _Spy:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.cookies = None

        monkeypatch.setattr(subscription.curl_cffi.requests, "AsyncSession", _Spy)
        monkeypatch.setattr(
            subscription, "get_egress", lambda: type("E", (), {"proxies_dict": lambda self: {"http": "http://gw"}})()
        )

        subscription.OpenAISubscriptionAdapter._new_curl_session(None)

        assert captured["proxies"] == {"http": "http://gw"}
        assert captured["curl_options"] == {CurlOpt.NOPROXY: ""}
