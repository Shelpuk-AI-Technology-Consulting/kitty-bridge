"""Dependency contract: what ``aiohttp`` promises ``_build_client_session``.

``.system_design/TEST_SUITE.md`` §6.2.4 · plan task **T-G8**
([KBR-84](https://shelpuk.atlassian.net/browse/KBR-84)).

Four facts are load-bearing, and every one of them is invisible in the code
that depends on it:

1. ``aiohttp.ClientSession(proxy=URL)`` routes every request through that
   proxy. ``_build_client_session`` sets it once — alongside its connector
   and timeout configuration — so a future call site cannot forget it.
2. ``aiohttp.ClientSession(proxy_auth=...)`` carries the credentials to the
   proxy as a ``Proxy-Authorization`` header.
3. **A per-request ``proxy=None`` cannot escape the session-level proxy.**
   The resolver reads ``if proxy is None: proxy = self._default_proxy``
   (``aiohttp/client.py:575-576``, measured on installed 3.13.5), so
   ``_session_for``'s containment design — proxied session for public
   destinations, direct session for loopback / private ones — is sound only
   while that short-circuit holds. Pinning it here is the row's headline
   claim.
4. Ambient ``HTTP_PROXY`` / ``HTTPS_PROXY`` are **not** honoured by default
   (``trust_env=False``). A user's shell cannot silently defeat the
   configured egress.

The opposite direction — a per-request explicit ``proxy=`` overrides the
session-level — is pinned too, so a future aiohttp release that reverses
either branch turns red rather than silently altering containment.

Every assertion runs against real local sockets, because a mock of a
dependency proves nothing about the dependency.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import BasicAuth, web

pytestmark = pytest.mark.l2

_TIMEOUT = 10.0


class _Recorder:
    """Counts requests and remembers the last one's shape."""

    def __init__(self) -> None:
        self.hits = 0
        self.method: str | None = None
        self.path: str | None = None
        self.proxy_authorization: str | None = None


async def _serve(handler: Callable[[web.Request], Any], port: int) -> web.AppRunner:
    """Start a local aiohttp app on *port* and return its runner.

    ``shutdown_timeout`` is deliberately short, as in the curl_cffi contract
    test: a teardown blocked on a live connection behaves differently across
    this repo's 3.10-3.13 matrix, so it is bounded here rather than left to
    the default.
    """
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


@pytest.fixture()
async def target() -> AsyncIterator[tuple[str, _Recorder]]:
    """A local stand-in for the upstream the bridge is reaching."""
    rec = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        rec.hits += 1
        rec.method = request.method
        rec.path = request.path
        return web.json_response({"ok": True})

    runner = await _serve(handler, 8791)
    try:
        yield "http://127.0.0.1:8791/echo", rec
    finally:
        await runner.cleanup()


@pytest.fixture()
async def gateway() -> AsyncIterator[tuple[str, _Recorder]]:
    """A local stand-in for the egress proxy (records every request)."""
    rec = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        rec.hits += 1
        rec.method = request.method
        rec.path = request.path
        rec.proxy_authorization = request.headers.get("Proxy-Authorization")
        return web.json_response({"ok": True})

    runner = await _serve(handler, 8792)
    try:
        yield "http://127.0.0.1:8792", rec
    finally:
        await runner.cleanup()


@pytest.fixture()
async def alternate_gateway() -> AsyncIterator[tuple[str, _Recorder]]:
    """A second local stand-in, for the per-request-override direction."""
    rec = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        rec.hits += 1
        return web.json_response({"ok": True})

    runner = await _serve(handler, 8793)
    try:
        yield "http://127.0.0.1:8793", rec
    finally:
        await runner.cleanup()


@pytest.fixture(autouse=True)
def _no_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ambient proxy variable so a developer's shell decides nothing.

    Per-variable tests below set *one* variable deliberately; the others stay
    absent. A mixed environment produces a pass / fail that cannot be
    attributed to one variable, which is the same hygiene rule the curl_cffi
    and botocore contract tests follow.
    """
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


class TestSessionLevelProxy:
    """R1 + R2: the session-level proxy and proxy_auth are honoured."""

    @pytest.mark.asyncio
    async def test_session_proxy_is_honoured(
        self, target: tuple[str, _Recorder], gateway: tuple[str, _Recorder]
    ) -> None:
        """A session built with ``proxy=URL`` routes every request through it."""
        url, direct = target
        proxy_url, gw = gateway

        async with aiohttp.ClientSession(proxy=proxy_url) as session, session.post(url, timeout=_TIMEOUT) as resp:
            await resp.read()

        assert (gw.hits, direct.hits) == (1, 0)

    @pytest.mark.asyncio
    async def test_session_proxy_auth_is_honoured(
        self, target: tuple[str, _Recorder], gateway: tuple[str, _Recorder]
    ) -> None:
        """``proxy_auth=`` reaches the gateway as ``Proxy-Authorization``."""
        url, direct = target
        proxy_url, gw = gateway
        auth = BasicAuth("kitty", "secret")
        expected = "Basic " + base64.b64encode(b"kitty:secret").decode("ascii")

        async with (
            aiohttp.ClientSession(proxy=proxy_url, proxy_auth=auth) as session,
            session.post(url, timeout=_TIMEOUT) as resp,
        ):
            await resp.read()

        assert gw.proxy_authorization == expected
        assert gw.hits == 1
        assert direct.hits == 0


class TestPerRequestCannotEscape:
    """R3: the row's load-bearing claim — containment rests on this."""

    @pytest.mark.asyncio
    async def test_per_request_proxy_none_cannot_escape(
        self, target: tuple[str, _Recorder], gateway: tuple[str, _Recorder]
    ) -> None:
        """A request issued with ``proxy=None`` still hits the session's proxy.

        The resolver at ``aiohttp/client.py:575-576`` reads
        ``if proxy is None: proxy = self._default_proxy`` — the literal
        ``None`` is the *missing-argument* sentinel, not an opt-out. A
        future aiohttp release that changed this to mean "explicitly no
        proxy" would silently invert containment; this test says so loudly.
        """
        url, direct = target
        proxy_url, gw = gateway

        async with (
            aiohttp.ClientSession(proxy=proxy_url) as session,
            session.post(url, proxy=None, timeout=_TIMEOUT) as resp,
        ):
            await resp.read()

        assert (gw.hits, direct.hits) == (1, 0)


class TestPerRequestExplicitProxy:
    """R4: the other branch — a per-request ``proxy=`` overrides the session's."""

    @pytest.mark.asyncio
    async def test_per_request_proxy_overrides_session_proxy(
        self,
        target: tuple[str, _Recorder],
        gateway: tuple[str, _Recorder],
        alternate_gateway: tuple[str, _Recorder],
    ) -> None:
        """The opposite direction of R3, pinned so a reversal is caught here too."""
        url, direct = target
        proxy_url, gw = gateway
        alt_proxy_url, alt_gw = alternate_gateway

        async with (
            aiohttp.ClientSession(proxy=proxy_url) as session,
            session.post(url, proxy=alt_proxy_url, timeout=_TIMEOUT) as resp,
        ):
            await resp.read()

        assert (gw.hits, alt_gw.hits, direct.hits) == (0, 1, 0)


class TestAmbientEnvIgnored:
    """R5: aiohttp's ``trust_env=False`` default keeps the shell out of it."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["HTTP_PROXY", "http_proxy"])
    async def test_ambient_http_proxy_is_ignored_by_default(
        self,
        target: tuple[str, _Recorder],
        gateway: tuple[str, _Recorder],
        monkeypatch: pytest.MonkeyPatch,
        var: str,
    ) -> None:
        """A user with that variable set in their shell still hits the configured proxy."""
        url, direct = target
        proxy_url, gw = gateway
        # A deliberately dead address: if the ambient variable won precedence,
        # the test would error with connection refused, not reach the gateway.
        monkeypatch.setenv(var, "http://127.0.0.1:1")

        async with aiohttp.ClientSession(proxy=proxy_url) as session, session.post(url, timeout=_TIMEOUT) as resp:
            await resp.read()

        assert (gw.hits, direct.hits) == (1, 0)


class TestTheAdapterAppliesTheContract:
    """R6: the contract is only worth having if the code uses it."""

    @staticmethod
    def _bridge_with_egress() -> Any:
        """A bare ``BridgeServer`` whose ``_egress`` is a minimal stand-in.

        The method under test reads only ``self._egress.proxy_url`` and
        ``self._egress.auth``; the production ``__init__`` wires a real
        :class:`~kitty.egress.EgressConfig`, which the contract does not need.
        """
        from kitty.bridge.server import BridgeServer

        server = BridgeServer.__new__(BridgeServer)
        server._egress = type(
            "E",
            (),
            {
                "proxy_url": "http://gw.example:8080",
                "auth": BasicAuth("u", "p"),
            },
        )()
        return server

    @staticmethod
    def _spy_aiohttp(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
        """Replace ``ClientSession`` and ``TCPConnector`` with capturing stubs.

        Both are stubbed so the R6 assertion stays about *kwargs pass-through*:
        a real ``TCPConnector`` would construct fine under the running loop but
        never be closed (the spy session discards it), leaking a
        ``ResourceWarning`` per test. Stubbing the connector keeps the test's
        scope clean of that artifact.
        """

        class _SpyClientSession:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        class _SpyConnector:
            def __init__(self, **kwargs: Any) -> None:
                pass

        monkeypatch.setattr(aiohttp, "ClientSession", _SpyClientSession)
        monkeypatch.setattr(aiohttp, "TCPConnector", _SpyConnector)

    @pytest.mark.asyncio
    async def test_build_client_session_sets_proxy_at_session_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_build_client_session(proxied=True)`` passes ``proxy=`` and ``proxy_auth=``."""
        captured: dict[str, Any] = {}
        self._spy_aiohttp(monkeypatch, captured)

        server = self._bridge_with_egress()
        server._build_client_session(proxied=True)

        assert captured["proxy"] == "http://gw.example:8080"
        assert captured["proxy_auth"] == BasicAuth("u", "p")

    @pytest.mark.asyncio
    async def test_build_client_session_omits_proxy_when_unproxied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``proxied=False`` must not leak a proxy or auth into the session."""
        captured: dict[str, Any] = {}
        self._spy_aiohttp(monkeypatch, captured)

        server = self._bridge_with_egress()
        server._build_client_session(proxied=False)

        assert "proxy" not in captured
        assert "proxy_auth" not in captured


def test_the_module_finds_known_positives() -> None:
    """Self-guard: the module cannot rot into a no-op.

    Counts the coroutine test methods across every test class so a silent
    deletion of a behavioural case shows up even if a class disappears
    entirely. The count is a floor, not a spec — adding cases raises it,
    deleting them breaches it. The floor is the current count (7), so a
    silent deletion of any one case is a red, not a slow rot.

    The module's own namespace is walked via ``sys.modules`` rather than
    re-imported through ``import tests.…``: ``tests/`` is a pytest rootdir,
    not a package (no ``__init__.py``), so that import only resolves when
    the repository root is on ``sys.path`` — true for a ``python -m pytest``
    run, false for the bare ``pytest`` the CI workflow invokes.
    """
    import inspect
    import sys

    behavioural: list[str] = []
    for cls in vars(sys.modules[__name__]).values():
        if not inspect.isclass(cls) or cls.__module__ != __name__:
            continue
        behavioural.extend(
            name for name, obj in inspect.getmembers(cls, inspect.iscoroutinefunction) if name.startswith("test_")
        )
    assert len(behavioural) >= 7, f"contract module lost its behavioural coverage; only {behavioural} remain"
