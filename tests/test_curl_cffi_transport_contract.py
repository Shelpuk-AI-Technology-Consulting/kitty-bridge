"""Dependency contract: what ``curl_cffi`` promises the OpenAI subscription leg.

``TEST_SUITE.md`` §6.2.4 owes a contract for ``curl_cffi``, whose declared range
is bounded at ``>=0.15,<0.17`` (KBR-283: the ceiling forces every version jump
through a deliberate upgrade; this suite is that upgrade's gate). §2.2's
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

The same precedence question now extends to ``HTTP_PROXY`` / ``HTTPS_PROXY`` /
``ALL_PROXY`` (KBR-85): the OAuth leg rides ``https://``, so any
scheme-scoped ambient proxy (or the ``ALL_PROXY``/``all_proxy`` catch-all
that applies to every scheme) that wins over ``proxies=`` would defeat the
configured gateway on the leg that matters. libcurl documents explicit
``CURLOPT_PROXY`` above the environment; the curl_cffi probes below pin
that order empirically, in both schemes.

Every assertion runs against real local sockets, because a mock of a dependency
proves nothing about the dependency.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable

import curl_cffi.aio as _curl_cffi_aio
import curl_cffi.requests
import pytest
from aiohttp import web
from curl_cffi import CurlOpt
from curl_cffi import curl as _curl_cffi_curl
from curl_cffi.requests.errors import RequestsError
from harness.connect_proxy import (
    CertFiles,
    ConnectProxy,
    TlsTarget,
    proxy_config,
)

from kitty.providers.openai_subscription import _CODEX_IMPERSONATE

pytestmark = pytest.mark.l2

_TIMEOUT = 10.0

_DEAD_PROXY_URL = "https://127.0.0.1:1"


class _Recorder:
    """Counts requests and remembers the last one's shape."""

    def __init__(self) -> None:
        self.hits = 0
        self.content_type: str | None = None
        self.user_agent: str | None = None
        self.body = ""


async def _serve(handler: Callable, port: int) -> web.AppRunner:
    """Start a local aiohttp app on *port* and return its runner.

    ``shutdown_timeout`` is deliberately short. ``cleanup()`` waits for in-flight
    handlers, so a test that deliberately stalls one would otherwise pay the
    handler's full sleep at teardown -- 30 s for a 1 s assertion. Worse, a
    teardown blocked on a live connection is the shape that behaves differently
    across this repo's 3.10-3.13 matrix, so it is bounded here rather than left
    to the default.
    """
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
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


@pytest.fixture()
async def ambient() -> AsyncIterator[tuple[str, _Recorder]]:
    """A stand-in for whichever ambient proxy variable a probe sets.

    A third listener, distinct from the configured gateway, so a probe
    that sets ``HTTP_PROXY`` / ``http_proxy`` / ``HTTPS_PROXY`` /
    ``https_proxy`` / ``ALL_PROXY`` / ``all_proxy`` can tell *which*
    proxy carried the request: the mapping's gateway, the ambient
    variable's proxy, or neither. A dead address would distinguish
    only "mapping won" from "something failed", and an error is a
    weaker signal than a hit count.
    """
    rec = _Recorder()

    async def handler(request: web.Request) -> web.Response:
        rec.hits += 1
        return web.json_response({"ok": True})

    runner = await _serve(handler, 8783)
    try:
        yield "http://127.0.0.1:8783", rec
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


class TestAmbientHttpProxy:
    """Precedence between ``proxies=`` and the ambient ``http_proxy``.

    ``NO_PROXY`` is host-scoped and *defeats* the mapping (the class above
    pins that exposure). ``http_proxy`` is scheme-scoped, and libcurl's
    documented order puts an explicitly set ``CURLOPT_PROXY`` — what
    ``proxies=`` becomes — above the environment. If a release ever flips
    that order, these probes turn red instead of letting a user's shell
    silently redirect provider traffic away from the configured gateway.

    Two libcurl quirks shape these probes:

    * **Casing.** On Linux/macOS libcurl reads **only the lowercase**
      ``http_proxy`` for ``http://`` targets; the uppercase ``HTTP_PROXY``
      is a deliberate no-op (the documented CGI-environment security
      exception). Uppercase names ARE honoured for every other scheme.
      The mapping probe therefore pins the lowercase form, and a
      dedicated per-platform probe pins the uppercase form's actual
      reading.
    * **Platform divergence.** curl_cffi 0.16.3's Windows build honours
      uppercase ``HTTP_PROXY`` for ``http://`` where Linux/macOS ignore
      it; the dedicated probe asserts the platform-divergent reading so
      a release that flips either direction on either platform turns red.

    The scheme-agnostic catch-all (``ALL_PROXY``/``all_proxy``) is pinned
    here too, at the bottom of the class — libcurl consults it for
    ``http://`` requests exactly as it does for ``https://``, so the
    https leg's pin does not cover this scheme. The class also carries
    the scoping negatives for this scheme: ``HTTPS_PROXY`` /
    ``https_proxy`` must not steer an ``http://`` request, and (in
    :class:`TestAmbientHttpsProxy`, on the other leg)
    ``HTTP_PROXY`` / ``http_proxy`` must not steer an ``https://`` one.
    """

    @pytest.mark.asyncio
    async def test_the_ambient_http_proxy_is_read_when_the_mapping_is_absent(
        self, target, proxy, ambient, monkeypatch
    ) -> None:
        """**The falsification control**: curl_cffi does read ``http_proxy``.

        Without the mapping, the ambient variable is the only proxy source —
        the request must land on the ambient listener. Without this test,
        the precedence probe below could pass vacuously: curl_cffi ignoring
        ``http_proxy`` entirely would look identical to the mapping winning.
        """
        url, direct = target
        proxy_url, gateway = proxy
        ambient_url, ambient_rec = ambient
        monkeypatch.setenv("http_proxy", ambient_url)

        await _session().post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (ambient_rec.hits, gateway.hits, direct.hits) == (1, 0, 0)

    @pytest.mark.asyncio
    async def test_an_ambient_http_proxy_does_not_defeat_the_mapping(self, target, proxy, ambient, monkeypatch) -> None:
        """The mapping is the last word when the ambient variable disagrees.

        Lowercase, because that is the only form libcurl reads for an
        ``http://`` request — see the class docstring.
        """
        url, direct = target
        proxy_url, gateway = proxy
        ambient_url, ambient_rec = ambient
        monkeypatch.setenv("http_proxy", ambient_url)

        await _session(proxies={"http": proxy_url, "https": proxy_url}).post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (gateway.hits, ambient_rec.hits, direct.hits) == (1, 0, 0)

    @pytest.mark.asyncio
    async def test_uppercase_http_proxy_for_an_http_target(self, target, proxy, ambient, monkeypatch) -> None:
        """Uppercase ``HTTP_PROXY`` behaviour is platform-divergent.

        libcurl's documented CGI-environment exception ignores uppercase
        ``HTTP_PROXY`` for ``http://`` requests (a CGI wrapper could set
        it on a victim's behalf), and Linux/macOS honour that. The
        curl_cffi 0.16.3 Windows build does not: it consults uppercase
        ``HTTP_PROXY`` for ``http://`` like every other scheme. Pinned
        per-platform so a release that flips either direction turns red
        — a Windows regression is as loud as a Linux one.

        Deliberately a no-mapping probe: on Linux/macOS the uppercase
        name never reaches curl_cffi, so a mapping-vs-``HTTP_PROXY``
        precedence cell would be trivially green there; the cell that
        needs pinning is the no-mapping one, which is what pins the
        reading itself. The mapping-wins rule for the lowercase form is
        pinned one probe above; the adapter's ``proxies=``-wins rule is
        the product-side guard, not part of this dependency contract.
        """
        url, direct = target
        proxy_url, gateway = proxy
        ambient_url, ambient_rec = ambient
        monkeypatch.setenv("HTTP_PROXY", ambient_url)

        await _session().post(url, data={"a": "b"}, timeout=_TIMEOUT)

        if sys.platform == "win32":
            # Windows build honours uppercase HTTP_PROXY; ambient wins.
            assert (ambient_rec.hits, gateway.hits, direct.hits) == (1, 0, 0)
        else:
            # Linux/macOS follow the CGI exception; direct wins.
            assert (direct.hits, ambient_rec.hits, gateway.hits) == (1, 0, 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["HTTPS_PROXY", "https_proxy"])
    async def test_an_ambient_https_proxy_is_not_consulted_for_an_http_target(
        self, target, ambient, monkeypatch, var: str
    ) -> None:
        """``HTTPS_PROXY`` / ``https_proxy`` are scheme-scoped: must not steer an ``http://`` request.

        No ``proxies=`` mapping is set here — deliberately. With a mapping
        in place, ``CURLOPT_PROXY`` wins over the environment and the
        request reaches the configured gateway whether or not the
        ambient variable was consulted for the ``http://`` scheme, so the
        probe would pass vacuously. Without one, the ambient variable is
        the only proxy source: if a release ever treated ``HTTPS_PROXY``
        or ``https_proxy`` as a catch-all for every scheme, this request
        would land on the ambient listener and the probe would go red.
        Both casings parametrized — libcurl reads both for ``https://``
        URLs, and a release honouring only one would still be a
        scheme-scoping regression.
        """
        url, direct = target
        ambient_url, ambient_rec = ambient
        monkeypatch.setenv(var, ambient_url)

        await _session().post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (direct.hits, ambient_rec.hits) == (1, 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["ALL_PROXY", "all_proxy"])
    async def test_the_ambient_all_proxy_is_read_when_the_mapping_is_absent(
        self, target, ambient, monkeypatch, var: str
    ) -> None:
        """**The falsification control** for the catch-all on the ``http://`` leg.

        Mirrors the lowercase-``http_proxy`` falsification at the top of
        this class, with the scheme-agnostic variable: ``ALL_PROXY`` /
        ``all_proxy`` apply to every scheme, so a release that read one
        casing but ignored the other would leave the parametrized
        precedence row green vacuously (the mapping wins either way) and
        the falsification must therefore cover both. The parametrization
        mirrors the https leg's falsification shape so the two halves of
        the catch-all pin look the same.
        """
        url, direct = target
        ambient_url, ambient_rec = ambient
        monkeypatch.setenv(var, ambient_url)

        await _session().post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (ambient_rec.hits, direct.hits) == (1, 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["ALL_PROXY", "all_proxy"])
    async def test_an_ambient_all_proxy_does_not_defeat_the_mapping(
        self, target, proxy, ambient, monkeypatch, var: str
    ) -> None:
        """The mapping is the last word when the catch-all disagrees, on ``http://`` too.

        ``ALL_PROXY``/``all_proxy`` are scheme-agnostic — libcurl consults
        them for ``http://`` requests exactly as it does for ``https://``
        — so the https leg's precedence pin (:class:`TestAmbientHttpsProxy`)
        does not cover this scheme. Both casings pinned: a release that
        honoured the catch-all for one scheme but not the other would be
        a containment-direction change worth a red test.
        """
        url, direct = target
        proxy_url, gateway = proxy
        ambient_url, ambient_rec = ambient
        monkeypatch.setenv(var, ambient_url)

        await _session(proxies={"http": proxy_url, "https": proxy_url}).post(url, data={"a": "b"}, timeout=_TIMEOUT)

        assert (gateway.hits, ambient_rec.hits, direct.hits) == (1, 0, 0)


class TestAmbientHttpsProxy:
    """Precedence between ``proxies=`` and the ambient ``HTTPS_PROXY``.

    The OpenAI OAuth leg rides ``https://auth.openai.com``, so an ambient
    ``HTTPS_PROXY`` / ``https_proxy`` / ``ALL_PROXY`` / ``all_proxy`` that
    wins over ``proxies=`` would defeat the configured egress gateway on
    the very leg the I3 invariant names. libcurl documents explicit
    ``CURLOPT_PROXY`` above the environment; the probes below pin that
    order for the resolved ``curl_cffi`` — both casings of the
    scheme-scoped variable plus the catch-all — using the harness
    CONNECT proxy and TLS target. (The botocore twin at
    ``tests/harness/test_botocore_transport_contract.py`` carries the
    equivalent contract for botocore on its own stack.)

    A dead ambient address is used so that *if* the ambient variable won
    the request would fail loudly with a connection refused against
    127.0.0.1:1, rather than silently arriving at an unrelated host.

    The class also carries the scheme-scoping negative for the
    ``http://`` variables on this leg (``HTTP_PROXY`` / ``http_proxy``
    must not steer an ``https://`` request), mirroring
    :class:`TestAmbientHttpProxy`'s ``HTTPS_PROXY`` / ``https_proxy``
    negative on its own scheme.
    """

    @pytest.fixture(autouse=True)
    def _patch_default_cacert(self, certs: CertFiles, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make curl_cffi trust the harness CA for both TLS legs.

        Two seams have to move:

        * ``curl_cffi.aio.DEFAULT_CACERT`` is bound at import time (a
          ``from .curl import DEFAULT_CACERT``), so patching only
          ``curl_cffi.curl.DEFAULT_CACERT`` does not reach AsyncSession.
        * ``curl_cffi.requests.session`` reads ``REQUESTS_CA_BUNDLE`` /
          ``CURL_CA_BUNDLE`` whenever ``verify`` is None (AsyncSession's
          default), and those values override the patched CA path. The
          developer whose shell sets either variable would otherwise see
          a cert error here, red but non-portable; mirror the precedent
          at ``test_egress_https_proxy.py`` and delete both.

        Patching both ``_curl_cffi_curl.DEFAULT_CACERT`` and
        ``_curl_cffi_aio.DEFAULT_CACERT`` covers the case where either
        side ever flips which side AsyncSession reads from.
        """
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
        monkeypatch.setattr(_curl_cffi_curl, "DEFAULT_CACERT", str(certs.ca))
        monkeypatch.setattr(_curl_cffi_aio, "DEFAULT_CACERT", str(certs.ca))

    @staticmethod
    def _https_target(tls_target: TlsTarget) -> str:
        """Build the ``https://`` URL the probes direct curl_cffi at."""
        return f"https://127.0.0.1:{tls_target.port}/oauth/token"

    @staticmethod
    def _https_mapping(connect_proxy: ConnectProxy) -> dict[str, str]:
        """Build the configured ``proxies=`` mapping the adapter actually passes."""
        config = proxy_config(connect_proxy.port)
        return config.proxies_dict()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["HTTPS_PROXY", "https_proxy"])
    async def test_an_ambient_https_proxy_does_not_defeat_the_mapping(
        self,
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        monkeypatch: pytest.MonkeyPatch,
        var: str,
    ) -> None:
        """The mapping is the last word when an ambient ``HTTPS_PROXY``/``https_proxy`` disagrees.

        Both casings are pinned: libcurl reads either for ``https://``
        requests, and a release that honoured only one of them would
        still be a containment-direction change worth a red test. The
        dead ambient address (``https://127.0.0.1:1``) makes the failure
        mode of the ambient variable winning loud rather than silent.
        """
        monkeypatch.setenv(var, _DEAD_PROXY_URL)

        await _session(proxies=self._https_mapping(connect_proxy)).post(
            self._https_target(tls_target), data={"a": "b"}, timeout=_TIMEOUT
        )

        assert len(connect_proxy.attempts) == 1, (
            f"ambient {var}={_DEAD_PROXY_URL} defeated proxies=: expected exactly one "
            f"CONNECT attempt against the harness proxy, got {len(connect_proxy.attempts)}. "
            "If this fails on a curl_cffi bump, the precedence contract has changed."
        )
        assert connect_proxy.attempts[0].authenticated is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["ALL_PROXY", "all_proxy"])
    async def test_an_ambient_all_proxy_does_not_defeat_the_mapping(
        self,
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        monkeypatch: pytest.MonkeyPatch,
        var: str,
    ) -> None:
        """``ALL_PROXY`` applies to every scheme and must also lose to ``proxies=``.

        The catch-all is the most dangerous of the four: a user with a
        single ``ALL_PROXY=`` in their shell intends it as a default for
        all schemes, and a containment regression that lets it beat the
        configured mapping would silently redirect every outbound curl
        request — including the OAuth leg — past the egress gateway.
        """
        monkeypatch.setenv(var, _DEAD_PROXY_URL)

        await _session(proxies=self._https_mapping(connect_proxy)).post(
            self._https_target(tls_target), data={"a": "b"}, timeout=_TIMEOUT
        )

        assert len(connect_proxy.attempts) == 1, (
            f"ambient {var}={_DEAD_PROXY_URL} defeated proxies=: expected exactly one "
            f"CONNECT attempt against the harness proxy, got {len(connect_proxy.attempts)}"
        )
        assert connect_proxy.attempts[0].authenticated is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"])
    async def test_an_ambient_https_proxy_is_read_when_the_mapping_is_absent(
        self,
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        monkeypatch: pytest.MonkeyPatch,
        var: str,
    ) -> None:
        """**The falsification control**: curl_cffi reads the env var in question.

        Parametrized over every ambient proxy variable the precedence
        probes above also cover — the two scheme-scoped variables
        (``HTTPS_PROXY``, ``https_proxy``) and the two scheme-agnostic
        catch-all variables (``ALL_PROXY``, ``all_proxy``): a release
        that stops reading any one of them must turn red here. Without
        this falsification, a release that ignored ``https_proxy`` (for
        example) would stay green on its precedence probe — the mapping
        wins either way — so the parametrized probe's failure is the
        only signal that *that* variable went missing.
        """
        monkeypatch.setenv(var, _DEAD_PROXY_URL)

        with pytest.raises(RequestsError) as exc_info:
            await _session().post(self._https_target(tls_target), data={"a": "b"}, timeout=_TIMEOUT)

        # The exact curl error code can vary by version; pin the proxy-attempted
        # substring instead so the test is robust to small wording changes.
        assert "over proxy 127.0.0.1" in str(exc_info.value), (
            f"expected curl to attempt the dead proxy and fail; got: {exc_info.value}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("var", ["HTTP_PROXY", "http_proxy"])
    async def test_an_ambient_http_proxy_is_not_consulted_for_an_https_target(
        self,
        connect_proxy: ConnectProxy,
        tls_target: TlsTarget,
        monkeypatch: pytest.MonkeyPatch,
        var: str,
    ) -> None:
        """``HTTP_PROXY`` / ``http_proxy`` are scheme-scoped: must not steer an ``https://`` request.

        The mirror of
        :meth:`TestAmbientHttpProxy.test_an_ambient_https_proxy_is_not_consulted_for_an_http_target`
        — without it, the scoping matrix is pinned in only one of its
        two directions, and a release that started consulting the
        ``http`` variable for ``https://`` requests would leave every
        other probe green while §5.5's closure claim no longer held.
        The ambient variable is set to a dead address with no
        ``proxies=`` mapping in place: if the release ever consulted the
        variable for the wrong scheme, the request dies loudly against
        127.0.0.1:1 instead of silently redirecting; if scheme-scoping
        holds, the request reaches the local TLS target directly.
        """
        monkeypatch.setenv(var, _DEAD_PROXY_URL)

        # No proxies=, no pytest.raises: a successful 200 is the contract.
        # If a release consults the variable for the wrong scheme, the
        # await raises RequestsError and the test fails loudly.
        response = await _session().post(self._https_target(tls_target), data={"a": "b"}, timeout=_TIMEOUT)

        assert response.status_code == 200, (
            f"ambient {var}={_DEAD_PROXY_URL} was consulted for an https:// request "
            f"(scheme-scoping broken); got status {response.status_code}"
        )


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
