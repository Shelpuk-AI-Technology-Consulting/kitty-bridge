"""The complete, falsified provider-aiohttp containment slice — T-E5 ([KBR-65]).

`.system_design/TEST_SUITE.md` §5.2.1, §5.2.2, §5.3, §5.4, §5.5, §7.2.2 ·
plan task **T-E5**
([KBR-65](https://shelpuk.atlassian.net/browse/KBR-65)) ·
`.requirements/20260916T205619Z_kbr65_t_e5_provider_aiohttp_containment_slice/REQUIREMENTS.md`.

The aiohttp slice (T-E2, :mod:`harness.test_aiohttp_containment_slice`) proved
§5.2.2's four phases for the bridge's own serving session, and the curl_cffi
slice (T-E3, :mod:`harness.test_curl_cffi_containment_slice`) proved them for
the OpenAI subscription's impersonating session. This module proves the same
phases for the **provider-aiohttp** transport — the two product paths §5.5
names that build an aiohttp session of their own and therefore bypass
``_session_for`` entirely:

* ``ollama_cloud``'s own serving session (the **serving leg**), driven through
  a real :class:`~kitty.bridge.server.BridgeServer` running the real
  :class:`~kitty.providers.ollama_cloud.OllamaCloudAdapter`;
* the ``openai_subscription`` OAuth **login** leg (:mod:`kitty.auth.openai_oauth`),
  which runs at startup, before anything else has been proven, and is the path
  most likely to fire on a fresh machine (§5.5) — driven through the product's
  own token-exchange coroutine, the same driving decision T-B1 recorded (the
  interactive half of ``run_oauth_flow`` waits five minutes for a browser).

One capability-report row (``provider_aiohttp``) covers both paths, so the
verdict gate tracks **nine** gated tests — the five §5.2.2 phase tests on the
serving leg plus the four OAuth-leg phases — rather than the siblings' five.
Gating the OAuth tests is deliberate: an ungated OAuth test could be deleted
without moving the verdict, which is exactly the vacuity the gate exists to
prevent.

**What carries the containment claim here.** The adapter honours
``provider_config["base_url"]``, so the drive points it at
``https://upstream.kitty-test.invalid:{port}`` — the non-loopback name the
proxy's resolve map translates (§5.3 trap 1). The adapter's session is built
with ``aiohttp_session_kwargs()`` **resolved in its own module's namespace**
(``from kitty.egress import aiohttp_session_kwargs`` at module import), so the
ambient egress global is what carries the proxy mapping, and the phase-3
falsification patches that module binding — a patch on ``kitty.egress`` would
be silently ignored, the exact fail-by-silence the falsification exists to
prevent. The OAuth drive builds its session through
``openai_oauth.aiohttp_session_kwargs()`` for the same reason: the drive must
resolve the same binding the phase's patch lands on, or the falsification is
vacuous. The static half of that claim — that ``run_oauth_flow`` builds its
session with ``aiohttp_session_kwargs()`` — is already pinned by
``tests/test_egress_coverage.py``.

**The direct route (§5.3).** The provider session resolves through
``socket.getaddrinfo`` exactly as the bridge's does, so
:class:`ProviderAiohttpContainment.direct_route` is the provider-aiohttp
resolver hook the ``ContainmentTransport`` protocol names: a self-contained
``try/finally`` patch on ``socket.getaddrinfo`` (T-E3's precedent for a
``monkeypatch``-free override — the protocol's ``direct_route(harness)``
signature has no fixture parameter). The mapping stays in scope on the
proxied phases too (§5.3 trap 2): a product that wrongly falls back to a
direct route still resolves the harness hostname to the recorder, and
§5.2.1's join detects the bypass instead of the negative assertion passing
vacuously on a DNS failure.

**TLS at the recorder.** The recorder is
:class:`~harness.provider_recorder.ProviderTlsRecordingUpstream` — the
provider vocabulary (``/api/chat`` and ``/oauth/token``) speaking TLS, taken
at construction because ``SealedNetwork``'s ``recorder_factory`` path calls
``start()`` with no arguments. Without TLS, aiohttp sends ``http://`` requests
in absolute form and the CONNECT-only proxy answers 405 (the T-E2 lesson).
Every drive takes the ``aiohttp_trusts_test_ca`` fixture so the adapter's
session and the drive-built OAuth session trust the harness CA on both hops.

**Python <3.11 floor.** The direct-leg tests (serving phase 1, OAuth direct)
and the static checks run on every supported Python. The proxied phases
(2/2b/3 on both legs) carry the same Python ≥3.11 ``skipif`` as the aiohttp
slice — aiohttp's TLS-in-TLS over stdlib asyncio landed in 3.11 (bpo-44011).
On interpreters below 3.11 the verdict gate in :mod:`harness.conftest` records
this slice's row as ``UNSUPPORTED`` with the documented floor reason (the
owner's scope decision on KBR-63, 2026-09-16), so the plan's "an outcome is
recorded" done-when holds on every supported interpreter. CI's 3.10 leg
exercises the writer end-to-end.

**The refresh leg is out of scope.** Since KBR-161 the OAuth **refresh** leg
runs on the adapter's impersonating ``curl_cffi`` session, which an aiohttp
recorder cannot observe; T-E3's serving-leg proof covers the shared
``_new_curl_session`` egress mapping both legs build with.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
``test_containment.py`` and the two sibling slices. The reason is §8.2's and
only §8.2's: ``l3`` is in ``PENDING_ACTIVATION_LAYERS``, so an ``l3`` marker
today would leave this slice's correctness checked by no job at all. T-K6
inherits the relocation for this file alongside the other two.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import ssl
import sys
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass

import aiohttp
import pytest

from harness.bridge import MODEL, InboundProtocol, inbound_path, marker, minimal_inbound_body
from harness.connect_proxy import (
    HARNESS_UPSTREAM_HOST,
    PROXY_PASSWORD,
    CertFiles,
    proxy_config,
    unattributable_peer_ports,
)
from harness.containment import (
    ContainmentTransport,
    Phase1Result,
    SealedNetwork,
    register_containment_transport,
    reset_for_test,
)
from harness.contract import WireFormat
from harness.provider_recorder import (
    OAUTH_TOKEN_SUFFIX,
    ProviderTlsRecordingUpstream,
)
from kitty.auth import openai_oauth
from kitty.egress import EgressConfig, get_egress, set_egress
from kitty.providers.ollama_cloud import OllamaCloudAdapter

#: The Python ≥3.11 floor the proxied phases skip below — the same threshold
#: the aiohttp slice's ``_AIOHTTP_NEEDS_311`` pins. Defined locally rather
#: than alias-imported so this file reads without an ``aiohttp``-named
#: constant: the threshold is shared, the transport is not.
_NEEDS_311 = sys.version_info < (3, 11)

#: Why the proxied §5.2.2 phases skip on Python <3.11. The aiohttp slice's
#: skipif carries a transport-specific reason (its own
#: aiohttp/stdlib-asyncio/bpo-44011 framing); the provider-aiohttp transport
#: is the same aiohttp stack, so the same framing applies — but the reason is
#: spelled in this module so a phase's pytest-skip report reads here, not in
#: a sibling file.
_SKIP_REASON = (
    "aiohttp requires Python 3.11 for TLS-in-TLS over stdlib asyncio (bpo-44011); "
    "the proxied §5.2.2 phases exercise exactly that shape"
)

#: A budget for one drive request (T-E1's value). The green path completes in
#: well under a second; the proxied-failure phases (2, 407) see the client
#: timeout fire while the bridge works through its ~30 s transport-grace
#: ladder, and the drive's ``status`` comes back as ``-1`` — which is what
#: those phases assert on.
_DRIVE_TIMEOUT = 10.0


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def sealed_network(certs: CertFiles) -> AsyncGenerator[SealedNetwork, None]:
    """One started ``SealedNetwork`` hosting a TLS provider recorder, torn down on exit.

    The factory hands :class:`~harness.provider_recorder.ProviderTlsRecordingUpstream`
    — Ollama ``/api/chat`` and OAuth ``/oauth/token`` vocabulary, TLS at
    construction (``SealedNetwork``'s factory path calls ``start()`` with no
    arguments) — into the *same* proxy+recorder pair the sibling slices use,
    so the §5.2.1 resolve map and lifecycle guards are shared, not re-derived.

    Each phase gets a **fresh** sealed network, for the reason the sibling
    fixtures record: phase 1's direct connection leaves a peer-port row in
    ``recorder.connections``; sharing one network would let phase 3's
    "unattributable port" assertion pass on phase 1's leak rather than on the
    injected bypass.

    Yields:
        The running harness.
    """

    def _factory(ctx: ssl.SSLContext) -> ProviderTlsRecordingUpstream:
        return ProviderTlsRecordingUpstream(default_format=WireFormat.OLLAMA_CHAT, ssl_context=ctx)

    net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs, recorder_factory=_factory)
    await net.start()
    try:
        yield net
    finally:
        await net.stop()


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    """Reset the capability-report singleton before every test in this module.

    The verdict test at the bottom records ``PROVEN`` for ``provider_aiohttp``;
    every earlier test must see an untouched report so its phase assertions do
    not depend on collection order. The session finalisers in
    :mod:`harness.conftest` run after every test in the process and write
    their own rows regardless of what any per-test reset did.
    """
    reset_for_test()


def _egress_for(net: SealedNetwork, password: str = PROXY_PASSWORD) -> EgressConfig:
    """Return the harness's egress configuration for ``net``'s proxy.

    Args:
        net: The sealed network the proxy belongs to.
        password: The proxy password; pass a wrong one for the 407 case.

    Returns:
        The configuration the adapter's session and the ``BridgeServer``
        both take.
    """
    return proxy_config(net.proxy.port, password=password)


def _drive() -> ProviderAiohttpContainment:
    """Build a fresh containment transport for one drive.

    ``ProviderAiohttpContainment`` is stateless, so a fresh instance per call
    is bookkeeping rather than necessity — but it keeps the drive's lifetime
    visible and mirrors the per-drive contract the protocol documents.

    Returns:
        A fresh transport.
    """
    return ProviderAiohttpContainment()


# ── The OAuth-leg seam (§5.3's harness-hostname rule, on the token URL) ────


@contextlib.contextmanager
def harness_oauth_token_url(harness: SealedNetwork) -> Iterator[str]:
    """Point the OAuth login leg's token endpoint at the **harness hostname**.

    The login leg reaches ``https://auth.openai.com/oauth/token`` through the
    module constant :data:`kitty.auth.openai_oauth.OAUTH_TOKEN_URL`, with no
    configuration channel of any kind. T-B1's
    :func:`harness.provider_aiohttp.oauth_token_endpoint` is the same
    read-swap-restore re-aimed: that seam yields the recorder's **loopback**
    base URL, which is correct for its conformance tests (no proxy involved)
    but would make §5.3 trap 1 vacuous here — a loopback token URL is
    addressed direct, and the containment assertions would pass while proving
    the opposite of their claim. This seam yields
    ``https://upstream.kitty-test.invalid:{port}/oauth/token`` — the
    non-loopback name the proxy's resolve map translates.

    **A silently-missed swap cannot reach the real host.** The harness
    proxy's resolve map carries the deny entries for ``auth.openai.com``
    (``_REAL_UPSTREAM_DENY_RESOLVE`` in :mod:`harness.containment`): a swap
    that stopped firing leaves the real constant in place, the proxy resolves
    the real hostname to an RFC-5735 blackhole and answers 502 — the phase
    fails loudly with no real egress incurred.

    Reading the constant before writing it is what makes the seam falsifiable:
    if the leg is ever rewritten to read its endpoint from somewhere else, this
    raises :class:`AttributeError` rather than swapping a name nothing
    consults and leaving a test to pass with an empty capture list.

    Args:
        harness: A **started** sealed network; its upstream URL is read here.

    Yields:
        The URL the login leg will now post to.

    Raises:
        AttributeError: When :mod:`kitty.auth.openai_oauth` no longer defines
            ``OAUTH_TOKEN_URL``.
    """
    url = f"{harness.upstream_base_url}{OAUTH_TOKEN_SUFFIX}"
    if not hasattr(openai_oauth, "OAUTH_TOKEN_URL"):
        raise AttributeError(
            "kitty.auth.openai_oauth no longer defines OAUTH_TOKEN_URL; the "
            "login leg reads its token endpoint elsewhere and this seam no "
            "longer redirects it"
        )
    original = openai_oauth.OAUTH_TOKEN_URL
    openai_oauth.OAUTH_TOKEN_URL = url
    try:
        yield url
    finally:
        # Restored in a `finally` rather than by pytest's monkeypatch, so the
        # seam is usable from anything — a containment drive included.
        openai_oauth.OAUTH_TOKEN_URL = original


# ── The containment transport (§5.2.2's extension interface) ──────────────


class ProviderAiohttpContainment(ContainmentTransport):
    """The provider-aiohttp containment transport: the adapter's own session.

    Drives one request through a real :class:`~kitty.bridge.server.BridgeServer`
    running the real :class:`~kitty.providers.ollama_cloud.OllamaCloudAdapter`,
    with the harness-hostname base URL, the ambient egress global, and the
    resolver hook in scope. Constructed per drive; no state lives on the
    instance.

    Attributes:
        name: The registry key, ``"provider_aiohttp"``.
    """

    name = "provider_aiohttp"

    @contextlib.contextmanager
    def direct_route(self, harness: SealedNetwork) -> Iterator[None]:
        """Put the provider-aiohttp resolver hook in scope for the drive (§5.3).

        The provider session resolves through ``socket.getaddrinfo`` exactly
        as the bridge's own session does — ``aiohttp``'s default
        (no-``aiodns``) build consults it in a worker thread — so the hook is
        the same patch :func:`harness.containment.monkeypatched_aiohttp_resolver`
        applies, but entered as a self-contained ``try/finally`` rather than
        through pytest's ``MonkeyPatch``: the protocol's
        ``direct_route(harness)`` signature has no fixture parameter, and
        T-E3's override set the precedent for a ``monkeypatch``-free
        implementation. This is the "provider-aiohttp session's resolver
        hook" the protocol's docstring names as this slice's override.

        Entered by **both** drives, so the mapping is in scope on the proxied
        phases too (§5.3 trap 2): a product that wrongly falls back to a
        direct route still resolves the harness hostname to the recorder, and
        §5.2.1's join detects the bypass instead of the negative assertion
        passing vacuously on a DNS failure.

        Args:
            harness: The sealed network the request will be driven against.

        Yields:
            ``None``.
        """
        original = socket.getaddrinfo

        def _patched(name: str, *args: object, **kwargs: object) -> object:
            """Return the mapped address for the harness hostname; defer the rest.

            Args:
                name: The host being resolved.
                *args: Forwarded to the original resolver.
                **kwargs: Forwarded to the original resolver.

            Returns:
                A single ``getaddrinfo``-shaped entry for the harness
                hostname, or whatever the original resolver returned.
            """
            if name == harness.upstream_host:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", harness.upstream_port))]
            return original(name, *args, **kwargs)  # type: ignore[arg-type]

        socket.getaddrinfo = _patched  # type: ignore[assignment]
        try:
            yield
        finally:
            socket.getaddrinfo = original  # type: ignore[assignment]

    async def drive_phase_1(
        self,
        harness: SealedNetwork,
        *,
        monkeypatch: pytest.MonkeyPatch,
        resolver_port: int | None = None,
    ) -> Phase1Result:
        """Drive one request through the bridge with egress off (§5.2.2 phase 1).

        The signature matches the ``ContainmentTransport`` protocol exactly.
        ``resolver_port`` is accepted and **ignored**: the protocol declares
        it as the falsification seam, and the aiohttp slice drives that seam
        through its monkeypatched resolver. This slice's resolver is the
        harness mapping baked into :meth:`direct_route` — there is no
        "broken resolver" variant to inject without a second override; this
        slice's falsification seam is the phase-3 ``aiohttp_session_kwargs``
        patch instead (T-E3 records the same decision).

        Args:
            harness: The sealed network the request is driven against.
            monkeypatch: Pytest's monkeypatch fixture; reserved for the
                calling test's own patches (the direct route manages its
                own scope).
            resolver_port: Ignored; present for protocol substitutability.

        Returns:
            A :class:`~harness.containment.Phase1Result` with the recorder's
            observations. With egress off and the resolver hook in scope,
            ``status == 200``, ``len(captures) == 1``, ``attempts == []``.
        """
        return await self._drive(harness, egress=None, monkeypatch=monkeypatch)

    async def drive_with_egress(
        self,
        harness: SealedNetwork,
        *,
        egress: EgressConfig,
        monkeypatch: pytest.MonkeyPatch,
        resolver_port: int | None = None,
    ) -> Phase1Result:
        """Drive one request through the bridge with egress configured (§5.2.2 phases 2/2b/3).

        The egress configuration reaches the adapter through the **ambient
        global** (:func:`kitty.egress.set_egress`), which the adapter's
        ``aiohttp_session_kwargs()`` call — resolved in its own module's
        namespace — reads when the session is first built. The prior global
        value is captured and restored exactly, so the drive does not own
        the process default. ``BridgeServer`` also receives ``egress=`` so
        its own outbound path honours the same configuration; the two
        channels are independent (the adapter's session is the one that
        carries the request).

        The resolver mapping stays in scope exactly as in phase 1 — see
        :meth:`direct_route` for the §5.3 trap-2 property that keeps the
        negative assertions honest.

        ``resolver_port`` is accepted and ignored — see
        :meth:`drive_phase_1` for the protocol-substitutability note.

        Args:
            harness: The sealed network the request is driven against.
            egress: The configuration to install. Typical value:
                ``_egress_for(harness)``; a wrong password produces the 407
                subcase.
            monkeypatch: Pytest's monkeypatch fixture; reserved for the
                calling test's own patches.
            resolver_port: Ignored; present for protocol substitutability.

        Returns:
            A :class:`~harness.containment.Phase1Result` with the recorder's
            observations.
        """
        return await self._drive(harness, egress=egress, monkeypatch=monkeypatch)

    async def _drive(
        self,
        harness: SealedNetwork,
        *,
        egress: EgressConfig | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Phase1Result:
        """The drive both entry points share; ``egress=None`` is phase 1.

        The context stack, innermost last: the egress global, the resolver
        hook. The bridge is constructed directly rather than through
        ``BridgeFixture``: the fixture's ``bind()`` returns the recorder's
        **loopback** base URL, and §5.3's whole point is that the adapter
        reaches the harness **by its non-loopback name** — the adapter
        honours ``provider_config["base_url"]``, so the harness hostname
        goes in there.

        Args:
            harness: The sealed network the request is driven against.
            egress: The configuration to install, or ``None`` for the direct
                leg.
            monkeypatch: Pytest's monkeypatch fixture (unused here; the
                parameter keeps the two entry points' signatures identical).

        Returns:
            A :class:`Phase1Result` with the bridge's status, the recorder's
            captures and connections after the request, and the proxy's
            attempts.
        """
        # Local import keeps the module-import surface tidy: the bridge
        # server depends on `kitty.bridge.server`, which the rest of the
        # test module surface (the report, the recorder) does not need.
        from kitty.bridge.server import BridgeServer

        saved_egress = get_egress()
        status: int = -1
        text: str = ""
        try:
            set_egress(egress)
            with self.direct_route(harness):
                # A fresh adapter per drive: the session (and its egress
                # mapping) is built lazily on the first request, so the
                # configuration above is what the session sees.
                adapter = OllamaCloudAdapter()
                server = BridgeServer(
                    None,  # type: ignore[arg-type]
                    adapter,
                    "harness-key",
                    model=MODEL,
                    provider_config={"base_url": harness.upstream_base_url},
                    egress=egress,
                )
                try:
                    bridge_port = await server.start_async()

                    sent = marker()
                    body = minimal_inbound_body(InboundProtocol.CHAT_COMPLETIONS, sent)

                    async with aiohttp.ClientSession() as client:
                        response = await client.post(
                            f"http://127.0.0.1:{bridge_port}{inbound_path(InboundProtocol.CHAT_COMPLETIONS)}",
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=_DRIVE_TIMEOUT),
                        )
                        text = await response.text()
                        status = response.status
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                    # The bridge either timed out or failed its own outbound
                    # connect — both shapes the test client's ``status``
                    # field cannot represent. Carry the exception in ``text``
                    # so a failing drive is diagnosable from its result alone.
                    status = -1
                    text = repr(exc)
                finally:
                    await server.stop_async()
        finally:
            set_egress(saved_egress)

        return Phase1Result(
            status=status,
            text=text,
            captures=list(harness.recorder.requests),
            connections=list(harness.recorder.connections),
            attempts=list(harness.proxy.attempts),
        )


# ── The OAuth-leg drive ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _OAuthDriveResult:
    """What one OAuth-leg drive observed, in one struct.

    The connection-level evidence (captures, connections, attempts) is read
    from the live harness objects after the drive, exactly as the serving
    leg's :class:`Phase1Result` snapshots them; this struct carries only what
    the drive itself learned.

    Attributes:
        ok: Whether the token exchange completed.
        error: The exception repr when ``ok`` is ``False``; ``""`` otherwise.
    """

    ok: bool
    error: str = ""


async def _drive_oauth(
    harness: SealedNetwork,
    transport: ProviderAiohttpContainment,
    *,
    egress: EgressConfig | None,
) -> _OAuthDriveResult:
    """Drive one OAuth login-leg token exchange against the sealed network.

    The session is built **through the product's own construction kwargs,
    resolved in the product's own module binding** —
    ``openai_oauth.aiohttp_session_kwargs()`` — because that is the binding
    the phase-3 falsification patches. A drive that resolved the kwargs
    through :mod:`kitty.egress` directly would be unaffected by an
    ``openai_oauth``-side patch, and the "bypass" it was meant to demonstrate
    would be a no-op (the falsification would be vacuously green). The static
    half of the claim — that ``run_oauth_flow`` builds its session with this
    helper — is pinned by ``tests/test_egress_coverage.py``; driving
    ``run_oauth_flow`` itself is impossible in CI (it opens a browser and
    waits five minutes for a human), which is the same reason T-B1 drives
    the token-exchange coroutines directly.

    The context stack mirrors the serving leg's: the egress global, the
    token-URL seam, the resolver hook — innermost last, the global restored
    in a ``finally`` so a failing drive cannot leave the process default
    swapped.

    Args:
        harness: The sealed network the exchange is driven against.
        transport: The containment transport whose :meth:`direct_route`
            resolver hook scopes the drive.
        egress: The configuration to install, or ``None`` for the direct
            leg.

    Returns:
        A :class:`_OAuthDriveResult`. The caller reads the harness's
        capture, connection and attempt logs for the containment
        assertions.
    """
    from kitty.auth import openai_oauth as _oauth  # noqa: PLC0415 -- the binding the drive must resolve

    saved_egress = get_egress()
    ok = False
    error = ""
    try:
        set_egress(egress)
        with harness_oauth_token_url(harness), transport.direct_route(harness):
            # Built inside the seam scope so the patched binding (if a
            # falsification is in flight) is what the session sees.
            http = aiohttp.ClientSession(**_oauth.aiohttp_session_kwargs())
            try:
                await _oauth._exchange_code_for_tokens("code", "verifier", "client-id", http)
                ok = True
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                ok = False
                error = repr(exc)
            finally:
                await http.close()
    finally:
        set_egress(saved_egress)

    return _OAuthDriveResult(ok=ok, error=error)


# ── The serving-leg seam's own falsification ──────────────────────────────


class TestHarnessOAuthTokenUrlSeam:
    """The token-URL swap is live, restores, and refuses a name nothing defines."""

    async def test_the_harness_hostname_seam_swaps_and_restores(self, sealed_network: SealedNetwork) -> None:
        """Inside the block the constant carries the harness hostname; after, the original."""
        original = openai_oauth.OAUTH_TOKEN_URL
        with harness_oauth_token_url(sealed_network) as url:
            assert url.startswith(f"https://{HARNESS_UPSTREAM_HOST}"), (
                f"seam yielded {url!r}, not the harness hostname: §5.3 trap 1, a loopback URL addresses the leg direct"
            )
            assert url.endswith(OAUTH_TOKEN_SUFFIX)
            assert url == openai_oauth.OAUTH_TOKEN_URL
        assert original == openai_oauth.OAUTH_TOKEN_URL, (
            "the seam did not restore the original constant: every later test in "
            "the process would post its token grants at the recorder"
        )

    async def test_the_seam_refuses_to_swap_a_name_nothing_defines(
        self, sealed_network: SealedNetwork, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A product rewrite that reads its endpoint elsewhere fails loudly, not silently."""
        monkeypatch.delattr(openai_oauth, "OAUTH_TOKEN_URL")
        with pytest.raises(AttributeError, match="OAUTH_TOKEN_URL"), harness_oauth_token_url(sealed_network):
            pass


# ── Phase 1 — positive control (§5.2.2 row 1) ─────────────────────────────


class TestPhase1PositiveControl:
    """Egress off: the adapter's session reaches the recorder directly; the proxy sees nothing.

    The control for everything below: without it, "the upstream received
    nothing" is satisfied just as well by a destination that is unreachable
    for an unrelated reason (§5.3 trap 2). Runs on every supported Python —
    the direct leg needs no TLS-in-TLS.
    """

    async def test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Green path on the TLS recorder: 200, one capture, real peer port, zero attempts."""
        result = await _drive().drive_phase_1(sealed_network, monkeypatch=monkeypatch)

        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert len(result.captures) == 1, f"recorder saw {len(result.captures)} capture(s)"
        assert result.captures[0].path == "/api/chat", (
            f"the adapter posted at {result.captures[0].path!r}, not /api/chat: the "
            "base-URL seam did not reach the adapter's own request path"
        )
        assert any(c.peer_port and c.peer_port > 0 for c in result.connections), (
            "recorder's connection log has no entry with a real peer port: the §5.2.1 join has nothing to bind to"
        )
        assert result.attempts == [], f"proxy saw {len(result.attempts)} CONNECT attempt(s) with egress off"


# ── Phase 2 — containment under proxy failure (§5.2.2 row 2) ──────────────


@pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
class TestPhase2ContainmentProxyDown:
    """Proxy down ⇒ upstream accepts zero connections, request fails, no direct fallback.

    The negative assertion §5.1 gap 2 and §5.2.2 row 2 specify. Bracketed by
    phase 1 (a reachable destination) per §5.3 trap 2. The drive's client
    timeout fires while the bridge works through its ~30 s transport-grace
    ladder, so ``status`` comes back as ``-1`` — which is what the first
    assertion reads.
    """

    async def test_proxy_down_leaves_the_recorder_with_zero_connections(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stop the proxy, drive once, assert the recorder holds nothing and the drive failed."""
        await sealed_network.proxy.stop()

        result = await _drive().drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network), monkeypatch=monkeypatch
        )

        assert result.status != 200, (
            f"bridge answered {result.status} with the proxy down: a 200 means the "
            "adapter reached the recorder anyway — the containment premise is broken"
        )
        assert result.connections == [], (
            f"recorder accepted {len(result.connections)} connection(s) with the proxy "
            "stopped: the adapter fell back to a direct route, which is the direct "
            "fallback §5.2.2 phase 2 exists to catch"
        )
        assert result.captures == [], f"recorder saw {len(result.captures)} capture(s) with the proxy stopped"
        assert result.attempts == [], f"proxy saw {len(result.attempts)} CONNECT attempt(s) after being stopped"


# ── Phase 2b — containment healthy (§5.2.2 row 2b + §5.2.1) ───────────────


@pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
class TestPhase2bContainmentHealthy:
    """Proxy up ⇒ every upstream connection joins a tunnel on the recorded source port.

    §5.2.1's join, exercised against a real proxied run. Requests-per-tunnel
    counts are deliberately not asserted — connection reuse may put many
    requests on one tunnel and a 407 puts none on any — so the assertion is
    at the connection level, via ``unattributable_peer_ports``.
    """

    async def test_proxy_up_every_peer_port_joins_a_tunnel(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three drives, every recorded peer port matches a tunnel source port."""
        egress = _egress_for(sealed_network)
        for _ in range(3):
            result = await _drive().drive_with_egress(sealed_network, egress=egress, monkeypatch=monkeypatch)
            assert result.status == 200, f"drive answered {result.status}: {result.text!r}"

        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        assert peer_ports, "recorder accepted no connections; the drive never reached it"
        assert unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts) == [], (
            f"recorder peer ports {peer_ports} not covered by tunnel source ports "
            f"{[a.source_port for a in sealed_network.proxy.attempts]}: §5.2.1's join broken"
        )
        # The swap-live check: every tunnel targeted the harness hostname. A
        # silently-wrong base URL would point the adapter at some other
        # public host, which the proxy would resolve and connect to — real
        # egress from a test, and the join below could not tell it from a
        # tunnel to the recorder.
        assert all(a.target.startswith(f"{HARNESS_UPSTREAM_HOST}:") for a in sealed_network.proxy.attempts), (
            f"proxy saw CONNECT targets {[a.target for a in sealed_network.proxy.attempts]}, "
            f"expected only {HARNESS_UPSTREAM_HOST!r}: the base-URL seam is not live"
        )

    async def test_failed_tunnel_contributes_no_upstream_connection(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """407 ⇒ the recorder holds nothing; the failed tunnel's attempt carries no port.

        A fresh adapter per drive (inside ``_drive``) means the wrong-password
        ``EgressConfig`` reaches a session that has never talked to the proxy.
        """
        result = await _drive().drive_with_egress(
            sealed_network,
            egress=_egress_for(sealed_network, password="wrong"),
            monkeypatch=monkeypatch,
        )

        assert result.status != 200, f"drive answered {result.status} on wrong proxy credentials"
        assert result.connections == [], (
            f"recorder accepted {len(result.connections)} connection(s) through a failed "
            "tunnel: failed tunnels must contribute none (§5.2.1)"
        )
        assert len(result.attempts) >= 1, "the proxy should have observed the 407 attempt"
        # Every attempt in this drive is unauthenticated (a 407), and
        # therefore none can carry a tunnel source port: the proxy answered
        # 407 before open_connection() could succeed, so §5.2.1's
        # "tunnel-establishment ⇒ source_port" key was never written.
        assert all(a.source_port is None for a in result.attempts), (
            f"some attempt carried a tunnel source port despite the failed authentication: "
            f"{[(bool(a.authenticated), a.source_port) for a in result.attempts]}"
        )


# ── Phase 3 — falsification (§5.2.2 row 3) ────────────────────────────────


@pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
class TestPhase3Falsification:
    """An injected product bypass makes the harness fail — the §1.4 rule for containment.

    The patch site is ``kitty.providers.ollama_cloud``, not ``kitty.egress``:
    the adapter does ``from kitty.egress import aiohttp_session_kwargs`` at
    import time, so ``_get_session`` resolves its own module's binding and a
    patch on the egress module is silently ignored. This is the exact
    fail-by-silence the falsification exists to prevent — the same shape
    T-E2's ``should_bypass`` patch and T-E3's ``get_egress`` patch pin on
    their own modules.
    """

    async def test_injected_bypass_makes_the_harness_report_it(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patched ``aiohttp_session_kwargs`` ⇒ the recorder sees a connection the proxy cannot explain.

        The patch makes ``_get_session`` build its session **without** the
        proxy kwargs while the egress configuration is still installed — a
        session built without the proxy, §5.2.2's second named bypass shape.
        The resolver hook is still in scope (it lives in ``direct_route``,
        not in the egress config), so the direct request genuinely reaches
        the recorder and the detection assertion has real evidence to read.
        """
        import kitty.providers.ollama_cloud  # noqa: PLC0415 -- needed before monkeypatch.setattr can find the attribute

        monkeypatch.setattr(kitty.providers.ollama_cloud, "aiohttp_session_kwargs", lambda: {})

        result = await _drive().drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network), monkeypatch=monkeypatch
        )

        # The bypass worked: the adapter reached the recorder directly (the
        # resolver mapping was still in scope) and answered 200.
        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert result.attempts == [], (
            f"proxy saw {len(result.attempts)} attempt(s) under the bypass — the adapter "
            "was expected to skip the proxy entirely"
        )
        # And the harness detected it: every peer port is unexplained by any
        # tunnel, which is precisely the assertion §5.2.1 pins.
        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        assert peer_ports, "recorder accepted no connections; the bypass was not exercised"
        unattributable = unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts)
        assert unattributable == peer_ports, (
            f"unattributable peer ports {unattributable} != all peer ports {peer_ports}: "
            "the harness failed to detect a deliberate bypass"
        )


# ── Drive with egress (R1) — the green path the phases rest on ────────────


class TestDriveWithEgress:
    """``ProviderAiohttpContainment.drive_with_egress`` — the phase-2/2b/3 entry point.

    The green-path exercise sends the adapter's session through the harness
    CONNECT proxy over TLS; it carries the same Python ≥3.11 skip as the
    proxied phases (bpo-44011 — see the module docstring). The static
    hostname/resolve-map check in this class does not drive the bridge, so
    it still runs on every supported Python.
    """

    @pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
    async def test_drive_with_egress_one_request_reaches_the_recorder_through_the_proxy(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Green path: egress enabled, proxy up, one request lands on the recorder.

        This is the baseline the phase-2/2b/3 assertions rest on. With the
        proxy up and authenticated, the adapter constructs its proxied
        session, the proxy records one authenticated CONNECT with a
        non-``None`` source port targeting the harness hostname, and the
        recorder's connection log shows the upstream connection on the
        matching source port — the §5.2.1 join key, exercised once.
        """
        egress = _egress_for(sealed_network)
        result = await _drive().drive_with_egress(sealed_network, egress=egress, monkeypatch=monkeypatch)

        assert isinstance(result, Phase1Result)
        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert len(result.captures) == 1, f"recorder saw {len(result.captures)} capture(s), expected exactly 1"
        assert any(c.peer_port and c.peer_port > 0 for c in result.connections), (
            "recorder's connection log has no entry with a real peer port"
        )
        assert len(result.attempts) == 1, f"proxy saw {len(result.attempts)} CONNECT attempt(s), expected exactly 1"
        assert result.attempts[0].authenticated is True
        assert result.attempts[0].source_port is not None, (
            "the proxy's tunnel source port is the §5.2.1 join key; it must be set "
            "for an authenticated, established tunnel"
        )
        assert result.attempts[0].target.startswith(f"{HARNESS_UPSTREAM_HOST}:"), (
            f"CONNECT target {result.attempts[0].target!r} is not the harness hostname: "
            "the base-URL seam is not live, and this tunnel may lead somewhere else"
        )
        # §5.2.1's join holds: the recorder's peer port equals the proxy's
        # tunnel source port — same kernel connection seen from both ends.
        peer_ports = {c.peer_port for c in result.connections if c.peer_port and c.peer_port > 0}
        assert result.attempts[0].source_port in peer_ports, (
            f"recorder's peer ports {peer_ports} do not include the proxy's tunnel "
            f"source port {result.attempts[0].source_port} — §5.2.1's join broken"
        )

    async def test_drive_targets_the_harness_hostname_not_loopback(self, sealed_network: SealedNetwork) -> None:
        """§5.3 trap 1: the upstream is addressed by the harness hostname, never loopback.

        A loopback-configured base URL would be reached without the proxy's
        name translation and the containment assertion would pass while
        proving the opposite of its claim. The adapter's base URL is the
        sealed network's ``https://upstream.kitty-test.invalid:{port}`` — the
        non-loopback name — and the proxy's resolve map is what maps that
        name to the recorder's loopback port.
        """
        assert sealed_network.upstream_base_url.startswith(f"https://{HARNESS_UPSTREAM_HOST}"), (
            f"upstream base url {sealed_network.upstream_base_url!r} is not the harness "
            f"hostname {HARNESS_UPSTREAM_HOST!r}: §5.3 trap 1, loopback addresses the "
            "bridge direct"
        )
        target = f"{HARNESS_UPSTREAM_HOST}:{sealed_network.upstream_port}"
        assert target in sealed_network.proxy.resolve, (
            f"proxy resolve map is missing {target!r}: the CONNECT target has no loopback translation"
        )


# ── The OAuth login leg — the startup path §5.5 will not leave out ────────


class TestTheOAuthLoginLegContainment:
    """The four containment phases for the OAuth login leg.

    The leg has no adapter and no bridge: the product reaches the network
    through the two token-exchange coroutines in :mod:`kitty.auth.openai_oauth`,
    and its session is built at startup with ``aiohttp_session_kwargs()``.
    The drives here build the session through that same module binding (see
    :func:`_drive_oauth` for why the binding, not the helper, is load-bearing)
    and drive the product's own coroutine, so the containment claim is about
    the product's construction kwargs and the product's request — not about a
    session the test happened to build differently.
    """

    async def test_the_oauth_login_leg_reaches_the_recorder_with_egress_disabled(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Direct leg (§5.2.2 phase 1 analogue): the token POST lands on the recorder.

        Runs on every supported Python — the direct TLS leg needs no
        TLS-in-TLS.
        """
        result = await _drive_oauth(sealed_network, _drive(), egress=None)

        assert result.ok, f"the token exchange failed: {result.error}"
        captures = [c for c in sealed_network.recorder.requests if c.path == OAUTH_TOKEN_SUFFIX]
        assert len(captures) == 1, f"recorder saw {len(captures)} /oauth/token capture(s), expected exactly 1"
        assert captures[0].peer_port, "containment cannot join a capture with no peer port"
        assert sealed_network.proxy.attempts == [], (
            f"proxy saw {len(sealed_network.proxy.attempts)} CONNECT attempt(s) with egress off"
        )

    @pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
    async def test_the_oauth_login_leg_joins_every_connection_to_a_tunnel(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Proxied (§5.2.2 phase 2b analogue): the POST tunnels and the join holds."""
        result = await _drive_oauth(sealed_network, _drive(), egress=_egress_for(sealed_network))

        assert result.ok, f"the token exchange failed through the proxy: {result.error}"
        captures = [c for c in sealed_network.recorder.requests if c.path == OAUTH_TOKEN_SUFFIX]
        assert len(captures) == 1, f"recorder saw {len(captures)} /oauth/token capture(s), expected exactly 1"
        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        assert peer_ports, "recorder accepted no connections; the drive never reached it"
        assert unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts) == [], (
            f"recorder peer ports {peer_ports} not covered by tunnel source ports "
            f"{[a.source_port for a in sealed_network.proxy.attempts]}: §5.2.1's join broken"
        )
        # The swap-live check: the tunnel targeted the harness hostname — a
        # missed `harness_oauth_token_url` swap would point the leg at
        # auth.openai.com, which the proxy's deny map blackholes (and this
        # assertion would name the wrong target).
        assert all(a.target.startswith(f"{HARNESS_UPSTREAM_HOST}:") for a in sealed_network.proxy.attempts), (
            f"proxy saw CONNECT targets {[a.target for a in sealed_network.proxy.attempts]}, "
            f"expected only {HARNESS_UPSTREAM_HOST!r}: the token-URL seam is not live"
        )

    @pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
    async def test_the_oauth_login_leg_connects_nowhere_with_the_proxy_down(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
    ) -> None:
        """Proxy down (§5.2.2 phase 2 analogue): the exchange fails; nothing arrives anywhere.

        The negative assertion for the startup leg: with the gateway
        stopped, the login must fail — not fall back to a direct route. The
        exchange fails fast (the proxy's listener is gone; the leg has no
        retry ladder), so this test is cheap.
        """
        await sealed_network.proxy.stop()

        result = await _drive_oauth(sealed_network, _drive(), egress=_egress_for(sealed_network))

        assert not result.ok, "the token exchange succeeded with the proxy down: a direct fallback"
        assert sealed_network.recorder.requests == [], (
            f"recorder saw {len(sealed_network.recorder.requests)} capture(s) with the "
            "proxy stopped: the login leg reached the upstream directly"
        )
        assert sealed_network.recorder.connections == [], (
            f"recorder accepted {len(sealed_network.recorder.connections)} connection(s) with the proxy stopped"
        )
        assert sealed_network.proxy.attempts == [], (
            f"proxy saw {len(sealed_network.proxy.attempts)} CONNECT attempt(s) after being stopped"
        )

    @pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
    async def test_an_oauth_leg_bypass_makes_the_harness_report_it(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patched ``openai_oauth.aiohttp_session_kwargs`` ⇒ the harness detects the bypass.

        The patch site is :mod:`kitty.auth.openai_oauth` — the module whose
        binding the drive's session construction resolves (see
        :func:`_drive_oauth`). A patch on ``kitty.egress`` would leave the
        drive unaffected, which is why the binding choice is load-bearing:
        the falsification must demonstrate that a session built without the
        proxy under a live egress config is **detected**, not that a patch
        somewhere did nothing.
        """
        monkeypatch.setattr(openai_oauth, "aiohttp_session_kwargs", lambda: {})

        result = await _drive_oauth(sealed_network, _drive(), egress=_egress_for(sealed_network))

        # The bypass worked: the leg reached the recorder directly (the
        # resolver hook was still in scope) and the exchange completed.
        assert result.ok, f"the bypassed exchange failed: {result.error}"
        assert sealed_network.proxy.attempts == [], (
            f"proxy saw {len(sealed_network.proxy.attempts)} attempt(s) under the bypass "
            "— the leg was expected to skip the proxy entirely"
        )
        # And the harness detected it: every peer port is unexplained by any
        # tunnel, which is precisely the assertion §5.2.1 pins.
        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        assert peer_ports, "recorder accepted no connections; the bypass was not exercised"
        unattributable = unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts)
        assert unattributable == peer_ports, (
            f"unattributable peer ports {unattributable} != all peer ports {peer_ports}: "
            "the harness failed to detect a deliberate bypass on the OAuth leg"
        )


# ── Verdict recording (R9) ────────────────────────────────────────────────
#
# Deliberately the LAST test class in the file: pytest's default collection
# order runs tests in declaration order within a file, so the verdict test
# below runs after every phase test and after the drive baseline. The
# recording itself lives in the session finaliser in
# `tests/harness/conftest.py` (`_record_provider_aiohttp_slice_verdict_at_session_end`),
# which runs after every test in the process; this class asserts the
# *precondition* the finaliser reads. See the aiohttp slice's identical
# block comment for the reordering-plugin caveat.


@pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
class TestSliceVerdict:
    """The verdict gate's precondition: every gated test actually ran and passed.

    Skipped on Python <3.11 alongside the proxied phases (bpo-44011 — see
    the module docstring): the assertion's premise — every gated test
    actually ran — is false on those interpreters, and a red verdict test
    would make CI red for a slice that has been honestly *not* proven on
    that Python rather than honestly proven.
    """

    def test_every_phase_actually_ran_and_passed(self) -> None:
        """One tracked outcome per gated test, all ``PASSED``.

        The assertion names any missing or non-passing phase so the fix is
        local: a rename means editing
        :data:`harness.conftest._PROVIDER_AIOHTTP_PHASE_TEST_NAMES`; a skip
        means the interpreter's floor was not met and the verdict is
        correctly not recorded.
        """
        from harness.conftest import (  # noqa: PLC0415 -- local like the siblings'
            _PROVIDER_AIOHTTP_PHASE_TEST_NAMES,
            _PhaseOutcome,
            _provider_phase_outcomes,
        )

        assert len(_provider_phase_outcomes) == len(_PROVIDER_AIOHTTP_PHASE_TEST_NAMES), (
            f"phase outcomes tracked for {len(_provider_phase_outcomes)} test(s), expected "
            f"{len(_PROVIDER_AIOHTTP_PHASE_TEST_NAMES)}: a phase test renamed or removed without "
            "updating the conftest's name set, or a phase never ran"
        )
        for name in sorted(_PROVIDER_AIOHTTP_PHASE_TEST_NAMES):
            assert name in _provider_phase_outcomes, f"phase {name!r} never ran"
            assert _provider_phase_outcomes[name] is _PhaseOutcome.PASSED, (
                f"phase {name!r} outcome is {_provider_phase_outcomes[name].value}: the slice "
                "verdict is correctly not recorded (T-E9's gate will see "
                "not_attempted until every phase passes on this interpreter)"
            )


# ── Registration ──────────────────────────────────────────────────────────


register_containment_transport(ProviderAiohttpContainment.name, ProviderAiohttpContainment)
