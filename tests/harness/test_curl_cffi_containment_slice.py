"""The complete, falsified curl_cffi containment slice — T-E3 ([KBR-63]).

`.system_design/TEST_SUITE.md` §5.2.1, §5.2.2, §5.3, §5.5, §7.2.1, §7.2.3 ·
plan task **T-E3**
([KBR-63](https://shelpuk.atlassian.net/browse/KBR-63)) ·
`.requirements/20260916T093044Z_kbr63_t_e3_curl_cffi_containment_slice/REQUIREMENTS.md`.

The aiohttp slice (T-E2, :mod:`harness.test_aiohttp_containment_slice`) proved
§5.2.2's four phases for the bridge's own serving session. This module proves
the same four phases for the **curl_cffi** transport — the ``AsyncSession``
`OpenAISubscriptionAdapter` builds for the Codex backend (§5.5's second row) —
and records the slice's ``proven`` verdict into the capability report.

**What carries the containment claim here.** A real
:class:`~kitty.bridge.server.BridgeServer` runs the real adapter; the drive
points the adapter's upstream constant at
``https://upstream.kitty-test.invalid:{port}/responses`` (a read-swap-restore
of ``_CODEX_BACKEND_URL``, T-B2's seam shape re-aimed at the harness hostname
— the adapter ignores ``provider_config["base_url"]``, §7.5.2), manages the
ambient egress global (the adapter reads ``get_egress()`` at session
construction, `src/kitty/providers/openai_subscription.py:_new_curl_session`),
and trusts the harness CA through the adapter's own ``CODEX_CA_CERTIFICATE``
seam. The OAuth session file comes from
:func:`harness.curl_cffi.seed_oauth_session` — the adapter reads
``cc_request["_resolved_key"]`` as a *path*, so a literal ``"harness-key"``
would crash.

**The direct route (§5.3).** curl resolves names itself, so the harness
hostname needs curl's own mapping: ``CURLOPT_RESOLVE`` carrying
``host:port:127.0.0.1``. :class:`CurlCffiContainment.direct_route` enters a
constructor patch on ``curl_cffi.requests.AsyncSession`` (the exact name the
product calls) so the session is *built* with the mapping — the session pools
curl handles (``max_clients``), so setting the option on one handle after
construction would miss the handles the pool actually draws. The pinned
curl_cffi 0.16.3 accepts no ``resolve=`` kwarg; ``curl_options={CurlOpt.RESOLVE:
[...]}`` is the only channel. The mapping stays in scope on the proxied phases
too (§5.3 trap 2): a product that wrongly falls back to a direct route still
resolves the harness hostname to the recorder, and §5.2.1's join detects the
bypass instead of the test passing vacuously on a DNS failure.

**Python <3.11 floor.** Phase 1 (direct leg) runs on every supported
Python. The proxied phases (2/2b/3) carry the same Python ≥3.11
``skipif`` as the aiohttp slice — the owner's scope decision on
KBR-63 (comment, 2026-09-16), imported as the standard wording from
T-E2's file. On interpreters below 3.11 the verdict gate in
:mod:`harness.conftest` records this slice's row as ``UNSUPPORTED``
with the documented floor reason, so the plan's "an outcome is
recorded" done-when holds on every supported interpreter and T-E9's
completeness gate (which accepts ``UNSUPPORTED`` as a permitted
partial delivery) sees a claimed row. CI's 3.10 leg exercises the
writer end-to-end.

**The refresh leg is out of scope.** The seeded OAuth session is fresh, so no
token refresh fires during a drive; the refresh leg's own containment drive is
T-E5/T-E8 scope. Both legs share ``_new_curl_session`` (KBR-161), so the
egress mapping proven here is the one the refresh leg builds with too.

**§7.2.1's failed-handshake shape is covered, not excluded.** T-B2 resolved
the recorder's connection-logging limitation at socket level (protocol-factory
logging, ``peer_port = -1`` until the handshake completes), so this slice's
bypass detection includes the connection that opens and then fails TLS
negotiation — the shape KBR-27 asked T-E3 to state. A ``proven`` verdict
therefore needs **no excluded-shape qualifier**; ``record()``'s contract
(no reason on ``proven``) is not bent to carry one.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
``test_containment.py`` and the aiohttp slice. The reason is §8.2's and only
§8.2's: ``l3`` is in ``PENDING_ACTIVATION_LAYERS``, so an ``l3`` marker today
would leave this slice's correctness checked by no job at all. T-K6 inherits
the relocation for this file alongside the other two.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from curl_cffi import CurlOpt
from curl_cffi import requests as curl_requests

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
from harness.curl_cffi import seed_oauth_session
from harness.curl_recorder import CODEX_RESPONSES_SUFFIX, CurlRecordingUpstream
from harness.test_aiohttp_containment_slice import (
    _AIOHTTP_NEEDS_311 as _NEEDS_311,
)
from harness.test_aiohttp_containment_slice import (
    _AIOHTTP_SKIP_REASON as _SKIP_REASON,
)
from kitty.egress import EgressConfig, get_egress, set_egress
from kitty.providers import openai_subscription
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def sealed_network(certs: CertFiles) -> AsyncGenerator[SealedNetwork, None]:
    """One started ``SealedNetwork`` hosting a curl_cffi recorder, torn down on exit.

    The factory hands :class:`~harness.curl_recorder.CurlRecordingUpstream` —
    OpenAI Responses vocabulary (the adapter speaks that upstream, P17) and
    socket-level connection logging (§7.2.1's failed-handshake shape stays
    visible) — into the *same* proxy+recorder pair the aiohttp slice uses, so
    the §5.2.1 resolve map and lifecycle guards are shared, not re-derived.

    Each phase gets a **fresh** sealed network, for the same reason T-E2's
    fixture records: phase 1's direct connection leaves a peer-port row in
    ``recorder.connections``; sharing one network would let phase 3's
    "unattributable port" assertion pass on phase 1's leak rather than on the
    injected bypass.

    Yields:
        The running harness.
    """

    def _factory(ctx: ssl.SSLContext) -> CurlRecordingUpstream:
        return CurlRecordingUpstream(ssl_context=ctx, default_format=WireFormat.OPENAI_RESPONSES)

    net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs, recorder_factory=_factory)
    await net.start()
    try:
        yield net
    finally:
        await net.stop()


@pytest.fixture
def curl_trusts_test_ca(monkeypatch: pytest.MonkeyPatch, certs: CertFiles) -> None:
    """Point the adapter's CA seam at the harness CA for the duration of a test.

    The adapter reads ``CODEX_CA_CERTIFICATE`` (Codex CLI's ``custom_ca.rs``
    precedence: it wins over ``SSL_CERT_FILE``), so the recorder's certificate
    — signed by the harness CA — verifies on both the direct and the proxied
    leg without touching the process certificate store.

    Args:
        monkeypatch: Pytest's monkeypatch fixture; the variable reverts on
            teardown.
        certs: The session's throwaway certificate set; ``ca`` is the CA the
            recorder's leaf is signed by.
    """
    monkeypatch.setenv("CODEX_CA_CERTIFICATE", str(certs.ca))


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    """Reset the capability-report singleton before every test in this module.

    The verdict test at the bottom records ``PROVEN`` for ``curl_cffi``; every
    earlier test must see an untouched report so its phase assertions do not
    depend on collection order. The session finalisers in
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
        The configuration the drive installs into the egress global and hands
        to ``BridgeServer``.
    """
    return proxy_config(net.proxy.port, password=password)


def _drive() -> CurlCffiContainment:
    """Build a fresh containment transport for one drive.

    ``CurlCffiContainment`` is stateless, so a fresh instance per call is
    bookkeeping rather than necessity — but it keeps the drive's lifetime
    visible and mirrors the per-drive contract the protocol documents.

    Returns:
        A fresh transport.
    """
    return CurlCffiContainment()


# ── The serving-leg seam (§7.5.2's custom-transport rule) ─────────────────


@contextlib.contextmanager
def harness_codex_url(harness: SealedNetwork) -> Iterator[str]:
    """Point the serving leg's upstream constant at the **harness hostname**.

    ``OpenAISubscriptionAdapter`` reaches its upstream through the module
    constant ``_CODEX_BACKEND_URL`` with no configuration channel of any kind
    (§7.5.2) — the same read-swap-restore T-B2's :func:`harness.curl_cffi.codex_backend_url`
    performs, re-aimed: that seam yields the recorder's **loopback** base URL,
    which would address the bridge direct and make §5.3 trap 1 vacuous. This
    seam yields ``https://upstream.kitty-test.invalid:{port}/responses`` — the
    non-loopback name the proxy's resolve map translates.

    Reading the constant before writing it is what makes the seam falsifiable:
    if the adapter is ever rewritten to read the URL from somewhere else, this
    raises :class:`AttributeError` rather than swapping a name nothing
    consults and leaving a test to pass with an empty capture list.

    Args:
        harness: A **started** sealed network; its upstream URL is read here.

    Yields:
        The URL the serving leg will now post to.

    Raises:
        AttributeError: When
            :mod:`kitty.providers.openai_subscription` no longer defines
            ``_CODEX_BACKEND_URL``.
    """
    url = f"{harness.upstream_base_url}{CODEX_RESPONSES_SUFFIX}"
    if not hasattr(openai_subscription, "_CODEX_BACKEND_URL"):
        raise AttributeError(
            "kitty.providers.openai_subscription no longer defines "
            "_CODEX_BACKEND_URL; the serving leg reads its upstream URL "
            "elsewhere and this seam no longer redirects it"
        )
    original: Any = openai_subscription._CODEX_BACKEND_URL
    openai_subscription._CODEX_BACKEND_URL = url
    try:
        yield url
    finally:
        # Restored in a `finally` rather than by pytest's monkeypatch, so the
        # seam is usable from anything — a containment drive included.
        openai_subscription._CODEX_BACKEND_URL = original


# ── The containment transport (§5.2.2's extension interface) ──────────────


class CurlCffiContainment(ContainmentTransport):
    """The curl_cffi containment transport: the adapter's own serving session.

    Drives one request through a real :class:`~kitty.bridge.server.BridgeServer`
    running the real :class:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter`,
    with the harness-hostname upstream swap, the ambient egress global, the
    adapter's CA seam, and the resolve mapping in scope. Constructed per
    drive; no state lives on the instance.

    Attributes:
        name: The registry key, ``"curl_cffi"``.
    """

    name = "curl_cffi"

    @contextlib.contextmanager
    def direct_route(self, harness: SealedNetwork) -> Iterator[None]:
        """Put curl's resolve mapping in scope for the drive (§5.3).

        The mapping is ``{host}:{port}:127.0.0.1:{port}`` for the harness
        hostname at the recorder's port, injected by patching
        ``curl_cffi.requests.AsyncSession`` — the exact constructor call the
        product makes (``_new_curl_session``) — so the session is **built**
        with the option. A post-construction ``setopt`` would miss the other
        handles the session's pool (``max_clients``) draws; the
        ``curl_options`` channel applies per request from the session, so
        every handle inherits it. The pinned curl_cffi 0.16.3 accepts no
        ``resolve=`` kwarg; ``curl_options={CurlOpt.RESOLVE: [...]}`` is the
        only channel.

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
        entry = f"{harness.upstream_host}:{harness.upstream_port}:127.0.0.1"
        real_cls = curl_requests.AsyncSession

        def _session_with_resolve(**kwargs: Any) -> Any:
            """Build the real session with the resolve entry merged in.

            Args:
                **kwargs: The product's own session kwargs
                    (``impersonate``, ``verify``, ``proxies``,
                    ``curl_options``); forwarded unchanged apart from the
                    resolve entry merged into ``curl_options``.

            Returns:
                The real ``AsyncSession`` the product would have built, plus
                the resolve mapping.
            """
            opts = dict(kwargs.pop("curl_options", None) or {})
            opts[CurlOpt.RESOLVE] = [entry]
            return real_cls(curl_options=opts, **kwargs)

        original = curl_requests.AsyncSession
        curl_requests.AsyncSession = _session_with_resolve  # type: ignore[assignment]
        try:
            yield
        finally:
            # Restored in a `finally` rather than by pytest's monkeypatch: the
            # protocol's `direct_route` takes no fixture, and the house style
            # (T-B2's seams) keeps these usable outside a test body.
            curl_requests.AsyncSession = original

    async def drive_phase_1(
        self,
        harness: SealedNetwork,
        *,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> Phase1Result:
        """Drive one request through the bridge with egress off (§5.2.2 phase 1).

        Args:
            harness: The sealed network the request is driven against.
            monkeypatch: Pytest's monkeypatch fixture; reserved for the
                calling test's own patches (the direct route manages its own
                scope).
            tmp_path: Where the seeded OAuth session file lives; the adapter
                reads ``cc_request["_resolved_key"]`` as a path.

        Returns:
            A :class:`Phase1Result` with the recorder's observations. With
            egress off and the resolve mapping in scope, ``status == 200``,
            ``len(captures) == 1``, ``attempts == []``.
        """
        return await self._drive(harness, egress=None, monkeypatch=monkeypatch, tmp_path=tmp_path)

    async def drive_with_egress(
        self,
        harness: SealedNetwork,
        *,
        egress: EgressConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> Phase1Result:
        """Drive one request through the bridge with egress configured (§5.2.2 phases 2/2b/3).

        The egress configuration reaches the adapter through the **ambient
        global** (:func:`kitty.egress.set_egress`), which
        ``_new_curl_session`` reads when the session is first built — the
        proxy mapping enters there, not through ``BridgeServer``. The prior
        global value is captured and restored exactly, so the drive does not
        own the process default. ``BridgeServer`` also receives ``egress=``
        so its own outbound path honours the same configuration; the two
        channels are independent.

        The resolve mapping stays in scope exactly as in phase 1. A product
        defect that wrongly falls back to a direct route therefore still
        resolves the harness hostname to the recorder, and §5.2.1's join
        detects the bypass instead of the test passing vacuously on a DNS
        failure (§5.3 trap 2).

        Args:
            harness: The sealed network the request is driven against.
            egress: The configuration to install. Typical value:
                ``_egress_for(harness)``; a wrong password produces the 407
                subcase.
            monkeypatch: Pytest's monkeypatch fixture; reserved for the
                calling test's own patches.
            tmp_path: Where the seeded OAuth session file lives.

        Returns:
            A :class:`Phase1Result` with the recorder's observations.
        """
        return await self._drive(harness, egress=egress, monkeypatch=monkeypatch, tmp_path=tmp_path)

    async def _drive(
        self,
        harness: SealedNetwork,
        *,
        egress: EgressConfig | None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> Phase1Result:
        """The drive both entry points share; ``egress=None`` is phase 1.

        The context stack, innermost last: the egress global, the
        harness-hostname upstream swap, the resolve mapping. A failure inside
        the drive propagates *after* the global is restored — the ``finally``
        ordering is what keeps the process default intact on a failing drive.

        Args:
            harness: The sealed network the request is driven against.
            egress: The configuration to install, or ``None`` for the direct
                leg.
            monkeypatch: Pytest's monkeypatch fixture (unused here; the
                parameter keeps the two entry points' signatures identical).
            tmp_path: Where the seeded OAuth session file lives.

        Returns:
            A :class:`Phase1Result` with the bridge's status, the recorder's
            captures and connections after the request, and the proxy's
            attempts.
        """
        # Local import keeps the module-import surface tidy: the bridge server
        # depends on `kitty.providers.*`, which the rest of the test module
        # surface (the report, the recorder) does not need.
        from kitty.bridge.server import BridgeServer

        # A fresh adapter per drive: the session (and its egress mapping) is
        # built lazily on the first request, so the configuration below is
        # what the session sees. A fresh OAuth session file per drive, because
        # `seed_oauth_session` stamps `now`-relative expiry — a stale file
        # from an earlier phase would fire the refresh leg, which is T-E5's
        # scope, not this slice's.
        oauth_key = seed_oauth_session(tmp_path)

        saved_egress = get_egress()
        status: int = -1
        text: str = ""
        try:
            set_egress(egress)
            with harness_codex_url(harness), self.direct_route(harness):
                adapter = OpenAISubscriptionAdapter()
                server = BridgeServer(
                    None,  # type: ignore[arg-type]
                    adapter,
                    oauth_key,
                    model=MODEL,
                    provider_config={},
                    egress=egress,
                )
                try:
                    bridge_port = await server.start_async()

                    # The same body the T-B2 round-trip drives: Responses
                    # inbound, trivial content, non-streaming — the recorder
                    # replies with its minimal SSE success and the bridge
                    # parses it. The marker is unique per drive so a capture
                    # from an earlier phase cannot satisfy a later assertion.
                    sent = marker()
                    body = minimal_inbound_body(InboundProtocol.RESPONSES, sent)

                    async with aiohttp.ClientSession() as client:
                        response = await client.post(
                            f"http://127.0.0.1:{bridge_port}{inbound_path(InboundProtocol.RESPONSES)}",
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


#: A budget for one drive request (T-E1's value). The green path completes in
#: well under a second; the proxied-failure phases (2, 407) see the client
#: timeout fire while the bridge works through its ~30 s transport-grace
#: ladder, and the drive's ``status`` comes back as ``-1`` — which is what
#: those phases assert on.
_DRIVE_TIMEOUT = 10.0


# ── The serving-leg seam's own falsification ──────────────────────────────


class TestHarnessCodexUrlSeam:
    """The upstream swap is live, restores, and refuses a name nothing defines."""

    async def test_the_harness_hostname_seam_swaps_and_restores(self, sealed_network: SealedNetwork) -> None:
        """Inside the block the constant carries the harness hostname; after, the original."""
        original = openai_subscription._CODEX_BACKEND_URL
        with harness_codex_url(sealed_network) as url:
            assert url.startswith(f"https://{HARNESS_UPSTREAM_HOST}"), (
                f"seam yielded {url!r}, not the harness hostname: §5.3 trap 1, a "
                "loopback URL addresses the bridge direct"
            )
            assert url == openai_subscription._CODEX_BACKEND_URL
        assert original == openai_subscription._CODEX_BACKEND_URL, (
            "the seam did not restore the original constant: every later test in "
            "the process would post to the recorder"
        )

    async def test_the_seam_refuses_to_swap_a_name_nothing_defines(
        self, sealed_network: SealedNetwork, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A product rewrite that reads its URL elsewhere fails loudly, not silently."""
        monkeypatch.delattr(openai_subscription, "_CODEX_BACKEND_URL")
        with pytest.raises(AttributeError, match="_CODEX_BACKEND_URL"), harness_codex_url(sealed_network):
            pass


# ── Phase 1 — positive control (§5.2.2 row 1) ─────────────────────────────


class TestPhase1PositiveControl:
    """Egress off: the bridge reaches the recorder directly and the proxy sees nothing.

    The control for everything below: without it, "the upstream received
    nothing" is satisfied just as well by a destination that is unreachable
    for an unrelated reason (§5.3 trap 2). Runs on every supported Python —
    the direct leg needs no TLS-in-TLS.
    """

    async def test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing(
        self,
        sealed_network: SealedNetwork,
        curl_trusts_test_ca: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Green path on the TLS recorder: 200, one capture, real peer port, zero attempts."""
        result = await _drive().drive_phase_1(sealed_network, monkeypatch=monkeypatch, tmp_path=tmp_path)

        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert len(result.captures) == 1, f"recorder saw {len(result.captures)} capture(s)"
        assert any(c.peer_port and c.peer_port > 0 for c in result.connections), (
            "recorder's connection log has no entry with a real peer port: the §5.2.1 "
            "join has nothing to bind to"
        )
        assert result.attempts == [], (
            f"proxy saw {len(result.attempts)} CONNECT attempt(s) with egress off"
        )


# ── Phase 2 — containment under proxy failure (§5.2.2 row 2) ──────────────


@pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
class TestPhase2ContainmentProxyDown:
    """Proxy down ⇒ upstream accepts zero connections, request fails, no direct fallback.

    The negative assertion §5.1 gap 2 and §5.2.2 row 2 specify. Bracketed by
    phase 1 (a reachable destination) per §5.3 trap 2. The drive's client
    timeout fires while the bridge works through its ~30 s transport-grace
    ladder (``_TRANSPORT_GRACE_DELAYS``, server.py), so ``status`` comes back
    as ``-1`` — which is what the first assertion reads.
    """

    async def test_proxy_down_leaves_the_recorder_with_zero_connections(
        self,
        sealed_network: SealedNetwork,
        curl_trusts_test_ca: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stop the proxy, drive once, assert the recorder holds nothing and the drive failed."""
        await sealed_network.proxy.stop()

        result = await _drive().drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network), monkeypatch=monkeypatch, tmp_path=tmp_path
        )

        assert result.status != 200, (
            f"bridge answered {result.status} with the proxy down: a 200 means the bridge "
            "reached the recorder anyway — the containment premise is broken"
        )
        assert result.connections == [], (
            f"recorder accepted {len(result.connections)} connection(s) with the proxy "
            "stopped: the bridge fell back to a direct route, which is the direct "
            "fallback §5.2.2 phase 2 exists to catch"
        )
        assert result.captures == [], (
            f"recorder saw {len(result.captures)} capture(s) with the proxy stopped"
        )
        assert result.attempts == [], (
            f"proxy saw {len(result.attempts)} CONNECT attempt(s) after being stopped"
        )


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
        curl_trusts_test_ca: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three drives, every recorded peer port matches a tunnel source port."""
        egress = _egress_for(sealed_network)
        for _ in range(3):
            result = await _drive().drive_with_egress(
                sealed_network, egress=egress, monkeypatch=monkeypatch, tmp_path=tmp_path
            )
            assert result.status == 200, f"drive answered {result.status}: {result.text!r}"

        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        assert peer_ports, "recorder accepted no connections; the drive never reached it"
        assert unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts) == [], (
            f"recorder peer ports {peer_ports} not covered by tunnel source ports "
            f"{[a.source_port for a in sealed_network.proxy.attempts]}: §5.2.1's join broken"
        )
        # The swap-live check: every tunnel targeted the harness hostname. A
        # missing or silently-failed `_CODEX_BACKEND_URL` swap would point the
        # adapter at the real `chatgpt.com`, which the proxy would resolve and
        # connect to — real egress from a test, and the join below could not
        # tell it from a tunnel to the recorder.
        assert all(a.target.startswith(f"{HARNESS_UPSTREAM_HOST}:") for a in sealed_network.proxy.attempts), (
            f"proxy saw CONNECT targets {[a.target for a in sealed_network.proxy.attempts]}, "
            f"expected only {HARNESS_UPSTREAM_HOST!r}: the upstream swap is not live"
        )

    async def test_failed_tunnel_contributes_no_upstream_connection(
        self,
        sealed_network: SealedNetwork,
        curl_trusts_test_ca: None,
        tmp_path: Path,
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
            tmp_path=tmp_path,
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

    The patch site is ``kitty.providers.openai_subscription``, not
    ``kitty.egress``: the adapter calls the ``get_egress`` name resolved in
    its own module's namespace, so a patch on the egress module is silently
    ignored. This is the curl_cffi analogue of T-E2's ``should_bypass`` patch
    landing on ``kitty.bridge.server`` — the exact fail-by-silence the
    falsification exists to prevent.
    """

    async def test_injected_bypass_makes_the_harness_report_it(
        self,
        sealed_network: SealedNetwork,
        curl_trusts_test_ca: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patched ``get_egress`` ⇒ the recorder sees a connection the proxy cannot explain.

        The patch makes ``_new_curl_session`` build its session **without**
        the proxy mapping while the egress configuration is still installed —
        a session built without the proxy, §5.2.2's second named bypass
        shape. The resolve mapping is still in scope (it lives in
        ``direct_route``, not in the egress config), so the direct request
        genuinely reaches the recorder and the detection assertion has real
        evidence to read.
        """
        monkeypatch.setattr(openai_subscription, "get_egress", lambda: None)

        result = await _drive().drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network), monkeypatch=monkeypatch, tmp_path=tmp_path
        )

        # The bypass worked: the bridge reached the recorder directly (the
        # resolve mapping was still in scope) and answered 200.
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
    """``CurlCffiContainment.drive_with_egress`` — the phase-2/2b/3 entry point.

    The green-path exercise sends the adapter's session through the harness
    CONNECT proxy over TLS; it carries the same Python ≥3.11 skip as the
    proxied phases (owner scope decision — see the module docstring). The
    static hostname/resolve-map check in this class does not drive the
    bridge, so it still runs on every supported Python.
    """

    @pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
    async def test_drive_with_egress_one_request_reaches_the_recorder_through_the_proxy(
        self,
        sealed_network: SealedNetwork,
        curl_trusts_test_ca: None,
        tmp_path: Path,
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
        result = await _drive().drive_with_egress(
            sealed_network, egress=egress, monkeypatch=monkeypatch, tmp_path=tmp_path
        )

        assert isinstance(result, Phase1Result)
        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert len(result.captures) == 1, (
            f"recorder saw {len(result.captures)} capture(s), expected exactly 1"
        )
        assert any(c.peer_port and c.peer_port > 0 for c in result.connections), (
            "recorder's connection log has no entry with a real peer port"
        )
        assert len(result.attempts) == 1, (
            f"proxy saw {len(result.attempts)} CONNECT attempt(s), expected exactly 1"
        )
        assert result.attempts[0].authenticated is True
        assert result.attempts[0].source_port is not None, (
            "the proxy's tunnel source port is the §5.2.1 join key; it must be set "
            "for an authenticated, established tunnel"
        )
        assert result.attempts[0].target.startswith(f"{HARNESS_UPSTREAM_HOST}:"), (
            f"CONNECT target {result.attempts[0].target!r} is not the harness hostname: "
            "the upstream swap is not live, and this tunnel may lead to the real "
            "chatgpt.com"
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

        A loopback-configured upstream would be reached without the proxy's
        name translation and the containment assertion would pass while
        proving the opposite of its claim. The adapter's upstream is the
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
            f"proxy resolve map is missing {target!r}: the CONNECT target has no "
            "loopback translation"
        )


# ── Verdict recording (R5) ───────────────────────────────────────────────
#
# Deliberately the LAST test class in the file: pytest's default collection
# order runs tests in declaration order within a file, so the verdict test
# below runs after every phase test and after the drive baseline. The
# recording itself lives in the session finaliser in
# `tests/harness/conftest.py` (`_record_curl_cffi_slice_verdict_at_session_end`),
# which runs after every test in the process; this class asserts the
# *precondition* the finaliser reads. See the aiohttp slice's identical
# block comment for the reordering-plugin caveat.


@pytest.mark.skipif(_NEEDS_311, reason=_SKIP_REASON)
class TestSliceVerdict:
    """The verdict gate's precondition: every phase actually ran and passed.

    Skipped on Python <3.11 alongside phases 2/2b/3 (the owner's scope
    decision — see the module docstring): the assertion's premise — every
    phase actually ran — is false on those interpreters, and a red verdict
    test would make CI red for a slice that has been honestly *not* proven
    on that Python rather than honestly proven.
    """

    def test_every_phase_actually_ran_and_passed(self) -> None:
        """One tracked outcome per phase, all ``PASSED``.

        The assertion names any missing or non-passing phase so the fix is
        local: a rename means editing
        :data:`harness.conftest._CURL_CFFI_PHASE_TEST_NAMES`; a skip means
        the interpreter's floor was not met and the verdict is correctly not
        recorded.
        """
        from harness.conftest import _CURL_CFFI_PHASE_TEST_NAMES, _curl_phase_outcomes, _PhaseOutcome

        assert len(_curl_phase_outcomes) == len(_CURL_CFFI_PHASE_TEST_NAMES), (
            f"phase outcomes tracked for {len(_curl_phase_outcomes)} test(s), expected "
            f"{len(_CURL_CFFI_PHASE_TEST_NAMES)}: a phase test renamed or removed without "
            "updating the conftest's name set, or a phase never ran"
        )
        for name in sorted(_CURL_CFFI_PHASE_TEST_NAMES):
            assert name in _curl_phase_outcomes, f"phase {name!r} never ran"
            assert _curl_phase_outcomes[name] is _PhaseOutcome.PASSED, (
                f"phase {name!r} outcome is {_curl_phase_outcomes[name].value}: the slice "
                "verdict is correctly not recorded (T-E9's gate will see "
                "not_attempted until every phase passes on this interpreter)"
            )


# ── Registration ──────────────────────────────────────────────────────────


register_containment_transport(CurlCffiContainment.name, CurlCffiContainment)
