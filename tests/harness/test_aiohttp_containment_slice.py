"""The complete, falsified aiohttp containment slice — T-E2 ([KBR-62]).

`.system_design/TEST_SUITE.md` §5.2.1, §5.2.2, §5.3, §5.4 · plan task **T-E2**
([KBR-62](https://shelpuk.atlassian.net/browse/KBR-62)) ·
`.requirements/20260915T135234Z_kbr62_t_e2_aiohttp_containment_slice/REQUIREMENTS.md`.

T-E1 ([KBR-61](https://shelpuk.atlassian.net/browse/KBR-61)) shipped the
sealed-network harness core — :class:`SealedNetwork`, the monkeypatched
aiohttp resolver, the per-transport capability report, and the
:class:`BridgeAiohttpContainment` whose :meth:`~.drive_phase_1` demonstrates
the §5.2.2 **phase 1** positive control for the bridge's own aiohttp serving
path. This module exercises the **remaining three phases** of §5.2.2 against
that same transport — proxy down ⇒ zero upstream connections, proxy up ⇒
every upstream connection attributable to a recorded tunnel source port,
and an injected product bypass that must make the harness fail — and records
the slice's ``proven`` verdict into the capability report.

**TLS at the recorder.** The KBR-61 design assumed the bridge's aiohttp
client CONNECT-tunnels for a plain-HTTP upstream; it does not — aiohttp only
CONNECTs for ``https://`` targets and sends ``http://`` targets in absolute
form, which the harness's CONNECT-only proxy answers with 405. The recorder
therefore speaks TLS (see :attr:`SealedNetwork.upstream_base_url`), the
bridge's outbound URL becomes ``https://upstream.kitty-test.invalid:{port}``,
and every proxied phase carries TLS-in-TLS. The ``aiohttp_trusts_test_ca``
fixture puts the harness CA in scope on every phase; phases 2/2b/3 skip below
Python 3.11 where aiohttp cannot do TLS-in-TLS over stdlib asyncio, per the
seam ``test_egress_https_proxy.py`` already established.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
``test_containment.py`` and ``test_vertical_slice.py``. The reason is §8.2's
and only §8.2's: ``l3`` is in ``PENDING_ACTIVATION_LAYERS``, so an ``l3`` marker
today would leave this slice's correctness checked by no job at all. §8.2
names ``test_containment.py`` as the T-E1 bullet; this file extends the same
convention and T-K6 inherits the relocation for both.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator

import pytest

from harness.conftest import _PHASE_TEST_NAMES, _phase_outcomes, _PhaseOutcome
from harness.connect_proxy import (
    HARNESS_UPSTREAM_HOST,
    PROXY_PASSWORD,
    CertFiles,
    proxy_config,
    unattributable_peer_ports,
)
from harness.containment import (
    BridgeAiohttpContainment,
    Phase1Result,
    SealedNetwork,
    reset_for_test,
)
from harness.contract import WireFormat
from kitty.egress import EgressConfig

#: aiohttp's TLS-in-TLS over stdlib asyncio — the shape every proxied phase
#: exercises — landed in Python 3.11 (bpo-44011). ``test_egress_https_proxy.py``
#: skips on the same seam; this slice inherits the floor and the reason.
_AIOHTTP_NEEDS_311 = sys.version_info < (3, 11)
_AIOHTTP_SKIP_REASON = "aiohttp requires Python 3.11 for TLS-in-TLS over stdlib asyncio (bpo-44011)"

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def sealed_network(certs: CertFiles) -> AsyncGenerator[SealedNetwork, None]:
    """One started ``SealedNetwork`` for the test, torn down on exit.

    Each phase gets a **fresh** sealed network. Phase 1's direct connection
    leaves a peer-port row in ``recorder.connections``; sharing one network
    would let phase 3's "non-empty unattributable-port" assertion pass on
    phase 1's leak rather than on the injected bypass — the vacuous pass
    §5.2.2's falsification control exists to prevent.

    Yields:
        The running harness.
    """
    net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs)
    await net.start()
    try:
        yield net
    finally:
        await net.stop()


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    """Reset the capability-report singleton before every test in this module.

    The verdict test at the bottom records ``PROVEN`` for ``bridge_aiohttp``;
    every earlier test must see an untouched report so its phase assertions
    do not depend on collection order.
    """
    reset_for_test()


def _egress_for(net: SealedNetwork, password: str = PROXY_PASSWORD) -> EgressConfig:
    """Return the harness's egress configuration for ``net``'s proxy.

    Args:
        net: The sealed network the proxy belongs to.
        password: The proxy password; pass a wrong one for the 407 case.

    Returns:
        The configuration the bridge takes as its ``egress=`` argument.
    """
    return proxy_config(net.proxy.port, password=password)


def _drive(net: SealedNetwork) -> BridgeAiohttpContainment:
    """Build a fresh containment transport for one drive.

    ``BridgeAiohttpContainment`` is stateless, so a fresh instance per call
    is bookkeeping rather than necessity — but it keeps the drive's lifetime
    visible and mirrors the per-drive contract the protocol documents.

    Args:
        net: The sealed network the drive targets (unused at construction;
            named so the call site reads as one sentence).

    Returns:
        A fresh transport.
    """
    return BridgeAiohttpContainment()


# ── Phase 1 — positive control (§5.2.2 row 1) ─────────────────────────────


class TestPhase1PositiveControl:
    """Egress off: the bridge reaches the recorder directly and the proxy sees nothing.

    The control for everything below: without it, "the upstream received
    nothing" is satisfied just as well by a destination that is unreachable
    for an unrelated reason (§5.3 trap 2). T-E1's copy of the same assertion
    lives in ``test_containment.py``; this one re-asserts on the now-TLS
    recorder so the slice is self-contained, and T-K6 relocates both.
    """

    async def test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Green path on the TLS recorder: 200, one capture, real peer port, zero attempts."""
        result = await _drive(sealed_network).drive_phase_1(sealed_network, monkeypatch=monkeypatch)

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


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
class TestPhase2ContainmentProxyDown:
    """Proxy down ⇒ upstream accepts zero connections, request fails, no direct fallback.

    The negative assertion §5.1 gap 2 and §5.2.2 row 2 specify. Bracketed by
    phase 1 (a reachable destination) per §5.3 trap 2: an unresolvable name
    would otherwise satisfy the negative test for the wrong reason.
    """

    async def test_proxy_down_leaves_the_recorder_with_zero_connections(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stop the proxy, drive once, assert the recorder holds nothing and the drive failed."""
        await sealed_network.proxy.stop()

        result = await _drive(sealed_network).drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network), monkeypatch=monkeypatch
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


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
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
            result = await _drive(sealed_network).drive_with_egress(
                sealed_network, egress=egress, monkeypatch=monkeypatch
            )
            assert result.status == 200, f"drive answered {result.status}: {result.text!r}"

        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        assert peer_ports, "recorder accepted no connections; the drive never reached it"
        assert unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts) == [], (
            f"recorder peer ports {peer_ports} not covered by tunnel source ports "
            f"{[a.source_port for a in sealed_network.proxy.attempts]}: §5.2.1's join broken"
        )

    async def test_failed_tunnel_contributes_no_upstream_connection(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """407 ⇒ the recorder holds nothing; the failed tunnel's attempt carries no port.

        A fresh bridge is constructed per drive because ``BridgeServer``
        caches ``_proxy_session`` — the wrong-password ``EgressConfig`` must
        reach a bridge that has never talked to the proxy.
        """
        result = await _drive(sealed_network).drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network, password="wrong"), monkeypatch=monkeypatch
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


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
class TestPhase3Falsification:
    """An injected product bypass makes the harness fail — the §1.4 rule for containment.

    The patch site is ``kitty.bridge.server``, not ``kitty.egress``: the
    bridge does ``from kitty.egress import should_bypass`` at import time, so
    ``_session_for`` resolves its own module's name and a patch on the egress
    module is silently ignored. This is the exact fail-by-silence the
    falsification exists to prevent.
    """

    async def test_injected_bypass_makes_the_harness_report_it(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patched ``should_bypass`` ⇒ the recorder sees a connection the proxy cannot explain.

        The explicit ``import kitty.bridge.server`` below makes the bridge
        module — and its module-level ``from kitty.egress import should_bypass``
        at line 56 — fully resolved before the patch lands. ``_session_for``
        calls ``should_bypass`` against its own module's binding, so the
        patch must land on ``kitty.bridge.server`` (not ``kitty.egress``) or
        the bridge silently keeps routing through the proxy and the bypass
        is invisible — the exact fail-by-silence the falsification exists
        to prevent.
        """
        import kitty.bridge.server  # noqa: PLC0415  -- needed before monkeypatch.setattr can find the attribute
        monkeypatch.setattr(kitty.bridge.server, "should_bypass", lambda url: True)

        result = await _drive(sealed_network).drive_with_egress(
            sealed_network, egress=_egress_for(sealed_network), monkeypatch=monkeypatch
        )

        # The bypass worked: the bridge reached the recorder directly (the
        # resolver mapping was still in scope) and answered 200.
        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert result.attempts == [], (
            f"proxy saw {len(result.attempts)} attempt(s) under the bypass — the bridge "
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


# ── Verdict recording (R5) ───────────────────────────────────────────────


@pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
class TestSliceVerdict:
    """The verdict gate's precondition: every phase actually ran and passed.

    Skipped on Python <3.11 alongside phases 2/2b/3 (TLS-in-TLS, bpo-44011):
    the assertion's premise — every phase actually ran — is false on those
    interpreters, and a red verdict test would make CI red for a slice
    that has been honestly *not* proven on that Python rather than
    honestly proven.

    The recording itself lives in the session finaliser in
    :mod:`tests.harness.conftest`, which runs **after** every test in this
    process — this class asserts the *precondition* the finaliser reads, so
    a phase rename that drifts from the conftest's name set fails loudly
    here rather than silently producing an unrecorded or falsely-`proven`
    verdict.
    """

    def test_every_phase_actually_ran_and_passed(self) -> None:
        """One tracked outcome per phase, all ``PASSED``.

        The assertion names any missing or non-passing phase so the fix is
        local: a rename means editing
        :data:`tests.harness.conftest._PHASE_TEST_NAMES`; a skip means the
        interpreter's TLS-in-TLS floor was not met and the verdict is
        correctly not recorded.
        """
        assert len(_phase_outcomes) == len(_PHASE_TEST_NAMES), (
            f"phase outcomes tracked for {len(_phase_outcomes)} test(s), expected "
            f"{len(_PHASE_TEST_NAMES)}: a phase test renamed or removed without "
            "updating the conftest's name set, or a phase never ran"
        )
        for name in sorted(_PHASE_TEST_NAMES):
            assert name in _phase_outcomes, f"phase {name!r} never ran"
            assert _phase_outcomes[name] is _PhaseOutcome.PASSED, (
                f"phase {name!r} outcome is {_phase_outcomes[name].value}: the slice "
                "verdict is correctly not recorded (T-E9's gate will see "
                "not_attempted until every phase passes on this interpreter)"
            )


# ── Drive with egress (R1) — the green path the phases rest on ────────────


class TestDriveWithEgress:
    """``BridgeAiohttpContainment.drive_with_egress`` — the phase-2/2b/3 entry point.

    The drive test below is skipped on Python <3.11 (TLS-in-TLS, bpo-44011):
    the green-path exercise sends the bridge's aiohttp session through the
    harness CONNECT proxy over TLS, and aiohttp's TLS-in-TLS over stdlib
    asyncio does not work below 3.11. The static hostname/resolve-map check
    in this class does not drive the bridge, so it still runs on every
    supported Python.
    """

    @pytest.mark.skipif(_AIOHTTP_NEEDS_311, reason=_AIOHTTP_SKIP_REASON)
    async def test_drive_with_egress_one_request_reaches_the_recorder_through_the_proxy(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Green path: egress enabled, proxy up, one request lands on the recorder.

        This is the baseline the phase-2/2b/3 assertions rest on. With the
        proxy up and authenticated, the bridge constructs its proxied session,
        the proxy records one authenticated CONNECT with a non-``None`` source
        port, and the recorder's connection log shows the upstream connection
        on the matching source port — the §5.2.1 join key, exercised once.

        Skipped on Python <3.11 (TLS-in-TLS, bpo-44011): the drive opens a
        TLS session to the harness CONNECT proxy and then wraps the recorder's
        TLS hop inside it, and aiohttp's stdlib-asyncio TLS-in-TLS only landed
        in 3.11.
        """
        egress = _egress_for(sealed_network)
        result = await _drive(sealed_network).drive_with_egress(
            sealed_network, egress=egress, monkeypatch=monkeypatch
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
        # §5.2.1's join holds: the recorder's peer port equals the proxy's
        # tunnel source port — same kernel connection seen from both ends.
        peer_ports = {c.peer_port for c in result.connections if c.peer_port}
        assert result.attempts[0].source_port in peer_ports, (
            f"recorder's peer ports {peer_ports} do not include the proxy's tunnel "
            f"source port {result.attempts[0].source_port} — §5.2.1's join broken"
        )

    async def test_drive_with_egress_targets_the_harness_hostname_not_loopback(
        self,
        sealed_network: SealedNetwork,
    ) -> None:
        """§5.3 trap 1: the upstream is addressed by the harness hostname, never loopback.

        ``should_bypass`` returns ``True`` for loopback and ``localhost``
        names, so a loopback-configured base URL would send the bridge direct
        and the containment assertion would pass while proving the opposite
        of its claim. The bridge's provider config is the sealed network's
        ``https://upstream.kitty-test.invalid:{port}`` — the non-loopback
        name — and the proxy's resolve map is what maps that name to the
        recorder's loopback port.
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
