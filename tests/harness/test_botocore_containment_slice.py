"""The complete, falsified botocore containment slice — T-E4 ([KBR-64]).

`.system_design/TEST_SUITE.md` §5.2.1, §5.2.2, §5.3, §5.4, §5.5 · plan task **T-E4**
([KBR-64](https://shelpuk.atlassian.net/browse/KBR-64)) ·
`.requirements/20260916T204434Z_kbr64_t_e4_botocore_containment_slice/REQUIREMENTS.md`.

T-E1 ([KBR-61](https://shelpuk.atlassian.net/browse/KBR-61)) shipped the
sealed-network harness core; T-E2
([KBR-62](https://shelpuk.atlassian.net/browse/KBR-62)) shipped the complete,
falsified aiohttp containment slice and the verdict-gate pattern this file
repeats for the botocore leg. The bedrock adapter reaches its upstream through
a boto3 client whose HTTP layer is botocore's, so every proxy and resolver
question the aiohttp slice answers has a botocore-specific answer here.

**TLS at the recorder and in the boto3 client's trust store.** The recorder
speaks TLS at the harness hostname
(see :attr:`SealedNetwork.upstream_base_url`); botocore's HTTPS client
verifies that hop against the harness CA via the ``AWS_CA_BUNDLE``
environment channel, which the drives set for the duration
(:meth:`BotocoreContainment._botocore_trusts_test_ca`). Unlike
``test_aiohttp_containment_slice.py``, **no Python ≥3.11 skip applies** on
the botocore phases: bpo-44011 is asyncio-specific, and botocore's HTTP
stack is synchronous urllib3 with TLS-in-TLS handled inside
``ssl.SSLContext.wrap_socket()`` on the worker thread
(``src/kitty/providers/bedrock.py`` calls the boto3 client from
``loop.run_in_executor``). The TLS-in-TLS path that 3.11 fixed in aiohttp's
asyncio stack does not appear on this leg.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
``test_aiohttp_containment_slice.py`` and ``test_botocore.py``. The reason is
§8.2's and only §8.2's: ``l3`` is in ``PENDING_ACTIVATION_LAYERS``, so an
``l3`` marker today would leave this slice's correctness checked by no job at
all. T-K6 inherits the relocation for all four sibling slices.
"""

from __future__ import annotations

import ssl
from collections.abc import AsyncGenerator

import pytest

from harness.botocore_containment import BotocoreContainment
from harness.connect_proxy import (
    AMBIENT_PROXY_ENV_VARS,
    HARNESS_UPSTREAM_HOST,
    PROXY_PASSWORD,
    CertFiles,
    proxy_config,
    unattributable_peer_ports,
)
from harness.containment import (
    Phase1Result,
    SealedNetwork,
    reset_for_test,
)
from harness.recorder import RecordingUpstream
from kitty.egress import EgressConfig

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def sealed_network(certs: CertFiles) -> AsyncGenerator[SealedNetwork, None]:
    """One started ``SealedNetwork`` for the test, torn down on exit.

    Each phase gets a **fresh** sealed network. Phase 1's direct connection
    leaves a peer-port row in ``recorder.connections``; sharing one network
    would let phase 3's "non-empty unattributable-port" assertion pass on
    phase 1's leak rather than on the injected bypass — the vacuous pass
    §5.2.2's falsification control exists to prevent.

    The recorder is the **botocore** one
    (:class:`~harness.botocore_recorder.BedrockRecordingUpstream`): the
    bridge's botocore leg speaks Bedrock Converse upstream, and the primary
    recorder answers only anthropic and chat-completions shapes. A request
    on the botocore leg answered in an anthropic shape would parse as
    garbage to the bedrock adapter and the drive would time out — the
    wrong-format failure ``test_botocore.py``'s own recorder factory exists
    to prevent.

    The factory follows the curl_cffi slice's shape (main's T-E3):
    ``SealedNetwork`` passes the harness's ready ``SSLContext`` to the
    factory at construction time, and the factory returns a recorder whose
    ``start()`` takes no arguments. ``BedrockRecordingUpstream`` inherits
    ``start(ssl_context=...)`` from ``RecordingUpstream`` — the
    ``BotocoreContainment`` module ships a small
    :class:`_TlsBedrockRecordingUpstream` wrapper that captures the
    context at construction and applies it at start, so the harness's
    factory contract and the bedrock recorder's lifecycle signature stay
    in agreement.

    Yields:
        The running harness.
    """
    from harness.botocore_containment import (
        _TlsBedrockRecordingUpstream as _TlsBotocoreRecorder,
    )
    from harness.contract import WireFormat

    def _factory(ctx: ssl.SSLContext) -> RecordingUpstream:
        """Build the botocore recorder with the harness CA bound at construction.

        Args:
            ctx: The harness's TLS context — leaf cert carries
                ``HARNESS_UPSTREAM_HOST`` in its SAN.

        Returns:
            A fully-constructed recorder whose ``start()`` applies the
            captured context.
        """
        return _TlsBotocoreRecorder(
            default_format=WireFormat.BEDROCK_CONVERSE,
            ssl_context=ctx,
        )

    net = SealedNetwork(
        WireFormat.BEDROCK_CONVERSE,
        certs=certs,
        recorder_factory=_factory,
    )
    await net.start()
    try:
        yield net
    finally:
        await net.stop()


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    """Reset the capability-report singleton before every test in this module.

    The verdict test at the bottom records ``PROVEN`` for ``botocore``;
    every earlier test must see an untouched report so its phase assertions
    do not depend on collection order.
    """
    reset_for_test()


@pytest.fixture(autouse=True)
def _isolate_ambient_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ambient proxy environment variable before every test.

    botocore's urllib3 reads ``HTTP_PROXY``, ``HTTPS_PROXY``, ``NO_PROXY``,
    ``ALL_PROXY`` and the three lowercase forms. ``kitty.egress`` documents
    the same (§5.5: "aiohttp ignores them unless ``trust_env=True``, while
    curl_cffi and botocore honour them"). A developer with ``HTTPS_PROXY=``
    in their shell would see phase 1 (egress off) attempt to tunnel through
    that env proxy — failing the "recorder saw the connection, proxy saw
    nothing" assertion for a reason that has nothing to do with
    containment. The phase-3 falsification has the mirror problem: patching
    ``get_egress`` to ``None`` does not stop botocore from honouring the
    ambient env, so the "bypass" would still be tunnelled (via the env
    proxy) rather than direct.

    The contract tests in ``test_botocore_transport_contract.py`` are the
    inverse — they *set* a subset of these to measure botocore's documented
    precedence — but this slice's containment phase tests need the
    environment clear.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
    """
    for var in sorted(AMBIENT_PROXY_ENV_VARS):
        monkeypatch.delenv(var, raising=False)


def _egress_for(net: SealedNetwork, password: str = PROXY_PASSWORD) -> EgressConfig:
    """Return the harness's egress configuration for ``net``'s proxy.

    Args:
        net: The sealed network the proxy belongs to.
        password: The proxy password; pass a wrong one for the 407 case.

    Returns:
        The configuration the bridge takes as its ``egress=`` argument.
    """
    return proxy_config(net.proxy.port, password=password)


def _drive(net: SealedNetwork) -> BotocoreContainment:
    """Build a fresh containment transport for one drive.

    ``BotocoreContainment`` is stateless, so a fresh instance per call
    is bookkeeping rather than necessity — but it keeps the drive's lifetime
    visible and mirrors the per-drive contract the protocol documents.

    Args:
        net: The sealed network the drive targets (unused at construction;
            named so the call site reads as one sentence).

    Returns:
        A fresh transport.
    """
    return BotocoreContainment()


# ── Phase 1 — positive control (§5.2.2 row 1) ─────────────────────────────


class TestPhase1PositiveControl:
    """Egress off: the botocore leg reaches the recorder directly and the proxy sees nothing.

    The control for everything below: without it, "the upstream received
    nothing" is satisfied just as well by a destination that is unreachable
    for an unrelated reason (§5.3 trap 2).
    """

    async def test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing(
        self,
        sealed_network: SealedNetwork,
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


class TestPhase2ContainmentProxyDown:
    """Proxy down ⇒ upstream accepts zero connections, request fails, no direct fallback.

    The negative assertion §5.1 gap 2 and §5.2.2 row 2 specify. Bracketed by
    phase 1 (a reachable destination) per §5.3 trap 2: an unresolvable name
    would otherwise satisfy the negative test for the wrong reason.

    No Python ≥3.11 skip applies — see the module docstring on botocore's
    synchronous urllib3 and the bpo-44011 floor's asyncio-specificity.
    """

    async def test_proxy_down_leaves_the_recorder_with_zero_connections(
        self,
        sealed_network: SealedNetwork,
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


class TestPhase2bContainmentHealthy:
    """Proxy up ⇒ every upstream connection joins a tunnel on the recorded source port.

    §5.2.1's join, exercised against a real proxied run. Requests-per-tunnel
    counts are deliberately not asserted — connection reuse may put many
    requests on one tunnel and a 407 puts none on any — so the assertion is
    at the connection level, via ``unattributable_peer_ports``.

    No Python ≥3.11 skip applies — see the module docstring.
    """

    async def test_proxy_up_every_peer_port_joins_a_tunnel(
        self,
        sealed_network: SealedNetwork,
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """407 ⇒ the recorder holds nothing; the failed tunnel's attempt carries no port.

        A fresh botocore client is constructed per drive because the bedrock
        adapter builds one per request (KBR-190) — the wrong-password
        ``EgressConfig`` reaches a client that has never talked to the
        proxy.
        """
        result = await _drive(sealed_network).drive_with_egress(
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


class TestPhase3Falsification:
    """An injected product bypass makes the harness fail — the §1.4 rule for containment.

    The patch site is ``kitty.providers.bedrock``, not ``kitty.egress``: the
    bedrock adapter does ``from kitty.egress import get_egress`` at import
    time, so ``_get_boto3_client`` resolves its own module's name and a patch
    on the egress module is silently ignored. This is the exact
    fail-by-silence the falsification exists to prevent, and the same shape
    the aiohttp twin documents for ``should_bypass`` on ``kitty.bridge.server``.

    No Python ≥3.11 skip applies — see the module docstring.
    """

    async def test_injected_bypass_makes_the_harness_report_it(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patched ``get_egress`` ⇒ the recorder sees a connection the proxy cannot explain."""
        import kitty.providers.bedrock  # noqa: PLC0415  -- needed before monkeypatch.setattr can find the attribute
        monkeypatch.setattr(kitty.providers.bedrock, "get_egress", lambda: None)

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


# ── Drive with egress (R1) — the green path the phases rest on ────────────


class TestDriveWithEgress:
    """``BotocoreContainment.drive_with_egress`` — the phase-2/2b/3 entry point.

    Both tests run on every supported Python: the green path exercises
    botocore's synchronous urllib3 through the harness CONNECT proxy over
    TLS, and the static hostname check does not drive the bridge at all.
    """

    async def test_drive_with_egress_one_request_reaches_the_recorder_through_the_proxy(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Green path: egress enabled, proxy up, one request lands on the recorder."""
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

        The bedrock adapter reaches the harness via
        ``provider_config["endpoint_url"]``, and ``should_bypass`` — which
        the three custom transports never consult, §5.5 — still decides at
        the *harness* level that a loopback base URL would not tunnel. The
        ``endpoint_url`` is spelled with the non-loopback harness hostname so
        a future code path that does consult ``should_bypass`` still sends
        the request through the proxy.
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


# ── §5.5 no-bypass property — botocore applies the proxy unconditionally ──


class TestNoBypassProperty:
    """§5.5: the four custom transports have no ``should_bypass`` consultation.

    The bedrock adapter applies ``Config(proxies=...)`` unconditionally and
    never imports ``kitty.egress.should_bypass``. A loopback or private
    destination — one aiohttp's bypass would skip — is **tunnelled** on
    botocore. This class exercises that property against a loopback base URL
    and asserts the proxy still received the CONNECT attempt.

    The argument is observation, not assertion over ``should_bypass``: there
    is nothing to call, only a proxy attempt that exists when a loopback
    destination would normally bypass. The T-E8 documentation task and the
    L1 property ``should_bypass`` never matches a public hostname will both
    record the asymmetry.
    """

    async def test_a_loopback_endpoint_is_tunnelled_anyway(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Drive to a loopback URL with egress on; the proxy still sees the CONNECT.

        The ``endpoint_url`` is rewritten for this drive to the recorder's
        loopback port (the same one the harness proxy resolves the harness
        hostname to). ``BotocoreAdapter`` does not consult
        ``should_bypass``; the URL is loopback, but ``Config(proxies=)`` is
        applied regardless, the CONNECT reaches the proxy, and the recorder
        answers through the tunnel.

        Args:
            sealed_network: The running harness.
            monkeypatch: Pytest's monkeypatch fixture.
        """
        from kitty.egress import set_egress
        from kitty.providers.bedrock import BedrockAdapter

        # Replace the harness-based URL with a loopback URL. The bridge
        # still routes through the proxy because the bedrock adapter's
        # egress mapping is unconditional. The proxy's resolve map does
        # not need to translate this loopback host — it tunnels the
        # connection as-is to ``127.0.0.1:{port}`` and the kernel does the
        # rest.
        loopback_url = f"https://127.0.0.1:{sealed_network.upstream_port}"
        set_egress(_egress_for(sealed_network))

        # No socket-level resolver override here: ``127.0.0.1`` resolves
        # naturally, and ``monkeypatched_aiohttp_resolver`` would override
        # every lookup of that name — including the proxy's own loopback
        # address — with the upstream port, misrouting the proxied hop to
        # the recorder instead of the proxy.
        with (
            BotocoreContainment()._botocore_trusts_test_ca(monkeypatch, sealed_network.ca_path),
        ):
            from kitty.bridge.server import BridgeServer

            server = BridgeServer(
                None,  # type: ignore[arg-type]
                BedrockAdapter(),
                resolved_key="harness-access-key:harness-secret-key",
                model="harness-model",
                provider_config={
                    "endpoint_url": loopback_url,
                    "region": "us-east-1",
                },
            )
            result = await BotocoreContainment()._drive(server, sealed_network)  # noqa: SLF001

        assert len(result.attempts) == 1, (
            f"expected exactly one CONNECT attempt against the loopback URL, got "
            f"{len(result.attempts)}: §5.5 says botocore applies the proxy "
            "unconditionally — a loopback destination must still tunnel"
        )
        assert result.attempts[0].authenticated is True
        assert result.attempts[0].source_port is not None
        assert result.status == 200, (
            f"bridge answered {result.status}: the loopback drive must reach the "
            "recorder through the proxy"
        )


# ── Verdict recording (R5) ───────────────────────────────────────────────
#
# Deliberately the LAST test class in the file: pytest's default collection
# order runs tests in declaration order within a file, so the verdict test
# below runs after every phase test and after the drive baseline. That
# ordering is what makes `TestSliceVerdict`'s assertion (one tracked outcome
# per phase, all PASSED) hold — a plugin that reorders tests would reorder
# this class ahead of the phase classes and the assertion would fail with
# "phase outcomes tracked for 0 test(s)". Keep this class last when adding
# new tests, and note the dependency here if a reordering plugin is ever
# adopted.


class TestSliceVerdict:
    """The verdict gate's precondition: every phase actually ran and passed.

    No Python ≥3.11 skip — botocore's synchronous urllib3 has no asyncio
    TLS-in-TLS floor (see the module docstring), so every phase runs on every
    supported Python. Skipping the verdict test on a hypothetical floor would
    leave the slice's verdict unrecorded where it should be recorded — the
    inverse of the aiohttp slice's situation.

    The recording itself lives in the session finaliser in
    :mod:`tests.harness.conftest`, which runs **after** every test in this
    process — this class asserts the *precondition* the finaliser reads, so
    a phase rename that drifts from the conftest's name set fails loudly
    here rather than silently producing an unrecorded or falsely-`proven`
    verdict.

    **Ordering.** This is the last test class in the file so pytest's
    default collection order runs it after every phase test; see the module
    comment above for what to do if a reordering plugin is ever adopted.
    """

    def test_every_phase_actually_ran_and_passed(self) -> None:
        """One tracked outcome per phase, all ``PASSED``.

        The assertion names any missing or non-passing phase so the fix is
        local: a rename means editing
        :data:`tests.harness.conftest._BOTOCORE_PHASE_TEST_NAMES`; a skip
        means the interpreter's TLS-in-TLS floor was not met and the verdict
        is correctly not recorded.
        """
        from harness import conftest as harness_conftest

        names = harness_conftest._BOTOCORE_PHASE_TEST_NAMES
        outcomes = harness_conftest._botocore_phase_outcomes
        assert len(outcomes) == len(names), (
            f"phase outcomes tracked for {len(outcomes)} test(s), expected "
            f"{len(names)}: a phase test renamed or removed without "
            "updating the conftest's name set, or a phase never ran"
        )
        for name in sorted(names):
            assert name in outcomes, f"phase {name!r} never ran"
            assert outcomes[name] is harness_conftest._PhaseOutcome.PASSED, (
                f"phase {name!r} outcome is {outcomes[name].value}: the slice "
                "verdict is correctly not recorded (T-E9's gate will see "
                "not_attempted until every phase passes on this interpreter)"
            )
