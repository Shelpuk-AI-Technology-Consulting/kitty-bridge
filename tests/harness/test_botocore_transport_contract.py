"""The botocore proxy contract — T-G11 / KBR-161 scope addition.

`.system_design/TEST_SUITE.md` §6.2.4 · plan task **T-G11** · scope added to
[KBR-64](https://shelpuk.atlassian.net/browse/KBR-64) by
[KBR-161](https://shelpuk.atlassian.net/browse/KBR-161).

**Why this file exists.** §6.2.4's botocore row carries an **expectation** —
"``Config(proxies=)`` is honoured and takes precedence over the environment.
It is botocore, not boto3, that implements this" — and an unverified
expectation is a defect waiting to be exposed by the next botocore bump.
Measured on botocore 1.34.46 (see
``botocore/endpoint.py:412-413``'s ``if proxies is None: ...`` short-circuit):
the explicit ``Config(proxies=...)`` reaches botocore's ``URLLib3Session``
before the env fallback runs, so the mapping wins. AWS's
[Configuration docs](https://docs.aws.amazon.com/boto3/latest/guide/configuration.html)
state the same contract.

KBR-161 measured the equivalent on curl_cffi and found the design
document's expectation **backwards**: ``proxies= + NO_PROXY=<matching>``
silently defeats the configured mapping, no error, no log line. Closed for
both curl_cffi legs by setting ``CURLOPT_NOPROXY`` at construction. This
file is the botocore twin — a probe that pins botocore's precedence as a
test, not as a comment.

**Layer.** Marked ``l2`` — these are fast contract tests that need real
sockets to reach a real recorder/proxy pair, but they drive through the
:class:`~harness.botocore_containment.BotocoreContainment` drive surface
and assert on the harness's own observability, so the harness (§5.5's
infra) is the boundary under test, not the product's own infra.

**Autouse ambient-env isolation.** Each test strips the eight ambient proxy
variables (``HTTP_PROXY``, ``HTTPS_PROXY``, ``NO_PROXY``, ``ALL_PROXY`` and
their lowercase forms). The contract probes set a *named* subset to isolate
the precedence question, but only with the others absent: a mixed
environment produces a pass or fail that cannot be attributed to one
variable. The ``_isolate_ambient_proxy_env`` fixture makes that isolation
shared with the botocore containment slice.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from harness.botocore_containment import BotocoreContainment
from harness.connect_proxy import (
    HARNESS_UPSTREAM_HOST,
    CertFiles,
    proxy_config,
)
from harness.containment import SealedNetwork
from harness.contract import WireFormat

#: Every ambient proxy environment variable that botocore's urllib3 reads.
#: Sharing the constant with the botocore containment slice means a future
#: addition (``HTTPS_PROXY``, ``no_proxy``, etc.) is one edit, not two.
_AMBIENT_PROXY_ENV_VARS: frozenset[str] = frozenset({
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
})

#: The proxy URL the contract probes point the bridge at. A dead address
#: (loopback port 1) so an ambient proxy that wins precedence fails loudly
#: with a connection refused, not silently answer 200. The harness proxy
#: is the only address that should reach the recorder.
_DEAD_PROXY_URL = "https://127.0.0.1:1"


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def sealed_network(certs: CertFiles) -> AsyncGenerator[SealedNetwork, None]:
    """One started ``SealedNetwork`` for the test, torn down on exit.

    Uses the botocore recorder so the bedrock adapter's request body
    parses against a recorder that speaks Converse. The harness TLS is
    unchanged: the botocore client trusts the harness CA via the drive's
    :meth:`BotocoreContainment._botocore_trusts_test_ca` patch.

    Yields:
        The running harness.
    """
    from harness.botocore_recorder import BedrockRecordingUpstream

    net = SealedNetwork(
        WireFormat.BEDROCK_CONVERSE,
        certs=certs,
        recorder_factory=BedrockRecordingUpstream,
    )
    await net.start()
    try:
        yield net
    finally:
        await net.stop()


@pytest.fixture(autouse=True)
def _isolate_ambient_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ambient proxy environment variable before every test.

    The contract probes that *set* ``NO_PROXY`` / ``HTTPS_PROXY`` /
    ``HTTP_PROXY`` only set the variable under test. The other variables
    are absent so the precedence question isolates to the tested variable:
    a mixed environment produces a pass or fail that cannot be attributed
    to one variable, and §6.2.4's pinning would then be wishful.

    Args:
        monkeypatch: Pytest's monkeypatch fixture.
    """
    for var in sorted(_AMBIENT_PROXY_ENV_VARS):
        monkeypatch.delenv(var, raising=False)


# ── Contract probes ───────────────────────────────────────────────────────


class TestConfigProxiesPrecedence:
    """``Config(proxies=...)`` wins over ambient ``NO_PROXY`` and ``HTTP(S)_PROXY``.

    The three positive cases share the same drive shape; only the ambient
    environment they set differs. The fourth test is the falsification
    control: when the explicit mapping is removed, the request must NOT
    reach the recorder through the configured proxy, so the contract
    tests can fail when the contract is broken.
    """

    async def test_config_proxies_routes_through_the_proxy(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Baseline: with no ambient proxy vars and our ``Config(proxies=...)``.

        The request goes through the harness proxy.
        """
        from kitty.egress import set_egress

        egress = proxy_config(sealed_network.proxy.port)
        set_egress(egress)

        result = await BotocoreContainment().drive_with_egress(
            sealed_network, egress=egress, monkeypatch=monkeypatch
        )

        assert len(result.attempts) == 1, (
            f"expected exactly one CONNECT attempt against the harness proxy, got "
            f"{len(result.attempts)}"
        )
        assert result.attempts[0].authenticated is True
        assert len(result.captures) == 1, (
            f"recorder saw {len(result.captures)} capture(s), expected exactly 1: "
            "the request did not reach the recorder through the proxy"
        )
        assert result.status == 200

    async def test_config_proxies_overrides_matching_no_proxy(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``NO_PROXY`` matching the destination does not defeat ``Config(proxies=...)``.

        Sets ``NO_PROXY`` to the harness hostname (the most dangerous
        variant — it matches the destination botocore would otherwise
        contact), and ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``ALL_PROXY`` to a
        dead address so any ambient bypass would fail loudly with
        ``ConnectionRefusedError``. The request must still arrive via the
        harness proxy: the explicit ``Config(proxies=...)`` wins.
        """
        from kitty.egress import set_egress

        monkeypatch.setenv("NO_PROXY", HARNESS_UPSTREAM_HOST)
        monkeypatch.setenv("no_proxy", HARNESS_UPSTREAM_HOST)
        monkeypatch.setenv("HTTP_PROXY", _DEAD_PROXY_URL)
        monkeypatch.setenv("HTTPS_PROXY", _DEAD_PROXY_URL)
        monkeypatch.setenv("http_proxy", _DEAD_PROXY_URL)
        monkeypatch.setenv("https_proxy", _DEAD_PROXY_URL)
        monkeypatch.setenv("ALL_PROXY", _DEAD_PROXY_URL)
        monkeypatch.setenv("all_proxy", _DEAD_PROXY_URL)

        egress = proxy_config(sealed_network.proxy.port)
        set_egress(egress)

        result = await BotocoreContainment().drive_with_egress(
            sealed_network, egress=egress, monkeypatch=monkeypatch
        )

        assert len(result.attempts) == 1, (
            f"ambient NO_PROXY={HARNESS_UPSTREAM_HOST} defeated Config(proxies=...): "
            f"expected exactly one CONNECT attempt, got {len(result.attempts)}. "
            "If this fails on a botocore bump, the precedence contract has changed."
        )
        assert result.attempts[0].authenticated is True
        assert len(result.captures) == 1
        assert result.status == 200

    async def test_config_proxies_overrides_ambient_https_proxy(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ambient ``HTTPS_PROXY`` set to a dead proxy does not defeat ``Config(proxies=...)``.

        Same shape as ``test_config_proxies_overrides_matching_no_proxy``
        with the precedence question rotated to ``HTTPS_PROXY`` — the most
        common ambient-proxy shape, and the one a CI runner with a corp
        proxy would exercise against us if botocore ever flipped.
        """
        from kitty.egress import set_egress

        monkeypatch.setenv("HTTPS_PROXY", _DEAD_PROXY_URL)
        monkeypatch.setenv("https_proxy", _DEAD_PROXY_URL)
        monkeypatch.setenv("HTTP_PROXY", _DEAD_PROXY_URL)
        monkeypatch.setenv("http_proxy", _DEAD_PROXY_URL)
        # No NO_PROXY — empty so a permissive matcher does not cancel the
        # ambient HTTPS_PROXY.
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)

        egress = proxy_config(sealed_network.proxy.port)
        set_egress(egress)

        result = await BotocoreContainment().drive_with_egress(
            sealed_network, egress=egress, monkeypatch=monkeypatch
        )

        assert len(result.attempts) == 1, (
            f"ambient HTTPS_PROXY defeated Config(proxies=...): expected exactly one "
            f"CONNECT attempt against the harness proxy, got {len(result.attempts)}"
        )
        assert result.attempts[0].authenticated is True
        assert len(result.captures) == 1
        assert result.status == 200

    async def test_falsification_without_config_proxies_the_proxy_sees_nothing(
        self,
        sealed_network: SealedNetwork,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``Config(proxies=...)`` is patched away, the request fails — proving the test can fail.

        Patches ``kitty.providers.bedrock.get_egress`` to return ``None`` so
        the bedrock adapter constructs a boto3 client without
        ``Config(proxies=...)``. With ambient env cleared by the autouse
        fixture, botocore has no proxy at all — the request fails.

        Args:
            sealed_network: The running harness.
            monkeypatch: Pytest's monkeypatch fixture.

        Returns:
            ``None``.
        """
        import kitty.providers.bedrock  # noqa: PLC0415  -- needed before monkeypatch.setattr can find the attribute
        from kitty.egress import set_egress

        # The patch site is the bedrock module's binding — the same
        # fail-by-silence shape T-E2 documents for ``should_bypass`` on
        # ``kitty.bridge.server``: the adapter does ``from kitty.egress
        # import get_egress`` at import time, so a patch on the egress
        # module alone is silently ignored.
        monkeypatch.setattr(kitty.providers.bedrock, "get_egress", lambda: None)
        set_egress(None)

        result = await BotocoreContainment().drive_with_egress(
            sealed_network,
            egress=proxy_config(sealed_network.proxy.port),
            monkeypatch=monkeypatch,
        )

        # The leak IS the evidence the contract broke. With
        # ``Config(proxies=...)`` patched away and the ambient env clear,
        # botocore has no proxy at all — it may still reach the recorder
        # directly (and answer 200) because the bedrock adapter's
        # ``endpoint_url`` names the harness hostname the kernel resolves
        # through the autouse resolver patch, but the request did NOT
        # traverse the configured proxy. Two observations prove the
        # contract is broken:
        #
        # 1. The proxy received **zero** CONNECT attempts — the request
        #    did not go through the proxy the test was supposed to be
        #    routing through.
        # 2. Every recorder peer port is unattributable to any tunnel
        #    — §5.2.1's join detects that a recorder connection has no
        #    matching proxy tunnel. With no tunnels at all, every peer
        #    port is unattributable.
        #
        # Together, those are the signal: the bridge's response is
        # secondary; what matters is whether the configured mapping was
        # actually used. The bridge can answer 200 either way; the
        # question is whether it answered **through the proxy**.
        from harness.connect_proxy import unattributable_peer_ports

        assert len(result.attempts) == 0, (
            f"proxy saw {len(result.attempts)} CONNECT attempt(s); expected zero: "
            "with Config(proxies=...) patched away and ambient env clear, "
            "botocore must not route through the proxy"
        )
        # The bridge may still reach the recorder directly (the loopback
        # endpoint is reachable without a proxy), in which case its
        # connections are real but unattributable. Either the recorder
        # is empty (the request never reached it) or every peer port is
        # unattributable to any tunnel — both are the same outcome from
        # the contract's perspective.
        peer_ports = [c.peer_port for c in sealed_network.recorder.connections]
        if peer_ports:
            unattributable = unattributable_peer_ports(peer_ports, sealed_network.proxy.attempts)
            assert unattributable == peer_ports, (
                f"recorder peer ports {peer_ports} have a tunnel entry but the "
                "proxy saw no CONNECT attempts: a tunnel cannot explain a "
                "connection when none was established — the contract is broken"
            )
