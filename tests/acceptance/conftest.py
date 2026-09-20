"""Shared fixtures for the L4 acceptance layer (T-J1, KBR-107).

The pytest-bdd generated tests are **synchronous** functions — pytest-bdd does
not natively run coroutine step functions — but the L3 harness
(:class:`~harness.bridge.BridgeFixture`) is async and its aiohttp session is
bound to the event loop that created it. To drive the L3 async surface from
sync steps without a per-step event loop (which would orphan the session), the
``bridge_session`` fixture here owns a single loop for the duration of one
scenario and yields the ``(loop, bridge)`` tuple to every step.

Steps consume the tuple and call ``loop.run_until_complete(coro)`` for each L3
op — most steps use ``bridge.post``. On teardown the fixture stops the bridge,
mirrors ``BridgeFixture.__aexit__``'s clean-path teardown-clean assertion, and
closes the loop. The outer ``try``/``finally`` closes the loop on **every**
exit path, including a raise from ``BridgeFixture.start()`` itself — reachable
per ``bridge.py:707-732`` (a profile-validation or bind failure), and the path
the round-2 review surfaced.

The conftest imports the L3 harness surface directly — the only behaviour the
acceptance layer owns is *how steps find L3*, never *how L3 behaves*. The
clean-path teardown lives in :func:`tests.acceptance.teardown.teardown_clean_path`
so the §1.4 falsification can wrap its call in ``pytest.raises``; fixture
teardown errors surface as test errors, not as exceptions a test can wrap, so
direct ``pytest.raises`` on the fixture would be impossible. The residual gap
is stated on the record: a future regression that *deletes the conftest's
``else`` call to the helper entirely* (rather than edits the helper itself)
would not be caught here — the helper is falsified directly, end-to-end. If
end-to-end conftest coverage becomes important, pytester's nested-run pattern
is the route.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from harness.bridge import BridgeFixture, WireFormat, transport
from harness.connect_proxy import CertFiles, proxy_config
from harness.containment import SealedNetwork
from harness.containment import WireFormat as ContainmentFormat
from teardown import teardown_clean_path

from kitty.egress import EgressConfig

#: The profile name the EG-3 scenario uses; asserted in the stderr check.
EG3_PROFILE_NAME = "bedrock-sso"

#: bpo-44011: aiohttp's TLS-in-TLS over stdlib asyncio landed in Python 3.11.
#: The EG-1 / EG-2 scenarios drive the proxied (TLS-in-TLS) shape, so the
#: same guard the L3 slice applies (``_AIOHTTP_NEEDS_311`` in
#: tests/harness/test_aiohttp_containment_slice.py) is mirrored here — the
#: alternative on 3.10 is a CI leg that fails on a known dependency shape.
_AIOHTTP_NEEDS_311 = sys.version_info < (3, 11)


def pytest_bdd_apply_tag(tag: str, function: object) -> object:
    """Map the ``@needs_python_311`` Gherkin tag to a version skipif.

    pytest-bdd calls this hook once per tagged scenario during collection,
    so the skip is a *collection-time* decision — no drive runs on 3.10
    and fails on the known bpo-44011 shape.

    Args:
        tag: The Gherkin tag above the scenario.
        function: The generated test function the tag applies to.

    Returns:
        The (possibly marker-decorated) test function.
    """
    if tag == "needs_python_311":
        return pytest.mark.skipif(
            _AIOHTTP_NEEDS_311,
            reason="aiohttp requires Python 3.11 for TLS-in-TLS over stdlib asyncio (bpo-44011)",
        )(function)  # type: ignore[operator]
    # Any other tag falls through unchanged — pytest-bdd's default
    # treatment applies (markers not recognised by this hook are still
    # recorded on the test item, so ``-m @tag`` selection works). Adding
    # a new branch here is the right path when a future scenario needs
    # a tag-to-marker mapping.
    return function


@pytest.fixture
def bridge_session() -> Iterator[tuple[asyncio.AbstractEventLoop, BridgeFixture]]:
    """Yield ``(loop, started BridgeFixture)`` for one acceptance scenario.

    Yields:
        A tuple of the asyncio loop and the :class:`BridgeFixture` the steps
        drive. The bridge is started on ``loop`` before the yield; on
        teardown — mirroring :meth:`BridgeFixture.__aexit__` exactly — the
        bridge is stopped, the transport's teardown is asserted only on the
        clean path, and the loop is closed last. The outer ``finally`` closes
        the loop on every exit, including ``BridgeFixture.start()`` raising.

    Notes:
        Function-scoped on purpose: pytest-bdd generates one test function per
        scenario, and a started ``BridgeFixture`` owns a real port the next
        scenario must not inherit.
    """
    loop = asyncio.new_event_loop()
    try:
        bridge = BridgeFixture(transport("aiohttp", WireFormat.ANTHROPIC_MESSAGES))
        loop.run_until_complete(bridge.start())
        try:
            yield loop, bridge
        except BaseException:
            # Test body raised. ``__aexit__`` skips the teardown-clean
            # assertion in this case, so the inner ``try`` mirrors it by
            # skipping too. Surface the test's own exception.
            loop.run_until_complete(bridge.stop())
            raise
        else:
            # Clean path. The shared ``teardown_clean_path`` is the single
            # implementation; the falsification test calls the same helper
            # with a deliberately broken transport.
            teardown_clean_path(bridge, loop)
    finally:
        # Closes the loop on every documented exit path:
        # - ``BridgeFixture.start()`` raising (reachable per bridge.py:707-732)
        # - the test body raising (after ``bridge.stop()`` runs)
        # - the clean path (after the teardown assertion runs)
        # - any future code that raises between fixture construction and the
        #   inner try block — the loop is always the conftest's, never the
        #   bridge's, and would strand itself otherwise.
        loop.close()


# ── T-J3 (KBR-109) fixtures: sealed network + refusing-profile start path ─


@pytest.fixture
def sealed_network(
    certs: CertFiles,
    aiohttp_trusts_test_ca: None,
) -> Iterator[tuple[asyncio.AbstractEventLoop, SealedNetwork]]:
    """Yield ``(loop, started SealedNetwork)`` for one EG scenario.

    The loop is owned here, not by pytest-asyncio: pytest-bdd's generated
    tests are synchronous, and the sealed network's aiohttp servers are bound
    to the loop that started them — the same constraint
    :func:`bridge_session` documents for the bridge fixture. A
    pytest-asyncio-managed fixture would run on a loop that is closed by the
    time the sync step body executes, stranding every subsequent drive.

    Args:
        certs: The harness throwaway TLS certificates (from
            ``harness.connect_proxy``, exposed suite-wide via
            ``pytest_plugins`` in tests/conftest.py).
        aiohttp_trusts_test_ca: The harness fixture that points the
            bridge's aiohttp client trust store at the harness CA. Without
            it the bridge's outbound TLS handshake to the recorder fails
            with ``ClientConnectorCertificateError`` — measured, not
            assumed: the first EG run without it timed out through three
            blip retries.

    Yields:
        The asyncio loop and the running harness — proxy + recording
        upstream sharing :data:`harness.connect_proxy.HARNESS_UPSTREAM_HOST`.
        Function-scoped on purpose: each EG scenario gets a fresh recorder,
        so a phase-1 peer-port row cannot leak into a phase-2 "zero
        connections" assertion across scenarios (the same trap T-E2
        documents in tests/harness/test_aiohttp_containment_slice.py).
    """
    loop = asyncio.new_event_loop()
    net = SealedNetwork(ContainmentFormat.ANTHROPIC_MESSAGES, certs=certs)
    try:
        loop.run_until_complete(net.start())
        yield loop, net
    finally:
        try:
            loop.run_until_complete(net.stop())
        finally:
            loop.close()


@pytest.fixture
def egress_for_sealed_network(
    sealed_network: tuple[asyncio.AbstractEventLoop, SealedNetwork],
) -> Iterator[EgressConfig]:
    """An :class:`EgressConfig` pointed at the sealed network's proxy.

    Args:
        sealed_network: The ``(loop, net)`` tuple the
            :func:`sealed_network` fixture yields; only ``net`` is read.

    Yields:
        The egress configuration the EG-1 and EG-2 steps hand to
        ``BridgeServer``. Built from the proxy the sealed network is
        already running so the two fixtures share one source of truth
        (the proxy URL the recorder expects is the proxy URL the bridge
        dials).
    """
    _, net = sealed_network
    yield proxy_config(net.proxy.port)


@pytest.fixture
def refusing_profile_start_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """Patch every collaborator ``bridge_runner.main`` reads for one EG-3 scenario.

    Mirrors :func:`tests.test_egress_start_path.start_path`. The duplication
    is deliberate: tests/acceptance/ has no ``l3`` job, and importing the L3
    fixture across the layer boundary would couple the pytest-bdd binding to
    that fixture's monkeypatch lifetime.

    Args:
        monkeypatch: Pytest's monkeypatch fixture; the patches revert on
            teardown.

    Yields:
        A dict the EG-3 step consults: ``captured`` records
        ``BridgeServer.__init__`` calls, ``egress`` is the egress config
        the guard rejects against, and ``profile_name`` is the asserted
        stderr substring.
    """
    from kitty import bridge_runner

    egress = EgressConfig(proxy_url="http://proxy.example:1234", username="u", password="s3cr3tpw")
    # ``main()`` re-resolves egress from the environment or the on-disk store
    # (bridge_runner.py line 113), so the fixture patches the resolver itself.
    monkeypatch.setattr("kitty.egress_store.resolve_egress", lambda **kwargs: egress)
    monkeypatch.setattr("kitty.egress._egress", egress, raising=False)

    captured = {"bridge_server_init_calls": 0}

    class _FakeProfileStore:
        def get_backend(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(
                name=EG3_PROFILE_NAME,
                provider="bedrock",
                provider_config={"region": "us-east-1"},
                auth_ref="dummy-ref",
                model="anthropic.claude-3-sonnet",
            )

    class _FakeCredentialStore:
        def __init__(self, backends: object = None) -> None:
            pass

        def get(self, ref: str) -> str:
            # SSO-shaped marker: makes BedrockAdapter.supports_egress return
            # False, which is what the guard refuses on.
            return "sso"

    class _BridgeServerSpy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["bridge_server_init_calls"] += 1

        async def start_async(self) -> int:
            raise SystemExit(0)

        async def stop_async(self) -> None:
            pass

    monkeypatch.setattr("kitty.profiles.store.ProfileStore", _FakeProfileStore)
    monkeypatch.setattr("kitty.credentials.store.CredentialStore", _FakeCredentialStore)
    # ``BridgeServer`` is imported at bridge_runner module level, so patch the
    # binding bridge_runner resolves against, not the source module.
    monkeypatch.setattr(bridge_runner, "BridgeServer", _BridgeServerSpy)
    monkeypatch.setattr(bridge_runner.asyncio, "run", lambda coro: coro.close())
    monkeypatch.setattr(sys, "argv", ["bridge_runner", "--profile", EG3_PROFILE_NAME])

    yield {"captured": captured, "egress": egress, "profile_name": EG3_PROFILE_NAME}
