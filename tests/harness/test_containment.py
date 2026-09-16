"""The sealed-network containment harness and its capability report.

`.system_design/TEST_SUITE.md` §5.2, §5.3, §5.4, §7.3 · plan task **T-E1**
([KBR-61](https://shelpuk.atlassian.net/browse/KBR-61)) ·
`.requirements/20260914T201610Z_kbr61_t_e1_containment_harness_core/REQUIREMENTS.md`.

This module tests the **harness core** T-E1 ships: the sealed network (proxy +
recording upstream addressed by ``HARNESS_UPSTREAM_HOST``), the monkeypatched
aiohttp resolver that gives the bridge's direct leg a name the public DNS
cannot resolve, the per-transport capability report initialised with every
transport ``not_attempted``, and the containment transport extension interface
with the bridge-aiohttp route registered as the default.

Phases 2 / 2b / 3 (proxy-down ⇒ zero connections, proxy-up ⇒ every connection
joins a tunnel, injected bypass into the product makes the harness fail) are
T-E2 ([KBR-62](https://shelpuk.atlassian.net/browse/KBR-62)). T-E1 ships the
machinery and demonstrates the **direct leg**, per §5.2.2 phase 1.

**Falsification (plan §1.4).** ``TestBridgeAiohttpContainment`` ships a
deliberately broken direct route: a resolver mapped to a port with no
listener. The harness's positive control fails — recorder receives zero
connections and the bridge answers an error — and the green assertion
"exactly one capture" is what catches it. Shipping the falsification here
fulfils §1.4 and is *not* T-E2's own phase-3 obligation, which injects a
bypass into the product.

**Layer.** No ``pytestmark``, so these take the ``l1`` path default, following
``test_bridge.py`` and ``test_vertical_slice.py``. The reason is §8.2's and
only §8.2's: ``l3`` is in ``PENDING_ACTIVATION_LAYERS``, so an ``l3`` marker
today would leave the containment harness's correctness checked by no job at
all. §8.2 names this module as the T-E1 bullet, so T-K6 inherits the
relocation. ``test_default_resolver_is_threaded``
([KBR-259](https://shelpuk.atlassian.net/browse/KBR-259)) pins the aiohttp
``DefaultResolver`` premise of ``monkeypatched_aiohttp_resolver``; it belongs
to the same l1 cohort, so the pin and the seam move together when T-K6
activates the Subsystem job.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import ssl
from collections.abc import AsyncGenerator

import aiohttp
import pytest

from harness import containment
from harness.connect_proxy import HARNESS_UPSTREAM_HOST, CertFiles, ConnectProxy
from harness.containment import (
    BridgeAiohttpContainment,
    CapabilityReport,
    Outcome,
    Phase1Result,
    SealedNetwork,
    get_containment_transport,
    monkeypatched_aiohttp_resolver,
    register_containment_transport,
    registered_containment_transports,
    reset_for_test,
)
from harness.containment import (
    instance as report_instance,
)
from harness.contract import WireFormat
from harness.recorder import RecordingUpstream

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def capability_report() -> CapabilityReport:
    """A fresh capability report per test, so unit assertions are independent.

    The subsystem-wide singleton (:func:`harness.containment.instance`)
    is the wire T-E2..T-E5 record into and T-E9's completeness gate reads.
    T-E1's own unit tests use a fresh report so a ``record()`` in one test
    cannot leak into another's assertion.
    """
    return CapabilityReport()


@pytest.fixture
async def sealed_network(certs: CertFiles) -> AsyncGenerator[SealedNetwork, None]:
    """One started ``SealedNetwork`` for the test, torn down on exit.

    Yields:
        The running harness.
    """
    net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs)
    await net.start()
    try:
        yield net
    finally:
        await net.stop()


# ── R2: the monkeypatched resolver ─────────────────────────────────────────


class TestMonkeypatchedResolver:
    """``socket.getaddrinfo`` maps the harness hostname and defers the rest (§5.3)."""

    def test_mapping_returns_the_target_sockaddr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The patched call for the harness hostname returns the mapped address."""
        with monkeypatched_aiohttp_resolver(monkeypatch, "upstream.kitty-test.invalid", 9001):
            entries = socket.getaddrinfo("upstream.kitty-test.invalid", 9001, type=socket.SOCK_STREAM)
            assert entries == [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 9001))]

    def test_deferral_keeps_other_names_unmapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Names outside the mapping fall through to the real resolver.

        ``127.0.0.1`` is an IP literal; the kernel resolves it without a DNS
        round trip, so this is a pure deferral check with no network.
        """
        with monkeypatched_aiohttp_resolver(monkeypatch, "upstream.kitty-test.invalid", 9001):
            entries = socket.getaddrinfo("127.0.0.1", 80, type=socket.SOCK_STREAM)
            assert any(entry[4] == ("127.0.0.1", 80) for entry in entries)

    def test_monkeypatch_teardown_restores_the_real_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After ``monkeypatch.undo()``, ``getaddrinfo`` behaves as before.

        ``monkeypatch`` is function-scoped: its reverts run at test teardown,
        not at context-manager exit. ``undo()`` is the seam to observe the
        revert synchronously; the assertion is what "restored" means in
        observable terms — the mapped name is no longer mapped, so the
        system call cannot return an address.
        """
        with monkeypatched_aiohttp_resolver(monkeypatch, "upstream.kitty-test.invalid", 9001):
            inside = socket.getaddrinfo("upstream.kitty-test.invalid", 9001, type=socket.SOCK_STREAM)
            assert inside[0][4] == ("127.0.0.1", 9001)
        monkeypatch.undo()

        # The real resolver, against an RFC-2606 .invalid name, raises. The
        # negative form is intentional: ``gaierror`` is what an un-mapped
        # .invalid lookup produces on Linux and macOS alike.
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("upstream.kitty-test.invalid", 9001, type=socket.SOCK_STREAM)

    def test_deferral_passes_a_hostname_through_to_the_real_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hostname-shaped deferral under the patch lands on the real resolver.

        AC-R2's "example.com returns the real result" maps cleanly to this:
        any name other than ``host`` is dispatched to the un-patched resolver.
        An RFC-2606 ``.invalid`` name raises ``gaierror`` from the real
        resolver — the same shape the negative test above asserts, here
        observed *under* the patch.
        """
        with (
            monkeypatched_aiohttp_resolver(monkeypatch, "upstream.kitty-test.invalid", 9001),
            pytest.raises(socket.gaierror),
        ):
            socket.getaddrinfo("other.invalid", 80, type=socket.SOCK_STREAM)

    def test_default_resolver_is_threaded(self) -> None:
        """Pin aiohttp's ``DefaultResolver`` to ``ThreadedResolver`` (TEST_SUITE.md §5.3).

        The direct-leg seam this class tests — ``monkeypatched_aiohttp_resolver``
        — works only because the connector the bridge builds resolves through
        ``socket.getaddrinfo``. With no resolver injected, ``TCPConnector``
        picks ``aiohttp.resolver.DefaultResolver``; in the no-``aiodns`` build
        that is ``ThreadedResolver``, whose ``resolve`` calls ``loop.getaddrinfo``.
        Adding ``aiodns`` ≥ 3.2 (the version whose ``DNSResolver`` exposes
        ``getaddrinfo``) flips the selection at aiohttp import time to
        ``AsyncResolver``, which resolves via c-ares and never calls
        ``socket.getaddrinfo``: the patch silently stops working and every
        §5.2.2 phase-1 positive control fails for the wrong reason.

        The message names both ``AsyncResolver`` and ``aiodns`` so a future
        failure points at the cause, and preempts the false-alarm path (a
        future aiohttp default that still routes through ``socket.getaddrinfo``):
        update this pin deliberately rather than delete it.
        """
        assert aiohttp.resolver.DefaultResolver is aiohttp.resolver.ThreadedResolver, (
            "aiohttp's DefaultResolver is "
            f"{aiohttp.resolver.DefaultResolver.__name__}, not ThreadedResolver: "
            "an aiodns install flips aiohttp.resolver.DefaultResolver to "
            "AsyncResolver, which resolves via c-ares and never calls "
            "socket.getaddrinfo, so monkeypatched_aiohttp_resolver "
            "(tests/harness/containment.py) silently stops working and every "
            "§5.2.2 phase-1 positive control fails for the wrong reason "
            "(TEST_SUITE.md §5.3). If a future aiohttp default still routes "
            "through socket.getaddrinfo, update this pin deliberately rather "
            "than delete it."
        )


# ── R3: the capability report ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_singleton() -> None:
    """Reset the capability-report singleton before every test in this module.

    The autouse fixture is declared at module scope (unindented between
    :class:`TestMonkeypatchedResolver` and :class:`TestCapabilityReport`),
    so its ``autouse=True`` applies to **every** test in the file, not just
    the class below it; a class-scoped autouse would be impossible to write
    in pytest, and the wording here is what the reader needs to know when
    they look for the seam.

    The singleton is the wire T-E2..T-E5 record into and T-E9's completeness
    gate reads; a sibling slice's verdict written earlier in the process would
    leave the per-test "initial state" assertion false. ``reset_for_test``
    replaces it with a fresh every-``not_attempted`` report before each test;
    the per-test :func:`capability_report` fixture is unchanged, so unit tests
    constructing a fresh :class:`CapabilityReport` directly are unaffected.
    """
    reset_for_test()


class TestCapabilityReport:
    """The in-process report the completeness gate reads (§5.3, §5.4)."""

    def test_initial_state_has_every_transport_not_attempted(self, capability_report: CapabilityReport) -> None:
        """Every §5.5 transport starts ``not_attempted``, so T-E9's gate is meaningful."""
        outcomes = {entry.outcome for entry in capability_report.entries().values()}
        assert outcomes == {Outcome.NOT_ATTEMPTED}

    def test_record_marks_an_entry_as_proven(self, capability_report: CapabilityReport) -> None:
        """A successful slice writes ``proven``; the verdict is one record call."""
        capability_report.record("bridge_aiohttp", Outcome.PROVEN)
        assert capability_report.outcome("bridge_aiohttp") is Outcome.PROVEN

    def test_record_unsupported_carries_a_reason(self, capability_report: CapabilityReport) -> None:
        """``unsupported`` is permitted for partial delivery and carries its reason."""
        capability_report.record("curl_cffi", Outcome.UNSUPPORTED, reason="curl resolve unsupported")
        entry = capability_report.entry("curl_cffi")
        assert entry.outcome is Outcome.UNSUPPORTED
        assert entry.reason == "curl resolve unsupported"

    def test_record_failed_carries_no_reason(self, capability_report: CapabilityReport) -> None:
        """``failed`` is a product defect; the verdict is the outcome itself."""
        capability_report.record("botocore", Outcome.FAILED)
        assert capability_report.outcome("botocore") is Outcome.FAILED

    def test_not_attempted_names_returns_remaining_pending(self, capability_report: CapabilityReport) -> None:
        """Names whose outcome is still ``not_attempted`` are the gate's input."""
        capability_report.record("bridge_aiohttp", Outcome.PROVEN)
        capability_report.record("curl_cffi", Outcome.UNSUPPORTED, reason="x")
        assert set(capability_report.not_attempted_names()) == {"provider_aiohttp", "botocore"}

    def test_record_an_unknown_transport_raises(self, capability_report: CapabilityReport) -> None:
        """Adding a new transport is a deliberate decision, not a silent default."""
        with pytest.raises(KeyError, match="not_a_real_transport"):
            capability_report.record("not_a_real_transport", Outcome.PROVEN)  # type: ignore[arg-type]

    def test_record_rejects_an_unregistered_outcome(self, capability_report: CapabilityReport) -> None:
        """An outcome outside the closed enum is a caller bug, not a verdict."""
        with pytest.raises(ValueError, match="outcome"):
            capability_report.record("bridge_aiohttp", "passed")  # type: ignore[arg-type]

    def test_singleton_exposes_every_transport_not_attempted(self) -> None:
        """``instance()`` hands back the same four rows on call.

        The all-``not_attempted`` assertion is scoped to **T-E1's** run: it
        states the report's *initial* state, which is what this ticket ships
        and what the future T-E9 gate reads. T-E2's own obligation is to
        record the first verdict into this singleton — from that commit
        onward, the assertion below is T-E2's to narrow, not T-E1's to
        defend.
        """
        report = report_instance()
        names = sorted(report.entries())
        assert names == ["botocore", "bridge_aiohttp", "curl_cffi", "provider_aiohttp"]
        outcomes = {entry.outcome for entry in report.entries().values()}
        assert outcomes == {Outcome.NOT_ATTEMPTED}
        # Same instance every call (in this process): the singleton is the
        # wire T-E2..T-E5 record into and T-E9's gate reads.
        assert report_instance() is report

    def test_require_completeness_raises_when_any_transport_pending(self, capability_report: CapabilityReport) -> None:
        """``require_completeness`` is what the future T-E9 completeness gate calls."""
        capability_report.record("bridge_aiohttp", Outcome.PROVEN)
        capability_report.record("curl_cffi", Outcome.PROVEN)
        capability_report.record("provider_aiohttp", Outcome.PROVEN)
        capability_report.record("botocore", Outcome.PROVEN)
        # No `pending` rows; the gate stays quiet.
        capability_report.require_completeness()

        fresh = CapabilityReport()
        fresh.record("bridge_aiohttp", Outcome.PROVEN)
        with pytest.raises(AssertionError) as excinfo:
            fresh.require_completeness()
        # The message names every pending transport so a CI failure is
        # diagnosable without a re-run.
        message = str(excinfo.value)
        for name in ("curl_cffi", "provider_aiohttp", "botocore"):
            assert name in message, f"missing pending transport {name!r} in gate message"

    def test_reset_for_test_replaces_the_singleton_with_every_not_attempted(self) -> None:
        """``reset_for_test`` returns the singleton to its initial state.

        The autouse fixture in this class already does this; the test is the
        contract the fixture rests on. A sibling slice that records a verdict
        into ``instance()`` and then sees another test's
        ``test_singleton_exposes_every_transport_not_attempted`` go green is
        what makes the singleton co-tenant-safe.
        """
        before = report_instance()
        before.record("bridge_aiohttp", Outcome.PROVEN)
        # Before resetting: at least one row is *not* ``not_attempted``.
        assert before.outcome("bridge_aiohttp") is Outcome.PROVEN

        reset_for_test()

        # After resetting: a fresh object, every row ``not_attempted``.
        assert report_instance() is not before
        assert set(report_instance().not_attempted_names()) == {
            "bridge_aiohttp",
            "curl_cffi",
            "provider_aiohttp",
            "botocore",
        }


# ── R1: the sealed-network harness ─────────────────────────────────────────


class TestSealedNetwork:
    """The proxy + recording upstream stood up together, addressed by the harness hostname."""

    async def test_starts_a_proxy_and_a_recorder_sharing_one_resolved_name(self, sealed_network: SealedNetwork) -> None:
        """``HARNESS_UPSTREAM_HOST:{port}`` resolves to the recorder via the proxy's map."""
        assert sealed_network.upstream_host == HARNESS_UPSTREAM_HOST
        assert sealed_network.upstream_port > 0

        target = f"{HARNESS_UPSTREAM_HOST}:{sealed_network.upstream_port}"
        assert target in sealed_network.proxy.resolve
        host, port = sealed_network.proxy.resolve[target]
        assert host == "127.0.0.1"
        assert port == sealed_network.recorder.port

    async def test_proxy_records_no_attempts_with_no_client(self, sealed_network: SealedNetwork) -> None:
        """A sealed network with no client reports zero CONNECT attempts."""
        assert sealed_network.proxy.attempts == []

    async def test_stopping_releases_the_proxy_port(self, certs: CertFiles) -> None:
        """After ``stop()`` the proxy's port is dead, and ``start()`` again binds a new one.

        T-E2's phase 2 stops the proxy mid-test; T-E1's own unit test shows the
        seam T-W5 already gives it keeps working through this harness too.
        """
        net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs)
        await net.start()
        old_port = net.proxy.port
        try:
            await net.stop()
            with pytest.raises(OSError):  # ConnectionRefusedError is an OSError subclass.
                await _connect_to("127.0.0.1", old_port)

            await net.start()
            try:
                assert net.proxy.port != 0
            finally:
                await net.stop()
        finally:
            await net.stop()  # Idempotent (T-W5's contract).

    async def test_stop_releases_the_recorder_even_if_the_proxy_stop_raises(self, certs: CertFiles) -> None:
        """The recorder stops even when the proxy's ``stop()`` raised (review M3).

        ``ConnectProxy.stop`` can raise ``TimeoutError`` on a stuck client. A
        teardown that stops there would leak the recorder's port for the rest
        of the session. The defect this test catches is a ``stop()`` that
        does not wrap the proxy's stop in ``try/finally`` — a sibling copy
        without the guard would leave the recorder running when the proxy's
        stop raised.
        """
        net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs)
        await net.start()
        recorder = net.recorder

        # Force the proxy's stop to raise so the guard is exercised.
        async def _raising_stop() -> None:
            raise TimeoutError("forced by test")

        net.proxy.stop = _raising_stop  # type: ignore[method-assign]
        with contextlib.suppress(TimeoutError):
            await net.stop()

        # `RecordingUpstream.stop()` clears its runner, and `port` then
        # raises — the observable that says "the recorder is really stopped",
        # not merely "harness forgot about it".
        with pytest.raises(RuntimeError, match="not running"):
            _ = recorder.port

    async def test_start_releases_the_recorder_when_the_proxy_start_raises(
        self, certs: CertFiles, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorder is released when the proxy's ``start()`` raised (review M3/N4).

        ``__aenter__`` propagates a failure and ``__aexit__`` is never called
        on the failed entry, so without the guard the recorder would hold its
        port for the rest of the session. The twin of the stop-side test
        above: inject a raising ``ConnectProxy.start``, capture the recorder
        ``SealedNetwork.start`` built, and observe that its runner is gone —
        the distinction between "harness forgot about it" and "it really
        stopped" is made by holding the recorder handle directly.
        """
        net = SealedNetwork(WireFormat.ANTHROPIC_MESSAGES, certs=certs)
        started: list[RecordingUpstream] = []

        class _CapturingRecorder(RecordingUpstream):
            async def start(self, *args: object, **kwargs: object) -> None:
                await super().start(*args, **kwargs)  # type: ignore[arg-type]
                started.append(self)

        monkeypatch.setattr(containment, "RecordingUpstream", _CapturingRecorder)

        async def _raising_start(_proxy: ConnectProxy, ssl_context: ssl.SSLContext) -> None:
            raise OSError("forced by test")

        monkeypatch.setattr(ConnectProxy, "start", _raising_start)

        with pytest.raises(OSError):
            await net.start()

        assert len(started) == 1, "the recorder SealedNetwork built was not observed"
        with pytest.raises(RuntimeError, match="not running"):
            _ = started[0].port

    async def test_factory_receives_the_target_ssl_context_and_starts_without_a_kwarg(
        self, certs: CertFiles, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A factory-built recorder is started with no ``ssl_context`` argument.

        ``CurlRecordingUpstream`` takes ``ssl_context`` at construction and its
        ``start()`` takes no arguments; ``RecordingUpstream`` takes the context
        at ``start()``. The factory contract hands the factory the **ready**
        ``SSLContext`` so the leaf-cert wiring stays in one place, and
        ``SealedNetwork.start`` calls ``await recorder.start()`` for a
        factory-built recorder (no ``ssl_context=`` kwarg) so neither shape
        breaks. The cert identity (not just the shape) is asserted via a
        ``server_ssl_context`` spy — the spy and ``SealedNetwork.start`` see
        the **same** ``SSLContext`` object, so ``is`` is a real identity check.
        """
        from harness.connect_proxy import server_ssl_context

        spy_ctx: list[tuple[Path, Path, ssl.SSLContext]] = []
        real_server_ctx = server_ssl_context

        def _recording_server_ctx(cert: Path, key: Path) -> ssl.SSLContext:
            ctx = real_server_ctx(cert, key)
            # SealedNetwork.start builds TWO contexts — the proxy's and the
            # recorder's — so the spy records the cert/key alongside and the
            # assertion matches on the target pair, not on call order.
            spy_ctx.append((cert, key, ctx))
            return ctx

        monkeypatch.setattr(containment, "server_ssl_context", _recording_server_ctx)

        seen_factory_ctx: list[ssl.SSLContext] = []
        seen_args: list[dict[str, object]] = []
        built_recorder: list[RecordingUpstream] = []

        class _FactoryRecorder(RecordingUpstream):
            def __init__(self, ssl_context: ssl.SSLContext, **kwargs: object) -> None:
                super().__init__(default_format=WireFormat.OPENAI_RESPONSES)
                self._ssl_context = ssl_context

            def __post_init__(self) -> None:
                # Bypass the base recorder's format allow-list — irrelevant
                # to the factory contract this test asserts.
                ...  # noqa: PIE790

            async def start(self, *args: object, **kwargs: object) -> None:
                seen_args.append(kwargs)
                # Delegate to the real start so the recorder actually binds a
                # port; the resolve map built by SealedNetwork reads
                # ``self._recorder.port``, which the base class populates.
                await RecordingUpstream.start(self, ssl_context=self._ssl_context)

        def _factory(ctx: ssl.SSLContext) -> _FactoryRecorder:
            seen_factory_ctx.append(ctx)
            rec = _FactoryRecorder(ctx)
            built_recorder.append(rec)
            return rec

        net = SealedNetwork(
            WireFormat.ANTHROPIC_MESSAGES,
            certs=certs,
            recorder_factory=_factory,  # type: ignore[arg-type]
        )
        await net.start()
        try:
            assert len(seen_factory_ctx) == 1, "the factory was not invoked exactly once"
            target_spy = [(k, c) for cert, k, c in spy_ctx if cert == certs.target_cert]
            assert len(target_spy) == 1, (
                f"SealedNetwork did not build a recorder SSLContext from target_cert "
                f"via the spy (spy recorded {[(cert, k) for cert, k, _ in spy_ctx]})"
            )
            key_used, ctx_for_target = target_spy[0]
            assert key_used == certs.target_key, (
                "SealedNetwork's target-cert ctx was built with the wrong key "
                f"(got {key_used}, expected {certs.target_key})"
            )
            assert seen_factory_ctx[0] is ctx_for_target, (
                "factory did not receive the SSLContext SealedNetwork built from the "
                "target cert/key — cert ownership must stay with SealedNetwork "
                f"(target cert carries {HARNESS_UPSTREAM_HOST!r} in its SAN)"
            )
            assert len(seen_args) == 1, "the factory's recorder was never started"
            assert "ssl_context" not in seen_args[0], (
                f"factory-built recorder's start() received ssl_context kwarg "
                f"({seen_args[0]!r}); CurlRecordingUpstream.start() takes no arguments"
                " — the factory binds the context at construction"
            )
            assert built_recorder, "no recorder was built by the factory"
            assert net.recorder is built_recorder[0], (
                "SealedNetwork's recorder is not the factory-built instance: cert/ctx "
                "ownership must flow through the factory, not be re-constructed"
            )
        finally:
            await net.stop()

    async def test_factory_owns_the_served_format(self, certs: CertFiles) -> None:
        """When a factory is given, ``fmt`` on the constructor is ignored.

        The factory is the T-E3/T-E4/T-E5 extension point and decides which
        wire format the recorder serves (the curl_cffi slice needs
        ``OPENAI_RESPONSES``; the botocore slice needs its own). Passing a
        different ``fmt`` to ``SealedNetwork`` should not override the
        factory's choice — the test asserts the recorder ends up with the
        factory's format.
        """
        class _FormatRecorder(RecordingUpstream):
            def __init__(self, ssl_context: ssl.SSLContext, **kwargs: object) -> None:
                super().__init__(default_format=WireFormat.OPENAI_RESPONSES)

            def __post_init__(self) -> None:
                # Same allow-list bypass as in the factory-contract test above: the
                # harness factory is format-agnostic and this test asserts ownership,
                # not the validity of a specific format under the base recorder's
                # product-level invariant.
                ...  # noqa: PIE790 — intentional no-op stub

        net = SealedNetwork(
            WireFormat.ANTHROPIC_MESSAGES,  # deliberately not the factory's format
            certs=certs,
            recorder_factory=lambda ctx: _FormatRecorder(ctx),  # type: ignore[arg-type, return-value]
        )
        await net.start()
        try:
            assert net.recorder.default_format is WireFormat.OPENAI_RESPONSES, (
                "factory-built recorder did not own the served format: the factory's "
                "default_format was overridden by the SealedNetwork fmt argument"
            )
        finally:
            await net.stop()


# ── R4: the containment transport extension interface ──────────────────────


class TestContainmentRegistry:
    """The protocol + registry T-E3..T-E5 plug into."""

    def test_bridge_aiohttp_is_registered_by_default(self) -> None:
        """``BridgeAiohttpContainment`` ships as the default registration."""
        assert "bridge_aiohttp" in registered_containment_transports()
        assert isinstance(get_containment_transport("bridge_aiohttp"), BridgeAiohttpContainment)

    def test_registering_a_duplicate_name_raises(self) -> None:
        """Two definitions for one transport is a coordination failure."""
        with pytest.raises(ValueError, match="already registered"):
            register_containment_transport("bridge_aiohttp", BridgeAiohttpContainment)

    def test_unknown_transport_lookup_raises(self) -> None:
        """Looking up an unregistered name is a caller bug, not a fallback."""
        with pytest.raises(LookupError):
            get_containment_transport("definitely_not_registered")


class TestBridgeAiohttpContainment:
    """``BridgeAiohttpContainment.drive_phase_1`` — the direct leg T-E2 builds on."""

    def test_direct_route_is_a_context_manager_over_sealed_network(self, sealed_network: SealedNetwork) -> None:
        """The protocol exposes ``direct_route``; the default implementation yields once.

        T-E3..T-E5 override this with their own transport-specific override
        (curl_cffi's ``--resolve`` mapping, botocore's ``endpoint_url``, the
        provider-aiohttp session's resolver hook). The bridge-aiohttp default
        is a no-op ``yield`` — the mechanism it needs (``socket.getaddrinfo``
        mapping) lives on ``drive_phase_1``, which enters
        ``self.direct_route(harness)`` so the override is in scope.
        """
        transport = BridgeAiohttpContainment()

        # The no-op ``direct_route`` is enterable and yields once; whatever
        # the with-body does runs inside.
        with transport.direct_route(sealed_network):
            pass  # Body ran; the with-block completed without raising.

    async def test_drive_phase_1_one_request_reaches_the_recorder_directly(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The bridge, egress off, resolver patched, posts once and lands on the harness.

        The recorder is TLS (see :attr:`SealedNetwork.upstream_base_url`), so
        the bridge's outbound aiohttp client must trust the harness CA — the
        fixture's job, on this hop and on the proxied phases' hop alike.
        """
        result = await BridgeAiohttpContainment().drive_phase_1(sealed_network, monkeypatch=monkeypatch)

        assert isinstance(result, Phase1Result)
        # Green-path expectations, stated separately so a defect that breaks
        # only one fails the test rather than masking another.
        assert result.status == 200, f"bridge answered {result.status}: {result.text!r}"
        assert len(result.captures) == 1, (
            f"recorder saw {len(result.captures)} capture(s), expected exactly 1: "
            "0 means the bridge reached some other upstream or the resolver mapping "
            "was bypassed; more than 1 means a retry ladder fired"
        )
        assert any(c.peer_port and c.peer_port > 0 for c in result.connections), (
            "recorder's connection log has no entry with a real peer port: the bridge's "
            "outbound connection was never logged, so the §5.2.1 join has nothing to bind to"
        )
        assert result.attempts == [], (
            f"proxy saw {len(result.attempts)} CONNECT attempt(s) with egress off: the "
            "direct leg went through the proxy by mistake, so phase 1 proved nothing"
        )

    async def test_drive_phase_1_with_a_broken_resolver_records_zero_connections(
        self,
        sealed_network: SealedNetwork,
        aiohttp_trusts_test_ca: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Falsification (plan §1.4): a wrong-port resolver must make the green path fail.

        The defect is "the resolver maps the harness hostname to a port with no
        listener". The bridge's aiohttp client attempts to connect, the
        kernel refuses; the test's outer ``aiohttp.ClientSession`` then times
        out at ``_DRIVE_TIMEOUT`` while :meth:`BridgeServer._make_upstream_request`'s
        :meth:`_wait_out_transport_blip` (server.py:8579) sleeps through
        ``_TRANSPORT_GRACE_DELAYS = (2.0, 4.0, 8.0, 16.0)`` (server.py:1030,
        summing to ``_TRANSPORT_GRACE_PERIOD = 30.0``, server.py:1029); the
        drive's ``status`` comes back as ``-1`` and
        ``text`` carries the ``TimeoutError`` repr. The recorder receives
        nothing in any of those shapes, which is what the green assertion
        ``captures == 1`` catches. The fixture is taken so the bridge's TLS
        verify step is in scope on the green path's notional branch — the
        broken-resolver branch never reaches TLS verify, but the
        consistency between this test and the green-path one is what the
        reader expects.

        **Cost: ~30 s.** The 10-second client timeout plus ``stop_async()``
        waiting out the bridge's in-flight retries is the price of driving
        a real bridge rather than a stub; the only way to make this test
        fast is to stop driving a real bridge, which would falsify a
        different claim.
        """
        broken_port = _find_closed_port()
        result = await BridgeAiohttpContainment().drive_phase_1(
            sealed_network, monkeypatch=monkeypatch, resolver_port=broken_port
        )

        # The phase-1 positive control failed; the bridge's reply is non-200
        # and the recorder holds no observation of the request.
        assert result.status != 200, (
            f"bridge answered {result.status} with a broken resolver: the positive control "
            "was supposed to fail, but a 200 means the resolver was bypassed"
        )
        assert result.captures == [], (
            f"recorder saw {len(result.captures)} capture(s) with a broken resolver: "
            "if the bridge really failed to reach it, the recorder must hold nothing"
        )


# ── Helpers (private) ─────────────────────────────────────────────────────


def _find_closed_port() -> int:
    """Reserve an ephemeral port and close it, returning a port whose socket is gone.

    Used by the falsification case to produce a number the client cannot dial.
    A subsequent ``connect()`` against this port is expected to fail with
    ``ConnectionRefusedError`` on every supported platform.

    Returns:
        A TCP port number that is no longer listening.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


async def _connect_to(host: str, port: int) -> None:
    """Open a TCP connection and close it immediately.

    Args:
        host: The host to dial.
        port: The port to dial.

    Raises:
        ConnectionRefusedError: When nothing is listening on ``port``.
    """
    _reader, writer = await asyncio.open_connection(host, port)
    writer.close()
