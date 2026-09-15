"""The sealed-network containment harness: core, capability report, extension interface.

`.system_design/TEST_SUITE.md` §5.2, §5.3, §5.4, §7.3 · plan task **T-E1**
([KBR-61](https://shelpuk.atlassian.net/browse/KBR-61)) ·
`.requirements/20260914T201610Z_kbr61_t_e1_containment_harness_core/REQUIREMENTS.md`.

What this module owns:

* **`SealedNetwork`** — proxy + recording upstream stood up together, the
  upstream addressed by :data:`harness.connect_proxy.HARNESS_UPSTREAM_HOST` at
  its own ephemeral port and the proxy's ``resolve`` map carrying exactly that
  ``host:port`` → ``127.0.0.1:port`` binding. Sibling slices (T-E2..T-E5) read
  the same :class:`~harness.connect_proxy.ConnectProxy` and recording-upstream
  the harness holds.
* **`monkeypatched_aiohttp_resolver`** — the **direct**-leg override for the
  bridge's own aiohttp sessions: ``socket.getaddrinfo`` mapped for the harness
  hostname, deferring every other name to the real resolver. It patches
  ``getaddrinfo`` rather than an aiohttp ``Resolver`` instance because
  ``_build_client_session`` builds its own ``TCPConnector`` with no injection
  point (§5.3). The default (no-``aiodns``) build selects ``ThreadedResolver``
  as ``DefaultResolver``, which reaches ``getaddrinfo`` in a worker thread; if
  ``aiodns`` is ever added, ``AsyncResolver`` is chosen instead and this
  patch has no effect — ``pyproject.toml`` pins no ``aiodns`` extra, so the
  seam holds today. ``/etc/hosts`` is left alone — no administrator rights
  on CI runners.
* **The per-transport capability report** — :class:`CapabilityReport` initialised
  with the four §5.5 transports (``bridge_aiohttp``, ``provider_aiohttp``,
  ``curl_cffi``, ``botocore``), every entry ``not_attempted``; ``record``
  accepts ``proven``, ``unsupported`` (with a reason) and ``failed``; the
  future T-E9 completeness gate reads what the slices wrote. T-E1 ships the
  machinery and the registry — **no verdicts**, since T-E2..T-E5 write those
  once their respective phases pass.
* **The containment transport extension interface** — :class:`ContainmentTransport`
  + :func:`register_containment_transport` + a default registration of
  :class:`BridgeAiohttpContainment` that demonstrates the seam by driving one
  request through the bridge's aiohttp serving path with egress off and the
  patched resolver in scope, asserting the recorder recorded the connection
  and the proxy recorded zero CONNECTs. T-E3–T-E5 register their own without
  editing this module.

Delivered as an async context manager rather than pytest fixtures: the harness
owns the lifetime of two servers the proxy `stop()` contract depends on (§7.3
notes Python 3.12.1+ ``wait_closed``), and an explicit manager makes that
lifetime visible in the caller. T-E2..T-E5 inherit the manager; the fixture
form would let a test stop only the proxy and leak the recorder.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol, runtime_checkable

import aiohttp
import pytest

from harness.connect_proxy import (
    HARNESS_UPSTREAM_HOST,
    CertFiles,
    ConnectAttempt,
    ConnectProxy,
    server_ssl_context,
)
from harness.contract import CapturedRequest, WireFormat
from harness.recorder import ConnectionRecord, RecordingUpstream

__all__ = [
    "BridgeAiohttpContainment",
    "CapabilityReport",
    "ContainmentTransport",
    "Outcome",
    "Phase1Result",
    "ReportEntry",
    "SealedNetwork",
    "get_containment_transport",
    "instance",
    "monkeypatched_aiohttp_resolver",
    "register_containment_transport",
    "registered_containment_transports",
]

#: Default wire format. The harness's recorder answers in this format and the
#: bridge fixture's ``AiohttpTransport`` registers Anthropic Messages as one
#: of its two formats. The choice does not enter the bridge's adapter
#: selection (it goes through ``provider_config["base_url"]``) but controls
#: the recorder's reply shape so the bridge parses a non-empty success.
_DEFAULT_FORMAT = WireFormat.ANTHROPIC_MESSAGES

#: A budget for one ``drive_*`` request. The green path (resolver mapped to the
#: recorder's port) completes well under a second; the falsification path
#: (closed port) sees the bridge answer a 5xx quickly too. Ten seconds is the
#: bridge fixture's own default and what the suite pays for an absent
#: upstream; one bad test cannot hold the gate much longer than its budget.
_DRIVE_TIMEOUT = 10.0


class Outcome(Enum):
    """One containment verdict a transport slice writes into the report.

    The four-valued enum is the closed set §5.4 specifies: ``not_attempted`` is
    the report's initial state, ``proven`` and ``unsupported`` are the two
    permitted end states, and ``failed`` is the product defect the report
    exists to surface.
    """

    NOT_ATTEMPTED = "not_attempted"
    PROVEN = "proven"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True)
class ReportEntry:
    """One row of the capability report.

    Attributes:
        outcome: The verdict the slice recorded.
        reason: An explanatory string for the ``unsupported`` outcome. ``None``
            for ``proven`` and ``failed``; the latter is a product defect and
            the verdict itself is the explanation.
    """

    outcome: Outcome = Outcome.NOT_ATTEMPTED
    reason: str | None = None


class CapabilityReport:
    """Per-transport verdict aggregator the completeness gate reads (§5.4).

    The report is a **closed** registry over the four §5.5 transports: adding
    a new entry to ``_TRANSPORT_NAMES`` is a deliberate decision, not a silent
    default, because an unmarked row survives the gate's "no ``not_attempted``
    rows remain" check without ever having been exercised. The closed set
    also lets ``record()`` reject unknown names loudly.
    """

    #: The transports §5.5 names. Listed in this order for stable iteration
    #: in failure diagnostics; the gate does not care.
    _TRANSPORT_NAMES: ClassVar[tuple[str, ...]] = (
        "bridge_aiohttp",
        "provider_aiohttp",
        "curl_cffi",
        "botocore",
    )

    def __init__(self) -> None:
        """Initialise one row per registered transport, every row ``not_attempted``."""
        self._entries: dict[str, ReportEntry] = {name: ReportEntry() for name in self._TRANSPORT_NAMES}

    def entries(self) -> dict[str, ReportEntry]:
        """Return every row in the report, in registration order.

        Returns:
            A shallow copy of the internal map. The copy is what callers see
            — the report's :meth:`record` updates its private state, and
            returning the live dict would let a reader hold a stale snapshot
            through a write and silently absorb a future verdict on a stale
            row.
        """
        return dict(self._entries)

    def entry(self, name: str) -> ReportEntry:
        """Return the row for ``name``.

        Args:
            name: A registered transport name.

        Returns:
            The row currently held for that transport.

        Raises:
            KeyError: When ``name`` is not a registered transport. Adding a
                transport is a deliberate decision, not a silent default.
        """
        if name not in self._entries:
            raise KeyError(f"{name!r} is not a registered containment transport")
        return self._entries[name]

    def outcome(self, name: str) -> Outcome:
        """Return the outcome currently held for ``name``.

        Args:
            name: A registered transport name.

        Returns:
            The outcome.

        Raises:
            KeyError: When ``name`` is not a registered transport.
        """
        return self.entry(name).outcome

    def record(self, name: str, outcome: Outcome, *, reason: str | None = None) -> None:
        """Mutate the row for ``name`` with ``outcome``.

        Args:
            name: A registered transport name.
            outcome: The verdict to write.
            reason: **Required** when ``outcome is UNSUPPORTED`` so a partial
                delivery has a self-explanatory diagnostic; **forbidden**
                (``None``) for every other outcome. ``failed`` is a product
                defect the verdict itself names; ``not_attempted`` and
                ``proven`` carry no reason by construction. A non-``None``
                reason on either of those three is rejected loudly — silently
                ignoring it would let a future caller lose a debug signal
                without noticing.

        Raises:
            KeyError: When ``name`` is not a registered transport.
            ValueError: When ``outcome`` is not an :class:`Outcome` value, or
                when the ``reason`` rule above is violated.
        """
        if name not in self._entries:
            raise KeyError(f"{name!r} is not a registered containment transport")
        if not isinstance(outcome, Outcome):
            raise ValueError(f"outcome must be an Outcome value, got {outcome!r}")

        if outcome is Outcome.UNSUPPORTED and reason is None:
            raise ValueError("outcome=unsupported requires a reason")
        if outcome is Outcome.FAILED and reason is not None:
            raise ValueError("outcome=failed must not carry a reason")
        if outcome in (Outcome.NOT_ATTEMPTED, Outcome.PROVEN) and reason is not None:
            raise ValueError(f"outcome={outcome.value} must not carry a reason")

        # `UNSUPPORTED` may carry a reason; everything else is fixed.
        normalised_reason: str | None = reason if outcome is Outcome.UNSUPPORTED else None
        self._entries[name] = ReportEntry(outcome=outcome, reason=normalised_reason)

    def not_attempted_names(self) -> tuple[str, ...]:
        """Return every transport whose verdict is still ``not_attempted``.

        Returns:
            The names, in registration order. T-E9's gate iterates this list
            and fails the run when it is non-empty.
        """
        return tuple(name for name, entry in self._entries.items() if entry.outcome is Outcome.NOT_ATTEMPTED)

    def require_completeness(self) -> None:
        """Raise when any transport's verdict is still ``not_attempted``.

        The future T-E9 completeness gate (§5.3) calls this; the call site is
        a separate ticket. T-E1 ships the assertion so the gate has a single,
        named seam to call — three separate tickets inventing their own
        completeness check is the kind of coordination failure the milestone
        structure exists to prevent.

        Raises:
            AssertionError: When at least one transport is still
                ``not_attempted``. The message names every pending transport
                (sorted for a deterministic diff against the report's
                registration order) so a CI failure is diagnosable without
                a re-run.
        """
        pending = self.not_attempted_names()
        if pending:
            raise AssertionError(f"containment completeness gate failed; verdicts still pending: {sorted(pending)}")


#: The in-process singleton T-E2..T-E5 reach through and T-E9 reads.
#: Initialised lazily so importing the module does not mutate the report.
_REPORT: CapabilityReport | None = None


def instance() -> CapabilityReport:
    """Return the process-wide capability report.

    Constructed lazily on first call. Subsequent calls in the same process
    return the same object, which is what the future T-E9 completeness gate
    reads and what T-E2..T-E5 record into: an in-process view that one
    process's slices have all written their verdict into.

    Returns:
        The same :class:`CapabilityReport` instance every call. T-E1's own
        unit tests construct a fresh :class:`CapabilityReport` directly so a
        ``record()`` in one test cannot leak into another's assertion.
    """
    global _REPORT
    if _REPORT is None:
        _REPORT = CapabilityReport()
    return _REPORT


# ── The monkeypatched resolver (§5.3 direct-leg seam) ──────────────────────


@contextlib.contextmanager
def monkeypatched_aiohttp_resolver(mp: pytest.MonkeyPatch, host: str, port: int) -> Iterator[None]:
    """Map ``host`` to ``(127.0.0.1, port)`` for the duration of the block.

    Args:
        mp: Pytest's monkeypatch fixture. The patch reverts on its teardown.
        host: The hostname to map. Lookups for this host return a single
            ``AF_INET`` entry whose sockaddr is ``(127.0.0.1, port)``.
        port: The loopback port the patch resolves to.

    Yields:
        ``None``. :mod:`socket`'s ``getaddrinfo`` is patched for the duration;
        every other name falls through to the un-patched resolver.

    The patch is process-wide and broad — the bridge's ``_build_client_session``
    builds its own ``TCPConnector`` with no injection point (§5.3). In the
    default (no-``aiodns``) build the connector's ``DefaultResolver`` is
    ``ThreadedResolver``, which reaches ``socket.getaddrinfo`` in a worker
    thread — the seam this patch takes. If ``aiodns`` is ever installed,
    ``DefaultResolver`` becomes ``AsyncResolver``, which bypasses
    ``socket.getaddrinfo`` entirely and the patch has no effect; a guard
    against that regression is a deliberate next change, not something this
    seam quietly absorbs (``pyproject.toml`` pins no ``aiodns`` extra today).

    ``monkeypatch.setattr(socket, "getaddrinfo", ...)`` reverts when ``mp``
    finalises, so the context manager is re-entry safe across the suite.
    """
    original = socket.getaddrinfo

    def _patched(name: str, *args: Any, **kwargs: Any) -> Any:
        """Return the mapped address for ``host``; defer everything else.

        Args:
            name: The host being resolved.
            *args: Forwarded to the original resolver.
            **kwargs: Forwarded to the original resolver.

        Returns:
            A single ``getaddrinfo``-shaped entry for ``host``, or whatever the
            original resolver returned.
        """
        if name == host:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return original(name, *args, **kwargs)

    mp.setattr(socket, "getaddrinfo", _patched)
    yield


# ── The sealed network (§5.3 sealed-network harness) ───────────────────────


class SealedNetwork:
    """A proxy + recording-upstream pair sharing one harness hostname.

    The pair is bound on loopback, then reachable by a single non-loopback
    name — :data:`harness.connect_proxy.HARNESS_UPSTREAM_HOST` — at the
    proxy's port. The proxy's ``resolve`` map carries the sealed network's
    name → upstream translation; the bridge's direct leg patches
    ``getaddrinfo`` for the same hostname.

    Attributes:
        recorder: The recording upstream the bridge, the proxy and the
            sibling slices all observe. Re-exposed rather than constructed
            per-slice so T-E2..E5 do not see two recorders.
        proxy: The local CONNECT proxy the bridge may route through. The
            "stop me mid-test" contract T-W5 ships and T-E2's phase 2 uses.
    """

    def __init__(
        self,
        fmt: WireFormat = _DEFAULT_FORMAT,
        *,
        certs: CertFiles,
    ) -> None:
        """Store the components without starting them.

        Args:
            fmt: The wire format the recorder answers in. The bridge's
                adapters honour ``provider_config["base_url"]``; ``fmt`` does
                not enter the adapter's choice. It controls the recorder's
                ``default_format`` so the recorder's reply matches the
                inbound protocol. The default — Anthropic Messages — is the
                one the fixture suite exercises end to end.
            certs: The session's throwaway TLS certificates. The proxy is a
                TLS server and must present one; the recorder binds plain HTTP
                and the bridge's ``aiohttp_trusts_test_ca`` fixture is *not*
                needed (the bridge reaches the recorder over plain HTTP, with
                only the client↔proxy hop TLS-wrapped). The seam is taken in
                here rather than from a pytest fixture because the harness is
                not a fixture itself — a test asks for ``SealedNetwork``
                explicitly.
        """
        self._fmt = fmt
        self._certs = certs
        self._proxy: ConnectProxy | None = None
        self._recorder: RecordingUpstream | None = None

    async def start(self) -> None:
        """Start the recorder and the proxy, in that order.

        The recorder must be listening before the proxy can resolve requests
        to it; the order is therefore fixed and recorded here, not at the call
        site, so sibling slices do not see a half-started harness.

        A failed proxy start releases the recorder — ``__aenter__`` propagates
        the exception and ``__aexit__`` is never called on the failed entry,
        so without this guard the recorder would hold its port for the rest of
        the session. ``BridgeFixture.start`` carries the same guard for the
        same reason.

        Raises:
            OSError: When the kernel refuses a port — a transient I/O error
                the caller may choose to retry. ``pytest.fail`` is *not* the
                right shape here; a connect-time failure of the local servers
                is the same family as a connect-time failure of any other
                network resource.
        """
        # Recorder first: the proxy's resolver map needs its port.
        recorder = RecordingUpstream(default_format=self._fmt)
        await recorder.start()
        self._recorder = recorder

        try:
            # Proxy second, with the resolve map keyed on the recorder's port.
            target = f"{HARNESS_UPSTREAM_HOST}:{self._recorder.port}"
            proxy = ConnectProxy(resolve={target: ("127.0.0.1", self._recorder.port)})
            await proxy.start(server_ssl_context(self._certs.proxy_cert, self._certs.proxy_key))
        except BaseException:
            await recorder.stop()
            self._recorder = None
            raise

        # Assigned only on success, so a failed start leaves both handles
        # ``None`` and a subsequent :meth:`stop` is a no-op rather than a
        # double-close.
        self._proxy = proxy

    async def stop(self) -> None:
        """Stop the proxy and the recorder, in reverse start order.

        Idempotent: a test that stops the proxy mid-run is still followed by
        the harness's teardown, and the seam T-W5 ships gives the proxy's
        ``stop()`` the same property.

        The recorder stops even if the proxy's stop raised — ``ConnectProxy.stop``
        can ``raise TimeoutError`` when its drain deadline (5 s) expires on a
        stuck client, and a teardown that stops there would leak the recorder.
        The recorder's stop is unconditional; the proxy's error is re-raised
        to the caller.
        """
        proxy, self._proxy = self._proxy, None
        recorder, self._recorder = self._recorder, None

        try:
            if proxy is not None:
                await proxy.stop()
        finally:
            if recorder is not None:
                await recorder.stop()

    async def __aenter__(self) -> SealedNetwork:
        """Start the pair on entry."""
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stop the pair on exit, regardless of whether the body raised.

        Args:
            *exc: The exception triple, unused; the ports are released either way.
        """
        await self.stop()

    @property
    def upstream_host(self) -> str:
        """Return the hostname the upstream is addressed by.

        Returns:
            :data:`harness.connect_proxy.HARNESS_UPSTREAM_HOST`, fixed.
        """
        return HARNESS_UPSTREAM_HOST

    @property
    def upstream_port(self) -> int:
        """Return the kernel-chosen loopback port the recorder bound.

        Returns:
            The recorder's port. ``0`` until :meth:`start` has run.

        Raises:
            RuntimeError: When the harness has not been started. The hint is
                what surfaces the half-started state -- the kernel does not
                mind a port-0 query, but the property's contract is "after
                start".
        """
        if self._recorder is None:
            raise RuntimeError("sealed network is not running; call start() first")
        return self._recorder.port

    @property
    def upstream_base_url(self) -> str:
        """Return the URL the bridge's adapter points at.

        Returns:
            ``http://HARNESS_UPSTREAM_HOST:{port}``. Plain HTTP at the
            upstream is a §5.3-shaped decision: the proxy terminates the
            TLS hop, and the upstream hop is the only place a recorder can
            observe the bridge's outbound socket peer port without a TLS
            handshake gate.
        """
        return f"http://{self.upstream_host}:{self.upstream_port}"

    @property
    def proxy_url(self) -> str:
        """Return the URL the bridge's proxy kwargs point at.

        Returns:
            ``https://127.0.0.1:{port}``. The proxy hop is the only TLS leg;
            T-W5 ships the local TLS server. The ``https`` scheme matters for
            the egress contract: aiohttp's ``_create_proxy_connection`` takes
            the scheme from the proxy URL, not the upstream URL.
        """
        if self._proxy is None:
            raise RuntimeError("sealed network is not running; call start() first")
        return f"https://127.0.0.1:{self._proxy.port}"

    @property
    def proxy(self) -> ConnectProxy:
        """Return the proxy the harness started.

        Returns:
            The proxy. Sibling slices record attempts; T-E2's phase 2 stops
            this object mid-test.
        """
        if self._proxy is None:
            raise RuntimeError("sealed network is not running; call start() first")
        return self._proxy

    @property
    def recorder(self) -> RecordingUpstream:
        """Return the upstream recorder the harness started.

        Returns:
            The recorder. Sibling slices and the green-path assertion read
            captures and connections off this object.
        """
        if self._recorder is None:
            raise RuntimeError("sealed network is not running; call start() first")
        return self._recorder


# ── The containment transport extension interface (R4) ─────────────────────


@dataclass(frozen=True)
class Phase1Result:
    """What a bridge-aiohttp phase 1 drive observes, in one struct.

    Attributes:
        status: The HTTP status the test client received from the bridge.
            A negative value means the bridge did not answer — the drive
            caught a client-side error (``OSError`` or ``asyncio.TimeoutError``)
            rather than an HTTP response.
        text: The raw response body, or the exception repr when ``status`` is
            negative. Text, not JSON: three of the four inbound surfaces
            answer with SSE.
        captures: The recorder's request captures after one request.
        connections: The recorder's connection log, including the request
            that carried the capture and any earlier probe the bridge made.
        attempts: The proxy's CONNECT attempts. Phase 1 (egress off) must
            leave this empty.
    """

    status: int
    text: str
    captures: Sequence[CapturedRequest]
    connections: Sequence[ConnectionRecord]
    attempts: Sequence[ConnectAttempt]


@runtime_checkable
class ContainmentTransport(Protocol):
    """What a containment route must provide to plug into the harness core.

    T-E3..T-E5 satisfy this and register themselves via
    :func:`register_containment_transport`; nothing in this module changes for
    them. The interface is the same shape T-W8's ``UpstreamTransport`` takes:
    attributes for identity, async methods for lifecycle and one drive entry
    point for the phase-1 positive control T-E1 demonstrates here.

    Attributes:
        name: The registry key, e.g. ``"curl_cffi"``.
    """

    name: str

    def direct_route(self, harness: SealedNetwork) -> contextlib.AbstractContextManager[None]:
        """Return a context manager that puts the **direct**-leg override in scope.

        The bridge-aiohttp default is a no-op ``yield`` — its own direct-route
        mechanism (``monkeypatched_aiohttp_resolver``) is applied inside
        :meth:`drive_phase_1` because the patch is via ``socket.getaddrinfo``
        and lives on the harness's :func:`monkeypatch` fixture. T-E3..T-E5
        override this to apply **their** per-transport direct-route overrides:
        curl_cffi's ``--resolve`` mapping, botocore's ``endpoint_url``, the
        provider-aiohttp session's resolver hook. ``drive_phase_1`` enters
        ``self.direct_route(harness)`` so the override is in scope for the
        drive's outbound call.

        The return type is a context manager, not the bare iterator a
        ``@contextmanager``-decorated function yields: every natural
        implementation is ``@contextlib.contextmanager``-decorated, and that
        decorator's return type is a ``_GeneratorContextManager``, not the
        generator itself. Declaring ``Iterator[None]`` here would make the
        decorator shape a mypy error in every sibling that follows.

        Args:
            harness: The sealed network the request will be driven against.

        Returns:
            The context manager to enter around the drive.
        """

    async def drive_phase_1(
        self,
        harness: SealedNetwork,
        *,
        monkeypatch: pytest.MonkeyPatch,
        resolver_port: int | None = None,
    ) -> Phase1Result:
        """Drive one request through the bridge, with egress off, observing the harness.

        Args:
            harness: The sealed network the request is driven against.
            monkeypatch: Pytest's monkeypatch fixture; the transport's
                ``direct_route`` patch reverts on its teardown.
            resolver_port: The port the **direct**-leg resolver maps the
                harness hostname to. Default is ``harness.upstream_port`` —
                the recorder's port — so the bridge reaches the harness.
                The falsification case uses a port with no listener to
                demonstrate the harness is not vacuous.

        Returns:
            A :class:`Phase1Result` with the recorder's observations. The
            caller is responsible for asserting on them.
        """


#: Builds a containment transport. T-E3..T-E5 supply their own factories;
#: this module ships the bridge-aiohttp one as the default registration.
ContainmentFactory = Callable[[], ContainmentTransport]

_CONTAINMENT_REGISTRY: dict[str, ContainmentFactory] = {}


def register_containment_transport(name: str, cls: type[ContainmentTransport]) -> None:
    """Register a containment transport under ``name``.

    Args:
        name: The registry key, e.g. ``"curl_cffi"``.
        cls: A class implementing :class:`ContainmentTransport`. Instances are
            built per drive.

    Raises:
        ValueError: When ``name`` is already registered. Silently replacing
            would let two slices disagree about which transport a name means.
    """
    if name in _CONTAINMENT_REGISTRY:
        registered = ", ".join(sorted(_CONTAINMENT_REGISTRY))
        raise ValueError(f"containment transport {name!r} is already registered; registered: {registered}")
    _CONTAINMENT_REGISTRY[name] = cls


def get_containment_transport(name: str) -> ContainmentTransport:
    """Build the registered containment transport for ``name``.

    Args:
        name: The registry key.

    Returns:
        A fresh :class:`ContainmentTransport` instance.

    Raises:
        LookupError: When ``name`` is not registered. The Python built-in
            ``LookupError`` is used so callers need not import a custom
            exception class to be told so.
    """
    try:
        factory = _CONTAINMENT_REGISTRY[name]
    except KeyError:
        registered = ", ".join(sorted(_CONTAINMENT_REGISTRY))
        raise LookupError(f"containment transport {name!r} is not registered; registered: {registered}") from None
    return factory()


def registered_containment_transports() -> tuple[str, ...]:
    """Return every registered containment transport name, sorted.

    Returns:
        The names registered in this process. Test files iterating this set
        should import their subjects explicitly, otherwise the contents
        reflect which modules the runner happened to import.
    """
    return tuple(sorted(_CONTAINMENT_REGISTRY))


class BridgeAiohttpContainment(ContainmentTransport):
    """The default containment transport: the bridge's own aiohttp serving path.

    Drives one request through a real :class:`~kitty.bridge.server.BridgeServer`
    pointed at the harness's recording upstream, with the monkeypatched
    resolver in scope and egress disabled. Constructed per drive; no state
    lives on the instance.

    Attributes:
        name: The registry key, ``"bridge_aiohttp"``.
    """

    #: A class attribute, ``"bridge_aiohttp"``. Matches the
    #: ``UpstreamTransport``-style naming pattern ``tests/harness/bridge.py``
    #: uses for its protocol members.
    name = "bridge_aiohttp"

    #: Same Anthropic-Messages body shape the bridge fixture uses. The body
    #: is intentionally trivial so the bridge's reply path — and only the
    #: reply path — is exercised; the test asserts the recorder observed the
    #: request, not that the bridge parsed an interesting body.
    _MODEL: ClassVar[str] = "harness-model"

    @contextlib.contextmanager
    def direct_route(self, harness: SealedNetwork) -> Iterator[None]:
        """Yield once — the bridge-aiohttp direct route has no override to apply.

        The bridge-aiohttp mechanism (``monkeypatched_aiohttp_resolver``) is
        applied inside :meth:`drive_phase_1` so the ``resolver_port``
        falsification seam lives on the method that uses it. T-E3..T-E5
        override this instead: curl_cffi's ``--resolve`` mapping, botocore's
        ``endpoint_url``, the provider-aiohttp session's resolver hook.

        Args:
            harness: The sealed network the request will be driven against.

        Yields:
            ``None``.
        """
        yield

    async def drive_phase_1(
        self,
        harness: SealedNetwork,
        *,
        monkeypatch: pytest.MonkeyPatch,
        resolver_port: int | None = None,
    ) -> Phase1Result:
        """Drive one request through the bridge with egress off (§5.2.2 phase 1).

        Args:
            harness: The sealed network the request is driven against.
            monkeypatch: Pytest's monkeypatch fixture; the resolver patch
                reverts on its teardown.
            resolver_port: The port the direct-leg resolver maps the harness
                hostname to. The default — ``harness.upstream_port`` — sends
                the bridge to the recorder. The falsification case passes a
                closed port; the broken-resolver seam lives here so the test
                file does not have to wrap the call in
                :func:`monkeypatched_aiohttp_resolver`.

        Returns:
            A :class:`Phase1Result` with the bridge's status, the recorder's
            captures and connections after the request, and the proxy's
            attempts. With egress off and the default resolver port,
            ``status == 200``, ``len(captures) == 1``, ``attempts == []``.
        """
        # Local import keeps the module-import surface tidy: the bridge
        # server depends on `kitty.providers.*`, which the rest of the test
        # module surface (e.g., the report) does not need.
        from kitty.bridge.server import BridgeServer
        from kitty.providers.custom_anthropic import CustomAnthropicAdapter

        target_port = harness.upstream_port if resolver_port is None else resolver_port

        # The per-transport direct route in scope around the drive, then the
        # harness's resolver mapping. The direct route is the override point
        # T-E3..E5 use; the resolver mapping is the bridge-aiohttp mechanism.
        with (
            self.direct_route(harness),
            # Scoped so a test that drives more than one request inside the
            # same monkeypatch fixture does not leave a stale mapping. The
            # teardown goes through ``monkeypatch``.
            monkeypatched_aiohttp_resolver(monkeypatch, harness.upstream_host, target_port),
        ):
            # `BridgeServer` is constructed directly rather than through
            # `BridgeFixture`: `AiohttpTransport.bind()` returns
            # `{"base_url": self._recorder.base_url}` — the loopback URL —
            # and §5.3's whole point is that the bridge reaches the
            # harness **by its non-loopback name**, not by the recorder's
            # own. Bypassing the fixture here is intentional, not lazy.
            adapter = CustomAnthropicAdapter()
            server = BridgeServer(
                None,  # type: ignore[arg-type]
                adapter,
                resolved_key="harness-key",
                model=self._MODEL,
                provider_config={"base_url": harness.upstream_base_url},
            )
            status: int = -1
            text: str = ""
            try:
                bridge_port = await server.start_async()

                # The body mirrors what `BridgeFixture.minimal_inbound_body`
                # builds for Messages, inlined to keep this module's import
                # surface minimal.
                body = {
                    "model": self._MODEL,
                    "messages": [{"role": "user", "content": f"kbr61-{uuid.uuid4().hex}"}],
                    "max_tokens": 16,
                    "stream": False,
                }

                async with aiohttp.ClientSession() as client:
                    response = await client.post(
                        f"http://127.0.0.1:{bridge_port}/v1/messages",
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=_DRIVE_TIMEOUT),
                    )
                    text = await response.text()
                    status = response.status
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                # The bridge either timed out or failed its own outbound
                # connect — both shapes the test client's ``status`` field
                # cannot represent. Carry the exception in ``text`` so a
                # failing drive is diagnosable from its result alone.
                status = -1
                text = repr(exc)
            finally:
                await server.stop_async()

            return Phase1Result(
                status=status,
                text=text,
                captures=list(harness.recorder.requests),
                connections=list(harness.recorder.connections),
                attempts=list(harness.proxy.attempts),
            )


# ── Default registration ──────────────────────────────────────────────────


register_containment_transport(BridgeAiohttpContainment.name, BridgeAiohttpContainment)
