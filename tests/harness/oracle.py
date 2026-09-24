"""The transparency oracle — makes Invariant I1 (message fidelity) testable.

`.system_design/TEST_SUITE.md` §3.3, §3.3.1, §3.3.2, §3.3.4, §4.3 C2, §7.4,
§10 · plan tasks **T-D1** (KBR-51) and **T-D2** (KBR-52).

§3.3 of the test-suite design names the oracle as "the single piece of new
infrastructure that makes I1 testable". Two assertions run on it; together they
are the test of I1:

* **§3.3.2 assertion 1** — every difference between the inbound projection and
  the captured upstream projection maps to a register row whose trigger the
  input met.
* **§3.3.2 assertion 2** — for each conditional row, an input that does not
  meet the trigger must show that row's mutation *absent*.

§4.3 C2 adds a third obligation the oracle owns: the **byte-level key-order
assertion** on the native passthrough path. JSON key order is exactly what a
provider fingerprints, so where kitty claims to be forwarding rather than
translating, key order is part of the contract (§3.3.5 names the route as the
property that decides this).

**§3.3.5 adds the fourth: the routing assertion (T-D2, KBR-52).** On Azure the
deployment id lives in the URL (P20) while P6 removes ``model`` from the body,
so two requests to two different deployments have byte-identical bodies — a
body-only comparison cannot tell them apart. When the caller supplies an
``expected_route`` — derived *independently*, from the profile and the
provider's published URL shape, never by calling ``build_base_url()`` /
``get_upstream_path()`` — every route component of the captured request must
match it. The authority-and-scheme normalisation is the caller's obligation
(§3.3.5: the harness serves ``http://127.0.0.1:<ephemeral>`` where a published
shape is ``https://`` on the provider's host, so the expectation is rewritten
with the recorder's own authority before comparison); path and query are
compared exactly as derived. With ``expected_route=None`` the assertion is
absent, not vacuous — T-D1's callers are unchanged.

**The oracle imports nothing from ``src/kitty``, and must not.** §3.3.1's
independent-oracle rule: a projection that asked kitty how to read a body
would inherit kitty's bugs. The oracle's readers, the contract, the register,
the corpus and the bridge fixture all live in :mod:`tests.harness`, and the
oracle reads only those.

**The signature matches §7.4.** ``inbound_format`` and ``captured_format``
are supplied by the harness caller from the *observed* wire shape (§3.3.4) —
not from a caller-supplied declaration from the adapter, and not derived
inside the oracle. The oracle's job is to use the selection, not to make it.

**The route signal is the trigger vocabulary, not the captured bytes.**
Native passthrough means :attr:`Trigger.NON_NATIVE_UPSTREAM_WIRE` is **not**
in ``triggers_met``. Under that condition the captured body's JSON key
order must equal the inbound body's, at every object level (§4.3 C2) —
key order, not bytes: the native branch rewrites ``model`` through M1,
a registered mutation, so a byte check would fail a body the design calls
correct. Using a non-existent ``NATIVE_UPSTREAM_WIRE`` enum member would
silently miss every native route — this is a known trip and is enforced
by the assertion's own wording.

**The first working version ships with one falsification case (plan §1.4).**
A mutated ``envelope.model`` with ``PROFILE_SETS_MODEL`` deliberately omitted
from ``triggers_met``, so M1 (whose pattern is ``envelope.model`` and whose
trigger is that trigger) does not claim the delta. The delta is unclaimed,
assertion 1 fails, and the oracle names ``envelope.model``. This proves the
diff *sees* the model field — §10 names "a projection that could not see the
model name" as one of the four harnesses that would have passed while
proving nothing. §9.2 G21 records the symmetric lever: a corpus entry that
*over*-declares makes assertion 1 claim every delta; this falsification uses
the same lever from the *under*-declaring direction.

**Layer.** No ``pytestmark``; tests default to ``l1`` per ``tests/layers.py``
and ``tests/harness/test_vertical_slice.py``'s precedent. ``l3`` activation
is T-K6's job.

**Response direction — KBR-59 (T-D10).** §3.3.1's last paragraph makes
response translation "a different claim [that] gets a different test",
and the plan splits the work as six request tasks against one reply
task (T-A7) with its own comparison task (T-D10). This module carries
its sibling entry point
:func:`assert_no_unclaimed_reply_mutation` (T-D10) — same §3.3.2
assertions, **omitting** §4.3 C2 (a reply body is the upstream's
serialised output, never a forwarded pass-through) and §3.3.5 (the
reply traverses the same connection the request did; routing is a
request-side concern). The reply-twin keeps the request-side
:func:`_run_assertions` and its KBR-307 ``provider_key`` plumbing
intact, parameterised by a ``diff`` callable so the request-side
diff (``_structural_diff``, hard-typed on :class:`Request`) and the
reply-side diff (``_structural_reply_diff``, hard-typed on
:class:`Reply`) share one assertion engine.

**Reply-side falsification case (§1.4).** A captured reply with an
injected, unrecognised top-level field (the response-direction
analogue of §3.3.1's 5th row) fails the run at the totality gate with
:class:`~harness.contract.ResidualFieldsError` — unmodified, no
wrapper. ``verify_total`` on a :class:`Reply` projection enforces it;
the test is the threshold check, not a new mechanism.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from harness import contract as c
from harness import register as r
from harness.contract import (
    CapturedReply,
    CapturedRequest,
    Conversation,
    Projection,
    Reply,
    ReplyProjection,
    Request,
    WireFormat,
    _redact_query,
    verify_total,
)

# --------------------------------------------------------------------------
# Public exception
# --------------------------------------------------------------------------


class OracleError(AssertionError):
    """An I1 breach detected by the oracle.

    Derives from ``AssertionError`` because it reports a failed check rather
    than a broken program: it reads as a test failure, not as a crash. The
    exception's :attr:`paths` lists the concrete delta paths the oracle found
    that violate the assertions; subclasses carry the additional context each
    failure mode requires.

    Attributes:
        paths: The concrete delta paths this failure names.
    """

    def __init__(self, message: str, *, paths: tuple[str, ...] = ()) -> None:
        """Store the paths alongside the assertion message.

        Args:
            message: The human-readable failure text, as ``AssertionError``.
            paths: The concrete delta paths this failure names.
        """
        super().__init__(message)
        self.paths = paths


class UnclaimedMutationError(OracleError):
    """§3.3.2 assertion 1 failed — at least one delta was unclaimed.

    Attributes:
        paths: The concrete delta paths no register row (whose trigger the
            input met) matched.
    """


class ConditionalRowFiredWithoutTriggerError(OracleError):
    """§3.3.2 assertion 2 failed — a conditional row's mutation appeared
    under a trigger that was not met.

    Attributes:
        paths: The concrete delta paths each violation landed at.
        row_id: The conditional register row id that fired without its trigger.
    """

    def __init__(
        self, message: str, *, paths: tuple[str, ...] = (), row_id: str = ""
    ) -> None:
        """Store the row id alongside the paths.

        Args:
            message: The human-readable failure text, as ``AssertionError``.
            paths: The concrete delta paths this failure names.
            row_id: The conditional register row id that fired.
        """
        super().__init__(message, paths=paths)
        self.row_id = row_id


class NativePassthroughKeyOrderError(OracleError):
    """§4.3 C2 failed — a native passthrough route's captured body reorders
    JSON keys relative to the inbound body.

    Attributes:
        paths: The concrete delta paths this failure names.
        inbound_preview: The first 200 bytes of the inbound body, for diff.
        captured_preview: The first 200 bytes of the captured body, for diff.
    """

    def __init__(
        self,
        message: str,
        *,
        paths: tuple[str, ...] = (),
        inbound_preview: bytes = b"",
        captured_preview: bytes = b"",
    ) -> None:
        """Store the two body previews.

        Args:
            message: The human-readable failure text, as ``AssertionError``.
            paths: The concrete delta paths this failure names.
            inbound_preview: The first 200 bytes of the inbound body.
            captured_preview: The first 200 bytes of the captured body.
        """
        super().__init__(message, paths=paths)
        self.inbound_preview = inbound_preview
        self.captured_preview = captured_preview


class RoutingMismatchError(OracleError):
    """§3.3.5 routing assertion failed — the captured request went elsewhere.

    Raised when an ``expected_route`` was supplied and one or more of its
    components (``method``, ``scheme``, ``host``, ``path``, ``query``) do not
    match the captured request. Path and query are where the Azure case
    lives; scheme and host are part of the comparison only after the caller's
    mandatory authority rewrite (§3.3.5's T-W4 scope addition) has
    substituted the recorder's own values into the expectation.

    Attributes:
        paths: The ``route.<component>`` paths that failed — spelled with
            :data:`tests.harness.contract.ROUTE_COMPONENTS` vocabulary so a
            future reader sees the same failure shape the register rows use.
    """

    def __init__(self, message: str, *, paths: tuple[str, ...] = ()) -> None:
        """Store the failing component paths alongside the message.

        Args:
            message: The human-readable failure text, naming every component
                that failed and the two values that disagreed.
            paths: The ``route.<component>`` paths that failed.
        """
        super().__init__(message, paths=paths)


# --------------------------------------------------------------------------
# Routing expectation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedRoute:
    """The §3.3.5 routing expectation a caller derives independently.

    The transparency oracle consumes the *route* alongside the body: on Azure
    the deployment id lives in the URL while P6 removes ``model`` from the
    body, so two requests to two different deployments have byte-identical
    bodies and a body-only comparison cannot tell them apart. The caller
    derives this value from the configured profile — provider, model,
    ``provider_config`` — using the provider's published URL shape, never by
    calling ``build_base_url()`` / ``get_upstream_path()``.

    **The authority rewrite is the caller's obligation, not this class's.**
    §3.3.5's T-W4 scope addition: the harness serves ``http`` on an ephemeral
    loopback port where a published shape is ``https`` on the provider's own
    hostname, so the expectation's scheme and host must already carry the
    recorder's values when they arrive here. Path and query are compared
    exactly as derived.

    Attributes:
        method: The HTTP method, ``"POST"`` for every bridge route today.
        scheme: The recorder's scheme, after the caller's rewrite.
        host: The recorder's host, port included, after the caller's rewrite.
        path: The route path component, asserted exactly.
        query: The route query component, asserted exactly.
    """

    method: str
    scheme: str
    host: str
    path: str
    query: str


# --------------------------------------------------------------------------
# Reader registry
# --------------------------------------------------------------------------


#: The reader registry, populated from the six landed readers. Importing this
#: module before the readers would leave the registry incomplete; the
#: :data:`_REGISTRY_GUARD` raises at import time if any :class:`WireFormat`
#: member is missing.
_REQUEST_PROJECTIONS: dict[WireFormat, Projection] = {}


def _register_projection(projection: Projection) -> None:
    """Insert one ``Projection`` into :data:`_REQUEST_PROJECTIONS`.

    Called once per reader at module import time. Re-registration is silent
    (no-op) so a test fixture that re-imports a reader does not warn.

    Args:
        projection: An instance of any class implementing
            :class:`~harness.contract.Projection`.
    """
    _REQUEST_PROJECTIONS[projection.wire_format] = projection


def _reader_for(fmt: WireFormat) -> Projection:
    """Return the registered reader for ``fmt``.

    Args:
        fmt: The wire format to read.

    Returns:
        The registered :class:`Projection`.

    Raises:
        RuntimeError: When no reader is registered for ``fmt`` — a missing
            reader is caught at module import time (see ``_REGISTRY_GUARD``),
            so reaching this branch indicates the caller supplied a format
            the suite does not yet cover. The exception type matches the
            guard's so a downstream caller that catches one catches both.
    """
    try:
        return _REQUEST_PROJECTIONS[fmt]
    except KeyError as exc:
        raise RuntimeError(
            f"harness/oracle.py: no Projection registered for {fmt!r}; "
            "every WireFormat member must have a reader — _REGISTRY_GUARD "
            "should have caught this at import time"
        ) from exc


def _REGISTRY_GUARD() -> None:
    """Assert every :class:`WireFormat` member has a registered reader.

    Called from the bottom of this module so a missing reader fails at
    import time, not at oracle call time. A reader that lands in a separate
    PR adds itself here by calling :func:`_register_projection` from its
    own module — a single source of truth that the missing-reader shape
    never silently passes.

    Raises:
        RuntimeError: When any :class:`WireFormat` member has no reader.
    """
    missing = [fmt for fmt in WireFormat if fmt not in _REQUEST_PROJECTIONS]
    if missing:
        raise RuntimeError(
            f"harness/oracle.py: missing Projection for {missing}; "
            "the landed reader must call _register_projection at import time"
        )


# --------------------------------------------------------------------------
# Reply projection registry (KBR-59 / T-D10)
# --------------------------------------------------------------------------

#: The set of wire formats the reply-direction oracle asserts are
#: *reachable* today. Five formats have a ``read_reply`` today; an
#: additional reply reader (KBR-312 lands Bedrock Converse) grows
#: this set in one place, and the import-time guard
#: :func:`_REPLY_REGISTRY_GUARD` catches "constant grew but reader not
#: yet landed" the first time pytest runs — the load-bearing case
#: (reviewer finding S8). The Bedrock Converse format is *not* in this
#: set until the reply reader lands.
_ASSERTABLE_REPLY_FORMATS: frozenset[WireFormat] = frozenset({
    WireFormat.ANTHROPIC_MESSAGES,
    WireFormat.CHAT_COMPLETIONS,
    WireFormat.GEMINI,
    WireFormat.OLLAMA_CHAT,
    WireFormat.OPENAI_RESPONSES,
})

#: Per-format reply projection registry, parallel to
#: :data:`_REQUEST_PROJECTIONS`. Central registration, mirroring
#: :func:`_register_projection`'s import-time call site.
_REPLY_PROJECTIONS: dict[WireFormat, ReplyProjection] = {}


def _register_reply_projection(projection: ReplyProjection) -> None:
    """Insert one :class:`~harness.contract.ReplyProjection` into
    :data:`_REPLY_PROJECTIONS`.

    Called once per reply reader at module import time. Re-registration
    is silent (no-op) so a test fixture that re-imports a reader does
    not warn. Mirrors :func:`_register_projection` so the request and
    reply registries use one shape.

    Args:
        projection: An instance of any class implementing
            :class:`~harness.contract.ReplyProjection`.
    """
    _REPLY_PROJECTIONS[projection.wire_format] = projection


def _reply_reader_for(fmt: WireFormat) -> ReplyProjection:
    """Return the registered reply reader for ``fmt``.

    Args:
        fmt: The wire format to read.

    Returns:
        The registered :class:`~harness.contract.ReplyProjection`.

    Raises:
        RuntimeError: When no reply reader is registered for ``fmt`` —
            either the caller supplied a format the suite does not yet
            cover, or the floor constant :data:`_ASSERTABLE_REPLY_FORMATS`
            grew without a reader landing (caught earlier at import time
            by :func:`_REPLY_REGISTRY_GUARD`, so reaching this branch
            indicates the caller reached past the guard).
    """
    try:
        return _REPLY_PROJECTIONS[fmt]
    except KeyError as exc:
        raise RuntimeError(
            f"harness/oracle.py: no ReplyProjection registered for {fmt!r}; "
            "every format in _ASSERTABLE_REPLY_FORMATS must have a reader — "
            "_REPLY_REGISTRY_GUARD should have caught this at import time"
        ) from exc


def _REPLY_REGISTRY_GUARD() -> None:
    """Assert :data:`_ASSERTABLE_REPLY_FORMATS` is a subset of the
    registered reply projections.

    The check is **superset**, not exact: an additional reply reader
    (e.g. Bedrock Converse via KBR-312) is welcome to grow the
    registry past the floor — what the guard catches is the converse,
    "the constant grew without a reader landing". The reviewer
    suggestion S8 named this as the load-bearing failure case to pin.

    Raises:
        RuntimeError: When a format is in :data:`_ASSERTABLE_REPLY_FORMATS`
            but no reader is registered for it.
    """
    missing = [
        fmt
        for fmt in _ASSERTABLE_REPLY_FORMATS
        if fmt not in _REPLY_PROJECTIONS
    ]
    if missing:
        raise RuntimeError(
            f"harness/oracle.py: _ASSERTABLE_REPLY_FORMATS grew but no "
            f"reply reader is registered for {missing}; the floor "
            "constant and the reader registration must move together"
        )


# --------------------------------------------------------------------------
# Public report object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleReport:
    """A small report the oracle attaches to the end of a successful run.

    :attr:`expected_route` records the routing expectation the caller
    supplied — since T-D2 (KBR-52) an expectation that reached the report has
    also been **asserted**: every component matched the captured request, or
    :class:`RoutingMismatchError` would have raised first. The report also
    records the projections and the diff so a future T-I8 cross-attempt
    comparison can read it without re-running the bridge.

    Attributes:
        expected_route: The caller-supplied routing expectation, or ``None``
            when the caller ran the body assertions alone.
        inbound_projection: The wire-independent form of the inbound capture.
        captured_projection: The wire-independent form of the captured body.
        deltas: The concrete delta paths the structural diff found, in walk
            order. Empty on a run with byte-identical projections.
    """

    expected_route: ExpectedRoute | None
    inbound_projection: Request
    captured_projection: Request
    deltas: tuple[str, ...]


# --------------------------------------------------------------------------
# Reply-direction report object (KBR-59 / T-D10)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplyOracleReport:
    """A small report the reply-direction oracle attaches to a successful run.

    Mirrors :class:`OracleReport` but carries :class:`Reply` projections
    rather than :class:`Request`. The two reports are separate dataclasses
    rather than one type with a ``kind`` discriminator — the call-site
    reviewer is told at the type level whether the comparison ran on the
    request or the reply side, and T-I8 (cross-attempt comparison) reads
    both as separate records. KBR-59.

    ``__hash__`` is set to ``None`` explicitly so a future
    set-or-dict use fails loud rather than inheriting-by-luck from
    dataclass auto-hash (the :class:`Reply` projection's parts are
    themselves unhashable, so a derived hash would ``TypeError`` at
    first use — making the limitation a property of the type, not an
    accident of attribute composition; reviewer suggestion S3).

    Attributes:
        inbound_projection: The wire-independent form of the inbound
            reply as captured on the agent side.
        captured_projection: The wire-independent form of the captured
            reply as observed on the upstream.
        deltas: The concrete delta paths the structural diff found, in
            walk order. Empty on a run with byte-identical projections.
    """

    inbound_projection: Reply
    captured_projection: Reply
    deltas: tuple[str, ...]

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Public oracle
# --------------------------------------------------------------------------


def assert_no_unclaimed_mutation(
    inbound: CapturedRequest,
    inbound_format: WireFormat,
    captured: CapturedRequest,
    captured_format: WireFormat,
    register: tuple[r.MutationRow, ...],
    triggers_met: frozenset[r.Trigger],
    *,
    expected_route: ExpectedRoute | None = None,
    provider_key: str = r.ALL_PROVIDERS,
) -> OracleReport:
    """Assert that every difference between the two projections is registered.

    Runs the four obligations the oracle owns, in order:

    1. **Totality gate** (§3.3.1, §7.4). Both projections pass through
       :func:`~harness.contract.verify_total`. A non-empty residual or a
       dropped field fails the run, naming the field. The structural diff is
       skipped — a key that residualises identically on both sides would
       otherwise produce zero deltas and the run would go green on a defect
       §3.3.1 calls out by name.
    2. **§3.3.2 assertion 1 + assertion 2** (§3.3.2). Every concrete delta
       path is classified against the register rows whose trigger is in
       ``triggers_met`` **and** whose site is reachable on the adapter
       ``provider_key`` names (KBR-307). Unclaimed deltas fail assertion
       1. Conditional rows whose trigger is *not* in ``triggers_met`` but
       whose anchor has an unclaimed delta fail assertion 2 — phrased so
       a coarser-anchor row (e.g. M3) does not false-fail when an M5 drop
       beneath it is legitimately claimed by M5's own trigger.
    3. **§4.3 C2 native passthrough key-order assertion.** When
       :attr:`~harness.register.Trigger.NON_NATIVE_UPSTREAM_WIRE` is *not* in
       ``triggers_met``, the captured body must equal the inbound body
       byte-for-byte. (The native route is the *absence* of the
       non-native trigger — there is no ``NATIVE_UPSTREAM_WIRE`` enum
       member, and using one would silently miss every native route.)
    4. **§3.3.5 routing assertion (T-D2, KBR-52).** When ``expected_route``
       is supplied, every route component of the captured request —
       ``method``, ``scheme``, ``host``, ``path``, ``query`` — must equal
       it. The component set is driven from
       :data:`~harness.contract.ROUTE_COMPONENTS` so it cannot drift, and
       any disagreement raises :class:`RoutingMismatchError` with the
       failing ``route.<component>`` paths. ``expected_route=None`` is
       the absence of the assertion, not a vacuous pass — T-D1's callers
       are unchanged. The routing falsification: a capture rerouted to
       another deployment with a byte-identical body reaches this step
       with every body obligation passed, and fails on routing and only
       on routing.

    Args:
        inbound: The agent's inbound request as observed on the wire.
        inbound_format: The format of ``inbound``, supplied by the harness
            from the observed wire shape (§3.3.4).
        captured: The request that reached the upstream, as observed on
            the recording upstream.
        captured_format: The format of ``captured``, supplied by the
            harness from the recorder's declared format.
        register: The Permitted-Mutation Register rows the oracle
            classifies against. Typically :data:`~harness.register.REGISTER`.
        triggers_met: The trigger vocabulary the input met. §9.2 G21: a
            corpus entry that over-declares here makes assertion 1 claim
            every delta; under-declaring is the symmetric lever and is the
            one T-D1's falsification uses.
        expected_route: The independently derived routing expectation
            (§3.3.5). When supplied, every component of the captured
            request's route must match it — method, scheme, host, path and
            query — after the caller has rewritten the expectation's
            authority with the recorder's own. ``None`` (every T-D1 caller)
            leaves the routing assertion out entirely.
        provider_key: The adapter this run is judging — narrows the
            runtime oracle's notion of "live on this adapter" by scoping
            each register row by its :attr:`~harness.register.MutationRow.scope`.
            Defaults to :data:`~harness.register.ALL_PROVIDERS`, the
            permissive sentinel — see the helpers' ``Args:`` for the
            call-site short-circuit. KBR-307.

    Returns:
        An :class:`OracleReport` recording the projections, the deltas, and
        the routing expectation. The report is what T-I8 reads on a
        successful run.

    Raises:
        UnreadableBodyError: When a reader cannot read a body. Bubbles from
            the contract unchanged.
        ProjectionTotalityError: When :func:`verify_total` rejects a
            projection. Bubbles from the contract unchanged.
        UnclaimedMutationError: When §3.3.2 assertion 1 fails. The
            exception's ``paths`` are the unclaimed concrete delta paths.
        ConditionalRowFiredWithoutTriggerError: When §3.3.2 assertion 2
            fails. The exception's ``row_id`` is the conditional row that
            fired without its trigger; ``paths`` are the offending deltas.
        NativePassthroughKeyOrderError: When §4.3 C2 fails. The exception
            surfaces both bodies' first 200 bytes for diff.
        RoutingMismatchError: When §3.3.5's routing assertion fails — an
            ``expected_route`` was supplied and some component of the
            captured request's route disagrees. The exception's ``paths``
            name the failing ``route.<component>`` paths.
    """
    # Step 1 — project both captures through the registered readers.
    inbound_projection = _reader_for(inbound_format).read_request(inbound)
    captured_projection = _reader_for(captured_format).read_request(captured)

    # Step 2 — totality gate. verify_total raises on non-empty residual or
    # dropped field; that raise is the oracle's first failure mode and
    # skips the structural diff.
    verify_total(inbound_projection)
    verify_total(captured_projection)

    # Step 3–5 — diff, claim matching, assertion 1, assertion 2.
    deltas = _run_assertions(
        inbound_projection, captured_projection, register, triggers_met, provider_key=provider_key
    )

    # Step 6 — §4.3 C2 native passthrough byte-level check. Route is the
    # *absence* of NON_NATIVE_UPSTREAM_WIRE in triggers_met.
    if r.Trigger.NON_NATIVE_UPSTREAM_WIRE not in triggers_met:
        _native_passthrough_check(inbound, captured)

    # Step 7 — §3.3.5 routing assertion (T-D2). Runs last so the body
    # failures — the louder diagnosis — surface first, and so the
    # falsification case (a rerouted capture with a byte-identical body)
    # fails on routing and only on routing: the body obligations have
    # already passed by the time this raises.
    if expected_route is not None:
        _routing_check(captured, expected_route)

    return OracleReport(
        expected_route=expected_route,
        inbound_projection=inbound_projection,
        captured_projection=captured_projection,
        deltas=tuple(deltas),
    )


def assert_no_unclaimed_reply_mutation(
    inbound: CapturedReply,
    inbound_format: WireFormat,
    captured: CapturedReply,
    captured_format: WireFormat,
    register: tuple[r.MutationRow, ...],
    triggers_met: frozenset[r.Trigger],
    *,
    provider_key: str = r.ALL_PROVIDERS,
) -> ReplyOracleReport:
    """Assert that every difference between the two reply projections is registered.

    The response-direction twin of :func:`assert_no_unclaimed_mutation`
    (KBR-59 / T-D10). §3.3.1's last paragraph makes response translation
    "a different claim [that] gets a different test", and the obligations
    — and omissions — are explicit.

    Runs three obligations, in order:

    1. **Totality gate** (§3.3.1, §7.4). Both projections pass through
       :func:`~harness.contract.verify_total`. A non-empty residual or a
       dropped field fails the run, naming the field, with the
       ``verify_total`` exception type **unmodified** — the entry point
       does not wrap it (reviewer suggestion S6).
    2. **§3.3.2 assertion 1 + assertion 2.** Threaded through
       :func:`_run_assertions` with :func:`_structural_reply_diff`; the
       shared claim-matching and conditional-violations machinery stays
       path-agnostic and untouched (reviewer decision B2).
    3. **(omitted) §4.3 C2 native-passthrough byte-level check.** The
       reply body is the upstream's serialised output, never a forwarded
       pass-through; key-order preservation is a request-side concern.
    4. **(omitted) §3.3.5 routing assertion.** Routing is part of the
       *request*: the reply traverses the same connection the request
       did, and the request oracle already verifies that. The entry
       point's signature has no ``expected_route`` parameter — a
       compile-time shape pin, not a runtime one.

    Args:
        inbound: The agent-side reply as observed on the wire.
        inbound_format: The format of ``inbound`` — supplied by the
            harness from the observed wire shape (§3.3.4).
        captured: The upstream-side reply as observed on the recording
            upstream (already reassembled from SSE if streamed; that is
            the boundary the reply readers' ``read_reply`` docstring
            carries — KBR-39 / T-A7).
        captured_format: The format of ``captured``.
        register: The Permitted-Mutation Register rows the oracle
            classifies against. Typically :data:`~harness.register.REGISTER`.
        triggers_met: The trigger vocabulary the input met.
        provider_key: The adapter this run is judging — narrows the
            runtime notion of "live on this adapter" via
            :func:`~harness.register.row_is_in_scope`. Defaults to
            :data:`~harness.register.ALL_PROVIDERS`, the permissive
            sentinel — the call-site short-circuit mirrors KBR-307
            verbatim. Keyword-only.

    Returns:
        A :class:`ReplyOracleReport` recording the projections and the
        deltas. The report is what future reply-side cross-attempt
        comparisons (T-I8) read on a successful run.

    Raises:
        UnreadableBodyError: When a reply reader cannot read a body.
            Bubbles from :class:`~harness.contract.ReplyProjection`
            unchanged.
        ProjectionTotalityError: When :func:`~harness.contract.verify_total`
            rejects a projection. Bubbles unchanged; the concrete type
            is :class:`~harness.contract.ResidualFieldsError` or
            :class:`~harness.contract.DroppedFieldsError`, not the
            parent (reviewer finding C8).
        UnclaimedMutationError: When §3.3.2 assertion 1 fails on the
            reply side.
        ConditionalRowFiredWithoutTriggerError: When §3.3.2 assertion 2
            fails on the reply side.
    """
    # Step 1 — project both captures through the registered reply readers.
    inbound_projection = _reply_reader_for(inbound_format).read_reply(inbound)
    captured_projection = _reply_reader_for(captured_format).read_reply(captured)

    # Step 2 — totality gate. verify_total raises unmodified on non-empty
    # residual or dropped field; that raise is the oracle's first
    # failure mode and skips the structural diff.
    verify_total(inbound_projection)
    verify_total(captured_projection)

    # Step 3–4 — diff via the reply twin, claim matching, assertion 1, assertion 2.
    deltas = _run_assertions(
        inbound_projection,
        captured_projection,
        register,
        triggers_met,
        diff=_structural_reply_diff,
        provider_key=provider_key,
    )

    return ReplyOracleReport(
        inbound_projection=inbound_projection,
        captured_projection=captured_projection,
        deltas=tuple(deltas),
    )


def _run_assertions(
    inbound_projection: Request | Reply,
    captured_projection: Request | Reply,
    register: tuple[r.MutationRow, ...],
    triggers_met: frozenset[r.Trigger],
    *,
    diff: Callable[[Any, Any], tuple[str, ...]] | None = None,
    provider_key: str = r.ALL_PROVIDERS,
) -> list[str]:
    """Run §3.3.2 assertions 1 and 2 over two already-projected requests.

    Extracted from :func:`assert_no_unclaimed_mutation` so unit tests can
    drive the assertions against synthesised :class:`Request` objects
    without paying for a body, a reader, or a bridge — the T-W9 precedent
    for attribution: a failing test must name the layer that failed.

    The ``diff`` callable is KBR-59's (T-D10) seam: the request oracle
    keeps ``None`` (resolved to :func:`_structural_diff`, hard-typed on
    :class:`Request`, at call time); the reply-twin passes
    :func:`_structural_reply_diff`, hard-typed on :class:`Reply`. The
    assertion-claiming machinery downstream is path-agnostic and stays
    shared — one engine, two diffs. ``None`` rather than the callable
    itself is the default because :func:`_structural_diff` is defined
    later in this module, so a module-level default value would raise
    :class:`NameError` at import time.

    Args:
        inbound_projection: The inbound request's projection.
        captured_projection: The captured request's projection.
        register: The Permitted-Mutation Register rows.
        triggers_met: The trigger vocabulary the input met.
        diff: The structural diff to drive. ``None`` (every existing
            call site — 25 in ``test_oracle.py``, plus the request
            oracle's private call site) resolves to
            :func:`_structural_diff` at call time, so the
            request-side behaviour is unchanged. KBR-59 / reviewer
            finding C7.
        provider_key: The adapter this run is judging — narrows the
            runtime oracle's notion of "live on this adapter" by scoping
            each row by its :attr:`~harness.register.MutationRow.scope`.
            Defaults to :data:`~harness.register.ALL_PROVIDERS`, the
            permissive sentinel — a caller that has no opinion passes
            through every row; the helper's body is **not** permissive
            when handed the sentinel (the sentinel is a valid value of
            ``row.scope``, not of ``provider_key``), so the helpers'
            default-empty behaviour is provided by an explicit call-site
            short-circuit, not by the helper. KBR-307.

    Returns:
        The concrete delta paths the structural diff found (also the input
        to claim matching), in walk order.

    Raises:
        UnclaimedMutationError: When assertion 1 fails.
        ConditionalRowFiredWithoutTriggerError: When assertion 2 fails.
    """
    # Resolve the diff at call time so the module-level default can be
    # ``None`` (KBR-59 — :func:`_structural_diff` is defined later in
    # this module, so it cannot be the function-arg default).
    if diff is None:
        diff = _structural_diff
    deltas = diff(inbound_projection, captured_projection)

    # Claim matching. For each delta, the rows whose trigger is met,
    # whose pattern matches, and whose site is reachable on the adapter
    # the oracle is judging are the claimers. The provider_key == ALL_PROVIDERS
    # short-circuit is what makes the default permissive: the helper is
    # not itself permissive when handed the sentinel (KBR-307).
    claimers: dict[str, tuple[str, ...]] = _claim_matching(
        deltas, register, triggers_met, provider_key=provider_key
    )
    unclaimed = [path for path in deltas if not claimers[path]]

    if unclaimed:
        raise UnclaimedMutationError(
            f"§3.3.2 assertion 1: {len(unclaimed)} unclaimed delta path(s); "
            f"first: {unclaimed[0]!r}; triggers_met={sorted(t.name for t in triggers_met)}",
            paths=tuple(unclaimed),
        )

    # Assertion 2: conditional rows whose trigger was *not* met, whose
    # anchored paths have an unclaimed delta, are violations.
    conditional_violations = _conditional_violations(
        register, triggers_met, deltas, provider_key=provider_key
    )
    if conditional_violations:
        # Surface one row at a time so the failure message names a specific
        # row id; the report keeps the full list.
        first_row_id, first_paths = conditional_violations[0]
        raise ConditionalRowFiredWithoutTriggerError(
            f"§3.3.2 assertion 2: conditional row {first_row_id!r} fired "
            f"without its trigger; first path: {first_paths[0]!r}",
            paths=tuple(first_paths),
            row_id=first_row_id,
        )

    return deltas


def _native_passthrough_check(inbound: CapturedRequest, captured: CapturedRequest) -> None:
    """Assert the native passthrough route preserves JSON key order.

    §4.3 C2: a provider can fingerprint the JSON serialiser from key
    ordering alone, so on the route that claims to be forwarding rather
    than translating, the captured body's key order must equal the
    inbound body's — at every object level, at every aligned position.

    **Key order, not bytes.** An earlier draft compared the raw bytes.
    That is too strong for the route's own registered mutations: the
    native branch shallow-copies the inbound body and rewrites ``model``
    through ``_normalize_model`` (register row M1, unconditional), so a
    run whose profile model differs from the agent's model differs in
    bytes while preserving key order exactly — and a byte check fails a
    body the design calls correct. §3.3.4 warns of precisely this: "a
    byte diff would fail constantly and teach people to ignore it".
    Values are assertion 1's business, through the projections; C2 owns
    only the ordering a serialiser fingerprint reads.

    **Structural divergence stops the walk.** Where the two bodies'
    shapes diverge (a key present on one side only, a list whose lengths
    differ, a type mismatch), there is no aligned position left to
    compare, and the divergence itself is a *content* difference —
    assertion 1's business through the projections, not an ordering
    fingerprint. The walk compares order while the shapes align and
    stops at the first place they do not.

    **Unparseable bodies fall back to bytes.** A native body that is not
    JSON cannot be walked for key order; byte equality is the only
    ordering claim left, and it is applied strictly.

    Args:
        inbound: The inbound capture.
        captured: The captured capture.

    Raises:
        NativePassthroughKeyOrderError: When the key order diverges (or,
            for unparseable bodies, when the bytes differ).
    """
    try:
        inbound_json = json.loads(inbound.body)
        captured_json = json.loads(captured.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Not walkable — the byte comparison is the only ordering claim
        # available, and it is strict.
        if inbound.body != captured.body:
            raise NativePassthroughKeyOrderError(
                f"§4.3 C2: native passthrough route's bodies are not JSON and "
                f"differ byte-for-byte; inbound={len(inbound.body)}B, "
                f"captured={len(captured.body)}B",
                paths=(),
                inbound_preview=inbound.body[:200],
                captured_preview=captured.body[:200],
            ) from None
        return

    diverged_at = _first_key_order_divergence(inbound_json, captured_json, "$")
    if diverged_at is not None:
        raise NativePassthroughKeyOrderError(
            f"§4.3 C2: native passthrough route reordered JSON keys; "
            f"first divergence at {diverged_at}",
            # No ``paths`` entry: the C2 failure is a raw-body observation,
            # not a projection delta, and §3.3.1a's route vocabulary
            # (``ROUTE_COMPONENTS``) names method/scheme/host/path/query —
            # no ``body`` component exists to anchor. The diverged-at JSON
            # path and the two body previews are the evidence.
            paths=(),
            inbound_preview=inbound.body[:200],
            captured_preview=captured.body[:200],
        )


def _routing_check(captured: CapturedRequest, expected: ExpectedRoute) -> None:
    """Assert the captured request went where the derived expectation says.

    §3.3.5: routing is part of the request, and the body cannot show it. The
    comparison runs over **every** route component — method, scheme, host,
    path, query — driven from :data:`tests.harness.contract.ROUTE_COMPONENTS`
    so the component set cannot drift from the contract vocabulary if a
    component is ever added. Scheme and host are compared on the strength of
    the caller's mandatory authority rewrite: the expectation must already
    carry the recorder's values when it arrives (§3.3.5's T-W4 scope
    addition), so a mismatch here means the caller skipped the rewrite or the
    request genuinely went to another authority.

    **Runs after the body obligations, by design.** The caller orders the
    routing assertion last so body failures — the louder diagnosis — surface
    first, and so the §3.3.5 falsification (a rerouted capture with a
    byte-identical body) fails on routing and only on routing.

    Args:
        captured: The captured request.
        expected: The independently derived routing expectation, with the
            recorder's authority already rewritten in.

    Raises:
        RoutingMismatchError: When any component disagrees. Every failing
            component is named in both places — the message carries each
            component's expected and captured values (the query redacted,
            the rest verbatim), and ``paths`` carries the matching
            ``route.<component>`` constants — so a misderived expectation
            and a misrouted request are distinguishable at a glance.
    """
    # Collect every disagreement before raising: one failure that names all
    # five components is worth more than five runs that each name one.
    mismatches: list[tuple[str, str, str]] = []
    for component in sorted(c.ROUTE_COMPONENTS):
        actual = getattr(captured, component)
        wanted = getattr(expected, component)
        if actual != wanted:
            mismatches.append((component, wanted, actual))
    if not mismatches:
        return
    paths = tuple(c.route_path(component) for component, _, _ in mismatches)
    # The ``query`` component carries wire-visible credentials when a profile's
    # ``base_url`` does (the KBR-143 merge brings them into the composed query).
    # Route them through the contract's redaction so a routing mismatch on
    # such a profile does not surface the credential into the pytest failure
    # message — the same redaction ``CapturedRequest.__repr__`` and the
    # diagnostic helper apply, now closing the last unmasked surface on the
    # oracle's output. Other components carry no secret and stay verbatim.
    detail = "; ".join(
        f"{component}: expected {_render(component, wanted)!r}, captured {_render(component, actual)!r}"
        for component, wanted, actual in mismatches
    )
    raise RoutingMismatchError(
        f"§3.3.5 routing assertion: {len(mismatches)} route component(s) "
        f"differ from the derived expectation — {detail}",
        paths=paths,
    )


def _render(component: str, value: str) -> str:
    """Render one route component's value for the routing-failure message.

    The ``query`` component may carry credentials; the other components carry
    none. The split matches the contract's redaction in
    :meth:`~tests.harness.contract.CapturedRequest.__repr__`, so the oracle's
    own failure surface honours the same masking.

    Args:
        component: The route component whose value is being rendered.
        value: The raw value as captured or expected.

    Returns:
        A display string safe for a log or an assertion diff.
    """
    if component == "query":
        return _redact_query(value)
    return value


def _first_key_order_divergence(a: Any, b: Any, at: str) -> str | None:
    """Return the JSON path of the first key-order divergence, or ``None``.

    Walks two parsed JSON values in parallel. Objects compare their key
    sequences (``list(keys)`` — insertion order is parse order on both
    sides) and recurse per key; lists walk element-wise at the same
    index; scalars carry no ordering and compare as aligned. The walk
    stops at structural divergence — different key sets, different
    lengths, different types — returning ``None`` there: a content
    difference is assertion 1's business, not an ordering fingerprint.

    Args:
        a: The inbound value.
        b: The captured value.
        at: The JSON path of this position, for the failure message.

    Returns:
        The JSON path of the first key-order divergence, or ``None`` when
        the orders agree everywhere the shapes align.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        # A key-SET difference is a content difference (a field the bridge
        # added or dropped — M1/M2-family business, assertion 1's, through
        # the projections). Only an equal set with a different SEQUENCE is
        # an ordering fingerprint.
        if set(a.keys()) == set(b.keys()) and list(a.keys()) != list(b.keys()):
            return at
        for key, a_value in a.items():
            if key not in b:
                continue  # content difference; nothing aligned to walk
            diverged = _first_key_order_divergence(a_value, b[key], f"{at}.{key}")
            if diverged is not None:
                return diverged
        return None
    if isinstance(a, list) and isinstance(b, list):
        # Element-wise at the same index; a length difference is a
        # content difference, not an ordering one, so the walk stops at
        # the shorter list (strict=False is the semantics, spelled out).
        for index, (a_item, b_item) in enumerate(zip(a, b, strict=False)):
            diverged = _first_key_order_divergence(a_item, b_item, f"{at}[{index}]")
            if diverged is not None:
                return diverged
        return None
    # Scalars, or structurally mismatched types: nothing to compare.
    return None


# --------------------------------------------------------------------------
# Structural diff
# --------------------------------------------------------------------------


def _structural_diff(inbound: Request, captured: Request) -> tuple[str, ...]:
    """Emit concrete delta paths between two :class:`Request` projections.

    Walks the envelope and the conversation. **The residual is not walked**,
    and needs no belt-and-braces: the totality gate (:func:`verify_total`)
    has already rejected any non-empty residual on either side before this
    function runs, so a residual delta is unreachable here by construction.
    A future change that loosens the gate would have to delete that check
    in this module — in one place, visibly — rather than quietly relying on
    a comment.

    Address forms:

    * Envelope fields use ``envelope.<field>`` and ``envelope.extra[<key>]``
      (§3.3.1a — ``extra`` is compared one wire key at a time, whole value).
    * Conversation turns use ``conversation.turns[<i>].role`` and
      ``conversation.turns[<i>].parts[<j>].<field>`` — index-based,
      per §7.4.1 rule 2 (residual keys are indexed).
    * Conversation tools use ``conversation.tools[<name>].<field>`` —
      by-name, per §3.3.1a (translators reorder; positional addressing
      would report a delta on every reorder).
    * Sampling uses ``conversation.sampling[<key>]``.
    * System uses ``conversation.system[<i>].<field>``.

    Args:
        inbound: The inbound projection.
        captured: The captured projection.

    Returns:
        A tuple of concrete delta paths in walk order. Empty on byte-equal
        projections.
    """
    deltas: list[str] = []

    # Envelope — leaf scalars compared directly; extra is one wire key at
    # a time, whole value (§3.3.1a).
    if inbound.envelope.model != captured.envelope.model:
        deltas.append(c.ENVELOPE_MODEL)
    if inbound.envelope.stream != captured.envelope.stream:
        deltas.append(c.ENVELOPE_STREAM)
    if inbound.envelope.store != captured.envelope.store:
        deltas.append(c.ENVELOPE_STORE)
    for key in sorted(set(inbound.envelope.extra) | set(captured.envelope.extra)):
        if inbound.envelope.extra.get(key) != captured.envelope.extra.get(key):
            deltas.append(c.extra_path(key))

    # Conversation — system, turns, tools, sampling.
    deltas.extend(_diff_system(inbound.conversation, captured.conversation))
    deltas.extend(_diff_turns(inbound.conversation, captured.conversation))
    deltas.extend(_diff_tools(inbound.conversation, captured.conversation))
    for key in sorted(set(inbound.conversation.sampling) | set(captured.conversation.sampling)):
        if inbound.conversation.sampling.get(key) != captured.conversation.sampling.get(key):
            deltas.append(c.sampling_path(key))

    return tuple(deltas)


def _diff_system(inbound: Conversation, captured: Conversation) -> Iterable[str]:
    """Diff the system blocks and the system role.

    ``system_role`` is compared first because M20 anchors it directly
    (``conversation.system_role`` — the Gemini ``systemInstruction`` role
    KBR-194 gave the grammar a slot for): a translated route that drops the
    role must surface as a positive delta at that path, not as a silent
    equivalence on the blocks.

    Args:
        inbound: The inbound projection's conversation.
        captured: The captured projection's conversation.

    Yields:
        Concrete delta paths.
    """
    if inbound.system_role != captured.system_role:
        yield c.SYSTEM_ROLE_PATH

    n = max(len(inbound.system), len(captured.system))
    for i in range(n):
        path = c.system_path(i)
        a = inbound.system[i] if i < len(inbound.system) else None
        b = captured.system[i] if i < len(captured.system) else None
        if a is None or b is None:
            yield path
            continue
        if a.text != b.text:
            yield c.system_path(i, "text")
        if a.cache_control != b.cache_control:
            yield c.system_path(i, "cache_control")


def _diff_turns(inbound: Conversation, captured: Conversation) -> Iterable[str]:
    """Diff the turns by index, walking each turn's parts.

    Args:
        inbound: The inbound projection's conversation.
        captured: The captured projection's conversation.

    Yields:
        Concrete delta paths.
    """
    n = max(len(inbound.turns), len(captured.turns))
    for i in range(n):
        turn_path = c.turn_path(i)
        a = inbound.turns[i] if i < len(inbound.turns) else None
        b = captured.turns[i] if i < len(captured.turns) else None
        if a is None or b is None:
            yield turn_path
            continue
        if a.role != b.role:
            yield c.turn_path(i, "role")
        # Walk parts — index-based, per §7.4.1 rule 2.
        yield from _diff_parts(i, a.parts, b.parts)


def _diff_parts(turn_index: int, a_parts: Sequence[Any], b_parts: Sequence[Any]) -> Iterable[str]:
    """Diff two ordered part lists, emitting paths for every differing field.

    Address form: ``conversation.turns[<i>].parts[<j>].<field>``.

    Args:
        turn_index: The turn index, used in the emitted path.
        a_parts: The inbound parts.
        b_parts: The captured parts.

    Yields:
        Concrete delta paths.
    """
    n = max(len(a_parts), len(b_parts))
    for j in range(n):
        path = c.part_path(turn_index, j)
        a = a_parts[j] if j < len(a_parts) else None
        b = b_parts[j] if j < len(b_parts) else None
        if a is None or b is None:
            yield path
            continue
        if type(a) is not type(b):
            yield path
            continue
        if isinstance(a, c.Text):
            if a.text != b.text:
                yield c.part_path(turn_index, j, "text")
            if a.cache_control != b.cache_control:
                yield c.part_path(turn_index, j, "cache_control")
            if a.video_metadata != b.video_metadata:
                yield c.part_path(turn_index, j, "video_metadata")
        elif isinstance(a, c.ToolUse):
            if a.name != b.name:
                yield c.part_path(turn_index, j, "name")
            if a.id != b.id:
                yield c.part_path(turn_index, j, "id")
            if a.arguments != b.arguments:
                yield c.part_path(turn_index, j, "arguments")
            if a.cache_control != b.cache_control:
                yield c.part_path(turn_index, j, "cache_control")
            if a.signature != b.signature:
                yield c.part_path(turn_index, j, "signature")
        elif isinstance(a, c.Thinking):
            if a.text != b.text:
                yield c.part_path(turn_index, j, "text")
            if a.signature != b.signature:
                yield c.part_path(turn_index, j, "signature")
        elif isinstance(a, c.Json):
            if a.value != b.value:
                yield c.part_path(turn_index, j, "value")
        elif isinstance(a, c.ToolResult):
            if a.tool_use_id != b.tool_use_id:
                yield c.part_path(turn_index, j, "tool_use_id")
            if a.is_error != b.is_error:
                yield c.part_path(turn_index, j, "is_error")
            if a.scheduling != b.scheduling:
                yield c.part_path(turn_index, j, "scheduling")
            if a.cache_control != b.cache_control:
                yield c.part_path(turn_index, j, "cache_control")
            yield from _diff_part_content(turn_index, j, a.content, b.content)
        elif isinstance(a, c.Opaque):
            if a.kind != b.kind:
                yield c.part_path(turn_index, j, "kind")
            if a.digest != b.digest:
                yield c.part_path(turn_index, j, "digest")
            if a.cache_control != b.cache_control:
                yield c.part_path(turn_index, j, "cache_control")
        elif isinstance(a, c.Image):
            # M24 (``conversation.turns[*].parts[*].video_metadata``) and M25
            # (``conversation.turns[*].parts[*].display_name``) anchor
            # fields the Gemini reader populates; the diff has to compare
            # them so a translation that drops either is a claimed delta,
            # not a silent equivalence.
            if a.digest != b.digest:
                yield c.part_path(turn_index, j, "digest")
            if a.media_type != b.media_type:
                yield c.part_path(turn_index, j, "media_type")
            if a.ref != b.ref:
                yield c.part_path(turn_index, j, "ref")
            if a.display_name != b.display_name:
                yield c.part_path(turn_index, j, "display_name")
            if a.video_metadata != b.video_metadata:
                yield c.part_path(turn_index, j, "video_metadata")
            if a.cache_control != b.cache_control:
                yield c.part_path(turn_index, j, "cache_control")
        else:
            # Defensive: a Part variant the diff does not know about.
            # Future variants fail closed by emitting the part path; the
            # register can claim it.
            yield path


def _diff_part_content(
    turn_index: int, part_index: int, a: Sequence[Any], b: Sequence[Any]
) -> Iterable[str]:
    """Diff a ToolResult's content list (Text/Image/Json/Opaque).

    The content is compared as a sequence; per-content-element fields are
    compared by element index.

    Args:
        turn_index: The enclosing turn index.
        part_index: The enclosing part index.
        a: The inbound content.
        b: The captured content.

    Yields:
        Concrete delta paths.
    """
    n = max(len(a), len(b))
    for k in range(n):
        path = c.part_path(turn_index, part_index, f"content[{k}]")
        x = a[k] if k < len(a) else None
        y = b[k] if k < len(b) else None
        if x is None or y is None:
            yield path
            continue
        if type(x) is not type(y):
            yield path
            continue
        if isinstance(x, c.Text):
            if x.text != y.text:
                yield path + ".text"
        elif isinstance(x, c.Image):
            if x.digest != y.digest:
                yield path + ".digest"
            if x.media_type != y.media_type:
                yield path + ".media_type"
            if x.ref != y.ref:
                yield path + ".ref"
            if x.display_name != y.display_name:
                yield path + ".display_name"
            if x.video_metadata != y.video_metadata:
                yield path + ".video_metadata"
        elif isinstance(x, c.Json):
            if x.value != y.value:
                yield path + ".value"
        elif isinstance(x, c.Opaque):
            if x.kind != y.kind:
                yield path + ".kind"
            if x.digest != y.digest:
                yield path + ".digest"


def _diff_tools(inbound: Conversation, captured: Conversation) -> Iterable[str]:
    """Diff the tool declarations by name (§3.3.1a — translators reorder).

    Args:
        inbound: The inbound projection's conversation.
        captured: The captured projection's conversation.

    Yields:
        Concrete delta paths.
    """
    by_name_in: dict[str, c.ToolDecl] = {t.name: t for t in inbound.tools}
    by_name_out: dict[str, c.ToolDecl] = {t.name: t for t in captured.tools}
    names = sorted(set(by_name_in) | set(by_name_out))
    for name in names:
        a = by_name_in.get(name)
        b = by_name_out.get(name)
        if a is None or b is None:
            yield c.tool_path(name)
            continue
        if a.description != b.description:
            yield c.tool_path(name, "description")
        if a.schema != b.schema:
            yield c.tool_path(name, "schema")
        if a.strict != b.strict:
            yield c.tool_path(name, "strict")
        if a.behavior != b.behavior:
            yield c.tool_path(name, "behavior")
        if a.type != b.type:
            yield c.tool_path(name, "type")
        if a.cache_control != b.cache_control:
            yield c.tool_path(name, "cache_control")


def _structural_reply_diff(inbound: Reply, captured: Reply) -> tuple[str, ...]:
    """Emit concrete delta paths between two :class:`Reply` projections.

    The response-direction twin of :func:`_structural_diff` (KBR-59 /
    T-D10). Walks ``parts`` and the bare ``stop_reason``. **Two fields
    are skipped unconditionally** (reviewer finding C10 retired the
    earlier "non-canonical value" qualifier):

    * :attr:`Reply.usage` — the design carries it "but excluded from
      the fidelity diff" (``contract.py:1318`` docstring); usage is
      provider-reported and never agent-supplied, so a difference
      carries no I1 information.
    * :attr:`Reply.stop_reason_raw` — carries the wire's own value on
      a non-canonical stop reason; it has no I1 information either.

    Neither generates a ``reply.usage[*]`` or ``reply.stop_reason_raw``
    delta. The residual is not walked, matching :func:`_structural_diff`'s
    rationale: the totality gate has already rejected a non-empty
    residual before this function runs.

    Address forms:

    * ``reply.parts[<i>].<field>`` — field-level, per §7.4.1 rule 2
      (residual keys are indexed; field names are stable across
      readers).
    * ``reply.parts[<i>]`` — bare-index, for a part-count mismatch or
      a kind mismatch at an index (two readers that disagree about a
      part boundary report a delta at the boundary and leave the
      fields unaddressed).
    * ``reply.stop_reason`` — the keyed literal, the M28 anchor.

    Args:
        inbound: The inbound reply projection.
        captured: The captured reply projection.

    Returns:
        A tuple of concrete delta paths in walk order. Empty on
        byte-equal projections.
    """
    deltas: list[str] = []

    # stop_reason — bare keyed literal (M28's anchor). The raw value is
    # skipped above; see the docstring.
    if inbound.stop_reason != captured.stop_reason:
        deltas.append(c.REPLY_STOP_REASON)

    # Parts — index-aligned pairwise diff. Length mismatch or kind
    # mismatch collapses to the bare index; same-kind field differences
    # address the field directly.
    n = max(len(inbound.parts), len(captured.parts))
    for i in range(n):
        in_part = inbound.parts[i] if i < len(inbound.parts) else None
        cap_part = captured.parts[i] if i < len(captured.parts) else None

        # Either side ran out, or the parts disagree on kind — the
        # §7.4.1 bare-index form. Reporting a bare-index delta here is
        # deliberate: a narrower field-level path would imply a
        # same-kind comparison that this branch already ruled out.
        if type(in_part) is not type(cap_part) or in_part is None or cap_part is None:
            deltas.append(c.reply_part_path(i))
            continue

        # Same kind; address every differing field.
        for field in dataclasses.fields(in_part):
            in_value = getattr(in_part, field.name)
            cap_value = getattr(cap_part, field.name)
            if in_value != cap_value:
                deltas.append(f"{c.reply_part_path(i)}.{field.name}")

    return tuple(deltas)


# --------------------------------------------------------------------------
# Claim matching + assertion 2
# --------------------------------------------------------------------------


def _claim_matching(
    deltas: Sequence[str],
    register: tuple[r.MutationRow, ...],
    triggers_met: frozenset[r.Trigger],
    *,
    provider_key: str = r.ALL_PROVIDERS,
) -> dict[str, tuple[str, ...]]:
    """Return, per delta, every register row id whose trigger is met and
    whose pattern matches.

    All claimers are recorded — not just the first one to match. A delta
    can match several rows simultaneously (§3.3.1a's prefix rule means
    coarse-anchor rows like M5 ``conversation.turns`` and narrow-anchor
    rows like M3 ``conversation.turns[*].parts[*]`` both match a
    part-level delta), and the assertion that consumes this map only
    needs the empty / non-empty distinction. The full list is what a
    reviewer reaches for when the run reports an unclaimed delta on a
    path they expected to be claimed.

    Args:
        deltas: The concrete delta paths to classify.
        register: The register rows.
        triggers_met: The trigger vocabulary the input met.
        provider_key: The adapter this run is judging — narrows the
            runtime notion of "live on this adapter" via
            :func:`~harness.register.row_is_in_scope`. Defaults to
            :data:`~harness.register.ALL_PROVIDERS`, the permissive
            sentinel — the call-site short-circuit makes the default
            permissive; the helper itself is **not** permissive when
            handed the sentinel (the sentinel is a valid value of
            ``row.scope``, not of ``provider_key``). KBR-307.

    Returns:
        A mapping from delta path to the tuple of claiming row ids. Empty
        tuple means unclaimed.
    """
    claimers: dict[str, list[str]] = {delta: [] for delta in deltas}
    active_rows = [row for row in register if row.trigger in triggers_met]
    if provider_key != r.ALL_PROVIDERS:
        active_rows = [row for row in active_rows if r.row_is_in_scope(row, provider_key)]
    for row in active_rows:
        if not row.is_projectable:
            continue
        for pattern in row.paths:
            for delta in deltas:
                if c.path_matches(pattern, delta):
                    claimers[delta].append(row.id)
    return {delta: tuple(rows) for delta, rows in claimers.items()}


def _conditional_violations(
    register: tuple[r.MutationRow, ...],
    triggers_met: frozenset[r.Trigger],
    deltas: Sequence[str],
    *,
    provider_key: str = r.ALL_PROVIDERS,
) -> list[tuple[str, list[str]]]:
    """Find conditional rows whose anchor carries a delta on an input that
    did not meet their trigger, under **specificity attribution**.

    For each conditional register row R whose trigger is *not* in
    ``triggers_met``, a delta matching one of R's anchored paths is a
    violation **unless a triggered row specifically claims it** — a
    triggered row whose pattern matches the delta and is *not a proper
    prefix* of one of R's matching patterns
    (:func:`~harness.contract.pattern_is_proper_prefix_of`). Three cases
    show the rule:

    * M16 (``…parts[*].cache_control``, triggered) and M3
      (``…parts[*]``, untriggered) both match a cache-breakpoint delta.
      M16's anchor is *finer* (M3's is a proper prefix of it), so M16
      specifically claims and M3 is exempt — a cache breakpoint the
      bridge legitimately stripped is not evidence that M3 fired.
    * M8 and M3 share the identical bare-part anchor. A part delta M8
      (triggered) explains is therefore exempt for M3 too: equal anchors
      co-claim, and neither defers to the other.
    * M5 (``conversation.turns``, triggered) and M3 (``parts[*]``,
      untriggered) both match a part-level delta — but M5's anchor *is*
      a proper prefix of M3's, so M5 does **not** specifically claim.
      The delta is a violation for M3. This is the case that keeps
      assertion 2 alive: assertion 1's claim matching cannot see it,
      because M5's broad anchor legitimately claims everything beneath
      it.

    **What this means for the caller.** The third case is why a
    complement input for conditional row R (§3.3.4, T-D8) must avoid
    co-triggering a *broader*-anchored conditional row on R's paths: on
    such an input the oracle reports R as fired whenever a delta lands
    at R's anchor, which is the observation the complement test exists
    to make. The oracle reports raw presence subject to the specificity
    rule; the corpus construction owns arranging the rest.

    **Scope filter (KBR-307).** R's iteration is also filtered by
    ``row_is_in_scope(row, provider_key)``: a conditional row whose site
    cannot execute on ``provider_key`` cannot "fire without its trigger"
    on that adapter, and the violation would be a false positive. The
    ``active`` set used for specificity-attribution is filtered the same
    way, so both halves of the oracle agree about what a row is allowed
    to fire on. The default-permissive short-circuit is at the
    ``provider_key == ALL_PROVIDERS`` boundary, the same shape as
    ``_claim_matching``'s.

    Args:
        register: The register rows.
        triggers_met: The trigger vocabulary the input met.
        deltas: The concrete delta paths the structural diff found.
        provider_key: The adapter this run is judging — narrows the
            runtime notion of "live on this adapter" via
            :func:`~harness.register.row_is_in_scope`, applied to both
            the iteration over conditional rows **and** the triggered
            ``active`` set. Defaults to
            :data:`~harness.register.ALL_PROVIDERS`, the permissive
            sentinel. KBR-307.

    Returns:
        A list of ``(row_id, [paths, ...])`` pairs, one per violation. Each
        violation's path list is the subset of ``deltas`` that land at the
        row's anchors without a specifically-claiming triggered row.
    """
    active = [row for row in register if row.trigger in triggers_met and row.is_projectable]
    if provider_key != r.ALL_PROVIDERS:
        active = [row for row in active if r.row_is_in_scope(row, provider_key)]

    violations: list[tuple[str, list[str]]] = []
    for row in register:
        if not row.conditional or row.trigger in triggers_met:
            continue
        if not row.is_projectable:
            continue
        if provider_key != r.ALL_PROVIDERS and not r.row_is_in_scope(row, provider_key):
            continue
        offending: list[str] = []
        for delta in deltas:
            matching_own = [p for p in row.paths if c.path_matches(p, delta)]
            if not matching_own:
                continue
            specifically_claimed = any(
                c.path_matches(tp, delta)
                and not any(c.pattern_is_proper_prefix_of(tp, rp) for rp in matching_own)
                for trow in active
                for tp in trow.paths
            )
            if not specifically_claimed:
                offending.append(delta)
        if offending:
            violations.append((row.id, offending))
    return violations


# --------------------------------------------------------------------------
# Reader registration
# --------------------------------------------------------------------------


# Importing the readers here would couple the oracle to every reader at
# import time, which is fine for T-D1's delivery but couples this module
# to the reader set. The registration is explicit so adding a reader is
# one line at the reader's own import site, mirroring the §7.4.1 reader
# contract.

from harness.reader_anthropic_messages import AnthropicMessagesProjection  # noqa: E402
from harness.reader_bedrock_converse import BedrockConverseProjection  # noqa: E402
from harness.reader_chat_completions import ChatCompletionsProjection  # noqa: E402
from harness.reader_gemini import GeminiProjection  # noqa: E402
from harness.reader_ollama import OllamaChatProjection as OllamaProjection  # noqa: E402
from harness.reader_responses import ResponsesProjection  # noqa: E402

_register_projection(AnthropicMessagesProjection())
_register_projection(BedrockConverseProjection())
_register_projection(ChatCompletionsProjection())
_register_projection(GeminiProjection())
_register_projection(OllamaProjection())
_register_projection(ResponsesProjection())

# Reply-direction readers (KBR-59 / T-D10) — central registration here,
# mirroring the request-side block above. Five formats have a
# `read_reply` today; an additional reader (e.g. KBR-312's Bedrock
# Converse reply reader) grows _ASSERTABLE_REPLY_FORMATS above and
# adds one line here.
from harness.reader_anthropic_messages import AnthropicMessagesReplyProjection  # noqa: E402
from harness.reader_chat_completions import ChatCompletionsReplyProjection  # noqa: E402
from harness.reader_gemini import GeminiReplyProjection  # noqa: E402
from harness.reader_ollama import OllamaChatReplyProjection  # noqa: E402
from harness.reader_responses import ResponsesReplyProjection  # noqa: E402

_register_reply_projection(AnthropicMessagesReplyProjection())
_register_reply_projection(ChatCompletionsReplyProjection())
_register_reply_projection(GeminiReplyProjection())
_register_reply_projection(OllamaChatReplyProjection())
_register_reply_projection(ResponsesReplyProjection())

_REGISTRY_GUARD()
_REPLY_REGISTRY_GUARD()


__all__ = [
    "ConditionalRowFiredWithoutTriggerError",
    "ExpectedRoute",
    "NativePassthroughKeyOrderError",
    "OracleError",
    "OracleReport",
    "ReplyOracleReport",
    "RoutingMismatchError",
    "UnclaimedMutationError",
    "assert_no_unclaimed_mutation",
    "assert_no_unclaimed_reply_mutation",
]
