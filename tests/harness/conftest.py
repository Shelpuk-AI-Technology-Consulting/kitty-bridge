"""Harness-level pytest conftest: shared fixtures and the slice verdict gates.

This conftest applies to every test under :mod:`tests/harness/`. It serves
three purposes, all narrow:

* the ``pytest_runtest_makereport`` hook populates per-phase outcome
  dictionaries for each registered slice — :data:`_phase_outcomes` /
  :data:`_phase_teardown_outcomes` for the **T-E2** (aiohttp) slice and
  :data:`_curl_phase_outcomes` / :data:`_curl_phase_teardown_outcomes` for
  the **T-E3** (curl_cffi) slice — dispatching by nodeid file prefix, which
  is what lets the two slices share method names without bleeding outcomes
  into each other's gates;
* the ``_record_slice_verdict_at_session_end`` and
  ``_record_curl_cffi_slice_verdict_at_session_end`` session-scope fixtures
  read their dictionaries at teardown, assert no foreign row was mutated,
  and write ``PROVEN`` iff their every phase is ``PASSED`` — or
  ``UNSUPPORTED`` (with the floor reason) when the Python <3.11 floor
  shape applies: phase 1 ran and passed, the proxied phases are
  setup-skipped. The plan T-E3 done-when ("an outcome is recorded") and
  T-E9's gate semantics ("``unsupported`` is a permitted outcome for
  partial delivery") are both honoured on every supported interpreter;
* the ``_enforce_containment_completeness`` session-scope fixture reads
  the singleton at finalisation and asserts every **landed** slice's
  verdict row is in a permitted state. Landed slices are those whose
  descriptor appears in :data:`_SLICES` — adding a sibling's descriptor
  there (T-E4, T-E5) tightens the gate with no edit to T-E9. The gate
  fixture is **defined first** in this file so its teardown is **last**
  (pytest reverses finalisation order for independent session-scope
  fixtures), which is the only ordering in which the slice finalisers'
  verdicts are already written when the gate reads them.

The mechanism is what makes the verdicts honest. A test that records
``PROVEN`` itself would pass with the four phases deleted, and on
Python <3.11 — where phases 2/2b/3 skip because of TLS-in-TLS — would record
``PROVEN`` after phase 1 alone. The hook is a falsifiable witness to "every
phase ran" — the verdict cannot be written without that witness agreeing.

**Descriptor shape.** Each :data:`_SLICES` tuple binds five fields, in this
order: ``(file_prefix, phase_names, outcomes, teardown_outcomes, verdict_row)``.
The trailing ``verdict_row`` is the :class:`~harness.containment.CapabilityReport`
row the slice's finaliser records into. :func:`_landed_verdict_rows` returns
the set of verdict rows registered across all descriptors, and the T-E9
gate passes that set to
:meth:`~harness.containment.CapabilityReport.require_completeness`
so a slice that has not yet landed (no descriptor) is exempt by design.

**A note for sibling slices (T-E4..T-E5).** The verdict recording is the
session-finaliser's job, and it runs **after** every test in the process,
not per file — so an autouse ``reset_for_test()`` in a sibling's test
module cannot erase a verdict that has not been written yet, and the
finaliser writes its own row regardless of what any earlier reset did.
The reset seam is for *per-test isolation* (each test sees an untouched
report), not for verdict erasure; sibling slices that follow this
template should put their own finaliser in this conftest (or in their own
scoped conftest) rather than calling ``record()`` from a test body, and add
their slice to :data:`_SLICES`. Each new finaliser extends the sibling-row
exemption pattern: a row is exempt only when its owning slice's gate passed
*and* the row currently reads ``PROVEN`` — the documented residual limit of
this structural check.
"""

from __future__ import annotations

import enum
import sys
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import Any

import pytest

from harness.containment import Outcome
from harness.containment import instance as report_instance


def _validate_slices(descriptors: Sequence[tuple[Any, ...]] | None = None) -> None:
    """Raise at import when any :data:`_SLICES` descriptor is malformed.

    A descriptor with the wrong arity would otherwise surface as a generic
    ``ValueError: not enough values to unpack`` at the **first matching
    phase test**, mid-session, with a traceback naming the test rather than
    the bad descriptor — and duplicated ``verdict_row`` or ``file_prefix``
    values would silently mis-record or mis-dispatch outcomes between two
    slices. Checking at conftest import fails the whole session with the
    offending descriptor named, which is the only message a future T-E4 /
    T-E5 author can act on without re-deriving the shape.

    Args:
        descriptors: The descriptors to check. Defaults to
            :data:`_SLICES`; parameterised so the unit tests can hand a
            synthetic malformed descriptor to each rejection branch
            without importing a broken conftest.

    Raises:
        ValueError: When a descriptor's arity is wrong, when its
            ``file_prefix`` or ``verdict_row`` is not a string, or when
            either value is duplicated across descriptors.
    """
    if descriptors is None:
        descriptors = _SLICES
    prefixes: list[str] = []
    rows: list[str] = []
    for descriptor in descriptors:
        if len(descriptor) != 5:
            raise ValueError(
                f"_SLICES descriptor {descriptor!r} binds {len(descriptor)} fields, expected 5: "
                "(file_prefix, phase_names, outcomes, teardown_outcomes, verdict_row)"
            )
        prefix, _names, _outcomes, _teardown_outcomes, verdict_row = descriptor
        if not isinstance(prefix, str):
            raise ValueError(f"_SLICES descriptor {descriptor!r} has a non-string file_prefix: {prefix!r}")
        if not isinstance(verdict_row, str):
            raise ValueError(f"_SLICES descriptor {descriptor!r} has a non-string verdict_row: {verdict_row!r}")
        prefixes.append(prefix)
        rows.append(verdict_row)
    duplicated_prefixes = sorted({p for p in prefixes if prefixes.count(p) > 1})
    if duplicated_prefixes:
        raise ValueError(f"_SLICES descriptors share file prefixes {duplicated_prefixes}: outcomes would mis-dispatch")
    duplicated_rows = sorted({r for r in rows if rows.count(r) > 1})
    if duplicated_rows:
        raise ValueError(f"_SLICES descriptors share verdict rows {duplicated_rows}: verdicts would mis-record")


def _landed_verdict_rows() -> frozenset[str]:
    """Return the verdict rows whose owning slice has a registered descriptor.

    A row absent from the returned set belongs to a slice that has not
    landed yet — its descriptor appears in :data:`_SLICES` when the slice
    ships, which is the auto-tightening property KBR-69's gate rests on:
    the session-end gate enforces exactly this set, so a sibling landing
    its descriptor immediately subjects its row to the gate with no edit
    to the gate itself.

    Returns:
        A ``frozenset`` (immutable for downstream callers, matching the
        module's other name-set constants).
    """
    return frozenset(verdict_row for *_rest, verdict_row in _SLICES)


def _run_completeness_gate() -> None:
    """Apply the T-E9 completeness check to the live singleton.

    Pure over :func:`report_instance` and :data:`_SLICES`: a unit test can
    drive this directly with a synthetic singleton state, and the autouse
    session-scope fixture calls it at session finalisation. Two short-circuits
    keep the gate from firing when it has nothing to say:

    * **No slice has recorded.** A run that touches the singleton only
      through ``reset_for_test()`` (e.g. ``tests.harness.test_containment``
      in isolation) leaves every row ``NOT_ATTEMPTED``; firing the gate
      would fail a green run for a defect that does not exist.
    * **No verdict row is registered as landed.** Defensive: today every
      :data:`_SLICES` entry has a verdict row, so this branch never fires
      in production, but the closed-set property means a malformed conftest
      that produced an empty landed set would fail in a noisier way at
      the seam rather than silently passing.

    Raises:
        AssertionError: Forwarded from
        :meth:`harness.containment.CapabilityReport.require_completeness`
        when any landed row is ``not_attempted`` or ``failed``. The message
        names every offending row, sorted, and groups them by defect kind.
    """
    singleton = report_instance()
    landed = _landed_verdict_rows()
    if not landed:
        return
    any_recorded = any(
        entry.outcome is not Outcome.NOT_ATTEMPTED for entry in singleton.entries().values()
    )
    if not any_recorded:
        return
    singleton.require_completeness(landed_rows=landed)


@pytest.fixture(scope="session", autouse=True)
def _enforce_containment_completeness() -> Iterator[None]:
    """Fail the session when a **landed** slice's verdict row is invalid.

    Calls :func:`_run_completeness_gate` at finalisation.

    **Ordering contract.** This fixture is defined **first** in this
    conftest on purpose. Pytest finalises independent session-scope
    fixtures in reverse definition order, so this fixture's teardown runs
    **after** the two slice finalisers (defined further down) have written
    their verdicts. Defining it last — the natural reading of "the gate
    runs after everything" — would make it finalise **first**, read a
    report no finaliser has touched yet, and fail every green session.

    Yields:
        ``None``, with the completeness check in the finalisation.
    """
    yield
    _run_completeness_gate()


class _PhaseOutcome(enum.Enum):
    """One phase test's outcome, tracked module-locally for the slice verdict.

    Tri-state, not a bare bool, because a **runtime-skipped** test must not
    count as "passed". ``@pytest.mark.skipif`` skips are decided at setup —
    the hook never reaches their ``call`` phase — so those are caught by the
    length check in the finaliser; this enum's ``SKIPPED`` branch catches the
    rarer in-test ``pytest.skip()`` calls.
    """

    PASSED = "passed"
    SKIPPED = "skipped"
    FAILED = "failed"


#: The phase tests whose outcomes gate the slice verdict. Test method names as
#: pytest reports them; renaming a phase test renames the gate, and a rename
#: that forgets this set makes the gate fail loudly (the verdict test names
#: the missing entry).
_PHASE_TEST_NAMES: frozenset[str] = frozenset({
    "test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing",
    "test_proxy_down_leaves_the_recorder_with_zero_connections",
    "test_proxy_up_every_peer_port_joins_a_tunnel",
    "test_failed_tunnel_contributes_no_upstream_connection",
    "test_injected_bypass_makes_the_harness_report_it",
})

#: The nodeid prefix every T-E2 phase test starts with. Matched alongside
#: :data:`_PHASE_TEST_NAMES` so a sibling slice (T-E3..T-E5) with a
#: coincidentally identical method name cannot bleed its outcome into
#: T-E2's verdict gate.
_SLICE_FILE_PREFIX = "tests/harness/test_aiohttp_containment_slice.py::"

#: The capability-report row this module is allowed to write. Named once so
#: the finaliser's write and the sibling-row precondition assertion below
#: cannot drift apart.
_VERDICT_ROW = "bridge_aiohttp"

# ── The T-E3 (curl_cffi) slice's gate ─────────────────────────────────────
#
# The phase-test names are §5.2.2's, identical to T-E2's set: the hook keys
# on the file prefix *alongside* the name (see below), so a coincidentally
# identical method name in the sibling file cannot bleed its outcome into
# this gate — the reason the prefix check exists at all.

#: The phase tests whose outcomes gate the curl_cffi slice verdict. Same five
#: §5.2.2 phases as T-E2's; disambiguated by :data:`_CURL_CFFI_SLICE_FILE_PREFIX`.
_CURL_CFFI_PHASE_TEST_NAMES: frozenset[str] = frozenset({
    "test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing",
    "test_proxy_down_leaves_the_recorder_with_zero_connections",
    "test_proxy_up_every_peer_port_joins_a_tunnel",
    "test_failed_tunnel_contributes_no_upstream_connection",
    "test_injected_bypass_makes_the_harness_report_it",
})

#: The nodeid prefix every T-E3 phase test starts with.
_CURL_CFFI_SLICE_FILE_PREFIX = "tests/harness/test_curl_cffi_containment_slice.py::"

#: The capability-report row the T-E3 slice is allowed to write.
_CURL_CFFI_VERDICT_ROW = "curl_cffi"

#: Populated by :func:`pytest_runtest_makereport` as the T-E3 phase tests
#: complete. Same contract as :data:`_phase_outcomes`, curl slice's own dict.
_curl_phase_outcomes: dict[str, _PhaseOutcome] = {}

#: The teardown-phase outcome for each tracked T-E3 phase test.
_curl_phase_teardown_outcomes: dict[str, _PhaseOutcome] = {}

#: Populated by :func:`pytest_runtest_makereport` as phase tests complete.
#: Read by the session-finaliser verdict gate below. Keyed on test method
#: name; the value is the *call*-phase outcome. A phase test whose teardown
#: fails is not "passed" for the slice-verdict's purposes, so the finaliser
#: also consults :data:`_phase_teardown_outcomes`.
_phase_outcomes: dict[str, _PhaseOutcome] = {}

#: The teardown-phase outcome for each tracked phase test, same keys as
#: :data:`_phase_outcomes`. Absent for a phase that never reached teardown
#: (setup-skipped phases, for instance), which the finaliser treats as
#: vacuously clean — pytest's overall verdict for such a test comes from
#: setup, not teardown.
_phase_teardown_outcomes: dict[str, _PhaseOutcome] = {}


#: Per-slice gate descriptors the hook iterates. Each tuple binds a
#: ``(file_prefix, phase_names, outcomes, teardown_outcomes, verdict_row)``
#: so a test's phase outcome is dispatched to the right slice by its
#: nodeid prefix — the only signal that lets the hook tell the T-E2 and
#: T-E3 slices apart, since the two slices share method names by design.
#: The trailing ``verdict_row`` is the row the slice's finaliser records
#: into; T-E9's completeness gate uses it as the registry of *landed*
#: slices, so a verdict row that has no descriptor here is exempt
#: (the slice hasn't landed yet — its descriptor will appear when the
#: slice ships).
_SLICES: tuple[
    tuple[str, frozenset[str], dict[str, _PhaseOutcome], dict[str, _PhaseOutcome], str],
    ...,
] = (
    (
        _SLICE_FILE_PREFIX, _PHASE_TEST_NAMES, _phase_outcomes, _phase_teardown_outcomes, _VERDICT_ROW,
    ),
    (
        _CURL_CFFI_SLICE_FILE_PREFIX, _CURL_CFFI_PHASE_TEST_NAMES,
        _curl_phase_outcomes, _curl_phase_teardown_outcomes, _CURL_CFFI_VERDICT_ROW,
    ),
)

# The descriptor table is the single source of truth for both the hook's
# dispatch and the T-E9 gate's "landed" set — a malformed row would corrupt
# both, so it is validated the moment the module loads, not at the first
# test that happens to match a bad prefix.
_validate_slices()


def _gate_passed(
    outcomes: dict[str, _PhaseOutcome],
    teardown_outcomes: dict[str, _PhaseOutcome],
    names: frozenset[str],
) -> bool:
    """Return whether a slice's gate is in the **all-passed** state.

    A gate passes iff every tracked phase ran and passed (length matches,
    every outcome is ``PASSED``, every teardown is ``PASSED``). The check is
    local to the dicts given — it never inspects the report — so it is
    order-independent: any finaliser can call it without knowing what the
    other has written.

    Args:
        outcomes: The slice's call-phase outcomes.
        teardown_outcomes: The slice's teardown-phase outcomes.
        names: The slice's expected phase names.

    Returns:
        ``True`` if the gate passes, ``False`` otherwise.
    """
    return (
        len(outcomes) == len(names)
        and all(o is _PhaseOutcome.PASSED for o in outcomes.values())
        and all(o is _PhaseOutcome.PASSED for o in teardown_outcomes.values())
    )


def _aiohttp_gate_passed() -> bool:
    """Whether T-E2's gate is in the all-passed state."""
    return _gate_passed(_phase_outcomes, _phase_teardown_outcomes, _PHASE_TEST_NAMES)


def _curl_gate_passed() -> bool:
    """Whether T-E3's gate is in the all-passed state."""
    return _gate_passed(_curl_phase_outcomes, _curl_phase_teardown_outcomes, _CURL_CFFI_PHASE_TEST_NAMES)


# ── The Python <3.11 floor shape (plan §8's "an outcome is recorded") ──────

#: Phase 1 — the direct leg — carries no ``skipif`` and runs on every
#: supported Python; both slices use the same method name for it.
_PHASE_1_NAME = "test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing"

#: The four proxied phases — 2, 2b's two tests, and 3 — which carry the
#: Python ≥3.11 ``skipif`` in both slices. Their **absence** from the
#: outcomes dict (setup-skipped, never reaching the hook's ``call`` phase)
#: is what distinguishes the floor shape from a deleted/renamed phase.
_PROXIED_PHASE_NAMES: frozenset[str] = frozenset({
    "test_proxy_down_leaves_the_recorder_with_zero_connections",
    "test_proxy_up_every_peer_port_joins_a_tunnel",
    "test_failed_tunnel_contributes_no_upstream_connection",
    "test_injected_bypass_makes_the_harness_report_it",
})

#: The reason text both finalisers record under the floor shape. Shared by
#: both slices (the floor is shared); references what a future maintainer
#: needs to re-derive the decision.
_FLOOR_UNSUPPORTED_REASON = (
    "Python <3.11: a proxied §5.2.2 phase did not run on this interpreter "
    "(phase 1 ran and came back clean). Owner scope decision recorded on "
    "KBR-63 (2026-09-16): record `unsupported` — a permitted partial "
    "delivery T-E9's completeness gate accepts — rather than leave the "
    "row `not_attempted`, which the gate rejects."
)


def _floor_unsupported_shape(
    outcomes: dict[str, _PhaseOutcome],
    teardown_outcomes: dict[str, _PhaseOutcome],
    *,
    phase_1_name: str = _PHASE_1_NAME,
    proxied_names: frozenset[str] = _PROXIED_PHASE_NAMES,
    version_info: tuple[int, ...] | None = None,
) -> bool:
    """Return whether the slice's run ended in the floor-skip shape.

    On Python <3.11 the proxied phases are setup-skipped by their
    ``skipif``; phase 1 — which carries no ``skipif`` — runs and may pass.
    That is the floor shape: the slice did real work (the direct leg
    against the recorder) but the §5.2.2 proxied phases never ran. The
    plan's T-E3 done-when ("an outcome is recorded") and T-E9's gate
    semantics ("``unsupported`` is a permitted outcome for partial
    delivery") are both honoured by recording ``UNSUPPORTED`` in this
    shape rather than leaving the row ``not_attempted``.

    **Phase 1 must have come back clean**, by the gate's own definition
    (:func:`_gate_passed` requires the same of every phase): a call that
    passed but a teardown that errored is "did not come back clean", not
    "partial delivery" — recording ``UNSUPPORTED`` with the reason
    "phase 1 passed" for such a run would be false. A phase-1 teardown
    failure therefore disqualifies the floor shape, and the row stays
    ``NOT_ATTEMPTED`` for T-E9 to surface as a real failure.

    Args:
        outcomes: The slice's call-phase outcomes dict.
        teardown_outcomes: The slice's teardown-phase outcomes dict. Phase
            1 must be present here and ``PASSED`` — the same standard
            :func:`_gate_passed` applies to every phase on the ≥3.11 path.
        phase_1_name: The direct-leg phase's name.
        proxied_names: The proxied phases whose **absence** from
            ``outcomes`` is the floor signal — absent, not ``FAILED``
            (a failure is a real failure, not a floor) and not missing
            from the name set (a rename is a defect the length check
            catches).
        version_info: The interpreter version to test against; defaults
            to the running interpreter. Parameterised so tests can
            simulate the 3.10 matrix without patching ``sys``.

    Returns:
        ``True`` iff the version is below the 3.11 floor AND phase 1 ran,
        passed its call, and came back clean on teardown AND every
        proxied phase is absent from ``outcomes``.
    """
    if version_info is None:
        version_info = sys.version_info[:3]
    if version_info >= (3, 11):
        return False
    if outcomes.get(phase_1_name) is not _PhaseOutcome.PASSED:
        return False
    # Same standard the gate applies on ≥3.11: a phase that passed its
    # call but errored on teardown did not "come back clean". Recording
    # `UNSUPPORTED` with the reason "phase 1 passed" for such a run would
    # be false — the row stays `NOT_ATTEMPTED` and T-E9 surfaces it.
    if teardown_outcomes.get(phase_1_name) is not _PhaseOutcome.PASSED:
        return False
    return all(name not in outcomes for name in proxied_names)


def _record_unsupported_if_floor_shape(
    outcomes: dict[str, _PhaseOutcome],
    teardown_outcomes: dict[str, _PhaseOutcome],
    verdict_row: str,
    *,
    version_info: tuple[int, ...] | None = None,
) -> bool:
    """Record ``UNSUPPORTED`` on ``verdict_row`` when the floor shape applies.

    Writes only when the row is still ``NOT_ATTEMPTED`` — a previously
    recorded verdict (a sibling finaliser's, or this slice's own on an
    earlier session phase) is never overwritten.

    Args:
        outcomes: The slice's call-phase outcomes dict.
        teardown_outcomes: The slice's teardown-phase outcomes dict —
            forwarded to :func:`_floor_unsupported_shape` so a phase-1
            teardown failure disqualifies the floor shape.
        verdict_row: The capability-report row this slice owns.
        version_info: Override for ``sys.version_info``; defaults to the
            current interpreter. Parameterised so tests can exercise the
            writer path on any interpreter matrix without patching ``sys``.

    Returns:
        ``True`` when the row was recorded — i.e. the floor shape applied
        and the row was still ``NOT_ATTEMPTED``. Returned for
        testability and readability; the finalisers call this as the
        last statement of their non-``PROVEN`` branch, so the value is
        not consulted by any caller today, but the contract is part of
        the recorder's shape (a ``False`` return always means "no write
        happened", which the unit tests pin).
    """
    if not _floor_unsupported_shape(outcomes, teardown_outcomes, version_info=version_info):
        return False
    current = report_instance().entries().get(verdict_row)
    if current is None or current.outcome is Outcome.NOT_ATTEMPTED:
        report_instance().record(verdict_row, Outcome.UNSUPPORTED, reason=_FLOOR_UNSUPPORTED_REASON)
        return True
    return False


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """Track phase test outcomes for both slice verdict gates.

    Fires once per phase (setup / call / teardown) of every test in the
    process; the item's nodeid prefix selects the owning slice, and the
    name set selects which phases count. Slices and unrelated harness tests
    leave the dictionaries alone. Both the **call** and the **teardown**
    phase are recorded: a phase whose body passed but whose ``finally``
    (say, a ``SealedNetwork.stop()`` drain) errored must not count as
    "the slice ran" for the verdict gate.

    Args:
        item: The test item being reported.
        call: The call phase's information.
    """
    outcome = yield
    rep = outcome.get_result()  # type: ignore[attr-defined]
    if rep.when not in ("call", "teardown"):
        return
    if rep.passed:
        verdict = _PhaseOutcome.PASSED
    elif rep.skipped:
        verdict = _PhaseOutcome.SKIPPED
    else:
        verdict = _PhaseOutcome.FAILED
    for prefix, names, outcomes, teardown_outcomes, _verdict_row in _SLICES:
        if item.nodeid.startswith(prefix) and item.name in names:
            target = outcomes if rep.when == "call" else teardown_outcomes
            target[item.name] = verdict
            return


@pytest.fixture(scope="session", autouse=True)
def _record_slice_verdict_at_session_end() -> Iterator[None]:
    """Record ``Outcome.PROVEN`` at session teardown iff every phase actually passed.

    The verdict is a **consequence** of the phases having run and passed, not
    something a test can write on its own behalf: the finaliser reads the
    slice's outcomes, which only :func:`pytest_runtest_makereport` fills in
    from real call reports. Four failure shapes, all of which leave the
    singleton's ``bridge_aiohttp`` row ``not_attempted`` — the claim T-E9's
    completeness gate reads, honest about what actually ran on this
    interpreter:

    - a phase test deleted or renamed → the length check fails;
    - a phase skipped at setup (``@pytest.mark.skipif``, the Python <3.11
      TLS-in-TLS path) → the hook never reaches that test's ``call``, so no
      entry is recorded and the length check fails;
    - a phase failed or runtime-skipped → the all-passed check fails;
    - a phase whose body passed but whose teardown errored → the
      teardown-outcomes check fails.

    **The floor shape is the one exception.** When the gate failed *because*
    the interpreter is below 3.11 — phase 1 ran and passed, the proxied
    phases are setup-skipped — the finaliser records ``UNSUPPORTED`` with
    the floor reason instead of leaving ``not_attempted``: the plan T-E3
    row's done-when ("an outcome is recorded") and T-E9's gate semantics
    both require it. A deleted/renamed phase or a real failure never takes
    this path; :func:`_floor_unsupported_shape` pins the difference.

    Before writing ``PROVEN``, the finaliser asserts the **other** rows are
    still ``not_attempted`` — the only rows the slice is allowed to mutate
    are its own verdict row and any sibling row whose **finaliser** has
    legitimately written ``PROVEN`` or (under the floor) ``UNSUPPORTED``.
    The checkable ceiling of this structural check is a rogue ``PROVEN`` on
    a sibling row, which the guard cannot distinguish from the legitimate
    finaliser write; that is the documented residual limit. A rogue
    ``FAILED``/``UNSUPPORTED`` write is still caught — only the ``PROVEN``
    coincidence is accepted.

    Yields:
        ``None``, with the verdict recording in the finaliser.
    """
    yield
    if _aiohttp_gate_passed():
        _record_proven_with_sibling_guard(
            _phase_outcomes,
            _phase_teardown_outcomes,
            _PHASE_TEST_NAMES,
            _VERDICT_ROW,
            _CURL_CFFI_VERDICT_ROW,
            _curl_gate_passed,
        )
        return
    # Gate did not pass. The floor-skip shape — phase 1 ran and the proxied
    # phases are setup-skipped on Python <3.11 — is the only other path to
    # satisfy the plan's T-E3 done-when ("an outcome is recorded") on every
    # supported interpreter; T-E9's gate accepts ``UNSUPPORTED`` as a
    # permitted partial-delivery verdict.
    _record_unsupported_if_floor_shape(_phase_outcomes, _phase_teardown_outcomes, _VERDICT_ROW)


def _record_proven_with_sibling_guard(
    outcomes: dict[str, _PhaseOutcome],
    teardown_outcomes: dict[str, _PhaseOutcome],
    names: frozenset[str],
    verdict_row: str,
    sibling_row: str,
    sibling_gate_passed_fn: Callable[[], bool],
) -> None:
    """Write the gate's ``PROVEN`` row, asserting no foreign row was mutated.

    The sibling-row exemption accepts ``PROVEN`` — and only ``PROVEN`` —
    when the sibling's gate also passed. A sibling row reading
    ``UNSUPPORTED`` never needs exempting here: this function is reached
    only through a passing gate, which on the reachable paths means the
    interpreter is ≥3.11 (below it the proxied phases are setup-skipped
    and the gate cannot pass), and the sibling's floor recorder no-ops on
    ≥3.11 — so a sibling ``UNSUPPORTED`` row cannot legitimately coexist
    with this ``PROVEN`` write, and if one is observed it is a defect the
    guard is right to surface.

    Order-independent: when the sibling's finaliser hasn't run yet, its
    row is ``NOT_ATTEMPTED`` and never enters the untouched list.

    Args:
        outcomes: The slice's call-phase outcomes.
        teardown_outcomes: The slice's teardown-phase outcomes.
        names: The slice's expected phase names.
        verdict_row: The capability-report row this slice writes.
        sibling_row: The capability-report row the sibling may write.
        sibling_gate_passed_fn: Zero-arg callable returning whether the
            sibling's gate passed.
    """
    if not _gate_passed(outcomes, teardown_outcomes, names):
        # Caller guards the PROVEN path; this assertion is the
        # belt-and-braces check for an unexpected internal call.
        raise AssertionError(f"{verdict_row!r} PROVEN path entered without a passing gate")
    entries = report_instance().entries()
    sibling = entries.get(sibling_row)
    sibling_legitimate = (
        sibling_gate_passed_fn()
        and sibling is not None
        and sibling.outcome is Outcome.PROVEN
    )
    untouched: list[str] = []
    for name, entry in entries.items():
        if name == verdict_row:
            continue
        if entry.outcome is Outcome.NOT_ATTEMPTED:
            continue
        if name == sibling_row and sibling_legitimate:
            continue
        untouched.append(name)
    if untouched:
        raise AssertionError(
            f"the slice must write only {verdict_row!r}, but sibling rows "
            f"{sorted(untouched)} were mutated: check the phase drives and any "
            "sibling slice that shares this process"
        )
    report_instance().record(verdict_row, Outcome.PROVEN)


@pytest.fixture(scope="session", autouse=True)
def _record_curl_cffi_slice_verdict_at_session_end() -> Iterator[None]:
    """Record ``Outcome.PROVEN`` for the T-E3 (``curl_cffi``) slice at session teardown.

    The T-E2 verdict gate's sibling; same contract: the verdict is a
    consequence of the phases having run and passed, not something a test
    writes on its own. The sibling-row guards on the two finalisers
    co-operate via the gate-passed and ``row == PROVEN`` exemption
    described on :func:`_record_slice_verdict_at_session_end` — order-
    independent, so either finaliser can run first without affecting the
    other.

    Yields:
        ``None``, with the verdict recording in the finaliser.
    """
    yield
    if _curl_gate_passed():
        _record_proven_with_sibling_guard(
            _curl_phase_outcomes,
            _curl_phase_teardown_outcomes,
            _CURL_CFFI_PHASE_TEST_NAMES,
            _CURL_CFFI_VERDICT_ROW,
            _VERDICT_ROW,
            _aiohttp_gate_passed,
        )
        return
    # Floor path — symmetric to the aiohttp finaliser above.
    _record_unsupported_if_floor_shape(_curl_phase_outcomes, _curl_phase_teardown_outcomes, _CURL_CFFI_VERDICT_ROW)


__all__ = [
    "_CURL_CFFI_PHASE_TEST_NAMES",
    "_CURL_CFFI_VERDICT_ROW",
    "_PHASE_TEST_NAMES",
    "_VERDICT_ROW",
    "_PhaseOutcome",
    "_curl_phase_outcomes",
    "_floor_unsupported_shape",
    "_phase_outcomes",
    "_phase_teardown_outcomes",
]
