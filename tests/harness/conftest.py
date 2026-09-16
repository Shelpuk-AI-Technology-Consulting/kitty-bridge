"""Harness-level pytest conftest: shared fixtures and the slice verdict gates.

This conftest applies to every test under :mod:`tests/harness/`. It serves
two purposes, both narrow:

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
  partial delivery") are both honoured on every supported interpreter.

The mechanism is what makes the verdicts honest. A test that records
``PROVEN`` itself would pass with the four phases deleted, and on
Python <3.11 — where phases 2/2b/3 skip because of TLS-in-TLS — would record
``PROVEN`` after phase 1 alone. The hook is a falsifiable witness to "every
phase ran" — the verdict cannot be written without that witness agreeing.

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
from collections.abc import Callable, Generator, Iterator

import pytest

from harness.containment import Outcome
from harness.containment import instance as report_instance


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
#: ``(file_prefix, phase_names, outcomes, teardown_outcomes)`` so a test's
#: phase outcome is dispatched to the right slice by its nodeid prefix —
#: the only signal that lets the hook tell the T-E2 and T-E3 slices apart,
#: since the two slices share method names by design.
_SLICES: tuple[
    tuple[str, frozenset[str], dict[str, _PhaseOutcome], dict[str, _PhaseOutcome]],
    ...,
] = (
    (_SLICE_FILE_PREFIX, _PHASE_TEST_NAMES, _phase_outcomes, _phase_teardown_outcomes),
    (_CURL_CFFI_SLICE_FILE_PREFIX, _CURL_CFFI_PHASE_TEST_NAMES, _curl_phase_outcomes, _curl_phase_teardown_outcomes),
)


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
    "Python <3.11: proxied §5.2.2 phases skip (TLS-in-TLS floor); phase 1 "
    "(direct leg) passed. Owner scope decision recorded on KBR-63 "
    "(2026-09-16): record `unsupported` — a permitted partial delivery "
    "T-E9's completeness gate accepts — rather than leave the row "
    "`not_attempted`, which the gate rejects."
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
        ``True`` when the row was recorded (the caller's normal ``PROVEN``
        path must then be skipped), ``False`` otherwise.
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
    for prefix, names, outcomes, teardown_outcomes in _SLICES:
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
