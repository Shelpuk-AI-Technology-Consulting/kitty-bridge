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
  and write ``PROVEN`` iff their every phase is ``PASSED``.

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
from collections.abc import Generator, Iterator

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

    Before writing, the finaliser asserts the **other** rows are still
    ``not_attempted`` — the only rows the slice is allowed to mutate are
    its own verdict row and any sibling row whose **finaliser** has
    legitimately written ``PROVEN``. The checkable ceiling of this
    structural check is a rogue ``PROVEN`` on a sibling row, which the
    guard cannot distinguish from the legitimate finaliser write; that is
    the documented residual limit. A rogue ``FAILED``/``UNSUPPORTED`` write
    is still caught — only the ``PROVEN`` coincidence is accepted.

    Yields:
        ``None``, with the verdict recording in the finaliser.
    """
    yield
    if not _aiohttp_gate_passed():
        return
    entries = report_instance().entries()
    # The sibling-row exemption: when the T-E3 (curl_cffi) slice's gate
    # passed AND its row currently reads ``PROVEN``, the value is the
    # legitimate work of :func:`_record_curl_cffi_slice_verdict_at_session_end`
    # — exempt it. Order-independent: if T-E3's finaliser hasn't run yet,
    # its row is ``NOT_ATTEMPTED`` and never enters the untouched list.
    curl_row = entries.get(_CURL_CFFI_VERDICT_ROW)
    curl_legitimate = (
        _curl_gate_passed() and curl_row is not None and curl_row.outcome is Outcome.PROVEN
    )
    untouched: list[str] = []
    for name, entry in entries.items():
        if name == _VERDICT_ROW:
            continue
        if entry.outcome is Outcome.NOT_ATTEMPTED:
            continue
        if name == _CURL_CFFI_VERDICT_ROW and curl_legitimate:
            continue
        untouched.append(name)
    if untouched:
        raise AssertionError(
            f"the T-E2 slice must write only {_VERDICT_ROW!r}, but sibling rows "
            f"{sorted(untouched)} were mutated: check the phase drives and any "
            "sibling slice that shares this process"
        )
    report_instance().record(_VERDICT_ROW, Outcome.PROVEN)


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
    if not _curl_gate_passed():
        return
    entries = report_instance().entries()
    # Symmetric to :func:`_record_slice_verdict_at_session_end` — exempt
    # the T-E2 ``bridge_aiohttp`` row when its finaliser legitimately wrote
    # ``PROVEN``.
    aiohttp_row = entries.get(_VERDICT_ROW)
    aiohttp_legitimate = (
        _aiohttp_gate_passed()
        and aiohttp_row is not None
        and aiohttp_row.outcome is Outcome.PROVEN
    )
    untouched: list[str] = []
    for name, entry in entries.items():
        if name == _CURL_CFFI_VERDICT_ROW:
            continue
        if entry.outcome is Outcome.NOT_ATTEMPTED:
            continue
        if name == _VERDICT_ROW and aiohttp_legitimate:
            continue
        untouched.append(name)
    if untouched:
        raise AssertionError(
            f"the T-E3 slice must write only {_CURL_CFFI_VERDICT_ROW!r}, but sibling "
            f"rows {sorted(untouched)} were mutated: check the phase drives and any "
            "sibling slice that shares this process"
        )
    report_instance().record(_CURL_CFFI_VERDICT_ROW, Outcome.PROVEN)


__all__ = [
    "_CURL_CFFI_PHASE_TEST_NAMES",
    "_CURL_CFFI_VERDICT_ROW",
    "_PHASE_TEST_NAMES",
    "_VERDICT_ROW",
    "_PhaseOutcome",
    "_curl_phase_outcomes",
    "_phase_outcomes",
    "_phase_teardown_outcomes",
]
