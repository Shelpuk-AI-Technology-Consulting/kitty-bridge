"""Harness-level pytest conftest: shared fixtures and the T-E2 verdict gate.

This conftest applies to every test under :mod:`tests/harness/`. It serves
two purposes, both narrow:

* the ``pytest_runtest_makereport`` hook populates per-phase outcome
  dictionaries :data:`_phase_outcomes` (call phase) and
  :data:`_phase_teardown_outcomes` (teardown phase), used by the **T-E2**
  slice's verdict gate to record ``Outcome.PROVEN`` only when every §5.2.2
  phase actually ran, passed, and came back clean on the current
  interpreter;
* the ``_record_slice_verdict_at_session_end`` session-scope fixture reads
  the dictionaries at teardown, asserts the sibling rows are untouched, and
  writes ``PROVEN`` iff every phase is ``PASSED``.

The mechanism is what makes AC-R5 honest. A test that records ``PROVEN``
itself would pass with the four phases deleted, and on Python <3.11 — where
phases 2/2b/3 skip because of TLS-in-TLS — would record ``PROVEN`` after
phase 1 alone. The hook is a falsifiable witness to "every phase ran" — the
verdict cannot be written without that witness agreeing.

**A note for sibling slices (T-E3..T-E5).** The verdict recording is the
session-finaliser's job, and it runs **after** every test in the process,
not per file — so an autouse ``reset_for_test()`` in a sibling's test
module cannot erase a verdict that has not been written yet, and the
finaliser writes its own row regardless of what any earlier reset did.
The reset seam is for *per-test isolation* (each test sees an untouched
report), not for verdict erasure; sibling slices that follow this
template should put their own finaliser in this conftest (or in their own
scoped conftest) rather than calling ``record()`` from a test body.
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

#: The T-E4 (botocore) slice's parallel phase-name set, file prefix and
#: verdict row. The three are the botocore twin of the T-E2 trio above and
#: exist so the T-E4 slice can record its own ``PROVEN`` verdict without
#: sharing mutable state with T-E2's gate: a shared dict would let a
#: renamed botocore phase corrupt the aiohttp gate, and a shared file
#: prefix cannot distinguish the two slices' phase tests.
_BOTOCORE_PHASE_TEST_NAMES: frozenset[str] = frozenset({
    "test_with_egress_disabled_the_recorder_records_the_connection_and_the_proxy_sees_nothing",
    "test_proxy_down_leaves_the_recorder_with_zero_connections",
    "test_proxy_up_every_peer_port_joins_a_tunnel",
    "test_failed_tunnel_contributes_no_upstream_connection",
    "test_injected_bypass_makes_the_harness_report_it",
})

#: The nodeid prefix every T-E4 phase test starts with. Matched alongside
#: :data:`_BOTOCORE_PHASE_TEST_NAMES` for the same sibling-isolation reason
#: :data:`_SLICE_FILE_PREFIX` is matched for T-E2.
_BOTOCORE_SLICE_FILE_PREFIX = "tests/harness/test_botocore_containment_slice.py::"

#: The capability-report row the T-E4 slice's finaliser writes. Distinct
#: from :data:`_VERDICT_ROW` so the two slices' verdicts cannot overwrite
#: one another.
_BOTOCORE_VERDICT_ROW = "botocore"

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

#: The T-E4 twin of :data:`_phase_outcomes`. Keyed the same way, mutated by
#: the same hook, read only by the botocore half of the session finaliser.
_botocore_phase_outcomes: dict[str, _PhaseOutcome] = {}

#: The T-E4 twin of :data:`_phase_teardown_outcomes`.
_botocore_phase_teardown_outcomes: dict[str, _PhaseOutcome] = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """Track phase test outcomes for the slice verdict gates.

    Fires once per phase (setup / call / teardown) of every test in the
    process; only items that are both in a tracked slice file *and* named in
    that slice's phase-name set get recorded, so sibling slices and
    unrelated harness tests leave the dictionaries alone. Both the **call**
    and the **teardown** phase are recorded: a phase whose body passed but
    whose ``finally`` (say, a ``SealedNetwork.stop()`` drain) errored must
    not count as "the slice ran" for the verdict gate.

    Args:
        item: The test item being reported.
        call: The call phase's information.
    """
    outcome = yield
    rep = outcome.get_result()  # type: ignore[attr-defined]
    if rep.when not in ("call", "teardown"):
        return

    # One dispatch, two slices. Each row pairs (file prefix, name set,
    # call-phase dict, teardown-phase dict); a test matching neither is
    # left alone. `tuple` unpacking keeps the T-E2 and T-E4 arms literally
    # identical so a future T-E3/T-E5 row is a one-line append.
    for prefix, names, calls, teardowns in (
        (_SLICE_FILE_PREFIX, _PHASE_TEST_NAMES, _phase_outcomes, _phase_teardown_outcomes),
        (
            _BOTOCORE_SLICE_FILE_PREFIX,
            _BOTOCORE_PHASE_TEST_NAMES,
            _botocore_phase_outcomes,
            _botocore_phase_teardown_outcomes,
        ),
    ):
        if not item.nodeid.startswith(prefix) or item.name not in names:
            continue
        if rep.passed:
            verdict = _PhaseOutcome.PASSED
        elif rep.skipped:
            verdict = _PhaseOutcome.SKIPPED
        else:
            verdict = _PhaseOutcome.FAILED
        if rep.when == "call":
            calls[item.name] = verdict
        else:
            teardowns[item.name] = verdict


@pytest.fixture(scope="session", autouse=True)
def _record_slice_verdict_at_session_end() -> Iterator[None]:
    """Record ``Outcome.PROVEN`` at session teardown iff every phase actually passed.

    The verdict is a **consequence** of the phases having run and passed, not
    something a test can write on its own behalf: the finaliser reads the
    per-slice outcome dictionaries, which only
    :func:`pytest_runtest_makereport` fills in from real call reports. Four
    failure shapes, all of which leave a slice's row ``not_attempted`` — the
    claim T-E9's completeness gate reads, honest about what actually ran on
    this interpreter:

    - a phase test deleted or renamed → the length check fails;
    - a phase skipped at setup (``@pytest.mark.skipif``, the Python <3.11
      TLS-in-TLS path) → the hook never reaches that test's ``call``, so no
      entry is recorded and the length check fails;
    - a phase failed or runtime-skipped → the all-passed check fails;
    - a phase whose body passed but whose teardown errored → the
      teardown-outcomes check fails.

    Each slice's verdict is written independently: the T-E2 gate reads
    :data:`_phase_outcomes` / :data:`_phase_teardown_outcomes` and writes
    :data:`_VERDICT_ROW`, and the T-E4 gate reads the botocore pair and
    writes :data:`_BOTOCORE_VERDICT_ROW`. One slice's phases skipping on an
    interpreter therefore records ``PROVEN`` for the other slice that did
    run, which is what T-E9 wants to see.

    Before writing, the finaliser asserts the sibling rows it is not about
    to write are still ``not_attempted`` — the only rows this module is
    allowed to mutate are :data:`_VERDICT_ROW` and
    :data:`_BOTOCORE_VERDICT_ROW`, and a sibling-row write is a defect this
    assertion surfaces rather than lets slip into T-E9's gate.

    Yields:
        ``None``, with the verdict recording in the finaliser.
    """

    def _slice_passed(
        calls: dict[str, _PhaseOutcome], names: frozenset[str], teardowns: dict[str, _PhaseOutcome]
    ) -> bool:
        """Return whether every phase test in ``names`` ran and passed cleanly.

        Args:
            calls: The call-phase outcomes, keyed on test name.
            names: The phase-name set the slice declared.
            teardowns: The teardown-phase outcomes, same keys as ``calls``.

        Returns:
            ``True`` only when the length check, the all-passed check and
            the teardown-clean check all hold.
        """
        return (
            len(calls) == len(names)
            and all(outcome is _PhaseOutcome.PASSED for outcome in calls.values())
            # A teardown failure after a passing body still means the test
            # did not come back clean, so the slice cannot claim "ran and
            # passed".
            and all(outcome is _PhaseOutcome.PASSED for outcome in teardowns.values())
        )

    yield

    # (phase names, call dict, teardown dict, verdict row) per slice. Two
    # independent gates over one report: a slice whose phases all skipped
    # records nothing for its own row but does not disturb the row of a
    # slice that did run on this interpreter.
    for names, calls, teardowns, row in (
        (_PHASE_TEST_NAMES, _phase_outcomes, _phase_teardown_outcomes, _VERDICT_ROW),
        (
            _BOTOCORE_PHASE_TEST_NAMES,
            _botocore_phase_outcomes,
            _botocore_phase_teardown_outcomes,
            _BOTOCORE_VERDICT_ROW,
        ),
    ):
        if not _slice_passed(calls, names, teardowns):
            continue
        writable = {_VERDICT_ROW, _BOTOCORE_VERDICT_ROW}
        entries = report_instance().entries()
        untouched = [
            name
            for name, entry in entries.items()
            if name not in writable and entry.outcome is not Outcome.NOT_ATTEMPTED
        ]
        if untouched:
            raise AssertionError(
                f"the slice finalisers may only write {sorted(writable)}, but sibling rows "
                f"{sorted(untouched)} were mutated: check the phase drives and any "
                "sibling slice that shares this process"
            )
        report_instance().record(row, Outcome.PROVEN)


__all__ = [
    "_BOTOCORE_PHASE_TEST_NAMES",
    "_BOTOCORE_SLICE_FILE_PREFIX",
    "_BOTOCORE_VERDICT_ROW",
    "_PHASE_TEST_NAMES",
    "_SLICE_FILE_PREFIX",
    "_VERDICT_ROW",
    "_PhaseOutcome",
    "_botocore_phase_outcomes",
    "_botocore_phase_teardown_outcomes",
    "_phase_outcomes",
    "_phase_teardown_outcomes",
]
