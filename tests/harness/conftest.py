"""Harness-level pytest conftest: shared fixtures and the T-E2 verdict gate.

This conftest applies to every test under :mod:`tests/harness/`. It serves
two purposes, both narrow:

* the ``pytest_runtest_makereport`` hook populates a per-phase outcome
  dictionary :data:`_phase_outcomes`, used by the **T-E2** slice's verdict
  gate to record ``Outcome.PROVEN`` only when every §5.2.2 phase actually
  ran and passed on the current interpreter;
* the ``_record_slice_verdict_at_session_end`` session-scope fixture reads
  the dictionary at teardown, asserts the sibling rows are untouched, and
  writes ``PROVEN`` iff every phase is ``PASSED``.

The mechanism is what makes AC-R5 honest. A test that records ``PROVEN``
itself would pass with the four phases deleted, and on Python <3.11 — where
phases 2/2b/3 skip because of TLS-in-TLS — would record ``PROVEN`` after
phase 1 alone. The hook is a falsifiable witness to "every phase ran" — the
verdict cannot be written without that witness agreeing.
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

#: Populated by :func:`pytest_runtest_makereport` as phase tests complete.
#: Read by the session-finaliser verdict gate below.
_phase_outcomes: dict[str, _PhaseOutcome] = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """Track phase test outcomes for the slice verdict gate.

    Fires once per phase (setup / call / teardown) of every test in the
    process; only items that are both in the T-E2 slice file *and* named in
    :data:`_PHASE_TEST_NAMES` get recorded, so sibling slices and unrelated
    harness tests leave the dictionary alone.

    Args:
        item: The test item being reported.
        call: The call phase's information.
    """
    outcome = yield
    rep = outcome.get_result()  # type: ignore[attr-defined]
    if rep.when != "call":
        return
    if not item.nodeid.startswith(_SLICE_FILE_PREFIX) or item.name not in _PHASE_TEST_NAMES:
        return
    if rep.passed:
        _phase_outcomes[item.name] = _PhaseOutcome.PASSED
    elif rep.skipped:
        _phase_outcomes[item.name] = _PhaseOutcome.SKIPPED
    else:
        _phase_outcomes[item.name] = _PhaseOutcome.FAILED


@pytest.fixture(scope="session", autouse=True)
def _record_slice_verdict_at_session_end() -> Iterator[None]:
    """Record ``Outcome.PROVEN`` at session teardown iff every phase actually passed.

    The verdict is a **consequence** of the phases having run and passed, not
    something a test can write on its own behalf: the finaliser reads
    :data:`_phase_outcomes`, which only :func:`pytest_runtest_makereport`
    fills in from real call reports. Three failure shapes, all of which leave
    the singleton's ``bridge_aiohttp`` row ``not_attempted`` — the claim
    T-E9's completeness gate reads, honest about what actually ran on this
    interpreter:

    - a phase test deleted or renamed → the length check fails;
    - a phase skipped at setup (``@pytest.mark.skipif``, the Python <3.11
      TLS-in-TLS path) → the hook never reaches that test's ``call``, so no
      entry is recorded and the length check fails;
    - a phase failed or runtime-skipped → the all-passed check fails.

    Before writing, the finaliser also asserts the sibling rows are still
    ``not_attempted`` — the only row this module is allowed to mutate is
    :data:`_VERDICT_ROW`, and a sibling-row write is a defect this assertion
    surfaces rather than lets slip into T-E9's gate.

    Yields:
        ``None``, with the verdict recording in the finaliser.
    """
    yield
    if (
        len(_phase_outcomes) == len(_PHASE_TEST_NAMES)
        and all(outcome is _PhaseOutcome.PASSED for outcome in _phase_outcomes.values())
    ):
        entries = report_instance().entries()
        untouched = [
            name
            for name, entry in entries.items()
            if name != _VERDICT_ROW and entry.outcome is not Outcome.NOT_ATTEMPTED
        ]
        if untouched:
            raise AssertionError(
                f"the T-E2 slice must write only {_VERDICT_ROW!r}, but sibling rows "
                f"{sorted(untouched)} were mutated: check the phase drives and any "
                "sibling slice that shares this process"
            )
        report_instance().record(_VERDICT_ROW, Outcome.PROVEN)


__all__ = [
    "_PHASE_TEST_NAMES",
    "_VERDICT_ROW",
    "_PhaseOutcome",
    "_phase_outcomes",
]
