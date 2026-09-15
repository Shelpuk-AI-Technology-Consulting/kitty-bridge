"""Harness-level pytest conftest: shared fixtures and the T-E2 verdict gate.

This conftest applies to every test under :mod:`tests/harness/`. It serves
two purposes, both narrow:

* the ``pytest_runtest_makereport`` hook populates a per-phase outcome
  dictionary :data:`_phase_outcomes`, used by the **T-E2** slice's verdict
  gate to record ``Outcome.PROVEN`` only when every §5.2.2 phase actually
  ran and passed on the current interpreter;
* the ``_record_slice_verdict_at_session_end`` session-scope fixture reads
  the dictionary at teardown and writes ``PROVEN`` iff every phase is
  ``PASSED``.

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

    Tri-state, not a bare bool, because a **skipped** test must not count as
    "passed": phases 2/2b/3 skip below Python 3.11 (TLS-in-TLS), and recording
    ``PROVEN`` there would let T-E9's completeness gate read a containment
    slice proven by one phase of four — the exact vacuous-verdict shape the
    capability report exists to prevent.
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

#: Populated by :func:`pytest_runtest_makereport` as phase tests complete.
#: Read by the session-finaliser verdict gate below.
_phase_outcomes: dict[str, _PhaseOutcome] = {}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, None, None]:
    """Track phase test outcomes; expose the call report on the item.

    Two narrow jobs:

    - store the ``call`` report on ``item.rep_call`` so a fixture finaliser
      can read what actually happened (the standard pytest hookwrapper
      idiom);
    - when the item is one of the T-E2 phase tests, record its outcome in
      :data:`_phase_outcomes` for the session-finaliser verdict gate.

    Args:
        item: The test item being reported.
        call: The call phase's information.
    """
    outcome = yield
    rep = outcome.get_result()  # type: ignore[attr-defined]
    if rep.when == "call":
        item.rep_call = rep  # type: ignore[attr-defined]
        if item.name in _PHASE_TEST_NAMES:
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
    fills in from real call reports. Deleting a phase test makes the gate's
    length check fail; skipping a phase (Python <3.11) records ``SKIPPED``
    and the gate's all-passed check fails. In both cases the singleton's
    ``bridge_aiohttp`` row stays ``not_attempted`` — the claim T-E9's
    completeness gate reads, and the one that is honest about what actually
    ran on this interpreter.

    Yields:
        ``None``, with the verdict recording in the finaliser.
    """
    yield
    if (
        len(_phase_outcomes) == len(_PHASE_TEST_NAMES)
        and all(outcome is _PhaseOutcome.PASSED for outcome in _phase_outcomes.values())
    ):
        report_instance().record("bridge_aiohttp", Outcome.PROVEN)


__all__ = [
    "_PHASE_TEST_NAMES",
    "_PhaseOutcome",
    "_phase_outcomes",
]
