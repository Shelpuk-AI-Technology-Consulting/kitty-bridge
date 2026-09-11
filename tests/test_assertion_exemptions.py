"""Tests for the assertion exemption registry.

``.system_design/TEST_SUITE.md`` §8.3 specifies a mechanism that exempts **one
named assertion** from a gating job, so that a guard whose scope is broader than
the one defect it exposes can land without turning the whole gate red.

The two cases the plan calls falsification (§1.4, plan task T-W7) are the pair
that separate this mechanism from the blanket amnesty §8 rejects:

* :func:`test_ratchet_does_not_suppress_a_later_failing_assertion` — an exempt
  assertion beside a second, healthy one must still fail the job;
* :func:`test_ratchet_fails_when_the_exempt_assertion_passes` — an exemption
  that has outlived its defect must fail the job.

Both are asserted **in process** rather than through a ``pytester`` sub-run. The
claim is "the exemption suppresses only what is inside its block", and an
exception escaping the block *is* how pytest fails a test; a second interpreter
would prove nothing further and would couple these tests to pytest's own plugin
loading.

This module is ``l1`` by the path default — pure functions and a context manager
over them, no sockets, no files, no clock. It declares no layer marker because
the default is right.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from exemptions import (
    EXEMPTION_PREFIX,
    EXEMPTIONS,
    Exemption,
    Outcome,
    UnexpectedExemptionPass,
    UnknownExemption,
    lookup_exemption,
    outcome_for,
    ratchet,
    registry_violations,
    unexpected_pass_message,
)

# A stand-in for the production registry. The falsification cases need rows, and
# the production registry ships empty (§8.3); tying these tests to whatever the
# first real row happens to be would make them fail the day that row is
# withdrawn, which is the opposite of what this mechanism is for.
SAMPLE_REGISTRY: Mapping[str, Exemption] = {
    "sample-only-header-subset": Exemption(
        assertion="the header set is a subset of what the agent sends natively",
        condition="the bridge adds a header the agent does not send",
        issue="KBR-8",
    ),
    "sample-only-inner": Exemption(
        assertion="the bridge opens one connection per session",
        condition="connection reuse differs from the native baseline",
        issue="KBR-8",
    ),
}

# Shaped like the real TR-1c assertion, so the tests below read as the thing
# being exempted rather than as `assert False` with extra steps. `assert False`
# is also a ruff B011 finding.
BRIDGE_ONLY_HEADERS = frozenset({"x-kitty-title"})
NATIVE_HEADERS: frozenset[str] = frozenset()


# ── The decisions, as pure functions ───────────────────────────────────────


def test_outcome_for_no_error_is_an_unexpected_pass() -> None:
    """An exempt assertion that raises nothing has outlived its defect."""
    assert outcome_for(None) is Outcome.UNEXPECTED_PASS


def test_outcome_for_an_assertion_error_is_the_expected_failure() -> None:
    """A failed assertion inside the block is the known defect."""
    assert outcome_for(AssertionError("the known defect")) is Outcome.EXPECTED_FAILURE


def test_outcome_for_an_assertion_error_subclass_is_the_expected_failure() -> None:
    """A subclass is still a failed assertion.

    pytest's assertion rewriting and several libraries raise subclasses of
    ``AssertionError``; an identity check on the type would amnesty the bare
    class and propagate every one of those, which is the same defect reported
    two different ways depending on who raised it.
    """

    class _RewrittenAssertionError(AssertionError):
        """A stand-in for a library's own ``AssertionError`` subclass."""

    assert outcome_for(_RewrittenAssertionError()) is Outcome.EXPECTED_FAILURE


@pytest.mark.parametrize(
    "error",
    [KeyError("headers"), RuntimeError("the bridge never started"), TypeError("bad fixture")],
    ids=["KeyError", "RuntimeError", "TypeError"],
)
def test_outcome_for_any_other_exception_propagates(error: Exception) -> None:
    """Anything that is not a failed assertion is not the known defect.

    Args:
        error: An exception a broken fixture might raise inside the block.
    """
    assert outcome_for(error) is Outcome.PROPAGATE


def test_unexpected_exemption_pass_is_not_an_assertion_error() -> None:
    """The strictness signal must not be amnestiable by another exemption.

    ``UnexpectedExemptionPass`` descends from ``Exception`` deliberately: were it
    an ``AssertionError``, a nested or adjacent ``ratchet`` block would classify
    it as its own expected failure and suppress it. A strictness signal that
    another exemption can swallow is not strictness.
    """
    assert not issubclass(UnexpectedExemptionPass, AssertionError)
    # And the registry miss, for the same reason: `ratchet()` resolves the row
    # outside its own block, so an enclosing exemption would otherwise amnesty
    # an unknown id and the registry would stop being the only source of them.
    assert not issubclass(UnknownExemption, AssertionError)


def test_the_unexpected_pass_message_names_the_id_the_issue_and_the_remedy() -> None:
    """The failure has to tell a maintainer with no context what to do.

    Whoever reads this in CI did not write the exemption and may never have seen
    the ticket. Naming only the id would send them grepping.
    """
    row = SAMPLE_REGISTRY["sample-only-header-subset"]

    message = unexpected_pass_message("sample-only-header-subset", row)

    # The literal, not `EXEMPTION_PREFIX in message`: that form is satisfied by
    # a prefix someone has emptied out.
    assert message.startswith("assertion-exemption:")
    assert EXEMPTION_PREFIX == "assertion-exemption:"
    assert "sample-only-header-subset" in message
    assert "KBR-8" in message
    assert "the header set is a subset of what the agent sends natively" in message
    # The remedy, not just the diagnosis, and named precisely: `"registry" in
    # message` is satisfied by any sentence that happens to use the word.
    assert "delete the row" in message
    assert "tests/exemptions.py" in message


def test_lookup_exemption_returns_the_named_row() -> None:
    """A known id resolves to its row."""
    row = SAMPLE_REGISTRY["sample-only-header-subset"]

    assert lookup_exemption("sample-only-header-subset", SAMPLE_REGISTRY) is row


def test_lookup_exemption_rejects_an_unknown_id() -> None:
    """An exemption that is not in the registry is a scattered amnesty.

    The message lists the ids that do exist, because the overwhelmingly likely
    cause is a typo or a row someone has already withdrawn.
    """
    with pytest.raises(UnknownExemption) as caught:
        lookup_exemption("no-such-id", SAMPLE_REGISTRY)

    assert "no-such-id" in str(caught.value)
    assert "sample-only-header-subset" in str(caught.value)


def test_lookup_exemption_says_so_when_the_registry_is_empty() -> None:
    """An empty registry is the goal state, and must read as one.

    Listing "known ids: " and nothing else invites the reader to conclude the
    lookup is broken. The registry ships empty (§8.3), so this is the message
    the first person to mistype an id will actually see.
    """
    with pytest.raises(UnknownExemption) as caught:
        lookup_exemption("sample-only-header-subset", {})

    assert "empty" in str(caught.value)


# ── The registry's own shape ───────────────────────────────────────────────


def test_registry_violations_passes_a_well_formed_registry() -> None:
    """A registry whose rows are all complete has no violations."""
    assert registry_violations(SAMPLE_REGISTRY) == []


@pytest.mark.parametrize(
    ("exemption_id", "row", "expected_fragment"),
    [
        (
            "sample-only-a",
            Exemption(assertion="  ", condition="a condition", issue="KBR-8"),
            "assertion",
        ),
        (
            "sample-only-b",
            Exemption(assertion="an assertion", condition="", issue="KBR-8"),
            "condition",
        ),
        (
            "no-ticket",
            Exemption(assertion="an assertion", condition="a condition", issue=""),
            "issue key",
        ),
        (
            "wrong-project",
            Exemption(assertion="an assertion", condition="a condition", issue="JIRA-8"),
            "issue key",
        ),
        (
            "   ",
            Exemption(assertion="an assertion", condition="a condition", issue="KBR-8"),
            "id",
        ),
        (
            "TR_1c_Header_Subset",
            Exemption(assertion="an assertion", condition="a condition", issue="KBR-8"),
            "kebab-case",
        ),
        (
            "sample-only-c",
            Exemption(
                assertion="an assertion", condition="a condition", issue="KBR-8 (wontfix)"
            ),
            "issue key",
        ),
        (
            "sample-only-d",
            Exemption(assertion="an assertion", condition="a condition", issue="KBR-8\n"),
            "issue key",
        ),
        (
            "t-g9-ok\n",
            Exemption(assertion="an assertion", condition="a condition", issue="KBR-8"),
            "id",
        ),
    ],
    ids=[
        "blank assertion",
        "blank condition",
        "no ticket",
        "wrong project",
        "blank id",
        "id in another convention",
        "issue key with trailing text",
        "issue key with a trailing newline",
        "id with a trailing newline",
    ],
)
def test_registry_violations_flags_a_malformed_row(
    exemption_id: str, row: Exemption, expected_fragment: str
) -> None:
    """Every field an entry must carry is checked, and named when it is missing.

    Args:
        exemption_id: The row's key in the registry.
        row: A row with exactly one thing wrong with it.
        expected_fragment: The word the violation line must contain, so that the
            test fails when the checker reports the wrong field. The ids are
            deliberately anonymous: every violation line embeds the id, so an id
            like ``"blank-condition"`` would satisfy the fragment on its own and
            a checker naming the wrong field would pass. The two trailing-newline
            rows are not padding: Python's ``$`` matches *before* a trailing
            newline, so a ``$``-anchored pattern accepts ``"KBR-8\n"``.

    This is deliberately proved against a **fabricated** malformed registry. The
    production registry ships empty, and a validator run over zero rows passes
    perfectly — §8's "green because it stopped looking", reproduced inside the
    check meant to prevent it.
    """
    violations = registry_violations({exemption_id: row})

    assert len(violations) == 1
    assert expected_fragment in violations[0]


def test_registry_violations_accepts_an_id_that_names_no_guard() -> None:
    """The prefix convention is not a check, and the suite says which it is.

    ``TEST_SUITE.md`` §8.3 lists it among the limits the mechanism cannot
    enforce. Without this test the omission reads as an oversight in
    ``registry_violations`` rather than as the recorded decision it is.
    """
    unprefixed = {
        "header-subset": Exemption(assertion="a", condition="c", issue="KBR-8"),
    }

    assert registry_violations(unprefixed) == []


def test_registry_violations_reports_every_offender() -> None:
    """All of them, not the first.

    At ~18 minutes a CI round, a check that surfaces one problem per round is a
    check people stop running.
    """
    violations = registry_violations(
        {
            "blank-assertion": Exemption(assertion="", condition="c", issue="KBR-8"),
            "no-ticket": Exemption(assertion="a", condition="c", issue=""),
        }
    )

    assert len(violations) == 2


def test_the_production_registry_cannot_be_added_to_at_runtime() -> None:
    """A guard cannot register its own row by assigning into the registry.

    The annotation says :class:`~collections.abc.Mapping`, which a type checker
    enforces and a plain ``dict`` does not. Rows are added by editing the
    literal in ``tests/exemptions.py``, where the list stays readable in one
    place — §8's whole reason for having a registry.
    """
    with pytest.raises(TypeError):
        EXEMPTIONS["smuggled-in"] = Exemption(  # type: ignore[index]
            assertion="an assertion", condition="a condition", issue="KBR-8"
        )

    assert "smuggled-in" not in EXEMPTIONS


def test_the_production_registry_is_well_formed() -> None:
    """Whatever rows the registry carries today, they are complete.

    A weaker claim than the test above and deliberately separate from it: this
    one passes vacuously while the registry is empty, which is exactly why it
    cannot be the only place the checker is exercised.
    """
    assert registry_violations(EXEMPTIONS) == []


# ── The mechanism ──────────────────────────────────────────────────────────
#
# Every case below asserts the EXACT type of what escapes, through a
# `BaseException` guard rather than through `pytest.raises`. That is not
# pedantry. `pytest.xfail()`, `pytest.skip()` and `pytest.fail()` raise
# `BaseException` subclasses that `pytest.raises(AssertionError)` does not
# catch, and the first two turn the surrounding test GREEN. A plausible wrong
# implementation -- suppress the assertion, then call `pytest.xfail()` -- is a
# test-wide amnesty, exactly what this mechanism exists to forbid, and it was
# observed passing an earlier version of these tests with the suite reporting
# "23 passed, 2 xfailed" and exit code 0.


def escaping_exception(body: Callable[[], None]) -> BaseException | None:
    """Run ``body`` and return whatever left it, including pytest's outcomes.

    Args:
        body: A zero-argument callable standing in for a guard's test body.

    Returns:
        The exception that escaped, or ``None`` when the body returned
        normally.

    Catches ``BaseException`` deliberately: ``pytest.xfail`` and ``pytest.skip``
    are how a wrong implementation would make a failing job look green, and a
    narrower guard cannot see them.
    """
    try:
        body()
    except BaseException as error:
        return error

    return None


def test_ratchet_suppresses_the_failure_of_the_assertion_it_names() -> None:
    """The known defect does not fail the job, and does not skip it either."""

    def a_guard_whose_only_failing_assertion_is_exempt() -> None:
        """Stand in for a guard that has caught nothing but its known defect."""
        with ratchet("sample-only-header-subset", _registry=SAMPLE_REGISTRY):
            assert BRIDGE_ONLY_HEADERS <= NATIVE_HEADERS, "the known defect"

    assert escaping_exception(a_guard_whose_only_failing_assertion_is_exempt) is None


def test_ratchet_does_not_suppress_a_later_failing_assertion() -> None:
    """Falsification 1 — an exemption covers one assertion, not the test.

    A scenario-wide exemption would make a broken fixture, a failing ``Given``
    step and an unrelated regression all invisible, indistinguishable from the
    known defect. That is worse than the red gate it was meant to avoid, so the
    mechanism is not fit to use until this case is observed failing.
    """

    def a_guard_with_one_exempt_and_one_healthy_assertion() -> None:
        """Stand in for a T-G9-shaped guard that has caught a real regression."""
        with ratchet("sample-only-header-subset", _registry=SAMPLE_REGISTRY):
            assert BRIDGE_ONLY_HEADERS <= NATIVE_HEADERS, "the known defect"

        assert [] == ["the turn"], "an unrelated regression"

    escaped = escaping_exception(a_guard_with_one_exempt_and_one_healthy_assertion)

    # The exact type, not `pytest.raises(AssertionError)`: an implementation
    # that xfailed the whole test would raise `XFailed`, which is not an
    # `AssertionError` and which reports the job green.
    assert type(escaped) is AssertionError
    assert "an unrelated regression" in str(escaped)


def test_ratchet_does_not_suppress_a_broken_fixture_inside_the_block() -> None:
    """An exemption says a claim is known false, not that anything may happen.

    A ``KeyError`` raised where the assertion's subject is being built is a
    broken fixture. Amnestying it would let the guard report the known defect
    while proving nothing at all.
    """
    captured: dict[str, str] = {}

    def a_guard_whose_fixture_is_broken() -> None:
        """Stand in for a guard whose capture never arrived."""
        with ratchet("sample-only-header-subset", _registry=SAMPLE_REGISTRY):
            assert captured["x-kitty-title"] == "absent"

    assert type(escaping_exception(a_guard_whose_fixture_is_broken)) is KeyError


def test_ratchet_does_not_amnesty_a_pytest_fail_verdict() -> None:
    """``pytest.fail`` is not an ``AssertionError``, so it is not exempt.

    The L2 guards this mechanism was built for report their verdict with
    ``pytest.fail`` and a diff today. The safe direction is the one taken — an
    exemption must never be able to convert a verdict into a pass — but the
    behaviour is pinned here so a downstream author finds it stated rather than
    discovers it in CI.
    """

    def a_guard_that_reports_its_verdict_with_fail() -> None:
        """Stand in for a T-G1-shaped README guard."""
        with ratchet("sample-only-header-subset", _registry=SAMPLE_REGISTRY):
            pytest.fail("the README table and the code disagree")

    assert type(escaping_exception(a_guard_that_reports_its_verdict_with_fail)) is pytest.fail.Exception


def test_ratchet_fails_when_the_exempt_assertion_passes() -> None:
    """Falsification 2 — an exemption cannot outlive the defect it describes.

    ``xfail(strict=True)`` semantics: when the assertion starts passing, the
    defect is fixed, and nobody has to remember to take the row out.
    """

    def a_guard_whose_defect_has_been_fixed() -> None:
        """Stand in for a guard the day its ticket closes."""
        with ratchet("sample-only-header-subset", _registry=SAMPLE_REGISTRY):
            assert NATIVE_HEADERS <= BRIDGE_ONLY_HEADERS, "the defect is fixed"

    escaped = escaping_exception(a_guard_whose_defect_has_been_fixed)

    assert type(escaped) is UnexpectedExemptionPass
    assert "KBR-8" in str(escaped)


def test_an_outer_exemption_cannot_swallow_an_inner_unexpected_pass() -> None:
    """The strictness signal survives being nested inside another exemption.

    This is the behaviour :func:`test_unexpected_exemption_pass_is_not_an_assertion_error`
    pins the type relationship for. Were ``UnexpectedExemptionPass`` an
    ``AssertionError``, the outer block would classify it as its own expected
    failure and suppress it — one fixed defect would hide another's exemption.
    """

    def two_nested_exemptions() -> None:
        """Stand in for a guard whose inner defect closed first."""
        with ratchet("sample-only-header-subset", _registry=SAMPLE_REGISTRY):
            with ratchet("sample-only-inner", _registry=SAMPLE_REGISTRY):
                assert NATIVE_HEADERS <= BRIDGE_ONLY_HEADERS, "the inner defect is fixed"

            assert BRIDGE_ONLY_HEADERS <= NATIVE_HEADERS, "the outer known defect"

    escaped = escaping_exception(two_nested_exemptions)

    assert type(escaped) is UnexpectedExemptionPass
    assert "sample-only-inner" in str(escaped)


def test_ratchet_rejects_an_unknown_id_before_running_the_block() -> None:
    """A registry miss must not be masked by the assertion failing anyway.

    Were the lookup deferred to the end of the block, an exemption naming a row
    that does not exist would pass silently for as long as its assertion kept
    failing — an amnesty granted by nothing, which is the state the registry
    exists to make impossible.
    """
    entered = False

    def a_guard_naming_a_row_that_does_not_exist() -> None:
        """Stand in for a guard whose exemption was withdrawn under it."""
        nonlocal entered
        with ratchet("no-such-id", _registry=SAMPLE_REGISTRY):
            entered = True
            assert BRIDGE_ONLY_HEADERS <= NATIVE_HEADERS, "the known defect"

    assert type(escaping_exception(a_guard_naming_a_row_that_does_not_exist)) is UnknownExemption
    assert not entered


def test_ratchet_defaults_to_the_production_registry() -> None:
    """The registry argument is a private test seam, not the calling convention.

    Guards write ``with ratchet("some-id"):`` and get the one registry §8
    requires; only this module passes its own, and the underscore says so.

    The precondition is derived from :data:`SAMPLE_REGISTRY` rather than written
    as a literal, so it keeps holding when a sample row is renamed and covers
    every sample row rather than this one.
    """
    assert not (SAMPLE_REGISTRY.keys() & EXEMPTIONS.keys())

    def a_guard_using_the_calling_convention() -> None:
        """Stand in for a guard, which never passes a registry of its own."""
        with ratchet("sample-only-header-subset"):
            pass

    assert type(escaping_exception(a_guard_using_the_calling_convention)) is UnknownExemption
