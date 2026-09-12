"""The assertion exemption registry, and the decisions taken over it.

``.system_design/TEST_SUITE.md`` §8 states the policy: a guard whose scope is
broader than the one defect it exposes may exempt **one named assertion** from a
gating job, and nothing else.  §8.3 specifies this mechanism.  Four contract
guards and one acceptance scenario (plan §16) are designed to land red on a
single assertion each, and none of them can land without it.

Three rules carry the whole of it:

* the exemption attaches to **one assertion**, delimited by a ``with`` block —
  setup, every ``Given`` step and every other assertion in the same test gate
  normally;
* only a failed assertion is amnestied — a broken fixture is not the known
  defect and fails the job;
* an **unexpected pass fails the job**, so an exemption cannot outlive the defect
  it describes.

Usage::

    from exemptions import ratchet

    with ratchet("tr-1c-header-subset"):
        assert bridge_headers <= native_headers

Every decision is a pure function — :func:`outcome_for`,
:func:`lookup_exemption`, :func:`registry_violations`,
:func:`unexpected_pass_message` — so each can be handed a deliberate defect, as
the implementation plan's §1.4 harness rule requires.  There is no pytest hook,
plugin or CI flag here: a test that uses an exemption is an ordinary test
carrying its ordinary layer marker, and no job's selection changes.

**Import note.** Imported as ``exemptions``, not ``tests.exemptions``: ``tests/``
has no ``__init__.py``, so pytest's ``prepend`` import mode puts that directory
on ``sys.path``.  :mod:`tests.layers` and ``tests/internal_key_scan.py`` are
imported the same way and the same caveat applies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, TracebackType

# Every failure this module raises is prefixed with this, so a CI summary can
# tell an exemption problem apart from the ordinary assertion failures around
# it -- all of which pytest reports the same way.
EXEMPTION_PREFIX = "assertion-exemption:"

# Jira keys in this project. Checked as a shape rather than against Jira: the
# suite must run offline, and a typo in the project prefix is the realistic
# mistake, not a well-formed key pointing at a closed ticket.
# `\Z`, not `$`: in Python `$` also matches before a trailing newline, so
# `"KBR-8\n"` would pass a `$`-anchored check.
_ISSUE_KEY = re.compile(r"^KBR-\d+\Z")

# Lowercase kebab-case. Enforced because five downstream tasks add rows
# independently and a registry whose ids are in three casings is a registry
# nobody greps successfully. Prefixing an id with its guard or scenario
# (`tr-1c-header-subset`, `t-g9-bedrock-user-agent`) is a convention and is NOT
# checked: an allow-list of prefixes would need editing for every new guard.
# TEST_SUITE.md §8.3 records that as a stated limit rather than a gap.
_EXEMPTION_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class Exemption:
    """One exempt assertion, and the reason it is exempt.

    Attributes:
        assertion: What the assertion claims, in the words a product owner would
            use — not its source text, which moves.
        condition: The condition under which it is expected to fail. This is the
            field that makes the row reviewable: a reader can ask whether the
            defect described is still the defect observed.
        issue: The Jira key of the defect, as ``KBR-<digits>``. An exemption
            without a ticket is an amnesty nobody owns.
    """

    assertion: str
    condition: str
    issue: str


# 🔴 The registry of ACKNOWLEDGED DEBT. One row per exempt assertion, and the
# list is supposed to trend towards zero.
#
# It shipped EMPTY until KBR-164. Every row below is the WINDOWS CELL of an
# assertion the platform legs found false on Windows and true on the other
# legs (TEST_SUITE.md §8.3, §8.4). Deliberately not counted in this comment:
# a count in prose is wrong the moment the next row lands, and nothing checks
# it -- the rule §8's own header states about naming things rather than
# counting them.
#
# Still unwritten, and still for the stated reason: TEST_SUITE.md §8 names
# TR-1c's header-subset assertion, pending G3's policy half (Q1) and keyed to
# KBR-8 until that shipped without closing parity -- but TR-1c is an acceptance
# scenario that does not exist yet (§6.4.1, plan task T-J2). A row for an
# assertion no test contains documents a fiction, and the unexpected-pass rule
# cannot catch that one, because nothing ever runs it.
#
# A `MappingProxyType`, not a bare dict, so the value matches the read-only
# annotation at runtime as well as to a type checker. `EXEMPTIONS[...] = ...`
# from a guard would otherwise pass `mypy` and register a row nobody can find by
# reading this file -- the one-registry invariant defeated by one line. Rows are
# added by editing the literal below.
#: Each row gates normally on the Linux and macOS legs — the parametrised-cell
#: shape §8.3 describes, not a blanket amnesty — so each fails the job the day
#: its own platform starts passing.
#:
#: They are exemptions rather than skips on purpose. A `skipif` here would hide
#: real defects behind a green leg, and §8 permits a platform skip only for
#: behaviour that *does not exist* on the platform. These assertions are not
#: inapplicable on Windows; they are **false** there, and that is a debt with a
#: ticket, not a platform difference.
#:
#: **The mechanism has now paid out once, which is worth recording.** KBR-188
#: held five rows here, all naming one cause: the recorder stamped `arrival`
#: with `time.monotonic()`, which on Windows under Python 3.12 is
#: `GetTickCount64()` at 15.625ms, so two adjacent requests shared a timestamp.
#: KBR-208 moved the stamp to `time.perf_counter()`, the five assertions began
#: passing on Windows, and every one of those rows failed the job through
#: `UnexpectedExemptionPass` — exactly the "fails the job the day its own
#: platform starts passing" contract above. That failure is the signal to
#: **delete the row**, not to widen it; all five are gone, along with the
#: `ratchet` plumbing at their call sites.
EXEMPTIONS: Mapping[str, Exemption] = MappingProxyType(
    {
        "recorder-responder-abort-emits-before-dropping": Exemption(
            assertion="a responder that aborts mid-stream still delivers a data frame first",
            condition="on Windows the abort wins the race against the first chunk, so the "
            "client sees response headers and no body",
            issue="KBR-189",
        ),
    }
)


class UnknownExemption(Exception):
    """Raised when a ``ratchet`` block names an id the registry does not hold.

    Descends from :class:`Exception` and deliberately not from
    :class:`AssertionError`, for the same reason as
    :class:`UnexpectedExemptionPass`: :func:`ratchet` resolves the row *outside*
    its own block, so an enclosing ``ratchet`` would otherwise classify a
    registry miss as its own expected failure and suppress it.
    """


class UnexpectedExemptionPass(Exception):
    """Raised when an exempt assertion passes, meaning its defect is fixed.

    Descends from :class:`Exception` and deliberately **not** from
    :class:`AssertionError`: were it an ``AssertionError``, a nested or adjacent
    ``ratchet`` block would classify it as its own expected failure and suppress
    it.  A strictness signal another exemption can swallow is not strictness.
    """


class Outcome(Enum):
    """What happened inside a ``ratchet`` block."""

    EXPECTED_FAILURE = "expected failure"
    UNEXPECTED_PASS = "unexpected pass"
    PROPAGATE = "propagate"


def outcome_for(error: BaseException | None) -> Outcome:
    """Classify what left a ``ratchet`` block.

    Args:
        error: The exception the block raised, or ``None`` when it completed.

    Returns:
        :attr:`Outcome.EXPECTED_FAILURE` for a failed assertion,
        :attr:`Outcome.UNEXPECTED_PASS` when nothing was raised, and
        :attr:`Outcome.PROPAGATE` for anything else.

    Only ``AssertionError`` is amnestied.  An exemption says "this claim is known
    to be false", not "this region of the test may do anything": a ``KeyError``
    raised while the assertion's subject is being built is a broken fixture, and
    suppressing it would let a guard report its known defect while proving
    nothing at all.  Subclasses count — pytest's rewriting and several libraries
    raise them, and an identity check would classify the same defect two
    different ways depending on who reported it.

    ``pytest.fail(...)``, ``pytest.xfail(...)`` and ``pytest.skip(...)`` are
    **not** amnestied: each raises a ``BaseException`` that is not an
    ``AssertionError``, so each propagates and fails the job.  That is the safe
    direction — an exemption must not be able to turn a test into a skip — but it
    surprises the author of a guard that reports its verdict with ``pytest.fail``
    and a diff, which is the established shape of the L2 guards in
    ``tests/test_github_actions.py`` and ``tests/test_pypi_packaging.py``.  A
    guard that needs an exemption states its verdict as an ``assert``.
    """
    if error is None:
        return Outcome.UNEXPECTED_PASS

    if isinstance(error, AssertionError):
        return Outcome.EXPECTED_FAILURE

    return Outcome.PROPAGATE


def lookup_exemption(exemption_id: str, registry: Mapping[str, Exemption]) -> Exemption:
    """Return the registry row an exemption names.

    Args:
        exemption_id: The id given to :func:`ratchet`.
        registry: The registry to resolve against.

    Returns:
        The matching :class:`Exemption`.

    Raises:
        UnknownExemption: When the registry holds no such id.

    An exemption that is not in the registry is precisely the scattered amnesty
    §8 forbids — it would grant cover with no ticket, no stated condition and
    nowhere to read how many are outstanding.  The message lists the ids that do
    exist, because the realistic cause is a typo or a row someone has already
    withdrawn.
    """
    if exemption_id in registry:
        return registry[exemption_id]

    # "known ids: " followed by nothing reads as a broken lookup. The registry
    # ships empty, so this is the message the first person to mistype an id
    # actually sees.
    known = ", ".join(sorted(registry)) if registry else "the registry is empty"

    raise UnknownExemption(
        f"{EXEMPTION_PREFIX} {exemption_id!r} is not in the registry, so it "
        f"exempts nothing. Known ids: {known}."
    )


def registry_violations(registry: Mapping[str, Exemption]) -> list[str]:
    """Return one human-readable line per malformed registry row.

    Args:
        registry: The registry to check.

    Returns:
        A message per offending row, naming the row and the field at fault.
        Empty when every row is complete.

    Every offender is reported rather than the first: a check that surfaces one
    problem per CI round is a check people stop running.
    """
    violations: list[str] = []

    for exemption_id, entry in registry.items():
        # The id is a row's only handle: it is what the `ratchet` call names and
        # what a maintainer greps for when the unexpected-pass failure fires.
        if not _EXEMPTION_ID.match(exemption_id):
            violations.append(
                f"{exemption_id!r} is not a usable exemption id; ids are "
                "lowercase kebab-case (and by convention, though not by this "
                "check, prefixed with the guard or scenario they belong to)"
            )

        if not entry.assertion.strip():
            violations.append(f"{exemption_id!r} names no assertion")

        # The condition is what makes the row reviewable later: without it a
        # reader cannot ask whether the defect described is still the one seen.
        if not entry.condition.strip():
            violations.append(f"{exemption_id!r} states no expected failure condition")

        if not _ISSUE_KEY.match(entry.issue):
            violations.append(
                f"{exemption_id!r} carries {entry.issue!r}, which is not a "
                "KBR-<number> issue key; an exemption without a ticket is an "
                "amnesty nobody owns"
            )

    return violations


def unexpected_pass_message(exemption_id: str, exemption: Exemption) -> str:
    """Build the failure text for an exempt assertion that has started passing.

    Args:
        exemption_id: The id the ``ratchet`` block named.
        exemption: Its registry row.

    Returns:
        The message :class:`UnexpectedExemptionPass` carries.

    Whoever reads this in CI did not write the exemption and may never have seen
    the ticket, so it states the assertion, the condition that was expected to
    break it, the issue, and the one action that clears it.
    """
    return (
        f"{EXEMPTION_PREFIX} the exempt assertion {exemption_id!r} passed. It "
        f"claims: {exemption.assertion}. It was expected to fail while: "
        f"{exemption.condition} ({exemption.issue}). The defect is fixed, so "
        "delete the row from the registry in tests/exemptions.py and let the "
        "assertion gate normally."
    )


class _Ratchet:
    """The context manager :func:`ratchet` returns.

    Written as an explicit ``__enter__``/``__exit__`` pair rather than with
    ``contextlib.contextmanager`` so that the three outcomes map one-to-one onto
    the return of :meth:`__exit__`: suppression is a returned ``True``, not a
    generator that happens not to re-raise.
    """

    def __init__(self, exemption_id: str, exemption: Exemption) -> None:
        """Store the resolved exemption.

        Args:
            exemption_id: The id the caller named.
            exemption: Its registry row, already resolved by :func:`ratchet`.
        """
        self._exemption_id = exemption_id
        self._exemption = exemption

    def __enter__(self) -> None:
        """Enter the exempt block.

        Returns:
            Nothing. The row is not handed back: no guard needs it, and a
            ``ratchet(...) as row`` form would invite the block to grow a second
            statement, which is the one thing the block must not do.
        """
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Apply the exemption to whatever left the block.

        Args:
            exc_type: The exception class, or ``None``.
            exc: The exception instance, or ``None``.
            traceback: The traceback, or ``None``.

        Returns:
            ``True`` to suppress the known defect, ``False`` to let anything else
            propagate.

        Raises:
            UnexpectedExemptionPass: When the block raised nothing, meaning the
                defect is fixed and the row must come out.
        """
        outcome = outcome_for(exc)

        if outcome is Outcome.UNEXPECTED_PASS:
            raise UnexpectedExemptionPass(
                unexpected_pass_message(self._exemption_id, self._exemption)
            )

        return outcome is Outcome.EXPECTED_FAILURE


def ratchet(exemption_id: str, *, _registry: Mapping[str, Exemption] | None = None) -> _Ratchet:
    """Exempt the single assertion inside the ``with`` block that follows.

    Args:
        exemption_id: The registry id of the assertion being exempted.
        _registry: The registry to resolve against, defaulting to
            :data:`EXEMPTIONS`. Underscored because it is a seam for this
            mechanism's own tests, which need rows while the production registry
            is empty. **A guard must never pass it**: a locally-supplied registry
            is an amnesty that no one can count by reading one file, which is the
            state §8 exists to prevent. Resolved in the body rather than as a
            default argument, so that rebinding :data:`EXEMPTIONS` takes effect
            instead of silently doing nothing.

    Returns:
        A context manager that suppresses the named assertion's failure and
        fails the test if it passes.

    Raises:
        UnknownExemption: When ``exemption_id`` is not in the registry. Raised
            **here**, before the block runs, rather than on the way out: a
            deferred lookup would let an exemption naming a row that does not
            exist pass silently for as long as its assertion kept failing.

    ``TEST_SUITE.md`` §8 writes this as ``@ratchet``, and pytest's own ``xfail``
    is a decorator.  A decorator is the wrong shape here, because it can only
    wrap a **whole test** — the scenario-wide amnesty §8 rejects, under which a
    broken fixture and every healthy assertion beside the exempt one all become
    invisible.  Assertion scope needs a block; the name is kept.
    """
    registry = EXEMPTIONS if _registry is None else _registry

    return _Ratchet(exemption_id, lookup_exemption(exemption_id, registry))
