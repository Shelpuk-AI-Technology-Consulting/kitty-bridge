"""The two-arm eval harness skeleton (KBR-110, plan task **T-K1**).

`TEST_SUITE.md` §6.4.3 specifies the paired-delta eval methodology and
`TEST_SUITE_IMPLEMENTATION_PLAN.md` §13 scopes what the skeleton owns:
pinning, two arms, the failure taxonomy, and the denominator. The
statistics and the decision rule are plan task **T-K3** and are
deliberately out of scope here.

The harness is **pure except for the injected ``clock`` keyword
argument** — it does no IO, launches no subprocess, touches no real
provider. The arm executor is the seam that does; whoever wires the
nightly (plan task T-K12) supplies it. Keeping the skeleton pure is
what lets its load-bearing rules — pinning, taxonomy, the denominator —
be judged at L1 and handed a deliberate defect (§1.4) without starting
a server.

**Import rule.** This module imports nothing from ``src/kitty``. The
independence precedent and its rationale are in
``tests/harness/failures.py``; the structural guard is
``test_the_eval_harness_imports_nothing_from_kitty`` in the file
beside this one. ``bridge.py`` is the one module in this package
permitted to import the product; this is not it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, TypeAlias

__all__ = [
    "ArmExecutor",
    "ArmSpec",
    "JSONValue",
    "ModelReply",
    "RawOutcome",
    "RunConfig",
    "TaskSpec",
    "TimedOut",
    "TrialCategory",
    "UnclassifiedError",
    "UpstreamFailure",
    "UpstreamRefusal",
    "Verdict",
    "classify",
    "validate_arms",
]

#: The shape ``sampling_overrides`` accepts: any JSON-serialisable scalar
#: or container. Named here so the field annotation and the constructor's
#: serialisability check describe one type, not two.
JSONValue: TypeAlias = "str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]"


class Verdict(Enum):
    """What an acceptance check returns for a single ``ModelReply``.

    Three values, deliberately closed: ``PASS`` (the answer is right),
    ``FAIL`` (the answer is wrong), ``REFUSED`` (the model declined or
    produced no usable answer). ``REFUSED`` is its own value rather
    than a special ``FAIL`` because §6.4.3 names it separately — a
    refusal is **evidence about the arm**, not just a wrong answer.
    """

    PASS = "pass"
    FAIL = "fail"
    REFUSED = "refused"


class TrialCategory(Enum):
    """The §6.4.3 failure taxonomy, plus ``SUCCESS`` and ``FAILED_ACCEPTANCE``.

    Every trial in a run lands in exactly one of these. ``SUCCESS`` and
    ``FAILED_ACCEPTANCE`` are the two paths a ``ModelReply`` can take
    after the acceptance check has spoken; the other five are the §6.4.3
    operational categories (refusal, upstream error, rate limit,
    timeout, harness fault). ``FAILED_ACCEPTANCE`` is added so the
    taxonomy is exhaustive over non-successes — a trial that is neither
    success nor one of the five operational categories still has a home.
    """

    SUCCESS = "success"
    FAILED_ACCEPTANCE = "failed_acceptance"
    REFUSAL = "refusal"
    UPSTREAM_ERROR = "upstream_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    HARNESS_FAULT = "harness_fault"


@dataclass(frozen=True)
class ModelReply:
    """The executor returned a model reply that needs an acceptance verdict.

    Attributes:
        reply: The model's reply, in whatever shape the executor's
            upstream returned. The acceptance check interprets it; the
            harness itself does not parse it.
    """

    reply: Any


@dataclass(frozen=True)
class UpstreamRefusal:
    """An upstream error the executor judged to be a behavioural refusal.

    The skeleton owns the seam — a dedicated outcome variant the
    executor can return when it recognises an upstream reply as a
    refusal-shaped error (e.g. a 400 with a content-moderation body).
    It does **not** own the heuristic for recognising one; provider
    body shapes belong to the executor author.
    """

    status: int
    body: Any


@dataclass(frozen=True)
class UpstreamFailure:
    """A non-refusal upstream error: a status code with a body.

    Status 429 lands in ``RATE_LIMIT``; everything else lands in
    ``UPSTREAM_ERROR``. A 400 with a content-moderation body should
    be wrapped as ``UpstreamRefusal`` by the executor instead, so a
    plain ``UpstreamFailure(400, ...)`` classifies to
    ``UPSTREAM_ERROR``.
    """

    status: int
    body: Any


@dataclass(frozen=True)
class TimedOut:
    """The executor hit the trial's wall-clock bound."""


@dataclass(frozen=True)
class UnclassifiedError:
    """An exception the executor caught but did not recognise.

    The runner wraps any exception escaping an executor call into this
    shape, so the per-trial loop never re-raises into the outer
    runner. Classifying here is the harness's one rule about harness
    faults: we caught it, we record it, the run continues.
    """

    exc: BaseException


#: The tagged union the executor returns. Distinct dataclasses — the
#: runner dispatches on ``isinstance`` rather than on a discriminator.
RawOutcome = ModelReply | UpstreamRefusal | UpstreamFailure | TimedOut | UnclassifiedError


@dataclass(frozen=True)
class RunConfig:
    """The settings that define one eval run, pinned at construction.

    §6.4.3: "Model id, provider, dataset revision, temperature and all
    sampling settings pinned and recorded with each run. An unpinned
    model makes the series meaningless." The dataclass makes that
    refusal mechanical: there is no way to construct a ``RunConfig``
    with a required field left as ``None``, and the error names the
    offending field so a maintainer reading a CI log knows which one
    the operator missed.

    ``sampling_overrides`` is the open extension surface for
    provider-specific knobs the fixed fields do not enumerate. It is
    pinned with everything else: its keys are recorded in sorted order
    so the digest is stable across dict construction orders, and its
    values must survive a JSON round-trip because the recorded digest
    is the only durable evidence of what was pinned.
    """

    #: The model identifier exactly as the provider publishes it.
    model_id: str
    #: The provider serving ``model_id``; a model id alone is ambiguous.
    provider: str
    #: Revision stamp of the task set the run used.
    dataset_revision: str
    #: Sampling temperature (§6.4.3 names it explicitly).
    temperature: float
    #: Nucleus-sampling ceiling, the standard companion to temperature.
    top_p: float
    #: Upper bound on reply length; matters to refusal/drop diagnoses.
    max_tokens: int
    #: Determinism hook for the model's own sampling.
    seed: int
    #: Samples per task per arm; §6.4.3 fixes N in advance. Must be >= 1.
    n_samples: int
    #: Wall-clock bound on a single trial, in seconds.
    deadline_seconds: float
    #: Provider-specific sampling knobs, pinned like the fixed fields.
    sampling_overrides: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        """Refuse any unset required field and validate the boundaries.

        Raises:
            ValueError: When any required field is ``None`` (the message
                names the field), when ``n_samples < 1``, or when
                ``sampling_overrides`` carries a value that cannot be
                serialised to JSON.
            TypeError: When ``sampling_overrides`` is not a ``Mapping``.
        """
        # Every required field unset is refused, naming the field. The
        # message is the only diagnostic a CI log gets, so the field name
        # is load-bearing and asserted by the falsification case (F1).
        for field in fields(self):
            if getattr(self, field.name) is None:
                raise ValueError(
                    f"RunConfig.{field.name} is unset; every field that defines "
                    "a run must be pinned (TEST_SUITE.md §6.4.3: an unpinned "
                    "model makes the series meaningless)"
                )

        # A zero-sample run has nothing to measure and no denominator to
        # hold; refusing it here keeps the boundary error at the site the
        # operator controls rather than deep in the runner.
        if self.n_samples < 1:
            raise ValueError(
                f"RunConfig.n_samples must be >= 1; got {self.n_samples}"
            )

        # The overrides map is pinned with the rest, which means its
        # contents have to reach the record intact — so a value that
        # cannot survive a JSON round-trip is refused here, at
        # construction, rather than silently dropped at write time.
        # Snapshot to a plain dict so any Mapping whose contents are
        # serialisable is accepted (a MappingProxyType or UserDict carries
        # JSON-clean bytes but the stdlib's JSON encoder refuses the
        # container itself, which would silently lose a perfectly valid
        # pin set).
        if not isinstance(self.sampling_overrides, Mapping):
            raise TypeError(
                f"RunConfig.sampling_overrides must be a Mapping; got "
                f"{type(self.sampling_overrides).__name__}"
            )
        try:
            json.dumps(dict(self.sampling_overrides))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RunConfig.sampling_overrides must be JSON-serialisable; "
                f"its recorded digest is the durable evidence of what was "
                f"pinned. {exc}"
            ) from exc


def classify(outcome: RawOutcome, verdict: Verdict | None = None) -> TrialCategory:
    """Map a single trial's outcome (and its verdict) to a trial category.

    §6.4.3 names the taxonomy: every non-success is classified and
    reported separately, and the categories are the diagnosis. The
    mapping table is in :mod:`harness.eval_harness` §2 of the
    requirements doc; this function is the implementation.

    Args:
        outcome: The tagged-union value the executor returned, or the
            runner's wrapped form of an exception.
        verdict: The acceptance check's judgement on the reply, used
            only when ``outcome`` is a :class:`ModelReply`. Other
            outcomes ignore it. ``None`` for a ``ModelReply`` is a
            programming error — the runner must always pass the verdict
            it computed, or skip ``classify`` and treat the trial as
            ``HARNESS_FAULT`` itself.

    Note:
        The mapping table lives in the design spec at
        ``.requirements/20260915T113051Z_eval_harness_skeleton/REQUIREMENTS.md``
        §2; this function is the implementation.

    Returns:
        The single :class:`TrialCategory` the trial lands in.

    Raises:
        ValueError: When ``outcome`` is a ``ModelReply`` and ``verdict``
            is ``None`` (the runner forgot to compute the verdict).
        TypeError: When ``outcome`` is not one of the five variants
            of :data:`RawOutcome`. A defensive guard so a future
            variant added without updating this function is reported,
            not silently dropped to ``HARNESS_FAULT`` (which would
            hide the omission).
    """
    if isinstance(outcome, ModelReply):
        if verdict is Verdict.PASS:
            return TrialCategory.SUCCESS
        if verdict is Verdict.FAIL:
            return TrialCategory.FAILED_ACCEPTANCE
        if verdict is Verdict.REFUSED:
            return TrialCategory.REFUSAL
        # No verdict was supplied for a ModelReply. A runner that calls
        # classify here has forgotten its own contract; raise rather
        # than classify to a default, which would let the omission pass
        # silently into the record.
        raise ValueError(
            "classify() requires a Verdict for a ModelReply; the runner "
            "must compute the acceptance check before classifying"
        )

    # Non-ModelReply outcomes ignore the verdict argument; each maps to
    # a single category by its own shape. No further branching is
    # needed for the §6.4.3 categories: a content-moderation 400 is
    # the executor's UpstreamRefusal, not a UpstreamFailure, and a 429
    # is the only rate-limit status the taxonomy names.
    if isinstance(outcome, UpstreamRefusal):
        return TrialCategory.REFUSAL
    if isinstance(outcome, UpstreamFailure):
        return TrialCategory.RATE_LIMIT if outcome.status == 429 else TrialCategory.UPSTREAM_ERROR
    if isinstance(outcome, TimedOut):
        return TrialCategory.TIMEOUT
    if isinstance(outcome, UnclassifiedError):
        return TrialCategory.HARNESS_FAULT

    raise TypeError(
        f"classify() received an outcome of unknown type "
        f"{type(outcome).__name__}; update this function when adding a new "
        f"RawOutcome variant"
    )


@dataclass(frozen=True)
class TaskSpec:
    """One eval task: a prompt, an id, and the check that judges a reply.

    The task set itself is plan task **T-K2** — independently authored,
    per §6.4.3 ("A model-generated test that the model's own code
    passes establishes nothing"). This class is only the shape a task
    has to satisfy so the harness can drive it; it ships no tasks.

    Attributes:
        id: Stable identifier the run record keys the trial rows by.
        prompt: What the arm is asked.
        acceptance_check: A callable judging one reply and returning a
            :class:`Verdict`. Per-task by design: what counts as a
            refusal is evidence about *this* task's context, and a
            generic heuristic would blur exactly the distinction
            §6.4.3 says is the diagnosis.
    """

    id: str
    prompt: str
    acceptance_check: Callable[[Any], Verdict]


#: The seam the runner calls once per trial. ``sample_index`` lets an
#: executor vary its behaviour across the N repetitions §6.4.3 requires.
ArmExecutor: TypeAlias = Callable[[TaskSpec, int], RawOutcome]


@dataclass(frozen=True)
class ArmSpec:
    """One arm of a paired run: a name for the record and its executor.

    Attributes:
        name: The arm's key in the run record's per-arm tallies. Must
            be unique within a run (the two-arm rule refuses duplicates,
            because two same-named arms would collapse into one tally
            and silently halve the evidence).
        executor: The :data:`ArmExecutor` that drives one trial.
    """

    name: str
    executor: ArmExecutor


def validate_arms(arms: Sequence[ArmSpec]) -> None:
    """Refuse any arm list that is not exactly two arms with distinct names.

    §6.4.3's measure is the *difference* between two arms — kitty and
    the direct provider. A single arm has nothing to compare against;
    a third arm changes what the comparison means; and two arms that
    share a name collapse into one per-arm tally, silently halving the
    evidence the record carries. The rule lives in its own function so
    the runner can call it as its first action — before any trial
    executes — and so a deliberate defect can be handed to it directly
    (§1.4).

    Args:
        arms: The arm list a run proposes to use.

    Raises:
        ValueError: When ``arms`` does not hold exactly two entries, or
            when the two entries share a name. The message states the
            rule, not just the count.
    """
    if len(arms) != 2:
        raise ValueError(
            f"an eval run needs exactly two arms (kitty and direct); got {len(arms)}"
        )

    first, second = arms
    if first.name == second.name:
        raise ValueError(
            f"the two arms must have distinct names; both are named {first.name!r}"
        )
