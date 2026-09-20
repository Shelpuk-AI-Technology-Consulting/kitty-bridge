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
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from typing import Any, TypeAlias

__all__ = [
    "ArmExecutor",
    "ArmSpec",
    "JSONValue",
    "ModelReply",
    "RawOutcome",
    "RunConfig",
    "RunRecord",
    "TaskSpec",
    "TimedOut",
    "TrialCategory",
    "TrialRecord",
    "UnclassifiedError",
    "UpstreamFailure",
    "UpstreamRefusal",
    "Verdict",
    "classify",
    "run_eval",
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
        #
        # Defaulted fields are skipped: a future ``field(default=None)``
        # on an optional pin would otherwise be rejected by this loop
        # even when the operator passed ``None`` intentionally.
        for field in fields(self):
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
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

    def digest(self) -> dict[str, JSONValue]:
        """Return the pinned settings as a JSON-compatible record.

        Returns:
            One key per field, in declaration order, with the
            ``sampling_overrides`` mapping re-keyed in sorted order so
            two dicts built with different insertion orders produce the
            same digest. This is the dict ``RunRecord.config`` carries;
            serialising it is what makes a run reproducible months
            later.
        """
        return {
            field.name: (
                # Sorted keys: the digest is the durable evidence of what
                # was pinned, and two runs that pinned the same overrides
                # in different insertion orders are the same run.
                dict(sorted(self.sampling_overrides.items()))
                if field.name == "sampling_overrides"
                else getattr(self, field.name)
            )
            for field in fields(self)
        }


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
        The mapping table is in ``.system_design/TEST_SUITE.md``
        §6.4.3; this function is the implementation.

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


#: The seam the runner calls once per trial. Async because the
#: vertical slice's executor awaits ``BridgeFixture.post``; making
#: the seam async from the start avoids an executor that wraps an
#: async call in ``asyncio.run`` and breaks the slice's own loop.
#: ``sample_index`` lets an executor vary its behaviour across the
#: N repetitions §6.4.3 requires.
ArmExecutor: TypeAlias = Callable[[TaskSpec, int], Awaitable[RawOutcome]]


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

    # Two arms with the same executor callable collapse the
    # bridge-vs-direct distinction without any visible signal — the
    # per-arm tally still has two distinct keys (names are different),
    # but every trial hits the same path. Refused at the gate so the
    # foot-gun trips before a single trial runs.
    if first.executor is second.executor:
        raise ValueError(
            "the two arms must have distinct executors; both share the same "
            "callable, which collapses the bridge-vs-direct distinction"
        )


@dataclass(frozen=True)
class TrialRecord:
    """One trial's outcome, as recorded in the run record.

    Attributes:
        arm: The arm this trial was run on (``"kitty"`` or ``"direct"``
            in the skeleton's two-arm shape).
        task_id: The :attr:`TaskSpec.id` the trial drove.
        sample_index: Which of the N samples this trial was. Zero-based,
            because ``range(n_samples)`` is what generated it.
        category: The :class:`TrialCategory` this trial landed in.
        duration_seconds: Wall-clock time spent in the trial, measured
            by the runner's injected ``clock``. Non-negative.
        detail: A short, JSON-safe diagnostic string. The category
            carries the diagnosis; ``detail`` is for a maintainer who
            wants to see *why* a refusal was a refusal, or what the
            upstream's body said. The skeleton keeps it small.
    """

    arm: str
    task_id: str
    sample_index: int
    category: TrialCategory
    duration_seconds: float
    detail: str


#: The per-arm × per-category tally: arm name → category → count,
#: keyed by the enum at runtime. The on-disk / JSON form (step 7's
#: ``to_json``) maps the enum to its ``value`` string.
TallyByArm = dict[str, dict[TrialCategory, int]]


@dataclass(frozen=True)
class RunRecord:
    """The artifact one run produces: the config digest, every trial, the tally.

    Attributes:
        config: The :meth:`RunConfig.digest` of the run's settings. The
            durable evidence of what was pinned (§6.4.3).
        scheduled: The number of trials the runner *intended* to run
            (``n_samples × len(tasks) × 2``). §6.4.3 calls the
            success-rate "Successes ÷ scheduled trials" specifically
            so refusals cannot be silently dropped from the denominator.
        trials: One :class:`TrialRecord` per intended trial, in the
            order the runner visited them (arm → task → sample_index).
            ``len(trials) == scheduled`` is a contract — the falsification
            case F2 catches a runner that drops non-successes.
        per_arm: The per-arm × per-category tally. ``per_arm[arm_name][TrialCategory.SUCCESS]``
            is the count for that cell; categories the arm never hit
            stay at their initial zero rather than being dropped, so a
            missing cell means "never populated", not "always zero".
    """

    config: dict[str, JSONValue]
    scheduled: int
    trials: tuple[TrialRecord, ...]
    per_arm: TallyByArm

    @property
    def pass_rate_per_arm(self) -> dict[str, float]:
        """Return successes ÷ per-arm scheduled for each arm, as a mapping.

        Returns:
            One float per arm in the run record's ``per_arm``. The
            runner's grid is rectangular (every arm runs every task
            ``n_samples`` times), so per-arm scheduled equals the
            total ``scheduled`` divided by the number of arms. ``0.0``
            when scheduled is non-zero and no trials succeeded; the
            empty mapping ``{}`` when the record has no arms.

        The per-arm denominator (not the total) is what §6.4.3's
        paired-delta measure consumes: the comparison is between
        arms, so each arm's rate must be self-consistent before the
        difference is meaningful.
        """
        if not self.per_arm:
            return {}
        per_arm_scheduled = self.scheduled / len(self.per_arm)
        return {
            arm: self.per_arm[arm].get(TrialCategory.SUCCESS, 0) / per_arm_scheduled
            for arm in self.per_arm
        }

    def to_json(self) -> bytes:
        """Serialise the record to JSON bytes for persistence.

        The on-disk form is the durable evidence of a run; T-K3's
        decision rule and T-K12's nightly will both read it. Indented
        (``indent=2``) so a maintainer can grep the file directly; the
        per-run size is bounded by the task set's size, and a
        readability-vs-bytes trade-off that costs bytes here buys
        legibility at every incident review.

        Returns:
            UTF-8-encoded JSON bytes. ``indent=2`` keeps the file
            diff-able and human-readable without an extra formatter.
        """
        return json.dumps(
            {
                "config": self.config,
                "scheduled": self.scheduled,
                "trials": [
                    {
                        "arm": trial.arm,
                        "task_id": trial.task_id,
                        "sample_index": trial.sample_index,
                        # Category travels as its enum value (the
                        # string), not as the enum's repr, so the
                        # on-disk format is independent of Python's
                        # enum printer.
                        "category": trial.category.value,
                        "duration_seconds": trial.duration_seconds,
                        "detail": trial.detail,
                    }
                    for trial in self.trials
                ],
                # Per-arm tally: enum key -> string key. The on-disk
                # form is JSON-safe; from_json rebuilds the enum keys.
                "per_arm": {
                    arm: {category.value: count for category, count in tally.items()}
                    for arm, tally in self.per_arm.items()
                },
            },
            indent=2,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> RunRecord:
        """Rebuild a :class:`RunRecord` from bytes :meth:`to_json` produced.

        Garbage input raises :class:`ValueError` — never silently
        returns a partial record. A future second consumer (T-K3) will
        validate the schema from outside this module; the harness's
        own inverse only proves round-trip-ability for the on-disk
        shape it wrote.

        Args:
            data: UTF-8 JSON bytes in the shape :meth:`to_json` emits.

        Returns:
            A :class:`RunRecord` whose ``__eq__`` matches the source.

        Raises:
            ValueError: When ``data`` is not valid JSON, when the
                top-level value is not a JSON object, when required
                keys are missing, or when a trial row names a category
                the taxonomy does not carry. The message names the
                specific failure so an operator reading a CI log
                knows what to fix.
        """
        try:
            raw = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"RunRecord.from_json: not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"RunRecord.from_json: top-level value is not a JSON object; "
                f"got {type(raw).__name__}"
            )

        required = {"config", "scheduled", "trials", "per_arm"}
        missing = required - set(raw)
        if missing:
            raise ValueError(
                f"RunRecord.from_json: missing required keys: {sorted(missing)}"
            )

        config = raw["config"]
        if not isinstance(config, dict):
            raise ValueError(
                f"RunRecord.from_json: 'config' must be an object; got {type(config).__name__}"
            )

        scheduled = raw["scheduled"]
        if not isinstance(scheduled, int) or isinstance(scheduled, bool):
            raise ValueError(
                f"RunRecord.from_json: 'scheduled' must be an int; got {type(scheduled).__name__}"
            )

        trials_raw = raw["trials"]
        if not isinstance(trials_raw, list):
            raise ValueError(
                f"RunRecord.from_json: 'trials' must be a list; got {type(trials_raw).__name__}"
            )
        # Each trial row is rebuilt with structural checks — key
        # presence and dict-ness. Value-type validation (e.g. ``arm``
        # being a string, ``sample_index`` being an int) belongs to
        # T-K3, the schema's second consumer. The harness's own inverse
        # only proves round-trip-ability for the shape it wrote.
        trials: list[TrialRecord] = []
        for index, row in enumerate(trials_raw):
            if not isinstance(row, dict):
                raise ValueError(
                    f"RunRecord.from_json: trial {index} is not an object; got {type(row).__name__}"
                )
            try:
                category = TrialCategory(row["category"])
            except ValueError as exc:
                # The propagated message names the bad value and the
                # valid set, which is what an operator reading a CI log
                # needs — but prefix it so the source is unambiguous.
                raise ValueError(
                    f"RunRecord.from_json: trial {index} names an unknown category: {exc}"
                ) from exc
            try:
                trials.append(
                    TrialRecord(
                        arm=row["arm"],
                        task_id=row["task_id"],
                        sample_index=row["sample_index"],
                        category=category,
                        duration_seconds=row["duration_seconds"],
                        detail=row["detail"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"RunRecord.from_json: trial {index} is missing key {exc.args[0]!r}"
                ) from exc
            # ``TrialRecord.duration_seconds`` is documented as
            # non-negative; the runner enforces it on write, so a
            # negative value in a parsed file is either a tampered
            # artifact or a bug somewhere else in the harness. The
            # docstring's contract is checked here so the violation
            # surfaces at the boundary ``from_json`` is, not later
            # when T-K3's consumer reads the field.
            if trials[-1].duration_seconds < 0:
                raise ValueError(
                    f"RunRecord.from_json: trial {index} has negative "
                    f"duration_seconds={trials[-1].duration_seconds}"
                )

        per_arm_raw = raw["per_arm"]
        if not isinstance(per_arm_raw, dict):
            raise ValueError(
                f"RunRecord.from_json: 'per_arm' must be an object; got {type(per_arm_raw).__name__}"
            )
        # Per-arm rebuild: convert string keys back to TrialCategory.
        # ``TrialCategory(key)`` raises ValueError on an unknown key,
        # which we let propagate so a corrupted file is loud.
        per_arm: TallyByArm = {}
        for arm, tally in per_arm_raw.items():
            if not isinstance(tally, dict):
                raise ValueError(
                    f"RunRecord.from_json: per_arm[{arm!r}] must be an object; "
                    f"got {type(tally).__name__}"
                )
            per_arm[arm] = {TrialCategory(category): count for category, count in tally.items()}

        return cls(
            config=config,
            scheduled=scheduled,
            trials=tuple(trials),
            per_arm=per_arm,
        )


async def run_eval(
    config: RunConfig,
    tasks: Sequence[TaskSpec],
    arms: Sequence[ArmSpec],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> RunRecord:
    """Drive the full grid, classify every trial, and emit the run record.

    The runner is **pure except for the injected ``clock`` keyword
    argument** (two calls per trial to populate ``duration_seconds``)
    and the executor calls it makes. It does not write to disk; the
    caller serialises the returned :class:`RunRecord` via its own
    methods (planned in step 7).

    Order of operations:

    1. :func:`validate_arms` — REQ 3's rule fires before anything else.
    2. Empty-task check — REQ 4's boundary.
    3. The grid loop: ``for arm in arms: for task in tasks: for
       sample in range(n_samples):``.
    4. Each trial: ``started = clock(); try { outcome = await
       arm.executor(...); verdict = task.acceptance_check(outcome.reply)
       if isinstance(outcome, ModelReply) else None; category =
       classify(outcome, verdict) } except Exception as exc { outcome =
       UnclassifiedError(exc); category = classify(outcome) }; duration
       = clock() - started``.

    The exception handler is the entire mechanism for F3 (REQ 7): an
    executor-side exception is wrapped, classified as
    :attr:`TrialCategory.HARNESS_FAULT`, and the run continues.

    Args:
        config: A pinned :class:`RunConfig`. ``n_samples`` and every
            required field are enforced at construction.
        tasks: The task set to drive. At least one task is required.
        arms: Exactly two :class:`ArmSpec` entries with distinct names.
            Validated by :func:`validate_arms` as the first action.
        clock: A zero-arg callable returning a non-decreasing float.
            Default ``time.monotonic``. Two calls per trial; the
            difference is the trial's ``duration_seconds``.

    Returns:
        A :class:`RunRecord` whose ``scheduled`` equals
        ``n_samples × len(tasks) × 2``, whose ``trials`` carries one
        row per intended trial, and whose ``per_arm`` is the per-arm
        × per-category tally.

    Raises:
        ValueError: From :func:`validate_arms` (bad arms list) or from
            the empty-task check.
    """
    # The two-arm rule fires first — a runner that runs trials against
    # an invalid arms list has built evidence it cannot key. The rule
    # is its own function so this is one call, not a copy of the logic.
    validate_arms(arms)

    # The empty-task boundary is here, not in RunConfig (where it has
    # no home) — a run with zero tasks has nothing to measure and no
    # denominator to hold; the runner is the place that knows what a
    # run *is*.
    if len(tasks) < 1:
        raise ValueError("an eval run needs at least one task; got 0")

    scheduled = config.n_samples * len(tasks) * len(arms)
    trials: list[TrialRecord] = []
    per_arm: dict[str, dict[TrialCategory, int]] = {
        arm.name: {category: 0 for category in TrialCategory} for arm in arms
    }

    # The grid. Per-trial try/except is the F3 mechanism: an executor
    # that raises or an acceptance check that raises is caught,
    # wrapped as UnclassifiedError, and classified HARNESS_FAULT. The
    # run continues regardless. Exception, not BaseException: an
    # operator interrupt (KeyboardInterrupt / SystemExit) propagates
    # so the operator's Ctrl-C still works during a long nightly.
    for arm in arms:
        for task in tasks:
            for sample_index in range(config.n_samples):
                started = clock()
                try:
                    outcome: RawOutcome = await arm.executor(task, sample_index)
                    verdict: Verdict | None = (
                        task.acceptance_check(outcome.reply)
                        if isinstance(outcome, ModelReply)
                        else None
                    )
                    category = classify(outcome, verdict)
                except Exception as exc:
                    outcome = UnclassifiedError(exc=exc)
                    verdict = None
                    category = classify(outcome)
                duration = clock() - started

                detail = _detail_of(outcome, verdict)
                trials.append(
                    TrialRecord(
                        arm=arm.name,
                        task_id=task.id,
                        sample_index=sample_index,
                        category=category,
                        duration_seconds=duration,
                        detail=detail,
                    )
                )
                per_arm[arm.name][category] += 1

    return RunRecord(
        config=config.digest(),
        scheduled=scheduled,
        trials=tuple(trials),
        per_arm=per_arm,
    )


def _detail_of(outcome: RawOutcome, verdict: Verdict | None) -> str:
    """Build the per-trial ``detail`` string.

    The category carries the diagnosis; ``detail`` is for a maintainer
    who wants to see *why* a refusal was a refusal, or what the
    upstream's body said. Kept short and JSON-safe — the on-disk
    record is the durable artifact.
    """
    if isinstance(outcome, ModelReply):
        return f"verdict={verdict.value if verdict is not None else 'none'}"
    if isinstance(outcome, UpstreamRefusal):
        return f"upstream_refusal status={outcome.status}"
    if isinstance(outcome, UpstreamFailure):
        return f"upstream_failure status={outcome.status}"
    if isinstance(outcome, TimedOut):
        return "timed_out"
    if isinstance(outcome, UnclassifiedError):
        # The category carries the diagnosis; ``detail`` records only
        # the exception class name, not the message. Exception messages
        # can carry URLs with embedded credentials (e.g. an aiohttp
        # ``ClientConnectorError`` carries the URL with userinfo in its
        # args), and the detail string lands in the persisted JSON
        # artifact T-K12 ships to disk. The harness-side log can carry
        # the full message; the durable record does not.
        return type(outcome.exc).__name__
    # Defensive: classify() raises on an unknown variant, so reaching
    # here means a new variant slipped past it. Mirror the same error.
    raise TypeError(
        f"_detail_of() received an outcome of unknown type "
        f"{type(outcome).__name__}; update this function when adding a new "
        f"RawOutcome variant"
    )
