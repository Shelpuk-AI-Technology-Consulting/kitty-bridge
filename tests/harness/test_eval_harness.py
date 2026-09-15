"""L1 tests for ``tests.harness.eval_harness`` (KBR-110, plan task T-K1).

Plan task **T-K1 — Eval harness skeleton** delivers the two-arm eval
harness; §6.4.3 specifies what every run must pin and how every
non-success must be classified. These tests are the contract that
``eval_harness.py`` has to satisfy at L1, and they include the three
§1.4 falsification cases the first working version of every harness
ships with.

**Layer.** This file defaults to ``l1`` by path (``tests/harness/`` is
not in :data:`tests.layers._PATH_DEFAULTS`, so the default is ``l1``);
no explicit marker is needed. The plan task is implemented in
``tests/harness/test_eval_harness_vertical_slice.py``, which carries the
``eval`` marker at module level so the end-to-end slice does not gate.

**New file, no CRLF trap.** Per ``kitty-bridge-tests-conftest-crlf``,
every file added to this repository writes LF; this file does too.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import harness.eval_harness as eval_harness
from harness.eval_harness import (
    ArmExecutor,
    ArmSpec,
    ModelReply,
    RawOutcome,
    RunConfig,
    RunRecord,
    TaskSpec,
    TimedOut,
    TrialCategory,
    UnclassifiedError,
    UpstreamFailure,
    UpstreamRefusal,
    Verdict,
    classify,
    run_eval,
    validate_arms,
)
from harness.test_contract import _KITTY_IMPORT

# ── RunConfig — pinning (REQ 1) ──────────────────────────────────────────────


def test_run_config_accepts_a_fully_pinned_instance() -> None:
    """REQ 1 — a fully-pinned config constructs and exposes every field."""
    config = RunConfig(
        model_id="claude-opus-4-1",
        provider="anthropic",
        dataset_revision="2026-09-12",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        seed=42,
        n_samples=4,
        deadline_seconds=30.0,
        sampling_overrides={"reasoning_effort": "medium"},
    )

    assert config.model_id == "claude-opus-4-1"
    assert config.provider == "anthropic"
    assert config.dataset_revision == "2026-09-12"
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.max_tokens == 1024
    assert config.seed == 42
    assert config.n_samples == 4
    assert config.deadline_seconds == 30.0
    assert config.sampling_overrides == {"reasoning_effort": "medium"}


def test_run_config_rejects_unset_required_field() -> None:
    """F1 (REQ 1) — every required field unset raises.

    §6.4.3 names this exact failure mode: "An unpinned model makes the
    series meaningless." The skeleton refuses to build a run record
    with an unpinned field rather than silently treating ``None`` as a
    default the operator forgot to set.
    """
    required: dict[str, object] = {
        "model_id": "claude-opus-4-1",
        "provider": "anthropic",
        "dataset_revision": "2026-09-12",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": 42,
        "n_samples": 4,
        "deadline_seconds": 30.0,
    }

    # A None in any one field is rejected, naming the offending field.
    for field_name in required:
        kwargs = dict(required)
        kwargs[field_name] = None

        with pytest.raises(ValueError) as exc_info:
            RunConfig(**kwargs, sampling_overrides={})  # type: ignore[arg-type]

        # The message must name the offending field — a maintainer reading
        # this in a CI log has no other way to see which one was missed.
        assert field_name in str(exc_info.value), (
            f"error for unset {field_name!r} should name the field; got {exc_info.value!r}"
        )


def test_run_config_rejects_n_samples_below_one() -> None:
    """REQ 1 boundary — a degenerate ``n_samples`` is refused at construction.

    The inclusive complement (``n_samples=1``) is the test that kills an
    off-by-one ``<= 1`` mutant; without it the falsification case would
    be vacuous at the boundary.
    """
    with pytest.raises(ValueError) as exc_info:
        _fully_pinned(n_samples=0)

    assert "n_samples" in str(exc_info.value)

    # The inclusive complement: 1 is legal, so the check must be ``< 1``,
    # not ``<= 1``. Without this, an off-by-one mutant survives.
    assert _fully_pinned(n_samples=1).n_samples == 1


def test_run_config_rejects_non_mapping_sampling_overrides() -> None:
    """REQ 1 type check — ``sampling_overrides`` must be a ``Mapping``."""
    with pytest.raises(TypeError) as exc_info:
        _fully_pinned(sampling_overrides=[("reasoning_effort", "medium")])  # type: ignore[arg-type]

    assert "sampling_overrides" in str(exc_info.value)


def test_run_config_rejects_non_serialisable_sampling_overrides_value() -> None:
    """REQ 1 serialisability — ``sampling_overrides`` values must round-trip through JSON.

    The recorded digest is the only durable evidence of what was pinned;
    a value that cannot be serialised would silently drop on the way to
    disk, so the check fires at construction rather than at write time.
    """
    with pytest.raises(ValueError):
        _fully_pinned(sampling_overrides={"fn": lambda: None})  # type: ignore[dict-item]

    with pytest.raises(ValueError):
        _fully_pinned(sampling_overrides={"set_val": {1, 2, 3}})  # type: ignore[dict-item]


# ── TrialCategory — classify() (REQ 5) ──────────────────────────────────────


def test_classify_verdict_pass_yields_success() -> None:
    """REQ 5 — a model reply the acceptance check accepts is SUCCESS."""
    assert classify(ModelReply(reply="OK"), Verdict.PASS) is TrialCategory.SUCCESS


def test_classify_verdict_fail_yields_failed_acceptance() -> None:
    """REQ 5 — an answer that fails the acceptance test is FAILED_ACCEPTANCE.

    "Failed acceptance" is the eval's quality signal: the arm answered
    but the answer was wrong. It is deliberately its own category so
    T-K3 can report it alongside the §6.4.3 operational ones.
    """
    assert classify(ModelReply(reply="nope"), Verdict.FAIL) is TrialCategory.FAILED_ACCEPTANCE


def test_classify_verdict_refused_yields_refusal() -> None:
    """REQ 5 — a model reply the acceptance check marks REFUSED is REFUSAL."""
    assert classify(ModelReply(reply="I cannot help"), Verdict.REFUSED) is TrialCategory.REFUSAL


def test_classify_upstream_refusal_yields_refusal() -> None:
    """REQ 5 — an executor pre-classified upstream refusal is REFUSAL.

    The skeleton owns the seam (``UpstreamRefusal`` is a dedicated
    ``RawOutcome`` variant) but **not** the heuristic for spotting one;
    provider-specific body shapes belong to the executor author.
    """
    assert classify(UpstreamRefusal(status=400, body={"error": "content_moderation"})) is TrialCategory.REFUSAL


def test_classify_upstream_failure_429_yields_rate_limit() -> None:
    """REQ 5 — a 429 upstream reply is RATE_LIMIT (§6.4.3's named category)."""
    assert classify(UpstreamFailure(status=429, body={"error": "rate_limited"})) is TrialCategory.RATE_LIMIT


def test_classify_upstream_failure_5xx_yields_upstream_error() -> None:
    """REQ 5 — any non-2xx, non-429 upstream reply is UPSTREAM_ERROR.

    The 400 case is the boundary the prose names: a 400 the executor
    cannot tell apart from any other upstream error lands as
    ``UpstreamFailure`` and classifies to ``UPSTREAM_ERROR``. A content-
    moderation 400 should arrive pre-classified as ``UpstreamRefusal``,
    not as a plain ``UpstreamFailure``. Asserting 400 here kills the
    ``==`` → ``<=`` off-by-one mutant the falsification culture requires.
    """
    assert classify(UpstreamFailure(status=400, body={"error": "bad"})) is TrialCategory.UPSTREAM_ERROR
    assert classify(UpstreamFailure(status=500, body={"error": "boom"})) is TrialCategory.UPSTREAM_ERROR
    assert classify(UpstreamFailure(status=503, body={"error": "unavailable"})) is TrialCategory.UPSTREAM_ERROR


def test_classify_timeout_yields_timeout() -> None:
    """REQ 5 — an executor-caught timeout is TIMEOUT."""
    assert classify(TimedOut()) is TrialCategory.TIMEOUT


def test_classify_unexpected_exception_yields_harness_fault() -> None:
    """REQ 5 — an exception the executor did not recognise is HARNESS_FAULT.

    HARNESS_FAULT is the one category that diagnoses *us*, not the
    upstream or the model. The runner catches executor-side exceptions
    and converts them; ``classify`` then turns the wrapped form into
    the category without re-raising.
    """
    assert classify(UnclassifiedError(exc=OSError("bridge bind failed"))) is TrialCategory.HARNESS_FAULT


def test_classify_model_reply_without_verdict_raises_value_error() -> None:
    """REQ 5 (guard) — a ModelReply without a verdict is a runner contract error.

    The runner is documented to compute the acceptance check before
    classifying; calling ``classify`` without a verdict means the
    runner forgot its own contract. The guard raises rather than
    silently defaulting, which would let the omission pass into the
    record as an ambiguous category.
    """
    with pytest.raises(ValueError, match="requires a Verdict"):
        classify(ModelReply(reply="anything"))


def test_classify_unknown_outcome_variant_raises_type_error() -> None:
    """REQ 5 (guard) — an outcome variant ``classify`` does not know is a programming error.

    The defensive guard exists so a future ``RawOutcome`` variant added
    without updating this function is reported loudly rather than
    silently dropping to ``HARNESS_FAULT`` (which would hide the
    omission entirely). The test pins the guard itself so a maintainer
    who replaces the ``raise`` with a silent default is caught.
    """
    with pytest.raises(TypeError, match="unknown type"):
        classify("not a RawOutcome")  # type: ignore[arg-type]


# ── Arms and the two-arm rule (REQ 3) ───────────────────────────────────────


async def _always_pass(task: TaskSpec, sample_index: int) -> RawOutcome:
    """A trivial async executor: every trial succeeds."""
    return ModelReply(reply=f"ok for {task.id} #{sample_index}")


def test_validate_arms_accepts_two_distinct_arms() -> None:
    """REQ 3 — exactly two arms with distinct names is the legal shape."""
    arms = [ArmSpec(name="kitty", executor=_always_pass), ArmSpec(name="direct", executor=_always_pass)]
    validate_arms(arms)  # must not raise


def test_validate_arms_rejects_a_single_arm() -> None:
    """REQ 3 — one arm is refused; §6.4.3's measure is the difference between two."""
    with pytest.raises(ValueError, match="exactly two"):
        validate_arms([ArmSpec(name="kitty", executor=_always_pass)])


def test_validate_arms_rejects_three_arms() -> None:
    """REQ 3 — three arms are refused for the same reason one is."""
    arms = [
        ArmSpec(name="kitty", executor=_always_pass),
        ArmSpec(name="direct", executor=_always_pass),
        ArmSpec(name="third", executor=_always_pass),
    ]
    with pytest.raises(ValueError, match="exactly two"):
        validate_arms(arms)


def test_validate_arms_rejects_duplicate_arm_names() -> None:
    """REQ 3 — two arms sharing a name are refused.

    The run record's per-arm tallies are keyed by name; two arms with
    the same name would collapse into one tally and silently halve the
    evidence.

    The two same-named arms carry **distinct** executors so a
    contrived conjunctive-condition mutant (``and first.executor is
    second.executor``) cannot satisfy the gate by accident.
    """
    arms = [
        ArmSpec(name="kitty", executor=_always_pass),
        ArmSpec(name="kitty", executor=_always_replying("alt")),
    ]
    with pytest.raises(ValueError, match="distinct"):
        validate_arms(arms)


# ── run_eval — the runner (REQ 4, 6, 7) ─────────────────────────────────────


async def test_run_eval_executes_the_full_grid() -> None:
    """REQ 4 — n_samples × len(tasks) × 2 trials executed and recorded."""
    config = _fully_pinned(n_samples=2)
    tasks = [_accepting_task("t1")]
    arms = _two_arms()

    record = await run_eval(config, tasks, arms)

    assert record.scheduled == 4  # 2 samples × 1 task × 2 arms
    assert len(record.trials) == 4
    # The grid is iterated arm → task → sample, so the trial ordering is
    # (kitty, t1, 0), (kitty, t1, 1), (direct, t1, 0), (direct, t1, 1).
    assert [trial.sample_index for trial in record.trials] == [0, 1, 0, 1]
    assert all(trial.category is TrialCategory.SUCCESS for trial in record.trials)


async def test_run_eval_rejects_empty_tasks() -> None:
    """REQ 4 boundary — a run with no tasks is refused before any trial executes.

    Per ``validate_arms``'s precedent, the rule fires first; an executor
    that raises on call would surface that here.
    """
    sentinel = False

    async def _raising(_task: TaskSpec, _sample_index: int) -> RawOutcome:
        nonlocal sentinel
        sentinel = True
        return ModelReply(reply="never reached")

    arms = _two_arms(_raising)

    with pytest.raises(ValueError, match="at least one task"):
        await run_eval(_fully_pinned(), [], arms)

    assert sentinel is False, "executor was called despite an empty task list"


async def test_run_eval_records_a_non_negative_duration_per_trial() -> None:
    """REQ 4 — every trial's ``duration_seconds`` is non-negative.

    A negative duration would mean the clock went backwards mid-trial,
    which is impossible for ``time.monotonic`` and a sign a fake clock
    is broken.
    """
    record = await run_eval(_fully_pinned(n_samples=2), [_accepting_task()], _two_arms())

    assert all(trial.duration_seconds >= 0.0 for trial in record.trials)


async def test_run_eval_durations_come_from_the_injected_clock() -> None:
    """REQ 4 — ``clock`` controls the recorded ``duration_seconds``.

    A stepped fake clock produces verbatim durations in the record,
    proving the clock is read and not approximated by ``time.monotonic``.
    """
    # n=2, 1 task, 2 arms → 4 trials, 8 clock calls (start, end each).
    # Pair the steps so each trial's duration is end - start of that pair.
    clock = _stepping_clock([0.0, 1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5])
    record = await run_eval(
        _fully_pinned(n_samples=2),
        [_accepting_task()],
        _two_arms(),
        clock=clock,
    )

    # Expected per-trial durations: 1.5, 1.5, 1.5, 1.5
    assert [trial.duration_seconds for trial in record.trials] == [1.5, 1.5, 1.5, 1.5]


async def test_run_eval_validation_fires_before_any_trial_executes() -> None:
    """REQ 3 + REQ 4 composition — the two-arm rule fires first.

    An invalid arm list with an executor that would record every call
    proves the validation refuses before the per-trial loop is reached.
    """
    sentinel = 0

    async def _counting(_task: TaskSpec, _sample_index: int) -> RawOutcome:
        nonlocal sentinel
        sentinel += 1
        return ModelReply(reply="called")

    bad_arms = [ArmSpec(name="kitty", executor=_counting)]  # only one arm

    with pytest.raises(ValueError, match="exactly two"):
        await run_eval(_fully_pinned(), [_accepting_task()], bad_arms)

    assert sentinel == 0, "validation passed but the executor was never called"


async def test_a_refusal_only_arm_leaves_pass_rate_at_zero_with_scheduled_intact() -> None:
    """F2 (REQ 6) — the denominator survives a refusal storm.

    §6.4.3: *"A bridge arm that refuses 90 of 100 tasks and answers the
    other 10 correctly scores 10%, which is the truth; excluding
    refusals would score it 100%."* The skeleton's runner holds the
    scheduled denominator regardless of how many trials return refusals.
    """
    config = _fully_pinned(n_samples=3)
    # A task whose acceptance check refuses every reply.
    refusing_task = TaskSpec(
        id="refuses",
        prompt="anything",
        acceptance_check=lambda _reply: Verdict.REFUSED,
    )
    record = await run_eval(config, [refusing_task], _two_arms())

    # Scheduled: 3 samples × 1 task × 2 arms = 6. Held intact.
    assert record.scheduled == 6
    assert len(record.trials) == 6
    # Every trial refused; no successes; pass rate 0.0 on both arms.
    assert all(trial.category is TrialCategory.REFUSAL for trial in record.trials)
    assert record.pass_rate_per_arm == {"kitty": 0.0, "direct": 0.0}


async def test_run_eval_classifies_a_trial_raising_oserror_as_harness_fault_and_continues() -> None:
    """F3 (REQ 7) — an executor exception is classified and the run continues.

    The defect it catches: a runner that crashes the whole run on one
    bad trial (surface as a job-level failure rather than a data point)
    or that swallows the exception into a silent skip (let a single
    broken executor look like an empty arm). Per REQ 7, the per-trial
    loop catches the exception, classifies it HARNESS_FAULT, and
    continues — scheduled intact, no unhandled exception escaping.
    """
    call_count = 0

    async def _flaky(_task: TaskSpec, sample_index: int) -> RawOutcome:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("simulated bridge bind failure")
        return ModelReply(reply=f"survivor #{sample_index}")

    record = await run_eval(_fully_pinned(n_samples=3), [_accepting_task()], _two_arms(_flaky))

    # The runner completes; no exception escapes.
    assert record.scheduled == 6  # 3 samples × 1 task × 2 arms
    assert len(record.trials) == 6

    # Per-arm × per-category tally: the first trial of the kitty arm
    # raised (HARNESS_FAULT); the rest succeeded.
    kitty_tally = record.per_arm["kitty"]
    direct_tally = record.per_arm["direct"]
    assert kitty_tally[TrialCategory.HARNESS_FAULT] == 1
    assert kitty_tally[TrialCategory.SUCCESS] == 2
    assert direct_tally[TrialCategory.SUCCESS] == 3
    assert direct_tally[TrialCategory.HARNESS_FAULT] == 0


async def test_run_eval_lets_operator_interrupt_propagate() -> None:
    """F3' — ``except Exception`` (not ``BaseException``) is the F3 mechanism.

    A future change that widened the per-trial catch to ``except
    BaseException`` would silently swallow ``KeyboardInterrupt`` /
    ``asyncio.CancelledError`` — operator signals that should abort a
    long nightly, not become a ``HARNESS_FAULT`` data point. The test
    raises ``asyncio.CancelledError`` (a ``BaseException`` subclass in
    3.8+) from an executor and asserts the runner re-raises it rather
    than wrapping it as ``UnclassifiedError``.
    """

    async def _canceller(_task: TaskSpec, _sample_index: int) -> RawOutcome:
        raise asyncio.CancelledError()

    arms = _two_arms(_canceller)

    with pytest.raises(asyncio.CancelledError):
        await run_eval(_fully_pinned(n_samples=1), [_accepting_task()], arms)


# ── RunRecord — JSON round-trip (REQ 2, 8) ──────────────────────────────────


async def _mixed_category_record() -> RunRecord:
    """Build a record whose trials hit several categories, for round-trip tests.

    Sample 0 produces a ``ModelReply`` the acceptance check accepts
    (SUCCESS). Sample 1 produces an ``UpstreamFailure(429)`` (RATE_LIMIT).
    Sample 2 produces an ``UpstreamRefusal`` (REFUSAL). The acceptance
    check accepts every ModelReply so a future change that drops
    sample-0's verdict trips the round-trip equality on the trial rows.
    """

    async def _mixed(_task: TaskSpec, sample_index: int) -> RawOutcome:
        if sample_index == 0:
            return ModelReply(reply="ok")
        if sample_index == 1:
            return UpstreamFailure(status=429, body={"error": "rate_limited"})
        return UpstreamRefusal(status=400, body={"error": "content_moderation"})

    task = TaskSpec(id="t1", prompt="anything", acceptance_check=lambda _r: Verdict.PASS)
    pinned = _fully_pinned(n_samples=3, sampling_overrides={"b_key": 2, "a_key": 1})
    return await run_eval(pinned, [task], _two_arms(_mixed))


async def test_run_record_json_round_trip() -> None:
    """REQ 8 — ``from_json(record.to_json())`` equals ``record``.

    Exercises every trial category the mixed executor produces, so the
    enum values, floats, ints and strings all survive the round-trip.
    """
    record = await _mixed_category_record()

    rebuilt = RunRecord.from_json(record.to_json())

    assert rebuilt == record


async def test_recorded_config_round_trips_through_json() -> None:
    """REQ 2 — the config digest carries every pinned setting after a JSON round-trip.

    The digest is the only durable evidence of what was pinned, so a
    round-trip that dropped a field would silently unpin the run.
    """
    record = await _mixed_category_record()
    rebuilt = RunRecord.from_json(record.to_json())

    assert rebuilt.config == record.config
    # Spot-check each field that defines the run.
    assert rebuilt.config["model_id"] == "claude-opus-4-1"
    assert rebuilt.config["n_samples"] == 3
    assert rebuilt.config["sampling_overrides"] == {"a_key": 1, "b_key": 2}


def test_sampling_overrides_keys_are_sorted_in_recorded_digest() -> None:
    """REQ 2 — overrides keys are re-keyed sorted so the digest is order-independent."""
    config = _fully_pinned(sampling_overrides={"zulu": 26, "alpha": 1, "mike": 13})

    digest = config.digest()
    overrides = digest["sampling_overrides"]
    assert isinstance(overrides, dict)  # type narrow: the union includes scalars

    assert list(overrides.keys()) == ["alpha", "mike", "zulu"]


async def test_run_record_from_json_rejects_garbage() -> None:
    """REQ 8 — garbage input raises ValueError, never a partial record.

    Four shapes of garbage: not JSON at all, JSON that is not an
    object, an object missing required keys, and a trial row naming a
    category the taxonomy does not carry. Each must be a loud typed
    error, not a silently-partial ``RunRecord``.
    """
    record = await _mixed_category_record()
    good = record.to_json()

    with pytest.raises(ValueError, match="not valid JSON"):
        RunRecord.from_json(b"this is not json at all {")

    with pytest.raises(ValueError, match="not a JSON object"):
        RunRecord.from_json(b"[1, 2, 3]")

    parsed = json.loads(good)
    del parsed["scheduled"]
    with pytest.raises(ValueError, match="missing"):
        RunRecord.from_json(json.dumps(parsed).encode("utf-8"))

    parsed = json.loads(good)
    parsed["trials"][0]["category"] = "not_a_real_category"
    # The propagated ``TrialCategory(...)`` message names the valid
    # set; the test pins that path, not a coincidental substring in the
    # bogus value itself.
    with pytest.raises(ValueError, match="is not a valid TrialCategory"):
        RunRecord.from_json(json.dumps(parsed).encode("utf-8"))


# ── Harness independence (REQ 9) ─────────────────────────────────────────────


def test_the_eval_harness_imports_nothing_from_kitty() -> None:
    """REQ 9 — the eval harness module's source contains no kitty import.

    A library that did would prove self-consistency, not fidelity
    (§3.3.1). The regex is imported from the canonical definition in
    :mod:`harness.test_contract`, not redefined — the failures-library
    guard (``tests/harness/test_failures.py``) imports the same
    constant, so a future form added there extends both guards
    together, and the canonical's own positive/negative control tests
    pin the pattern against silent drift.
    """
    source = eval_harness.__file__
    assert source is not None, "eval_harness module has no source path"

    body = Path(source).read_text(encoding="utf-8")
    # A non-trivial body length — an empty file would pass vacuously,
    # the same trap the failures guard avoids.
    assert len(body) > 1000, "read no meaningful source; the guard would pass vacuously"

    offending = [line.strip() for line in body.splitlines() if _KITTY_IMPORT.search(line)]
    assert offending == [], f"eval_harness.py must not import kitty: {offending}"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _fully_pinned(**overrides: object) -> RunConfig:
    """Build a fully-pinned ``RunConfig`` with the given overrides applied."""
    defaults: dict[str, object] = {
        "model_id": "claude-opus-4-1",
        "provider": "anthropic",
        "dataset_revision": "2026-09-12",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": 42,
        "n_samples": 4,
        "deadline_seconds": 30.0,
        "sampling_overrides": {},
    }
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def _accepting_task(task_id: str = "t1") -> TaskSpec:
    """A task whose acceptance check accepts any reply."""
    return TaskSpec(id=task_id, prompt="say something", acceptance_check=lambda _reply: Verdict.PASS)


def _two_arms(executor: object = _always_pass) -> list[ArmSpec]:
    """Two legal arms sharing the given executor."""
    return [ArmSpec(name="kitty", executor=executor), ArmSpec(name="direct", executor=executor)]  # type: ignore[arg-type]


def _always_replying(word: str) -> ArmExecutor:
    """Build an async executor whose every reply carries ``word``."""
    async def _reply(task: TaskSpec, sample_index: int) -> RawOutcome:
        return ModelReply(reply=f"{word} for {task.id} #{sample_index}")
    return _reply


def _stepping_clock(steps: list[float]) -> Callable[[], float]:
    """A fake clock handing out the given values in order.

    Two calls per trial (start, end), so a run of N trials consumes
    2N values. Raising when exhausted is the point: a runner that
    calls the clock a different number of times than the contract
    says should fail loudly, not wrap around.
    """
    yielded = iter(steps)

    def _next() -> float:
        return next(yielded)

    return _next
