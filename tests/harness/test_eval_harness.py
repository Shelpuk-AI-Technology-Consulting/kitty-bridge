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

import pytest

from harness.eval_harness import (
    ModelReply,
    RunConfig,
    TimedOut,
    TrialCategory,
    UnclassifiedError,
    UpstreamFailure,
    UpstreamRefusal,
    Verdict,
    classify,
)

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
