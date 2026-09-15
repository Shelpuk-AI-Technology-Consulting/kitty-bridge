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

from harness.eval_harness import RunConfig

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
    """REQ 1 boundary — a degenerate ``n_samples`` is refused at construction."""
    with pytest.raises(ValueError) as exc_info:
        _fully_pinned(n_samples=0)

    assert "n_samples" in str(exc_info.value)


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
