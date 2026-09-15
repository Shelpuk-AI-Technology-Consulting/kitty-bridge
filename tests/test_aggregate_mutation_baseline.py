"""T-H1 (KBR-88) — the per-group aggregator is not rot.

``scripts/aggregate_mutation_baseline.py`` reads mutmut's per-file
``.meta`` JSONs and buckets each mutant by the scope registry's
patterns. The aggregator is mutmut-coupled and version-tolerant
(imports ``mutmut.stats.status_by_exit_code`` from the installed
mutmut with a static fallback), so the thing most likely to drift
is the JSON shape it consumes (``exit_code_by_key``) and the bucket
routing.

Five tests pin the contract, plus a sixth for the unknown-exit-code path:

* the JSON shape — what the script reads from each ``.meta`` file,
  what key names it tolerates, and what it does with malformed input;
* the bucket routing — given a known mutant key in a known target
  group, it lands in that group with the expected status;
* the score formula — a hand-computed ratio the script's table
  matches.

**Layer.** L2 — the subject is a config-like artifact (the aggregator
plus its scope) that consumes a data format edited by mutmut and
edited by hand. Two artifacts edited separately, held against each
other.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the aggregator by file path rather than package name: `scripts/`
# is not a package and is deliberately NOT added to `sys.path` — one
# file, not a whole directory of script entry points. The module must
# be registered in ``sys.modules`` before ``exec_module`` — Python
# 3.13's ``@dataclass`` resolves the class's namespace through
# ``sys.modules`` while decorating ``Stat``.
_spec = importlib.util.spec_from_file_location(
    "aggregate_mutation_baseline",
    _REPO_ROOT / "scripts" / "aggregate_mutation_baseline.py",
)
assert _spec is not None and _spec.loader is not None
agg = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_mutation_baseline"] = agg
_spec.loader.exec_module(agg)
del _spec


def _meta(tmp_path: Path, *, exit_codes: dict[str, int]) -> Path:
    """Write a one-mutant ``.meta`` JSON under ``tmp_path``."""
    meta = tmp_path / "src" / "kitty" / "validation.py.meta"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"exit_code_by_key": exit_codes}))
    return meta


def test_json_shape_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The aggregator reads ``exit_code_by_key``; a missing key is empty.

    Pins the JSON shape mutmut writes: each per-file ``.meta`` file
    holds a top-level ``exit_code_by_key`` mapping mutant names to
    exit codes. A file without that key contributes zero mutants
    rather than crashing the run.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    # No exit_code_by_key key at all.
    (tmp_path / "src" / "kitty" / "validation.py.meta").parent.mkdir(parents=True)
    (tmp_path / "src" / "kitty" / "validation.py.meta").write_text("{}")
    stats, _unknown = agg.bucket_mutants(
        agg.collect_meta_files(),
        {g: agg.patterns_for(g) for g in agg.TARGET_GROUPS},
    )
    # No mutants → all-zero counts.
    for name, s in stats.items():
        if name.startswith("__"):
            continue
        assert s.total == 0, f"{name}: expected 0 mutants, got {s.total}"


def test_bucket_routing_for_a_known_mutant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real-shaped key lands in the right target group with the right status.

    Uses ``kitty.providers.openai_subscription.x__convert_content_types__mutmut_1``
    — the exact key shape the registry derives the specific-function
    pattern from. The exit code 1 maps to "killed" in mutmut's table.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    _meta(
        tmp_path,
        exit_codes={
            "kitty.providers.openai_subscription.x__convert_content_types__mutmut_1": 1,
        },
    )
    stats, _unknown = agg.bucket_mutants(
        agg.collect_meta_files(),
        {g: agg.patterns_for(g) for g in agg.TARGET_GROUPS},
    )
    g = stats["openai_subscription"]
    assert g.killed == 1, f"openai_subscription killed count: {g.killed}"
    assert g.survived == 0
    assert g.timeout == 0
    assert g.total == 1
    # And nothing landed in the wrong group or in __unmatched__.
    other_groups = {n: s for n, s in stats.items() if n not in ("openai_subscription", "__total__", "__unmatched__")}
    for n, s in other_groups.items():
        assert s.total == 0, f"{n} should be empty, got {s.total}"
    assert stats["__unmatched__"].total == 0


def test_score_formula_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand-computed score matches the script's badge formula.

    Five mutants: three killed (exit 1), one survived (exit 0), one
    timeout (exit 24). Badge formula: (killed + timeout) / tested =
    (3 + 1) / 5 = 0.8. ``no_tests`` and ``suspicious`` dilute the score;
    ``skipped`` drops out of the denominator.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    _meta(
        tmp_path,
        exit_codes={
            f"kitty.validation.x__foo_{i}__mutmut_1": code
            for i, code in enumerate([1, 1, 1, 0, 24], start=1)
        },
    )
    stats, _unknown = agg.bucket_mutants(
        agg.collect_meta_files(),
        {g: agg.patterns_for(g) for g in agg.TARGET_GROUPS},
    )
    g = stats["supporting"]  # kitty.validation is in supporting
    assert g.killed == 3
    assert g.survived == 1
    assert g.timeout == 1
    assert g.total == 5
    assert g.tested == 5  # no skipped → tested == total
    assert g.score == pytest.approx(0.8)


def test_score_formula_drops_skipped_from_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skipped mutant drops out of BOTH numerator and denominator.

    Pins mutmut's choice: skipped mutants were filtered out before
    running (their exit code 34 carries no behaviour verdict), so
    including them in either side would distort the score. Three
    mutants: two killed (exit 1), one skipped (exit 34).
    score = (2 + 0) / (3 - 1) = 2/2 = 1.0.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    _meta(
        tmp_path,
        exit_codes={
            "kitty.validation.x__foo_1__mutmut_1": 1,
            "kitty.validation.x__foo_2__mutmut_1": 1,
            "kitty.validation.x__foo_3__mutmut_1": 34,
        },
    )
    stats, _unknown = agg.bucket_mutants(
        agg.collect_meta_files(),
        {g: agg.patterns_for(g) for g in agg.TARGET_GROUPS},
    )
    g = stats["supporting"]
    assert g.killed == 2
    assert g.skipped == 1
    assert g.total == 3
    assert g.tested == 2  # total minus skipped
    assert g.score == pytest.approx(1.0)


def test_unknown_exit_code_buckets_as_suspicious_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An exit code not in the table is bucketed as suspicious AND warns.

    Pins the round-2 review finding: the previous fallback was silent.
    The mutant is counted (suspicious), so the score is not silently
    dropped, and a stderr summary names the unknown code so a future
    mutmut release that re-numbers an exit code is visible to whoever
    runs the aggregator.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    unknown_code = 99  # not in mutmut's table or our static fallback
    _meta(
        tmp_path,
        exit_codes={
            "kitty.validation.x__foo_1__mutmut_1": unknown_code,
            "kitty.validation.x__foo_2__mutmut_1": 1,  # known: killed
        },
    )
    stats, unknown = agg.bucket_mutants(
        agg.collect_meta_files(),
        {g: agg.patterns_for(g) for g in agg.TARGET_GROUPS},
    )
    g = stats["supporting"]
    assert g.suspicious == 1, f"unknown code should bucket as suspicious, got {g.suspicious}"
    assert g.killed == 1
    assert unknown == {unknown_code: 1}
    # Run the warner and assert the stderr message names the code.
    agg._warn_unknown_exit_codes(unknown)
    captured = capsys.readouterr()
    assert repr(unknown_code) in captured.err
    assert "warning" in captured.err.lower()


# ── main() guard branches ──────────────────────────────────────────────
#
# Each test pins one of ``main``'s three loud-failure exit codes (1)
# and the no-input exit code (2). The composition layer (how
# bucket_mutants results turn into the loud-failure contract) was
# uncovered by the five bucket/routing tests above; a regression that
# drops a guard would not be caught there.
#
# The three loud-failure branches:
#   1. total == 0 (this section's first test) — a non-deferred group
#      matches no mutants, the registry's patterns are mis-scoped.
#   2. no_tests > 0 — the L1 selection never touches the trampoline.
#   3. not_checked > 0 — an interrupted run left unexamined mutants.
# All three exit 1; main()'s no-input exit code is 2.


def test_main_exits_one_when_a_group_matches_zero_mutants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A non-deferred group with zero matching mutants fails loud.

    Pins the mis-scope guard (round-6 review: the third loud-failure
    branch had no test). Seed only mutants that match NO group's
    patterns — a module outside the registry — so every non-deferred
    group lands at ``total == 0`` and ``main`` must reject the run:
    the recorded baseline's per-group rows would otherwise read as
    measured when nothing was scoped at all.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    _meta(
        tmp_path,
        exit_codes={
            "kitty.unknown_module.x__foo__mutmut_1": 1,
        },
    )
    rc = agg.main()
    captured = capsys.readouterr()
    assert rc == 1, f"main should exit 1 when a group matches nothing, got {rc}"
    assert "zero mutants matched its patterns" in captured.err
    # The offending group is named: translators_and_engine is the first
    # non-deferred group in declaration order.
    assert "translators_and_engine" in captured.err


def test_main_exits_one_when_a_group_has_no_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A group whose mutants all land in ``no_tests`` fails loud.

    Pins the L1-completeness guard: if mutmut's trampoline never
    fires for any mutant in a target group, the L1 suite is missing
    coverage of those methods. ``main`` prints the table and exits 1
    with a stderr line naming the group.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    # A synthetic mutant key for `kitty.validation` (supporting group)
    # with exit code 33 — no_tests per the static fallback.
    _meta(
        tmp_path,
        exit_codes={
            "kitty.validation.x__foo__mutmut_1": 33,
        },
    )
    rc = agg.main()
    captured = capsys.readouterr()
    assert rc == 1, f"main should exit 1 when a group has no_tests, got {rc}"
    assert "supporting" in captured.err
    assert "no_tests" in captured.err
    # The table still prints so the operator sees what was produced.
    assert "| supporting |" in captured.out


def test_main_exits_one_when_a_group_has_unchecked_mutants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An interrupted run leaves ``not_checked`` mutants; ``main`` fails loud.

    Pins the run-completeness guard: every non-deferred group should
    be fully tested when the recording stops. A single ``None``
    exit code in the ``.meta`` JSON is enough to trigger the
    failure for the whole group.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    _meta(
        tmp_path,
        exit_codes={
            "kitty.validation.x__foo__mutmut_1": None,  # not_checked
        },
    )
    rc = agg.main()
    captured = capsys.readouterr()
    assert rc == 1, f"main should exit 1 on not_checked, got {rc}"
    assert "not_checked" in captured.err
    assert "supporting" in captured.err


def test_main_exits_two_when_no_meta_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """No ``.meta`` files under mutants/ → exit 2 (distinct from guard failures).

    The operator's first action is different — re-run mutmut, not fix
    the scope — so the exit code is distinct from the guard-failure 1.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    rc = agg.main()
    captured = capsys.readouterr()
    assert rc == 2, f"main should exit 2 with no .meta files, got {rc}"
    assert "mutmut run" in captured.err


def test_main_exits_zero_when_a_group_is_fully_tested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Every mutant in scope has a known exit code and no no_tests →
    exit 0, table prints, no stderr noise.

    One killed mutant per non-deferred target group, so every group
    clears its three guards.
    """
    monkeypatch.setattr(agg, "_MUTANTS_ROOT", tmp_path)
    _meta(
        tmp_path,
        exit_codes={
            # supporting: whole-module kitty.validation
            "kitty.validation.x__killed__mutmut_1": 1,
            # openai_subscription: specific top-level fn
            "kitty.providers.openai_subscription.x__convert_content_types__mutmut_1": 1,
            # model_context: whole-module
            "kitty.providers.model_context.x__resolve_catalog__mutmut_1": 1,
            # egress: whole-module
            "kitty.egress.x__should_bypass__mutmut_1": 1,
            # translators_and_engine: whole-module kitty.bridge.engine
            "kitty.bridge.engine.x__map_finish_reason__mutmut_1": 1,
            # provider_hooks: cross-module (any class carrying the hook)
            "kitty.providers.anthropic.xǁAnthropicAdapterǁtranslate_to_upstream__mutmut_1": 1,
        },
    )
    rc = agg.main()
    captured = capsys.readouterr()
    assert rc == 0, f"main should exit 0 on a clean run, got {rc}"
    assert captured.err == "", f"unexpected stderr on clean run: {captured.err!r}"
    for group in agg.TARGET_GROUPS:
        if group in agg.DEFERRED_GROUPS:
            continue
        assert f"| {group} |" in captured.out


def test_render_markdown_table_renders_but_does_not_fail_on_unmatched(
    capsys: pytest.CaptureFixture,
) -> None:
    """``__unmatched__`` mutants appear in the table but do not fail the run.

    Pins the explicit asymmetry in the design: ``__unmatched__`` is a
    signal of mis-scope (a hand-added glob for a file outside the
    registry, or vice versa), but it is not a guard-failure condition
    — the run completed cleanly. The aggregator emits the row so a
    reader sees the count, and the per-group guards catch the
    actual mis-scope (zero-total group).
    """
    # Stat.total is a computed property: supply field values, not the
    # computed total. 5 unexamined mutants = not_checked=5.
    stats = {group: agg.Stat() for group in agg.TARGET_GROUPS}
    stats["__total__"] = agg.Stat(killed=1, survived=1)
    stats["__unmatched__"] = agg.Stat(not_checked=5)  # 5 unexamined mutants
    table_lines = agg.render_markdown_table(stats)
    assert "__unmatched__" in table_lines
