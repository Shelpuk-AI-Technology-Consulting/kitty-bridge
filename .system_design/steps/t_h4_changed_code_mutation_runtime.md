---
id: t_h4_changed_code_mutation_runtime
depends_on: [KBR-88]
---

# T-H4 — Measure changed-code mutation runtime, resolving Q11 (KBR-92)

Plan row: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §11 **T-H4**. Design:
`TEST_SUITE.md` §6.1, the "Cadence" bullet. Jira: **KBR-92**.

Depends on T-H1 (KBR-88, Done): the `[tool.mutmut]` config and the
`tests/mutmut_scope.py` scope registry this measurement builds on.
Q11's own framing governs the deliverable: the nightly-only mutation
cadence is *provisional pending a number* — per-PR changed-code mutation
testing is adopted only if a measured `mutmut run` restricted to the
functions a representative PR touches lands inside the budget the fast
gate can absorb. Rejecting it without measuring is an assumption, not a
decision.

## Plan

1. `scripts/measure_changed_code_mutation.py` — map a git rev-range onto
   the §6.1 scope registry (`tests/mutmut_scope.py`): changed
   `src/kitty/` files → dotted modules; unified-0 diff hunks →
   post-image touched lines; AST body spans → touched
   `(class, function)` pairs; registry intersection → touched `Target`
   rows and their mutmut fnmatch patterns. `--run` executes the scoped
   `mutmut run` timed (monotonic clock), recording box load, teeing
   output to a log, and emitting a JSON summary; without it, a dry-run
   prints the patterns and the exact command.
2. Representative PR (owner-confirmed): **KBR-285, PR #234, merge commit
   `30a91a0`** — the broad case (5 source files across 4 §6.1 target
   groups). Ground truth for the unit fixture: whole-module scope for
   `kitty.bridge.{messages,responses,gemini,engine}`, plus
   `content_classifiers`' `_cc_chunk_carries_content` and
   `BridgeServer._is_empty_cc_response`; six registry targets total.
3. Measure for real on the dev box (owner-confirmed: load documented, not
   deferred); record wall-clock, load conditions, and the budget
   comparison in `MUTATION_BASELINE.md`; mark Q11 ANSWERED in
   `TEST_SUITE.md` §11 and update §6.1's cadence bullet with the number
   and the decision.

## Delivered (2026-09-22)

1. **`scripts/measure_changed_code_mutation.py`** — the measurement
   tooling. Pure core: `git diff` hunks → post-image touched lines →
   AST body spans → touched `(class, function)` pairs (module-level +
   class-method depth, decorator-inclusive spans, nested defs
   attributed to their enclosing method — mutmut's mangler depth);
   registry intersection expanding whole-module rows per touched def
   and honouring cross-module `provider_hooks` rows against
   `only_mutate`; function-level mangled patterns per Q11's verbatim
   "restricted to the **functions**" framing. Runner: `subprocess.Popen`
   in its own process group under a wall-clock cap (default 3600 s =
   the CI job cap), SIGTERM → grace → SIGKILL, JSON summary with load
   and monotonic-clock wall time. An empty intersection refuses
   `--run` so zero positional patterns can never silently fall through
   to the full `only_mutate` scope.
2. **`tests/test_measure_changed_code_mutation.py`** (L1, 17 tests) —
   the pure mapping, the runner contract through an injected fake, the
   empty-intersection refusal, and the KBR-285 ground truth pinned by
   `tests/data/kbr285_diff_snapshot.json` (a committed snapshot of the
   analyzer's verified output for `30a91a0^1..30a91a0`, generated once
   by a throwaway and hand-verified against the actual diff — the
   oracle is the real diff, not a free-form list).
3. **The measurement** (see `MUTATION_BASELINE.md`'s KBR-92 section):
   representative PR KBR-285 (`30a91a0`); 10 §6.1-scope functions
   covered, 1864 mutants matched; generation 112 s; the clean test
   (the full L1 selection) 22.8 min standalone; the full local run
   OOM-killed during stats under documented box contention.
   **Structural finding:** positional patterns narrow only the
   per-mutant phase — generation + clean test + stats are identical to
   the nightly's, and the clean test alone is at the fast gate's
   entire budget. **Q11 answered: nightly-only retained.** The tooling
   remains valid for any future selection mechanism that can scope the
   clean test; a CI-side confirmation measurement is the flagged
   follow-up.
