# Kitty Bridge — Mutation Testing Baseline

**Status: PROVISIONAL.** See the caveat below before treating any number
here as a reference.

## What this records

Per-target-group mutation scores for the scope TEST_SUITE.md §6.1
specifies, produced by `mutmut` (3.x). The scope is defined in
`tests/mutmut_scope.py` — the machine-readable form of §6.1's table —
and the mutmut configuration lives in `pyproject.toml`
(`[tool.mutmut]`). `tests/test_mutmut_scope.py` guards the registry
against source drift.

This baseline is what [KBR-91](https://shelpuk.atlassian.net/browse/KBR-91)
(T-H3, per-component thresholds + nightly reporting) builds its
thresholds on. Until that ticket lands, no number here gates anything.

## 2026-09-21 — KBR-290: the §8.2 socket/process-binding modules leave mutmut's selection

[KBR-290](https://shelpuk.atlassian.net/browse/KBR-290) added one
`--ignore <path>` row per module in
`tests/socket_binding_l1_modules.py::SOCKET_BINDING_L1_MODULES` (the fifteen
§8.2 socket/process-binding modules) to `[tool.mutmut]
pytest_add_cli_args_test_selection`, so the nightly mutation run never
depends on loopback socket timing (the KBR-266 clean-test stall's root
cause). The Fast job's `pytest -m "l1 or l2"` selection is unchanged — every
module still runs on every push (and carries KBR-272's 120 s timeout mark on
the gate once that PR lands). Three
consequences for reading the numbers this file records:

1. **Kills vanish.** Tests in the deselected modules stop associating with
   mutants, so mutants they would have killed now show as `survived`. This
   is the effect §6.1 already anticipates ("kills currently credited through
   substantively-L3 tests vanish on re-measure") and is the desired honesty,
   not a regression.
2. **`no_tests` dilution.** Mutants covered *only* by the deselected tests
   reclassify to `no_tests` (exit codes 5/33) and stay in the score
   denominator (`tested = total − skipped − not_checked`). The groups most
   affected are `compaction_and_pairing` — 532–596 L1 tests associated with
   each method before the change, several contributed by
   `tests/harness/test_bridge.py`, `test_vertical_slice.py`,
   `test_containment.py` and `test_provider_aiohttp.py` — and `egress`,
   which overlaps with `tests/test_egress_https_proxy.py`. KBR-91's
   thresholding must treat the next per-group re-measure as the new
   baseline, not as a regression against the rows above.
3. **Cache invalidation on the first post-merge run.** Verified on the
   mutmut 3.8.0 installed in this venv (pyproject declares `mutmut>=3.0,<4`;
   the fingerprint + invalidation behaviour below holds across the pinned
   range per the mutmut 3.x source): mutmut fingerprints
   `pytest_add_cli_args_test_selection` (`test_selection` in
   `config_fingerprint`); a change resets all cached verdicts and forces a
   full stats recollection and per-mutant re-run
   (`_apply_config_change_invalidation`). The "re-running resumes from here"
   property of `mutmut run` does **not** apply across this change — budget
   the first nightly as a full re-run.

The deselection set is held against `pyproject.toml` by
`tests/test_socket_binding_l1_mutation_exclusion.py` (l2); the same registry
is the source of truth for KBR-272's timeout-mark guard.

## 2026-09-17 — KBR-73 (T-F4) note: egress-group tests strengthened, TOTAL not re-measured

[KBR-73](https://shelpuk.atlassian.net/browse/KBR-73) added five property
tests to the `egress` group's L1 selection (`tests/test_egress_properties.py`),
which should raise the group's kill rate — the earlier baseline
(75.8%, 30 survivors) predates them. The **numbers below are not
re-measured** for this change: a full-group mutmut run costs more than
this task's budget, and the consumer of the refreshed numbers is
KBR-91 (T-H3), which will re-run the baseline when it derives the
per-component thresholds. No claim is made that the score improved,
only that the surviving-mutant list the next run produces will be
judged against a stronger suite. `tests/mutmut_scope.py` is unchanged
— `kitty.egress` and `kitty.egress_guard` were already targets.

## How to reproduce

```console
# dev extras carry mutmut>=3.0,<4; mutmut 3.x requires fork (Linux, or WSL).
uv pip install -e '.[dev]'
mutmut run <patterns...>          # patterns from `patterns_for()` below
mutmut export-cicd-stats          # aggregate CI/CD stats
python scripts/aggregate_mutation_baseline.py   # per-group table
```

The recorded invocation below passes, on the `mutmut run` command line,
the union of `patterns_for(g)` over every group in `tests/mutmut_scope`.
`source_paths` cannot carry function wildcards (mutmut treats its entries
as filesystem paths), so the narrowing is positional and lives in the
registry, not in `pyproject.toml`.

## The run

| | |
|---|---|
| Mutmut | 3.8.0 |
| Commit | `b680edd` for the six measured groups; `compaction_and_pairing` row from [KBR-266](https://shelpuk.atlassian.net/browse/KBR-266) at commit `4f38421` (see the [note below](#compaction_and_pairing-no-longer-deferred-score-still-pending)) |
| Date | 2026-09-15 (b680edd); 2026-09-16 (KBR-266, partial) |
| Host | Linux dev workstation, 8 CPU. b680edd was recorded idle; the KBR-266 per-mutant test phase ran under sibling-session contention |
| Selection | `pytest -m l1` minus `tests/test_internal_keys_not_sent_upstream.py` (see the deselect rationale below) |
| Score formula | mutmut badge formula as this aggregator computes it: `(killed + timeout) / tested`, where `tested = total − skipped − not_checked`. Timeouts count as kills; `no_tests` and `suspicious` dilute the score; `skipped` and `not_checked` drop out of the denominator (unexamined mutants are neither kills nor evidence of a gap) |

| Group | Total | Tested | Killed | Survived | Timeout | No tests | Suspicious | Not checked | Score |
|---|---|---|---|---|---|---|---|---|---|
| translators_and_engine | 4040 | 3084 | 2103 | 971 | 10 | 0 | 0 | 956 | 68.5% |
| compaction_and_pairing | 648 | 0 | -- | -- | -- | -- | -- | 648 | _pending_ — see [the note below](#compaction_and_pairing-no-longer-deferred-score-still-pending) |
| provider_hooks | 1045 | 1026 | 860 | 166 | 0 | 0 | 0 | 19 | 83.8% |
| model_context | 183 | 73 | 43 | 30 | 0 | 0 | 0 | 110 | 58.9% |
| openai_subscription | 474 | 474 | 236 | 238 | 0 | 0 | 0 | 0 | 49.8% |
| bedrock_transport | _pending_ | -- | -- | -- | -- | -- | -- | -- | _pending_ — added by KBR-89 (T-H2); not yet measured; KBR-91 (T-H3) re-measures and sets the threshold row |
| ollama_transport | _pending_ | -- | -- | -- | -- | -- | -- | -- | _pending_ — added by KBR-90 (T-H5) as the P19 / `OllamaCloudAdapter._ollama_body` sibling of `bedrock_transport`; not yet measured; KBR-91 (T-H3) re-measures and sets the threshold row |
| egress | 125 | 124 | 94 | 30 | 0 | 0 | 0 | 1 | 75.8% |
| supporting | 773 | 686 | 445 | 241 | 0 | 0 | 0 | 87 | 64.9% |
| **TOTAL** | **7288** | **5467** | **3781** | **1676** | **10** | **0** | **0** | **1821** | **61.4%** of tested (648 in `compaction_and_pairing` pending; `bedrock_transport` added by KBR-89 and `ollama_transport` by KBR-90, re-measured by KBR-91) |

Column meanings, so the numbers recompute from the formula:

* **Total** — every mutant the registry's patterns match in the
  generated `.meta` files (the §6.1 scope at commit `b680edd`).
* **Tested** — Total minus `not_checked`: mutants that actually ran.
* **Not checked** — mutants still pending when the recording run was
  stopped (`exit_code = None` in the `.meta` files). mutmut caches
  results, so re-running `mutmut run` with the same patterns resumes
  from here; see the paragraph below the table.

**In progress at recording time.** `translators_and_engine` (76% tested),
`provider_hooks` (98%), `model_context` (40%), `egress` (99%), and
`supporting` (89%) are not 100% — the remaining mutants are in functions
whose l1 tests take minutes each, and the workstation run plateaued at
5,467 / 6,640 in-scope mutants over ~2 hours of testing. mutmut caches
results, so re-running `mutmut run` with the same patterns resumes from
the current state and completes the table — KBR-91's nightly job is the
natural home for that.

## `compaction_and_pairing`: no longer deferred, score still pending

[KBR-266](https://shelpuk.atlassian.net/browse/KBR-266) lifted the group
out of `DEFERRED_GROUPS` by landing `# pragma: no mutate block` markers
on every def/class in `server.py` except the seven `BridgeServer`
methods the group names (pinned by
`tests/test_mutmut_scope.py::test_server_py_pragma_scheme_marks_everything_but_the_registered`,
generalised by [KBR-285](https://shelpuk.atlassian.net/browse/KBR-285)
to derive the expected set from every registry row naming
`kitty.bridge.server`).
`server.py` rejoined `only_mutate`, and mutmut generates mutants for
those methods: **648 mutants in scope, generated in 63 seconds**
(previously the whole file mutated to 354 MB / 5.7 M lines and the
generating worker did not finish in 25 minutes).

**Why the score row is still `_pending_`, not measured.** The run that
generated the mutants recorded the test-to-mutant associations (the
stats phase completed; 532–596 L1 tests associate with each of the five
compaction methods), but the subsequent per-mutant test phase did not
run to completion: the clean-test run that precedes it exercises L1
tests which make real upstream calls, and under sibling-session
contention on the recording workstation those calls stalled in
`CLOSE_WAIT` with no per-test timeout bound
(`pytest-timeout` joined the dev extra with KBR-272; the marks bound
the socket-binding modules at 120 s). The stats cache is
`mutants/mutmut-stats.json`; re-running `mutmut run` with the group's
patterns on an idle workstation resumes from it and skips the ~20
minute stats phase. Recording the score is the re-measure obligation
below, and [KBR-91](https://shelpuk.atlassian.net/browse/KBR-91)'s
per-group threshold rule applies to every group in §6.1's table — the
table is not complete until the seventh row measures.

**Consequence for the thresholds.** [KBR-91](https://shelpuk.atlassian.net/browse/KBR-91)
(T-H3) cannot set a per-group threshold for `compaction_and_pairing`
until the score above records. Its "≥ 85% killed per target group" rule
applies to every group in §6.1's table, and the table is not complete
until the seventh row measures.

## Why the numbers are provisional

[The KBR-88 comment of 2026-09-08](https://shelpuk.atlassian.net/browse/KBR-88)
records the concern, and §6.1 §"Test selection" states the rule: mutation
testing measures the **L1** suite. The socket/process modules §8.2
enumerates sit at the `l1` path default — deliberately, because only
`l1` and `l2` are gated today and reclassifying them before the
Subsystem job exists (T-K6,
[KBR-115](https://shelpuk.atlassian.net/browse/KBR-115)) would remove them
from every gate. They are substantively L3 tests running under the L1
marker.

**The consequence is that this baseline is optimistic, not merely
different.** When KBR-115 moves those modules to `l3`, the kills they
currently earn will move with them, and the per-group scores will drop.
A group reading ≥ 85% here has **not** passed its threshold; the
threshold check is T-H3's job, run after the re-measure.

Re-measure after KBR-115 lands. A number from this file treated as the
reference after that point is the outcome to avoid.

## What is NOT in this baseline

- The socket/process modules TEST_SUITE.md §8.2 enumerates — they
  run under `-m l1` today and dilute the numbers; see the provisional
  caveat. §8.2 is the authoritative enumeration, and a future reader
  reconciling the baseline's drift against the doc should look there,
  not in this file.
- `tests/cli/test_stream_encoding.py` (38 child interpreters per run) — its
  cost is paid once at stats collection, not per mutant: `only_mutate`
  excludes `kitty.io_encoding` from generation, so no mutant carries those
  tests' associations. §8.2's concern is solved structurally, not by a
  `--deselect`.
- `tests/test_internal_keys_not_sent_upstream.py` — excluded from
  mutmut's selection via `--ignore` in `pytest_add_cli_args_test_selection`
  (the L1 gate still runs it). mutmut's clean test run fails on this
  file: mutmut-generated string mutations in
  `MessagesTranslator._translate_assistant_message` write
  `_THINKING_BLOCKS` in upper case, and the test's `startswith("_")`
  scan sees the uppercase key as leaked before the strip can run. Every
  parametrized case of `test_no_internal_key_reaches_upstream` fails
  for the same reason, so one `--deselect` only shifts the failure to
  the next case. Investigating the trampoline interaction further is
  its own ticket, not T-H1 scope.
- `also_copy = ["README.md", "scripts", "openapi"]` ("scripts" KBR-266;
  "openapi" KBR-272) — `tests/test_aggregate_mutation_baseline.py`
  imports `scripts/aggregate_mutation_baseline.py` by filesystem path
  at module level, so a fresh `mutants/` tree needs `scripts/` inside
  it or test collection fails before the clean run starts. The entry
  postdates the recorded b680edd run (the L2 test landed in KBR-88
  review round 7); every run on main needed it from then on. The
  directory form (not the single file) is deliberate: mutmut 3.8.0's
  file branch (`shutil.copy2`) does not create the destination's
  parent directory and dies on a fresh `mutants/` tree; the directory
  branch (`copytree(dirs_exist_ok=True)`) does.
- `do_not_mutate_patterns` is unused, and the
  `# pragma: no mutate block` markers on `server.py` (KBR-266) are the
  only suppression mechanism in scope. Either mechanism silently
  shrinks a group's measured surface: the marker scheme is pinned by
  `tests/test_mutmut_scope.py::test_server_py_pragma_scheme_marks_everything_but_the_registered`,
  but `do_not_mutate_patterns` changes have no guard — the per-group
  TOTAL against the previous run is the only thing that would notice.
- `mutate_only_covered_lines` is **not** enabled — known crash class on
  single-init C extensions (mutmut #528, fixed in #566); kitty imports
  `curl_cffi` and `boto3`. Do not enable without verifying the fix first.

## Survivor triage

Every survivor in this baseline is triaged under
[KBR-91](https://shelpuk.atlassian.net/browse/KBR-91) (T-H3) using §6.1's
rule: (a) a survivor revealing a missing assertion → strengthen the test;
(b) revealing untested behaviour → add a test; (c) genuinely equivalent →
suppress at the site with `# pragma: no mutate` **and a comment saying
why**. Never dismiss a survivor silently.

## 2026-09-22 — KBR-92 (T-H4): changed-code run measured, Q11 answered conservatively

[KBR-92](https://shelpuk.atlassian.net/browse/KBR-92) resolves Q11 by
measurement, not assertion. The question is whether `mutmut run` scoped to
the functions a representative PR touches would fit inside the per-PR
gate's budget, given that the gate already runs ~18.5 minutes per Python
version. The scope is the strict reading of `TEST_SUITE.md` §6.1's cadence
bullet: "restricted to the **functions** a representative PR touches",
**not** whole-module (which would silently re-test ~4,000 translator
mutants and answer a different question).

**Representative PR.** KBR-285 (PR #234, merge commit `30a91a0`) — the
broad worst case. Five source files changed; 11 def bodies touched;
10 of them §6.1-scope (the `BridgeServer._stream_chat_completions`
def is touched but is not in any §6.1 registry row, so it is out of the
measurement by design — the nightly doesn't measure it, and a per-PR
run must match the nightly's coverage):

```
kitty.bridge.messages.translator.xǁMessagesTranslatorǁtranslate_response__mutmut_*
kitty.bridge.messages.translator.xǁMessagesTranslatorǁtranslate_stream_chunk__mutmut_*
kitty.bridge.responses.translator.xǁResponsesTranslatorǁtranslate_response__mutmut_*
kitty.bridge.responses.translator.xǁResponsesTranslatorǁtranslate_stream_chunk__mutmut_*
kitty.bridge.responses.translator.x__extract_text_parts__mutmut_*
kitty.bridge.gemini.translator.xǁGeminiTranslatorǁtranslate_response__mutmut_*
kitty.bridge.gemini.translator.xǁGeminiTranslatorǁtranslate_stream_chunk__mutmut_*
kitty.bridge.gemini.translator.x__extract_text_parts__mutmut_*
kitty.bridge.server.x__cc_chunk_carries_content__mutmut_*
kitty.bridge.server.xǁBridgeServerǁ_is_empty_cc_response__mutmut_*
```

The mapping (rev-range → §6.1-scope function patterns, with the registry's
`mangled_patterns` semantics reproduced per-def) lives in
`scripts/measure_changed_code_mutation.py` and is pinned by 23 L1 unit
tests (the authoritative count is `grep -c '^def test_' tests/test_measure_changed_code_mutation.py` — this prose count has drifted four times across rounds 2–6; the grep is the source of truth)
tests in `tests/test_measure_changed_code_mutation.py`; the KBR-285
fixture in `tests/data/kbr285_diff_snapshot.json` is the analyzer's
verified output for `30a91a0^1..30a91a0` (generated once by a throwaway,
hand-verified against the actual diff, committed).

**What a changed-code `mutmut run` actually costs.** `mutmut`'s positional
patterns only filter which mutants get **tested**, not which get
**generated** — `only_mutate` bounds generation to every registered
source file, the test selection (`pytest_add_cli_args_test_selection`)
bounds the clean test, and positional patterns only select mutants in the
per-mutant test phase. The run therefore has the same three fixed-cost
phases the nightly pays, plus a scoped per-mutant phase:

| Phase | Cost on this box | Notes |
|---|---|---|
| Generation (full `only_mutate`) | **112 s** (run log: `done in 111774ms (41 files mutated, 53 ignored, 0 unmodified)`) | Identical to the nightly; the 10 positional patterns do not narrow it |
| Clean test (full L1 selection) | **1374 s = 22.8 min** (standalone measurement: `pytest -m l1 --ignore …` against 7009 tests, 0 failures; wall 1368.69 s under load avg 0.4–2.8, median ~1.0) | Identical to the nightly; the 16 `--ignore` rows are mutmut-only, the L1 gate runs every selected file |
| Stats (test→mutant associations) | Proportional to nightly; killed before completion locally | Identical to the nightly |
| Per-mutant test (scoped to the 10 patterns) | **1864 mutants** generated and matched; per-mutant time bounded by the baseline's ~1.3 s/mutant | The only phase a per-PR gate saves on |

**Why this answers Q11 in the conservative direction without a completed
local run.** The fixed-cost row of the table — generation + clean test +
stats — is identical to the nightly, and the clean test alone is already
22.8 minutes on this box under modest load (the fast gate's per-Python
version budget is ~18.5 minutes; the CI runner's pace is faster than
this workstation, but the L1 selection itself is the workload). A
per-PR mutation job would re-pay that fixed cost every push, on top of
the per-mutant marginal. Even if the per-mutant phase were zero, the job
already meets the budget head-on; with even a modest per-mutant cost
it is structurally over.

The local full run did not complete: the box was OOM-killed during the
stats phase at ~13 minutes total elapsed (4.6 GiB used / 233 MiB free /
4.4 GiB in swap, with 11 sibling Claude sessions, another worktree's
pytest, and a GitHub Actions runner process resident). The local box
cannot currently complete the measurement reliably — the documented
"kill under parallel sessions" memory applies, plus an additional OOM
hazard from the stats phase's coverage-traced pytest. The CI measurement
is the follow-up (see [KBR-92](https://shelpuk.atlassian.net/browse/KBR-92)
comment for the recommendation); on an idle runner the clean-test cost
is expected to land at the fast-gate leg's pace (~15–16 minutes),
confirming that the fixed cost alone is at the budget edge.

**Decision (KBR-92, 2026-09-22).** Keep mutation testing on the existing
nightly cadence. A per-PR changed-code `mutmut run` is structurally
infeasible under the current `pytest_add_cli_args_test_selection` and
`only_mutate` configuration because the positional patterns narrow only
the per-mutant phase; the fixed cost is shared with the nightly and
already meets the gate budget head-on. Q11 is ANSWERED.

**What would change the answer.** Either a per-test-selection that scopes
the clean test to the touched files' tests (which the current selection
does not support — `pytest_add_cli_args_test_selection` is a CLI list,
not a function-level filter), OR a measurement on an idle CI runner
showing the full per-PR run lands inside 18.5 minutes (the CI leg's
budget; structurally unlikely given the fixed cost's load-independent
character). Both are out of scope for T-H4.
