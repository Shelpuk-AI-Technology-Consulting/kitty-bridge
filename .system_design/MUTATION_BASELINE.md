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
| Commit | `b680edd` |
| Date | 2026-09-15 |
| Host | Linux dev workstation, 8 CPU, idle |
| Selection | `pytest -m l1` minus `tests/test_internal_keys_not_sent_upstream.py` (see the deselect rationale below) |
| Score formula | mutmut badge formula as this aggregator computes it: `(killed + timeout) / tested`, where `tested = total − skipped − not_checked`. Timeouts count as kills; `no_tests` and `suspicious` dilute the score; `skipped` and `not_checked` drop out of the denominator (unexamined mutants are neither kills nor evidence of a gap) |

| Group | Total | Tested | Killed | Survived | Timeout | No tests | Suspicious | Not checked | Score |
|---|---|---|---|---|---|---|---|---|---|
| translators_and_engine | 4040 | 3084 | 2103 | 971 | 10 | 0 | 0 | 956 | 68.5% |
| compaction_and_pairing | -- | -- | -- | -- | -- | -- | -- | -- | _deferred_ |
| provider_hooks | 1045 | 1026 | 860 | 166 | 0 | 0 | 0 | 19 | 83.8% |
| model_context | 183 | 73 | 43 | 30 | 0 | 0 | 0 | 110 | 58.9% |
| openai_subscription | 474 | 474 | 236 | 238 | 0 | 0 | 0 | 0 | 49.8% |
| egress | 125 | 124 | 94 | 30 | 0 | 0 | 0 | 1 | 75.8% |
| supporting | 773 | 686 | 445 | 241 | 0 | 0 | 0 | 87 | 64.9% |
| **TOTAL** | **6640** | **5467** | **3781** | **1676** | **10** | **0** | **0** | **1173** | **69.3%** |

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

## Deferred groups

The registry's `DEFERRED_GROUPS` names groups whose baseline is pending,
not measured. Today that is **`compaction_and_pairing`** — the seven
BridgeServer methods §6.1 lists as "Compaction and pairing".

**Why.** mutmut generates mutants per *file*, not per function. Those
seven methods live in `src/kitty/bridge/server.py`, a ~7,900-line file;
mutmut's mutated copy of it reaches 354 MB / 5.7M lines and the
generating worker did not finish in 25 minutes on an idle 8-CPU
workstation. Mutating only the seven methods needs `# pragma: no mutate
block` markers on everything else in the file — a source change that
belongs to its own ticket, not to baseline configuration. The group is
therefore excluded from `only_mutate` in `pyproject.toml` (with the
reason inline) and named in the registry's `DEFERRED_GROUPS` until
someone lands those markers.

**Consequence for the thresholds.** [KBR-91](https://shelpuk.atlassian.net/browse/KBR-91)
(T-H3) cannot set a per-group threshold for `compaction_and_pairing`
until this gap closes. Its "≥ 85% killed per target group" rule applies
to every group in §6.1's table, and the table is not complete until the
seventh row measures.

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
