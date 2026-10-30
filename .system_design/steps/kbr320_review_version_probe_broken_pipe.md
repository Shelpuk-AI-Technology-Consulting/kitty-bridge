---
id: kbr320_review_version_probe_broken_pipe
depends_on: []
---

# KBR-320 — Review job exit 120: broken pipe in the version probe

## What

Repair the `Record which bridge version ran` step in
`.github/workflows/claude-code-review.yml`. The probe
`VERSION=$("$KITTY" --version 2>&1 | head -n 1)` leaves kitty (a CPython
console script) upstream of `head`, which exits after the first line and
closes the pipe; `kitty --version` prints a second line, so kitty's
remaining write or final flush raises `BrokenPipeError`, and CPython's
documented interpreter-cleanup rule changes the exit status to 120
(bpo-5319 / bpo-29130). Under the runner's
`bash --noprofile --norc -e -o pipefail` the failing pipeline aborts the
step before its summary line: zero output, exit 120, absorbed by the
step's `continue-on-error: true` but painted as a red `X` annotation on a
merge-gating check. Measured: the exact pipeline returns 120 in 60/60
local runs; three CI occurrences on 2026-10-18/19 (runs 39161761143,
39174082633, 39239198146).

The fix captures the whole version output into a variable first, then
takes the first line — no early-closing reader stays upstream of the
writer. The root cause is recorded in the workflow file as a comment at
the fix, including the distinction from the prompt-size E2BIG mechanism
the ticket also names (that one is real and already guarded by the
`Build review prompt` redaction budget).

No job-level retry is added: the only observed exit-120 mechanism is
eliminated at source, the only E2BIG mechanism is already budget-mitigated
(its incidents predate the `Build review prompt` redaction budget that was
built because of them), and a new trigger surface on a public-repository
merge gate is not speculative-proof. The ticket's acceptance criteria
offer fix *or* retry; this takes the fix.

Design-record location: this area's "To Be" lives in the workflow file's
own comments — `claude-code-review.yml` carries its design decisions
inline, evidenced and dated, and `.github/review/rules/ci.md` pins the
invariants. `SYSTEM_DESIGN.md` deliberately stays at bridge architecture
altitude and does not restate per-step workflow behaviour; this step file
plus the in-workflow comment at the fix are the record, so the next reader
does not look for a section that house convention puts in the workflow
itself.

## Why

- The step's whole contract is "a bad release is visible, not confused
  with the change under review" (`.github/review/rules/ci.md`). An
  exit-120 annotation on the version step muddies exactly that separation,
  which is how KBR-319's triage conflated infrastructure noise with the
  merge-gating failure.
- Capturing before truncating is the only shape that cannot regress into
  the broken pipe: any pipe into `head` re-creates it, which is why the
  static pin forbids the shape rather than the exact line.
- The stub in the test flushes the first line, delays, then prints the
  second — the measured shape of the real output — so the behavioral case
  is deterministic rather than a scheduling race.

## Tests

- `tests/test_review_workflow_cli_contract.py`:
  - `TestTheVersionProbeDoesNotBreakItsOwnPipe::test_the_step_body_survives_a_two_line_version_output`
    — extracts the step's `run:` body from the workflow YAML, runs it the
    way the runner does (`bash -e -o pipefail`) against the two-line stub,
    asserts exit 0, `Bridge:` on stdout, and the version stamped into the
    summary file.
  - `TestTheVersionProbeDoesNotBreakItsOwnPipe::test_the_probe_leaves_no_writer_upstream_of_head`
    — static pin: the `--version` capture line carries no pipe at all, so
    no truncating reader (`head`, `tail`, `sed -n 1p`, …) can re-create
    the broken pipe.
