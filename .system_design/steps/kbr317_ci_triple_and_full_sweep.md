---
id: kbr317_ci_triple_and_full_sweep
depends_on: [kbr317_readme_remote_subscription]
---

# CI triple + full sweep + PR

KBR-317, step 6 of 6. Pre-push verification and the pull request.

## Why

The lint/import/type triple runs in CI **before** pytest (see the
`ruff-lint-gate` memory), so a green local pytest proves nothing if lint is
red. The full sweep (whole tree, not subdirs) is the repo's pre-push
contract, and the `server.py` one-function-scope mypy trap means mypy must
run locally too.

## To Be

1. `uv sync --extra dev` (the bare `uv sync` trap strips dev tools).
2. `ruff check .`
3. `lint-imports`
4. `mypy src/kitty`
5. Full `pytest` (whole tree, isolated — no concurrent reviewer agents per
   the concurrency memory).
6. `python scripts/regenerate_step_index.py` exits 0.
7. Branch + commit + push + open PR; post the Jira comment; watch PR
   checks and reviewer comments.

## Acceptance

- All six local checks exit 0.
- PR open with a description that carries product context for the PO.
- Jira comment records what was implemented and how.

## Implementation notes

- Local CI triple before push: `ruff check src/ tests/` clean (the `.tmp/`
  scratch findings are untracked and never reach CI), `lint-imports` 5/5
  kept, `mypy src/kitty` clean, step index exit 0.
- **Line-ending churn caught pre-commit** (per the
  `serena_edits_rewrite_line_endings` memory): one Serena edit rewrote the
  CRLF `main.py` to LF — the diff showed 1730 changed lines for a ~50-line
  edit. Restored to CRLF; the committed diff is content-only.
- Code review round 1 (code-reviewer subagent): APPROVED-pending-fixes —
  1 critical (FR-4 plain-copy form unimplemented), 3 high (vacuous L2
  request-shape tests, cancellation landing outside the try, missing
  to_dict equality test), plus warnings (dispatch-rule duplication →
  `select_device_flow` in `kitty.auth.headless`; bare `kitty auth`
  implicit-openai default restored; vacuous dispatch test strengthened;
  README transcript completed). All addressed; 146 auth+cli tests green
  afterwards.
- Test-infra lessons recorded in the device-flow step file
  (aioresponses: yarl.URL keys, consumed-once registrations; no-op sleep
  stubs freeze the loop's clock and break timeout/cancellation tests).
