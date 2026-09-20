---
id: kbr286_claude_md_review_resolution_discipline
depends_on: []
---

# KBR-286 — Claude Code PR review resolution discipline

Plan: `REQUIREMENTS.md` at
`.requirements/20260918T203713Z_claude_md_pr_review_resolution_discipline/REQUIREMENTS.md`.
Jira: **KBR-286** ([link](https://shelpuk.atlassian.net/browse/KBR-286)).
Design: repo-root `CLAUDE.md` (new), `tests/test_project_claude_md_review_discipline.py`
(new).

## What the task does

Writes down, in the repository, how an agent must behave when it takes on a
GitHub code-review conversation: enumerate every open conversation and inline
thread before answering any; post one PR-level comment per round stating what
was addressed and implemented and what was deferred; reply in thread where
detail helps; resolve every conversation that was taken and fixed; and for a
conversation it will not address, post an in-thread reply naming the reason
and leave the conversation open (never silently resolved, never resolved
without a stated reason).

Until now the behaviour lived nowhere for this repo: the "project-level
CLAUDE.md" Claude Code loads is the user's home file, outside every git
repository, so it cannot be versioned, reviewed, or PR'd, and it applies to
all of the user's projects rather than to this one. The same silent-resolution
opacity this change removes has been observed in practice (PR #177, PR #212).

The change:

1. A new repo-root `CLAUDE.md` carrying the instruction (the first
   project-level instruction; future ones follow the same file).
2. `/CLAUDE.md` removed from `.gitignore` — the line was added in commit
   `b201181` (2026-04-27) without rationale as an incidental one-line side
   effect of an unrelated provider-grouping change, and made the ticket
   unshippable.
3. A new L2 contract guard, `test_project_claude_md_review_discipline.py`,
   pinning the file's presence, tracking, ignore-status, and the four
   mandated behaviours, with anchors asserted inside the review-resolution
   section only, whitespace-normalised so a re-wrap cannot break them, and
   the file read with `encoding="utf-8"` (the KBR-266 cp1252 trap).
4. The guard registered in `test_layer_selection.py`'s explicit-L2 set
   (KBR-280 rule), with its module docstring first line declaring it a
   contract guard so the §6.2 scan finds it.

## Implementation notes

### What landed (2026-09-18)

1. `CLAUDE.md` — the "Resolving GitHub code reviews" section states the four
   mandated behaviours as numbered rules; the CI review bot (comments only,
   never resolves) is scoped out in the section's closing paragraph.
2. `.gitignore` — the `/CLAUDE.md` line removed; `git check-ignore CLAUDE.md`
   exits 1.
3. `tests/test_project_claude_md_review_discipline.py` — `pytestmark =
   pytest.mark.l2`; `TestProjectClaudeMdShips` (exists / tracked via
   `git ls-files --error-unmatch` / not ignored via `git check-ignore`, exit
   1 is the only acceptable code) and `TestTheReviewResolutionSection`
   (parametrized over the four behaviours, one test per behaviour so a red
   run names its cause).
4. `tests/test_layer_selection.py` — the explicit-L2 set gained the new file
   with a KBR-286 comment.

Implementation-time findings:

* The first green run failed two clause anchors because `CLAUDE.md` wraps at
  ~78 columns and the anchor phrase straddled a line break. Wrapping is
  formatting, not behaviour: `_review_section` now collapses whitespace runs
  before the anchor search. Caught by the red phase, exactly as intended.
* AC2's clause-removal checks are dev-time mutation checks, per the §6.2
  self-check pattern: the four clauses were each removed by hand and the
  guard was re-run to watch the corresponding test fail, before the final
  text was restored. The ignore-line and missing-file red states were
  observed live during the TDD loop itself.
* The whole-suite layer fixtures need `schemathesis` importable (two openapi
  test files fail collection otherwise); on this worktree it had to be
  installed into the venv by hand — a local-environment gap, not a repo
  defect, and pre-existing on `main`.

Gates: `test_project_claude_md_review_discipline.py` 8/8 ·
`test_layer_selection.py` 19/19 (both pins green, the §6.2 contract-guard
scan finds the new file) · review-scripts suite 738/738 · step-index
validator unchanged (its only red remains the pre-existing `t_g6 → t_w8`
dangling dep on `main`, tracked in KBR-278's open follow-up; this step adds
no dangling dep) · `ruff check` on touched files clean.

### Review rounds (PR #230)

* **Internal code review round 1** (REQUEST_CHANGES): (a) clause-(b) anchor
  `("in-thread",)` also matched rule 4's wording, so removing rule 3 did
  not trip the guard — re-anchored to rule-3-unique wording and verified
  with a whole-rule deletion probe (each of rules 1–4 removed in turn →
  every removal detected); (b) the `.gitignore` edit had flipped 9
  content-unchanged lines LF→CRLF — rewritten byte-exact to the merge-base
  content minus the removed line; (c) the §6.2.3 register row was missing —
  added, per the KBR-216/KBR-280 precedent.
* **Internal code review round 2** (APPROVE): all three round-1 fixes
  independently re-verified; one documentary nit in a commit message
  deferred as not worth rewriting pushed history.
* **Automated reviewer round 3** (1 warning, addressed in commit `ee4e217`):
  plain `git check-ignore` short-circuits tracked files out of the ignore
  consultation, so the guard was green in exactly the re-ignore scenario it
  exists to detect. Verified empirically (with `/CLAUDE.md` re-added:
  plain form exits 1, `--no-index` form exits 0); fixed with `--no-index`;
  the same false claim corrected in the module docstring, the class
  docstring, the method comment, and the §6.2.3 row; a re-ignore probe
  confirms the corrected guard fails with the line re-added. The inline
  thread was replied to and resolved; the PR-level round comment states
  one finding addressed, nothing deferred.
* **Automated reviewer round 4** (4 findings across two reviewer passes,
  addressed in commit `f27517c`): (a) three further sub-clause holes
  confirmed empirically before the fix — every anchor made unique to its
  rule's wording, re-probed after; (b) ten-plus sites in `.github/review/`
  still said `.gitignore` excludes `CLAUDE.md` — prose + the
  `FORWARD_LOOKING_LITERALS` excuse + the `ALLOWED` fixture pins updated
  in lockstep (one accidental CLAIMS-window trip caught and reworded);
  (c) the "Self-guarded" claim gained an in-suite section-scoping check
  (`test_the_section_extraction_does_not_leak_preamble_text`); (d) the
  exit-128 failure message branched so a broken git reports the git
  failure, not a phantom ignore pattern. All four threads replied to and
  resolved.
* **Merge-conflict rebase (2026-09-19)**: main advanced through KBR-282 /
  KBR-87 / KBR-52 / KBR-268; the §6.2.3 table re-piled three times
  (KBR-282 widened the neighbouring row twice under my feet) — resolved by
  keeping the neighbour's current row and my KBR-286 row, latest version.
* **Automated reviewer round 5** (4 findings on the rebased head, this
  commit): (a) `scripts/regenerate_step_index.py` and `tests/test_step_index.py`
  cited a `§System-design discipline` CLAUDE.md section the tracked file
  has never carried (the citation predates KBR-286 and pointed at the
  per-developer home file) — citations now name KBR-278 directly; (b)
  three further sub-clause anchors added (rule 1's disposition assignment,
  rule 2's "addressed and implemented" and "posted once per round"), each
  probed after; (c) the `_CLAUSES` comment reworded so section-scoping is
  attributed to the in-suite probe and uniqueness to the hand-run rewrite
  probes; (d) this step file's stale counts and round list updated.
* **CI**: all 14 checks pass on the rebased head (test matrix
  3.10–3.13 × Linux/macOS/Windows, `review`, `ci-required`, CodeQL,
  Analyze, review-scripts, review_replies, update-metadata).

## Cross-references

* `CLAUDE.md` — "Resolving GitHub code reviews".
* `tests/test_project_claude_md_review_discipline.py` — the guard.
* `tests/test_layer_selection.py` — explicit-L2 expected set.
* `.requirements/20260918T203713Z_claude_md_pr_review_resolution_discipline/REQUIREMENTS.md`
  — decisions D1–D4 awaiting the reporter's confirmation on the PR.

## Status

Implemented; five review rounds closed (two internal, three automated), all
14 CI checks pass on the rebased head, nine inline threads resolved (0 open,
1 reply-and-resolve per the shipped rule). PR #230 remains open awaiting the
reporter's confirmation of decisions D1–D4 plus any further review before
merge.
