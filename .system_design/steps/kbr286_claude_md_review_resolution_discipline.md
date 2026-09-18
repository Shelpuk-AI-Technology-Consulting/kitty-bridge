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

Gates: `test_project_claude_md_review_discipline.py` 7/7 ·
`test_layer_selection.py` 19/19 (both pins green, the §6.2 contract-guard
scan finds the new file) · step-index validator unchanged (its only red
remains the pre-existing `t_g6 → t_w8` dangling dep on `main`, tracked in
KBR-278's open follow-up; this step adds no dangling dep) · `ruff check` on
touched files clean.

## Cross-references

* `CLAUDE.md` — "Resolving GitHub code reviews".
* `tests/test_project_claude_md_review_discipline.py` — the guard.
* `tests/test_layer_selection.py` — explicit-L2 expected set.
* `.requirements/20260918T203713Z_claude_md_pr_review_resolution_discipline/REQUIREMENTS.md`
  — decisions D1–D4 awaiting the reporter's confirmation on the PR.

## Status

Implemented; code review pending (the PR carries decisions D1–D4 flagged for
the reporter).
