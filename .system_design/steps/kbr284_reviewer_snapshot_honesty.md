---
id: kbr284_reviewer_snapshot_honesty
depends_on: []
---

# KBR-284 — Reviewer snapshot honesty

Plan: `REQUIREMENTS.md` at
`.requirements/20260918T181500Z_reviewer_snapshot_honesty/REQUIREMENTS.md`.
Jira: **KBR-284** ([link](https://shelpuk.atlassian.net/browse/KBR-284)).
Design: `.github/review/README.md` ("What the reviewer is given"),
`fetch_conversation.py` module docstring and `render` docstring.

## What the task does

Closes the failure class where a review round asserts, confidently, that
prose which exists does not exist — because its conversation snapshot
predated it. Observed live on PR #225 (2026-09-18): the round-3 review's
`conversation_notes` claimed "no author-side prose" while sixteen author
replies sat fourteen minutes old; the job's fetch had run before they
landed, so the claim was true of the snapshot and false of the pull
request, and nothing in the span or the notes disclosed the boundary.

The change:

1. `fetch_conversation.render` gains an opt-in keyword-only `fetched_at`
   parameter and emits a `# Snapshot` header in all four rendering paths:
   excerpt, complete copy, and both empty-conversation branches. Counts
   describe the fetch, not the budget, and are derived inside
   `_snapshot_header` from the entries `render` already receives.
2. `fetch_conversation.main` stamps the fetch time once, **immediately
   before the first `_api`/`_threads` call** — a strict lower bound:
   anything posted at or after that instant is unknown to the review — and
   passes it on both `render` calls. It does not derive or pass counts.
3. `REVIEW_PROMPT.md` adds the matching rule: the span is a snapshot;
   anything posted at or after its timestamp is invisible to this review;
   never assert absence about the pull request — only about the snapshot,
   citing its timestamp; post-snapshot contributions are unknown, not
   silence.
4. `.github/review/README.md` documents the header, so the subsystem's own
   design doc agrees with the behaviour. The `# Snapshot` heading is a
   public constant (`SNAPSHOT_HEADER`) and the prompt⇄script⇄README
   wiring tests import it; the heading shape joins `_FORGED_HEADING` so a
   forged snapshot header in a comment body is neutralised like the entry
   headings.

Non-goal (scoped as a follow-up, not filed as a ticket): re-fetching
after the model returns and disclosing mid-run drift on the posted
review — a separate increment with its own payload plumbing; the
snapshot rule is the floor. Tracked as a scope note on the KBR-125 epic
rather than a standalone ticket.

## Implementation notes

### What landed (2026-09-18)

1. `fetch_conversation.py` — `SNAPSHOT_HEADER = "# Snapshot"` public
   constant beside `FULL_COPY_NAME`; `_snapshot_header(fetched_at, entries)`
   helper (counts derived inside from the entries `render` already receives,
   over a chronological view, distinct kind strings verbatim in order of
   first appearance, `contributions: 0` when the fetch is empty);
   keyword-only `fetched_at: str | None = None` on `render`; header inserted
   in all four render paths (excerpt, complete copy, both empty branches),
   conditional so the no-snapshot rendering stays byte-identical; the
   empty-branch silence sentence is past perfect ("Nothing had been said…")
   only when the header bounds it; `|snapshot` joined `_FORGED_HEADING`;
   `main()` stamps `datetime.now(timezone.utc)` immediately before the first
   `_api` call (strict lower bound, D5) and passes it to both `render`
   calls.
2. `test_review_scripts.py` — `SnapshotHeaderTests`, nine tests: header
   shape/order on the excerpt path; byte-identical header between excerpt
   and complete copy (three-line extraction, bounded by line count so the
   omission notice cannot desync it); empty branches with past-perfect
   silence; counts over the full fetch under a tight budget; kind order of
   first appearance; AC5 golden equality (pre-change byte sequence pinned);
   AC6 end-to-end through `main()` with monkey-patched `_api`/`_threads`
   (one stamp, two files equal, counts from payloads, ±60 s wall clock);
   AC7/AC8 wiring against the public constant; R8 forged-heading
   neutralisation.
3. `REVIEW_PROMPT.md` — the snapshot rule paragraph in "Read the
   conversation before anything else": the header names the instant the
   fetch began; anything at or after it is invisible, including replies
   written mid-run; absence is claimed as of the snapshot, never about the
   pull request.
4. `.github/review/README.md` — "What the reviewer is given" documents the
   header and the lower-bound rule; test-count line 729 → 738.
5. `.github/workflows/ci.yml` + `pyproject.toml` — the repo's own
   `RecordedTestCountTests` guard caught the other two 729 quotes in the
   first full-suite run; both updated to 738.

Implementation-time findings:

* The AC2 extraction first grabbed the excerpt's omission notice (the
  header block is followed by different content per path); fixed by
  extracting exactly the three header lines, bounded by line count.
* The AC6 fixture first used a bodyless COMMENTED review — dropped by
  `_entry` by design (`keep_empty` only for APPROVED/CHANGES_REQUESTED),
  so the counts line could never name it; the fixture now gives the review
  a body.
* `_snapshot_header` counts over `sorted(entries, …)`: production always
  arrives sorted (`collect` sorts), and the counts should describe the same
  order the rendered span reads in regardless of caller order.

Gates: review-script suite 738/738 OK · `ruff check .` — zero new errors
(20 pre-existing on untouched lines, the documented unbounded dev-pin
drift; this PR's files clean) · `lint-imports` 5 kept, 0 broken ·
`mypy src/kitty` 94 files, no issues.

## Cross-references

* `.github/review/scripts/fetch_conversation.py` — `render`, `main`.
* `.github/review/REVIEW_PROMPT.md` — "Read the conversation before
  anything else".
* `.github/review/README.md` — "What the reviewer is given".
* `.github/review/tests/test_review_scripts.py` — `SnapshotHeaderTests`
  (new), beside the existing `render` test classes.

## Status

Implemented; code review round 1 returned one finding (this file's own
"What the task does" still described the pre-review design — post-fetch
stamp, caller-supplied counts), fixed in the same pass (2026-09-18).
system-design-reviewer ran three rounds on REQUIREMENTS.md (one blocker on
stamp direction, six concerns, three suggestions — all folded; final round
returned "Ready for implementation"). code-reviewer verified every gate
(suite 738/738, ruff zero new errors, lint-imports 5/0, mypy clean) and
every AC trace before the step-file sweep. Implementation notes above
record the two implementation-time findings and the count-guard catch.
