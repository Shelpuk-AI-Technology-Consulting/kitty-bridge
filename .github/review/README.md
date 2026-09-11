# Automatic pull request review

Every pull request against `main` is reviewed by Claude Code, running in
`.github/workflows/claude-code-review.yml`. The review is posted as a GitHub
review with inline comments on the lines it is about, and **a pull request that
was not reviewed fails the check** — a green tick has to mean a review actually
happened.

This file is for the two people who need it: whoever sets the workflow up, and
whoever answers what it finds.

---

## What the reviewer is given

| Piece | What it is |
|---|---|
| `REVIEW_GUIDE.md` | the repository-wide contract — what this product is, what a change must never weaken, how to rank severity |
| `REVIEW_PROMPT.md` | the task: read the conversation, then `README.md`, then the diff; sweep eight dimensions; return structured output |
| `rules/*.md` | per-component rules, selected by which files the pull request touched |
| `scripts/select_rules.py` | the map from changed paths to rule files, including the fan-out that makes a shared-module change pull in its consumers' rules |
| `schemas/review_findings.schema.json` | the findings contract the model's output is validated against |

The pull request's own conversation — description, comments, inline threads and
previous rounds — is fetched and put in the prompt **before** the diff, so the
reviewer can tell an addressed finding from an ignored one. It is fenced as
untrusted input: an attempt to instruct the reviewer through a comment is itself
reported as a critical finding.

⚠️ **The reviewer's specification is `README.md` plus `.system_design/`.**
`.gitignore` excludes `/.requirements/` and `/CLAUDE.md`, so neither exists in a
CI checkout and the reviewer must not be pointed at them. `/.system_design/` was
un-ignored when the design documents were first committed, and the selector
already matched those paths, so a change under them is reviewed with rules
loaded. The traceability target is therefore the README (the user-facing
interface contract), the design documents (the internal one), and the pull
request description — and `REVIEW_GUIDE.md` says exactly that rather than
pointing at a document that is not there.

---

## Answering a review

**Reply in the thread, verify it published, then resolve.**

```bash
gh api -X POST repos/{owner}/{repo}/pulls/{pr}/comments/{comment_id}/replies -f body='...'
gh api repos/{owner}/{repo}/pulls/{pr}/comments --paginate \
  --jq '[.[]|select(.in_reply_to_id=={comment_id})]|length'
```

🔴 **Not `addPullRequestReviewThreadReply`.** That GraphQL mutation attaches the
reply to *your pending review* — a draft only you can see — while
`resolveReviewThread` still works. The thread then reads as answered to you and
as closed in silence to everybody else. It reports success and says nothing.
Check for a `PENDING` review before resolving:

```bash
gh api graphql -f query='{repository(owner:"OWNER",name:"NAME"){pullRequest(number:PR){reviews(first:100){nodes{state}}}}}' \
  --jq '[.data.repository.pullRequest.reviews.nodes[]|select(.state=="PENDING")]|length'
```

Publish a draft you already wrote with `submitPullRequestReview(event:COMMENT)`.

⚠️ **Resolving is not answering.** The `review_replies` job in
`.github/workflows/ci.yml` blocks the merge when a resolved thread's comments are
all by one author, because that state has two causes and they are
indistinguishable from outside: the finding was dismissed without a word, or a
reply was written and never published. If a thread genuinely needs no answer —
your own question, a note on your own change — say so in the thread in one
sentence and then resolve it. That clears the check.

The gate reads the API live, so a re-run is enough; **do not push an empty commit
to clear it**, because that re-triggers the billed review.

### Disagreeing with a finding

Say so in the thread. The reviewer reads the conversation on the next round and
is told, in `REVIEW_PROMPT.md`, that an explanation may resolve a finding it
would otherwise raise — and equally that disagreement is information, not a
verdict: if the code is still wrong, the finding still stands however firmly
somebody argued.

---

## Setting it up

The reviewer is configured entirely by **organisation** Actions secrets and
variables, shared with the repository this system came from. Nothing is
repository-local.

| Setting | Kind | What it is |
|---|---|---|
| `KITTY_CREDENTIALS_JSON` | org secret | the Kitty Bridge credential store |
| `KITTY_EGRESS_JSON` | org secret | the egress gateway, as kitty's own `egress.json`, in its versioned envelope |
| `KITTY_PROFILES_JSON` | org variable | the kitty profile(s): endpoint and model |

To enable the workflow on this repository:

1. Grant this repository access to the three settings above (organisation
   settings → Secrets and variables → Actions → each item → repository access).
   All three were already granted when this workflow landed.
2. Nothing to do for runners — every job uses a GitHub-hosted label
   (`ubuntu-latest` for the review, the aggregate, the metadata refresh and the
   test matrix; `ubuntu-slim` for the two review-system jobs), so no runner group
   has to be granted.

   ⚠️ **Why not a self-hosted fleet:** a runner group carries an *"Allow public
   repositories"* setting that is **off by default**, and this repository is
   public. Opening a shared fleet to a public repository means anyone who can
   open a pull request gets closer to those machines; taking the hosted runner
   avoids that decision entirely.

   ⚠️ **If a job is queued and never starts, suspect the label first.** An
   unreachable runner label is completely silent — no error, no annotation, no
   timeout. Both labels used here are GitHub's own standard public ones, so
   neither depends on an organisation grant.

   🔴 **`ubuntu-slim` has a 15-minute job ceiling that no setting can raise.**
   GitHub's runner reference: *"The job timeout for single-CPU runners is 15
   minutes."* Upstream, the review job ran there with `timeout-minutes: 60` and
   was killed mid-run five times before anyone paired the two lines. It runs on
   `ubuntu-latest` here, whose ceiling is six hours. The two review-system jobs
   stay on `ubuntu-slim`, which is what that label is for. A test fails if any
   job ever declares a cap its runner will not honour — including a matrix job,
   which is held to the LOWEST ceiling among the labels its matrix can produce.
3. Open a pull request. The `review` check appears on it.
4. Once it has run green once, make `Claude Code Review` and `ci-required`
   required status checks on `main` — **after reading the two consequences
   below**, because both are easier to decide before enabling than after.

### Two things to know before making the check required

🔴 **A fork pull request never reports this check at all.** The job runs only for
a head branch in this repository, and a job whose `if:` is false reports no
status — so a required check stays pending for ever and only an admin can merge.
This repository is public, so fork contributions are a real path. Requiring the
check is a decision to merge them by admin override.

🔴 **A broken `kitty-bridge` release blocks every merge.** See the next section.

### The egress guarantee

**The reviewer reaches the model only through the configured egress gateway.**
That is not a default that can be left off — five separate things enforce it, and
`rules/ci.md` describes each one and why neither of the two redundant-looking
checks subsumes the other. If the gateway does not resolve, **no review is
attempted and the check fails**; it never quietly falls back to the runner's own
network path.

### 🔴 This repository is Kitty Bridge, and the reviewer runs on it

The workflow installs the **released** `kitty-bridge` from PyPI on every run —
not the pull request's own checkout. Reviewing with the PR's own bridge was
rejected: a broken change would then break its own review, and the failure would
be reported as a provider fault rather than as the change's own.

Three things follow, and the third is the one that costs time if it is not known
in advance:

1. **A green review says nothing about whether the bridge still works.** The
   reviewer never exercises the code it is reviewing.
2. **A bad release freezes every merge.** The last release is a live dependency
   of a required check. If it cannot resolve a gateway, every pull request fails
   review — including the one that fixes it. There is no in-repo bypass, and that
   is deliberate: a `skip-review` label would be reachable by anyone with write
   access and would defeat the gate it is meant to survive. **The break-glass is
   an admin merge.** The run summary records the installed `kitty --version` so a
   failure is attributable to a release rather than to the change.
3. **A change to the CLI surface the workflow drives breaks the reviewer one
   release later.** `tests/test_review_workflow_cli_contract.py` is the guard for
   that, and it is the only one there can be — by (1), no review can catch it.

### If the check goes red

The pull request gets a comment saying what happened, and the run summary carries
a per-attempt table. The outcomes mean different things:

- **`exhausted`** — the provider could not serve the request (quota, credentials,
  a transient error). Nothing is wrong with the change. Top up or wait, then
  re-run. The job already retried once by itself.
- **`fatal`** — the workflow or its settings. The diagnostic names which. The
  most common causes are one of the three `KITTY_*` settings missing or not valid
  JSON, and an egress document that parses but disables the gateway. Check the
  `kitty --version` in the run summary against the latest release before
  suspecting the settings.
- **`cancelled`** — the run did not finish. Either it hit the job cap, or a newer
  commit superseded it (the workflow cancels an in-flight review on a new push).
  Nothing is wrong with the change and nothing is wrong with the workflow: if a
  newer commit superseded this run, its own review reports; otherwise re-run.
  This appears in the run summary only — no comment is posted, so that a rapid
  push sequence does not comment once per superseded run.

One case is worth knowing in advance because it is easy to misdiagnose: **if the
job starts failing at `Install kitty-bridge and Claude CLI`, that is the runner
or the release, not the settings.** It is reported as `fatal` and points at the
`KITTY_*` settings, which will all be fine.

And one that is not a failure at all: **a check that never appears, or appears
and stays queued for ever, is a runner-label problem, not a review problem.** See
step 2 above.

---

## Changing the review system

`.github/workflows/ci.yml` runs `tests/test_review_scripts.py` on every pull
request — 642 tests over the selector, the classifier, the notices, the redactor,
the schema and the workflow's own wiring, plus the workflow parser, the
runner-ceiling table and the `ci-required` aggregation. Run them locally the same
way:

```bash
python .github/review/tests/test_review_scripts.py
```

Not through `unittest discover`: discovery imports the start directory as a
package and `.github` is not a valid package name. The suite runs on a bare
interpreter with no installed dependencies, on purpose — a broken review workflow
has to be diagnosable without provisioning anything first — so do not add a test
there that imports a third-party package. That includes `pyyaml`, which the
repository's own `tests/test_github_actions.py` uses freely and this suite may
not.

**Adding a source directory means adding it to `scripts/select_rules.py`.** A
path no pattern matches is reviewed with **zero rule files loaded**, and nothing
goes red to say so; the review just gets quietly shallower. A test asserts that
every tracked file in the repository selects at least one rule — per file, not
per directory, because a directory-level check passes while one file in it
selects nothing.

Every script here except `select_rules.py` is carried verbatim from the
repository this system came from, so fixes stay portable in both directions.
Changing one is a fork — say why, and consider upstreaming it.
