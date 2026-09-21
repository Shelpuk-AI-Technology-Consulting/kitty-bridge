---
id: kbr294_provision_live_probe_credentials
depends_on: []
---

# KBR-294 — Provision paid provider credentials for the owner-side live probes

Ticket: [KBR-294](https://shelpuk.atlassian.net/browse/KBR-294). Requirements:
`.requirements/20260921T155257Z_provision_live_probe_credentials/REQUIREMENTS.md`.
Design: `TEST_SUITE.md` §8.7 (added in this step).

## What

Provisioned the two paid credentials that gate KBR-246 (thinking-carriage
live-verify) and KBR-252 (image/document live-verify), into the owner box's
**kitty config — deliberately outside the repository** (2026-09-21):

- Profile `anthropic-firstparty` (provider `anthropic`, model `claude-haiku-4.5`,
  `https://api.anthropic.com`) — a first-party `sk-ant-…` key.
- Profile `opencode-go` (provider `opencode_go`, model `mimo-v2.5`,
  `https://opencode.ai/zen/go`) — an OpenCode Go paid key.

Both were written through kitty's own stores (`CredentialStore.set` +
`ProfileStore.save` — base64 + 0600 + atomic writes, not hand-edited JSON), a
digest round-trip confirms the values survive the encode, and both pass
kitty's own pre-flight (`validate_api_key`) with `valid=True` and
`warning=None`. Scope of that evidence, precisely: the empty-warning state
rules out the timeout / connection-error paths (`validate_api_key` returns
`valid=True` *with* a non-empty warning there), so the upstream was reached
and the probe was not rejected as 401/403 — but for `opencode_go` the probe
itself is rejected pre-auth (400 MissingSessionID: the CC key-probe carries
no `x-opencode-session` header, owner-replicated 2026-09-21), so pre-flight
proves **reachability**, not key authentication, on that adapter. The
key-authentication evidence for both profiles lives in the owner's KBR-252
live session (real model calls, outcomes recorded on that ticket), not in
`validate_api_key`.
Neither profile is a member of any balancing pool and neither is default, so
paid credits burn only in an explicitly launched probe session — with the
caveat that "not default" is a current-state property, not an invariant:
deleting the default profile auto-promotes the first remaining profile as
the new default (README, profile-management notes), so re-verify the
not-default property after any profile deletion.

The code change on this ticket is **documentation only** (the §8.7 note in
`TEST_SUITE.md`): the ticket's scope is credential acquisition; the probes
themselves stay on KBR-246 / KBR-252 and run owner-side, per the KBR-238 AC-4
precedent.

## Why the keys live in kitty config and not in CI or the repo

`§8.6` shows where the nightly `agent_live` and `eval` jobs get their
credentials — the `KITTY_CREDENTIALS_JSON` CI secret. That channel serves
recurring, framework-controlled jobs and deliberately withholds secrets from
fork PRs. The live-verify sessions KBR-246 / KBR-252 need are different:

- They are **owner-performed one-offs** on the box that owns the account, run
  against the real upstream to settle a specific behavioural question (thinking
  carriage in KBR-228, image carriage in KBR-222). The probe script attached
  to KBR-228 is the template; KBR-294 does not write it.
- The dependency on the keys is **visible on the backlog** as a credential
  ticket that `blocks` the verification tickets, so the gate cannot be
  silently dropped.
- The keys themselves are committed nowhere — committed files, PR descriptions,
  and Jira comments are all banned by AC-3. A CI secret reaches a PR comment
  only if a developer pastes it; a box-local credential store, with the keys
  served by the same code path a `kitty` launch already exercises, removes that
  hazard by construction.

## What this step does not touch

- **No probe code** is written here. The probe script is attached to KBR-228
  and the verification sessions are owner-side; both are explicitly out of
  scope per the ticket's "credential acquisition only" framing.
- **No redaction helper** is added. Two existing rules together cover the
  credential echo surface: `ProviderAdapter.redact_url_for_display` drops
  userinfo outright and replaces every URL query value and the fragment with
  a fixed `****` mask (parameter names kept for diagnostic value, length not
  preserved — covering the URL-echo paths at `validation.py:66` and
  `BridgeServer._debug_url`); and `BridgeServer._debug_headers`, keyed on
  `_CREDENTIAL_NAME_STROKES = ("auth","key","token","cookie","secret","signature")`
  and codified in `SYSTEM_DESIGN.md` §9.2, masks a header's value when its
  **name**, lowercased, contains any stroke — a name-only rule, deliberately
  divergent from the URL rule's mask-everything principle because a value
  rule would redact the whole header dump. The corollary a probe author must
  know: a credential echoed as the *value* of a header whose name carries no
  stroke is **not** masked. The header rule is the one that actually
  protects `x-api-key` and `Authorization: Bearer` echoes — those travel in
  headers rather than URLs. The probe script's own printing is owned by
  KBR-246 / KBR-252.
- **The CI cadence is unchanged.** §8.6's nightly jobs continue to source from
  the `KITTY_CREDENTIALS_JSON` secret; the box-local profiles in §8.7 are for
  owner sessions only.

## Residue / open items

- **KBR-294 AC-1's account-class line** (strict prefix check on / off) is
  owner-known from the KBR-238 probes (2026-09-13); recorded on the ticket
  once the owner states it. Until then KBR-294 stays *In Progress* on AC-1's
  account-class clause alone — the key-availability clauses (ticket AC-1 and
  AC-2), the no-leak clause (AC-3) and the pool-exclusion criterion
  (REQUIREMENTS.md AC-4) are all met.
- **Stale `validate_api_key` docstring observed while provisioning.** The
  docstring (lines 95–107 of `src/kitty/validation.py`) says "Anthropic,
  Bedrock, Vertex" are skipped as custom transports, but `use_custom_transport`
  defaults `False` (the only overrides are `bedrock` / `ollama_cloud` /
  `openai_subscription`), so `validate_api_key` actually runs for `anthropic`
  and `opencode_go`. Pre-flight for both new profiles therefore posts a real
  request; `valid=True` with `warning=None` proves the upstream was reached
  and the probe was not 401/403 — for `anthropic` that is authentication
  evidence, while for `opencode_go` the probe is rejected pre-auth (400
  MissingSessionID) so it proves reachability only (see the What paragraph).
  Noted
  here as an observation; a docstring fix is **not this ticket's scope** —
  flag with the owner if you want it ticketed.

- **Real key state after provisioning (owner-verified, 2026-09-21).** Both
  credentials exist and decode to the keys the KBR-252 owner session used,
  but neither account was immediately probe-ready: the first-party Anthropic
  account is usage-capped until 2026-10-01 (400 on every request) and
  `opencode-go`'s subscription gate rejects model-route requests (403) —
  owner-recorded on KBR-252 from its live session. This confirms §8.7's
  gate framing empirically: keys present and validate-clean is necessary
  but not sufficient for the verification tickets to run.

- **Runbook — rotation, reprovision, cleanup.** On key rotation: revoke at
  the provider console, then drop the existing profile via `kitty profile`'s
  interactive flow (`_delete_profile_flow` at
  `src/kitty/cli/profile_cmd.py:455-509` — also clears the orphaned
  `auth_ref` in `credentials.json`), then re-run the provisioning ingest and
  annotate this step's date on rotation. The same delete-then-add is the
  clean reprovision path; the ingest script refuses to re-create a profile
  that already exists, by design. The pre-change backups
  `~/.config/kitty/{profiles,credentials}.json.bak-kbr294-20260921T155257Z`
  are retained until the first successful probe session on either new
  profile confirms behaviour, then deleted (the store's own atomic-write
  backup on damage is unrelated to these).

- **Deferred follow-up — tracked-file credential-shape guard.** AC-3's
  enforcement today is a one-shot grep on the worktree; the repo gains no
  durable invariant from that. A structural test scanning tracked files
  for key-shaped literals (`sk-ant-[A-Za-z0-9_-]{20,}`, `ghp_`, `gho_`,
  long Bearer JWTs) — sitting beside `tests/test_internal_key_completeness.py`
  and the §6.2.3 internal-key guards — would enforce AC-3 structurally.
  Deferred: the ticket is explicitly credential-acquisition-only and adding
  a new structural test belongs on its own ticket. Flagged here so the
  choice is a stated deferral, not an omission.
