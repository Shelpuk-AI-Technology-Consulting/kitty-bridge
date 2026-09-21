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
kitty's own pre-flight (`validate_api_key`): a non-401/403 response proves the
key authenticated. Neither profile is a member of any balancing pool and
neither is default, so paid credits burn only in an explicitly launched probe
session.

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
- **No redaction helper** is added: `redact_url_for_display` (the style AC-3
  names) already covers every URL-echo path (`validation.py:66`,
  `BridgeServer._debug_url`, the DEBUG-log redaction policy in
  `SYSTEM_DESIGN.md` §9), and the probe script's own printing is owned by
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
  request and a non-401/403 is genuine auth proof. Noted here as an
  observation; a docstring fix is **not this ticket's scope** — flag with
  the owner if you want it ticketed.
