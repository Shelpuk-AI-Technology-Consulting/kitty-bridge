---
id: balancing_subscription_spawn_identity
depends_on: [kbr319_launcher_eligibility, kbr319_exact_messages_relay]
---

# Balancing-pool subscription spawn identity

## Planned delivery

A Claude-launched balancing pool containing an `anthropic_subscription` member must spawn the
agent from that member's profile, so Claude Code presents its own logged-in OAuth identity on the
raw subscription relay. Today `launch_async` builds the spawn configuration from the launch
profile — the pool's first member — so a mixed pool injects the first member's API key and the
`kitty-bridge-token` placeholder into the child; the relay forwards both to
Anthropic. When the first member is a non-Anthropic provider, the key is one
Anthropic never issued, which rejects them with 401, the subscription member
parks for 900 s, and the
client sees `503 All 1 backends rejecting the credentials (401/403)` while every API-key member
of the pool sits healthy but unreachable for that request. When the first member is itself an
`anthropic` key profile, the forwarded key is one Anthropic did issue, so the outcome is not
established as a 401 — the request may be accepted and the subscription member's share billed to
that key, a silent variant of the same defect. Either way the child presented a foreign identity
instead of Claude Code's session.

Requirements, acceptance criteria, and the test plan live in
`.requirements/20261020T135901Z_balancing_subscription_spawn_identity/REQUIREMENTS.md`; the
To-Be contract is recorded in `SYSTEM_DESIGN.md` §16.1/§16.2.

## Decisions and why

- **Spawn from the subscription member rather than strip synthetic headers in the relay.** With
  `ANTHROPIC_API_KEY` set, Claude Code presents the API key instead of its OAuth token, so there
  would be no real identity left to forward; the child must run with its own session, which is
  exactly the KBR-319 direct-launch contract reused for pools.
- **Select in `launch_async`, no signature change.** The pool (backends) is already in scope at
  the `build_spawn_config` call site; every launch path (sync and async) funnels through it, and
  non-Claude pools cannot carry a subscription member (they are filtered before launch), so other
  adapters are unaffected.
- **Accept the model-hint change.** `ANTHROPIC_MODEL` and the default-model aliases now hint the
  subscription member's model for mixed pools. On the translated path the hint is inert (the
  bridge rewrites the request model to the selected backend's), but the raw subscription relay
  posts the child's bytes unchanged and never normalizes the model — so on that path the hint is
  what determines the model Anthropic receives, which is exactly why it had to move.

## Implementation notes

Implemented 2026-10-20 on `fix/balancing-subscription-spawn-identity`.

- `launch_async` (src/kitty/cli/launcher.py) selects `spawn_profile`: the first
  `anthropic_subscription` member of `backends` when the adapter is `claude`
  (the same gate that arms `allow_subscription_relay`), else the launch profile.
  No signature change; `launch()` forwards `backends` on every branch, so the
  sync wrapper shares the path.
- L1 tests in tests/test_launch_orchestrator.py
  (`TestMixedPoolSubscriptionSpawnIdentity`): mixed pool spawns neither
  `ANTHROPIC_API_KEY` nor `ANTHROPIC_AUTH_TOKEN` (parent-seeded values included)
  and the session settings file pins both empty; a subscription-led pool matches
  the direct-launch identity (`ANTHROPIC_MODEL` = the member's model); a
  non-Claude adapter keeps the launch profile; a subscription-free pool keeps
  the synthetic first-member credentials.
- Accepted trade-off: `ANTHROPIC_MODEL` and the default-model aliases hint the
  subscription member's model for mixed pools — decisive on the raw relay path
  (bytes are forwarded unchanged, no model normalization), inert on the
  translated path (the request model is rewritten per selected backend).
- Deferred follow-up: the terminal all-unhealthy 503 for a parked subscription
  member still uses the API-key-dialect headline; a subscription-aware headline
  (remediation `claude auth login`) was recorded in SYSTEM_DESIGN.md §16.1 as a
  deliberate non-goal of this stage.
- Review rounds (CI bot, five rounds): credential now travels with the spawn
  profile (`_select_spawn_profile` returns the pair, so a subscription member
  never carries the first member's key into `build_spawn_config`), and the
  mixed-pool test records the credential handed to `build_spawn_config`,
  asserting it is the member's own (empty) key — the pairing is pinned (verified
  by reverting the pairing and watching the test fail); the
  subscription-led test launches with the subscription profile itself — the only
  shape `_launch_target_balancing` can produce — and the reachable mixed-pool
  test also asserts the model hint; README documents the pool behavior; the
  pre-fix outcome is qualified (a non-Anthropic first member's key is rejected
  with 401 and parks the member; an `anthropic`-key first member yields a key
  Anthropic accepts, so the defect there is silent over-billing rather than a
  503), and the remedy wording is `claude auth login`.
- Two further limits are recorded rather than fixed: a pool listing several
  subscription members hints the first one's model (on the raw path the model
  travels inside the forwarded bytes, so a later member's differing model cannot
  be honoured — the L1 test `test_two_subscription_members_hint_follows_the_first`
  pins it), and an `auth` park is a fixed 900 s cooldown that only expiry
  clears, so `claude auth login` does not put the member back into the pool
  immediately.
