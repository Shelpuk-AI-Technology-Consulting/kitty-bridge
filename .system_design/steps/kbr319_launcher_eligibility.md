---
id: kbr319_launcher_eligibility
depends_on: [kbr319_subscription_profile]
---

# KBR-319 — Subscription launcher eligibility

## Planned delivery

Centralize the rule that `anthropic_subscription` profiles are usable only by the Claude launcher. Direct non-Claude and generic bridge requests fail before credential resolution or egress. A non-Claude balancing launch removes subscription members before backend construction; an empty eligible result is a configuration error. A Claude raw subscription attempt may fail over only to another subscription member, never to an API-key translation path.

## Decisions and why

- **Filter before backend construction.** An excluded profile must not influence credential lookup, egress checks, context-window calculation, selected spawn configuration, or random selection.
- **Fail closed for no eligible member.** Sending traffic through a fabricated/default member would violate the profile's declared policy and hide an operational configuration error.
- **Defend at the bridge boundary too.** CLI routing is the normal path, but embedded callers cannot bypass the product boundary by constructing incompatible backends directly.

## Implementation notes

Landed on feat/kbr-319-anthropic-subscription. Central `require_launcher_eligible_profile` /
`filter_launcher_eligible_profiles` (src/kitty/profiles/eligibility.py) gate direct launches before credential
resolution and filter balancing pools before backend construction, failing closed on empty pools. Subscription launches
bypass credential resolution and preflight validation; ClaudeAdapter omits and clears both Anthropic credential env
vars. BridgeServer gained default-deny `allow_subscription_relay`, set only for the Claude launcher. Reviewed:
code-reviewer APPROVE (R3/R4/R5/R6 + R11).

Review round (2026-10-18, PR #294 CI review): generic `kitty bridge` had missed the shared gate. A direct
subscription profile falsely reported a missing API key; a balancing pool tried to read the first subscription member
before filtering. `_run_bridge` now invokes `require_launcher_eligible_profile(profile, None)` before a credential
lookup, and `_run_bridge_balancing` invokes `filter_launcher_eligible_profiles(..., None)` before member wiring,
with the existing no-members-eligible wording for bridge mode. Three receiver tests pin direct refusal, an
all-subscription pool, and a mixed pool that resolves only the eligible member's credential.

Code-review follow-up (same round): the background runner (`kitty bridge start`, `bridge_runner.main`) was the
sibling arm with the identical gap; it now applies the same gate and filter before its credential reads, pinned by
tests/test_bridge_runner_eligibility.py (direct refusal, empty eligible pool, mixed pool wiring only the eligible
member).
