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

Pending.
