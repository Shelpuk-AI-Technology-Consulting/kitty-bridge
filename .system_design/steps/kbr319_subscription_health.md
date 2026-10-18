---
id: kbr319_subscription_health
depends_on: [kbr319_exact_messages_relay]
---

# KBR-319 — Subscription relay health and cooldowns

## Planned delivery

Route raw relay failures through the existing backend health state. The failure carrier preserves status, a bounded body, and parsed `Retry-After`: rate limits honor retry hints, authentication failures park as `auth`, plan failures park as `entitlement`, and transport or other upstream failures use existing kinds. Streaming failures before emitted content use the same classifier.

## Decisions and why

- **Reuse the existing circuit breaker.** A second health model would produce divergent selection, observability, cooldown, and recovery behavior for a profile that is otherwise a normal balancing member.
- **Claude Code owns token refresh.** Kitty only relays the request it receives; a session/auth failure is surfaced and parked rather than refreshed from a copied credential.

## Implementation notes

Landed on feat/kbr-319-anthropic-subscription. The relay balancing loop parks failures through the shared precedence
(cloudflare > entitlement > rate_limit > auth > hard; transport parked by the loop) with Retry-After-honoring
cooldowns, fails over only within subscription members (`require_subscription` threaded through selection, skipping
the cooldown gamble per R9), and returns the existing all-unhealthy 503 when none remain. 6 L3 tests in
tests/bridge/test_subscription_relay_health.py. Reviewed: code-reviewer APPROVE (R9).
