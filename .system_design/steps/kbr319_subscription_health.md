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

Review round (2026-10-18, PR #294 CI review): read timeouts are now the one classified non-parking failure — the
handler answers the translated path's 504 `api_error` envelope without parking or failing over (owner decision:
mirror the translated Messages path's timeout semantics; a timeout is a slow request, not a sick backend). Recorded
in SYSTEM_DESIGN §16.1; two L3 tests pin the envelope and the empty health state.

Review round 2 (same day): auth-class relay rejections now advise the Claude Code session (`claude auth login`)
instead of the shared translator's API-key text; a 403 the classifier calls entitlement reads as a plan denial, and
the Cloudflare branch keeps the shared message. Usage attribution now works in both stream modes: an Anthropic→
attribution key mapping feeds `_log_usage` (the raw block's `input_tokens`/`output_tokens` were being read as zeros),
and the streaming pump reads `message_start`/`message_delta` usage counters from complete SSE lines while forwarding
bytes untouched — a mid-stream failure still records what it saw. The `Retry-After` parsing rule moved to the shared
`kitty.providers.base.parse_retry_after_seconds`; both subscription consumers delegate to it. Tests: the 401/403
envelopes (names `claude auth login`, never `kitty setup`), non-streaming and streaming token attribution with the
byte-exactness assertions kept, and the existing parser pins through the delegator.

Review round 3 (same day): single-backend transport failures (a dropped wire to Anthropic) now answer the same 500
`api_error` envelope the translated non-streaming arm uses, instead of the middleware's bridge-shaped 500 — the arm
sits last so the timeout keeps its 504, and anything that is not a transport blip still propagates as a bug. The
streaming usage pump's line buffer is capped at the translators' `_SSE_MAX_LINE_BYTES` (drop and resync; forwarded
bytes untouched). Subscription setup/profile creation now offers `claude-sonnet-4-6` as the default model instead of
falling through to the OpenRouter-format prompt, pinned in both flows' tests.

Review round 4 (same day): the relay now honours the per-request `TransportGrace` window every
translated route rides out (README Balanced Profiles promises it pool-wide) — same-member retries
while the window lasts, park-and-fail-over once spent, and the same grace in single-backend mode
before the handler's 500 envelope. Tests: two-blips-then-success with no parking and no hop
(grace knobs patched instant), and a single-backend call-count pin (initial attempt + one retry per
grace delay). The SSE line-cap test now asserts the drop's warning via caplog, making the cap
regression-visible (the round-3 test alone passed with the cap deleted).
