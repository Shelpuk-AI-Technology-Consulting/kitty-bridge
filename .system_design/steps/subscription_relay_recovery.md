---
id: subscription_relay_recovery
depends_on: [brotli_subscription_relay, mixed_pool_subscription_relay_failover, kbr319_subscription_health]
---

# A rejected subscription relay attempt no longer stalls a mixed pool

## Planned delivery

Three rules, confirmed by the product owner on 2026-11-04 and recorded in SYSTEM_DESIGN §16.1:
recover a thinking-signature rejection on the same member, fall over to a translated member when no
subscription sibling is left, and park nobody for a rejection of the request itself.

## Decisions and why

SYSTEM_DESIGN §16.2 holds the reasoning: why the earlier "raw requests stay in the subscription
subset" rule was reversed, and why recovery reacts to Anthropic's refusal instead of filtering by
signature shape.

## Implementation notes

Landed in `src/kitty/bridge/server.py`: `_relay_attempt_recovering_thinking`,
`_strip_thinking_not_issued`, `_anthropic_sse_signature`, `_fall_over_to_translated`, the
`may_fall_over` parameter and `None` return of `_relay_subscription_messages`, and the
request-scoped refusal mark. Signatures are recorded from both relay reply shapes. F4's "two 400s"
counter is gone from the relay: nothing is parked for a request-shaped rejection, so a retry there
could not end, and it was reproduced posting indefinitely on 413/422 in review before it shipped.

Tests: `tests/bridge/test_subscription_relay_recovery.py`, written red first (12 of the first 15
failed with the live 503). Coverage is loopback against mocked upstreams.

Verified live on 2026-11-04 with an interactive Claude Code 2.1.318 in tmux, on a scratch-config
pool of `minimax-mid` + `anthropic` (OAuth relay, real upstreams). First run, on the code before review
round 2, which later changed only the strip counter, where the recorded set is cleared, and a type check
in that path:

- Before the fix the same session answered `503 All 1 backends` on the first turn that reached
  Anthropic with MiniMax reasoning in history.
- After it: 30 requests, 19 taking the arrival relay. The first with foreign reasoning was refused
  once at 17:16:26 (``messages.5.content.0: Invalid `signature` ``), recovered on the same member
  with no park, and all 18 later ones were accepted, Bash tool loops included, with no `API Error` on
  screen. Anthropic-signed reasoning stayed in those requests and was accepted (`sig_other` grew from
  1 to 5). `anthropic` completed 20 of 21 attempts.
- Assistant turns left with `content: []` were accepted, since Claude Code stores thinking as its own
  assistant message.

Second run, on the final code: MiniMax was rate-limited, so no foreign reasoning entered the history and
the signature path did not run again. 31 requests, no `API Error` on screen, and `anthropic` completed 30
of 31 attempts. The relay-to-translated fall-over engaged live at 17:34:18 after the startup 429, but
MiniMax was rate-limited too, so that request ended in the recovery hold rather than a translated reply.
A translated member actually serving a fall-over is covered by the loopback tests only.

Open, deliberately not fixed here:

- At session start Claude Code sends a `max_tokens: 1` "quota" request. Anthropic answered it 429 on
  the relay, parking the member 300 s; the cause of that 429 is not established. MiniMax's reply to the
  same request raised `'NoneType' object is not iterable` in the translated path, parking that member
  too (also seen in unmodified 1.21.0). A small pool can be briefly fully parked by this.
- When the crossing itself fails, the 503 speaks for the subscription subset ("All 1 backends").
