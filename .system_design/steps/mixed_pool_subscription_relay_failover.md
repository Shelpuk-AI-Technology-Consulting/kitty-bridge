---
id: mixed_pool_subscription_relay_failover
depends_on: []
---

# Mixed pool: translated failover must never dispatch the subscription member, and crosses into the raw relay as a last resort

Live diagnosis 2026-10-29 (pool `my`, `anthropic` member): the translated Messages
stream-failover loop selected the `anthropic_subscription` member, dispatched it
through the API-key transport with its by-design empty key, received a guaranteed
401, and parked it 900 s. The member has never served a request. Requirements:
`.requirements/20261029T184032Z_subscription_relay_failover/REQUIREMENTS.md`.
Design of record: `SYSTEM_DESIGN.md` §16 (extended by this step).

## Why

§16's spawn-identity fix gave the subscription member a working path only as an
INITIAL pick (the relay engages on `_active_provider`). Failover selections were
left unfiltered, so the rotation still lands on the member mid-walk — on a path
that cannot authenticate it by construction (`key = ""`; Kitty never stores
subscription OAuth material). Two defects follow: a guaranteed-401 dispatch (false
`auth` park, 900 s out of rotation), and a failover target that cannot ever
succeed. The fix keeps §16's one-way crossing doctrine for the relay→API-key
direction and adds the mirror discipline for the translated path: translated
failover selections skip subscription members by default
(`allow_subscription=False`), while the Messages routes gain an explicit,
pre-emission-only LAST-RESORT crossing into the raw relay when the translated walk
is exhausted — which is the overflow role the pool configuration intends the
subscription member to play. The crossing reuses the relay's own loop (its
confinement, health classifier, F4 rule, terminal envelopes) rather than
duplicating it.

## To Be

- `_get_next_backend`/`_select_backend` accept `allow_subscription: bool = False`;
  default selections skip `anthropic_subscription` backends (both the healthy pass
  and the all-unhealthy candidates pass).
- `_select_backend_or_hold` passes `allow_subscription=True` (initial picks
  unchanged on all four routes; non-Messages routes keep their
  `_subscription_route_rejection` guards).
- `_relay_last_resort(request, body)`: eligibility (balancing mode, relay allowed,
  ≥1 healthy subscription member, nothing written to the client) → select the
  subscription member (`require_subscription=True`) → return
  `_relay_subscription_messages(request, body, await request.read())`; logs one
  INFO crossing line.
- Wired at the nine Messages terminals: `_handle_messages`' error arm and both
  empty gates; `_stream_messages`' plain error and catch-all terminals; the
  custom arm's error and empty terminals; and the native and translated empty
  ladders (count guard: `_handle_messages` 3, `_stream_messages` 6). Request-shaped
  400/404/413/422 failures are barred: F4 establishes that the body itself is
  bad, so relaying it would bench a healthy subscription member and turn the
  non-retryable client error into a retryable-looking 503.

## Verification

- L1 selection matrix; L3 aiohttp: crossing serves original bytes with Claude
  Code's headers and un-rewritten model; post-emission never crosses; terminal
  envelope unchanged when the member is parked; pools without a subscription
  member are behaviorally untouched (full sweep).

## Implementation notes (2026-10-29, delivered)

- `allow_subscription` threads through `_get_next_backend`, `_select_backend`, and
  `_any_healthy_backend`; all three filter passes plus the all-unhealthy gamble pool share the
  visibility (the gamble was a third selection expression the plan's wording missed — a
  fully-cooled mixed pool raised into a member dispatch without it; owner-approved R11).
- Guard parity (R10) fell out of the shared default rather than per-site edits: guards and
  selections both exclude by default, so the unfiltered walk sites stay consistent untouched.
- The non-streaming crossing is wired at BOTH error arms (owner-approved R13): a walk whose
  last attempt is a custom-transport member re-raises `ProviderError` through
  `except Exception`, which must cross like `UpstreamError`. The two arms merged into one with
  an `isinstance` branch (consecutive, no other handling between them).
- Owner-approved semantics: relay's own envelope wins when the crossing fails (R12); the
  original-bytes/uncompacted trade-off accepted (R14); the non-Messages terminal shift (earlier
  503, member absent from causes) recorded as intended (R15).
- The stub selection fakes grew the `allow_subscription` kwarg — signature-only: the four
  `_deterministic_draw` stubs in `tests/bridge/test_balancing_server.py`, the eight
  `_fixed_select` stubs in `tests/bridge/test_empty_response_retry.py`, the three in
  `tests/bridge/test_post_emission_no_failover.py`, and `_pinned_backend` in
  `tests/bridge/test_concurrent_backend_selection.py` (the latter three surfaced only in the
  full-suite run — file-isolated runs never invoke the stub with the new kwarg).
- Tests: `tests/bridge/test_mixed_pool_subscription_failover.py` (14: selection matrix 7,
  crossing 7). Deterministic via `random.seed(1)` (plain member drawn first) where order
  matters; ladder sleeps patched per the empty-hold test pattern.

## Implementation notes — review round 2 (2026-10-29, owner direction: address relevant findings)

- Empty-response terminals cross too: all five Messages empty terminals (streaming plain
  exhaustion, custom-transport exhaustion, native streaming ladder exhaustion, both non-streaming
  gates) offer `_relay_last_resort` before the 502 terminal — they are pre-emission walk-exhaustion
  points like the error terminals. A count guard pins one crossing call per handler
  (`_handle_messages` 3, `_stream_messages` 5).
- Crossing failures render the relay's envelope from every arm via `_relay_failure_envelope`
  (arrival path deduped onto it): a stall answers 504 on the non-streaming arm AND the streaming
  plain arm — the latter previously leaked into the dispatch-loop catch-all and surfaced as a
  mangled 502 (found by the new parity test, not by the review, which had marked the streaming
  arms safe). Non-transport failures still propagate.
- Recovery-hold finding rebutted with evidence: the candidates pass computing
  `AllBackendsUnhealthyError.retry_after` admits the member under `allow_subscription=True`
  (server.py:2495's skip is three-condition and does not fire), so the hold DOES wait for a
  member recovering inside the 300 s window — pinned by
  `TestRecoveryHoldVisibility::test_hold_waits_for_the_subscription_member_recovery` (900/120 park
  → one 120 s hold → recovery). No code change; the finding's premise was a misread of the
  triple-negative condition.

## Implementation notes — review of the round-1 docs push (fde4285), applied 2026-10-29

- The reviewer caught that §16/step-doc/README claimed a "streaming custom transport" crossing
  terminal that was never actually wired: both round-1 streaming sites sit in the plain-transport
  branch. Wired now — the custom arm's walk-exhaustion error terminal offers
  `_relay_last_resort` (envelope-wrapped like every site); its empty terminal was already wired in
  round 2. Count guard updated to `_handle_messages` 3 / `_stream_messages` 6; new streaming twin
  test drives `_FailingCustomProvider` exhaustion through the relay.
- Round-1's "streaming custom arm honours the hint" claim was likewise false in code: the mark
  passed no `cooldown` while all three sibling routes did. Fixed to
  `cooldown=self._retry_after_from_exc(exc)` and pinned by
  `TestMessagesCustomArmRetryAfter` (stub raises http_status 429 with `retry_after` 42 → park is
  42 s, kind `rate_limit`). Root cause of both: the round-1 doc claims were written from the plan,
  not verified against the merged hunk — the KBR-247 lesson again.

## Implementation notes — review round 4 (on 1ad24af), applied 2026-10-30

- Request-shaped failures are barred from the crossing (`_crossing_forbidden_for`,
  400/404/413/422): F4 already establishes the body is bad, and relaying it parked a healthy
  subscription member on Anthropic's identical 400 while turning the non-retryable client error
  into a 503. Wired at all four error-terminal crossings; auth/rate-limit/upstream statuses stay
  eligible (member-shaped). Pinned by the F4 cell (two 400ing members → 400 envelope, member
  healthy, no relay call), the streaming single-400 cell, and a 500-still-crosses control.
- The five empty-terminal crossings now wrap the relay in `_relay_failure_envelope` like the
  error terminals — a stall there 504s instead of reaching the 500 handler or (on the streaming
  arm) re-dispatching a second full relay POST (README promises one attempt). §16's "every
  crossing site" sentence is now true by construction; the empty-gate stall test pins it.
- Doc residues fixed: the test module docstring's "currently parks", the Bedrock map_error
  Google docstring (changed symbol, was undocumented), and this file's To Be bullet (three
  terminals → nine, with the bar).
