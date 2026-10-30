---
id: translated_rate_limit_retry_after
depends_on: []
---

# Translated 429 arms honor the provider Retry-After hint

## Planned delivery

`SYSTEM_DESIGN.md` §6.1 promises "`rate_limit` from the provider's own retry hint", but only
part of the surface kept it. This step closes the gap on the Messages route: the plain
streaming arm parses `Retry-After` (`parse_retry_after_seconds`) and parks for the hint;
the shared non-streaming classification attaches the hint to the balancing 429
(`UpstreamError.retry_after`) and both of its mark sites (failover loop, final ladder) pass it
as the cooldown; `_stream_messages`' custom arm passes `cooldown=self._retry_after_from_exc(exc)`
like its three sibling routes. Absent, unparseable, or HTTP-date hints keep the fixed default;
each applied hint logs one DEBUG line. The sibling routes' plain arms keep the fixed default —
out of scope (the diagnosed pool's traffic is Messages-only), and §6.1's wording is qualified
accordingly.

Requirements and the test plan live in
`.requirements/20261029T184032Z_translated_rate_limit_retry_after/REQUIREMENTS.md`.

## Decisions and why

- Scope: the live observation (2026-10-29, pool `my`) was MiniMax members on the Messages route
  flapping at 429 with hints ignored; widening to all four routes' plain arms is left as a
  possible follow-up rather than folded in unasked.
- The single-member non-stream tests double as cap probes: the hint bypasses the 30 s
  single-backend cooldown cap, the hintless path does not — pinning both sides of the branch.

## Implementation notes (2026-10-29)

- Implemented in `server.py`: the streaming status-classification block, the balancing 429 raise
  site, both non-stream mark sites, and the Messages custom arm's mark.
- Tests: `tests/bridge/test_translated_rate_limit_retry_after.py` (4). The non-stream tests
  collapse the post-exhaustion empty ladder (`_EMPTY_RETRY_DELAYS`/`_EMPTY_FINAL_DELAYS`)
  following `test_chat_completions_non_streaming_empty_hold.py`'s pattern; every scenario parks
  both members (or uses one), so no assertion depends on weighted-selection order.

## Implementation notes — review round 2 (2026-10-29)

- The shared parser now caps the hint at one hour (`parse_retry_after_seconds`,
  `_RETRY_AFTER_HINT_CAP = 3600` in `providers/base.py`): a broken `Retry-After: 86400` header
  parks a member 3600 s, not a day; a genuinely long limit re-parks on its next 429. Pinned at
  both layers (parser unit test + hinted-429 integration test).
- Sibling plain arms on the other three routes remain out of scope as recorded above (owner
  sign-off at development time); the cap, being in the shared parser, covers them the day their
  hints are wired.
