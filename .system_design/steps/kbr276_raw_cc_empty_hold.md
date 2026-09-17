---
id: kbr276_raw_cc_empty_hold
depends_on: []
---

# KBR-276 — Widen the CC pre-emission hold to raw Chat Completions-wire upstreams

## What

Drop the converter gate (`release = stream_converter is not None`) from
`_stream_chat_completions`'s per-attempt prologue, so the KBR-248 pre-emission
hold engages on every Chat Completions-wire upstream on `/v1/chat/completions`
— raw-CC providers (OpenAI, OpenRouter, DeepSeek, any plain-POST backend) as
well as the converted Messages-wire route KBR-248 covered. The `release`
variable is removed; `_hold_or_write`'s first branch simplifies from
`if has_content or not release:` to `if has_content:`. Update the closure
docstring, the hold-prologue comment, the forward-chunk comment, the
`_cc_chunk_carries_content` docstring ("converted line" → "line"), and the
three KBR-254 re-dispatch buffer-list comments. Add
`tests/bridge/test_raw_cc_empty_hold.py` (the raw-CC mirror of the KBR-248
suite) and replace `SYSTEM_DESIGN.md` §5.4's KBR-248 converter-gate paragraph
with the KBR-276 decision record; add the amendment footer + "Completed by
KBR-276" note to `TEST_SUITE.md`.

## Why

A content-less completion from a raw-CC upstream reached the client as a
well-formed skeleton — role chunk, finish chunk, `[DONE]` — because the first
chunk set `has_content` and permanently disarmed the empty-response ladder.
KBR-248 fixed the identical defect on the converted route but deliberately
left the raw-CC widening to a product decision (imposing ~80 s of retry
latency on every content-less completion for plain-POST providers was a trade
that ticket did not authorise). KBR-276 is the ticket where that decision was
made; the owner's direction to implement settles it as "widen". Everything
downstream — the ladder, the D5 10 MiB cap with its fail-open, the non-JSON
fail-open, and the route-wide D4 exhaustion terminal
(`type: "empty_response"` + `[DONE]`, KBR-248) — already existed route-wide;
only the gate was in the way. Dropping the gate reuses all of it unchanged,
and the classifier `_cc_chunk_carries_content` already handles raw CC chunks
(it is documented robust to arbitrary parsed upstream JSON).

**Why not the alternative ("keep the skeleton"):** a well-formed skeleton is
the worst kind of failure — the agent sees a turn that succeeded with nothing
in it, the backend is marked healthy, and a balancing pool can keep routing to
the broken upstream. The converted route (KBR-248) and the sibling routes
(KBR-155/227, KBR-235, KBR-250) already retry empty completions; keeping the
raw-CC route as the sole skeleton route preserved a defect class the product
had otherwise closed, for a latency cost that only bites when the reply was
empty anyway.

**Scope-out, deliberate (recorded in §5.4):** the custom-transport branch
this route re-dispatches into on cross-class failover (KBR-254) synthesises
its own CC stream and has no hold; an empty completion from such a failover
target still delivers a skeleton within the crossing bound. Same defect one
branch over; its own ticket.

## Implementation notes (2026-09-17, PR #221)

- The gate's removal is the entire code change: `release` deleted from the
  prologue, the closure, and the three buffer-list comments; no ladder, cap,
  or terminal change. Diff in `server.py` is comments + one branch condition.
- Tests in `tests/bridge/test_raw_cc_empty_hold.py` parametrise over
  `OpenAIAdapter` / `CustomOpenAIAdapter` (both resolve
  `stream_converter = None`; both default to
  `https://api.openai.com/v1/chat/completions`; `requires_custom_url` is only
  read in a 404 error-rebuild path, not at startup). AC-1/AC-4 watched
  failing pre-fix (`calls == 1`); AC-2 (`client_body == upstream_body` —
  byte equality holds because the raw path's translator is identity and the
  handler re-emits `f"{line}\n\n"` verbatim) and AC-3 pin the no-regression
  surface.
- The D4-exhaustion test pins `_MAX_RETRIES = 0` +
  `_EMPTY_FINAL_DELAYS = [0.01]` → `max_attempts = 2` against the harness's
  `len(bodies) + 2 = 3` registered aioresponses callbacks.
- Reviewers (system-design-reviewer, then code-reviewer APPROVE) surfaced:
  the custom-transport scope-out (now recorded in §5.4), the stale
  `_cc_chunk_carries_content` docstring (fixed), the usage-note clause
  (exhausted empty ladders attribute no usage — pre-existing, shared), the
  C3(i)/T-I8 widening obligation (recorded in the TEST_SUITE completion
  note), the terminology convergence on "raw-CC", and the amendment footer
  on KBR-248's paragraph. Forward-only, not in this PR: T-I8 widening, and
  the harness injector doc-tick (`tests/harness/failures.py:~583`,
  `tests/harness/test_failures.py:~428` still describe the pre-KBR-276
  skeleton behaviour).
- Full gate green: `ruff check .`, `mypy src`, `lint-imports` 5/5,
  `pytest -q` 8122 passed / 14 skipped / 37 deselected, `tests/harness/`
  2397 passed, mutmut scope + corpus roundtrip + CRLF lint guards green.
- `scripts/regenerate_step_index.py` does not exist in the repo (known gap;
  see the auto-memory note), so the step index was not regenerated.
