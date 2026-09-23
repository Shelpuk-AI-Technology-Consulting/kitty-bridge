---
id: kbr306_native_messages_arm_empty
depends_on: [kbr304_native_provider_empty_hold]
---

# KBR-306 — Non-streaming silent-skeleton residual: the native-Messages arm of `/v1/messages`

Ticket: [KBR-306](https://shelpuk.atlassian.net/browse/KBR-306) (KBR-136 epic, High,
labels `silent-skeleton` + `found-by-kbr-304`). The last silent-skeleton-shaped cell
on the non-streaming path after KBR-300 (PR #267) and KBR-304 (PR #273, merged
`146eece` 2026-09-23) closed the four CC-shaped cells. Requirements:
`.requirements/20260923T160753Z_kbr306_native_messages_arm_empty/REQUIREMENTS.md`.

## What

The `if cc_response.get("type") == "message"` branch of
`BridgeServer._handle_messages` non-streaming (`src/kitty/bridge/server.py:5426`)
currently ships the ladder-exhausted empty native Messages reply as a billed,
healthy-marked `200`: after `_request_with_retry` walks the empty ladder via
`_is_non_retryable_reply` → the Messages-shaped arm of `_is_empty_cc_response`,
the handler's `if` branch checks only `_messages_truncation_before_content` (D3)
— which catches only `max_tokens` / `model_context_window_exceeded` stop reasons
on a content-less reply — and falls through to `result = cc_response`, billing
the empty completion, marking the backend healthy (in balancing mode), and
returning the empty `content: []` as HTTP `200`. A native Anthropic provider
answering `/v1/messages` non-streaming in its own Messages shape with
`content: []` or a thinking-only block is the defect.

The fix mirrors KBR-298/300/304 exactly: inside the `if` branch, after the D3
truncation check and before `result = cc_response`, the parsed native reply is
judged through `self._is_empty_cc_response` — whose Messages-shaped arm
(already present and pinned by the streaming twin `PreambleHold._block_start_releases`,
Q14 D1) decides — and a judged-empty completion ends in the route's D4 terminal
(`server.py:5492-5502`'s byte-image: bare-JSON `502` +
`_NATIVE_EMPTY_REPLY_MESSAGE` + `reason: "empty_response"`), before
`_audit_response_tool_use` / `_log_usage` / `_mark_backend_healthy`. One client
branch, `(502, reason=empty_response)`, covers `/v1/messages` in both stream
modes.

## Why

- *One gate after the D3 check, byte-mirror of the elif D4.* Same surgical
  shape as KBR-298/300/304. The ladder already detects Messages-shaped empties
  (`_is_non_retryable_reply` → `_is_empty_cc_response` Messages arm); the new
  gate sits where it can only see a content-bearing reply or the
  ladder-exhausted empty one. No new ladder logic, no parallel gate, no
  structural change.

- *One D4 discriminator across stream modes.* The D4 body literal — bare-JSON
  `502`, `_NATIVE_EMPTY_REPLY_MESSAGE`, `reason: "empty_response"` — is
  byte-identical to the elif's at `server.py:5492-5502` and the streaming S11
  terminal at `server.py:7077-7087`. One client branch covers the route in
  both stream modes; the ticket's design-review question (whether the native
  arm needs a distinct discriminator) is answered: it does not.

- *D3 ordering preserved.* `_messages_truncation_before_content` catches a
  `max_tokens` / `model_context_window_exceeded` stop reason on a content-less
  reply and renders the request-shaped `400` (KBR-235) before the new empty
  gate. The truncated reply is non-retryable in the ladder
  (`_is_non_retryable_reply`'s second clause), so the handler sees it on
  attempt 1 and D3 fires at once — entry count 1, no usage billed, no
  healthy-mark. The new gate does not run on this branch.

- *The `if` branch's structural comment stays as-is.* The elif's comment —
  *"The gate sits in the translated arm, so a native Messages reply (the `if`
  above) is structurally unreachable here"* — is factually true after this fix
  (the elif gate IS in the translated arm; a Messages-shaped reply takes the
  `if` path). What changes is the §5.4 recorded-residual sentence; the
  structural fact in the elif's comment does not.

- *Two carry-overs, recorded so they are not read as accidents:*
  - *Whitespace-only text is not judged empty* (PO confirmation 2026-09-23):
    the Messages arm's `text != ""` (no `.strip()`) mirrors the streaming
    hold's release semantics. A native reply whose only content is a
    whitespace-only text block reaches the client as `200` in both stream
    modes. Widening it would require changing BOTH judges together — a
    behaviour change on a streaming path this ticket explicitly records as
    fine, so it stays open.
  - *A thinking-only native reply takes the ladder* (the KBR-287/293/297/298/300
    reasoning-only trade-off, carried through the KBR-277/285 judge family):
    the arm counts a thinking block as empty, consistent with the streaming
    hold. A tool_use block counts as content (D1: any block whose type is
    not text/thinking/redacted_thinking is content).

## Tests

`tests/bridge/test_messages_native_non_streaming_empty_hold.py` — the KBR-300/304
harness shape in a sibling file (the physical-mirror-as-divergence-guard
convention, same as each KBR-300 route file):

- `_MessagesLauncher`, `_NativeOpenAIAdapter`, `_post()` — physically mirrored
  from `tests/bridge/test_messages_raw_cc_non_streaming_empty_hold.py`.
  The stub deliberately leaves `upstream_wire_shape` at the inherited
  `WireShape.CHAT_COMPLETIONS` (the base-class invariant is a
  production-adapter rule; the test exercises the gate's dispatch dimension
  only).
- Messages-shaped canned bodies flow through the
  `_native_messages_request + type == "message"` guard in
  `_make_upstream_request` as-is — no `translate_from_upstream` — so the
  handler sees the exact JSON the route receives and dispatches into the
  `if` branch.
- The recording seam captures `_log_usage` / `_mark_backend_healthy` so
  the empty-arm guarantees are asserted, not assumed.

Seven tests, three RED on the unfixed code (defect: `assert 200 == 502`)
plus four pass-through regression pins:

- `test_an_empty_native_messages_non_streaming_completion_ends_in_the_d4_terminal`
  — FR-1 / AC-FR-1.1, AC-FR-1.2, AC-FR-1.3, AC-FR-7.1. RED pre-fix.
- `test_a_thinking_only_native_messages_non_streaming_completion_ends_in_the_d4_terminal`
  — FR-2 / AC-FR-2.1. RED pre-fix.
- `test_a_content_bearing_native_messages_non_streaming_completion_reaches_the_client`
  — FR-3 / AC-FR-3.1. Pass-through regression pin.
- `test_a_tool_use_only_native_messages_non_streaming_completion_releases_the_verdict`
  — FR-4 / AC-FR-4.1. Pass-through regression pin.
- `test_a_truncated_native_messages_non_streaming_completion_ends_in_the_d3_terminal`
  — FR-5 / AC-FR-5.1. Pass-through regression pin.
- `test_an_empty_native_messages_non_streaming_attempt_crosses_to_a_healthy_plain_peer`
  — FR-6 / AC-FR-6.1. Pass-through regression pin.
- `test_an_empty_native_pool_exhausts_into_the_d4_terminal_without_a_healthy_mark`
  — FR-1 in balancing mode / AC-FR-1.4 — the meaningful balancing-mode
  "no healthy-mark" pin. RED pre-fix.

The balancing-mode test 7 is the meaningful pin for
`healthy_log == []` on the empty arm: the single-backend test 1's
`healthy_log == []` assertion is vacuous (the route's `_mark_backend_healthy`
site guards on `if self._backends and self._current_backend_idx >= 0`,
which is always False in single mode — the recorded KBR-300 asymmetry).

## Implementation notes

- **One gate, one branch, +31 lines.** The server-side diff is the new
  empty gate inside the `if cc_response.get("type") == "message":` branch
  of `_handle_messages` (`src/kitty/bridge/server.py:5447`): judge
  `self._is_empty_cc_response(cc_response)` after the D3 truncation check
  and before `result = cc_response`; on a judged-empty reply return the
  route's D4 terminal — `web.json_response` with `_NATIVE_EMPTY_REPLY_MESSAGE`
  + `reason: "empty_response"`, HTTP `502` — byte-identical to the elif's
  D4 at `server.py:5492-5502` and the streaming S11 terminal at
  `server.py:7077-7087`. The gate comment records the mirror rule, the
  ladder-walk precedence, the byte-image citations, and both carry-overs.
  The elif's structural comment ("structurally unreachable here") is
  untouched — it remains factually true (the elif is in the translated
  arm; a Messages-shaped reply takes the `if` path). The §5.4
  recorded-residual sentence is the doc that changed, not the elif's
  inline comment.

- **Pre-fix RED evidence.** The three empty-arm tests
  (`test_an_empty_native_messages_non_streaming_completion_ends_in_the_d4_terminal`,
  `test_a_thinking_only_native_messages_non_streaming_completion_ends_in_the_d4_terminal`,
  `test_an_empty_native_pool_exhausts_into_the_d4_terminal_without_a_healthy_mark`)
  failed on the unfixed code with `assert 200 == 502` — the ladder's log
  read `"Empty upstream response after 5 attempts, returning fallback"`
  (single-backend; the log line's wording is historical, KBR-300's
  recorded note) and the handler's un-gated `result = cc_response`
  shipped the empty `content: []` body as a billed `200`. Pre-fix
  `usage_log` length was 1 (billed once); test 7's pre-fix
  `healthy_log` was `[backend_idx]` (the balancing-mode mark fired). The
  four pass-through pins (content-bearing, tool-use-only, D3 truncation,
  `[native, plain]` crossing) passed under both predicates — they are
  regression pins, not RED cells (KBR-304's per-test-mechanism lesson:
  name which tests went RED and why the others do not).

- **Balancing-mode exhaustion accounting.** Test 7's draw pin is
  `[[0], [1], [0], [0]]` — initial selection, failover after the first
  empty, and one selection per final retry (`len(_EMPTY_FINAL_DELAYS)`)
  — and its upstream entry count is
  `len(backends) + len(_EMPTY_FINAL_DELAYS)` (`2 + 2 = 4`), unlike the
  single-backend bound `len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`
  (balancing walks one attempt per backend with no intra-backend empty
  retries, then takes only the final-retry loop).

- **No new imports.** The fix uses `_NATIVE_EMPTY_REPLY_MESSAGE` and
  `logger`, both already module-level in `server.py`; the test file's
  imports are the raw-CC sibling's set (no third-party additions).

- **Verification.** Seven tests green post-fix
  (`tests/bridge/test_messages_native_non_streaming_empty_hold.py`, ~1 s);
  312 passed / 0 failed across the fourteen affected sibling files (all
  KBR-298/300/304 empty-hold files, `test_native_provider_reply_shape.py`,
  `test_native_empty_stream.py`, `test_pairing_truncation_properties.py`,
  `test_empty_response_retry.py`, `test_empty_response_reasoning_properties.py`,
  `test_raw_cc_empty_hold.py`) in ~4.5 min wall-clock;
  `ruff check` clean; `mypy src/kitty` clean (94 files);
  `python3 scripts/regenerate_step_index.py` exits 0. No CI-relevant
  environment staleness (fresh venv on the fresh branch).

- **No system-design follow-up filed.** With this gate every
  non-streaming silent-skeleton cell on every route is judged: CC-shaped
  replies by the elif (KBR-298/300/304), native-Messages-shaped replies
  by the `if` branch (KBR-306). The whitespace-only-text carry-over is
  the only recorded open shape on the non-streaming path — and it is
  consistent across stream modes (both judges use `!= ""`), documented
  in §5.4 and REQUIREMENTS.md as deliberate.
