---
id: kbr298_messages_custom_transport_non_streaming_empty_hold
depends_on: [kbr297_messages_custom_transport_empty_hold]
---

# KBR-298 — Empty-completion hold on the non-streaming `/v1/messages` custom-transport path

Ticket: [KBR-298](https://shelpuk.atlassian.net/browse/KBR-298) (KBR-136 epic, High).
The non-streaming residual KBR-297 recorded in `SYSTEM_DESIGN.md` §5.4 ("Non-streaming
residual (out of scope)") and in this directory's `kbr297_messages_custom_transport_empty_hold.md`.
Requirements: `.requirements/20260921T153557Z_kbr298_messages_custom_transport_non_streaming_empty_hold/REQUIREMENTS.md`.

## What

Judge the non-streaming translated arm. `BridgeServer._handle_messages`' `stream: false`
branch takes `_request_with_retry(cc_request)` — for a `use_custom_transport` provider the
call resolves through `_make_upstream_request` → `provider.make_request`, and the ladder
inside `_request_with_retry` already retries a judged-empty completion (single:
`len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1` attempts; balancing: `n_backends`,
empties never marking a backend unhealthy). Its exhaustion arm returns the empty response
(grep anchor: "translator will add fallback text"), and the handler — with no emptiness
gate — hands it to `MessagesTranslator.translate_response`, whose defensive fallback
fabricates `_EMPTY_ASSISTANT_FALLBACK_TEXT` as the model's reply, bills it through
`_log_usage`, and marks the broken backend healthy.

The fix sits in the handler's translated arm (the `else` of
`cc_response.get("type") == "message"`), **gated on `self._active_provider.use_custom_transport`**:
a judged-empty `cc_response` returns the route's D4 terminal —
`_make_error_response({"type": "error", "error": {"type": "api_error", "message":
_NATIVE_EMPTY_REPLY_MESSAGE, "reason": "empty_response"}}, status=502)` — before
`translate_response` can run; no `_log_usage`, no `_mark_backend_healthy`. The native
Messages arm is untouched (its D3 truncation check already governs it).

## Why (decisions)

- **Judge after the ladder, not per attempt.** The non-streaming route's retry structure
  already walks the empty ladder inside `_request_with_retry`; the ticket says to mirror
  that walk and invent no second ladder. The judge therefore sits after the walk returns —
  it can only see a content-bearing reply or the exhausted empty one. This differs in
  position from the streaming twin (per-attempt pre-emit judge) because the non-streaming
  branch is atomic per attempt by construction.
- **Status 502, decided against the route's non-streaming error contract.** The route's
  non-streaming statuses are 400 (malformed input, D3 truncation), the preserved upstream
  status (`UpstreamError`), 500 (unknown exceptions). None covers "the upstream answered
  with nothing usable on every attempt" — the canonical 502 case. Route-internal
  consistency agrees: this route's streaming D4 (plain `empty_no_finish` arm and the
  KBR-297 custom segment alike) is bare-JSON 502 with the same `reason: "empty_response"`
  discriminator, so one client branch, `(502, reason=empty_response)`, covers the route in
  both stream modes.
- **Custom-transport gate only.** The ticket scopes the cell. The plain-POST (raw-CC)
  non-streaming cell has the same post-ladder shape — the ticket attributes it to KBR-277,
  but KBR-277 widened only the `reasoning_content` predicate and left the exhaustion
  terminal untouched — so it is filed as a follow-up ticket rather than widened here.
- **Reasoning-only stays ladder-taking.** Neither custom adapter's non-streaming parser
  surfaces reasoning (Bedrock `translate_from_upstream` reads only `text`/`toolUse` blocks;
  Ollama Cloud reads only `message.content`/`message.tool_calls`; the subscription's
  `_parse_sse_to_response` reads only text deltas and function-call items) — the same
  accepted trade-off KBR-287/293/297 pin.

## Verification

Bridge-level tests in the KBR-287/293/297 harness shape
(`tests/bridge/test_messages_custom_transport_non_streaming_empty_hold.py`): real
`BridgeServer`, `stream: false`, parametrised over the three `use_custom_transport`
adapters, scripted responses fed through the adapters' **real** parse step
(`translate_from_upstream` / `_parse_sse_to_response`), content oracle parsed from the JSON
response body (never raw substrings), watched red on the unfixed code — the red observation
on the empty/exhaustion/reasoning cells is the fabricated-fallback 200, the corrected As-Is.

## Implementation notes

- Landed on `fix/kbr-298-messages-custom-transport-nonstreaming-empty-hold`.
  Falsification baseline: **3 red / 10 green** on the unfixed code — the red
  cells were the AC-FR-2 empty-exhaustion cells for all three adapters, each
  showing the corrected As-Is (the 5-attempt walk logging "returning
  fallback", then the fabricated-fallback **200**); AC-FR-1/3/4 (content,
  tool-call, reasoning-recovery) were green sanity cells because the ladder
  already recovers on a non-empty second attempt — the defect fires only at
  exhaustion. AC-FR-6 (crossing) was green pre-fix too (the crossing is
  pre-existing walk behaviour; the test pins the fix must not break it).
  Post-fix: 13/13 new, 239/239 adjacent focused
  (`test_messages_custom_transport_empty_hold` streaming twin,
  `test_raw_cc_empty_hold`, `test_empty_response_retry`,
  `test_balancing_server`, `test_pairing_truncation_properties`,
  `test_native_empty_stream`), ruff + mypy clean on `server.py`.
- One mid-implementation correction, same class as KBR-297's
  `_is_empty_cc_response` method note: `_make_error_response` is a **local
  function inside `_stream_messages`** (grep anchor: "Build a Messages API
  error response for pre-stream failures"), not a module-level helper — the
  first cut raised `NameError` in `_handle_messages`, caught immediately by
  the test run. The non-streaming D4 uses `web.json_response` directly, the
  handler's own error idiom.
- Design review (system-design-reviewer, 2026-09-21): design sound, no
  blockers; five worth-applying items folded in — exact gate placement named
  (the `else:` of the native `if`, after the D3 check); AC-FR-4 widened to
  all three adapters (Bedrock's Converse `reasoningContent` blocks parse to
  empty the same way); the recording seam (`on_build` patching `_log_usage` /
  `_mark_backend_healthy`) and the balancing-pool construction spelled out in
  the Testing Plan; FR-8 tightened so the §5.4 closure paragraph must name
  the raw-CC sibling as the remaining open cell with its follow-up ticket.
  The reviewer also confirmed the 502 choice changes no Claude Code retry
  behaviour (the Messages client retries any 5xx).
- Harness divergence from the streaming twin, deliberate: `_EMPTY_RETRY_DELAYS`
  is collapsed here (`[0.01, 0.01]`, lengths untouched) because the
  non-streaming walk reads it for mid-ladder sleeps — the streaming custom
  segment never reads it, which is why the KBR-297 test left it alone. The
  AC-FR-2 bound assertion reads `len(_EMPTY_RETRY_DELAYS) +
  len(_EMPTY_FINAL_DELAYS) + 1`, never a literal (F30 coupling).
- `_request_with_retry_single`'s "returning fallback" log line is left
  unchanged on purpose: its wording remains accurate for the raw-CC sibling
  routes this ticket leaves open; the route-scoped empty-arm warning added in
  the handler supersedes it for the custom-transport cell.
- Sibling follow-up filed: **KBR-300** (High) — the raw-CC × non-streaming
  exhaustion cell, same defect class, with the mixed-pool both-empty hole
  recorded in §5.4. The KBR-298 ticket's "KBR-277 closed the non-streaming
  half on the raw-CC route" attribution was corrected: KBR-277 widened only
  the predicate.
