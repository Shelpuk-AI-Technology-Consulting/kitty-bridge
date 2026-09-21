---
id: kbr300_non_streaming_silent_skeleton_sweep
depends_on: [kbr298_messages_custom_transport_non_streaming_empty_hold]
---

# KBR-300 — Non-streaming silent-skeleton sweep: the raw-CC `/v1/messages` cell + the three sibling routes

Ticket: [KBR-300](https://shelpuk.atlassian.net/browse/KBR-300) (KBR-136 epic, High).
The residual KBR-298 recorded in `SYSTEM_DESIGN.md` §5.4 ("Custom-transport gate only"
decision bullet: "the raw-CC × non-streaming cell … stays open here and is filed as
KBR-300") and the three sibling handlers the same sweep widened to.
Requirements: `.requirements/20260921T224600Z_kbr300_non_streaming_silent_skeleton_sweep/REQUIREMENTS.md`.

## What

The same post-ladder exhaustion defect KBR-298 closed on one cell exists on four
non-streaming cells: the `/v1/messages` raw-CC cell (the KBR-298 gate is transport-gated,
so `use_custom_transport = False` falls through to `translate_response`'s fabricated
fallback), and the non-streaming arms of `_handle_responses`, `_handle_gemini`, and
`_handle_chat_completions` (no gate at all — Responses/Gemini fabricate fallback text via
their translators; Chat Completions ships the empty CC body verbatim as a billed,
healthy-marked `200` skeleton). `_request_with_retry`'s built-in empty ladder already
walks every cell; the defect is the exhaustion arm's terminal.

The fix is one gate per handler, judged after the ladder, in the translated/verbatim arm,
before `translate_response` (or, on Chat Completions, before `_log_usage` and the verbatim
return):

- `_handle_messages`: the KBR-298 elif drops its `use_custom_transport` conjunct and takes
  the ticket's single predicate — `not self._active_provider.use_native_messages and
  self._is_empty_cc_response(cc_response)` — so one gate covers both the KBR-298 cell
  (still closed) and the raw-CC cell. The KBR-298 comment above the elif is updated to
  record the widening.
- `_handle_responses` / `_handle_gemini`: the same predicate gates a `web.json_response`
  D4 terminal immediately before `translator.translate_response(...)`.
- `_handle_chat_completions`: the same predicate gates immediately before `_log_usage` /
  the verbatim return.

Per-route D4 body (all HTTP `502`, all before `_log_usage` and `_mark_backend_healthy`):

- Messages — `{"type": "error", "error": {"type": "api_error", "message":
  _NATIVE_EMPTY_REPLY_MESSAGE, "reason": "empty_response"}}` (KBR-298's body, unchanged).
- Responses — `{"type": "error", "error": {"code": "empty_response", "message":
  _NATIVE_EMPTY_REPLY_MESSAGE, "reason": "empty_response"}}` (KBR-293's streaming
  `code: "empty_response"` discriminator, carried to the non-streaming JSON shape with the
  `reason` marker).
- Gemini — `{"error": {"code": 502, "message": _NATIVE_EMPTY_REPLY_MESSAGE, "reason":
  "empty_response"}}` (byte-mirror of KBR-293's Gemini streaming SSE error payload).
- Chat Completions — `{"error": {"message": _NATIVE_EMPTY_REPLY_MESSAGE, "type":
  "empty_response"}}` (byte-mirror of KBR-287's Chat Completions streaming D4).

## Why

- *One gate per handler, not parallel gates.* The ticket's scope item 3: the three
  sibling handlers need the gate regardless of transport class, and
  `use_custom_transport = False` is the default for raw-CC — a second, transport-keyed
  gate on the messages handler would be a parallel structure to maintain. Widening the
  existing elif is a one-token edit with the KBR-298 tests as the regression net.
- *The `use_native_messages is false` conjunct is the ticket's literal predicate.* Native
  providers stay on today's behaviour on every route (the KBR-298 native-arm carve-out,
  generalised); `_make_upstream_request` already returns CC for them on the sibling
  routes, so the conjunct is a pure behavioural carve-out, not a shape guard. Concretely
  on `/v1/messages`: a native Anthropic provider whose `_native_messages_request` was
  cleared by the KBR-237 tool-use-format fallback answers in CC form; an empty reply
  there falls through the widened elif into `translate_response`'s fabricated fallback
  (today's behaviour, deliberately preserved). Widening to cover native providers is
  one conjunct drop per handler; **proposed follow-up, not yet filed — PO to decide**.
- *Per-route D4 body mirrors each route's streaming D4.* One client branch per route
  across both stream modes — the same route-internal-consistency argument KBR-298
  recorded for the messages 502.
- *Judge after the ladder.* The non-streaming walk is atomic per attempt; the judge can
  only see a content-bearing reply or the exhausted empty one. Carried over from KBR-298's
  decision record.

## Reasoning-only trade-off

`BridgeServer._is_empty_cc_response`'s CC arm counts a non-empty string
`reasoning_content` (KBR-277). On raw-CC cells a reasoning-only reply is therefore judged
non-empty and released (the route's translator maps it); on custom-transport cells no
non-streaming parser surfaces reasoning (`BedrockAdapter.translate_from_upstream` reads
only `text`/`toolUse`; `OllamaCloudAdapter` reads only `message.content`/`message.tool_calls`;
`OpenAISubscriptionAdapter._parse_sse_to_response` reads only text deltas and function-call
items), so a reasoning-only reply parses to the empty shape and the ladder runs. The
accepted KBR-287/293/297/298 trade-off carries over unchanged.

## Tests

Four new files, the KBR-287/293/297/298 harness shape (real `BridgeServer`, scripted raw
upstream shapes through the adapters' real parsers, content oracles parsed from the JSON
body, recording seam for `_log_usage`/`_mark_backend_healthy`):

- `tests/bridge/test_messages_raw_cc_non_streaming_empty_hold.py` — the raw-CC cell
  (plain `OpenAIAdapter`, `aioresponses` upstream), five tests.
- `tests/bridge/test_responses_non_streaming_empty_hold.py` — Responses × both
  transports (custom × 3 parametrised + plain), per-cell content/empty/tool/reasoning
  plus the mixed-pool crossing.
- `tests/bridge/test_gemini_non_streaming_empty_hold.py` — Gemini × both transports,
  same shape.
- `tests/bridge/test_chat_completions_non_streaming_empty_hold.py` — Chat Completions ×
  both transports, same shape (verbatim-body oracles).

The merged KBR-298 file (`tests/bridge/test_messages_custom_transport_non_streaming_empty_hold.py`)
is unchanged and re-pins the custom-transport cell against the widened elif.

## Implementation notes

(appended after implementation)
