---
id: kbr297_messages_custom_transport_empty_hold
depends_on: [kbr293_sibling_custom_transport_holds]
---

# KBR-297 — Empty-completion hold on the `_stream_messages` `use_custom_transport` segment

Ticket: [KBR-297](https://shelpuk.atlassian.net/browse/KBR-297) (KBR-136 epic,
label `found-by-kbr-293`). The follow-up gap KBR-293 recorded in
`SYSTEM_DESIGN.md` §5.4 — `/v1/messages` is the last streaming leg of the
silent-skeleton class. Reviewed by `system-design-reviewer` 2026-09-21 (one
blocker corrected: the As-Is wire shape is fabricated fallback text, not a
silent empty turn — the ticket's "zero content blocks" wording was wrong; the
design of record is `SYSTEM_DESIGN.md` §5.4 + this step file).

## What

Port KBR-287's judge-first discipline to the last streaming leg — the
`use_custom_transport` branch of `BridgeServer._stream_messages`
(`src/kitty/bridge/server.py`, the branch around the
`if self._active_provider.use_custom_transport:` check inside the KBR-249
dispatch loop — re-locate by text; the citations on this ticket were
re-derived on the current tree). The branch parses the upstream response into
a complete Chat Completions dict (`parse_stream_to_cc_response` for Bedrock /
Ollama Cloud; `OpenAISubscriptionAdapter._parse_sse_to_response` for the
subscription) and translates it through `MessagesTranslator`; the judge runs
on the parsed response with `_is_empty_cc_response` (the KBR-285 lockstep
whole-response twin of `_cc_chunk_carries_content`).

Unlike KBR-287 / KBR-293, the branch is atomic (parse → translate → emit, no
per-chunk streaming), so no chunk-list synthesis is needed — the
whole-response judge is the natural shape, and no chunk synthesis is
introduced. The KBR-293 route-translation step is also unnecessary here: this
route's wire is already Messages events; `translate_response` produces them
directly.

The four KBR-287 review-settled decisions carry over verbatim: judge-first
not hold-walk; `empty_response` not `cross_class_exhaustion`; one
class-agnostic empty-arm `_select_backend()`; attempt bound
`n_backends + len(_EMPTY_FINAL_DELAYS)` with the final-delay prologue, the
exception-path gates staying at `n_backends - 1`, and the exception-path log
denominators following `n_backends` (mirroring the KBR-293 twin). The
exhaustion terminal is **the Messages route's own D4 shape** (the plain
path's `empty_no_finish` terminal — re-locate by text):
a bare JSON `_make_error_response({"type": "error", "error": {"type":
"api_error", "message": _NATIVE_EMPTY_REPLY_MESSAGE, "reason":
"empty_response"}}, status=502)`.

It is **not** copied from the SSE siblings — this route defers `sr.prepare()`
and answers pre-stream failures with a bare JSON error (§5.3 S8), so the D4
terminal must be the JSON shape, not an in-stream SSE error event followed by
a lifecycle close.

## Why the terminal shape differs from the SSE siblings

`/v1/chat/completions`, `/v1/responses`, and `/v1/gemini` all eagerly call
`sr.prepare()` at the top of their handlers, so by the time the ladder runs
the client has received `200 + text/event-stream` and an empty D4 must be
delivered **in-stream** (an SSE error event followed by a lifecycle close
on Responses / Gemini; a JSON error on Chat Completions). `/v1/messages`
defers prepare (the `_ensure_prepared` helper) so a D4 before any emit can
be a bare JSON response with the right HTTP status (§5.3 S8). The plain
path's own `empty_no_finish` arm is the precedent: JSON 502 with the same
`error.reason == "empty_response"` discriminator the route uses for the
mid-stream empty-after-content case. Same discriminator, two terminal shapes
by stream position — a deliberate asymmetry recorded in `SYSTEM_DESIGN.md`
§5.3 S8.

## Corrected defect description (tickets vs. code)

The ticket's "zero content blocks" wording described the defect incorrectly.
Today an empty completion on this branch does **not** deliver a silent turn —
it delivers **fabricated fallback text**. `MessagesTranslator.translate_response`
has a defensive fallback (`src/kitty/bridge/messages/translator.py:766-771`,
"never emit thinking-only or empty assistant output") that appends a `text`
block carrying `_EMPTY_ASSISTANT_FALLBACK_TEXT` — *"Upstream model returned
an empty response. Please retry. If the context is full, use /clear to reset
the conversation."* — plus a `(provider, model, after N attempts)` suffix on
a balancing pool. `_log_usage` then records the fabricated turn as a billed
completion. KBR-297 closes this: the judge pre-emits and the ladder walks
before `_fallback_assistant_text` can run, so no fallback text ever reaches
the client and no usage is billed for a judged-empty completion.

## Reasoning-only pin

Neither parser surfaces reasoning (the KBR-293 fidelity callout records
this for the subscription; `OllamaCloudAdapter.parse_stream_to_cc_response`
only reads `delta.content` and `delta.tool_calls`; `BedrockAdapter` mirrors
Ollama Cloud). A reasoning-only completion parses to a message with no
content / tool_calls / function_call / refusal / reasoning_content — judged
empty by `_is_empty_cc_response`, walks the ladder, the recovered "hello"
attempt's text reaches the client. Reasoning never appears on the wire.

**One-step reasoning-widening asymmetry (recorded):** on this route the
judge reads the parsed message directly and
`MessagesTranslator.translate_response` already maps
`message.reasoning_content` → a `thinking` block. So a future parse-widening
**alone** would regain reasoning fidelity here — unlike the siblings, whose
chunk-synthesis also projects only `content` / `tool_calls` and would need
its own widening.

## Whitespace-drift consequence (recorded)

`_is_empty_cc_response`'s CC arm keeps its documented `.strip()` on string
`content` (the KBR-277 paragraph records the drift as pre-existing), while
the sibling routes' judge (`_cc_chunk_carries_content`) uses `!= ""`. On this
route the custom segment now takes the `.strip()` side: a whitespace-only
completion ladders to the 502 D4 terminal here, while the same shape is
delivered on `/v1/chat/completions` and the KBR-293 siblings — and within
this same route, the plain-POST streaming hold (chunk-predicate semantics)
would deliver it too. Aligning the two predicates is not this ticket's scope.

## Empty-arm liveness omission (deliberate)

The empty arm's sleeps (the pool-less backoff and the final-delay prologue)
do not call `_raise_if_client_gone()` although the helper exists in this
handler and the plain path uses it before each native upstream attempt. The
KBR-287/293 empty arms omit it too; adding a raise path into a segment that
has no `ClientDisconnectedError` handler today is beyond this ticket's scope.
A dead client plus a broken upstream burns the ~60 s ladder; the omission is
recorded so it can be revisited as a cross-cutting pass over all four custom
segments rather than quietly persisting.

## Non-streaming residual (out of scope)

The non-streaming `/v1/messages` × custom-transport path (through
`_request_with_retry`) still returns the empty response for
`translate_response` to dress in fallback text and still marks the backend
healthy on success — pre-existing, a separate defect class from the
streaming silent turn. KBR-297 is scoped to the **streaming** custom segment
only; the non-streaming residual is recorded here so the "last streaming
leg" claim stays honest.

## Verification

Bridge-level tests in the KBR-287 / KBR-293 harness shape
(`tests/bridge/test_messages_custom_transport_empty_hold.py`): real
`BridgeServer` over aiohttp, canned bytes through the real parse step,
parametrised over the three `use_custom_transport` adapters, content oracle
parsed from Messages-API events (`content_block_delta` /
`content_block_start` — never raw substrings). Watched red pre-fix (the
red observation on the empty / exhaustion / reasoning cells is the
fabricated fallback-text turn, not a silent skeleton — the falsification the
ticket requires), green post-fix.

`SYSTEM_DESIGN.md` §5.4 KBR-293 paragraph — the deferred `/v1/messages`
sentence (re-locate by text — the ticket's "content-less Messages turn"
wording must also be corrected, since the actual wire is fabricated text,
not silence) is retired and replaced with a paragraph recording the fix
outcomes and the four carried-over review decisions, the corrected defect
description, the whitespace-drift consequence, the reasoning-widening
asymmetry, the empty-arm liveness omission, and the non-streaming residual.

## Implementation notes (2026-09-21)

- Landed on `fix/kbr-297-messages-custom-transport-empty-hold`.
  Falsification baseline: **9 red / 6 green** on the unfixed code
  (empty-ladder ×3, reasoning-only ×2, exhaustion ×3, crossing ×1 red;
  content-bearing ×3 and tool-call-only ×3 green sanity cells). The red
  cells showed the **fabricated-fallback-text 200** — the corrected As-Is
  (fallback text, not a silent turn) confirmed empirically before the fix.
  Post-fix: 15/15 new, 92/92 focused (all four custom-transport files +
  post-emission), 59/59 balancing, mypy clean on `server.py`.
- One mid-implementation correction: `_is_empty_cc_response` is a
  **method** (`self._is_empty_cc_response(...)`), unlike the sibling
  routes' module-level `_cc_chunk_carries_content`; the first T3 cut
  raised `NameError`, caught immediately by the test run. Recorded in the
  REQUIREMENTS §8 reference list so the next reader does not repeat it.
- Full-suite round 1 caught the one real behavioural consequence outside
  the new tests — the **third instance of the same stale-stub pattern**
  (KBR-254's sibling stubs, then KBR-293's corrections, now these two):
  `tests/bridge/test_balancing_server.py::TestBalancingAllCustomTransport::
  test_messages_stream_uses_custom_transport` and
  `::test_streaming_skips_backends_without_stream_request` fed
  **Responses-API SSE to a `BedrockAdapter`** (wrong wire; parsed to an
  empty completion), so pre-fix they passed only on the fabricated
  fallback text. Corrected to what real Bedrock writes (CC-SSE with
  content): test 1 keeps its "hi" text as CC chunks, test 2 reuses the
  shared `_fake_hello_cc_stream` helper. What the two tests originally
  proved (streaming uses `stream_request`, not aiohttp; streaming skips
  backends without `stream_request`) is unchanged — the corrections
  strengthen the oracles from status-only to content-bearing.
- Design-reviewer confirmation pass (2026-09-21): all corrections
  verified faithful; the liveness-omission disposition explicitly endorsed
  (today a `_raise_if_client_gone()` raise in the empty arm would
  propagate to the handler catch-all — a pointless 500 attempt on a dead
  socket; doing it properly means adding a `ClientDisconnectedError` arm
  this segment does not have — a cross-cutting change over all four
  custom segments, recorded for a future pass).
- The `_original_body` pop in the empty arm's plain-crossing branch is a
  defensive no-op on this route (only the Responses handler sets
  `_original_body`); kept for uniformity with the KBR-293 twin and this
  method's exception-path crossing site. A reader should not go hunting
  for where it gets set here.
- Code-review round 1: **APPROVE**, no critical findings. One shared gap
  flagged for visibility: the empty arm's **stay-custom refresh branch**
  (`_resolved_key` / `_provider_config` refreshed + `continue`) is
  untested on all four custom segments — KBR-287/293 never tested it
  either; FR-7 pins the crossing path, so the refresh path is
  review-verified only. Candidate for the same future cross-cutting pass
  as the liveness omission (a four-segment symmetric test would need a
  two-custom-backend pool with per-backend scripts in each file).
