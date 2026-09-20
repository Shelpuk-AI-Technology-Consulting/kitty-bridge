---
id: kbr293_sibling_custom_transport_holds
depends_on: [kbr287_custom_transport_empty_hold]
---

# KBR-293 — Sibling custom-transport holds on `/v1/responses` and `/v1/gemini`

Ticket: [KBR-293](https://shelpuk.atlassian.net/browse/KBR-293) (KBR-136 epic,
label `found-by-kbr-287`). Design: `SYSTEM_DESIGN.md` §5.4 KBR-293 paragraph.
Requirements: `.requirements/20260920T220520Z_kbr293_sibling_custom_transport_holds/REQUIREMENTS.md`.

## What

Port KBR-287's judge-first custom-transport discipline from
`_stream_chat_completions` to its two siblings, `_stream_responses` and
`_stream_gemini`, plus the one step KBR-287 did not need: the synthesised CC
chunk list is **translated through the route's own translator**
(`ResponsesTranslator` / `GeminiTranslator.translate_stream_chunk`) before any
write, because on these two routes the route's wire is Responses / Gemini
events, not CC chunks — Bedrock and Ollama Cloud emit CC-SSE on every route,
and the Codex subscription emits Responses-SSE, so `/v1/gemini` receives the
wrong wire from all three providers and `/v1/responses` from two of three.

## Why the route-translation step exists (KBR-287 did not need it)

On `/v1/chat/completions` the synthesised chunk list **is** the route's wire —
writing it directly is correct. On the siblings the same list must be converted
through the route's translator exactly as the plain-POST path converts each
upstream CC line. Skipping that step would ship CC-SSE bytes to a Codex or
Gemini client — the pre-fix defect for the content path, which is broader than
the empty-completion skeleton the ticket names.

## Fidelity callout

`/v1/responses` × `OpenAISubscriptionAdapter` was a native Responses-SSE
passthrough pre-fix (reasoning summaries included). Post-fix it runs
parse → synthesise → translate, and `_parse_sse_to_response` drops reasoning
summaries — the same trade-off KBR-287 accepted on `/v1/chat/completions` ×
subscription. Rejected alternative: a wire-aware branch preserving passthrough
when the provider's native wire matches the route (two write paths, a brittle
per-pair heuristic). Regaining fidelity later means lifting reasoning into the
parse projection — a parse-step widening, not a branch fork. See REQUIREMENTS
§7.

## Verification

Bridge-level tests in the KBR-287 harness shape
(`tests/bridge/test_responses_custom_transport_empty_hold.py`,
`tests/bridge/test_gemini_custom_transport_empty_hold.py`): real
`BridgeServer` over aiohttp, canned bytes through the real parse step,
parametrised over the three `use_custom_transport` adapters, content oracles
parsed from the route's protocol events (never raw substrings — the KBR-249
vacuous-oracle trap). Watched red pre-fix (the falsification the ticket
requires), green post-fix.

## Implementation notes (2026-09-20)

- Landed on `fix/kbr-293-sibling-custom-transport-hold`. Falsification
  baseline: 13/14 red on `/v1/responses`, 14/14 on `/v1/gemini` (the single
  pre-fix green cell — subscription tool-calls on `/v1/responses` — is the
  native passthrough satisfying the parsed oracle; its empty/usage defects
  were red). Post-fix both files green plus KBR-287's 14 and the balancing
  file's 59.
- The design review (system-design-reviewer, no blockers) caught one real
  test-oracle bug before it shipped: a raw `"role": "assistant"`
  substring-absence check would have failed on the FIXED wire (the route's
  lifecycle items carry that role) — replaced with the parsed
  `object: "chat.completion.chunk"` absence check.
- The one-function-scope mypy trap fired for real: the success-exit rewrite
  dropped the `upstream_status: int | None` annotation and a later
  `= None` in the plain path went red. Restored; mypy clean.
- The 30 s `test_streaming_skips_backends_without_stream_request` slowness
  (KBR-249-era, `/v1/messages`) is pre-existing — verified identical on
  main's server.py; out of scope here.
- Full-suite round 2 caught the one real behavioural consequence outside the
  new tests: `tests/bridge/test_post_emission_no_failover.py`'s
  Q14(a) test (KBR-247-era) pinned "failure after bytes does not fail
  over" on these two branches. Post-fix the premise is obsolete by
  construction — collected bytes never reach the socket before the verdict —
  so the test was rewritten to pin the stronger guarantee (failing attempt's
  bytes never ship; failover serves the next backend). The Q14(a) rule
  itself is unchanged on every incrementally-streaming path.
