---
id: openai_subscription_400_diagnostics
depends_on: []
---

# OpenAI subscription member: surface the 400 body, reproduce, fix from evidence

Live diagnosis 2026-10-29 (pool `my`, `openai` member, model `gpt-5.6-terra`):
every dispatch fails `OpenAI subscription error 400: ` with an empty body in the
message and parks 300 s; the member has never served a request. The empty log
line hides the cause (model id vs request shape vs session). Requirements:
`.requirements/20261029T184032Z_openai_400_diagnostics/REQUIREMENTS.md`.

## Why

The custom-transport failure sites log `exc` only, and the error's `str` drops the
upstream body. A permanently-400 member is invisible to the operator without the
body, and balancing launches never pre-probe members, so the diagnosis must come
from the failure path itself. The fix order is evidence-first: render the bounded
body, reproduce with a one-request direct probe, and only then correct the
confirmed cause (config-level model id or bridge-level shape) — no speculative
model or translation changes.

## To Be

- The 400's upstream body reaches the debug log bounded (500 chars) on every
  custom-transport stream failure; empty and absent bodies log nothing (the
  body is carried as-is, not redacted — a 400 body can echo request material,
  which is why it is DEBUG-only).
- Probe evidence recorded under the requirements folder names the cause.
- The follow-up fix lands behind a pinning test.

## Verification

- L1 error-rendering unit (present/empty/non-JSON/oversized, redaction), L3
  stream-failure logging, probe evidence, full sweep.

## Implementation notes (2026-10-29, diagnostic half delivered)

- `ProviderError.upstream_body: str | None` (bounded 500, JSON-encoded) attached by
  `OpenAISubscriptionAdapter.map_error` (status wording moved to `_map_error_by_status`);
  empty body stays `None` so the helper is silent.
- `BridgeServer._log_custom_transport_failure_body` renders it at DEBUG — the body goes to
  DEBUG only, not the warning (owner-approved: a 400 body can echo request material and must
  not print at default visibility). All FOUR sites call the helper (Responses, Messages,
  Gemini, Chat Completions); an AST drift guard
  (`tests/bridge/test_custom_transport_failure_body.py`) pins the pairing.
- The warning line's wording and level are unchanged.
- Tests: `tests/providers/test_openai_subscription.py::TestMapError` (+3: incident-shape
  carriage, 500-char bound, empty-body None) and the drift guard (3).
- Probe + evidence-based correction remain open (owner-action; the probe is what settles
  model id vs request shape vs session).

## Implementation notes — review round 2 (2026-10-29)

- Body carriage widened to the custom-transport siblings: `BedrockAdapter.map_error` and
  `OllamaCloudAdapter.map_error` attach the bounded body via the new shared
  `ProviderAdapter._bounded_upstream_body` (the 500-char bound lives in one place;
  `OpenAISubscriptionAdapter` was moved onto it). Both siblings are `use_custom_transport`
  providers, so their stream failures reach the same four DEBUG sites.
- Non-streaming twin wired: the balancing walk's generic failure arm
  (`_request_with_retry_balancing`, the arm that parks the member) calls the helper after its
  `Backend attempt ... failed` INFO line; an AST guard pins the pairing. Other routes'
  single-backend non-stream raises stay uncovered — recorded, not accidental.
- Step "To Be" wording corrected the same day (bound-only, as-is carriage, silent on empty).
