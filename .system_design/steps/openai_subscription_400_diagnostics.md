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

- The 400's upstream body reaches the debug log bounded and redacted on every
  custom-transport stream failure (`<no body>` when empty).
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
