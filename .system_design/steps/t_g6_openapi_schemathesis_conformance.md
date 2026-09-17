---
id: t_g6_openapi_schemathesis_conformance
title: T-G6 — OpenAPI 3.1 document and schemathesis conformance
jira: KBR-82
depends_on:
  - t_w8_bridge_fixture_core_and_transport_extension_interface
---

# T-G6 — OpenAPI 3.1 document and schemathesis conformance

## Scope

An OpenAPI 3.1 document for the eight routes the bridge registers in bridge mode,
plus a schemathesis conformance run, a per-protocol registration-matrix guard,
and the per-route ingress normalisers that harden the 500s KBR-159 and the
pre-flight measured. All 500s the conformance run could reach on day one are
fixed in this task, per the plan's "no sibling leaves the suite red waiting".

## Delivered

* `openapi/kitty-bridge.yaml` — the schema. Dual-mode 200s (JSON + SSE) on the
  three routes that branch on `body.stream`; SSE-only on `:streamGenerateContent`,
  JSON-only on `:generateContent`. No 500 is documented — any 500 the conformance
  run finds is a real defect, not a contract.
* `tests/test_openapi_conformance.py` — the conformance run, gated normally.
  Schemathesis 4.x's `Case.call_and_validate` is sync (uses `requests`); the call
  is wrapped in `asyncio.to_thread` so the bridge's aiohttp loop stays free.
  `base_url=` is passed per call because `SchemathesisConfig` on 4.x has no
  `base_url` field. `stream` is bounded to `enum: [false]` in the schema's
  request properties: the recording upstream replies non-streaming, and a
  `stream: true` case would spin the bridge's empty-response retry ladder
  against it.
* `tests/test_route_registration_matrix.py` — the per-protocol registration
  matrix plus the schema↔routes agreement guard. aiohttp regex converters
  (`{model:.*}`) are normalised to `{model}` before set-equality. A third
  guard pins the auth-off precondition on the fixture (a future fixture change
  passing `keys_file` would make the conformance run vacuously green).
* `tests/test_openapi_schema.py` — the schema's structural guards.
* `tests/test_responses_normalizer.py` — the four KBR-159 500s pinned as
  known positives with `error.reason == "invalid_input"`; the three
  confirmed-clean shapes pinned with positive controls.
* `tests/test_route_preflight.py` — the pre-flight measured four additional
  500s (one per non-`/v1/responses` POST route); each is fixed by a per-route
  ingress normaliser (`_normalize_messages_request`, `_normalize_chat_completions_request`,
  `_normalize_gemini_request`) added in `src/kitty/bridge/server.py`, raising a
  route-specific error class rendered as the dialect's 400 envelope.
* `src/kitty/bridge/responses/translator.py` — `normalize_responses_request`
  hardening: `tools` must be a list of objects (function tools must carry a
  non-empty `name` and an object `parameters`; other tool shapes stay
  unvalidated — the translator skips them on purpose), and a reasoning item's
  `summary` must be a list of objects carrying a string `text`.
  **The `function_call_output` shape is deliberately not validated here** —
  KBR-169's `_drop_orphan_response_outputs` owns it (orphan-drop, 200), and
  duplicating it would break `tests/bridge/test_bridge_server_openai_subscription.py`.
* `pyproject.toml` — `schemathesis>=4.9,<5` under `[project.optional-dependencies].dev`.

## Decisions and why

* **`base_url=` per call, not on the schema config.** `SchemathesisConfig` on
  4.27.3 has no `base_url` field; the per-case kwarg is the supported wiring.
  Verified empirically.
* **`asyncio.to_thread` around `call_and_validate`.** The call is sync; the
  bridge runs aiohttp on the `pytest-asyncio auto` loop. Without the thread
  shift the test deadlocks.
* **`stream` bounded to `false` in the schema.** The recorder does not produce
  streams; `stream: true` bodies spin the empty-response retry ladder for
  ~35 s per case. Streaming is exercised by L3 bridge tests, not this gate.
* **`{model:.*}` → `{model}` normalisation before set-equality.** Same
  character-exact-match trap KBR-9 hit on the README endpoint table.
* **Per-route normalisers, not a shared one.** Each dialect's 400 envelope
  differs (Anthropic `error.type`; OpenAI `error.code`; Gemini
  `error.code` + `error.status`), and each route's shape contract differs
  (`messages` vs `contents`). One shared validator would tie its error text
  to one dialect.
* **KBR-169 wins on `function_call_output`.** Its orphan-drop (200 with the
  unpaired item silently removed) is the shipped contract; hardening the same
  shape to 400 would duplicate its job and break its test.

## Implementation notes

* The pre-flight found the four additional 500s in one sweep, because the
  plan's pre-flight step (F11) asked for the measurement *before* the
  conformance run gates. The four normalisers and their tests were written
  in the same task.
* The `pytest-asyncio auto` mode treats a sync test depending on an async
  fixture oddly — the conformance test is written as `async def` and
  hypothesis health-check suppression for the function-scoped fixture is
  declared explicitly.
* Hypothesis' shrink phase does not respect `max_examples`; `no_shrink=True`
  plus `phases=[Phase.generate, Phase.target]` keep a single failing case
  from generating dozens of 10-second-timed-out shrinking attempts.
* **Code-review round 1:** the first launch-mode matrix test was subset-only
  (`expected ⊆ routes`), so a leaked `/v1/messages` into the RESPONSES_API
  branch passed — falsified by the reviewer. The walker now locates each
  `if protocol == BridgeProtocol.X:` dispatch body and asserts the *exact*
  route set per branch. The `_launch_dispatch` helper it replaced looked in
  the bridge-mode `if`'s `else:` arm, which does not exist in this method
  (the dispatch lives at method-body level after the `return`).
* **Code-review round 1:** two diverging implementations of
  `_normalise_aiohttp_path` consolidated onto the stronger one (handles
  mid-path and nested-brace placeholders); `test_openapi_schema.py` imports
  it from `test_route_registration_matrix.py`.
* **Code-review round 1:** the SSE 200 schema descriptions now state
  plainly that the L2 gate bounds `stream` to `false` and drives only the
  JSON path — the SSE schema is for human readers, not for the gate.
