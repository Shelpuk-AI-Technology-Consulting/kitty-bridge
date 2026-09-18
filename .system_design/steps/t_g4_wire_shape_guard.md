---
id: t_g4_wire_shape_guard
depends_on: [KBR-30, KBR-40, KBR-41, KBR-42]
---

# T-G4 — Wire-shape honesty guard at the wire boundary (KBR-80)

Plan row: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §10 **T-G4**. Design:
`TEST_SUITE.md` §6.2.3, "Wire-shape honesty". Jira: **KBR-80**.

Depends on T-W7 (KBR-30), T-B1 (KBR-40), T-B2 (KBR-41), T-B3 (KBR-42) —
all Done at the time this step landed. The `depends_on` uses the Jira keys
because the plan-task IDs (T-W7, T-B1…) have no step files of their own;
this is the first step file in the directory.

## Delivered (2026-09-17)

1. **`tests/test_wire_shape_honesty_wire.py`** (new, L2) — the wire form of
   the honesty guard. Captures the body at the §3.2.3 serialization boundary
   for each custom-transport adapter and pairs the captured classification
   against the declaration, per the same discipline the hook sweep uses:

   * `bedrock` — stubbed boto3 client; captures the kwargs to
     `client.converse` after the P18 pops (`modelId`, `stream`). Declared
     `OTHER` (Converse).
   * `ollama_cloud` — stubbed aiohttp session; captures the `json=` to
     `session.post` after the P19 overwrite. Declared `CHAT_COMPLETIONS`.
   * `openai_subscription` — captures at the §3.2.3 body builders
     (`_cc_to_responses` CC-origin, `_prepare_responses_body`
     Responses-origin), plus a structural pass-through pin that drives
     `make_request` with a stubbed curl_cffi session and a real
     (fresh-token) OAuth session file and asserts the posted `json`
     equals the builder output exactly.  The comparison is plain
     dict equality, which catches value changes for present keys,
     additions/removals, and list-element reorders — but not
     top-level dict-key reorders (Python `dict.__eq__` is order-
     insensitive), so the docstring states the gap rather than
     overclaiming.  Declared `RESPONSES`.
   * Falsification per §1.4: one deliberate-defect case per capture, all
     defects introduced adapter-side (the guard's subject), each
     demonstrating the mismatch is visible.
   * `provider_config` sweep (R2): `minimax_token` with and without
     `{"native_messages": True}`, paired against the declaration.
   * Native-passthrough (R3): for each adapter with
     `use_native_messages` True, drives `translate_to_upstream` with a
     Messages-shaped `_native_messages_request=True` body and asserts both
     the wire shape and body preservation (input minus
     `ProviderAdapter._INTERNAL_KEYS`).
   * `test_every_custom_transport_adapter_has_a_wire_capture` — the
     captured set is asserted against `CUSTOM_TRANSPORT_ADAPTERS`, so a
     fourth custom transport forces a decision (capture + set update).

2. **`src/kitty/providers/openai_subscription.py`** — **the guard caught a
   live defect.** `OpenAISubscriptionAdapter` inherited
   `upstream_wire_shape = CHAT_COMPLETIONS` from `OpenAIAdapter`, but every
   body it ships is OpenAI Responses (both builders). Fixed to
   `RESPONSES`, KBR-7's atomic pattern (fix + guard together, red evidence
   in the PR — three tests failed pre-fix, green post-fix). No consumer
   impact: all four `server.py` readers of the declaration are
   unreachable for custom transports — the thinking round-trip repair
   (~5075, inside `_stream_messages`'s plain-transport branch) and the
   pre-write thinking carrier in `_upstream_body_for` (9817) both sit
   behind the `use_custom_transport` dispatch (4674 / 9973);
   `_serves_messages_wire` (9777) returns False for this adapter under
   either value; and the streaming-converter selection (9849) is
   consulted only from the three plain-transport branches (3784 /
   6423 / 7716) — under RESPONSES it would return a converter if
   reached, but the custom-transport branches (3585 / 6272 / 7493)
   dispatch to `stream_request` without consulting it.

3. **`tests/test_wire_shape_honesty.py`** — the hook sweep now names its
   exemption: `HOOK_DEAD_ON_REQUEST_PATH = {"openai_subscription"}` (the
   hook is never the shipped body there), asserted non-empty and a subset
   of `CUSTOM_TRANSPORT_ADAPTERS` by
   `test_hook_dead_on_request_path_is_a_named_subset_of_custom_transport_adapters`.
   `bedrock` / `ollama_cloud` stay in the hook sweep: their transports
   mutate the hook's output with a scalar change that does not alter the
   shape family, so the hook form remains meaningful — and the wire form
   now observes their shipped bytes. Module docstring non-claims 1–3
   updated to point at the wire form.

## Why this shape

* **Capture at the boundary, not the hook** — §3.2.3's whole point: on
  `openai_subscription` the hook is dead on the request path, so a
  hook-observing guard reports clean while inspecting a body the adapter
  never sends.
* **Adapter-side defects in the falsifications** — the guard's subject is
  the adapter's honesty; a defect injected into the stub would test the
  stub, not the adapter.
* **`provider_config` + native-passthrough in the wire file** — the plan
  row inherits them ("properties of the construction, not of the
  boundary"), and the native passthrough check is not vacuous: it drives
  each adapter's real `_native_messages_request` branch and asserts body
  preservation, which is what the bridge's forward path relies on.

## Verification

Local gate: `ruff check .` clean, `mypy src` clean, `lint-imports` clean
(5 contracts kept), targeted suite (wire + hook files) 98 passed / 1
skipped (the named hook-sweep exemption). Full `pytest -q` run before
landing; CI is the authority for the Windows/macOS legs (plan §1.3).

## Notes

* `scripts/regenerate_step_index.py` shipped after this step (KBR-278) —
  the validator now exists and runs on every step-file change. The `depends_on`
  above is validated under the D7 Jira-key exemption (the four `KBR-nn`
  entries match `^KBR-\d+$` and are accepted as cross-references without
  resolution). Before the validator existed, the dependency graph above was
  validated by inspection.
* The `stream_request` (streaming) half of each custom transport was not
  driven: the body builders are shared with `make_request`, and the only
  transport-side difference is the `stream` flag (P18 pops it in both; P19
  sets True in `stream_request`), which the existing provider tests pin.
  Stated limit, not an omission.
