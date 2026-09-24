---
id: t_d7_provider_aiohttp_oracle_slice
depends_on: [KBR-38, KBR-40, KBR-51, KBR-52, KBR-307]
---

# T-D7 — Oracle slice: provider-aiohttp transport

Jira: **KBR-57** ([link](https://shelpuk.atlassian.net/browse/KBR-57)).
Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §7, row T-D7. Design:
`TEST_SUITE.md` §3.3.2, §3.3.4, §3.3.5, §7.2.2, §7.4 (settlement in the new
§7.4.5). Requirements:
`.requirements/20260924T083854Z_td7_provider_aiohttp_oracle_slice/REQUIREMENTS.md`.
Sibling contracts inherited from `t_d4_default_transport_slice.md`
(routing-derivation restriction, skip-table pattern, `_SENTINEL_ROUTE_PATH`
pin) — acked on the KBR-57 ticket.

## What the task does

Drives the transparency oracle end-to-end on the provider-aiohttp transport
for the `ollama_cloud` adapter — the third of the three
`use_custom_transport = True` adapters the default aiohttp recording upstream
never sees (§3.3.4). A real `BridgeFixture` drives **every**
Anthropic-Messages corpus entry against `ProviderRecordingUpstream`; the
captured OLLAMA_CHAT body is read by `reader_ollama` and compared against the
inbound Messages projection with all four obligations active (§3.3.1 totality,
§3.3.2 assertions 1 + 2, §3.3.5 routing). `provider_key` is derived from the
binding (KBR-307's seam); `Trigger.ALWAYS` stays in `triggers_met` so P19's
scope gate is the discriminating filter. The OAuth **login** leg — the
ticket's other Done-when half — is covered recorder-level on the same
instance (§7.2.2 dual-product load; no projection comparison, per §7.5's
decided question).

## P19's observation posture (the slice's one non-obvious decision)

The recorder captures **after** `_ollama_body` writes `stream` from the
transport's endpoint-mode decision — but on a consistent drive the overwrite
is **value-preserving** (the inbound body carries the same flag or none; the
readers' shared normalisation reads an absent flag as its wire default), so
the honest bytes produce no `envelope.stream` delta. This is a wire fact, not
a test gap — the same shape T-D6 recorded for P18's pops. The row is
therefore **armed, not observed by absence**: every driven run keeps P19
trigger-eligible and in scope, and `TestP19ClaimMachinery` proves the claim
machinery by flipping the captured value (claimed with `ALWAYS` met;
`UnclaimedMutationError` naming `envelope.stream` with it omitted — P19 is
the unique ALWAYS-claimant for that address on `ollama_cloud`). The
scope-**exclusion** half stays KBR-307 AC4's: a transport bound to one
adapter cannot drive a foreign one.

## Findings (skip table; owner tracking on the KBR-57 scope-addition comment)

Five entries pass clean (`deltas == ("envelope.model",)`); nine are
skip-tabled:

- **K57-F1** (`plain_turn`, `effort_configured`) — the CC→Ollama translation
  drops the Messages body's top-level `context_management` / `output_config`
  / `thinking`, merges the system blocks, and flattens multi-part content
  with `"\n"` joins; eight unclaimed paths per entry (probe-verified
  identical).
- **K57-F2** (`tool_use_and_tool_result`, `compaction_budget_over`) —
  **reader gap**: the adapter emits `id`/`type` on request-side
  `tool_calls`; the published Ollama `ChatRequest` declares `function` only
  on that sub-shape and `reader_ollama` (T-A6) has no slot, so the totality
  gate fails before any delta is classified. Go's decoder tolerates the
  extra keys upstream; resolution (reader slot vs adapter stop) is the
  owner's.
- **K57-F3** (`tool_result_under_limit`, `tool_result_over_limit`) —
  `conversation.turns[2]` dropped; T-D4's F3.d shape, new on this route.
- Framing (`tools_declared`, `system_prompt_over_window_compacts_normally`)
  and the `compaction_budget_under` calibration gap (M3's coarse `parts[*]`
  anchor vs M5's pruned texts at the 800 K budget) — no finding marker,
  T-D4's pattern.

## Falsification (plan §1.4)

Three, all running in the suite and all self-falsifying
(`pytest.raises` / else-raise): the **path pin**
(`_SENTINEL_ROUTE_PATH == OllamaCloudAdapter().get_upstream_path(...)`), the
**routing falsification** (real route passes every body obligation, then a
sentinel-wrong path must raise `RoutingMismatchError` naming `route.path` —
T-D2's reroute-with-byte-identical-body shape), and the **P19 claim
machinery** pair described above. The probe rounds of 2026-09-24
(`.scratch/probe_corpus_provider_aiohttp.py`, `probe_p19_claim.py`,
`probe_full_deltas.py`) are the red-then-green evidence: every skip row and
both P19 behaviours were observed empirically before the committed tests
encoded them.

## Companion sites

The module binds a real `BridgeServer` + `ProviderRecordingUpstream` (two
sockets per run), so it joins the §8.2 set: `tests/socket_binding_l1_modules.py`
entry (after the T-D6 row, before the KBR-10 paragraph entry), the registry
docstring's bullet count, the guard test renamed to
`test_registry_has_exactly_seventeen_entries` (count pin 17), the pyproject
mutmut `--ignore` row in registry order, and the §8.2 bullet (T-D6's as the
template) plus the §8.2 opening sentence's module count (Sixteen →
Seventeen) and three numbers in the §8.2 enumeration sentence — the new
entry makes each stale; the sentence was internally consistent before.

## Status

Implemented on `feat/kbr-57-oracle-slice-provider-aiohttp` off `origin/main`
(2026-09-24). 19 tests: 10 passed, 9 skipped (5 clean corpus entries + pin +
routing falsification + 2×P19 + OAuth leg; 9 skip-tabled corpus entries).
Design review (system-design-reviewer, 2026-09-24, two rounds — 8 concerns +
5 suggestions, then 4 re-review residue items, all applied) converged; code
review pending. No register, reader, adapter or product changes in the diff
(T-D6 merge policy).

### Implementation notes

- The trigger set is `entry.triggers_met ∪ {NON_NATIVE_UPSTREAM_WIRE,
  PROFILE_SETS_MODEL, ALWAYS}` (+ `OVER_COMPACTION_BUDGET` above the 800 000
  literal) — T-D4's `_triggers_met` plus the ALWAYS KBR-307 requires.
- The budget literal `_DEFAULT_PROFILE_BUDGET_CHARS = 800_000` is derived
  from `get_model_context_tokens("ollama_cloud", "harness-model", None)` →
  `DEFAULT_CONTEXT_TOKENS = 200_000` × 4; recorded beside the literal (the
  driver is a judge; no `src/kitty` import beyond the sentinel pin).
- The P19/OAuth falsification entry selection is corpus-derived (smallest
  clean entry whose inbound body does not ask for `stream: true`), so corpus
  renames flow through; today it resolves to `no_output_config`.
- `no_output_config` is the P19 subject because the entry with inbound
  `stream: true` (e.g. `format_example`) makes the flip a no-op — the flip
  must change the value for the delta to exist.
- The OAuth-leg test reuses the corpus-derived clean entry for the bridge
  half and drives `openai_oauth._exchange_code_for_tokens` through
  `oauth_token_endpoint(subject.recorder)` — the login leg only; the refresh
  leg is T-B2's (KBR-161 split).
