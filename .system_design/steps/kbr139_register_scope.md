---
id: kbr139_register_scope
depends_on: [KBR-51]
---

# KBR-139 (T-W3a) — Register scope: which adapters each row is reachable on

Plan: `REQUIREMENTS.md` at
`.requirements/20260921T115922Z_register_scope/`. Design of record after this
step: `TEST_SUITE.md` §3.2.4 (rewritten by this step from "Scope is not
carried here" to the carried-column state).

## What

`MutationRow` gains a required `scope: tuple[str, ...]` — the provider registry
keys on which the row's site is reachable, with the explicit sentinel
`register.ALL_PROVIDERS = "*"` for the endpoint-level rows that every adapter
can reach. A helper `row_is_in_scope(row, provider_key)` interprets the
sentinel in exactly one place, for the two consumers this field exists for
(T-G2's completeness guard, KBR-79; T-D8's complement checker, KBR-60).

Two pure guards mirror the existing `row_shape_problems` / `unresolved_sites`
family, both read by AST and never by import (§3.3.1's independent-oracle
rule):

- `provider_registry(src_root)` — reads `providers/registry.py`'s
  `_registry` dict literal (key → adapter class name); raises rather than
  returning an empty result.
- `scope_problems(rows, registry, symbols)` — per row: non-empty; the
  sentinel never mixed with keys; every key a real registry key; and the
  site↔scope sibling check (every registry key whose adapter class is defined
  in a file the row's sites name must appear in scope).

## The reachability-vs-survival decision (KBR-160)

**Scope is reachability of the site**: a key is in scope when at least one
request path through that adapter executes the row's site. It is deliberately
**not** "the row's effect survives to the capture boundary". KBR-160 is the
standing counterexample: M1 was true at its site, trigger, conditionality and
scope while `OpenAISubscriptionAdapter._prepare_responses_body` discarded the
row's effect one layer down. Survival is not statically derivable (a consumer
between site and boundary can silently undo a row) and is asserted at the
§3.2.3 boundary instead — the KBR-160 pattern; generalising it to the
bridge-level rows × the three custom-transport adapters remains open and
belongs to the boundary, not to scope data.

## Derivation (all 79 live rows; method, not just results)

Read per adapter: class hierarchy (`providers/*.py`), each
`translate_to_upstream` / `build_upstream_headers` override, each `super()`
or explicit base-class delegation, `use_native_messages` overrides, the
endpoint handlers' translator gating (`server.py` — `/v1/messages` skips
`MessagesTranslator` when the active adapter is native; `/v1/responses` and
`/v1beta` never skip), and §6.2.3's custom-transport rule (the request path of
`openai_subscription` / `bedrock` / `ollama_cloud` never reaches
`translate_to_upstream`).

| Fact (source) | Rows | Scope |
|---|---|---|
| Server-level sites in `server.py`, and translator sites with no adapter carve-out (Gemini + Responses endpoints never skip; `normalize_responses_request` runs unconditionally; the M12 fallback constants are reachable via the Responses half on every adapter) | M1–M8, M10–M12, M14, M15, M17–M25 | `"*"` |
| `_convert_native_to_cc_format` runs only when `_native_messages_request` is set (server.py:5048), i.e. adapters with `use_native_messages` true (`anthropic` is **not** one — base returns False) | M9, M9a, M9b | `custom_anthropic`, `minimax_token`, `zai_coding` |
| `MessagesTranslator.translate_request` / `carry_tool_choice_and_metadata` skipped on the two adapters that hardcode native (`custom_anthropic.py:77`, `zai_anthropic.py:72`); `minimax_token` is profile-driven so it stays in | M16, M26, P34, P35 | all 23 minus `custom_anthropic`, `zai_coding` (21) |
| `ProviderAdapter._INTERNAL_KEYS` strip consults the base constant on every adapter | P1 | `"*"` |
| `_ZaiBase.translate_to_upstream`, inherited by both zai CC adapters | P2a, P2b | `zai_regular`, `zai_coding_cc` |
| `OpenAIAdapter.translate_to_upstream`; `openai_subscription` inherits it but its custom transport never calls it (§6.2.3) | P4 | `openai` |
| `AnthropicAdapter.translate_to_upstream` + `_translate_assistant_msg` + `_translate_tools`: `super()` delegation at `custom_anthropic.py:97`, `minimax_token.py:135`, `zai_anthropic.py:91`; explicit `AnthropicAdapter.translate_to_upstream(self, …)` at `opencode.py:882` for Messages-routed models (KBR-258 measured four; `opencode_go` is reachable by the same delegation read, so five — and the KBR-258 comment block above P26 in `register.py` is rewritten to name the five, so the data does not contradict its own scope values; `zai_anthropic` there is the class behind the `zai_coding` registry key, verified, so five is final) | P5a–P5f, P5e, P26–P30 | `anthropic`, `custom_anthropic`, `minimax_token`, `zai_coding`, `opencode_go` |
| `_inject_empty_reasoning_content` callers (ticket's own P8 example, confirmed by AST) | P8 | `kimi`, `custom_openai`, `zai_regular`, `zai_coding_cc` |
| `build_upstream_headers` rows: `custom_anthropic`/`minimax_token` inherit the Anthropic hook, `opencode_go` reaches it via `build_upstream_headers_for_model` on Messages models only (`opencode.py:856–860`); `zai_coding` overrides it (its own row P9f); `ollama_cloud` overrides it (excluded from P9h) | P9a–P9h | per site (P9e: `anthropic`, `custom_anthropic`, `minimax_token`, `opencode_go`) |
| Custom-transport whole-body translations and their in-transport mutations | P11–P13, P14–P25, P31–P33, P36–P38, P42 | the site's adapter |
| Route helpers | P6, P20, P21 | `azure` / `azure` / `vertex` |

The site↔scope subset guard mechanically holds the cheap direction of this
table (a row whose site names `mimo.py` cannot claim a scope without `mimo`);
the delegation direction stays a reviewed fact per the row comments.

## Why a required field, not a default

`scope` has **no default**. A default of `"*"` would reproduce P8's
over-scoping shape silently for every future row author who forgets the
argument — the register's culture is anti-silent-default for exactly this
reason (`not_projectable_reason` is required-when-escaping, not defaulted).
79 one-line edits is the visible cost and is the point: each row's scope was
a reviewed decision.

## Recorded decisions from the requirements review

- **P9e structural check deferred** (the PR #133 review proposal: "row site
  names class-method X ∧ a subclass defines X ∧ the row's prose names that
  subclass → require the row's review to cover it"). It guards a
  markdown-prose→review axis orthogonal to scope data. The per-row delegation
  comments plus the site↔scope subset guard carry the reachable-delegation
  facts; if prose drift surfaces, file it as its own follow-up. Recorded in
  §3.2.4's rewrite.
- **KBR-160 survival generalisation gets a durable home**: §3.2.4's rewrite
  names the open item (bridge-level rows × the three custom-transport adapters
  — which bridge-level rows survive to the custom transports' capture
  boundary?) as a ticket-able follow-up, so it does not evaporate with this
  step file.
- **Future tightening of the subset guard, deliberately not built now**: the
  check is file-level ("every registry key whose class file a site names must
  be in scope"). A future row whose site names a *specific* class in a
  multi-class file (`zai.py` defines `ZaiRegularAdapter` and
  `ZaiCodingAdapter`) would be over-forced to carry both keys — the P8
  over-scoping shape. The tightening is class-aware extraction (class-level
  when the site's class is a registry class, file-level fallback for base
  classes like `ProviderAdapter`/`_ZaiBase`). No live row exercises the
  divergence, so the speculative branch is declined until one does; this note
  is the record.

## Falsification cases (plan §1.4)

- A synthetic row whose scope names a key absent from the AST-read registry →
  `scope_problems` names the row and the key.
- A synthetic row mixing `"*"` with keys, and one carrying `()` → named.
- A synthetic row whose site names `mimo.py` while scope omits `mimo` → named
  by the subset check.
- The AST reader: a hand-written mini-registry source is read into
  key→class pairs (parser live); a registry-shaped source without a readable
  `_registry` dict raises (no silent empty).
- `row_is_in_scope` truth table.

## Implementation notes

- **What landed (2026-09-21).** `ALL_PROVIDERS = "*"` sentinel; `MutationRow.scope`
  (required, declared between `design_ref` and `not_projectable_reason` for the
  dataclass-ordering constraint — all call sites are keyword-form, verified);
  `row_is_in_scope`; `provider_registry` (AST reader of
  `providers/registry.py`, `RegisterSourceError` on an unreadable dict — never
  an empty result); `scope_problems` (shape / key-validity / site↔scope
  subset, pure, mirroring `row_shape_problems`). All 79 register rows scoped
  (via the three shared constants `_NATIVE_MESSAGES_ADAPTERS`,
  `_TRANSLATED_MESSAGES_ADAPTERS`, `_ANTHROPIC_FAMILY` plus inline tuples);
  the 9 synthetic `test_oracle.py` rows carry `scope=("*",)`. Per-row comments
  on P2a, P4, P8, P9e, P26–P30's block comment; the KBR-258 comment block
  rewritten to five adapters (measured-vs-derived distinction recorded).
- **Gates.** `ruff check .` clean · `mypy src/kitty` clean (94 files) ·
  `lint-imports` 5/5 kept · `pytest tests/harness/` 2552 passed 0 failed
  (136 s, pre-doc-edit run; re-run green after §3.2.2/§3.2.4 edits) ·
  `test_register.py` + `test_register_agreement.py` 117 passed after the
  design-doc edits.
- **Tests.** L1 (`test_register.py::TestTheScopeColumn`, 7): sentinel truth
  table, explicit-scope membership, and the five pinned derivation facts
  (P8's four callers, P5a's five-adapter family, M16's 21-adapter exclusion of
  the hardcoded-native pair, M9a's native trio, P4 excluding
  `openai_subscription`). L2 (`test_register_agreement.py::
  TestEveryScopeNamesRealProviders`, 13): live-register guard, registry-reader
  self-guard (23 keys + named members incl. the `zai_coding` →
  `ZaiAnthropicAdapter` file-class split), four falsification cases (bogus
  key / mixed sentinel / empty scope / site-scope omission), synthetic
  mini-registry positive control, and six `RegisterSourceError` refusals
  (unreadable source, empty dict literal, `**`-unpacking entry, non-literal
  key, non-string key, non-class value) — one per refusal path.
- **Design doc.** §3.2.4 rewritten to the carried-column state (reachability
  definition, KBR-160 survival boundary + open follow-up, P9e deferral, three
  derived facts, guard description); schema sentence "plus two"; §3.2.5 table
  gains the scope⇄registry row; P5d/P5f notes sharpen survival-vs-reachability;
  P26–P30 prose and `register.py`'s KBR-258 comment both move to five adapters;
  §6.2.3's completeness guard names the scope filter.
- **Deliberately not done.** The class-aware subset-guard tightening (recorded
  above); survival assertions at the custom-transport boundaries (KBR-160
  follow-up, home in §3.2.4); any markdown⇄scope mechanical reconciliation
  (same authority split as `site`).
- **Review trail.** Requirements: two `system-design-reviewer` rounds (7 warnings;
  one in-data contradiction). Diff: `code-reviewer` (two required fixes + polish
  landed pre-push). CI reviewer: round 1 — comment the sentinel's subset-check
  exemption (72d2144); round 2 — pin the native-trio constant against a literal
  (fe70e55); round 3 — step-file L2 count 8 → 9 (this commit).
- **Status.** PR #254 open; not merging (standing instruction).

