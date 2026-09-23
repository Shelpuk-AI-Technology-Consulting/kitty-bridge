---
id: kbr307_t_d11_scope_enforcement
depends_on: [KBR-51, KBR-53, KBR-139]
---

# KBR-307 — T-D11 Scope enforcement: claim matching filters by row.scope

Jira: **KBR-307** ([link](https://shelpuk.atlassian.net/browse/KBR-307)).
Design: `TEST_SUITE.md` §3.3.1a, §3.3.2; register scope data added by
**KBR-139** (`row_is_in_scope` at `register.py:382`). Requirements:
`.requirements/20260923T103055Z_kbr307_t_d11_scope_enforcement/REQUIREMENTS.md`.

## What the task does

Closes the data-side half of the register-coverage family: every
register row whose `scope` excludes `provider_key` is filtered out of the
runtime oracle surface (`_claim_matching` and `_conditional_violations`)
before it can contribute a claim. P17 is the canonical example — an
ALWAYS-triggered, `scope=("openai_subscription",)` row that claims
`envelope.stream`/`envelope.store` — but the rule is general.

The companion rules (T-D8 / KBR-60 trigger-complement coverage checker;
KBR-140 trigger predicates) live in their own tickets; this one is the
data-side counterpart.

## Decisions

- **`provider_key` is a keyword-only parameter, not a capture attribute.**
  A `CapturedRequest` carries no adapter binding; the binding is external
  (`transport.bind()` returns `(adapter, provider_config)`). Reading it
  would couple the oracle to a transport and regress the L1/T-D1
  discipline (the oracle projects, classifies, asserts — no IO, no
  transport). Keyword-only is the cleanest seam and is consistent with
  `expected_route` already living on the same signature.
- **Default to `r.ALL_PROVIDERS`, not a required argument.** Existing
  callers keep working without modification; the default mirrors the
  helper's `ALL_PROVIDERS in row.scope` semantics and is permissive when
  the caller has no opinion. AC3 pins this with a documented sentinel.
- **Both helpers filter, not just `_claim_matching`.** §3.3.2 assertion
  2's specificity-attribution computes "live on this adapter" from the
  same notion as assertion 1's claim matching. Filtering one but not the
  other creates a contradiction the runtime oracle cannot surface — AC2
  pins it.
- **Two driven cases (AC4, AC5), not one.** AC4 names the failure mode
  the ticket exists to prevent. AC5 names the per-adapter property the
  design claims: without it, a future bug that makes the filter global
  (e.g. dropping all narrowly-scoped rows) passes AC4 and silently
  breaks the openai_subscription route.
- **Driven tests use the harness's default aiohttp transport on
  `WireFormat.ANTHROPIC_MESSAGES`** rather than re-hosting
  `AnthropicAdapter` via `redirected()`. P17's scope excludes every
  adapter except `openai_subscription`, so the simpler drive is honest
  about scope semantics and matches T-D1's vertical-slice precedent.
- **No pytestmark** — files default to `l1`. T-K6 still owns the
  promotion to gating infrastructure.

## Implementation notes

Implemented (2026-09-23); branch `feat/kbr-307-t-d11-scope-enforcement`
off `origin/main`.

- **`provider_key` is keyword-only on all three oracle entry points**
  (`assert_no_unclaimed_mutation`, `_run_assertions`, `_claim_matching`,
  `_conditional_violations`), defaulting to `r.ALL_PROVIDERS`. The
  default is permissive **only** via an explicit call-site short-circuit
  (`if provider_key != r.ALL_PROVIDERS: filter`); the helper itself is
  not permissive when handed the sentinel — review round 1 caught this
  (`row_is_in_scope(row, "*")` is `False` for every narrow-scoped row).
- **`_conditional_violations` filters both row-sets** — the triggered
  `active` set used for specificity-attribution *and* the iteration over
  untriggered conditional rows. The ticket named only the specificity
  set; the iteration filter closes the same hazard on the assertion-2
  half (P22/P25, P26, the P5 family are narrow-scoped conditional rows).
- **Driven case derives `provider_key` from the binding**
  (`fixture.transport.bind()[0].provider_type` → `custom_anthropic`),
  not a hardcoded `"anthropic"` — the ticket's "e.g. anthropic" is
  illustrative, and the derivation seam is what the corpus runner
  (KBR-55/56/57) will use.
- **The openai_subscription companion stays projection-level.** The
  curl_cffi transport's binding needs TLS certs + OAuth seeding
  (`test_bridge.py:62`); the ceremony is KBR-55/56/57's scope, and AC5's
  rule pins at the cheapest layer that can prove it.
- **`triggers_met = frozenset({ALWAYS, PROFILE_SETS_MODEL})` for the
  driven case** — ALWAYS keeps P17 trigger-eligible (the scope gate is
  the only thing that can keep it from claiming); PROFILE_SETS_MODEL
  pins M1 to the model delta so the harness profile's rewrite does not
  surface as a stray failure.

### Red evidence (planted removal, both filters, T-D3 R6 precedent)

With the `_claim_matching` scope filter short-circuited to `pass` (the
loop over `active_rows` unchanged) **and** the `_conditional_violations`
iteration filter planted the same way, five of the seven
`TestScopeFilter` cases fail for the right reason:
`test_claim_matching_filters_rows_outside_provider_scope` (P17 appears
in the claimers on `anthropic`),
`test_conditional_violations_filters_iteration` and
`test_run_assertions_forwards_provider_key_to_conditional_violations`
(`ConditionalRowFiredWithoutTriggerError` naming `Z-SCOPE-COND` on
`provider_key="anthropic"` — the row whose iteration filter is planted),
`test_run_assertions_raises_unclaimed_when_scope_filter_removes_the_only_claimer`
(no raise where the raise is required), and
`test_run_assertions_p17_claims_on_openai_subscription` (the anthropic
half stops raising). Review round 2's blocker on the
assertion-1-first ordering was folded in before the red run: cases 2 and
2b use a **two-row** fixture (broader triggered claimer + narrow
conditional R) so the discriminator reaches assertion 2 instead of being
masked behind assertion 1's raise. The remaining two cases
(`test_default_provider_key_is_all_providers_permissive`,
`test_scope_filter_removal_breaks_claim_matching`) pass by construction
under both plantings. Filters restored; L1 gate green (59 passed);
harness sweep green (2638 passed, 2 skipped); ruff and mypy green.
