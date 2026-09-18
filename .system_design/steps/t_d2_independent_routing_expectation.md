---
id: t_d2_independent_routing_expectation
depends_on: [KBR-36, KBR-51]
---

# T-D2 — Independent routing expectation, plus its falsification

Jira: **KBR-52** ([link](https://shelpuk.atlassian.net/browse/KBR-52)).
Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §7, row T-D2. Design:
`TEST_SUITE.md` §3.3.5 (including the T-W4 scope addition on authority and
scheme normalisation). Requirements:
`.requirements/20260918T230000Z_td2_routing_expectation/REQUIREMENTS.md`.

## What the task does

Makes the transparency oracle consume the route. T-D1's
`assert_no_unclaimed_mutation` accepted an `expected_route` parameter and
recorded it without asserting; this task adds the assertion and its first
independent derivation, on the Azure route — the one provider where routing is
invisible to a body-only comparison (P6 removes `model` from the body, the
deployment id lives in the URL, so two deployments share byte-identical
bodies).

The oracle compares captured `method`/`scheme`/`host`/`path`/`query` against
the expectation component-by-component and raises the new
`RoutingMismatchError` on any mismatch. The expectation itself is computed in
the test from the profile using Azure's *published* URL shape — never by
calling `build_base_url()` / `get_upstream_path()` — with the authority and
scheme rewritten to the harness recorder's (§3.3.5's mandatory normalisation;
a published `https://<resource>.openai.azure.com` cannot match a
`http://127.0.0.1:<ephemeral>` recorder by construction).

## Falsification (plan §1.4)

A captured request whose deployment path segment is replaced while the body
stays byte-identical must fail the oracle — and fail on routing (the body
obligations pass first, which the `RoutingMismatchError` type itself proves).
This is the case a body-only oracle cannot see.

## Decisions

- **The derivation reimplements two behaviours rather than observing them**
  (§3.3.5 names both): Azure's cut of a pasted full `base_url` at the
  `/openai/deployments/` marker (`_cut_deployment_segment` — the
  Azure-specific operation behind §3.3.5's generic KBR-134 paragraph; a
  profile whose `base_url` already ends in the full endpoint reaches the same
  destination), and the model-half rule (deployment = prefix-stripped profile
  model when the profile names one, else the prefix-stripped inbound model —
  `_route_model`'s rule, KBR-127).
- **`api-version=2024-10-21` is pinned as a literal in the derivation.** The
  query is part of Azure's published shape and P20 claims `route.query`; a
  bump of the adapter's `_API_VERSION` is a routing-visible wire change, and
  this literal is what notices it. The alternative — deriving the value from
  the adapter's constant — would violate the independent-derivation rule.
- **The routing assertion runs after the three existing obligations.** Body
  failures are the louder diagnosis, and the ordering makes the falsification
  clean: the byte-identical body passes assertions 1–3, so the routing error
  is the only thing that can fire. The comparison is driven from
  `sorted(contract.ROUTE_COMPONENTS)` and failures carry the `route.*`
  constants in `paths`, so the component set cannot drift from the contract
  vocabulary.
- **`expected_route=None` callers are unchanged** — every T-D1 test keeps its
  behaviour; the assertion exists only when the caller supplies an
  expectation. The parameter and the `OracleReport.expected_route` field are
  both narrowed to `ExpectedRoute | None` in the same change — this is the
  filling-in moment, and no external callers pass anything else.
- **A test-local transport subclass binds `AzureOpenAIAdapter`** rather than
  extending `_ADAPTER_FOR_FORMAT`: that map is keyed by *wire format* and
  Azure's format is Chat Completions, already taken by `custom_openai`. The
  `bind()` seam is the extension point the harness documents for exactly this.
- **The positive driven case uses a prefixed profile model**
  (`azure/<deployment>`), so the derivation's prefix-strip is exercised
  through the bridge and not only in the pure derivation test — and the case
  foreshadows T-D3's seventh falsification (KBR-127), which is **deliberately
  out of scope here**: its auth-scheme leg is a header claim and its
  body-shape leg is the projection diff, both outside the routing assertion.
  T-D3 owns it.
- **No pytestmark** — the file defaults to l1 per the T-D1 precedent
  (`test_oracle_driven.py`); the §3.4 table calls this surface L3, and T-K6
  owns the l3 activation, so no `test_layer_selection.py` registration is
  needed.

## Status

Implemented (2026-09-18); PR #238 draft on `feat/kbr-52-routing-expectation` off
`origin/main`. Code review round 2 APPROVED (requirements coverage complete,
falsification proven by exception type); design review on the R3 rewording
APPROVED with a mechanically-checkable tightening applied to the requirement
and the test module docstring. Awaiting CI (6-leg matrix) and the repository's
own reviewer (skipped while the PR is draft).
