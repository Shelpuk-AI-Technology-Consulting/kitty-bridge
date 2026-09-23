---
id: t_d4_default_transport_slice
depends_on: [KBR-43, KBR-51, KBR-52]
---

# T-D4 — Oracle slice: default aiohttp transport

Jira: **KBR-54** ([link](https://shelpuk.atlassian.net/browse/KBR-54)).
Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §7, row T-D4. Design:
`TEST_SUITE.md` §3.3.4 (triggers + complements across representative models;
parametrised over transport), §3.3.5 (routing), §7.4 (oracle / wire projections).
Sibling design settlement: appended to §7.4 as "What T-D4 settled".
Requirements:
`.requirements/20260923T084304Z_td4_default_transport_slice/REQUIREMENTS.md`.

## Inter-task contract (T-D5–T-D7 inheritance)

The sibling slices inherit this slice's three exposed contracts. The contracts are
listed here so the inheritors (and a future reviewer) can confirm alignment without
re-reading T-D4's full design record.

| Contract | Shape | Inheritors |
|---|---|---|
| Routing-derivation restriction | Literal comparison on path/query; recorder's authority rewritten in via `urlsplit(transport.recorder.base_url)`; query is `""` when the adapter carries no endpoint query. Recorded in the test module docstring + KBR-54 scope-addition comment. | **KBR-55** (T-D5, curl_cffi / openai_subscription), **KBR-56** (T-D6, botocore / bedrock), **KBR-57** (T-D7, provider-aiohttp / ollama_cloud) |
| Skip-table pattern | A module-level constant `_CORPUS_SKIP_TABLE: dict[str, str]` mapping `entry_id` to a single composed skip message (the unclaimed-delta path or framing rationale, plus the F3 marker or sibling ticket id); the driver consults it before driving and `pytest.skip`s on hit with the message verbatim. Local to the test module, not `tests/exemptions.py`. | KBR-55, KBR-56, KBR-57 (each populates with their own transport's findings) |
| `_SENTINEL_ROUTE_PATH` constant pattern | A constant asserted equal to the adapter's `get_upstream_path(model)` so a future adapter rename fails the test loudly with both values in the message. | KBR-55, KBR-56, KBR-57 (each pins their own adapter's endpoint shape) |

The one-way-inheritance risk: KBR-55 (T-D5) and KBR-56 (T-D6) are already **In
Progress** in Jira as of 2026-09-23. KBR-54's scope-addition comment asks each
inheritor to ack on their own ticket before review lands, so any divergence surfaces
in T-D4's PR conversation rather than as a post-merge surprise.

## What the task does

Drives the bridge end-to-end against the transparency oracle on **every** inbound
Anthropic-Messages corpus entry, using the `custom_openai` default-transport adapter
(Chat Completions upstream — the non-native route, where unannounced body changes are
most likely to hide), with all four oracle obligations active: §3.3.1 totality,
§3.3.2 assertions 1 + 2, §3.3.5 routing.

T-D1 (`test_oracle_driven.py`) proved the same obligations on one synthetic body; T-D4
extends the proof across the golden corpus — the corpus-driven slice the design §3.3.4
calls for and the plan §7 row T-D4 names. T-D5–T-D7 (curl_cffi / botocore /
provider-aiohttp) inherit this slice's three contracts above; T-D9 (full matrix)
extends them across all 20 default adapters.

## Decisions

- **No special small-context profile.** The harness's default `"harness-model"` resolves
  via `get_model_context_tokens` (`src/kitty/providers/model_context.py:362–414`) to the
  `DEFAULT_CONTEXT_TOKENS = 200_000` fallback (`model_context.py:15`), giving a derived
  budget of **800 000 chars** (200k × 4). `compaction_budget_over`'s 2.85 MB body is over
  that budget, so M5 fires on the default profile and the entry passes the oracle with all
  M5 deltas claimed. An F1.c draft proposed a `redirected(...)` + small-context profile to
  force M5 to fire; the design review (2026-09-23) showed the premise was numerically false
  (the reviewer's blocker 1) and the mechanism depended on the priority-2
  `provider_config["context_window"]` override rather than the deterministic,
  product-documented seam. Dropped.
- **Routing authority is the recorder's, not the bridge's.** The bridge and the recording
  transport bind **different** ephemeral ports; using `fixture.base_url` (the bridge's URL)
  produces a `route.host` mismatch on every captured request. The fix is
  `urlsplit(fixture.transport.recorder.base_url)` — the transport's recorder binds the
  port the captured `Host` header carries. Same shape T-D2's `_drive_azure` uses.
- **The routing derivation is the literal KBR-52 shape, restricted to parameter-free
  `base_url`s.** The default aiohttp binding's `provider_config["base_url"]` carries no
  query, so the literal comparison is correct here. The KBR-143 base-URL query-merge
  rule is not exercised; T-D5–T-D7 inherit and may need to extend the derivation when
  their adapters carry endpoint-side queries (Azure, Vertex). The restriction is recorded
  in the test module docstring and on KBR-54 as a Jira scope-addition comment (mirroring
  the KBR-52 pattern of 2026-09-19).
- **The expected path is `/chat/completions` — `custom_openai`'s published endpoint shape.**
  Pinned by `_SENTINEL_ROUTE_PATH` (F6.a) and by a behavioural falsification (F6.b): the
  driven slice's captured request with a sentinel-wrong expected path must raise
  `RoutingMismatchError` naming `route.path` — the T-D2 reroute-with-byte-identical-body
  shape, applied to the L3-driven surface.
- **The skip table is the slice's authored restriction.** T-D1's module docstring
  anticipated that the corpus-driven run would surface register gaps whose rows
  "belong in their own PRs". T-D4 makes the anticipated gap visible in CI: each known
  gap is named with its unclaimed-delta path and its sibling ticket id (or `new
  finding`), and the entry is `pytest.skip`'d with that message. The table is local to
  the test module, not `tests/exemptions.py` — §8.3's registry is platform skips, a
  different subject.
- **Four new findings the empirical pass surfaces:** `context_management` and `metadata`
  drops on the CC adapter from a Messages body (the Codex P23 row covers
  `openai_subscription` only); `messages[N].reasoning_content` residualised by the CC
  reader (KBR-285 widened the CC content classifier but did not touch `reasoning_content`);
  `conversation.turns[2]` dropped on the CC adapter from a Messages body carrying a
  50 000-char `tool_result` (pairing-validation drop, no register row). Each is named on
  KBR-54's scope-addition comment for owner tracking; the fix is out of scope here.
- **A corpus-calibration gap on `compaction_budget_under`.** The entry was calibrated to
  the 2.8 M-char static threshold (per `tests/corpus/README.md`), but the default profile
  resolves to 800 000-char budget. The entry is no longer a clean M5 complement (M5 fires,
  M3 also fires on the boundary tool_result, violating the entry's
  `triggers_absent: tool_result_over_limit`). Recorded as a calibration gap; the fix is
  a corpus-regeneration ticket (sibling, TBD).
- **No pytestmark.** The slice defaults to `l1` per the T-D1 / T-D2 precedent; the
  §3.4 table calls this surface L3, and T-K6 owns `l3` activation. When T-K6 lands,
  `@pytest.mark.l3` is added; until then, the file runs as `l1` and the
  `pytest -m "l1 or l2"` fast gate covers it.
- **T-D5–T-D7 inherit this slice's three contracts** (the table at the top of this file).

## Falsification (plan §1.4 harness rule)

Two falsifications ship with the slice, both running in the suite:

1. **`TestRoutingFalsification`** (F6.b) — drives the smallest **clean** corpus entry
   once through the bridge, takes the captured request, runs the oracle with the expected
   path rewritten to a sentinel wrong path. The oracle must raise `RoutingMismatchError`
   naming `route.path`. The body obligations pass first by ordering (§3.3.5: routing
   runs last so the louder body diagnosis surfaces first), so the routing error is the
   only thing that can fire — the byte-only oracle cannot see a path mismatch, but the
   driven slice with `expected_route` active must, and this case proves it. The "clean
   entry" selection iterates the corpus and picks the first one not in
   `_CORPUS_SKIP_TABLE`, so a future corpus entry rename or skip-table move does not
   silently break the falsification's body-obligations-pass-first ordering.
2. **`TestExpectedRoutePathIsLiteralAndPinned`** (F6.a) — asserts
   `_SENTINEL_ROUTE_PATH == CustomOpenAIAdapter().get_upstream_path("harness-model")`.
   A future adapter rename (or a derivation that drifts to a hardcoded wrong path) fails
   this test loudly with both values in the message.

The skip-table's first row (`plain_turn` with the F3 findings) is also a falsification in
the §1.4 sense: a known defect the oracle **catches and names**, the gap and its marker
recorded, the test green because the catch is the proof. A slice that silently passed on
`plain_turn` would be the §1.4 shape — a green test proving nothing — and is the bug the
skip-with-named-marker pattern rules out. (That is distinct from `TestRoutingFalsification`
above: the routing falsification drives a **clean** entry so the body obligations pass
first; the skip-table's first row is the *body* falsification — the catch on `plain_turn`
*is* the proof — and the two must not be conflated.)

## Status

Implemented (2026-09-23); PR pending on `feat/kbr-54-oracle-default-slice` off
`origin/main`. Design review (system-design-reviewer, 2026-09-23) and code review
(code-reviewer, 2026-09-23) both APPROVED; the design review's four blocker-level
defects and all concerns applied (F1.c premise re-measured against
`get_model_context_tokens`; F1.b seam fixed to the recorder's authority; F1.d's
missing `ProjectionTotalityError` import resolved; F4 skip table rebuilt from the
corrected empirical probe), and the code reviewer's one required fix (dead `_drive_one`
helper) plus all suggestions applied. CI gates green: `ruff check .`, `mypy src/kitty`,
`lint-imports` 5/5 contracts kept, `pytest -m "l1 or l2"` 2603 passed / 10 skipped on
Linux 3.13. Awaiting human review; not merged.

### Implementation notes

- The `_DEFAULT_PROFILE_BUDGET_CHARS = 800_000` literal in the test module is derived
  from `get_model_context_tokens("custom_openai", "harness-model", None)` resolving to
  `DEFAULT_CONTEXT_TOKENS = 200_000` × 4 chars/token; recorded beside the literal so
  the derivation is auditable without an `src/kitty` import (the driver is a judge).
- The `_SENTINEL_ROUTE_PATH` literal is pinned against
  `CustomOpenAIAdapter().get_upstream_path("harness-model")` (the model arg is for
  signature parity; the adapter's base implementation ignores it). A future adapter
  that honours the model arg in the path would surface immediately.
- The skip table's first row is `plain_turn`; it is the §1.4 **body** falsification's
  load-bearing entry — a known gap the oracle catches and names, the test green because
  the catch is the proof (the same shape T-D2's "reroute with byte-identical body"
  follows). `TestRoutingFalsification` is a separate §1.4 case: it drives a **clean**
  entry (`format_example`, the smallest non-skipped one — selection iterates rather than
  hardcoding the id) so the routing assertion's "body obligations pass first" ordering
  has a body that actually passes them.
- Empirical baseline: `.scratch/probe_corpus.py` (corrected version, 2026-09-23). Six
  clean entries pass with `report.deltas == ("envelope.model",)` (the M1
  PROFILE_SETS_MODEL rewrite); `compaction_budget_over` produces 383 deltas, first
  `envelope.model`, tail M5-pruned turn texts. The `>= 200` lower bound in the test
  has margin; the count is documented in the test docstring for traceability.
