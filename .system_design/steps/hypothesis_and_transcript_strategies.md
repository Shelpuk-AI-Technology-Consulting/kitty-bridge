---
id: hypothesis_and_transcript_strategies
depends_on: []
---

# T-F1 — `hypothesis` + shared transcript strategies

Plan row: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §6.1 **T-F1**. Design:
`TEST_SUITE.md` §6.1, the §6.1 property row. Jira: **KBR-70**.

## Why

The §6.1 property list (compaction, pairing/truncation, egress,
anomaly, translator) needs property-based tests running against pure
logic. Property tests for the *transcript-shaped* targets compose
scenarios from valid request bodies, and need to vary far enough
through the space that one example test could not cover. This module
ships the reusable substrate: whole-request, single-message,
single-block, and tool-definition strategies for Anthropic Messages,
Chat Completions, and OpenAI Responses, plus a per-format `problems()`
reporter so a generated body can be handed back and asserted
well-formed.

## Implementation notes

- **`tests/harness/transcripts.py`** (the substrate). Composable
  Hypothesis strategies: `messages_request`, `cc_request`, the
  Responses twin, plus the per-block and per-tool-definition
  primitives and the `*_problems()` reporters that state what "valid"
  means per format. The Responses strategies were added with T-F3
  (KBR-72) to support the L1 properties for the Responses twins of
  pairing and truncation; they ship in the same module because the
  wire rule mirrors the CC / native pair and the per-format
  reporters share their scaffolding.
- **CI Hypothesis profile in `tests/conftest.py`**. Registered and
  loaded when `CI` is set: `derandomize=True` (a CI failure replays
  deterministically), `deadline=None` (the slow Windows/macOS
  fast-gate legs would flake otherwise), `suppress_health_check=
  [HealthCheck.too_slow]`, `database=None` (so a developer's local
  failure does not replay on a runner). The registration is
  session-scoped rather than per-module so every property test gets
  the same treatment deterministically; the local developer keeps
  hypothesis's default randomized profile plus the on-disk example
  database when `CI` is unset.
- **The carve-out is recorded in the module's own docstring**
  (`tests/harness/transcripts.py` lines 23–35): T-F4's egress
  properties (`should_bypass`, `parse_proxy_url`, `EgressConfig`)
  and T-F5's `describe_tool_input_anomaly` property do *not*
  consume this module — they need address / hostname / proxy-URL
  strategies (T-F4) and JSON-schema-conforming input strategies
  (T-F5). Those tasks ship their own local strategies; pre-empting
  their design here would couple unrelated work to a substrate that
  does not exist when they start.
- **Consumers.** `tests/bridge/test_compaction_properties.py` (T-F2),
  `tests/bridge/test_pairing_truncation_properties.py` (T-F3),
  `tests/bridge/test_messages_translator_property.py` (T-F6), plus
  the substrate's own conformance at
  `tests/harness/test_transcripts.py` — which judges the substrate's
  generators the way `test_corpus_thresholds.py` judges the corpus
  builders: "the strategies produce bodies that the per-format
  reporters accept."

## Verification

- `pytest tests/harness/test_transcripts.py -q` green.
- `pytest -m l1 tests/bridge/test_compaction_properties.py tests/bridge/test_pairing_truncation_properties.py tests/bridge/test_messages_translator_property.py -q`
  green.
- `ruff check` clean; `mypy src/kitty` clean.
- Full fast-gate subset (`pytest -m "not agent_smoke and not agent_live and not eval and not load"`) green across the six test-matrix legs.

## Status

Delivered in `feat/kbr-70-t-f1-hypothesis-strategies` (PR #159, merged
on `origin/main`). KBR-278 adds this step file retroactively so that
the step-index validator it ships has a graph node for the dependency
the KBR-74 step file (`describe_tool_input_anomaly_property.md`,
PR #213) declares in its body and now declares in its YAML —
`depends_on: [hypothesis_and_transcript_strategies]`.
