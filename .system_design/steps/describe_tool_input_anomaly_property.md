---
id: describe_tool_input_anomaly_property
depends_on: []
---

# `describe_tool_input_anomaly` property test (T-F5)

T-F5 of the test-suite implementation plan (`TEST_SUITE_IMPLEMENTATION_PLAN.md`
§9) / [KBR-74](https://shelpuk.atlassian.net/browse/KBR-74). Replaces the
example-based coverage of the tool-input anomaly detector with one property test
that proves the no-false-positive guarantee across the detector's resolvable
JSON-Schema subset, plus a stand-down-respected property for the composition-
keyword path, plus a falsification control that pins the property's bite
against an always-`None` stub (the §1.4 harness rule).

## Why

The detector (kitty-bridge#33) warns when an upstream returns a `tool_use`
whose `input` contradicts the tool's declared schema. A warning that fires on
ordinary traffic trains operators to ignore it; the detector's whole value is
that *a warning means something*. The example-based suite pins the two finding
rules and the stand-down paths on hand-picked cases. The §6.1 property row for
this unit — *"never reports an anomaly for input that validates against the
declared schema"* — is the *generality* claim those examples do not, and
cannot, prove. Without it, the detector could be edited (say, dropping the
`missing` guard) and the suite would stay green.

The detector deliberately stands down when the schema carries a composition
keyword (`oneOf`, `anyOf`, `allOf`, `not`, `if`, `$ref`), because the real
constraint lives somewhere it does not resolve. The property must respect
that — and the stand-down itself must be pinned, since "returns `None` because
the input happens to be valid" is observationally indistinguishable from
"returns `None` because the detector declines to look". P2 covers that.

## Scope decision: hand-rolled generator, not `hypothesis-jsonschema`

Recorded in `TEST_SUITE.md` §6.1 alongside the property row, per the ticket.
Reasons: the detector resolves a small, fixed subset (`type: "object"`, root
`properties`, root `required`, optional `additionalProperties`); a generator
over exactly that subset produces inputs whose validity is provable by
construction and runs in `~80` lines. `hypothesis-jsonschema` generates
against the full schema language, including the composition keywords the
detector declines — any such draw hits the stand-down and never exercises the
resolve-and-compare half, so the extra coverage is wasted at the price of a
new dev dependency. Same posture as T-F4's local egress strategies
(KBR-73): the T-F1 substrate's own docstring carves this task out
(`tests/harness/transcripts.py` lines 23–35), and the strategies ship
locally with the property.

## Implementation notes

- **Three properties, not one.** P1 is the ticket's claim on the resolvable
  subset. P2 pins the stand-down: for any schema carrying a composition
  keyword, the detector returns `None` even for a violating input — so the
  stand-down is *unconditional*, not the side effect of validity. P3 is the
  §1.4 falsification control: a constructed wrapped payload that the
  detector's own rules must catch (asserted non-`None` with the envelope
  wording), plus a negative control that the P1 predicate is not vacuous
  — expressed by patching `describe_tool_input_anomaly` to return `None`
  always and watching the property fail (the "the scan actually finds
  something" pattern).
- **Generator correctness by construction.** Every (schema, input) pair in P1
  is drawn together: one `st.tuples(schema, input)` style composite emits both
  sides from a single draw, so validity is a property of how the strategies
  are *composed*, not of a separate post-hoc validator. Each strategy's
  docstring names the rule it enforces (root-required ⊆ root-properties,
  present-key-type-match, additionalProperties respected). The shape mirrors
  the T-F1 substrate's "valid body" rule (transcripts.py §"What 'valid' means
  here"), where validity is structural and reported by the substrate, not
  bolted on by the consumer.
- **Composition-keyword path is a `st.sampled_from` over the six keywords**
  from `tool_audit._COMPOSITION_KEYWORDS`, each composited with a non-empty
  resolvable reading so the violating input is meaningful (e.g. a required
  field absent) — not a vacuous input that the detector would have ignored
  anyway.
- **Layer.** File lives under `tests/bridge/`; the path default assigns `l1`
  (same posture as `test_compaction_properties.py`, `test_pairing_truncation_properties.py`),
  so no `pytestmark` is added. The repo-wide l1 gate picks it up without
  ceremony.
- **Hypothesis conventions.** `@settings(max_examples=200)` per property
  (matching T-F2's settings), CI profile from `tests/conftest.py` (derandomise
  + no deadline + suppress `HealthCheck.too_slow`), the local developer keeps
  the default randomized profile plus the example DB — neither is touched.
- **No product-code changes.** `src/kitty` is not in the diff.

## Verification

- `pytest -m l1 tests/bridge/test_tool_audit_properties.py -q` green
  (18 tests: P1a/P1b/P1c × property + required-only example, P2 × 6
  keywords, P4 × 4 shapes, P5 × 3 arms, P3 control).
- Mutation spot-checks (both reverted; failure signatures recorded in the
  PR):
  - *Spot-check A* — drop the `not missing` precondition (fire on
    `unexpected` alone): P1a and P1b fail, plus the required-only example
    (whose `ghost` key is exactly the shape the mutation misreports).
    This is why the P1b discriminator arm is non-negotiable.
  - *Spot-check B* — remove the composition-keyword stand-down: P2 fails
    across all six keyword parametrisations.
- `ruff check` clean (B905 `zip(strict=True)` — sizes are drawn equal, so
  the strict form is a free invariant); `ruff format` applied;
  `mypy src/kitty` clean; `lint-imports` clean (5 kept, 0 broken).
- `scripts/regenerate_step_index.py` does not exist in the repo (recorded
  in memory since the first step file, KBR-67/PR #197); the step was added
  without machine-validated `depends_on`.

## Status

Implemented in `feat/kbr-74-t-f5-tool-input-anomaly-property`. PR open;
watching reviewer + CI, then maintainer merge.
