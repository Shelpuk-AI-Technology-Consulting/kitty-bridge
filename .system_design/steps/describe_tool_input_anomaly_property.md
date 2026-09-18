---
id: describe_tool_input_anomaly_property
depends_on: [hypothesis_and_transcript_strategies]
---

# `describe_tool_input_anomaly` property test (T-F5)

T-F5 of the test-suite implementation plan (`TEST_SUITE_IMPLEMENTATION_PLAN.md`
§9) / [KBR-74](https://shelpuk.atlassian.net/browse/KBR-74). Replaces the
example-based coverage of the tool-input anomaly detector with one property test
that proves the no-false-positive guarantee across the detector's resolvable
JSON-Schema subset, plus a stand-down-respected property for the composition-
keyword path, plus a §1.4 falsification control that pins the property's bite
against a **report-happy** regression (the "naive reporter" — patches the
detector with "report on any undeclared root key" and watches P1's exact
assertion expression raise). The miss-class is intentionally omitted: the
example suite already covers positive findings.

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

- **Four properties plus a §1.4 falsification control** (the auto-reviewer's
  warning on an earlier draft caught a stale paragraph here: the shipped
  P3 does not duplicate the example-suite positive findings; see "Status"
  at the bottom for the resolution). P1 is the ticket's no-false-positive
  claim on the resolvable subset, with three arms — broad (P1a),
  discriminator (P1b, always emits an undeclared permitted key so the
  "drop `not missing`" mutation is visible), strict (P1c). P2 pins the
  composition-keyword stand-down with a violating input so the `None` is
  provably the stand-down's, not validity's. P4 is the both-conditions
  rule example class — kitty-bridge#33's "must not later be relaxed into
  an or" discipline, carried into this file. P5 is the generator
  self-conformance audit against a hand-rolled draft-07 validator. P3 is
  the §1.4 falsification control: a monkeypatched **report-happy**
  ("report on any undeclared root key") detector makes P1's exact
  assertion expression raise against a valid input that carries one extra
  key — proving the no-false-positive predicate is live, not vacuous. The
  miss class is **deliberately omitted**: it is already
  pinned by `tests/bridge/test_tool_use_audit.py`'s positive findings, and
  per the test-development skill's Part A.10 ("Do not add an assertion
  another test already covers") the property file does not duplicate it.
- **Generator correctness by composition, audited separately.** Every
  (schema, input) pair is drawn together in `_valid_pair` — one
  ``@st.composite`` draw fixes the property names, their types, the
  required set and the ``additionalProperties`` posture, then builds the
  input from those same names. Validity is a property of how the
  strategies are *composed*. P5 is the *audit* on that construction, kept
  out of the property bodies: inlining it would re-derive the generator
  inside the property and mask a shared bug. Same posture as
  `tests/harness/test_transcripts.py` judging the T-F1 substrate's own
  conformance.
- **Draft-07 subtyping rules enforced at the value level.** Booleans are
  not integers or numbers in JSON Schema (Python's ``bool`` is an ``int``
  subclass — the trap); NaN/Infinity are not valid JSON at all (the same
  JSON-strict rule the T-F1 substrate enforces via ``allow_nan=False``);
  the validator's `_value_matches_declared_type` pins both.
- **Composition-keyword path is a `parametrize` over the six keywords**
  from `tool_audit._COMPOSITION_KEYWORDS` (imported, not hard-coded — a
  keyword added to the source list auto-joins the property), each paired
  with a non-empty resolvable reading and an input that violates the
  resolvable reading (an unexpected root key + a missing required key) so
  the `None` is provably the stand-down's, not validity's.
- **Layer.** File lives under `tests/bridge/`; the path default assigns `l1`
  (same posture as `test_compaction_properties.py`, `test_pairing_truncation_properties.py`),
  so no `pytestmark` is added. The repo-wide l1 gate picks it up without
  ceremony.
- **Hypothesis conventions.** `@settings(max_examples=200)` per property
  (matching T-F2's settings), CI profile from `tests/conftest.py` (derandomise
  + no deadline + suppress `HealthCheck.too_slow`), the local developer keeps
  the default randomized profile plus the example DB — neither is touched.
- **Module-attribute call path so P3 can monkey-patch.** The test file
  imports `from kitty.bridge import tool_audit` and calls
  `tool_audit.describe_tool_input_anomaly(...)` — never a direct function
  reference — so `monkeypatch.setattr(tool_audit, "describe_tool_input_anomaly", ...)`
  takes effect at every call site. Same precedent in
  `tests/bridge/test_tool_use_auditor.py`.
- **No product-code changes.** `src/kitty` is not in the diff.

## Verification

- `pytest -m l1 tests/bridge/test_tool_audit_properties.py -q` green
  (18 tests: P1a/P1b/P1c × property + required-only example = 4; P2 × 6
  keywords = 6; P4 × 4 shapes = 4; P5 × 3 arms = 3; P3 control = 1).
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
- Full fast-gate subset on Linux (`pytest -m "not agent_smoke and not
  agent_live and not eval and not load"`): **8038 passed, 13 skipped, 37
  deselected** in 26 min (2026-09-17). All six CI test-matrix legs pass
  (3.10/3.11/3.12 ubuntu, 3.13 ubuntu, 3.12 macos, 3.12 windows), plus
  review, CodeQL, Analyze (python/actions), review-scripts,
  review_replies, update-metadata, ci-required.
- `scripts/regenerate_step_index.py` did not exist when this step shipped
  (recorded in memory since the first step file, KBR-67/PR #197), so the
  step was added without machine-validated `depends_on`. KBR-278 landed
  the validator and backfilled the `depends_on` to
  `[hypothesis_and_transcript_strategies]`, converting the body-text
  rationale into an actually-honest YAML edge.

## Status

Implemented in `feat/kbr-74-t-f5-tool-input-anomaly-property` (PR #213).
**Auto-reviewer's round-1 and round-2 warnings addressed in this PR** —
both flagged the same root cause: a *concept iteration* changed P3's
mechanism (from a miss-class control to a report-happy-regression
control), but the descriptive artifacts were not all swept in the same
pass. The test file's docstring, REQUIREMENTS.md, and the PR body were
updated; the step file lagged. Two commits (`badcb52`, `c3059a3`)
corrected the step file's Implementation notes, and a third round-2 fix
corrected the opening summary paragraph (which still described the
pre-iteration design). The miss class is omitted in
the shipped P3 because the example suite already covers positive
findings, and per the test-development skill's Part A.10 ("Do not add
an assertion another test already covers") the property file does not
duplicate it. PR open; CI green; awaiting human reviewer + maintainer
merge.
