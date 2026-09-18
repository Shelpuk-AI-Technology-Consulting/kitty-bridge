---
id: kbr279_ollama_request_reader_strict_function_name
depends_on: [KBR-267]
---

# KBR-279 — Ollama request reader: strict `function.name` check

Plan: `REQUIREMENTS.md` at
`.requirements/20260917T185118Z_kbr279_ollama_request_reader_strict_function_name/REQUIREMENTS.md`.
Jira: **KBR-279** ([link](https://shelpuk.atlassian.net/browse/KBR-279)).
Design: `TEST_SUITE.md` §3.3.1b (losslessness argument for `name`),
§7.4.1 (within-module anti-drift — one rule, one place).
Predecessor: **KBR-267** ([link](https://shelpuk.atlassian.net/browse/KBR-267),
PR #215, merged 2026-09-17) added the strict check on the reply side and
filed this as its future-work follow-up.

## What the task does

Closes the within-module drift between the request and reply directions
of the Ollama reader on the "name must be a non-empty string" rule for
`tool_calls[*].function.name`. The request reader currently routes
`name` through `_typed_leaf` and falls back to `""` for absent /
wrongly-typed leaves (`reader_ollama.py:1057`), violating
`contract.decode_arguments`'s losslessness argument at
`contract.py:935-941`. The reply projection (KBR-267) already raises
`UnreadableBodyError` on the same condition.

The change:

1. Adds a module-level helper `_require_tool_call_name` to
   `tests/harness/reader_ollama.py`.
2. Wires the request `_read_tool_calls` and the reply `_read_message`
   pre-check to the same helper.
3. Adds `TestNameRequired` to `tests/harness/test_reader_ollama.py` with
   three falsifications + a control, mirroring
   `tests/harness/test_reader_ollama_reply.py::TestNameRequired`.
4. Updates `TEST_SUITE.md` §7.4.2 rule 7 row 2's tool-name outcome: the
   row still prescribes `ToolUse(name="")`, which is dead letter across
   shipped readers (CC and Gemini raise on a non-string name; the Ollama
   reply direction has raised since KBR-267). The row's `fileData.fileUri`
   half is unaffected.

The strict check is a no-op on real wire traffic — every published
Ollama `/api/chat` request and reply example in the harness carries a
non-empty `function.name` — so the change closes a malformed-input
divergence, not a real-traffic one.

## Implementation notes

### Implementation-time finding — row 2's Outcome is NOT dead letter

The design-review pass (two rounds) proposed changing row 2's Outcome to
**raise**, on the grounds that CC, Gemini and the Ollama reply direction
already raise. Verification during implementation found the missing
reader: **Responses** (`reader_responses.py:1040-1052`) implements the
row verbatim — residualises absent/null/non-string `name` at its own
path and projects `ToolUse(name="")` — with a comment citing the same
losslessness argument the row cites. Anthropic raises on a missing
tool_use name (`reader_anthropic_messages.py:750-751`); CC and Gemini
raise only on a **non-string** name (both accept `""`).

The landed row edit therefore keeps the Outcome cell (residualise +
project) and extends the Because cell to record the deliberate stricter
raises per reader, with line references. The general prescription stays
the rule; the raise outcome stays the recorded deviation.

### What landed (2026-09-17)

1. `_require_tool_call_name(function, fn_path) -> str` in
   `tests/harness/reader_ollama.py` — the shared helper; docstring
   carries the two invariants from the plan (ordering; cross-reader
   residual delta for T-D8).
2. Request `_read_tool_calls`: `name = _typed_leaf(...) or ""` replaced
   by the helper call; `Raises:` docstring updated.
3. Reply `_read_message` pre-check: inline per-entry name check replaced
   by the helper call (guards for non-Mapping entry / non-Mapping
   function preserved, so the ordering invariant holds); class docstring
   and `_read_message` docstring updated.
4. `TestNameRequired` in `tests/harness/test_reader_ollama.py` —
   published multi-turn body as class constant, json round-trip deep
   copy per test, three falsifications + control. TDD red first
   (3 × "DID NOT RAISE"), green after (1).
5. `TEST_SUITE.md` §7.4.2 rule 7 row 2 Because cell extended (see the
   finding above).

Gates: `ruff` ✓ · `lint-imports` ✓ (5 kept, 0 broken) · `mypy src/kitty`
✓ (94 files) · full suite — see Jira/PR for the final count.

## Cross-references

* `tests/harness/reader_ollama.py` — `_read_tool_calls` (~1057), `_read_message` (~413-446 reply pre-check), `_typed_leaf` (1298).
* `tests/harness/test_reader_ollama.py` — `TestToolCalls` (488), `TestPublishedExamples::test_with_history_with_tools` (214-275).
* `tests/harness/test_reader_ollama_reply.py` — `TestNameRequired` (501).
* `tests/harness/contract.py:935-941` — losslessness argument.
* `.system_design/TEST_SUITE.md` §3.3.1b, §7.4.1.

## Status

Implemented and review-approved (2026-09-17). system-design-reviewer ran
two rounds on REQUIREMENTS.md (8 findings folded in) plus one
confirmation round; code-reviewer returned APPROVE after verification of
all eight acceptance criteria and the full suite (8117 passed, 14
skipped, 0 failed on Linux, 26m42s). Implementation-time correction:
the design-reviewer-pass proposed changing §7.4.2 row 2's Outcome to
"raise" on the grounds that the general residualise+project prescription
was dead letter; verification found the Responses reader implements it
verbatim (`reader_responses.py:1044-1047`), so the row's Outcome was
preserved and its Because cell was extended to record the deliberate
stricter raises per reader.