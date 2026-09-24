---
id: kbr59_t_d10_response_direction_comparison
depends_on: [KBR-39, KBR-51, KBR-307, KBR-194, KBR-195, KBR-257, KBR-267, KBR-311, KBR-312]
---

# KBR-59 — T-D10 Response-direction comparison

Jira: **KBR-59** ([link](https://shelpuk.atlassian.net/browse/KBR-59)).
Design: `TEST_SUITE.md` §3.3.1 (last paragraph — "Response translation is
tested separately"), §3.3.2, §7.4. Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md`
§7 (T-D10 row). Requirements: `.requirements/20260924T084657Z_kbr59_response_direction_comparison/REQUIREMENTS.md`.

## What the task does

Lands the response-direction half of the I1 transparency oracle: a
sibling entry point `assert_no_unclaimed_reply_mutation` in
`tests/harness/oracle.py` that compares a `CapturedReply` pair
projected through the existing five `ReplyProjection` readers,
asserting every concrete delta is claimed by a register row whose
trigger is in `triggers_met`. Lands three new register rows
**M27/M28/M29** (the first free ids after KBR-195's M21/M22/M23)
that today trip the oracle on deliberate, documented mutations
that no row claims — the response-direction Gemini `functionCall.id`
echo (KBR-257), the Ollama canonical tool-calling `reply.stop_reason`
delta, and the rare Thinking+ToolUse part-ordering delta (KBR-267).

## Decisions

- **Diff parameterisation, not unification, not fork.** The existing
  `_structural_diff` is hard-typed on `Request`. `_run_assertions` is
  generalised to take a positional `diff` callable; the request
  oracle's call site (`oracle.py:466`) becomes `_run_assertions(
  _structural_diff, ..., provider_key=provider_key)`, one
  non-semantic line. The reply oracle passes `_structural_reply_diff`.
  _claim_matching / _conditional_violations stay untouched (they
  are already path-agnostic). Recorded as decision (b) in the spec.

- **M27/M28/M29, not M21/M22/M23.** KBR-194/195 already own
  `register.py:1034/1048/1063`; reusing those ids fails
  `test_row_ids_are_unique`,
  `test_no_two_rows_are_indistinguishable`, and
  `register_disagreements`. Reviewer finding B1.

- **Trigger rename.** `CAPTURED_TOOL_CALL_ID_PRESENT`, not
  `GEMINI_INBOUND_ID_PRESENT` — the trigger reads the *captured* CC
  reply, not the inbound Gemini request, so the old name was
  direction-misleading. Reviewer finding C4.

- **`ArrangingBy.RESPONSE` on every new trigger.** Required by
  `test_every_non_always_trigger_has_an_arranged_by` and
  `test_classification_matches_the_specified_table` (the F1 spec).
  Reviewer finding C1.

- **Narrow scope on every new row.** M27 → `("google_aistudio",
  "vertex")`; M28/M29 → `("ollama_cloud",)`. Per §3.2.2 ("scope =
  reachability of site"), the bridge sites that perform the mutation
  live on one provider; broad `ALL_PROVIDERS` overstates the
  reachable set. Reviewer finding C3.

- **M29 trigger is wire-level post-C5.** Ollama
  `message.thinking` non-empty AND `message.tool_calls` non-empty.
  The complement is a corpus entry the corpus author can write.

- **Reply-registry guard is a membership-superset check**
  (`_ASSERTABLE_REPLY_FORMATS <= _REPLY_PROJECTIONS`). Future reply
  readers (KBR-312 = Bedrock) grow the constant in one place.
  Reviewer suggestion S1.

- **Reader-side registration pinned to the five reader modules.**
  `tests/harness/reader_<format>.py` calls
  `_register_reply_projection(cls())` at import time, mirroring
  the request-side `_register_projection(...)` calls at
  `oracle.py:1234`. Reviewer suggestion S5.

- **`ReplyOracleReport` is a separate frozen dataclass with
  `__hash__ = None` explicit.** Two dataclasses > one discriminated
  one: the call-site reviewer at the type level is told what it
  compares (`Reply`, not `Request`). S2/S3.

- **SSE stream reassembly and Bedrock Converse reply reader are
  siblings.** Filed as **KBR-311** (High, blocked-by KBR-59) and
  **KBR-312** (Medium, blocked-by KBR-59) respectively; the reply
  oracle's contract specifies complete replies, so the comparison
  drives the non-streaming path first; KBR-311 + KBR-312 grow the
  corpus into L3 coverage.

## Implementation notes

_(filled in during the PR — red evidence, planted-removal
discoveries, gate results, review round notes; see KBR-307's
"Implementation notes" block for the shape.)_

### Landed 2026-09-24 — branch `feat/kbr-59-response-direction-comparison` off `origin/main`

- **Reply-side entry point**: `assert_no_unclaimed_reply_mutation`
  in `tests/harness/oracle.py` mirrors the request oracle's contract
  minus §4.3 C2 (the reply body is the upstream's output, never a
  forwarded pass-through) and §3.3.5 (the reply traverses the same
  connection the request did; routing is a request-side concern).
  The entry point does **not** catch `verify_total`'s exceptions —
  the diagnostic type is preserved unmodified.

- **Diff parameterisation** (reviewer B2): `_run_assertions` gains a
  keyword-only ``diff: Callable[[Any, Any], tuple[str, ...]] | None``
  defaulting to ``None``, resolved at call time to
  :func:`_structural_diff` for backward compatibility. The new
  :func:`_structural_reply_diff` ships as a sibling, hard-typed on
  :class:`Reply`. The 25 existing call sites in ``test_oracle.py``
  keep working unchanged — the regression test
  ``test_run_assertions_keeps_default_diff_argument`` pins this.

- **Reply registry**: `_ASSERTABLE_REPLY_FORMATS`,
  `_REPLY_PROJECTIONS`, `_register_reply_projection`,
  `_reply_reader_for`, `_REPLY_REGISTRY_GUARD` (a *superset* check
  per reviewer S1). Five readers register centrally, mirroring the
  request-side block at the bottom of `oracle.py`. Bedrock Converse
  raises `RuntimeError` at call time (KBR-312 sibling).

- **`c.reply_part_path` extension** (S7): the helper gains an
  optional `field_name` parameter so M27's anchor reads
  ``reply.parts[*].id`` from the helper rather than string concat.

- **Three register rows** (`register.py`):
  - **M27** — Gemini response-direction `functionCall.id` echo
    (KBR-257). Trigger `CAPTURED_TOOL_CALL_ID_PRESENT`,
    `ArrangingBy.RESPONSE`, scope `("google_aistudio", "vertex")`,
    anchored at `reply.parts[*].id`.
  - **M28** — Ollama canonical tool-calling `reply.stop_reason`
    delta. Trigger `OLLAMA_CANONICAL_TOOL_CALLING_REPLY`,
    `ArrangingBy.RESPONSE`, scope `("ollama_cloud",)`, anchored at
    `REPLY_STOP_REASON`.
  - **M29** — Ollama vs CC part ordering on rare
    Thinking+ToolUse replies (KBR-267 comment 3.2). Trigger
    `REPLY_CONTAINS_THINKING_AND_TOOL_USE`, `ArrangingBy.RESPONSE`,
    scope `("ollama_cloud",)`, anchored at `reply.parts[*]`
    (broad, per §7.4.1).

- **Three new `Trigger` members**, each with a wire-level docstring.
  The `ArrangingBy.RESPONSE` enumeration at `register.py:138-140`
  now lists "M6, M8, M9, M12, M17, M27, M28, M29".

- **Companion sites in `test_register.py`** (C11/C12): the F1
  dict at line 650 gains the three new triggers; the
  `len(r.REGISTER) == 80` count pin becomes `83`; the docstring
  above (lines 132-156) gains the +3 narrative. The M-row count
  in `test_register_agreement.py:303` becomes `30` (was 27).

- **Two `_SHAPES` entries** in `test_register.py`:
  `(reply.parts[*].id, reply.parts[2].id)` (M27's anchor),
  `(reply.parts[*], reply.parts[2])` (M29's anchor).

- **Sibling tickets filed**: **KBR-311** (SSE stream reassembly,
  High, `Blocks` KBR-59) and **KBR-312** (Bedrock Converse reply
  reader, Medium, `Blocks` KBR-59). Both linked in Jira.

- **Gates**: `ruff check .` — one pre-existing error in
  `.scratch/update_pr260_body.py` (out of scope, scratch file).
  `mypy src` — clean. `lint-imports` — clean.
  `pytest tests/harness/` — 2681 passed, 10 skipped,
  3 deselected, 5 warnings in 193.76s, exit 0.

