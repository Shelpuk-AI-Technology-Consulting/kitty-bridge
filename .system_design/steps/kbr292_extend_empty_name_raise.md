---
id: kbr292_extend_empty_name_raise
depends_on: [KBR-281]
---

# KBR-292 — Bedrock Converse request, Responses request, Responses reply: extend KBR-281's raise to the empty tool-call name

Plan: `REQUIREMENTS.md` at
`.requirements/20260920T215802Z_extend_empty_tool_call_name_raise/`.

## What

Close the silent-`""` defect in the three readers KBR-281 left open. The design
reviewer chose **option (a)** — raise, mirroring KBR-281's settled rule — over
option (b) (residualise the empty string at the name's path). All three readers
raise `UnreadableBodyError` on absent / empty / non-string tool-call names:

- Bedrock Converse request `_read_tool_use`: the residualise-and-project branch
  is replaced by a per-module `_require_tool_call_name(name, path)` helper
  (Gemini-style signature), and its docstring drops the sentinel-`""`-name
  language that no longer holds.
- Responses request `_read_function_call`: the residualise branch and its
  now-false comment are replaced by the module's `_require_tool_call_name`;
  the `Raises:` section is added to the method docstring.
- Responses reply `_read_output_item` `function_call` branch: the
  non-string-only inline check routes through the same helper, so `""` raises
  too and the message says "non-empty string name".

The helper seam becomes: Ollama, Gemini, Bedrock Converse, Responses (both
directions) via per-module `_require_tool_call_name`; Chat Completions and
Anthropic inline — `git grep _require_tool_call_name tests/harness/` lists four
helper modules.

Declarations keep the residualise posture deliberately, with a per-branch
split on the projection: Gemini `FunctionDeclaration` and Responses
`FunctionTool` project `ToolDecl(name="")` from the residualised leaf, while
Converse `toolSpec` omits the declaration. All three admit `""` silently —
recorded in §7.4.2 row 2's Because cell as a sibling gap, not settled here.

## Why option (a) and not (b)

The losslessness argument (`contract.py:935-941`) does not distinguish `""`
from absent — both are "no usable name", so KBR-281's rule extends mechanically.
Under (b) the Responses reply reader would raise on absent/non-string but
residualise `""` — one field, two outcomes, no recorded reason. And the oracle
exists to make readers agree: (a) ends with seven tool-call readers on one
posture, (b) with the reply reader an outlier in all three shapes. KBR-281's
review had also recorded that the "Bedrock/Responses visible residual"
precedent option (b) would lean on did not, in fact, exist.

## Verification

- New `TestNameRequired` classes in `tests/harness/test_reader_bedrock_converse.py`,
  `tests/harness/test_reader_responses.py`, `tests/harness/test_reader_responses_reply.py`:
  absent / empty / non-string raise + published-example-shaped control, each
  watched failing (or watched passing as a pin) before the fix, empty case
  failing as a silent clean projection.
- Pin-listed existing-test dispositions (KBR-281's AC-7 discipline):
  `test_reader_bedrock_converse.py::test_a_non_string_tool_use_name_residualises`
  deleted (coverage moved to `TestNameRequired`); `test_reader_responses.py::`
  `test_a_wrongly_typed_call_id_or_name_residualises_rather_than_being_coerced`
  rewritten to its `call_id`-only form.
- Docs: `TEST_SUITE.md` §7.4.2 rule 7 row 2 (Outcome column and Because cell)
  records the seven-reader settlement and the four-helper seam;
  `contract.py`'s `decode_arguments` docstring bumps "four strict readers" to
  seven and scopes the residualise exception to Gemini declarations.
- `pytest tests/harness/test_reader_bedrock_converse.py
  tests/harness/test_reader_responses.py tests/harness/test_reader_responses_reply.py`
  green; full suite, `ruff check`, `lint-imports`, `mypy src/kitty` green on
  Linux.

## Implementation notes

(design review: option (a) chosen; all findings applied to REQUIREMENTS.md —
line-range correction, declaration-`""` scope note, explicit test dispositions,
stale-comment and docstring items. Serena's language server was unavailable in
this session's worktree; investigation used built-in search tools instead.)
