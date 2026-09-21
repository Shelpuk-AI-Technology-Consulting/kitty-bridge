---
id: kbr295_tighten_declaration_empty_name_raise
depends_on: [kbr292_extend_empty_name_raise]
---

# KBR-295 — Tighten the empty-`""` admission on tool-*declaration* branches (Gemini `FunctionDeclaration`, Responses `FunctionTool`, Converse `toolSpec`) to raise

Plan: `REQUIREMENTS.md` at
`.requirements/20260921T114216Z_tighten_declaration_empty_name_raise/`.

## What

Close the deliberate sibling KBR-292 recorded: the seven tool-*invocation*
readers raise uniformly (KBR-281 + KBR-292), but declaration sites admitted
`""` silently. After this change, the **seven invocation readers** and the
**six declaration branches** raise `UnreadableBodyError` on absent / empty
/ null / non-string: Gemini `FunctionDeclaration`, Responses
`FunctionTool`, Converse `toolSpec`, Chat Completions `function.name`,
Anthropic `tools[].name`, Ollama `tools[].function.name`.

The scope widened from the ticket's original three readers: the design
review (system-design-reviewer, 2026-09-21) found the identical silent-`""`
shape on the Chat Completions, Anthropic, and Ollama declaration branches
(type-only raise in CC/Anthropic so `""` slipped through; no raise at all
in Ollama, which projected `ToolDecl(name="")` via `_typed_leaf(...) or ""`).
The ticket's acceptance sketch said "tighten under the design verdict";
the verdict is six declaration branches, so the oracle does not disagree
across formats.

- `reader_gemini._read_required_name` — replaced the
  residualise-and-`""`-project body with a thin aliased-view wrapper over
  `_require_tool_call_name` (one spelling of the rule per module);
  `residual` dropped from the signature (raise path never writes it).
- `reader_responses._read_tool` `function` branch — replaced the
  residualise-and-`""`-project branch with the module's
  `_require_tool_call_name`; `Raises:` section added to the method
  docstring (KBR-292 precedent).
- `reader_bedrock_converse._read_tool_specification` — replaced the
  residualise-and-`return None` name branch with the module's
  `_require_tool_call_name`; the function still returns `None` from
  `inputSchema` and other field failures (asymmetry pinned by
  `test_input_schema_failure_still_omits_the_declaration`).
- `reader_chat_completions._read_tools` — `or not name` added to the
  type-only raise (mirror of the KBR-281 invocation-site spelling).
- `reader_anthropic_messages._read_tools` — same `or not name` extension.
- `reader_ollama._read_tools` — `_typed_leaf(...) or ""` replaced by the
  module's `_require_tool_call_name`.

Helper seam unchanged (still four modules via per-module
`_require_tool_call_name`; Chat Completions and Anthropic raise inline in
both directions, per §7.4.1's within-module anti-drift rule).

## Why

The KBR-292 design review explicitly recorded the three Gemini / Responses /
Converse declaration branches as a deliberate sibling. The widened CC /
Anthropic / Ollama arms surfaced in the KBR-295 design review's
sibling-arms sweep — an oracle settled on three of six declaration branches
would disagree across formats, the exact coordination failure this row's
general prescription exists to prevent.

The option chosen is **(a) raise**, mirroring KBR-281 + KBR-292.
Format-independent losslessness argument; uniformity is safer than
per-branch asymmetry.

## Implementation notes

(2026-09-21) TDD: thirteen new `TestDeclarationNameRequired` cases first
(Gemini 4 / Responses 4 / Converse 5, the fifth pinning the Converse
name-vs-inputSchema asymmetry), watched failing on `origin/main`
("DID NOT RAISE"), then the three reader sites tightened; then the widened
scope (CC / Anthropic / Ollama, twelve more cases; CC and Anthropic red only
on the empty shape — their absent/non-string postures already raised —
Ollama red on all three). Pin-listed dispositions per AC-11: three old
pin tests deleted (Gemini, Responses, Converse), the
`functionDeclaration.name` row removed from Gemini's
`TestEveryOptionalLeafFailsClosed` table (the table's scope no longer
includes `name`). Three helper docstrings updated; three reader doc
channels updated (`_read_function_declarations` Raises, `_read_tool`
Raises added, `_read_tool_specification` Returns narrowed + Raises
added); three invocation `TestNameRequired` docstrings refreshed
(they said the declaration sibling was deliberate). `TEST_SUITE.md`
§7.4.2 row 2 records the ten-site settlement; `contract.decode_arguments`
names the six declaration branches as siblings of the seven invocation
readers.