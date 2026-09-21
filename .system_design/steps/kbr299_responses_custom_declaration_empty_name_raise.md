---
id: kbr299_responses_custom_declaration_empty_name_raise
depends_on: [kbr295_tighten_declaration_empty_name_raise]
---

# KBR-299 — Responses reader's custom / built-in / MCP declaration branch: tighten the empty-`""` `name` admission to raise

Plan: `REQUIREMENTS.md` at
`.requirements/20260922T094500Z_responses_custom_declaration_empty_name_raise/`.

## What

Close the last silent-`""` site in the Responses tool-name rule. After
this change, the seven invocation readers, the six declaration branches,
and the three sub-branches of `ResponsesProjection._read_tool` (custom,
built-in, `mcp`) all raise on absent / empty / non-string — except the
three sub-branches, which **deliberately keep** the kind-derived-label
posture for absent / null / non-string `name` and raise only on `""`. The
narrower posture is pinned by six control tests (absent + non-string
per shape) so a future widening of the guard fails loudly.

- `reader_responses._read_tool` non-function fallback (custom / built-in
  / MCP): one inline check before the kind dispatch — `isinstance(name, str)
  and not name` — raising `UnreadableBodyError` with the
  `c.residual_key(path, "name") must be a non-empty string name`
  spelling (same as `_require_tool_call_name`'s helper). Absent / null /
  non-string `name` keeps `str(kind)` (custom / built-in) or
  `_mcp_tool_name(label)` / `"mcp"` (mcp).
- The `function` branch (closed by KBR-295) is untouched; helper seam
  unchanged — the module now carries two raise spellings: the helper's
  three-shape strict check on the function branch, plus this inline
  empty-only check on the fallback. That asymmetry is deliberate and
  documented in the helper's docstring (the KBR-299 finding-4 fix).

The **recorded why** for the permissive absent posture (per the
design-discipline rule, taken from the ticket's own rationale): the
kind-derived label is a *wire-derived identity* — for built-ins, absent
`name` **is** the published shape (built-in declarations carry no `name`
field at all), so raising on absent would reject legal wire traffic. For
`mcp`, the identity derives from `server_label`. For `custom`, the schema
makes `name` required, but this ticket tightens only the `""` shape and
leaves the absent posture for a future settlement rather than widening
the blast radius. An empty string is different in kind — it is a wire
value that *names nothing*.

## Why

KBR-281 + KBR-292 settled the seven invocation readers on raise;
KBR-295 settled the six declaration branches on the same rule. The
sibling-arms sweep in KBR-295 found three more silent sites on
declaration branches outside Responses (Chat Completions, Anthropic,
Ollama) and closed them in the same pass — the design precedent is
format-uniform disagreement prevention, not per-branch asymmetry. The
`_read_tool` non-function fallback (custom / built-in / MCP) was the
last three silent sites KBR-295 did not name, because KBR-295's ticket
text named the `function` branch only. KBR-299 closes them with the
narrower (empty-only) raise the ticket text prescribes, leaving
absent / null / non-string permissive because of the schema / published
shape facts above.

## Implementation notes

(2026-09-22) TDD: nine `TestCustomDeclarationNameRequired` cases first
(six controls — custom / built-in / mcp × absent and non-string — and
three empty-shape raises), the three empty-shape tests watched failing
on `origin/main` (`DID NOT RAISE` + projected declaration in the verbose
context: `ToolDecl(name="", ...)` for custom / built-in;
`ToolDecl(name="mcp:<server_label>", ...)` for mcp where the empty wire
field is silently overwritten), then the one-inline-check fix in
`_read_tool`'s fallback; then the docstring updates (`_read_tool`'s
`Raises:`, `contract.decode_arguments`' "six declaration branches" list,
and `_require_tool_call_name`'s "one spelling" → "second, deliberately
narrower spelling" note); then the `TEST_SUITE.md` §7.4.2 row 2
"Because" cell addition (anchor-name citations, no line ranges, per
the KBR-292 lesson). Pin-listed dispositions per AC-5: the three
`TestToolDeclarations` controls at lines 1801-1833 stay green (no edit).
Sibling scope (KBR-303, Medium, Epic A child) filed in the same pass per
the design review's sibling-arms sweep — covers `_normalise_tool_choice`'s
empty-name admissions and the adjacent `server_label=""` →
`ToolDecl(name="mcp:")` shape at the same `_read_tool` branch.
`ruff check`, `lint-imports`, `mypy src/kitty`, full pytest suite green on
Linux.
