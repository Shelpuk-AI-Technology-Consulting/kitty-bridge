---
id: kbr303_tool_choice_and_mcp_server_label_empty_raise
depends_on:
  - kbr296_responses_custom_declaration_empty_name_raise
---

# KBR-303 — Close the silent `tool_choice.name=""` admission across the five readers that carry the shape, plus the Responses MCP `server_label=""` declaration slip

Plan: `.requirements/20260923T000000Z_kbr303_tool_choice_and_mcp_server_label_empty_raise/REQUIREMENTS.md`.

## What

Tighten the empty-string shape on the **selection surface** of every
reader that carries a `tool_choice` by-name selector, and the
**declaration surface** of the Responses reader's MCP sub-branch.
The same `isinstance(x, str)` check that admits `""` lives on five
readers today:

1. Responses (`_normalise_tool_choice`,
   `reader_responses.py:526-532` mcp + `534-536` by-name).
2. Chat Completions (`_read_tool_choice`,
   `reader_chat_completions.py:1369-1380`).
3. Anthropic (`_read_tool_choice`,
   `reader_anthropic_messages.py:456-459`).
4. Converse (`_read_tool_choice`,
   `reader_bedrock_converse.py:606-613`).
5. Gemini (`_read_tool_choice`,
   `reader_gemini.py:1171-1175`,
   `allowedFunctionNames[0]`).

Plus the Responses **declaration** MCP sub-branch
(`_read_tool` mcp arm at `reader_responses.py:1310-1312`).

After this change, every empty `name` / `server_label` on the eight
sites raises `UnreadableBodyError` with the family message spelling
(`f"{path} must be a non-empty string name"` or
`...server_label"`) so the family greps together via
`git grep "must be a non-empty string" tests/harness/`. The
absent / non-string postures stay deliberately permissive — residualise
on the selection surface, narrow-to-server on the Responses mcp
selection, `"mcp"` fallback on the Responses declaration mcp sub-branch.

## Why

KBR-303's ticket text named only the Responses reader's selection and
declaration slips. The wider sweep the design reviewer
(system-design-reviewer, 2026-09-23) ran on `origin/main`
(`247bc71`) found the same defect class on four more readers — per
the `multi_round_review_sweep_rule` memory, every widening on one arm
demands the sweep across sibling arms with the same shape. KBR-292 and
KBR-295 both widened under the design verdict (KBR-292: 4 → 7
invocation readers; KBR-295: 3 → 6 declaration branches), so KBR-303
follows the same precedent.

The settlement posture — **empty raises, absent / non-string stays
deliberate** — matches the family verdict KBR-281 / KBR-292 / KBR-295 /
KBR-299 settled: `""` for a name is not a lossless projection
(`contract.decode_arguments`), and `"tool:"` corresponds to no tool.
The permissive postures stay because absent / non-string are *identified*
losses (the kind-derived label, the residualised selector, the
narrow-to-server projection, the `"mcp"` fallback), not silent slips.

## Why Medium (not High like KBR-299)

The declaration-side defect KBR-299 closed costs a *tool its identity*
— it cannot be paired with its result or addressed by a register row.
The selection-side defect costs a *selection its referent* — `"tool:"`
corresponds to no tool, a lower-stakes loss on the envelope-extra
surface. Same defect family, smaller blast radius; the ticket invites
PO re-triage.

## Implementation notes

Implemented in PR (branch
`feat/kbr-303-tool-choice-empty-selector-and-mcp-server-label`,
off `origin/main` 2026-09-23). Eight sites tightened across five
readers (Responses selection + declaration, Chat Completions,
Anthropic, Bedrock Converse, Gemini) and one declaration branch
(Responses MCP `server_label=""` slip). Every absent / non-string
posture pinned by a control test, mirroring KBR-299's
six-controls pattern, so a future widening of the empty-only guard
fails loudly.

**Per-reader fix sites (re-derived against the landed branch):**
- `tests/harness/reader_responses.py` — `_normalise_tool_choice`
  (mcp branch raises on `server_label=""` and `name=""`; by-name
  branch raises on `name=""`); `_read_tool`'s mcp sub-branch raises
  on `server_label=""`.
- `tests/harness/reader_chat_completions.py:1369-1380` — extends
  the existing `isinstance(name, str)` raise to `or not name`.
- `tests/harness/reader_anthropic_messages.py:456-459` — same
  `or not name` extension.
- `tests/harness/reader_bedrock_converse.py:606-613` — adds
  empty-only raise inside the `isinstance(target.get("name"), str)`
  truthiness check.
- `tests/harness/reader_gemini.py:1171-1175` — adds empty-only
  raise inside the `isinstance(allowed[0], str)` truthiness check;
  multi-element restrictions keep the residualise posture.

**Family message spelling:**
`f"{path} must be a non-empty string name"|server_label"`. The
existing CC / Anthropic / Gemini messages were updated to embed the
path prefix so `git grep "must be a non-empty string" tests/harness/`
lists every site (27 occurrences post-merge).

**Test additions (24 total):**
- `TestToolChoiceEmptySelectorRaises` (5 cases incl. one
  interaction pin for the `server_label=""` + `name=""` order) +
  8 controls in `TestToolChoiceNormalisation` + 3 declaration tests
  in `TestToolDeclarations` for Responses.
- 2 parametrised raise cases for Chat Completions.
- 1 raise case for Anthropic.
- 1 raise case + 2 controls for Bedrock Converse.
- 1 raise case + 1 multi-element control for Gemini.

**Design doc updates:**
- `.system_design/TEST_SUITE.md` §7.4.2 rule 7 row 2 — KBR-303
  extension paragraph added, replacing the "filed as KBR-303"
  forward reference with the landed state; General-prescription
  pointer in the same cell extended to the selection surface across
  the five readers plus the Responses declaration mcp
  `server_label` sub-branch.
- `_require_tool_call_name` docstring at
  `tests/harness/reader_responses.py:276-306` extended to record
  the third inline spelling (selection surface) and the fourth
  (declaration mcp `server_label` shape).

**Gates at merge-time:** `ruff check src tests` clean; `mypy
src/kitty` clean (94 files); harness tests pass (2619 passed, 2
skipped, 3 deselected).
