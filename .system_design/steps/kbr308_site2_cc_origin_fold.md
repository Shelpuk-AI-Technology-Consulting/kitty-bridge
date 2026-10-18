---
id: kbr308_site2_cc_origin_fold
depends_on:
  - kbr308_site1_hop1_carriage
---

# CC-origin carry in `translate_to_upstream` — the adapter fold (R2, R3, AC-2)

KBR-308 site 2. The `/v1/chat/completions` route hands its raw CC body
to `AnthropicAdapter.translate_to_upstream` without translation. The
fold pre-step reads a CC body's plain `cache_control` shapes (top-level,
tool-decl, system-part, assistant-message-object, tool-call,
tool-message-object) and folds each into the **same** KBR-296 carriage
slots the adapter's restore code already reads, when the carriage
doesn't already carry the value (carriage wins — kitty's own translators
are authoritative). Plus the system blocks-form carve so a CC system
content-part carrying a marker reaches the wire as a block with the
marker in place, not a joined string.

## Why

KBR-199 measured six drop sites at the adapter's CC-origin rebuild:
top-level, system-part, assistant-message-object, tool-call,
tool-message-object (and the user-message-object case — deferred). The
KBR-296 restore side reads them via the same carriage keys the M9
rebuild and (after `kbr308_site1_hop1_carriage`) the hop-1 translator
write. Folding is a small pre-step at the top of `translate_to_upstream`
that consults the raw CC shapes when the carriage is silent.

The fold never mutates the caller's `cc_request`: the request dict is
shared across retries, and the strips are copy-on-write for the same
reason (KBR-296 R7 lesson). Local copies only.

## Carriers (six, reusing KBR-296 slots)

1. `cc_request["cache_control"]` → local `_cache_control` if absent
   → wire top-level.
2. `tools[i]["cache_control"]` → local `_tool_cache_controls[name]`
   per-name, merged → wire tool def.
3. System content-parts carrying a marker **and**
   `self.forwards_thinking_signature` is True → carve: emit the system as
   a list of blocks with markers attached (same carve the
   `build_user_content_message` flag flip makes at hop 1); unmarked
   parts keep the joined string byte-for-byte. **The gate is load-bearing**:
   on `minimax_token` and `opencode_go`'s Messages-routed models
   (`forwards_thinking_signature=False`; MiniMax rejects `cache_control`
   on system blocks outright, `minimax_token.py:29-30`) the join stands —
   the G43 scope-out is preserved on the CC-origin path too (design-review
   round 1 blocker fix). `_anthropic_system` carriage path unchanged and
   still wins on the signature-binding adapters.
4. Assistant-message-object `cache_control` → message-level
   `_cache_control` if absent → restored onto the joined text block.
5. `tool_calls[i]["cache_control"]` → local
   `_tool_call_cache_controls[i]` per-index → restored onto the
   rebuilt `tool_use` block.
6. Tool-message-object `cache_control` → message-level `_cache_control`
   if absent → restored onto the `tool_result` block.

User content parts are already restored (kept cases from the prior
KBR-296 work — `{**part}` text spread + image conditional); no change.

## Deferred (PO decision 2026-09-24)

A `cache_control` on a **user-message object** with **string content**
stays dropped: Anthropic's wire has no message-level slot, no published
CC dialect defines the shape, and any placement would invent a block
choice with no fidelity basis. P28's user-half stays a live claim;
named in the row-comment amendment (delivered by
`kbr308_register_and_suite_docs`).

## Fold-site inline comments (DQ-B)

Two short comments at the fold sites encode the non-obvious invariants
so a reader of just the code can reconstruct them:

- "Carriage wins": a raw CC shape is consulted only when the carriage is
  absent — the carriage is kitty's own translator output and is
  authoritative.
- "No request mutation": the fold builds local copies; the request
  dict is shared across retries (KBR-296 R7 lesson).

## CB-2 updates (R7)

The given-breakpoint matrix widens: top-level, tool-declaration,
system-part, tool-call become **kept**; **two new kept cases join** —
assistant-message-object and tool-message-object (the message-object
shapes, distinct from the existing content-part-relocated case); the
existing `message-object-dropped` case is renamed
`user-message-object-deferred-dropped` and pins the **deferred** drop
deliberately; a minimax/opencode-style pin (any
`forwards_thinking_signature=False` adapter) asserts the
system-content-part case keeps the joined-string drop on that route
(the R3 gate's CB-2 cell). The round-trip headline widens to the full
survival set.

## Cross-class strip hygiene (R5)

`ollama_cloud._translate_messages` and `opencode.translate_to_upstream`'s
CC passthrough branch forward no message-level carriage keys verbatim
(KBR-296 R7 already verified for these keys; this re-runs the check at
hop 1 where the translator now writes them, recorded in the step
implementation notes).

## Out of scope

- New internal keys — the carriages are KBR-296's; no
  `_INTERNAL_KEYS`/`_INTERNAL_MESSAGE_KEYS` edits, no harness mirror
  edit.
- A user-message-object carry — explicitly out (PO decision above).
- Native passthrough route — unchanged.
- Responses and Gemini ingress — unchanged; they don't rebuild
  through the Anthropic adapter.

## Implementation notes (2026-09-24)

The fold pre-step lives as fallback reads at each restore site (top-level
in `translate_to_upstream`, tools in `_translate_tools`, assistant message
text and per-tool_use in `_translate_assistant_msg`, tool_result in
`_tool_result_block`); the system blocks-form carve lives in the system
rebuild loop and is gated on `self.forwards_thinking_signature` exactly
as the `_anthropic_system` restore is — preserving the G43/minimax
scope-out on the CC-origin path. The fallbacks read the raw CC shape
when the carriage is absent (carriage wins — DQ-B). Inline comments at
every restore site cite the DQ-B rationale and the hop-1 dual-writer
provenance. CB-2: 29 passing (matrix widened to nine cases — five flips,
two kept, two new — plus the intermediate guards rewritten for the
broader cargo set and the new `test_system_content_part_drop_is_preserved_on_minimax_token`
gate pin). The CB-2 module docstring + intermediate guards' docstrings
record the KBR-308 narrowing from "the adapter restores nothing" to
"the adapter restores every carrier the rebuild can express, except
the deliberately deferred user-message-object case (P28 user-half)".
