---
id: kbr308_site1_hop1_carriage
depends_on:
  - KBR-296
---

# Hop-1 translator writes the KBR-296 carriages (R1, AC-1)

KBR-308 site 1. `MessagesTranslator.translate_request` writes the six
carriages the adapter's KBR-296 restore side already reads, so the
hop-1 path carries `cache_control` breakpoints for every carrier the
rebuild can express. The opt-in keyword on the shared user-content
builder is flipped to `True` for the hop-1 call site.

## Why

KBR-258/KBR-263 closed the register half (P26–P30 + M16's fourth
path); KBR-296 closed the M9 product half. The primary translated
route — every request `translate_request` handles — still strips the
breakpoint. The cost is the same as the M9 fallback's: Anthropic
prices a cache read at ~0.1× base input, so a stripped marker
re-bills the agent's stable prefix at ~10× the cached rate on every
turn. The carry reuses the keys the adapter already reads; no new
carriage vocabulary, no fork with the M9 path, drift is impossible.

## Carriers (six)

1. `messages_request["cache_control"]` → `result["_cache_control"]` (top-level).
2. `tools[i]["cache_control"]` → `result["_tool_cache_controls"]`
   name-keyed (P30 vocabulary).
3. `_user_content_message(..., carry_cache_control=True)` (the KBR-296
   keyword-only flag flipped on at hop 1).
4. Assistant text blocks → message-level `_cache_control`,
   last-marked-wins.
5. `tool_use` blocks → message-level `_tool_call_cache_controls`,
   index-keyed.
6. `tool_result` blocks → message-level `_cache_control` on the CC tool
   message.

`system` / `document` / nested `tool_result` content ride unchanged
(`_anthropic_system`, `_documents`, verbatim forward).

## CB-1 update (R6)

Seven absence tests invert; the inversion is **partial** because the
harness detector skips `_KITTY_CARRIAGE_KEYS`:

- `user_text`, `image`: flip red on their own — the markers ride the CC
  parts, which `find_breakpoints` sees.
- `tool`, `assistant_text`, `tool_use`, `tool_result`, `top_level`:
  become carriage-cargo drill-ins (whole-body absence assertions would
  stay green vacuously — the document-test pattern at
  `test_messages_translator.py:621-632` is the model).
- `system`, top-level key-set test: assertions keep (carriage-skipped /
  underscored key); docstrings rewritten.
- Class docstring: updated to the accurate partial-inversion count.
- `document`, `tool_result_nested`: stay green unchanged.

Two new tests mirror KBR-296's AC-8: a text-only user turn with a
marked text block keeps the parts-form carve; an assistant turn with
two differently marked text blocks keeps the last-marked-wins
breakpoint.

## CB-2 round-trip update (R7)

`test_the_default_anthropic_route_delivers_only_the_restored_system_carrier`
widens to the full survival set; the two intermediate guards update
for part-level markers and the new carriages; the given-breakpoint
matrix widens per R7.

## Out of scope

- The internal-key registries and harness mirror (`_INTERNAL_KEYS`,
  `_INTERNAL_MESSAGE_KEYS`, `_KITTY_CARRIAGE_KEYS`) — unchanged; the
  keys were registered by KBR-296, the guard stays green unchanged.
- `minimax_token` system scope-out — recorded deliberately, untouched.
- The native passthrough route — unchanged.
- Register rows — count pins unchanged; row-comment amendments roll
  into `kbr308_register_and_suite_docs`.

## Implementation notes (2026-09-24)

The six carries landed in the same wire-up order as the TDD steps:
top-level → tools → tool_result (via the tool-message rebuild path,
with the nested-list content forwarded verbatim like hop 1) → parts
via `_user_content_message(..., carry_cache_control=True)` →
assistant text last-marked-wins + tool_use index-keyed. Test
assertions drill into the carriage keys for the carriage-riding
sites (the harness detector skips `_KITTY_CARRIAGE_KEYS`; whole-body
absence would stay green vacuously after the carry, per the KBR-308
design-review round-1 finding); drill into the CC part for text/image
(the `_documents` re-addressing double-counts via whole-body
find_breakpoints, also round-1 finding). The class docstring is
updated to the accurate partial-inversion breakdown. The
`build_user_content_message` docstring is rewritten to name both
call sites, retire the "default off" clause for KBR-308 (both
production callers now pass `True`), and carry the KBR-200 OpenRouter
caveat. CB-1: 14 passing (12 inverted + 2 new AC-8 siblings).
