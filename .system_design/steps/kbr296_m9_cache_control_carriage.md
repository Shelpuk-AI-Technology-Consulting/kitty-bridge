---
id: kbr296_m9_cache_control_carriage
depends_on: [KBR-271, KBR-200, KBR-258, KBR-263]
---

# KBR-296 — M9 fallback converter carries `cache_control` through every carrier the rebuild can express

Plan: `REQUIREMENTS.md` at
`.requirements/20260922T224819Z_kbr296_m9_cache_control_carriage/REQUIREMENTS.md`.
Jira: **KBR-296** ([link](https://shelpuk.atlassian.net/browse/KBR-296)),
child of epic **KBR-197**. Design: `TEST_SUITE.md` §3.2.1 (rows M9a/M9b),
§3.3.1a, §6.2.3, §9.2 G43 + the G37/G38 closure template.
Predecessors: **KBR-271** (register half, PR #245, merged 2026-09-21 —
M9a/M9b rows and open gap G43), **KBR-200** (CB-3 wire suite pinning today's
M9 behaviour), **KBR-258** / **KBR-263** (hop-1 register twins, product
halves deliberately out of scope).

## What the task does

The product half of KBR-271's split. The M9 fallback converter
(`server._convert_native_to_cc_format`) rebuilds an Anthropic Messages body
as a Chat Completions intermediate after a `tool_use` format error on a
native provider; the adapter's CC→Messages rebuild then drops every
`cache_control` breakpoint except the two carriers the converter already
carries (`system` via `_anthropic_system`, restored only on the two
`forwards_thinking_signature=True` native adapters; `document` via
`_documents`). This ticket carries the remaining eight sites through the
same carriage-and-restore pattern:

1. **Carry-side** (`server._convert_native_to_cc_format`): top-level
   `body["cache_control"]` → `_cache_control`; `tools[i].cache_control` →
   `_tool_cache_controls` (**name-keyed** — P30-aligned, survives
   reordering); part-level breakpoints → re-attached to the CC parts via an
   opt-in on the shared user-content builder (hop 1 byte-identical — the
   ticket's KBR-258/KBR-263 scope-out); assistant text → `_cache_control`
   on the CC assistant message (last-marked-wins); `tool_use` →
   `_tool_call_cache_controls` (index-keyed) on the CC assistant message;
   `tool_result` → `_cache_control` on the CC tool message; list-form
   `tool_result.content` forwarded verbatim (no flatten — hop 1's
   KBR-198/KBR-199 nested preservation restored on this path).
2. **Restore-side** (`AnthropicAdapter.translate_to_upstream` and its
   helpers): each carriage read back onto the rebuilt Anthropic body.
   Attempt-0 parity, no new flags: the value already reached this upstream on
   the first attempt (native passthrough ships the raw body verbatim), so
   the retry cannot newly 400.
3. **Hygiene**: new keys registered in `_INTERNAL_KEYS` /
   `_INTERNAL_MESSAGE_KEYS` so the strips keep them off wires that do not
   consume them.
4. **`minimax_token` system carrier**: scoped deliberately (AC-2's second
   option), recorded in G43's closure text — the restore stays coupled to
   `forwards_thinking_signature` (KBR-228's signature-binding contract) and
   MiniMax's reference rejects `cache_control` on system blocks outright
   (`src/kitty/providers/minimax_token.py:29-30`), so widening would 400
   the retry.
5. **CB-3 suite** (`tests/bridge/test_native_passthrough_cache_breaks.py`):
   `_SURVIVES_ON_ADAPTER` widened (10 sites on `zai_anthropic`/
   `custom_anthropic`, 9 on `minimax_token`); the `tool_result_nested`
   mechanism pin inverted (content array survives, not the flatten).
6. **Register untouched in shape**: no new/removed rows, count pins
   unchanged (AC-4); M9a/M9b row comments + §3.2.1 table cells amended in
   lockstep (the KBR-228/KBR-263 row-text amendment precedent) to record
   the post-fix behaviour; **G43 closes** in §9.2 per the G37/G38 template
   (AC-5).

## Implementation notes

### What landed (2026-09-23)

1. `src/kitty/bridge/server.py` (`_convert_native_to_cc_format`):
   - `_cache_control` (top-level carriage) — automatic-caching form,
     attempt-0 parity.
   - `_tool_cache_controls` — **name-keyed** `{tool_name: breakpoint}`
     (design-review C4: aligns with the register's P30 vocabulary
     `conversation.tools[<name>].cache_control` and survives any future
     normalisation that reorders tools; the adapter looks up by
     `func.get("name")`).
   - `tool_result` block-level `_cache_control` on the CC tool message;
     list-form `tool_result.content` forwarded verbatim (the flatten that
     defeated KBR-198/KBR-199 nested preservation is gone).
   - Assistant joined-text `_cache_control`, **last-marked-wins** (DQ3:
     the latest breakpoint is the effective cache write; Claude Code marks
     one breakpoint per text run today, so the common case is a no-op and
     the rare multi-marked case a strict superset); attached only when the
     join is non-empty (an empty join rebuilds no text block to host it).
   - `_tool_call_cache_controls` index-keyed on the CC assistant message
     (1:1 with the rebuilt `tool_calls` list, same order — verified by
     review: nothing between converter and adapter can mutate it).
   - The shared user-content builder call opts in with
     `carry_cache_control=True`.
2. `src/kitty/bridge/messages/translator.py`
   (`build_user_content_message`): keyword-only `carry_cache_control:
   bool = False`. When on, text/image parts carry their block's
   `cache_control` and a marked text-only turn keeps parts-list form (a
   joined string would lose the breakpoint); the carve is **conditional** —
   unmarked text-only turns keep the joined string (attempt-0 parity;
   `test_text_only_content_unchanged` pins it). Default off keeps hop 1
   byte-identical (R4; CB-1 green unchanged).
3. `src/kitty/providers/anthropic.py`: restores in
   `translate_to_upstream` (top-level `_cache_control` →
   `anthropic["cache_control"]`), `_translate_tools` (name-keyed lookup,
   optional parameter defaulting to `None` — hop-1-free callers
   unaffected), `_tool_result_block` (message-level `_cache_control` →
   block `cache_control`), `_translate_assistant_msg` (text block +
   index-keyed `tool_use` restores). Part-level restores were already in
   place (`anthropic.py:714` spread, `:718-719` explicit image copy).
4. `src/kitty/providers/base.py`: `_INTERNAL_KEYS` += `_cache_control`,
   `_tool_cache_controls`; `_INTERNAL_MESSAGE_KEYS` += `_cache_control`,
   `_tool_call_cache_controls`.
5. `tests/harness/cache_breakpoints.py`: `_KITTY_CARRIAGE_KEYS` mirror
   extended with all three distinct strings (`_cache_control`,
   `_tool_cache_controls`, `_tool_call_cache_controls` — the detector's
   `"cache_control" in key` substring rule would otherwise surface each as
   a phantom finding). `tests/harness/test_cache_breakpoints.py`: the
   mirror-falsification tests cover request-level and message-level cargo;
   the old `test_finds_a_private_prefixed_key` pin (which asserted
   `_cache_control` is FOUND — the KBR-198-era contract) moved to an
   unregistered key (`_my_cache_control`). Its red was exactly the
   drift-correction the mirror docstring promises.
6. `tests/harness/register.py` + `.system_design/TEST_SUITE.md` §3.2.1:
   M9a/M9b row comments and table cells amended **in lockstep** — the
   KBR-271 measurement preserved as the record, the post-fix behaviour
   named, claim dormancy stated. Count pins unchanged
   (`register_disagreements` pins membership/conditional/order, not
   prose). §9.2 **G43 closed** per the G37/G38 template.
7. `tests/bridge/test_native_format_fallback.py`: two AC-8 gap tests —
   marked text-only user turn (parts-list carve, full round trip to the
   wire) and assistant two-marked-text-blocks (last-marked-wins).
8. `tests/bridge/test_messages_translator_property.py`: the pinned
   `_REQUEST_LEVEL_INTERNAL_KEYS` snapshot literal extended with the two
   new registry keys (case (a) of its own docstring: genuinely not hop-1
   translator outputs).

### R7 / AC-7 — cross-class failover hygiene verification (recorded)

The M9 retry always targets the same provider (`self._active_provider`,
`server.py:~4601, ~6046`), so a cross-class leak is hypothetical; the
verification is defensive. Confirmed by reading both targets:

- `ollama_cloud._translate_messages` (`src/kitty/providers/ollama_cloud.py:154-183`):
  rebuilds every message from **named keys only** (`role`, `content`,
  `tool_calls`, `thinking` — via `_flatten_content` for content); unknown
  message-level keys (`_cache_control`, `_tool_call_cache_controls`)
  cannot survive the rebuild. Its `translate_to_upstream` builds `result`
  from named keys only, so the request-level keys never copy either.
- `opencode.translate_to_upstream` (`src/kitty/providers/opencode.py:864-885`):
  Messages-routed models delegate to `AnthropicAdapter` (consumes the
  carriages); Responses-routed models go through `_cc_to_responses`
  (rebuild); Chat Completions-routed models delegate to the base
  `ProviderAdapter.translate_to_upstream`, which strips both levels
  (top-level `_INTERNAL_KEYS` spread + `_strip_internal_message_keys`).

No strip addition was needed; no other adapter forwards unknown
message-level keys.

### Gates

Full bridge + providers + harness sweep: **5417 passed, 5 skipped,
3 deselected** (23 min). CI triple green: `ruff check .` (all checks
passed), `lint-imports` (5 kept, 0 broken), `mypy src/kitty` (success,
94 files). `scripts/regenerate_step_index.py` exit 0. Register pins
(`test_register.py` + `test_register_agreement.py` + `test_oracle.py`)
162 passed. CB-3 matrix 30/30 cells green; CB-1 hop-1 suite green
unchanged; the L1 sibling guard
`test_cache_control_is_not_a_member_of_internal_keys` green (survives by
the underscore difference — KBR-296 registers `_cache_control`, not
`cache_control`).

### Review trail

- Design review (system-design-reviewer), round 1:
  APPROVE-WITH-COMMENTS — 3 blockers (register row-text amendment
  precedent B1; `_KITTY_CARRIAGE_KEYS` mirror gap B2; DQ4 rationale
  rewrite B3), 3 concerns (conditional text-only carve C1; string-form
  tool_result pin verified non-flipping C2; two fixture-unexercised gaps
  → AC-8 C3), 2 decide-and-document (name-keying adopted C4; R7
  verification required C5), suggestions S1–S3 adopted. All folded in.
- Design review round 2: CONFIRMED with two items (mirror must cover all
  three key strings — adopted; matrix-count wording — adopted).
- Code review (code-reviewer): REQUEST CHANGES → documentation-only —
  W1 (this implementation-notes section, including the R7 record above),
  W2 (step file staleness — this rewrite), W3 (stale `#:` block above
  `_SURVIVES_ON_ADAPTER` — rewritten). Suggestion S1 adopted
  (`t.get("name", "")` consistency in the carriage loop); S2 declined
  (the two-loop split preserves the pre-existing rebuild block — surgical
  diff).

## Status

Implemented (2026-09-23); PR open, awaiting CI + review. Not merged by
this session — owner merges.
