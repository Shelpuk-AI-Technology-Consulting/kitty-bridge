---
id: kbr271_m9_fallback_cache_breaks
depends_on: [KBR-200, KBR-258, KBR-263]
---

# KBR-271 — M9 fallback converter drops `cache_control` breakpoints: register the drops

Plan: `REQUIREMENTS.md` at
`.requirements/20260920T220257Z_kbr271_m9_fallback_cache_breaks/REQUIREMENTS.md`.
Jira: **KBR-271** ([link](https://shelpuk.atlassian.net/browse/KBR-271)).
Design: `TEST_SUITE.md` §3.2.1 (the new M9a/M9b rows), §3.3.1a (path
vocabulary — keyed literal, wildcard patterns already listed for M16/P28),
§9.2 G43 (the residual product defect and the two scope-outs).
Predecessors: **KBR-200** (CB-3 suite, PR #159, merged 2026-09-13) pinned
the M9 fallback's wire behaviour; **KBR-258** (PR #173, merged 2026-09-15)
closed G37 by landing the CC-route twin rows P26–P30; **KBR-263** (PR #188,
merged 2026-09-16) closed G38 by extending M16 with the fourth path
(`envelope.extra[cache_control]`).

## What the task does

Closes the M9-fallback half of the KBR-197 prompt-cache-fidelity epic's
register-side surface. The M9 fallback converter
(`server._convert_native_to_cc_format`) drops every `cache_control`
breakpoint except the two carriers it carries through (`system` via
`_anthropic_system` carriage, restored on `forwards_thinking_signature`
adapters; `document` via `_documents`). The drops at the tool
*declaration*, every content part (`image`, user/assistant text, `tool_use`,
`tool_result` — the converter flattens `tool_result.content` to a string
before any carriage, defeating the Messages translator's KBR-198/KBR-199
nested preservation), and the top-level automatic-caching form were
unclaimed at the register; no row anchored on the part-level or top-level
`cache_control` at M9's site, so the under-claiming direction §3.3.1a
calls unrecoverable would fire the day an oracle run reaches the site.

The change:

1. Adds two unconditional register rows at M9's site:
   - **M9a** — `paths=(c.extra_path("cache_control"),)` — the M9 twin of
     M16's fourth path (KBR-263 / G38).
   - **M9b** — `paths=(c.tool_path(c.WILDCARD, "cache_control"),
     c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"))` — tool
     declarations plus content parts (the converter flattens every block
     it touches).
2. Both rows: site `("kitty/bridge/server.py:_convert_native_to_cc_format",)`,
   trigger `NON_NATIVE_UPSTREAM_WIRE` (M16-family documentation symmetry;
   the alternative `NATIVE_TOOL_USE_FORMAT_ERROR` forces `conditional=True`
   with a §3.3.2 complement no corpus entry can author — the fallback is
   reachable only on the native route, so the ROUTE-kind trigger is false
   on the only path that reaches the site today), `conditional=False`.
3. Updates `.system_design/TEST_SUITE.md`:
   - §3.2.1 intro paragraph (row counts: 23→25 request-path rows; 25→27 live).
   - §3.2.1 table: M9a and M9b rows after M9.
   - §3.2.2 unconditional sentence: adds M9a, M9b adjacent to their trigger-
     family neighbours (the parser reads the ids as a set; placement is
     for human-reader tracking).
   - §9.2: open gap **G43** recording the residual product defect (the
     product-side carry-through is epic KBR-197's boundary) **plus the
     two drops these rows cannot claim**: the `system` carrier on
     `minimax_token` (the adapter declares `forwards_thinking_signature=False`,
     so the carriage is never restored — a drop on a native route, outside
     `NON_NATIVE_UPSTREAM_WIRE`'s reach) and the nested `tool_result` block
     (residualises before register matching; §3.3.1a's vocabulary has no
     path for it — M16 records the same fact for its own paths).
4. Updates the four count pins:
   - `tests/harness/test_register.py`: total rows 77 → 79 (+2), with the
     docstring's provenance chain extended to "+2 over the pre-KBR-271
     count is M9a and M9b (KBR-271)" and the stale "73 live rows / 48
     provider-level" numbers corrected in the same edit.
   - `tests/harness/test_register_agreement.py`: M-rows 25 → 27;
     unconditional ids 46 → 48; the two falsification-fragment tests'
     `.replace()` literals updated to track the new sentence (the same
     "second copy nothing compares" hazard §3.2.4 records, in miniature).
5. No product code changes. The converter's behaviour is pinned, not
   fixed (the ticket's explicit scope-out).

## Implementation notes

(To be filled in after the implementation lands — the format is
post-hoc, per the CLAUDE.md: "After completing an implementation task,
append implementation notes to the step file.")
