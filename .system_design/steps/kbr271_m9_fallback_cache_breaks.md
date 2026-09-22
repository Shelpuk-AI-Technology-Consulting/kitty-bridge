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

### What landed (2026-09-21)

1. `tests/harness/register.py`: two `MutationRow` entries after M9.
   - **M9a** — `paths=(c.extra_path("cache_control"),)` (keyed literal, no
     `_SHAPES` entry — P26's reason). Twin of M16's fourth path.
   - **M9b** — `paths=(c.tool_path(c.WILDCARD, "cache_control"),
     c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"))` (both patterns
     already in `_SHAPES` for M16/P28). Twin of M16's three carrier paths;
     adds the tool-declaration path the ticket's draft omitted — the CB-3
     suite's `tool` site is the tool *declaration* (`("tools", 0)`), a
     different address from the part carriers.
   - Both rows: site `("kitty/bridge/server.py:_convert_native_to_cc_format",)`,
     trigger `NON_NATIVE_UPSTREAM_WIRE`, `conditional=False`. Trigger
     rationale per the design review: documentation symmetry with the hop-1
     twins, not reachability — the fallback is native-route-only (all four
     call sites gate on `_native_messages_request`), so the ROUTE-kind
     trigger is false on the only path that reaches the site today; the
     rows are anticipatory in P28's sense, and both overlap facts
     (site-blind matching against M16; anticipatory) are stated in the
     row comments so no future reader re-derives them from `oracle.py`.
     The alternative trigger (`NATIVE_TOOL_USE_FORMAT_ERROR`) was rejected:
     sharing M9's trigger forces `conditional=True` and owes a §3.3.2
     assertion-2 complement no corpus entry can author.
2. `TEST_SUITE.md`: §3.2.1 intro paragraph (23→25 request-path rows; 25→27
   live), the two table rows after M9, §3.2.2's unconditional sentence
   (M9a/M9b inserted adjacent to their trigger-family neighbours — the
   parser reads the ids as a set, placement is for human-reader tracking),
   and §9.2's open gap **G43** (not the ticket's G39 — taken by KBR-226)
   recording the residual product defect plus the two drops the rows
   cannot claim: the `system` carrier on `minimax_token`
   (`forwards_thinking_signature=False`, native route, outside the
   trigger's reach) and the nested `tool_result` block (residualises
   before register matching; the §3.3.1a vocabulary has no path for it).
3. Count pins (four): `test_register.py` total 77→79 with the docstring
   provenance chain extended ("+2 over the pre-KBR-271 count is M9a and
   M9b") and the stale "73 live rows / 48 provider-level" numbers
   corrected in the same edit; `test_register_agreement.py` M-rows 25→27,
   P-rows 52 unchanged, unconditional 46→48.
4. Falsification-fragment drift prevention: the two `.replace()` literals
   in `test_register_agreement.py`'s unconditional-list negative controls
   hard-code the sentence's opening fragment; both updated to track the
   new sentence (the §3.2.4 "second copy nothing compares" hazard, in
   miniature — the negative controls went silently vacuous at first run
   and the failures named the fragment, not the register).

Gates: register suites + CB-3 wire suite 154 passed (watched red first:
count pin 79≠77, "M9a unpublished", order divergence — all for the right
reason); full `tests/harness/` suite 2526 passed / 2 skipped; register-
adjacent guards 50 passed; markdown-consuming contract tests 194 passed.
`scripts/regenerate_step_index.py` exit 0. A whole-suite background run
was killed by the OS for memory, not by a failure; the targeted sweep
covers every suite that reads the register or the design document.

Design review: system-design-reviewer ran one round on REQUIREMENTS.md
(no blockers; five concerns — trigger rationale inverted vs the
reachability gate, "false I1 breach" framing unfireable, system-carrier
scope-out wrong on minimax_token, nested tool_result silently unclaimed,
KBR-285 provenance wrong — all folded in before implementation, plus
three suggestions adopted: stale docstring numbers, §3.2.2 placement,
the two overlap facts in the row comments).

## Status

Implemented (2026-09-21); PR [#245](https://github.com/Shelpuk-AI-Technology-Consulting/kitty-bridge/pull/245)
open, awaiting CI + review.
