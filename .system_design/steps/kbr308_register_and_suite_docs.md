---
id: kbr308_register_and_suite_docs
depends_on:
  - kbr308_site2_cc_origin_fold
---

# Register row comments + TEST_SUITE.md cells amended in lockstep (R8, R9, AC-3)

KBR-308 step 3 — the register prose and TEST_SUITE.md §3.2.1 cells catch
up with the landed carry, per the KBR-228/KBR-263/KBR-296 row-text
amendment precedent.

## Why

The register rows are the documentation-of-record for the KBR-258/KBR-263
measurement. They land regardless of the product fix; their claims become
dormant the day the carry ships (matching is site-blind, so M16 claims the
same address on any route where the carry runs). The row comments name the
post-fix behaviour so a future sweep doesn't mistake the dormant claim for
a live drop. Count pins pin membership/`conditional`/order, not comment
prose — the amendments are prose-only.

## What changes

- M16, P26, P27, P28, P29, P30 row comments in
  `tests/harness/register.py` amended: each names the post-fix carry and
  the route it restored (hop-1 or CC-origin).
- P28's comment additionally names the **deferred** user-message-object
  half (live claim, no wire slot, no published dialect).
- TEST_SUITE.md §3.2.1 cells amended in lockstep (same text).
- The KBR-200 OpenRouter caveat (part-level `cache_control` honoured
  natively on OpenRouter's CC dialect; Anthropic family restores; other
  CC dialects unknown) named in the row text or the
  `build_user_content_message` docstring.
- Count pins unchanged: row count, `conditional` flags, order, §3.2.2
  unconditional list.

## Verification

- `tests/harness/test_register.py` + `test_register_agreement.py` green
  (prose-only amendments are free).
- `.venv/bin/python scripts/regenerate_step_index.py` exit 0.
- Companion-site sweep (auto-memory `register-edit-companion-sites`):
  §3.2.1/§3.2.2 tables, unconditional sentence, count pins,
  falsification-fragment `.replace()` literals, §3.2.5 omission catalogue,
  step-file implementation notes.

## Implementation notes (2026-09-24)

M16's TEST_SUITE.md cell and `register.py` comment each grew a final
paragraph naming the KBR-308 carry at hop 1 (six carriers, two new
AC-8 sibling tests, the deliberate user-message-object deferral,
the cross-reference to the CB-1 suite update). P26-P30 each grew a
one-paragraph amendment recording the site-2 fold at the matching
restore point (the load-bearing `forwards_thinking_signature` gate on
P27, the tool-message-object fold on P28, the assistant-message-object
+ tool-call fold on P29, the name-keyed + raw fallback on P30, the
top-level carriage fallback on P26) plus the matching CB-2 case name
and the full-value (`{"type": "ephemeral", "ttl": "1h"}`) discriminant.
Harness mirror docstring + its inline comment attribute the carriage
keys to **both writers** (KBR-296's M9 site and KBR-308's hop-1 site)
and tighten the drift-correction framing. `register.py` and
`TEST_SUITE.md` are kept byte-identical by hand across each row's text
amendment. Count pins: 122 register-pin tests green, all
six falsification-fragment guards green, `_KITTY_CARRIAGE_KEYS`
mirror guard green unchanged (no new keys; reused the KBR-296 set).
