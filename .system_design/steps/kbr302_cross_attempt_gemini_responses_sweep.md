---
id: kbr302_cross_attempt_gemini_responses_sweep
depends_on: []
---

# KBR-302 — Cross-attempt sweep: Gemini + Responses stream handlers (T-I8 sibling sites)

## What

Widen `tests/harness/failures.py` to `{GEMINI, OPENAI_RESPONSES}` and re-drive
the §4.3 C3 cross-attempt contract — (i) byte-identical repeats, (ii) M9 trigger
+ complement, M8 probe + one complement — on `_stream_responses` and
`_stream_gemini`. Update `tests/bridge/test_cross_attempt_content_l3.py`'s
module docstring to drop the "next ticket" pointer and name the completed
coverage.

## Why

KBR-100 / T-I8 pinned §4.3 C3 on two of the four stream handlers that carry
sibling copies of the retry/empty/M9 arms. The Responses (`server.py:4648`)
and Gemini (`server.py:7896`) sites are unsampled — a regression on either
passes every cross-attempt test today. The `multi_round_review_sweep_rule`
memory names this exact gap class.

## How

1. **`tests/harness/failures.py` widening.** Module-private Gemini and
   Responses chunk builders (same style as the existing `_cc_*` /
   `_anthropic_*` helpers). `frames_for(WireFormat.GEMINI, point)` /
   `frames_for(WireFormat.OPENAI_RESPONSES, point)` for the four
   `InjectionPoint`s. `_SERVED_FORMATS` widened; `_check_fmt` message
   updated; per-format branches in `success`, `empty_response`,
   `error_status`, `cloudflare_block`, `context_too_large`, `drop_at`. No
   import from `src/kitty`. Deterministic byte sequences; abort
   flush-before-drop honoured. Update the module docstring.
2. **`tests/harness/test_failures.py` pins.** Same shape of falsification
   pins for the new formats (builder determinism, `frames_for` table,
   served-set error) — the §6.3 "every harness carries a negative" rule.
3. **`tests/bridge/test_cross_attempt_content_l3.py` L3 cells.** Reuse the
   KBR-100 `_assert_byte_identical_repeat` oracle (via a thin
   `_Upstream`-requests → `CapturedRequest` adapter). Mirror the
   streaming-recovery grid's launcher + `_StubProvider` + `_Upstream` pattern
   for the inbound-protocol dimension (the `BridgeFixture` path can't drive
   Gemini/Responses today because it always passes `None` as the launcher,
   registering only the default Messages route). Add a sibling
   `_ResponsesLauncher(_StubLauncher)` for the Responses inbound route.
   Eight new tests:
   - (i) transport-blip × {Responses, Gemini}
   - (i) empty-response × {Responses, Gemini}
   - (ii) M9 trigger + complement × {Responses, Gemini}
   - (ii) M17 trigger × {Responses, Gemini} + one complement. **Not M8**: the
     carrier-repair site lives only on `_stream_messages`
     (`_is_thinking_roundtrip_error` has exactly one call site,
     server.py:6154); the sibling arm on the other three handlers is the M17
     thinking-signature strip (`_recover_rejected_thinking` →
     `_strip_thinking_blocks`). The KBR-100 file's M8 cell already covers
     `_stream_messages`; this ticket adds M17 coverage on the two unsampled
     routes.
4. **Module docstring update.** Replace the "next ticket" pointer with a
   statement of completed coverage and an updated handler×coverage table.
5. **Pre-push gate.** `ruff check . && lint-imports && mypy src/kitty`
   clean; the Fast gate (`-m l1 or l2`) passes; the L3 file passes via
   bare local pytest (`-m l3` selection); the full local pytest on the
   tree is green.

## Out of scope (named)

- The recorder's `minimal_success_*` stays two-format — the KBR-302 tests
  never call them, and the §6.3.1 grid consumers that do own their own
  byte-sequence widening (the KBR-99 follow-up).
- M6 and M17 re-driving — owned by their existing fixtures
  (`test_tc4_corpus_recovery_l3.py`, etc.).
- The pre-content `event: error` shape and the chosen-stop-reason
  parameterisation on `drop_at` — KBR-99's scope; the cross-attempt cells
  need only the four shape families named in the ticket's "Done when"
  (empty, content, tool_use/format-error, drop).
