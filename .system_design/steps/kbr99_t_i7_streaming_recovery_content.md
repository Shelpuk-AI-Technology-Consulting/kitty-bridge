---
id: kbr99_t_i7_streaming_recovery_content
depends_on: [KBR-43, KBR-83, KBR-31]
---

# T-I7 — Streaming recovery: content guarantees, not just grammar (KBR-99)

Plan row: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §12 **T-I7**. Design:
`TEST_SUITE.md` §6.3.1 + §11 Q14 (D1–D7). Requirements:
`.requirements/20260921T115834Z_kbr99_t_i7_streaming_recovery/REQUIREMENTS.md`.

Depends on T-B4 (KBR-43, the scripted failure library), T-G7 (KBR-83, the SSE
grammar floor) and T-W8 (KBR-31, the bridge fixture) — all Done when this
step opened.

## Scope (owner decisions 2026-09-21)

Beyond the core four-injection-point oracles, the ticket carries four product
fixes, each confirmed by the product owner in the opening conversation of
this task:

1. **Gemini in-stream-error exhaustion writes its terminal diagnostic** —
   one `{"error": {"code": 502, ...}}` SSE event before the break (the
   KBR-247 Gemini convention). Filed by the KBR-247 review (2026-09-21).
2. **Translated `/v1/messages` exhaustion unifies on D4** — the
   finish-chunk empty arm stops exhausting into M12 fallback text; the
   streaming M12 site is retired (M12 stays live non-streaming).
3. **Streaming D3 lands on the translated route** — a truncation
   finish-chunk empty stream fails at once with
   `400 reason: "<stop_reason>_before_content"`.
4. **Messages-wire pre-content error quarantine parity** — the empty ladder
   charges `_get_stream_error_cooldown`, closing the difference KBR-241's D2
   amendment kept.

The translated transport-drop ending (`end_turn` + `message_stop`, KBR-183
D2) stays as the documented residue and is pinned as such.

## Delivered (2026-09-21)

1. **`tests/bridge/test_streaming_recovery_content.py`** (new, l1) — the
   §6.3.1 grid oracle plus the four fix scenarios, driven by the sibling's
   local scripted-aiohttp harness (reused — no second harness surface):
   - Grid (R1, green at base): pre-emission clean failover on the translated
     route and the native Messages passthrough; post-emission timeout
     endings on AFTER_TEXT, MID_TOOL_ARGUMENTS, BEFORE_TERMINAL with
     content-identity assertions (no duplicated text via the
     `_assert_no_duplicated_content` helper).
   - Pins (R6, green at base): the D2 residue on AFTER_TEXT and
     MID_TOOL_ARGUMENTS transport drops; the BEFORE_TERMINAL two-way shape
     set; the single-backend translated timeout ending; the streaming 413
     failover (balancing) / upstream-error surface (single); the CC empty
     ladder on a stream that forwards NOTHING (no role chunk).
   - Falsification (§1.4): `TestGridOracleFalsification` proves the
     no-duplication oracle catches a replayed-text defect.
   - Fix regressions (each red at base, green with its fix): R2 Gemini
     terminal diagnostic; R3 D4 unification on translated Messages; R4
     streaming D3 (`finish_reason: "length"` and the literal
     `"model_context_window_exceeded"` pass-through); R5 Messages-wire
     pre-content error quarantine parity.
2. **Product fixes in `src/kitty/bridge/server.py`**:
   - R2 (S10): `_stream_gemini`'s `stream_error + events_emitted` arm now
     writes one `{"error": {"code": 502, ...}}` SSE event before breaking.
   - R3 (S11): translated `/v1/messages` empty-stream ladder unifies on
     D4 — the `if empty_no_finish:` guard around the D4 return is dropped,
     retiring the streaming M12 fallback path. The translator's fallback
     synthesis is untouched; M12 stays live on non-streaming replies.
   - R4 (S12): streaming D3 on the translated route. New helper
     `_chunk_finish_reason` captures the raw CC `finish_reason` at
     buffering time (main loop and the flush twin); inside the empty gate,
     after the post-emission arm, the raw spelling maps through
     `TranslationEngine.map_finish_reason` and a member of
     `_NATIVE_TRUNCATING_STOP_REASONS` returns the shared
     `_d3_truncation_error_body` 400, ending the ladder on that attempt.
   - R5 (S13): the Messages-wire pre-content error ladder charges
     `_get_stream_error_cooldown` on the failing backend at ladder entry
     when the hold recorded an upstream error event, closing the D2
     amendment's kept-difference and matching the CC-wire in-stream error
     cooldown.
3. **Design-of-record updates** (`SYSTEM_DESIGN.md` §5.3 S10–S13, §5.4
   residue bullet, `TEST_SUITE.md` §6.3.1 + §11 D2 amendment, KBR-235
   boundaries, §7.2.1 remainder, `TEST_SUITE_IMPLEMENTATION_PLAN.md` §12
   T-I7 row annotation, register row M12 notes) — committed ahead of the
   code (3e1ce0e, 899df4e, 1445d93).
4. **Test sweep** of the fixes: the KBR-235-era empty-response pins and
   the KBR-241-era quarantine pins moved with each fix to the new contract
   (the tests themselves are owned by their original PRs; the sweep-rule
   says the move rides with the change). 292 tests across the affected
   suites pass locally; the project's authoritative verdict is CI.
