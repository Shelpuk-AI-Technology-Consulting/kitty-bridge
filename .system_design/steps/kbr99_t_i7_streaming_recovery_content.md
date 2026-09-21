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

## Delivered

(implementation notes appended at completion)
