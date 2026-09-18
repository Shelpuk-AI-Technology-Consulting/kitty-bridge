---
id: kbr277_non_streaming_cc_reasoning_content
depends_on: [KBR-248, KBR-155]
---

# KBR-277 — Non-streaming `_is_empty_cc_response` reads `reasoning_content`

Jira: [KBR-277](https://shelpuk.atlassian.net/browse/KBR-277). Parent: KBR-136.
Plan row: TBD (filed as a follow-up to KBR-248 rather than a tracked plan
task; the KBR-248 PR description already listed this as a separate ticket).
Design: `SYSTEM_DESIGN.md` §5.4 (KBR-248 area). The four emptiness oracles are enumerated in `TEST_SUITE.md` §7.2.1; the underlying reference for the asymmetry is §4.3 C3.

Depends on:

* **KBR-248** (Done, PR #208) — supplied the streaming-side `_cc_chunk_carries_content`
  predicate and the converted-route hold that releases on `reasoning_content`. This
  ticket mirrors the same predicate on the non-streaming side.
* **KBR-155** (Done, PR #60) — supplied `PreambleHold` and Q14 D1–D7, which define
  what "content" means on the Messages wire. The Messages-shaped arm of
  `_is_empty_cc_response` already conforms to D1; only the CC arm diverged.

(Per CLAUDE.md, `scripts/regenerate_step_index.py` does not exist in the repo
— see memory `missing_step-index-validator`. No machine validation of the
`depends_on` graph runs for this step.)

## Implementation notes

Delivered 2026-09-17 (PR pending):

1. **`src/kitty/bridge/server.py`** — `_is_empty_cc_response`'s Chat
   Completions arm gained `has_reasoning = isinstance(reasoning_content, str)
   and reasoning_content != ""`, the literal mirror of
   `_cc_chunk_carries_content`'s last clause (`server.py:1483`), added as the
   last conjunct so mutation testing side-by-side catches divergence. The
   Messages-shaped arm is untouched (D1-consistent). Docstring expanded to
   name both arms, the mirror, and the pre-existing `.strip()` vs `!= ""`
   drift on `content`.
2. **`tests/bridge/test_empty_response_retry.py`** — seven L1 unit tests in
   `TestEmptyResponseDetection` (truthy / empty-string / whitespace /
   non-string / alongside-content / alongside-tool_calls / missing-key) and
   the L3 bridge-level mirror of the KBR-248 streaming pin in
   `TestEmptyResponseNonBalancing` (`test_non_streaming_reasoning_only_response_succeeds_first_attempt`,
   collapsed delays, one upstream reply registered, asserts calls == 1 and
   reasoning reaches the client).
3. **`tests/bridge/test_empty_response_reasoning_properties.py`** (new) — the
   L1 Hypothesis property: `_cc_chunk_carries_content` agrees with
   `_is_empty_cc_response` on the `reasoning_content` axis, `tool_calls`
   varied over `{None, [], [{id: "t1"}]}`, `content` held to the two
   absent/empty shapes the two predicates agree on (documented drift
   excluded from the domain).
4. **`README.md`** — the empty-reply FAQ's "One asymmetry to note" sentence
   replaced by the positive claim (reasoning-only replies succeed on the
   first attempt on both routes; KBR-248 + KBR-277).
5. **`SYSTEM_DESIGN.md` §5.4** — the "Known asymmetry, deliberate" bullet
   replaced by the KBR-277 closure paragraph naming the shared predicate and
   explicitly recording the `content` drift as documented drift.

**Watched-fail evidence:** pre-fix, the truthy / whitespace unit tests and
the property all failed on `reasoning_content` values the streaming side
accepts; the bridge-level test failed through the retry ladder (second
upstream call hit an unregistered `aioresponses` route → connection-blip
retries → 500). Post-fix, 9/9 new tests green; 137 adjacent tests
(`test_empty_response_retry`, `test_messages_wire_translated_streams`,
`test_native_empty_stream`) green.