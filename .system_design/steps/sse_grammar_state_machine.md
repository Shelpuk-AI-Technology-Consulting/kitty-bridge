---
id: sse_grammar_state_machine
depends_on: []
---

# SSE grammar state machine

T-G7 of the test-suite implementation plan (KBR-83, Epic G). A downstream
grammar state machine over the byte stream the bridge writes to agents, plus
falsification and bridge-driven suites.

## Why

The existing tests assert per-event shapes and response *content*, but nothing
rejects a structurally illegal *sequence* — a `message_stop` before
`content_block_stop`, a Responses delta without an open item, data after
`[DONE]`. Real-world proxies have broken Codex CLI and Claude Code in exactly
this way (out-of-order Responses events made Codex silently discard entire
streams). T-I7 (KBR-99) explicitly owns content invariants; ordering is this
task's grammar.

## Design decisions (see the task's REQUIREMENTS.md for the full table)

- **Four protocols, not three.** The plan row said "three streaming
  protocols"; the bridge serves four inbound SSE routes today. A guard that
  skips one protocol leaves that agent unguarded.
- **Five-way classification, fixed precedence.** `malformed` > `truncated`
  (open structure at finish) > `complete_sentence` > `error_terminal` >
  `truncated` (no terminal). The first draft's "every stream is a complete
  sentence" premise was false: the bridge's native Messages-wire close-out
  writes one error event with forwarded blocks left open — an honest
  truncation, not a defect.
- **Independence.** `tests/harness/sse_grammar.py` imports nothing from
  `src/kitty`; the guard reuses `_KITTY_IMPORT` from `harness.test_contract`
  with positive and negative controls. `StreamProtocol` duplicates
  `InboundProtocol` deliberately (importing `harness.bridge` would pull
  `kitty`); the bridge-driven module asserts the two agree.
- **Amended shapes accepted.** G39 parallel blocks closing out of order, G40
  overlapping Responses items, KBR-242 lazy lifecycle (no `response.created`
  on the D4 path), KBR-250 `[error, response.completed(incomplete)]`, KBR-241
  pre-content errors answered as JSON (the verbatim-forward shape is gone),
  KBR-236 empty-verdict-after-content (blocks closed + one trailing error).

## Implementation notes

- `tests/harness/sse_grammar.py` — four grammar classes over a shared SSE
  framing base (incremental UTF-8 decode, blank-line frame split, sticky
  malformed flag, diagnostic naming the offending kind and frame position).
  `grammar_for(protocol)` factory; `classify_response(protocol, status, body)`
  dispatches `json_error` vs stream parse.
- `tests/harness/test_sse_grammar_falsification.py` — hand-built malformed
  sequences (each asserting the diagnostic) + positive controls for the
  amended shapes + the import-discipline guard over a `frozenset` of module
  paths. l1 by path (harness default).
- `tests/bridge/test_sse_grammar.py` — real `BridgeFixture` drives each §4
  row: both Messages paths (translated CC-upstream, native Anthropic-upstream),
  Responses over CC, CC passthrough, Gemini over CC. 22 parametrised cases
  (1 enum-agreement + 9 translated + 8 native + 2 Responses + 1 CC +
  1 Gemini). `pytest.mark.l2`.
  A `fast_stall` fixture patches `_STREAM_READ_TIMEOUT`, `_BACKOFF_BASE`,
  `_EMPTY_FINAL_DELAYS`, `_TRANSPORT_GRACE_PERIOD` and `_TRANSPORT_GRACE_DELAYS`
  — without it the empty-response ladder (20 s + 40 s final delays) and the
  30 s transport grace collide with the client timeout.
- The translated-path `drop_at(BEFORE_TERMINAL)` cell is genuinely two-way:
  whether the injected finish chunk is processed before the abort-induced read
  exception is not deterministic. The test accepts `complete_sentence` or
  `truncated` for that cell — never malformed, never a JSON error.
- Design docs amended in the same PR: `TEST_SUITE.md` §6.2.2 (taxonomy + four
  protocols) and the plan's T-G7 row.
- The empty-verdict-after-content shape is covered by the falsification
  positive control plus the existing pinned test
  (`TestEmptyVerdictAfterContent`); driving it through `BridgeFixture` +
  `CustomOpenAIAdapter` does not exercise the same close-out path, so the
  bridge-driven module does not duplicate it.
- `scripts/regenerate_step_index.py` does not exist in the repo (recorded in
  memory); the `INDEX.md` step was shipped without machine validation, same as
  KBR-67.

## Verification

- `pytest tests/harness/test_sse_grammar_falsification.py -q` green (45 cases).
- `pytest tests/bridge/test_sse_grammar.py -q` green (22 cases, l2).
- `ruff check` clean on the three new files.
- Broader harness+bridge tiers green (no regressions).
