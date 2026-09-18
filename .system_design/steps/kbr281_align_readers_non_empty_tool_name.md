---
id: kbr281_align_readers_non_empty_tool_name
depends_on: [KBR-279]
---

# KBR-281 — Align Chat Completions, Gemini, Anthropic readers on the non-empty tool-call name

Plan: `REQUIREMENTS.md` at
`.requirements/20260918T204933Z_align_readers_non_empty_tool_name/`.

## What

Extend the strict tool-call `name` rule KBR-267/KBR-279 landed for the two Ollama readers to the
three remaining readers that project a `ToolUse`: Chat Completions (request `_read_tool_calls`,
reply `tool_calls`, reply legacy `function_call`), Gemini (reply `_read_part`, request
`_read_function_call`), and Anthropic (`_read_block`, one site shared by both directions). A
tool-call `name` that is absent, empty, or not a string raises `UnreadableBodyError` naming the
path, at all six sites. `TEST_SUITE.md` §7.4.2 rule 7 row 2 is rewritten to record the settled
rule; the per-reader Because cell is updated honestly (Bedrock Converse request and Responses
request residualise absent/non-string and are silent on `""`; Responses reply raises on
absent/non-string).

## Why

`contract.decode_arguments`'s losslessness argument (`contract.py:935-941`): `""` for a name is
not a lossless projection — it claims a tool *named* empty-string, and a call nobody can name
cannot be paired with its result or addressed by a register row. The three readers already raise
on absent and non-string names; accepting `""` silently was the one remaining silent path, and the
same defect the Ollama readers stopped projecting in KBR-267/279. Uniformity is the point of the
harness: two readers that disagree on a malformed input report a delta on content nobody changed.

**Why raise and not residualise (the ticket's Open Question).** The ticket offered outcome (b):
residualise the empty string and project, "mirroring Bedrock / Responses". An empirical probe run
during the requirements phase showed that premise is false — Bedrock Converse and Responses are
**silent** on `""` (their `isinstance(name, str)` guards pass; only absent/non-string residualise).
Option (b) therefore had no precedent to mirror within this ticket's scope, and implementing it
would have meant inventing a posture no reader has. Outcome (a) extends raises that already exist
at every site for the neighbouring name shapes; the design reviewer confirmed (a) in review
round 1.

**Why Gemini's request direction changes absent/non-string behavior too.** It routed its name
through the residualising `_read_required_name` helper — which also serves `FunctionDeclaration`
(tool *declaration*) names, where the residualise posture is deliberate (§3.3.1b, T-A3) and stays.
Uniformity across a module's directions (§7.4.1) requires the request direction to match its own
reply direction, so the request tool-call site stops routing through the helper and both sites
share a new `_require_tool_call_name`, named to match Ollama's so the strict-name helpers grep
together. The declaration path and its pins are untouched.

**Why the residualise posture stays for Bedrock Converse and Responses.** Their residualise+
project behavior on absent/non-string is the general prescription of §7.4.2 rule 7 row 2
(§3.3.1b's sentence), recorded as the deliberate deviation now that four readers raise. Their
silent `""` case is the same defect class as this ticket but lives in readers this ticket does not
cover; it is recorded honestly in the row rather than fixed here.

## Verification

- New `TestNameRequired` classes (request direction) in `tests/harness/test_reader_chat_completions.py`,
  `tests/harness/test_reader_gemini.py`, `tests/harness/test_reader_anthropic_messages.py`; four-case
  coverage (absent/empty/non-string raise + published-example control) in
  `tests/harness/test_reader_chat_completions_reply.py` (both reply sites) and
  `tests/harness/test_reader_gemini_reply.py` — each empty case watched failing (silent projection)
  before the fix.
- `pytest tests/harness/test_reader_chat_completions.py tests/harness/test_reader_chat_completions_reply.py
  tests/harness/test_reader_gemini.py tests/harness/test_reader_gemini_reply.py
  tests/harness/test_reader_anthropic_messages.py tests/harness/test_reader_anthropic_messages_reply.py`
  green; declaration-name pins (`TestEveryOptionalLeafFailsClosed.CASES["functionDeclaration.name"]`,
  the no-usable-name-declaration residualise test) untouched and green.
- `pytest` full suite, `ruff check .`, `lint-imports`, `mypy src/kitty` — all green on Linux.
- `python scripts/regenerate_step_index.py` exits 0 (or fails only on the pre-existing dangling
  `t_g6` → `t_w8` edge recorded on main, which this step may not edit).

## Implementation notes

(round 1 review: RESOLUTION option (a); all findings applied — see REQUIREMENTS.md AC-7 for the
exact list of existing test bodies replaced.)
