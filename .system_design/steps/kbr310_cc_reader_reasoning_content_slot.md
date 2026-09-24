---
id: kbr310_cc_reader_reasoning_content_slot
depends_on: [kbr285_widen_cc_content_classifier, t_d4_default_transport_slice]
---

# KBR-310 — CC reader slot for `reasoning_content` on assistant messages

Ticket: [KBR-310](https://shelpuk.atlassian.net/browse/KBR-310) (KBR-16 Epic D,
High, labels `fidelity` + `found-by-kbr-54` + `silent-skeleton`). F3.c
(`TEST_SUITE.md`). Requirements:
`.requirements/20260924T082808Z_kbr310_cc_reader_reasoning_content_slot/REQUIREMENTS.md`.

## What

The Chat Completions request reader (`tests/harness/reader_chat_completions.py`,
T-A2) did not model `reasoning_content` on assistant messages. The reply
direction did (the P8 complement — `test_reader_chat_completions_reply.py::test_reasoning_content_projects_as_thinking`); the request direction's
`_read_one_message` consumed only `{role, content, tool_calls, name, refusal}`,
so a `messages[N].reasoning_content` key residualised at its path and
`contract.verify_total` raised `ProjectionTotalityError`. The bridge does
not strip the field — `ProviderAdapter._inject_empty_reasoning_content` writes
`""` on every assistant message once thinking is active (Kimi / Z.AI /
custom-OpenAI reject without it), and the Messages→CC translator sets
`_thinking_enabled = True` for a body carrying `thinking: {"type": "adaptive"}`
(`src/kitty/bridge/messages/translator.py:484-489`). Empirically (reproducer
`.tmp/repro_kbr310.py` against `554a653`): the default-transport slice's
`tool_use_and_tool_result` corpus entry captured a CC body whose assistant
turn carried `reasoning_content: ""`; the reader residualised it; the oracle
run died on totality. The KBR-54 slice's `_CORPUS_SKIP_TABLE` excluded the
entry with the F3.c message — fidelity-defensive, not fidelity-correct.

KBR-285's widening comment named exactly the exclusion: "non-first-party on
OpenAI-shaped backends". The bridge is the carrier of `reasoning_content` on
this route (P8's injection is registered and deliberate), so the reader's
residual posture was the asymmetry, not a fidelity defence.

## Why option (a) over (b)

Option (a) — add the slot. Option (b) — strip the field + register row. (b)
breaks Kimi/Z.AI/custom-OpenAI: their wire contracts reject the request
without the injected carrier. The field is a registered, deliberate mutation
those providers require; the honest fix is to model what the wire actually
carries, mirroring the reply direction's established posture.

## Delivered (this branch)

- **`tests/harness/reader_chat_completions.py`** — `_read_one_message` assistant
  branch: `reasoning_content` (string, empty included) projects a `Thinking`
  part appended after the call/text content; the key is added to the consumed
  set. A non-string value keeps the fail-closed posture the unmodelled
  `audio`/`function_call` keys take — residualises at `messages[N]
  .reasoning_content`, `verify_total` raises. The slot is assistant-only
  (P8 injects here; published request shapes place the field on assistant
  messages; a `user`/`tool`-turn carrier residualises on evidence).
- **`tests/harness/test_reader_chat_completions.py`** — four TDD tests
  (R1a-R1d): with-text, empty (P8's exact injection), non-string
  residualise, user-turn residualise. R1a/R1b were RED before the fix;
  R1c/R1d were already GREEN under the fail-closed posture.
- **`tests/corpus/tool_use_and_tool_result.json`** — `triggers_met` gains
  `"thinking_signalled_or_inferred"`. Body file untouched (byte-identical;
  `body_sha256` unchanged). The agent's `thinking: {"type": "adaptive"}` is
  the inbound signal; the translator sets `_thinking_enabled = True` from it,
  P8's site fires, the injection lands — the entry under-declared the trigger.
- **`tests/harness/test_oracle_default_slice.py`** — the F3.c skip row is
  removed; the entry is re-skipped with the F3.a/F3.b reason pointing at the
  KBR-309 sibling. F3.c itself is resolved (totality passes for the entry; the
  KBR-309 drops are what keep it out of the green run). F3.c's row in the
  table moves to a CLOSED strike-through in `TEST_SUITE.md`.

## Implementation notes

- The slot's positional placement (after content parts, not first) is the
  diff-minimisation choice: a positionally-diffed run sees exactly one delta
  (`conversation.turns[i].parts[j]`) when the inbound lacks the part; leading
  with it would shift every existing part and manufacture a delta per
  position. The reply direction reads `reasoning_content` after `tool_calls`
  too — the same "call/text content first, reasoning last" convention in both
  directions of one module.
- The corpus trigger declaration (rather than widening P8 to fire on
  `adaptive_thinking_keys_present`) keeps P8's row tied to its actual
  condition ("thinking signalled or inferred") and treats the entry as the
  site of the under-declaration. Widening the row would couple P8 to P5d's
  `output_config` carrier condition and change the row's meaning for every
  other profile.
- `_with_thinking_carrier` (`src/kitty/bridge/server.py:727-779`) is a
  post-translate repair on the serialised upstream body; it is independent
  of this reader slot (the slot reads the wire the bridge sends; the repair
  mutates that body only after a 4xx rejection). The corpus slice exercises
  the recorder's minimal success path, where the repair never fires, so
  the two are decoupled in the empirical pass; on a real upstream that
  rejects with a thinking round-trip error, the repair short-circuits
  (`server.py:761-762` returns `None` if `reasoning_content is not None`)
  and P8's row continues to claim the deltas.
- F3.c reopens under one observable: an `assistant` message carrying a
  non-empty `reasoning_content` reaches the reader (a multi-turn DeepSeek
  echo-back body). The slot's MIT-style model already handles it — the part is
  `Thinking(text=...)`, exactly the shape the reply direction projects. The
  diff will compare against the inbound Messages-side `Thinking` part from
  the prior assistant turn (which the Anthropic reader projects with
  `signature`); the positional offset is the diff-shape concern, claimed at
  P8's wildcard part path. The KBR-54 slice doesn't yet carry such an entry
  (T-C2 is the corpus coverage). KBR-310 does not add a corpus entry — the
  reader slot is the scope; a corpus entry is T-C2's job.