---
id: kbr301_gemini_generation_config
depends_on: []
---

# KBR-301 — carry the remaining published `generationConfig` sampling fields on the Gemini ingress; register the five named drops

Plan: `REQUIREMENTS.md` at
`.requirements/20260922T224102Z_kbr301_gemini_generation_config/`.

## What

`GeminiTranslator.translate_request` now reads **all eleven** published
`generationConfig` sampling fields (the harness reader's `_SAMPLING_KEYS`,
`tests/harness/reader_gemini.py:138-149`), up from KBR-213's five:

- `candidateCount` → `n` — guard `isinstance(v, int) and not isinstance(v, bool)`.
- `presencePenalty` → `presence_penalty`, `frequencyPenalty` →
  `frequency_penalty` — `_NUMBER` guard (`int | float`, bool excluded).
- `seed` → `seed` — `(int,)` not bool.
- `responseLogprobs` → `logprobs` — `(bool,)` strict; an int `1` is
  rejected, mirroring the harness reader's strict typing
  (`reader_gemini.py:148`).
- `logprobs` → `top_logprobs` — `(int,)` not bool.

The last two are the **name collision** the ticket calls the only design
question: Gemini `logprobs` (integer count) and `responseLogprobs`
(boolean flag) map onto the *other* spelling on the CC side; carrying
either onto its own spelling would silently break every logprobs request
on a Gemini route.

The five format-specific control fields named in the ticket —
`responseMimeType`, `responseSchema`, `thinkingConfig`,
`mediaResolution`, `speechConfig` — stay dropped: Chat Completions
declares no equivalent, and the harness reader's own comment
(`reader_gemini.py:152-162`) records why no fold onto `response_format`
exists. The omission is registered as gap row **G44** in
`TEST_SUITE.md` §9.2 (deferred Before T-D5, G28's shape), with the nine
newer `_GENERATION_EXTRA_KEYS` entries recorded as out-of-scope in the
row text.

## Why

KBR-213 fixed two of the five-then-known dropped sampling fields and
filed the rest. The harness reader (the I1 oracle's projection) has
always projected all 25 published keys — the production side was the
only seam whose inbound projection and outbound body disagreed. The
design of record for the guard discipline is the harness reader's
`_typed_leaf` (`reader_gemini.py:915-960`): presence + type + the
bool/int subclass exclusion; production mirrors it inline, the same way
KBR-213's `topK` guard does.

`TEST_SUITE.md` is pinned by `tests/harness/test_reader_gemini.py::TestSchemaAgreement`
at `SCHEMA_VERSION == "20260910"` — the 25-key set is test-enforced, so
the production mappings cannot silently drift from the projection.

## Implementation notes

(2026-09-22) TDD: 35 new tests in
`tests/test_gemini_translator.py::TestTranslateRequestGenerationConfig`
(5–6 per new mapping: lands / zero-lands / none-omitted / bool-omitted /
wrong-type-omitted; the two collision tests assert the positive landing
AND the wrong-address absence, so a swap turns exactly one red; one
drop test pinning the five control fields absent from the CC body; one
all-eleven combined reproduction). 14 watched failing on the base
(`KeyError` on the missing CC keys — red for the right reason); 20
already green (the omission pins hold today). Then the production edit:
four simple mappings + the collision pair, two banner comments, one
drop-pointer comment, inserted after KBR-213's `topK` block.

Self-caught test bug during the green run: the combined test's
wire-leak list originally included `seed` and `logprobs` — but `seed`
is the one sampling key whose Gemini and CC spellings are identical,
and `logprobs` legitimately appears on the CC body when the request
carries `responseLogprobs` (the collision mapping working as designed).
Both removed from the wire-only list with an explanatory comment; the
collision swap detection lives in the per-field tests, which send one
member of the pair and assert the wrong address absent.

Code-review additions: the drop test gained `assert "response_format"
not in cc` (+ a `MEDIA_RESOLUTION_HIGH` value check) — a mutation probe
during review proved the plausible fold mutant (rewrite
`responseMimeType` onto `response_format`) escaped every key-absence
check, so the docstring's "would turn this test red" claim was false
until that assertion landed; `test_frequency_penalty_int_lands_on_
frequency_penalty` added for coverage symmetry with presencePenalty's
int arm; bool-loop asserts carry `f"boolean={boolean!r} leaked"`
messages for failure attribution; sub-banners converted to the file's
box-drawing style.

`ruff check .`, `lint-imports`, `mypy src/kitty`, full pytest suite
green on Linux.
