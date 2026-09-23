---
id: kbr305_rebuild_trio_sampling_keys
depends_on:
  - kbr301_gemini_generation_config
---

# KBR-305 — Carry the six KBR-301 sampling keys at hop 2 on the rebuild-trio adapters (Anthropic family + Bedrock + Ollama Cloud)

Plan: `.requirements/20260923T161010Z_kbr305_rebuild_trio_sampling_keys/REQUIREMENTS.md`.

## What

Close the hop-2 leg of KBR-301. KBR-301 made `GeminiTranslator.translate_request`
carry **all eleven** published Gemini `generationConfig` sampling fields
onto the Chat Completions body at the bridge's first hop. On the
**verbatim-forwarder** routes those fields now reach the upstream; on the
three **rebuild-from-allowlist** adapters they still die at hop 2, because
each rebuild reads only the keys its destination wire accepts.

Per-adapter disposition, verified against each destination's published
schema or source (Anthropic Messages `platform.claude.com/docs/en/api/messages`
accepts none of the six; Bedrock `InferenceConfiguration` is exactly
`{maxTokens, temperature, topP, stopSequences}`;
[Ollama `ChatRequest` + `Options`](https://raw.githubusercontent.com/ollama/ollama/main/api/types.go)
on main carries `Logprobs bool`/`TopLogprobs int` top-level and
`Seed`/`PresencePenalty`/`FrequencyPenalty` under `options`):

| CC key | Anthropic family (5 entries) | Bedrock | Ollama Cloud |
|---|---|---|---|
| `seed` | drop | drop | carry → `options.seed` |
| `presence_penalty` | drop | drop | carry → `options.presence_penalty` |
| `frequency_penalty` | drop | drop | carry → `options.frequency_penalty` |
| `logprobs` (bool) | drop | drop | carry → top-level `logprobs` |
| `top_logprobs` (int) | drop | drop | carry → top-level `top_logprobs` |
| `n` | drop | drop | drop (no Ollama equivalent) |

## Why

The bridge level already carries the six. The remaining defect class is
G28's (`top_k` dropped on every non-Anthropic-family route, KBR-178) — the
rebuild allowlists read no internal key for these six, so they die at hop 2
the same way `top_k` did pre-KBR-178. KBR-301 explicitly recorded the
residue as gap row G44's "second residue":

> The six newly carried keys survive only on the verbatim forwarders (the
> OpenAI family et al.); the rebuild-from-allowlist adapters drop them at
> hop 2 — `AnthropicAdapter.translate_to_upstream` reads only
> `model`/`max_tokens`/`stream`/`temperature`/`top_p`/`stop`/`_top_k` from
> the CC body, `BedrockAdapter` the same rebuild shape, `OllamaCloudAdapter`
> carries only `temperature`/`top_p`/`top_k` into `options` — so on those
> routes a Gemini client's `seed`/`n`/`logprobs` request still dies, one
> hop later than before KBR-301.

The T-D5 corpus will drive such a request through the rebuild trio, and
the §3.3.1 oracle would then report a **false** I1 breach on a deliberate
drop.

`G44`'s row text already names the remedy options: either widen G28's row
or land a sibling row at the same anchor. We land a new provider-level
row P43 at the adapter sites. Reasoning recorded in `REQUIREMENTS.md`:
the P31/P32 split ("paths must be true of every site a row names") would
otherwise force one row per adapter family, but the mutmut scope
(`TEST_SUITE.md` §6.1 — `kitty.providers.* translate_to_upstream`) gives
the L1 carry suite the discriminating duty on the five ollama carries,
and the owner confirmed the one-row shape on 2026-09-23. The accepted
consequence is spelled: P43 has no discriminating power on the five
carried keys for the `ollama_cloud` route — the L3 oracle stays green on
a carry regression there, and the L1 suite carries that duty.

The reader widening of `tests/harness/reader_ollama.py` is the L3-oracle
hard blocker: once the adapter forwards `logprobs`/`top_logprobs` as
top-level Ollama-body keys, the reader's else-branch sends unknown
top-level keys to `residual`, and a non-empty residual raises
`ResidualFieldsError` at the oracle's `verify_total` totality gate — so
T-D5 cannot drive such a corpus entry at all until the reader widens.
The reader is total over the published format (the documented "no
inner-type validation" posture, `reader_ollama.py:622-625`, mirrored from
`reader_chat_completions.py:613-615`), independent of whether the bridge
forwards the keys.

## Implementation notes

Implemented in PR (branch `fix/kbr-305-rebuild-trio-sampling-keys`,
off `origin/main` 2026-09-23).

**Production edit — `src/kitty/providers/ollama_cloud.py::translate_to_upstream`.**
The three options keys (`seed`, `presence_penalty`, `frequency_penalty`)
join the existing `temperature`/`top_p`/`top_k` loop with the loop's
existing `key in cc_request and cc_request[key] is not None` guard. Two
top-level carries added: `logprobs` strict bool (mirrors hop-1
`translator.py:288`); `top_logprobs` `int` and not `bool` (mirrors
`translator.py:291`). `n` stays dropped. 22 new tests in
`TestOllamaCloudSixSamplingKeys` (`tests/test_provider_ollama_cloud.py`):
6 parametrisations on the three options keys (incl. `0`/`0.0`/`-0.25`
to pin that the loop is presence+non-null, not truthiness);
`logprobs: True` and `False` both land (the one falsy-but-meaningful
value in the six); `top_logprobs: 5` lands; bool-typed `top_logprobs`
does NOT land; null-omitted and absent-omitted controls; `n` stays
dropped and the adapter invents no `num_predict`/`num_choices`
spelling.

**Reader widening — `tests/harness/reader_ollama.py`.** New
`_TOP_LEVEL_SAMPLING_KEYS = frozenset({"logprobs", "top_logprobs"})`
constant and a new elif in `_project` (sibling to the existing
`_PUBLISHED_EXTRA_KEYS` branch, before `options`). Projection is
verbatim — the family's documented "no inner-type validation"
posture (`reader_ollama.py:622-625`, mirrored from
`reader_chat_completions.py:613-615`). 5 new tests in
`TestOllamaTopLevelLogprobs` (`tests/harness/test_reader_ollama.py`):
`logprobs: True`/`False` both project to `conversation.sampling[logprobs]`
(no residual); `top_logprobs: 5` projects to `[top_logprobs]`;
bool-typed `top_logprobs` projects verbatim to `[top_logprobs] = True`;
absent → absent.

**Drop pins — Anthropic family + Bedrock.** 6 new tests in
`TestAnthropicFamilySixSamplingKeysDrop`
(`tests/test_provider_anthropic.py`) — parametrised over the five
family adapters (AnthropicAdapter + 4 delegates, with OpenCodeGoAdapter
on its Messages-routed model `minimax-m2.7`, confirmed in
`_MESSAGES_MODELS` not `_RESPONSES_MODELS`) — assert none of the six
keys reach the Messages body, and that no renamed Messages spelling
(`num_choices`, `response_logprobs`) is invented. 2 new tests in
`TestBedrockSixSamplingKeysDrop` (`tests/test_provider_bedrock.py`)
assert none of the six keys reach the Converse body (either top-level
or under `inferenceConfig`), and that no camelCase Converse spelling
(`numChoices`, `presencePenalty`, `frequencyPenalty`, `topLogprobs`,
etc.) is invented — the assertion runs at `_bedrock_body`, the §3.2.3
capture boundary.

**Register row P43 — `tests/harness/register.py` + `.system_design/TEST_SUITE.md`.**
A new `MutationRow` with six `c.sampling_path(...)` entries, scope
`_ANTHROPIC_FAMILY + ("bedrock", "ollama_cloud")`,
`Trigger.ALWAYS`, three-site tuple. The exact tuple is pinned by a
new assertion in `test_register.py::test_row_ids_are_unique`
(`register_disagreements` compares ids/conditionality/order but not
path content; without the pin a data-side drift goes uncaught). The
markdown row P43 carries the per-adapter split (Anthropic family +
Bedrock drop all six; Ollama Cloud carries five, drops only `n`) and
the ⚠️-marked P31/P32 departure (one row, dormant claims on Ollama
for the five carries; the L1 carry suite is the discriminating guard
on that route). The §3.2.2 closing paragraph's unconditional list
extended (`P43` added). Count pins updated 79→80 (total), 52→53 (P),
48→49 (unconditional list). `G44` §9.2 cell updated to record the
remedy taken (sibling-row P43, not the G28-widening alternative);
first residue (five control fields) remains open with its deferral.

**Mutation probe results.** Six deliberate mutations were
independently applied and reverted (review discipline); each was
caught:

| Mutation | Site | Caught by |
|---|---|---|
| Carry `n` as `options.num_predict` | `OllamaCloudAdapter.translate_to_upstream` | `TestOllamaCloudSixSamplingKeys::test_n_stays_dropped` |
| Restore `seed` on Anthropic | `AnthropicAdapter.translate_to_upstream` | `TestAnthropicFamilySixSamplingKeysDrop::test_each_key_stays_absent` (5 of 6 family parametrisations) |
| Reader mis-project `logprobs`/`top_logprobs` to `envelope.extra` | `reader_ollama._project` | `TestOllamaTopLevelLogprobs` (4 of 5 tests) |
| Truthiness guard on `logprobs` (swallows `False`) | `OllamaCloudAdapter.translate_to_upstream` | `test_logprobs_lands_top_level[False]` |
| Drop the bool exclusion on `top_logprobs` (leaks `True` as int 1) | `OllamaCloudAdapter.translate_to_upstream` | `test_top_logprobs_bool_does_not_land` |
| Drop `is not None` on the options loop (leaks `None`) | `OllamaCloudAdapter.translate_to_upstream` | `test_options_key_null_omitted` (3 parametrisations) |

**Third-party verification.** Ollama `ChatRequest` + `Options` claims
read live against `ollama/api/types.go` on main (2026-09-23):
`Logprobs bool json:"logprobs,omitempty"` (line 173),
`TopLogprobs int json:"top_logprobs,omitempty"` (line 178),
`Seed int`, `PresencePenalty float32`, `FrequencyPenalty float32` under
`Options` — and **no `n`/`num_choices` field anywhere** (the ticket's
"options.n" half is wrong; verified absent in the struct). Anthropic
Messages verified against `platform.claude.com/docs/en/api/messages`:
no `seed`/`n`/penalties/logprobs in the schema. Bedrock
`InferenceConfiguration` verified against AWS API reference: exactly
`{maxTokens, temperature, topP, stopSequences}`.

**Gates at merge-time:** `ruff check .` clean; `mypy src/kitty` clean
(94 source files); `lint-imports` clean; full `pytest` clean (9073
passed, 16 skipped, 45 deselected, 36 warnings in 30:32); harness
register guards clean (122/122).