---
id: kbr223_bedrock_list_content_converse
depends_on: [KBR-89, KBR-222, KBR-264, KBR-169]
---

# KBR-223 — Bedrock list-form content → Converse blocks

Jira: [KBR-223](https://shelpuk.atlassian.net/browse/KBR-223) (High; parent KBR-136 "Bug
Fixes"). Design: `SYSTEM_DESIGN.md` §13. Spec:
`.requirements/20260921T180000Z_bedrock_list_content_converse/REQUIREMENTS.md`.

Scope folds in four ticket comments: KBR-222 (user-branch image mapping owned here),
KBR-169 (list-form truncation gap owned here), KBR-264 (schema oracle pattern reusable),
KBR-89 (oracle re-homed onto `_bedrock_body`).

## Implementation plan (from the REQUIREMENTS, in order)

1. **Shared mapper** in `src/kitty/providers/bedrock.py`: module-level
   `_converse_content_blocks(content)` + `_cc_part_to_converse_block(part)` — pure, Google
   docstrings, table 13.1 as the spec.
2. **`_translate_tool_result_msg`** calls the mapper (R1).
3. **User branch** of `translate_to_upstream` calls the mapper (R2); update the two
   KBR-222 tests to the real-mapping expectation.
4. **`_tool_result_content_size`** in `server.py`; wire into the three truncation sites
   (R4). Watched failing first.
5. **KBR-264 oracle re-driven** through `_bedrock_body`; new shape branches (R3).
6. **Docs** (R5): update the two flipped gap tests' names/docstrings; TEST_SUITE §3.2.1
   M3/M4 prose.
7. Verification: `ruff`, `lint-imports`, `mypy src/kitty`, fast gate, full local suite.

## Verification (target)

- `ruff check .` clean, `mypy src` clean, `lint-imports` clean.
- New + updated L1 tests pass; the two flipped gap tests assert the new posture; the
  string-arm Hypothesis oracles pass unchanged.
- A `git diff --stat` that traces to R1–R5 only.

## Delivered (2026-09-21)

1. **Mapper** — `bedrock.py` gained module-level `_converse_content_blocks` +
   `_cc_part_to_converse_block` (+ `_decode_base64_payload`,
   `_image_block_from_media_type`, `_image_from_data_url`, `_document_block` and the
   two media-type tables). Part kinds: text; Anthropic-native base64 image; CC
   `image_url` data-URL (dict **or** plain-string URL spelling); Anthropic base64
   document (`name` = `title` or `"document"`; DocumentBlock requires it). Parsing per
   spec: lowercase scheme/format, `image/jpg`→`jpeg`, whitespace stripped,
   `b64decode(validate=True)`, drop on error. Non-dict / wrong-typed parts → `None`
   → dropped; all-dropped → `[{"text": ""]` fallback.
2. **Wire-ins** — `_translate_tool_result_msg` and the user branch of
   `translate_to_upstream` call the mapper; the string arms are byte-identical to
   pre-fix output; non-str-non-list user content still passes through untouched
   (pre-existing posture, unchanged).
3. **Truncation** — `server.py` gained `_tool_result_content_size` (string → len;
   list → serialized JSON len; unmeasurable → -1, F33 parity). Wired into
   `_truncate_oversized_tool_results` (both arms), `_truncate_oversized_responses_outputs`,
   `_compact_messages` step 1 (CC-shape only; M4's posture pinned by a comment). An
   over-limit list collapses to the standard notice string.
4. **Oracle** — KBR-264's `test_assistant_reasoning_block_passes_bedrock_schema_validation`
   re-driven through `_bedrock_body` (re-attaching `modelId`); new
   `TestBedrockListContentOracle` adds tool-result (text / image+document) positives
   and verbatim-copy negatives, plus user-branch positive + negative — same
   `validate_parameters`-on-`Converse.input_shape` oracle.
5. **Docs** — `SYSTEM_DESIGN.md` §13 (design of record); TEST_SUITE §3.2.1 M3/M4 rows
   and §9.2 G42 updated to the closed state (bedrock removed from G42's still-dropping
   list; `ollama_cloud`/`openai_subscription` corridors stay); flipped tests
   renamed/documented: the two KBR-222 flatten tests became real-mapping tests,
   `TestNonToolContentUntouched` keeps three "survives" pins on structured-free
   fixtures, and `TestStructuredContentTruncated` asserts the new truncation posture.
6. **Verification** — `ruff` clean; `lint-imports` 5/5 kept; `mypy src/kitty` clean
   (94 files); bedrock file 112 passed; truncation suites (stage11 + property
   file's truncation classes) all green; Fast gate + full suite on CI.

## Notes

- The Hypothesis generators for the truncation property are string-only (verified:
  substrate strategies draw string tool content, generators append string pairs), so
  the `_apply_*_truncation` replicas stayed string-only; both sides agree by being
  string-only. The pin lives in REQUIREMENTS R4.
- DocumentBlock's `name` has no length/pattern constraint on the installed model
  (measured); `title or "document"` needs no UUID fallback.
- An empty `toolResult.content` list is validator-clean (measured) — the
  `[{"text": ""}]` fallback is defence-in-depth for the documented shape, not a
  client-side requirement.
- **Mutation surface.** The new pure helpers widen the `bedrock_transport` mutation
  surface; flag for [KBR-91](https://shelpuk.atlassian.net/browse/KBR-91)'s
  re-measure after this PR lands. No action in this PR — the bedrock_transport
  baseline row is `_pending_`.
