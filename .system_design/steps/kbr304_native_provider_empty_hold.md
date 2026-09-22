---
id: kbr304_native_provider_empty_hold
depends_on: [kbr300_non_streaming_silent_skeleton_sweep]
---

# KBR-304 — Non-streaming silent-skeleton residual: native-provider backends (the KBR-300 follow-up)

Ticket: [KBR-304](https://shelpuk.atlassian.net/browse/KBR-304) (KBR-136 epic, High).
The native-provider carve-out KBR-300 recorded as a deliberate non-fix and filed
as this ticket. Requirements: `.requirements/20260922T223855Z_kbr304_native_provider_empty_hold/REQUIREMENTS.md`.

## What

The four KBR-300 non-streaming gates use the literal predicate

```
not self._active_provider.use_native_messages and self._is_empty_cc_response(cc_response)
```

The `not use_native_messages` conjunct deliberately excluded native
(`use_native_messages = True`) providers — the KBR-298 native-arm carve-out,
generalised to the three sibling routes. Four cells remained open on
`origin/main` after KBR-300 merged (PR #267):

- `_handle_messages` × the KBR-237 tool-use-format fallback cell — a native
  Anthropic provider whose `_native_messages_request` was cleared by the
  KBR-237 fallback answers in CC form; an empty reply there falls through the
  widened elif into `MessagesTranslator.translate_response`'s fabricated
  fallback. The cell KBR-300 §5.4 named.
- `_handle_responses` × native providers — `ResponsesTranslator`'s
  fabricated fallback ships as a `200`, billed once.
- `_handle_gemini` × native providers — `GeminiTranslator` ships an empty
  text part (`candidates[0].content.parts[0].text == ""`), billed once.
- `_handle_chat_completions` × native providers — verbatim CC body ships as a
  billed empty `200` skeleton.

KBR-304 drops the `not use_native_messages` conjunct from each of the four
gates; the new predicate is `self._is_empty_cc_response(cc_response)`. One
conjunct per handler, no new ladder logic, no per-route D4 body change. The
native-Messages arm of `/v1/messages` (the `if cc_response.get("type") ==
"message"` branch) stays structurally out of the gate — the widening affects
only CC-shaped replies.

Per-route D4 bodies unchanged from KBR-300 (route-protocol-native `502`
shapes, recorded in `REQUIREMENTS.md` D3).

## Why

- *One-conjunct drop per handler, in place.* Same surgical shape as KBR-300:
  one token removed at each of the four gate sites. No new ladder logic, no
  parallel gate, no structural change.
- *Reuse the KBR-300 harness, no new third-party configuration.* Each of
  the four KBR-300 test files gains a native-provider variant driven by a
  test-only `_NativeOpenAIAdapter(OpenAIAdapter)` stub whose only override
  is `use_native_messages -> True`. The stub deliberately leaves
  `upstream_wire_shape` at the inherited `CHAT_COMPLETIONS` — the base-class
  invariant `use_native_messages ⇒ WireShape.MESSAGES` is a
  production-adapter rule (`src/kitty/providers/base.py:643-657`), and the
  test exercises only the gate's conjunct dimension.
- *CC-shape fallback on /v1/messages is reached structurally, not via the
  KBR-237 fallback machinery.* The KBR-237 fallback is streaming-only
  (`server.py:4599`, `6045`, `7846`, `9467`); on non-streaming the handler
  sets `_native_messages_request = True` for a native provider and the
  reply's shape decides the branch. A CC-shaped reply from a native
  provider lands in the elif regardless of whether the KBR-237 fallback
  cleared the flag — the cell is reached by structural shape, not by the
  streaming-only fallback. The earlier doc draft's "the KBR-237 fallback
  path is exercised naturally" wording was inaccurate; the system-design
  reviewer caught it and the doc was reworded.
- *Reasoning-only trade-off carries over unchanged.* Native providers on the
  sibling routes answer in CC form, so a `reasoning_content`-only reply is
  judged non-empty (KBR-277) and released — same behaviour as the existing
  raw-CC reasoning test. The native parametrisation does not add a
  reasoning-only cell; the existing KBR-300 raw-CC reasoning test on each
  route already pins the trade-off.

## Tests

Four sets of new tests across the four KBR-300 test files, sixteen tests in
all:

- `tests/bridge/test_messages_raw_cc_non_streaming_empty_hold.py` —
  `test_a_content_bearing_native_completion_reaches_the_client`,
  `test_an_empty_native_completion_ends_in_the_d4_terminal`,
  `test_a_tool_call_only_native_completion_releases_the_verdict`,
  `test_an_empty_native_attempt_crosses_to_a_healthy_plain_peer`.
- `tests/bridge/test_responses_non_streaming_empty_hold.py` — the same four
  tests, on `/v1/responses`.
- `tests/bridge/test_gemini_non_streaming_empty_hold.py` — the same four
  tests, on `/v1/gemini`.
- `tests/bridge/test_chat_completions_non_streaming_empty_hold.py` — the same
  four tests, on `/v1/chat/completions`.

The KBR-300 custom-transport cells, the KBR-300 plain-CC cells, and the merged
KBR-298 messages file are unchanged and re-pin the non-native behaviour
against the widened gate.

The three sibling-route files' `_post_plain()` helper gains a
`provider_factory=OpenAIAdapter` keyword-only parameter; the messages file's
`_post()` helper already takes a `provider` positional, so no helper change
was needed there.

## Implementation notes

- **One-conjunct drop per handler, four sites.** Server-side diff is four
  one-token edits: each gate changes from
  `if not self._active_provider.use_native_messages and self._is_empty_cc_response(cc_response):`
  to `if self._is_empty_cc_response(cc_response):`. The four gate comments
  were rewritten together to retire the "one-conjunct drop away" / KBR-237
  cell notes KBR-300 recorded.
- **Per-AC pre-fix evidence on `an_empty_native_completion_ends_in_the_d4_terminal`.**
  The captured log on the unfixed code reads
  `"Empty upstream response after 5 attempts, returning fallback"` — the M12
  fabricated fallback firing. That is the KBR-300 carve-out's As-Is
  behaviour, present on every native route today. The gate's RED step
  observes this directly.
- **`healthy_log` pre-fix asymmetry on the Chat Completions cell.** The CC
  non-streaming arm does not call `_mark_backend_healthy` (the KBR-300
  recorded asymmetry). The system-design reviewer caught a doc error where
  the AC-FR-4.2 pre-fix clause claimed `healthy_log` length 1; the correct
  pre-fix description is `usage_log` length 1, `healthy_log` empty. The
  existing KBR-300 chat-completions test docstring carries the same error
  (it asserts `usage_log == []` and `healthy_log == []` for the post-fix
  without recording the asymmetry) — out of scope to retouch in this PR.
- **One library not imported.** No new imports across the four test files —
  the `_NativeOpenAIAdapter` stub uses `OpenAIAdapter`'s imports as-is.
- **No system-design follow-up filed.** The widened gate covers every
  provider class on every route. The native-Messages arm of `/v1/messages`
  remains structurally out of the gate (the `if type == "message"` branch
  handles it without translating); the recorded residual is unchanged.
- **Verification:** the four KBR-300 test files (145 existing tests) plus
  the KBR-298 messages file and the merged `tests/bridge/test_empty_response_retry.py`
  suite ran 161 passed, 0 failed, 0 deselected (~4 minutes wall-clock); the
  16 new tests went RED on the unfixed code (`status == 200` from the M12
  fallback) and GREEN after the four conjunct drops. `ruff check .`,
  `mypy src/kitty` clean on the diff. `python3 scripts/regenerate_step_index.py`
  exits 0.