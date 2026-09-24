---
id: kbr309_messages_cc_extra_drops
depends_on: [KBR-54, KBR-184, KBR-214, KBR-218, KBR-224, KBR-258, KBR-263]
---

# KBR-309 — Messages→CC translation drops `context_management` and `metadata` (no register row)

Jira: **KBR-309** ([link](https://shelpuk.atlassian.net/browse/KBR-309)).
Design: `TEST_SUITE.md` §3.2.1, §3.2.2, §3.3.1b (CC reader's `_PUBLISHED_EXTRA_KEYS`),
§3.3.2 (oracle), §7.4.4 (slice composition). Requirements:
`.requirements/20260924T083605Z_kbr309_messages_cc_extra_drops/REQUIREMENTS.md`.

## What the task does

Closes the KBR-54 scope-addition follow-up: the corpus-driven default-transport
slice (T-D4, `tests/harness/test_oracle_default_slice.py`) reported four
unclaimed deltas on `plain_turn` and `effort_configured` —
`envelope.extra[context_management]`, `envelope.extra[metadata]`,
`conversation.turns[0].parts[0].text`, `conversation.turns[0].parts[1]`. The
ticket's AC-2 requires the four to vanish from the slice's skip table.

The fix has three parts, none in `src/kitty/`:

1. **`_triggers_met` includes `Trigger.ALWAYS`.** M26 — the existing row of
    record for the Anthropic-family `metadata` drop (KBR-184 / G31) — is
    `conditional=False` and was dormant because the slice's helper didn't
    include `Trigger.ALWAYS`. Adding ALWAYS activates it without any
    register change. The `tests/harness/test_oracle_driven.py` precedent
    (KBR-307) already includes ALWAYS at its driven-slice call site; T-D4
    simply missed it.
2. **M27 (new)**: `envelope.extra[context_management]` drop on the translated
    Messages route. The field is Anthropic **beta** (requires
    `anthropic-beta: context-management-2025-06-27`); the bridge builds
    upstream headers from scratch and forwards no inbound agent header
    (§4.2 C1), so a KBR-224-style restore would 400 at the upstream. The
    drop is the design posture, not a bug.
3. **M28 (new)**: the multi-text-part user-turn join collapse. The captured
    CC body carries one text part whose text differs from the inbound's
    first text part (`parts[*].text` delta) and whose collection is one
    shorter (`parts[*]` whole-part delta). KBR-296 deliberately scoped the
    `carry_cache_control=True` extension to the M9 fallback path; KBR-258 /
    KBR-263 closed the cache_control follow-ups as register rows rather than
    a hop-1 expansion, and KBR-309 closes the join's surface deltas the same
    way. The bare `parts[*]` anchor over-claims (would also claim a *deleted*
    part — §3.3.1's own falsification case); the trade-off is documented in
    the row text.

## Decisions

- **No `src/kitty/` changes.** Every drop is the design of record: `context_management`
  needs the beta header (not in scope); `metadata`'s drop is M26's charter (KBR-184);
  the join is KBR-222's deliberate posture with KBR-296's deliberate M9-only
  scope. The ticket's hint at "the natural fix pattern is the same as
  KBR-224's" was correct for the *idea* but the natural pattern fails for
  `context_management` without the beta header and is exactly the shape
  KBR-258 / KBR-263 already rejected for the cache_control follow-ups.
- **M28's bare `parts[*]` anchor.** The user-approved trade. The owner is
  on record that the alternative (a `carry_cache_control` expansion to
  hop 1) is KBR-258/KBR-263 territory and resolved as register rows.
  Equal-anchor co-claim with M3/M4/M8/M17 is the §3.3.1 docstring's third
  example: those rows are conditional, M28 is not, and the co-claim is
  exempt from a real bug by construction. Documented in M28's reason cell.
- **M28's two-site tuple** (`build_user_content_message` + `MessagesTranslator.translate_request`):
  the row's surface is the join inside `build_user_content_message`, but the
  join is invoked from `_translate_user_message` inside
  `MessagesTranslator.translate_request`. The first site names the
  mutation surface; the second names the translator that drives it.
  The `site ↔ scope` guard accepts both.
- **`_EXPECTED_CLAIMED_DELTAS` is per-entry, not a single tuple.** Both
  `plain_turn` and `effort_configured` carry `model: "MiniMax-M3"` while
  the harness profile resolves to `"harness-model"`, so both ship
  `envelope.model` as a delta. The captured body's `MiniMax-M3`-shaped
  normalisation keeps them identical at ten deltas. Pinning the walk order
  matters because `_structural_diff` walks deterministically (envelope →
  `envelope.extra` → `conversation.system` → `conversation.turns`), and a
  drift in walk order or claim would turn the tuple into a falsification
  surface.
- **`Trigger.ALWAYS` in `_triggers_met` is safe on the six passing entries.**
  None of the six currently-passing corpus entries (`format_example`,
  `m6_recovery_oversized_paired`, `m6_recovery_oversized_paired_streaming`,
  `m5_irreducible_single_final_turn`, `no_output_config`,
  `compaction_budget_over`) carry `metadata`, `context_management`, or a
  multi-text-part user turn; M26 / M27 / M28's claims are dormant on their
  bodies. Adding ALWAYS flips M26 onto the four entries that need it
  (plain_turn, effort_configured) and is no-op on the rest. Verified by
  re-running the slice: 10 passed / 6 skipped after the revert, identical
  to before for the six passing entries (they still assert
  `report.deltas == ("envelope.model",)`).

## Companion sites

Six silent companion sites updated alongside the two new rows (KBR-271's
lesson, recorded in `~/.claude/.../MEMORY.md` "Register-edit companion sites"):

| Site | Old | New |
|---|---|---|
| `tests/harness/test_register.py:132` docstring | "80 live rows" | "82 live rows" |
| `tests/harness/test_register.py:135` docstring | "27 bridge-level rows" | "29 bridge-level rows" |
| `tests/harness/test_register.py:158` count pin | `== 80` | `== 82` |
| `tests/harness/test_register.py:_SHAPES` | (no `text` entry) | `(part_path(WILDCARD, WILDCARD, "text"), part_path(2, 0, "text"))` |
| `tests/harness/test_register_agreement.py:303` M-count pin | `== 27` | `== 29` |
| `tests/harness/test_register_agreement.py:310` unconditional count pin | `== 49` | `== 51` |
| `.system_design/TEST_SUITE.md` §3.2.1 row-count prose | "Twenty-eight rows… twenty-seven live" | "Thirty rows… twenty-nine live" |
| `.system_design/TEST_SUITE.md` §3.2.1 table | (no M27 / M28 entries) | M27 + M28 rows |
| `.system_design/TEST_SUITE.md` §3.2.2 unconditional list at line 275 | (no M27 / M28) | appended `, M27, M28` |

The `test_register_agreement.py` `test_register_holds_every_live_row` /
`test_the_parser_reads_both_tables` /
`test_an_unconditional_list_that_omits_a_row_is_caught` /
`test_an_unconditional_list_naming_a_row_that_does_not_exist_is_caught`
guards flip red on any future drift in either direction.

## Implementation notes

Implemented (2026-09-24); branch `fix/kbr-309-messages-cc-extra-context-metadata`
off `origin/main` `554a653`.

Verified at merge:
- `ruff check .` clean on `src tests` (the `.cache/` tree is gitignored; the
  leftover `.cache/curl_cffi_probe/` artifacts are pre-existing, not from
  this ticket).
- `mypy src/kitty` clean (no `src/kitty/` changes touched).
- `pytest tests/harness/test_register.py tests/harness/test_register_agreement.py tests/harness/test_oracle_default_slice.py tests/harness/test_oracle.py tests/harness/test_oracle_driven.py` —
  123 + 49 + 10 (6 skipped) + 59 = 241 passed / 6 skipped.
- `pytest -m "l1 or l2"` — 9050 passed / 22 skipped / 117 deselected
  (the agent_smoke / agent_live / eval / load markers are excluded per
  `pyproject.toml`'s `addopts`).
- `scripts/regenerate_step_index.py` exits 0 (this step file added).

Open follow-ups filed separately, deliberately out of scope:
- **KBR-310** (the sibling ticket already filed from KBR-54's close-out):
  `messages[N].reasoning_content` residualised by the CC reader (no slot
  in T-A2's grammar; KBR-285 didn't touch it). F3.c in the skip table.
- The `conversation.turns[2]` whole-tool_result-turn drop on `tool_result_under_limit` /
  `tool_result_over_limit` is filed as a scope-add on KBR-272 (its
  `compaction_and_pairing` territory). F3.d in the skip table.
- The `compaction_budget_under` calibration gap (README documents the 2.8 M
  static threshold; the runtime default-profile budget is 800 K chars, so
  the entry is no longer a clean M5 complement) is medium, no related To Do.
- The framing-gap exclusions (`tools_declared` role-system-in-messages;
  `system_prompt_over_window_compacts_normally` 400 at ingress) are
  entry-shape issues, not product defects — no ticket.
- `context_management` beta-header carriage (the rationale that drives
  M27's drop rather than a KBR-224-style restore) is a separate product
  decision, owned elsewhere.