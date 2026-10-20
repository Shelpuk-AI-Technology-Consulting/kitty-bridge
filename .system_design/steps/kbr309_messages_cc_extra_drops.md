---
id: kbr309_messages_cc_extra_drops
depends_on: [KBR-54, KBR-184, KBR-214, KBR-218, KBR-224, KBR-258, KBR-263]
---

# KBR-309 — claim the Messages→CC `context_management` drop (register row M33) and activate M26 for `metadata`

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
2. **Register row M33** (first-drafted M27, renumbered M30 → M33 across the
    KBR-55 and KBR-59 id collisions — see the merge-conflict note):
    `envelope.extra[context_management]` drop on the translated Messages
    route. The field is Anthropic **beta** (requires
    `anthropic-beta: context-management-2025-06-27`); the bridge builds
    upstream headers from scratch and forwards no inbound agent header
    (§4.2 C1), so a KBR-224-style restore would 400 at the upstream. The
    drop is the design posture, not a bug.
3. **Parts-join row withdrawn** (first-drafted M28, renamed M31, dropped in
    `37c282b`). The first draft registered the multi-text-part user-turn
    join collapse (`parts[*].text` + bare-`parts[*]` deltas). KBR-308 then
    expanded `carry_cache_control=True` to hop 1, closing the join on every
    cache_control-bearing multi-part turn — the two `conversation.turns[*]`
    deltas no longer occur on `plain_turn` / `effort_configured` at all —
    and the bare `parts[*]` anchor would over-claim a *deleted* part
    (§3.3.1's own falsification case,
    `test_oracle_curl_cffi.py::test_inbound_content_survives_while_introduced_is_caught`).
    With the deltas gone and the anchor net-negative, the row was withdrawn
    rather than landed.

## Decisions

- **No `src/kitty/` changes.** Every drop is the design of record: `context_management`
  needs the beta header (not in scope); `metadata`'s drop is M26's charter (KBR-184);
  the join is KBR-222's deliberate posture with KBR-296's deliberate M9-only
  scope. The ticket's hint at "the natural fix pattern is the same as
  KBR-224's" was correct for the *idea* but the natural pattern fails for
  `context_management` without the beta header and is exactly the shape
  KBR-258 / KBR-263 already rejected for the cache_control follow-ups.
- **Parts-join row withdrawn rather than anchored.** The first draft carried
  the user-approved bare-`parts[*]` trade; KBR-308's hop-1
  `carry_cache_control=True` expansion (landed on main while the PR was
  open) removed the join's deltas on the slice entries entirely, and the
  anchor's over-claim on §3.3.1's deleted-part falsification case made the
  row net-negative — so it was withdrawn instead of landed. KBR-258 /
  KBR-263's register-row precedent for the cache_control follow-ups
  stands.
- **`_EXPECTED_CLAIMED_DELTAS` is per-entry, not a single tuple.** Both
  `plain_turn` and `effort_configured` carry `model: "MiniMax-M3"` while
  the harness profile resolves to `"harness-model"`, so both ship
  `envelope.model` as a delta. The captured body's `MiniMax-M3`-shaped
  normalisation keeps them identical at eight deltas. Pinning the walk order
  matters because `_structural_diff` walks deterministically (envelope →
  `envelope.extra` → `conversation.system` → `conversation.turns`), and a
  drift in walk order or claim would turn the tuple into a falsification
  surface.
- **`Trigger.ALWAYS` in `_triggers_met` is safe on the six passing entries.**
  None of the six currently-passing corpus entries (`format_example`,
  `m6_recovery_oversized_paired`, `m6_recovery_oversized_paired_streaming`,
  `m5_irreducible_single_final_turn`, `no_output_config`,
  `compaction_budget_over`) carry `metadata`, `context_management`, or a
  multi-text-part user turn; M26 / M33's claims are dormant on their
  bodies. Adding ALWAYS flips M26 onto the four entries that need it
  (plain_turn, effort_configured) and is no-op on the rest. Verified by
  re-running the slice: 10 passed / 6 skipped after the revert, identical
  to before for the six passing entries (they still assert
  `report.deltas == ("envelope.model",)`).

## Companion sites

The companion sites of record after the 2026-10-18 merge of `origin/main` —
main had already moved to 86 rows / 33 bridge-level (KBR-55's M27–M29 + KBR-59's
M30–M32); the table records what KBR-309's **landed** row changes against that
baseline (KBR-271's lesson, recorded in
`~/.claude/.../MEMORY.md` "Register-edit companion sites"):

| Site | Old (at merge base) | New (landed) |
|---|---|---|
| `tests/harness/test_register.py` docstrings | "86 live rows" / "33 bridge-level rows" | "87 live rows" / "34 bridge-level rows" |
| `tests/harness/test_register.py` count pin | `== 86` | `== 87` |
| `tests/harness/test_register_agreement.py` M-count pin | `== 33` | `== 34` |
| `tests/harness/test_register_agreement.py` unconditional count pin | `== 51` | `== 52` (main's 51 + M33) |
| `tests/harness/test_register_agreement.py` unconditional spot-check set | without `"M33"` | with `"M33"` |
| `tests/harness/register.py` `REGISTER` tuple | 33 bridge-level live + 53 provider-level | M33 row appended after M32 |
| `.system_design/TEST_SUITE.md` §3.2.1 prose | "thirty-three live" | "thirty-five rows in all / thirty-four live" |
| `.system_design/TEST_SUITE.md` §3.2.1 table | (no M33) | M33 row appended after M32 |
| `.system_design/TEST_SUITE.md` §3.2.2 unconditional list | (no M33) | appended `M33` |
| `tests/harness/test_oracle_default_slice.py` annotations | `M26` | `M26/M33` (module-level overview, `_EXPECTED_CLAIMED_DELTAS` comments) |
| `tests/harness/test_register.py:_SHAPES` | (no KBR-309 text entry) | unchanged — the first-draft text entry was withdrawn with the parts-join row (the tuple it was added for was itself tautological; `_SHAPES`' own block comment records that) |

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
  M33's drop rather than a KBR-224-style restore) is a separate product
  decision, owned elsewhere.

## Merge-conflict note (2026-10-18)

PR #284 conflicted with `origin/main` twice over row ids. First rebase
(b6dbe3c): KBR-55 took M27/M28/M29, so the context_management row went
M27 → M30 and the first-draft M28 (parts-join) row was **dropped** — KBR-308
had already closed the join via `carry_cache_control=True` at hop 1, and the
bare-`parts[*]` anchor would over-claim §3.3.1's deleted-part falsification
case. Second (this merge of origin/main): KBR-59 landed its response-direction
rows on **M30/M31/M32**, so the context_management row is now **M33** (next
free id), appended last in both the `REGISTER` tuple and the §3.2.1 table.
All companion sites re-pinned: REGISTER 87 (34 bridge-level live M-rows +
53 provider-level), markdown M-count 34, unconditional list 52 (main's 51 +
M33; KBR-59 rows are all conditional), slice annotations `M26/M33`. This step
Row-id history: first-draft M27 → M30 (KBR-55 collision) → **M33**
(KBR-59 collision, landed); first-draft M28 → M31 → withdrawn. **M33 is a
request-side row**,
so the "next reply-side row id" noted in the KBR-59 record moves from M33 to
**M34** (the KBR-58/T-D9 corpus slice inherits that).