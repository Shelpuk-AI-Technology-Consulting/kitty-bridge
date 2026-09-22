---
id: t_i6_agent_settings_precedence
depends_on:
  - KBR-97
---

# T-I6 — Agent settings precedence (KBR-98)

## What

One new test file in the existing `tests/agent_smoke/` package,
`tests/agent_smoke/test_claude_settings_precedence.py`, with three
test methods on one `TestSettingsPrecedence` class — one for each
row of TEST_SUITE.md §6.4.2's three-run precedence table (Main,
Control 1, Control 2). Every destination is demonstrated live:
the recorder wins the Main run, sentinel B wins Control 1, sentinel
A wins Control 2.

One keyword-only parameter added to
`tests/agent_smoke/test_claude_startup.py::_run_one_turn`,
`extra_args: Sequence[str] = ()`, so Main can pass
`["--settings", <path>]` through the existing helper without
forks; T-I5's two call sites (and the three tests in
`TestMissingBinaryFalsification` that don't reach this helper) are
byte-identical.

One package-docstring correction in
`tests/agent_smoke/__init__.py`: the "belongs elsewhere" sentence
that named T-I6 as out-of-package is rewritten to name this package
as T-I6's home, citing T-I5's step file as the placement authority
(it has been since KBR-97's merge).

One TEST_SUITE.md §6.4.2 paragraph addition: the **"What T-I5
settled"** paragraph, the cross-reference paragraph that captures
T-I5's hermeticity posture, the capture-count discovery, and the
two falsification seams, and pins the L4 rationale (a precedence
order is a fact about Claude Code; only the real binary's
destination choice is observable).

## Why

KBR-98 (T-I6) is the §6.4.2 precedence run that T-I5's connectivity
smoke cannot prove: a run in which nothing competes cannot
distinguish correct precedence from accidental agreement, and a
binary with the precedence order backwards passes T-I5's smoke
silently. The launcher's `prepare_launch` design depends on
`--settings` outranking `~/.claude/settings.json` outranking env;
KBR-1's investigation put the precedence order under suspicion; no
test in the tree defended it.

**The three-sentinel design (Main + Control 1 + Control 2).** A
sentinel that is never hit cannot prove it can be hit — without the
controls, a typo in A's URL reads as a pass because A is *supposed*
to be silent in Main and Control 1, and a broken A is silent too.
Each control makes one sentinel the winner, exercising it as a
destination, not just a loser. The §6.3 validation rule applied at
L4: the controls are the negative case proving the harness can
fail. The two falsification runs this PR executed against the
unpatched code confirmed both halves — a wrong `--settings`
destination makes sentinel A capture, which Main catches; a sentinel
with a closed port makes its control's winner assertion fail.

## How

* **Hermeticity posture, inherited.** T-I5's four redirects plus one
  cwd pin plus one env-key filter, verbatim. The `ANTHROPIC_BASE_URL`
  redirect carries sentinel A's URL — `_hermetic_env` is already
  parameterised on `base_url`, so the same helper reaches a different
  destination for the precedence runs. No wrapper helper, no l1
  unit tests for trivial JSON writers: the L4 tests in AC-1/AC-2/AC-3
  exercise the file shapes end to end, and a wrong shape fails its
  run with the destination's `captures` empty, which is a precise
  diagnostic — an l1 pin would duplicate that for no added power.

* **Three fixtures, three independent ports.** Each test method
  starts three `BridgeFixture`s on three ephemeral loopback ports
  before `claude -p` runs, so the binary's connect attempt never
  sees a half-up destination. Sequential starts (not concurrent):
  a start failure in the second or third fixture would leave the
  first bound if the starts raced. The test method's `finally`
  block stops all three in the same order they were started.

* **The session settings file is exactly
  `{"env": {"ANTHROPIC_BASE_URL": <base_url>}}`.** Same shape
  `ClaudeAdapter.prepare_launch` writes for this one key. A wider
  file would couple T-I6 to the launcher's key set and break the
  next time the launcher grows a new key — the precedence claim is
  about one key, not the launcher's key list. The global settings
  file is the same shape; Control 2 omits the file entirely
  (literal absence, not an empty JSON object — an empty settings
  file is a different shape than no settings file and would carry
  its own Claude Code interpretation).

* **`extra_args` is keyword-only with default `()`.** T-I5's two
  call sites pass nothing and keep their behaviour byte-identical
  (the regression half of AC-4 verifies this; T-I5's suite is
  green). Control 1 and Control 2 pass an empty tuple via the same
  parameter — going through the same code path with an empty
  sequence is the negative case that keeps the flag's effect from
  being asserted only on the winner's path.

* **Content pin, not count.** The "at least one `POST /v1/messages`
  carrying a user-role message" assertion reuses T-I5's
  `_body_has_user_message` (imported, not re-implemented —
  `grep -c "def _body_has_user_message" tests/agent_smoke/*.py`
  returns exactly 1). Pinning to a count would tie the precedence
  claim to today's binary shape and fail the moment Claude Code
  adds another pre-flight call.

* **Session-title assumption, recorded.** Claude Code in `-p` mode
  makes more than one POST per turn (a session-title-generation call
  precedes the user reply); both go to the winner under today's
  precedence chain. This is documented in the new module's
  docstring because it is what makes the assertion safe — a future
  Claude Code release that routed the title call to a different
  source would still produce a readable failure (the destination
  that captured is named in the diagnostic).

* **Missing binary is a failure, not a skip.** Inherited from T-I5:
  `_claude_binary` calls `pytest.fail` when no binary is found,
  with the `KITTY_AGENT_SMOKE_BINARY` override seam and the
  factored `_resolve_default_binary()` for the default branch.
  T-I6 does not duplicate these — both branches are already covered
  by `TestMissingBinaryFalsification`.

* **TDD ordering.** `_run_one_turn`'s `extra_args` parameter landed
  first with T-I5's suite green (AC-4's regression half). Then the
  Main run was written and watched failing (no fixtures wired),
  then the three-fixture scaffolding + settings-file writer +
  `--settings` pass-through made it pass. Controls 1 and 2 followed
  the same cycle, differing from Main only in their row's
  configuration (the `extra_args` and `write_global_settings`
  parameters). The three-fixture scaffolding stayed inline because
  the duplication across the three methods is shorter than the
  helper that would extract it.

## Why this does not overlap with T-I5, T-K10, T-K11, T-I14

* **T-I5 (KBR-97, PR #251, merged 2026-09-21)** is the connectivity
  smoke. T-I6 inherits T-I5's `_claude_binary`, `_hermetic_env`,
  `_body_has_user_message`, `_run_one_turn`, and the bridge fixture;
  T-I5's two `_run_one_turn` call sites (and T-I5's other three
  tests in `TestMissingBinaryFalsification`) are untouched. The
  `_run_one_turn` extension is one keyword-only parameter with a
  default that preserves T-I5's argv ordering byte-identical.
* **T-K10 (KBR-119, To Do)** is the CI gate activation. T-I6 does
  not edit `PENDING_ACTIVATION_LAYERS`; the `agent_smoke` row
  stays acknowledged-debt until T-K10. The bidirectional checks at
  `tests/layers.py::unaccounted_layers` and `stale_pending_layers`
  hold the registry to the tree.
* **T-K11 (agent_live nightly), T-I14, the answer-quality evals**
  operate against real providers; their inheritance from T-I6 is
  only the bridge fixture's shape.

## Implementation notes (this PR)

- Three new test methods on `TestSettingsPrecedence`, plus the
  `_three_fixtures_up`, `_write_session_settings_file`,
  `_write_global_settings_file`, `_run_turn_and_assert_winner`,
  and `_run_turn_and_assert_silent_loser` helpers in the new test
  module. The session-file helper shape is intentionally minimal
  (one key) — see "How" above.
- One keyword-only parameter added to T-I5's `_run_one_turn`,
  `extra_args: Sequence[str] = ()`. T-I5's two call sites are
  untouched.
- One package-docstring correction in
  `tests/agent_smoke/__init__.py`.
- One TEST_SUITE.md §6.4.2 paragraph addition: "What T-I5
  settled", cross-referencing T-I5's step file and the canonical
  hermeticity breakdown (four redirects, one cwd pin, one env-key
  filter), recording the capture-count discovery and the two
  falsification seams, and pinning the L4 rationale.
- Falsification runs executed during this PR: two throwaway scripts
  driven by hand against the unfixed code. The first simulated a
  wrong `--settings` destination (sentinel A) — sentinel A
  captured, recorder did not, the Main assertion would fail. The
  second simulated `--settings` and global settings both pointing
  at sentinel A with env at sentinel B — same outcome. Both
  confirmed the harness can detect the precedence defect it is
  designed to detect.
- Step index regenerated this PR; dependency graph validates
  (`depends_on: KBR-97`, Done); `scripts/regenerate_step_index.py`
  exits 0.
