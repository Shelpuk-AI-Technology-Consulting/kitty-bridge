---
id: t_i5_agent_startup_smoke
depends_on:
  - KBR-31
  - KBR-32
  - KBR-43
  - KBR-216
---

# T-I5 — Agent startup smoke (KBR-97)

## What

One new test category, `agent_smoke`, populated by three tests at
`tests/agent_smoke/test_claude_startup.py`:

* `TestAgentStartupSmoke::test_a_pinned_claude_code_runs_one_turn_and_reaches_the_bridge`
  — the §6.4.2 connectivity claim. The test launches a real `claude -p`
  against a real `BridgeServer` in front of the T-W4 recorder, on loopback,
  with a dummy `ANTHROPIC_API_KEY`, `HOME` and `CLAUDE_CONFIG_DIR` both
  redirected into `tmp_path`, and
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` killing the binary's own
  statsig/sentry/update traffic. It asserts the binary exits 0 and at least
  one `POST /v1/messages` capture carries the user's turn (the user turn is
  pinned by *content*, not by index — see "Why" below).
* `TestMissingBinaryFalsification::test_a_missing_binary_fails_rather_than_skips`
  — the §8 falsification (KBR-132 shape): a nonexistent
  `KITTY_AGENT_SMOKE_BINARY` must raise `pytest.fail`, not skip.
* `TestMissingBinaryFalsification::test_an_existing_override_path_is_honoured`
  — the positive control proving the override seam is non-vacuous.

Layer-marker wiring: `tests/layers.py::_PATH_DEFAULTS` gains the row
`("tests/agent_smoke/", "agent_smoke")`, with a parametrized test in
`tests/test_layer_markers.py::TestDefaultLayerForPath::test_it_maps_a_path_to_its_layer`
covering it (and the sibling `tests/acceptance_old/` negative case already
there covers the prefix-precision direction).

## Why

TEST_SUITE.md §6.4.2 carves out two distinct agent-boundary claims:
*connectivity* (this task) and *settings precedence* (T-I6 / KBR-98). The
suite as it stood had only `agent_live` E2E tests, which prove neither
because they require real provider credentials and skip when those are
absent. The bridge was proven against a synthetic driver (T-W9) and against
real providers (the E2E suite), but never against a real Claude Code binary
in the hermetic configuration the per-PR gate is supposed to enforce.

Q12 — "how is a pinned Claude Code binary supplied to CI?" — is answered
([KBR-216](https://shelpuk.atlassian.net/browse/KBR-216), 2026-09-12, the
PO comment on that ticket and on KBR-97 of the same date): CI installs the
official CLI from `https://claude.ai/install.sh` at an exact version
(`2.1.238`), at run time, so redistribution does not arise and the per-PR
gate is viable. The four operational facts that follow from that answer are
recorded in TEST_SUITE.md §8.6 (the *inventory* table and the "Two limits"
paragraph that follows it, around lines 5441-5452 in the source), and this
step inherits them without re-justifying them.

**How many requests one turn makes.** §6.4.2's sentence says *"its
request"*, singular. Claude Code in `-p` mode makes **more than one** POST
per turn: a session-title-generation call before the user reply (observed
on Claude Code 2.1.276 during the first TDD cycle; the binary's own stderr
names the two `query_source` values, `generate_session_title` and the
default). This step therefore asserts connectivity with
`len(captures) >= 1` and pins the *user* turn by content
(`_body_has_user_message`), not by index and not by count — pinning the
count would fail the moment Claude Code adds another pre-flight call, and
that is the wrong signal for a connectivity smoke. The same fact reshaped
REQUIREMENTS.md R1, so the two artifacts agree.

## How

* **Hermetic by construction.** Four redirects, each closing a different
  leak (the module docstring names them):
  * the bridge fixture owns the recorder (T-W8:
    `tests/harness/bridge.py::AiohttpTransport` with
    `WireFormat.ANTHROPIC_MESSAGES`); the fixture's `_KEY` is the fixed
    dummy. No credential store is consulted.
  * `HOME` → `tmp_path`: Claude Code keeps a fifth file, `~/.claude.json`,
    that it writes *outside* the `CLAUDE_CONFIG_DIR` it honours
    (`anthropics/claude-code#25762`); `HOME` is the coarse redirect that
    covers it.
  * `CLAUDE_CONFIG_DIR` → `tmp_path`: the documented fine redirect
    (https://code.claude.com/docs/en/env-vars).
  * `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`: the binary's own
    statsig/sentry/update traffic is non-essential and would make
    "no network beyond loopback" false for the child. The repo already
    uses this flag for its own hermetic runs
    (`tests/integration/test_tmux_disconnect.py:141`,
    `scripts/capture_corpus_t_c1.py:179`).
* **Binary discovery uses the same fallback chain kitty uses in
  production.** `kitty.launchers.discovery.discover_binary("claude")` is
  the production helper; §8.6 records that `~/.local/bin/claude` is not on
  `PATH`, so a plain `PATH` lookup would not find a CLI that is present.
  **PATH-first precedence, stated:** on a dev machine where `claude` on
  `PATH` resolves to a kitty wrapper, the wrapper's spawn config overrides
  `ANTHROPIC_BASE_URL` at process level — the captures would be empty and
  the test fails loudly, naming the path it tried. The connectivity claim
  still holds; the raw-binary assertion is the wrapper-overrides case,
  which the `HOME` and `CLAUDE_CONFIG_DIR` redirects cover (a wrapper's
  settings injection lands in the redirected config dir, not the
  developer's).
* **Missing binary is a failure, not a skip.** TEST_SUITE.md §8 line 4524
  (*"Skips are failures in a gating job. If `agent_smoke` cannot find its
  pinned binary, or `l3` cannot start its proxy, the job fails. A gating
  job that goes green because it ran nothing is the most expensive kind of
  false confidence."*) is honoured: the test calls `pytest.fail(...)`.
  **The automatable seam:** poisoning `PATH` cannot simulate the absence —
  `discover_binary` falls back to `~/.local/bin`, where the real binary
  sits — so `_claude_binary` takes a `KITTY_AGENT_SMOKE_BINARY` env
  override, and `TestMissingBinaryFalsification` drives it. This is the
  KBR-132 shape ("restore the skip, the test must go **red**"), and the
  reason `SKIP_AGENT_SMOKE` was rejected as a mechanism: a skip silently
  bypasses the §8 rule, and no falsification case can catch a silent
  bypass. A developer without the binary installed who deliberately runs
  `pytest -m agent_smoke` sees the failure message; a bare `pytest` never
  collects the category (the `addopts` expression excludes
  `RESOURCE_DEPENDENT_LAYERS`), so nobody who did not opt in is blocked.
* **Version-agnostic.** §8.6 records that the CLI pin is a hand-maintained
  pairing with the floating `@v1` tag of
  `anthropics/claude-code-action`, and "a job asserting a *specific* CLI
  version rests on a human having noticed". The test asserts no version;
  if the binary's wire shape changes, the capture-shape test catches it;
  if the binary's behaviour regresses, the `agent_live` E2E and the
  answer-quality evals catch it.
* **The recorder's auto-200 default is sufficient for `-p` mode — via the
  bridge's adapter, not directly.** The recorder's `minimal_success_body`
  is "the fewest keys that still read as a success **to the bridge**"
  (T-W4). Claude Code is the bridge's *downstream* client, so what it
  parses is the bridge's `CustomAnthropicAdapter` translation of that
  minimal body, not the body itself. That two-hop path is exactly what
  the connectivity claim covers: the exit-0 assertion is the runtime
  evidence the translation is acceptable, and nothing here re-proves it.
* **TDD discipline.** Layer-marker wiring is test-first: the parametrized
  case in `test_layer_markers.py` is added before the `_PATH_DEFAULTS`
  row. The startup test is implemented in small cycles, each watching
  the test fail for the new reason before making it pass — the first
  cycle's discovery (`-p` makes more than one POST) is recorded under
  "Why" above.
* **`--dangerously-skip-permissions` is required, not optional.** In
  `-p` non-interactive mode the binary has no TTY, so a permission
  prompt would fail rather than hang. Skipping the permission step makes
  the `-p` turn complete in any case where the binary would otherwise
  need a human — and the smoke's claim is connectivity, not permissions;
  a permissions-shaped test belongs elsewhere.

## Why this does not overlap with T-I6, T-K10, T-K11, T-I14

* **T-I6 (KBR-98) is settings precedence**, three runs with three winners
  (the recorder plus two sentinels) so a backwards implementation cannot
  pass accidentally. T-I5 is the single-run "the binary even reaches the
  bridge" proof that precedes it; T-I6's three-sentinel design inherits
  T-I5's bridge fixture and test harness.
* **T-K10 (KBR-119) is the CI gate activation.** It adds the `agent_smoke`
  job to `.github/workflows/tests.yml` with the right
  `--require-category=agent_smoke` plumbing and drops the `agent_smoke`
  entry from `PENDING_ACTIVATION_LAYERS`. The bidirectional checks at
  `tests/layers.py::unaccounted_layers` (a populated layer no job runs and
  nothing acknowledges) and `tests/layers.py::stale_pending_layers` (a
  registry entry outliving its reason) are what keep this debt registered
  rather than silent between T-I5's merge and T-K10's. Until T-K10 lands,
  T-I5's tests are exercised by `pytest -m agent_smoke` and by local
  developer runs — not by every PR.
* **T-K11 (agent_live nightly), T-I14, and the answer-quality evals**
  operate against real providers and need credentials; their inheritance
  from T-I5 is only the bridge fixture's shape.

## Implementation notes (2026-09-21, this PR)

- One new test file in a new directory; one row in `_PATH_DEFAULTS`; one
  parametrized case in `test_layer_markers.py`; one package docstring.
  Three files, all surgical, none invasive.
- The capture assertion checks the request path (`/v1/messages`) and the
  shape of the inbound body (`messages` array, role `user`) on **at least
  one** capture, and does not pin the count. It does NOT re-prove what
  T-W8's `assert_teardown_clean` already pins — the transport's declared
  format and the recorder's path selection — because re-proving it here
  would create a duplicate assertion that drifts the next time T-W8
  changes. Streaming-vs-JSON response shape is likewise out of scope: the
  exit-0 assertion is the runtime evidence the bridge's translation is
  acceptable to Claude Code, and a response-shape claim belongs to the
  acceptance scenarios, not to a smoke.
- The `_KEY` constant in `tests/harness/bridge.py` is the *upstream* key
  the bridge sends to the recorder; the dummy `ANTHROPIC_API_KEY` this
  test sets is what Claude Code sends *to the bridge* as its inbound
  `x-api-key`. Two different credentials on two different hops; the
  bridge does not validate the inbound one, so any string works. The
  requirement text says "dummy" and the test says which dummy.
- The `~/.local/bin/claude` location is the one this box has (Claude Code
  2.1.276). The CI workflow pins `2.1.238`; the tests do not assert on
  the version, so the local and CI binaries are interchangeable.
- First-cycle TDD discovery, recorded in the module docstring and above:
  Claude Code `-p` mode makes more than one POST per turn (a
  session-title-generation call precedes the user reply), so the
  capture-count assertion is `>= 1` with the user turn pinned by content.
- Step index regenerated 2026-09-21; dependency graph validates
  (`depends_on: KBR-31, KBR-32, KBR-43, KBR-216`, all Done);
  `scripts/regenerate_step_index.py` exits 0.
