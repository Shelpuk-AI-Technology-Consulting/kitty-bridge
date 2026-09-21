---
id: t_i4_background_bridge_ownership
depends_on: []
---

# T-I4 — Background bridge ownership (KBR-96)

Jira: [KBR-96](https://shelpuk.atlassian.net/browse/KBR-96). Parent: KBR-21
(Epic I — Lifecycle and subsystem, L3). Plan row: §12 T-I4. Harness:
`TEST_SUITE.md` §6.3.2, row "Background bridge owned by another user".

Depends on: nothing (plan tier 0). KBR-176, KBR-219 and KBR-220 are closed;
KBR-219's BrokenPipeError defect is consolidated here ("Closed as
consolidated — NOT as fixed … Track the fix there") and is part of this step.

## What

Two deliverables in one step:

1. **Pin the ownership behaviour** the product already has. A background
   bridge running under another user account — a state file whose recorded
   PID `probe_pid` classifies `UNKNOWN` — is **not stopped**
   (`stop_bridge` refuses and leaves the state file), **not restarted**
   (`restart_bridge` aborts in its stop phase), and has **no second bridge
   started beside it** (`start_bridge` refuses; only `UNKNOWN` +
   *unreachable* is stale, and that complement keeps its current
   behaviour). Tests in `tests/cli/test_background_bridge_ownership.py`,
   marked `l3`.

2. **Fix the consolidated BrokenPipeError defect** (product owner,
   2026-09-21: "DEVNULL once the bridge reports ready"). The child
   (`kitty.bridge_runner`) points pipe-shaped fds 1/2 at `os.devnull` once
   `start_async` has reported ready (`kitty.io_encoding.relinquish_output_streams`,
   called after `await server.start_async()`); the pre-ready no-TLS warning
   — the documented kill site, which fires before the state file is written —
   swallows `BrokenPipeError` from a dead reader and dup2's devnull on
   catch. Regression test: a real child with `stdout=PIPE` whose parent's
   read end closes before the warning must still reach ready.

## Why these shapes

* **In-process management calls.** `probe_pid`'s `UNKNOWN` outcome is a
  kernel-level EPERM that cannot be reproduced without a second account, and
  a `monkeypatch` cannot cross the CLI-subprocess boundary the KBR-220
  fixture uses (where `SystemExit` becomes a returncode). The tests import
  `kitty.bridge.manage` and call the four verbs directly, patching only the
  classification. Everything else — child process, socket, state file,
  management code path — is real.
* **fd-state oracle, not behaviour, for devnull.** `logging.Handler.handleError`
  and `warnings._showwarnmsg_impl` both swallow `OSError` (CPython issue
  5971), so every post-ready stderr writer already survives a dead pipe on
  unfixed code. "The bridge keeps serving" cannot distinguish fixed from
  unfixed; `/proc/<pid>/fd/{1,2}` → `os.devnull` can. Linux-only via readlink,
  skipped elsewhere; the portable fd-level unit tests of the helper live at
  l1 (`tests/test_io_encoding.py`) so the fast gate covers them on all six
  legs from day one.
* **Falsification** (plan §1.4): the ALIVE-misclassification control (a
  broken guard that reads a foreign PID as ours must make `stop_bridge` stop
  the bridge) runs in the suite; AC-4's watched-failing regression is the
  second deliberate defect the harness detects.

## Falsification-fragment literals

`.replace()` pair used by the falsification control (grep these literals when
editing this file or the test module — a silent mismatch makes the control
vacuous):

* `probe_pid` → the only seam the tests patch
* `ProcessLiveness.UNKNOWN` → the classification that means "another user's"

## Verification

* `pytest tests/cli/test_background_bridge_ownership.py -q` — green.
* `pytest tests/test_io_encoding.py -q` — green.
* `pytest tests/cli/test_bridge_state_location.py -q` — still green (the
  KBR-220 harness is untouched by the product change).
* AC-4 watched failing on the unfixed code (2026-09-21, this worktree)
  before the fix landed.
* `ruff check`, `lint-imports`, `mypy src/kitty`, `pytest -q` (bare —
  includes l3), `pytest -m "l1 or l2" -q` (fast-gate shape) — green.

## Implementation notes

**Product change** (`src/kitty/io_encoding.py`,
`src/kitty/bridge_runner.py`, `src/kitty/bridge/server.py`):

* `kitty.io_encoding.relinquish_output_streams()` opens `os.devnull` lazily,
  dup2's it onto fd 1 and fd 2 **only where those fds are FIFOs** (S_ISFIFO
  guard), is closed-fd-safe (`except OSError` on `fstat`), and is naturally
  idempotent because after the first call the fd names a character device,
  not a pipe. The devnull fd is opened at most once and always closed in a
  `finally`.
* `bridge_runner.run()` calls `relinquish_output_streams()` immediately after
  `await server.start_async()` returns — the ready transition. Justification
  is in D9.
* `server.py start_async`'s no-TLS warning (the `if self._should_warn_no_tls():`
  block — the block whose `print` writes the warning, not a literal line
  number, since line numbers drift: it sat at `server.py:3289` when written
  and at `server.py:3465-3476` after the rebase onto `main`) is wrapped in
  `try / except BrokenPipeError: relinquish_output_streams()`. The dup2 on
  catch is what kills the interpreter-exit flush that would otherwise raise a
  second `BrokenPipeError` during shutdown.

**Tests added**:

* `tests/cli/test_background_bridge_ownership.py` (l3, 10 tests, marked at
  module level): ownership against foreign + serving + unreachable PIDs; the
  `process_survives` falsification control (patch ALIVE-never-DEAD is wrong;
  the modelled defect is the *ownership collapse only*, with death detection
  intact); the KBR-219 BrokenPipeError regression (real child with
  stdout=PIPE, parent's read end closed immediately, child must still reach
  ready); the post-ready fd-state oracle via `/proc/<pid>/fd/{1,2}` → devnull
  (Linux-only, `pytest.skip` elsewhere); the file-spawned guard-regression
  test that pins the pipes-only scoping.
* `tests/test_io_encoding.py` (l1, 4 tests): portable fd-level coverage of
  the helper (`stat.S_ISCHR` as the "is devnull" oracle; written-bytes-land
  as the "file untouched" oracle; closed-fd no-raise; second-call
  idempotent).

**Tests surfaced to the design documents**:

* `TEST_SUITE.md` §8.2: the four sentence edits the design review named
  (`Twelve` → `Thirteen` in the first paragraph, `Eleven modules` → `Twelve
  modules` in the second, KBR-220 bullet's `one module` → `first of the
  modules`, new KBR-96 bullet appended).
* `SYSTEM_DESIGN.md` §1.4: new decision **D9** (the readiness devnull +
  warnings-site audit + write-site scope justification + runit/s6 residual +
  why-not-log-file + l3-vs-l1 rationale).

**Module-level markers** (`pytestmark = pytest.mark.l3` at module level for
`test_background_bridge_ownership.py`) — every test inherits the marker;
both layers module as exactly one layer (T-W1).

**Generated step index** kept in sync: `python
scripts/regenerate_step_index.py` exits 0.

**Gates** (Linux, local): `ruff check` no new findings (full repo count
unchanged at 48 from the baseline), `lint-imports` silent, `mypy src/kitty`
clean, `pytest tests/cli/test_background_bridge_ownership.py tests/test_io_encoding.py
-m l3 -q` green (14 in ~16 s); the l3 selection runs only on this box until
T-K6 activates the Subsystem job, per the acknowledged debt in
`PENDING_ACTIVATION_LAYERS`. Fast-gate-shaped `pytest -m "l1 or l2" -q` is
checked at PR run time.
