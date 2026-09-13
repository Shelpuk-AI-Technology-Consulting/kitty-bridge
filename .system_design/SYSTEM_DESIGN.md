# Kitty Bridge — System Design

**Status:** To Be. **Partial by design:** this document starts with the areas work has actually
touched and grows as later work reaches others. What the *test suite* must prove is in
[`TEST_SUITE.md`](TEST_SUITE.md), and how that gets built is in
[`TEST_SUITE_IMPLEMENTATION_PLAN.md`](TEST_SUITE_IMPLEMENTATION_PLAN.md). This file describes the
product's own components.

Symbols are named, not line-numbered, for the reason `TEST_SUITE.md` gives.

---

## 1. Background bridge management

Traces to [KBR-220](https://shelpuk.atlassian.net/browse/KBR-220) and the Windows stop-handler
defect fixed with it. Related: KBR-154, KBR-176 and KBR-219 (how `start` reads and waits on the
child), and KBR-180 (probing a PID on Windows).

### 1.1 Components

| Component | Role |
|---|---|
| `kitty bridge start / stop / restart / status` (`kitty.cli.main`) | The user's commands. They resolve paths and call `kitty.bridge.manage`. |
| `kitty.bridge.manage` | `start_bridge` spawns the child and waits for its state file. `stop_bridge` signals the recorded PID and removes the file. `bridge_status` classifies the recorded PID. |
| `kitty.bridge_runner` | The background bridge process, spawned by `start_bridge` or by a service unit (`kitty.bridge.service`). kitty's second entry point. |
| `kitty.bridge.server.BridgeServer` | `start_async` writes the state file once listening. `stop_async` removes it, and so does the crash handler. |
| `kitty.bridge.state` | The file format, and `default_state_path()`, the one resolver for where it lives. |
| `kitty.bridge.stop_signals` | `install_stop_handlers`, used by every loop that runs a bridge until told to stop. |

### 1.2 The state-file contract

`bridge_state.json` is how every management command finds a bridge it did not start in-process.
It records `pid`, `host`, `port`, `profile`, `started_at` and `tls`.

- **One location:** `~/.config/kitty/bridge_state.json` on every platform, from
  `kitty.bridge.state.default_state_path()`, read on each call.
- **Who resolves it:** the CLI's four commands call the resolver. `manage` uses it when a caller
  passes no path. The runner uses it when it is given no `--state-file`.
- **Hand-off:** `start_bridge(state_path=X)` spawns the runner with `--state-file X`, so the child
  writes exactly the file the parent polls, whatever `X` is.
- **Service-started bridges** (systemd `--user`, a LaunchAgent) pass only `--config`, so they
  write to the default, which is where `status` and `stop` read.
- `bridge.yaml` is **not** part of this contract. It stays in `platformdirs.user_config_dir("kitty")`
  and always reaches the child explicitly via `--config`.

```
kitty bridge start ──default_state_path()──► start_bridge(state_path=P)
                                                   │ spawn: python -m kitty.bridge_runner --state-file P --config …
                                                   ▼
                                    bridge_runner ──► BridgeServer(state_file=P).start_async() ──writes──► P
kitty bridge status / stop ──default_state_path()──► reads P
service unit ──► bridge_runner --config … (no flag) ──default_state_path()──► writes P
```

### 1.3 Stop signals

A bridge loop waits on an `asyncio.Event` that SIGINT or SIGTERM sets, then runs `stop_async`.
`install_stop_handlers` registers both where the loop supports it. Where it does not (Windows:
CPython's base loop raises `NotImplementedError`), it registers nothing and returns `False`.
Nothing calls `loop.add_signal_handler` directly; `tests/test_stop_handler_call_sites.py` holds
that line.

### 1.4 Decisions, and why

Numbered as in the KBR-220 requirements and PR. D4 there, which folded the Windows fix into the same change, was a scope decision and is not repeated here.

| # | Decision | Why, and the rejected alternative |
|---|---|---|
| D1 | The state file lives at `~/.config/kitty` on every platform, not in `user_config_dir` | Every background bridge has always written there. Moving it would orphan the state of bridges already running at upgrade and need migrate-or-report code. `user_config_dir` (beside `bridge.yaml`) is more idiomatic. **Accepted cost:** on macOS, Windows and XDG Linux the state sits apart from `bridge.yaml`, as the keys file and logs (`kitty.bridge.config`) already do. *Product owner, 2026-09-13.* |
| D2 | A shared resolver **and** an explicit `--state-file` hand-off | The resolver alone leaves `start_bridge(state_path=X)` silently ignored by the child. That ignored parameter is why no test saw KBR-220: every test passed the same path to both sides. The flag alone would not cover service-started bridges. *Product owner, 2026-09-13.* |
| D3 | No migration of an "old" state file | The new home is the only place a bridge has ever written state: the runner line is unchanged since `bridge_runner.py` was created. The old CLI location only ever received `bridge_state.json.start.lock`, a zero-byte lock. It is left alone: deleting a lock an older kitty may hold is the riskier act, and it blocks nothing because the lock now sits beside the new state path. |
| D5 | Where a loop cannot register signal handlers, register nothing: no `signal.signal` fallback | On Windows, `kitty bridge stop` ends the process with `TerminateProcess`, which no handler can intercept, and `stop_bridge` removes the state file itself. Ctrl+C in a foreground bridge still raises `KeyboardInterrupt`, which `asyncio.run` turns into cancellation, so `finally: stop_async()` still runs. A thread-to-loop signal bridge would add complexity for no visible gain. |
| D6 | `stop_signals` lives in `kitty.bridge`, not a top-level leaf | Both callers (`kitty.cli.main`, `kitty.bridge_runner`) may already import `kitty.bridge`. A top-level leaf would need its own import-linter contract and an entry in every "every sibling" list. |

### 1.5 Known limits (recorded, not fixed here)

- **A Windows "background" bridge shares the user's console.** `start_new_session=True` is
  POSIX-only, so closing that console or pressing Ctrl+C in it ends the bridge.
- **A Windows bridge never shuts down gracefully when stopped.** `TerminateProcess` skips
  `stop_async`, so an opt-in session summary (`KITTY_SESSION_SUMMARY`) is not written.
- **Services running as another account are outside the contract.** The NSSM script sets no
  account, so the service runs as LocalSystem with a different home and profile directory. A
  LaunchAgent with `KeepAlive` is restarted by launchd straight after `kitty bridge stop`.
- **`start` waits 5 seconds.** A bridge slower than that is reported as not ready and left
  running (KBR-176).

---

## 2. Agent launch path: `kitty [flags] <profile> <agent> [agent args]`

Traces to no ticket yet (the tmux-wrap work, 2026-09-13). The *background* bridge of §1 is a
separate mode; this section is the foreground one every interactive session uses.

```
shell ─► kitty.cli.main.main
           │ parse kitty flags (parse_known_args); unknown flags go to the agent
           │ resolve egress, route the command (CLIRouter)
           │ ── §3: Claude Code with --tmux? maybe hand over to tmux here ──
           ▼
         _launch_target ─► kitty.cli.launcher.launch_async
                             validate key ─► start BridgeServer in process (127.0.0.1:<port>)
                             ─► write Claude settings file ─► spawn agent (inherits the TTY)
                             ─► wait ─► restore settings ─► stop bridge
```

- **On this path the bridge lives inside the `kitty` process.** When the agent exits, kitty stops
  the bridge; when kitty dies, the bridge dies.
- **Kitty installs no `SIGHUP` handler**, so closing an SSH terminal kills kitty and its bridge
  even if the agent survives elsewhere.
- **For Claude Code, kitty passes its settings two ways:** environment variables, and a
  `--settings <tmpfile>` flag whose `env` block repeats them.

## 3. Surviving an SSH disconnect: `--tmux` on the Claude Code agent

### 3.1 What Claude Code's own `--tmux` does (read from the v2.1.269 binary, checked against live sessions)

- `claude -w <name> --tmux[=classic]` creates the git worktree **in the outer process**, then runs
  `tmux new-session -A -s <session> -c <worktree> -e CLAUDE_CODE_TMUX_*=… -- <claude> <args minus
  -w/--tmux>` and waits for that tmux client. `--tmux` alone uses iTerm2 native panes when it
  detects iTerm2; `--tmux=classic` always uses plain tmux.
- **Session name:** `<basename of the canonical repo root>_worktree-<name, "/"→"+">`, then every
  `/` and `.` → `_`. The canonical root is the main checkout even when run from a linked worktree
  (kitty uses the parent of `git rev-parse --git-common-dir`, resolved against the working
  directory).
- **Name forms:** after `-w`/`--worktree`, the next token unless it starts with `-`; or the text
  after `--worktree=`. No name → `<adj>-<noun>-<4 base-36 chars>` (adjectives `swift bright calm
  keen bold`, nouns `fox owl elm oak ray`). `#N`, a GitHub PR URL or a GitLab MR URL → `pr-N`.
- If `$TMUX` is set, Claude creates the session detached, switches to it, and the outer process
  returns.
- Every other argument, kitty's `--settings <tmpfile>` included, is forwarded to the inner
  Claude.
- The real Claude is a child of the **tmux server**. It gets the server's environment, not the
  caller's (tmux 3.4 manual, *GLOBAL AND SESSION ENVIRONMENT*): the caller's environment is copied
  only when the server first starts.

**The problem under kitty:** the outer `claude` is kitty's child, kitty stays on the SSH terminal,
and the bridge dies with it. The tmux-hosted Claude keeps running against a dead port.


### 3.2 To Be: kitty wraps itself in tmux

`kitty.cli.tmux_wrap` decides; `kitty.cli.main.main` calls it after routing and before
`_launch_target` (plain and balancing profiles alike), so routing and egress-resolution errors
still print in the user's own terminal.

The wrap is considered when the agent is Claude Code and its args contain `--tmux` or
`--tmux=classic`. Then, in this order:

| Situation | Behaviour |
|---|---|
| No `-w`/`--worktree` token, Windows, stdin or stdout not a TTY, or not a git repo | Pass the args through unchanged. Claude reports its own error, or runs as before. |
| `$TMUX` set (already inside tmux) | Remove `--tmux` and launch in place. The pane already survives a disconnect. |
| tmux missing, or `tmux -V` below 3.2 | Error, exit 1. |
| A session with that name exists and carries kitty's marker | Print that this command's profile and flags are ignored, then `tmux attach-session -t =<name>`. |
| A session with that name exists without the marker | Error naming the session and both remedies (attach, kill). Exit 1. |
| Otherwise | Print `Session: <name> (reattach: tmux attach -t <name>)`, write the environment file, run the command below, wait. |

```
tmux new-session -s <name> -c <cwd> -e KITTY_TMUX_WRAPPED=1 -e KITTY_TMUX_ENV_FILE=<path>
     [-e PYTHONPATH=… and the other interpreter-start variables that are set]
     -- <sys.executable> -m kitty.cli.tmux_inner <kitty argv, --tmux removed, generated name inserted>
```

When the tmux client returns: if the live session's `KITTY_TMUX_ENV_FILE` is this launch's file, the
session is ours and kitty prints the reattach hint again; otherwise kitty deletes the file if it is
still there. Checking ownership, not mere existence, covers two launches racing for one name: the
loser's `new-session` fails, the winner's session exists, and the loser's file would otherwise stay. Kitty exits with the
client's exit code, **which is tmux's, not the inner kitty's** (a pane that exits 3 ends the
session and the client exits 0, measured on tmux 3.4).

Inside the session, `kitty.cli.tmux_inner` (standard-library imports only until the environment is
in place) reads the file, deletes it in a `finally`, replaces `os.environ` with its contents except
the variables tmux owns (T7), and runs `kitty.cli.main.main`. On a non-zero exit or an exception it
prints `kitty exited with code N - press Enter to close` and waits for a line or end of input.

### 3.3 Decisions, and why

| # | Decision | Why, and the rejected alternative |
|---|---|---|
| T1 | Kitty wraps **itself**, not just Claude | The bridge is in-process on this path (§2); only moving kitty moves the bridge. Rejected: reuse the background bridge of §1. It has a different lifecycle (a state file, one shared bridge, explicit stop), so a session would leave a bridge running after it ends or need its own cleanup. *Product owner, 2026-09-13.* |
| T2 | `--tmux` is removed before re-running kitty | Left in, the Claude inside kitty's session sees `$TMUX`, creates a second session detached and returns at once, so kitty stops the bridge. |
| T3 | Claude still creates the worktree; kitty only computes the name | No duplicated worktree logic. |
| T4 | Session name follows Claude's rule exactly | Re-running the command reattaches instead of starting a second copy, and the name is the one users already see. *Product owner.* |
| T5 | No name → kitty generates one Claude's way and inserts it; PR forms → `pr-N` | Kitty needs the name before tmux starts; Claude would pick its random name too late. The name is printed before tmux starts, because after a disconnect the client never returns to print it, and re-running a nameless command generates a new name. *Product owner.* |
| T6 | The environment goes through a private file, not `tmux -e` | A running tmux server does not pass the caller's environment to new sessions, and kitty depends on far more than it reads directly: `HOME` and `XDG_*` (profile and credential stores), `AWS_*` (Bedrock), `TMPDIR`, proxy variables, plus everything Claude inherits (`CLAUDE_CONFIG_DIR`, locale). A hand-kept list would drift, and `-e` values stay visible in `ps` for the whole attach. The file is JSON (`ensure_ascii`, so undecodable bytes round-trip), created `0600` with `mkstemp`, and deleted by the inner kitty as soon as it is read, or by the outer kitty once the session is gone. **Accepted cost:** the environment (which may include API keys) sits on local disk, readable only by the user, between tmux starting and the inner kitty reading it. *Product owner.* |
| T6a | Interpreter-start variables (`PYTHONPATH`, `PYTHONHOME`, `PYTHONUSERBASE`, `PYTHONUTF8`, `PYTHONIOENCODING`, `LANG`, `LC_ALL`, `LC_CTYPE`) also go through `-e` | Python applies them before any code runs, so the file arrives too late for them. Without `PYTHONPATH` an installation that relies on it cannot even import `kitty.cli.tmux_inner`, the file is never read, and nothing holds the pane. They are not secrets. |
| T7 | Inside the session, `TMUX`, `TMUX_PANE`, `TERM`, `TERM_PROGRAM`, `TERM_PROGRAM_VERSION` and `KITTY_TMUX_WRAPPED` keep tmux's values, not the file's | The file holds the *outer* terminal's values. Restoring the outer `TMUX` (unset) would make the inner kitty wrap itself again, and the outer `TERM`/`TERM_PROGRAM` misdescribe the terminal Claude is now in. |
| T8 | An unmarked session with the same name is refused | Because T4 matches Claude's names, a plain `claude --tmux` session, or one left by the bug this fixes, would otherwise be silently attached: a Claude that bypasses kitty and bills the user's own account. The marker is `KITTY_TMUX_WRAPPED=1` in the session environment (`new-session -e`), read with `tmux show-environment -t =<name>`. *Product owner.* |
| T9 | A failed inner kitty holds the pane until Enter | Otherwise the session ends with the pane and every launch error (bad key, egress block, missing binary, Claude refusing the name) disappears. It does not give the outer kitty the inner exit code; that stays tmux's (§3.2). Ctrl-C is not held: the user asked to stop. *Product owner.* |
| T10 | Both `--tmux` and `--tmux=classic` trigger the wrap | Identical on Linux over SSH, the case this exists for. **Accepted cost:** on macOS with iTerm2, `--tmux` under kitty gives plain tmux instead of native panes. *Product owner.* |
| T11 | tmux ≥ 3.2 required | `new-session -e` arrived in 3.2; Claude Code uses the same cut-off. An unparseable version string (e.g. `openbsd-7.4`) is treated as supported. |
| T12 | The inner command is `sys.executable -m kitty.cli.tmux_inner` | Runs the same installation as the outer kitty whatever the server's `PATH` holds, and never the kitty *terminal*'s binary of the same name. A separate entry module keeps environment loading and the pane hold out of `main`, and lets the environment be replaced before modules that read it at import time (`Path.home()` in `kitty.launchers.claude`) are imported. |
| T13 | Spawn and wait for tmux, rather than `exec` | Kitty can then delete a leftover environment file and print the reattach hint. |
| T14 | Non-TTY and Windows pass through | An attached tmux session needs a terminal, and Windows has no native tmux; before this feature both launched fine without the wrap. The pass-through checks run before the inside-tmux check, so a non-TTY run inside tmux keeps `--tmux` exactly as before this feature. |

### 3.4 Verification

- **L1** (`tests/cli/test_tmux_wrap.py`, `tests/cli/test_tmux_inner.py`): every decision in §3.2 as
  pure functions with injected environment, filesystem, TTY/platform flags and process runner.
- **Live end-to-end** (`tests/integration/test_tmux_disconnect.py`, layer `agent_live`), run by
  `.github/workflows/tmux-disconnect.yml` on same-repository, non-draft pull requests whose changes
  touch the wrap. Activating `agent_live` for this job removes it from `PENDING_ACTIVATION_LAYERS`;
  the nightly T-K11 job for the other agent tests is still owed (*product owner, 2026-09-13*). The
  workflow is **not** a required check: path-filtered, it would sit pending on every other PR.

  The test starts `kitty claude -w <n> --tmux=classic` under `script` (a pseudo-terminal), waits for
  readiness (the bridge's `/healthz` answers 200 on the port in Claude's `ANTHROPIC_BASE_URL`), hangs
  the terminal up (`script` killed, which closes the pseudo-terminal so the kernel sends SIGHUP to
  the session on it; `script`'s stdin is a pipe the test holds open until then, because `script`
  forwards end-of-input into the terminal), then proves that the outer kitty exited and: the session exists, a
  `kitty.cli.tmux_inner` process is alive, `/healthz` still answers, and a prompt sent with
  `tmux send-keys` asking for the sum of two random numbers produces that sum in `capture-pane`.
  **Negative control:** the same readiness then hang-up against `kitty claude -w <n2>` without
  `--tmux` leaves no live kitty. With `KITTY_TMUX_E2E=1` a missing prerequisite **fails** the test;
  without it the test skips (developer machines).

  CI specifics, each for a reason:

  | Rule | Why |
  |---|---|
  | Installs the pull request's checkout, not PyPI | The claim is about this code. (The review job's PyPI choice, `.github/review/rules/ci.md`, answers a different question.) |
  | **Egress is enforced below kitty:** the test runs as a dedicated unprivileged user whose outbound traffic an `iptables` owner-match rule drops except loopback, DNS and the egress gateway's resolved address and port; a self-check proves a direct connection from that user fails before the test starts | The PR's kitty decides routing at runtime with the organisation's real credentials, so kitty's own `egress show` cannot be the guarantee: a PR that breaks egress would also break that check. Scoping the rule to one user keeps the runner's own connection to GitHub working. The review job's other enforcers (`configure_kitty.py`'s shape check, `KITTY_EGRESS_PROXY: ""`, `egress show`) are kept as well. *Product owner, 2026-09-13.* |
  | **Everything the test touches belongs to the test user:** a copy of the checkout in its home, its own virtualenv with the checkout installed, Claude Code installed by the review job's exact line and pin (`bash -s -- 2.1.238`) into its `~/.local/bin`, and kitty's three documents written by `configure_kitty.py` run as that user. `sudo` passes only the three `KITTY_*_JSON` values (for configuration) and, for the test, `KITTY_EGRESS_PROXY=""`, `KITTY_TMUX_E2E=1`, `LANG`, `TERM` and a `PATH` of the test user's own directories. `kitty egress show` runs as that user. All installs happen before the firewall goes up, and **the firewall goes up before any pull-request code sees a credential**: the gateway address is read from the secret by the runner's own interpreter, then `configure_kitty.py` and `egress show` run | `sudo -u` resets the environment and a different user cannot write into the runner's checkout (`-w` writes `.git/worktrees`) or read its `~/.config`. Checking egress for one user and running the test as another would prove nothing. One hand-maintained Claude pin, already guarded both ways by the §8.6 version arm. |
  | The firewall covers IPv4 **and** IPv6 (`iptables` and `ip6tables` owner-match rules), the self-check tries both families, and rules are inserted silently | An IPv4-only rule leaves IPv6 open. The self-check probes IPv4 before the rules (it must connect, or the later failure proves nothing) and after (it must not); IPv6 is confirmed by checking the reject rule exists, because the runner has no IPv6 route and a failed IPv6 probe would prove nothing. Rule listings and failed-rule echoes print the gateway address from a secret. **Accepted costs:** DNS through the loopback resolver stub stays open (name lookups only); a gateway whose address changes after the rules are inserted makes the job fail closed, as a flaky red run rather than a leak. |
  | The exact command is `pytest tests/integration/test_tmux_disconnect.py -m "agent_live" --strict-markers --require-category=agent_live -rfEs` | Naming the file keeps `tests/integration/test_agent_e2e.py` out of this job. The module skips entirely without `KITTY_TMUX_E2E=1` (it spends tokens and edits `~/.claude.json`), and with it every missing prerequisite fails, so `--require-category` cannot pass on a skipped test. `-rfEs`, not `-rs`: `-r` replaces pytest's default `fE`. |
  | tmux is installed by the job | The `ubuntu-latest` image does not guarantee it. |
  | The test seeds the test user's `~/.claude.json` (restoring any previous file afterwards): `hasCompletedOnboarding`, `projects[<repo>].hasTrustDialogAccepted`, and the resolved key's last 20 characters in `customApiKeyResponses.approved`. The child environment drops every `CLAUDE_CODE_*` variable and sets `DISABLE_AUTOUPDATER=1` | Observed in a live trial (Claude Code 2.1.269, fresh home): `-w` refuses outright without trust, then Claude stops at "Detected a custom API key" (kitty sets both `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`), and it auto-installed an update. An inherited `CLAUDE_CODE_TMUX_SESSION` made the trial Claude report the caller's session. The review job never meets any of this because it runs `-p`. |
  | No `--debug`/`--debug-file`; only the port is parsed out of `/proc/<pid>/environ`; failure messages name the failed check and never include pane text | The debug log, Claude's environment and the pane (which can show kitty's egress refusal naming the gateway) all carry secrets into a public job log. |
  | The fork guard is the same as the review job's, and the §8.6 fork arm is extended to every job binding a `KITTY_*` secret | Today that arm checks only jobs using the review action. |

### 3.5 Known limits

- A stale `$TMUX` (a terminal opened from tmux that has since left it) makes kitty launch in place,
  without protection.
- A reattach ignores the new command's profile and flags; kitty says so but does not compare them.
- Activating `agent_live` for one file leaves `tests/integration/test_agent_e2e.py` selected by no
  job, and nothing but §8.2's prose and plan task T-K11 records that.
- The session-name rule (§3.1) was read from Claude Code 2.1.269. Nothing detects a change in a
  later Claude release; the cost of drift is a name mismatch, not a failure.
- If the inner Python cannot start at all (e.g. the interpreter was removed), nothing holds the
  pane; the outer kitty still deletes the environment file.
