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
child), KBR-180 (probing a PID on Windows), and KBR-231 (detaching the child from the launching
console on Windows).

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
| D7 | A missing keys file means auth off; a named-but-missing one refuses to start with a clear error | Before the fix a fresh install could not start a background bridge at all (`parse_keys_file`'s `FileNotFoundError`). Auth off matches the foreground bridge, `kitty claude` and the README; the default file still enables auth when it exists, so installs relying on it keep exactly the behaviour they had. Rejected: requiring a keys file — background would be the only mode demanding a hand-created secrets file. *Product owner, 2026-09-14 (KBR-230).* |
| D8 | On Windows the background bridge child is started detached: `DETACHED_PROCESS \| CREATE_NEW_PROCESS_GROUP` | `start_new_session=True` is POSIX-only, and CPython's Windows `Popen` accepts it and ignores it, so a "background" bridge kept the launcher's console: Ctrl+C there, or closing the window, ended a bridge the user was told runs in the background (KBR-231). Observed on the Windows leg first, per the ticket's first acceptance criterion; the probe lives beside its guard in `tests/bridge/test_bridge_management.py::TestTheWindowsConsoleDetachment`. `DETACHED_PROCESS` gives the child no console at all, so no console event of any console can reach it — the Windows analogue of the `setsid()` `start_new_session` runs on POSIX. `CREATE_NEW_PROCESS_GROUP` additionally disables Ctrl+C group-wide — scoped claim, since `CTRL_BREAK` is always delivered, but a detached child has no console to receive any of it on. `CREATE_NO_WINDOW` (the ticket's alternative) was rejected: it detaches the child from the launcher's console but gives it a hidden console of its own -- a `conhost.exe` per bridge for a daemon that needs no console at all -- and beside `DETACHED_PROCESS` the vendor docs state it is ignored anyway. The flags are integer literals in `manage.py` because `subprocess` imports those names from `_winapi` on Windows only, and the decision (`background_spawn_kwargs`) returns key-disjoint dicts because POSIX `Popen` raises `ValueError` on a nonzero `creationflags`. Scope: `start_bridge` only — service units run `bridge_runner` directly under a manager that already detaches them. |
| D9 | The child gives up its parent's pipes at ready: `bridge_runner` calls `io_encoding.relinquish_output_streams()` once `start_async` has returned, pointing pipe-shaped fds 1/2 at `os.devnull`; the pre-ready no-TLS warning swallows a `BrokenPipeError` and does the same on catch | A `kitty bridge start` that gives up (its 5-second window, KBR-176) exits and closes the child's pipe; every later child write then raised `BrokenPipeError` (KBR-219). The warning print is the documented kill site — it fires *before* the state file, so devnull-at-ready alone cannot save it; the except-branch dup2 is safe precisely because a `BrokenPipeError` proves the reader is gone, and an interpreter-exit flush would otherwise raise the same error a second time (Python's own note-on-SIGPIPE pattern). **Write-site audit:** the warning is the only unprotected pre-ready stderr writer on the success path — every other pre-ready stderr print in `bridge_runner.main` precedes `sys.exit(1)` (a BrokenPipeError there changes a clean exit line into a traceback; the child was dying anyway), and every post-ready writer routes through `logging` or `warnings`, both of which wrap their writes in `except OSError: pass` (CPython issue 5971) — which is why the devnull assertion is an fd-state oracle (`/proc/<pid>/fd/N`), not a behavioural one. **Pipes only:** systemd `--user` hands the child a journal `AF_UNIX` socket, launchd a file or `/dev/null`, NSSM a file or null, supervisord a temp file — none are FIFOs, so the guard never touches them. Residual: runit/s6-style daemontools-family supervisors erect a FIFO between the service and the logger (`runsv(8)`); under those the guard would replace the log pipe with devnull and lose the log stream. Accepted — kitty ships unit installers only for systemd `--user`, NSSM and launchd. **Why not a log file:** KBR-176's rejection stands — a new on-disk file that may hold sensitive text was declined there, and the product owner confirmed devnull-silence over a retained destination (2026-09-21, KBR-96). |

### 1.5 Known limits (recorded, not fixed here)

- **A Windows bridge never shuts down gracefully when stopped.** `TerminateProcess` skips
  `stop_async`, so an opt-in session summary (`KITTY_SESSION_SUMMARY`) is not written.
- **Services running as another account are outside the contract.** The NSSM script sets no
  account, so the service runs as LocalSystem with a different home and profile directory. A
  LaunchAgent with `KeepAlive` is restarted by launchd straight after `kitty bridge stop`.
- **`start` waits 5 seconds.** A bridge slower than that is reported as not ready and left
  running (KBR-176).

### 1.6 Configuration and authentication of a background bridge

Traces to [KBR-230](https://shelpuk.atlassian.net/browse/KBR-230). `bridge.yaml` always reaches the
child via `--config` (§1.2); `kitty.bridge.config.load_bridge_config` resolves it. Authentication is
the keys file, and the decision table has four rows:

| The configuration says | The background bridge does |
|---|---|
| `keys_file:` names a file that exists | Loads it; every route requires a valid Bearer key (`BridgeServer._auth_middleware`), `/healthz` included. |
| `keys_file:` names a file that is missing | Refuses to start: one clear line naming the path and the `bridge.yaml` that named it, exit 1, no traceback. An explicit configuration is never silently ignored. |
| Nothing named, `~/.config/kitty/bridge_keys.txt` exists | Loads the default file; auth on as in row 1. |
| Nothing named, no default file | Starts with auth **off** — every route, `/healthz` included, answers without credentials. |

`resolve_keys_file` (`kitty.bridge.config`) computes the effective file for the two "nothing named"
rows and for `kitty bridge config`'s display, which prints `(none — auth disabled)` for the last one
and marks a named-but-missing file `(not found — start will be refused)`, so row 2 never displays
like row 1.
`bridge_runner` owns row 2's refusal, beside its other startup errors (egress, profile, API key):
the server has no printing convention, and the foreground — which never passes `keys_file` — is
unaffected. A bare `bridge_runner` without `--config` keeps auth off; every real entry point
(`start`, `restart`, the service units) passes `--config`.

The table is evaluated **once, at start**: a keys file created or removed while a bridge runs
changes nothing until `kitty bridge restart`. Falsy values (`keys_file:` with nothing after it,
`""`, `false`, `0`) count as nothing named. A named path that exists but cannot be read as a keys
file — a directory, a permission error, malformed content — fails as it always has (an exception
from `parse_keys_file`); only a *missing* file gets row 2's one-line refusal. And row 4 combined
with a non-loopback `host` is a real exposure, chosen with eyes open: any host that can reach the
machine can use the bridge and its upstream key — set `keys_file` to prevent that.

Why auth off rather than a required keys file: on every install before this fix a fresh start
crashed with a traceback, so background mode worked for no new user, while the foreground and
`kitty claude` run without a keys file and the README's only mention of one treats auth as
conditional. Honouring the existing default keeps
every working install unchanged, and the trust model does not move: the default bind is 127.0.0.1.
*Product owner, 2026-09-14 (KBR-230, option (a)).*

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
  `--settings <tmpfile>` flag whose `env` block repeats them. The env block also carries
  `ENABLE_CLAUDEAI_MCP_SERVERS=false` (KBR-245): Claude Code's documented per-session
  opt-out from claude.ai MCP connectors, which suppresses the banner Claude Code prints when
  `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` shadow the user's claude.ai OAuth login, and
  `CLAUDE_CODE_TMPDIR=tempfile.gettempdir()` (KBR-314): Claude Code's documented override for its
  internal temp dir; without it Claude Code's cross-session messaging daemon falls back to `/` as
  its socket dir, which fails its ownership check on boxes where `/` is neither user- nor root-owned
  and surfaces a remediation hint at every launch.

### 2.1 Launcher lifecycle and crash recovery (KBR-93, KBR-268)

`launch_async` calls `adapter.prepare_launch(...)` before spawning and
`adapter.cleanup_launch(...)` from both a `finally` block and an `atexit`
handler (`cli/launcher.py:241-347`). `SIGKILL`, OOM, and power loss bypass
both, which is what the crash-recovery tier below repairs; a `SIGTERM` landing
in the pre-handler window has the same outcome and is accepted
(`TEST_SUITE.md` §6.3.2). The two adapters patch different surfaces:

- **Claude** (KBR-93) no longer patches any user-owned file: it writes a
  per-session settings file passed as `--settings`, so the user-global
  `~/.claude/settings.json` is only read (stale-value warning) and concurrent
  sessions cannot disturb each other. Its backup trio
  (`save/load/delete_settings_backup`) and `kitty cleanup`'s phase-1 exact
  restore serve damage written by pre-per-session versions.
- **Kilo** (KBR-268) must patch in place: Kilo CLI reads one global config,
  `~/.config/kilo/kilo.json`, and has no per-session settings flag. kitty
  injects a `provider.kitty` block (loopback bridge URL + session API key)
  and overwrites the top-level `model` key with `kitty/<model>`.

The Kilo crash-recovery contract, all of it marker-gated by
`_kilo_kitty_values_present` (loopback `provider.kitty.baseURL`, or a
top-level `model` prefixed `kitty/`; a remote-URL `provider.kitty` alone does
**not** count — a user who named their own provider `kitty` must not have
their config auto-restored):

1. **Clean-capture backup.** `prepare_launch` writes the byte-exact original
   to `~/.config/kitty/kilo-config-backup.json` before patching — but only
   when the captured original carries no kitty markers. *Why:* without this
   rule, the crash → relaunch sequence captures the *patched* file as the
   "original" and silently destroys the only good backup
   (system-design review blocker, 2026-09-18). The invariant this preserves:
   **the backup file never contains kitty markers.** A malformed original
   parses as `{}` and counts as clean — the user's bytes are still the honest
   original.
2. **Ownership-checked restore.** `cleanup_launch` restores (and deletes the
   backup) only when the captured original is clean **and** the current
   `kilo.json` is readable and marker-bearing. Polluted captured original
   (a concurrent session's patch), clean current file (user hand-edit),
   missing file, unreadable file → file and backup left alone. *Why
   marker-presence rather than Claude's per-session injected values:* the
   whole region kitty touches is kitty-owned, and the one case per-session
   precision would add — restoring over a same-config sibling's patch — is
   bounded (that patch is equally dead; the true original wins either way).
   *Why not a `str`-subclass prepare contract like Claude's `_SessionSnapshot`:*
   Kilo's adapter is new; carrying the injected values on the returned string
   would buy precision this analysis shows is unobservable.
3. **`kitty cleanup` Kilo arm.** `run_kilo_cleanup` (in `cleanup_cmd.py`,
   called alongside `run_cleanup` from `_run_cleanup`; the worse exit code
   wins): backup present → exact byte restore (`newline=""` on both legs, the
   KBR-262 contract) when the current config carries markers or is
   missing/unreadable; readable config without markers → the backup is stale,
   delete it, config untouched; no backup → no-op. Restore I/O failures print
   one `Error:` line and exit 1 — there is no heuristic phase to fall
   through to. *Why the unreadable-config verdict is the deliberate opposite
   of `cleanup_launch`'s:* a live session must never guess, so it leaves an
   unreadable file alone; `kitty cleanup` is an explicit repair request, so
   the exact backup wins.

**Accepted residuals** (both documented, both the same class Claude accepts):
a crash of a from-scratch session (no pre-existing `kilo.json`) leaves the
kitty-only file behind — cleanup cannot distinguish it from user-authored
content without a backup; and damage from pre-fix kitty versions (or after a
hand-deleted backup) needs a manual fix. **Rejected alternative** (product
owner, 2026-09-18): a heuristic strip arm for backupless damage — rejected
because a backupless heuristic could remove `provider.kitty` but could never
restore the overwritten `model` key, i.e. it can only half-repair (minimal-fix
precedent KBR-260).

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
| T15 | `run_captured` decodes captured child output with `encoding="utf-8", errors="replace"`, never bare `text=True` | `text=True` decodes the parent side of the pipes with the system ANSI codepage (cp1252 on `windows-latest`); `PYTHONIOENCODING` does not change it, and one child byte the codepage cannot represent raises `UnicodeDecodeError` into every caller (zooba/eryksun, cpython#105312; observed on the Windows leg as KBR-265's scope expansion). Forcing UTF-8 with a non-strict handler makes the read-back locale-independent and crash-free. **Known cost, deliberate:** on a POSIX repo whose absolute path contains a non-UTF-8 byte, the decoded `git rev-parse --git-common-dir` carries U+FFFD and the resulting tmux session name is mangled; reattach still works because the mangling is deterministic, whereas pre-fix the same input crashed the launch outright. The other callers (`tmux -V` parsed by `parse_tmux_version`, the `MARKER` / `ENV_FILE_VARIABLE` equality checks on `show-environment`) all degrade gracefully from a crash to a benign skip or an existing error message — ASCII-only on the happy path, regex-no-match / equality-fail on the hostile path. `errors="replace"` matches the prevailing byte-drain pattern across `bridge/server.py`, `preamble_hold.py`, `tool_audit.py` and the providers (the lone `backslashreplace` decode is KBR-154's `manage.py:667` diagnostic drain); the triage note chose it explicitly. The child-side alternative (`PYTHONIOENCODING=utf-8:replace` in the spawned environment) was rejected: `run_captured` runs commands we do not own, and the kwarg shape is one place per call with no environment plumbing. The companion test-side read-back (the console-break launcher's `Popen`, KBR-265) carries the same kwargs, held by an L1 structural test asserting both keyword literals in the spawning method's source — the KBR-275 guard pattern; it is deliberately refactor-sensitive (extracting the `Popen` into a helper trips it), which is the guard doing its job. **Obsolescence horizon:** Python 3.15 (PEP 686, UTF-8 mode by default) makes bare `text=True` decode UTF-8, so the `encoding="utf-8"` half becomes the platform default — but `errors="replace"` stays load-bearing, because the default error handler remains `strict` and an invalid byte would still raise. |

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
  | **Everything the test touches belongs to the test user:** a copy of the checkout in its home, its own virtualenv with the checkout installed, Claude Code installed by the review job's exact line and pin (`bash -s -- 2.1.238`) into its `~/.local/bin`, and kitty's three documents written by `configure_kitty.py` run as that user. `sudo` passes only the three `KITTY_*_JSON` values (for configuration) and, for the test, `KITTY_EGRESS_PROXY=""`, `KITTY_TMUX_E2E=1`, `LANG`, `TERM` and a `PATH` of the test user's own directories. `kitty egress show` runs as that user. All installs happen before the firewall goes up, and **the firewall goes up before any pull-request code sees a credential**: the gateway address is read from the secret by the runner's own interpreter, then `configure_kitty.py` and `egress show` run | `sudo -u` resets the environment, **except that the runner image sets `XDG_CONFIG_HOME` and its siblings to the runner's home system-wide and `sudo` re-applies them**, so every kitty-facing command as the test user sets them to that user's home explicitly (observed in the first CI runs: kitty looked for its stores in a directory the test user cannot read). A different user also cannot write into the runner's checkout (`-w` writes `.git/worktrees`) or read its `~/.config`. Checking egress for one user and running the test as another would prove nothing. One hand-maintained Claude pin, already guarded both ways by the §8.6 version arm. |
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

---

## 4. Protocol translators: stream item positioning

Traces to [KBR-226](https://shelpuk.atlassian.net/browse/KBR-226) and
[KBR-240](https://shelpuk.atlassian.net/browse/KBR-240). Related: KBR-221, KBR-232 (the
Anthropic-*upstream* translated adapters — a different seam, §5). This section is seeded by
KBR-240's area and grows as later work reaches the rest of the translator layer.

### 4.1 Components

| Component | Role |
|---|---|
| `MessagesTranslator` (`kitty.bridge.messages.translator`) | Claude Code's Messages wire ⇄ Chat Completions. Streams Anthropic SSE; positions content blocks by **index**. |
| `ResponsesTranslator` (`kitty.bridge.responses.translator`) | Codex's Responses wire ⇄ Chat Completions. Streams Responses SSE (`kitty.bridge.responses.events` formatters); positions output items by **`output_index`**. |
| `GeminiTranslator` | Gemini CLI's wire ⇄ Chat Completions. Positional `parts[]` — no index in its grammar, so no allocation contract (checked by KBR-226's sibling audit). |
| `kitty.bridge.engine.ToolCallBuffer` | Assembles streamed tool-call arguments; one per upstream call. |

On these routes the upstream always speaks Chat Completions (native-Anthropic providers skip
the translators entirely — see the M2 row of `TEST_SUITE.md` §3.2). CC's stream carries its
own authorial `tool_calls[].index`; the *inbound* wire's positional slots are kitty's to
allocate — the upstream authorial number and the downstream positional slot are different
things that happen to look alike when both start at 0.

### 4.2 The To Be contract: one counter per translator, allocated at open

Each streaming translator owns **one counter** for its wire's positional slot. A slot is
allocated when an output item **opens** — the first reasoning delta, the first content
delta, each newly seen tool-call id — and every event referencing that item carries the
recorded slot, on both the streaming path and the two closing paths (the finish chunk, and
the EOF-without-finish fallback: `finalize_interrupted_stream` /
`synthesize_completed_events`).

- Every opened item gets the next distinct, increasing slot; one added/done pair per item.
- **The CC `tool_calls[].index` is a routing key only** — which `ToolCallBuffer` and meta an
  argument delta lands in — never the downstream slot.
- **Decided stream shape** (G39, and KBR-240 keeps it): items opened out of order may
  overlap in time and need not close in order — clients key items by id and position by
  slot — but each slot opens once, closes once, closes only after it opened, and carries no
  delta outside its own window.
- **Responses specifics** (KBR-240): `response.function_call_arguments.delta` /
  `...done` carry the owning call's `output_index` — the vendor grammar defines it as a
  required field (OpenAI SDK types, verified 2026-09-14); kitty omitted it. And
  `response.completed`'s `output` array is ordered by allocated `output_index`, so a client
  aligning array position with indices reads it correctly.

### 4.3 Decisions, and why

| # | Decision | Why, and the rejected alternative |
|---|---|---|
| X1 | A shared per-translator counter allocated at open, not a slot derived from upstream numbers | Deriving the Responses `output_index` from CC's tool-call index (the pre-fix behaviour) collides the moment two item kinds are live: text and reasoning were pinned to 0 and the first call took CC's 0, so any pair of the three item kinds claimed one slot and two `output_item.done` events closed it. The downstream slot positions *our* items; anchoring it to an upstream authorial number leaves it undefined whenever an item opens outside the anchor's frame. Mirrors KBR-226's Messages fix — one mechanism per wire, not one per defect. |
| X2 | `output_index` added to the arguments events rather than left omitted | The vendor grammar requires the field on both events (verified against the generated SDK types). A client positioning by `output_index` — the client class the defect is about — needs it there as much as on `output_item.*`. Additive: existing clients tolerate the extra field. |
| X3 | `response.completed`'s `output` sorted by slot, not by emission order | Opening order and slot order diverge once text can open before reasoning (or a call before text). The completed array is the client's canonical final view; positional clients read it by position. Two lines; removes the last positional surprise. |
| X4 | On the response direction, carry the upstream CC `tool_calls[].id` to the emitted `functionCall` part; do **not** synthesise | Mirror of KBR-195 (request-side): the response direction used to read only `tc["function"]["name"]` and `tc["function"]["arguments"]`, dropping the upstream id by omission. KBR-195's request-side `or`-echo only works if the client received an id to echo back, which the response direction never gave it. Rule is **emit when present, omit when absent** (Gemini `FunctionCall.id` is optional per `v1beta`); synthesis stays on the request side, where Chat Completions requires an id. **Presence tests differ on purpose and the asymmetry is load-bearing:** the response side uses `is not None` (the contract is `Optional[str]` — emit the wire value verbatim), the request side uses `or` (an empty string is no usable id, synthesise). Emitting `id: ""` verbatim would round-trip into the request-side `or` and synthesise, mis-pairing the loop this rule closes. The streaming open branch is **name-keyed** per §4.2 — the id riding the opening (name-bearing) delta is stored; an id on a later delta is deliberately ignored. The Gemini reader already reads `functionCall.id` faithfully (KBR-36), so the fix is translator-only and no reader change is owed. The register row that would claim `reply.parts[*].id` waits on T-D10's response-direction Reply projection ([KBR-59](https://shelpuk.atlassian.net/browse/KBR-59), To Do) — the row to add there is the response-direction mirror of M18/M19, conditional on the upstream chunk carrying an id; a `NOT_PROJECTABLE` row is not an option (KBR-195 §8: the reader projects `ToolUse.id` on the request side, and the response side is symmetric). |

### 4.4 Verification

- **L1** (`tests/bridge/test_messages_translator.py::TestParallelToolCallBlockIndices`,
  `tests/bridge/test_responses_translator.py`): distinct increasing slots, interleaved
  argument routing by per-call meta, one close per slot at its own slot, a later item at
  the next free slot, EOF fallback, reset.
- **L1** (`tests/test_gemini_translator.py::TestTranslateResponseToolCallIdEcho`,
  KBR-257): sync present/absent echo and the streaming carry from the opening delta
  to the finish emit. Two of three tests red at the base revision before the fix
  (KBR-221 plan §16 discipline); the absent case is the regression guard.
- **Server-level** (`tests/bridge/test_parallel_tool_use_stream.py`,
  `tests/bridge/test_responses_output_index_stream.py`): the client-visible byte stream
  walked end to end against the decided shape.

### 4.5 Known limits (recorded, not fixed here)

- A repeated id-chunk for an already-open CC tool-call index re-enters the open branch and
  stays malformed (G39's scope-out, shared with the Responses translator).

---

## 5. Response translation: the four stream handlers

Traces to [KBR-227](https://shelpuk.atlassian.net/browse/KBR-227) and
[KBR-232](https://shelpuk.atlassian.net/browse/KBR-232). What the suite must prove about this
area is in `TEST_SUITE.md` (invariant I1 and register rows M12/M17); this section is the
components and the rule — the *upstream* seam that feeds §4's translators.

### 5.1 Components

| Component | Role |
|---|---|
| `BridgeServer._stream_messages` | The `/v1/messages` inbound stream. On a Messages-wire upstream it forwards the raw SSE (KBR-227); on a Responses-wire upstream it converts each line through `OpenCodeGoResponsesCCStreamConverter` first (KBR-274); otherwise it translates CC chunks to Messages events. |
| `BridgeServer._stream_responses` / `_stream_chat_completions` / `_stream_gemini` | The Codex, Chat Completions and Gemini inbound streams. On a Messages-wire upstream they convert (KBR-232); otherwise they translate CC chunks to their protocol. |
| `BridgeServer._serves_messages_wire` | The one answer to "does this request's upstream speak Anthropic Messages?". Every branch that decides how a Messages-wire stream is handled asks it — a change to the rule cannot reach one site and miss another. |
| `AnthropicCCStreamConverter` (`kitty.providers.anthropic`) | The stateful Anthropic-SSE → Chat Completions-chunk converter. One instance per upstream attempt. |
| `MessagesTranslator` / `ResponsesTranslator` / `GeminiTranslator` | The CC-chunk → client-protocol translators. They never see Anthropic events: the converter or the raw forward sits upstream of them. |

### 5.2 The rule

Each handler asks `_serves_messages_wire(cc_request)` **once per attempt**, after backend
selection. On `/v1/messages` a yes means forward the upstream's bytes unchanged — the client
already speaks the upstream's protocol, and conversion would drop thinking signatures
(KBR-227). On the other three a yes means feed every `data:` line through
`AnthropicCCStreamConverter` and let the converted lines re-enter the same per-line body a
Chat Completions upstream's would: finish buffering, the empty-response ladder, usage
attribution and in-stream error detection are all the handler's existing, already-proven
logic. A no means the handler consults `_stream_converter_for`: a Responses-wire upstream
(KBR-137) has its lines converted through `OpenCodeGoResponsesCCStreamConverter` and the
converted lines re-enter the same per-line body (KBR-274 — before it, `/v1/messages` walked
the six-step empty ladder on a healthy stream because no converter was wired there); any
other non-Messages wire behaves byte-identically to the pre-KBR-232 code.

### 5.3 Decisions, and why

| # | Decision | Why, and the rejected alternative |
|---|---|---|
| S1 | Convert on the three non-Messages protocols; forward only on `/v1/messages` | Only `/v1/messages` shares the upstream's wire. Conversion there would lose signatures (KBR-227); forwarding on the other three would hand clients Anthropic SSE they cannot read. KBR-274 narrowed "forward only" to *Messages-wire* upstreams: `/v1/messages` on a Responses-wire upstream (the OpenCode Go route) now converts, because the alternative was handing the handler Responses events its CC translator silently ignored — six empty-ladder retries and a 502 on a healthy stream. |
| S2 | A stateful converter class, not a stateless per-event map | A `tool_use` block's `input_json_delta` fragments have no meaning without the `content_block_start` that allocated the block's `tool_calls` index. The stateless map is precisely why every tool call was lost (KBR-232). |
| S3 | Converted lines re-enter the handler's existing per-line body | The alternative — a parallel write path — forks the finish/empty/usage/error logic per protocol. The converter's `[DONE]` sentinel and malformed-line passthrough are byte-identical outputs, so the body's residual `translate_upstream_stream_event` call sites stay harmless; an L1 test pins that identity as a contract, not a coincidence. |
| S4 | Gate and converter re-evaluated per attempt | A failover can land on a Chat Completions-wire backend mid-handler; a stale converter would mangle its Chat Completions stream. |
| S5 | `thinking_delta` → `reasoning_content`; signatures dropped | The Chat Completions wire has no signature slot, so preservation is impossible; M17's strip-and-retry recovers the round-trip rejection instead (KBR-238). |
| S6 | The three loops run wider by the strip budget, with an attempt correction | Same rationale KBR-238 recorded on `_stream_messages`: a strip gets its attempt back, so the empty-response schedule is not pulled forward. |
| S7 | `_stream_responses` opens the lifecycle lazily, on the first non-finish write of each attempt | `translate_stream_start` and `translate_stream_chunk` draw from the same `_seq` counter, so translating the lifecycle after the first chunk had been translated would put `sequence_number` 3 and 4 on the wire ahead of the chunk's 0, 1, 2 — the translation therefore runs speculatively at attempt start, before any chunk, and the two strings are written on the first real event and invalidated at every `translator.reset()` inside the loop (KBR-242; gap G41). Writing eagerly, before the first chunk is translated, was rejected: an all-finish first chunk is how an empty response presents, and publishing the lifecycle before the empty verdict is known would put a half-open lifecycle on the wire exactly where the failover ladder is about to retire the attempt. A purely-empty attempt publishes nothing, so KBR-247's `events_emitted` model survives; the exhausted-ladder fallback stays an empty 200 (KBR-235's territory); the error paths never open the lifecycle. Tests: `tests/bridge/test_responses_stream_lifecycle.py`, and the KBR-240 walk's opening + exact-`sequence_number` assertions |
| S8 | Cross-class re-dispatch (`_stream_messages` KBR-249; `_stream_responses`, `_stream_gemini`, `_stream_chat_completions` KBR-254): when a plain-POST branch's failover selects a `use_custom_transport` provider, the function re-enters the custom-transport branch with the failover-selected provider as its own initial selection | The plain-POST branch cannot drive a `use_custom_transport` provider via `session.post(...)`; the bridge must speak the protocol that matches the selected backend's class. Without re-dispatch the failover silently delivers an empty `200` (the bug KBR-235 exposed). The custom-transport branch's symmetric `custom → plain` fall-through — `src/kitty/bridge/server.py:4185-4204` (the cross-mode select with the three pops) then `4265-4270` (the `continue` entering the plain block) — is unchanged in behaviour. The re-dispatch bound is `(2 * n_backends) + 1` per request so a pathological cooldown-expiry ping-pong surfaces an honest error instead of looping; the cap-hit error carries the route's D4 discriminator set to `"cross_class_exhaustion"` — `code` (plus the parent `reason: "cross_class_exhaustion"` marker) on Responses, `reason` on Gemini, `type` on Chat Completions — so a client that branches on its route's discriminator can tell the cap-hit apart from `empty_response` and `upstream_error` (D4, KBR-241 / KBR-247 / KBR-250). Chat Completions does **not** carry `reason` — its route's D4 discriminator is `type` alone (see §5.4). The three sibling handlers prepare their SSE response eagerly (`sr.prepare()` before the dispatch loop), so unlike `/v1/messages` — which defers prepare and answers a cap-hit with a bare JSON `502` — a sibling cap-hit surfaces as the route's in-stream terminal event followed by the handler's existing post-loop. |
| S9 | Cross-class re-dispatch reuses the failover-selected provider, never re-selects | Re-selecting would consume a new draw from the deterministic test stub and break the pinned two-draw invariant; semantically, the failover already chose — the branch re-enters with that choice intact. |
| S10 | Gemini's in-stream-error exhaustion arm writes its terminal diagnostic (KBR-99) | The arm ran log + `break`: the stream ended `status="incomplete"` with no event, and a Gemini CLI session saw the stream just stop — the one silent arm, since Responses writes its `upstream_error` SSE event (KBR-247) and Chat Completions ships its clean marker. The fix is the route's own convention, one `data: {"error": {"code": 502, ...}}` event before the break — no new shape invented. A `finalize_interrupted_stream` analogue was rejected: Gemini's `streamGenerateContent` SSE has no typed completion event, so there is nothing to synthesize; the single error event is the whole closing vocabulary the route has. |
| S11 | Translated `/v1/messages` exhaustion unifies on D4 (KBR-99) | KBR-235 left the route's exhaustion split: a contentless reply *with* a finish chunk exhausted into M12 fallback text while a no-finish stream got the D4 `502 empty_response` — the same upstream failure producing a normal-looking assistant turn or an error depending on an accident of the upstream's chunking. Q14(a)'s rationale already says a `200` carrying substituted text is the one thing the route must never produce, so both arms now exhaust into D4. The translator's fallback synthesis is untouched: M12 stays live on non-streaming replies and the other inbound protocols, where the substitution is a decided product behaviour rather than a stream-recovery outcome; only the streaming write path stops delivering it. The alternative — one more retry for the finish-chunk case — was rejected: the ladder already ran to exhaustion by the time the branch is reached. |
| S12 | Streaming D3 lands on the translated route (KBR-99) | A truncation (`max_tokens` / context window) before content is a failure no retry can improve — D3's native-route rationale verbatim — yet the translated route retried it and fallback-ized it. The stop reason is available in the buffered finish events, so the check adds no plumbing: when the empty verdict holds with nothing emitted and the finish events carry a truncation stop reason, the handler returns the `400` `invalid_request_error` with `reason: "<stop_reason>_before_content"` and the ladder ends on that attempt (a `400` also stops the agent re-sending, where a `5xx` invites exactly that). D4 stays the exhaustion for every other stop reason; the post-emission arm is untouched — a truncation seen after bytes reached the client is Q14(a)'s ending, not D3's. |
| S13 | The Messages-wire pre-content error ladder charges the stream-error cooldown (KBR-99) | KBR-241's amendment kept two differences deliberately; the health-model half meant a persistently-erroring backend kept drawing ~1/n of the attempts on a balancing pool while its CC-wire sibling cooled under `_get_stream_error_cooldown`. The parity fix reuses that cooldown unchanged on the hold's error ladder; the recognition rule (name line or `data.type`) and the exhaustion payload (the provider's error re-embedded, `reason: "upstream_error"`) from the D2 amendment are untouched. A separate Messages-wire cooldown table was rejected: two ladders for one failure class is precisely the drift the parity exists to close. |

### 5.4 Known limits

- On `/v1/messages` a converted Responses stream's role-only opening chunk
  (`response.created`) translates to no events and starts no lifecycle, so the FI-8.3
  truncation guard stays inactive until real content lands; a completed-but-content-less
  Responses stream enters the empty-response ladder through the existing
  `translator.response_was_empty` branch, exactly as a content-less CC stream does
  (KBR-274).
- **KBR-276 closed the raw-CC gap on `/v1/chat/completions`.** KBR-248
  (closed in PR #205) added the hold converter-gated
  (`release = stream_converter is not None`) and explicitly deferred widening
  it to raw-CC upstreams: a content-less completion from OpenAI, OpenRouter,
  DeepSeek, or any plain-POST CC backend reached the client as a well-formed
  skeleton (role chunk → finish → `[DONE]`), the empty-response ladder could
  not fire there, and the backend was marked healthy. KBR-276 authorises the
  widening: the hold now engages unconditionally on the route (the converter
  gate is dropped — the classifier `_cc_chunk_carries_content` works on raw
  CC chunks the same way it works on converter-emitted ones), so an empty
  raw-CC attempt is pre-emission and the existing ladder fires. The mechanism
  is the mirror of KBR-248's: non-content Chat Completions lines (the role
  chunk, the finish chunk, `[DONE]`) are withheld until the first
  content-bearing delta (non-empty `content`, `tool_calls`, or
  `reasoning_content`); a content-bearing stream is byte-identical to
  today's output because the held preamble flushes ahead of the first
  content line. The byte cap (`PreambleHold.MAX_HELD_BYTES`, D5 fail-open),
  the non-JSON fail-open, the ladder reachability, and the D4 exhaustion
  terminal (`type: "empty_response"` + `[DONE]`, backend not marked
  healthy) are unchanged. **Accepted cost:** every content-less raw-CC
  completion now pays the empty ladder's retry latency
  (`_EMPTY_RETRY_DELAYS` + `_EMPTY_FINAL_DELAYS`, ~80 s on a single-backend
  pool) instead of being delivered as a skeleton. A pre-content in-stream
  `error` chunk on a raw-CC upstream now takes the in-stream failover arm
  (pre-emission) rather than the "error after content" arm — the same
  semantics the converted route has had since KBR-248. **Scope-out,
  deliberate (KBR-248 → KBR-276); closed by KBR-287:** the plain-POST
  branch's hold covered the converted route and every raw Chat Completions
  upstream; the `use_custom_transport` segment the KBR-254 cross-class
  re-dispatch routes into synthesised its own CC stream and had no hold,
  so an empty completion from a `use_custom_transport` failover target
  still delivered the skeleton within the crossing bound. KBR-287 retired
  that scope-out (the fix is recorded below). **Usage note:** usage a
  discarded
  empty attempt carried is never attributed (the D4 exhaustion terminal
  logs no completion); pre-existing behaviour shared with the converted
  route. The reasoning asymmetry the KBR-248 record called deliberate is
  closed by KBR-277 below; pinned streaming-side by
  `test_a_reasoning_only_raw_cc_prefix_releases_the_hold`. **KBR-285 closed the content-set gap on `/v1/chat/completions`.** Both
  `_cc_chunk_carries_content` (the streaming hold's release predicate) and
  `BridgeServer._is_empty_cc_response` (the non-streaming detector's Chat
  Completions arm) widened from KBR-248/KBR-277's three-shape set to six:
  non-empty string `content`, non-empty list `content` (multimodal parts),
  non-empty `tool_calls` list, truthy dict legacy `function_call`, non-empty
  string `refusal`, non-empty string `reasoning_content`. The two predicates
  stayed as physical functions with a deliberate byte-for-byte mirror (the
  KBR-277 pattern) — the mirror is the divergence guard, mutation-tested
  side-by-side in the new `content_classifiers` group. On the
  `/v1/messages` translated route, `MessagesTranslator` carries the same
  six shapes: `translate_stream_chunk` coerces list `content` via
  `_extract_text_content`, treats `refusal` as text, and maps legacy
  `function_call` onto the `tool_calls` machinery with a synthesised
  index/id and accumulating arguments; `translate_response` maps a legacy
  `message.function_call` to one `tool_use` block. The auto-reviewer
  surfaced that the **same consumer-side gap** lived on the two sibling
  translators that share the same widening path — `ResponsesTranslator`
  (`/v1/responses`, Codex CLI) and `GeminiTranslator`
  (`/v1beta/...:streamGenerateContent`, Gemini CLI). Both are extended in
  this PR with the same three-shape coercion, parallel
  `translate_stream_chunk` / `translate_response` for each translator
  (per-translator list-extract helpers, matching the existing
  `_strip_thinking_tags` precedent). Pre-fix, `/v1/responses` non-stream
  + list `content` would have crashed the handler's catch-all as `500
  internal_error` (TypeError in `_strip_thinking_tags(content)` when
  `content` is a list), and `/v1/responses` + `/v1beta` streams + refusal-
  only or legacy-`function_call`-only replies would still have tripped the
  ladder (their `response_was_empty` counted only the pre-widening shapes).
  `TranslationEngine`'s `_FINISH_REASON_MAP` learned the legacy
  `"function_call"` value so its stop_reason maps to `"tool_use"` the
  same way `"tool_calls"` does; Gemini's `_CC_TO_GEMINI_FINISH` learned
  `"function_call": "STOP"` (Gemini v1beta ends tool turns on STOP, so the
  default mapping already lands right). **OpenAI-spec vs OpenAI-compat
  tension:** OpenAI's first-party `ChatCompletionStreamResponseDelta`
  declares `content: string | null` (list content is **not** in the
  first-party spec); the widening's list-`content` clause rests on
  OpenAI-*compatible* multimodal backends (vLLM serving image-capable
  models, OpenRouter for image outputs). The two first-party shapes the
  ticket names — `refusal` and the deprecated `function_call` — are
  confirmed in OpenAI's OpenAPI schema. **Pre-existing drift kept,
  deliberately:** the
  non-streaming `content` clause uses `.strip()` (whitespace-only content
  is empty) while the streaming predicate's `content` clause uses `!= ""`
  (whitespace-only content is content). The widening's four new clauses
  take the streaming-side spelling on both sides so no new drift is
  minted. Tests: `tests/bridge/test_raw_cc_empty_hold.py` (streaming,
  three new "does-not-fire" tests), `tests/bridge/test_empty_response_retry.py`
  (non-streaming unit + bridge twin), `tests/bridge/test_messages_translator.py`
  (translator coercion), `tests/bridge/test_responses_translator.py`
  (sibling-route coercion, Codex CLI), `tests/test_gemini_translator.py`
  (sibling-route coercion, Gemini CLI),
  `tests/bridge/test_empty_response_reasoning_properties.py` (agreement
  property extended to the three new axes).
- **KBR-287 closed the last leg — the `use_custom_transport` segment of
  `_stream_chat_completions`** (grep anchor: `# Custom-transport providers
  return Responses API SSE but CC clients`). A content-less completion from
  Bedrock, Ollama Cloud, the Codex subscription — any adapter resolving
  `use_custom_transport = True`, reached as the initial draw or through a
  KBR-254 cross-class re-dispatch — had synthesised its own CC chunk
  sequence and written it unconditionally: role chunk → finish → `[DONE]`,
  the ladder unable to fire, the backend staying healthy while a balancing
  pool kept routing to it. The branch now records its synthesised
  payloads/lines and applies one up-front verdict through the shared
  `_cc_chunk_carries_content` before any write: content-bearing → write
  every line in synthesis order (byte-identical wire output); content-free →
  write nothing and take the ladder. Four decisions the review settled,
  each against a plausible alternative:
  - *Judge-first, not an incremental hold-walk.* The plain-POST hold
    buffers because a streaming branch does not know the future when its
    first line arrives; this branch parses the entire upstream response
    before emitting, so there is no unknown future to buffer against. A
    `held` buffer here would be write-deferral with extra state, and the D5
    `MAX_HELD_BYTES` cap could never fire (the only holdable lines are the
    synthesised role/finish/`[DONE]`, tiny against 10 MiB) — the cap and
    the non-JSON fail-open are satisfied **by construction**. This
    deliberately simplifies the ticket's mechanism wording, which assumed
    the plain-POST line-arrival shape.
  - *The ladder ends in `empty_response`, not the ticket's
    `cross_class_exhaustion`.* The ticket's acceptance named the cap-hit
    terminal; reusing it would break §5.3 S8's promise that a client
    branching on this route's `type` can tell the crossing-cap hit (a
    pathological ping-pong — a configuration problem) apart from
    `empty_response` (the upstream returned nothing — transient), and
    would make the same all-attempts-empty failure carry different
    discriminators depending on pool composition. The branch emits
    `_NATIVE_EMPTY_REPLY_MESSAGE` + `type: "empty_response"` + `[DONE]`,
    uniform with the plain-POST twin; the backend is not marked healthy
    and no usage is logged (a discarded empty attempt is not a
    completion).
  - *The empty arm's backend selection is class-agnostic, mirroring the
    plain-POST idiom* (grep anchor: `Check for empty response
    (pass-through: no content bytes written)`). A custom-first tier pair
    was rejected: empties never mark a backend unhealthy, so a mixed pool
    [custom-empty, plain-good] would have spent every attempt re-selecting
    among customs and never tried the plain backend — worst exactly in the
    cross-class scenario this ticket exists for. The selected provider's
    class decides: custom → re-normalise + refresh the
    `_resolved_key`/`_provider_config` keys + next attempt; plain → pop
    the three custom keys + the branch's fall-through (grep anchor:
    `Cross-mode failover: entering standard streaming path`). The
    custom→plain crossing is uncapped like the exception path's arm;
    termination is bounded by the reverse direction — only plain→custom
    crossings `continue` the dispatch loop, capped at `(2 * n_backends) +
    1`, and a custom→plain fall-through happens at most once per pass.
  - *The attempt bound is `n_backends + len(_EMPTY_FINAL_DELAYS)`, not the
    plain-POST `(_MAX_RETRIES + 1) * n_backends + len(...)`.* The
    plain-POST bound bakes in that branch's transport-error ladder (6
    attempts on a single-backend pool with `_MAX_RETRIES = 3`); the custom
    branch's transport errors ladder within `n_backends` via its own
    exception path (grep anchor: `Custom-transport failover: attempt`),
    so its "original" budget is the failover walk and only the empty
    ladder extends it (3 attempts single-backend, all against the same
    provider — empties never mark a backend unhealthy). The
    final-delay prologue mirrors the plain-POST loop's (grep anchor:
    `Empty upstream response: final retry in`); the exception path's
    `attempt < n_backends - 1` gate keeps its meaning.
  Two structural facts the next reader needs: the synthesis projects only
  `content` and `tool_calls` — neither parser surfaces
  `reasoning_content`, so a reasoning-only completion from a custom
  transport synthesises the empty shape and ladders (the reasoning was
  never delivered pre-fix either; the projection gap is not this ticket's
  to close); and of KBR-285's widening set only the **list-content**
  clause is reachable here (refusal and legacy dict `function_call` are
  never projected), consumed at the same predicate the plain-POST branch
  uses — one classifier, both branches, lockstep by construction. Usage
  logging is log-on-release: the content path logs exactly as today,
  including on client disconnect (the branch parses atomically, so usage
  is fully known regardless of client state; the plain-POST
  never-log-on-disconnect is a structural consequence of incremental
  arrival, not a policy to copy). Tests:
  `tests/bridge/test_custom_transport_empty_hold.py`
  (the KBR-276 harness shape, canned bytes through the branch's real
  parse step, parametrised over `BedrockAdapter` / `OllamaCloudAdapter` /
  `OpenAISubscriptionAdapter`; the ticket's `vertex` mention is a ticket
  correction — `VertexAIAdapter` is a plain-POST OpenAI-compatible
  passthrough on this tree and was already held by KBR-276). **The CI
  round-1 review closed a second gap in the same change:** Bedrock's
  `stream_request` emits translated CC-SSE bytes, which the branch's
  Responses-SSE fallback cannot read — so every Bedrock completion
  parsed content-free, pre-KBR-287 the bridge answered with a content-free
  skeleton regardless of the completion's real content, and the first
  KBR-287 cut made that a guaranteed ladder-to-terminal failure.
  `BedrockAdapter.parse_stream_to_cc_response` (mirroring
  `OllamaCloudAdapter`'s) is the fix: the branch now judges Bedrock's
  real content, the botocore harness's bridge-driven test
  (`test_a_streamed_request_via_the_bridge_yields_content_and_finish_reason`)
  asserts content and finish_reason reach the client, and the same
  single-predicate rule holds — the parser feeds `_cc_chunk_carries_content`
  through the synthesis like every other adapter.
- **KBR-293 closed the sibling legs — the `use_custom_transport` segments of
  `_stream_responses` and `_stream_gemini`** (grep anchors:
  `KBR-293: collect, don't write` in both handlers). The falsification the
  ticket required came out worse than the skeleton class alone: on
  `/v1/responses` the segments piped provider bytes through `_tracked_write`
  unconditionally, so an empty completion from any of the three
  custom-transport adapters delivered a skeleton with the ladder unable to
  fire (13/14 pre-fix bridge-level tests red), and on `/v1/gemini` the wire
  was wrong for **every** adapter — Bedrock/Ollama Cloud emit CC-SSE on all
  routes, and the subscription (no `_original_body` on the Gemini route)
  emits Responses-SSE — so even content-bearing completions never arrived as
  Gemini events (14/14 red). Both segments now run the KBR-287 judge-first
  shape — collect (`_collect`, not `_tracked_write`; the `_bytes_written`
  guard drops as vacuous), parse (`parse_stream_to_cc_response` dispatch /
  `_parse_sse_to_response` fallback), synthesise the CC chunk list (the
  KBR-287 payload shape ported physically — the KBR-277/KBR-285
  mirror-as-divergence-guard convention; no shared helper was extracted),
  judge through the same `_cc_chunk_carries_content` call site — plus the
  one step KBR-287 did not need: **the synthesis is translated through the
  route's own translator** (`ResponsesTranslator.translate_stream_chunk` /
  `GeminiTranslator.translate_stream_chunk`) before any write, because on
  these routes the route's wire is Responses/Gemini events, not CC chunks.
  The KBR-287 four review-settled decisions carry over verbatim: judge-first
  not hold-walk; the ladder ends in the route's `empty_response` D4
  discriminator (`code` on Responses via `responses_format_error` +
  `synthesize_completed_events(status="incomplete")`, `reason` on Gemini via
  the 502 error event — not `cross_class_exhaustion`, per §5.3 S8);
  class-agnostic empty-arm select (custom → refresh keys + continue, keep
  `_original_body` on Responses; plain → pop keys + the fall-through
  `break`); attempt bound `n_backends + len(_EMPTY_FINAL_DELAYS)` with the
  final-delay prologue. Usage is log-on-release (`_log_usage` outside the
  disconnect guard); neither content arm marks the backend healthy (KBR-287
  parity). Three facts make reusing the handler-level translator
  state-leak-safe, and a future refactor must preserve all three: empty
  attempts never translate (the judge is pre-emission), the plain→custom
  crossing sites reset the translator before re-entry (KBR-254), and the
  content arm ends the request — the lifecycle opening is written
  unconditionally there, not lazily, because the verdict is already known.
  **§11 Q14(a) on these two branches is now satisfied by construction, not
  by a guard:** pre-fix the `_bytes_written` flag stopped a post-write
  failure from failing over (a second backend's bytes would splice into the
  first's); post-fix no collected byte reaches the socket before the
  verdict, so a failing attempt's partial bytes are discarded and the
  failover proceeds — the KBR-247-era tests
  (`tests/bridge/test_post_emission_no_failover.py`,
  `TestCustomTransportFailureAfterBytes`) pin the stronger guarantee (the
  failed attempt's bytes never ship; the next backend's content is the only
  content), and the Q14(a) rule itself is unchanged everywhere bytes still
  stream incrementally (the plain-POST paths).
  **Fidelity callout the PO signed off via PR review:** `/v1/responses` ×
  subscription was a native Responses-SSE passthrough pre-fix (reasoning
  summaries included); post-fix it runs parse → synthesise → translate, and
  `_parse_sse_to_response` drops reasoning. A **reasoning-only** completion
  from the subscription on this route flips from *delivered* (pre-fix) to
  *ladder → `empty_response` terminal* (post-fix) — deliberate, pinned by
  `test_a_reasoning_only_custom_transport_completion_takes_the_ladder[openai_subscription]`;
  the wire-aware passthrough alternative was rejected (two write paths, a
  per-pair wire heuristic, and no empty-ladder guard for that cell).
  Regaining fidelity later means lifting reasoning into the parse
  projection — a parse-step widening, not a branch fork. **Recorded
  asymmetries, deliberate:** the custom segments' transport-error terminal
  (the `except` arm) still ships its error event without a lifecycle close,
  unlike the plain path's catch-alls which fall through to the post-loop
  synthesize — pre-existing, not this ticket's defect class; and
  `/v1/messages`' custom segment — closed by KBR-297 below; the smaller-class
  sibling defect. KBR-254's sibling-test stubs were corrected to what real
  adapters emit (`_fake_hello_cc_stream`; the Responses-SSE stub pinned what
  the branch accepted, not what Bedrock writes). Tests:
  `tests/bridge/test_responses_custom_transport_empty_hold.py`,
  `tests/bridge/test_gemini_custom_transport_empty_hold.py` (the KBR-287
  harness shape; content oracles parsed from route-protocol events, never
  raw substrings — the KBR-249 vacuous-oracle trap).
- **KBR-297 closed the last streaming leg — the `use_custom_transport` segment
  of `_stream_messages` / `/v1/messages`.** Ticket correction recorded here: the
  pre-fix wire did **not** deliver a silent empty turn. It delivered
  **fabricated fallback text** — `MessagesTranslator.translate_response`'s
  defensive fallback (`src/kitty/bridge/messages/translator.py:766-771`,
  "never emit thinking-only or empty assistant output") appended a `text`
  block carrying `_EMPTY_ASSISTANT_FALLBACK_TEXT` — *"Upstream model
  returned an empty response. Please retry. If the context is full, use
  /clear to reset the conversation."* — plus a `(provider, model, after N
  attempts)` suffix on a balancing pool, and `_log_usage` billed the
  fabricated turn. The bridge put words in the model's mouth, the
  empty-response ladder could not fire, and a balancing pool kept routing to
  the broken upstream. The fix ports KBR-287/293's judge-first shape: after
  `parse_stream_to_cc_response` (Bedrock / Ollama Cloud) or
  `_parse_sse_to_response` (subscription), the branch runs a single
  up-front verdict through `BridgeServer._is_empty_cc_response` (the
  KBR-285 lockstep whole-response twin of `_cc_chunk_carries_content`).
  Content-bearing → translate + emit, unchanged. Content-free → held; ladder
  per the KBR-287/293 shape (class-agnostic `_select_backend()` —
  custom → refresh `_resolved_key` / `_provider_config` + `continue`; plain
  → pop the three custom keys + `break` into the dispatch-loop fall-through;
  pool-less `elif` → `_BACKOFF_BASE` exponential backoff); exhaustion →
  **the route's own D4 terminal**, the bare JSON `_make_error_response` with
  `_NATIVE_EMPTY_REPLY_MESSAGE` + `reason: "empty_response"` and HTTP 502
  — this route defers `sr.prepare()`, so the D4 is a JSON response, not
  the SSE siblings' in-stream error event, per §5.3 S8. Attempt bound
  `n_backends + len(_EMPTY_FINAL_DELAYS)` with the final-delay prologue;
  exception-path gates stay at `n_backends - 1`; exception-path log
  denominators follow `n_backends` (mirroring the KBR-293 twin). Usage is
  log-on-release (`_log_usage` after emit); empty attempts never reach
  `_log_usage` and never reach `translate_response`, so the fallback text
  is never fabricated and no usage is billed for a judged-empty completion.
  Reasoning-only completions take the ladder — neither parser surfaces
  reasoning, the accepted trade-off KBR-287/293 pin. **One-step
  reasoning-widening asymmetry:** on this route the judge reads the parsed
  message directly and `translate_response` already maps
  `message.reasoning_content` → a `thinking` block, so a future
  parse-widening **alone** would regain reasoning fidelity here — unlike
  the siblings, whose chunk-synthesis also projects only `content` /
  `tool_calls` and would need its own widening. **Whitespace-drift
  consequence:** `_is_empty_cc_response`'s CC arm keeps its documented
  `.strip()` on string `content` (the pre-existing drift this paragraph
  also records above for KBR-277), while the sibling routes' judge uses
  `!= ""`. On this route the custom segment now takes the `.strip()` side:
  a whitespace-only completion ladders to the 502 D4 terminal here, while
  the same shape is delivered on `/v1/chat/completions` and the KBR-293
  siblings — and within this same route, the plain-POST streaming hold
  would deliver it too. Aligning the two predicates is not this ticket's
  scope. **Deliberate omission:** the empty arm's sleeps (pool-less backoff
  and final-delay prologue) do not call `_raise_if_client_gone()` although
  the helper exists in this handler — the KBR-287/293 empty arms omit it
  too, and a dead client plus a broken upstream burns the ~60 s ladder.
  Recorded so the omission can be revisited as a cross-cutting pass over
  all four custom segments. The non-streaming `/v1/messages` × custom-transport
  residual this paragraph used to record as out of scope is closed by KBR-298
  below. Tests: `tests/bridge/test_messages_custom_transport_empty_hold.py`
  (the KBR-287/293 harness shape; content oracles parsed from Messages-API
  events, never raw substrings; `_EMPTY_ASSISTANT_FALLBACK_TEXT` absence
  asserted explicitly as the fabricated-text defect).
- **KBR-298 closed the non-streaming `/v1/messages` × custom-transport cell —
  the residual KBR-297 recorded here as out of scope.** Ticket correction
  recorded, same shape as KBR-297's: the pre-fix wire did not deliver a
  silent turn either. `_request_with_retry`'s built-in empty ladder
  (single-backend: `len(_EMPTY_RETRY_DELAYS) + len(_EMPTY_FINAL_DELAYS) + 1`
  attempts; balancing: `n_backends`, empties never marking a backend
  unhealthy) already retried a judged-empty completion, and its exhaustion
  arm returned the empty response by design (grep anchor: "translator will
  add fallback text") — which the handler, with no emptiness gate, handed to
  `translate_response`'s defensive fallback. The client received a
  **fabricated** `_EMPTY_ASSISTANT_FALLBACK_TEXT` reply as a `200`,
  `_log_usage` billed it, and `_mark_backend_healthy` kept the broken
  upstream in rotation. The fix: in `_handle_messages`' translated arm — the
  `else:` branch of the `cc_response.get("type") == "message"` check, so a
  native Messages reply (and its D3 truncation 400) is structurally
  unreachable — the parsed `cc_response` is judged through
  `BridgeServer._is_empty_cc_response` (the KBR-285 lockstep whole-response
  judge, KBR-297's streaming twin) **before** `translate_response`. KBR-298
  gated on `self._active_provider.use_custom_transport`; KBR-300 widened the
  predicate to `not self._active_provider.use_native_messages and
  self._is_empty_cc_response(cc_response)` so the same gate covers the
  raw-CC cell too (see the KBR-300 paragraph below). Content-bearing →
  translate + respond, unchanged. Judged-empty → the route's D4 terminal as a
  non-streaming JSON error: `web.json_response` with
  `_NATIVE_EMPTY_REPLY_MESSAGE` + `reason: "empty_response"`, HTTP `502` —
  **no** `_log_usage`, **no** `_mark_backend_healthy` on a judged-empty
  completion. Three decisions the design review settled:
  - *Judge after the ladder, not per attempt.* The non-streaming route's
    retry structure already walks the empty ladder inside
    `_request_with_retry`; the ticket says to mirror that walk and invent no
    second ladder. The judge sits after the walk returns, where it can only
    see a content-bearing reply or the exhausted empty one — unlike the
    streaming twin's per-attempt pre-emit judge, because the non-streaming
    branch is atomic per attempt by construction
    (`_make_upstream_request` → `provider.make_request` returns one complete
    response or raises).
  - *Status 502, decided against the route's non-streaming error contract.*
    The route's non-streaming statuses are 400 (malformed input, D3
    truncation), the preserved upstream status (`UpstreamError`), 500
    (unknown exceptions) — none covers "the upstream answered, with nothing
    usable, on every attempt", the canonical 502 case. Route-internal
    consistency agrees: this route's streaming D4 (the plain
    `empty_no_finish` arm and KBR-297's custom segment alike) is bare-JSON
    502 with the same `reason: "empty_response"` discriminator, so one
    client branch, `(502, reason=empty_response)`, covers the route in both
    stream modes. Claude Code's Messages client retries any 5xx, so 502 vs
    500 changes no retry behaviour — the choice is semantic.
  - *Custom-transport gate only.* The ticket scopes the cell. The raw-CC ×
    non-streaming cell has the same post-ladder shape — the ticket
    attributed it to KBR-277, but KBR-277 widened only the
    `reasoning_content` predicate and left the exhaustion terminal
    untouched. The raw-CC cell was filed as **KBR-300** and is now closed
    there: KBR-300 widens the elif's predicate to
    `not self._active_provider.use_native_messages and self._is_empty_cc_response(cc_response)`,
    so the same gate covers both the KBR-298 cell and the raw-CC cell —
    one location, one comment, no parallel structures to maintain.
  The reasoning-only accepted trade-off carries over (none of the three
  custom parsers' non-streaming paths surface reasoning:
  `BedrockAdapter.translate_from_upstream` reads only `text`/`toolUse`
  blocks; `OllamaCloudAdapter` reads only `message.content` /
  `message.tool_calls`; the subscription's `_parse_sse_to_response` reads
  only text deltas and function-call items). The empty arm logs a
  route-scoped `logger.warning` mirroring KBR-297's;
  `_request_with_retry_single`'s own "returning fallback" line is left
  unchanged — its wording is now historical (the raw-CC sibling routes
  KBR-298 leaves open were closed by KBR-300 below; the comment above the
  KBR-298 code site records the widening). Tests:
  `tests/bridge/test_messages_custom_transport_non_streaming_empty_hold.py`
  (the KBR-287/293/297 harness shape; scripted raw upstream shapes fed
  through the adapters' real non-streaming parsers; content oracles parsed
  from the JSON response body, never raw substrings; the recording seam
  captures `_log_usage` / `_mark_backend_healthy` so the empty-arm
  guarantees are asserted, not assumed).
- **KBR-300 closed the four non-streaming silent-skeleton cells the
  KBR-298 closure left open.** KBR-248/276/287/293/297 closed the streaming
  emission paths; KBR-298 closed the non-streaming `/v1/messages` ×
  custom-transport cell; this ticket closes the three sibling cells
  (`_handle_responses`, `_handle_gemini`, `_handle_chat_completions`,
  both transport classes) and the raw-CC cell on `/v1/messages` itself
  (the KBR-298 elif gated on `use_custom_transport` and left
  `use_custom_transport = False` reaching `translate_response`'s
  fabricated fallback). Each handler's judge sits after the ladder
  (`_request_with_retry`'s existing walk, unchanged), in the
  translated/verbatim arm, before `translate_response` (and before
  `_log_usage` / `_mark_backend_healthy`). Single predicate per handler:
  `not self._active_provider.use_native_messages and self._is_empty_cc_response(cc_response)`
  — the messages elif widens in place (no parallel gate); the three
  siblings add one gate each. **Four per-route D4 body shapes, each
  mirroring its route's streaming D4 family** (route-internal consistency
  across stream modes, one client branch per route): `/v1/messages` keeps
  KBR-298's bare-JSON `502` + `_NATIVE_EMPTY_REPLY_MESSAGE` +
  `reason: "empty_response"` (`type: "error"`, `error.type: "api_error"`);
  `/v1/responses` adds `reason: "empty_response"` over the streaming
  `code: "empty_response"` discriminator (`error.code` / `error.reason`,
  no top-level `type` — **ticket correction:** the ticket spelled the body
  wrapped in `type: "error"`, but that wrapper is the Anthropic Messages
  error convention (KBR-298's shape), not the Responses route's: all eight
  other non-streaming error envelopes on `/v1/responses` return
  `{"error": {...}}` only, and OpenAI's documented Responses error shape
  puts the error class inside `error`; the wrapper was dropped in PR
  review round 1 so the route's D4 follows its own protocol's convention
  like the Gemini and Chat Completions cells do); the discriminator
  `code: "empty_response"` appears in both stream modes on this route,
  with `reason` a non-streaming-only marker — its JSON depth differs by
  wire (top level in the streaming SSE event data, nested under `error`
  in the non-streaming body), which is each wire's native envelope
  convention, not a divergence.
  `/v1/gemini` byte-mirrors the streaming SSE error
  payload (`code: 502` integer, no top-level `type`); `/v1/chat/completions`
  byte-mirrors KBR-287's streaming D4 (`error.type: "empty_response"`,
  verbatim CC carries `type`, not `reason`). All four `502`; all four
  return *before* `_log_usage` and `_mark_backend_healthy` (with the
  CC-arm asymmetry on `_mark_backend_healthy` recorded in the gate
  comment — the CC non-streaming body returns before the healthy-mark
  site; the asymmetry is pre-existing and unrelated to the widening).
  Reasoning-only trade-off carries over unchanged: custom-transport
  non-streaming parsers drop reasoning
  (`BedrockAdapter.translate_from_upstream` reads only `text`/`toolUse`;
  `OllamaCloudAdapter` reads only `message.content`/`message.tool_calls`;
  `OpenAISubscriptionAdapter._parse_sse_to_response` reads only text
  deltas and function-call items) → ladder-taking; raw-CC cells carry
  `reasoning_content` non-empty in the wire shape → the KBR-277
  predicate counts it as content → released on the first attempt. Tests:
  `tests/bridge/test_messages_raw_cc_non_streaming_empty_hold.py` (the
  raw-CC cell on `/v1/messages`); `tests/bridge/test_responses_non_streaming_empty_hold.py`,
  `tests/bridge/test_gemini_non_streaming_empty_hold.py`,
  `tests/bridge/test_chat_completions_non_streaming_empty_hold.py` (each
  route × both transport classes, the KBR-287/293/297 harness shape; plain
  cells use `aioresponses` on the OpenAI default endpoint, custom cells
  monkeypatch `make_request` and feed canned raw shapes through the
  adapters' real non-streaming parsers). The pre-existing
  `test_non_streaming_exhausts_retries_emits_fallback` /
  `test_all_backends_empty_emits_fallback` /
  `test_non_streaming_final_retries_fallback` tests in
  `tests/bridge/test_empty_response_retry.py` pinned the raw-CC fallback
  behaviour KBR-300 retires; they were rewritten in-PR to pin the D4
  expectation, mirroring the streaming-side KBR-99 (S11) precedent.
- **KBR-304 closed the native-provider residual KBR-300 recorded.** KBR-300's
  gate predicate — `not use_native_messages and _is_empty_cc_response` —
  carved out native providers as the deliberate non-fix ("proposed follow-up,
  not yet filed"); that follow-up is now filed and implemented. Each of the
  four gates' predicate is now `self._is_empty_cc_response(cc_response)` —
  the `use_native_messages` conjunct is gone, so a native provider's empty
  `cc_response` on the three sibling routes and on the KBR-237
  tool-use-format fallback arm of `/v1/messages` (the cell the KBR-300
  paragraph named) ends in the same per-route D4 `502` as non-native
  providers, with no usage billed and no healthy-mark. One conjunct dropped
  per handler, no new ladder logic, no per-route D4 body change. The
  native-Messages arm of `/v1/messages` — the `if cc_response.get("type") ==
  "message"` branch, where the reply arrives in native Messages shape —
  stayed structurally out of the gate (it never reaches the elif) and was
  the recorded residual for both KBR-300 and KBR-304 — closed by KBR-306
  below. Tests: each
  of the four KBR-300 test files gained a native-provider variant
  (`test_an_empty_native_completion_ends_in_the_d4_terminal` +
  `test_an_empty_native_attempt_crosses_to_a_healthy_plain_peer` +
  content/tool pins) driven by a test-only `_NativeOpenAIAdapter` stub
  whose only override is `use_native_messages -> True`; the stub
  deliberately leaves `upstream_wire_shape` at the inherited
  `CHAT_COMPLETIONS` (the base-class invariant is a production-adapter
  rule; the test exercises the conjunct dimension only).
- **KBR-306 closed the last cell: the native-Messages arm of `/v1/messages`
  non-streaming.** After KBR-304 every CC-shaped non-streaming empty reply
  on every route was judged, but the `if cc_response.get("type") ==
  "message"` branch of `_handle_messages` — the arm a native Anthropic
  provider reaches when it answers in its own Messages shape — shipped the
  ladder-exhausted empty reply as a billed, healthy-marked `200`:
  `result = cc_response` fell through to the audit/bill/mark block, and
  `_messages_truncation_before_content` (D3) could not catch a genuinely
  empty `end_turn`. The fix mirrors KBR-298/300/304 exactly: inside the
  `if` branch, after the D3 truncation check and before
  `result = cc_response`, the parsed native reply is judged through
  `self._is_empty_cc_response` — whose Messages-shaped arm (pinned by the
  streaming twin `PreambleHold._block_start_releases`, Q14 D1) decides —
  and a judged-empty completion ends in the route's D4 terminal
  (`server.py:5510-5520`'s byte-image: bare-JSON `502` +
  `_NATIVE_EMPTY_REPLY_MESSAGE` + `reason: "empty_response"`), before
  `_audit_response_tool_use` / `_log_usage` / `_mark_backend_healthy`. One
  client branch, `(502, reason=empty_response)`, now covers `/v1/messages`
  in both stream modes — the D4 body is byte-identical across the
  non-streaming elif, the native `if` arm, and the streaming S11 terminal.
  The ladder is unchanged: `_is_non_retryable_reply` already judges
  Messages-shaped empties through the same arm, so
  `_request_with_retry_{single,balancing}` walked these attempts before
  this ticket; the gate sits where it can only see a content-bearing reply
  or the exhausted empty one. D3 keeps its ordering (a
  `max_tokens` / `model_context_window_exceeded` stop reason on a
  content-less reply is non-retryable in the ladder and renders the
  request-shaped `400` before the new gate). Two carry-overs are
  deliberate, recorded here so they are not read as accidents:
  - *Whitespace-only text is not judged empty* (PO confirmation
    2026-09-23): the Messages arm's `text != ""` (no `.strip()`) mirrors
    the streaming hold's release semantics; a native reply whose only
    content is a whitespace-only text block reaches the client as `200`
    in both stream modes. Widening it would require changing BOTH the arm
    and the streaming hold together to preserve the documented mirror —
    a behaviour change on a streaming path this ticket explicitly
    records as fine, so it stays open.
  - *A thinking-only native reply takes the ladder* (the KBR-287/293/
    297/298/300 reasoning-only trade-off, carried through the
    KBR-277/285 judge family): the arm counts a thinking block as empty,
    consistent with the streaming hold, which also does not release on a
    thinking block. A tool_use block counts as content (D1: any block
    whose type is not text/thinking/redacted_thinking is content).
  Tests:
  `tests/bridge/test_messages_native_non_streaming_empty_hold.py` (the
  KBR-300/304 harness shape in a sibling file: a real in-process
  `BridgeServer` against an `aioresponses` upstream; the
  `_NativeOpenAIAdapter` stub physically mirrored from the raw-CC file;
  Messages-shaped canned bodies that flow through the
  `_native_messages_request + type == "message"` guard as-is; the
  recording seam capturing `_log_usage` / `_mark_backend_healthy`; three
  RED tests on the unfixed code — empty, thinking-only, and the
  balancing-mode `[native, native]` exhaustion whose `healthy_log == []`
  is the meaningful "no healthy-mark" pin, the single-backend assertion
  being vacuous by the recorded asymmetry; four pass-through pins —
  content-bearing, tool-use-only, D3 truncation, and the
  `[native, plain]` crossing — pass under both predicates). The existing
  R4 pin in `tests/bridge/test_native_provider_reply_shape.py`
  (`test_claude_code_still_gets_the_upstream_messages_reply`, three real
  native adapters) continues to hold and is the stronger content-bearing
  pass-through pin.
- **KBR-277 closed the non-streaming half.**
  `BridgeServer._is_empty_cc_response`'s Chat Completions-shaped arm now reads
  `message.reasoning_content` with the same `isinstance(..., str) and ... != ""` rule
  `_cc_chunk_carries_content` applies to `delta.reasoning_content`.
  The two predicates agree on the `reasoning_content` axis — the property test in
  `tests/bridge/test_empty_response_reasoning_properties.py` pins this. The
  Messages-shaped arm is unchanged and continues to mirror
  `PreambleHold._block_start_releases` (Q14 D1): a Messages thinking block does not
  count as content, consistent with the streaming hold, which also does not release on
  a thinking block. **Pre-existing drift on `content`, now documented here:** the
  CC arm's `content` check uses `.strip()` (whitespace-only content is empty) while
  the streaming predicate's `content` check uses `!= ""` (whitespace-only content is
  content). `TEST_SUITE.md` D6 covers only the empty-string case; aligning the two
  would change product behaviour outside KBR-277's scope (a whitespace-only `content`
  would stop being treated as empty) and is left in place.
- In-stream error failover on `/v1/chat/completions` needs a backend pool; pool-less the
  error surfaces to the client (which is still the fix: the per-event translator used to
  swallow the error and deliver a truncated success).
- **KBR-254: cross-class dispatch fix on the three siblings.** The KBR-249 fix
  on `_stream_messages` left the same defect in `_stream_responses`,
  `_stream_gemini`, and `_stream_chat_completions` (recorded in §5.4 of
  `SYSTEM_DESIGN.md` since KBR-249 as "the fix shape is identical, a sibling
  ticket per handler is owed"). KBR-254 applies the fix: each sibling now
  sits inside a `while True:` dispatch loop with `_crossings` /
  `_max_crossings = (2 * n_backends) + 1` and a transport-class crossing
  guard at every plain-POST `_select_backend()` failover site — three per
  handler, mirroring the KBR-249 shape on `_stream_messages` (see §5.3 S8
  and §5.4 for the per-route cap-hit wire shapes):

  - `_stream_responses` — the three plain-POST failover sites that
    KBR-249 recorded at pre-KBR-254 line numbers
    `src/kitty/bridge/server.py:~3605, 3770, 3804`. Each now sits behind
    a guard inside the dispatch loop. The cross-mode fall-through at
    `:3395` is the symmetric custom→plain path, not a crossing site.
  - `_stream_gemini` — three plain-POST failover sites pre-recorded at
    `:~5818, 5959, 5980`.
  - `_stream_chat_completions` — three plain-POST failover sites
    pre-recorded at `:~6825, 7024, 7054`.

  To re-locate the now-guarded sites against current source, run
  `git grep -n "KBR-254: cross-class failover"`; matches inside the
  custom-transport branch's cross-mode fall-through are not crossings.
  Per-route cap-hit terminal SSE event shape (recorded in the PR
  description for KBR-254):

  - `/v1/responses`: `error` event with `code: "cross_class_exhaustion"`
    plus `reason: "cross_class_exhaustion"`, then the existing
    `responses_format_error(...)` + `synthesize_completed_events` loop.
  - `/v1beta/...:streamGenerateContent`: `data: {"error":{"code":502,
    "message":"...","reason":"cross_class_exhaustion"}}\n\n` then the
    existing `write_eof`. The asymmetry from the Messages route — which
    returns a bare JSON `502` on cap-hit because `sr.prepare()` is
    deferred there — is intentional: the three siblings prepare their SSE
    response eagerly, so the cap-hit surfaces as an in-stream terminal
    event followed by the handler's existing post-loop, and the status
    line on the wire is `200`.
  - `/v1/chat/completions`: `data: {"error":{"message":"...","type":
    "cross_class_exhaustion"}}\n\n` followed by `data: [DONE]\n\n`, then
    `write_eof`.
- **The one remaining post-emission ending asymmetry is deliberate (KBR-99
  records it as pinned residue) — and it is scoped to drops that land before
  any finish chunk is read.** On the translated `/v1/messages` branch a
  post-emission **transport drop** at the AFTER_TEXT or MID_TOOL_ARGUMENTS
  cells closes `end_turn` + `message_stop` rather than the Q14 error event —
  KBR-183's decision D2, kept so the branch's truncated answer reads as a
  completed turn to Claude Code (verified 10/10 per cell, 2026-09-21). On
  the BEFORE_TERMINAL cell the residue does **not** hold: the finish chunk's
  streaming path auto-resets the translator the moment it is read
  (`messages/translator.py:1037`), so when the drop surfaces afterwards
  `finalize_interrupted_stream()` finds no live message and returns `[]`,
  and the transport branch falls to its error-event fallback with the block
  left open (truncated); on the clean-EOF race the buffered close-out
  flushes first (complete sentence). That cell is T-G7's genuinely two-way
  shape set `{complete_sentence, truncated}`, verified 15/15 truncated under
  the exception path — the grammar floor already composes on it, and KBR-99's
  grid pins the shape set rather than one internal path. Every other
  post-emission failure on every route ends in the route's error event
  (native Messages via KBR-183; Responses via KBR-247; Gemini via S10; Chat
  Completions via its `upstream_error` marker). `TEST_SUITE.md` §6.3.1 owns
  the assertion; a change here is an owner decision that must move the pin,
  the §6.3.1 residue paragraph, and §11's D2 amendment together.
## 6. Backend health, cooldowns, and the arrival recovery hold

### 6.1 The state machine as it stands

Each backend of a balancing pool carries a health record
(`BridgeServer._backend_health`). A failure marks it unhealthy for a cooldown whose
length depends on the kind: `rate_limit` from the provider's own retry hint,
`hard`/`cloudflare` from `backend_cooldown` (default 300 s; capped at 30 s for a
one-backend pool), `transport`/`stream` on an escalating ladder, `auth` for 900 s,
`entitlement` for 24 h. On request arrival `_select_backend()` picks among healthy
backends (backup tier held out while a primary lives); when **every** backend is
cooling it either gambles on a near-expiry backend (soonest recovery ≤ 60 s,
`_ALL_UNHEALTHY_FAST_FAIL_THRESHOLD`) or raises `AllBackendsUnhealthyError`, which the
four protocol handlers answer with an immediate per-protocol 503 carrying
`Retry-After` and the per-backend causes.

### 6.2 KBR-243: hold the arrival while recovery fits inside the window

An immediate 503 makes Claude Code abandon the turn even when the outage is seconds
from ending; the operator's session then stalls until a human re-sends. **To Be:**
when selection fails on arrival, `_select_backend_or_hold()` holds the request while
the soonest expiry fits strictly inside `_RECOVERY_HOLD_WINDOW` (300 s) of the arrival
— sleep to the expiry, re-select, proceed — and answers with the today-shaped 503
built from the **latest** failure only once no recovery fits. `/stats` gains
`recovery_holds` (hold starts; `all_backends_unhealthy` keeps counting raise events,
including raises a hold then recovers). Decisions, and why:

- **Arrival only.** The reported failure mode is the arrival 503 (the ticket's log:
  the agent quits at second zero). Mid-ladder exhaustion is a different path whose
  post-emission side is governed by the Q14/KBR-163 rule that the bridge never retries
  or holds once bytes reached the client; blanket mid-ladder holds would have to be
  proven per transport and are deliberately out of scope.
- **Uniform across cooldown kinds.** The hold does not filter `rate_limit` from
  `hard`/`transport`/`cloudflare` — the same uniformity as the existing 60 s
  near-expiry gamble, which also does not care why a backend cools. `auth` (900 s)
  and `entitlement` (24 h) exceed the window and 503 immediately; in the narrow band
  where their *remainder* fits, one ≤300 s hold precedes the 503 — acceptable for a
  session that is already dead.
- **No polling: sleep exactly to the expiry.** Cooldown expiry is deterministic
  monotonic arithmetic, so waking at `retry_after` wastes nothing. The integer
  truncation in `remaining` can wake the loop ≤1 s early; the re-selection then exits
  through the existing near-expiry gamble, which is what that branch is for.
- **Strict window.** A recovery exactly 300 s away 503s immediately: a 300 s silent
  hold sits on common client idle-timeout edges and buys nothing over an immediate
  503 carrying `Retry-After: 300`. The window is hard-coded from the ticket
  ("within the next 300 seconds"); a configuration knob waits for an operator asking
  for one.
- **`recoverable=False` on the no-stream-capable raise.** That raise fabricates
  `retry_after=300` (§6.1's streaming filter — unreachable at arrival today, kept as
  hardening): its number is not a recovery time, so the hold must never sleep on it.
- **Jitter, clamped.** Each hold sleeps `min(retry_after + jitter, window − elapsed)`
  with a 0–2 s jitter so a herd of held sessions does not converge on the one
  just-recovered backend in a single instant; the clamp keeps every hold inside the
  window the formula promised.
- **The client is protected while it waits.** Before each hold the request body
  is drained best-effort (aiohttp caches it, so the handler's later parse is
  unchanged and later iterations are no-ops) so the client is not left stalled
  mid-upload; the transport is polled
  before the first sleep and after each wake — aiohttp runs with
  `handler_cancellation` off, so this is the only way to notice a hang-up
  (the `_raise_if_client_gone` pattern, KBR-241's `PreambleHold` being the sibling
  hold) — and a gone client ends the hold with no upstream request fired for it.
- **Client-side deadlines.** The binding deadline for a held request is Claude Code's
  streaming first-byte watchdog (~5 min unset default); `API_TIMEOUT_MS` (10 min
  default) is the outer cap. A client that aborts re-sends, and the re-sent request
  re-enters the hold benignly — a new arrival under the same formula. SSE keep-alive
  pings would let holds run longer still, but are deferred as an accepted cost
  (keep-alive semantics differ across the four served protocols, and inventing bytes
  on three of them is an I2 exposure); revisit if field reports show clients idling
  out mid-hold.
- **Empty-pool profiles are untouched.** A profile without a backends list never
  raises on arrival and never holds; a one-element balancing list holds like any
  pool. Graceful shutdown may wait behind held sessions up to aiohttp's
  `shutdown_timeout` (60 s default) — accepted, bounded.

Tests: `tests/bridge/test_all_backends_unhealthy.py` — `TestRecoveryHold` (L1, the
stepped-on-sleep fake clock) and `TestArrivalHoldWiring` (HTTP against the real app;
a real 1 s hold, jitter patched).

## 7. The interactive-terminal guard

### 7.1 The contract as it stands

A process that calls `kitty.tui.prompts.can_interact()` (the prompt-and-menu guard shared
by `check_tty()` and `kitty.tui.menu`) must answer truthfully whether **both** standard
streams belong to a terminal the child can read keys from and draw on. **As Is:**
`can_interact()` answers `sys.stdin.isatty() and sys.stdout.isatty()`. On POSIX that is
exact — `/dev/null` is not a character device — so no Linux or macOS run has ever seen
the defect. On Windows `isatty()` checks for *any* character device, and `NUL` is one
(CPython [bpo-28654](https://bugs.python.org/issue28654); confirmed 2026-09-14), so
`kitty auth openai > NUL`, or any parent process that spawns an interactive command with
`stdin=DEVNULL, stdout=DEVNULL`, passes the guard and dies inside `prompt_toolkit` with
`NoConsoleScreenBufferError` the moment it tries to build a Win32 screen buffer over a
non-console stdout. KBR-187 fixed the menus' copy of the guard (raised on a piped
stdout), KBR-204 found `check_tty` still held the stdin-only reading, and KBR-218 is the
NUL-shaped hole KBR-204's `isatty-AND` reading cannot close.

### 7.2 To Be: ask the console, not the file system

On `win32`, `can_interact()` asks the console API whether each standard handle is one.
`kernel32.GetStdHandle(-10)` (`STD_INPUT_HANDLE`) and `(-11)` (`STD_OUTPUT_HANDLE`) return
the process's standard handles; `kernel32.GetConsoleMode(handle, byref(DWORD))` succeeds
only when the handle names a real console — for a pipe, a file, or `NUL` it fails (the
documented return value is zero with `GetLastError() == ERROR_INVALID_HANDLE`; the home
code does not branch on the error code because fail-closed is the safe direction for a
guard). On every other platform the answer is the unchanged
`sys.stdin.isatty() and sys.stdout.isatty()`. The answer stays in exactly one place:
`check_tty()` and `kitty.tui.menu` keep importing `can_interact` and cannot drift.

**STD_ERROR_HANDLE (-12) is deliberately not consulted.** The prompt path renders to
stdout and `print_error` (`tui/display.py`) reports to stderr, so consulting the error
handle would add a third ctypes call without adding signal. KBR-187, KBR-204 and KBR-218
all live on stdin/stdout; keeping that symmetry means the next reader does not have to
reason about an edge the product has no use for.

### 7.3 Decisions, and why

- **One shared answer.** The KBR-187/KBR-204 history is two fixes to two copies of the same
  predicate; the third fix goes into the one that the other two now consult
  (`can_interact`), so the regression surface is `O(1)`, not `O(callers)`. The KBR-204
  falsification pins this: it asserts `menu.can_interact is real` after a `prompts` patch,
  and is the regression detector if anyone re-introduces a local copy.
- **`GetConsoleMode`, not `isatty`, on Windows.** `isatty()` is a CRT probe that classifies
  by file type; `GetConsoleMode` is a Windows-console probe that classifies by API.
  Classifying by API is the only answer that distinguishes a console from the next
  character device the OS opens, because the *next* device the kernel might grow
  character-device semantics for is not a console either. Reading the API also matches what
  prompt_toolkit does at the crash site (`Win32Output.get_win32_screen_buffer_info` calls
  the same family of `kernel32` functions), so the guard and the consumer agree on what a
  console is.
- **Fail-closed.** Any `GetConsoleMode` failure — not a console, no handle, the process
  started `DETACHED_PROCESS` and `GetStdHandle` returns `NULL`/`INVALID_HANDLE_VALUE` —
  returns False. A guard that declines interactivity for a daemon started without a console
  is correct: a daemon never calls `check_tty`, and a guard that returned True for a
  console-less start would be lying to prompt_toolkit, which would then crash on the next
  step anyway. **The "daemon never calls this" guarantee is contractual for
  `kitty.bridge`** (the `Bridge must not import leaf or CLI modules` import-linter
  contract in `pyproject.toml` forbids `kitty.tui`) **and conventional for
  `kitty.bridge_runner`** — `bridge_runner` is a sibling top-level module, not a
  descendant of `kitty.bridge`, and the import-linter contracts do not reach it (the
  `kitty.io_encoding` contract's own comment names exactly this gap as the reason its
  list enumerates every sibling). KBR-231's background bridge is the `bridge_runner`
  case; its shutdown path does not consult `check_tty` today, and adding a prompt there
  is held only by code review, not by the gate.
- **POSIX untouched.** `/dev/null` is not a character device; the `isatty`-AND reading is
  already exact there and rewriting it would be a no-op with a real risk of breaking the
  four Linux CI legs that depend on it.
- **`WinDLL("kernel32")` instantiation lives inside the win32 branch body, not at module
  scope.** `ctypes` is portable and its `import` is fine at any scope, but `ctypes.WinDLL`
  only exists on Windows — hoisting the `WinDLL(...)` call to module scope would
  `AttributeError` at import time and break every `kitty.tui.prompts` consumer
  (`kitty.tui.menu`, `kitty.cli.{setup,auth,egress,profile}_cmd`) on POSIX. The `bridge/
  manage.py::_probe_pid_windows` pattern (`WinDLL` instantiation inside the function,
  after the `sys.platform != "win32": raise` narrowing) is the local precedent and is
  mirrored here.
- **Adjacent `isatty` calls left alone.** Three other call sites in `src/kitty` still read
  `isatty()` for decisions other than the interactive-prompt guard, and have the same
  Windows `NUL` lies today: `cli/tmux_wrap.py:402` decides whether to wrap the agent in a
  tmux client, `cli/launcher.py:290` decides whether to inherit the parent's stdin into
  the spawned agent (`stdin_arg = sys.stdin if sys.stdin.isatty() else None`), and
  `tui/display.py:169` decides whether to render a progress bar. None is the
  interactive-prompt guard, and KBR-218 is scoped to `can_interact()`; touching them in
  the same change would mix the ticket's hole with three distinct decisions, and an
  `isatty` reading that lies about `NUL` is the documented Windows behaviour these call
  sites have always lived with. Recorded as a known limit so the decision is reviewable.
- **Harmonising side effect.** On Windows, `kitty auth openai < NUL` at a real console
  used to pass the guard and EOF-loop inside the prompt; after the fix it refuses with
  exit 2, matching the POSIX behaviour that has always been correct. This is the intended
  consequence of the same fail-closed rule that handles `NUL`/`NUL`, and is preempted
  here so a user filing "now my one-NUL redirect doesn't work" lands on this paragraph.

### 7.4 Verification

Tests extend the existing KBR-204 harness in `tests/cli/test_stream_encoding.py` (l1 by
path default; the file's docstring records the §8.2 reason for the marker choice), a unit
case in `tests/tui/test_prompts.py`, and the seam extension documented in §7.5. Five
cases:

1. A premise case that the child really sees `isatty == (True, True)` on Windows and
   `(False, False)` on POSIX with `stdin=DEVNULL, stdout=DEVNULL`. The child prints its
   report **to stderr** so the parent can read it back even with stdout discarded.
2. The product test: `kitty egress` with both streams discarded → exit 2, TTY diagnosis on
   **stderr specifically**, no traceback. Green on POSIX at base (the guard already
   refuses `/dev/null`), **red on Windows until the fix**, green on Windows after. The
   red's shape pins the mechanism: a `_CRASH_MARKERS`-bearing traceback from
   prompt_toolkit. A TimeoutExpired there would mean the premise is wrong (a hang, not
   the crash the ticket documents) — investigate the harness before trusting a fix.
3. A harness falsification: the child patches `prompts.can_interact` to `lambda: True`
   (the lie Windows tells natively for `NUL`) while `menu.can_interact` stays bound to
   the real predicate, then runs `kitty egress`. The harness asserts the pass-through
   signature (exit 0, no diagnosis, no crash) — so the product test's exit-2 assertion is
   proven to be what stands between the user and silent nothing. **Red on the Windows leg
   at the test commit too** — at base the menu's own (real) predicate is also blind for
   `NUL`, so the child crashes inside the menu rather than declining silently — and green
   on every leg after the fix. A watcher seeing both new tests red at the test commit is
   seeing the expected evidence, not a harness defect.
4. The probe's decision is a one-line pure function `_handle_attached(raw_mode: int) ->
   bool` (the `bool(raw_mode)` interpretation, per the
   `tests/bridge/test_bridge_management.py:842-854` house pattern that keeps ctypes as
   plumbing and decisions as pure functions), and `tests/tui/test_prompts.py` pins it on
   every leg: `0` → False, any nonzero → True. Separately, the same file parametrises
   the four `(stdin_value, stdout_value)` combinations in `(True, False)`, patches
   `_query_console_mode` (the plumbing) with
   `side_effect=iter([1 if stdin_value else 0, 1 if stdout_value else 0])`, and leaves
   `_handle_attached` real so it interprets each raw value — `can_interact()` returns the
   AND, `(True, True)` → True; the three other cells → False. This is the **positive
   direction** of the fix: every subprocess case above exercises the refusal branch,
   so without this truth table a regression that makes `can_interact()` always False
   on `win32` (silently refusing every real console) ships green against the whole
   suite. The ctypes plumbing (`_query_console_mode` — `WinDLL("kernel32",
   use_last_error=True)`, `GetStdHandle`, `GetConsoleMode`) is Windows-only inside its
   own body and is exercised by the Windows pytest leg; the `restype` MUST be declared
   pattern from `bridge/manage.py:147-149` keeps the HANDLE truncation trap out of
   the helper.
5. The existing "simulated interactivity" test seams (`_mock_tty` helpers and the
   `True,True`/`False`-patch inlines across the seven test files) extend to patch
   `kitty.tui.prompts._handle_attached` alongside `isatty`, in **both directions**:
   positive sites patch it True (so the Windows CI leg reads "interactive" when the
   test says so), negative sites patch it False. The negative direction matters on a
   developer's Windows machine, not in CI: a `patch isatty=False` refusal test runs
   against the *real* console probe there, and with a real console attached the probe
   says True, so the test fails — deterministic on CI, red on a dev box, exactly the
   environment-dependence the repo does not accept. (Found by the PR review
   classifier; the initial wording claimed False-only patches were safe because the
   CI probe also said False.) Without the extension, `can_interact()` no longer reads
   `isatty` on Windows and every existing "the prompt proceeds" / "the command
   refuses" unit test is decided by the environment rather than the patch.

The `_run_child` runner gains a `stdout` pass-through (defaulting to capture, as today)
and `_asked_for_a_terminal` tolerates a discarded stdout by reading `completed.stdout or
b""`; this is one line and leaves every existing KBR-204 case byte-identical.

### 7.5 Known limits

- The three adjacent `isatty` call sites (`cli/tmux_wrap.py:402`,
  `cli/launcher.py:290`, `tui/display.py:169`) still read `isatty()` directly for their
  own decisions and carry the same `NUL` lies on Windows. §7.3 records why they are out
  of KBR-218's scope; if a user-visible defect surfaces at one of them it gets its own
  ticket, scoped to that decision alone.
- The "simulated interactivity" test seams (§7.4 case 5) stay patched to the
  `_handle_attached` shape, in both directions (True for proceed-expectation, False for
  refusal-expectation): a future change to `can_interact()` that consults a *different*
  oracle must extend the same seams again, or the Windows CI leg will regress. The list
  (as of writing — the grep rule in §7.4 case 5 is the load-bearing invariant, this is
  the snapshot) is `tests/tui/test_prompts.py::_mock_tty`,
  `tests/tui/test_menu.py::_mock_tty`, `tests/tui/test_setup_wizard.py::_mock_tty`,
  the inline `True,True` block in
  `tests/tui/test_profile_menu.py::test_table_includes_backup_column`, the nine
  `True,True` blocks across `tests/tui/test_egress_menu.py` (five in
  `TestConfigureFlow`, two in `TestRemoveFlow`, two in `TestMenuShape`) plus its
  `test_non_tty_is_rejected` False site, `tests/cli/test_auth_cmd.py::_mock_tty` plus
  its two `run_oauth_for_provider` / `run_auth_openai` False sites, and the four
  `stdin.isatty=False` sites plus the `True,True` block in
  `tests/test_cli_main.py::TestNonTTYExit`. `tests/tui/test_live_checklist.py` is
  excluded — its `stdout.isatty=True` patches drive `display.py:169`'s rendering
  decision (an adjacent site left alone per §7.3), not the guard, and the seam
  extension does not apply to it.

---

## 8. Bedrock adapter — Converse reasoningContent emission

Traces to [KBR-264](https://shelpuk.atlassian.net/browse/KBR-264). Read alongside
`TEST_SUITE.md` §3.3 (transparency oracle) and §7.2 (bedrock recorder), which own the bridge's
view of the Converse wire; this section owns the adapter's view of one content-block spelling.

### 8.1 The rule

`kitty.providers.bedrock.BedrockAdapter._translate_assistant_msg` emits the assistant
`reasoningContent` block in the **schema-correct** spelling:

```
{"reasoningContent": {"reasoningText": {"text": <reasoning>}}}
```

The two branches — present `reasoning_content` (populated inner `text`) and the
`_thinking_enabled=True` empty-injection branch (inner `text` carries `""`) — use the same
nested spelling. There is no top-level `text` inside `reasoningContent`: the published botocore
`bedrock-runtime` `ReasoningContentBlock` union is exactly `{reasoningText, redactedContent}`,
and `reasoningText` carries `text`. The installed service model (botocore `1.43.93` in the dev venv; the pinned version
is recorded in the upstream lockfile, which `uv.lock` does not commit in this worktree) is
the authority; the reader (T-A5, KBR-37) and the adapter agree on that authority. (The
reader's `TestSchemaAgreement::test_reasoning_content_block_members_match_the_live_service_model`
asserting the union lives on PR #177's branch, OPEN at the time of writing; this section's
`validate_parameters` oracle in §8.2 stands on its own against the installed botocore.)

### 8.2 The contract oracle

L1 asserts the adapter's emission against the live service model, not against a hand-typed
schema. The call:

```python
from botocore.session import Session
from botocore.validate import validate_parameters

input_shape = (
    Session()
    .get_service_model("bedrock-runtime")
    .operation_model("Converse")
    .input_shape
)
validate_parameters(emitted_converse_request, input_shape)
```

A `botocore.exceptions.ParamValidationError` on the old spelling — `Unknown parameter in
messages[0].content[0].reasoningContent: "text", must be one of: reasoningText, redactedContent`
— is the rejection that motivates this fix, and it is the regression sentinel: the same call
on the new spelling returns no error. The oracle lives in `tests/test_provider_bedrock.py`
(the adapter's own L1 file), not in `tests/harness/`, because the claim under test is the
*adapter's emission*, not the reader's view of one.

### 8.3 Decisions, and why

- **Two-branch symmetry.** Both the present-reasoning and the
  `_thinking_enabled`-injected-empty branches use the same nested spelling — the fix
  *preserves* the symmetry both branches already had (they previously shared the wrong
  spelling) while correcting both. Letting the branches diverge would re-create the
  rejection the moment one of them is exercised against the live model.
- **No content-block reordering, no new blocks.** The fix changes the spelling of one
  member; the surrounding list (`text`, `toolUse`, …) is untouched. Per
  `TEST_SUITE.md` §3.2.3, the bedrock recorder observes the body *after* the transport's
  `modelId` / `stream` pops (P18); the adapter's emission is what reaches the recorder on
  the hook path.
- **Contract oracle at L1, not L2.** The reader's similar test
  (`TestSchemaAgreement::test_reasoning_content_block_members_match_the_live_service_model`)
  is in `tests/harness/test_reader_bedrock_converse.py` (the L2 reader file). The adapter's
  counterpart lives at L1 because the claim is the *adapter's* shape, not the schema as a
  shared vocabulary. Keeping the two close to the code they assert on is the
  `tests/layers.py` default-layer-by-path rule; duplicating the oracle into a harness file
  would break that rule without buying locality.
- **No harness register row.** No row in the Permitted-Mutation Register (TEST_SUITE.md §3.2)
  pins the old `{"reasoningContent": {"text": …}}` spelling; the only mention of
  `reasoningContent` in TEST_SUITE.md is the schema content-block member name. The fix does
  not change a register row and does not need a new one — a register row would be the right
  place for a *product decision* about whether to carry reasoning, not for a schema-typo
  fix.

## 9. DEBUG-log redaction policy

Traces to [KBR-156](https://shelpuk.atlassian.net/browse/KBR-156) (closed-as-folded into
[KBR-73](https://shelpuk.atlassian.net/browse/KBR-73)); recorded before the code changes land,
per the project's design-first discipline.

### 9.1 The rule

**DEBUG-level logs are held to the same redaction standard as user-facing messages.** Every log
line the bridge emits through the `kitty.bridge` logger — the logger `--debug`'s FileHandler
writes to `~/.cache/kitty/bridge.log` — must not contain a query value, a userinfo component,
or a credential, verbatim. What this buys: a user attaching `bridge.log` to a bug report, the
project's own troubleshooting guidance, cannot leak a gateway key by doing so. What it costs:
a masked query value tells the developer "this parameter was sent" without telling them what it
said — acceptable, because the URL's path and parameter names (the routing facts a debug log
exists to answer) survive, and the credential is the one thing a user cannot afford to publish.

This deliberately overrides the diagnostic-quality argument recorded on KBR-156: the log exists
to answer "where did the request actually go", and the answer stays intact — scheme, host,
port, path, parameter names. Only the values are masked.

### 9.2 The two redactors, and the header rule

Two redactors already exist, each owning one URL class:

- `EgressConfig.masked()` — the **proxy** URL (userinfo present by design, password masked).
- `ProviderAdapter.redact_url_for_display` — the **upstream** URL (userinfo dropped, every
  query value masked, fragment masked whole).

KBR-143's stated principle, in `providers/base.py`, is that *values are masked indiscriminately
rather than by name, because telling a credential from a routing parameter means guessing, and
a guess that is wrong once leaks a key.* Headers cannot take the URL rule (mask every value)
without destroying the diagnostic — non-credential headers are most of what a developer reads
in a header dump — so the header rule becomes: **mask a header's value when its name, lowercased,
contains any of `auth`, `key`, `token`, `cookie`, `secret`, `signature`.** This is the same
principle, applied to a surface where the credential signal lives in the *name* rather than the
position: it catches `x-goog-api-key`, `x-auth-token`, `anthropic-api-key` and any future
credential-bearing header without enumerating them, and it mirrors the
`_CF_COOKIE_PREFIX` pattern already in `providers/openai_subscription.py`. The divergence from
KBR-143's wording is deliberate and recorded here so a future reader does not "fix" one to
match the other.

### 9.3 The sweep, and what was cleared

Swept every DEBUG/INFO site that logs a URL, a header dict, or a `provider_config`. Sites are
described by symbol/regex, not line number, because line numbers drift and the structural guard
(`tests/test_egress_log_redaction.py`, L2) keys on the log-call text.

**Redacted by this task** — the six leaking sites, all on the `kitty.bridge` logger:

- `logger.debug("Upstream POST → %s", url)` — four sites, one per protocol handler. Routed
  through `BridgeServer._debug_url` (a `@staticmethod` wrapping `redact_url_for_display`).
- `logger.debug("Request headers: %s", dict(request.headers))` — the inbound request's
  `authorization` header carries the bridge's own API key. Header rule applied.
- `logger.debug("Upstream response headers: %s", dict(upstream.headers))` — `set-cookie` and
  friends. Header rule applied.

**Cleared, with reasons** — reviewed and left as written:

- `model_context_sync.py` — logs `REMOTE_OVERRIDES_URL`, a static public URL, no credential.
- `logger.info("Bridge server started on ...")` — the bridge's own listen host and port.
- `logger.debug("Capping single-backend cooldown ...")` and
  `logger.debug("Selected backend: ...")` — cooldown numbers, profile/provider/model names,
  health booleans, indices. No credential.
- `Request body:` / `Translated CC request:` dumps — user conversation content. This is the
  very thing the debug log exists to inspect; masking it would defeat the log's purpose, and a
  secret a user pasted into their own conversation is not a credential kitty holds.
- `model_context.py:400` warning — provider and model names only.
- `providers/openai_subscription.py` `_log_cf_cookies` and the "Filtered N non-CF cookies" log
  — already redacted by design (name allowlist + 8-char value truncation) and, decisively, on
  the `kitty.providers.openai_subscription` logger, which `_setup_debug_logging` does **not**
  attach the file handler to (it attaches to `kitty.bridge` and `kitty.providers.model_context`
  only). These lines never reach `bridge.log`.
- `providers/openai_subscription.py:352` — the TLS impersonation profile name, not a credential.

### 9.4 Enforcement

The redaction is enforced structurally, not just behaviourally, because "a new handler forgot
to call the helper" is the failure mode this policy exists to prevent and a property test of
the helper cannot see it. `tests/test_egress_log_redaction.py` (L2, following the
`tests/test_egress_coverage.py` pattern) scans `bridge/server.py` for the two *literal* log
shapes the six known sites use — `logger.debug("Upstream POST → %s", …)` and
`logger.debug("… headers: %s", …)` — asserts each routes through the corresponding helper,
and asserts its own scan finds the six known sites so it cannot rot into a no-op.

**The deliberate limit, recorded so a future reader does not rely on the broader claim:** the
guard's patterns are anchored on those two literal format strings. A future log call that
carries a URL or header dict under a *different* format string — `logger.debug("POST %s",
url)`, `logger.info("upstream_url: %s", url)` — does **not** match and would ship unredacted.
The guard pins the shapes that exist today; a new shape needs its pattern added to
`_REDACT_SITE_PATTERNS` in the same change that adds the call. Widening the regex to any
`logger.debug(… %s, url)` shape was considered and rejected: it would sweep non-URL `%s`
arguments (message ids, model names) into the redaction and force every call site to carry an
exemption comment, which is the failure mode the design avoids.

## 10. Profile schema: one base-URL channel

### 10.1 The rule

A profile carries its base URL in **`provider_config["base_url"]`**, and nowhere else. The
wizards (`src/kitty/cli/profile_cmd.py:171-176`, `src/kitty/cli/setup_cmd.py:79-84`) write
the channel the user types into; pre-flight (`src/kitty/validation.py:60`) and the five
provider adapters that honour it (`custom_openai`, `custom_anthropic`, `minimax`, `ollama`,
`ollama_cloud`) read it from there. `Profile.base_url` — once a typed field on the schema —
is gone. A profile whose raw input carries a non-`None` top-level `base_url` raises
`ValidationError` with a message naming `provider_config["base_url"]`; a `None` or absent
key is unchanged behaviour.

### 10.2 Decisions, and why

**D9** (KBR-158): `provider_config["base_url"]` is the only channel. *Product owner,
2026-09-17.*

- The alternative channels were: (a) **make `Profile.base_url` authoritative and migrate**
  — inverts the channel TEST_SUITE §7.5.2 already names authoritative, requires relaxing the
  HTTPS-only / 2083-char / normalisation constraints (local vLLM/LM Studio/Ollama endpoints
  use `http://`), changes all five reader sites plus both wizards, and needs a profile
  migration. The wizards can validate the URL at write time; a typed channel buys little
  over the value-aware rejection the schema already runs. (b) **type `provider_config`
  instead** — a per-provider discriminated-union design (Vertex reads `project_id/
  location`; Azure reads `resource`/`api-version`; etc.), far larger than a Low-priority
  bug ticket should carry, and duplicates per-adapter checks such as minimax's
  `provider_config.base_url must be a non-empty http:// or https:// URL`. (c) **silent
  deletion** — the field was never read, so deleting it without a rejection changes
  nothing behaviourally. Rejected: the ticket's own acceptance clause ("never silently
  ignored") requires the rejection; the cost is one `@model_validator(mode='before')` that
  fires only on a non-`None` value, and the pointed message is the *safer* outcome for any
  user who hand-edited top-level `base_url` (today's silent ignore routes their request to
  the provider's default endpoint, which is the wrong place to be wrong).
- **The rejection is value-aware, not key-presence.** `store.py` serialises profiles with
  `model_dump(mode="json")` and no `exclude_none`; every existing `profiles.json` file on
  disk carries `"base_url": null` for the deleted optional field. A key-presence check
  would raise on load for every legacy profile, and `store._deserialize_entry`'s broad
  `except Exception` (F44 invalid-entry precedent) would silently drop every user's
  profile list — the never-silently-ignored failure mode aimed at every user at once. A
  serialized `null` means "not set" (the field was `Optional` with default `None`), so the
  check is `data.get("base_url") is not None`; only a typed value — the user who actually
  set a URL — gets the pointed message. The validator guards with `isinstance(data, dict)`
  because `mode='before'` also receives model instances, on which the key test would
  silently fall through.

### 10.3 Where the message reaches whom — and where it does not

The rejection fires where a `Profile` is constructed in code (`Profile(...)`,
`Profile.model_validate(dict)`). A hand-edited *store file* carrying a non-`null`
top-level `base_url` is dropped by `store._deserialize_entry`'s existing invalid-entry
handling (`logger.warning("Skipping invalid profile entry in store")` + skipped entry) —
the F44 contract the store already applies to any invalid entry. Surfacing per-entry
validation messages from the store would mean changing that contract and is out of
scope here; the canonical decision record in `.system_design/steps/
kbr158_delete_dead_profile_base_url.md` carries the same note for the next reader.

### 10.4 Verification

- `grep -rn "base_url" src/kitty/profiles/schema.py` returns only the validator's
  message/docstring lines.
- `python -c "from kitty.profiles.schema import Profile; assert 'base_url' not in
  Profile.model_fields"` exits 0.
- `pytest tests/test_profile_schema.py` is the single L1 home for the claims; no L2/L3/L4
  test is added (the "one channel carries the base URL" half of the acceptance clause is
  pinned by field absence — no resolver can read what does not exist — so no resolver-side
  test is needed).
- Cross-references: `TEST_SUITE.md §7.5.2` records `provider_config["base_url"]` as
  "the product's own channel" and now no longer cites the deleted field; the harness
  fixture docstring at `tests/harness/bridge.py:profile_for` and the
  `src/kitty/launchers/base.py:build_spawn_config` docstring are updated to the same
  post-deletion state.



## 11. Credential store

Traces to [KBR-87](https://shelpuk.atlassian.net/browse/KBR-87) (T-G12, §6.2.4's `keyring`
row + the KBR-154 scope addition). This section is the To Be state of
`src/kitty/credentials/`; TEST_SUITE.md §6.2.4 owns the dependency-contract test for
`keyring`.

### 11.1 Components

- **`store.py`** — `CredentialBackend` (abstract `get`/`set`/`delete`), `CredentialError`,
  `CredentialNotFoundError`, and `CredentialStore`: a fallback chain; `get` returns the
  first non-`None` result (`.strip()`ped); `resolve` raises `CredentialNotFoundError` when
  every backend returns `None`. **A backend error is a stop, not a miss:**
  `CredentialStore.get` catches nothing, so `None` advances the fallback chain while a
  raised `CredentialError` aborts it. With today's single-backend chains the difference is
  unobservable; the invariant is recorded so the ticket that wires a second backend
  inherits it rather than rediscovers it.
- **`file_backend.py`** — `FileBackend`: JSON `{ref: base64}` under
  `platformdirs.user_config_dir("kitty")/credentials.json` (or an explicit path), guarded
  by a `filelock` (5 s timeout, F38), written atomically (`mkstemp` + `os.replace`) with
  POSIX `0600`/`0700`. F37: a file that is not valid **JSON** is backed up
  (`*.corrupt.<ts>.<pid>`); on the success branch the store restarts empty behind a
  CRITICAL log, on the failure branch (read-only mount, etc.) it raises
  `CredentialError` instead of silently writing `{}` over the damaged original —
  same silent-loss argument as the file-level guards, applied to F37. KBR-291 extends
  the same guarantee to the other two file-level damage shapes: an invalid-UTF-8 file
  and a valid-JSON-non-dict file are backed up the same way; `get` raises
  `CredentialError` at the boundary, while `set`/`delete` proceed from `{}` — the
  write path forgives because the recovery command the error names reaches a `set`
  call, and raising there would crash the very command the message points at. The
  CRITICAL log + backup fire on both paths, so nothing is silent.
- **`keyring_backend.py`** — `KeyringBackend`: delegates to the `keyring` package with
  service name `"kitty"`. `get` swallows every exception to `None`; `set` wraps failures in
  `CredentialError` (F39 — headless Linux without D-Bus raises `NoKeyringError`); `delete`
  suppresses `keyring.errors.PasswordDeleteError` only. **Dormant by construction:** no
  production site wires it into a `CredentialStore` — every construction site uses
  `CredentialStore(backends=[FileBackend(...)])`. It is the exported OS-native option; the
  §6.2.4 contract pins what it would get from `keyring` the day it is wired.

### 11.2 The corruption contract (KBR-87, KBR-291)

`FileBackend.get` distinguishes **absent** from **corrupt** at two layers:

- **Per-ref** (KBR-87): ref not in the store → `None` ("no credential"); ref present
  but the stored value is not decodable → `CredentialError` naming the ref, chained
  (`raise ... from`) from the cause. Undecodable means one of the four measured
  shapes: not valid base64 (`binascii.Error`), a non-ASCII string (`ValueError` —
  raised by `b64decode`'s ASCII-encode step, independent of the `validate=` flag),
  decoded bytes not valid UTF-8 (`UnicodeDecodeError`), or a non-string stored value
  (`TypeError` — the file is user-editable JSON). An explicit JSON `null` value is
  the **absent** spelling, not corruption: `set()` never writes it and
  `data.get(ref)` returns `None` for it, so it takes the absent branch by
  construction — pinned by test. `binascii.Error` and `UnicodeDecodeError` both
  subclass `ValueError`, so the handler is `except (ValueError, TypeError)` — exactly
  as narrow as naming the subtypes, and complete over every measured shape.
- **File-level** (KBR-291): the file's bytes are not valid UTF-8, or its JSON parses
  to a non-dict (top-level list, string, number, bool, null) — both raise
  `CredentialError` naming the file at the backend boundary. Shape (b) chains from
  the `UnicodeDecodeError`; shape (a) does not chain (the JSON parsed cleanly, so
  there is no underlying exception). F37 (invalid JSON) has a two-branch contract:
  the success branch (backup rename succeeded) is unchanged — `get` returns `None`,
  the file is reset to `{}` behind a CRITICAL log; the failure branch raises
  `CredentialError` honestly rather than writing `{}` over the still-damaged
  original.

**Where the signal lands — per call site.** The raise is only half the contract; a signal
nobody receives is a traceback on every startup path:

- **`egress_store.resolve_egress`** — catches `CredentialError` around the stored-gateway
  password read and re-raises its existing `ValueError` (chained, cause text included).
  This is load-bearing: `cli/main.py` wraps `resolve_egress` in `except ValueError` with a
  deliberate carve-out keeping `kitty egress` / `kitty cleanup` reachable when the stored
  gateway is broken — without this receiver, `kitty egress`, the very command the message
  tells the user to run, would die at startup.
- **`bridge_runner.py`** (both branches — single profile and balancing members) — the
  same `Error: …` + exit treatment as `cli/main.py`; this is the background-bridge
  startup path, where the child's output lands in the service journal and a raw
  traceback is precisely the diagnostic failure the contract exists to prevent.
  (Added in review round 1 — the first receiver survey grepped `cli/` only and missed
  these two sites; the omission was an oversight, not a scoping decision.)
- **`cli/launcher.py`** — the launch path's `except CredentialNotFoundError` widens to
  include `CredentialError`: same clean `Error: …` + exit 1.
- **`cli/main.py`** (three profile-resolution sites) — `cred_store.get` is wrapped so the
  corruption message replaces what would otherwise be a raw traceback where today a clean
  `No API key found for profile X` + exit fires.
- **`cli/profile_cmd._find_reusable_auth_ref`** — a corrupt profile is skipped (treated as
  not reusable) so the setup wizard stays reachable; re-entry *is* the recovery.
- **`cli/doctor_cmd.py`** (both credential checks) — corruption is reported as a failed
  check with the corruption message; the diagnostic tool must not crash on the condition it
  exists to diagnose.

**Why fail-loud rather than warn-and-return-None.** Collapsing "damaged" into "absent"
surfaces both as `CredentialNotFoundError` ("no API key for profile X"), which sends the
user to re-enter a key they already have — the KBR-134/KBR-154 diagnostic family, where the
misleading message costs more than the underlying fault. A CRITICAL-log precedent already
exists for file-level corruption (F37); per-ref corruption now raises through the same
exception hierarchy `KeyringBackend.set` already uses (F39). *Product owner decision,
2026-09-19 (raise `CredentialError`; warn-and-None and a sentinel result type were the
rejected alternatives — the former is indistinguishable at the boundary, the latter breaks
the `str | None` interface at five call sites).*

**Why `validate=True`.** `set` writes pure base64 alphabet, so `validate=True` accepts
everything the store itself writes and rejects hand-edited or damaged values that the
default silently truncates into plausible garbage. The load-bearing arm is pinned by
test: `"ab@=="` strips to the length-valid `"ab=="` and silently decodes to the
single byte `b"i"` under `validate=False` — corruption read back as a plausible
single-character credential — while `validate=True` rejects it outright.

**Why the write path forgives file-level damage (KBR-291).** `set`/`delete` swallow the
file-level `CredentialError` (`_read_raw_for_write` treats a damaged file as empty) while
`get` propagates it. The recovery command the error message names (`kitty setup`) reaches
a `set` call — if `set` raised, the wizard would crash on the very write the user is
performing, re-creating the diagnostic failure this contract exists to eliminate. The
CRITICAL log and backup still fire from `_read_raw` before the write path forgives, so
nothing is silent. `CredentialStore.delete` is already best-effort
(`contextlib.suppress(Exception)`).

**Why `_read_raw_for_write` re-raises when `self._path` still exists.** When `os.replace`
fails (e.g., a read-only bind mount on the credentials directory), the damaged file
remains at `self._path` and the CRITICAL log claims the original was preserved at a
backup that does not exist. The next `set` swallowing the exception would overwrite the
damaged original with no backup anywhere — silent credential loss accompanied by a
confident false promise. `_read_raw_for_write` checks `self._path.exists()` after the
raise: present (backup failed) → propagate, the recovery command reports the failure;
absent (backup succeeded) → return `{}`, the write proceeds. Neither the shape (a) arm
nor the shape (b) arm writes `{}` after the backup — shape (a) because doing so would
erase the `self._path.exists()` signal the guard reads, and shape (b) because the bytes
cannot be decoded and recreating `{}` adds nothing. Both rely on that single file-system
check to distinguish "backup succeeded" from "backup failed".

**Why the F37 path also raises on backup failure.** F37's success branch (invalid JSON
with a successful backup) is unchanged: `get` returns `None`, the file is reset to
`{}`, no raise. The failure branch (invalid JSON with a failed backup) now raises
`CredentialError` rather than writing `{}` over the still-damaged original — the same
silent-loss argument as the file-level guards, applied to the pre-existing F37 path.
The `_back_up_damaged_file` helper returns the rename's success; F37 only writes
`{}` and returns when the rename succeeded. Acceptance criterion 3's "F37 unchanged"
holds for the success branch; this paragraph records the explicit failure-branch
contract change.

**Why wizard `cred_store.set` sites wrap `CredentialError`.** The seven wizard set
sites (`setup_cmd.py`, `profile_cmd.py`, `auth_cmd.py`, `egress_cmd.py`) wrap
`cred_store.set(...)` in `try/except CredentialError: print_error(...); exit`. Without
the wrappers, a backup-failed raise would propagate as a Python traceback at the
recovery command — the KBR-154 diagnostic family on the very path the error names.
The wrappers produce the same clean `Error: …` + exit the receiver map produces for
every `get` site.

### 11.3 The keyring dependency contract

§6.2.4's rule for a dependency whose behaviour varies by platform **by design** ("where no
stable neighbour exists") is to record which mechanism was chosen. For `keyring` the
chosen mechanism is: the declared floor `>=23.0` **plus a contract on the resolution
mechanics**, not on live native services. `tests/test_keyring_backend_contract.py` (L2)
pins, all measured against the installed release:

1. the module-level API (`get_password`/`set_password`/`delete_password`) delegates to
   `keyring.get_keyring()` — so pinning `get_keyring()` pins where credentials go;
2. `PYTHON_KEYRING_BACKEND` selects the backend (`keyring.core.load_env()` — a public
   module function — is pinned, not the private `_detect_backend` ordering);
3. resolution always lands on a `keyring.backends.*` class — on every platform, including
   headless Linux where it is the `fail.Keyring` fallback;
4. per-platform native class where the native service is reachable: macOS Keychain
   (developer machines only unless pyobjc is installed — the bare `keyring>=23.0`
   dependency does not carry it, so on the macOS CI leg this arm skips with a stated
   reason rather than pretending coverage), Windows Credential Manager (pywin32-ctypes is
   a base dependency, so the Windows leg genuinely asserts);
5. `keyring.errors.PasswordDeleteError` ⊂ `KeyringError` — the exception family
   `KeyringBackend.delete` suppresses.

**Deliberately not asserted:** a specific native class on Linux unconditionally. On a
D-Bus-equipped developer box the SecretService backend classifies and resolution lands on
the chainer, not `fail.Keyring`; asserting the fallback unconditionally would be red for
environmental, not contract, reasons — the mirrored form of the trap the ipaddress
contract's docstring records. The fallback is asserted only behind a SecretService-
unavailable guard.

### 11.4 Verification

- `tests/test_keyring_backend_contract.py` (L2) — the five §6.2.4 pins above plus the
  no-op self-guard (sync-test count floor; the aiohttp twin's guard counts coroutine
  methods and does not transfer verbatim).
- `tests/test_credential_store.py::TestFileBackend` — the corruption arms (parametrised:
  invalid base64, non-ASCII string, invalid UTF-8, non-string), absent-vs-corrupt, and
  per-ref isolation; the F37 backup tests in
  `tests/credentials/test_stage7_credentials.py` are unchanged and stay green.
- `tests/test_credential_store.py::TestFileLevelCorruption` (KBR-291) — the file-level
  arms: invalid-UTF-8 bytes (shape b, constructed values, chained from
  `UnicodeDecodeError`) and valid-JSON-non-dict payloads (shape a, parametrised over
  list/string/number/bool/null, no chaining), backup content + CRITICAL log pinning
  path and backup path, `set`/`delete` after damage (the write-path forgiveness),
  the write-path propagation when the backup rename fails (read-only-mount case,
  `os.replace` monkeypatched to raise, `set` must not overwrite the damaged
  original), the F37 backup-failure raise (the round-3 contract change — F37 no
  longer silently resets when the rename failed), the success-branch message naming
  the real backup path, and the F37 unchanged regression pin.
- `tests/test_egress_store.py` — the corrupt stored gateway password raises the documented
  `ValueError` (chained, naming `kitty egress`), the twin of the existing
  missing-credential test.
- `tests/test_doctor_cmd.py` / the profile-wizard reuse scan — corruption reported, flow
  stays reachable.

## 12. Request ingress trust boundary

Each bridge-protocol POST route runs a per-protocol normalizer at the trust boundary,
**before** any translator or upstream call. The normalizer enforces the contract the
downstream path can rely on: "every shape the translator iterates or subscripts is the
shape the schema says." Anything else is rejected with the dialect's 400 envelope
([KBR-82](https://shelpuk.atlassian.net/browse/KBR-82), §6.2.1 — *the bridge must never
500 on a malformed body*). The schemathesis conformance run is the discoverer; the
L2 pins in `tests/test_route_preflight.py` and sibling files are the deterministic
regression gate.

### 12.1 The four normalizers and their scopes

| Normalizer | Protocol | Required-shape checks (→ 400) | Tolerated-as-absent (omitted from the body) |
|---|---|---|---|
| `_normalize_messages_request` (server.py) | `/v1/messages` (Anthropic Messages) | body, `model`, `messages`, `messages[*]`, `tools`, `tools[*]`, `tools[*].name` (non-empty string) | — (required by the Anthropic contract; missing `model`/`messages` 400s) |
| `_normalize_chat_completions_request` (server.py) | `/v1/chat/completions` (OpenAI Chat Completions) | body, `messages`, `messages[*]` | `tools` (deliberately not validated: CC tools are passed through to the upstream without iteration; measured 200 on non-array tool bodies; CC tools nest `name` under `function`, so a flat `tool["name"]` check would 400 every legitimate CC body — there is no measured CC tools 500 to prevent) |
| `normalize_responses_request` (`responses/translator.py`, [KBR-144](https://shelpuk.atlassian.net/browse/KBR-144)) | `/v1/responses` (OpenAI Responses) | body, `input` as string-or-list, list elements as objects, plus the five `/v1/responses` shapes closed with KBR-159 | — |
| `_normalize_gemini_request` (server.py) | `/v1beta/...:{generate,streamGenerate}Content` | body, `contents` (list of objects) — extended with [KBR-288](https://shelpuk.atlassian.net/browse/KBR-288) to also reject required-shape defects in `contents[*].parts`, `parts[*]`, `parts[*].functionCall`/`functionResponse` member defects, `tools[*]`, `tools[*].functionDeclarations` and its members | `systemInstruction` (KBR-82), `generationConfig` (KBR-82), `toolConfig` subtree (KBR-82); `tools` itself when non-list (KBR-288 — schema permits absence) |

### 12.2 Dialect 400 envelopes

Each normalizer raises a dialect-specific `Invalid*Request(ValueError)`. The handler
catches it and returns the route's published envelope (already documented in
`openapi/kitty-bridge.yaml`):

- **Anthropic Messages**: `{"type": "error", "error": {"type": "...", "message": "..."}}`.
- **OpenAI Chat Completions**: `{"error": {"code": "invalid_request", "message": "..."}}`.
- **OpenAI Responses**: `{"error": {"code": "invalid_input", "message": "..."}}` (the
  `reason: "invalid_input"` discriminator introduced with KBR-144).
- **Gemini**: `{"error": {"code": <int 400>, "message": "...", "status": "INVALID_ARGUMENT"}}`.

The OpenAPI document declares each of these as the published 400; no undocumented
statuses. The schemathesis schema-conformance check pins this contract — a guard that
catches a new 400 the schema does not declare as well as a 500 the schema forbids.

### 12.3 Disposition policy — 400 for required, tolerate-as-absent for optional

The normalizers deliberately split the disposition of malformed user input, not by
crash class but by **whether the field is required by the schema**.

- **Required shapes** (the request cannot proceed without them, or a non-empty
  iteration over the value is the contract the downstream path expects): **400**. A
  missing or wrong-typed `contents`, a `contents[i]` that is not a `Content` object,
  a `parts[j]` that is not a `Part` object, a `tools[*].name` that is missing —
  these are client mistakes. The OpenAPI document calls the client out for the wrong
  shape; the bridge's job is to surface that cleanly, not to silently reshape.
- **Optional shapes** (`systemInstruction`, `generationConfig`, `toolConfig` on the
  Gemini side; `tools` itself on Gemini): **tolerated as absent**. The schema
  documents them as optional with a default; an optional field whose shape is
  wrong is the same as the field being omitted. Reshaping silently keeps permissive
  Gemini clients reachable (some send partial tools arrays; some omit `systemInstruction`
  entirely) and keeps the KBR-82 conformance run from spiking on every minor shape
  drift.

**The split, not blanket tolerance.** A blanket "tolerate everything" rule would 500
no client but would also fail the schemathesis schema-conformance check the moment a
fuzzer sends an iterate-able-required shape (the documented `contents[*].parts[*]`
is required when `contents` is non-empty — a tolerated-as-absent `parts` would
silently drop a real conversation turn, breaking I1 message-fidelity rather than
breaking I2 indistinguishability).

**Why not blanket 400 either.** The schemathesis fuzzer surfaces the same class of
shape defect on optional envelopes (`generationConfig: true`, `toolConfig: 5`) as on
required ones; blanket 400 would 400 a real client whose implementation drifted a
single optional field. The KBR-82 product-owner decision — split, not blanket —
locks the policy in for every widening this ticket or its siblings propose.

### 12.4 Gemini ingress — KBR-288 widening

[KBR-288](https://shelpuk.atlassian.net/browse/KBR-288) closes the Gemini request
path. Two things happened between the ticket's filing and this work: the
skip-never-raise commits on `main` (post-KBR-82 follow-ups) made the translator
silently skip most container-shape defects the ticket's probe list names, and
empirical probing at HEAD (`.scratch/probe_kbr288.py` at requirements time) found
the crash classes that survived those guards. The ticket therefore delivers two
things:

1. **Genuine crash closure** — shapes that still 500 at HEAD:

| Shape | Crash at HEAD |
|---|---|
| `contents[i].role` present and non-hashable (list / dict) | `TypeError: unhashable type` (`_ROLE_MAP.get(role, role)`) |
| `contents[i].parts[*].text` present and non-string | `TypeError: sequence item 0: expected str instance` (all three text branches: user, assistant, assistant-thought) |
| `tools[i].functionDeclarations` present and `null` | `TypeError: 'NoneType' object is not iterable` (`tool.get("functionDeclarations", [])` — the default does not fire when the key is present-null) |
| `tools[i].functionDeclarations` present and `true` | `TypeError: 'bool' object is not iterable` (same site) |
| `systemInstruction.parts[*].text` present and non-string | `TypeError: sequence item 0` (`_extract_text` filters non-dict parts but joins unguarded text) |

2. **Disposition tightening** — the Wave-2 silent-skip shapes become boundary
   400s, per §12.3's required-shape policy (a silent drop inside a required
   chain is an I1 message-fidelity hit, not a clean outcome):

| Shape | Old disposition | New disposition | Rationale |
|---|---|---|---|
| `contents[i].role` present-and-non-string | crash (see above) | 400 INVALID_ARGUMENT | required Content member, documented `type: string` (PO decision 2026-09-21) |
| `contents[i].parts` not a list | tolerated-skip | 400 INVALID_ARGUMENT | required; translator iterates it |
| `contents[i].parts[*]` not a dict | tolerated-skip | 400 INVALID_ARGUMENT | required; per-part member access |
| `contents[i].parts[*].text` non-string | crash (see above) | 400 INVALID_ARGUMENT | required Part member, documented `type: string` |
| `parts[*].functionCall` non-dict / without string `name` | tolerated-skip | 400 INVALID_ARGUMENT | required-when-present member; `fc["name"]` read |
| `parts[*].functionResponse` (mirror of functionCall) | tolerated-skip | 400 INVALID_ARGUMENT | mirror |
| `tools[i]` not a dict | tolerated-skip | 400 INVALID_ARGUMENT | required Tool member |
| `tools[i].functionDeclarations` present-null or non-list-non-dict | tolerated-skip (string/dict iterate empty) / crash (null, bool) | 400 INVALID_ARGUMENT | required list member; present-null defeats the `.get(..., [])` default |
| `tools[i].functionDeclarations[*]` not a dict / without string `name` | tolerated-skip | 400 INVALID_ARGUMENT | required FunctionDeclaration member; `fd["name"]` read |

The optional envelopes keep their tolerated disposition, with one
crash-proofing fix inside the translator: `_extract_text` gains an
`isinstance(p.get("text"), str)` filter so a non-string text inside a
tolerated `systemInstruction` envelope does not crash the join — a tolerated
envelope gets no boundary check, so the translator carries the leaf guard.

**Key absence vs wrong type.** Every "400" row above fires on a *present*
value whose documented type is violated. A *missing* key stays tolerated
throughout — `{"contents": [{"role": "user"}]}` (no `parts`) proceeds with an
empty parts list, exactly as KBR-82 left it: there is no crash class on
absence (`content.get("parts", [])` handles it), and requiring the key would
400 bodies the pre-KBR-288 contract accepted. The §12.3 fidelity argument
("a tolerated-as-absent `parts` would silently drop a real conversation
turn") is about a *present-but-wrong-typed* `parts` being silently treated
as absent — not about the key's absence.

Helpers (`_validate_gemini_content`, `_validate_gemini_part`,
`_validate_gemini_tools`) keep the call sites readable. Each helper is
covered by route-level L2 pins in `tests/test_route_preflight.py`
(per-shape `assert status == 400` plus the dialect-envelope check).

**Ordering dependency (verified):** validation runs first in `_handle_gemini`
— normalizer, then translate, truncate, compaction, size check — so no 400 can
be preempted by a later failure, and `translate_request` has exactly one call
site (immediately after the normalizer), which is what makes the
boundary-invariant claim structural rather than aspirational. The 400 shapes
are all ones a schema-conforming Gemini CLI cannot send; the conformance
fuzzer's distribution is the only producer.

### 12.5 Verification

- `tests/test_route_preflight.py` (L2) — per-protocol pins; the Gemini rows carry
  every To Be-table row of §12.4 with a positive control on `:generateContent`,
  plus one tightening shape and one crash-class shape on `:streamGenerateContent`
  (the shared-handler proof), asserting HTTP `status == 400`, `error.code` is
  an integer (any int — a regression writing `code: 500` over HTTP 400 would
  still pin), `error.status == "INVALID_ARGUMENT"`, and no traceback leak.
- `tests/test_openapi_conformance.py` (L2) — the schemathesis conformance run; the
  discoverer. The 400 envelope is schema-documented so widening the normalizer does
  not introduce an undocumented status.
- `tests/test_openapi_schema.py` (L2) — pins the `INVALID_ARGUMENT` enum value.
- `src/kitty/bridge/gemini/translator.py::translate_request` then raises no
  exception on any user-supplied input: container shapes arrive
  boundary-validated; the leaf-value reads (role, text, functionCall/functionResponse/
  functionDeclaration names, tool names) carry local isinstance guards — the
  structural-review criterion the ticket names. (`carry_gemini_tool_choice` already
  satisfies this for the `toolConfig` subtree.) The same shape now covers the
  eleven published `generationConfig` sampling fields: KBR-213 carried five,
  [KBR-301](https://shelpuk.atlassian.net/browse/KBR-301) carried the six
  remaining (`candidateCount`→`n`, `presencePenalty`, `frequencyPenalty`,
  `seed`, and the collision pair `responseLogprobs`→`logprobs` /
  `logprobs`→`top_logprobs`), each guarded presence+type with the bool/int
  subclass exclusion the harness reader's `_typed_leaf` uses
  (`TEST_SUITE.md` §3.3.1b is the closed-set authority; gap row G44
  registers the five named format-specific control fields that stay dropped).

**Hop-2 landing on the rebuild-trio — [KBR-305](https://shelpuk.atlassian.net/browse/KBR-305).**
The six KBR-301 sampling keys reach the CC body. On the **verbatim-forwarder**
routes the body reaches the upstream with the keys intact. On the three
rebuild-from-allowlist adapters — `AnthropicAdapter` (and its four delegates
`zai_coding`, `custom_anthropic`, `minimax_token`, `opencode_go` on
Messages-routed models), `BedrockAdapter`, `OllamaCloudAdapter` — the
allowlists read only the keys their destination wire accepts. The per-adapter
disposition, verified against each destination's published schema or source
(Anthropic Messages `platform.claude.com/docs/en/api/messages` accepts none
of the six; Bedrock `InferenceConfiguration` is exactly
`{maxTokens, temperature, topP, stopSequences}`;
[Ollama `ChatRequest` + `Options`](https://raw.githubusercontent.com/ollama/ollama/main/api/types.go)
on main carries `Logprobs bool`/`TopLogprobs int` top-level and
`Seed`/`PresencePenalty`/`FrequencyPenalty` under `options` — but no
`n`/`num_choices` field anywhere):

| CC key | Anthropic family | Bedrock | Ollama Cloud |
|---|---|---|---|
| `seed` | drop | drop | carry → `options.seed` |
| `presence_penalty` | drop | drop | carry → `options.presence_penalty` |
| `frequency_penalty` | drop | drop | carry → `options.frequency_penalty` |
| `logprobs` (bool) | drop | drop | carry → top-level `logprobs` |
| `top_logprobs` (int) | drop | drop | carry → top-level `top_logprobs` |
| `n` | drop | drop | drop (no Ollama equivalent) |

**Ticket correction, recorded.** The ticket's text states "Ollama Cloud's
`/api/chat` accepts `seed` and `options.n`". The `options.n` half is wrong —
Ollama's Go source declares no `n`/`num_choices` field anywhere, so `n`
cannot be carried and is drop-and-register on Ollama like the others; the
five other Ollama keys are real (Go struct + `docs.ollama.com/api/chat` for
`logprobs`/`top_logprobs`). **Ticket scope clarification.** The two logprob
carries are accepted-but-inert upstream — ollama.com accepts the keys but
serves logprobs from local models only (maintainer statement,
ollama/ollama#13638), and `OllamaCloudAdapter.translate_from_upstream` does
not forward response logprobs either way. The carry is chosen because (a)
forwarding keeps both projections agreeing on those addresses with no
claimed delta at the wire, and (b) the carry starts working with no further
change if Ollama ever serves them on cloud. The reader widening
(`tests/harness/reader_ollama.py`) is owed either way because the published
`ChatRequest` carries the keys — the reader is total over the published
format, independent of whether the bridge forwards them. The
register-row P43 (per the test suite's design of record) lands the
conditional carry/drop split at the rebuild-trio sites; the row text records
the split and the accepted P31/P32-rule departure (L1 suite carries the
discriminating duty on the five ollama carries per `TEST_SUITE.md` §6.1's
mutmut scope).

### 12.6 Out of scope

- The response direction (`translate_response`, `translate_stream_chunk`) processes
  upstream CC bodies, not user input, and is not fuzzer-reachable.
- The Responses, Chat Completions, and Messages ingress normalizers — already
  hardened by KBR-82, KBR-144, KBR-159, KBR-169. KBR-288 covers Gemini only.
- The deeper nightly fuzz job ([KBR-116](https://shelpuk.atlassian.net/browse/KBR-116),
  T-K7) — the discoverer for ongoing shape drift; this ticket closes the KBR-82
  conformance-suite-discoverable crash set on the Gemini path.

---

## 13. Bedrock adapter — list-form CC content → Converse blocks

Traces to [KBR-223](https://shelpuk.atlassian.net/browse/KBR-223). Read alongside §8
(Converse reasoningContent emission), which owns the adapter's view of one content-block
spelling, and `TEST_SUITE.md` §3.2.1 (M3/M4 register rows), whose truncation prose records
the list-form gap this section closes. This section owns the adapter's view of *list-form*
content — tool results and user turns — and the truncation sites that count their bytes.

### 13.1 The rule

`kitty.providers.bedrock.BedrockAdapter` translates list-form CC content part-by-part into
Converse blocks via a shared module-level mapper. The four part kinds that matter, and
their outputs:

| CC part | Converse block | Notes |
|---|---|---|
| `{"type":"text","text":T}` | `{"text": T}` | text only |
| `{"type":"image_url","image_url":{"url":"data:image/<fmt>;base64,<b64>"}}` | `{"image":{"format":<fmt>,"source":{"bytes":<decoded>}}}` | fmt ∈ {png,jpeg,gif,webp}; http(s) URLs dropped |
| `{"type":"image","source":{"type":"base64","media_type":"image/<fmt>","data":<b64>"}}` | same image block | Anthropic-native shape arrives via the Messages→CC translator's tool_result passthrough (`tr.get("content","")`) |
| `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":<b64>"}}` | `{"document":{"format":"pdf","name":<title or "document">,"source":{"bytes":<decoded>}}}` | DocumentBlock requires `name` and `source` per the installed service model |

Anything else — non-data-URL images, `file` sources, unknown types, undecodable payloads —
is **dropped**, the established posture for unmappable shapes (KBR-222,
`build_user_content_message`'s `file` source drop). If every part drops, the result falls
back to `[{"text": ""}]` — the empty list is validator-clean (measured), but a non-empty
block list is the shape the service is documented to take, and the fallback costs one line.

Two call sites:

* `_translate_tool_result_msg` — list-form `content` becomes `toolResult.content`. The
  pre-fix behaviour was a verbatim copy that left CC `{"type": …}` keys in place; botocore's
  parameter validator rejected the body and `client.converse` failed the turn locally.
* The user branch of `translate_to_upstream` — list-form user content becomes a list of
  Converse blocks in part order. This replaces KBR-222's provisional flatten (which joined
  text parts and dropped images): one Converse `text` block per text part, in order. On
  hop-1 routes `build_user_content_message` only assembles a parts list when a non-text part
  is present, so the common Messages-ingress turn is unaffected; a CC-native agent sending a
  text-only parts list sees a wire-visible change from a single joined block to one block
  per part.

### 13.2 The truncation shared extractor

Three sites check tool-result payloads for the 50,000-char limit: `_truncate_oversized_tool_results`
(CC + Anthropic-native arms), `_truncate_oversized_responses_outputs` (Responses arm), and
`_compact_messages` step 1 (CC arm, compaction-engaged only). All three now share one
module-level size extractor in `server.py`:

```python
def _tool_result_content_size(content: str | list) -> int:
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, ensure_ascii=False))
    except (TypeError, ValueError):
        return -1  # F33-style: not measurable → leave alone
```

An over-limit list-form payload is replaced by the same `[Tool output truncated — original
size: N chars]` notice string the string branch uses (the list collapses to the notice).
String-arm behaviour is byte-identical to today, so the existing string-only Hypothesis
oracles (`_expected_truncation_count`, `_apply_cc/native/responses_truncation`) keep
passing unchanged — the change only *widens* what the sites consider.

This closes the list-form gap recorded in `TEST_SUITE.md` §3.2.1 (M3, M4) as the KBR-169
"ships untruncated" symptom — KBR-223 owns the gap per KBR-169's adjacent-scope comment.

### 13.3 The contract oracle

L1 asserts the adapter's emission against the live service model — the same oracle pattern
§8 adopts for `reasoningContent`. The oracle lives on `_bedrock_body` (the pure builder
extracted by KBR-89), so the validation is unit-pure (no hook envelope, no boto3 client):

```python
input_shape = (
    botocore.session.Session()
    .get_service_model("bedrock-runtime")
    .operation_model("Converse")
    .input_shape
)
model_id, body = adapter._bedrock_body(cc_request)
botocore.validate.validate_parameters({"modelId": model_id, **body}, input_shape)
```

The oracle covers four branches:

1. The KBR-264 reasoning branches (populated and `_thinking_enabled`-injected-empty),
   re-driven through `_bedrock_body` with the same negative controls on the pre-fix
   `{"reasoningContent": {"text": …}}` spelling.
2. A tool result with list-form content — text + image + document variants — passes.
3. The pre-fix verbatim copy of that list content — `[{"type":"text","text":…}]` etc. —
   raises `botocore.exceptions.ParamValidationError`.
4. A user turn with an image part passes; the verbatim CC parts list for that turn raises.

### 13.4 Decisions, and why

- **Map late in the adapter, not normalize at ingress.** KBR-169's adjacent-scope comment
  offered "normalize the list-form shape once at the ingress boundary" as the alternative.
  Rejected: ingress normalization to strings would destroy the image/document payloads
  this section exists to carry (the same loss KBR-222 temporarily accepted on the user
  branch), and it would change the CC wire shape every other adapter reads. The Bedrock
  adapter is the only place the Converse block union is the target shape.
- **Drop unmappable parts rather than fail the turn.** Matches KBR-222's established
  posture (`build_user_content_message` drops `file` sources with a recorded reason). A
  hard failure on an unmappable part would recreate this ticket's defect class one level
  down. The all-parts-dropped fallback to `[{"text": ""}]` keeps the emitted list
  non-empty, matching the documented shape — the client-side validator accepts `[]` and
  `{"text":""}` (measured on the installed model), so the fallback is defence-in-depth
  against a service-side rule the oracle can't see.
- **Truncation collapses a list to the notice string rather than trimming within parts.**
  The string branch's contract (one notice, size recorded) is preserved; trimming inside
  part lists would need per-kind size budgets and would still change the shape. The size
  measure is the serialized JSON length — what actually ships on the wire — not the
  concatenated text length, so image/document payloads are counted too.
- **Oracle on `_bedrock_body`, not `translate_to_upstream`.** KBR-89's scope addition names
  the builder as the oracle's home: it is the pure function whose return value is the
  shipped body kwargs, so the validation is unit-pure and tight. Re-attaching `modelId`
  is the only envelope the oracle needs.
- **Shared size extractor in `server.py`, not `bridge/`.** The three call sites already
  live in `server.py`; a shared helper belongs beside them. Putting it in `bridge/` would
  invert the dependency direction (a translation-time helper in a transport module).
- **No new register row, no new wire-shape class.** The fix widens what M3/M4 consider
  (their triggers become content-shape-aware, not string-only); TEST_SUITE.md §3.2.1's M3/M4
  prose is updated to the closed state but the rows themselves stay. The Bedrock adapter
  continues to declare `WireShape.OTHER`; the emitted shape family is unchanged.

---

## 14. OpenAI OAuth login: browser flow and headless device-code flow

Traces to [KBR-317](https://shelpuk.atlassian.net/browse/KBR-317). The `openai_subscription`
provider authenticates through OpenAI's OAuth surface with the same `client_id` as the
Codex CLI (`app_EMoamEEZ73f0CkXaXp7hrann`). Two login paths now exist; both land in the
same `OAuthSession` shape, so the downstream `id_token` → API key exchange, the
credential store, and the `openai_subscription` adapter are unchanged.

### 14.1 Components

| Component | Module | Responsibility |
|---|---|---|
| `run_oauth_flow` (`kitty.auth.openai_oauth`) | **changed** | Single entry point for both flows; gains keyword-only `device_auth: bool \| None = None` and dispatches (§14.3). Browser path otherwise unchanged. |
| `run_oauth_device_flow` (`kitty.auth.openai_oauth`) | **new** | RFC 8628 device-authorization grant against `auth.openai.com`; mirrors `codex-rs/login/src/device_code_auth.rs`. |
| `is_headless_environment` (`kitty.auth.headless`) | **new** | Pure predicate over `os.environ`: `CI`, `SSH_CONNECTION`, `SSH_CLIENT`, `SSH_TTY`, or (Linux) neither `DISPLAY` nor `WAYLAND_DISPLAY` set. Variable list from `codex-rs/core/src/env.rs` at revision `1fc72c6` (file since moved; pinned revision cited in openai/codex#9253); the headless-preference pattern itself is current-main behaviour in `codex-rs/cli/src/login.rs::run_login_with_device_code_fallback_to_browser`. |
| CLI flag plumbing (`kitty.cli.main::_run_auth`, `kitty.cli.auth_cmd`) | **changed** | `kitty auth openai --device-auth` / `--no-device-auth`, forwarded as `device_auth: bool \| None` through `run_auth_openai` → `run_oauth_for_provider` → `run_oauth_flow`. |

**Entry points served** (all four reach the dispatch through
`run_oauth_for_provider`): `kitty auth openai` (`auth_cmd.py`), the profile
wizard's OpenAI path (`profile_cmd.py` `_create_profile_flow`), the
profile-edit re-auth branch (`profile_cmd.py`), and `kitty setup`
(`setup_cmd.py`). Only `kitty auth openai` has a flag surface; the three
interactive entry points inherit auto-detect.

### 14.2 The wire contract (as confirmed against codex-rs @ main, 2026-10)

Device-code login against `auth.openai.com` (the OAuth issuer):

1. `POST {issuer}/api/accounts/deviceauth/usercode` — body
   `{"client_id": "<CLIENT_ID>"}`, `Content-Type: application/json`.
   Response: `{"device_auth_id": str, "user_code": str, "interval": str-digits}`.
   The `interval` is a string of digits (codex deserialises it with
   `s.trim().parse::<u64>()`); the `user_code` key may also arrive as
   `usercode` (codex aliases both).
2. The CLI prints `{issuer}/codex/device` as the verification URL and the
   `user_code` as a one-time code (expires in 15 minutes per codex's prompt).
3. `POST {issuer}/api/accounts/deviceauth/token` — body
   `{"device_auth_id": str, "user_code": str}` — polled every `interval`
   seconds. **403 / 404 mean "still pending" → keep polling** (up to 15 min
   total). **Other non-2xx → fail fast with the status in the message.**
   Success returns
   `{"authorization_code": str, "code_challenge": str, "code_verifier": str}` —
   the **server** mints the PKCE pair for the device flow (unlike the browser
   flow, where the client mints it).
4. `POST {issuer}/oauth/token` (the standard token endpoint) — authorization-code
   grant with the server-issued PKCE pair and
   `redirect_uri = {issuer}/deviceauth/callback` (a **server-side** redirect
   target, not the user's loopback).
   Response: the same token shape as the browser flow
   (`access_token`, `id_token`, `refresh_token`, `account_id`, …).
5. `id_token` → API key exchange proceeds as in the browser flow (best-effort,
   non-fatal on failure).

Why the device path needs no local callback: the authorization code is
delivered to the CLI by the polling endpoint itself, so the browser never has
to reach the machine running kitty.

### 14.3 Selection rule

| Condition | Path taken |
|---|---|
| `--device-auth` passed | Device-code flow, regardless of environment. |
| `--no-device-auth` passed | Browser flow, regardless of environment (used with `ssh -L 1455:localhost:1455` when the user prefers the tunnel). |
| Neither flag, headless per §14.2's predicate | Device-code flow. |
| Neither flag, not headless | Browser flow (today's default). |

**Why auto-detect is kitty's default when the `codex login` subcommand's
default is browser-with-hint:** the ticket's premise is that a remote-VM
user should not need to discover a flag to authenticate. Codex itself
prefers device-code when headless in its other login paths
(`run_login_with_device_code_fallback_to_browser`, TUI onboarding); kitty
extends that preference to its single shared entry point. The device-code
protocol is mirrored exactly; the selection default is a recorded product
choice.

### 14.4 Decisions, and why

- **Mirror codex-rs's device protocol rather than design a bespoke UX.** The
  ticket names the Codex CLI implementation as the reference and OpenAI has
  already shipped, debugged, and documented this exact protocol for the same
  `client_id`. A bespoke "paste the callback URL" variant (Claude Code's
  approach) would be a different protocol from what OpenAI's
  `/codex/device` page implements, and would split kitty's UX from the
  tool it is mirroring.
- **Dispatch inside `run_oauth_flow`, not in the CLI layer.** There are four
  entry points into `run_oauth_for_provider` (§14.1); only one has a flag
  surface. A `device_auth` kwarg with a `None` default threaded through
  `run_oauth_for_provider` / `run_auth_openai` gives all four auto-detect
  with no extra wiring, keeps the selection rule in one place, and
  preserves the existing call-site behaviour when the kwarg is omitted.
  The signature change is additive and keyword-only.
- **`is_headless_environment` lives in its own `kitty.auth.headless` module.**
  It is a pure predicate over the environment with no dependency on aiohttp,
  asyncio, or the OAuth module; giving it its own home keeps it importable
  from tests and from future callers (e.g. tmux-wrap warnings) without
  dragging in the OAuth machinery.
- **Fail fast on unexpected statuses and transport errors, poll only on
  403/404.** Matches codex. A 401 or 5xx on the polling endpoint, or a DNS
  failure mid-poll, is a server-side or network problem the user cannot fix
  by waiting; only "not authorised yet" (403/404) is a legitimate reason to
  keep the loop alive. The user re-runs the command.
- **Interval policy deliberately diverges from codex.** Upstream
  serde-defaults a missing `interval` to 0 and its poll loop
  `Duration::from_secs(interval).min(remaining)` then busy-loops the token
  endpoint for up to 15 minutes. Kitty's posture: missing field → 5 s
  default; `interval` ≤ 0 or unparseable → typed
  `OAuthDeviceCodeIntervalError` (fail fast); sleep floor `max(interval, 1)`;
  final sleep clamped to the remaining budget. Recorded so a future reader
  does not "fix" the clamp to match upstream.
- **15-minute hard cap on polling, not `OAUTH_TIMEOUT_SECONDS`.** Codex uses
  15 minutes for the device flow because the user has to switch machines,
  open a browser, and type a code — a fundamentally slower interaction than
  clicking a link. Reusing the 300 s browser timeout here would time out
  legitimate logins.
- **The prompt is kitty-owned, printed to stderr, and carries the
  workspace-admin sentence.** Codex's `device_code_prompt` string is the
  structural base (a test fixture, not a contract); kitty adds the admin
  sentence because personal vs workspace entitlement is a question codex's
  prompt does not have to answer. Stderr matches where the browser path
  already prints its "if the browser did not open" hint. The device path
  does not print the browser path's "Opening browser for ChatGPT
  login…" line — that message is wrong for a no-browser flow.
- **No secret values in logs or exception messages.** §9's structural
  redaction guard is scoped to `kitty.bridge` loggers and does not cover
  `kitty.auth.*`, so the requirement is stated directly on the new code:
  DEBUG lines carry status codes and counts only, and OAuth error bodies
  are redacted before landing in an exception message (some token-endpoint
  errors echo request parameters verbatim).
- **No change to the token shape or persistence layer.** Both flows produce
  the same `OAuthSession`; the credential store, the `auth_ref` indirection,
  and the `openai_subscription` adapter are untouched. This is the reason
  the change is confined to `kitty/auth/` + CLI plumbing.

### 14.5 Verification

- L1 unit tests for `is_headless_environment` (table-driven over
  monkeypatched `os.environ`).
- L1 unit tests for the device-response parsers (interval as whitespace-
  padded string, `user_code`/`usercode` alias, 403/404 pending, other
  statuses fail fast) and the interval edge cases (missing → 5 s default;
  `"0"`, negative, non-numeric → `OAuthDeviceCodeIntervalError`).
- L1 redaction tests assert no device secret appears in DEBUG output.
- L2 contract test pins kitty's JSON request/response shapes from §14.2 as
  a literal fixture (kitty's contract; upstream drift needs a deliberate
  fixture update, not a test failure).
- L3 subsystem test drives `run_oauth_device_flow` end-to-end over mocked
  HTTP (aioresponses): happy path, 404-on-usercode, poll-then-success,
  poll-timeout (injected shrunken budget), transport exception mid-poll,
  partial token response → typed error, cancellation, error body
  containing the auth code → message without it, final-sleep clamp
  (monkeypatched `asyncio.sleep`), session close-if-created.
- CLI tests assert `--device-auth` / `--no-device-auth` route to the right
  path with the environment detection stubbed; the device path emits the
  device prompt and not the "Opening browser" line; `check_tty` still
  gates the command.

### 14.6 Known limits (recorded, not fixed here)

- ChatGPT **Business/Team/Enterprise** workspaces must have device-code
  authentication enabled by a workspace admin
  (`https://chatgpt.com/admin/permissions`, "Enable device code
  authentication for Codex CLI"); personal Plus/Pro accounts have it on by
  default (toggle at `https://chatgpt.com/#settings/Security` when
  disabled). This is an OpenAI-side entitlement; kitty cannot detect or
  bypass it. Surfaced in the device prompt (§14.4) and in the README.
- **Pre-existing redirect-URI/port-scan mismatch (not this ticket's scope).**
  `REDIRECT_URI` is hardcoded to `http://localhost:1455/auth/callback`
  while the callback server scans ports 1455–1459; if 1455 is occupied and
  kitty binds 1456, the browser's redirect lands on whatever holds 1455 and
  the flow times out with no hint. Recorded because §14.3's
  `--no-device-auth` tunnel fallback makes the failure reachable; the
  README's tunnel recipe carries the "port 1455 must be free" caveat.
  Fixing it (bind-then-build-URL is impossible — OpenAI registers only
  1455 for this client — so the fix would be refusing non-1455 binds with
  a clear error) belongs to a separate ticket.
- The verification URL is a **fixed string** (`{issuer}/codex/device`),
  not per-login. Codex has the same property; the one-time code is the
  per-login secret, not the URL.
- The server-issued PKCE pair means the device flow's code_verifier is
  chosen by OpenAI, not by kitty. This is codex's protocol choice, not a
  kitty one; it is what makes the polling endpoint the sole delivery
  channel for the authorization code.

## 15. Impersonated Codex CLI identity and OpenAI's client-version gates

Traces to [KBR-318](https://shelpuk.atlassian.net/browse/KBR-318). The
`openai_subscription` provider does not present itself as kitty: every
request to `chatgpt.com/backend-api/codex/responses` claims to be a Codex
CLI install, carrying the CLI's `User-Agent` and its `version` header.
OpenAI version-gates individual models on that identity — a request whose
claimed version is below a model's gate is rejected with **HTTP 400 and an
empty body**, with no hint that the version was the problem. The gate first
bit on `gpt-6-astra` (requires Codex CLI ≥ 0.153.0) while kitty claimed
`0.128.0`; the failure surfaced as a user-facing 400, not a red CI run.
Login into the provider is §14's concern; this section is the request
identity once logged in.

### 15.1 Components

| Component | Module | Responsibility |
|---|---|---|
| `CODEX_CLI_VERSION` (`kitty.codex_identity`) | **existing, re-pointed** | The single literal for the impersonated CLI version — `"0.154.0"` as of v1.18.0. Every version field kitty sends OpenAI derives from it. |
| `build_codex_user_agent` (`kitty.codex_identity`) | **existing** | Builds `codex_cli_rs/{version} ({os} {release}; {machine})` from the constant. Also consumed by the OAuth token leg's `token_request_headers` (`kitty.auth.oauth_session`) — a bump moves that identity too, beyond this section's logged-in-request scope. |
| `OpenAISubscriptionAdapter._build_codex_headers` (`kitty.providers.openai_subscription`) | **existing** | Sends `version: CODEX_CLI_VERSION` on every Codex-backend request; the UA rides alongside it. |
| `TestTheIdentityMeetsOpenAisClientGates` (`tests/test_upstream_identity_consistency.py`) | **new** | The tripwire: asserts the parsed constant stays ≥ the lowest known client gate (`(0, 153, 0)`, Astra's), with the bump instructions carried in the assertion message. |

### 15.2 Decisions, and why

- **One literal, two derived fields, never a second copy.** The `version`
  header and the `User-Agent` both read the module constant, so a bump
  moves every consumer together. The KBR-8 two-source defect — two
  independently-editable version literals that can silently disagree — is
  pinned shut: a single-source test asserts the version literal appears
  exactly once in `src/`, and header ⇄ constant ⇄ UA agreement tests
  assert both fields derive from the module (§15.3).
- **The guard is a floor, not an early-warning system.** It fails when the
  constant drops below the lowest *known* gate (a regression or a careless
  edit). It cannot detect a *new* gate raise: both sides of the comparison
  are literals in this repository, and live upstream probing is banned for
  `tests/`. A future gate raise is discovered the way this one was — a
  user-facing failure traced back to the identity — and the deliberate
  response is: bump `CODEX_CLI_VERSION`, raise the floor in the test, cite
  the new gate's provenance in the test docstring.
- **The constant is living by design.** OpenAI gates models by client
  version and rotates the list independently of kitty releases; each
  future gate needs a conscious bump. The tripwire exists to make that a
  test failure with instructions rather than a customer's 400. The
  version is also pinned to never be kitty's own release train
  (`kitty.__version__ != CODEX_CLI_VERSION`), because the two numbers
  move on independent schedules.
- **Why 0.154.0 specifically.** ≥ 0.153.0 satisfies the Astra gate;
  0.154.0 is the value the fix was validated against empirically (the
  identical request rejected at `0.128.0` succeeded end-to-end at
  `0.154.0`). The then-current stable (0.162.1) was deliberately *not*
  adopted: only a version validated against the real backend ships, not
  merely the newest release — the same question the next gate raise will
  face.

### 15.3 Verification

- L1 module tests pin the UA format (version + running platform) and that
  the impersonated version is not kitty's release train
  (`tests/test_codex_identity.py`).
- L1 consistency tests pin `version` header ⇄ constant ⇄ UA agreement
  (`tests/providers/test_openai_subscription.py::TestCodexHeaders`).
- L1 single-source test asserts the version literal appears exactly once
  in `src/` (`tests/test_oauth_leg_identity.py`) — the KBR-8 regression
  pin.
- L1 tripwire: the floor guard (§15.1), watched red at `0.128.0` and
  green at `0.154.0` — a development-time observation recorded in the
  KBR-318 ticket, not a committed artifact.

### 15.4 Known limits (recorded, not fixed here)

- **A new gate raise is invisible to CI until a user hits it.** This is a
  consequence of the no-live-probing rule, accepted deliberately: the
  discovery channel is a user-facing 400 with an empty body, and the
  response is the bump-and-raise-floor procedure from §15.2.
- **The gate value's provenance is community documentation**, not an
  official API reference ("Codex && Work GPT-6-ASTRA…" ChatGPT community
  thread; confirmed empirically). It is cited in the tripwire's
  docstring so the next maintainer can re-verify before trusting it.
- **Gate raises are expected to recur.** Recorded so a future reader does
  not mistake the recurrence for a defect in the design: the constant is
  a deliberate, periodically-bumped value, not a set-and-forget one.

---

## 16. Anthropic subscription relay

Traces to [KBR-319](https://shelpuk.atlassian.net/browse/KBR-319). This is a Claude Code
subscription feature, not API-key Anthropic support: the subscription request must remain Claude
Code's request while Kitty supplies local routing and the established balancing safety controls.

### 16.1 The contract

`anthropic_subscription` is a distinct, tokenless profile type. During profile setup, editing, and
`kitty doctor` diagnostics (both the checklist and single-profile paths), Kitty calls
`claude auth status --json` and accepts only a JSON object whose `loggedIn` is `true`,
whose `authMethod` is `"claude.ai"`, and whose `subscriptionType` is a non-empty string. `email`
is optional. These four names are the complete allowlist: unknown values, command stdout/stderr, and
the decoded object are never logged, stored, or displayed. The profile retains a generated UUIDv4
`auth_ref` only for schema compatibility; no CredentialStore value exists for it, and subscription
launch paths bypass credential resolution and validation. Claude Code continues to own storage and
refresh, which keeps macOS Keychain and non-macOS credential-file differences outside Kitty's
credential model.

A direct subscription profile may run only under `kitty … claude`. Codex, Gemini, Kilo, default
profile-first non-Claude routing, and generic bridge mode reject it before credential lookup,
egress, bridge creation, or child spawn. A subscription Claude launch sets no `ANTHROPIC_API_KEY`
and no `ANTHROPIC_AUTH_TOKEN` (and clears both from the child environment; the session settings file
additionally pins both empty, because Claude Code merges `--settings` per variable against user
scope and an omitted key would let a user-scope value — direct Claude Code API use — stand);
API-key profile launches keep setting them. The same rule governs a Claude-launched balancing
pool: if the pool contains an `anthropic_subscription` member, `launch_async` builds the spawn
configuration from the first such member's profile — paired with that member's own (empty)
credential — instead of the launch (first-member) profile, so the child presents Claude Code's
own OAuth identity and the relay forwards real session credentials. Pre-fix, the child carried
the first member's real stored API key plus the synthetic placeholder token, and the relay
forwarded both to Anthropic. When the first member is a non-Anthropic provider, the key is one
Anthropic never issued: Anthropic answered 401 and parked the member for 900 s — the failure this
change was raised from. When the first member is itself an `anthropic` API-key profile, the
forwarded key is one Anthropic did issue, so no 401 follows from it: the request may be accepted
and the subscription member's share billed to that key — a silent variant of the same defect
(which credential Anthropic honours when the child presents both a key and the placeholder token
is not established here). The switch is gated on the Claude
adapter (the same condition that arms the relay), so no other launcher can pick up a subscription
profile even if a caller hands it one. The child's credentials never authenticate a
non-subscription member — the translated path builds upstream headers from each member's own
resolved key — so this is purely an identity switch. One accepted side effect, deliberate rather
than cosmetic: `ANTHROPIC_MODEL` and the default-model aliases now hint the subscription member's
model. On the translated path the hint is inert — the bridge rewrites the request model to the
selected backend's — but the raw relay posts the child's bytes unchanged and never normalizes the
model, so on that path the hint is what determines the model Anthropic receives; that is exactly
why it had to move. A pool that lists several `anthropic_subscription` members hints the first
one's model: those members share the one session identity, and on the raw path the model travels
inside the forwarded bytes, so a later member's differing model cannot be honoured — pool
subscription members should agree on a model. The trade's known failure mode: a mixed pool now
rests on a live Claude Code
session. With an expired or absent session the child either fails with Claude Code's own login
error — in which case no request reaches the bridge at all — or its token-less request 401s at
Anthropic and parks the subscription member as `auth` —
the terminal 503 then speaks the credential-rejection headline, whose remediation is
`claude auth login`, not a key change (pre-fix, the pool's API-key members still served most
requests). The park does not end when the user signs in again: an `auth` park runs a fixed 900 s
cooldown that only expiry clears, so after `claude auth login` the member rejoins the pool only
when that cooldown lapses, and the API-key members keep serving until then. Recording a
subscription-aware headline for that terminal is a deliberate non-goal
here. For a non-Claude balancing launch, Kitty excludes
subscription members before it constructs backend state; remaining members keep their configured
order and backup flags. No remaining member is a configuration error, not a fallback to an
unapproved provider. A Claude-launched mixed pool may retry or fail over a raw subscription request
only to another subscription member, so that request never crosses into an API-key translation path.
A BridgeServer guard preserves this boundary for embedded callers.

For an eligible subscription backend on `/v1/messages`, the bridge reads the inbound JSON bytes
once, validates only enough to return the normal invalid-JSON response, then posts those same bytes
using `data=`. It does not call model normalization, a translator, compaction, truncation, provider
normalization, synthetic upstream authentication, or token refresh. End-to-end incoming headers —
including Claude Code's OAuth `Authorization`, `anthropic-beta`, `anthropic-version`, and any
incoming `x-api-key`, retained when Claude Code sent it — pass through untouched; only synthetic
injection is forbidden, and Kitty never injects `x-api-key`. Authority/Host,
hop-by-hop headers, and HTTP length/framing are the sole exceptions: they cannot survive the
localhost-to-Anthropic hop and aiohttp regenerates them. The raw relay is not available through Chat
Completions, Responses, or Gemini ingress.

Raw relay failures retain status, a bounded response body, and parsed `Retry-After` before entering
the existing backend-health state. Rate limits use that hint when present; OAuth failures park as
`auth`; plan denials park as `entitlement`; transport and other upstream failures use their existing
kinds. Non-streaming and pre-emission SSE failures share this classifier. A stalled upstream is the
one classified non-parking failure: before the first byte reaches the client, the relay answers the
translated Messages streaming path's 504 `api_error` envelope, parking no backend and failing over
to none — one timeout semantics for both stream modes. This deliberately diverges from the
translated path's non-streaming arms (a 500 envelope single-backend; park-and-failover when
balancing): the equivalence to mirror is the streaming arm's, and a stall after emission closes the
stream per the established post-emission rule rather than answering 504. Transport blips are not a
divergence: the relay rides out the same per-request `TransportGrace` window every translated route
honours — same-member retries while it lasts, park-and-fail-over once it is spent, and in
single-backend mode the same grace before the handler's 500 envelope. The existing selection,
failover, all-unhealthy response, healthz/stats, and recovery-hold paths remain the one system for
every balancing member.

### 16.2 Decisions, and why

- **Ask `claude auth status --json`, not Claude Code's private credential storage.** The command
  reports the supported cross-platform status without Kitty reading, storing, or accidentally
  logging OAuth material. The strict four-field reader allowlist prevents a future Claude Code
  status addition from becoming a new secret-display/storage channel. Parsing
  `~/.claude/.credentials.json` would fail on a Keychain-backed macOS session and couple Kitty to
  an implementation detail Claude Code may change.
- **Retain a UUID with no credential record.** The existing schema and profile-store contract use a
  UUIDv4 `auth_ref`; making it optional would widen every profile consumer. A provider-specific
  launch bypass keeps compatibility while establishing the invariant that this UUID cannot be used
  as an API key, upstream header, or child environment value.
- **A separate provider type, not an `anthropic` option.** API-key Anthropic constructs its own
  headers and can translate requests; subscription traffic must preserve Claude Code's identity and
  bytes. Combining them would make a security-relevant distinction implicit.
- **Byte-preserve the body and every end-to-end header.** Decoding/re-serializing JSON changes wire
  bytes even when values are equal. Rebuilding headers risks dropping OAuth opt-ins. `x-api-key` is
  retained if Claude Code sent it; only synthetic injection is forbidden. Routing and framing fields
  are excluded only because they become invalid at the new TCP destination.
- **Filter a balancing pool before all startup work.** Reading excluded credentials or using an
  excluded first member for the spawn configuration would violate the Claude-only policy even if
  random selection later skipped it.
- **Spawn a mixed pool from its subscription member's profile.** One child process presents one
  credential set, and the raw relay forwards exactly what the child sent: a synthetic
  `ANTHROPIC_AUTH_TOKEN` and a member API key would reach Anthropic, where a key Anthropic never
  issued is rejected with 401 — parking the member for 900 s, commonly with no other subscription
  member in the pool to fail over to — a 503 for the client while eight healthy members sat
  unused (an `anthropic` API-key first member instead yields a key Anthropic accepts, so the
  defect there is silent over-billing rather than a 503; §16.1 records both). The alternative
  (injecting the stored credential) is unavailable by construction: Kitty never stores or
  refreshes OAuth material, so Claude Code's own session is the only identity the subscription
  member can present. Building the spawn config from the subscription member therefore keeps one
  rule — "a subscription launch presents Claude Code's identity" — instead of a second,
  pool-specific auth path in the relay.
- **Keep raw subscription attempts inside their own balancing subset.** The raw OAuth request has no
  permitted translation/authentication fallback. Allowing it to fail over to an ordinary member
  would silently abandon the byte-exact subscription contract, so a mixed Claude pool can only
  choose another subscription member for that request.
- **Reuse the health state, not a subscription-specific circuit breaker.** Two circuits would diverge
  in cooldown, failover, observability, and recovery semantics for otherwise equivalent members.

### 16.3 Verification

- L1 tests cover status parsing (including unknown/secret-shaped extra fields and other spawn
  failures), provider/launcher eligibility, profile menus, and failure kinds.
- L1 tests pin the doctor session checks in both execution paths (the credential store is never
  consulted for a subscription profile) and the edit flow's verify-only session re-check.
- L2 tests prove provider/schema/menu registration and that Kitty profile/credential output never
  stores OAuth token material or a subscription credential record.
- L3 aiohttp tests compare captured raw body bytes and headers for normal and SSE Messages relays,
  then exercise subscription-only cooldown/failover, 429 retry hints, entitlement, recovery
  through BridgeServer, and the pre-emission timeout's 504/no-park/no-failover semantics.
- The step records are `.system_design/steps/kbr319_subscription_profile.md`,
  `kbr319_launcher_eligibility.md`, `kbr319_exact_messages_relay.md`,
  `kbr319_subscription_health.md`, and `balancing_subscription_spawn_identity.md`; each receives
  implementation notes when its stage lands.


## 17. Model context-window resolution

### 17.1 Components

- `src/kitty/providers/model_context.py` — the two resolution entry points,
  `get_model_context_tokens` (single backend) and
  `get_balancing_min_context_tokens` (pool budget = the smallest member
  window). Both catalogs and the default live here.
- `src/kitty/providers/model_context_overrides.json` — the git-tracked pin
  catalog: model id → verified context window in tokens — cloud-native ids
  for most models, plus the OpenRouter-dialect `minimax-m3.1-flash:free`
  (§17.3 D3).
- `src/kitty/providers/model_metadata.json` — the OpenRouter-generated
  table, refreshed wholesale by `scripts/script_update_model_metadata_table.py`
  (needs `OPENROUTER_API_KEY`; the script never writes the pin catalog).
- `src/kitty/providers/model_context_sync.py` — fetches the pin catalog from
  the repo's raw GitHub URL at most once per 24 h and caches it at
  `REMOTE_OVERRIDES_CACHE_PATH`; a valid, collision-free cache replaces the
  packaged catalog wholesale, so a merged pin reaches existing installs
  without a release.

### 17.2 The precedence chain and the matcher

`get_model_context_tokens` resolves in order: pin catalog →
`provider_config["context_window"]` → metadata table →
`DEFAULT_CONTEXT_TOKENS` (200,000). Both catalogs share one matcher
(`_match_catalog` / `_resolve_catalog`): exact lowercase id, a
`/`-delimited suffix in either direction (`gpt-4o` finds `openai/gpt-4o`;
`z-ai/glm-5.2` finds `glm-5.2`), then one first-segment strip retry. An
ambiguous match is terminal, never retried; a **synced** catalog whose keys
share a final segment is rejected wholesale rather than loaded — the packaged
catalog is loaded as shipped, and CI holds its tail uniqueness instead.

The budget feeds `BridgeServer._get_max_context_chars` (per-request
request-size budget, chars = tokens × 4, capped at `_MAX_REQUEST_CHARS`,
driving compaction and oversized-request handling) and the launcher's
`build_spawn_config` (the window the agent is told about).

### 17.3 Decisions, and why

| ID  | Decision                                                                                                                                                                             | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1  | A pin catalog outranks both the metadata table and a hand-written `provider_config["context_window"]`, with the shadowed setting announced at INFO (`_log_shadowed_context_window`)   | OpenRouter's metadata drifts on refresh, and operator-written windows go stale silently — KBR-151 sized a conversation from a window the model does not have. The pin is the reviewed, source-verified value; INFO (not WARNING) because the precedence is deliberate visibility, not an alarm (KBR-170).                                                                                                                                                                                                                                                                                                                                                                                                                 |
| D2  | Pin values follow the per-model verified upstream value, not a power-of-two convention                                                                                               | `MiniMax-M3` is pinned 1,048,576 because OpenRouter reports 1,048,576 for it; `MiniMax-M3.1-Flash` and its `:free` variant are pinned 1,000,000 because MiniMax's API overview and OpenRouter both report 1,000,000 for that model (verified 2026-10-21). Rounding either way would embed a wrong number.                                                                                                                                                                                                                                                                                                                                                                                                               |
| D3  | The OpenRouter `:free` spelling gets its own pin key (`minimax-m3.1-flash:free`) instead of relying on the metadata table                                                            | OpenRouter serves M3.1 Flash only as `minimax/minimax-m3.1-flash:free`; the matcher's suffix rule is `/`-delimited and never crosses the `:free` suffix, and the packaged metadata snapshot predates the model, so without the key the OpenRouter path resolves the 200k default and compacts ~5× too early. A metadata refresh would sweep in dozens of unrelated entries behind an API key for one row of value; the owner chose the two-key pin (2026-10-21). Cost, accepted: until the snapshot is refreshed the `:free` pin is the only source for that spelling, so removing it must be accompanied by a refresh — the pinning test makes the pin's presence explicit either way.                                            |
| D4  | Pins use the bare cloud-native id, not the OpenRouter `vendor/id` form                                                                                                               | The bare key resolves exact matches (Anthropic-dialect paths: custom endpoints, coding plans) and vendor-prefixed queries alike through the suffix rule, so one row serves every path that names the model natively; a prefixed key would serve only the one dialect. The `:free` suffix is part of OpenRouter's id, not a vendor prefix, so it cannot be stripped by the first-segment retry — it stays in the key.                                                                                                                                                                                                                                                                                                        |

### 17.4 Verification

- `tests/test_model_context.py` — synthetic-data precedence, matcher,
  collision-rejection and remote-cache tests (L1).
- `tests/test_model_context_packaged_catalog.py` — the real packaged
  catalogs: `EXPECTED_CATALOG` equality with the shipped file, the
  provider-variant matrix, vendor-prefixed and `:free` suffix spellings, and
  tail uniqueness over both catalogs (L2). Every `EXPECTED_CATALOG` entry
  carries its verified sources in the module docstring with a date.
- `tests/test_model_context_sync.py` — remote-cache replacement, TTL, and
  failure fall-through to the packaged catalog.

### 17.5 Known limits (recorded, not fixed here)

- A model absent from both catalogs resolves 200,000; the resolver logs the
  fall-back once per model (`_log_default_fallback`) rather than per request.
- The metadata snapshot is refreshed weekly by `model-metadata.yml` (which
  opens a pull request); between refreshes new OpenRouter-only models
  (including new `:free` variants) resolve the default unless pinned.
- The runtime sync fetches over the network; a machine that never reaches
  GitHub runs on the packaged catalog indefinitely.
