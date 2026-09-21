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
  `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` shadow the user's claude.ai OAuth login.

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
  `/v1/messages`' custom segment (parse → `translate_response` → Messages
  events) still has no emptiness gate, so a content-less completion there
  delivers a content-less Messages turn — the smaller-class sibling defect,
  deferred. KBR-254's sibling-test stubs were corrected to what real
  adapters emit (`_fake_hello_cc_stream`; the Responses-SSE stub pinned what
  the branch accepted, not what Bedrock writes). Tests:
  `tests/bridge/test_responses_custom_transport_empty_hold.py`,
  `tests/bridge/test_gemini_custom_transport_empty_hold.py` (the KBR-287
  harness shape; content oracles parsed from route-protocol events, never
  raw substrings — the KBR-249 vacuous-oracle trap).
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
