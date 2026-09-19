---
id: kbr268_kilo_crash_recovery
depends_on: [KBR-93, KBR-262]
---

# KBR-268 — Kilo launcher crash recovery (backup + `kitty cleanup` arm)

Jira: [KBR-268](https://shelpuk.atlassian.net/browse/KBR-268). Parent: KBR-123.
Harness: `TEST_SUITE.md` §6.3.2 (CLI with real filesystem and processes).

Depends on:

* **KBR-93** (Done, PR #162) — established the Claude crash-recovery pattern
  this ticket mirrors: the backup trio, the ownership-checked restore, and
  the phase-1 exact restore in `kitty cleanup`.
* **KBR-262** (Done, PR #186) — established Kilo's byte-identity contract
  (`newline=""` on capture and restore) that the backup write and both
  restore legs must carry over.

## What

Give the Kilo launcher the crash-recovery tier Claude already has. Today a
`SIGKILL` mid-`kitty kilo` leaves `~/.config/kilo/kilo.json` routed at a dead
bridge (kitty provider block + one-time API key + `model: kitty/<m>`)
indefinitely. Three changes:

1. `KiloAdapter.prepare_launch` persists the byte-exact original to
   `~/.config/kitty/kilo-config-backup.json` before patching — **only** when
   the captured original is clean (no kitty markers): the clean-capture rule
   that prevents the crash-then-relaunch clobbering of the true backup.
2. `KiloAdapter.cleanup_launch` restores — and deletes the backup — only when
   this session still owns the file (captured original clean, current file
   readable and marker-bearing); every other combination leaves file and
   backup alone.
3. `kitty cleanup` grows a Kilo arm, `run_kilo_cleanup` (called alongside
   `run_cleanup` from `_run_cleanup`): backup present → exact restore when
   the current config carries kitty markers or is missing/unreadable; stale
   backup → delete it, config untouched; no backup → no-op. Restore I/O
   failures print one `Error:` line and exit 1.

## Why

* **Backup-only, no heuristic strip arm** (product owner, 2026-09-18): Claude's
  heuristic can restore fully because kitty only *adds* env keys it can later
  remove. Kilo is structurally different — kitty *overwrites* the shared
  `model` key — so a backupless heuristic could remove `provider.kitty` but
  could never restore the user's pre-session model. A partial repair was
  rejected; minimal-fix precedent KBR-260.
* **Backup path `~/.config/kitty/kilo-config-backup.json`** (product owner):
  Claude's `_DEFAULT_BACKUP_PATH` precedent — all kitty-owned state in one
  directory, never a foreign file inside Kilo's own config dir.
* **Clean-capture rule** (system-design review, blocker): without it the
  crash → relaunch sequence captures the *patched* file as the "original" and
  silently destroys the only good backup. The backup file therefore never
  contains kitty markers.
* **Marker-presence ownership, not per-session injected values** (design
  review): the whole region kitty touches is kitty-owned, so "current file
  carries kitty markers" is the ownership signal; this avoids introducing a
  `str`-subclass prepare contract to a fresh adapter. The one case per-session
  precision would add — restoring over a same-config sibling session's patch —
  is bounded: that patch is equally dead, and the true original wins either
  way.
* **Opposite verdicts on an unreadable current file, recorded**: a live
  session's `cleanup_launch` must never guess, so it leaves an unreadable file
  alone; `kitty cleanup` is an explicit repair request, so it lets the exact
  backup win. Both are Claude parity.

## Accepted residuals

* A crash of a **from-scratch** session (no pre-existing `kilo.json`) leaves
  the kitty-only file behind; cleanup cannot distinguish it from a
  user-authored file without a backup.
* Damage from **pre-fix** kitty versions, or after a user deletes the backup
  file by hand, needs a manual fix.

## Status

Implemented (2026-09-18); PR open, awaiting CI + review.

## Implementation notes

### What landed (2026-09-18)

1. `src/kitty/launchers/kilo.py` — `_DEFAULT_BACKUP_PATH`
   (`~/.config/kitty/kilo-config-backup.json`), the
   `save/load/delete_kilo_config_backup` trio (lazy default resolution,
   `newline=""`, reusing `claude._atomic_write_text` — single-sourced like
   `cleanup_cmd` already does), and the `_kilo_kitty_values_present` detector
   (loopback `provider.kitty.baseURL` via the single-sourced
   `claude._is_local_kitty_url`, or `kitty/` model prefix; remote-URL
   provider alone does not count).
2. `KiloAdapter.prepare_launch` — clean-capture rule: backup written only
   when the normalised parsed original (`{}` for malformed/non-dict) carries
   no markers; marker-bearing capture logs the `kitty cleanup` warning and
   skips the write. Detector runs on the already-normalised dict — never a
   fresh `json.loads`, whose `ValueError` would escape the orchestrator's
   `OSError`-only guard.
3. `KiloAdapter.cleanup_launch` — ownership matrix: polluted captured
   original → leave file+backup, warn; clean original + marker-bearing
   readable current → restore (`newline=""`) then delete backup; clean
   original + clean/missing/unreadable current → leave both, info log.
   Backup deletion sits **after** the successful restore write.
4. `src/kitty/cli/cleanup_cmd.py` — `_DEFAULT_KILO_CONFIG_PATH`,
   `_get_kilo_backup_path` (lazy single-source, mirrors `_get_backup_path`),
   and `run_kilo_cleanup(settings_path=...)`: backup-first read (OSError →
   `Error:` line + exit 1), missing/unreadable config counts as crashed
   (restore wins), readable marker-free config → stale backup deleted,
   config untouched; no backup → strict no-op exit 0.
5. `src/kitty/cli/main.py` `_run_cleanup` — both arms always run (Claude,
   then Kilo); `sys.exit(max(claude_code, kilo_code))`.
6. Tests — `tests/test_kilo_config.py` (+22 tests: trio, detector truth
   table, prepare backup matrix incl. relaunch guard, cleanup ownership
   matrix, CRLF leg; autouse `_DEFAULT_BACKUP_PATH` redirect keeps every
   test off the real home); `tests/cli/test_cleanup_cmd.py`
   (`TestRunKiloCleanup`, +10 tests incl. the no-backup negative control and
   the portable directory-staged unreadable backup); `tests/test_cli_main.py`
   (`_cli_run` patches both arms and yields the tuple; +1 worst-of-both-arms
   exit-code test; the four `as run` sites unpack both mocks and assert the
   Kilo arm).
7. Docs — `SYSTEM_DESIGN.md` §2.1 (new "Launcher lifecycle and crash
   recovery" section: both adapters, the contract, decisions + why),
   `TEST_SUITE.md` §6.3.2 Kilo row, this step file.

### Verification

* Touched files: 224 passed / 4 skipped (Windows-only legs).
* Full local suite: green (see PR description for the exact counts).
* `ruff check` + `ruff format --check` clean on all six touched code files.
* Line-ending audit vs merge base: `kilo.py` and `cleanup_cmd.py` remain
  100% CRLF (210→383 and 251→335 CR lines, zero LF-only lines) — no ending
  churn.
* Step-index validator: exits 1 with **only** the pre-existing
  `t_g6 → t_w8` dangling dep (red on main before this task; KBR-278 D2:
  developer-side only). No new failures.

### Known review notes

* Two pre-existing observations recorded, not fixed: `run_cleanup`'s ~45
  lines of unreachable duplicated code (`cleanup_cmd.py:208-251`) and
  Claude's dormant `save_settings_backup` writer (zero `src/` callers since
  the per-session refactor — its backup trio + phase-1 cleanup serve
  pre-per-session damage). Both out of scope per the ticket.
