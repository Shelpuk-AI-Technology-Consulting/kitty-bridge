---
id: kbr314_claude_code_tmpdir
depends_on: []
---

# KBR-314 — inject `CLAUDE_CODE_TMPDIR` from the Claude launcher

Ticket: [KBR-314](https://shelpuk.atlassian.net/browse/KBR-314) (Bug, Medium,
parent [KBR-136](https://shelpuk.atlassian.net/browse/KBR-136) Bug Fixes epic).
Requirements:
`.requirements/20260924T112935Z_kbr314_claude_code_tmpdir/REQUIREMENTS.md`.

## What

`ClaudeAdapter.build_spawn_config` adds `CLAUDE_CODE_TMPDIR` to its
`env_overrides` (value `tempfile.gettempdir()`) and the key is added to
`_SETTINGS_ENV_OVERRIDE_KEYS` (so `prepare_launch` writes it into the
per-session `--settings` file) and to `_KITTY_INJECTED_KEYS` (so `kitty
cleanup` strips stale values from a crashed legacy
`~/.claude/settings.json`). The pre-existing
`test_settings_and_cleanup_lists_are_identical` enforces the lockstep.

## Why

Claude Code's cross-session messaging daemon picks `/` as its socket dir
when neither `XDG_RUNTIME_DIR` nor `CLAUDE_CODE_TMPDIR` is set and no
user-owned temp dir is discoverable; on the affected box `/` is
`1000:1001 0755`, the ownership check rejects it, and Claude Code prints
the remediation hint at every launch. The hint itself names
`CLAUDE_CODE_TMPDIR` as the first fix; Claude Code's env-var reference
documents it as the override for internal temp files (Claude Code
appends `/claude-<uid>/` on Unix, `\claude\` on Windows). The bridge's
`ClaudeAdapter` already owns Claude Code's launch env, so this is one
entry at one seam.

## Scope

`ClaudeAdapter` only. Codex / Gemini / Kilo / OpenCode unchanged. The
launcher base surface, `SpawnConfig`, `prepare_launch` /
`cleanup_launch` / `settings_cli_args`, the per-session file's atomicity,
the `env_clear` list, and the `cli_args` list are all untouched.

## Implementation notes

_(appended after the PR lands)_

- PR #288, branch `fix/kbr-314-claude-code-tmpdir` off `origin/main` `adae476`,
  commit `3fb18f1`.
- Production delta: 2 lines in `src/kitty/launchers/claude.py` (one entry in
  `_SETTINGS_ENV_OVERRIDE_KEYS`, one entry in `build_spawn_config`'s
  `env_overrides`); 1 line in `src/kitty/cli/cleanup_cmd.py` (one entry in
  `_KITTY_INJECTED_KEYS`). The two list additions land in lockstep; the
  pre-existing `test_settings_and_cleanup_lists_are_identical` is the
  structural guarantee.
- Tests: four new L1 unit tests (R1 in `TestClaudeCodeTmpdir`, R2's two
  membership assertions in `TestInjectedKeyListsInSync`, R3 in the new
  `TestClaudeCodeTmpdirInSessionFile`, R4 in `test_cleanup_cmd.py::test_run_cleanup_strips_claude_code_tmpdir`).
  All four watched RED before the production edit; the lockstep test
  continues to pass.
- Verification on this box (uid 1001): `tempfile.gettempdir()` → `/tmp`,
  `/tmp` owner `root:root` mode `1777`. Claude Code's ownership check
  ("owned by you or root") accepts root-owned → the fix silences the
  error in the common case on every supported platform.
- Reviewer rounds: 2 system-design rounds (both Majors resolved).
  Code-reviewer and reviewer-bot verdicts pending on PR.
- Accepted residual (per Requirements doc Design Notes): a user who exports
  a hostile `TMPDIR` gets that value forwarded verbatim and Claude Code
  may still print the error. A CWD fallback (only reachable when every
  standard temp dir is unwritable) would silence the error at the cost of
  a `claude-<uid>/` directory appearing in the repo — recorded, not fixed.
- Cross-platform note: on macOS `tempfile.gettempdir()` returns a per-user
  `/var/folders/...` path; on Windows `%TEMP%` is per-user. Both pass
  Claude Code's ownership check. CI matrix covers all three.
- Follow-up owed (none filed in this PR): if Claude Code's daemon ever
  surfaces the original ownership error to its own log even when
  `CLAUDE_CODE_TMPDIR` is set, that's upstream
  `claude-code#90908`-class behaviour — Claude Code's decision about its
  own socket, not the bridge's.
