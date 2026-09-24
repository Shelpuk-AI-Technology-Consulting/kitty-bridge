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
