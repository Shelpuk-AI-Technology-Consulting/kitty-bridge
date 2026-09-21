---
id: kbr265_cp1252_decode_hardening
depends_on: []
---

# KBR-265 — Harden parent-side reads against the locale-codepage decode crash

Jira: [KBR-265](https://shelpuk.atlassian.net/browse/KBR-265). Parent: KBR-136.
Requirements: `.requirements/20260920T220620Z_kbr_265_cp1252_decode_hardening/REQUIREMENTS.md`.
Design: `SYSTEM_DESIGN.md` §3.3 (T15 — the `run_captured` read-back hardening).

Depends on: **KBR-275** (Done, PR #209) and **`dc47728`** — both already landed. KBR-275 retired
the recurring `JSONDecodeError` the original ticket hypothesised (the atomic
`.tmp` + `os.replace` shape for the launcher's console snapshot, with
`test_the_launcher_publishes_its_console_snapshot_atomically` as its structural guard). `dc47728`
retired the `state.json` race with the bounded content-aware poll in `_read_bridge_pid`. This step
addresses the **scope expansion** the owner's 2026-09-17 triage note (KBR-265 comment) identified:
the `cp1252`/UTF-8-strict reader-thread crash at the two named `text=True` sites. Nothing in this
step depends on either fix to land first; both are Done and the dependency is informational only,
hence `depends_on: []`.

## What

Pin `encoding="utf-8", errors="replace"` on two parent-side read-back sites so the parent no longer
decodes a child's output with the locale codepage in strict mode:

* `src/kitty/cli/tmux_wrap.py::run_captured` — `subprocess.run(..., text=True)` →
  `subprocess.run(..., text=True, encoding="utf-8", errors="replace")`.
* `tests/bridge/test_bridge_management.py::TestTheWindowsConsoleDetachment::test_a_background_bridge_survives_a_console_break_in_its_launching_console`
  — `Popen(..., text=True)` → `Popen(..., text=True, encoding="utf-8", errors="replace")`.

Pin the shape with two L1 tests:

* `tests/cli/test_tmux_wrap.py::test_run_captured_survives_an_undecodable_byte` —
  behavioural, red before R1 (`UnicodeDecodeError` on byte `0x90`), green after. Runs on every leg
  (byte `0x90` is undefined in cp1252 and an invalid UTF-8 continuation byte alone, so the defect
  reproduces deterministically everywhere — `cpython#105312`, zooba/eryksun).
* `tests/bridge/test_bridge_management.py::TestTheWindowsConsoleDetachment::test_the_console_break_launcher_popen_decodes_with_errors_replace`
  — structural guard, mirrors KBR-275's `test_the_launcher_publishes_its_console_snapshot_atomically`.
  Reads the kwargs of the existing `Popen` call and pins them, so every leg holds the shape even
  though only the Windows leg can run the console-break scenario.

## Why this and not the obvious alternative

The obvious alternative is `PYTHONIOENCODING=utf-8:replace` in the *child* environment. That only
solves the case where the child is a Python we own — `run_captured` runs arbitrary tmux/git
commands whose environment we do not own, and the ticket explicitly asks for the `Popen`/`run`
kwarg shape. The kwarg shape also wins on simplicity: one place per call, no environment plumbing.
