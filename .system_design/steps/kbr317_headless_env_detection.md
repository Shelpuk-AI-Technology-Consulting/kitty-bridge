---
id: kbr317_headless_env_detection
depends_on: []
---

# Headless environment detection (`kitty.auth.headless`)

KBR-317, step 1 of 6. The pure predicate that decides whether the device-code
login path should be the default. Mirror of
`codex-rs/core/src/env.rs::is_headless_environment` (fetched 2026-10-14).

## Why

`kitty auth openai` currently always starts a localhost callback server. On a
remote CLI-only VM the browser is on a different machine, so the OpenAI redirect
to `http://localhost:1455/auth/callback` can never reach kitty and the flow
times out after 300 s. The fix adds a device-code path; this step provides the
headless detection that selects it automatically, so a user on an SSH session
does not have to know a flag exists.

The predicate lives in its own module (`kitty.auth.headless`), not inside
`openai_oauth`, so it stays a pure `os.environ` read with no aiohttp/asyncio
imports and is trivially unit-testable with `monkeypatch`.

## Upstream source (verified 2026-10-14)

The variable list is grounded in
`codex-rs/core/src/env.rs::is_headless_environment` from codex revision
`1fc72c647fd52e3e73d4309c3b568d4d5fe012b5` (the pinned revision cited in
issue openai/codex#9253; the file has since moved). The current-main
behaviour of preferring device-code when headless lives in
`codex-rs/cli/src/login.rs::run_login_with_device_code_fallback_to_browser`
(line 367-373) and in the TUI onboarding gate at
`codex-rs/tui/src/onboarding/auth/headless_chatgpt_login.rs`.

## To Be

```python
# kitty/auth/headless.py
def is_headless_environment() -> bool: ...
```

Returns `True` when any of:

- `CI` set and non-empty
- `SSH_CONNECTION`, `SSH_CLIENT`, or `SSH_TTY` set and non-empty
- On Linux: neither `DISPLAY` nor `WAYLAND_DISPLAY` set or non-empty

Returns `False` otherwise (e.g. a desktop with `DISPLAY=:0` and no SSH vars).

## Acceptance

- Table-driven unit test over `monkeypatch.setenv/delenv` covering each
  variable individually, combinations, empty-string vs unset, and the
  Linux-only branch.
- No dependency on `aiohttp` / `asyncio` in the module.

## Implementation notes

- `src/kitty/auth/headless.py`: `is_headless_environment()` + private
  `_env_var_set` (empty/whitespace counts as unset, mirroring codex's
  `env_var_set`). DISPLAY/Wayland branch is `sys.platform.startswith("linux")`
  — NOT `os.name == "posix"` — because macOS has no DISPLAY by default but
  has a GUI (caught in self-review before the first test run).
- `tests/auth/test_headless.py`: 16 tests, table-driven; autouse fixture
  establishes a **desktop baseline** (`DISPLAY=:0`, no SSH/CI) because the
  CI box itself is a headless Linux host — without the baseline the
  "not headless" assertions are untestable. Darwin/Windows branches pinned
  via `monkeypatch.setattr(sys, "platform", ...)`.
