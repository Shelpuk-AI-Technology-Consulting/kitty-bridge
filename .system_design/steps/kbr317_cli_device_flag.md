---
id: kbr317_cli_device_flag
depends_on: [kbr317_oauth_flow_dispatch]
---

# CLI flags: `kitty auth openai --device-auth` / `--no-device-auth`

KBR-317, step 4 of 6. Adds the user-facing flags to the auth command and the
profile wizard.

## Why

Auto-detection covers the common case, but users who will set up an SSH
tunnel (`ssh -L 1455:localhost:1455`) need a way to force the original
browser flow, and users on a desktop that happens to have SSH vars set (e.g.
VS Code Remote) need a way to force the device flow. Codex exposes the same
escape hatch (`codex login --device-auth`); kitty mirrors it.

## To Be

- `kitty auth openai --device-auth` forces the device flow.
- `kitty auth openai --no-device-auth` forces the browser flow.
- Neither flag → auto-detect (step 3's rule).
- **The three interactive entry points** (`kitty profile` wizard,
  profile-edit re-auth, `kitty setup`) get auto-detect only; they have no
  flag surface and gain none in this change. (See step 3's enumeration.)

The flags are argparse booleans on the existing `kitty auth openai`
subcommand; no new parser framework. `_run_auth` in `main.py` parses the
extra args after the literal `openai` token and forwards the choice to
`run_auth_openai`.

## Acceptance

- CLI tests assert each flag routes to the right path with the environment
  detection stubbed.
- `kitty auth openai --help` shows both flags.
- The wizard's, profile-edit's, and `kitty setup`'s OpenAI paths continue
  to work unchanged and inherit auto-detect via the shared dispatch.

## Implementation notes

- `main.py::_run_auth` rewritten: `args[0] != "openai"` → print + `sys.exit(1)`
  (previously it printed and returned 0 — an unknown provider was not an
  error; corrected as part of the rewrite). `args[1:]` parsed by a local
  argparse with a mutually-exclusive group; `--device-auth` → True,
  `--no-device-auth` → False, neither → None (auto-detect).
- The wizard, profile-edit, and `kitty setup` paths needed **zero** changes:
  the defaulted kwarg threads auto-detect through the shared dispatch.
- `tests/cli/test_auth_device_flag.py`: 6 tests (no-flag → None, both
  flags, mutual exclusion → SystemExit(2), unknown provider → SystemExit(1),
  `--help` → SystemExit(0)). `_tty_required` is a *sync* context manager —
  the stub needed `__enter__`/`__exit__`, not the async protocol.
