---
id: kbr317_oauth_flow_dispatch
depends_on: [kbr317_device_code_usercode_poll]
---

# Dispatch: browser vs device flow inside `run_oauth_flow`

KBR-317, step 3 of 6. Wires the device flow into the existing entry point so
both CLI paths (auth command + profile wizard) get it without a signature
change.

## Why

There are **four** call sites of `run_oauth_for_provider` in the repo
(grepped 2026-10-14):

- `src/kitty/cli/auth_cmd.py:48` — `run_auth_openai` (the `kitty auth openai`
  command, the only surface that takes CLI flags).
- `src/kitty/cli/profile_cmd.py:206` — `_create_profile_flow` (the profile
  wizard's "OpenAI ChatGPT Plan" path).
- `src/kitty/cli/profile_cmd.py:386` — the profile-edit re-auth branch.
- `src/kitty/cli/setup_cmd.py:92` — `kitty setup`.

Only the first has a CLI-flag surface; the other three are interactive
menus. Keeping dispatch inside `run_oauth_flow` (with a `device_auth` kwarg
threaded through `run_oauth_for_provider`) means **all four** auto-detect
on a headless box with no extra wiring, and the one CLI surface gets the
flag. The selection rule lives in one place; the call graph stays
one-deep.

## To Be

`run_oauth_flow(http=None, *, device_auth: bool | None = None) -> OAuthSession`:

- `device_auth=True` → device flow.
- `device_auth=False` → browser flow.
- `device_auth=None` (default) → `is_headless_environment()` decides.

`run_oauth_for_provider(..., device_auth: bool | None = None)` and
`run_auth_openai(..., device_auth: bool | None = None)` thread the kwarg
through. Default `None` everywhere preserves the existing call-site
behaviour for the three non-CLI surfaces.

The existing browser path is untouched; the device path is the new sibling.
Two new exceptions: `OAuthDeviceCodeDisabledError` (usercode endpoint
returned 404 — mirrors codex's one-line error) and
`OAuthDeviceCodeIntervalError` (`interval` field missing-parseable as ≤0
or non-numeric — see step 2's interval policy).

The device path must NOT print the existing
`print_status("Opening browser for ChatGPT login...")` line — that message
is wrong for a no-browser flow. The device path's own prompt (step 2's
`format_device_code_prompt`) replaces it.

## Acceptance

- Unit tests: `device_auth=True` short-circuits to the device path;
  `device_auth=False` never touches the device endpoints;
  `device_auth=None` defers to `is_headless_environment` (stubbed).
- The existing browser-flow tests continue to pass unchanged.
- A new L3 test asserts the device path emits the device prompt to stderr
  and does NOT emit "Opening browser for ChatGPT login…".

## Implementation notes

- `run_oauth_flow(http=None, *, device_auth=None)`: dispatch at the top —
  `device_auth if device_auth is not None else is_headless_environment()`.
- `run_oauth_for_provider(..., *, device_auth=None)` and
  `run_auth_openai(..., *, device_auth=None)` thread the kwarg.
- The stdout `print_status("Opening browser for ChatGPT login...")` line is
  now conditional: printed only when the browser path is the predicted one
  (same predicate, re-evaluated — a pure function, so caller and dispatch
  cannot disagree). The device path's stderr prompt announces itself; no
  extra stdout line was added for it.
- The four existing browser-path tests in `test_openai_oauth.py` that call
  bare `run_oauth_flow()` now pass `device_auth=False` — on a headless CI
  box the unforced call would (correctly) take the device path and their
  browser mocks would never fire. The mock factory in
  `tests/cli/test_auth_cmd.py` gained the `device_auth` kwarg for the same
  reason.
- `tests/auth/test_openai_oauth_dispatch.py`: 5 tests pinning the routing
  with both flow functions stubbed to raise a marker naming the selected
  path.
