---
id: kbr87_keyring_backend_contract
depends_on: []
---

# KBR-87 — T-G12: keyring backend resolution contract (+ FileBackend corruption disambiguation)

Ticket: [KBR-87](https://shelpuk.atlassian.net/browse/KBR-87). Plan: `REQUIREMENTS.md` at
`.requirements/20260918T223610Z_kbr_87_keyring_backend_contract/`; design record:
`SYSTEM_DESIGN.md` §11; dependency row: `TEST_SUITE.md` §6.2.4 (`keyring`).

## Why

`pyproject.toml` declares `keyring>=23.0` and `KeyringBackend` hands it every credential
read and write blind — nothing observes which backend keyring resolved. Resolution varies
by platform **by design** (macOS Keychain, Windows Credential Manager, Linux Secret
Service, `fail.Keyring` fallback), so §6.2.4's no-stable-neighbour rule applies: pin the
resolution *mechanics* and record the choice, rather than a version floor alone or an
unreachable native-service assertion.

The same ticket carries two scope additions from its Jira comments: the KBR-154 diagnostic
family's `FileBackend.get()` gap (corrupt stored value indistinguishable from absent —
both surfaced as "no API key for profile X"), and the §6.2.4 aiohttp row's missing
`Landed (KBR-84)` annotation (its botocore twin was already fixed on `main` by KBR-85).

## What landed

- `tests/test_keyring_backend_contract.py` (L2): the five pins — public-API delegation to
  `keyring.get_keyring()`, `PYTHON_KEYRING_BACKEND` selection via `keyring.core.load_env()`,
  resolution always lands on a `keyring.backends.*` class, per-platform native class behind
  availability guards, `PasswordDeleteError` ⊂ `KeyringError`. Each pin's docstring names
  its own falsification. Self-guard counts sync behavioural tests (the aiohttp twin's
  coroutine count does not transfer).
- `FileBackend.get()` raises `CredentialError` (chained, ref named) for present-but-
  undecodable values via `except (ValueError, TypeError)` and `b64decode(validate=True)`;
  `None` now means absent only (an explicit JSON `null` is the absent spelling). Receivers
  at every site that produced a clean message before: `egress_store.resolve_egress`
  (re-raises the documented `ValueError`, preserving the recovery-command carve-out),
  `bridge_runner.py` both branches (added in review round 1 — the first survey grepped
  `cli/` only), `cli/launcher.py`, three `cli/main.py` sites,
  `profile_cmd._find_reusable_auth_ref` (skip — re-entry is the recovery), both
  `doctor_cmd` checks (report — the diagnostic tool must not crash on its subject).
- §6.2.4: aiohttp row annotated; keyring row annotated with the mechanism sentence; G11's
  keyring cell updated.

## Gotchas (measured, for the next reader)

- keyring's backend discovery enumerates **every loaded subclass of
  `keyring.backend.KeyringBackend`** — a test-defined backend at priority ≥ 1 wins
  `init_backend()`'s `max(by_priority)` for the whole process the moment its module is
  imported. The contract's `_MemoryKeyring` therefore sits at priority -2: invisible to
  auto-detection, reachable only through explicit `set_keyring`.
- `b64decode`'s non-ASCII rejection is a plain `ValueError`, not `binascii.Error` — the
  narrow-looking `(ValueError, TypeError)` tuple is the complete one.
- `keyring.backends.macOS.Keyring.priority` raises environment-specific errors
  (`PermissionError` on hardened runners, `RuntimeError` without the Security API) — the
  availability guard catches `Exception`, because any failure means "not available", which
  is the only fact the guard needs.
- The bare `keyring>=23.0` dependency does not carry pyobjc, so the macOS per-platform arm
  skips on the macOS CI leg; only the Windows arm genuinely asserts (pywin32-ctypes is a
  base dependency). Recorded in the contract's docstring so a permanently-skipping arm is
  not mistaken for coverage.
