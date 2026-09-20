---
id: kbr291_file_backend_f37_residuals
depends_on: []
---

# KBR-291 — FileBackend F37 residuals: file-level corruption contract

Ticket: [KBR-291](https://shelpuk.atlassian.net/browse/KBR-291) —
"FileBackend F37 residuals: invalid-UTF-8 file crashes the launch path;
valid-JSON-non-dict silently resets (and the next set destroys the
original)". Plan: `REQUIREMENTS.md` at
`.requirements/20260920T215552Z_file_backend_f37_residuals/`.

## Why

KBR-87 closed the per-ref half of the credential corruption contract
(`CredentialError` at `FileBackend.get`'s boundary for present-but-
undecodable values, with the receiver map producing the clean message
at every call site) and explicitly recorded two file-level residuals
in `SYSTEM_DESIGN.md` §11.1 as "not fixed" rather than silently absorbing
them. Both residuals live in the same `_read_raw` site:

- **Shape (b)** — invalid-UTF-8 bytes. `read_text(encoding="utf-8")` raises
  `UnicodeDecodeError` (⊂ `ValueError`, not `OSError`); the outer
  `except (FileNotFoundError, json.JSONDecodeError, OSError)` does not
  catch it; the error propagates through every receiver, producing raw
  tracebacks on the launch path, in the wizard, and in the doctor.
- **Shape (a)** — valid JSON that parses to a non-dict (top-level list,
  string, number, bool, null). The existing `isinstance(result, dict)`
  branch in `_read_raw` returns `{}` silently — no backup, no log. The
  next `set` overwrites the file with no trace of the original.

Both are pre-existing KBR-154 residuals (the diagnostic family the
contract exists to prevent) and KBR-87's `SYSTEM_DESIGN.md` §11.1
honesty document already named them.

## What landed

- `src/kitty/credentials/file_backend.py` — `_read_raw` restructured into
  three concerns: (a) read bytes (`UnicodeDecodeError` now backs up via
  `os.replace` and raises `CredentialError` chained from the
  `UnicodeDecodeError`); (b) parse JSON (F37 path verbatim); (c) validate
  shape (`isinstance(result, dict)` — non-dict backs up via `os.replace`
  and raises `CredentialError` with no chain because the JSON parsed
  cleanly). The F37 invalid-JSON path is unchanged. Both new shapes
  emit a CRITICAL line naming the file path and the backup path.
  Shape (a) does **not** write `{}` after the backup (D3's no-write-
  empty pattern extends to shape a — see the gotcha below);
  `_read_raw_for_write()` private helper swallows the file-level
  `CredentialError` for the write path when the backup succeeded
  (`self._path` is absent), but propagates when the backup failed
  (`self._path` still holds the damaged original — the read-only-mount
  case). `set`/`delete` proceed from `{}` on the success branch; on
  the failure branch, the recovery command sees the honest message
  rather than silently overwriting the user's data with no backup
  anywhere. `FileBackend.get` lets the exception propagate to the
  KBR-87 receiver map.
- `tests/test_credential_store.py::TestFileLevelCorruption` (L1) —
  shape (b) `CredentialError` + chain pin + backup + CRITICAL log
  content (path + backup path) + `set` after damage; shape (a)
  parametrised over list/string/number/bool/null with the same pins
  plus the no-chain pin (the JSON parsed cleanly, so `__cause__` is
  `None`); `delete` after damage parametrised over both shapes; F37
  regression pin (`test_f37_invalid_json_path_is_unchanged`).
- `SYSTEM_DESIGN.md` §11.1 — drop the residuals paragraph; describe the
  KBR-291 extension. §11.2 — split the per-ref / file-level halves and
  name shape (a) and shape (b). §11.2 — new paragraph "Why the write path
  forgives file-level damage" recording D6. §11.4 — add the
  `TestFileLevelCorruption` row to the verification list.

## Gotchas (measured, for the next reader)

- `UnicodeDecodeError` chains **shape (b)** but **shape (a)** does not.
  Shape (a)'s JSON parsed cleanly, so there is no underlying exception
  to preserve — `raise CredentialError(...)` without `from` is correct,
  and the regression pin is `test_error_does_not_chain_for_shape_a`.
- `os.replace` is bytes-level, so the shape (b) backup preserves the
  original bytes verbatim without re-encoding — recreating `{}` after
  shape (b) damage adds nothing (the bytes can't be decoded anyway).
  Shape (a)'s bytes are valid UTF-8, but the file is also not written
  `{}` after the backup: the asymmetric design (F37 invalid-JSON writes
  `{}` because it returns `{}`; shapes a/b do not write `{}` because
  they raise) lets `_read_raw_for_write`'s `self._path.exists()` guard
  distinguish "backup succeeded" from "backup failed" without inspecting
  the exception message. If a future refactor re-adds `_write_raw({})`
  to shape (a), the guard will misfire on the success branch and the
  read-only-mount regression pin
  (`test_write_path_propagates_when_backup_could_not_be_made`) will
  fail.
- The write-path forgiveness (`_read_raw_for_write`) is the load-bearing
  reason the recovery command (`kitty setup`, `kitty egress`, profile
  wizard) stays reachable: every one of those commands reaches a
  `cred_store.set(...)` call. If `set` propagated file-level
  `CredentialError`, the wizard would crash on the very write the user
  is performing — re-creating the KBR-154 diagnostic failure this
  contract exists to eliminate.
- `pytest.raises(match=...)` uses `re.search`, so a regex-special path
  (e.g., on Windows: `C:\…`) would need `re.escape(...)`. The test file
  uses tmp paths under `/tmp/...`, so this is not exercised today.
- The receiver map (eight call-site groups, ten handler clauses across
  six files — counted as the ticket counts) is unchanged: the new
  shapes raise `CredentialError`, which the KBR-87 receivers already
  handle. The seven wizard `cred_store.set` sites are write-path and
  intentionally outside the receiver map.
- `credentials/file_backend.py` is not in `pyproject.toml`'s
  `only_mutate` list and not in `mutmut_scope.TARGET_GROUPS` — a
  standing project decision (the L1 tests assert observable behaviour
  at the boundary, which is what `mutmut` would score anyway if scope
  were widened). Recorded in REQUIREMENTS.md D5.

## Follow-ups (not this ticket)

- The "eight sites" vs "ten handler clauses" count is reconciled in the
  doc but the canonical phrasing across `SYSTEM_DESIGN.md` §11.2 and
  the KBR-87 step file is "ten handler clauses across six files" — the
  ticket text's "eight" is a per-group count, the step file's "ten" is
  a per-`except` count. If the ticket is re-read in isolation the
  inconsistency is visible; both phrasings are correct, only the
  counting rule differs.
- `mutmut_scope` does not track `credentials/file_backend.py`. If the
  project ever decides to widen `only_mutate`, this ticket's
  `TestFileLevelCorruption` is the seed for the per-test mutation
  pin.