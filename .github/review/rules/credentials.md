# Rule: credentials, auth flows and profiles (`credentials/**`, `auth/**`, `profiles/**`)

One rule, because a profile's `auth_ref` is meaningless without the store it
dereferences: a change to either side has to be judged against the other. This
covers the credential store and its two backends (keyring, file), the OAuth and
PKCE flows, and the profile schema, store and resolver.

## Invariants

- **A secret is stored by reference, never in a config file.** `profiles.json`
  and `egress.json` hold opaque ids; the values live in the credential store. A
  diff that writes a key, token or password into a profile or egress document is
  a **critical** finding — those files are the ones users paste into issues.
- **A secret must not reach a log, an exception message, a `repr`, a `str`, a
  `kitty doctor` line, or the terminal.** Check `__repr__` and `__str__` on any
  type the diff adds that can hold one; the default dataclass `repr` prints
  every field, and that is how a key ends up in a traceback.
- **File-backed credentials are base64-encoded, not encrypted.** That is a
  deliberate, documented trade-off, not an oversight — do not raise it as a new
  finding on every pull request. What *is* a finding is a change that makes it
  worse: widening the file's permissions, moving it somewhere world-readable, or
  adding a second copy of the value elsewhere.
- **The keyring backend must degrade to the file backend, not to nothing.** A
  machine with no keyring service is the normal case on a headless VM. A change
  that turns an unavailable keyring into a crash, or into a silent no-op that
  loses the credential, is critical in opposite directions.

## Profiles

- **The on-disk format has no version field and no migration step.** `kitty` is
  upgraded in place over a config directory the previous version wrote. A schema
  change that makes an existing `profiles.json` fail to load is a **critical**
  finding. A new field must have a default.
- **The documented lifecycle rules are behaviour, not implementation detail**,
  and README states each one:
  - deleting a regular profile removes it from every balancing profile, and a
    balancing profile that drops below two members is deleted entirely;
  - deleting the default profile promotes the first remaining profile;
  - editing a profile's key creates a **new** credential entry and leaves
    profiles sharing the old one untouched;
  - the `backup` flag lives on the profile, not on the membership.

  A change to any of these needs the README section moved with it. A change that
  contradicts one without moving it is drift — quote both sides.
- **Reserved names exist to keep routing unambiguous.** `setup`, `doctor`,
  `codex`, `claude`, `gemini`, `kilo`, `profile`, `bridge`, `egress` and their
  siblings cannot be profile names. A new command or launcher target that does
  not join that list creates a profile that shadows it. See `rules/cli.md`.
- **Names are stored lowercase and matched case-insensitively.** A change that
  normalises on one side only splits a user's profile in two.
- **Orphaned credentials.** Deleting a profile should not leave its credential
  behind, and deleting a credential should not leave a profile pointing at
  nothing. A resolver that raises on a dangling `auth_ref` is better than one
  that returns `None` and lets an empty key reach the provider — check which the
  diff produces.

## Auth flows

PKCE and OAuth are protocol implementations with exact requirements. Check the
verifier is generated with a CSPRNG and is long enough, the challenge method is
what the provider expects, the state parameter is compared, and the redirect
target is not attacker-influenceable. A refresh path that can loop, or that
leaves the stored token in a half-written state on failure, is a finding.

## Layering

`pyproject.toml` forbids `kitty.profiles` and `kitty.credentials` from importing
`kitty.cli`, `kitty.launchers` or `kitty.bridge`. They are leaves.

## Tests

A secrets change with no test is a finding, and the useful test is the negative
one: assert the value is **absent** from the rendered output, the log, the
exception text. `tests/test_credential_store.py` and
`tests/credentials/test_stage7_credentials.py` are where these live.

## Severity

- **Critical** — a secret in a config file, log, message, `repr` or terminal; a
  schema change that breaks an existing config; a keyring fallback that loses a
  credential silently.
- **Warning** — a documented lifecycle rule changed without README; a missing
  reserved name; a dangling-reference path; an auth-flow parameter you could not
  confirm against the provider's requirement.
- **Suggestion** — naming, duplication between the two backends, docstrings.
