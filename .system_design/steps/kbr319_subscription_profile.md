---
id: kbr319_subscription_profile
depends_on: []
---

# KBR-319 — Subscription profile declaration and local session check

## Planned delivery

Add the distinct `anthropic_subscription` provider declaration and a tokenless local Claude Code session-status reader. The reader accepts only `loggedIn: true`, `authMethod: "claude.ai"`, a non-empty `subscriptionType`, and optional `email`; unknown status fields, raw stdout/stderr, and parsed payloads never reach logs, storage, or UI. Setup/profile flows display only allowlisted account/subscription details and retain a generated schema-compatible UUID with no CredentialStore record instead of an OAuth token or Kitty-owned OAuth session.

## Decisions and why

- **Use Claude Code status, not its credential storage.** Claude Code owns platform-specific OAuth storage and refresh. Invoking its documented status command works across its supported storage backends and avoids Kitty copying credentials.
- **Keep the provider distinct from `anthropic`.** API-key Anthropic and a Claude.ai subscription have incompatible identity and request contracts; one provider type would make the security boundary implicit.
- **Validate local state only.** Setup never sends a synthetic subscription request, because the feature is an exact relay and only Claude Code may construct the request.

## Implementation notes

Landed on feat/kbr-319-anthropic-subscription. `read_claude_subscription_status` (src/kitty/auth/claude_subscription.py)
invokes `claude auth status --json` with a 30 s timeout; accepts only the four allowlisted fields; returns redacted
reasons for missing binary, failure, timeout, invalid/non-object JSON, logged-out, and wrong-method/empty-type states.
Profile keeps a UUIDv4 `auth_ref` with no CredentialStore record. 15 L1 tests in tests/test_anthropic_subscription.py.
Reviewed: code-reviewer APPROVE (R2 coverage + redaction).

Review round (2026-10-18, PR #294 CI review): the reader now turns every other spawn `OSError` (for example,
a non-executable Claude binary) into a fixed, path-free remediation message instead of allowing a wizard traceback.
`kitty doctor` now treats the Claude Code session as the subscription profile's credential in both the checklist and
single-profile paths; it never asks CredentialStore for the intentionally absent UUID. The profile edit credential action
is now a session re-check, not an API-key prompt: no key is stored and `auth_ref` remains unchanged. Tests pin the
PermissionError remediation, both doctor paths, and successful/failed session re-checks.

Design-review round (same day): `session_detail` (beside the reader) is the one place the allowlisted
email/subscription-type join happens (was five copies across doctor, profile, and setup flows). The edit flow's
re-check runs before the model prompt, so a failed check under "Both" aborts before any typed input is discarded
(test pins that no prompt fires). The edit-flow docstring states the subscription arm's verify-only contract.
