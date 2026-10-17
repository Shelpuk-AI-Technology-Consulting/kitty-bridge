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

Pending.
