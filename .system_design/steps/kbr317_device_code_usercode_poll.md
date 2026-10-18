---
id: kbr317_device_code_usercode_poll
depends_on: [kbr317_headless_env_detection]
---

# Device-code OAuth flow: usercode + polling + token exchange

KBR-317, step 2 of 6. The wire-level implementation of the device flow in
`kitty.auth.openai_oauth`, mirroring
`codex-rs/login/src/device_code_auth.rs` (fetched 2026-10-14).

## Why

The browser-based authorization-code flow requires the browser and the CLI to
share a loopback, which is false on a remote VM. The device flow moves the
authorization-code delivery to a polling endpoint, so no local server is
needed. OpenAI already implements and documents this protocol for the same
`client_id` kitty uses; mirroring it exactly is the ticket's stated intent
and keeps kitty's UX identical to the upstream tool.

## To Be

```python
# kitty/auth/openai_oauth.py (new symbols)
OAUTH_DEVICE_USERCODE_URL  # https://auth.openai.com/api/accounts/deviceauth/usercode
OAUTH_DEVICE_TOKEN_URL     # https://auth.openai.com/api/accounts/deviceauth/token
OAUTH_DEVICE_REDIRECT_URI  # https://auth.openai.com/deviceauth/callback
OAUTH_DEVICE_VERIFICATION_URL  # https://auth.openai.com/codex/device
DEVICE_FLOW_MAX_WAIT_SECONDS  # 900 (15 minutes, matches codex)

async def run_oauth_device_flow(http=None) -> OAuthSession: ...
def format_device_code_prompt(verification_url: str, user_code: str) -> str: ...
```

Wire contract (see SYSTEM_DESIGN §14.2):

1. `POST {issuer}/api/accounts/deviceauth/usercode` body
   `{"client_id": CLIENT_ID}` →
   `{"device_auth_id": str, "user_code": str, "interval": str}` (the
   `user_code` key may arrive as `usercode`; `interval` is a string of
   digits, whitespace tolerated).
2. Poll `POST {issuer}/api/accounts/deviceauth/token` body
   `{"device_auth_id", "user_code"}` every `interval` seconds. 403/404 =
   still pending, keep polling up to 15 min. Other non-2xx fail fast with
   the status. 200 → `{"authorization_code", "code_challenge", "code_verifier"}`
   (server-issued PKCE pair).
3. Exchange at `POST {issuer}/oauth/token` (authorization_code grant) with
   `redirect_uri = OAUTH_DEVICE_REDIRECT_URI` and the server-issued PKCE pair.
4. The resulting token dict feeds the existing
   `OAuthSession.from_token_response` + the best-effort `id_token` → API key
   exchange — identical to the browser flow's tail.

Redaction: `user_code`, `device_auth_id`, and the authorization code must
never appear in DEBUG logs (extends the §9 redaction policy to the three new
short-lived secrets).

## Interval policy (kitty-owned, deliberate divergence from codex)

Upstream `codex-rs/login/src/device_code_auth.rs` declares
`interval: u64` with `#[serde(default, …)]` and a custom deserialiser that
parses the string field — meaning (a) a missing field defaults to `0` and (b)
a non-numeric value causes the whole `usercode` request to fail. The poll
loop is `Duration::from_secs(interval).min(remaining)`, so `interval = 0`
busy-loops the token endpoint for up to 15 min.

Kitty's posture (chosen, recorded):

- Missing `interval` field → use a 5 s default (KBR-317 design decision;
  reasonable middle ground between codex's busy-loop and a too-slow 30 s).
- `interval` parses to 0 or fails to parse → raise typed
  `OAuthDeviceCodeIntervalError` (no busy-loop, no silent default).
- Sleep floor: `max(interval, 1)` to prevent busy-looping on a server that
  returns 0 with a parse-success.
- Final-sleep clamp: `min(sleep, MAX_WAIT - elapsed)` (matches codex) so
  the loop never overshoots the 15-min cap.

## Session lifecycle

`run_oauth_device_flow(http: aiohttp.ClientSession | None = None)` must
follow the same close-if-created pattern as the browser flow:
`close_http = http is None; http = http or aiohttp.ClientSession(...); try
... finally: if close_http: await http.close()`. Tests will exercise
both call shapes via aioresponses.

## Acceptance

- L1 unit tests for the response parsers: interval as padded string,
  `usercode` alias, success/pending/error dispositions.
- L2 contract test pins the JSON request/response shapes against the codex
  source snapshot.
- L3 subsystem test with `aioresponses` covering: happy path,
  404-on-usercode fast-fail, poll-then-success, poll-timeout, Ctrl+C
  cancellation, and the prompt format (ANSI + plain).
- `run_oauth_device_flow` returns a fully populated `OAuthSession`.

## Implementation notes

- Constants: `OAUTH_DEVICE_USERCODE_URL`, `OAUTH_DEVICE_TOKEN_URL`,
  `OAUTH_DEVICE_REDIRECT_URI`, `OAUTH_DEVICE_VERIFICATION_URL`,
  `DEVICE_FLOW_MAX_WAIT_SECONDS = 900`,
  `DEVICE_FLOW_MIN_INTERVAL_FLOOR = 1.0`,
  `DEVICE_FLOW_DEFAULT_INTERVAL = 5`.
- New exceptions: `OAuthDeviceCodeDisabledError` (usercode 404 — the
  server's "not enabled" signal), `OAuthDeviceCodeResponseError`
  (unparseable/missing-field usercode body; NOT the same as disabled —
  split during implementation), `OAuthDeviceCodeIntervalError`
  (interval present but ≤ 0 or unparseable), `OAuthDeviceCodeCancelledError`
  (Ctrl+C).
- `parse_device_user_code_response` / `parse_device_token_poll` never echo
  secret values into error messages (status codes and field names only).
- `format_device_code_prompt`: kitty-owned (codex structure + admin
  sentence + anti-phishing footer), ANSI-coloured, printed to stderr by
  the flow.
- **Sentinel bug found and fixed by the L3 poll tests**: the original loop
  assigned every parse result to the `poll_result` sentinel, so a
  *pending* result exited the loop and hop 3 fired with empty fields.
  Fix: only a `kind == "success"` result is stored. The poll-timeout and
  clamp tests now both guard this.
- **Redaction leak found and fixed by the L3 redaction test**: the shared
  `_exchange_code_for_tokens` embedded raw error-body text in the
  exception; OAuth error bodies can echo the authorization code. The
  helper gained `error_redactions: tuple[str, ...] = ()` (default empty
  preserves the browser path byte-for-byte); the device path passes the
  code + verifier.
- `_exchange_code_for_tokens` gained keyword-only `redirect_uri`
  (default = the localhost constant) — browser path untouched.
- aiohttp session: single try/finally, close-if-created (the first
  implementation had two finally blocks and closed the session before
  hop 3 — caught by the L3 tests).
- Test-mock lessons: aioresponses' registrations are consumed once — a
  pending-poll loop needs N registrations (or it raises ConnectionError,
  which the flow wraps as OAuthAuthorizationError and misleads the
  assertion); no-op `asyncio.sleep` stubs stop the loop's monotonic clock
  from advancing, so timeout tests must let sleeps run for real.
