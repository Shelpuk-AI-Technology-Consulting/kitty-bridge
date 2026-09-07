# Rule: the egress gateway (`egress.py`, `egress_guard.py`, `egress_store.py`)

A static egress gateway puts every kitty install behind one IP so the provider
sees a single stable client. README documents it as a product feature; for review
purposes it is a **containment boundary**.

## The invariant

**When an egress gateway is configured, no traffic to the upstream provider may
leave the machine by any other path.** Not "most traffic", not "the bridge's
traffic": if a gateway is configured and a request could reach the provider
directly, that is a **critical** finding.

This component is the only one in the repository whose failure mode is *silent
success*. A request that should have been proxied and simply was not looks
identical, from every observable in the system, to one that was. Nothing goes
red. The user finds out when the provider sees an unexpected IP — which may be
never, or may be an account suspension.

So review changes here against the failure, not against the happy path.

## What to check

- **Fail-closed is the contract.** README: *"If a profile uses a transport that
  cannot honour the proxy, kitty refuses to start and tells you which profile,
  rather than quietly connecting from the machine's own IP."* AWS Bedrock in SSO
  mode is the named example, because botocore resolves those credentials outside
  the proxied client. A change that turns a refusal into a warning, a log line,
  or a fallback is critical. A **new** transport or client added anywhere in the
  repository that the guard does not know about is the same defect arriving from
  the other direction — ask whether the guard was extended.
- **Every HTTP client must be covered.** The repository uses more than one
  (`aiohttp` and `curl_cffi` at least). A proxy honoured by one and ignored by
  the other is a hole that only opens for the providers using the second. Check
  which client the changed code path uses.
- **`proxy=""` is not "no proxy" — it is a bug.** An empty string loads as a
  valid-looking configuration and then connects directly, because the client
  ignores it. Any code path that can produce an empty or whitespace proxy URL and
  treat it as configured is critical.
- **Resolution order is `--egress-proxy` flag, then `KITTY_EGRESS_PROXY`, then
  the store.** A change to that order changes which of three sources wins on a
  machine where more than one is set, and the environment variable is the one an
  ambient value can occupy without anyone intending it. Flag a reorder that the
  pull request does not explain.
- **Local endpoints are deliberately not tunnelled.** Loopback, private ranges
  and instance metadata connect directly, because a rented proxy cannot reach the
  user's own network — README says so and it is correct. Do **not** raise that as
  a leak. Do check that the predicate deciding "local" cannot be widened by an
  attacker-influenced value, and that a DNS name resolving into a private range
  is handled the way the diff claims.
- **CONNECT, not TLS termination.** TLS runs end to end between kitty and the
  provider, so the gateway moves encrypted bytes and never sees a key or a
  prompt. A change that terminates TLS at the proxy, or that would let it, breaks
  the privacy claim README makes and is critical.

## Secrets

The gateway password goes to the credential store under an opaque id, never into
`egress.json`, and is masked wherever kitty prints the gateway. A diff that puts
the password in a config file, a log line, an exception message, an error
returned to the user, or a `kitty doctor` / `kitty egress show` table is
critical. The proxy **address** and **username** are also sensitive in CI
context — see `rules/ci.md`.

## Tests

An egress change with no test is a finding, and the useful test asserts the
negative: that the request went through the proxy, or that the run refused to
start. `tests/test_egress_fail_closed.py`, `test_egress_coverage.py` and
`test_egress_https_proxy.py` exist to hold exactly these. Weakening one of them
to make a diff pass is a **critical** finding.

## Severity

- **Critical** — any path by which configured egress can be bypassed; a
  fail-closed refusal downgraded; a credential rendered; a weakened test.
- **Warning** — a resolution-order or predicate change with no stated reason; a
  new client path whose coverage you could not confirm.
- **Suggestion** — message wording, structure.
