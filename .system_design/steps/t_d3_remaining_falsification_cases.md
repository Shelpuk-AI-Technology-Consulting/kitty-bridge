---
id: t_d3_remaining_falsification_cases
depends_on: [KBR-51, KBR-52]
---

# T-D3 — Remaining oracle falsification cases

Jira: **KBR-53** ([link](https://shelpuk.atlassian.net/browse/KBR-53)).
Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §7, row T-D3. Design:
`TEST_SUITE.md` §3.3.1, §3.3.1a, §3.3.5. Requirements:
`.requirements/20260922T000000Z_kbr53_td3_remaining_falsification/REQUIREMENTS.md`.

## What the task does

Completes the oracle's falsification suite so the oracle may gate acceptance
(T-J2) and the Subsystem job (T-K6) — plan §1.4 forbids promoting the oracle
into gating infrastructure before its falsification suite is complete.
T-D1 delivered the changed-model case; T-D2 the Azure deployment-segment
case; this task delivers the remaining six, all running in the suite.

1–4. §3.3.1's four unwritten body cases — flipped `stream` (unclaimed
   `envelope.stream`), deleted tool `description` (unclaimed
   `conversation.tools[get_weather].description`), stripped `strict` where
   no register row applies (unclaimed `.strict`), and an injected
   `x-kitty-trace` field (non-empty residual, fails closed via
   `c.ResidualFieldsError`).
5. §3.3.1a's anchor-discipline tripwire (the KBR-25 comment on KBR-53; the
   paired obligation is KBR-26's): with the production register and P15's
   trigger met, a deleted tool `description` is an unclaimed delta at the
   deep path — proving the `.strict`-anchored P15 does not swallow it. A
   fixture control with a coarse `conversation.tools[*]` row demonstrates
   the prefix rule the tripwire guards against.
6. §3.3.5's seventh case (the KBR-52 comment on KBR-53; the defect is
   KBR-127): driven through the bridge on an `opencode_go` profile with
   `model = opencode/minimax-m2.5`, one test asserts path (`/v1/messages`),
   auth scheme (`x-api-key`, not Bearer) and body shape (Anthropic Messages
   reader) agree — each leg alone passes the KBR-127 defect; only the
   conjunction catches it.

## Decisions

- **R1–R3 wholesale-under-declare the trigger vocabulary**
  (`triggers_met=frozenset()`), T-D1's posture: P17 is not the only
  `envelope.stream` claimer (M11, P18, P19 also anchor it), so selective
  omission cannot ensure none claim.
- **R5a is the discipline tripwire; R5b is the mechanism control.** R5a's
  expected raise is what detects a coarse re-anchoring of P15 (the raise
  silently stops happening). R5b proves the prefix rule is real, so R5a's
  failure to raise is evidence, not a test bug.
- **R6 drives through `redirected(OpenCodeGoAdapter(), …)`** — the adapter
  does not read `provider_config["base_url"]`, so the `redirected()` seam
  (`tests/harness/bridge.py:314`) is the only thing that re-hosts it at the
  recorder; the `{"base_url": …}` config channel alone would leave the
  bridge posting at the real opencode.ai.
- **R6's auth-scheme leg asserts captured headers directly** — the oracle
  does not diff headers (§3.3.1a, §3.2.2); the independently derived
  expectation (Messages → `x-api-key`) is what the captured headers must
  agree with.
- **No pytestmark** — the files default to l1 per the T-D1/T-D2 precedent;
  T-K6 owns the l3 activation.

## Status

Implemented (2026-09-22); implementation on
`feat/kbr-53-oracle-falsification-cases` off `origin/main`. Six cases
land in two files (`tests/harness/test_oracle.py::TestFalsificationSuite`
+ a `TestTotalityGate` extension, and
`tests/harness/test_oracle_routing.py::TestPrefixedProfileModel`), all
six passing in the suite, no production code touched.

KBR-127 red evidence captured by temporarily reverting the two
`_route_model(cc_request)` call sites in `BridgeServer._build_upstream_url`
(`server.py:11280`) and `BridgeServer._upstream_headers` (`server.py:11401`)
to the pre-KBR-127 form `self._active_model or ""`. The driven R6 test
fails with `harness.bridge.MisdeclaredFormatError: transport
'opencode-go-aiohttp' declares anthropic_messages but
['/zen/go/v1/chat/completions'] select another format` — the path leg
flips to the Chat Completions endpoint (because
`get_upstream_path('opencode/minimax-m2.5')` falls through to
`/v1/chat/completions` when the bare-name check is fed the raw prefixed
string), and the bridge fixture's own teardown guard catches it before
the test body can reach the assertion. Code restored; R6 passes again
on the unmodified code. The body leg (the Messages body shape) stays
intact under the defect — `translate_to_upstream` already read
`cc_request["model"]` before KBR-127 — which is exactly the conjunction
the seventh case is built to catch: a single leg is defensible in
isolation.

Three reviewer rounds on the requirements doc (one blocker — the
`redirected()` seam — and seven concerns) applied before implementation.
