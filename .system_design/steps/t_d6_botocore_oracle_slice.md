---
id: t_d6_botocore_oracle_slice
depends_on: [KBR-51, KBR-52, KBR-42, KBR-37]
---

# T-D6 — Oracle slice: botocore transport

Jira: **KBR-56** ([link](https://shelpuk.atlassian.net/browse/KBR-56)).
Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §7, row T-D6. Design:
`TEST_SUITE.md` §3.3.2, §3.3.4, §3.2.3, §7.2.4, §7.4. Requirements:
`.requirements/20260922T224150Z_t_d6_botocore_oracle_slice/REQUIREMENTS.md`.

Dependencies (all Done at implementation time): [KBR-51](https://shelpuk.atlassian.net/browse/KBR-51)
(oracle core), [KBR-52](https://shelpuk.atlassian.net/browse/KBR-52) (routing
expectation), [KBR-42](https://shelpuk.atlassian.net/browse/KBR-42) (botocore
recorder), [KBR-37](https://shelpuk.atlassian.net/browse/KBR-37) (Bedrock
Converse reader).

## What the task does

Drives the transparency oracle end-to-end on the botocore transport for the
`bedrock` adapter, the custom-transport adapter where the least-inspected
serialisation code lives (§3.3.4). A real `BridgeFixture` posts a minimal
Anthropic Messages body — asking for one model while the profile pins a
different one, so M1's rewrite is observable rather than a no-op — through
the bridge against a `BedrockRecordingUpstream`; the captured **wire** body
(post-`translate_to_upstream`, post-`_bedrock_body` — i.e. after register
row P18's `modelId` / `stream` pops) is read by `reader_bedrock_converse`
and compared against the inbound Messages projection by
`assert_no_unclaimed_mutation`. The slice ships with its falsification case
per plan §1.4.

## Why this matters (product context)

Three adapters — `bedrock`, `ollama_cloud`, `openai_subscription` — set
`use_custom_transport = True` and never reach the default `_make_upstream_request`,
so the aiohttp recording upstream never sees their bodies. The oracle's
"every provider" claim is false for all three until each has its own recorder
slice. T-D6 closes the botocore half. The other two halves are T-D5
(curl_cffi / openai_subscription) and T-D7 (provider-aiohttp / ollama_cloud).

## The §3.2.3 boundary, stated precisely

`modelId` is a **URI parameter** on the Converse wire
(`POST /model/{modelId}/converse`), not a body key — the harness reader
reads `envelope.model` from `CapturedRequest.path`
(`reader_bedrock_converse.py`, `PUBLISHED_TOP_LEVEL_KEYS`) — and
`translate_to_upstream` never emits `stream` into the body
(`bedrock.py:621-624`; P18's `stream` pop is defensive). The captured body
therefore carries neither key, and the test asserts that as a positive
wire-shape observation documenting the boundary — not as a P18
falsification.

## Falsification (plan §1.4)

**T-D1's changed-model pattern, adapted to bedrock** (empirically verified
by probe before being committed): the profile model is distinct from the
inbound body's, so the captured URI carries the profile's model and the
`envelope.model` delta is real. With `PROFILE_SETS_MODEL` in
`triggers_met` the delta is claimed and the run is green with
`deltas=('envelope.model',)`; with the trigger deliberately omitted the
delta is unclaimed and the oracle raises `UnclaimedMutationError` naming
`envelope.model` — the falsification test asserts exactly that, and its
`else` branch raises so a silently-green run cannot pass.

**A P18 pop-regression is not observable through this slice**, and the
reason is a wire fact, not a test gap: the unpopped `modelId` collides with
boto3's `converse(modelId=…, **body)` call as a duplicate keyword argument
(`bedrock.py:659`), the bridge 500s with `ProviderError` **before the
recorder captures anything** (probe: status 500, `N_CAPTURES: 0`), and the
drive's own `status == 200` / one-capture assertions are what would surface
it. An earlier draft of this step file described a monkeypatch
falsification asserting the captured payload would carry `modelId` — that
was empirically wrong on both counts (no payload is captured; even on the
green path `modelId` lives in the URI, never the body) and is recorded here
so the wrong narrative does not return.

## Decisions

- **Capture via `BedrockRecordingUpstream`, not a `before-send` stub.**
  T-B3's recorder is the harness's per-transport observation point
  (§7.2.4's "harness reaches upstream through five distinct client
  configurations"; §3.3.4's "parametrised over transport"). The recorder
  already captures the wire body after the pops — exactly the §3.2.3
  boundary. A `before-send` short-circuit (KBR-78's T-G9 seam) would observe
  the same bytes but bypass the recorder's conformance suite and its
  round-trip evidence, so the recorder is the right wiring and T-D1's
  precedent agrees.
- **Driven through a real `BridgeFixture`, like T-D1.** The recorder +
  fixture prove the wire is what the bridge sends end-to-end; a
  hand-crafted captured body would let a boto3-only defect pass.
- **Distinct profile model in both tests** (T-D1's lever). The fixture's
  default profile model equals `minimal_inbound_body`'s default, so an
  equal-model positive is a no-op rewrite and a trivially-green run —
  probe-verified. With the distinct model the rewrite is real, the
  positive is green with `deltas=('envelope.model',)` (claim machinery
  genuinely exercised), and the falsification (trigger omitted) fires.
- **Minimal synthetic body, not a corpus entry.** T-D1's docstring
  names the rationale: the corpus-driven run against every entry is
  T-D4's deliverable. The conditional rows reachable on the bedrock
  route — P33 (`BEDROCK_FORCES_AUTO_TOOL_CHOICE`, bedrock-scoped) and
  P34 (`disable_parallel_tool_use`, Messages-translator-scoped, not
  bedrock-only) — owe their trigger cases and assertion-2 complements
  to **T-D5's corpus entries** (T-D8 owns the corpus-wide check). The
  minimal body carries no `tools`, no `tool_choice` and no
  `disable_parallel_tool_use`, so neither trigger is met here.
- **§3.3.2 assertion 2 is deferred, on the record.** This slice
  exercises assertion 1 only; T-D5's corpus entries carry the per-row
  complements and T-D8 the corpus-wide check.
- **Routing (§3.3.5) is out of scope, flagged.** The ticket's Done-when
  is body-only; no plan row assigns bedrock's routing assertion to a
  task. The gap is named in the PR for the owner rather than silently
  absorbed.
- **Merge policy.** No register or reader changes in this PR; the PR
  merges only because the driven run is green on first execution. If a
  future re-run surfaces a register gap, the failing case is evidence
  for a sibling ticket and this slice's gate holds the merge until it
  lands.
- **No `pytestmark` (l1 default).** T-D1's docstring: "§3.4 calls this
  surface L3 and T-K6 owns the l3 activation." No
  `test_layer_selection.py` registration.
- **§8.2 socket-binding registry.** The file binds real loopback
  sockets (a `BridgeServer` + the recorder, two per run), so it joins
  `SOCKET_BINDING_L1_MODULES` with its companion sites (the KBR-290
  `--ignore` row, the guard's count pin, the §8.2 bullet). T-D1's
  `test_oracle_driven.py` has the same shape and is an inherited
  pre-KBR-272 gap — flagged for the owner, not silently fixed here.

## Status

Implemented on `feat/kbr-56-t-d6-botocore-oracle-slice` off `origin/main`
(2026-09-22). Two tests, both green locally (`2 passed`), CI triple green
(`ruff check .`, `lint-imports`, `mypy src/kitty`). Reviewed by the
system-design-reviewer and code-reviewer subagents; the CRITICAL/HIGH
findings (tautology narrative, unreachable monkeypatch falsification,
non-load-bearing `PROFILE_SETS_MODEL`) were resolved on the probe-verified
narrative above. PR opened; awaiting review; not merged.
