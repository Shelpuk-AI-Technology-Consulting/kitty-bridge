---
id: t_d5_curl_cffi_oracle_slice
depends_on: [KBR-35, KBR-41, KBR-51, KBR-52, KBR-34, KBR-149, KBR-171, KBR-184, KBR-185]
---

# T-D5 — Oracle slice: curl_cffi transport (openai_subscription)

Jira: **KBR-55** ([link](https://shelpuk.atlassian.net/browse/KBR-55)).
Plan: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §7, row T-D5. Design:
`TEST_SUITE.md` §3.3.2 (the two assertions), §3.2.2 P13–P17 (the
load-bearing rows this slice exercises), §3.2.3 (the serialization
boundary `_cc_to_responses` / `_prepare_responses_body` lives at),
§7.2.3 (T-B2's curl_cffi recorder, harness TLS, the refresh leg),
§7.6 (the proven-vertical-slice pattern this slice follows for the
custom-transport case). Requirements:
`.requirements/20260923T084139Z_t_d5_curl_cffi_oracle_slice/REQUIREMENTS.md`.

The ticket carries **three pieces of work**, two of them recorded in the
KBR-55 ticket comments as scope-adds so they live on this PR rather than
spinning their own:

1. **The curl_cffi slice** — a driven end-to-end test that runs the
   transparency oracle against `openai_subscription`, capturing the
   unconditional row set P13 / P14 / P17 / P22-when-met / P23 / P24 /
   P25-when-met and the tool-schema boundary P15 / P16.
2. **Six corpus entries** — the trigger case and §3.3.2 assertion-2
   complement for P34 and P35 (two trigger shapes for P35), and P33's
   complement (T-D6 owns P33's trigger case, per KBR-184's comment).
3. **Three register rows** (M27, M28, M29) closing G28 / G29 / G30,
   re-derived from the landed readers per KBR-184's re-derivation
   precedent.

## What the task does

Lands the L3 oracle run for `openai_subscription` end-to-end. The
`curl_cffi` transport is the **second** of the three custom-transport
adapters (§3.2.3): its request path never reaches
`translate_to_upstream` (`OpenAISubscriptionAdapter`'s hook is "as dead
there as §3.2.3 says translate_to_upstream is"), so an
aiohttp-recording oracle run catches nothing on this provider — the
bodies the recorder captures are the ones `_cc_to_responses`
(CC-origin) and `_prepare_responses_body` (Responses-origin) ship,
**after** the strict Codex allowlist is applied.

The slice is **sibling to T-D4 (default-transport slice) and T-D6
(botocore slice)**, not a successor: each consumes the oracle interface
(KBR-51) plus its own transport's prerequisites (T-B1 / T-B2 / T-B3).
T-D5's recorder is KBR-41's `CurlCffiTransport` /
`CurlRecordingUpstream`; the product seam is the module-constant swap
`codex_backend_url`; none of this is new under T-D5 and none of it
changes.

Per KBR-55 comment 2, the conditional rows KBR-184 landed (P33, P34,
P35) **all promised their corpus cases with T-D5**. The promise is in
the row comments themselves; landing T-D5 closes it for the REQUEST
triggers whose complements are corpus-authorable on the Messages route.
P33's trigger case is the bedrock slice's deliverable (T-D6), because
it requires driving the bedrock adapter, not the curl_cffi one.

## Falsification (plan §1.4)

Per §3.3.1 and §3.2.3, the slice ships the falsification cases the
oracle-driven shape requires:

- **The P13-drop-set falsification.** Driven end-to-end, with the
  inbound body carrying **all fourteen** CC sampling parameters
  (hand-built; `minimal_inbound_body` carries none of them, which is
  the §7.6 vacuity trap — "the driven request did not carry the thing
  the claim was about"). `dataclasses.replace(P13, paths=())` must
  red-line the run naming the bare `conversation.sampling` anchor.
- **The P17-override falsification.** Removing P17's claim red-lines
  the run naming `envelope.stream` / `envelope.store`.
- **§3.3.3's distinguishing property.** An inbound body carrying
  `kitty-bridge` must arrive **byte-identical** in the captured body
  (asserted directly on the capture, not via the oracle); a separate
  oracle run against a capture whose body is mutated to insert a part
  with no inbound counterpart must raise `UnclaimedMutationError` —
  the synthetic M13-string shape KBR-5 inherits from
  `tests/bridge/test_vendor_token_guard.py` and T-G5 takes over.
- **The routing falsification.** A capture whose path is rewritten to
  a different one in memory must fail on routing and only on routing,
  so a run whose body obligations have already passed still red-lines.

## Decisions

- **One slice file, seven tests, one helper.** The test file is
  `tests/harness/test_oracle_curl_cffi.py` — three driven happy-path
  cases (CC origin, Responses origin, routing) and four falsification
  cases (P25 override, P17 override, §3.3.3, routing). One
  `_post_one(...)` helper deduplicates the
  fixture/post/capture pattern across all seven (the reviewer's
  MUST-FIX-10 finding: the "single call site" premise was wrong, and
  seven repetitions of the same setup is the kind of drift T-W9's
  precedents exist to prevent). The pattern matches T-D1's
  `test_oracle_driven.py`, not T-D4's matrix, because T-D5 is *one*
  adapter (one transport). The L3 default matrix is T-D9's.
- **Layer is `l1` path-default**, matching T-D1
  (`test_oracle_driven.py`) and T-D2 (`test_oracle_routing.py`).
  §3.4 calls the surface L3, and T-K6 owns the activation. Today an
  `l3` marker deselects the test in the Fast gate
  (`two-pytest-gates-deselect-acceptance` memory); path-default
  keeps it in the gate today, with the same consequence every T-D
  sibling carries. The L3 activation swap is T-K6's separate
  ticket.
- **Harness TLS** is wired (not skipped), matching §7.2.3's
  conformance posture: `transport("curl_cffi",
  WireFormat.OPENAI_RESPONSES, ssl_context=<server_ssl_context>,
  ca_cert=<harness_ca_path>)`, where the two fixtures come from the
  harness `certs` fixture and the harness CA path KBR-41 already
  provides.
- **The CC-origin path is the primary path the slice drives.** P13
  (the CC builder's 14 sampling drops) plus P17 (`stream: true` +
  `store: false`) plus P22 (when `effort` is present) plus P24 (the
  CC builder's 13 non-sampling fields) are the load-bearing
  CC-origin set. The Responses-origin path is **a second driven
  case in the same file**, because P14, P15, P16, P25 trigger
  against the inbound wire's Responses shape.
- **The driven body is deliberately minimal but does NOT use
  `minimal_inbound_body` unmodified.** The body deliberately
  carries **none** of `cache_control` (M16), `metadata` (M26),
  `top_k` (M27 / G28), non-empty `stop_sequences` (M28 / G29), or
  string-form `stop` (M29 / G30) — so those rows' triggers are
  unmet and their absence is provable. The two falsification
  bodies, by contrast, *are* hand-built and carry exactly the
  thing the row being falsified claims (the 14 sampling
  parameters for P13, the Responses-shape fields for P17).
- **The recorder is KBR-41's** — `tests/harness/curl_recorder.py` /
  `tests/harness/curl_cffi.py`. The recorder's
  `CONFORMANCE_CASES` pre-names `OPENAI_RESPONSES` and
  `/v1/responses`; nothing under T-D5 changes the recorder.
- **The product seam is the existing module-constant swap
  `codex_backend_url`** — not a `provider_config["base_url"]`
  override, per §3.2.3 and §7.2.3's precedent for
  custom-transport adapters. **`bind()` returns the recorder's own
  URL in place of the constant** (§7.2.3's seam), so the captured
  path is the recorder's `CODEX_RESPONSES_SUFFIX` (`"/responses"`),
  NOT the published `_CODEX_BACKEND_URL`'s pathname
  (`/backend-api/codex/responses`). The routing derivation must
  read the seam's output, not the published constant — the
  reviewer's BLOCKING-2 finding.
- **`expected_route` is derived independently of the adapter.** Per
  T-D2, computed in the test from the profile and the
  `codex_backend_url` seam's output, never by calling
  `build_base_url()` / `get_upstream_path()` — with the authority
  and scheme rewritten to the recorder's (§3.3.5). KBR-127's
  model-prefix rule is **not** cited here — it applies to
  adapters that compose a URL from base + path (Azure, Vertex);
  `openai_subscription` posts to a fixed URL and the seam decides
  the destination (the reviewer's MUST-FIX-14 finding). The
  `codex_backend_url` seam being the only path into the recorder
  **is** the load-bearing routing fact.
- **The six corpus entries live as fixtures in `tests/corpus/`.**
  Each is a sibling file pair; the manifest declares its
  `wire_format`, `triggers_met`, `triggers_absent`,
  `inbound_protocol` and the same shape of `captured_from` /
  `captured_at` every corpus entry carries (synthetic entries use
  empty strings per the `no_output_config` precedent). The six
  entries are, verbatim (the reviewer's BLOCKING-8 finding — pin
  the composition):
  - `p34_parallel_false_omitted_trigger.{json,body}`
  - `p34_parallel_false_omitted_complement.{json,body}`
  - `p35_tool_choice_omitted_no_tools_trigger.{json,body}`
  - `p35_tool_choice_omitted_forcing_anthropic_tool_trigger.{json,body}`
  - `p35_tool_choice_omitted_complement.{json,body}`
  - `p33_bedrock_auto_tool_choice_complement.{json,body}`
  P33's trigger case is **T-D6's** (per the P33 row's own comment).
  **T-D5 authors these fixtures; T-D8 owns the §3.3.2 enforcement
  that consumes them.** The slice's own oracle runs do not
  exercise assertion-2 against P33 / P34 / P35 — those require
  the bedrock and Anthropic reader paths the slice does not
  drive (the reviewer's MUST-FIX-16 finding).
- **G28 / G29 / G30 rows land alongside.** All three are
  re-derived from the §9.2 prose by reading the landed readers,
  with the honest counts per KBR-184's precedent:
  - **M27 (G28) — `top_k` dropped on the non-Anthropic-family
    routes.** Site is `MessagesTranslator.translate_request`
    (the `_top_k` mint at `translator.py:417-418`). Trigger
    `ANTHROPIC_TOP_K_PRESENT` (new enum member, REQUEST,
    `ArrangingBy.REQUEST`), `conditional=False` per §9.2
    ("unconditional in P13's sense — fires wherever the field
    is present, so it owes no §3.3.2 assertion-2 complement").
    Paths anchored at `conversation.sampling[top_k]` — the
    narrowest address the Anthropic reader projects (T-A1's
    `_SAMPLING_KEYS` maps `top_k → top_k`). Scope =
    **18** adapters, derived as
    `_TRANSLATED_MESSAGES_ADAPTERS ∖ {anthropic, minimax_token,
    opencode_go}` — the three Anthropic-family adapters whose
    Messages-routed restore runs (`custom_anthropic` and
    `zai_coding` never reach the translator, so they are not
    in `_TRANSLATED_MESSAGES_ADAPTERS` to begin with). P13's
    bare-`CONVERSATION_SAMPLING` anchor matches the same delta
    on `openai_subscription`, but the specificity rule
    (`oracle._conditional_violations`) keys on the narrower
    anchor, so M27 specifically claims there.
  - **M28 (G29) — empty `stop_sequences` omitted.** Site is
    `MessagesTranslator.translate_request` (the truthy-only
    guard at `translator.py:409-411`). Trigger
    `EMPTY_STOP_SEQUENCES` (new enum member, REQUEST,
    `ArrangingBy.REQUEST`), `conditional=True` (the non-empty
    value is the complement — a non-empty list is carried as-is,
    so §3.3.2 assertion 2 owes a corpus entry). Paths anchored
    at `conversation.sampling[stop]` (the Anthropic reader's
    `stop_sequences → stop` rename at
    `reader_anthropic_messages.py:85`). Scope = the same **18**
    as M27.
  - **M29 (G30) — string-form `stop` rewritten into a list.**
    Site is `kitty/bridge/server.py:_normalize_cc_stop`. Trigger
    `_ALWAYS`; `conditional=False` (a body already in the list
    form meets the row as a no-op — M15's exact shape);
    `paths=(NOT_PROJECTABLE,)` with the
    `not_projectable_reason` binding T-A2 / KBR-34 to read
    `stop: "END"` and `stop: ["END"]` into the identical
    `Request` (§9.2's exact clause). Scope = `(ALL_PROVIDERS,)`
    because the CC-ingress seam is not an adapter property
    (KBR-139 reachability of the site).
  - §9.2's stale reference to "Row **M17**" for G30 is void —
    KBR-232 reserved M17 for the thinking strip, so the landed
    id is **M29**, with a row comment naming the stale
    reference.
- **One-shot assert of register shape.** The M27 / M28 / M29 edit
  happens in one commit alongside the §3.2.1 markdown table rows
  (the L2 agreement guard red-lines a one-sided change); the
  §3.2.2 unconditional-id sentence grows **M27 and M29** (the
  two unconditional new rows) and **omits M28**. The L2 guard's
  diff against the registry AST is what holds the scope honest
  (KBR-139).
- **No new helpers in the register module.** The three new
  `MutationRow` entries use the existing dataclass and the
  existing `c.sampling_path` / `c.NOT_PROJECTABLE` vocabulary.

## Implementation notes

Filling in after the PR lands. Will append here rather than write a new
file, mirroring `kbr139_register_scope.md`'s convention.

### What shipped (2026-09-23, pre-PR commit)

- **`tests/harness/test_oracle_curl_cffi.py`** — the slice module, 7
  tests at `l1` path-default. Three happy-path (CC origin, Responses
  origin, routing), one routing falsification, three row-level
  falsifications (P17 override, P25 override, §3.3.3 distinguishing
  property). Harness-TLS wiring via the `certs` fixture +
  `server_ssl_context`; the transport's `CODEX_CA_CERTIFICATE` env
  wiring is what makes the adapter trust the harness CA (KBR-41).
- **One helper** `_post_one(...)` deduplicates the fixture/post/capture
  pattern across the seven tests (MUST-FIX-10 finding).
- **The oracle's `report.deltas` is the full delta list, not the
  unclaimed subset** — T-D1's `custom_openai` route happens to have
  zero legitimate deltas on a minimal body, but T-D5's
  `openai_subscription` route has three (`envelope.stream` +
  `envelope.store` via P17's forced streaming, and
  `conversation.sampling[max_tokens]` via M1's profile-model rewrite
  claimed by P13's bare anchor). A green run is "the call returned
  normally"; the happy-path test pins the three-path list so a
  regression that widens the delta set turns red rather than silent.
- **The P13 falsification was replaced with a P25 falsification.**
  Driving the fourteen CC sampling parameters onto an Anthropic
  Messages body residualises at the Anthropic reader (`temperature`
  and friends are CC-only keys, not Anthropic reader sampling keys),
  which turns the run red on the *inbound* side before P13 can
  fire. P13's bare-`conversation.sampling` anchor is a defensive
  claim — its drops are unreachable on the natural
  Anthropic→CC→Responses path because the Messages→CC translator
  only forwards `temperature` / `top_p` and `_cc_to_responses` builds
  the shipped body from named keys, so no sampled value reaches the
  wire for P13 to drop. P25 (allowlisted-but-falsy) *is* reachable
  on the Responses-origin path via `include: []`, and that
  falsification exercises the same structural claim (removing a row
  makes the run red).
- **§3.3.3's distinguishing property lands in the slice** per the
  KBR-51 pattern, with the two halves: (i) an inbound body carrying
  `kitty-bridge` arrives byte-identical in the captured body
  (direct assertion on the capture, not via the oracle);
  (ii) the captured body mutated to add an `input_text` turn with
  no inbound counterpart must raise `UnclaimedMutationError` —
  the orphan turn lands at `conversation.turns[0].parts[1]` per
  §3.3.1b's merge rule (consecutive same-role turns form one turn),
  and no triggered row claims that path. The prior attempt used a
  content type the Responses reader does not consume
  (`"text"`), which residualised at the reader instead of
  reaching the diff — fixed to `input_text`.
- **`test_corpus.py` grew** `TestTheKbr55Td5CorpusEntries` —
  the requirements doc's AC 5 as a committed-corpus guard: P34
  ≥1 trigger + ≥1 complement, P35 ≥2 triggers (case 1 + case 2)
  + ≥1 complement, P33 ==0 trigger + ≥1 complement.

### Deferred out of scope (with the reason)

- **P33's trigger case** — T-D6's deliverable per the P33 row's own
  comment (the bedrock trigger requires driving the bedrock
  adapter, which the curl_cffi slice does not own).
- **P13's falsification via the natural route** — see the P25
  replacement note above; P13's bare-anchor claim is defensive
  (its drops are unreachable on the natural path).

## Status

Implemented (2026-09-23); PR not yet opened. Four commits on
`feat/kbr-55-t-d5-curl-cffi-oracle-slice`, rebased onto current
`origin/main` at `5f59da3`:

| SHA | Title |
|---|---|
| `a6c5be8` | register rows M27 / M28 / M29 closing G28 / G29 / G30 |
| `200aeba` | six corpus entries — P34 trigger + complement, two P35 triggers + complement, P33 complement |
| `5bbcefa` | driven curl_cffi oracle slice — 7 tests, harness TLS, 3 happy + 4 falsification |
| `5e61121` | post-rebase ruff fixes (F541 + E501) — applied at the reviewer's suggestion 1; suggestion 2 (corpus body trailing `\n`) applied in a follow-up commit; suggestion 3 (stale line citations in M27/M28 row comments) applied together with the SHA refresh in a follow-up commit. |

The requirements doc and step file have been through one
system-design-reviewer round (9 BLOCKING + 7 MUST-FIX findings, all
applied), then one code-reviewer round (APPROVE, with three
SUGGESTIONs applied pre-PR).
