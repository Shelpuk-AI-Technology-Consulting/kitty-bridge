# Kitty Bridge — Test Suite Implementation Plan

**Implements:** [`TEST_SUITE.md`](TEST_SUITE.md). That document says what the suite must prove and
why; this one says who can build what, in what order, without waiting on each other.
**Traces to:** [KBR-2](https://shelpuk.atlassian.net/browse/KBR-2).
**Status:** Plan. No Jira issues created yet — the `T-` ids here are plan-local and become ticket
summaries when the plan is accepted.

> **On identifiers.** Every task id is prefixed `T-`. `TEST_SUITE.md` already uses bare `C1–C6`
> for observable channels, `F1–F5` for findings, `G1–G19` for gaps, `I1–I3` for the invariants,
> and `M`/`P` for register rows. Unprefixed task ids collided with all of them — the first draft
> of this plan said "KBR-6 → G3" where the design says "KBR-6 → G15". The prefix is the fix, and
> the **Design** column carries the reverse link.

---

## 0. How this plan is organised

The design has one shape problem for delivery: almost every interesting test depends on shared
infrastructure that does not exist. Build it in the wrong order and one person writes harnesses
for three weeks while everyone else waits.

So the plan is arranged by **what unblocks the most people soonest**, not by importance:

- **§2 Wave 0** is the enablement set — eight small artifacts, plus the handful of tasks that
  genuinely need nothing. Until it lands, parallelism is capped around ten. After it, around
  fifteen.
- **§3–§13** are epics. Each task is atomic, independently landable, and carries its dependencies,
  acceptance criteria and design reference.
- **§14** is the wave plan and the two long chains. **§15** lists tasks blocked on decisions
  rather than on code. **§16** covers the open defects. **§17** covers sequencing risk.

**Read §1 before picking up any task.** The definition of done is unusual and it is the thing most
likely to be got wrong.

---

## 1. Rules every task obeys

### 1.1 Definition of done

1. The deliverable exists and satisfies the acceptance criteria in its row.
2. `ruff`, `lint-imports`, `mypy src/kitty` and the full suite pass on Python 3.10–3.13.
3. Every new test carries exactly one layer marker (**T-W1**) so it lands in exactly one CI job.
4. Google-style docstrings on every class, method and function; module docstrings; block comments
   explaining *why*. Test code is code.
5. **It can land on `main` alone.** No task may leave the suite red waiting for a sibling. A check
   that fails because of a known product defect lands with an entry in the exemption registry
   (**T-W7**) — not disabled, not omitted.

### 1.2 The harness rule — a harness is not done until it has been seen to fail

This is the throughline of the design and the reason it took four review rounds. Repeatedly,
proposed tests would have passed without proving anything: an oracle composed of two functions
that were not inverses; a containment test asserting nothing arrived at a destination nothing
could reach; a guard proving a function was *called* when the enforcement was the branch after it;
a projection that could not see the model name, in a product whose purpose is changing the model
name.

So for every task that builds a harness, assertion or guard:

> **Acceptance requires a committed falsification case: a deliberate defect the harness must
> detect, running in the suite, not demonstrated once by hand.**

Rows below name the specific case where the design specifies one; otherwise the author picks one
and records it in the PR.

### 1.3 Flags

| Flag | Meaning |
|---|---|
| **src** | Touches `src/kitty/`. Higher risk; needs a `code-reviewer` pass and a note in the PR |
| **ci** | Touches `.github/workflows/`. Verify on a branch first — a broken gate is worse than a missing one |
| **blocked** | Cannot start until an open question (`TEST_SUITE.md` §11) is answered. See §15 |
| **partial** | Can start and land, but a stated part of its acceptance waits on a decision |
| **defect** | Designed to land red, with an exemption removed by the defect's own ticket. See §16 |
| **resolves** | Its output answers an open question |

### 1.4 Sizing

**S** ≈ half a day · **M** ≈ 1–2 days · **L** ≈ 3–5 days. Nothing is larger than L; anything that
looks larger is split. Sizes assume the author has read the referenced design section.

---

## 2. Wave 0 — the enablement set

**Everything here is small, and nearly everything else waits on some of it.** Treat Wave 0 as one
push with several people on it, not a queue.

| ID | Task | Depends on | Unblocks | Design | Size |
|---|---|---|---|---|---|
| **T-W1** | Layer markers and pytest config | — | Everything | §8 | M |
| **T-W2** | Request types and the projection protocol | — | Epic A, then the oracle | §3.3.1 | S |
| **T-W3** | The Permitted-Mutation Register as data | T-W2 | Oracle, register guard, corpus loader | §3.2 | M |
| **T-W4** | Recorder protocol + primary aiohttp recorder | — | Epic B, T-W8, all of L3 | §7.2 | M |
| **T-W5** | Shared CONNECT proxy fixture | — | All of Epic E | §7.3 | M |
| **T-W6** | Corpus capture procedure, scrubber, loader | T-W3 | All of Epic C | §7.1 | M |
| **T-W7** | The exemption registry | T-W1 | Every guard that lands red | §8 | M |
| **T-W8** | Bridge-under-test fixture | T-W4 | ~15 tasks across D, E, G, I, J, K | §6.3.1 | M |

**Genuinely needs nothing, start day one:** T-W1, T-W2, T-W4, T-W5, T-E6, T-E7, T-F1, T-I1, T-I3,
T-I4, T-K1. Everything else in Wave 0 is one hop behind.

### T-W1 — Layer markers and pytest config

Register `l1`, `l2`, `l3`, `acceptance`, `agent_smoke`, `agent_live`, `eval`, `load`. A meta-test
asserts **every collected test carries exactly one** — zero or two both fail.

**Do not hand-edit 2,880 test functions across 139 files.** That is the worst possible first
commit when a dozen people are about to branch, and it conflicts with every one of them. Assign
the default in `pytest_collection_modifyitems` by path, and require an explicit marker only where
the default is wrong. The meta-test is what stops the defaulting rotting.

Also reconcile the existing `--runslow` skip-by-default mechanism with **T-K8** ("a gating job
whose prerequisite is missing fails") — today `slow` tests silently skip, which is the pattern
T-K8 forbids.

*Done when:* removing a needed marker fails the meta-test; `pytest -m l1` collects a strict subset
of `pytest`. *Size:* M — larger than it looks, because of the defaulting rules and the `--runslow`
reconciliation.

### T-W2 — Request types and the projection protocol

`Request(envelope, conversation, residual)`, `Envelope`, `Conversation`, `Turn`, `Part` variants,
and the `Projection` protocol. No readers — this is the shared vocabulary.

Includes the **totality rule** as a reusable assertion: every key classifies into envelope,
conversation or residual, and a non-empty residual raises. Shipping it here means all six readers
inherit one semantics rather than six interpretations.

*Done when:* a stub reader that silently drops an unknown key fails the totality assertion — the
falsification case. *Size:* S.

### T-W3 — The register as data

Design §3.2 is a markdown table; the oracle takes `register` and `triggers_met` as arguments and
the L2 completeness guard diffs against the same rows. **It has to become data, or both consume a
prose table by hand.**

One entry per row M1–M14 and P1–P21: id, site symbol, trigger predicate, the projection field it
touches (`envelope.model`, `conversation.tools[].strict`, …), conditional or unconditional, design
anchor. Also defines the **trigger vocabulary** T-W6's loader indexes by.

*Done when:* a test asserts the markdown table and the data agree in both directions.
Falsification: delete a row from either side. *Size:* M.

### T-W4 — Recorder protocol and the primary aiohttp recorder

`CapturedRequest(method, scheme, host, path, query, headers, body)` with original header casing and
order, arrival timestamp, and **the peer port of the accepted connection** — the join key for
containment (§5.2.1). Plus the aiohttp implementation serving Anthropic Messages and Chat
Completions.

**Ships one minimal valid success response per protocol.** Without it, any request driven through a
real `BridgeServer` falls into the retry and failover paths — which are themselves body-mutating
(M6, M8, M9) — and no consumer could obtain a clean baseline. The *failure* library is **T-B4**.

Peer-port and casing capture are part of the **protocol**, enforced by a conformance test every
Epic B recorder must pass — not restated per recorder and forgotten.

*Done when:* casing and order survive capture; peer port is recorded; the conformance test exists
and a recorder that lowercases headers fails it. *Size:* M.

### T-W5 — Shared CONNECT proxy fixture

Extract `_ConnectProxy` and `_TlsTarget` from `tests/test_egress_https_proxy.py` into a shared
fixture. Extend it to record **the outbound source port of each tunnel** and to be stoppable
mid-test.

An extraction, not a rewrite: `test_egress_https_proxy.py` is the strongest existing asset in the
suite and must pass unchanged against the extracted fixture.

*Done when:* the existing module still passes; tunnel source port is recorded; stopping the proxy
produces a connection failure, not a hang. *Size:* M.

### T-W6 — Corpus capture procedure, scrubber and loader

The capture procedure, a **credential scrubber**, and a loader exposing entries **by the register
triggers they meet** (hence T-W3).

The scrubber is load-bearing: a fixture file is as public as the repository, and captured
transcripts carry prompts, file contents and API keys.

Also **names an owner and a cadence for corpus refresh**, tied to Claude Code releases. Design
§7.1 warns an un-refreshable corpus becomes a museum of a protocol nobody speaks; nothing else in
this plan schedules that.

*Done when:* the scrubber removes an API key, a bearer token and an absolute home path from a
synthetic transcript; a CI lint fails on an unscrubbed fixture. *Size:* M.

### T-W7 — The exemption registry

Four guards and two acceptance scenarios are *designed* to land red (§16). They need the exemption
mechanism before any of them can land, which makes this **enablement, not acceptance plumbing** —
it sat in Epic J in the first draft, which stranded three Wave-0 tasks.

Plain pytest, not BDD-specific: the Epic G guards are ordinary `l2` tests. An entry names **one
assertion**, its expected failure condition and its ticket. Setup and every other assertion in the
same test gate normally. **An unexpected pass fails the job** (`xfail(strict=True)` semantics). One
registry file.

*Done when:* a test with an exempt assertion **and** a second failing non-exempt assertion fails;
an exempt assertion that starts passing fails. Those two are the falsification cases — without
them this is the blanket amnesty the design rejects. *Size:* M.

### T-W8 — Bridge-under-test fixture

A profile/backend factory plus a real `BridgeServer` started against a named recorder, torn down
cleanly. Around fifteen tasks across Epics D, E, G, I, J and K need exactly this; `conftest.py`
offers only `sample_profile_dict` and `unused_tcp_port` today, and each of the ~40 files under
`tests/bridge/` builds its own. Without this task, whoever picks up T-D1 builds it, and T-E3,
T-G2 and T-I8 each build another.

Constructs a profile for any adapter × model × transport, points its backend at a recorder host,
starts the server in bridge mode, exposes both.

*Done when:* one fixture call yields a running bridge for any registered adapter, and a test
asserts teardown releases the port. *Size:* M.

---

## 3. Epic A — Wire projections

Six independent readers, parallel after **T-W2**. Each imports **nothing from `src/kitty/bridge`**
— that independence is the point (§3.3.1).

| ID | Reader | Depends on | Design | Size |
|---|---|---|---|---|
| **T-A1** | Anthropic Messages | T-W2 | §3.3.1 | M |
| **T-A2** | Chat Completions | T-W2 | §3.3.1 | M |
| **T-A3** | OpenAI Responses | T-W2 | §3.3.1 | M |
| **T-A4** | Gemini — **consumes the URL as well as the body**; model and operation live in the path | T-W2 | §3.3.5 | M |
| **T-A5** | Bedrock Converse | T-W2 | §3.3.1 | M |
| **T-A6** | Ollama `/api/chat` | T-W2 | §3.3.1 | S |

**Acceptance, all six:** round-trips that format's **published examples** into `Request` with an
empty residual — validated against the published schema, never against kitty's output. A reader
validated against kitty's output inherits kitty's bugs and the oracle becomes circular.

**Falsification, all six:** a body with one unrecognised key produces a non-empty residual and
fails the totality assertion.

**Deliberately *not* in scope here:** "residual empty across the whole corpus." That needs the
corpus and belongs to **T-D8**, which owns coverage. Putting it in T-A1's acceptance made the
projections silently depend on Epic C and dragged the corpus onto the critical path.

---

## 4. Epic B — Recorders and scripted responses

All depend on **T-W4** and must pass its recorder conformance test — including peer-port capture,
which T-E4's tunnel join needs from every recorder, not only the primary.

| ID | Task | Depends on | Design | Size |
|---|---|---|---|---|
| **T-B1** | Provider-aiohttp recorder — `ollama_cloud` and the `openai_subscription` **OAuth legs**, which run at startup before anything else is proven | T-W4 | §5.5, §7.2 | M |
| **T-B2** | curl_cffi-reachable recorder with harness TLS — the only place `_cc_to_responses` output (P13, P17) is observable | T-W4 | §7.2 | M |
| **T-B3** | botocore endpoint-override recorder — captures the Converse payload **after** the transport's `modelId`/`stream` pops (P18) | T-W4 | §3.2.3, §7.2 | M |
| **T-B4** | Scripted failure library — SSE variants, error statuses, Cloudflare blocks, empty responses, context-too-large rejections, mid-stream disconnect at each of the four §6.3.1 injection points | T-W4 | §6.3.1, §7.2 | M |

---

## 5. Epic C — Golden corpus

Parallel after **T-W6**. Coverage is driven by the register: every conditional row needs a trigger
case **and** a complement.

| ID | Entries | Depends on | Covers | Size |
|---|---|---|---|---|
| **T-C1** | Plain turn, tools declared, `tool_use`, `tool_result` | T-W6 | Complement for most conditional rows; M7 | S |
| **T-C2** | Thinking, image, `system` with `cache_control` | T-W6 | P2a/b, P5c–e, P8, M8 | S |
| **T-C3** | Tool result under/over 50,000 chars; transcript under/over the compaction budget | T-W6 | M3, M4, M5 — triggers and complements | M |
| **T-C4** | 400/413 recovery against a **balancing** profile; system prompt alone over the window; a single final turn over the budget | T-W6 | M6, M13, the irreducible-set case | M |
| **T-C5** | The vendor-string regression entry — a turn reading `Please explain how kitty-bridge works` | T-W6 | §3.3.3 | S |
| **T-C6** | `max_tokens` above/below 4096 × streaming/non-streaming; a malformed body | T-W6 | P7, P13, the L2 fuzz path | S |
| **T-C7** | Native Claude Code baseline — headers **and** connection pattern | T-W6 | Baselines for design channels C1b (header parity) and C5 (connection lifecycle) — consumed by T-I12 and T-I9 | M |

**T-C4's M6 entry must be authored against a balancing profile.** `_compact_with_tighter_budget`
is called only from `_request_with_retry_balancing`; a single-backend profile never reaches it and
the entry would silently never fire.

**T-C4 and T-C6 are deliberately synthesised, not captured.** A system prompt larger than the
window, and a malformed body, do not occur in a real session on demand — so design §7.1's
"real, not synthetic" rationale does not apply to them. Record that reasoning beside the fixtures;
the scrubber and lint rules still apply.

**Scheduling note:** T-C1–T-C7 share one capture-and-secret-review pass. Seven tasks, not seven
independent parallel slots.

---

## 6. Epic D — The fidelity oracle (I1)

| ID | Task | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|
| **T-D1** | Oracle core | T-W2, T-W3, T-W4, T-W8, T-A1, T-A2, T-C1 | `assert_no_unclaimed_mutation` runs end to end on one default-transport adapter with both §3.3.2 assertions. **Accepts `expected_route`** so T-D2 is a filling-in, not a signature change. **Includes the byte-level key-order assertion on the native passthrough path** — the one place the comparison is not projected | §3.3, §4.3 C2 | L |
| **T-D2** | Independent routing expectation | T-D1, T-A4 | Expected host/path/query computed from the profile using the provider's *published* URL shape — **never** by calling `build_base_url()` / `get_upstream_path()` | §3.3.5 | M |
| **T-D3** | Falsification suite | T-D1, T-D2 | Six cases, all failing the oracle, in the suite: changed model · flipped `stream` · deleted tool description · stripped `strict` · injected metadata field (non-empty residual) · **changed Azure deployment segment with a byte-identical body** | §3.3.1, §3.3.5 | M |
| **T-D4** | Parametrise — default aiohttp transport | T-D1, T-D2, T-A1–T-A4, T-B4, T-C1–T-C6, T-W8 | The 20 default-transport adapters at their representative models; `opencode_go` per route | §3.3.4 | L |
| **T-D5** | Parametrise — curl_cffi | T-D4, T-B2, T-A3 | `openai_subscription` serving path, observing P13–P17 | §3.3.2 | M |
| **T-D6** | Parametrise — botocore | T-D4, T-B3, T-A5 | `bedrock`, observing the Converse payload after P18 | §3.3.2 | M |
| **T-D7** | Parametrise — provider aiohttp | T-D4, T-B1, T-A6 | `ollama_cloud` after P19; the subscription OAuth leg | §3.3.2 | M |
| **T-D8** | Coverage checker | T-D4, T-W3, T-C1–T-C6 | A meta-test failing when any conditional register row lacks a trigger case or a complement. **Also owns "residual empty across the whole corpus"** for all six readers | §3.3.4 | M |

**T-D3 lands immediately after T-D1/T-D2, not at the end.** It is what distinguishes an oracle from
a decoration.

**T-D4 was one L covering four transports.** Split per transport so each is landable and each
extends the parametrisation, rather than one task with thirteen predecessors on the critical path.

---

## 7. Epic E — Containment (I3)

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-E1** | Hostname harness + aiohttp direct route | | T-W5, T-W4, T-W8 | Upstream addressed as `upstream.kitty-test.invalid`; direct leg resolves via a monkeypatched aiohttp resolver — **not** `/etc/hosts`, which needs admin rights and is unavailable on most CI runners | §5.3 | L |
| **T-E2** | curl_cffi and botocore direct routes | | T-E1, T-B2, T-B3 | curl's `resolve` mapping and a botocore `endpoint_url` override. **A transport that cannot be given a working direct route is reported `unproven`, not passed** | §5.3, §5.5 | M |
| **T-E3** | Phase 1 — positive controls | | T-E1, T-E2 | With egress **disabled**, each transport reaches the upstream directly and the connection is recorded | §5.2.2 | M |
| **T-E4** | Phase 2/2b — containment and tunnel correlation | | T-E3, T-W5 | Proxy down ⇒ **zero** upstream connections and the request fails. Proxy up ⇒ every accepted connection joins a tunnel on the recorded source port, N requests per tunnel allowed, failed tunnels contributing none | §5.2.1, §5.2.2 | L |
| **T-E5** | Phase 3 — falsification | | T-E4 | A deliberately injected bypass — patched `should_bypass`, or a session built without the proxy — makes the harness fail | §5.2.2 | M |
| **T-E6** | Guard **enforcement** | | — | Each of the five start paths driven with a rejecting configuration asserts **no server starts**: no listening socket, non-zero exit. Falsification: a variant keeping the `egress_block_reason()` call and discarding its return value must make these fail. **No recorder needed — nothing reaches upstream by construction** | §6.2.3 | M |
| **T-E7** | AST start-path domination guard | | — | Every `BridgeServer(` construction is dominated by an `egress_block_reason(` call at AST level, not file level. Falsification: move a construction above its guard call in the same file | §5.1, §6.2.3 | M |
| **T-E8** | Local bypass, fail-closed, and the transport asymmetry | | T-E1 | Loopback/`localhost` provider connects directly **on bridge sessions**; a `supports_egress() == False` profile blocks startup **and the message names the profile** (EG-3's assertion); and the complement — those destinations **are** tunnelled on curl_cffi, botocore and provider-aiohttp, which have no bypass. Pinning the asymmetry stops a future "fix" quietly adding one | §5.5, §5.2.2 | M |

**T-E6 and T-E7 are complements and both are needed.** T-E7 proves the guard is *called*; T-E6
proves its answer is *obeyed*. Deleting `if egress_error: return 1` passes T-E7 and fails T-E6.
Both need nothing and belong in Wave 0.

---

## 8. Epic F — L1 component and property

| ID | Task | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|
| **T-F1** | `hypothesis` in the dev extra + a shared strategy library for Messages/CC transcripts | — | Strategies reusable by T-F2–T-F5 | §6.1 | M |
| **T-F2** | Compaction properties | T-F1 | Identity below budget · no orphaned pair · idempotent · **output ≤ budget unless the surviving set is irreducible**, stated with the exception | §6.1 | M |
| **T-F3** | Pairing and truncation properties | T-F1 | No orphan in either shape; truncation identity below the limit, bounded above it | §6.1 | M |
| **T-F4** | Egress properties | T-F1 | Private ranges bypassed; **no hostname outside the `localhost` family bypassed**; `parse_proxy_url` round-trip; **structural** redaction — the password component of `masked()` is exactly the mask, never a substring test | §5.3, §6.1 | M |
| **T-F5** | `describe_tool_input_anomaly` property | T-F1 | Never reports an anomaly for input valid against the declared schema | §6.1 | S |
| **T-F6** | Translator semantic property via the projections | T-F1, T-A1, T-A2 | `project_messages(inbound)` equals `project_cc(translate_request(inbound))` — the projections, never a translator round-trip | §3.3.1, §6.1 | M |

**T-F4's `should_bypass` property is load-bearing beyond L1**: it is the premise T-E1's harness
rests on. If someone adds name resolution to `should_bypass`, T-F4 fails and T-E1's design is
re-examined — rather than T-E1 quietly going vacuous.

---

## 9. Epic G — L2 contract

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-G1** | README ⇄ code table guards | defect | T-W7 | Endpoint, attribution-header, env-var and logging-flag tables. Lands red against the endpoint table → exemption (KBR-9 / gap G6) | §6.2.3 | M |
| **T-G2** | Register completeness — coverage over the captures | | T-W3, T-D4 | Per adapter × model × transport, the captured delta equals the union of that adapter's rows whose triggers the input met. **A meta-assertion over T-D4–T-D7's captures, marked `l3`** — it does not re-drive the wire, and it must not put real sockets in the fast gate | §6.2.3 | M |
| **T-G3** | Internal-key completeness AST guard | defect | T-W7 | Every `_`-prefixed key written into a request dict in `bridge/**` is in `_INTERNAL_KEYS`. Lands red → exemption (KBR-6 / gap G15) | §6.2.3 | M |
| **T-G4** | Wire-shape honesty guard | defect | T-W7, T-W8, T-B1–T-B3 | `upstream_wire_is_messages_api` agrees with the observed output shape per adapter × representative model. Lands red → exemption (KBR-7 / gap G16) | §6.2.3 | M |
| **T-G5** | Bridge-introduced vendor token guard | defect | T-D1, T-W7, T-C5 | No **bridge-introduced** content contains `kitty` in any casing, scoped by the projection diff — **and, in the same run**, an inbound turn containing `kitty-bridge` survives byte-identically. A harness that cannot do both at once has not solved the problem. Lands red → exemption (KBR-5 / gap G14) | §3.3.3, §6.2.3 | M |
| **T-G6** | OpenAPI 3.1 document + schemathesis | | T-W8 | Five POST routes plus `/healthz`, `/stats`, `/v1/models`, **targeting a bridge started in bridge mode**; a separate guard asserts the per-protocol registration matrix | §6.2.1 | L |
| **T-G7** | SSE grammar state machine | | T-B4, T-W8 | Every stream — including error streams, mid-stream failover and the empty-response fallback — is a sentence in the Anthropic event grammar, across all three streaming protocols | §6.2.2 | M |
| **T-G8** | Dependency behaviour contracts | | T-W5 | aiohttp session proxy across `>=3.11,<3.14`; `curl_cffi` `proxies=` precedence over ambient `HTTP_PROXY`/`NO_PROXY`; `botocore Config(proxies=)` precedence; `keyring` backends. **Also declares `botocore` explicitly** — a containment guarantee currently rests on an undeclared transitive dependency | §6.2.4 | M |
| **T-G9** | **Header contract** | defect | T-W7 | Per adapter, an **exact** header set — names present, names absent, casing, value shape. Plus: no adapter derives a `User-Agent` or version header from `kitty.__version__`, and where both a UA version and a `version` header are sent they agree. **This is the test that catches KBR-8.** Lands red → exemption (gap G3) | §4.3 C1 | M |

---

## 10. Epic H — Mutation validation

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-H1** | `mutmut` config, L1 selection, baseline | | T-W1 | `[tool.mutmut]` with array `source_paths` and `pytest_add_cli_args_test_selection = ["-m", "l1"]`; a recorded baseline per target group | §6.1 | M |
| **T-H2** | Extract pure payload builders | **src** | T-B3 | `_bedrock_body(...)` and `_ollama_body(...)` extracted from `make_request`/`stream_request`. **Depends on T-B3 because "behaviour unchanged" needs a characterisation test** — T-B3 is what makes the post-mutation Converse payload observable | §6.1 | M |
| **T-H3** | Per-component thresholds + CI reporting | ci | T-H1, T-H2, T-F2–T-F6 | ≥ 85% killed **per target group**, not one aggregate; survivor triage documented; `mutmut export-cicd-stats` in the nightly job | §6.1 | M |
| **T-H4** | Measure changed-code mutation runtime | resolves Q11 | T-H1 | Timed `mutmut run` restricted to a representative PR's functions, against the fast gate's budget | §6.1 | S |

---

## 11. Epic I — L3 subsystem

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-I1** | Settings lifecycle | | — | Normal exit, `SIGTERM`, `SIGKILL` + `kitty cleanup`; global settings byte-identical before and after; `_kitty_values_present` fires only on kitty-written state | §6.3.2 | M |
| **T-I2** | Concurrent sessions | | T-I1 | Two sessions, separate `--settings` files, neither touches `~/.claude/settings.json`, the second does not disturb the first (issue #22) | §6.3.2 | M |
| **T-I3** | `prepare_launch` failure fails the launch | | — | When the session file cannot be written, launch **fails** rather than running on the user's own Anthropic credentials | §6.3.2 | S |
| **T-I4** | Background bridge ownership | | — | A bridge owned by another user is not stopped, not restarted, no second bridge starts beside it | §6.3.2 | S |
| **T-I5** | Agent startup smoke | blocked Q12 | T-W4, T-W8, T-B4 | Pinned real Claude Code binary, one non-interactive turn against the recorder, clean exit | §6.4.2 | M |
| **T-I6** | Agent settings precedence | blocked Q12 | T-I5 | Three runs, three winners: session file beats both; without it the global file beats the environment; without either the environment wins. Every sentinel demonstrated live | §6.4.2 | M |
| **T-I7** | Streaming recovery — content | partial Q14 | T-B4, T-G7, T-W8 | Four injection points; no duplicated text, no tool-call id reused across attempts, no spliced arguments, exactly one terminal outcome. **Positive post-emission oracle waits on Q14**; until then the negatives only | §6.3.1 | L |
| **T-I8** | Cross-attempt content and cadence | | T-D1, T-W8, T-B4 | Transport-blip and empty-response retries byte-identical; M6, M8, M9 and failover re-normalisation each fire only under their own trigger | §4.3 C3 | M |
| **T-I9** | Connection lifecycle baseline | | T-W4, T-W8, T-C7 | Distinct TCP connections per N-turn session against the native capture; reported baseline, ratcheted | §4.3 C5 | M |
| **T-I10** | `_backend_context` isolation | | T-W8 | Concurrent requests never observe each other's backend selection. Deterministic — belongs here, not in the load profile | §6.3.1 | S |
| **T-I11** | Failover, disconnect, error envelopes | | T-B4, T-W8 | Streaming failover mid-response; client disconnect releases the upstream connection without marking the backend unhealthy; all-backends-unhealthy 503 in each protocol's native envelope; oversized request in the protocol's own error shape | §6.3.1 | M |
| **T-I12** | Fingerprint parity report | | T-C7, T-W8, T-G9 | kitty's header set compared against the native baseline; **reported with a ratchet, not gating**, until gap G3 closes | §4.3 C1b | M |
| **T-I13** | Side traffic | | T-W4, T-W8 | `GET /healthz` and `GET /stats` never cause an upstream request; a launch contacts the provider only for the agent's turns plus pre-flight validation, pinned as a declared exception; `kitty --no-validate` removes it | §4.3 C6 | M |

---

## 12. Epic J — L4 acceptance

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-J1** | `pytest-bdd` wiring and step definitions | | T-W1, T-W7 | Gherkin runs under pytest; steps bind to the L3 harnesses rather than re-implementing them. The exemption mechanism itself is T-W7 | §6.4.1 | M |
| **T-J2** | TR scenarios — fidelity and indistinguishability | defect | T-J1, T-D4, T-C7, T-G9 | TR-1, TR-1c, TR-2, TR-3, TR-4. TR-1c and TR-4 land with exemptions (KBR-8, KBR-5). **No egress dependency — none of the TR scenarios involves containment** | §6.4.1 | M |
| **T-J3** | EG scenarios — containment | | T-J1, T-E3, T-E4, T-E6, T-E8 | EG-0 (the reachability control), EG-1, EG-2, EG-3. T-E8 supplies EG-3's "names the profile" assertion | §6.4.1 | M |

---

## 13. Epic K — Evals, load and CI wiring

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-K1** | Eval harness skeleton | | — | Two arms; model, provider, dataset and sampling pinned and recorded per run; failure taxonomy captured per arm | §6.4.3 | M |
| **T-K2** | Independently authored task set | | T-K1 | Coding tasks whose acceptance tests are **written by a person**. A model-generated test passing the model's own code shares its misunderstandings | §6.4.3 | L |
| **T-K3** | Statistics and decision rule | blocked Q4, Q13 | T-K1, T-K2 | Successes ÷ **scheduled** trials; repetition; confidence interval; pre-registered margin; symmetric exclusion rule; missing-data ceiling that **voids** the run | §6.4.3 | M |
| **T-K4** | Load rig and baseline | | T-W4, T-W8, T-B4 | Fixed workload and named runner class; bridge-added latency p50/p95, TTFB, completion and error rates, bounded RSS, socket/fd recovery; streaming and buffered paths measured separately | §6.4.4 | L |
| **T-K5** | `load.yml` reusable + publish dependency | ci | T-K4 | `publish.yml` `needs:`-gates on a successful load run **for the tag commit**; a nightly caller warns but does not substitute | §8 | M |
| **T-K6** | `tests.yml` gains Subsystem and Acceptance | ci | T-W1, T-E4, T-J3, T-I5 | Both gate PRs and releases through the one reusable workflow. **T-I5 is a dependency because `agent_smoke` is in the Acceptance selection**: if it never lands the selection collects nothing, pytest exits 5, and T-K8 turns that into a hard failure. Either T-I5 lands first, or T-K6 ships with `acceptance` only and T-I5 adds `agent_smoke` | §8 | M |
| **T-K7** | Nightly workflows | ci | T-H3, T-K1 | Deep (mutation, schemathesis at high `--max-examples`), Agent-live, Eval. **Agent-live runs the existing `tests/integration/test_agent_e2e.py`, which needs no pinned binary** — so this does not wait on Q12. None of these gate | §8 | M |
| **T-K8** | Skip-is-failure enforcement | ci | T-K6 | A gating job whose prerequisite is missing **fails**. A green tick meaning "we did not test this" is worse than a red one | §8 | S |

---

## 14. Sequencing

### 14.1 Dependency graph, condensed

```
T-W1 ──────────────────────────────────────► everything (markers)
T-W1 ──► T-W7 ──► T-G1, T-G3, T-G4, T-G5, T-G9, T-J1
T-W2 ──► T-A1..T-A6 ──┐
T-W3 ─────────────────┼──► T-D1 ──► T-D2 ──► T-D3
T-W4 ──► T-B1..T-B4 ──┤          └─► T-D4 ──► T-D5/6/7, T-D8, T-G2 ──┐
T-W4 ──► T-W8 ────────┤                                              ├──► T-J2 ──┐
T-W6 ──► T-C1..T-C7 ──┘                                 T-C7, T-G9 ──┘           │
                                                                                 │
T-W5 ──► T-E1 ──► T-E2 ──► T-E3 ──► T-E4 ──► T-E5                                │
                    └────► T-E8 ──┐                                              │
              T-E6, T-E7 ─────────┴──► T-J3 ──► T-K6 ──► T-K8 ◄──────────────────┘

independent from day one:  T-F1 ──► T-F2..T-F5      T-I1 ──► T-I2
                           T-I3, T-I4, T-K1, T-E6, T-E7
```

### 14.2 Waves

| Wave | Content | Parallel capacity |
|---|---|---|
| **0** | T-W1–T-W8, T-E6, T-E7, T-F1, T-I1, T-I3, T-I4, T-K1 | ~10 — T-W* is the constraint, staff it first |
| **1** | T-A1–T-A6, T-B1–T-B4, T-C1–T-C7, T-E1, T-F2–T-F5, T-G1, T-G3, T-G8, T-G9, T-H1, T-H2, T-I2, T-K2 | **~15**, the widest point — though T-C1–T-C7 share one capture pass and are not seven independent slots |
| **2** | T-D1, T-E2, T-E3, T-F6, T-G4, T-G6, T-I10, T-I13, T-J1 | ~9 |
| **3** | T-D2, T-D3, T-D4, T-E4, T-E8, T-G5, T-G7, T-H3, T-I8, T-I9, T-I11, T-I12, T-K4 | ~12 |
| **4** | T-D5, T-D6, T-D7, T-D8, T-E5, T-G2, T-I5, T-I6, T-I7, T-J2, T-J3, T-K5 | ~11 |
| **5** | T-K6, T-K7, T-K8, T-K3, T-H4 | ~5 — CI wiring plus the decision-blocked tail |

### 14.3 The two long chains

Two chains of comparable length; the containment one is longer. Both end at CI wiring, not at the
acceptance scenario — a scenario that gates nothing is not delivered.

```
Containment  T-W4(M) → T-W8(M) → T-E1(L) → T-E2(M) → T-E3(M) → T-E4(L) → T-J3(M) → T-K6(M) → T-K8(S)
             ≈ 17–18 days

Fidelity     T-W2(S) → T-A1(M) → T-D1(L) → T-D2(M) → T-D4(L) → T-J2(M) → T-K6(M) → T-K8(S)
             ≈ 15 days
```

They converge only at **T-K6**. Nothing in one waits on anything in the other before that, so
**running them as two streams with different owners roughly halves elapsed time on the two hardest
parts of the suite** — the single biggest scheduling lever in this plan.

Four consequences worth acting on:

1. **T-W2, T-W4, T-W5 and T-W8 head the chains and are all small.** They go first, and no two of
   them to the same person.
2. **T-A1 and T-A2 gate T-D1**, so those two projections go before the other four, which can be
   built during T-D1's construction.
3. **T-E1 is the riskiest task in the plan** and sits second in the longer chain. If it slips,
   containment slips one-for-one. Start it earliest, review it hardest.
4. **T-K6 is a convergence point and a single point of failure.** Both chains stop there; it
   should not be scheduled as an afterthought.

---

## 15. Blocked on decisions, not on code

| Question | Blocks | Cost of leaving it open |
|---|---|---|
| **Q4** + **Q13** — eval margin, compaction baseline | T-K3 | The eval runs and cannot conclude. T-K1/T-K2 proceed |
| **Q10** — irreducible final turn | T-F2's exact bound, TR-3's wording, register rows M3–M7/M13 | T-F2 lands with the observed-behaviour property and is revised once — but it must be revised **together with** TR-3 and the register |
| **Q11** — per-PR mutation cadence | Nothing. **T-H4 answers it** | T-H3 lands nightly-only; a later change moves it |
| **Q12** — pinned Claude Code binary | T-I5, T-I6 — and therefore T-K6's `agent_smoke` selection | **The most expensive open question.** It blocks the only test of a third-party behaviour the product depends on, and it reaches into CI wiring |
| **Q14** — post-emission stream recovery | T-I7's positive assertions | T-I7 lands asserting only the negatives: catches corruption, cannot confirm correctness |

**Q1** (agent identity) blocks no task but decides whether T-I12 and T-J2's TR-1c exemption ever
become gating. **Q5, Q6, Q7** affect register rows and residual-risk wording, not delivery.

---

## 16. Relationship to the open defects

Five defects are filed. The dependency runs one way only:

- **The guard lands first, red, with an exemption.** Writing it *after* the fix proves nothing
  about whether it would have caught the defect — and the fix then has no regression test that
  ever failed.
- **The defect's ticket removes the exemption**, as its acceptance criterion. T-W7's
  unexpected-pass rule makes forgetting impossible.

| Defect | Design gap | Guard that lands red | Exemption removed by |
|---|---|---|---|
| KBR-5 — vendor string upstream | G14 | T-G5, T-J2 (TR-4) | KBR-5's fix |
| KBR-6 — internal keys on the wire | G15 | T-G3 | KBR-6's fix |
| KBR-7 — wire-shape mismatch | G16 | T-G4 | KBR-7's fix |
| KBR-8 — contradictory client versions | G3 | T-G9, T-J2 (TR-1c) | KBR-8's fix |
| KBR-9 — README endpoint table | G6 | T-G1 | KBR-9's fix |

---

## 17. Risks in the sequencing

**T-E1 is the hardest single task and the easiest to underestimate.** Direct-route resolution needs
a different mechanism per transport, and T-E2 is explicit that a transport which cannot be given
one is reported **unproven** rather than passed. Splitting T-E1/T-E2 means a stuck transport is one
blocked M, not a blocked L.

**The corpus is the quiet long pole.** T-C1–T-C7 are individually small but need real captured
sessions, a secret review, and an owner for refresh. Start T-W6 on day one even though nothing
visibly depends on it yet.

**T-D4–T-D7 and T-G2 are adjacent and easy to duplicate.** T-D4–T-D7 own driving the wire and
capturing; T-G2 owns asserting register-row coverage over those captures, as a meta-assertion. Two
teams implementing "per adapter × model × transport" independently is the most likely wasted work
in this plan — and if T-G2 re-drives the wire under an `l2` marker it puts real sockets in the
fast gate.

**T-H2 is the only source change.** Keep it out of every test PR so a revert is cheap. It now
depends on T-B3 so "behaviour unchanged" has evidence behind it rather than assertion.

**The existing suite already runs ~18.5 minutes per Python version.** Wave 3 onward adds real
sockets to the PR gate. T-K6 should land with a measured runtime; if the gate crosses the point
where people route around it, the marker split is the mechanism for pulling work back to nightly —
that is what T-W1 buys beyond tidiness.
