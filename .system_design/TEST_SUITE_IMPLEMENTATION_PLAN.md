# Kitty Bridge — Test Suite Implementation Plan

**Implements:** [`TEST_SUITE.md`](TEST_SUITE.md). That document says what the suite must prove and
why; this one says who can build what, in what order, without waiting on each other.
**Traces to:** [KBR-2](https://shelpuk.atlassian.net/browse/KBR-2).
**Status:** Plan. No Jira issues created — the `T-` ids are plan-local and become ticket summaries
when the plan is accepted.

> **Identifiers.** Every task id is prefixed `T-`. `TEST_SUITE.md` already uses bare `C1–C6` for
> observable channels, `F1–F5` for findings, `G1–G19` for gaps and `I1–I3` for the invariants. An
> earlier draft collided with all of them. The **Design** column carries the reverse link.

> **The epic tables in §4–§13 are the single source of truth for dependencies.** §14's tiers and
> chains are *derived* from them — an earlier draft had a graph, a wave table and a duration
> estimate that were three different schedules.

---

## 1. How to use this plan

### 1.1 Schedule by readiness, not by wave

§14 groups tasks into tiers, but **a tier is a derived view, not a gate**. A task is available the
moment its own dependencies are done. Nobody waits for a tier to finish.

### 1.2 Milestone 0 is contracts plus one proven slice

Almost every interesting test needs shared infrastructure that does not exist. Milestone 0 (§3)
delivers the **shared contracts** — types, schemas, protocols, extension interfaces — plus **one
working vertical slice with a falsification case**, so later streams integrate against an
interface proven once rather than against a promise.

Give each shared file a named integration owner. Give each stream its own modules.

### 1.3 Definition of done

1. The deliverable satisfies the acceptance criteria in its row.
2. `ruff`, `lint-imports`, `mypy src/kitty` and the full suite pass on Python 3.10–3.13.
3. Every new test carries exactly one layer marker (**T-W1**).
4. Google-style docstrings throughout; block comments explaining *why*. Test code is code.
5. **It lands on `main` alone**, without leaving the suite red waiting for a sibling.

### 1.4 The harness rule

Four review rounds on the design produced four harnesses that would have passed while proving
nothing: an oracle composed of two functions that were not inverses; a containment test asserting
nothing arrived at a destination nothing could reach; a guard proving a function was *called* when
the enforcement was the branch after it; a projection that could not see the model name, in a
product whose purpose is changing the model name.

So:

> **The first working version of every harness ships with at least one falsification case — a
> deliberate defect it must detect, running in the suite.** The dedicated falsification task adds
> the rest, and **is a hard prerequisite for that harness gating anything downstream.**

An earlier draft made falsification mandatory in the definition of done and optional in the
dependency graph, so both major harnesses could have become gating infrastructure before their
falsification suites landed. The dependency rows below close that.

### 1.5 Flags and sizing

| Flag | Meaning |
|---|---|
| **src** | Touches `src/kitty/`; needs a `code-reviewer` pass |
| **ci** | Touches `.github/workflows/`; verify on a branch first |
| **blocked** | Cannot start until an open question is answered (§15) |
| **partial** | Lands, but a stated part of its acceptance waits on a decision |
| **defect** | Relates to an open defect (§16) |
| **resolves** | Its output answers an open question |

**S** ≈ 0.5 day · **M** ≈ 1.5 days · **L** ≈ 4 days.

---

## 2. Streams

After Milestone 0, seven streams advance independently, each owning its own modules.

| Stream | Epics | Needs from Milestone 0 |
|---|---|---|
| **Fidelity** | A, D | Types, register, recorder, bridge fixture |
| **Containment** | E | CONNECT proxy, bridge fixture |
| **Corpus** | C | Corpus format and scrubber |
| **Properties** | F | Nothing beyond `hypothesis` |
| **Contracts** | G | Exemption registry, proxy fixture |
| **Lifecycle** | I | Bridge fixture — several tasks need nothing |
| **Evaluation** | H, K | Markers |

---

## 3. Milestone 0 — shared contracts and one proven slice

| ID | Task | Depends on | Design | Size |
|---|---|---|---|---|
| **T-W1** | Layer markers, CI selection rules, per-category collection checks | — | §8, §8.1, §8.2 | M |
| **T-W2** | **The input contract** — request/capture types, the projection protocol, the path vocabulary and the normalisation rules | — | §3.3.1, §3.3.1a, §3.3.1b | ~~S~~ **M** |
| **T-W3** | Register schema and data | T-W2 | §3.2 | M |
| **T-W4** | Recorder implementation — primary aiohttp recorder | T-W2 | §7.2 | M |
| **T-W5** | Shared CONNECT proxy fixture | — | §7.3 | M |
| **T-W6** | Corpus format, capture procedure, scrubber, loader | T-W3 | §7.1 | M |
| **T-W7** | Assertion exemption registry | T-W1 | §8 | M |
| **T-W8** | Bridge fixture **core + transport extension interface** | T-W4 | §6.3.1 | M |
| **T-W9** | **The proven vertical slice** | T-W1, T-W2, T-W4, T-W8 | §6.3.1 | S |

**T-W1 — markers and CI selection.** Register `l1`, `l2`, `l3`, `acceptance`, `agent_smoke`,
`agent_live`, `eval`, `load`; a meta-test asserts every collected test carries exactly one. **Do
not hand-edit 2,880 test functions** — default by path in `pytest_collection_modifyitems`, and
require an explicit marker only where the default is wrong.

This also establishes the **CI selection mechanism and per-category collection checks up front**,
not in Epic K: parallel authors need the divided test command from day one. And a category that
must be non-empty needs its own check — an earlier draft claimed `pytest -m "acceptance or
agent_smoke"` would exit 5 when agent-smoke tests were missing. **That is wrong.** Once acceptance
tests exist the expression collects them and passes happily with zero agent-smoke coverage. Also
reconcile the existing `--runslow` silent skip with the same rule.

**T-W2 — the input contract.** One owner for everything a projection reads and a recorder produces:
`Request(envelope, conversation, residual, consumed, source)`, `Envelope`, `Conversation`, `Turn`,
`Part` variants, `Reply` for the response direction, **`CapturedRequest(method, scheme, host, path,
query, headers, body)`**, `CapturedReply`, the closed `WireFormat` enum, the `Projection` and
`ReplyProjection` protocols, **the path vocabulary and its pattern matcher (§3.3.1a)**, and **the
cross-format normalisation rules (§3.3.1b)**.

**Delivered in `tests/harness/contract.py`.** The **package** `tests/harness/` is the home for
T-W4's recorder, T-W5's proxy fixture, T-W6's corpus loader and T-W8's bridge fixture — each in its
own module beside the contract, not inside it. `contract.py` defines shapes and rules; it captures
nothing and reads no bodies.

**The path vocabulary and the normalisation rules were added after a design review**, and both are
here for the reason `CapturedRequest` is: T-W3 cannot fill its "projection field it touches" column
without a path form, and T-D1 must emit the same strings when it reports a delta — neither can
define the vocabulary without the other agreeing. Likewise six readers written by six authors
produce incomparable output unless the canonical roles, placements and sampling spellings are fixed
once. Both are exactly the coordination problem Milestone 0 exists to remove.

**T-W3 inherits two acceptance criteria from this work.**

1. Every row M1–M14 and P1–P21 carries either a path in T-W2's vocabulary or `not projectable`
   **with a reason**, asserted against the register data so a new row cannot escape it. T-W2 proves
   the vocabulary is *expressive*; the row-by-row assignment is T-W3's, because a copy of that
   table inside T-W2 would be a second source of truth that stays green while the register moves.
2. **Every row is anchored at the *narrowest* path that covers its effect.** A pattern is a prefix
   (§3.3.1a), so a coarser anchor silently claims every delta beneath it — anchoring P15 at
   `conversation.tools[*]` rather than `conversation.tools[*].strict` would claim a *deleted tool
   description*, which is one of §3.3.1's own five oracle falsification cases. The matcher cannot
   detect this, by construction; **T-D3 gains the paired falsification case** — mutate a field
   beneath a registered row's anchor and assert the oracle still fails.

**`CapturedRequest` lives here, not in T-W4.** An earlier draft titled this task "request/capture types" while T-W4 defined `CapturedRequest` with no dependency between them — so the recorder author and the Gemini reader author (who needs the URL, §3.3.5) could have started against two different readings of the same contract. That is precisely the coordination problem Milestone 0 exists to remove.

Includes the **totality rule**: every key classifies into envelope, conversation or residual, and a
non-empty residual raises. *Falsification:* a stub reader that drops an unknown key fails it.

**T-W3 — register.** One entry per row M1–M14 and P1–P21: id, site symbol, trigger predicate, the
projection field it touches, conditional or not, design anchor. Defines the trigger vocabulary
T-W6 indexes by. *Falsification:* delete a row from the markdown or the data; the agreement test
fails.

**T-W4 — recorder implementation.** Implements T-W2's `CapturedRequest` — original casing and order,
arrival timestamp, and **the peer port of the accepted connection** — for the primary aiohttp
recorder. It consumes the contract rather than defining it. Ships **one minimal valid success
response per protocol** — without it any request
driven through a real bridge falls into the retry paths, which are themselves body-mutating. The
failure library is T-B4. Peer-port and casing capture are enforced by a **conformance test every
Epic B recorder must pass**.

**T-W5 — CONNECT proxy.** Extract `_ConnectProxy`/`_TlsTarget` from
`tests/test_egress_https_proxy.py`; add tunnel source-port recording and mid-test stoppability. An
extraction, not a rewrite: that module must pass unchanged.

**T-W6 — corpus.** Format, capture procedure, credential scrubber, and a loader indexing entries
by register trigger. Names an **owner and cadence for refresh**. *Falsification:* an unscrubbed
fixture fails a CI lint.

**T-W7 — exemption registry.** Plain pytest, not BDD-specific. An entry names **one assertion**,
its expected failure condition and its ticket; everything else in the test gates normally; an
unexpected pass fails (`xfail(strict=True)`). *Falsification:* a second, non-exempt failing
assertion in the same test must still fail the job.

**T-W8 — bridge fixture core and extension interface.** A profile/backend factory and a real
`BridgeServer` started against a recorder, for the **default aiohttp transport only**, plus the
**extension interface** custom transports plug into. An earlier draft promised "any registered
adapter" while depending only on T-W4 — which would have meant implementing T-B1–T-B3 inside the
shared fixture. Defining the interface here lets recorder authors integrate without editing the
core.

**T-W9 — the proven vertical slice.** Drive one request from the bridge fixture into the recorder
and assert the capture is complete and **type-compatible with T-W2's declared contract** — the
recorder's output must satisfy what a projection expects to read. Small, but it is the first
moment the contracts are known to compose rather than merely to exist.
*Falsification:* a recorder that drops the query string fails it.

---

## 4. Epic A — Projections

| ID | Task | Depends on | Design | Size |
|---|---|---|---|---|
| **T-A1** | Anthropic Messages reader | T-W2 | §3.3.1 | M |
| **T-A2** | Chat Completions reader | T-W2 | §3.3.1 | M |
| **T-A3** | OpenAI Responses reader | T-W2 | §3.3.1 | M |
| **T-A4** | Gemini reader — **consumes the URL as well as the body** | T-W2 | §3.3.5 | M |
| **T-A5** | Bedrock Converse reader | T-W2 | §3.3.1 | M |
| **T-A6** | Ollama `/api/chat` reader | T-W2 | §3.3.1 | S |
| **T-A7** | **Response-direction `Reply` projection** — parts, stop reason, usage | T-W2 | §3.3.1 | M |

**All readers:** validated against that format's **published examples**, never against kitty's
output — a reader validated against kitty's output inherits kitty's bugs and the oracle becomes
circular. *Falsification:* one unrecognised key produces a non-empty residual.

"Residual empty across the whole corpus" belongs to **T-D8**, not here; putting it in T-A1's
acceptance made the projections silently depend on Epic C.

---

## 5. Epic B — Recorders

Each task delivers a recorder **and its bridge-fixture integration** through T-W8's extension
interface, and must pass T-W4's conformance test — including peer-port capture, which T-E2's
tunnel join needs from every recorder.

| ID | Task | Depends on | Design | Size |
|---|---|---|---|---|
| **T-B1** | Provider-aiohttp recorder + integration — `ollama_cloud`, and the `openai_subscription` **OAuth legs** that run at startup | T-W4, T-W8 | §5.5, §7.2 | M |
| **T-B2** | curl_cffi recorder + integration, harness TLS — the only place `_cc_to_responses` output is observable | T-W4, T-W8 | §7.2 | M |
| **T-B3** | botocore recorder + integration — captures the Converse payload **after** the transport's `modelId`/`stream` pops | T-W4, T-W8 | §3.2.3, §7.2 | M |
| **T-B4** | Scripted failure library — SSE variants, errors, Cloudflare, empty responses, context-too-large, mid-stream disconnect at each of the four injection points | T-W4 | §6.3.1 | M |

---

## 6. Epic C — Corpus

| ID | Entries | Depends on | Covers | Size |
|---|---|---|---|---|
| **T-C1** | Plain turn, tools declared, `tool_use`, `tool_result` | T-W6 | Complement for most conditional rows; M7 | S |
| **T-C2** | Thinking, image, `system` with `cache_control` | T-W6 | P2a/b, P5c–e, P8, M8 | S |
| **T-C3** | Tool result under/over 50,000 chars; transcript under/over the compaction budget | T-W6 | M3, M4, M5 | M |
| **T-C4** | 400/413 recovery on a **balancing** profile; a conversation whose only remaining turn is an unpaired tool result; a single final turn over budget | T-W6 | M6, irreducible set (M13 withdrawn by KBR-5). **The corpus entry changed**: "system prompt alone over the window" does not empty the conversation — measured in KBR-5, see F3 | M |
| **T-C5** | The vendor-string entry — `Please explain how kitty-bridge works` | T-W6 | §3.3.3 | S |
| **T-C6** | `max_tokens` above/below 4096 × streaming/non-streaming; a malformed body | T-W6 | P7, P13, fuzz path | S |
| **T-C7** | Native Claude Code baseline — headers and connection pattern | T-W6 | Baselines for design channels C1b and C5 | M |

**T-C4's recovery entry must use a balancing profile** — `_compact_with_tighter_budget` is reached
only from `_request_with_retry_balancing`. **T-C4 and T-C6 are deliberately synthesised**: a
system prompt over the window and a malformed body do not occur in a real session on demand, so
design §7.1's "real, not synthetic" rationale does not apply; record that beside the fixtures.

**T-C1–T-C7 share one capture-and-secret-review pass** — seven tasks, not seven parallel slots.

---

## 7. Epic D — Fidelity oracle

**T-D4–T-D7 are siblings, not a chain.** Each consumes the oracle interface plus its own
transport's prerequisites. An earlier draft made T-D5–T-D7 wait for the full 20-adapter default
matrix, serialising unrelated provider coverage.

| ID | Task | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|
| **T-D1** | Oracle core **+ first falsification case** | T-W2, T-W3, T-W8, T-W9, T-A1, T-A2, T-C1 | Both §3.3.2 assertions on one adapter; accepts `expected_route` so T-D2 is a filling-in; includes the byte-level key-order assertion on the native passthrough path; **a changed model must fail the oracle** | §3.3, §4.3 C2 | L |
| **T-D2** | Routing expectation **+ its falsification** | T-D1, T-A4 | Route derived from the profile using the provider's *published* URL shape, never `build_base_url()`; a changed Azure deployment segment with a byte-identical body must fail | §3.3.5 | M |
| **T-D3** | Remaining falsification cases | T-D1, T-D2 | Flipped `stream`, deleted tool description, stripped `strict`, injected metadata field, **plus a mutation *beneath* a registered row's anchor** (§3.3.1a — a pattern is a prefix, so an over-coarse anchor silently claims what it should not, and only this case catches it). **Hard prerequisite for T-J2** — the oracle does not gate acceptance until its falsification suite is complete | §3.3.1, §3.3.1a | M |
| **T-D4** | Default-transport slice | T-D1, T-D2, T-B4 | One representative default adapter, end to end | §3.3.4 | M |
| **T-D5** | curl_cffi slice | T-D1, T-D2, T-B2, T-A3 | `openai_subscription`, observing P13–P17 | §3.3.2 | M |
| **T-D6** | botocore slice | T-D1, T-D2, T-B3, T-A5 | `bedrock`, observing the Converse payload after P18 | §3.3.2 | M |
| **T-D7** | Provider-aiohttp slice | T-D1, T-D2, T-B1, T-A6 | `ollama_cloud` after P19; the subscription OAuth leg | §3.3.2 | M |
| **T-D9** | Full default-transport matrix | T-D4, T-A3, T-A4, T-C2, T-C3, T-C4, T-C6 | All 20 default adapters at representative models; `opencode_go` per route | §3.3.4 | L |
| **T-D10** | Response-direction comparison | T-D1, T-A7 | `Reply` projections compared on the response path — a separate claim from request fidelity, tested separately | §3.3.1 | M |
| **T-D8** | Coverage checker | T-D9, T-D5, T-D6, T-D7, T-W3, T-A5, T-A6, T-A7, T-C5 | Fails when any conditional register row lacks a trigger case or a complement. **Owns "residual empty across the whole corpus" for all seven readers** | §3.3.4 | M |

---

## 8. Epic E — Containment

**T-E2 is one complete, falsified aiohttp slice.** T-E3–T-E5 extend it per transport as siblings.
An earlier draft made a single positive-control task wait on every transport's direct route, so a
difficult botocore route delayed even aiohttp containment — contradicting this plan's own claim
that a stuck transport blocks only its own task.

**A transport task is done when it records an outcome, not when it succeeds.** `proven`, `unsupported`
(the harness cannot give that transport a direct route — design §5.3 requires this be reported,
not hidden) or `failed` (a route exists and containment does not hold, which is a product defect
and gets its own ticket). An earlier draft defined these tasks as complete only when all four
phases passed, and then had T-E9 — the report whose entire purpose is recording that a transport
could **not** be proven — depend on all three succeeding. The failure report could not run in the
one case it existed for. T-E1 now ships the report; T-E9 only checks it is complete.

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-E1** | Harness core, aiohttp direct route, extension interface, **capability report** | | T-W5, T-W8, T-W9 | Upstream addressed as `upstream.kitty-test.invalid`; aiohttp direct leg via a monkeypatched resolver — **not** `/etc/hosts`, unavailable on CI runners. **Ships the per-transport capability report**, which starts with every transport `not-attempted` | §5.3 | L |
| **T-E2** | **aiohttp containment slice, complete and falsified** | | T-E1 | Phase 1 positive control; proxy down ⇒ zero connections; proxy up ⇒ every connection joins a tunnel on the recorded source port; **and an injected bypass makes the harness fail** | §5.2.1, §5.2.2 | L |
| **T-E3** | curl_cffi route and slice | | T-E1, T-E2, T-B2 | **An outcome is recorded**: `proven` (all four phases pass), `unsupported` (the harness cannot give this transport a direct route — with the reason), or `failed` (a route exists and containment does not hold — a product defect, filed) | §5.3, §5.5 | M |
| **T-E4** | botocore route and slice | | T-E1, T-E2, T-B3 | Same three outcomes as T-E3, for botocore | §5.3, §5.5 | M |
| **T-E5** | Provider-aiohttp route and slice | | T-E1, T-E2, T-B1 | Same three outcomes, for the provider sessions and the OAuth leg | §5.5 | M |
| **T-E6** | Guard **enforcement** | | — | Five start paths with a rejecting configuration assert **no server starts**. Falsification: a variant keeping the call and discarding its return value must fail these. **No recorder needed** | §6.2.3 | M |
| **T-E7** | AST start-path domination | | — | Every `BridgeServer(` construction dominated by an `egress_block_reason(` call at AST level | §5.1 | M |
| **T-E8** | Local bypass, fail-closed, transport asymmetry | | T-E2, T-B1, T-B2, T-B3 | Loopback bypass on bridge sessions; a `supports_egress() == False` profile blocks startup **and names the profile**; and the complement — those destinations **are** tunnelled on the three custom transports, which have no bypass. Needs the **recorders** to observe tunnelling, not the containment slices, so a stuck transport does not block it | §5.5 | M |
| **T-E9** | Containment completeness gate | | T-E1 | Asserts **every transport has a recorded outcome and none is `not-attempted`**. `unsupported` is a permitted outcome for partial delivery; `failed` is not. It reads the report T-E1 ships — **it does not depend on any transport succeeding** | §5.3 | S |

---

## 9. Epic F — Properties

| ID | Task | Depends on | Design | Size |
|---|---|---|---|---|
| **T-F1** | `hypothesis` + shared transcript strategies | — | §6.1 | M |
| **T-F2** | Compaction properties — including **output ≤ budget unless the surviving set is irreducible** | T-F1 | §6.1 | M |
| **T-F3** | Pairing and truncation properties | T-F1 | §6.1 | M |
| **T-F4** | Egress properties — private ranges; **no hostname outside the `localhost` family bypassed**; **structural** password redaction, never a substring test | T-F1 | §5.3, §6.1 | M |
| **T-F5** | `describe_tool_input_anomaly` property | T-F1 | §6.1 | S |
| **T-F6** | Translator semantic property **via the projections** | T-F1, T-A1, T-A2 | §3.3.1 | M |

**T-F4 is load-bearing beyond L1**: it pins the premise T-E1's harness rests on.

---

## 10. Epic G — Contracts

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-G1** | README ⇄ code table guards | defect | T-W7 | Endpoint, attribution-header, env-var, logging-flag tables (KBR-9) | §6.2.3 | M |
| **T-G2** | Register coverage meta-assertion | | T-W3, T-D9, T-D5, T-D6, T-D7 | Over T-D4–T-D9's captures — **it does not re-drive the wire**, and is marked `l3` so it cannot put sockets in the fast gate | §6.2.3 | M |
| **T-G3** | Internal-key completeness AST guard | defect | — | Every `_`-prefixed key written into any dict in `bridge/**` or `providers/**` is in `_INTERNAL_KEYS` (KBR-6). **Delivered by KBR-6**, which fixed the defect and so landed the guard green: the T-W7 dependency existed only to let it land red, and is dropped. Its complementary delegation check is discharged behaviourally by KBR-6's registry-parametrised regression test — see §6.2.3 | §6.2.3 | M |
| **T-G4** | Wire-shape honesty guard **at the wire** | defect | T-W7, T-B1, T-B2, T-B3 | The declaration agrees with the shape observed at the §3.2.3 boundary, per adapter × model × transport. **The hook-level form already landed with KBR-7** (`tests/test_wire_shape_honesty.py`) and is not this row. What remains is the three `use_custom_transport` adapters, for which nothing at the hook observes the bytes that ship: `openai_subscription` never invokes `translate_to_upstream` on the request path (P13–P17), and `bedrock` and `ollama_cloud` mutate the hook's body in the transport (P18, P19) — those two mutations do not change the shape family, so there the wire check is precautionary. Plus `provider_config`-constructed adapters and native-passthrough requests | §6.2.3 | M |
| **T-G5** | Bridge-introduced vendor token guard | defect | T-D1, T-W7, T-C5 | No bridge-introduced content names kitty — **and in the same run** an inbound turn containing `kitty-bridge` survives byte-identically. **KBR-5 is already fixed**, so this guard has no live positive fixture: inherit the synthetic historical M13 string from `tests/bridge/test_vendor_token_guard.py`, which also stands in for this row until the oracle lands | §3.3.3 | M |
| **T-G6** | OpenAPI 3.1 + schemathesis | | T-W8 | Five POST routes plus `/healthz`, `/stats`, `/v1/models`, **targeting bridge mode**; plus a per-protocol registration-matrix guard | §6.2.1 | L |
| **T-G7** | SSE grammar state machine | | T-B4, T-W8 | Every stream, including error streams and mid-stream failover, is a sentence in the grammar | §6.2.2 | M |
| **T-G8** | aiohttp proxy contract | | T-W5 | Session-level proxy across `>=3.11,<3.14`; a per-request `proxy=None` cannot escape it | §6.2.4 | S |
| **T-G9** | **Header contract** | defect | T-W7 | Exact header set per adapter — names, absences, casing, value shape. No `User-Agent` or version header derived from `kitty.__version__`; where both are sent they agree. **This catches KBR-8** | §4.3 C1 | M |
| **T-G10** | curl_cffi proxy contract | | T-W5 | `proxies=` honoured; precedence over ambient `HTTP_PROXY`/`NO_PROXY` | §6.2.4 | S |
| **T-G11** | botocore proxy contract + **declare `botocore`** | | T-W5 | `Config(proxies=)` precedence over the environment; and the dependency is declared, since a containment guarantee currently rests on an undeclared transitive | §6.2.4 | S |
| **T-G12** | `keyring` backend contract | | — | Backend resolution on each supported platform | §6.2.4 | S |

---

## 11. Epic H — Mutation validation

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-H1** | `mutmut` config, L1 selection, baseline | | T-W1 | Array `source_paths`, `pytest_add_cli_args_test_selection = ["-m", "l1"]`, baseline per target group | §6.1 | M |
| **T-H2** | Extract `_bedrock_body` | **src** | T-B3 | Payload shaping out of `make_request`/`stream_request`, behind T-B3's characterisation capture | §6.1 | M |
| **T-H5** | Extract `_ollama_body` | **src** | T-B1 | Same for `ollama_cloud`, behind **its own** characterisation capture | §6.1 | M |
| **T-H3** | Per-component thresholds + nightly reporting | ci | T-H1, T-H2, T-H5, T-F2, T-F3, T-F4, T-F5, T-F6 | ≥ 85% killed **per target group**, not one aggregate | §6.1 | M |
| **T-H4** | Measure changed-code mutation runtime | resolves Q11 | T-H1 | Timed against the fast gate's budget | §6.1 | S |

**T-H2 and T-H5 are separate** — one refactor per module, each behind the characterisation test
that makes *its* bytes observable. Bundled, the Ollama half would have had no evidence.

---

## 12. Epic I — Lifecycle and subsystem

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-I1** | Settings lifecycle | | — | Normal exit, `SIGTERM`, `SIGKILL` + `kitty cleanup`; global settings byte-identical | §6.3.2 | M |
| **T-I2** | Concurrent sessions | | T-I1 | Separate `--settings` files; neither touches the global file (issue #22) | §6.3.2 | M |
| **T-I3** | `prepare_launch` failure fails the launch | | — | Never proceeds on the user's own credentials | §6.3.2 | S |
| **T-I4** | Background bridge ownership | | — | Not stopped, not restarted, no second bridge | §6.3.2 | S |
| **T-I5** | Agent startup smoke | blocked Q12 | T-W8, T-W9, T-B4 | Pinned Claude Code binary, one turn, clean exit | §6.4.2 | M |
| **T-I6** | Agent settings precedence | blocked Q12 | T-I5 | Three runs, three winners; every sentinel demonstrated live | §6.4.2 | M |
| **T-I7** | Streaming recovery — content | partial Q14 | T-B4, T-G7, T-W8 | Four injection points; no duplicated text, no reused tool-call id, no spliced arguments. Positive oracle waits on Q14 | §6.3.1 | L |
| **T-I8** | Cross-attempt content and cadence | | T-D1, T-W8, T-B4 | Blip and empty-response retries byte-identical; M6, M8, M9 and failover re-normalisation each fire only on trigger | §4.3 C3 | M |
| **T-I9** | Connection lifecycle baseline | | T-W8, T-C7 | Distinct connections per session vs the native capture; ratcheted | §4.3 C5 | M |
| **T-I10** | `_backend_context` isolation | | T-W8 | Deterministic; belongs here, not in load | §6.3.1 | S |
| **T-I11** | Failover, disconnect, error envelopes | | T-B4, T-W8 | Mid-stream failover; disconnect releases upstream without marking unhealthy; 503 in each native envelope; oversized in the protocol's error shape | §6.3.1 | M |
| **T-I12** | Fingerprint parity report | | T-C7, T-W8, T-G9 | Header set vs the native baseline; **reported with a ratchet, not gating**, until gap G3 closes | §4.3 C1b | M |
| **T-I13** | Side traffic | | T-W8 | `/healthz` and `/stats` cause no upstream request; pre-flight pinned as a declared exception; `--no-validate` removes it | §4.3 C6 | M |
| **T-I14** | **Expand live-agent coverage to five Claude Code scenarios** | | — | Plain turn, tool-using turn, multi-turn with tool results, extended thinking, and a session crossing the compaction threshold. The existing file has two cases; scheduling it nightly does not expand it | §6.4.2 | M |

---

## 13. Epic J — Acceptance · Epic K — Evaluation, load, CI

| ID | Task | Flags | Depends on | Done when | Design | Size |
|---|---|---|---|---|---|---|
| **T-J1** | `pytest-bdd` wiring and step definitions | | T-W1, T-W7 | Steps bind to the L3 harnesses rather than re-implementing them | §6.4.1 | M |
| **T-J2** | TR scenarios | defect | T-J1, T-D3, T-D9, T-C7, T-G9 | TR-1, **TR-1b**, TR-1c, TR-2, TR-3, TR-4. Depends on T-D3 because the oracle may not gate acceptance before its falsification suite is complete. No egress dependency — no TR scenario involves containment | §6.4.1 | M |
| **T-J3** | EG scenarios | | T-J1, T-E2, T-E6, T-E8 | EG-0 (the reachability control), EG-1, EG-2, EG-3. T-E2 already carries its falsification | §6.4.1 | M |
| **T-K1** | Eval harness skeleton | | — | Two arms; model, provider, dataset, sampling pinned per run; failure taxonomy per arm | §6.4.3 | M |
| **T-K2** | Independently authored task set | | T-K1 | Acceptance tests **written by a person** | §6.4.3 | L |
| **T-K3** | Statistics and decision rule | blocked Q4, Q13 | T-K1, T-K2 | Successes ÷ **scheduled** trials; interval; pre-registered margin; symmetric exclusions; a missing-data ceiling that **voids** the run | §6.4.3 | M |
| **T-K4** | Load rig and baseline | | T-W8, T-B4 | Fixed workload, named runner class; latency, TTFB, completion and error rates, bounded RSS, socket recovery; streaming and buffered measured separately | §6.4.4 | L |
| **T-K5** | `load.yml` reusable + publish gate | ci | T-K4 | `publish.yml` `needs:`-gates on a load run **for the tag commit** | §8 | M |
| **T-K6** | Activate the Subsystem job | ci | T-W1, T-E2, T-D3, T-D4 | `l3` gates PRs and releases. **T-D3 is required, not just T-D4**: activating the job promotes the oracle into gating infrastructure, and §1.4 forbids that before its falsification suite exists. Fixing only T-J2's dependency left this hole open | §8 | S |
| **T-K7** | Deep nightly — mutation and schema fuzzing | ci | T-H3, T-G6 | Both exist before the job claims to run them | §8 | S |
| **T-K8** | Attach the per-category checks to each job | ci | T-W1, T-K6 | **The mechanism is T-W1's** — `--require-category`, and the test pairing it to every job's marker expression, landed there. What is left here is attaching a flag per category as each job activates, and removing that layer from `PENDING_ACTIVATION_LAYERS`. Narrowed after T-W1 delivered the enforcement rather than only the vocabulary | §8.1 | S |
| **T-K9** | Activate the Acceptance job | ci | T-J2, T-J3, T-K6 | `acceptance` gates PRs and releases | §8 | S |
| **T-K10** | Activate the `agent_smoke` category | ci, blocked Q12 | T-W1, T-I5, T-I6 | Its own required category — **it does not block Subsystem or Acceptance**, and it does not wait for them either: the T-K9 dependency was delay with no shared prerequisite behind it. It requires **T-I6 as well as T-I5**, because startup connectivity alone would let the category go green without proving the settings precedence that is the whole reason it exists | §8 | S |
| **T-K11** | Agent-live nightly | ci | T-I14 | Runs the **expanded** five-scenario coverage, not the two existing cases | §8 | S |
| **T-K12** | Eval nightly | ci | T-K3 | Runs once the decision rule exists; alerts, never gates | §8 | S |

**CI activation is incremental.** Each job turns on when its first independently runnable slice
lands, and each required category carries its own collection check. `agent_smoke` stays pending
Q12 as a separate category so it cannot hold up Subsystem or Acceptance.

---

## 14. Derived schedule

Computed from the dependency columns above, not written by hand. 98 tasks; the graph is acyclic
and every dependency resolves. **Recompute after any dependency change.**

### 14.1 What these numbers are, and are not

The durations below are **dependency-only lower bounds**. They assume every task starts the moment
its predecessors finish, which in turn assumes enough people to run every ready task in parallel.
They are not a delivery forecast, and two facts in this plan actively break that assumption:

- **T-C1–T-C7 share one capture-and-secret-review pass** (§6) — they are seven tasks but not seven
  parallel slots.
- **Five shared files need a single integration owner** (§18), which serialises work that the
  graph shows as parallel.

An earlier draft said "two owners halve elapsed time." **That does not follow from this
calculation** — the chain lengths are lower bounds under unlimited staffing, and halving is a
statement about resourcing that this graph cannot support. What the graph *does* support is
narrower and still useful: the fidelity and containment chains share no task until T-K9, so they
never block each other.

### 14.2 Readiness tiers

A tier is the earliest point a task *could* start, not a batch to wait for. **43 of 98 tasks are
available in tiers 0–2.**

| Tier | Count | Tasks |
|---|---|---|
| **0** | 12 | T-E6 T-E7 T-F1 T-G12 T-I1 T-I3 T-I4 T-I14 T-K1 T-W1 T-W2 T-W5 |
| **1** | 21 | T-A1–T-A7 T-F2–T-F5 T-G8 T-G10 T-G11 T-H1 T-I2 T-K2 T-K11 T-W3 T-W4 T-W7 |
| **2** | 9 | T-B4 T-F6 T-G1 T-G9 T-H4 T-J1 T-K3 T-W6 T-W8 (T-G3 delivered by KBR-6) |
| **3** | 18 | T-B1 T-B2 T-B3 T-C1–T-C7 T-G6 T-G7 T-I10 T-I11 T-I13 T-K4 T-K12 T-W9 |
| **4** | 10 | T-D1 T-E1 T-G4 T-H2 T-H5 T-I5 T-I7 T-I9 T-I12 T-K5 |
| **5** | 8 | T-D2 T-D10 T-E2 T-E9 T-G5 T-H3 T-I6 T-I8 |
| **6** | 11 | T-D3 T-D4 T-D5 T-D6 T-D7 T-E3 T-E4 T-E5 T-E8 T-K7 T-K10 |
| **7** | 3 | T-D9 T-J3 T-K6 |
| **8** | 4 | T-D8 T-G2 T-J2 T-K8 |
| **9** | 1 | T-K9 |

### 14.3 The chains

| Milestone | Longest chain to it | Days |
|---|---|---|
| Oracle core proven (**T-D1**) | T-W2 → T-W3 → T-W6 → T-C1 → T-D1 | 9.0 |
| Containment slice proven (**T-E2**) | T-W2 → T-W4 → T-W8 → T-W9 → T-E1 → T-E2 | 13.0 |
| Fidelity matrix complete (**T-D9**) | … → T-D1 → T-D2 → T-D4 → T-D9 | 16.0 |
| Fidelity acceptance (**T-J2**) | … → T-D9 → T-J2 | 17.5 |
| **Everything gating (T-K9)** | … → T-J2 → T-K9 | **18.0** |

> **Recomputed 2026-09-07.** T-W2 was **S** and is now **M** (§3, and KBR-25's requirements §0.3):
> the path vocabulary and the normalisation rules were added to it after a design review. T-W2
> heads every chain above, so **each gained 1.0 day**. §14 opens with "recompute after any
> dependency change"; a size change on the head of every chain is one. No ordering changed — T-W2
> was already first and alone.

**The critical path runs through the corpus**, not through containment:
`T-W2 → T-W3 → T-W6 → T-C1 → T-D1 → T-D2 → T-D4 → T-D9 → T-J2 → T-K9`. Four sequential tasks
precede the oracle, because T-D1 needs one corpus entry to run against. The corpus is not a
background risk; it is the front of the longest chain.

**A previously claimed optimisation has been withdrawn.** An earlier draft suggested letting T-D1
run against a synthetic transcript instead of T-C1, "taking about two days off the front." That
number was asserted, not computed. Recomputing it against the current graph gives **zero days
saved**: dropping the T-D1 → T-C1 edge makes T-W9 the limiting predecessor at exactly the same
length, so the chain merely moves from `T-W3 → T-W6 → T-C1` to `T-W4 → T-W8 → T-W9`. The corpus
dependency is not what makes this path long — the path is long either way. The lever does not
exist, and the plan should not offer it.

### 14.4 Consequences worth acting on

1. **T-W2 is the single most blocking task.** It heads both chains and it owns the whole input
   contract (§3). It goes first, alone, and lands before the streams branch. **Sized M, not S** —
   a design review established that without the path vocabulary T-W3 cannot be written and without
   the normalisation rules the six readers produce incomparable output, so both moved into it. The
   day that added is on every chain (§14.3), and it is the cheapest place in the plan to spend it:
   the alternative is eleven tasks each solving the same problem differently.
2. **T-W3 and T-W6 follow immediately** — they sit second and third on the critical path, ahead of
   any test.
3. **T-E1 is the riskiest single task**; containment carries ~5 days of slack against the critical
   path, so a slip there is absorbed rather than fatal. That is a change from the previous draft,
   where the two chains were nearly equal.
4. **T-K9 is the convergence point.** Both chains stop there; it is not an afterthought.

---

## 15. Blocked on decisions

| Question | Blocks | Cost of leaving it open |
|---|---|---|
| **Q4** + **Q13** | T-K3, and therefore T-K12 | The eval runs and cannot conclude; T-K1/T-K2 proceed |
| **Q10** | T-F2's exact bound, TR-3's wording, register rows M3–M7 | T-F2 lands with the observed-behaviour property and is revised **together with** TR-3 and the register. Narrowed by KBR-5: M13 is withdrawn, so "keep current behaviour" is no longer an option for that row |
| **Q11** | Nothing — **T-H4 answers it** | T-H3 lands nightly-only |
| **Q12** | T-I5, T-I6, T-K10 | The settings-precedence claim has no per-PR proof. **It no longer blocks Subsystem or Acceptance** — T-K10 is a separate category |
| **Q14** | T-I7's positive assertions | T-I7 lands asserting only the negatives |

**Q1** blocks no task but decides whether T-I12 and TR-1c ever gate. **Q5–Q7** affect register
rows and wording, not delivery.

---

## 16. Defects and regression evidence

An earlier draft required every guard to **merge red** before its fix. That is stronger than the
goal requires and it delays production fixes — KBR-5's acceptance scenario sits in tier 9 while
the design ranks the defect priority 0.

**The goal is evidence that the regression test fails on the unfixed revision — not a separate red
merge.** So:

- **Default: one atomic PR** containing the regression test **and** the fix, with evidence in the
  PR that the test fails at the base revision and passes with the fix. A guard written after the
  fix with no such evidence is still not acceptable — that is what the rule exists to prevent.
- **Exemption path** only where the fix must genuinely follow later, or where the guard's scope is
  broader than one defect. T-G1, T-G3, T-G4, T-G5 and T-G9 each cover a class of check beyond a
  single defect, so they may land first with a single-assertion exemption.
- **Later acceptance scenarios pass normally** if the defect is already fixed. T-J2 does not need
  TR-1c and TR-4 to be red; it needs them to be correct.

| Defect | Design gap | Broader guard | Fastest route |
|---|---|---|---|
| ~~KBR-5~~ | ~~G14~~ | T-G5, TR-4 | **DONE 2026-09-07** — atomic fix + tests, red at base revision. TR-4's exemption withdrawn |
| KBR-6 | G15 | T-G3 | Atomic fix + test now |
| KBR-7 · **CLOSED** | G16 | T-G4 | Route taken: atomic fix + hook-level guard landed together, red evidence in the PR. T-G4 still owns the wire boundary. (Status convention: `TEST_SUITE.md` §9.2.) |
| KBR-8 | G3 | T-G9, TR-1c | Atomic fix + test now |
| KBR-9 | G6 | T-G1 | Atomic fix + test now |

---

## 17. Design-to-deliverable coverage

Every design requirement has an owner. "Existing" means the current suite already covers it.

| Design | Requirement | Owner |
|---|---|---|
| §3.2 | The register, machine-readable | T-W3 |
| §3.3.1 | Six request projections | T-A1–T-A6 |
| §3.3.1 | Response-direction `Reply` projection and comparison | T-A7, T-D10 |
| §3.3.1 | Oracle falsification suite | T-D1, T-D3 |
| §3.3.2 | Both oracle assertions | T-D1 |
| §3.3.3 | Bridge-introduced scoping + the survival complement | T-G5 |
| §3.3.4 | Trigger complements across models | T-D8, T-C1–T-C6 |
| §3.3.5 | Routing in the oracle + deployment falsification | T-D2 |
| §4.3 C1 | Header contract | T-G9 |
| §4.3 C1b | Fingerprint parity baseline and report | T-C7, T-I12 |
| §4.3 C2 | Key-order on the native path | T-D1 |
| §4.3 C3 | Cross-attempt content | T-I8 |
| §4.3 C5 | Connection lifecycle | T-C7, T-I9 |
| §4.3 C6 | Side traffic | T-I13 |
| §5.2.1 | Tunnel correlation | T-E2 |
| §5.2.2 | Three phases + falsification, per transport | T-E2–T-E5, T-E9 |
| §5.3 | Hostname harness, per-transport resolution | T-E1–T-E5 |
| §5.4 | Guard enforcement and domination | T-E6, T-E7 |
| §5.5 | Per-transport containment and the bypass asymmetry | T-E3–T-E5, T-E8 |
| §6.1 | Properties · mutation validation | T-F1–T-F6 · T-H1–T-H5 |
| §6.2.1 | OpenAPI, schemathesis, registration matrix | T-G6 |
| §6.2.2 | SSE grammar | T-G7 |
| §6.2.3 | Register, internal-key, wire-shape, vendor-token, docs guards | T-G1–T-G5 — T-G3 delivered by KBR-6, and the **hook-level** half of wire-shape honesty by KBR-7; T-G4 still owns the wire boundary |
| §6.2.4 | Four dependency contracts | T-G8, T-G10, T-G11, T-G12 |
| §6.3.1 | Bridge-with-sockets scenarios | T-I7–T-I11 |
| §6.3.2 | CLI lifecycle | T-I1–T-I4 |
| §6.4.1 | Gherkin scenarios | T-J1–T-J3 |
| §6.4.2 | Agent smoke · precedence · **five live scenarios** | T-I5 · T-I6 · T-I14 |
| §6.4.3 | Eval harness, task set, statistics | T-K1–T-K3 |
| §6.4.4 | Load rig and gate | T-K4, T-K5 |
| §7.1–§7.4 | Corpus, recorders, proxy, oracle | T-W6/Epic C, T-W4/Epic B, T-W5, T-W2/T-D1 |
| §8 | Markers, selection, job activation | T-W1, T-K6–T-K12 |
| §5.1 | Real-socket transport proof across three stacks | **Existing** — `tests/test_egress_https_proxy.py`, extended by T-W5 |
| §6.2.3 | A structural scan that asserts its own positives | **Existing** — `tests/test_egress_coverage.py`, the pattern T-E7 copies |

---

## 18. Risks

**T-E1 is the hardest single task**, and T-E2's slice depends on it entirely. Splitting the
per-transport routes into T-E3–T-E5 means a stuck transport is one blocked M and an entry in
T-E9's unproven report, not a blocked chain.

**The corpus is on the critical path**, not merely a background risk (§14.3). T-C1–T-C7 need real
sessions, a secret review and a named refresh owner, and share one capture pass — while
`T-W2 → T-W3 → T-W6 → T-C1` is the front of the longest chain. Staff it first and capture T-C1
first. There is **no shortcut**: §14.3 shows that dropping the corpus dependency saves nothing,
because the alternative predecessor chain is exactly as long.

**T-D9 and T-G2 are adjacent.** T-D4–T-D9 own driving the wire and capturing; T-G2 asserts
register coverage over those captures as a meta-assertion. Implementing "per adapter × model ×
transport" twice is the most likely wasted work here — and if T-G2 re-drives the wire under an
`l2` marker it puts real sockets in the fast gate.

**T-H2 and T-H5 are the only source changes.** Keep them out of test PRs so a revert is cheap.

**The suite already runs ~18.5 minutes per Python version.** Real sockets reach the PR gate at
T-K6. It should land with a measured runtime; the marker split is the mechanism for pulling work
back to nightly if the gate stops being fast.

**Shared files need a named integration owner** — T-W2, T-W3, T-W4, T-W8 and T-W9 are touched by
several streams. Everything else lives in stream-owned modules.
