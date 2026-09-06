# Kitty Bridge — Test Suite Design

**Status:** To Be. Describes the suite Kitty Bridge should have, not the one it has.
**Scope:** The whole product, with Claude Code as the primary agent.
**Traces to:** [KBR-2](https://shelpuk.atlassian.net/browse/KBR-2).
**Method:** The four-layer model from the `test-development` skill, instantiated for this
codebase.

**On citations.** Symbols are named, not line-numbered. `server.py` is 6,463 lines and line
numbers rot on the next edit; `grep -n` or a symbol lookup recovers them, and a stale number is
worse than none.

> **Five defects were found while writing this document, three of them live breaches of the
> invariants it defines** (§4.4). Each is filed as its own Jira bug — KBR-5 through KBR-9 —
> and none is fixed by this change. That the *design* work found them, before a single test
> was written, is the argument for the design work.
>
> **Standing policy:** any defect this suite reveals — by a failing test, by observed
> behaviour, or by reading the code — is filed as a Jira ticket when it is found, not
> collected in a document. §4.4 is a summary of tickets, not a substitute for them.

---

## 0. How to read this

§1 states what the suite exists to prove. §2 says which layer proves what. §3–§5 specify the
three invariants that carry the product's commercial promise; they are the reason this document
exists and they are where the new work is. §6 specifies each layer. §7 specifies the shared
infrastructure. §9 is the honest gap list.

If you are adding a test and want one rule: **prove each behaviour at the lowest layer that can
prove it** (§2.2).

---

## 1. What the suite must prove

Kitty Bridge sits on the wire between a coding agent and an LLM provider. That position creates
three promises whose breach is not a bug but a product failure, and they are not currently
stated anywhere as checkable claims.

| ID | Invariant | Breach looks like |
|---|---|---|
| **I1** | **Message Fidelity** — the bridge forwards the agent's message content unchanged except for mutations on an explicit register | The agent's prompt silently loses a tool result; the model answers a question the user did not ask |
| **I2** | **Bridge Indistinguishability** — nothing the upstream provider can observe reveals that Kitty Bridge is in the path | A coding-plan provider fingerprints bridge traffic and blocks the account |
| **I3** | **Egress Containment** — when an egress gateway is configured, no provider-bound traffic reaches upstream except through it, and kitty refuses to start when it cannot honour that | Ten machines the operator believed shared one IP present ten; or one request in a thousand leaks the real address |

Alongside these sit the ordinary correctness claims any proxy needs: protocol translation is
faithful, failover and retry behave, credentials never leak, the agent's config files are
restored after a crash. Those are well served by the existing suite (§9.1) and are specified
here only where the invariants touch them.

**Why invariants first.** A test suite organised by module tells you the code does what it does.
A test suite organised by invariant tells you the product keeps its promises. Kitty's promises
are unusual — they are about *absence* (nothing changed, nothing visible, nothing leaked) — and
absence is not provable by adding more example tests. It needs differential and structural
tests, which have to be designed deliberately. That is §3–§5, and it is how §4.4's findings
surfaced.

---

## 2. The layer model

### 2.1 The four layers, instantiated

```
L4  Product     Claude Code ─► kitty launcher ─► BridgeServer ─► provider ─► upstream
                proves: a developer's session through kitty is indistinguishable,
                        faithful, and contained — end to end, with real binaries
                covers: Gherkin acceptance · real-agent E2E · answer-quality evals · load
                speed:  minutes · a handful · nightly and pre-release

L3  Subsystem   [BridgeServer + real sockets + recording upstream + real CONNECT proxy]
                [kitty CLI + real filesystem + real child process]
                proves: the pieces wire up correctly against real infrastructure
                covers: what actually leaves the socket · settings-file lifecycle ·
                        concurrent sessions · crash and recovery · failover on real TCP
                speed:  ~100ms–seconds · dozens · every pull request

L2  Contract    endpoint schemas ⇄ handlers · README tables ⇄ code · dependency behaviour
                proves: two things that must agree still agree
                covers: SSE event grammar · Messages API conformance · env-var and
                        header registers · docs/code sync · proxy semantics per transport
                speed:  fast and isolated · every pull request

L1  Component   translators · compaction · tool pairing · should_bypass · parse_proxy_url
                proves: a unit's logic is right for every input that matters
                covers: unit · property-based · validated by mutation testing
                speed:  milliseconds · thousands · every pull request
```

### 2.2 The allocation rule

Prove each behaviour at the lowest layer that can prove it. Worked examples from this codebase:

| Claim | Layer | Why not higher / lower |
|---|---|---|
| `should_bypass("http://10.0.0.5/")` is `True` | **L1** | Pure function of a string. A subsystem test proving it would be 1000× slower and no more conclusive. |
| Compaction never orphans a `tool_result` from its `tool_use` | **L1** (property) | `_compact_messages` is deterministic given its inputs. The property holds for all of them; an example test only samples. It is *not* a pure function — it calls `self._validate_tool_call_pairing` and reads its budget from `self._backends` / `self._active_model` — so the test must construct a server, not call a free function. |
| The README's endpoint table names the routes the code registers | **L2** | Both artifacts are readable statically. No server needed. |
| A `curl_cffi` session built with `proxies=` proxies, and its precedence over `NO_PROXY` | **L2** | A claim about a dependency, not about kitty. Belongs in a dependency contract test (§6.2.4). |
| No connection reaches the upstream host except from the proxy | **L3** | Requires real sockets. Cannot be proven by inspecting code, and does not need a real agent. |
| Two concurrent `kitty claude` sessions do not disturb each other's settings | **L3** | Requires the real filesystem and two real processes. |
| A developer running `kitty claude` gets a working Claude Code session against Z.AI | **L4** | Only a real agent binary exercises the real request shapes. |
| Compaction has not degraded the model's answers | **L4** (eval) | Semantic, nondeterministic, and only observable end to end. |

Two consequences, stated because they are routinely violated in suites of this size:

**Do not re-prove lower-layer claims higher up.** The L4 acceptance scenario for I1 asserts the
session works. It does not re-check that `_validate_tool_call_pairing` drops orphans — L1 owns
that.

**When a bug escapes, add the test at the lowest layer that would have caught it.** Add a
higher-layer smoke test only when the wiring itself was at fault. kitty-bridge#33 (a malformed
`tool_use` forwarded in silence) is the pattern: the detector is L1, the "the auditor sees what
the client sees" claim is L3.

---

## 3. Invariant I1 — Message Fidelity

> The bridge forwards the agent's message content to the upstream provider unchanged, except
> for mutations named on the Permitted-Mutation Register, each of which fires only under its
> stated trigger.

### 3.1 The problem with testing this the obvious way

The obvious approach is a golden-file test: record the upstream body for a fixed input, commit
it, and fail on any diff. It is the wrong tool here. Golden files fail on *every* change,
including intended ones, so they get regenerated reflexively and stop meaning anything. Worse,
they say nothing about inputs nobody recorded.

The design instead makes the *permitted* set explicit and tests the complement. Anything not on
the register is a violation, for every input, not just the recorded ones. The register becomes a
reviewed artifact: adding a mutation means adding a row, and a reviewer sees a mutation being
introduced rather than a golden file being refreshed.

### 3.2 The Permitted-Mutation Register

Every place kitty changes the agent's request between the inbound HTTP request and the upstream
body. Established by reading `src/kitty/bridge/server.py` and all 23 adapters in
`src/kitty/providers/`.

#### 3.2.1 Bridge-level

Eleven request-path rows (M1–M11) plus one response-path row (M12) and one substitution (M13).

| # | Mutation | Site | Trigger | Why it is necessary |
|---|---|---|---|---|
| M1 | Replace `model` with the profile's model, then provider-normalise it | `BridgeServer._normalize_model` | Always, when the profile sets a model | This is the product. The agent asks for one model; the profile decides what actually runs. |
| M2 | Translate the agent's protocol → Chat Completions | `MessagesTranslator` / `ResponsesTranslator` / `GeminiTranslator` `.translate_request` | Provider has `use_native_messages == False`, or the agent speaks Responses/Gemini | The upstream speaks a different protocol. Skipped entirely for native-Anthropic providers such as `zai_coding`. |
| M3 | Truncate a tool result over 50,000 chars (`_TOOL_RESULT_TRUNCATION_LIMIT`) | `_truncate_oversized_tool_results` | A single tool result exceeds the limit | A single oversized result can exceed the model's window on its own. |
| M4 | Truncate a tool result over the same limit, again, inside compaction | `_compact_messages` step 1 | Compaction ran **and** a `role: "tool"` message's string content exceeds the limit | Second pass, CC-shape only. Distinct from M3: M3 is unconditional pre-processing, M4 fires only once compaction is already engaged. |
| M5 | Compact the message history | `_apply_compaction` → `_compact_messages` | Serialized messages exceed the model-derived budget | Without it the upstream rejects the request outright. |
| M6 | Re-compact at half budget and re-send the same backend | `_compact_with_tighter_budget`, called only from `_request_with_retry_balancing` | Upstream returned 400/413 **and** `_is_context_too_large_error` **and** `_is_oversized_request` | Recovery from a rejection kitty's own budget estimate failed to prevent. **Balancing profiles only** — `_request_with_retry` (single backend) has no compaction recovery. Also an I2 exception; see §4.3 C3. |
| M7 | Drop orphan `tool_result` blocks | `_validate_tool_call_pairing` | A `tool_result` has no matching `tool_use` after compaction | An orphan triggers upstream error 2013 and fails the turn. |
| M8 | Add a thinking-carrier block and re-send the same backend | `_repair_thinking_roundtrip` / `_with_thinking_carrier` | This backend rejected this transcript for a thinking round-trip mismatch (issue #32) | Avoids one rejected round-trip per turn against backends that require it. Also an I2 exception. |
| M9 | Convert a native Messages body to CC format and re-send the same backend | `_convert_native_to_cc_format`, then a re-run of `_normalize_model` and `normalize_request` | Upstream returned a `tool_use` format error on the native path | Fallback that keeps the session alive rather than failing the turn. Also an I2 exception. |
| M10 | Inject the model from the URL path into the body | `_handle_gemini` | Gemini protocol only | Gemini carries the model in the path, not the body; `_normalize_model` needs it in the body to override it. |
| M11 | Force `stream: False` | `_handle_gemini` | Gemini protocol, non-streaming `:generateContent` | The Gemini translator defaults `stream=True`; the non-streaming endpoint must not open an SSE stream. |
| M12 | Substitute fallback assistant text | `_EMPTY_ASSISTANT_FALLBACK_TEXT` in `bridge/messages/translator.py` **and** `bridge/responses/translator.py` | Upstream returned an empty response | **Response-side**, not part of the eleven request-path rows. |
| M13 | **Discard the conversation and substitute a `[Kitty Bridge: …]` user message** | `_compact_messages` post-condition | No non-system message survives compaction (F25/F26 path — the system prompt alone exceeds the window) | Produces a legible error instead of a confusing upstream 400. **Qualitatively unlike M5**: M5 shrinks history, M13 replaces it — and it writes the product's name into the upstream body. See finding F3. |

#### 3.2.2 Provider-level

`ProviderAdapter` gives every adapter three hooks that can reshape the body:
`normalize_request`, `translate_to_upstream`, and the `_INTERNAL_KEYS` strip. Rolling 23
adapters into one row would make the register unfalsifiable — "the provider overrides it" is a
trigger no test can fail — so each material mutation gets its own row.

| # | Mutation | Site | Trigger | Why it is necessary |
|---|---|---|---|---|
| P1 | Strip kitty's internal metadata keys | `ProviderAdapter._INTERNAL_KEYS` via `translate_to_upstream` | Always | These keys are kitty's own; forwarding them is both an I1 and an I2 breach. **The set is incomplete — see F4.** Note it also strips `base_url`, which is *not* kitty-internal: it is defence-in-depth against a URL override arriving in the body, and is the one entry that could discard a field a caller meant. |
| P2a | Inject `thinking: {"type": "enabled"}` | `_ZaiBase.translate_to_upstream` (`ZaiRegularAdapter`, `ZaiCodingAdapter`) | `_thinking_enabled` truthy, or a reasoning effort other than `none` | Z.AI's wire format for a signal the agent sent differently. |
| P2b | Inject `thinking: {"type": "disabled"}` | same | `_thinking_enabled is False` **or** effort `== "none"` | The `else` branch. Listed separately because the oracle must not treat one as covering the other. |
| P3 | Inject `reasoning: {"effort": …}` | `OpenRouterAdapter` | `_reasoning_effort` present | OpenRouter's spelling of the same signal. |
| P4 | Inject `reasoning_effort` | `OpenAIAdapter` | `_reasoning_effort` present | OpenAI's spelling. |
| P5a | Default `max_tokens` to `_DEFAULT_MAX_TOKENS` (4096) | `AnthropicAdapter.translate_to_upstream` | Agent omitted `max_tokens` | The Messages API requires it. |
| P5b | Join system blocks with `\n` | same | Multiple system blocks | Messages API takes one system string. |
| P5c | **Raise** `max_tokens` to at least 1025 and set `thinking.budget_tokens = max_tokens - 1` | same | Thinking enabled | Anthropic requires `budget_tokens >= 1024` and `< max_tokens`. **User-visible**: it increases the agent's own `max_tokens`. |
| P5d | Map `_thinking_adaptive` → `thinking: {"type":"adaptive"}` and `_effort` → top-level `effort` | same | Those keys present | Passthrough of an agent signal. |
| P5e | Inject an empty `{"type":"thinking","thinking":""}` block into assistant messages | `AnthropicAdapter._translate_assistant_msg` | Assistant message lacks one while thinking is active | The Anthropic-path analogue of P8. **A message-content change**, not a parameter change. |
| P6 | Remove `model` from the body | `AzureOpenAIAdapter` | Always | Azure selects the model by deployment id in the URL; the body field is rejected. |
| P7 | **Cap the agent's `max_tokens` at 4096** | `FireworksAdapter.normalize_request` | Non-streaming request with `max_tokens > 4096` | Fireworks rejects non-streaming requests above 4096. **User-visible** as shortened output. |
| P8 | Inject empty `reasoning_content` into assistant messages | `ProviderAdapter._inject_empty_reasoning_content`, called from `KimiCodeAdapter`, `_ZaiBase`, `CustomOpenAIAdapter` | Thinking signalled **or** inferred from prior `reasoning_content` via `_detect_thinking_from_messages` | Those providers reject the request without it. The *inferred* trigger matters: it fires with no signal from the agent at all. |
| P9a | Set `User-Agent` to `claude-code/1.0` | `KimiCodeAdapter`, `BytePlusAdapter`, `MimoAdapter` `.build_upstream_headers` | Always, on those three | Those providers 403 without a recognised coding-agent user-agent. Central to I2 — F1. |
| P9b | Remove `Authorization`, add `api-key` | `MimoAdapter.build_upstream_headers` | Always | MiMo does not use Bearer auth. An auth-**scheme** change §4.3 C1's exact-set assertion must encode. |
| P9c | Synthesise a Codex CLI `User-Agent` and a `version` header | `OpenAISubscriptionAdapter` | Always | Impersonation required by the subscription endpoint. **The two disagree — see F1.** |
| P10 | Set `reasoning_split = True` | `MiniMaxAdapter.normalize_request` | **Unconditionally** | Makes MiniMax return thinking in `reasoning_details` instead of inline tags. Unconditional, so exempt from §3.3 assertion 2. |
| P11 | Translate CC → Bedrock Converse | `BedrockAdapter.translate_to_upstream` | Always, on `bedrock` | A third upstream wire format M2 does not name. Custom transport — see §3.3.2. |
| P12 | Translate CC → Ollama `/api/chat` | `OllamaCloudAdapter.translate_to_upstream` | Always, on `ollama_cloud` | A fourth wire format. Custom transport. |

**Conditional rows are the point.** Every row whose trigger is a condition must be provably
*inert* when that condition is absent — the sharpest form of "unless absolutely necessary", and
what §3.3 assertion 2 tests. M1, M2, M10, P1, P6, P9a–c, P10, P11 and P12 are unconditional by
design and are exempt from that assertion.

**Register maintenance.** The register is the specification. A pull request that adds a mutation
site without adding a row fails the L2 register guards (§6.2.3).

### 3.3 The transparency oracle

The single piece of new infrastructure that makes I1 testable.

**What it is.** A test harness that runs a request through the real `BridgeServer` into a
recording fake upstream, then diffs the captured upstream body against the inbound agent body
and classifies every difference against the register.

```
inbound agent body ──► BridgeServer ──► recording upstream
        │                                      │
        └───────── structural diff ────────────┘
                          │
                classify each delta
                          │
     ┌────────────────────┴────────────────────┐
claimed by a register row              unclaimed  ──► FAIL
whose trigger was present                    │
            │                          unclaimed delta is the
    assert the row's                   whole point of the oracle
    stated shape held
```

**The two assertions, in priority order.**

1. **No unclaimed delta.** Any difference between inbound and upstream bodies must map to a
   register row. This is the invariant.
2. **No mutation without its trigger.** For each conditional row, a corpus entry that does not
   meet the trigger must show that row's mutation *absent*. This is what stops M3/M5 quietly
   becoming unconditional.

#### 3.3.1 The oracle is scoped by wire shape — and why

A direct inbound-vs-upstream diff is only meaningful where the two bodies are in the same
protocol. On a Chat-Completions-wire provider the upstream body is CC while the inbound body is
Anthropic Messages: *every* field differs, every delta is claimed by M2, and assertion 1 passes
trivially while proving nothing. A test that cannot fail is worse than no test.

| Path | Adapters | Comparison | Why |
|---|---|---|---|
| **Native Messages wire** | `anthropic`, `zai_coding`, `custom_anthropic`, `minimax_token`, and `opencode_go` **for its Messages models only** | Inbound body vs. upstream body, directly | Same protocol both sides. "Unchanged" is literally meaningful, and this is the path the Jira premise names. |
| **Chat Completions wire** | the remaining adapters | Inbound vs. **round-trip** (`translate_request` → `translate_response` back to the agent's protocol) | The claim that survives translation is semantic preservation, not byte equality. |
| **Other wire** | `bedrock` (Converse), `ollama_cloud` (`/api/chat`) | Round-trip through that adapter's own pair | Neither Messages nor CC. Round-trip mode needs the adapter's own inverse, not the shared one (P11, P12). |

**The scoping predicate is per-request, not per-adapter.** `upstream_wire_is_messages_api` is the
natural selector, but it cannot be trusted as an adapter-level constant: `OpenCodeGoAdapter`
inherits `True` from `AnthropicAdapter` while its `translate_to_upstream` returns a **Chat
Completions** body for every model outside `_MESSAGES_MODELS`. See finding F5. Until that is
fixed, the oracle must select on the observed body shape, and an L2 guard must assert the
property agrees with the actual output shape for every adapter × representative model.

#### 3.3.2 The oracle must be parametrised over transport

`openai_subscription`, `bedrock` and `ollama_cloud` set `use_custom_transport = True` and never
go through `_make_upstream_request`, so §7.2's aiohttp recording upstream never sees their
bodies. The oracle is parametrised over transport exactly as §5.5's containment harness is:
an aiohttp recording server for the default path, a `curl_cffi`-reachable one for the
subscription provider, and a botocore endpoint override for Bedrock. Without this, §3.4's "every
provider" rows are false for three adapters.

**Diffing semantics.** Structural, not byte-level. JSON key order and whitespace are not part of
the agent's meaning and vary with serialisation; a byte diff would fail constantly and teach
people to ignore it. The oracle compares parsed JSON with a canonical ordering, and reports
deltas as JSON-pointer paths so a failure names the exact field. The one exception is key
ordering on the native path, which §4.3 C2 asserts for I2 reasons.

**What it does not do.** It does not judge whether a translation is *semantically* correct — that
a `tool_use` block became the right `tool_calls` entry is an L1 translator claim. The oracle
answers a narrower question: was anything changed that nobody declared.

### 3.4 Where I1 is proven

| Claim | Layer | Test |
|---|---|---|
| Compaction preserves `tool_use`/`tool_result` atomicity, in both CC and native shapes | L1 property | `compact(m)` contains no orphan |
| Compaction output fits the budget **unless the system block alone exceeds it** | L1 property | The guaranteed-fit loop breaks out while still over budget when it cannot shrink further; the honest property must say so |
| The last turn survives **unless pairing validation drops it**, in which case M13 fires | L1 property | Stated with the exception, or it fails on day one and gets weakened |
| Compaction is identity below the budget | L1 property | Short-circuit at the `original_size <= compaction_threshold` guard — M5's trigger, tested directly |
| Truncation is identity below `_TOOL_RESULT_TRUNCATION_LIMIT` | L1 property | Same shape, for M3 and M4 |
| Protocol round-trip preserves semantic content | L1 property | Text, tool calls, tool results and their ids survive translation both ways |
| A **below-threshold** corpus entry on a native-wire provider shows only M1 and P1 | L2 | Scoped to below-threshold deliberately: above it, M3/M5/M7 and any triggered `normalize_request` row apply to the native path too |
| No unclaimed delta, over the whole corpus, on every native-wire provider and model | **L3** | Transparency oracle (§3.3.1), parametrised per transport (§3.3.2) |
| Round-trip fidelity over the whole corpus, on every CC-wire and other-wire provider | **L3** | Oracle, round-trip mode |
| A real Claude Code session's tool calls execute correctly through kitty | L4 | Real-agent E2E (§6.4.2) |
| Compaction has not degraded answer quality | L4 eval | §6.4.3 |

---

## 4. Invariant I2 — Bridge Indistinguishability

> Nothing the upstream provider can observe about a request reveals that Kitty Bridge is in the
> path, or that the client is anything other than the coding agent it claims to be.

### 4.1 Why this is the load-bearing invariant

A coding plan is priced for a coding agent. A provider that can tell bridge traffic from native
agent traffic can throttle it, block it, or terminate the account — and the user finds out
mid-session. This is the invariant with commercial consequences, so it gets an adversarial test
design: a fake upstream that actively tries to fingerprint, rather than a test that checks a
list of headers we happened to think of. That posture is what surfaced F3 and F4.

### 4.2 The observable channels

| Channel | What it exposes today | Verdict |
|---|---|---|
| **C1 — Request headers** | `build_upstream_headers()` constructs the set from scratch; no inbound agent header is forwarded. Four adapters supply a coding-agent `User-Agent` (P9); every other provider — including `zai_coding`, whose set is exactly `Authorization`, `anthropic-version`, `content-type` — sends aiohttp's default. | **Gap, and inconsistent.** F1. |
| **C2 — Request body** | The register's mutations (§3.2), JSON key ordering produced by kitty's serialisation, **the literal string `[Kitty Bridge: …]` (M13)**, and **`_effort` / `_thinking_adaptive`, which are kitty-internal and reach the wire**. | **Breached.** F3 and F4. |
| **C3 — Cross-attempt content and cadence** | Retries (`_MAX_RETRIES = 3`), failover, transport-blip re-connects, the empty-response ladder — and **four** paths that send a *different body* on a later attempt (M6, M8, M9, and failover re-normalisation). | §4.3 C3. Four declared exceptions. |
| **C4 — Transport fingerprint** | TLS/ALPN/HTTP-2 signature of aiohttp, unlike the agent's own client. `curl_cffi` is already used for the OpenAI subscription provider precisely because that provider fingerprints TLS. | **Accepted residual risk.** §4.5. |
| **C5 — Connection lifecycle** | `_build_client_session` uses `TCPConnector(limit=…, force_close=True)` — a fresh TCP and TLS connection for **every** upstream request, no keep-alive reuse. The agent's client does not behave that way. | **Gap.** A cheap, non-TLS fingerprint — arguably more detectable than C4. §4.3 C5. |

### 4.3 Test specifications

**C1 — Header contract (L2).** Assert, per adapter, an exact header set: names present, names
absent, **casing**, and the value shape of each. Exact-set rather than subset-contains, because
a subset assertion cannot catch a *new* header being added — precisely the failure mode. The
forbidden set is asserted explicitly and includes any header whose name or value contains
`kitty` in any casing, and the bridge's own `X-Kitty-*` attribution headers (downstream-only;
they must never appear upstream).

Casing is part of the assertion, not decoration: `ZaiAnthropicAdapter` sends capitalised
`Authorization` beside lowercase `anthropic-version` and `content-type`, while the base adapter
sends `Authorization`/`Content-Type`. `MimoAdapter` removes `Authorization` entirely (P9b). A
provider can fingerprint any of that.

**Header *order* is deliberately not asserted.** What reaches the wire is aiohttp's ordering, not
the agent's, and pinning it would pin a dependency's internals. §7.2 records header order only
so the fixture can *report* it against the native baseline (C1b), not so a test asserts it.

Two further C1 assertions arising from F1:

- No adapter's `User-Agent` or version header may be derived from `kitty.__version__`.
- Where an adapter sends both a user-agent version and a `version` header, the two must agree.

**C1b — Fingerprint parity (L3).** Compare kitty's header set against a captured Claude Code
native set (§7.1 captures both). Assert kitty's is a *subset*, and report the difference. Today
the difference is large; the test's job is to make it visible and stop it growing, not to fail
the build on day one — a reported baseline with a ratchet, becoming a gate once G3 closes.

**C2 — Body shape (L3).** Covered by the transparency oracle (§3.3), plus two assertions the
oracle does not give for free:

- **No forbidden token in the body.** The same `kitty` token check C1 applies to headers, applied
  to the serialized upstream body. This is what M13 breaches, and what nothing currently checks —
  C1's forbidden set covers headers only, and M13's string is in a message.
- **Key order preservation on the native path.** A provider can fingerprint the JSON serialiser
  from key ordering alone. This is the one place the oracle compares bytes rather than structure,
  and it applies only where kitty claims to be forwarding rather than translating.

**C3 — Cross-attempt content and cadence (L3).** The design must not overstate this: kitty sends
a different body on a later attempt in **four** distinct situations, not one.

| Path | What changes between attempts | Same backend? |
|---|---|---|
| M6 — compaction recovery | Body re-compacted at half budget | Yes |
| M8 — thinking-carrier repair | `messages` rewritten in place | Yes |
| M9 — native→CC fallback | Whole body converted, then re-normalised | Yes |
| Backend failover | `_normalize_model` and `normalize_request` re-run, so a same-host different-key sibling still gets a different body | No (different backend, possibly same host) |

The assertions:

- *(i)* Transport-blip retries and empty-response retries — the two that repeat a request
  unchanged — must be **byte-identical** to the attempt they repeat: same body, same headers, no
  added retry-count or correlation header.
- *(ii)* Each of the four paths above is a **declared exception**: assert each fires only under
  its own trigger and never otherwise. A provider that hashes bodies can see all four; whether to
  close any of them is Q6.

**C5 — Connection lifecycle (L3).** Count distinct TCP connections the recording upstream accepts
across an N-turn session and compare against a native Claude Code capture. Reported as a
baseline first, like C1b. `force_close=True` exists to prevent port exhaustion, so closing this
gap is a real trade-off, not an oversight — recorded in §4.5 until Q7 is decided.

**C6 — Side traffic (L3).** Assert `GET /healthz` and `GET /stats` never cause an upstream
request, and that a launch sequence contacts the provider only for the agent's own turns plus
credential pre-flight validation. Pre-flight is a real upstream request the agent did not make;
the test pins it as a *declared* exception and asserts `kitty --no-validate` removes it (Q2).

### 4.4 Findings

Five defects surfaced while writing this document. **None is fixed by this change**; each needs
its own ticket. F3, F4 and F5 are live breaches of invariants defined above.

- **F1 — Agent identity is handled per-provider, not by policy.** *(KBR-8.)* Upstream headers are built from
  scratch, so Claude Code's `user-agent`, `x-app`, `anthropic-beta` and `x-stainless-*` never
  reach the provider. Four adapters compensate ad hoc (P9): `KimiCodeAdapter`, `BytePlusAdapter`
  and `MimoAdapter` hard-code `User-Agent: claude-code/1.0` — Kimi's carries a comment recording
  the string was on the provider's allowlist as of 2026-04-18 — and `OpenAISubscriptionAdapter`
  synthesises a Codex CLI identity. Everywhere else, including `zai_coding`, aiohttp's default
  goes instead.
  **The subscription adapter contradicts itself in a single request:** its user-agent is
  `codex_cli_rs/{kitty.__version__}` (currently `1.9.0`) while its `version` header is the
  constant `0.128.0`. A client claiming to be Codex CLI 1.9.0 *and* 0.128.0 at once is a one-line
  detection rule — and the user-agent tracks kitty's release train, so it changes with every
  kitty release and with nothing else. Tracked as G3, KBR-8 and Q1.
- **F2 — The README's endpoint table does not match the router.** *(KBR-9.)* README documents
  `POST /v1/gemini/generateContent`; `_register_routes` registers
  `/v1beta/models/{model}:generateContent` and `:streamGenerateContent`, and the README omits
  `GET /v1/models`. Exactly the drift the L2 docs⇄code layer exists to catch. Tracked as G6
  and KBR-9.
- **F3 — The product's own name is written into the upstream request body.** *(KBR-5.)* When compaction
  cannot preserve any non-system message, `_compact_messages` discards the conversation and
  substitutes a user message reading
  `[Kitty Bridge: Unable to compact conversation — the system prompt is too large relative to the
  model's context window. Use /clear to reset the conversation.]`. That message goes upstream via
  `_apply_compaction`. **A direct breach of I2** — the provider sees the vendor name in the
  request — and a fidelity mutation qualitatively unlike M5, since it replaces the conversation
  rather than shrinking it. The intent (a legible error rather than an opaque 400) is sound; the
  delivery is not. Register row M13; tracked as G14, KBR-5 and Q9.
- **F4 — Kitty-internal keys reach the upstream body on every Chat-Completions-wire provider.**
  *(KBR-6.)*
  `MessagesTranslator.translate_request` writes `_effort` and `_thinking_adaptive` into the CC
  request, but neither is a member of `ProviderAdapter._INTERNAL_KEYS`, so the default
  `translate_to_upstream` — which strips only that frozenset — forwards both. Confirmed
  empirically across `openai`, `openrouter`, `zai_regular`, `fireworks`, `minimax` and `novita`:
  each emits `['_effort', '_thinking_adaptive']` upstream. **A breach of both I1 and I2**, and
  the exact thing P1 exists to prevent. Underscore-prefixed keys no public API defines are an
  unmistakable proxy signature. Note that `_reasoning_effort` and `_thinking_enabled`, written by
  the same function, *are* in the set — so this is an omission, not a design choice. Tracked as
  G15 and KBR-6.
- **F5 — `OpenCodeGoAdapter.upstream_wire_is_messages_api` is wrong for most of its models.**
  *(KBR-7.)* It
  inherits `True` from `AnthropicAdapter`, but its `translate_to_upstream` returns a Chat
  Completions body for every model outside `_MESSAGES_MODELS`. `ProviderAdapter`'s own docstring
  says the property "describes the shape that actually goes on the wire" and that anything
  shaping the serialized body must branch on it — so a `True` that is false for most models is a
  latent defect in the thinking-repair path (M8) as well as a trap for the oracle's scoping
  (§3.3.1). Tracked as G16 and KBR-7.

### 4.5 Accepted residual risk

**C4 — transport fingerprint.** Full parity is not achievable on the aiohttp serving path. A
provider determined to fingerprint TLS can distinguish an aiohttp client from the agent's own
runtime, and matching it would mean routing every provider through `curl_cffi` — a substantial
change to the serving path for a threat no provider is currently known to apply to this traffic.
Recorded so a future incident is a known gap rather than a surprise. The README's existing
guidance ("use a CONNECT proxy, not a TLS-terminating one") already depends on this reasoning.

**C5 — connection lifecycle,** until Q7 is decided. `force_close=True` is a deliberate defence
against port exhaustion; removing it to gain keep-alive parity trades one operational risk for
one detection risk.

---

## 5. Invariant I3 — Egress Containment

> When an egress gateway is configured, no provider-bound traffic from the agent or the bridge
> reaches the upstream except through that gateway. When kitty cannot honour the gateway for
> every backend that will serve traffic, it refuses to start.

### 5.1 What is already proven, and what is not

**Fail-closed is built and tested.** `egress_guard.egress_block_reason()` checks every backend,
not just the first, and is called from **five** sites in three files — `bridge_runner.py` (two),
`cli/launcher.py`, and `cli/main.py` (two). `tests/test_egress_fail_closed.py` covers the guard's
logic.

**Real-socket transport proof already exists, and is strong.** `tests/test_egress_https_proxy.py`
(601 lines) stands up a local TLS CONNECT proxy that enforces Basic auth and **records every
`CONNECT` it sees**, plus a local TLS target, and performs real TLS handshakes across all three
transport stacks kitty uses: aiohttp (driving the real `egress_cmd._probe`), `curl_cffi` with the
exact `proxies=` mapping `openai_subscription` passes, and urllib3 shaped as
`botocore.httpsession._get_proxy_manager` builds it. This is the strongest asset in the area and
the foundation the rest of §5 builds on rather than replaces.

**Three things are missing, and they are the gap.**

1. **`BridgeServer`'s own request path is untested.** The existing module drives
   `egress_cmd._probe` — the function behind `kitty egress test` — not `_session_for` /
   `_make_upstream_request`. The serving path, which carries every byte of every conversation,
   has no equivalent proof.
2. **No negative assertion anywhere.** Nothing asserts that with the proxy *down*, the upstream
   receives **zero** connections. Without it, a bridge that proxies most of the time and falls
   back to a direct route on error would pass every test in the suite.
3. **The existing start-path guard is file-granular.** `tests/test_egress_coverage.py` asserts
   that a file constructing `BridgeServer` also *contains* a call to `egress_block_reason`.
   `cli/main.py` already holds two start paths, so a third added to that file would pass
   unguarded. §6.2.3 specifies the AST-level replacement.

### 5.2 The sealed-network harness

An L3 harness that closes gaps 1 and 2, built by extending `tests/test_egress_https_proxy.py`'s
`_ConnectProxy` and `_TlsTarget` rather than writing new infrastructure — that proxy already
records CONNECT attempts, which is the observation the harness needs.

```
       ┌──────────── kitty BridgeServer ────────────┐
       │                                            │
       │   direct session ──X (must see nothing)    │
       │   proxy  session ──────────┐               │
       └────────────────────────────┼───────────────┘
                                    ▼
                     recording CONNECT proxy  ── CONNECT log
                                    │
                                    ▼
                     recording fake upstream  ── request log
```

**The assertions.**

1. **Correlation, not peer address.** Every request the fake upstream records must correlate to
   a `CONNECT` in the proxy's log for the same target and connection, and the two counts must be
   equal. *Peer address cannot carry this*: with bridge, proxy and upstream all on loopback in
   one test process, a proxied and a direct connection both present `127.0.0.1`, and both source
   ports are ephemeral. Correlating on the proxy's own log discriminates; peer address does not.
   (Binding the proxy's outbound socket to `127.0.0.2` is an alternative, but it needs an `lo0`
   alias on macOS — a platform constraint the correlation approach avoids.)
2. **Negative — traffic stops rather than leaks.** With the proxy stopped and egress configured,
   the upstream accepts **zero** connections and the request fails. This is the assertion that
   proves containment; assertion 1 alone is satisfied by a bridge that proxies *sometimes*.
3. **Local bypass still works.** A loopback or `localhost` provider (a local Ollama) connects
   directly and is not tunnelled — a rented proxy cannot reach the caller's LAN. **Applies to
   the bridge's own sessions only**; see §5.5.
4. **Fail-closed.** A profile whose adapter returns `supports_egress() == False` (Bedrock in SSO
   mode) prevents startup, and the message names the profile.

### 5.3 The addressing trap — and why the harness uses a hostname

**This is the constraint that makes or breaks the harness, and it is not obvious.**

`egress.should_bypass()` returns `True` for loopback, private and link-local destinations, so
those connect directly. The naive harness binds a fake upstream on `127.0.0.1` — which
`should_bypass` sends *direct*, so the test proxies nothing and passes vacuously while proving
the opposite of what it claims. Binding on a Docker network does not help: `172.16.0.0/12` is
private and is also bypassed.

The escape is in `should_bypass`'s own design. It reads `urlsplit(url).hostname` and:

1. returns `True` for `localhost` and any `.localhost` suffix — **checked first, by name**;
2. returns `True` for an IP literal in a loopback, private or link-local range;
3. for anything else, returns `False` **without resolving it** — deliberately, to avoid a DNS
   round trip per request.

So the harness must address the fake upstream by a **hostname outside the `localhost` family**.
`upstream.kitty-test.invalid` is a good choice: RFC 2606 guarantees `.invalid` never resolves
publicly, which is precisely why the harness must supply the mapping itself.

**Name resolution, per leg.** The two legs differ, and only one needs work:

- *Proxied leg* — **no resolution needed.** aiohttp's `_create_proxy_connection` resolves the
  *proxy* and then issues `CONNECT upstream.kitty-test.invalid:443`; the test's own proxy
  resolves the target. The harness owns the resolver because the harness is the proxy.
- *Direct/bypass leg* — needs a local override. Use a monkeypatched aiohttp resolver, **not** an
  `/etc/hosts` entry: hosts edits need administrator rights and are unavailable on most CI
  runners, and `_build_client_session` constructs its own `TCPConnector` with no resolver
  injection point.

**The property that keeps the harness honest.** An L1 property test pins the premise so a future
change to `should_bypass` cannot silently make the harness vacuous:

> For every hostname that is neither an IP literal **nor `localhost` nor a `.localhost`
> suffix**, `should_bypass` returns `False`.

The `localhost` exclusions are not a caveat bolted on — `should_bypass` matches them explicitly,
before it ever tries `ipaddress.ip_address`, and a property stated without them fails on day one
and gets weakened, which removes the guard.

**A narrower user-visible consequence.** Because names outside the `localhost` family are not
resolved, a user whose local model server is reached by a **LAN hostname** — not `localhost`, not
an IP — will have that traffic tunnelled to a proxy that cannot reach it. The common
`http://localhost:11434` configuration is bypassed correctly. An L1 test documents the edge;
whether to change it is Q3.

### 5.4 Where I3 is proven

| Claim | Layer | Test |
|---|---|---|
| `should_bypass` classifies loopback / private / link-local / `localhost` family / public / hostname correctly | L1 + property | Enumerate IPv4 and IPv6 private ranges via `ipaddress`; assert the §5.3 hostname property |
| `parse_proxy_url` round-trips credentials, including percent-encoded `@` and `:` | L1 property | `parse(url_with_credentials(cfg)) == cfg` |
| No `EgressConfig` representation leaks the password | L1 property | `password not in repr(cfg) + str(cfg) + cfg.masked()` for all generated passwords |
| Every outbound HTTP client is egress-aware; no source assigns `HTTP_PROXY`; no session trusts the environment | L2 structural | **Exists:** `tests/test_egress_coverage.py` |
| **Every** `BridgeServer` construction is dominated by an `egress_block_reason` call | L2 structural | **Must be strengthened** — the existing guard is file-granular (§5.1 gap 3) |
| An `https://` proxy carries real traffic on all three transport stacks | L2/L3 | **Exists:** `tests/test_egress_https_proxy.py` |
| Proxy semantics under each dependency's version range | L2 | §6.2.4 |
| Nothing reaches upstream except via the proxy, **from the bridge's own serving path** | **L3** | Sealed-network harness (§5.2) |
| Stopping the proxy stops the traffic — no direct fallback | **L3** | §5.2 assertion 2 |
| Containment holds for each custom transport | **L3** | §5.5 |
| kitty refuses to start when a backend cannot be proxied | L2 + L4 | Guard unit test + scenario EG-3 |
| A developer's whole session presents one IP | L4 | Scenario EG-1 |

### 5.5 Per-transport containment

`_session_for` and `should_bypass` govern **only** `BridgeServer`'s own aiohttp sessions. Four
other outbound paths exist, and each applies the proxy **unconditionally, without consulting
`should_bypass`**:

| Path | Client | How the proxy is applied |
|---|---|---|
| `openai_subscription` — serving | `curl_cffi.AsyncSession` | `proxies=egress.proxies_dict()` |
| `openai_subscription` — OAuth token legs | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` |
| `bedrock` | boto3 / botocore | `BotoConfig(proxies=egress.proxies_dict())` |
| `ollama_cloud` | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` |

Three consequences the rest of §5 must not paper over:

1. **§5.2 assertion 3 is false for these paths.** They have no bypass, so a loopback or private
   destination *is* tunnelled. Arguably safer, but different — the design must say so rather than
   imply uniformity.
2. **The sealed-network harness proves nothing about them** unless parametrised over the
   transport. It must run over `{bridge aiohttp session, provider aiohttp session, curl_cffi
   session, botocore client}` — the shape `tests/test_egress_https_proxy.py` already uses, which
   is a further reason to extend that module rather than start fresh.
3. **The OAuth leg runs at startup**, before anything else has been proven, and is the one most
   likely to fire on a fresh machine. It must not be left out.

**An untested interaction.** `kitty.egress`'s own docstring records that the three stacks
disagree about `HTTP_PROXY`/`HTTPS_PROXY`: aiohttp ignores them unless `trust_env=True`, while
curl_cffi and botocore honour them. Kitty never sets those variables — but the *user's shell*
may have. Nothing tests what happens when an ambient `HTTP_PROXY` or `NO_PROXY` disagrees with
the configured egress on the two stacks that read the environment. §6.2.4 pins it.

---

## 6. Layer specifications

### 6.1 L1 — Component and property

**Scope.** Pure logic: the three translators, the translation engine, compaction and tool-call
pairing, model normalisation, `should_bypass` and `parse_proxy_url`, profile schema and
resolver, the tool-use anomaly detector, and every `ProviderAdapter` payload builder.

**Tools.** `pytest` (present) + `hypothesis` (**to add**, dev extra only).

**Command.** `pytest tests/ -q`

**Validation.** Mutation testing (below). This is the layer whose strength is actually measured.

**Property tests to add.** Example-based tests sample; these translators and the compactor are
exactly the kind of total function where properties pay.

| Unit | Property |
|---|---|
| `MessagesTranslator` | Semantic round-trip: text, tool ids, tool names and tool results survive Messages → CC → Messages |
| `_compact_messages` | Identity below budget · output ≤ budget **unless the system block alone exceeds it** · no orphaned pair · last block preserved **unless pairing validation drops it (then M13 fires)** · idempotent |
| `_validate_tool_call_pairing` | Output contains no `tool_result` without a `tool_use`, in both message shapes |
| `_truncate_oversized_tool_results` | Identity below the limit · output ≤ limit · non-tool-result content untouched |
| `should_bypass` | Every address in a private range is bypassed · the §5.3 hostname property |
| `parse_proxy_url` / `EgressConfig` | Credential round-trip · password never in any string form |
| `describe_tool_input_anomaly` | Never reports an anomaly for input that validates against the declared schema |

The two weakened properties are stated with their exceptions on purpose. A property that fails on
day one gets weakened by whoever is on the rota, and a weakened property guards nothing.

**Mutation testing.** Line coverage cannot tell a real assertion from `assert result is not
None`. `mutmut` closes that gap.

- **Tool:** `mutmut` (3.x; requires `fork`, so it runs on Linux CI — on Windows it needs WSL).
  Configured in `pyproject.toml` under `[tool.mutmut]`, where `source_paths` and
  `pytest_add_cli_args_test_selection` take **arrays**.
- **Command:** `mutmut run`, then `mutmut browse` to triage and `mutmut export-cicd-stats` for
  the CI score.
- **In scope — not the whole codebase:** `src/kitty/bridge/messages/`,
  `src/kitty/bridge/responses/`, `src/kitty/bridge/gemini/`, `src/kitty/bridge/engine.py`,
  `src/kitty/bridge/tool_audit.py`, `src/kitty/egress.py`, `src/kitty/profiles/`,
  `src/kitty/validation.py`.
- **Why a subset.** Mutation testing on 22,700 lines would take hours and produce a survivor list
  nobody reads. The subset is chosen by one rule: **modules whose silent misbehaviour breaches an
  invariant**. A mutation surviving in the compactor means the suite would not notice kitty
  eating a tool result — an I1 breach. A mutation surviving in the TUI menu means a cosmetic
  defect. Expand when the first list is clean.
- **Budget:** ≥ 85% killed on the in-scope set. Below that, the layer is not trusted.
- **Triage rule:** (a) a survivor revealing a missing assertion → strengthen the test; (b)
  revealing untested behaviour → add a test; (c) genuinely equivalent → suppress at the site with
  `# pragma: no mutate` **and a comment saying why**. Never dismiss a survivor silently.
- **Cadence:** nightly and pre-release. Per-PR would add tens of minutes to a gate that must stay
  fast.

### 6.2 L2 — Contract

**Scope.** Any two artifacts that must agree but are deployed, edited or upgraded separately.

**Command.** `pytest tests/contract -q`

**Validation.** Every structural guard must assert that its own scan finds known positives, so it
cannot rot into a no-op. `tests/test_egress_coverage.py` already does this
(`test_the_scan_actually_finds_something`, `test_the_scan_finds_the_known_start_paths`) and is
the pattern to copy.

#### 6.2.1 Bridge endpoint schemas

Publish an OpenAPI 3.1 document covering the **five** POST routes registered in bridge mode —
`/v1/chat/completions`, `/v1/messages`, `/v1/responses`,
`/v1beta/models/{model}:generateContent` and `:streamGenerateContent` — plus `GET /healthz`,
`/stats` and `/v1/models`. Test the handlers against it with `schemathesis` (4.x; pytest
integration via `@schema.parametrize()` and `case.call_and_validate()`).

**The job must target a bridge started in bridge mode** (`self._adapter is None`).
`_register_routes` registers only the routes matching `self._adapter.bridge_protocol` when an
agent is launching — one route for Messages, Responses and Chat Completions, **two** for Gemini,
and `GET /v1/models` in bridge mode only. A conformance run against a `kitty claude` bridge would
see a single route, pass, and leave the rest unvalidated. A separate guard asserts the
per-protocol registration matrix.

Checks that matter here: `not_a_server_error` (the bridge must never 500 on a malformed body),
`response_schema_conformance`, `status_code_conformance`.

**Why publish a schema for a local proxy nobody integrates against?** Because the *agents*
integrate against it, and they are third parties on their own release cycle. The schema is the
written form of "what Claude Code may send us," and the artifact against which a Claude Code
update can be checked. It also gives the fuzzer a target, which is how the malformed-input paths
get exercised at all.

#### 6.2.2 SSE event grammar

The Anthropic streaming format is a grammar, not a schema: `message_start` …
`content_block_start` / `content_block_delta`* / `content_block_stop` … `message_delta`,
`message_stop`. A malformed sequence breaks Claude Code in ways a per-event schema check cannot
see.

Test as a state machine over the byte stream the bridge writes: every stream it produces —
including error streams, failover mid-stream, and the empty-response fallback — must be a
sentence in that grammar. Applies to all three streaming protocols.

#### 6.2.3 Register and docs ⇄ code

Structural guards in the style of `tests/test_egress_coverage.py`.

| Guard | Asserts |
|---|---|
| **Register completeness — shape diff, not write scan** | For every registered adapter, feed a fixed CC request through `normalize_request` + `translate_to_upstream` and assert the resulting key-set and value deltas are **exactly** the union of that adapter's §3.2.2 rows. A write-scan cannot do this job: every provider mutation except P7 and the `normalize_request` overrides happens by *building and returning* a new dict, so a scan for writes to `cc_request` sees none of them — and would have missed F4 entirely. |
| **Internal-key completeness** | AST-scan `bridge/**` for every `_`-prefixed key written into a request dict, and assert each is a member of `_INTERNAL_KEYS`. **This is the guard that catches F4.** The complementary check — that each `translate_to_upstream` override delegates to `super()` or excludes `_INTERNAL_KEYS` — is necessary but not sufficient: every override *does* strip the set today; the set is what is wrong. |
| **Wire-shape honesty** | For every adapter × representative model, assert `upstream_wire_is_messages_api` agrees with the shape `translate_to_upstream` actually returns. Catches F5. |
| **Forbidden vendor token** | No string literal reachable into a request body or upstream header contains `kitty` in any casing. Catches F3. |
| **Start-path domination** | Every `BridgeServer(` construction is dominated by an `egress_block_reason(` call **at AST level**, not merely co-located in the same file. `cli/main.py` already holds two start paths (§5.1 gap 3). |
| **Env-var register** | `_SETTINGS_ENV_OVERRIDE_KEYS` and `_CONFLICTING_ENV_VARS` (`launchers/claude.py`) match what `build_spawn_config` emits and what the README documents. |
| **Endpoint table** | The README endpoint table matches `_register_routes`. Catches F2. |
| **Attribution-header table** | The README's `X-Kitty-*` table matches `_attribution_headers()`, and none of those names can reach `build_upstream_headers()`. |
| **Flag table** | The README logging-flag table matches the CLI parser. |

**Why guard the README specifically.** For a CLI tool the README *is* the interface
specification — it is what a user configures against. A drifted README is a defect with the same
user impact as a drifted API, and F2 shows it has already happened.

#### 6.2.4 Dependency behaviour contracts

Small, fast tests pinning third-party behaviour the invariants rest on, so a dependency bump
fails here with a clear message rather than in production. The pin situation is worse than a
glance suggests:

| Dependency | Declared pin | What must be pinned by test |
|---|---|---|
| `aiohttp` | `>=3.11,<3.14` | A session built with `proxy=`/`proxy_auth=` proxies, and a per-request `proxy=None` cannot escape it. `_build_client_session` sets the proxy at session level precisely so no call site can forget it, and `_session_for` depends on a request being unable to opt out. |
| `curl_cffi` | `>=0.7` — **unbounded** | `proxies=` is honoured; its precedence over ambient `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` (this stack *does* read the environment); the impersonation target still exists. |
| `botocore` | **not declared at all** — arrives transitively via `boto3>=1.34` | `Config(proxies=)` is honoured and takes precedence over the environment. It is botocore, not boto3, that implements this. An undeclared dependency owning a containment guarantee is worse than an unbounded one. |
| `keyring` | `>=23.0` | Backend resolution on each supported platform. |

The ambient-environment cases are not hypothetical: `kitty.egress`'s docstring records the
divergence, and a user with `HTTP_PROXY` set in their shell exercises it on two of three stacks.

### 6.3 L3 — Subsystem

**Scope.** One subsystem plus its real infrastructure. Two here: the bridge with real sockets,
and the CLI with the real filesystem and real child processes.

**Tools.** `pytest` + real `aiohttp` servers + the CONNECT proxy from
`tests/test_egress_https_proxy.py`. `testcontainers` only if a scenario genuinely needs process
isolation; a local proxy and upstream do not.

**Command.** `pytest tests/subsystem -q`

**Validation.** Each harness must carry a **negative** case proving it can fail — the sealed
network's proxy-down assertion, the oracle's "mutation present without its trigger" case. A
subsystem harness with no negative is this layer's characteristic failure mode; §3.3.1 and §5.3
each show how easily one goes vacuous.

#### 6.3.1 Bridge with real sockets

| Scenario | Assertion |
|---|---|
| Transparency oracle over the corpus (§3.3), per wire shape and per transport | No unclaimed delta; no conditional mutation without its trigger |
| Sealed network (§5.2), parametrised per transport (§5.5) | Everything correlates to a CONNECT; nothing when the proxy is down |
| Cross-attempt content (§4.3 C3) | Transport-blip and empty-response retries byte-identical; each of M6, M8, M9 and failover re-normalisation fires only under its own trigger |
| Connection lifecycle (§4.3 C5) | Distinct-connection count per session, against the native baseline |
| Forbidden vendor token (§4.3 C2) | No `kitty` token in any upstream header or body, including the M13 path |
| Streaming failover mid-response | The client sees one well-formed stream; `/stats` is authoritative for attribution, per the README's own caveat about first-byte headers |
| Client disconnect during a stream | Upstream connection released; the backend not marked unhealthy for a client-side fault |
| All backends unhealthy | The 503 arrives in each protocol's native error envelope |
| Oversized request | Rejected with the protocol's own error shape, not a raw 413 |

#### 6.3.2 CLI with real filesystem and processes

| Scenario | Assertion |
|---|---|
| Two concurrent `kitty claude` sessions | Each gets its own `--settings` temp file; neither touches `~/.claude/settings.json`; the second's start does not disturb the first (issue #22) |
| Normal exit | Session settings file removed; user's global settings byte-identical to before |
| `SIGTERM` | `atexit` path restores; same assertion |
| `SIGKILL`, then `kitty cleanup` | Recovery from the backup file; the `_kitty_values_present` heuristic fires only on kitty-written state |
| `prepare_launch` cannot write the file | Launch **fails**. It must not proceed — a session without the settings file would silently run on the user's own Anthropic credentials, which is both a fidelity and a billing failure |
| Background bridge owned by another user | Not stopped, not restarted, no second bridge started beside it |

**Why a real child process rather than a mock.** The settings/env precedence between
`--settings`, the process environment and `~/.claude/settings.json` is Claude Code's behaviour,
not kitty's. A mock would assert kitty's belief about that precedence — exactly the assumption
under suspicion in the KBR-1 investigation. Only a real spawn tests it.

### 6.4 L4 — Product

**Tools.** `pytest-bdd` for the Gherkin layer (**to add**, dev extra only — no BDD runner is
currently a dependency); `pytest` for the real-agent E2E; separate runners for evals and load.

**Command.** `pytest tests/acceptance -q` · `pytest tests/integration -v --runslow` · the eval
and load runners are separate entry points.

**Validation.** The paired-delta band (§6.4.3), and the requirement that every scenario binds to
an L3 harness rather than re-implementing one — an acceptance test that grows its own assertions
has drifted into L3 and should be moved down.

#### 6.4.1 Gherkin acceptance

The invariants written as scenarios a product owner can read and sign.

```gherkin
Feature: The upstream provider cannot tell Kitty Bridge is there

  Scenario: TR-1  Kitty leaves no fingerprint in the request
    Given a profile using the Z.AI coding plan
    When Claude Code sends a turn through kitty
    Then the request the provider receives contains no header, field or value
         naming kitty
    And the header set is a subset of what Claude Code sends natively

  Scenario: TR-2  A short turn reaches the provider unchanged
    Given a conversation well inside the model's context window
    When Claude Code sends a turn through kitty
    Then the provider receives the agent's messages with no content altered
    And the only difference from the agent's own request is the model name

  Scenario: TR-3  A long turn is altered only as far as necessary
    Given a conversation that exceeds the model's context window
    When Claude Code sends a turn through kitty
    Then history is compacted only as far as the budget requires
    And no tool result is separated from the call that produced it
    And the most recent turn is preserved in full

  Scenario: TR-4  An unrecoverable conversation fails without naming kitty upstream
    Given a system prompt larger than the model's context window
    When Claude Code sends a turn through kitty
    Then the user is told the conversation cannot be compacted
    And the provider receives no message naming kitty

Feature: Configured egress cannot be bypassed

  Scenario: EG-1  Every request presents the gateway's address
    Given a configured egress gateway
    When Claude Code runs a session through kitty
    Then every connection the provider accepts arrived through the gateway

  Scenario: EG-2  Traffic stops rather than leaks
    Given a configured egress gateway that has become unreachable
    When Claude Code sends a turn through kitty
    Then the turn fails with a clear error
    And the provider receives nothing

  Scenario: EG-3  An unproxyable profile stops the launch
    Given a configured egress gateway
    And a profile whose transport cannot honour it
    When the user runs kitty
    Then kitty refuses to start and names the profile
```

TR-4 is the acceptance form of F3: it states the outcome the user needs (a legible error) and the
one they must not pay for it with (the vendor name upstream), without prescribing the fix.

Step definitions bind to the L3 harnesses, so the acceptance layer stays thin: it composes, it
does not re-implement.

#### 6.4.2 Real-agent end-to-end

`tests/integration/test_agent_e2e.py` exists and spawns real agent binaries. Keep it out of the
default `pytest` run — it requires four agent CLIs, live credentials and live network, and a
suite that cannot run on a laptop or in CI stops being trusted. Promote it to a **nightly** job
with credentials in CI secrets, and extend it from two cases to cover, for Claude Code
specifically: a plain turn, a tool-using turn, a multi-turn session with tool results, an
extended-thinking turn, and a session that crosses the compaction threshold.

#### 6.4.3 Answer-quality evals

Compaction (M5, M6, M13) and truncation (M3, M4, P7) are the mutations that can degrade an answer
without breaking any structural assertion. Nothing below L4 can detect that.

Design: a fixed set of coding tasks with objective pass criteria (the produced code compiles; the
test it was asked to write passes). Run each through kitty and directly against the same
provider, and compare pass rates. The metric is the **delta**, not the absolute rate — the
absolute rate is a property of the model and moves for reasons unrelated to kitty. Sample size
fixed in advance, run nightly, alert on a delta beyond the band (Q4).

**Why a delta rather than a threshold.** An absolute threshold on a nondeterministic system
produces a flaky gate, and a flaky gate at L4 is worse than none: it trains people to re-run. The
paired comparison cancels the model's own variance.

#### 6.4.4 Load

A concurrency and duration profile — many parallel sessions against one bridge, a long streaming
response, and a sustained session — asserting the connection-pool limit
(`KITTY_BRIDGE_CONN_LIMIT`, default 500) and `force_close=True` behave as intended, that memory
does not grow across a long stream, and that the per-request `_backend_context` ContextVar does
not leak between concurrent requests. Pre-release, not per-PR.

---

## 7. Shared test infrastructure

### 7.1 Golden Claude Code transcript corpus

**What.** Real `POST /v1/messages` bodies captured from actual Claude Code sessions, committed as
fixtures, with the native upstream headers and connection pattern Claude Code produces captured
alongside them (the baselines C1b and C5 compare against).

**Why real rather than synthetic.** Hand-written fixtures encode our belief about what Claude
Code sends. That belief is the thing most likely to be wrong, and it drifts every time Anthropic
ships a release. Captured bodies are evidence.

**Required entries, at minimum:** plain text turn · turn with `tools` declared · assistant
`tool_use` · user `tool_result` · extended thinking · image content block · `system` with
`cache_control` · a transcript above the compaction budget · a single tool result above 50,000
chars · a transcript that provokes the 400/413 recovery path (**must run against a balancing
profile** — M6 exists only in `_request_with_retry_balancing`) · a system prompt alone larger
than the window, to reach M13 · a malformed body for the fuzz path.

**Traps.** Captured transcripts contain prompts, file contents and API keys. The capture
procedure must scrub credentials and the corpus must be reviewed before commit; a fixture file is
as public as the repository. The corpus needs a refresh cadence tied to Claude Code releases and
a recorded capture procedure — an un-refreshable corpus becomes a museum of a protocol nobody
speaks any more.

### 7.2 Recording fake upstream

An `aiohttp` server that speaks the Anthropic Messages API and the Chat Completions API, records
every request in full (method, path, **headers with original casing and order**, raw body bytes,
arrival timestamp, and a connection identifier), and replays scripted responses including SSE
streams, error statuses, Cloudflare blocks, empty responses, context-too-large rejections and
mid-stream disconnects.

It is the substrate for I1 (§3.3), I2 (§4.3) and I3 (§5.2). **Casing** is asserted by C1; **order**
is recorded for the C1b baseline report only, not asserted — what reaches the wire is aiohttp's
ordering, not the agent's. The connection identifier supports C5's distinct-connection count and
§5.2's CONNECT correlation. Peer address is deliberately **not** used for containment (§5.2
assertion 1).

Per §3.3.2 and §5.5, equivalents are needed for the non-aiohttp transports: a `curl_cffi`-reachable
recorder and a botocore endpoint override.

### 7.3 Recording CONNECT proxy

**Already exists.** `tests/test_egress_https_proxy.py` contains `_ConnectProxy` (enforces Basic
auth, records every `CONNECT` as a `ConnectAttempt`) and `_TlsTarget`, with throwaway
certificates on ephemeral ports. Extend it rather than build a second: expose it as a shared
fixture, add the ability to stop it mid-test for §5.2's negative assertion, and keep the CONNECT
log addressable for correlation.

### 7.4 Transparency oracle

Specified in §3.3. Implemented as a pytest fixture wrapping the recording upstream, exposing one
assertion — `assert_no_unclaimed_mutation(inbound, recorded, register, mode)` where `mode` is
`native`, `round_trip_cc` or `round_trip_other` per §3.3.1 — so a new corpus entry, provider or
transport costs one parametrisation, not a new test.

---

## 8. CI cadence

| Job | Contents | Trigger | Gate? |
|---|---|---|---|
| **Fast** | `ruff`, `lint-imports`, `mypy src/kitty`, L1 + L2 across Python 3.10–3.13 | every push and PR | **Yes** |
| **Subsystem** | L3: oracle, sealed network, settings lifecycle, concurrency | every PR | **Yes** |
| **Deep** | mutation testing, schemathesis at high `--max-examples`, extended property runs | nightly | Report + ratchet |
| **Product** | real-agent E2E, answer-quality evals | nightly | Alert, not gate |
| **Release** | everything above plus load | tag push, via the existing reusable `tests.yml` | **Yes** |

The existing arrangement — `publish.yml` and `ci.yml` both calling one reusable `tests.yml` so a
release runs exactly the checks a PR ran — is correct and is extended rather than replaced. The
new L3 job joins the same reusable workflow; the nightly jobs get their own, because a release
must not wait on an LLM eval.

---

## 9. Gap register

### 9.1 What the current suite already does well

2,880 test functions across 139 Python files under `tests/`; ~50,700 lines of test to ~22,700
lines of source. Broad L1 coverage of translators, providers, profiles, credentials and the TUI.
Three assets stand out and are built on rather than replaced:

- `tests/test_egress_https_proxy.py` — real TLS handshakes through a local recording CONNECT
  proxy, across all three transport stacks. The strongest existing proof of anything in this
  document.
- `tests/test_egress_coverage.py` — AST/regex structural guard over `src/` that also asserts its
  own scan finds known positives, so it cannot rot into a no-op.
- `tests/test_github_actions.py` — treats the workflow definitions as testable artifacts.

Four Python versions in CI, with `mypy` and `import-linter` as gates rather than reports.

### 9.2 What is missing

| ID | Gap | Today | Target | Priority |
|---|---|---|---|---|
| **G14** | **F3 — the vendor name goes upstream in the body (M13)** — KBR-5 | Live I2 breach | Fix the message; forbidden-token guard (§6.2.3) + TR-4 | **0** |
| **G15** | **F4 — `_effort` / `_thinking_adaptive` reach the wire** — KBR-6 | Live I1+I2 breach on every CC-wire provider | Add both to `_INTERNAL_KEYS`; internal-key completeness guard (§6.2.3) | **0** |
| **G16** | **F5 — `OpenCodeGoAdapter` misdeclares its wire shape** — KBR-7 | Latent defect in the M8 path; trap for the oracle | Make the property per-model; wire-shape honesty guard | **1** |
| **G1** | I1 is unstated and untested | No definition of "unchanged"; mutation sites discoverable only by reading 6,463 lines | Register (§3.2) + oracle (§3.3) | **1** |
| **G2** | No-bypass unproven **for the bridge's serving path**; no negative assertion; start-path guard is file-granular | `test_egress_https_proxy.py` proves the transports and drives `egress_cmd._probe` | Sealed-network harness (§5.2) per transport (§5.5) + AST start-path guard | **1** |
| **G3** | I2 partially breached (F1) — KBR-8 | Identity ad hoc per adapter; the subscription adapter reports two different versions in one request | Header contract + parity baseline, then a policy and a code fix | **2** |
| **G4** | L1 strength unmeasured | Line coverage only | `mutmut` ≥ 85% on the in-scope set | **2** |
| **G8** | No corpus of real agent traffic | Synthetic fixtures encode our assumptions | Golden corpus (§7.1) | **2** |
| **G10** | Custom-transport containment untested | Proven at transport level, never through the bridge; ambient `HTTP_PROXY`/`NO_PROXY` untested; the OAuth leg untested | §5.5 + §6.2.4 | **2** |
| **G5** | No contract layer | No published schema; SSE grammar unchecked | OpenAPI + `schemathesis` + grammar state machine | **3** |
| **G6** | Docs drift undetected (F2) — KBR-9 | README endpoint table already wrong | README ⇄ code guards | **3** |
| **G7** | No property-based tests | All example-based | `hypothesis` on the §6.1 list | **3** |
| **G9** | C5 unmeasured | `force_close=True` gives a per-request connection pattern unlike the agent's | Connection-count baseline | **3** |
| **G11** | Dependency behaviour unpinned; `curl_cffi` unbounded and **botocore undeclared** | Containment rests on an undeclared transitive dependency | Dependency contract tests (§6.2.4) + declare botocore | **3** |
| **G12** | Product layer effectively absent | 2 E2E tests, never run in CI | Nightly job, extended to 5 Claude Code cases | **4** |
| **G13** | No answer-quality signal | Compaction, M13 and the Fireworks cap can degrade output invisibly | Paired delta eval | **4** |

**Order of work.** G14 and G15 are priority 0: they are live breaches of the product's stated
promise, both are small code fixes, and each has a cheap guard that stops it recurring. Then G1
and G2 — the two invariants with the least coverage, sharing §7.2's recording upstream as their
foundation — with G16 alongside because the oracle's scoping depends on it. G8 unblocks G1's
corpus; G10 rides along with G2 once the harness is parametrised. G3's measurement lands with G1;
G3's *fix* is its own ticket. G4 and G7 reinforce an existing layer and can run in parallel. G5,
G6, G9, G11 are cheap and independent. G12 and G13 are last: most expensive to run, least caught
per hour.

---

## 10. Design rationale

Recorded per the repo's system-design discipline: the reasoning, especially where the choice was
not the obvious one.

**A permitted-mutation register instead of golden files (§3.1).** Golden files fail on every
change, get regenerated reflexively, and prove nothing about unrecorded inputs. The register
inverts it: the permitted set is small, explicit and reviewed, and everything else fails for all
inputs. The cost is maintenance — hence the L2 guards.

**Per-adapter register rows instead of one "providers may normalise" row (§3.2.2).** A trigger of
"the provider overrides it" is unfalsifiable, and a register with an unfalsifiable row does not
constrain the 23 adapters where most of the reshaping happens. Splitting it costs twenty lines of
table and buys a testable claim — and writing those rows out is what surfaced P5c, P7 and P10.

**The register guard is a shape-diff harness, not a source scan (§6.2.3).** Almost every provider
mutation happens by building and returning a new dict, so a scan for writes to `cc_request` sees
none of them. Feeding a fixed request through each adapter and diffing the output against that
adapter's rows is falsifiable and catches added-and-returned mutations. A source scan would have
missed F4.

**The oracle is scoped by observed wire shape, not by the adapter's own property (§3.3.1).** On a
CC-wire provider every field differs and every delta is claimed by the translation row, so a
direct diff passes without proving anything. And the natural selector cannot be trusted:
`OpenCodeGoAdapter` declares a Messages wire while emitting Chat Completions for most models
(F5). Selecting on the observed shape keeps assertion 1 falsifiable regardless.

**Structural diff, except key order on the passthrough path (§3.3, §4.3 C2).** Byte-comparing
JSON fails on serialisation noise and trains people to ignore red. But key *order* is exactly what
a provider fingerprints, so where kitty claims to be forwarding rather than translating, order is
part of the contract. The asymmetry is deliberate — and it stops at the body: header order is
aiohttp's, not the agent's, so pinning it would pin a dependency's internals.

**CONNECT-log correlation instead of peer address (§5.2).** With bridge, proxy and upstream on
loopback in one process, a proxied and a direct connection both present `127.0.0.1` and both
source ports are ephemeral — peer address cannot discriminate. Correlating each recorded request
to a recorded `CONNECT` can, and it avoids the `127.0.0.2` alias macOS needs configured.

**A hostname outside the `localhost` family, never an IP literal, for the fake upstream (§5.3).**
`should_bypass` bypasses loopback, private and `localhost`-suffixed destinations, so the obvious
`127.0.0.1` harness proves the opposite of what it claims — silently. The premise is pinned by an
L1 property test, stated *with* the `localhost` exclusions so it does not fail on day one and get
weakened.

**Extend the existing CONNECT proxy rather than build one (§7.3).** `test_egress_https_proxy.py`
already owns a recording proxy across all three transport stacks. A second would duplicate the
hard part and risk the two drifting on exactly the behaviour they both exist to pin.

**Two L1 properties are stated with their exceptions (§6.1).** "Output ≤ budget" and "the last
turn survives" are both false as absolutes — the compactor breaks out while still over budget
when it cannot shrink further, and M13 replaces the last turn. Stating the honest version keeps
the properties enforceable; stating the clean version guarantees they get weakened by whoever is
on the rota.

**Mutation testing on a subset (§6.1).** Whole-codebase mutation testing on 22,700 lines produces
a survivor list nobody reads and a nightly job nobody waits for. The subset rule — modules whose
silent misbehaviour breaches an invariant — keeps the output actionable.

**A paired delta for answer quality, not an absolute threshold (§6.4.3).** Absolute thresholds on
LLM output are flaky, and a flaky gate at L4 trains people to re-run rather than investigate. The
paired comparison cancels the model's own variance.

**Real-agent E2E stays out of the default run (§6.4.2).** It needs four CLIs, live credentials and
live network. A default suite that cannot run on a developer's laptop stops being run, and then
stops being trusted. Nightly with CI secrets is the right home.

**C1b and C5 start as reported baselines, not gates (§4.3).** Both differences are large today. A
gate that fails on day one gets disabled on day one. A ratcheted baseline makes the number visible
and stops it growing while the fix is done properly.

**TLS fingerprint parity is accepted residual risk, not omitted (§4.5).** Closing it means routing
every provider through `curl_cffi`, a large change to the serving path for a threat no provider is
currently known to apply here. Recording it means a future incident is a known gap, not a surprise.

---

## 11. Open questions for the product owner

Answers belong in this document. They are not invented here.

**Q1 — How faithful should the agent's identity be (F1, G3)?** Three options, materially
different: (a) forward a curated allowlist of the agent's real headers, uniformly, so every
provider sees a genuine coding agent; (b) send a neutral, stable identity that is neither kitty
nor Claude Code; (c) leave it ad hoc and treat I2 as "no kitty fingerprint" rather than "looks
like the agent." Option (a) is the literal reading of KBR-2 but carries provider-compatibility
risk (some providers reject unknown `anthropic-beta` values) and would replace three working
hard-coded strings with a general mechanism. Whichever is chosen, the current state is not a
policy — it is four independent workarounds, one of which reports two different client versions
in the same request. This decides both the I2 target and the C1b gate.

**Q2 — Is pre-flight credential validation an acceptable I2 exception?** It is an upstream request
the agent never made, and `--no-validate` already exists to suppress it. Declared exception, or
suppressed by default when egress is configured?

**Q3 — Should `should_bypass` resolve hostnames (§5.3)?** Not resolving saves a DNS round trip per
request but means a LAN-hostname-configured local model server gets tunnelled to a proxy that
cannot reach it. `localhost` is already handled by name. Keep the trade-off, or resolve-and-cache?

**Q4 — What is the acceptable answer-quality delta (§6.4.3)?** A number is needed for the alert
band. Zero is not achievable on a nondeterministic system.

**Q5 — Should the Fireworks `max_tokens` cap (P7) be visible to the user?** It silently shortens
the model's output. P5c raises `max_tokens` in the other direction. A one-line warning would make
both visible at the cost of noise. A product call, not a test-design one, but the register
surfaced it.

**Q6 — Which of the four body-changing retry paths are acceptable (§4.3 C3)?** M6, M8, M9 and
failover re-normalisation each send a different payload on a later attempt, and a provider that
hashes bodies sees all four. The alternative in each case is to fail the turn, which is worse for
the user. Declare all four as exceptions, or close some?

**Q7 — Should `force_close=True` stay (C5, §4.5)?** It prevents port exhaustion but produces a
connection pattern unlike the agent's own. Keep, or trade for keep-alive parity?

**Q8 — What is the repo's policy on tracking `.system_design/`?** `.gitignore` ignored both
`.system_design/` and `.requirements/`. KBR-2 asks for this document at a tracked path *and* for a
PR, which cannot both hold while the directory is ignored; this change therefore un-ignores
`.system_design/` and leaves `.requirements/` ignored as per-task working material. If that is
wrong, say so — and note the repo has no `SYSTEM_DESIGN.md` at all, so the request path, provider
registry and failover state machine are undocumented. A separate ticket seems right.

**Q9 — What should the unrecoverable-compaction message say (F3, G14)?** It must stay legible to
the user and stop naming the product upstream. Options: a vendor-neutral string; routing the
error to the agent as an HTTP error instead of a synthetic assistant turn; or keeping the text but
returning it downstream only. The third is probably right — the message is for the user, and it
currently reaches the one audience it was never meant for.
