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

Twelve request-path rows (M1–M11 and M15), one response-path row (M12), and the routing row **M14**
(§3.3.5), which is listed here because the destination is a mutation surface the body cannot show.
Fifteen rows in all. The former substitution row M13 is **withdrawn** — KBR-5 replaced it with a
downstream error, so it mutates nothing — leaving **fourteen live** bridge-level rows.

| # | Mutation | Site | Trigger | Why it is necessary |
|---|---|---|---|---|
| M1 | Replace `model` with the profile's model, then provider-normalise it | `BridgeServer._normalize_model` | Always, when the profile sets a model | This is the product. The agent asks for one model; the profile decides what actually runs. **"Always" was false on one route until KBR-160.** On `openai_subscription`'s Responses path the adapter built the shipped body out of `cc_request["_original_body"]` — the *raw inbound* body — so M1 ran, computed the profile's model, and had it discarded one layer down; the agent's model shipped, silently. `OpenAISubscriptionAdapter._prepare_responses_body` now reads `cc_request.get("model", "gpt-5.4")`, the same expression its sibling `_cc_to_responses` already used, which is why the other routes into that provider were never affected. The row's **site is still `_normalize_model` alone**: the adapter *consumes* M1's output, it does not perform M1's mutation, so this closed a discrepancy between the register and the code without changing a single register field. What the episode does show is that `paths=(envelope.model,)` is a claim about the §3.2.3 capture boundary that a downstream consumer can silently break, and that nothing but a test at that boundary notices. |
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
| M12 | Substitute fallback assistant text | `_EMPTY_ASSISTANT_FALLBACK_TEXT` in `bridge/messages/translator.py` **and** `bridge/responses/translator.py` | Upstream returned an empty response | **Response-side**, not part of the twelve request-path rows. |
| ~~M13~~ | **Withdrawn — no longer a mutation.** Was: discard the conversation and substitute a `[Kitty Bridge: …]` user message. | `_compact_messages` / `_apply_compaction` post-condition | No non-system message survives | **Closed by KBR-5.** The post-condition now raises `CompactionFailedError` and the handler returns a protocol-native 400 downstream; nothing is substituted, so there is no mutation left to register. The row is kept struck through rather than deleted so a reader of finding F3 can still find it. **The trigger recorded here was wrong** — see F3. |
| M14 | **Replace the destination entirely** — scheme and host are built from the profile by `build_base_url()`; the path by `get_upstream_path(_route_model(cc_request))` — the **request's normalized model**, which is the normalized profile model when there is one and the agent's model when there is not. `_route_model` is the single place that answers this; the auth scheme (P9/P20) and the thinking carrier read it too, and the adapter reads the same key for the body (KBR-127 — it was the raw profile model, so path and body could route differently; and on `openai_subscription`'s Responses path the adapter read the *inbound* body's model instead until KBR-160, which was harmless for routing only because that provider posts to a fixed URL and derives no header from the model). Base and path are then **composed** by `ProviderAdapter.compose_upstream_url`, not concatenated (KBR-143). | `BridgeServer._build_upstream_url` | Always | The agent addressed a loopback bridge; the request has to reach the real provider. Listed because **the destination is a mutation surface the body cannot show**: on Azure an identical body sent to the wrong deployment path is a different request entirely (§3.3.5). **The query is part of the mutation, not a passenger** (KBR-143): the endpoint joins the *path* component and the two queries merge, the endpoint's parameters winning a name clash and the base URL's others surviving unaltered. A row naming only "path" would let an oracle derive `route.query` and still not know which side owns a clash. The base URL's fragment is carried through and never sent, since no HTTP client puts one on the wire — so an oracle deriving `route.*` from the profile must expect it on the composed URL and absent from the request line. **The composed URL is redacted before it is echoed** into the 404 diagnostic or a pre-flight failure (`redact_url_for_display`): query values and the fragment are masked, which is an I2-adjacent containment property, not a fidelity one — nothing about the request changes. The composition helper is shared with `kitty.validation.validate_api_key` and `OllamaCloudAdapter._build_url`, but **this row's site is the bridge alone**: pre-flight's probe is not a request the agent made, and the register describes what happens to the agent's request. |
| M15 | Rewrite a string `input` into the single-item list form `[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": <s>}]}]` | `normalize_responses_request` (`bridge/responses/translator.py`), called from `_handle_responses` before the body forks | Always | OpenAI's `CreateResponse` defines the two forms as the **same request**: `input` is `oneOf` a string (*"a text input to the model, equivalent to a text input with the `user` role"*) or an array, and everything downstream reads the array. Fires on every request reaching the handler; a body already in the array form meets the row with a **no-op** rather than avoiding it, so there is no complement state for §3.3.2 assertion 2 to arrange, which is why it is unconditional. Listed rather than omitted because the rewrite is real bytes at the `curl_cffi` boundary of §3.2.3, where `_original_body` **is** this body; the projection cannot express the difference, so the row takes §3.3.1a's escape for P16's reason. **KBR-144.** |

#### 3.2.2 Provider-level

`ProviderAdapter` gives every adapter three hooks that can reshape the body:
`normalize_request`, `translate_to_upstream`, and the `_INTERNAL_KEYS` strip. Rolling 23
adapters into one row would make the register unfalsifiable — "the provider overrides it" is a
trigger no test can fail — so each material mutation gets its own row.

**Header rows record *deviations from the base header set*, and headers are never diffed against
the inbound request.** §4.2 C1 records that `build_upstream_headers()` builds the upstream set from
scratch and forwards no inbound agent header, so there is no "unchanged except" claim to make about
headers and no inbound counterpart to diff against — such a diff would report every header on every
adapter. I1's subject is the agent's **message content**. The baseline is therefore
`ProviderAdapter.build_upstream_headers` — `Authorization: Bearer <key>` and
`Content-Type: application/json` — and a P9 row names what an adapter **adds to, removes from, or
re-spells in** that set. A header matching the baseline in name, casing and value *shape* is not a
mutation even when the value differs: every adapter substitutes the profile's credential for the
agent's, which is M14, not a per-adapter effect. That is why `Content-Type` and `Authorization`
carry no row anywhere — including on `openai_subscription`, whose `Authorization` is still
`Bearer <opaque>`, and which does not override `build_upstream_headers` at all: its curl_cffi
transport calls `_build_codex_headers` instead, so the inherited hook is as dead there as §3.2.3
says `translate_to_upstream` is.

**The consumer of a header row is §4.3 C1's exact-set assertion, not §3.3.2 assertion 1.** A header
added without a row is caught by C1's exact-set assertion failing, never by the oracle. Coverage is
partial today — see gap G22.

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
| P20 | **Encode the request's normalized model as the deployment id in the URL path** | `AzureOpenAIAdapter.get_upstream_path` | Always | The counterpart of P6: what P6 removes from the body reappears in the path. A register that records only P6 makes the model look *dropped* when it was *moved*, and leaves the move unchecked. Before KBR-127 this read *the profile's* model, so a profile written `azure/my-deploy` addressed a `/deployments/azure/my-deploy/` segment that cannot exist. |
| P21 | Encode `project_id` and `location` in the base URL | `VertexAIAdapter.build_base_url` | Always | Vertex addresses a project-scoped endpoint. Same class as P20: routing carried outside the body. |
| P7 | **Cap the agent's `max_tokens` at 4096** | `FireworksAdapter.normalize_request` | Non-streaming request with `max_tokens > 4096` | Fireworks rejects non-streaming requests above 4096. **User-visible** as shortened output. |
| P8 | Inject empty `reasoning_content` into assistant messages | `ProviderAdapter._inject_empty_reasoning_content`, called from `KimiCodeAdapter`, `_ZaiBase`, `CustomOpenAIAdapter` | Thinking signalled **or** inferred from prior `reasoning_content` via `_detect_thinking_from_messages` | Those providers reject the request without it. The *inferred* trigger matters: it fires with no signal from the agent at all. |
| P9a | Set `User-Agent` to `claude-code/1.0` | `KimiCodeAdapter`, `BytePlusAdapter`, `MimoAdapter` `.build_upstream_headers` | Always, on those three | Those providers 403 without a recognised coding-agent user-agent. Central to I2 — F1. |
| P9b | Remove `Authorization`, add `api-key` | `MimoAdapter.build_upstream_headers` | Always | MiMo does not use Bearer auth. An auth-**scheme** change §4.3 C1's exact-set assertion must encode. |
| P9c | Synthesise a Codex CLI `User-Agent` and a `version` header, and add `Accept: text/event-stream` | `OpenAISubscriptionAdapter._build_codex_headers` / `._build_user_agent` | Always | Impersonation required by the subscription endpoint; `Accept` is the header half of P17's forced streaming — the Codex backend is streaming-only. **The two versions disagree — see F1.** The conditional `ChatGPT-Account-Id` this site also sets has no row yet; see G22 |
| P10 | Set `reasoning_split = True` | `MiniMaxAdapter.normalize_request` | **Unconditionally** | Makes MiniMax return thinking in `reasoning_details` instead of inline tags. Unconditional, so exempt from §3.3.2 assertion 2. |
| P11 | Translate CC → Bedrock Converse | `BedrockAdapter.translate_to_upstream` | Always, on `bedrock` | A third upstream wire format M2 does not name. Custom transport — see §3.3.4. |
| P12 | Translate CC → Ollama `/api/chat` | `OllamaCloudAdapter.translate_to_upstream` | Always, on `ollama_cloud` | A fourth wire format. Custom transport. |
| P13 | **Drop fourteen Chat-Completions-only parameters** — `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `response_format`, `stop`, `n`, `stream_options`, `seed`, `logit_bias` | `OpenAISubscriptionAdapter._cc_to_responses` | Always, on the **CC-origin** path | The Codex backend applies strict allowlist validation and rejects them with 400. **User-visible**: `max_tokens` and `temperature` silently do nothing on this provider. Logged at DEBUG. |
| P14 | **Drop every parameter outside the Codex allowlist**, notably `max_output_tokens` | `OpenAISubscriptionAdapter._prepare_responses_body` | Always, on the **Responses-origin** path | Same backend restriction, different input shape — this path receives a Responses body, so the parameter is spelled `max_output_tokens`, not `max_tokens`. A single row cannot cover both paths; the sets differ. |
| P15 | **Strip `strict` from every tool declaration** | `_prepare_responses_body` | Always, on the Responses-origin path | The Codex backend rejects it. A change to the **tool schema** the agent declared, not to a sampling parameter — a different kind of fidelity mutation and worth its own row. |
| P16 | Rewrite content types `input_text` → `output_text` | `_convert_content_types`, called from `_prepare_responses_body` | Always, on the Responses-origin path | The Codex backend validates content types strictly. **Message-content mutation.** |
| P17 | Inject `stream: True` and `store: False` | `_cc_to_responses` and the Responses-origin body builder | **Unconditionally**, both subscription paths | The Codex backend is streaming-only; kitty reassembles a non-streaming reply from the SSE. Note `stream: True` **overrides a non-streaming client request** — the subscription-path analogue of M11. |
| P18 | Remove `modelId` and `stream` from the Converse payload | `BedrockAdapter` transport (`make_request` / `stream_request`) | Always, on `bedrock` | boto3 takes the model id as a call argument and selects streaming by choosing `converse` vs `converse_stream`, so both must leave the body. Applied **in the transport, after `translate_to_upstream`**. |
| P19 | Overwrite `stream` | `OllamaCloudAdapter` transport (`make_request` sets `False`, `stream_request` sets `True`) | Always, on `ollama_cloud` | The transport, not the caller, decides which Ollama endpoint mode is used. Applied **after `translate_to_upstream`** has already set it from the request. |

**Conditional rows are the point.** Every row whose trigger is a condition must be provably
*inert* when that condition is absent — the sharpest form of "unless absolutely necessary", and
what §3.3.2 assertion 2 tests, with the trigger complements §3.3.4 requires. M1, M2, M10, M14, M15, P1,
P6, P9a, P9b, P9c, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20 and P21 are unconditional
by design and are exempt from that assertion.

**M14, P20 and P21 were missing from that list until KBR-26**, while their own trigger cells read
`Always`. A row that always fires has no complement, so assertion 2 would have demanded a corpus
entry nobody could ever write. The list is also written out id by id rather than abbreviated as a
range: `P9a–c` names three rows in one token, and §3.2.4's guard has to either guess at the
expansion or drop two rows from the comparison. It refuses the notation instead.

**Register maintenance.** The register is the specification. A pull request that adds a mutation
site without adding a row fails the L2 register guards (§6.2.3).

#### 3.2.3 Serialization paths — where the register must actually be checked

The register is only enforceable at the point where bytes are handed to a transport. That point
is **not** `translate_to_upstream` for any of the three custom-transport providers, and assuming
it is hid seven rows in the first draft: on `openai_subscription` the hook is **never invoked on
the request path at all** — `_cc_to_responses` builds the Responses body inside the transport
(P13–P17) — while on `bedrock` and `ollama_cloud` the hook builds the body and the transport then
**mutates it** (P18, P19).

| Path | Adapters | Where the final bytes are decided |
|---|---|---|
| Bridge aiohttp | the 20 default-transport adapters | The `json=` body passed to `session.post` in `_make_upstream_request` / `_open_upstream_stream` |
| curl_cffi | `openai_subscription` | Built **inside the transport**: `_cc_to_responses` (CC-origin) or `_prepare_responses_body` (Responses-origin). The adapter hook's output is not what ships. |
| botocore | `bedrock` | Built by `translate_to_upstream`, then **mutated** in `make_request` / `stream_request` (P18). Capture after the mutation. |
| provider aiohttp | `ollama_cloud` | Built by `translate_to_upstream`, then **mutated** in `make_request` / `stream_request` (P19). Capture after the mutation. |

Every guard and every oracle run in this document targets the right-hand column, never the hook
that precedes it. Where the body is built in the hook and mutated in the transport, "after the
mutation" is the boundary — capturing the hook's return value would miss P18 and P19 exactly as
it missed P13.

#### 3.2.4 The register as data

§3.2.1 and §3.2.2 are the register's reviewed prose. `tests/harness/register.py` is its
machine-readable form, and three consumers read the second rather than the first: the oracle
(§3.3, which §7.4 gives `register` and `triggers_met` as arguments), the coverage checks (T-G2,
T-D8), and the corpus loader (T-W6, which indexes captured sessions **by trigger**).

**Authority is split, deliberately.** The markdown remains the reviewed record of *why* each
mutation is necessary — a reviewer reads a table, not a tuple. The data is what tests execute.
Ids and the conditional/unconditional classification are **mechanically reconciled** between the
two by an L2 guard. Paths and sites are **not**, because the tables have no path column and their
Site cells are prose (M4's reads "step 1", P17's "and the Responses-origin body builder", P2b's
simply "same"). Sites are checked against the **source tree** instead, which is the stronger
check: it catches a renamed mutation site, which no comparison against a prose cell could.

**The Trigger column is reconciled by nothing, and that is the third state.** Editing a Trigger
cell produces no disagreement. Those cells are prose of the same kind as Site — "Always, on the
**CC-origin** path", "Serialized messages exceed the model-derived budget" — while the data's
`Trigger` is a closed vocabulary, and comparing them needs a mapping that would itself be a third
artifact to keep in step. What holds a trigger honest instead is the classification check: a row
whose trigger is `ALWAYS` may not be conditional, and §3.2.2's unconditional list is compared id
by id. Beyond that the Trigger column is reviewed, not tested, until G21 closes.

**The schema is the six fields plan §3 names, plus one.** `id`, `site`, `trigger`, `paths`,
`conditional`, `design_ref` — and `not_projectable_reason`, required exactly when `paths` carries
the §3.3.1a escape, because an escape without a reason is a row nothing can falsify.

**A trigger is a name, not a callable.** The obvious reading of "trigger predicate" is a function
of the inbound request. It cannot work: M6 fires on an upstream 400, M8 on a rejected thinking
round-trip, M9 on an upstream tool-use format error, M12 on an empty upstream response. None is a
property of the request. §7.4 settles it — the oracle is *given* `triggers_met`, so the register's
job is to name the conditions and the test that drove the request declares which it arranged.

**`conditional` is a second field, not a consequence of the trigger.** It answers one question:
does §3.3.2 assertion 2 apply — must a corpus entry exist in which this row's mutation is provably
**absent**. A trigger is either a property of the *route* or a property of the *request*, and only
the second can be varied by a corpus entry. P13's `CC_ORIGIN_PATH` is a route property; every
request on that route meets it, so there is no complement to write. M2 and M10 are the same shape
and were already exempt, which is why this reads as a rule rather than a P-row exception.

**Two fields that must agree are cross-checked in the data**, because leaving them uncompared
reproduces the D1 defect inside `register.py`: a row carrying `ALWAYS` may not be conditional, and
rows sharing a trigger must agree on whether it is.

**Expected shapes are T-D1's, not the register's.** P2a and P2b both land on
`envelope.extra[thinking]` and differ only in the value injected — `{"type": "enabled"}` against
`{"type": "disabled"}`. The §3.3 diagram's third step, "assert the row's stated shape held", is
therefore written per-row in oracle test code. The register says *where* a mutation may appear and
*when*; it does not say what the value must be. M3/M4 and P5a/P5c have the same property.

**Scope is not carried here, and the site does not supply it.** §6.2.3's completeness guard and
T-D8 both need to know, per adapter, which rows are reachable. Reading that off the site's class is
wrong twice: P8's site is `ProviderAdapter._inject_empty_reasoning_content`, a **base class**
method that reads as all 23 adapters while only four call it (`kimi`, `custom_openai`,
`zai_regular`, `zai_coding_cc`); and P5a–d's site is `AnthropicAdapter.translate_to_upstream`,
which four subclasses override *and conditionally delegate back to*, so the row is also reachable
on `custom_anthropic`, `zai_coding`, `minimax_token` and `opencode_go` — which no static rule over
the class hierarchy finds. Authoring scope now would ship data **nothing in T-W3 could prove
wrong**, since no wire-level capture exists yet to contradict a bad entry, and that is what
plan §1.4's harness rule forbids. It is filed as **KBR-139** rather than guessed at.

**A declared trigger is not a verified one**, and that is a known gap — see G21 in §9.2.

#### 3.2.5 What T-W3's guards own, and what they do not

§6.2.3's register row and this section are easy to read as one thing. They are two, at two layers:

| Guard | Layer | Task | Reads |
|---|---|---|---|
| Data ⇄ §3.2 markdown — ids and conditionality | L2 | **T-W3** | Two files. No sockets |
| Data ⇄ source tree — every site resolves | L2 | **T-W3** | The AST of `src/kitty`. No sockets |
| Register completeness — projected delta at the wire equals the triggered rows | **L3** | T-G2 | Captures from T-D4–T-D9 |

The first two are what make the register *well-formed*. Only the third makes it *true*, and it
cannot run until a recorder and an oracle exist.

**None of the three proves the register is *complete*.** They prove the data and the document say
the same thing, and that every site named still exists. A mutation the product performs and
*neither* artifact records is invisible to all of them — only the wire-level guard can catch that,
and it needs a recorder and an oracle. Two such omissions are already known and filed: G22
(headers) and G23 (`openai_subscription`'s `reasoning` injection), the second found by walking the
subscription request path by hand while writing the data.

**A header row's `paths` are checked by none of the three.** Under §3.2.2's header rule they are
consumed by §4.3 C1's exact-set assertion, which does not exist yet — so for P9a, P9b and P9c only
the id, the conditionality and the sites are under test today. `headers[user-agent]` is a reviewed
claim, not yet a tested one.

### 3.3 The transparency oracle

The single piece of new infrastructure that makes I1 testable.

**What it is.** A test harness that runs a request through the real `BridgeServer` into a
recording upstream, projects both the inbound agent body and the captured upstream body into a
**wire-independent semantic form**, and classifies every difference against the register.

```
inbound agent body ─────► BridgeServer ─────► recording upstream (final wire bytes)
        │                                                 │
   project()                                         project()
        │                                                 │
        ▼                                                 ▼
  Conversation                                      Conversation
        └──────────────── structural diff ────────────────┘
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

#### 3.3.1 The comparison is an independent projection, never a round-trip

An earlier draft of this design specified comparing the inbound body against a **round-trip**
through the production translators — `translate_request` then `translate_response`. That is
wrong twice over, and the reason is worth recording so nobody proposes it again.

**It is a category error.** `MessagesTranslator.translate_request` maps a Messages *request* to a
Chat Completions *request*. `MessagesTranslator.translate_response` maps a Chat Completions
*response* — a body whose payload is `choices` — back to a Messages *response*. They are not
inverses and never were: composing them takes a conversation in and returns an assistant reply,
so the input history cannot survive the trip. The same objection applies to any "inverse pair"
assembled from the provider adapters.

**Even a real inverse would prove the wrong thing.** Using the production translator to check the
production translator establishes self-consistency, not fidelity. A translator that drops the
same field in both directions round-trips perfectly. The oracle must not be written in terms of
the code under test.

**The projection must be total, or it hides exactly what it is looking for.** A projection that
keeps only the semantic conversation cannot see a changed model, a dropped `stream`, a stripped
tool `description`, or an injected metadata field — both sides project identically and the "no
unclaimed delta" assertion passes. That blind spot would swallow **M1**, the model override that
is the product's entire purpose, along with P6, P15, P17, P18 and P19. So the projection covers
the whole request and accounts for every field:

```
Request                          -- the complete request; nothing is dropped silently
  envelope:     Envelope         -- routing and control
  conversation: Conversation     -- semantic content
  residual:     {path: value}    -- anything the reader did not account for
  consumed:     {key}            -- top-level body keys the reader DID account for
  source:       {key: value}     -- the mapping the reader parsed

Envelope
  model, stream, store, and every other control field the format defines
  in `extra`, keyed by the wire key
  (Bedrock's modelId normalises onto `model`; Responses' store lives here too)

Conversation
  system:   ordered text parts
  turns:    ordered [ Turn(role, parts) ]        -- role is `user` or `assistant`, only
  tools:    ordered [ ToolDecl(name, description, schema, strict) ]
  sampling: a CLOSED set of fifteen canonical keys (§3.3.1b) -- declared, may be absent

Part = Text(str)
     | ToolUse(name, arguments, id?)
     | ToolResult(content, tool_use_id?, is_error)   -- content: [ Text | Image | Json | Opaque ]
     | Thinking(text, signature?)
     | Image(digest?, media_type?, ref?)
     | Json(value)
     | Opaque(kind, digest?)
```

**`consumed` is why a dropped key is detectable.** A reader that *drops* an unknown key produces
an **empty** residual, so "the residual must be empty" would pass it — and T-W2's own falsification
case is a stub reader that drops an unknown key. Totality is decidable only against the source
body, so the projection records what the reader claimed to handle. `consumed` covers top-level keys
and catches drops; the **path-keyed** residual covers nesting and fails closed, which matters
because Gemini puts every sampling parameter under `generationConfig` and Converse nests
`inferenceConfig` and `toolConfig`.

> **The boundary, stated so nobody over-reads a green run.** `consumed` holds *top-level* keys, so
> a reader that claims `generationConfig` and **silently drops** `topK` inside it **passes**
> `verify_total`. Catching that would require the contract to walk the body itself — making it a
> second reader, which the independent-oracle rule forbids. What closes it instead: each reader's
> own L1 tests against its format's published examples (§7.4), and **T-D8**, which owns "residual
> empty across the whole corpus" for all seven readers. T-W2 pins this boundary with its own test,
> so it stays a decision rather than an assumption.

**Optional ids, because a format may carry none.** Where an id is absent, pairing is by tool name
and the k-th unanswered call of that name in the most recent assistant turn. A required id would
force such a reader to synthesise one and show a delta on every tool turn.

> ⚠️ **Corrected by T-A4 (KBR-36).** This section previously read "two formats have none" and named
> Gemini as one of them, on the strength of Google's Cloud / Agent-Platform reference. The
> **Developer API** surface kitty actually serves differs: `v1beta`'s `FunctionCall` publishes an
> optional `id` ("If populated, the client to execute the `function_call` and return the response
> with the matching `id`") and `FunctionResponse` an optional `id` the client populates to match —
> verified against the discovery document at revision `20260910`. Optional either way, so the
> *decision* to make `ToolUse.id` and `ToolResult.tool_use_id` optional is unchanged and still
> right; only its stated reason was wrong. The Gemini reader therefore **reads the wire id when one
> is sent** and falls back to the name-and-position rule when it is not. KBR-36's own acceptance
> asked for this to be confirmed rather than assumed, which is why it is recorded here rather than
> left as a reader's private finding.

**`ToolResult.content` is wider than text and images**, because Converse's `toolResult.content`
carries `json` (the common case), `document`, `video` and `searchResult`, Anthropic's carries
`document` and `search_result`, and Gemini's `functionResponse.response` is a bare struct. `Json`
carries structured results; `Opaque(kind, ...)` keeps the rest **detectable** without modelling six
vendors' block zoos, with `kind` a canonical snake_case name rather than the wire's spelling.

**An empty block is a part with an empty string, never nothing.** P5e injects an empty `thinking`
block and P8 an empty `reasoning_content`; P8's trigger is conditional and *inferred*, so §3.3.2
assertion 2 needs its absence to be observable. `Thinking.signature` carries what M8's carrier
repair manipulates.

**`Image.digest` is the lowercase hex SHA-256 of the decoded bytes**, with `media_type` excluded
from it and carried separately, so a changed media type is its own delta. Gemini's
`fileData.fileUri` has no bytes: `digest` is then absent and `ref` holds the URI. Unpinned, the
Messages reader and the Chat Completions reader would produce different digests for one image.

The contract lives in `tests/harness/contract.py` (T-W2). The **package** `tests/harness/` is the
home of T-W4's recorder, T-W5's proxy fixture, T-W6's corpus loader and T-W8's bridge fixture, each
in its own module beside it — `contract.py` itself captures nothing and reads nothing.

**Unknown fields fail closed.** Each reader must classify **every** key in the body into exactly
one of: mapped to the envelope, mapped to the conversation, or residual. A non-empty `residual`
on either side **fails the run** — it is not reported as a diff and it is not ignored. An
unaccounted field is precisely where an unregistered mutation hides, and a reader that quietly
skips what it does not recognise is a reader that cannot prove completeness. Adding a field to a
wire format therefore forces a deliberate decision: map it, or declare it ignored with a reason.

**Every register row names the field it touches.** M1 is `envelope.model`; P17 is
`envelope.stream` and `envelope.store`; P15 is `conversation.tools[*].strict`; P13 is
`conversation.sampling`. Without that, "claimed by a register row" is a judgement call rather
than a lookup.

#### 3.3.1a The path vocabulary

T-W2 owns the string form, because it has **two** consumers that must agree exactly: a delta the
oracle reports (§3.3.4), and the "projection field it touches" column of every register row
(T-W3). Neither can define it without the other agreeing. `headers[<name>]` is the one form with
only the second consumer; §3.2.2 says why.

| Path form | Names |
|---|---|
| `envelope.model` · `envelope.stream` · `envelope.store` | The named control fields |
| `envelope.extra[<wire key>]` | A format-specific control field — P2a `thinking`, P3 `reasoning`, P4 `reasoning_effort`, P10 `reasoning_split` |
| `conversation.system[<i>]` | One system text part |
| `conversation.turns[<i>].role` · `.parts[<j>]` | A turn, or one part of it |
| `conversation.tools[<name>].description` · `.schema` · `.strict` | A tool declaration, **by name** |
| `conversation.sampling[<key>]` | One sampling parameter |
| `conversation.turns` · `.system` · `.tools` · `.sampling` | A **whole collection** — M5, M6 and M7 rewrite the turns, P5b joins the system blocks, §3.3.1 pins P13/P14 to the bare `sampling` |
| `headers[<name>]` | A header — P9a, P9b, P9c, and §4.3 C1. **Not produced by the projection diff**: `Request` carries no headers and no inbound header is forwarded, so this form addresses a per-adapter *deviation from the base header set* (§3.2.2), never a delta between two projections |
| `residual[<path>]` | An unclassified value |
| `reply.parts[<i>]` · `reply.stop_reason` · `reply.usage[<key>]` | The response direction — M12, T-D10 |
| `route.method` · `.scheme` · `.host` · `.path` · `.query` | The route (§3.3.5) — M14, P20, P21 |

**Tools are addressed by name, not index**, because translators reorder and filter declarations; a
positional path would report a delta whenever the order changed and the declaration did not.

**Two kinds of path, and a matcher.** A register row writes a **pattern** with the `[*]` wildcard
(`conversation.tools[*].strict` — every tool); a delta is **concrete**
(`conversation.tools[get_weather].strict`). §3.3.2 assertion 1 is literally a match of one against
the other, so T-W2 supplies the predicate rather than leaving T-W3 to write patterns and T-D1 a
matcher that agree only by luck.

**A pattern is a prefix.** It names its node and everything beneath it, at any depth — so a row
anchored at `conversation.turns[*].parts[*]` claims
`conversation.turns[2].parts[0].signature`, which is what M8's carrier repair produces.

The rule is deliberately **asymmetric**, and the asymmetry is the reason to prefer it:
under-claiming manufactures a *false* I1 breach, failing the run over a mutation that **is**
registered; over-claiming is silent. So the error the matcher can make is the recoverable one — but
it is recoverable only if the register is written carefully:

> ⚠️ **A row must be anchored at the *narrowest* path that covers its effect.** A coarser anchor
> silently claims every delta beneath it. Anchoring P15 at `conversation.tools[*]` rather than
> `conversation.tools[*].strict` would claim a *deleted tool description* — which is one of
> §3.3.1's own five oracle falsification cases. The matcher cannot catch that; **T-W3's anchoring
> discipline and T-D3's falsification case (mutate a field beneath a registered anchor and assert
> the oracle still fails) are what keep it honest.**

**The prefix stops at a bracket.** Bracket contents are literal and are never re-parsed — which is
what makes `residual[generationConfig.topK]` legal — so a pattern naming a *parent key* does not
claim paths nested under it. `residual[generationConfig]` does **not** match
`residual[generationConfig.topK]`; `residual[*]` and the bare `residual` both do. This is a second
rule sitting beside the first and it is the one that surprises.

**`envelope.extra` is diffed one wire key at a time.** The value under a wire key is compared
**whole**: a difference anywhere inside it is reported at `envelope.extra[<wire key>]`, never at a
dotted sub-path. `envelope.extra[thinking.budget_tokens]` is **not** a path this vocabulary
defines and a reader must not emit one. The dotted-bracket form exists for `residual` alone,
because §3.3.1 gives nesting to the residual deliberately — `consumed` covers top-level keys, the
path-keyed residual covers what nobody classified. A key in `extra` *has* been classified, so its
address is the key.

This is what makes P5c's anchor correct rather than lucky: `AnthropicAdapter.translate_to_upstream`
writes `thinking` whole — `{"type": "enabled", "budget_tokens": max_tokens - 1}` — over whatever
the agent sent, so the delta is at the key. P2a, P2b, P3, P4, P5d and P10 are anchored the same way
for the same reason. Under the other spelling every one of those six rows would match nothing and
§3.3.2 assertion 1 would report a false I1 breach on six *registered* mutations — the under-claiming
direction this section warns is the unrecoverable one. `extra_path()` **enforces** the rule: a key
containing a dot raises, so a reader cannot emit the nested form by accident.

**The cost, recorded so it is not discovered later.** `envelope.extra[<key>]` is the narrowest
address the vocabulary offers, so a row anchored there claims everything inside that key by
construction. **T-D3's falsification case — mutate a field beneath a registered anchor and assert
the oracle still fails — therefore cannot be sited under `envelope.extra`.** It needs a path with
addressable depth: `conversation.tools[*]` versus `conversation.tools[*].strict`, which is the
example this section already gives.

`[*]` is the wildcard. `[]` is accepted as its **legacy spelling**, because §3.3.1 wrote P15 as
`conversation.tools[].strict` before this vocabulary existed and a row carried over in the old
notation must not silently match nothing. An unbalanced bracket **raises** rather than mis-splitting
the path.

**`not projectable` is a legal value for the register's field column, and it requires a reason.**
P16 uses it — the `input_text`/`output_text` tag is redundant with the turn's role, so carrying it
would put one vendor's spelling into a wire-independent form — as do the whole-body protocol
translations M2, M9, P11 and P12, **P1** for the reason below, and **M15**, whose two spellings of
a Responses `input` are one request (KBR-144). An empty cell would leave those
rows silently unfalsifiable; an explicit value with a reason does not.

⚠️ **`residual` is never a legal register anchor.** P1 strips kitty's internal keys, which no
reader maps, so its effect can only ever appear *as* a residual — which makes `residual` look like
the natural anchor. It is the opposite. A bare collection claims its members, P1's trigger is
`Always`, and a non-empty residual **fails the run before register matching happens at all**
(§3.3.1). Such a row could therefore only ever claim a delta the oracle was supposed to stop at:
the injected `x-kitty-trace` field that is one of §3.3.1's five mandatory falsification cases, and
a real internal-key leak — the defect P1 exists to prevent. P1 takes the escape instead.

**The index builders accept the wildcard (KBR-26).** `system_path`, `turn_path`, `part_path` and
`reply_part_path` take `WILDCARD` where they take a position, because a register row writes a
pattern over every turn and part — M3, M4, M8, P5e and P8 all do — where a delta writes concrete
indices. Both come from one builder, or the spelling drifts between T-W3 and T-D1, which is the
drift this vocabulary exists to prevent. An index that is neither a position nor the wildcard
raises: `mypy` covers `src/kitty` only, so a typo would otherwise build a path that looks concrete
and matches nothing.

#### 3.3.1b Normalisation rules the six readers share

The claim that "a conversation is a conversation" holds only if six independently written readers
agree on a canonical form. They are six separate tasks, so the agreement is part of the contract.

- **Roles** are `user` or `assistant`, and nothing else. Gemini's `model` maps to `assistant`.
- **System instructions lift into `Conversation.system`**, never into a turn — from a dedicated
  field (Messages, Converse, Gemini `systemInstruction`), a `role: "system"` message, a
  `role: "developer"` message, or Responses' `instructions` field.
- **A tool result is a `ToolResult` part inside a `user` turn**, by a **merge rule**: a maximal run
  of consecutive tool results forms one turn; an immediately following non-tool user message merges
  into it; `ToolResult` parts come first; consecutive same-role turns merge. *An orphan tool result
  still projects, in the turn where it occurred* — M7 exists to drop orphans, so a reader that
  raised on one would fail instead of producing the delta that names it.

  **The four clauses are an ordered pipeline.** `ToolResult` parts come first *within the turn the
  **run** and **following-message** clauses build*, never as a re-sort after the same-role merge —
  so `ToolResult → user text → ToolResult` projects as `[ToolResult, Text, ToolResult]`: two runs,
  the first absorbing the text that follows it, then concatenated. Re-sorting afterwards would
  hoist a result ahead of text the agent sent **before** it — moving history the bridge did not
  move — and because paths are index-based the invented delta would land on every part of that
  turn and every turn after it. The rule gives a run of results one home; it does not reorder
  history. Clause 3 is not thereby idle: it governs the formats that carry text and results inside
  **one message**, where the run has no natural boundary — Anthropic Messages is that case, and
  §7.4.1 records what each format's reader does with the pipeline.

  A lift rule ("into the user turn that follows the assistant turn") does **not** work: the standard
  Chat Completions exchange ends `assistant(tool_calls) → tool → tool`, with no following user
  message at all. And because paths are index-based, any disagreement about turn boundaries reports
  a delta on *every* subsequent turn.
- **`modelId` and Azure's deployment id normalise onto `envelope.model`**, or P18 and P6/P20 cannot
  be expressed as `envelope.model` and a *moved* field looks *dropped*.
- **Sampling normalises to the Chat Completions spelling**, onto this **closed set of fifteen** —
  the fourteen P13 drops, plus `top_k`, which Gemini and Converse carry and Chat Completions does
  not:

  `temperature` · `top_p` · `top_k` · `max_tokens` · `max_completion_tokens` ·
  `frequency_penalty` · `presence_penalty` · `logprobs` · `top_logprobs` · `response_format` ·
  `stop` · `n` · `stream_options` · `seed` · `logit_bias`

  Responses' `max_output_tokens` maps onto `max_tokens`; `max_completion_tokens` stays distinct,
  because P13 drops it in its own right, so a reader must not collapse both. A key outside the set
  that the reader **recognises as a declared control field of that format** maps to
  `envelope.extra[<wire key>]` — only an *unrecognised* key residualises. Without that split,
  Gemini's `generationConfig.responseSchema` and Converse's `guardrailConfig` would fail the run as
  unaccounted fields, on the two formats the CC-shaped set was not derived from.

  The set is **enforced**, not merely declared: `Conversation` rejects a non-canonical sampling key
  the way `Turn` rejects a role outside `user`/`assistant`. Six readers cannot quietly disagree
  about whether `n` is sampling.
- **`extra` is keyed, never nested.** A reader emits one entry per wire key and the oracle
  compares its value whole. No reader emits `envelope.extra[<key>.<subkey>]`; nesting belongs to
  the residual (§3.3.1a). Six readers cannot quietly disagree about whether `thinking.budget_tokens`
  has an address of its own.
- **Tool-call arguments decode through one shared rule.** Chat Completions and Responses both
  carry them as a JSON *string*. Absent, `null`, empty or whitespace decodes to `{}`; a value that
  is valid JSON but not an object, or not valid JSON at all, decodes to `{}` **and residualises at
  that argument's own path**. Never raises.

  **Why an absent `arguments` does not residualise though an absent `name` does**, the schema
  requiring both: the test is whether the projection can represent the absence *losslessly*.
  `ToolUse.arguments` is a mapping defaulting to empty, and `{}` is a true statement about the
  call — seen and classified, therefore accounted for, the same ground on which `STOP_REASONS`
  gives `other` its escape instead of the residual. `ToolUse.name` is a `str` with no such value:
  `""` claims a tool *named* empty-string, and a call nobody can name cannot be paired or
  addressed. Apply that test to every required field, not only these two.

  Six readers cannot quietly disagree about what `arguments: ""` means, and it is not
  hypothetical — `openai_subscription.py:OpenAISubscriptionAdapter._cc_to_responses` writes
  `func.get("arguments", "")`. The residual *path* is format-specific and stays each reader's own;
  only the decode and the fail-closed policy are shared. Tracked for pinning as code beside
  `image_digest` — KBR-174.
- **`tool_choice`** unifies four wire keys — CC/Messages `tool_choice`, Converse's
  `toolConfig.toolChoice`, Gemini's `functionCallingConfig.mode` — onto
  `envelope.extra["tool_choice"]`, with the **value** normalised to `auto` · `any` · `none` ·
  `tool:<name>`. This is the one deliberate exception to keying `extra` by the wire key, because
  four spellings name one concept.
- **On the response direction**, `stop_reason` is `end_turn` · `max_tokens` · `stop_sequence` ·
  `tool_use` · `error` · `other`, where `other` keeps the wire's own string in
  **`Reply.stop_reason_raw`** — Gemini adds `SAFETY` and `RECITATION`, and a closed set with no
  escape would fail the run on a legitimate safety-blocked reply.

  **Not in the residual**, and the reason generalises: a non-empty residual *fails the run*, so
  building an escape out of the residual defeats the escape. A value mapped to `other` has been
  seen and classified — it is accounted for. The residual means only *nobody has looked at this*.

  **The pairing is enforced, both ways.** `other` without `stop_reason_raw` is rejected, because a
  reader that maps both `SAFETY` and `RECITATION` to a bare `other` has discarded exactly what
  T-D10 needs; and a `stop_reason_raw` beside a canonical reason is rejected as a stale leftover.
  An invariant stated only in a docstring is a comment, not a rule — the same posture the closed
  vocabularies take.

  **`usage` is carried but excluded from the diff**: it is provider-reported, never agent-supplied,
  so a difference carries no I1 information.

Then write one **hand-written reader per wire format** — Anthropic Messages, Chat Completions,
OpenAI Responses, Gemini, Bedrock Converse, Ollama `/api/chat` — each written directly against
that format's published shape and **importing nothing from `src/kitty/bridge`**. Six small
readers, each independent of whatever kitty code produces that format. The oracle compares
`project(inbound)` with `project(captured_upstream_bytes)`, field by field, across all three
parts.

This is the independent-oracle rule, and it is what makes the check meaningful across a protocol
boundary: a Messages body and a Chat Completions body are not comparable as JSON, but their
projections are directly comparable, because a conversation is a conversation.

**The oracle gets its own falsification control.** The same discipline §5.2.2 phase 3 applies to
the containment harness applies here: a fidelity oracle never shown to fail is indistinguishable
from one that cannot fail. Five injected mutations must each produce a failure, and they run as
part of the suite, not once by hand:

| Injected | Must be caught as |
|---|---|
| Change the model sent upstream to a value no profile set | unclaimed `envelope.model` delta |
| Flip `stream` on an otherwise unchanged request | unclaimed `envelope.stream` delta |
| Delete one tool's `description` | unclaimed `conversation.tools[].description` delta |
| Strip `strict` from a tool where no register row applies | unclaimed `conversation.tools[].strict` delta |
| Inject an unrecognised `x-kitty-trace` field into the body | non-empty `residual` — fails closed |

The last of these is also what keeps the vendor-token check (§3.3.3) honest: bridge-added
metadata that the projection discarded could never have been scanned for a vendor string.

**Response translation is tested separately**, with its own projection (`Reply(parts, stop_reason,**Response translation is tested separately**, with its own projection (`Reply(parts, stop_reason,
usage)`) over the response direction. It is a different claim and it gets a different test.

#### 3.3.2 The two assertions

1. **No unclaimed delta.** Every difference between the two projections must map to a register
   row whose trigger the input met. This is the invariant.
2. **No mutation without its trigger.** For each conditional row, an input that does *not* meet
   the trigger must show that row's mutation **absent**. This is what stops M3/M5 quietly
   becoming unconditional, and it is why §3.3.4 insists on trigger complements.

#### 3.3.3 Bridge-introduced content is what gets the vendor-token check

The projection also settles a problem a naive I2 check cannot (§4.3 C2). "The upstream body must
not contain the string `kitty`" is **wrong** — a user may legitimately ask Claude Code to explain
kitty-bridge, a repository path may contain the word, or a tool result may quote this very
document. All of that must reach the provider **unchanged**, or I1 is broken in the act of
defending I2. The two invariants would be in direct conflict.

Diffing projections separates the two cases cleanly:

- A part present in the upstream projection with a counterpart in the inbound projection is
  **agent-supplied**. It is never inspected for vendor tokens, whatever it contains.
- A part with no inbound counterpart is **bridge-introduced**. Only that set is scanned.

The regression case is explicit, and belongs in the corpus (§7.1): an inbound turn whose text is
`Please explain how kitty-bridge works` must survive byte-identically, **and** an injected vendor
message must still be caught in the same run. A harness that cannot do both at once has
not solved the problem.

Since KBR-5 there is no *live* injected vendor message to use as the positive half — M13 was the
only one. The fixture is therefore the synthetic historical M13 string held in
`tests/bridge/test_vendor_token_guard.py`, which T-G5 inherits. A guard whose positive control
disappeared with the defect it caught is a guard that has quietly stopped working.

#### 3.3.4 Scoping, triggers and transports

**Scoped by observed wire shape, per request.** The declared wire shape is the natural selector
but must not be trusted as an adapter-level constant. `OpenCodeGoAdapter` used to inherit `True`
from `AnthropicAdapter` while emitting Chat Completions for every model outside
`_MESSAGES_MODELS` (F5, KBR-7); that is fixed, and the declaration is now per-model. **The
decision here stands regardless**, for the reason given in §7.4 rather than because of that one
bug: an oracle must not ask the code under test what shape it emitted, and a declaration is a
claim, not an observation. So the oracle selects the projection by the shape actually observed on
the wire, and the L2 guard (§6.2.3) separately asserts that the declaration agrees with that
shape for every adapter × representative model.

**Triggers and their complements, across representative models.** A single fixed request cannot
establish register completeness. For every conditional row the corpus must contain a case that
meets the trigger and a case that does not, and both must run against every adapter for which
the row is reachable — including each adapter's distinct model classes where routing differs
(`opencode_go` Messages vs. CC models; `fireworks` streaming vs. non-streaming for P7).

**Parametrised over transport.** `openai_subscription`, `bedrock` and `ollama_cloud` set
`use_custom_transport = True` and never reach `_make_upstream_request`, so an aiohttp recording
upstream never sees their bodies. The oracle runs against the recorder appropriate to each
transport (§7.2). Without this, "every provider" is false for three adapters — and those three
are where the least-inspected serialization code lives (§3.2.3).

**Diffing semantics.** Structural, over projections. JSON key order and whitespace are not part
of the agent's meaning and vary with serialisation; a byte diff would fail constantly and teach
people to ignore it. Deltas are reported as paths into the `Conversation` so a failure names the
exact turn and part. The one byte-level exception is key ordering on the native passthrough
path, which §4.3 C2 asserts for I2 reasons.

**What it does not do.** It does not judge whether a translation is *semantically apt* — that a
`tool_use` block became the right `tool_calls` entry is an L1 translator claim. The oracle
answers a narrower question: was anything changed that nobody declared.


#### 3.3.5 Routing is part of the request, and the body cannot show it

The envelope fixed the missing-body-fields problem, but a body-only oracle still cannot see where
the request went. Three providers carry routing outside the body:

| Provider | What lives in the URL | Consequence |
|---|---|---|
| Azure | The deployment id, which **is** the request's normalized model (P20) — and P6 deliberately removes `model` from the body | Two requests to two different deployments have **byte-identical bodies**. A body-only oracle cannot tell them apart, so a misrouted request is invisible. |
| Vertex | `project_id` and `location` (P21) | The account being billed is a URL component |
| Gemini | Model and operation (`:generateContent` vs `:streamGenerateContent`) in the inbound path | M10 lifts the model into the **outbound Chat Completions** body precisely because it is not in
the inbound one to begin with |

So the oracle takes the **whole captured request** — method, scheme, host, path, query, headers,
body — and asserts routing separately from content.

**The routing expectation is derived independently.** It is computed in the test from the
configured profile — provider, model, `provider_config` — using the provider's *published* URL
shape, not by calling `build_base_url()` / `get_upstream_path()`. The model half of that
derivation is `normalize_model_name(profile.model)` when the profile names a model, and the
model the agent asked for when it does not: since KBR-127 the path resolves from
`cc_request["model"]`, so deriving it from the profile's raw string would encode the very defect
KBR-127 removed. Asking the code under test where it meant to go and then checking it went there
proves nothing; this is the same independent-oracle rule §3.3.1 applies to bodies.

**One normalisation the independent derivation has to reproduce (KBR-134).** `build_base_url()`
strips a trailing endpoint suffix from a user-configured base URL, because users routinely paste
the full endpoint their provider's documentation shows and the bridge would otherwise compose a
doubled path. A profile whose `base_url` already ends in `/chat/completions` therefore reaches the
same destination as one that does not. T-D2 must apply the same rule when it computes the expected
route, or that profile reports a routing mismatch against a request that went exactly where it
should. This is the awkward edge of the independent-derivation rule — the expectation has to
reimplement a behaviour rather than observe it — and it is recorded here because the alternative is
T-D2 discovering it as a failing test with no obvious cause.

**Falsification control.** Alongside the five body cases in §3.3.1, a sixth: change the Azure
deployment segment in the captured path while leaving the body byte-identical. The oracle must
fail. A seventh, from KBR-127: give the profile a prefixed model (`opencode/minimax-m2.5`) and
assert path, auth scheme and body shape agree — the defect showed a correct body reaching a
correct-looking host at the wrong path under the wrong auth, which each of the three checked alone
would pass. Without these cases there is no evidence the routing assertion is wired to anything.

**T-D2 must normalise the authority and scheme before comparing, and this is not optional.**
A published URL shape is `https://…` on the provider's own hostname; the harness serves
`http://127.0.0.1:<ephemeral>`. T-W4's recorder captures the `Host` header **verbatim,
port included** — correct, because that is what the client sent, and because the ephemeral
port is the only thing distinguishing one recorder from another. The consequence is that
`route.host` and `route.scheme` mismatch **by construction** on every comparison unless the
expectation is rewritten with the harness's own authority and scheme. Path and query need no
such treatment, and they are where the Azure case lives. Stated here because nothing else
assigns this, and a reader discovering it at T-D2 time would reasonably conclude the recorder
was wrong.

The recorders already capture method, path and query (§7.2). The gap was that the assertion did
not consume them.

### 3.4 Where I1 is proven

| Claim | Layer | Test |
|---|---|---|
| Compaction preserves `tool_use`/`tool_result` atomicity, in both CC and native shapes | L1 property | `compact(m)` contains no orphan |
| Compaction output fits the budget **unless the surviving set is irreducible** | L1 property | See §6.1 — the honest exception is wider than an oversized system block |
| The last turn survives **unless it is itself truncated (M3/M4) or dropped by pairing validation (M7), in which case the request is refused downstream** | L1 property | Stated with all three exceptions, or it fails on day one. Since KBR-5 the third case raises `CompactionFailedError` rather than substituting a turn |
| **No upstream request is ever made after a `CompactionFailedError`** | L1 + L3 | The load-bearing invariant behind KBR-5. Note it is *not* "the request is left untouched": `_apply_compaction` raises after it has already replaced `cc_request["messages"]`, so the guarantee is about the absence of an upstream call, not about the request dict |
| Compaction is identity below the budget | L1 property | Short-circuit at the `original_size <= compaction_threshold` guard — M5's trigger, tested directly |
| Truncation is identity below `_TOOL_RESULT_TRUNCATION_LIMIT` | L1 property | Same shape, for M3 and M4 |
| Each wire projection reads its format correctly | L1 | The projections are test code and get their own tests — against published format examples, not against kitty's output |
| A **below-threshold** corpus entry on a native-wire provider shows only M1 and P1 | L2 | Scoped to below-threshold deliberately: above it, M3/M5/M7 and any triggered provider row apply to the native path too |
| No unclaimed delta, over the whole corpus, on every adapter × representative model × transport | **L3** | Transparency oracle (§3.3) |
| The request reaches the destination the profile implies — host, path and query | **L3** | §3.3.5. Independently derived from the profile, never from `build_base_url()` |
| A changed deployment path with an unchanged body is caught | **L3** | §3.3.5 falsification control — the case a body-only oracle cannot see |
| An inbound `kitty` string survives while an injected vendor message is caught | **L3** | §3.3.3 regression case |
| A real Claude Code session's tool calls execute correctly through kitty | L4 | Real-agent E2E (§6.4.2) |
| Compaction has not degraded answer quality | L4 eval | §6.4.3 |

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
| **C1 — Request headers** | `build_upstream_headers()` constructs the set from scratch; no inbound agent header is forwarded. Four adapters supply a coding-agent `User-Agent` (P9a, P9c); every other provider — including `zai_coding`, whose set is exactly `Authorization`, `anthropic-version`, `content-type` — sends aiohttp's default. | **Gap, and inconsistent.** F1. |
| **C2 — Request body** | The register's mutations (§3.2), JSON key ordering produced by kitty's serialisation, ~~the literal string `[Kitty Bridge: …]` (M13)~~ **— fixed, KBR-5** — and **`_effort` / `_thinking_adaptive`, which are kitty-internal and reach the wire**. With M13 gone the only bridge-introduced literal left in the body is `[Tool output truncated — original size: N chars]` (M3/M4): still a viable fingerprint, it simply does not name the product. | **Still breached by F4.** F3 closed. |
| **C3 — Cross-attempt content and cadence** | Retries (`_MAX_RETRIES = 3`), failover, transport-blip re-connects, the empty-response ladder — and **four** paths that send a *different body* on a later attempt (M6, M8, M9, and failover re-normalisation). | §4.3 C3. Four declared exceptions. |
| **C4 — Transport fingerprint** | TLS/ALPN/HTTP-2 signature of aiohttp, unlike the agent's own client. `curl_cffi` is already used for the OpenAI subscription provider precisely because that provider fingerprints TLS. | **Accepted residual risk.** §4.5 — including the narrower, *provider-specific* residual KBR-161 leaves on the OpenAI login leg, which the general argument does **not** cover. |
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
the build on day one — a reported baseline with a ratchet (the monotonic kind, not the §8.3
exemption), becoming a gate once G3 closes.

**C2 — Body shape (L3).** Covered by the transparency oracle (§3.3), plus two assertions the
oracle's projection makes possible and a flat scan cannot:

- **No vendor token in *bridge-introduced* content.** The check is emphatically **not** "the
  serialized body must not contain `kitty`". A user may ask Claude Code to explain kitty-bridge;
  a path may contain the word; a tool result may quote this document. All of that must reach the
  provider unchanged — stripping it to satisfy I2 would breach I1, and the two invariants would
  be in direct conflict. The oracle's projection diff already separates agent-supplied parts
  from bridge-introduced ones (§3.3.3); only the latter are scanned. M13 was caught because it
  had no inbound counterpart, while the user's sentence is not, because it does. M13 is fixed;
  the worked example stands because the *next* bridge-introduced string will be caught the same way.
- **Key order preservation on the native path.** A provider can fingerprint the JSON serialiser
  from key ordering alone. This is the one place the comparison is byte-level rather than
  projected, and it applies only where kitty claims to be forwarding rather than translating.

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

Five defects surfaced while writing this document. **None was fixed by this change** — it added
documentation only; each needed its own ticket. F3, F4 and F5 were live breaches of invariants
defined above. **F5 has since been fixed under KBR-7**, together with the hook-level half of its
guard; see its entry below and §6.2.3. The rest remain open.

- **F1 — Agent identity is handled per-provider, not by policy.** *(KBR-8 — the self-contradiction
  **fixed**; the policy gap remains.)* Upstream headers are built from
  scratch, so Claude Code's `user-agent`, `x-app`, `anthropic-beta` and `x-stainless-*` never
  reach the provider. Four adapters compensate ad hoc (P9a, P9c): `KimiCodeAdapter`, `BytePlusAdapter`
  and `MimoAdapter` hard-code `User-Agent: claude-code/1.0` — Kimi's carries a comment recording
  the string was on the provider's allowlist as of 2026-04-18 — and `OpenAISubscriptionAdapter`
  synthesises a Codex CLI identity. Everywhere else, including `zai_coding`, aiohttp's default
  goes instead.
  **The subscription adapter contradicted itself in a single request — FIXED (KBR-8,
  2026-09-11).** Its user-agent was `codex_cli_rs/{kitty.__version__}` while its `version` header
  was the constant `0.128.0`. A client claiming to be Codex CLI 1.9.1 *and* 0.128.0 at once is a
  one-line detection rule, and the user-agent tracked kitty's release train, changing with every
  kitty release and with nothing else. **The finding's own text demonstrates it:** the defect was
  filed reading `1.9.0` and measured reading `1.9.1`, moved by a kitty release and nothing else.
  `_build_user_agent` now reads `codex_identity.CODEX_CLI_VERSION`, the same constant the
  `version` header carries, so the two agree by construction. **KBR-161 moved that constant out of
  the adapter** into the leaf `kitty.codex_identity`, because the OAuth token legs in `kitty.auth`
  present the same identity and cannot import `kitty.providers` back; both consumers read it at
  call time, never through a module-level alias, so patching one name moves all six fields. The behavioural guard is
  `tests/test_upstream_identity_consistency.py`, which sweeps every registered adapter through a
  mirror of `BridgeServer._build_upstream_headers` — plus the subscription adapter's
  `_build_codex_headers`, since it overrides no hook and would otherwise be invisible. It stands
  in until T-G9 / KBR-78 lands the exact-set contract, exactly as
  `tests/bridge/test_vendor_token_guard.py` stands in until T-G5.
  **What KBR-8 did not close:** identity is still ad hoc per adapter — three hard-coded
  `claude-code/1.0` strings, one synthesised Codex identity, and aiohttp's default everywhere else.
  That is the policy half of G3, and it waits on Q1.
- **F2 — The README's endpoint table does not match the router.** *(KBR-9.)* README documents
  `POST /v1/gemini/generateContent`; `_register_routes` registers
  `/v1beta/models/{model}:generateContent` and `:streamGenerateContent`, and the README omits
  `GET /v1/models`. Exactly the drift the L2 docs⇄code layer exists to catch. Tracked as G6
  and KBR-9.
- **F3 — FIXED (KBR-5, 2026-09-07).** The product's own name was written into the upstream request
  body. **Two corrections were made to this finding while fixing it, both measured against the
  running code:**
  1. **The trigger stated below is wrong.** The guaranteed-fit fallback always keeps at least one
     non-system block, so a surviving user turn always defeats the post-condition — a large system
     prompt *cannot* cause this. The set is emptied only by `_validate_tool_call_pairing` removing
     an unpaired tool result: a corrupt conversation, typically one where an empty upstream SSE
     response was recorded as a complete tool message. The same wrong trigger appeared in register
     row M13 and in the §7.1 corpus row, and both are corrected. The two existing tests for this
     path (`TestCompactionPostCondition`) never reached it and passed vacuously.
  2. **There was a second, unguarded site.** `_apply_compaction` re-runs pairing validation *after*
     `_compact_messages` returns, and `_compact_messages` short-circuits below the compaction
     threshold — so a below-threshold conversation could be emptied with no post-condition
     anywhere, and a system-only body went upstream. Closed in the same change.

  The fix: `_compact_messages` and `_apply_compaction` raise `CompactionFailedError`; the four
  handlers render a protocol-native HTTP 400 carrying `error.reason == "compaction_failed"`. The
  recovery path (`_compact_with_tighter_budget`) fails over to the next backend **without** marking
  it unhealthy, since no second upstream request was made and cooling the pool down for one corrupt
  conversation would 503 every concurrent session.

  **The recovery-path guard is defence in depth, not a reachable path — measured.** Pre-flight
  `_apply_compaction` strips every orphan tool result and raises if that empties the conversation,
  so anything arriving at the recovery re-compaction is already well-paired; `_compact_messages`
  groups `tool_use`/`tool_result` atomically and its guaranteed-fit fallback always keeps one
  non-system block. Four oversized shapes were driven through both stages and none reached the
  post-condition from recovery. The guard stays because `_compact_with_tighter_budget` does **not**
  re-run pairing validation, so a future change could make it reachable, and because the invariant
  should hold wherever compaction runs — but the handler tests for that site inject the exception
  deliberately, and say so, rather than pretending a fixture provokes it. Original finding, for the
  record:

  When compaction
  cannot preserve any non-system message, `_compact_messages` discarded the conversation and
  substituted a user message reading
  `[Kitty Bridge: Unable to compact conversation — the system prompt is too large relative to the
  model's context window. Use /clear to reset the conversation.]`. That message goes upstream via
  `_apply_compaction`. **A direct breach of I2** — the provider saw the vendor name in the
  request — and a fidelity mutation qualitatively unlike M5, since it replaced the conversation
  rather than shrinking it. The intent (a legible error rather than an opaque 400) was sound; the
  delivery was not. Register row M13 (withdrawn); tracked as G14, KBR-5 and Q9 (answered).
- **F4 — Kitty-internal keys reach the upstream body on every Chat-Completions-wire provider.**
  *(KBR-6.)*
  `MessagesTranslator.translate_request` writes `_effort` and `_thinking_adaptive` into the CC
  request, but neither is a member of `ProviderAdapter._INTERNAL_KEYS`, so the default
  `translate_to_upstream` — which strips only that frozenset — forwards both. Confirmed
  empirically by sweeping **all 23 registry entries** (KBR-6) on the translated Messages path, for
  an input carrying `effort` and `thinking: {"type": "adaptive"}`, each adapter constructed from an
  empty `provider_config`: 17 emit `['_effort', '_thinking_adaptive']` from
  `translate_to_upstream`, and 16 of those put that body on the wire. That is a sweep over
  adapters and still a sample over models and configs — `opencode_go` routes by model and
  `minimax_token` by config, so the regression test parametrises over adapter × route.
  The seventeenth, `openai_subscription`, is saved by a later stage —
  `_cc_to_responses` and `_prepare_responses_body` rebuild from an allowlist — which is §6.2.3's
  "the hook is not the wire" running in the opposite direction. An earlier draft of this finding
  named six providers; that was a sample, not a sweep. **A breach of both I1 and I2**, and
  the exact thing P1 exists to prevent. Underscore-prefixed keys no public API defines are an
  unmistakable proxy signature. Note that `_reasoning_effort` and `_thinking_enabled`, written by
  the same function, *are* in the set — so this is an omission, not a design choice. Tracked as
  G15 and KBR-6.
- **F5 — `OpenCodeGoAdapter.upstream_wire_is_messages_api` was wrong for most of its models.**
  *(KBR-7 — **fixed**.)* It
  inherited `True` from `AnthropicAdapter`, while its `translate_to_upstream` returned a Chat
  Completions body for every model outside `_MESSAGES_MODELS`. `ProviderAdapter`'s own docstring
  says the property "describes the shape that actually goes on the wire" and that anything
  shaping the serialized body must branch on it — so a `True` that was false for most models was a
  latent defect in the thinking-repair path (M8) as well as a trap for the oracle's scoping
  (§3.3.4). Tracked as G16 and KBR-7. The declaration is now per-model —
  `upstream_wire_is_messages_api_for_model(model)` mirrors `translate_to_upstream`'s own routing,
  and the bare property reports the adapter's default (Chat Completions) route — and both bridge
  repair sites read it from the `cc_request` the adapter routes on. The hook-level honesty guard
  landed with the fix; §6.2.3 records what it does and does not prove. §3.3.4's decision stands
  regardless: the oracle still selects on the observed shape, because an oracle must not ask the
  code under test what shape it emitted.

### 4.5 Accepted residual risk

**C4 — transport fingerprint.** Full parity is not achievable on the aiohttp serving path. A
provider determined to fingerprint TLS can distinguish an aiohttp client from the agent's own
runtime, and matching it would mean routing every provider through `curl_cffi` — a substantial
change to the serving path for a threat no provider is currently known to apply to this traffic.
Recorded so a future incident is a known gap rather than a surprise. The README's existing
guidance ("use a CONNECT proxy, not a TLS-terminating one") already depends on this reasoning.

**C4a — the OpenAI login leg, specifically (KBR-161).** The general C4 argument above does
**not** apply to this provider: `curl_cffi` is already a dependency, already constructed, and
already in use in this very adapter, so the cost that justifies accepting C4 elsewhere is absent.
KBR-161 therefore moved the **recurring** OAuth traffic — `_refresh` and `_exchange_api_key`, which
`get_valid_api_key` drives on every Codex request — onto an impersonating `curl_cffi` session, and
all four token POSTs now carry the Codex `User-Agent` from `kitty.codex_identity`.

The **interactive login** leg (`_exchange_code_for_tokens`, `_exchange_id_token_for_api_key`) stays
on aiohttp. It is reached from `cli/auth_cmd.py`, which has no adapter and no curl session; giving
it one means building an impersonating session inside the sign-in path, where a regression is
user-visible and blocks login outright. **State the residual honestly:** this is not an absent
exposure. Site 4 carries `Authorization: Bearer <access_token>`, so the request is bound to the
account, and it leaves by the same egress IP as every later API call. A genuine Codex CLI has no
non-Codex TLS handshake anywhere in its history (`codex-rs/login/src/auth/default_client.rs` builds
one client and `server.rs` posts `/oauth/token` through it), so a kitty account carries exactly one,
at signup, permanently associated with it. Weaker in frequency than the pre-KBR-161 pattern, not
weaker in kind — accepted on sign-in-regression cost alone. `tests/test_oauth_leg_identity.py`
guards the identity half and says explicitly that it asserts nothing about the fingerprint.

**Ambient `NO_PROXY` on the curl transport — closed, not accepted (KBR-161).** Moving the refresh
leg onto `curl_cffi` would have imported an egress exposure the aiohttp leg did not have: aiohttp
ignores proxy environment variables without `trust_env=True`, `curl_cffi` reads them, and — measured
against the resolved 0.16.3 — a `NO_PROXY` matching the upstream host **silently defeats an explicit
`proxies=` mapping**, with no error and no log line. That already applied to the API leg, so the
move would have put refresh tokens and exchanged API keys through a hole that previously leaked only
prompt content. `CURLOPT_NOPROXY`, set explicitly at session construction, wins over the
environment; `_new_curl_session` now sets it for **both** sessions, and
`tests/test_curl_cffi_transport_contract.py` pins the exposure, the remedy, and the complement.
The same hole remains open on the **botocore** (Bedrock) path, which is not this ticket's.

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
stands up a local TLS CONNECT proxy that enforces Basic auth and **records every `CONNECT` it
sees**, plus a local TLS target — both since T-W5 shared from
`tests/harness/connect_proxy.py` (§7.3) — and performs real TLS handshakes across all three
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

An L3 harness that closes gaps 1 and 2, built on `tests/harness/connect_proxy.py`'s
`ConnectProxy` and `TlsTarget` rather than writing new infrastructure — that proxy already
records CONNECT attempts, which is the observation the harness needs.

Those two classes were `_ConnectProxy` and `_TlsTarget`, private to
`tests/test_egress_https_proxy.py`, until **T-W5 ([KBR-28])** extracted them and registered their
fixtures as a pytest plugin; §7.3 records what that delivered. `tests/test_egress_https_proxy.py`
now imports them and keeps only `target_url`, the one seam specific to `kitty egress test`.

```
       ┌──────────── kitty BridgeServer ────────────┐
       │                                            │
       │   direct session ──X (must see nothing)    │
       │   proxy  session ──────────┐               │
       └────────────────────────────┼───────────────┘
                                    ▼
                     recording CONNECT proxy  ── tunnel log
                                    │
                                    ▼
                     recording fake upstream  ── connection + request log
```

#### 5.2.1 Correlate connections to tunnels, not requests to CONNECTs

An earlier draft asserted that the count of upstream requests must equal the count of `CONNECT`
attempts. That is wrong in both directions and would reject correct behaviour:

- **One tunnel can carry many requests.** A transport with connection reuse issues a single
  `CONNECT` and then sends N HTTP requests through it. The bridge's own aiohttp sessions use
  `force_close=True` so they happen to be 1:1 today, but curl_cffi and botocore need not be —
  and a test must not silently depend on `force_close`, which C5 and Q7 may well change.
- **One CONNECT can carry no requests.** A rejected proxy authentication (407) or a failed TLS
  negotiation produces a `CONNECT` attempt and no HTTP request at all.

The correct assertion is at the **connection** level: every TCP connection the upstream accepts
must be attributable to a successful tunnel through the proxy, with any number of requests
riding on it, and failed tunnels contributing no upstream connections.

**The join.** `ConnectAttempt` recorded target and authentication status only, which is not enough
to identify a connection at both ends. **T-W5 added `source_port`** — the proxy's outbound source
port for each tunnel it opens, and `None` where no tunnel was opened; the recording upstream
records the peer port of each accepted connection. Joining on that port identifies each upstream
connection with the tunnel that created it. An upstream connection with no matching tunnel port is
a bypass, and it is the only thing this assertion needs to catch.
`unattributable_peer_ports()` in the same module states the assertion once, so each transport
slice inherits it rather than re-deriving it.

(Peer *address* cannot do this job: with bridge, proxy and upstream on loopback in one process,
proxied and direct connections both present `127.0.0.1`.)

**The premise, which is a property of the wiring and not a law.** A source port identifies a
connection only because every leg terminates on the same destination `ip:port`, so the kernel will
not hand the same port to a second connection while the first is in `TIME_WAIT`. Point a future
leg at a *different* destination and a direct connection may legitimately draw a live tunnel's
source port and be attributed to it. The resulting error is a false negative — a breach
unreported — never a false positive, so it degrades safety rather than stability. Re-derive this
before adding a second upstream port. The join is also over **connections**, never requests: a
request-level log must be reduced to its distinct connections first, since one tunnel may carry
many requests.

#### 5.2.2 The three phases, per transport

The negative assertion is the one that matters, and on its own it is dangerously easy to satisfy
for the wrong reason — see §5.3. It must be bracketed by a positive control before it and a
falsification control after it. All three run for **every** transport in §5.5, not just the
bridge's own aiohttp session.

| Phase | Setup | Assertion | What it rules out |
|---|---|---|---|
| **1. Positive control** | Egress **disabled** | The upstream is reachable **directly**, and records the connection | That the destination is unreachable for some unrelated reason — name resolution, firewall, a mis-scripted fake. Without this, phase 2 proves nothing. |
| **2. Containment** | Egress enabled, proxy **stopped** | The upstream accepts **zero** connections, and the request fails | A direct fallback on proxy failure |
| **2b. Containment, healthy** | Egress enabled, proxy running | Every upstream connection joins to a tunnel (§5.2.1) | A partial bypass under normal operation |
| **3. Falsification control** | Egress enabled, proxy running, **a bypass deliberately introduced** (a patched `should_bypass` returning `True`, or a session built without the proxy) | The harness **fails** | That the harness is incapable of detecting a bypass at all |

Phase 3 is not optional decoration. A containment harness that has never been shown to fail is
indistinguishable from one that cannot fail, and §5.3 is a worked example of exactly that trap.

**Two further assertions**, unchanged in substance:

- **Local bypass still works.** A loopback or `localhost` provider (a local Ollama) connects
  directly and is not tunnelled — a rented proxy cannot reach the caller's LAN. **Bridge sessions
  only**; §5.5 explains why this is false for the custom transports.
- **Fail-closed.** A profile whose adapter returns `supports_egress() == False` (Bedrock in SSO
  mode) prevents startup, and the message names the profile.

### 5.3 The addressing trap — and why the harness needs a positive control

**Two traps live here, and the second is created by the fix for the first.**

**Trap 1 — a loopback destination is bypassed.** `egress.should_bypass()` returns `True` for
loopback, private and link-local destinations, so those connect directly. A harness that binds
its fake upstream on `127.0.0.1` is sent *direct*, proxies nothing, and passes vacuously while
proving the opposite of its claim. A Docker network does not help: `172.16.0.0/12` is private and
also bypassed.

The escape is in `should_bypass`'s own design. It reads `urlsplit(url).hostname` and:

1. returns `True` for `localhost` and any `.localhost` suffix — **checked first, by name**;
2. returns `True` for an IP literal in a loopback, private or link-local range;
3. for anything else, returns `False` **without resolving it** — deliberately, to avoid a DNS
   round trip per request.

So the harness addresses the fake upstream by a **hostname outside the `localhost` family**.
`upstream.kitty-test.invalid` is a reasonable choice: RFC 2606 guarantees `.invalid` never
resolves publicly, so the name cannot escape the test environment.

**Trap 2 — an unresolvable name satisfies the negative test for the wrong reason.** That same
guarantee is the problem. If the harness supplies name resolution only for the aiohttp leg, then
on curl_cffi or botocore a genuinely broken implementation could attempt a direct connection,
fail at DNS, and the upstream would still record **zero connections**. The negative assertion
passes; containment was never demonstrated. The test would be green on a product that leaks on
every other transport.

This is why §5.2.2 phase 1 exists and why it is mandatory per transport: **before** asserting
that nothing arrives, the harness must have shown that something *can* arrive directly for that
exact transport and destination. A negative result is only evidence when the positive is
possible.

**Name resolution, per leg.**

- *Proxied leg* — **no resolution needed by the client.** aiohttp's `_create_proxy_connection`
  resolves the *proxy* and issues `CONNECT upstream.kitty-test.invalid:443`; the test's own proxy
  resolves the target. The harness owns the resolver because the harness is the proxy.
- *Direct leg* — needs a local override, **for every transport, not only aiohttp**. aiohttp takes
  a monkeypatched resolver (`/etc/hosts` needs administrator rights and is unavailable on most CI
  runners, and `_build_client_session` builds its own `TCPConnector` with no injection point).
  curl_cffi and botocore need their own equivalents — curl's `resolve` mapping and a botocore
  `endpoint_url` override respectively. If a transport cannot be given a working direct route,
  its phase-1 control cannot pass, and its negative assertion must be reported as **unproven**
  rather than counted as a pass.

**The property that keeps the harness honest.** An L1 property test pins the premise so a future
change to `should_bypass` cannot silently make the harness vacuous:

> For every hostname that is neither an IP literal **nor `localhost` nor a `.localhost`
> suffix**, `should_bypass` returns `False`.

The `localhost` exclusions are not a caveat bolted on — `should_bypass` matches them explicitly,
before it ever tries `ipaddress.ip_address`, and a property stated without them fails on day one
and gets weakened, which removes the guard.

An IPv4-mapped literal such as `::ffff:10.0.0.5` **is** an IP literal, so it takes the address
branch and this property never reaches it. Its classification is a stdlib behaviour, pinned in
§6.2.4 — including the note that one of the three terms the disjunction reads is not independently
stable across interpreter patch releases.

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
| No `EgressConfig` representation leaks the password | L1 property | Structural: the password component of `masked()` is exactly the mask. **Not** a substring test — see §6.1 |
| Every outbound HTTP client is egress-aware; no source assigns `HTTP_PROXY`; no session trusts the environment | L2 structural | **Exists:** `tests/test_egress_coverage.py` |
| **Every** `BridgeServer` construction is dominated by an `egress_block_reason` call | L2 structural | **Must be strengthened** — the existing guard is file-granular (§5.1 gap 3) |
| The guard's rejection is **enforced** — no server starts on a rejecting configuration | **L3** | §6.2.3. Structural domination proves the call, not the branch that acts on it |
| An `https://` proxy carries real traffic on all three transport stacks | L2/L3 | **Exists:** `tests/test_egress_https_proxy.py` |
| Proxy semantics under each dependency's version range | L2 | §6.2.4 |
| The destination is reachable directly with egress **disabled**, on every transport | **L3** | §5.2.2 phase 1 — the positive control. Without it the row below proves nothing (§5.3) |
| Nothing reaches upstream except via the proxy, **from the bridge's own serving path** | **L3** | Sealed-network harness (§5.2.2 phase 2b) |
| The harness detects a deliberately injected bypass | **L3** | §5.2.2 phase 3 — the falsification control. A containment harness never shown to fail is indistinguishable from one that cannot |
| Stopping the proxy stops the traffic — no direct fallback | **L3** | §5.2.2 phase 2 |
| Containment holds for each custom transport | **L3** | §5.5 |
| kitty refuses to start when a backend cannot be proxied | L2 + L4 | Guard unit test + scenario EG-3 |
| A developer's whole session presents one IP | L4 | Scenario EG-1 |

### 5.5 Per-transport containment

`_session_for` and `should_bypass` govern **only** `BridgeServer`'s own aiohttp sessions. Four
other outbound paths exist, and each applies the proxy **unconditionally, without consulting
`should_bypass`**:

| Path | Client | How the proxy is applied |
|---|---|---|
| `openai_subscription` — serving | `curl_cffi.AsyncSession` | `proxies=` + `CURLOPT_NOPROXY` |
| `openai_subscription` — OAuth **refresh** leg | its own `curl_cffi.AsyncSession` | `proxies=` + `CURLOPT_NOPROXY` |
| `openai_subscription` — OAuth **login** leg (`kitty.auth.openai_oauth`) | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` |
| `bedrock` | boto3 / botocore | `BotoConfig(proxies=egress.proxies_dict())` |
| `ollama_cloud` | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` |

**Why the refresh leg has its own session rather than sharing the serving one (KBR-161).** Not for
identity — both come from one builder, `_new_curl_session`, so they cannot drift apart on
impersonation target, CA bundle or egress mapping. For two reasons that have nothing to do with what
the provider sees: an `AsyncSession` owns a bounded pool of curl handles (`max_clients`, 10) and a
streaming completion holds its handle for the life of the stream, so a refresh sharing that pool
would queue behind in-flight completions **while holding `OAuthSession._refresh_lock`**, blocking
every request for that account; and cookies are host-scoped, so one jar would hold
`auth.openai.com`'s cookies for the process lifetime and replay them across every account the bridge
serves. The two legs address different hosts, so sharing buys nothing observable and costs both.

Three consequences the rest of §5 must not paper over:

1. **The local-bypass assertion in §5.2.2 is false for these paths.** They have no bypass, so a
   loopback or private destination *is* tunnelled. Arguably safer, but different — the design
   must say so rather than imply uniformity.
2. **The sealed-network harness proves nothing about them** unless parametrised over the
   transport. It must run over `{bridge aiohttp session, provider aiohttp session, curl_cffi
   session, botocore client}` — the shape `tests/test_egress_https_proxy.py` already uses, which
   is a further reason it was that module's proxy T-W5 extracted rather than a fresh one.
3. **The OAuth leg runs at startup**, before anything else has been proven, and is the one most
   likely to fire on a fresh machine. It must not be left out. Since KBR-161 it is **two** paths on
   two stacks: the login leg on aiohttp, and the refresh leg on `curl_cffi` — and the `curl_cffi`
   one fires on every subsequent request, so that is the transport the harness must exercise first.

**An untested interaction.** `kitty.egress`'s own docstring records that the three stacks
disagree about `HTTP_PROXY`/`HTTPS_PROXY`: aiohttp ignores them unless `trust_env=True`, while
curl_cffi and botocore honour them. Kitty never sets those variables — but the *user's shell*
may have. **For `curl_cffi` this is now measured and closed** (KBR-161): a matching `NO_PROXY` beat
`proxies=` outright, and `_new_curl_session` now sets `CURLOPT_NOPROXY` so the mapping is the last
word; `tests/test_curl_cffi_transport_contract.py` pins both directions. **`botocore` is untouched and
unmeasured.** §6.2.4's row expects `Config(proxies=)` to take precedence over the environment — but
that is exactly the shape of expectation the `curl_cffi` row carried until KBR-161 measured it and
found the reverse. Until someone probes it, the Bedrock path's behaviour under an ambient
`NO_PROXY` is unknown, not known-good. **KBR-173.**

---

## 6. Layer specifications

### 6.1 L1 — Component and property

**Scope.** Pure logic: the three translators, the translation engine, compaction and tool-call
pairing, model normalisation, `should_bypass` and `parse_proxy_url`, profile schema and
resolver, the tool-use anomaly detector, every `ProviderAdapter` payload builder, and the wire
projections the oracle depends on (§3.3.1 — test code, but code the whole of I1 rests on).

**Tools.** `pytest` (present) + `hypothesis` (**to add**, dev extra only).

**Command.** `pytest -m l1 -q` — a marker, not `pytest tests/ -q`, which runs every layer and so
cannot be the L1 selection (§8).

**Validation.** Mutation testing (below).

**Property tests to add.**

| Unit | Property |
|---|---|
| `MessagesTranslator` | Semantic round-trip **through the projections, not through the translator pair**: `project_messages(inbound)` equals `project_cc(translate_request(inbound))`. See §3.3.1 for why the translator pair cannot serve as its own oracle. |
| `_compact_messages` | Identity below budget · no orphaned pair · idempotent · **output ≤ budget unless the surviving set is irreducible** (below) |
| `_validate_tool_call_pairing` | Output contains no `tool_result` without a `tool_use`, in both message shapes |
| `_truncate_oversized_tool_results` | Identity below the limit · output ≤ limit · non-tool-result content untouched |
| `should_bypass` | Every address in a private range is bypassed, **including the IPv4-mapped form of each range** (§6.2.4 pins why that is a stdlib claim and not a kitty one) · the §5.3 hostname property |
| `parse_proxy_url` / `EgressConfig` | Credential round-trip · the redaction property below |
| `describe_tool_input_anomaly` | Never reports an anomaly for input that validates against the declared schema |
| Wire projections (§3.3.1) | Each reads its format correctly, tested against published format examples — never against kitty's own output |

**The compaction budget property, stated honestly.** "Output ≤ budget" is false, and the
exception is wider than an earlier draft claimed. The guaranteed-fit loop drops head blocks, then
tail blocks, and `break`s while still over budget once it can shrink no further — when only the
system message and a single tail block remain. So it exceeds the budget whenever the **surviving
set is irreducible**, which includes a small system message plus one oversized final user turn,
not only an oversized system block. The property must be stated that way or it fails on the first
run and gets weakened by whoever is on the rota.

**This is a product question, not just a test-wording question.** What *should* happen when the
final turn alone will not fit? Today the request goes upstream over budget and is rejected there,
or — when nothing sendable survives — KBR-5's downstream 400 refuses it. Neither is obviously
right, and neither has been decided. The register,
the properties and the Gherkin must agree on one answer — see Q10. Until it is decided, the
document records current behaviour as *observed*, explicitly not as *approved*.

**The redaction property, stated so it cannot false-fail.**
`password not in repr(cfg) + str(cfg) + cfg.masked()` is wrong: `masked()` returns
`scheme://user:****@host`, so a password that happens to equal a substring of the host or
username fails the assertion despite correct masking. Password `proxy` with host `proxy.example`
reproduces it. Assert the **structure** instead — parse `masked()` and assert its password
component is exactly the mask — and test percent-encoded and URL-embedded forms as separate
cases. For the broader "does the password reach a log" sweep, generate a distinctive sentinel
that cannot collide with any other field.

**Where the budget itself comes from — and why it is not `normalize_model_name`.** TR-3's Given
is "a conversation that exceeds the model's context window". That window is a *resolved* value,
not a given: `_get_max_context_chars` asks `get_model_context_tokens(provider, model, config)`,
which searches two catalogs — the overrides catalog (`model_context_overrides.json`, or a newer
revision **synced from GitHub at runtime**) and the OpenRouter-derived `model_metadata.json`.
Until KBR-151 this document assumed the window was simply known. It was not: three of the four
query/key spellings missed, and a miss is silent.

- **Both catalogs are matched by one rule** (`_resolve_catalog`): exact, then query-as-suffix-of-key,
  then key-as-suffix-of-query, then one retry with the query's leading vendor segment removed.
  Before KBR-151 each catalog had one half of that rule, in opposite directions, so a profile
  written `azure/gpt-4o` resolved the 200,000-token default instead of 128,000 — the bridge then
  believed it had ~56% more room than the model has and sent an oversized request rather than
  compacting. That is an I1 breach produced by arithmetic, not by translation, which is why it
  went unnoticed by everything in §3.
- **Not via the adapter's `normalize_model_name`**, which the ticket originally proposed. That
  method translates into the **provider's** dialect; the catalogs are keyed in **OpenRouter's**.
  Measured across all 23 providers: it would fix 5,336 lookups and break 754 — 395 on `anthropic`
  (its `normalize_model_name` replaces dots with hyphens, and the catalog ids carry the dots) and
  359 on `vertex` (its version prepends `google/`). It would also stand a second, differently
  normalized model string beside `cc_request["model"]`, which KBR-127 made the single source of
  truth for every routing decision — see M14 and P20.
- **The invariant the matcher rests on: no two keys in a catalog end in the same segment.** Two
  such keys both match one query, so every lookup for that model goes ambiguous and falls to the
  default. Verified exhaustively, and the rule must be the **final** segment, not the part after
  the first separator: the looser reading leaves 640 ambiguous combinations, and the two coincide
  only while every key has at most two segments — true of both catalogs today, which is exactly
  how a guard written on the loose rule would look correct and stop protecting the moment a
  three-segment id appeared. The two catalogs are guarded differently **because they change through different
  doors** — `model_metadata.json` moves by a refresh script landing in a pull request, so a **test**
  suffices (L2, `test_model_context_packaged_catalog.py`); the overrides catalog can be replaced
  from the network with no release and no review, so a colliding revision is **rejected at load**
  in favour of the packaged file. Enforce where a file can change unreviewed; test where it cannot.
- **Ambiguity is terminal.** A catalog that cannot identify the entry declines and the next
  priority source answers; it does not retry a shorter string and return a number it has just
  called ambiguous.
- **A fall-through to `DEFAULT_CONTEXT_TOKENS` is logged**, at `INFO`, once per `(provider, model)`.
  The budget is recomputed on every request, so an unconditional line would be one per turn — and
  the silence is precisely what let KBR-151 live undetected. `INFO` not `WARNING`: an unknown model
  is routine and correct for a profile naming a local or private model.
- **Open, referred onward:** the overrides catalog outranks a profile's hand-written
  `provider_config["context_window"]`, and since KBR-151 one key captures every prefixed spelling
  of its model. Whether a *network-synced* catalog should outrank the one setting the operator
  typed is a question about operator authority, not about prefixed names, and is filed as
  [KBR-170](https://shelpuk.atlassian.net/browse/KBR-170) rather than decided inside a bug fix.

**Mutation testing.** Line coverage cannot tell a real assertion from `assert result is not
None`. `mutmut` closes that gap.

- **Tool:** `mutmut` (3.x; requires `fork`, so it runs on Linux CI — on Windows it needs WSL).
  Configured in `pyproject.toml` under `[tool.mutmut]`, where `source_paths` and
  `pytest_add_cli_args_test_selection` take **arrays**.
- **Test selection:** `pytest_add_cli_args_test_selection = ["-m", "l1"]`. Mutation testing
  measures the L1 suite; letting it run L3 subsystem tests would make each mutant minutes long
  and attribute kills to the wrong layer.
- **Scope — narrow, but it must include the code the rationale is about.** An earlier draft
  justified the subset by "a mutation surviving in the compactor means the suite would not notice
  kitty eating a tool result", then excluded `server.py`, where the compactor lives. Corrected
  scope, using `mutmut`'s function wildcards rather than whole modules:

  | Target | Why |
  |---|---|
  | `kitty.bridge.messages.*`, `kitty.bridge.responses.*`, `kitty.bridge.gemini.*`, `kitty.bridge.engine` | Translation — I1 |
  | `kitty.bridge.server._compact_messages*`, `_compact_with_tighter_budget*`, `_validate_tool_call_pairing*`, `_truncate_oversized_tool_results*`, `_apply_compaction*`, `_normalize_model*`, `_get_max_context_chars*` | Compaction and pairing — the I1 core, and the thing the rationale was always about |
  | `kitty.providers.*` `translate_to_upstream` / `normalize_request` / `build_upstream_headers` | The register's provider half — I1 and I2 |
  | `kitty.providers.model_context.*` | Where the compaction budget is actually resolved since KBR-151. The `_get_max_context_chars*` row above now covers a dispatcher: it reads `_active_model` and hands both catalogs to `_resolve_catalog`. A mutation in the matcher — dropping the tail retry, collapsing ambiguity into a miss — changes every budget in the product and would not be caught by any target listed above |
  | `kitty.providers.openai_subscription._cc_to_responses*`, `_prepare_responses_body*`, `_convert_content_types*`, `_build_user_agent*` | P13–P17 and the F1 user-agent. These are where the subscription path's real body is built; omitting them lets the score stay healthy while nothing detects a regression in the mutations this design only just registered |
  | `kitty.egress`, `kitty.egress_guard` | I3, including the startup guard |
  | `kitty.bridge.tool_audit`, `kitty.profiles.*`, `kitty.validation` | Supporting correctness |

  Still excluded: the TUI, the CLI wiring, the retry/health state machine. A mutation surviving
  in a menu is a cosmetic defect; one surviving in the compactor is a silent invariant breach.

- **P18 and P19 need a refactor before they can be mutation-tested.** Both live inside
  `make_request` / `stream_request`, which open network connections, so `mutmut` cannot reach them
  from an L1 selection. Extract the payload shaping into a pure builder — `_bedrock_body(...)`,
  `_ollama_body(...)` — leaving the network method to call it. The builder is then unit-testable
  and mutation-testable at L1, while the wire-capture test at L3 continues to prove the real bytes.
  This is the "design for testability" rule applied where the current factoring is what blocks the
  test, not the test that is hard to write.

- **Per-component thresholds, not one aggregate.** ≥ 85% killed **per target group** above. A
  single aggregate over a large surface lets a weak component hide behind a strong one — and the
  weak components here are exactly the invariant-critical ones.
- **Triage rule:** (a) a survivor revealing a missing assertion → strengthen the test; (b)
  revealing untested behaviour → add a test; (c) genuinely equivalent → suppress at the site with
  `# pragma: no mutate` **and a comment saying why**. Never dismiss a survivor silently.
- **Cadence — to be set by measurement, not assertion.** The full scoped run is nightly. Whether
  a **changed-code** mutation run also fits the per-PR gate is an open question with a numeric
  answer: measure the wall-clock of `mutmut run` restricted to functions touched by a
  representative PR, and adopt it per-PR if it lands inside the budget the fast gate can absorb.
  Rejecting per-PR mutation testing without that measurement is an assumption, not a decision.
  Tracked as Q11.

### 6.2 L2 — Contract

**Scope.** Any two artifacts that must agree but are deployed, edited or upgraded separately.

**Command.** `pytest -m l2 -q`

**Validation.** Every structural guard must assert that its own scan finds known positives, so it
cannot rot into a no-op. `tests/test_egress_coverage.py` already does this
(`test_the_scan_actually_finds_something`, `test_the_scan_finds_the_known_start_paths`) and is
the pattern to copy.

**And a contract pins what the code reads, never what it merely tolerates.** Pinning a value that is
itself version-dependent enforces whatever the author's interpreter happened to say; the fix is to
stop depending on it and pin the stable neighbour. Worked example and the precondition in §6.2.4.

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

**The register guard runs at the serialization boundary (§3.2.3), not at `translate_to_upstream`.**
An earlier draft specified feeding a request through `normalize_request` + `translate_to_upstream`
and diffing the result. That misses every transformation inside a custom transport — and P13 is
exactly that case: on `openai_subscription` `translate_to_upstream` is never called on the request
path, and `_cc_to_responses` builds the Responses body from `cc_request` directly, dropping
fourteen parameters. A guard checking the hook would have reported the adapter clean while
inspecting a body it never sends.

| Guard | Asserts |
|---|---|
| **Register completeness — shape diff at the wire** | For every adapter × representative model × transport, capture the body at the §3.2.3 boundary and assert the projected delta from the input is exactly the union of that adapter's register rows whose triggers the input met. **One fixed request is not sufficient** — each conditional row needs a trigger case and a complement case (§3.3.4), and adapters that route by model need one input per route. |
| **Internal-key completeness** | AST-scan `bridge/**` **and `providers/**`** for every `_`-prefixed key written into **any** dict — the scan cannot narrow to request bodies, and must not try; see below — and assert each is a member of `_INTERNAL_KEYS`. **This is the guard that catches F4 (KBR-6).** The complementary check — that each `translate_to_upstream` override delegates or excludes the set — is necessary but not sufficient: every override strips it correctly today; the set itself is what is wrong. Two scoping rules make the scan sound; both are stated below. |
| **Wire-shape honesty** | For every adapter × representative model, assert the adapter's declared wire shape agrees with the shape the body is actually written in. Catches F5 (KBR-7). **Two boundaries, two owners.** The *hook* form — observing `translate_to_upstream`'s return value — landed with **KBR-7** as `tests/test_wire_shape_honesty.py`, together with the fix: it asserts the per-model declaration `upstream_wire_is_messages_api_for_model(model)` against the emitted body, and the bare property against an explicitly declared default-route model. It guards itself so it cannot rot — the classifier is pinned against known Messages, Chat Completions and Converse bodies (including a Converse body with no tools, since the tools axis is what separates Converse from Messages); every registry key must be represented; every route of a model-routing adapter must be represented; and the custom-transport set is asserted rather than narrated. The *wire* form — observing the body at the §3.2.3 boundary — is **T-G4 / KBR-80** and is **not** delivered: for the three `use_custom_transport` adapters nothing in the hook-level guard observes the bytes that ship. On `openai_subscription` `translate_to_upstream` is never invoked on the request path at all — `_cc_to_responses` builds the Responses body inside the transport (P13–P17) — so there the exemption is load-bearing. On `bedrock` and `ollama_cloud` the transport mutates the hook's body afterwards (P18, P19); neither mutation changes the body's *shape family*, so for those two the exemption is precautionary — a guard must observe the shipped bytes, not infer them. The hook form also does not cover adapters constructed with `provider_config`, native-passthrough requests, whether a non-Messages body is *well-formed* (the declaration is a boolean, so Chat Completions and "neither" are collapsed) — **T-G4 inherits that one**, because it is a property of the declaration and not of the boundary — or whether the routing table matches the provider's published endpoint table (**KBR-126 — now closed**; see the Endpoint-table row below). The declaration stays boolean deliberately: its consumer is binary (`_repair_thinking_roundtrip` picks between exactly two carriers), so widening it to an enum would change the repair's contract rather than this declaration's. If a routing adapter ever gains a third wire, the boolean must be **replaced**, not extended — a `False` meaning "Responses" would be F5 again in a new costume. **KBR-126 is the near miss that clarifies that rule rather than breaking it.** `OpenCodeGoAdapter.get_upstream_path` now reports three paths, so the sentence appears to bite; it does not, because the declaration describes the body `translate_to_upstream` *emits*, and for the four `/v1/responses` models it emits none — it raises `UnsupportedModelError`. There is no third wire to declare, only a refusal, and the declaration's sole consumer takes the body as its first argument, so it is unreachable for those models. The replacement obligation transfers intact to **KBR-137**, which is the change that actually emits a Responses body. |
| **Bridge-introduced vendor token** | No content the bridge *introduces* into a request body or header contains `kitty` in any casing. Scoped by the projection diff (§3.3.3), never a flat scan of the serialized body — a flat scan would fail on a user legitimately writing the word, and "fixing" that would breach I1. Caught F3 (KBR-5). **F3 is now fixed, so this guard has no live positive fixture left**: its positive control is the synthetic historical M13 string held in `tests/bridge/test_vendor_token_guard.py`, which T-G5 inherits. That file is also the defect-scoped stand-in until T-G5 lands — it scans **source literals** against an allowlist, never traffic, so it does not fall into the flat-scan trap this row warns about. |
| **Start-path domination** | Every `BridgeServer(` construction is dominated by an `egress_block_reason(` call **at AST level**, not merely co-located in the same file. `cli/main.py` already holds two of the five start paths (§5.1 gap 3). **Necessary but not sufficient — see below.** |
| **Env-var register** | `_SETTINGS_ENV_OVERRIDE_KEYS` and `_CONFLICTING_ENV_VARS` (`launchers/claude.py`) match what `build_spawn_config` emits and what the README documents. |
| **Provider routing table ⇄ provider docs** | For every model the provider publishes, the adapter routes to the endpoint the provider serves it on. Landed with **KBR-126** as `tests/data/opencode_go_endpoints.json` (a snapshot of OpenCode Go's published endpoint table, carrying `source_url`, `verified_utc` and a note on what a keyed probe would add) plus `tests/test_opencode_endpoint_table.py`. The snapshot is an **oracle**, deliberately not the routing table itself: deriving `_MESSAGES_MODELS` from it at import would remove the duplication and add a worse failure mode, since a missing or corrupt data file would silently route everything to the default endpoint — the defect, reintroduced invisibly. The checker is a pure function (`check_routing`) so the negative cases can hand it a deliberate defect, and it compares **set equality in both directions**: a constant naming a model the provider has *stopped* serving on a route is exactly as wrong as one it never started routing, and that is the shape KBR-126 actually was. **Honest limit:** snapshot and constants are written in the same commit, so a green run proves self-consistency, not agreement with the provider; no stronger evidence is reachable without a paid key, because an unauthenticated probe of either endpoint returns `401 AuthError` (auth precedes dialect). Hence gap **G24**. |
| **`validation_model` reachability** | For every adapter that is not `use_custom_transport`, the path `validate_api_key` posts to and the bare `build_upstream_headers` agree on a dialect the key-check ping is written in. Landed with KBR-126 as `tests/test_validation_model_routing.py`. The ping body — `{model, messages, max_tokens, stream}` — is **simultaneously valid Chat Completions and valid Anthropic Messages**, which is why `anthropic`, `custom_anthropic`, `minimax_token` and `zai_coding` validate against `/v1/messages` and work. So the rule is *not* "`validation_model` must be Chat-Completions-routed": it is that path and auth must match. Pointing `opencode_go` at a Messages-routed model — the fix KBR-126's own ticket suggested — leaves the headers `Bearer` and fails every key check, and a path-only guard would pass it. |
| **Endpoint table** | The README endpoint table matches `_register_routes`. Catches F2 (KBR-9). |
| **Attribution-header table** | The README's `X-Kitty-*` table matches `_attribution_headers()`, and none of those names can reach any `build_upstream_headers()`. |
| **Flag table** | The README logging-flag table matches the CLI parser. |


**Scoping the internal-key scan (1): `providers/**` is in scope, not only `bridge/**`.** An earlier
draft scanned `bridge/**` alone, on the reasoning that the translators are where internal keys are
minted. They are not the only place. `ProviderAdapter.normalize_request` **mutates `cc_request` in
place** and is called on the live serving path — 34 call sites in `server.py`, six adapters
overriding it — so `providers/**` owns a live minting hook of its own. `providers/kimi.py:57`
writes `_thinking_enabled` from `build_request`, a second such hook on the public adapter
interface, dormant today in that `build_request` has no call site under `src/`. The invariant is
about what reaches the wire, not about which directory performed the write, so a `bridge/**`-only
scan leaves the `normalize_request` path unguarded. `providers/**` is green today, which is the
cheapest moment to adopt it — adding scope to a guard that is already red is a migration, adding
it now is a line.

**The scan is deliberately over-approximate.** No AST scan can tell a Chat-Completions body from
any other dict, so the scan's real population is *every* `_`-prefixed key written into *any* dict
in the scanned files. It must not filter by the target variable's name: name-based **inclusion**
(considering only targets called `cc_request` / `body` / `result`) is the same defect as the
name-based **exclusion** rejected below, with the sign flipped, and it would miss a leak written
into `payload["_x"] = 1`. When the next non-body underscore write appears — a cache key, a stats
dict — the escape hatch is a named exclusion with a stated reason, **never** an `_INTERNAL_KEYS`
entry. `server.py:1187` (`self.__dict__["_provider_config"] = value`) is already such a write, and
passes only because `_provider_config` happens to be a set member for unrelated reasons; it is a
warning of the pressure, not a precedent.

**Scoping the internal-key scan (2): an aiohttp `web.Request` is not a request body.** The naive
form of this scan — every subscript assignment whose key is a `_`-prefixed string constant —
reports three false positives, all in `BridgeServer._auth_middleware`: `_key_id`, `_profile_name`
and `_mapped_profile`. Those are written onto the **inbound** `aiohttp.web.Request` object, which
aiohttp supports as a request-scoped mapping, and are read back by the access logger. They are
never serialized to any provider.

The scan must therefore exclude subscript targets that resolve to a parameter annotated
`web.Request`, and must key that exclusion on the **annotation**, not on the variable being named
`request` — the codebase uses `request` for both kinds of object, and a name-based rule would
excuse a genuine leak written into a variable that happened to be called `request`.

The tempting alternative — adding those three names to `_INTERNAL_KEYS` to make the scan pass —
is wrong and must not be taken. `_INTERNAL_KEYS` is applied to a Chat-Completions body and is
documented as "keys that must never be sent upstream"; these three never enter such a body at all.
Adding them would make the set describe something it does not govern, and would then silently
excuse a real leak if one of those names were ever written into an actual request body.

**The exclusion is annotation-*seeded* but name-*applied*, so it must retire when the name is
rebound.** An annotation binds a name once; every later use of that name is matched textually.
`request = await request.json()` leaves `request` holding a parsed request *body*, and a naive
implementation goes on excluding writes to it — at which point the rule has silently become the
name-based one this section forbids, reachable in four moves (annotate, reassign, write, ship).
The same applies to a nested `def inner(request)` that re-declares the name unannotated: it must
*not* inherit the enclosing scope's exclusion, even though closures otherwise should. The scan
therefore refuses to trust an annotation for any name the scope **rebinds anywhere in its body**.

**Decided per scope, not per statement — and that is the design decision, not an implementation
detail.** The first version enumerated binding *statements* and was corrected four times in review:
annotated assignment, then `async for` and `async with`, then augmented assignment,
`except`/`import` aliases and tuple targets, then `match` captures and `class`. That list cannot be
completed, because Python keeps adding to it and each addition silently re-opens the hole. Asking
instead "does this scope rebind the name at all" terminates: a plain name is bound by a `Name` in a
`Store`/`Del` context, and the handful of forms that carry the name as a bare string —
`except`, `import`, `match`, `def`, `class` — is closed and short. Comprehension targets, which the
statement-by-statement version had explicitly given up on, fall out for free.

The rule is deliberately coarse and not flow-sensitive: a write *before* the rebinding is reported
too. Over-reporting costs a review comment; under-reporting hides a leak. No handler in `server.py`
rebinds `request` today, so the coarseness costs nothing now, and the guard's real job is to hold
the day one does.

**The counterweight matters more than the rule.** A `Subscript` or `Attribute` target is **not** a
rebinding. `request["_key_id"] = ...` writes *through* the name, and the `Name` node inside it
carries a `Load` context — so it is excluded from the bound set by the same mechanism rather than by
a special case. Had it counted, the exclusion would retire on the very statement it exists to
suppress and the middleware's three request-scoped writes would invert into reported leaks. Widening
the rebinding rule is safe; widening it carelessly is not, so both cases carry their own tests.

Because an exclusion that stops matching is indistinguishable from a guard that has quietly gone
blind, the exclusion carries its own assertion: **the scan must fail if the `web.Request`
exclusion matches nothing.**

The premise — that an aiohttp `web.Request` is a `MutableMapping` supporting `__setitem__` — was
verified against **aiohttp 3.13.5**, the version in this project's environment. (The
`AppKey`/`NotAppKeyWarning` deprecation applies to `Application`, not `Request`.) The version is
named so that an aiohttp bump that changes this is a visible decision rather than a silent one.

**The complementary delegation check, and why it is not implemented structurally.** The row above
calls the "each `translate_to_upstream` override delegates or excludes the set" check *necessary*.
It is discharged behaviourally instead, by the registry-parametrised regression test (KBR-6): that
test is parametrised over `providers.registry._registry` × wire route, so a newly added adapter is
covered the day it is registered, whereas a structural delegation check must be taught about each
new override. The behavioural form is the stronger of the two. This is recorded rather than left
implicit because the structural check would otherwise be silently orphaned.

**Calling the guard is not enforcing it.** `egress_block_reason()` *returns a reason*; it stops
nothing. Enforcement is the branch that follows — `if egress_error: print(...); return 1`. Delete
that branch and every structural check above still passes while an unproxyable profile starts
normally and leaks the machine's own address, which is the entire failure mode I3 exists to
prevent.

The structural guard is therefore paired with a behavioural one at L3:

- **Per entry point.** Each of the five start paths (`bridge_runner.py` ×2, `cli/launcher.py`,
  `cli/main.py` ×2) is driven with a configuration the guard rejects, and the assertion is that
  **no server starts** — no listening socket, non-zero exit — not merely that a message was
  printed.
- **Falsification control.** A variant that keeps the `egress_block_reason()` call and discards
  its return value must make these tests **fail**. Without it, the suite cannot distinguish
  enforcement from decoration.

This complements the guard's own unit test (which checks the returned reason) and acceptance
scenario EG-3 (which checks the user-facing behaviour). The unit test proves the guard decides
correctly; this proves the decision is obeyed.

Every guard asserts its own scan finds known positives, so none can rot into a no-op —
`tests/test_egress_coverage.py` already does this and is the pattern to copy.

**Why guard the README specifically.** For a CLI tool the README *is* the interface
specification — it is what a user configures against. A drifted README is a defect with the same
user impact as a drifted API, and F2 shows it has already happened.

#### 6.2.4 Dependency behaviour contracts

Small, fast tests pinning dependency behaviour the invariants rest on — and the
ordinary-correctness behaviour whose drift the gate cannot see — so an upgrade fails here with
a clear message rather than in production. The pin situation is worse than a glance suggests:

| Dependency | Declared pin | What must be pinned by test |
|---|---|---|
| `aiohttp` | `>=3.11,<3.14` | A session built with `proxy=`/`proxy_auth=` proxies, and a per-request `proxy=None` cannot escape it. `_build_client_session` sets the proxy at session level precisely so no call site can forget it, and `_session_for` depends on a request being unable to opt out. |
| `curl_cffi` | `>=0.7` — **unbounded** | **Landed (KBR-161): `tests/test_curl_cffi_transport_contract.py`.** `proxies=` is honoured; `data=dict` form-encodes (the token grants depend on it); an explicit `User-Agent` beats the one `impersonate=` injects (the whole of KBR-161's fix depends on it); the impersonation target still exists. On the environment, the measured answer is the *opposite* of what this row assumed: a matching ambient `NO_PROXY` **defeats** `proxies=`, and only `CURLOPT_NOPROXY` overrides it — both directions pinned, since a future release reversing either must turn red rather than silently change containment. |
| `botocore` | **not declared at all** — arrives transitively via `boto3>=1.34` | `Config(proxies=)` is honoured and takes precedence over the environment. It is botocore, not boto3, that implements this. An undeclared dependency owning a containment guarantee is worse than an unbounded one. |
| `keyring` | `>=23.0` | Backend resolution on each supported platform. |
| CPython `ipaddress` | `requires-python = ">=3.10"` — **minor only, no patch floor** | Two consumers. **I3:** `should_bypass` reads `is_loopback or is_private or is_link_local`, so the disjunction's verdict on the IPv4-mapped form of each range must be pinned — see the masking note below. **Liveness:** `_connect_target` reads `IPv6Address.ipv4_mapped` (the mapped `IPv4Address` for `::ffff:x.x.x.x`, `None` otherwise) and `is_unspecified` for `0.0.0.0` and `::`. What must **not** be pinned is `IPv6Address("::ffff:0.0.0.0").is_unspecified`: CPython [gh-122792](https://github.com/python/cpython/issues/122792) changed it mid-branch, so its value is a property of the patch release, and the code is written not to read it. |

The ambient-environment cases are not hypothetical: `kitty.egress`'s docstring records the
divergence, and a user with `HTTP_PROXY` set in their shell exercises it on two of three stacks.

**The standard library is a dependency, and it is declared to the wrong precision.** Three of the
other rows name a version range; `botocore` names none at all. The interpreter is a third shape —
declared, but only to the *minor*, so `requires-python` admits both sides of a behaviour change that
moved at a patch boundary. That is how KBR-146 reached `main` green: `ipaddress` answered one way on
the runner and the other way on a stock Ubuntu 24.04 developer box, and nothing in the tree asserted
which answer was being relied upon.

**A contract pins what the code reads, never what it merely tolerates.** The pre-fix
`_connect_target` would have been pinned by asserting `is_unspecified` is true for the mapped
wildcard — red on the 35 supported releases that predate the backport (3.10.0–3.10.15,
3.11.0–3.11.10, 3.12.0–3.12.6, 3.13.0) — so the contract would have enforced the defect rather than
caught it. The rule this row establishes: when a dependency's behaviour is version-dependent, stop
depending on it and pin the **stable neighbour** instead; pinning the moving value only relocates
the failure. **Where no stable neighbour exists** — `keyring`'s backend resolution varies by
platform by design, and `curl_cffi`'s impersonation targets come and go — the remaining options are
a version floor or a runtime feature check, and the row must record which was chosen. A floor was
rejected here for the reason a floor is usually wrong: it drops supported users to settle a question
the code no longer asks. `tests/test_ipaddress_contract.py` holds the contract and says in its own
docstring which property it refuses to assert and why.

**Why forcing the property is a faithful stand-in for an old interpreter.** `ipv4_mapped` itself was
measured identical on 18 releases spanning all four supported branches, which is what makes the L1
test's forced `is_unspecified` a reproduction of a pre-backport interpreter rather than a resemblance
to one. Everything else in the mapped path reads the same on both sides of gh-122792.

**`should_bypass` survives gh-122792 by masking, not by independence — and that is a premise, not an
accident.** `ipaddress.ip_address("::ffff:169.254.1.1").is_link_local` flips at the *same* four
boundaries as `is_unspecified` (measured: `False` on 3.10.13–15, 3.11.8–10, 3.12.4–6 and 3.13.0;
`True` from 3.10.16, 3.11.11, 3.12.7 and 3.13.1). The disjunction is stable only because
`is_private` delegates to the mapped address on every supported release and IPv4's `is_private`
already covers `169.254.0.0/16`. So I3's bypass decision *does* read a version-dependent value, and
is saved by a sibling term. Anyone who later splits that disjunction, narrows it, or logs per term
re-opens the patch dependence in the containment direction — which is why the contract pins the
disjunction's verdict on the mapped forms rather than the individual terms. Upstream's own
motivation for gh-122792 was "folks using IP address filtering before establishing a connection",
which is precisely what `should_bypass` is.

**What this contract does not prove, and what carries it instead.** `.github/workflows/tests.yml`
names bare minor versions and `actions/setup-python` resolves each to the newest patch, so this
module is only ever evaluated on the **new** side of every such boundary. It can therefore catch
*forward* drift — a future interpreter changing a value we read — and cannot catch a value that
differs on an older patch a user is actually running, which is the shape KBR-146 had. That half
rests entirely on the L1 forced-property test, which holds both sides inside one run. Recorded as
gap **G25** rather than closed by a matrix entry: `setup-python` does accept an exact patch version,
so one pinned job would do it, and that is a CI-spend decision for the product owner rather than a
change this defect's fix should make on its own authority.

### 6.3 L3 — Subsystem

**Scope.** One subsystem plus its real infrastructure. Two here: the bridge with real sockets,
and the CLI with the real filesystem and real child processes.

**Tools.** `pytest` + real `aiohttp` servers + the CONNECT proxy from
`tests/test_egress_https_proxy.py`. `testcontainers` only if a scenario genuinely needs process
isolation; a local proxy and upstream do not.

**Command.** `pytest -m l3 -q`

**Validation.** Each harness must carry a **negative** case proving it can fail — the sealed
network's proxy-down assertion, the oracle's "mutation present without its trigger" case. A
subsystem harness with no negative is this layer's characteristic failure mode; §3.3.1 and §5.3
each show how easily one goes vacuous.

#### 6.3.1 Bridge with real sockets

**Command.** `pytest -m l3 -q`

| Scenario | Assertion |
|---|---|
| Transparency oracle over the corpus (§3.3), per adapter × model × transport | No unclaimed delta; every conditional row exercised with trigger **and** complement |
| Bridge-introduced vendor token (§4.3 C2) | An inbound `kitty` string survives byte-identically; an injected vendor message is caught |
| Sealed network (§5.2), all three phases, per transport (§5.5) | Positive control passes; zero connections with the proxy down; the falsification control fails the harness |
| Cross-attempt content (§4.3 C3) | Transport-blip and empty-response retries byte-identical; each of M6, M8, M9 and failover re-normalisation fires only under its own trigger |
| Connection lifecycle (§4.3 C5) | Distinct-connection count per session, against the native baseline |
| **Streaming recovery — content, not just grammar** (below) | Four injection points; no duplication, no replayed tool calls, no spliced arguments |
| Client disconnect during a stream | Upstream connection released; the backend not marked unhealthy for a client-side fault |
| All backends unhealthy | The 503 arrives in each protocol's native error envelope |
| Oversized request | Rejected with the protocol's own error shape, not a raw 413 |
| `_backend_context` isolation | Concurrent requests never observe each other's backend selection. **Deterministic, so it belongs here, not in the load profile.** |

**Streaming recovery needs content guarantees, not only a well-formed stream.** "The client sees
one well-formed stream" is necessary and nowhere near sufficient: a failover that replays text
the client already received, re-emits a `tool_use` block under a fresh id, or splices tool-call
arguments assembled from two different attempts produces a stream in which **every SSE event is
syntactically valid** and the conversation is corrupt. Claude Code will act on a duplicated tool
call.

Inject the failure at four points, forced deterministically with barriers or a scripted event
sequence rather than timing:

| Injection point | Assertion |
|---|---|
| Before any downstream byte | Clean failover; the client sees one complete stream from the second backend |
| After text has been emitted | No text the client already received is repeated; the transcript reads as one message |
| Mid `input_json_delta`, tool arguments partly sent | Arguments are never a splice of two attempts. **The acceptance oracle here is undecided — Q14.** Until it is answered this row asserts only the negative (no silent merge, no reused id across attempts), which is weaker than the row needs to be |
| After content, before the terminal event | Exactly one terminal outcome reaches the client; `message_stop` is not duplicated or omitted |

Each case asserts tool-call **identity** (ids stable within an attempt, never reused across
attempts) and a single terminal outcome.

**The post-emission semantics are a prerequisite, and they are not decided.** Once bytes have
reached the client, what a correct recovery even *looks like* is a product decision, not a test
detail: abandon and re-open, fail the turn, or something else. Writing "whichever the agreed
semantics say" into a test specification leaves it without an acceptance oracle — the same defect
this document objects to elsewhere. It is tracked as **Q14** rather than left as prose, so the
gap is visible in the question list where decisions are collected, not buried in a table.

`/stats` remains authoritative for attribution after a mid-stream failover, per the README's own
caveat that the headers name whoever produced the first byte.

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
currently a dependency); `pytest` for the agent tests; separate runners for evals and load.

**Commands.** `pytest -m acceptance -q` · `pytest -m agent_smoke -q` · `pytest -m agent_live -q`
· the eval and load runners are separate entry points (§8).

**Validation.** Every scenario binds to an L3 harness rather than re-implementing one — an
acceptance test that grows its own assertions has drifted into L3 and should be moved down.

#### 6.4.1 Gherkin acceptance

The invariants written as scenarios a product owner can read and sign.

```gherkin
Feature: The upstream provider cannot tell Kitty Bridge is there

  Scenario: TR-1  Kitty introduces no fingerprint of itself
    Given a profile using the Z.AI coding plan
    When Claude Code sends a turn through kitty
    Then no content the bridge added to the request names kitty

  # One exempt assertion: header-subset, pending G3's policy half (Q1). See the exemption registry in 8.
  Scenario: TR-1c  Kitty's headers are a subset of the agent's own
    Given a profile using the Z.AI coding plan
    When Claude Code sends a turn through kitty
    Then the header set is a subset of what Claude Code sends natively

  Scenario: TR-1b  The agent may talk about kitty freely
    Given a profile using the Z.AI coding plan
    And a prompt whose text is "Please explain how kitty-bridge works"
    When Claude Code sends that turn through kitty
    Then the provider receives that text unchanged

  Scenario: TR-2  A turn needing no adaptation reaches the provider unchanged
    Given a native-wire provider and no thinking signal in the request
    And a conversation inside the model's context window
    And no tool result larger than the truncation limit
    When Claude Code sends a turn through kitty
    Then the provider receives the agent's messages with no content altered
    And the only difference from the agent's own request is the model name
    # Preconditions matter: on a translated wire M2 rewrites everything, and a
    # thinking signal triggers P2a, P5c-e and P8 content injection. Cf. 3.4.

  Scenario: TR-3  A long turn is altered only as far as necessary
    Given a conversation that exceeds the model's context window
    When Claude Code sends a turn through kitty
    Then history is compacted only as far as the budget requires
    And no tool result is separated from the call that produced it
    And the most recent turn is preserved, unless its own tool result exceeds
         the 50,000-char truncation limit, or it is dropped because its
         tool_result lost the tool_use that produced it

  # No exemption. KBR-5 is fixed, so every assertion gates normally.
  Scenario: TR-4  An unrecoverable conversation fails without naming kitty upstream
    Given a conversation whose only remaining turn is a tool result with no matching tool call
    When Claude Code sends that turn through kitty
    Then the user is told the conversation cannot be compacted
    And the provider receives nothing at all
    # The Given was "a system prompt larger than the model's context window" until
    # KBR-5 measured it: that shape does not empty the conversation, so the
    # scenario could never have reached the behaviour it names. See F3.

Feature: Configured egress cannot be bypassed

  Scenario: EG-0  The destination is reachable directly when egress is off
    Given no egress gateway configured
    When kitty sends a request on each supported transport
    Then the provider records the connection
    # Control. Without it, EG-2 can pass because nothing could ever arrive.

  Scenario: EG-1  Every request arrives through the gateway
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

**One scenario carries an assertion-level exemption.** The Acceptance job gates every PR (§8), so
an assertion about behaviour the product does not yet have would make `main` red on day one — and
a permanently red gate gets disabled, taking the working scenarios with it.

**TR-4's exemption is withdrawn (2026-09-07):** KBR-5 is fixed, so its no-vendor-content assertion
gates normally. **TR-1c's remains, and KBR-8 closing did not lift it (2026-09-11).** KBR-8 fixed the
*self-contradiction* — two client versions in one request — not header *parity*: twenty of the
twenty-three adapters still send no `User-Agent` at all, so kitty's set is not yet a subset of the
agent's. The exemption is therefore re-keyed from KBR-8 to **G3's policy half (Q1)**, with the C1b
baseline it depends on owned by T-C7 and T-I12. Left pointing at KBR-8 it would read as a row whose
defect is closed, which the unexpected-pass rule cannot correct because the assertion still fails
for a different reason.

The exemption covers **one assertion**, never the scenario. In TR-1c it is the header-subset
assertion (KBR-8). Every other step in
those scenarios — setup, the `Given` clauses, and any other `Then` — gates normally, so a broken
fixture or an unrelated regression still fails the build. An unexpected pass also fails, forcing
the exemption off when the defect closes. The full policy is in §8 and the registry in §8.3;
this section does not restate it, so the two cannot drift apart.

**Three scenarios were corrected against the implementation, not the other way round.**

- **TR-1 / TR-1b** — the original TR-1 said the request "contains no header, field or value
  naming kitty", which forbids a user from asking about the product. TR-1 now constrains only
  content the bridge introduced; TR-1b pins the complement, so the pair cannot be satisfied by
  stripping user text.
- **TR-2** — "a short turn reaches the provider unchanged" was false: M3 tool-result truncation
  is unconditional pre-processing and fires on a short conversation containing one 50,000-char
  tool result. The precondition is now explicit.
- **TR-3** — "the most recent turn is preserved in full" was false: the last turn is subject to
  truncation and to pairing validation. It now states what the code does.

**TR-3 and the compaction properties still depend on an undecided product question** (Q10): what
*should* happen when the final turn alone exceeds the budget. This Gherkin records current
behaviour; it does not ratify it. When Q10 is answered, TR-3 and register rows M3–M7 and the
§6.1 properties change together or not at all.

#### 6.4.2 Agent-boundary tests

Two distinct things, currently conflated, with different costs and different cadences.

**Agent smoke — per PR, hermetic.** Two distinct cases, and only the second proves the claim.

*Startup smoke.* A **pinned real Claude Code binary** runs one non-interactive turn against the
scripted recorder (§7.2) — no live provider, no credentials, no network. The binary starts,
resolves the bridge URL, its request arrives, it exits cleanly. This proves connectivity.

*Precedence.* Connectivity is **not** the claim. The claim is that kitty's `--settings` file wins
over the process environment and over `~/.claude/settings.json` — a fact about *Claude Code's*
behaviour, not kitty's, and the assumption under suspicion in the KBR-1 investigation. A run in
which nothing competes cannot distinguish correct precedence from accidental agreement: an
implementation with the order backwards passes it.

So the test creates a genuine conflict and points every loser at a **sentinel recorder** of its
own:

| Source | Points at | Expected |
|---|---|---|
| `ANTHROPIC_BASE_URL` in the child's environment | sentinel A | receives **nothing** |
| `env.ANTHROPIC_BASE_URL` in `~/.claude/settings.json` (a temp `HOME`) | sentinel B | receives **nothing** |
| kitty's per-session `--settings` file | the real recorder | receives **the request** |

Assert on all three: the request arrives at the recorder **and** both sentinels stay silent. A
test that checks only the recorder cannot tell "precedence is correct" from "the request went
everywhere."

*The controls.* A sentinel that is never hit proves nothing unless it can be hit — and one
control exercises only one sentinel. Two more runs are needed, so that **each destination wins
under its own configuration**:

| Run | Session settings | `~/.claude/settings.json` | Environment | Expected winner |
|---|---|---|---|---|
| Main | present | present (B) | present (A) | the recorder |
| Control 1 | absent | present (B) | present (A) | **B** |
| Control 2 | absent | absent | present (A) | **A** |

Control 1 alone leaves A unexercised, so a typo in A's URL would read as a pass in the main run —
A is *supposed* to be silent there, and a broken A is silent too. Three runs, three winners, and
every sentinel demonstrated live.

Pinning it in CI has real questions attached — which distribution, how tightly version-pinned, licensing for
redistribution in a CI image — recorded as Q12 rather than assumed away.

**Agent live — nightly.** `tests/integration/test_agent_e2e.py` as it exists: real binaries, real
credentials, real providers. Keep it out of the default run — it needs four agent CLIs and live
network, and a suite that cannot run on a laptop stops being trusted. Nightly with CI secrets,
extended from two cases to cover, for Claude Code: a plain turn, a tool-using turn, a multi-turn
session with tool results, an extended-thinking turn, and a session crossing the compaction
threshold.

**Neither job may skip silently.** A required job whose prerequisite is missing must **fail**,
not pass-by-skipping. A green tick that means "we did not test this" is worse than a red one.

#### 6.4.3 Answer-quality evals

Compaction (M5, M6) and truncation (M3, M4, P7, P13) can degrade an answer without breaking
any structural assertion. Nothing below L4 can detect that. But the method has to be honest about
what it can establish.

**What pairing does and does not buy.** Running the same task through kitty and directly against
the same provider removes *task* variance — task difficulty is held constant. It does **not**
cancel the model's independent sampling on each call. Two samples of the same prompt differ. So
the design needs repetition and an interval, not a single paired run and a single number.

| Element | Specification |
|---|---|
| **Tasks** | A fixed set with **independently authored** acceptance tests. A model-generated test that the model's own code passes establishes nothing — the same misunderstanding can be present in both. |
| **Repetition** | N samples per task per arm, N fixed in advance and large enough for the interval below to be narrower than the margin. |
| **Pinning** | Model id, provider, dataset revision, temperature and all sampling settings pinned and recorded with each run. An unpinned model makes the series meaningless. |
| **Statistic** | Difference in pass rate between arms, with a confidence interval — not a point estimate. |
| **Decision rule** | A **pre-registered regression margin**: the alert fires when the interval excludes a delta smaller than the margin. Chosen before the data, not after. |
| **Primary outcome** | **Successes ÷ scheduled trials.** Not successes ÷ completed trials. A bridge arm that refuses 90 of 100 tasks and answers the other 10 correctly scores 10%, which is the truth; excluding refusals would score it 100% and report a catastrophic regression as a clean run. |
| **Failure taxonomy** | Every non-success is classified and reported separately — refusal, upstream error, timeout, rate limit, harness fault — per arm. The categories are the diagnosis; they are not deductions from the denominator. A refusal is **evidence about the arm**: changed instructions or lost context cause refusals, and both are exactly what compaction can do. |
| **Exclusions** | Only for pre-defined infrastructure incidents, applied **symmetrically to both arms** by a rule written before the run. Every exclusion is reported with its reason. |
| **Missing-data ceiling** | If exclusions plus harness faults exceed a fixed fraction of scheduled trials, the run is **void**, not adjusted. Below that ceiling the comparison stands; above it there is no comparison to make. |

**The compaction arm needs a different baseline.** For an input that exceeds the model's context
window, the direct-provider arm does not produce a worse answer — it produces a 400. There is
nothing to compare. The baseline for compaction cases must be something that can actually answer:
kitty against a larger-context model, or kitty with compaction thresholds relaxed. Which one is
Q13.

Q4 asks for the margin. It is one input to this design, not a substitute for it.

#### 6.4.4 Load

The earlier draft said "many parallel sessions", "sustained", and "behave as intended". None of
that is a gate — it is the "as appropriate" this document forbids elsewhere. Specified properly:

| Element | Specification |
|---|---|
| **Workload** | C concurrent sessions × R requests, Poisson arrivals at rate λ, duration D, response sizes drawn from the corpus. C, R, λ and D fixed in the profile and versioned with it. |
| **Hardware** | Named runner class and core/memory count recorded with every result. A latency number without a machine is not a number. |
| **Streaming vs. buffered** | Measured separately. The bridge streams incrementally on the SSE paths but **buffers whole responses** on others (non-streaming `_make_upstream_request`, the subscription provider's SSE-to-response parse). "Memory does not grow across a long stream" is false as a blanket claim and must be scoped to the incremental paths. |

**Metrics and gates:**

- **Bridge-added latency** — p50 and p95 of (through-kitty − direct-to-fake-upstream), against a
  fixed ceiling.
- **Time to first output byte** on streaming paths, against a fixed ceiling.
- **Completion rate** and **error rate** by class, against fixed floors.
- **Bounded memory** — peak RSS stays within a fixed multiple of the largest single buffered
  response on buffered paths; flat within a fixed band on incremental paths.
- **Resource recovery** — open sockets and file descriptors return to baseline within a fixed
  window after the run, proving `force_close` and the connection-limit behave under saturation.

The ceilings and floors are numbers this document does not invent: they come from a first
baseline run, recorded and then ratcheted — again the monotonic kind, not the §8.3 exemption.
**`_backend_context` isolation moved to L3** (§6.3.1)
— it is deterministic and does not need a load rig to prove.

## 7. Shared test infrastructure

### 7.1 Golden Claude Code transcript corpus

**What.** Real `POST /v1/messages` bodies captured from actual Claude Code sessions, committed as
fixtures, with the native upstream headers and connection pattern Claude Code produces captured
alongside them (the baselines C1b and C5 compare against).

**Why real rather than synthetic.** Hand-written fixtures encode our belief about what Claude
Code sends. That belief is the thing most likely to be wrong, and it drifts every time Anthropic
ships a release. Captured bodies are evidence.

**Coverage is driven by the register, not by intuition.** Every conditional row in §3.2 needs a
corpus entry that meets its trigger **and** one that does not (§3.3.4); a corpus that only
contains trigger cases cannot support assertion 2. At minimum:

| Entry | Exercises |
|---|---|
| Plain text turn | Baseline; complement for every conditional row |
| Turn with `tools` declared | Tool-declaration projection |
| Assistant `tool_use` / user `tool_result` | Pairing, M7 |
| Extended thinking | P2a/P2b, P5c, P5d, P8, M8 |
| Image content block · `system` with `cache_control` | Projection fidelity on non-text parts |
| Tool result just under and just over 50,000 chars | M3/M4 trigger **and** complement |
| Transcript just under and just over the compaction budget | M5 trigger and complement |
| Transcript provoking the upstream 400/413 recovery | M6 — **must run against a balancing profile**; `_request_with_retry` has no compaction recovery |
| System prompt alone larger than the window | The Q10 behaviour. **Not** the compaction-failure path: KBR-5 measured this shape and it leaves the user turn intact |
| Only remaining turn is a tool result with no matching tool call | The compaction-failure path — the real trigger for what M13 used to do (F3) |
| A single final turn larger than the budget | The irreducible-set case (§6.1) |
| **A turn whose text is `Please explain how kitty-bridge works`** | The §3.3.3 regression case — must survive byte-identically while a bridge-introduced vendor string is still caught in the same run |
| `max_tokens` above and below 4096, streaming and non-streaming | P7 trigger and complement; P13 |
| Malformed body | The L2 fuzz path |

**Traps.** Captured transcripts contain prompts, file contents and API keys. The capture
procedure must scrub credentials and the corpus must be reviewed before commit; a fixture file is
as public as the repository. The corpus needs a refresh cadence tied to Claude Code releases and
a recorded capture procedure — an un-refreshable corpus becomes a museum of a protocol nobody
speaks any more.

### 7.2 Recording upstreams — one per transport

The bridge reaches upstream through **five** distinct client configurations (§3.2.3, §5.5), and
an aiohttp recorder observes only one of them. One recorder per configuration, all presenting the
same interface to the tests:

| Recorder | Serves | Observes |
|---|---|---|
| aiohttp server — bridge sessions | the 20 default-transport adapters | The primary; speaks Anthropic Messages and Chat Completions |
| aiohttp server — provider sessions | `ollama_cloud`, and the `openai_subscription` **OAuth token legs** | Those adapters build their own sessions and never touch `_session_for`, so the bridge recorder never sees them. The OAuth leg runs at startup, before anything else has been proven (§5.5) |
| curl_cffi-reachable server | `openai_subscription` serving path | Must terminate TLS with the harness certificate; the only place `_cc_to_responses` output (P13, P17) can be seen |
| botocore endpoint override | `bedrock` | Points the client at the local recorder rather than AWS; observes the Converse payload **after** the transport's `modelId`/`stream` pops (P18) |

Each records every request in full: method, scheme, host, path, **query**, **headers with
original casing and order**, raw body bytes, arrival timestamp, and — for containment — **the
peer port of the accepted connection**, which is the join key against the proxy's tunnel log
(§5.2.1). The routing fields are not decoration: §3.3.5 asserts on them, and on Azure they carry
the only difference between two otherwise identical requests. Each replays
scripted responses: SSE streams, error statuses, Cloudflare blocks, empty responses,
context-too-large rejections, and disconnects at each of §6.3.1's four injection points.

**Casing** is asserted by C1; **order** is recorded for the C1b baseline report only, not
asserted — what reaches the wire is the client library's ordering, not the agent's. Peer
*address* is deliberately not used for containment (§5.2.1).

#### 7.2.1 What T-W4 settled — read this before writing another recorder

The primary recorder is `tests/harness/recorder.py`; the contract every recorder is
judged against is `tests/harness/recorder_conformance.py`, and plan §5 requires T-B1–T-B3 to
pass it. The following were established by probe on the pinned stack (aiohttp 3.13.5) and are
not matters of taste — each is a way a recorder can look correct and lie.

| Field | Read it from | Never from |
|---|---|---|
| Headers | `raw_headers`, decoded **latin-1** | `request.headers`; and never `utf-8`, which refuses obs-text |
| Path | `rel_url.raw_path` | `request.path` — percent-**decoded** |
| Query | `rel_url.raw_query_string` | `request.query_string` — percent-**decoded** |
| Authority | the `Host` header itself; `""` when absent | `request.host` — falls back to `socket.getfqdn()`, **inventing the build machine's name**, and that fallback differs across the `>=3.11,<3.14` pin range. `request.url` lower-cases the host |
| Peer port | `transport.get_extra_info("peername")[1]` | `request.remote` — address only |

Three server-construction facts, each of which fails as a bare client-side
`ConnectionResetError` with no traceback if got wrong:

- **`client_max_size=0`**, supplied through a `request_factory` — it is a `BaseRequest`
  argument, *not* a `Server`/`RequestHandler` one. The 1 MiB default turns §7.1's
  over-budget corpus entries into a harness **413**, which is M6's own trigger, so the
  harness would manufacture the compaction the oracle exists to detect. Zero disables the
  check outright; a fixed ceiling is something a later corpus entry outgrows, and the
  comparison is `>=`. The recorder then buffers bodies unbounded, which is acceptable only
  because it is a loopback double whose clients are the harness's own.
- **`auto_decompress=False`**. Left on, a `Content-Encoding: gzip` body is captured
  decompressed *beside a header saying it is compressed* — an internally inconsistent
  capture, and C1 asserts on that header.
- **Connection logging via `web.Server.connection_made(handler, transport)`**, which is the
  only public seam that sees a connection carrying **no** request — §5.2.1's bypass shape,
  and structurally invisible in any request list. Wrapping the protocol object is
  impossible: `RequestHandler` defines `__slots__`. Key the log on the **handler object**;
  never on `id(handler)`, whose addresses CPython reuses, which would reintroduce the
  port-reuse aliasing one level down, and never on a `WeakKeyDictionary`, because
  `RequestHandler` is not weak-referenceable.

**Limitation.** With a TLS-terminating site, `connection_made` fires *after* the handshake,
so a connection that fails negotiation is not logged — and §5.2.1 names a failed TLS
negotiation as a real bypass. **T-B2 and T-E2 must observe at socket level or accept this
explicitly.**

**A recorder produces no `CapturedReply`.** The reply is the script; the test already knows
it. T-A7 and T-D10 need not reopen this contract to ask for one.

**The reply's format is chosen by a path *suffix*** — `/v1/messages` or `/messages` for
Anthropic Messages, `/chat/completions` for Chat Completions — because a recorder is
impersonating a provider and providers dispatch on the URL. An exact match would serve
`anthropic` and `custom_anthropic` and get Azure, vertex, ollama, opencode and
zai_anthropic wrong. §3.3.4's "select by the shape observed on the wire" governs the
**oracle's** choice of projection and must not be wired to this decision. An unmatched path
falls back to a declared default and is **recorded**, failing the fixture at teardown: a
wrong-format reply is not a loud failure but an apparently empty response, and that costs
the 80-second retry ladder.

**One thing no bridge-side judgement checks.** §4.3 C3's emptiness oracles are keyed on the
**upstream** format: `_is_empty_cc_response` for non-streaming, `translator.response_was_empty`
plus the `has_content` byte flag for Chat Completions streams — and **nothing at all** for a
native Anthropic Messages stream, which `server.py:3616` forwards to the client byte-for-byte.
A recorder's Anthropic SSE success is therefore guarded only by §6.2.2's grammar. That is a
property of the product, not of the harness: an upstream returning a well-formed but
contentless Anthropic stream reaches Claude Code with none of the retry the Chat Completions
path has.

### 7.3 Recording CONNECT proxy

**Delivered by T-W5 ([KBR-28]) in `tests/harness/connect_proxy.py`**, extracted from
`tests/test_egress_https_proxy.py` rather than written a second time — two proxy implementations
is how two harnesses come to disagree about what "tunnelled" means. That module keeps its five
tests, unchanged down to the collected node ids, and drives all three transport stacks through the
extracted fixture; it is the regression evidence for the extraction.

What it provides:

- **The fixtures**, registered as a pytest plugin from `tests/conftest.py`, so a test asks for
  `connect_proxy` or `tls_target` by name and imports nothing.
- **`ConnectAttempt.source_port`** — the local port of the proxy's outbound socket for each tunnel
  it opens, and `None` where no tunnel was opened (rejected auth, or an unreachable upstream).
  This is §5.2.1's join key; target and auth status alone cannot identify a connection at both
  ends.
- **`unattributable_peer_ports(peer_ports, attempts)`** — §5.2.1's assertion, stated once so each
  transport slice inherits it rather than re-deriving it. It refuses an unset peer port rather
  than guessing: a recorder that never populated `CapturedRequest.peer_port` would otherwise make
  containment unfalsifiable in whichever direction the default happened to fall. *This is an
  addition to T-W5's two named deliverables, made because §1.4 requires this delivery to ship a
  falsification case and a falsification case needs a checkable assertion.* It does not discharge
  **T-E2's** phase-3 obligation, which injects a bypass into the product rather than into the test.
- **`ConnectProxy.stop()` and `TlsTarget.stop()`** — mid-test stoppability for §5.2.2 phase 2.
  Both abort live connections rather than closing them (a TLS `close()` waits out
  `ssl_shutdown_timeout`, 30s by default) and both retry until the listener has actually finished
  closing. Python 3.12.1 changed `asyncio.Server.wait_closed()` to block until every connection is
  dropped while 3.10 and 3.11 return immediately, so a teardown that merely closes the listener
  hangs on half the support matrix and passes on the other half.
- **The proxied-leg resolver** — `HARNESS_UPSTREAM_HOST` (`upstream.kitty-test.invalid`, §5.3) in
  the target certificate's SAN, and an empty-by-default `ConnectProxy.resolve` map consulted
  before the outbound connection. The harness owns the resolver because the harness is the proxy;
  a client tunnelling to a `.invalid` name never resolves it itself, so this is the only place the
  name can be mapped. **Shipped here, not in T-E1**, because T-E1 would otherwise have to edit a
  module five tickets consume — the coordination problem Milestone 0 exists to remove.

Still **T-E1's** (KBR-61): the per-transport **direct**-route override §5.2.2 phase 1 needs, and
which transport gets which route. T-W5 ships the seam, not the policy.

**A missing `openssl` fails, it does not skip.** `certs` is shared infrastructure, and §8's rule
is that a skip in a gating job is a failure: a suite that quietly stops proving containment
because a tool is absent is indistinguishable from one that proves it.

[KBR-28]: https://shelpuk.atlassian.net/browse/KBR-28

### 7.4 Wire projections and the transparency oracle

**The projections (§3.3.1)** are the load-bearing piece: one hand-written reader per wire format —
Anthropic Messages, Chat Completions, OpenAI Responses, Gemini, Bedrock Converse, Ollama
`/api/chat` — each mapping a serialized body to the common `Conversation` form and **importing
nothing from `src/kitty/bridge`**. They are test code that the whole of I1 rests on, so they get
their own L1 tests, written against each format's published examples rather than against kitty's
output.

**The oracle** is a pytest fixture wrapping the recorders, exposing one assertion over **complete
requests**, not bodies:

```
CapturedRequest(method, scheme, host, path, query, headers, body,
                arrival?, peer_port?)      -- the last two are T-W4's to populate (§5.2.1)
CapturedReply(status, headers, body)

Projection      -- wire_format: WireFormat ; read_request(CapturedRequest) -> Request
ReplyProjection -- wire_format: WireFormat ; read_reply(CapturedReply)    -> Reply

verify_total(projected)                     -- the totality rule; Request or Reply

assert_no_unclaimed_mutation(inbound:  CapturedRequest, inbound_format,
                             captured: CapturedRequest, captured_format,
                             register, triggers_met,
                             expected_route)   # derived from the profile, independently
```

`WireFormat` is a **closed** enumeration of the six formats above. §7.4 already notes that a
boolean declaration cannot select among six projections; a bare string has the opposite failure,
where six reader authors spell one format three ways and the format-keyed lookup silently misses.

**Two protocols, not one with two methods.** §3.3.1 makes response translation "a different claim
[that] gets a different test", and the plan splits the work as six request readers (T-A1–T-A6)
against one reply task (T-A7) with its own comparison (T-D10).

**`arrival` and `peer_port` sit on `CapturedRequest` but belong to T-W4.** They serve containment's
tunnel join (§5.2.1), and no fidelity assertion reads them. They live on the shared type rather than
on a T-W4 subclass so that T-W4, T-B1–T-B3 and T-E2 consume one type instead of two.

**Projection values are not hashable** — `__hash__` is set to `None` on every one, deliberately, so
the limitation is total rather than data-dependent. §3.3.3's counterpart matching must therefore be
an **order-aware multiset** match, not a set difference: a set-based implementation would pass on
text-only fixtures and raise on the first corpus entry carrying a `ToolUse`. Capture types *are*
hashable; only projections are not.

**A body that cannot be read raises `UnreadableBodyError`**, named in the contract so that T-C6's
malformed corpus entry is distinguishable from an I1 breach without catching bare `Exception`.

**Bodies alone cannot prove correct routing** (§3.3.5). Both formats are supplied by the harness
from the observed wire shape (§3.3.4), never read from the adapter's own declaration. That
declaration was unreliable (F5, KBR-7, since fixed) — but the decision does not rest on that: an
oracle must not ask the code under test what it did, and a **boolean** declaration cannot
select among the six projections listed above in any case. A new corpus entry, adapter, model
route or transport costs one parametrisation, not a new test.

#### 7.4.1 What every reader does with a field the grammar cannot carry

Six request readers and one reply task — seven authors — are written against §3.3.1's grammar,
and the grammar is deliberately narrower than the six wire formats. §3.3.1 fixes the *shapes*;
§3.3.1b fixes the *normalisation*. This section fixes the remaining ten decisions, each of which was reached
writing **T-A1** and each of which six later authors would otherwise answer differently. A
disagreement here is not a style difference: paths are index-based, so two readers that disagree
about a part boundary report a delta on every subsequent part.

**Where a projection lives.** `tests/harness/reader_<format>.py`, one module per `WireFormat`
member — **six**, not seven — with `test_reader_<format>.py` beside it. Flat, next to
`contract.py`, which is what §3.3.1 already says of every harness module: "each in its own module
beside it". An earlier draft of this section put them in a `projections/` package; two readers
were written against the two conventions within a day of each other, which is the coordination
failure this section exists to prevent, so the design's own existing wording wins.

`WireFormat` has six members and Epic A has seven tasks because T-A7 is the *reply* direction
across formats, not a seventh format: each format's `ReplyProjection` lands in that format's
existing module, beside its `Projection`. One module per format, both directions, because the two
share that format's vocabulary and nothing else does.

**Which of these rules are request-side only.** The turn merge and the tool-result-string rule
address `Conversation`, which `Reply` does not have. Everything else below binds both directions.

**Fail closed at every depth.** §3.3.1's "unknown fields fail closed" is not a top-level rule. A
key the mapping does not consume residualises under its path, whatever its depth. `verify_total`
cannot see past the top level (§3.3.1 records that boundary and why closing it would make the
contract a second reader), so this is the rule that closes it, and it is each reader's own L1
tests that hold it.

**Residual *keys* are the body's own path, indexed, and never the delta spelling.** Two rules,
and the second is the one that surprises:

1. A wholly-unclassified **top-level** key is keyed by its bare name — `x-kitty-trace`, never
   `residual[x-kitty-trace]`. `verify_total` computes `set(consumed) | set(residual)` against
   `set(source)`, so the wrapped form misses `source` and raises `DroppedFieldsError`, naming the
   wrong defect. `residual_path()` renders a *delta path* for the oracle to report; it never
   builds this mapping.
2. A nested key is keyed by its path from the body root with **array positions as indices** —
   `tools[0].type`, `messages[2].content[0].cache_control`, `system[0].cache_control`. It does
   **not** inherit §3.3.1a's by-name tool addressing. That convention exists because "translators
   reorder and filter declarations", which is a property of a *comparison*; a residual key is
   never matched against a register pattern, so the reason does not apply and one rule is better
   than two. T-D8 diffs residual key sets across all six readers and index-here/name-there is
   exactly the drift this section exists to stop.

**`Opaque` consumes its block, and carries a payload digest.** A block type the grammar does not
model projects as `Opaque(kind=…, digest=…)` where:

- `kind` is the wire `type` converted to snake_case. Anthropic's spellings (`document`,
  `search_result`, `redacted_thinking`, `server_tool_use`) are already canonical; Converse writes
  `searchResult` for the same thing, so **the cross-vendor alias table lands in this section with
  the first reader that needs one** (T-A5), rather than being invented twice.
- `digest` is exactly
  `hashlib.sha256(json.dumps(rest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()`,
  where `rest` is the block without `type` and without `cache_control`.
- the block's every other key is **consumed** — nothing beneath it residualises — while
  `cache_control` residualises under its path exactly as it does on a modelled block.

Each clause is load-bearing. A bare `Opaque("document")` makes two different documents project
identically, so a swapped or truncated document produces no delta at all — and §3.3.1 put
`digest` on `Opaque` precisely to keep unmodelled content "detectable". Residualising every
payload key instead would fail the run on every `document` block, which is not a defect signal but
the grammar's known limit. **Canonical JSON rather than the raw wire slice**, because a translator
that reorders keys must not change the digest; that is the whole reason the recipe is not
`sha256(raw_block_bytes)`. **`ensure_ascii` is pinned** because its default is `True` while the
surrounding prose says UTF-8: an author who "helpfully" passes `False` gets a different digest for
the same block, and it would surface only on non-ASCII content. **`cache_control` is excluded**
so that one field behaves the same way everywhere — inside the digest it would produce a delta
with no named cause, on a path where the same field on a modelled block produces a diagnosis.

> **The cost, recorded so it is not discovered later.** The digest is over *that format's* JSON, so
> one document carried from Messages to Converse digests differently and shows a cross-format
> delta no mutation caused. Modelling six vendors' block zoos is what §3.3.1 declined to do, so
> the alternative is not on offer. **T-D9's cross-format matrix is where this will first bite**,
> and the fix, when it is needed, is a per-kind payload rule here — not six readers each
> inventing one.

**A wrongly-typed leaf residualises — it is neither coerced nor raised on — and the rule is
general.** It binds *every* optional leaf, not the ones a bug happened to be found in: the
contract validates only `Turn.role`, `Conversation.sampling` and `extra["tool_choice"]`, so an
unguarded leaf declared `str | None` carries a dict silently with an empty residual. A reader
should apply it through one helper, so the next field added inherits it. `str(7)` and
`dict(["ab", "cd"])` invent a value the agent never sent, and a silently nulled tool description is
indistinguishable from the deletion §3.3.1's own falsification set injects. Raising is the other
wrong answer: it blinds the oracle to everything else in a request it could otherwise diff, and
`verify_total` cannot see a nested coercion because `consumed` is top-level only. So the field
residualises at its own path, the projection carries the grammar's absent value in its place, and
the run fails with the field named.

Structural failures are the exception and still raise `UnreadableBodyError` — a role outside
`user`/`assistant`, a content block with no type — because there is no partial projection to
salvage: the turn cannot be built at all.

**A tool-result string is always `Text`, never `Json`.** `Json` is for a format that carries a
structured value natively — Converse's `toolResult.content.json`, Gemini's
`functionResponse.response`. A Messages or Chat Completions tool result whose content is the
*string* `'{"price": 259.75}'` is text that happens to parse. Without this fixed once, one reader
parses and another does not, and every JSON-shaped tool result shows an unclaimed delta on the
Messages ↔ Chat Completions comparison the oracle rests on.

**§3.3.1b's merge rule is an ordered pipeline, and the last step is never a re-sort.** Its four
clauses run in order: a maximal run of consecutive tool results forms one turn; an immediately
following non-tool user message merges into it; `ToolResult` parts come first **within the turn
those first two clauses build**; then consecutive same-role turns merge. Read as four independent
rules — "merge everything, then hoist every result in every user turn" — it produces a different
conversation, and a wrong one:

> `tool_result → user(text) → tool_result` must project as `[ToolResult, Text, ToolResult]`. A
> re-sort after the merge gives `[ToolResult, ToolResult, Text]`, hoisting a result ahead of text
> the agent sent **before** it — moving history the bridge did not move. Because paths are
> index-based, that invented delta lands on every part of the turn and on every turn after it.
> **Two readers were written against the two readings within a day of each other**, which is why
> the ordering is now stated here rather than inferred from the bullet's sentence order.

**Where each clause does its work depends on the format, and clause 3 is the one that moves.**
Chat Completions and Responses deliver results contiguously in their own messages or input items,
so a run is delimited by the wire itself and clause 3 is satisfied *vacuously* — every member of
the run is already a result. Anthropic Messages carries text and results inside **one message**, so
the run has no natural boundary and clause 1 cannot do the work by splitting; clause 3 does it
instead, **per message and for a `user` message only**.

That scoping is the whole distinction. Per message, `tool_result → user(text) → tool_result`
keeps three groups and concatenates to `[ToolResult, Text, ToolResult]`. Applied to the *merged*
turn it gives `[ToolResult, ToolResult, Text]` — the defect above. And omitting it entirely is
also wrong: a single Anthropic message of `[text, tool_result]` would project `[Text, ToolResult]`
where the Chat Completions and Responses readers both produce `[ToolResult, Text]` for the same
content, which is a delta no mutation caused.

> **What the merge hides**, stated because §3.3.1 requires a projection's blind spots to be on the
> record: a mutation whose only effect is to split or join two consecutive same-role turns
> produces no delta, which weakens §3.3.2 assertion 2 for M5, M6 and M7, whose complement case
> must show the mutation *absent*. Accepted, because without the merge the Messages and Chat
> Completions readers disagree about turn boundaries on the standard
> `assistant(tool_calls) → tool → tool` exchange, and that disagreement reports a false delta on
> every subsequent turn.

**A key outside the published schema is recognised only on stated evidence, and there are two
kinds.** The default is the published schema: §3.3.1's independent-oracle rule means a reader is
written against the format's own documentation, not against what kitty happens to emit. Two narrow
exceptions, and they are not interchangeable:

1. **The client demonstrably sends it.** Then it is a recognised control field and maps to
   `envelope.extra[<wire key>]`, with a comment naming the evidence. Residualising it would fail
   the run on every real request, which is a harness defect and not a finding.

   The worked example is Anthropic Messages' top-level **`effort`**. Claude Code sends it and
   `MessagesTranslator` reads it straight off the inbound body
   (`src/kitty/bridge/messages/translator.py`: `if "effort" in messages_request`), so it is the
   *agent's* field, not kitty's. Anthropic's API reference does not list it — the feature page
   spells the concept `output_config.effort` — which is why the rule is about evidence rather than
   about the reference table.

2. **Only kitty emits it, and a register row names it.** Then it maps the same way, because
   residualising it would fail the run *before* register matching happens and the row could never
   be claimed — the under-claiming direction §3.3.1a calls unrecoverable.

   **No key currently exercises this branch**, and that is worth saying: an earlier draft of this
   section presented it as the general case using `effort` as its example, which is wrong twice
   — `effort` is client-sent, and generalising from it would have told six authors to ask "does a
   register row name this?" about fields no row names, such as `context_management` and
   `output_config`, and to residualise them.

A key kitty emits that **no** register row names is an *unregistered* mutation. That is the defect
the oracle exists to find, and it must residualise.

**The third outcome §3.3.1 promises does not exist yet.** §3.3.1 says adding a field to a wire
format "forces a deliberate decision: map it, or declare it ignored with a reason". The contract
implements the first and the residual; there is **no reader-side declared-ignored mechanism** —
`NOT_PROJECTABLE` is a sentinel for a register row's `paths` tuple, not something a reader can
say. Until T-W2 adds one, a field the grammar cannot carry has only the residual, and the residual
fails the run.

That is the correct signal and it is also a deadline. Claude Code sets a block-level
`cache_control` on nearly every request and the grammar has no slot for it, so **T-C2's corpus
entry — `system` with `cache_control` — fails the first oracle run.** Tracked as a blocking edge
onto T-D1, not as a note here.

#### 7.4.2 What T-A4 settled — seven more rules, and which readers each one binds

§7.4.1 fixed ten decisions writing T-A1. Writing **T-A4** (Gemini, [KBR-36]) reached seven more, each
of which the remaining authors would otherwise answer differently, and each recorded here for the
same reason: paths are index-based and `envelope.extra` is keyed, so two readers that disagree
report a delta on content nobody changed.

**Two of the seven are Google-specific and five are not**, which matters because only the five are a
standing obligation on the rest of Epic A. **Rule 7 is different in kind from the other six**: it is
not a new question this format raised but one the three shipped readers have already answered three
different ways, which makes it the only rule here that is also a correction.

| Rule | Binds |
|---|---|
| 1 — the route as a reader input | Gemini alone; no other format puts routing in the inbound URL (§3.3.5) |
| 2 — a nested control field flattens to its leaf published key | **T-A5**, whose `inferenceConfig` and `toolConfig` nest the same way |
| 3 — ProtoJSON's two spellings, and case-insensitive enum values | Google formats; a **Vertex** reader, if one is ever added, inherits it |
| 4 — `envelope.extra` keyed by the *published* spelling | every reader of a format with more than one legal spelling, so today rule 3's set |
| 5 — a capability toggle the wire does not name is control, not a `ToolDecl` | **T-A5** and **T-A6** |
| 6 — digest the payload, not the carrier, where the field name discriminates | any format whose content union is discriminated by field name rather than by a `type` member |
| 7 — a union member's own value is wrong: raise, residualise, or drop? | **every** reader; three answers are already shipped |

**1. The route is a reader input, and only for what the body cannot show.** Gemini alone puts the
model and the operation in the URL (§3.3.5), so its reader derives `envelope.model` from the path
segment and `envelope.stream` from the operation suffix — `:streamGenerateContent` versus
`:generateContent`, and **never** from `?alt=sse`, which selects SSE framing over JSON-array framing
for a method that streams either way. The **query string is not read at all**: `verify_total`
compares `consumed | residual` against the *body*, so a query key in either account would be
reported as a claim on a key the body does not have. Asserting the route is T-D2's.

A path that is not a published generate route raises `UnreadableBodyError`. That widens a type §7.4
describes as "a body that cannot be read" to cover a *route* problem, which is defensible only on
the inbound direction, where the path is the client's — and it is stated here because T-D1 uses that
one exception type to tell a malformed corpus entry from an I1 breach.

**2. A nested control field is addressed by its leaf published key.** Gemini puts sampling under
`generationConfig` and the tool choice under `toolConfig.functionCallingConfig`; Converse nests
`inferenceConfig` and `toolConfig` the same way, so T-A5 inherits this. `extra_path()` **raises** on
a dotted key, which leaves exactly two dotless candidates, and the container loses:
`envelope.extra[generationConfig]` cannot collide and survives a schema revision, but §3.3.1a
compares an `extra` value **whole**, so it would collapse fourteen independently registrable fields
into one address and make any row anchored there claim all of them — the coarse-anchor failure
§3.3.1a warns about by name. The leaf key wins on the narrowest-anchor rule.

The cost is a flat namespace assembled from several nested objects, so **each reader that flattens
owes a test that its `extra` key sets are pairwise disjoint.** The namespace is not naturally
disjoint — Gemini publishes `mediaResolution` on both `GenerationConfig` and `Part`, and the only
reason there is no clash is that the `Part` one residualises. A collision would be introduced by a
*schema revision*, not by a request, and the loser would overwrite the winner with no residual and
no delta.

**3. Google's JSON has two legal spellings of every field, and both must read.** Gemini's wire
format is ProtoJSON, whose parsers "accept both the lowerCamelCase name … and the original proto
field name" (`protobuf.dev/programming-guides/json/`). This is not theoretical: Google's own
published examples mix them freely — `system_instruction`, `function_declarations`, `tool_config`,
`file_data` and `response_mime_type` in snake_case, beside `generationConfig`, `stopSequences`,
`maxOutputTokens` and `topP` in camelCase. A reader that knew only the schema's spelling would
residualise the other and **fail the run on Google's own published example**, which §7.4.1 already
calls "a harness defect and not a finding". Enum *values* are matched case-insensitively for the
same reason: the published `FunctionCallingConfig.mode` enumeration is upper case and Google's
`function_calling.sh` sends `"mode": "auto"`.

Where one object carries both spellings of one field, the **published** spelling is read and the
other residualises. Not "the first wins": §7.4.1 designs key order out of the projection elsewhere —
"canonical JSON rather than the raw wire slice, because a translator that reorders keys must not
change the digest" — and resolving by position would put it back, projecting one semantic body two
ways depending on which alias a serialiser emitted first.

**4. `envelope.extra` is keyed by the *published* wire key.** §3.3.1b says "keyed by the wire key",
which named one thing until rule 3; it now names two. A register row can name only one, and T-D9's
matrix needs one, so the published lowerCamelCase name is the address and the snake_case original
resolves onto it. This is the one place where `extra` and the residual diverge deliberately:
`consumed` and residual keys stay in the **wire** spelling, because `verify_total` compares them
against the body's own keys.

**5. A server-side capability toggle in the tools array is control, not a tool declaration.**
Gemini's `Tool` message carries eight of them beside `functionDeclarations` — `googleSearch`,
`codeExecution`, `urlContext`, `fileSearch`, `computerUse`, `googleMaps`, `googleSearchRetrieval`,
`mcpServers` — each an unnamed toggle object such as `{"googleSearch": {}}`. They map to
`envelope.extra[<key>]`.

**This departs from T-A3**, which makes a non-`function` Responses tool a `ToolDecl` whose name is
the tool type, and the departure is the point: a Responses built-in tool *has* a name to be
addressed by, and §3.3.1a's tool paths are by name. Gemini's has none, so a `ToolDecl` would have to
invent one — putting a vendor spelling into a form whose purpose is wire independence — while
residualising would fail the run on every request that enables Google Search. The distinguishing
question is therefore **"does the wire name this tool?"**, not "is it built in".

**6. Digest the payload, not the carrier, where the field name is the discriminator.** §7.4.1's
recipe says `rest` is "the block without `type` and without `cache_control`", which assumes a block
discriminated by a `type` member. A Gemini `Part` is a union discriminated by **field name** and
defines no `cache_control`, so the digest is taken over the *payload object* — `part["executableCode"]`
— and neither exclusion has anything to remove. Sibling members on the same part residualise, which
is exactly the role `cache_control` plays on a `type`-discriminated block. The recipe itself is
unchanged, `ensure_ascii` included.

**7. When a union member's own value is wrong: raise, residualise, or drop — and never drop.** Three
readers have shipped and all three answer differently, which is the coordination failure §7.4.1
exists to prevent, so it is settled here rather than left to T-A5 and T-A6 to pick a precedent from.
The distinction is **what the bad value is a value *of***:

| The wrong value is… | Outcome | Because |
|---|---|---|
| the value that **is** the part — `Part.text`, a `Thinking`'s text, `Opaque.kind` | **raise** `UnreadableBodyError` | the grammar has no absent value to fall back to, and `Text("")` fabricates an empty part — which is *meaningful* here, since P5e and P8 both inject one |
| a **required field** of a part — a tool `name` | **residualise**, project the part with `""` | §3.3.1b settles it in those words: "an absent `name` *does* residualise … a call nobody can name cannot be paired or addressed" |
| a **payload** the reader cannot canonicalise — base64 that does not decode | **residualise the leaf**, project the part with the grammar's absent value | `Image.digest` is `str \| None`, so an absent value exists, and §7.4.1: "raising is the other wrong answer: it blinds the oracle to everything else in a request it could otherwise diff" |
| the **member itself**, where the schema declares an object and the wire sent a scalar — `{"functionCall": 7}` | **raise** | there is no value to put in the position, and the position cannot be vacated |

**The line between the last two rows is where the member sits, not how bad the value is.** A
container under the **envelope** — `generationConfig`, `toolConfig` — residualises whole when it is
not an object, because every envelope field has an absent value and the rest of the request still
projects. A container that **is a part or a turn** raises, because a part must occupy its index and
the grammar offers nothing to put there: residualising it would leave the position empty, which is
the drop this rule forbids. State the question as *"can the projection still fill this position?"*
and every case above falls out of it.

**No branch ever returns *no part*.** That is the load-bearing half, and it is where two of the three
shipped readers are wrong: `reader_responses.py` returns `None` for both a wrongly-typed `input_text`
and an undecodable data URL, which drops the part and shifts every later part's index — §7.4.1's own
warning that "that invented delta lands on every part of the turn and on every turn after it". A
reader that cannot read a part must still *occupy its position*.

> **Reconciliation owed.** `reader_anthropic_messages.py` raises on undecodable base64 where this
> rule residualises, and `reader_responses.py` drops a part where this rule keeps it. Both predate
> this section. T-A4 is the reference implementation; the two landed readers need conforming, and
> that is a change to shipped code rather than a note, so it is tracked as its own ticket.

> **What this reader leaves on the record.** Six fields the format publishes, real clients send, and
> the grammar cannot carry now residualise and so **fail the first oracle run** — the same shape as
> the `cache_control` deadline above, and tracked as its own defect rather than as a note here. The
> sharpest is `thoughtSignature` on a `functionCall` part, which Gemini 3 *requires* clients to echo
> back verbatim. Unlike `cache_control` the grammar nearly has the slot — `contract.py` already says
> `Thinking.signature` carries "Anthropic's `signature` or Gemini's `thoughtSignature`" — so the fix
> is small and specific rather than open-ended.
>
> A second consequence, on the oracle rather than the reader: `GeminiTranslator` **discards** the
> inbound tool-call id and synthesises one per call (`_make_tool_call_id`). Now that the §3.3.1 correction above
> projects the wire id, every tool turn from a client that populates one shows a delta with no
> register row to claim it. That is a correct oracle finding, not a reader defect, and it needs a
> row or a ticket before T-D9 runs.

### 7.5 The bridge fixture

`tests/harness/bridge.py` — plan task **T-W8** ([KBR-31]). The counterpart of §7.2 on the
inbound side: §7.2 says what a recording upstream must observe, this says how a **real
`BridgeServer`** is put in front of one. Around fifteen tasks across Epics D, E, G, I, J and K
need it, and `tests/conftest.py` offers only `unused_tcp_port` today. Measured in-repo: of the
**62** test modules under `tests/bridge/`, **38** start a real `BridgeServer` and most build
their own stub adapter, fake upstream and `post()` helper; 25 of those *also* intercept the
upstream with `aioresponses` rather than a real socket, and 24 start no server at all.

**This is the one `tests/harness/` module that imports the product.** `contract.py` and
`recorder.py` each carry a structural guard forbidding any `kitty` import, because §3.3.1's
independent-oracle rule says a reader that asked kitty how to parse a body would inherit
kitty's bugs. That rule governs what *judges* a request. This module *starts* the thing under
test, so it must import `kitty.bridge.server`; the guards are per-module and nothing here
weakens them. Its own structural guard is a different claim — that it registers no Epic B
transport (below).

**What it delivers.**

| Piece | What it is |
|---|---|
| `UpstreamTransport` | The **extension interface**: `name`, `format`, `start`, `stop`, `bind`, `captures`, `connections`, `assert_teardown_clean`. **Open question for T-B1:** `format` is typed `WireFormat`, which §3.3.1 closes at six — and the `openai_subscription` **OAuth token leg** §7.2 assigns it is none of them. T-B1 decides whether that leg is a separate transport, a recorder outside this interface, or a seventh format; named here so the decision is deliberate |
| The registry | `register_transport` / `transport(name, fmt, *, responder)` / `registered_transports` |
| `AiohttpTransport` | The **only** transport registered here, over §7.2's `RecordingUpstream` |
| `redirected(adapter, origin, provider_config)` | Re-hosts any **default-transport** adapter onto a recorder, keeping path and query |
| `profile_for` / `backend_for` | Valid `Profile` objects and `(adapter, key, profile)` backend triples already pointed at a transport |
| `pin_backend_order` | The determinism seam for balancing mode |
| `InboundProtocol`, `inbound_path`, `minimal_inbound_body` | The agent-facing side: route and smallest body per inbound protocol |
| `BridgeFixture` | An async context manager: a real `BridgeServer`, single-backend or balancing, against a started transport |
| `assert_transport_reaches_its_recorder(transport, *, protocol=None)` | The **integration conformance check**, run over every registered transport by a meta-test. Takes a transport **instance**, not a registry name — so the falsification defects need never be registered, and an author whose format has no matching inbound route names one |
| `marker()`, `protocol_for()` | The per-request marker, and the inbound route matching an upstream format |
| `MisdeclaredFormatError`, `TransportTimeout` | What a teardown check and a timed-out request raise |

**Naming.** One list of captured requests, spelled `captures` on a transport and `requests` on
`RecordingUpstream` (§7.2's own name, unchanged). `CapturedRequest` is the element type.
`assert_teardown_clean` is deliberately *not* called `check`: `recorder_conformance.py` already
owns `check_*` for "the contract every recorder is judged against" (§7.2.1), and an Epic B author
told to "implement `check()`" would reasonably read it as that suite.

#### 7.5.1 Two axes, not one: inbound protocol and upstream format

They are independent, and translating between them is what the bridge *is*. `WireFormat`
(six values) describes what a **recorder** serves; `InboundProtocol` (four —
`chat_completions`, `messages`, `responses`, `gemini`) describes the **bridge route an agent
posts to**. `BEDROCK_CONVERSE` and `OLLAMA_CHAT` have no inbound route at all, and the oracle
(§3.3) exists precisely because the two sides differ. A single parameter for both would work
only for the pass-through pairs T-W8 happens to ship and would mislead every reader after that,
so the fixture keeps two vocabularies.

#### 7.5.2 `bind()` is the seam, and redirection **re-hosts** rather than replaces

A transport returns the `(ProviderAdapter, provider_config)` pair that points a bridge at *its
own* recorder. The obvious alternative — the fixture reading `recorder.base_url` and building the
config itself — works for the one transport T-W8 ships and for none of the other three: botocore
takes an **endpoint override**, curl_cffi terminates TLS with the harness certificate, and the
`openai_subscription` OAuth leg is a different session entirely (§7.2). Defining the interface
here is what lets T-B1–T-B3 register a transport without editing a module fifteen tickets consume.

**`provider_config["base_url"]` is the product's own channel, and it reaches five adapters.**
Read from every file under `src/kitty/providers/`: `ProviderAdapter.build_base_url` ignores
`provider_config` and returns `default_base_url`. Four adapters override it to read that key —
`custom_anthropic`, `custom_openai`, `ollama`, `minimax` — and **`ollama_cloud` honours the same
key by a different route**, resolving it inside its own custom transport rather than through
`build_base_url`, which an enumeration of `build_base_url` overrides misses. `vertex` and
`minimax_token` override it to read *different* keys, and the remaining ~14 never override it.
So the channel cannot redirect the "20 default-transport adapters" §7.2 names, which is the first
thing T-D1's "per adapter × model × transport" hits.

**`redirected(adapter, origin, provider_config)` closes the gap by overriding `build_base_url`** —
the single method the bridge calls for the destination (`server.py`, `_build_upstream_url`).
Overriding `default_base_url` instead, the convention `tests/bridge/` established by hand, is
**not** equivalent: the six adapters that override `build_base_url` ignore it.

**It substitutes scheme and authority only, and keeps path, query and fragment.** Replacing the
whole base URL is the obvious implementation and it is wrong, for a reason §3.3.5 already states
from the other side: T-D2 must rewrite the **authority and scheme** of its expected route before
comparing, and "path and query need no such treatment". Some adapters put routing *in the base
URL's path* — `vertex` returns
`https://{location}-aiplatform.googleapis.com/{version}/projects/{project_id}/locations/{location}`,
and §3.3.5 records that for Vertex "the account being billed is a URL component" (P21). A
whole-URL replacement deletes that path, while T-D2's independently-derived expectation still
carries it, so P21 would report a routing mismatch against a request that went exactly where the
harness sent it — indistinguishable from a real misroute.

The trap is that the obvious check passes: **Azure survives a whole-URL replacement**, because
`get_upstream_path(model)` carries its deployment id and `?api-version=`, not the base URL. Azure
is §3.3.5's lead example, so a seam validated against it looks correct and silently breaks the
row two below. Re-hosting makes the binding side and the comparison side the same rule, which is
a stronger guarantee than either stated alone.

**The redirect runs the adapter's real `build_base_url` once, at redirect time.** Two consequences
worth knowing. It preserves whatever the adapter computes from `provider_config`, which is the
point. And `build_base_url` is also a **validator** — `validation.py` calls it for pre-flight, and
Vertex raises `ProviderError("Vertex AI requires 'project_id' …")` from inside it — so a redirected
adapter raises that at redirect time and no longer raises it per request. A test meaning to
exercise pre-flight validation must call the adapter, not the redirected copy.

**The default transport never uses `redirected()`.** It binds `custom_anthropic` or
`custom_openai` precisely because those two honour the product's own channel, so the default path
exercises that channel and the helper is exercised separately — it exists for the ~14 adapters
that honour no key.

**This applies to default-transport adapters.** The three custom-transport adapters never call
`build_base_url` at all: `ollama_cloud` resolves the key inside its own request path,
`openai_subscription` uses a module constant, and `bedrock` takes a botocore endpoint override.
Redirecting those is T-B1–T-B3's business, through `bind()` — which is exactly why `bind()`, and
not a base-URL helper, is the interface.

**`provider_config` arrives by two routes.** In balancing mode `_get_next_backend` passes
`profile.provider_config`; in single-backend mode the same branch returns the constructor kwarg.
`Profile.base_url` is **neither**: the resolver never reads it, and it is typed `HttpsUrl`, which
a loopback recorder serving `http://` cannot satisfy. The fixture also encapsulates the
constructor's `adapter=None` positional and its `# type: ignore[arg-type]`, which 38 modules
currently repeat.

**No pytest fixture, and not in `pytest_plugins`.** T-W5's proxy is published that way because
a test asks for `connect_proxy` by name and cannot construct one. This module is different on
both counts: a test must choose a wire format and a bridge shape anyway, so
`BridgeFixture(transport("aiohttp", fmt))` says more than a fixture name would — and importing it
pulls in `server.py`, measured at **0.72 s**, which a global plugin charges to every pytest
invocation including those that never start a bridge.

**No `host=` on the factory.** An earlier draft added one "so T-E1 can rebind to §5.3's
non-loopback name". It cannot be justified: §7.3 gives T-E1 non-loopback addressing by
**resolution** — `HARNESS_UPSTREAM_HOST` in the target's SAN and the proxy's `resolve` map, "the
only place the name can be mapped" — not by a bind address, and a CI runner may have no
non-loopback address to bind. It is dropped by the same standard that admits `responder` and
`connections`: a seam is paid for by a named consumer, and T-E1 can add one when it has a need
a resolver cannot serve.

#### 7.5.3 Balancing is part of the core, and undisciplined balancing is vacuous

`BridgeServer` has two shapes, and `_select_backend`, failover, reserve-tier selection and
`_backend_context` isolation are unreachable through the single-backend constructor — four Epic I
tickets would each have to build the second shape themselves.

Selection is **weighted-random** (`random.choices(tier, weights=…)`), so a balancing test without
a seam may never try the backend it is about to assert on. That is not hypothetical:
`tests/bridge/test_opencode_responses_refusal.py` records that its two-backend test "would pass
with the refusal handler deleted, **which is how this was found**". `pin_backend_order` ships the
round-robin seam once, rather than four Epic I tickets re-deriving the same monkeypatch. It takes
a `monkeypatch` and is therefore callable only from a test; the conformance check below is
single-backend and never needs it. It patches `random.choices` on the **stdlib module object**
reached through `kitty.bridge.server.random`, so for the duration every caller in the process gets
round-robin, not only the bridge; `monkeypatch` reverting it is what contains that.

**All members share the fixture's one transport, and therefore one recorder.** Distinct models per
member are what make the selection observable. A test needing members on *separate* recorders
builds its own `backends` list from `backend_for`; per-backend upstream *behaviour* — T-I8's
blip-then-success, T-I11's mid-stream failover — is scripted through a stateful `responder` under
`pin_backend_order`, not through separate transports. Stated here so four Epic I tickets inherit
the shape rather than each rediscovering its limit.

#### 7.5.4 Why the fixture carries its own conformance check

A bridge fixture's characteristic failure is **silent**, and it is §1.4's own shape. Measured
against `origin/main`, single-backend, non-streaming, one inbound request:

| Wiring | Inbound result | Elapsed | Captures |
|---|---|---|---|
| Pointed at its recorder | 200 | 0.01 s | 1 |
| Pointed at a closed port | 500 | **30 s** | 0 |
| Pointed at a **different live** upstream | **200** | fast | **0** |
| Recorder answers **400** | 400 | 0.00 s | 1 |
| Recorder answers **500** | 500 | 7 s | 4 (retried) |

Row 3 is the dangerous one. Nothing about the request fails, and every assertion the suite builds
on the fixture — "no unclaimed delta" (§3.3), "zero unattributable connections" (§5.2.1), every
cross-attempt and lifecycle claim in §6.3.1 — is then quantified over an **empty** capture list
and passes vacuously. So T-W8 ships the check that makes the wiring a checkable claim rather than
a promise.

`assert_transport_reaches_its_recorder` drives one request and asserts four things. Each is
paired with a defect the **other three pass** — the only argument that establishes
non-redundancy — and each defect is a **real transport instance**, driven through the real check
and running in the suite. They are deliberately **not** registered: the check takes an instance,
so a shared registry never holds four things that are wrong on purpose — which would otherwise
force the completeness meta-test below to carry a skip-list, the one mechanism it exists to
forbid:

| # | Assertion | The defect only it catches |
|---|---|---|
| 1 | Exactly one capture | **The decoy**: `bind()` points the bridge at a different live recorder. Status 200, teardown clean, and this transport's capture list is empty |
| 2 | The marker is in the capture's body | **The blind capture**: a recorder that counts the request and stores no body — §1.4's "projection that could not see the model name" |
| 3 | The inbound status is 200 | **The refusal**: upstream answers 400. Measured at one capture, marker present, teardown clean, downstream 400 — the request arrived correctly and the client was still failed |
| 4 | `assert_teardown_clean()`, run by the **fixture's** `__aexit__` | **The mis-declared format**: a transport declaring one format while binding an adapter that posts to the other. The recorder dispatches by path **suffix**, so it replies in the format the *path* named, the adapter parses it, `unmatched` stays empty — one correct capture, marker present, 200, and the declared `format` was never under test |

Row 4 is why the fourth defect is a transport and not a unit test on
`assert_teardown_clean()`. A unit test proves that function raises; it does not prove anything
**calls** it — and plan §1.4's own list of past harness failures includes "a guard proving a
function was *called* when the enforcement was the branch after it". Row 1 and row 4 are as far
apart as row 1 and row 2: one fails with an **empty** capture list, the other with a **complete
and correct** one.

**Row 4 is asserted once, by the fixture, and deliberately not repeated inside the conformance
check.** An earlier draft did both. Running the falsification procedure on it — delete each
assertion, confirm exactly its own case goes red — showed the repeat was an assertion **no defect
could falsify**: `__aexit__` raises first either way, so removing it changed nothing. An
assertion nothing can kill is the thing §1.4 objects to, so it went. The teardown call is
falsified instead by `_MisdeclaredTransport`, and a separate case asserts the fixture makes that
call on the clean path.

**An upstream 500 is deliberately not one of the defects.** Measured at four captures and seven
seconds, it also trips assertion 1, so it would not be orthogonal. A 400 is the clean one.

**The marker is in the body, and that is the opposite of `recorder_conformance.py`'s rule on
purpose.** T-W4's conformance markers travel in a **header** because the defects *it* injects
destroy every other location, and a body marker would make correlation fail with "no capture
found" instead of failing the named check. T-W8's marker must prove the **user's content**
survived translation to the upstream, which a header cannot show. Both rules are right for their
own subject; neither is a general lesson, and an author must not "fix" one to match the other.
The marker is regenerated per call, so a capture left by an earlier fixture cannot satisfy it.

**The registry earns its indirection by being iterable — and the iteration must be complete.** A
name-keyed lookup would otherwise be equivalent to importing the class, since a name resolves only
after something imported the module that registered it. What it buys is a **meta-test that runs
the check over every registered transport**, so an Epic B author gets the conformance check by
registering rather than by remembering. But "iterate whatever is registered" is not enough:
under a selective run or a per-file invocation the registering module may never be imported in
that process, and the meta-test would pass over a set of one while reporting success for a
category it never ran — the failure `--require-category` exists to prevent. So the meta-test iterates an explicit
`CONFORMANCE_CASES` table — one row per Epic B ticket, carrying the registering module, the
registry name, **the upstream format to construct with, and the inbound route that exercises
it** — imports every row's module, and asserts the registered set **equals** the expected one.

The last two fields are not decoration. The bridge *translates*, so the inbound route cannot be
derived from the upstream format: `BEDROCK_CONVERSE` (T-B3) and `OLLAMA_CHAT` (half of T-B1) have
no inbound route at all. And a single hardcoded format — the first draft — raises `ValueError` at
construction for **all three** Epic B transports, since none of them serves Anthropic Messages.
"Inherits the check by registering" was false for every author it was written for until the row
carried both. T-W8's structural guard is separate and pins what
*this module* registers, against a module-local constant and an AST scan, never against global
registry state.

**`responder` is a hook, and scripted replies are stateful closures.** Every consumer named for it
needs the reply to change between attempts — §6.3.1's four injection points, T-I8's
blip-then-success, and the `empty_replies` / `error_replies` counters `tests/bridge/` already
uses. `Responder`'s signature permits a closure over mutable state, and that is the intended
answer; no setter is added to a Protocol fifteen tickets consume.

**Timeouts bound the wait and must not destroy the diagnostic.** aiohttp's client default is
`total=300` seconds (confirmed on the pinned 3.13.5), so a hung binding costs five minutes per
test. `post()` defaults to **10 seconds** and converts a client timeout into an assertion naming
the transport, the elapsed time, the captures so far and the 30 s / 80 s reference values — a bare
`TimeoutError` keeps none of that. A test that means to observe the empty-response ladder
(§7.2.1) raises the timeout deliberately.

**Teardown releases the ports and never substitutes an exception.** Client sessions close before
`BridgeServer.stop_async`, which cleans up the runner but does not abort live connections — the
`wait_closed` split §7.3 handles for the proxy, and the reason the support matrix is
3.10–3.13. `assert_teardown_clean()` runs only on the clean path: raised from `__aexit__` over a
failing body it would replace the assertion the test was making, and the falsification cases
assert on message content. The conformance check does **not** call it as well: `__aexit__` runs
before the check returns either way, so a repeat inside the block would be an assertion no defect
could falsify — which is how the repeat was found and removed.

**`assert_teardown_clean()` is stated for every transport, not just this one.** Its obligation is
"every request I received was answered in the format I declare"; how a transport knows that is its
own business, since none of the other three can consult `RecordingUpstream`'s suffix table. The
aiohttp transport uses T-W4's `format_for_path` — the **pure** lookup, split out of
`_format_for` for this, because `_format_for` appends to `unmatched` and an assertion must not
mutate the evidence it judges.

**What T-W8 does not settle.** The one-upstream-request assertion here is scoped to the binding
the conformance check drives, because a check tolerating a retry ladder could not tell a working
binding from a broken one. **T-W9** still owns it as a product claim for one request driven end
to end, together with capture **type-compatibility with T-W2's declared contract**, the
`has_content` cell of §7.2.1's table, and the wall-clock bound on the empty ladder; T-W9 depends
on T-W1/T-W2/T-W4/T-W8 and no corpus task, so corpus-wide quantification is **T-D1's**, not its.
§5.3's non-loopback addressing stays with T-E1/T-E2 — loopback is correct for this fixture and is
not a containment decision. **T-K4** is not served by this fixture: §7.2.1's unbounded capture
retention is acceptable for one request per test and not for a sustained load run, which needs
its own bound.

**Existing tests are not migrated.** The 38 modules that build their own bridge stay as they are;
this fixture is for new work. Rewriting them is neither in T-W8 nor scheduled elsewhere, and
saying so here stops a future reader reading it as an unpaid debt.

[KBR-31]: https://shelpuk.atlassian.net/browse/KBR-31

---

## 8. CI cadence

One executable selection matrix. Every test carries exactly one layer marker
(`l1`, `l2`, `l3`, `acceptance`, `agent_smoke`, `agent_live`, `eval`, `load`), so a job is
defined by its marker expression and no test can fall between two jobs or into both.

| Job | Selection | Trigger | Gates a PR? | Gates a release? |
|---|---|---|---|---|
| **Fast** | `ruff`, `lint-imports`, `mypy src/kitty`, then `pytest -m "l1 or l2" -q` on Python 3.10–3.13 | push, PR | **Yes** | **Yes** |
| **Subsystem** | `pytest -m l3 -q` | PR | **Yes** | **Yes** |
| **Acceptance** | `pytest -m "acceptance or agent_smoke" -q` | PR | **Yes** | **Yes** |
| **Deep** | mutation testing (§6.1), schemathesis at high `--max-examples`, extended property runs | nightly | No | No |
| **Agent live** | `pytest -m agent_live -q`, credentials from CI secrets | nightly | No | No |
| **Eval** | answer-quality runner (§6.4.3) | nightly | No | **No** — alerts only |
| **Load** | load runner (§6.4.4) | pre-release | No | **Yes** |

**The release gate is the union of the four "gates a release" rows, and nothing else.** An
earlier draft said Release runs "everything above plus load" and then, a paragraph later, that a
release must not wait on an LLM eval. Both cannot hold. Evals and the live-agent job are
**alerting**, not gating: they are nondeterministic and depend on a third party's availability,
and a release that can be blocked by someone else's rate limiter is not a release process.

**`ratchet` exempts one named assertion, not a scenario.** A scenario-wide exemption is a
blanket amnesty: a broken fixture, a failure in a `Given` step, or an unrelated regression inside
that scenario all become invisible, indistinguishable from the known defect. That is a worse
outcome than the red gate it was meant to avoid.

The exemption is therefore narrow and accountable:

- It attaches to **one assertion**, named, with its expected failure condition and its issue key
  (TR-1c's header-subset assertion → G3's policy half, Q1 — **re-keyed from KBR-8 on 2026-09-11**
  when KBR-8 shipped without closing parity; see §6.4.1. TR-4's no-vendor-content assertion → KBR-5
  was the only other entry and was **withdrawn on 2026-09-07** when KBR-5 shipped, which leaves the
  registry one row long **in the state this document describes** — the length it is supposed to
  trend towards, and not a count of what is in the file today; see §8.3).
- **Setup and every other assertion in the scenario gate normally.** If TR-4 cannot reach the
  bridge, the job fails — that is not the known defect.
- **An unexpected pass fails the job.** When the assertion starts passing, the defect is fixed
  and the exemption must come off; `xfail(strict=True)` semantics, so nobody has to remember.
- All exemptions live in one registry with their issue keys, so the list is short, visible and
  obviously temporary rather than scattered through the suite.

A gate that is red for a known reason on the day it is introduced does not survive contact with a
release. A gate that is green because it stopped looking is worse.

**Skips are failures in a gating job.** If `agent_smoke` cannot find its pinned binary, or `l3`
cannot start its proxy, the job fails. A gating job that goes green because it ran nothing is the
most expensive kind of false confidence.

**One exception, stated so the rule is honest.** A **platform or interpreter** skip is permitted:
`tests/test_launcher_discovery.py` skips POSIX cases on Windows, and the matrix is what covers
them. A **resource-availability** skip is not. The suite had one — `tests/bridge/test_bridge_state.py`
called `pytest.skip("openssl not available")` inside a gating job, the exact shape this rule
forbids — filed as KBR-132 rather than quietly grandfathered, and **closed on 2026-09-11**.

**How it is upheld there now.** `tests/bridge/tls_certs.py` owns the certificate generation the
bridge TLS tests need, and every non-success exit from it goes through `pytest.fail`: a missing
binary, a non-zero exit, a timeout. `tests/bridge/test_tls_certs.py` is the check on that, and its
falsification case — restore the skip, the test must go **red** — is what makes it a detector
rather than a decoration. That case is written out in the module rather than left to a
`pytest.raises`, because `Failed` and `Skipped` are *sibling* classes: `pytest.raises(Failed)` does
not catch a `Skipped`, so the obvious spelling would have reported the reintroduced defect as a
*skipped* test, passing the gate. A detector that fails by skipping is the defect wearing the
uniform of its own guard.

**A consequence, stated because it is now load-bearing:** `openssl` is an **environment
prerequisite of the Fast job**, not something a test may probe for. Forbidding the
resource-availability skip and requiring the runner to provide the resource are the same statement.
`ubuntu-latest` ships it; a change of runner image, a container job, or a non-Ubuntu matrix entry
has to keep it, and the fix for a red gate is to install `openssl`, never to reinstate the skip.

**Why that is not handled by deselection, which is this document's other answer for a missing
resource.** §8.1's `RESOURCE_DEPENDENT_LAYERS` excludes a whole **layer** whose resource a
developer's machine may not have — a pinned agent binary, live credentials, a load rig — so a bare
`pytest` reports those as *deselected*, not skipped. That mechanism is layer-granular by
construction, and `openssl` is needed by *some tests inside* `l1` rather than by a layer — stated
as that property rather than as a count of modules, which is the kind of number that goes stale
silently. A resource used by part of a gating layer has only two possible treatments: skip it,
which §8 forbids, or require it.
The two rules are consistent, and the seam between them is worth naming because the obvious
reading of §8.1 suggests a third option that does not exist.

**The rule itself is still only prose.** Nothing checks that a *future* test does not do what
`test_bridge_state.py` did. The property holds today — no test in a gating layer calls
`pytest.skip`, and every `skipif` left there tests `sys.platform`, `os.name`, `sys.version_info` or
`hasattr(signal, ...)` — but that is an observation about the present, not a mechanism, and it is
the kind of observation that stops being true without anyone noticing. Filed as
[KBR-138](https://shelpuk.atlassian.net/browse/KBR-138), by the same reasoning that filed KBR-132:
the rule is only worth stating if the gap between it and its enforcement has an owner.

### 8.1 The selection mechanism

Delivered by T-W1, ahead of the jobs that use it, because parallel authors need the divided test
command from day one.

- **`tests/layers.py`** owns the vocabulary and every decision over it as a **pure function**:
  the path default, the exactly-one predicate, the required-category predicate, and which layers
  a marker expression positively selects. Pure so that each can be handed a deliberate defect —
  plan §1.4 makes that mandatory, and a decision entangled with a pytest hook cannot be given one.
- **`tests/conftest.py`** wires them with a single `wrapper=True`
  `pytest_collection_modifyitems`. Defaults are applied **before** pytest's `-m` deselection —
  otherwise `-m l1` deselects a suite whose files carry no markers — and the category check runs
  **after** it, or it counts tests the job will not run. A wrapper gets that ordering from the
  hook protocol rather than from plugin registration order.
- **`--require-category=NAME`**, repeatable, fails the run when a named layer collected nothing.
  This is the fix for the `or` problem above, and every job must pass one per layer its
  expression selects. A test enforces that pairing across every workflow.
- **`--layer-report=PATH`** dumps the collected items and their layers as JSON, which is how the
  whole-suite checks reason about labelling without re-deriving it.

**The default is by path, and the vocabulary is by marker.** Thousands of tests predate the scheme
and are not edited one at a time: `tests/integration/**` defaults to `agent_live`, everything else
to `l1`, and a file names its own layer only where that is wrong. An unrecognised path falls back
to `l1` — a new corner of the tree joining the fast gate uninvited is visible and cheap, whereas
one joining a nightly job is invisible until something ships broken.

**`--runslow` is gone.** It attached `pytest.mark.skip` to the 32 live-agent tests, so the gating
job collected them, skipped them, and reported green — the failure this section names, running in
production. They are `agent_live` now, and a bare `pytest` excludes them through an `addopts`
marker expression instead. The difference is not cosmetic: the run now reports *32 deselected*, a
statement about what was **selected**, where it used to report *32 skipped*, a statement about
tests that were supposed to run and did not.

**Two hand-maintained copies of one list is a defect in waiting**, so the `addopts` expression is
asserted equal to one derived from `RESOURCE_DEPENDENT_LAYERS` — the layers needing a resource CI
has and a developer's machine may not. `agent_smoke` is on that list before it has a single test,
because §6.4.2 launches a real pinned binary and the task that makes the category live should not
have to rediscover the rule.

**`--strict-markers` is passed on the command line, never through `addopts`.** pytest 9.0 silently
ignores it there; 9.1 honours it (upstream issue 14442). With `pytest>=8.0` and no upper bound, a
config-file placement would mean the gate behaves differently for two contributors looking at the
same tree, which is worse than not having it.

### 8.2 Activation is incremental, and the gap is on the record

The matrix above describes the finished state. Today only the Fast job exists, so `l3`,
`acceptance`, `agent_smoke`, `agent_live`, `eval` and `load` are selected by **no job at all**.

That is a real hole and it is the one this mechanism could most easily hide: before the split,
`pytest -q` ran everything, so a subsystem test written tomorrow ran in CI. After it, that test
runs nowhere — silently, because a job nobody has written cannot go red.

`PENDING_ACTIVATION_LAYERS` is the answer: a registry mapping each not-yet-run layer to the plan
task that activates it. It is checked in **both** directions, which is what stops it becoming a
standing amnesty:

- a layer holding tests that no job selects and that is **not** in the registry fails the suite;
- a layer in the registry that a job **does** now select also fails, so an entry cannot outlive
  its reason.

A consequence worth stating: **a test may not be moved to `l3` before the Subsystem job exists.**
Roughly six modules under `tests/` bind real sockets or spawn processes and are `l1` by default
today — `test_egress_https_proxy.py` foremost among them, and since T-W5 the shared fixture it was
extracted into plus `tests/harness/test_connect_proxy.py`, which must move **with** it: an
extraction and its own regression evidence landing in two different jobs would leave one proving
the other in a run that no longer includes it. Reclassifying them is correct and is T-K6's
business, together with the job that runs them; doing it earlier would remove them from every
gate. T-H1 must take that reclassification into account before it measures a mutation
baseline, because it selects on `l1`.

**Six modules are bulleted below, and `tests/cli/test_stream_encoding.py` (KBR-10) is described
after them — seven in all, named here so T-K6 inherits a list rather than a search** — the count
is what T-K6 and T-H1 plan against. (The bullet count and the KBR-10 paragraph were already
drifting apart before T-W8 added two; spelling out both is what stops the next addition
guessing which set it joins.)

- **KBR-132:** `tests/bridge/test_tls_certs.py` spawns a real `openssl` in one of its five cases.
  KBR-132 deliberately did **not** move it — the rule above applies to a test fixing a skip defect
  exactly as it applies to any other.
- **T-W4 (KBR-27):** `tests/harness/test_recorder.py` and
  `tests/harness/test_recorder_falsification.py` both bind real sockets.
  `tests/harness/test_recorder_conformance.py` is genuinely `l1` — its checks are pure functions
  over data and it opens nothing. The two socket-binding modules together run in **~1 second**,
  measured, which is the number the fast-gate budget should carry until T-K6 moves them.
- **T-W8 (KBR-31):** `tests/harness/test_bridge.py` and
  `tests/harness/test_bridge_falsification.py` each start a real `BridgeServer` **and** a
  recorder, most cases one of each. Together they run in **~1.6 seconds**, measured (1.1 s and
  0.5 s), which is the number the fast-gate budget should carry until T-K6 moves them. Worth
  knowing while planning that move: the timeout case cost **30 seconds** until its responder was
  released explicitly, because `stop_async` waits for in-flight upstream handlers rather than
  aborting them — the same property §7.3 handles deliberately for the proxy.
- **KBR-144:** `tests/bridge/test_responses_string_input.py` starts a real `BridgeServer` on an
  ephemeral port in four of its classes, following the existing convention of
  `tests/bridge/test_crash_resilience.py` rather than inventing a second one. The whole module
  runs in **~0.6 seconds**, measured, of which the socket-binding cases are ~0.1.

KBR-10 added the largest one: `tests/cli/test_stream_encoding.py` spawns **35 child interpreters**
per run, ×4 Python versions. It has no choice — the behaviour it proves is that kitty survives a
hostile *interpreter start-up encoding*, and `PYTHONIOENCODING` is read before any in-process test
exists, so a real child is the only oracle. Each spawn is short (the whole file runs in ~14s), but
T-H1 should note that mutation testing over `l1` will re-pay that cost per mutant, and may want to
deselect this file from the mutation baseline rather than from the gate.

**The load gate has to be wired, not merely declared.** The table above marks Load as gating a
release, but `publish.yml` currently depends only on the reusable `tests.yml`. Putting the load
run in "its own workflow" would leave publication free to proceed while load fails — or while it
never ran at all for that commit, which is the more likely failure. A row in a table is not a
dependency.

The arrangement, made explicit:

- `tests.yml` — reusable. Gains the Subsystem and Acceptance jobs. Called by `ci.yml` (push, PR)
  and by `publish.yml`.
- `load.yml` — **reusable**, and `publish.yml` calls it too, `needs:`-gated on the tag commit, so
  publication cannot complete without a successful load run **for that exact commit**. A
  scheduled nightly caller of the same reusable workflow gives early warning; it does not
  substitute for the release call, because a nightly result belongs to a different commit.
- Nightly Deep, Agent-live and Eval workflows stand alone and gate nothing.

That keeps the property the existing setup gets right — a release runs exactly the checks a PR
ran, from one definition — while extending it to the one gate that runs only at release time.

**A note on the existing runtime.** The suite already takes ~18.5 minutes per Python version, ×4
versions. The new L3 work adds real sockets to that gate. If the fast gate stops being fast,
people route around it, so the marker split above is also the mechanism for keeping the per-PR
path bounded — and the mutation-cadence measurement (Q11) has to be taken against that budget,
not against an empty one.

### 8.3 The exemption mechanism

Delivered by T-W7 (KBR-30), ahead of the guards that need it: T-G1, T-G4, T-G5, T-G9 and T-J2
each cover a class of check broader than the one defect they happen to expose, so each is
expected to land red on one assertion (plan §16). T-J1 is the blocked *dependency* rather than a
guard — it is the pytest-bdd wiring T-J2's scenarios are written against.

**`tests/exemptions.py`** holds the registry and the decisions over it, in the shape §8.1 uses
for the selection rules: every decision is a **pure function** — `outcome_for`,
`lookup_exemption`, `registry_violations`, `unexpected_pass_message` — so each can be handed a
deliberate defect, per plan §1.4. There is no pytest hook, no plugin and no new CI flag; a test
that uses an exemption is an ordinary test carrying its ordinary layer marker.

```python
# tests/exemptions.py — the one registry
EXEMPTIONS: Mapping[str, Exemption] = {
    "tr-1c-header-subset": Exemption(
        assertion="the header set is a subset of what Claude Code sends natively",
        condition="the bridge sends a User-Agent the agent does not",
        issue="KBR-8",
    ),
}
```

```python
# the guard. Imported as `exemptions`, not `tests.exemptions`: `tests/` has no
# `__init__.py`, so pytest's `prepend` import mode puts it on `sys.path` —
# `tests/layers.py` carries the same caveat in its module docstring.
from exemptions import ratchet

with ratchet("tr-1c-header-subset"):
    assert bridge_headers <= native_headers
```

**Two words spelled the same.** `ratchet(...)` here is the **exemption**: binary, gating, and it
fails when its assertion starts passing. A *"ratcheted baseline"* in §4.3 C1b, §6.3.1 and §6.4.4,
and in plan tasks T-I9 and T-I12, is a **different and non-gating** mechanism: record a number,
report it, tighten it monotonically. The collision predates this section — §8 above already calls
the exemption `@ratchet` — and the name is kept because §8 is the section the guards are written
against. T-I9 and T-I12 must not import this symbol; their plan rows say so.

**Ids are lowercase kebab-case**, and that much is checked, because five tasks add rows
independently. Prefixing an id with the guard or scenario it belongs to is a convention, not a
check — see the stated limits below. **One row per assertion, not per defect:** two guards over
KBR-8 take two rows, since each must be withdrawn on the day its own cell starts passing.

**Parametrised guards.** The blocked consumers are per-adapter (T-G9) and per adapter × model ×
transport (T-G4), and the defect is usually one cell of that matrix. Exempt the cell, not the
parametrisation:

```python
exemption = (
    ratchet("t-g9-openai-subscription-user-agent")
    if adapter == "openai_subscription"
    else nullcontext()
)
with exemption:
    assert headers(adapter) <= native_headers(adapter)
```

The other adapters gate normally; the exempt one still fails the job the day it starts passing.
The row's `condition` column names the cell in prose, because nothing else does. **In a
parametrised guard a cell is an assertion:** three exempt cells take three rows, so each can be
withdrawn on the day its own cell starts passing, and the count in the registry stays a count of
outstanding defects.

Six decisions in that mechanism depart from the obvious option, and are recorded because the
obvious option is what a future reader will otherwise assume was intended:

- **A context manager, not a decorator.** The paragraph above writes `@ratchet`, and a decorator
  is what pytest's own `xfail` would give. But a decorator can only wrap a **whole test**, which
  is exactly the scenario-wide amnesty this section rejects: it would hide a broken fixture and
  every healthy assertion beside the exempt one. Assertion scope needs a block. The name is kept.
- **Only `AssertionError` is amnestied; every other exception propagates.** An exemption says
  "this claim is known to be false", not "this region of the test may do anything". A `KeyError`
  inside the block is a broken fixture and fails the job. `pytest.fail()`, `pytest.xfail()`,
  `pytest.skip()` and **a missed `pytest.raises()`** are not `AssertionError`s and are therefore
  **not** amnestied — the safe direction, since an exemption that could turn a test into a skip is
  the failure §8 names two paragraphs above, but a surprise for the author of an L2 guard that
  reports its verdict with `pytest.fail` and a diff, which is the shape
  `tests/test_github_actions.py` uses today. **A guard that needs an exemption states its verdict
  as an `assert`**: catch with `try`/`except` and assert on what you caught, rather than leaning on
  `pytest.raises` or `pytest.fail`.
- **An unexpected pass raises a non-`AssertionError` exception.** `UnexpectedExemptionPass`
  descends from `Exception`, deliberately not from `AssertionError`, so that a nested or adjacent
  `ratchet` block can never swallow it. A strictness signal that another exemption can amnesty is
  not strictness.
- **The registry seam is private.** `ratchet` takes an underscored `_registry` keyword so that
  this mechanism's own tests can have rows while the production registry is empty. **A guard must
  never pass it.** A public `registry=` would be a documented back door into the one-registry
  invariant: a local amnesty nobody can count by reading one file.
- **No orphan-row scan.** A row whose assertion has since been deleted is stale documentation,
  but it grants amnesty to nothing and so cannot hide a failure; the mechanism against an
  exemption outliving its defect is the unexpected-pass rule. A text scan of `tests/**` for unused
  ids would buy tidiness at the price of a false positive on any id named in a comment.
  Considered and declined, not overlooked.
- **No terminal reporting of fired exemptions.** Visibility is the registry file, which is what
  §8 asks for — "one registry … short, visible". A per-run summary would be a `conftest.py` hook,
  and the value of this mechanism is precisely that it changes no job's collection or selection.
  The residual risk is real and accepted: a CI log gives no sign that an exemption fired.

**Two limits the mechanism cannot enforce, stated so they are not mistaken for guarantees.**

- **The block must hold exactly one assertion** and the statements that build its subject.
  Nothing detects a second: once the first assertion fails, the second is never evaluated, and a
  failure it would have reported is invisible — a miniature of the blanket amnesty §8 rejects.
  The enforcement is review, and the rule is in `.github/review/rules/python-tests.md`.
- **The `condition` column is documentation, not an assertion.** Any `AssertionError` from the
  block is amnestied, including one raised by an unrelated regression that happens to break the
  same assertion. The mitigation is the single-statement scope above, not the column.
- **The id prefix is a convention.** `registry_violations` checks the casing, not that an id
  names its guard or scenario, and it cannot check that a parametrised guard took one row per
  exempt cell. An allow-list of permitted prefixes would need editing for every new guard, which
  buys tidiness at the price of friction on the path this mechanism exists to keep open.

**The unexpected-pass rule only fires in a job that runs the test.** A row attached to a test on
a layer in `PENDING_ACTIVATION_LAYERS` (§8.2) — `l3` and `acceptance` today, which is where T-G4,
T-G5 and T-J2 land — is unchecked until that job is activated, so there the exemption *can*
outlive its defect. The task that activates the job owns re-checking the rows on its layer, in
the same way §8.2 makes each pending layer someone's named handoff.

**The registry ships empty, today.** The row this section names — TR-1c's header-subset
assertion, KBR-8 — belongs to an acceptance scenario that does not exist yet (§6.4.1, delivered
by T-J2 downstream of T-J1). A registry row for an assertion no test contains documents a
fiction, and the guard against a fiction cannot be the unexpected-pass rule, because nothing ever
runs it. Whichever of T-G1, T-G4, T-G5, T-G9 or T-J2 lands first adds the first row. §6.4.1 and
§6.2.3 are not amended to say so: this document states the To-Be state, in which those rows
exist, and §8's row-count parenthetical now says which state it is counting.

That makes the registry-shape check itself vulnerable to §8's own "green because it stopped
looking": a validator run over zero rows passes perfectly. So `registry_violations` is proved
against a **fabricated malformed registry** rather than against the production one, and the
production registry is asserted clean as a separate, weaker claim.

## 9. Gap register

### 9.1 What the current suite already does well

2,880 test functions across 139 Python files under `tests/`; ~50,700 lines of test to ~22,700
lines of source. Broad L1 coverage of translators, providers, profiles, credentials and the TUI.
Three assets stand out and are built on rather than replaced:

- `tests/test_egress_https_proxy.py` — real TLS handshakes through a local recording CONNECT
  proxy, across all three transport stacks. The strongest existing proof of anything in this
  document. Since T-W5 the proxy, the target and the certificates are shared from
  `tests/harness/connect_proxy.py`; the tests stayed here.
- `tests/test_egress_coverage.py` — AST/regex structural guard over `src/` that also asserts its
  own scan finds known positives, so it cannot rot into a no-op.
- `tests/test_github_actions.py` — treats the workflow definitions as testable artifacts.

Four Python versions in CI, with `mypy` and `import-linter` as gates rather than reports.

### 9.2 What is missing

**Status convention.** A closed gap keeps its row — the reasoning is still worth reading — with
**CLOSED** in the *Gap* column, *Today* in the past tense, and Priority `—`, so a scan by priority
does not surface work that is done. `TEST_SUITE_IMPLEMENTATION_PLAN.md` §16 mirrors it.

| ID | Gap | Today | Target | Priority |
|---|---|---|---|---|
| ~~**G14**~~ | ~~**F3 — the vendor name goes upstream in the body (M13)** — KBR-5~~ | **CLOSED 2026-09-07** | Post-condition raises; handlers render a downstream 400; defect-scoped source-literal guard (`tests/bridge/test_vendor_token_guard.py`) stands in until T-G5 | — |
| **G15** | **F4 — `_effort` / `_thinking_adaptive` reach the wire** — KBR-6 | Live I1+I2 breach on every CC-wire provider | Add both to `_INTERNAL_KEYS`; internal-key completeness guard (§6.2.3); regression test at `BridgeServer._upstream_body_for`. Residual: `openai_subscription` alone builds its body independently of that boundary (allowlisted, hence never leaked); `bedrock` and `ollama_cloud` call `translate_to_upstream` inside their transports, so the assertion reaches their wire. Carried by T-G2 over T-D4–T-D9's captures | **0** |
| **G16** | **F5 — `OpenCodeGoAdapter` misdeclared its wire shape** — KBR-7 · **CLOSED** | Was a latent defect in the M8 path and a trap for the oracle | Done: the declaration is per-model, both repair sites branch on it, and the **hook-level** honesty guard landed with the fix. The **wire-level** guard remains T-G4 / KBR-80 | — |
| **G24** | **No staleness alarm on the provider endpoint snapshot** | KBR-126 checked in `tests/data/opencode_go_endpoints.json` as the routing oracle. Nothing detects that the provider has since changed its table: §8's determinism rules exclude both mechanisms that could — a networked check and a clock. Refreshing it is a human act | Accepted trade-off, recorded rather than fixed: a networked alarm makes CI depend on a third party's uptime and turns green into a statement about today's weather. Revisit only if the provider publishes a machine-readable endpoint table — today's `/v1/models` carries ids only, no endpoints, and still lists retired aliases | **3** |
| **G23** | **`openai_subscription` injects `reasoning` from `_reasoning_effort`, unregistered** — KBR-149 | Three sites in `providers/openai_subscription.py` set `reasoning: {"effort": …}` from kitty's internal key. Structurally identical to P3 and P4, and **P4 cannot cover it**: §3.2.3 records that `translate_to_upstream` never runs on this adapter's request path. Unlike G22 this is a **request-body** row feeding §3.3.2 assertion 1, so the moment T-D5 drives a corpus entry carrying a reasoning effort the oracle reports a *false* I1 breach on a deliberate mutation — the under-claiming direction §3.3.1a calls unrecoverable | Add P22: trigger `REASONING_EFFORT_PRESENT`, conditional, anchored at `envelope.extra[reasoning]`. Needs a trigger case and a complement in the corpus. **Before T-D5** | **1** |
| **G22** | **Register header coverage is partial and inconsistent** — KBR-148 | Rows exist for four adapters (P9a ×3, P9b, P9c). At least six more deviate from the base header set with none: `AnthropicAdapter` and its three subclasses plus `ZaiAnthropicAdapter` (`x-api-key` / `anthropic-version` / lowercase `content-type`), `AzureOpenAIAdapter` (`api-key` on the non-Entra credential), and `OllamaAdapter`, which drops `Authorization` entirely — the same shape as P9b, which *does* have a row. `openai_subscription` additionally sets a conditional `ChatGPT-Account-Id` no row names | One row per deviation; `ChatGPT-Account-Id` becomes P9d, conditional, with a claimless-`id_token` fixture for its assertion-2 complement. Then §4.3 C1's per-adapter expectation is *reviewable against the register* instead of written from scratch — which is what stops C1 reproducing the ad-hockery F1 names | **2** |
| **G21** | **A declared trigger is never verified** — KBR-140 | §7.4 hands the oracle `triggers_met` as an argument and §3.3.2 asserts only that a row is **absent** when its trigger is not met. Nothing asserts a trigger declared met actually fired, so a corpus entry that over-declares makes assertion 1 claim every delta — the oracle reports green on a bridge that is rewriting messages. The same author writes the entry and its trigger index (T-W6), so the mechanism has no second reader | Roughly fifteen triggers are decidable from the inbound request; give those an optional predicate and have T-D8 require the declaration to agree with it. M6, M8, M9 and M12 depend on an upstream response and stay declaration-only — the stated residual risk. Blocked on T-A1/T-A2, since a predicate needs a projected request to read | **1** |
| ~~**G20**~~ | ~~**OpenCode Go's routing table does not match the provider** — KBR-126~~ | **CLOSED 2026-09-11.** Was: eight models served on `/v1/messages` against two routed there, four `/v1/responses` models with no route, and a `validation_model` that had left the catalogue — ten of twenty-eight models broken, presenting to the user as a false auth failure because the provider answers an unsupported model with `401` | Done: table refreshed against the provider's published list (re-verified live 2026-09-11); `/v1/responses` models route truthfully and are refused at serialization with `UnsupportedModelError` naming KBR-137, which owns the route itself; `validation_model` replaced and the whole class guarded registry-wide; snapshot oracle checked in. The balancing pool is explicitly protected from the refusal, mirroring the `CompactionFailedError` precedent | — |
| **G19** | Routing was outside the register and outside the oracle | The destination is built from the profile (M14, P20, P21); a body-only check cannot see a misrouted Azure deployment | §3.3.5 — whole-request oracle with an independently derived route | **1** |
| **G17** | Undecided behaviour for an irreducible final turn | Compaction emits an over-budget request, or (since KBR-5) the bridge refuses it downstream; neither was designed | Answer Q10, then align M3-M7, the 6.1 properties and TR-3 together | **2** |
| **G18** | P13-P19 - seven transport-level mutations, unregistered in the first draft | Necessary (the Codex backend and boto3 require them) but invisible above DEBUG, and unreachable by a guard placed at `translate_to_upstream` | Rows P13-P19; boundary corrected in 3.2.3; Q5 decides user visibility | **3** |
| **G21** | §8's skip rule is stated in prose and nothing checks it — KBR-138 | Found while closing KBR-132. The known breach is fixed, and every skip left in a *gating* layer is a platform or interpreter one — but that is an observation, not a mechanism, and the next resource-availability skip written into `l1`, `l2`, `l3` or `acceptance` re-creates the same silent-green defect | A check over the collected suite that fails on a resource-availability skip in a gating layer, with a planted skip as its falsification case (§1.4). It must be **layer-aware**: `tests/integration/test_agent_e2e.py` holds three legitimate resource skips (missing credentials, profile, agent binary) that are legal only because they sit in `agent_live`, so a flat grep would report them and be turned off. Two further questions: static sweep or runtime hook, and whether a permitted skip is recognised by condition shape or declared by marker | **3** |
| **G1** | I1 is unstated and untested | No definition of "unchanged"; mutation sites discoverable only by reading 6,463 lines | Register (§3.2) + oracle (§3.3) | **1** |
| **G2** | No-bypass unproven **for the bridge's serving path**; no negative assertion; start-path guard is file-granular | `test_egress_https_proxy.py` proves the transports and drives `egress_cmd._probe` | Sealed-network harness (§5.2) per transport (§5.5) + AST start-path guard | **1** |
| **G3** | I2 partially breached (F1) — KBR-8 · **fix landed, gap open** | Identity is still ad hoc per adapter. The subscription adapter no longer reports two different versions in one request: **KBR-8 fixed that on 2026-09-11**, and `tests/test_upstream_identity_consistency.py` guards both halves of §4.3 C1's F1 assertions across every registered adapter | Remaining: the exact-set header contract (T-G9 / **KBR-78**) and the parity baseline (T-C7, T-I12), then a policy — Q1 | **2** |
| **G4** | L1 strength unmeasured | Line coverage only | `mutmut` ≥ 85% **per target group** on the §6.1 scope | **2** |
| **G8** | No corpus of real agent traffic | Synthetic fixtures encode our assumptions | Golden corpus (§7.1) | **2** |
| **G10** | Custom-transport containment untested | **Partly closed (KBR-161).** Ambient `NO_PROXY` on `curl_cffi` is now measured, closed with `CURLOPT_NOPROXY`, and pinned; the OAuth refresh leg is covered by the same builder and contract. **Still open:** containment is proven at transport level, never through the bridge; `botocore`'s behaviour under an ambient `NO_PROXY` is **unmeasured** (its §6.2.4 row expects precedence, which is the assumption measurement falsified for `curl_cffi` — KBR-173); the aiohttp login leg is unproven end to end | §5.5 + §6.2.4 | **2** |
| **G5** | No contract layer | No published schema; SSE grammar unchecked | OpenAPI + `schemathesis` + grammar state machine | **3** |
| **G6** | Docs drift undetected (F2) — KBR-9 | README endpoint table already wrong | README ⇄ code guards | **3** |
| **G7** | No property-based tests | All example-based | `hypothesis` on the §6.1 list | **3** |
| **G9** | C5 unmeasured | `force_close=True` gives a per-request connection pattern unlike the agent's | Connection-count baseline | **3** |
| **G11** | Dependency behaviour unpinned; `curl_cffi` unbounded, **botocore undeclared** and the interpreter declared to the minor only | One of five §6.2.4 contracts has landed — the stdlib `ipaddress` one (KBR-146). The four transport contracts and the `botocore` declaration remain, so containment still rests on an undeclared transitive dependency | Dependency contract tests (§6.2.4) + declare botocore | **3** |
| **G25** | **§6.2.4 contracts are only ever evaluated on the newest patch of each minor** — KBR-146 | `tests.yml` names bare minor versions and `actions/setup-python` resolves each to the newest patch. Every dependency contract therefore proves forward drift only; a value that differs on an older patch a user runs — the shape KBR-146 had — is invisible to the gate. Today that half rests on one L1 test that forces the property both ways, which works because the surrounding behaviour was measured stable, and does not generalise to a contract whose neighbours have not been | One job pinned to the oldest supported patch (`setup-python` accepts an exact version, so it is one job, not four). Deferred as a CI-spend decision, not a technical one | **3** |
| **G12** | Product layer effectively absent | 2 E2E tests, never run in CI | Nightly job, extended to 5 Claude Code cases | **4** |
| **G13** | No answer-quality signal | Compaction and the Fireworks cap can degrade output invisibly | Paired delta eval | **4** |

**Order of work.** G14 and G15 are priority 0: they are live breaches of the product's stated
promise, both are small code fixes, and each has a cheap guard that stops it recurring. Then G1
and G2 — the two invariants with the least coverage, sharing §7.2's recording upstream as their
foundation. (G16 was originally scheduled alongside them; it is closed, and per §3.3.4 the
oracle's scoping never depended on it.) G8 unblocks G1's
corpus; G10 rides along with G2 once the harness is parametrised. G3's measurement lands with G1;
G3's *fix* is its own ticket. G4 and G7 reinforce an existing layer and can run in parallel. G5,
G6, G9, G11 are cheap and independent. G12 and G13 are last: most expensive to run, least caught
per hour.

---

## 10. Design rationale

Recorded per the repo's system-design discipline: the reasoning, especially where the choice was
not the obvious one.

**Pin what the code reads, not what it tolerates (§6.2.4).** A dependency contract that asserts a
version-dependent value enforces the author's interpreter rather than the product's requirement —
and, for KBR-146, would have enforced the defect. Where the dependency offers a stable neighbour the
answer is to read that instead; where it does not, the row records whether a floor or a runtime
check was chosen.

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

**The oracle is scoped by observed wire shape, not by the adapter's own property (§3.3.4).** On a
CC-wire provider every field differs and every delta is claimed by the translation row, so a
direct diff passes without proving anything. And the natural selector must not be trusted: it is a
claim by the code under test, and it is a boolean where §7.4 needs six projections. It was also in
fact wrong (`OpenCodeGoAdapter` declared a Messages wire while emitting Chat Completions for most
models — F5, since fixed), but the first two reasons stand without it. Selecting on the observed
shape keeps assertion 1 falsifiable regardless.

**Structural diff, except key order on the passthrough path (§3.3, §4.3 C2).** Byte-comparing
JSON fails on serialisation noise and trains people to ignore red. But key *order* is exactly what
a provider fingerprints, so where kitty claims to be forwarding rather than translating, order is
part of the contract. The asymmetry is deliberate — and it stops at the body: header order is
aiohttp's, not the agent's, so pinning it would pin a dependency's internals.

**A hostname outside the `localhost` family, never an IP literal, for the fake upstream (§5.3).**
`should_bypass` bypasses loopback, private and `localhost`-suffixed destinations, so the obvious
`127.0.0.1` harness proves the opposite of what it claims — silently. The premise is pinned by an
L1 property test, stated *with* the `localhost` exclusions so it does not fail on day one and get
weakened.

**Extend the existing CONNECT proxy rather than build one (§7.3).** `test_egress_https_proxy.py`
already had a recording proxy exercised across all three transport stacks. A second would
duplicate the hard part and risk the two drifting on exactly the behaviour they both exist to pin.
T-W5 carried this out: the proxy moved to `tests/harness/connect_proxy.py` and that module's five
tests kept passing against it, unchanged down to their collected node ids.

**Two L1 properties are stated with their exceptions (§6.1).** "Output ≤ budget" and "the last
turn survives" are both false as absolutes — the compactor breaks out while still over budget
when it cannot shrink further, and the last turn can be dropped outright (after which KBR-5
refuses the request downstream). Stating the honest version keeps
the properties enforceable; stating the clean version guarantees they get weakened by whoever is
on the rota.

**Mutation testing on a subset (§6.1).** Whole-codebase mutation testing on 22,700 lines produces
a survivor list nobody reads and a nightly job nobody waits for. The subset rule — modules whose
silent misbehaviour breaches an invariant — keeps the output actionable.

**A paired delta for answer quality, not an absolute threshold (§6.4.3).** Absolute thresholds
on LLM output are flaky, and a flaky gate at L4 trains people to re-run rather than investigate.
Pairing holds *task difficulty* constant, which removes the largest nuisance term. It does **not**
cancel the model's independent sampling on each call — which is why §6.4.3 specifies repetition,
a confidence interval and a pre-registered margin rather than a single delta. The eval is a
measurement, not a comparison of two numbers.

**Real-agent E2E stays out of the default run (§6.4.2).** It needs four CLIs, live credentials and
live network. A default suite that cannot run on a developer's laptop stops being run, and then
stops being trusted. Nightly with CI secrets is the right home.

**C1b and C5 start as reported baselines, not gates (§4.3).** Both differences are large today. A
gate that fails on day one gets disabled on day one. A ratcheted baseline makes the number visible
and stops it growing while the fix is done properly.

**TLS fingerprint parity is accepted residual risk, not omitted (§4.5).** Closing it means routing
every provider through `curl_cffi`, a large change to the serving path for a threat no provider is
currently known to apply here. Recording it means a future incident is a known gap, not a surprise.

**The oracle compares independent projections, never a translator round-trip (§3.3.1).** The
production translators are not inverses — one maps request to request, the other response to
response — so the round-trip an earlier draft specified could not have run at all. And even a
genuine inverse would only demonstrate self-consistency: a translator that drops a field in both
directions round-trips perfectly. Hand-written projections, importing nothing from the bridge,
are the only form of this check that can fail for the right reason.

**The vendor-token check is scoped to bridge-introduced content (§3.3.3, §4.3 C2).** A flat scan
of the serialized body for `kitty` puts I1 and I2 in direct conflict: a user asking Claude Code
to explain kitty-bridge would fail the I2 check, and satisfying it by stripping their words would
breach I1. Scoping by the projection diff dissolves the conflict — the user's sentence has an
inbound counterpart, M13's message did not.

KBR-5's own guard could not wait for the oracle, so it takes the other way out of the same
conflict: it scans **source literals** against an allowlist rather than serialized traffic. Agent
content is never inspected, so the conflict cannot arise — at the cost of catching only strings
that are literals in the bridge's own source, which is why it is a stand-in and not the answer.

**The register is enforced at the serialization boundary, not at `translate_to_upstream`
(§3.2.3).** On the subscription provider those hooks return a Chat Completions shape and
`_cc_to_responses` drops fourteen parameters afterwards, inside the custom transport. A guard
placed at the hook would have called that adapter clean — which is how P13 went unregistered in
the first draft.

**Containment gets a positive control and a falsification control (§5.2.2).** The `.invalid`
hostname that solves the loopback-bypass trap creates a second one: an unresolvable destination
makes "zero upstream connections" true for a broken implementation too. Proving direct
reachability first, and then proving the harness can detect a deliberately injected bypass, is
what turns a green result into evidence.

**Connections are joined to tunnels, not counted against requests (§5.2.1).** Connection reuse
puts many requests on one tunnel and a failed CONNECT puts none on any, so equal counts would
reject correct behaviour — and would quietly encode today's `force_close=True` into a test, which
Q7 may change.

**Two compaction claims and three Gherkin scenarios were corrected against the code (§6.1,
§6.4.1).** "Output ≤ budget except for an oversized system block", "a short turn is unchanged"
and "the latest turn survives in full" were all false. Where the correction exposed an undesigned
behaviour rather than a wording slip it became Q10, instead of being written up as intended
behaviour. A design document that ratifies whatever the code happens to do is not a specification.

**Mutation scope now contains the code its own rationale is about (§6.1).** The subset was
justified by "a mutation surviving in the compactor", and then excluded `server.py`, where the
compactor lives. Function-level wildcards bring it in without dragging in the retry state
machine, and per-component thresholds stop a weak invariant-critical component hiding behind a
strong one in an aggregate.

**Evals are specified as a measurement, not a comparison (§6.4.3).** Pairing removes task
variance; it does not remove the model's independent sampling variance, so a single paired run
and a single delta cannot support a decision. Repetition, a pinned configuration, a confidence
interval and a pre-registered margin are the minimum that makes a nightly signal actionable —
and independently authored acceptance tests, because a model-generated test passing the model's
own code shares the model's misunderstandings.

**Load has numbers or it is not a gate (§6.4.4).** "Behave as intended" is the "as appropriate"
this document forbids elsewhere. The ceilings come from a recorded baseline run and are then
ratcheted; streaming and buffered paths are measured separately, because the blanket claim that
memory does not grow is false on the paths that buffer a whole response.

---

## 11. Open questions for the product owner

Answers belong in this document. They are not invented here. Q10-Q14 are prerequisites for the
implementation work they name — each blocks a test whose acceptance oracle depends on it.

**Q1 — How faithful should the agent's identity be (F1, G3, KBR-8)?** Three options, materially
different: (a) forward a curated allowlist of the agent's real headers, uniformly, so every
provider sees a genuine coding agent; (b) send a neutral, stable identity that is neither kitty
nor Claude Code; (c) leave it ad hoc and treat I2 as "no kitty fingerprint" rather than "looks
like the agent." Option (a) is the literal reading of KBR-2 but carries provider-compatibility
risk (some providers reject unknown `anthropic-beta` values) and would replace three working
hard-coded strings with a general mechanism. Whichever is chosen, the current state is not a
policy — it is four independent workarounds, one of which reports two different client versions
in the same request. This decides both the I2 target and the C1b gate.

**Q2 — Is pre-flight credential validation an acceptable I2 exception?** It is an upstream
request the agent never made, and `--no-validate` already exists to suppress it. Declared
exception, or suppressed by default when egress is configured?

**Q3 — Should `should_bypass` resolve hostnames (§5.3)?** Not resolving saves a DNS round trip
per request but means a LAN-hostname-configured local model server gets tunnelled to a proxy that
cannot reach it. `localhost` is already handled by name. Keep the trade-off, or resolve-and-cache?

**Q4 — What regression margin for answer quality (§6.4.3)?** A pre-registered number, chosen
before the data. This is one input to the eval design, not a substitute for it — Q13 and the
repetition and interval choices in §6.4.3 have to be settled alongside it.

**Q5 — Should silently dropped or overridden request parameters be visible to the user (P5c, P7, P13–P19)?**
Fireworks caps `max_tokens` down, Anthropic raises it up, the Codex backend drops fourteen
parameters including `max_tokens` and `temperature` outright and strips `strict` from every tool
declaration, and both custom transports overwrite `stream`. Every one is necessary — the
upstream rejects the request otherwise — and every one means a setting the agent sent silently
does nothing. Today each logs at DEBUG. A one-line warning at launch would make it visible at
the cost of noise.

**Q6 — Which of the four body-changing retry paths are acceptable (§4.3 C3)?** M6, M8, M9 and
failover re-normalisation each send a different payload on a later attempt, and a provider that
hashes bodies sees all four. The alternative in each case is to fail the turn, which is worse for
the user. Declare all four as exceptions, or close some?

**Q7 — Should `force_close=True` stay (C5, §4.5)?** It prevents port exhaustion but produces a
connection pattern unlike the agent's own. Keep, or trade for keep-alive parity? §5.2.1 is
deliberately written not to depend on the answer.

**Q8 — What is the repo's policy on tracking `.system_design/`?** `.gitignore` ignored both
`.system_design/` and `.requirements/`. KBR-2 asks for this document at a tracked path *and* for
a PR, which cannot both hold while the directory is ignored; this change therefore un-ignores
`.system_design/` and leaves `.requirements/` ignored as per-task working material. If that is
wrong, say so — and note the repo has no `SYSTEM_DESIGN.md` at all, so the request path, provider
registry and failover state machine are undocumented. A separate ticket seems right.

**Q9 — ANSWERED by the product owner, 2026-09-07.** Return the error **downstream only**, never as
a synthetic upstream turn — the third option below — and the downstream message **names the
product**, since it never leaves the user's machine and naming the component tells them which part
of their stack is speaking. Implemented in KBR-5: the post-condition raises `CompactionFailedError`
and each handler renders a protocol-native 400. The message text was also rewritten, because the
original blamed the system prompt and prescribed `/clear`, and F3 establishes that neither is
right. *Original question:* what should the unrecoverable-compaction message say (F3, G14,
KBR-5)? It must stay
legible to the user and stop naming the product upstream. Options: a vendor-neutral string;
routing the error to the agent as an HTTP error instead of a synthetic assistant turn; or keeping
the text but returning it downstream only. The third is probably right — the message is for the
user, and it currently reaches the one audience it was never meant for.

**Q10 — What should happen when the final turn alone exceeds the budget (§6.1, TR-3)?** Today
compaction gives up and emits an over-budget request, which the provider rejects, or — since
KBR-5 — the bridge refuses it downstream with a 400. Neither was designed; both are what the loop happens to do when it can
shrink no further. Options: truncate the final turn's content, fail fast with a clear local
error, or keep current behaviour and document it deliberately. **Whatever is chosen, register
rows M3–M7, the §6.1 compaction properties and TR-3 move together — M13 has since been
withdrawn by KBR-5, which narrows this question rather than answering it.** The design records
current behaviour as observed, explicitly not as approved, until this is answered.

**Q11 — Does changed-code mutation testing fit the per-PR gate (§6.1)?** Answerable by
measurement, not opinion: time `mutmut run` restricted to the functions a representative PR
touches, against the budget the fast gate can absorb given it already runs ~18.5 minutes per
Python version. Adopt per-PR if it fits, nightly-only otherwise. The current nightly-only choice
is provisional pending that number.

**Q12 — How is a pinned Claude Code binary supplied to CI (§6.4.2)?** The per-PR agent smoke
needs a real Claude Code, not an arbitrary child process, because the claim under test is Claude
Code's own settings precedence. Which distribution, pinned how, and is redistribution inside a CI
image acceptable? If it is not, the settings-precedence claim has no per-PR proof, and that
limitation should be stated rather than papered over.

**Q13 — What is the baseline for the compaction arm of the evals (§6.4.3)?** For an over-context
input the direct-provider arm returns a 400, so there is no answer to compare against.
Candidates: kitty against a larger-context model, or kitty with compaction relaxed. The choice
determines what a regression in that arm actually means.

**Q14 — What is a correct stream recovery after bytes have reached the client (§6.3.1)?** Failover
before the first downstream byte is unambiguous. After text has been emitted, or mid tool-call
arguments, there is no obvious right answer: abandon the partial block and re-open under a new
id, fail the turn and let the agent retry, or something else. Until this is decided the L3 row
can assert only the negatives — no duplicated text, no reused tool-call id across attempts, no
spliced arguments — which catches corruption but cannot confirm correct behaviour. This is the
one place in the design where a test is specified without a full acceptance oracle, and it is
recorded here rather than papered over.
