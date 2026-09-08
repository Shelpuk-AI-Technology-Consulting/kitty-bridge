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

Eleven request-path rows (M1–M11) plus one response-path row (M12). The former substitution row
M13 is **withdrawn** — KBR-5 replaced it with a downstream error, so it mutates nothing.

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
| ~~M13~~ | **Withdrawn — no longer a mutation.** Was: discard the conversation and substitute a `[Kitty Bridge: …]` user message. | `_compact_messages` / `_apply_compaction` post-condition | No non-system message survives | **Closed by KBR-5.** The post-condition now raises `CompactionFailedError` and the handler returns a protocol-native 400 downstream; nothing is substituted, so there is no mutation left to register. The row is kept struck through rather than deleted so a reader of finding F3 can still find it. **The trigger recorded here was wrong** — see F3. |
| M14 | **Replace the destination entirely** — scheme, host and path are built from the profile by `build_base_url()` + `get_upstream_path()` | `BridgeServer._build_upstream_url` | Always | The agent addressed a loopback bridge; the request has to reach the real provider. Listed because **the destination is a mutation surface the body cannot show**: on Azure an identical body sent to the wrong deployment path is a different request entirely (§3.3.5). |

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
| P20 | **Encode the profile's model as the deployment id in the URL path** | `AzureOpenAIAdapter.get_upstream_path` | Always | The counterpart of P6: what P6 removes from the body reappears in the path. A register that records only P6 makes the model look *dropped* when it was *moved*, and leaves the move unchecked. |
| P21 | Encode `project_id` and `location` in the base URL | `VertexAIAdapter.build_base_url` | Always | Vertex addresses a project-scoped endpoint. Same class as P20: routing carried outside the body. |
| P7 | **Cap the agent's `max_tokens` at 4096** | `FireworksAdapter.normalize_request` | Non-streaming request with `max_tokens > 4096` | Fireworks rejects non-streaming requests above 4096. **User-visible** as shortened output. |
| P8 | Inject empty `reasoning_content` into assistant messages | `ProviderAdapter._inject_empty_reasoning_content`, called from `KimiCodeAdapter`, `_ZaiBase`, `CustomOpenAIAdapter` | Thinking signalled **or** inferred from prior `reasoning_content` via `_detect_thinking_from_messages` | Those providers reject the request without it. The *inferred* trigger matters: it fires with no signal from the agent at all. |
| P9a | Set `User-Agent` to `claude-code/1.0` | `KimiCodeAdapter`, `BytePlusAdapter`, `MimoAdapter` `.build_upstream_headers` | Always, on those three | Those providers 403 without a recognised coding-agent user-agent. Central to I2 — F1. |
| P9b | Remove `Authorization`, add `api-key` | `MimoAdapter.build_upstream_headers` | Always | MiMo does not use Bearer auth. An auth-**scheme** change §4.3 C1's exact-set assertion must encode. |
| P9c | Synthesise a Codex CLI `User-Agent` and a `version` header | `OpenAISubscriptionAdapter` | Always | Impersonation required by the subscription endpoint. **The two disagree — see F1.** |
| P10 | Set `reasoning_split = True` | `MiniMaxAdapter.normalize_request` | **Unconditionally** | Makes MiniMax return thinking in `reasoning_details` instead of inline tags. Unconditional, so exempt from §3.3.2 assertion 2. |
| P11 | Translate CC → Bedrock Converse | `BedrockAdapter.translate_to_upstream` | Always, on `bedrock` | A third upstream wire format M2 does not name. Custom transport — see §3.3.4. |
| P12 | Translate CC → Ollama `/api/chat` | `OllamaCloudAdapter.translate_to_upstream` | Always, on `ollama_cloud` | A fourth wire format. Custom transport. |
| P13 | **Drop fourteen Chat-Completions-only parameters** — `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `response_format`, `stop`, `n`, `stream_options`, `seed`, `logit_bias` | `OpenAISubscriptionAdapter._cc_to_responses` | Always, on the **CC-origin** path | The Codex backend applies strict allowlist validation and rejects them with 400. **User-visible**: `max_tokens` and `temperature` silently do nothing on this provider. Logged at DEBUG. |
| P14 | **Drop every parameter outside the Codex allowlist**, notably `max_output_tokens` | `OpenAISubscriptionAdapter._prepare_responses_body` | Always, on the **Responses-origin** path | Same backend restriction, different input shape — this path receives a Responses body, so the parameter is spelled `max_output_tokens`, not `max_tokens`. A single row cannot cover both paths; the sets differ. |
| P15 | **Strip `strict` from every tool declaration** | `_prepare_responses_body` | Always, on the Responses-origin path | The Codex backend rejects it. A change to the **tool schema** the agent declared, not to a sampling parameter — a different kind of fidelity mutation and worth its own row. |
| P16 | Rewrite content types `input_text` → `output_text` | `_convert_content_types`, called from `_prepare_responses_body` | Always, on the Responses-origin path | The Codex backend validates content types strictly. **Message-content mutation.** |
| P17 | Inject `stream: True` and `store: False` | `_cc_to_responses` and the Responses-origin body builder | **Unconditionally**, both subscription paths | The Codex backend is streaming-only; kitty reassembles a non-streaming reply from the SSE. Note `stream: True` **overrides a non-streaming client request** — the subscription-path analogue of M11. |
| P18 | Remove `modelId` and `stream` from the Converse payload | ``BedrockAdapter` transport (`make_request` / `stream_request`)` | Always, on `bedrock` | boto3 takes the model id as a call argument and selects streaming by choosing `converse` vs `converse_stream`, so both must leave the body. Applied **in the transport, after `translate_to_upstream`**. |
| P19 | Overwrite `stream` | `OllamaCloudAdapter` transport (`make_request` sets `False`, `stream_request` sets `True`) | Always, on `ollama_cloud` | The transport, not the caller, decides which Ollama endpoint mode is used. Applied **after `translate_to_upstream`** has already set it from the request. |

**Conditional rows are the point.** Every row whose trigger is a condition must be provably
*inert* when that condition is absent — the sharpest form of "unless absolutely necessary", and
what §3.3.2 assertion 2 tests, with the trigger complements §3.3.4 requires. M1, M2, M10, P1,
P6, P9a–c, P10, P11, P12, P13, P14, P15, P16, P17, P18 and P19 are unconditional by design and
are exempt from that assertion.

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

**Optional ids, because two formats have none.** Gemini's `functionCall`/`functionResponse` carry
no id; pairing there is by tool name and the k-th unanswered call of that name in the most recent
assistant turn. A required id would force those readers to synthesise one and show a delta on every
tool turn.

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
(T-W3). Neither can define it without the other agreeing.

| Path form | Names |
|---|---|
| `envelope.model` · `envelope.stream` · `envelope.store` | The named control fields |
| `envelope.extra[<wire key>]` | A format-specific control field — P2a `thinking`, P3 `reasoning`, P4 `reasoning_effort`, P10 `reasoning_split` |
| `conversation.system[<i>]` | One system text part |
| `conversation.turns[<i>].role` · `.parts[<j>]` | A turn, or one part of it |
| `conversation.tools[<name>].description` · `.schema` · `.strict` | A tool declaration, **by name** |
| `conversation.sampling[<key>]` | One sampling parameter |
| `conversation.turns` · `.system` · `.tools` · `.sampling` | A **whole collection** — M5 and M13 rewrite the turns, P5b joins the system blocks, §3.3.1 pins P13/P14 to the bare `sampling` |
| `headers[<name>]` | A header — P9a, P9b, P9c, and §4.3 C1 |
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

`[*]` is the wildcard. `[]` is accepted as its **legacy spelling**, because §3.3.1 wrote P15 as
`conversation.tools[].strict` before this vocabulary existed and a row carried over in the old
notation must not silently match nothing. An unbalanced bracket **raises** rather than mis-splitting
the path.

**`not projectable` is a legal value for the register's field column, and it requires a reason.**
P16 uses it — the `input_text`/`output_text` tag is redundant with the turn's role, so carrying it
would put one vendor's spelling into a wire-independent form — as do the whole-body protocol
translations M2, M9, P11 and P12. An empty cell would leave those rows silently unfalsifiable;
an explicit value with a reason does not.

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
| Azure | The deployment id, which **is** the profile's model (P20) — and P6 deliberately removes `model` from the body | Two requests to two different deployments have **byte-identical bodies**. A body-only oracle cannot tell them apart, so a misrouted request is invisible. |
| Vertex | `project_id` and `location` (P21) | The account being billed is a URL component |
| Gemini | Model and operation (`:generateContent` vs `:streamGenerateContent`) in the inbound path | M10 lifts the model into the body precisely because it is not there to begin with |

So the oracle takes the **whole captured request** — method, scheme, host, path, query, headers,
body — and asserts routing separately from content.

**The routing expectation is derived independently.** It is computed in the test from the
configured profile — provider, model, `provider_config` — using the provider's *published* URL
shape, not by calling `build_base_url()` / `get_upstream_path()`. Asking the code under test where
it meant to go and then checking it went there proves nothing; this is the same independent-oracle
rule §3.3.1 applies to bodies.

**Falsification control.** Alongside the five body cases in §3.3.1, a sixth: change the Azure
deployment segment in the captured path while leaving the body byte-identical. The oracle must
fail. Without this case there is no evidence the routing assertion is wired to anything.

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

- **F1 — Agent identity is handled per-provider, not by policy.** *(KBR-8.)* Upstream headers are built from
  scratch, so Claude Code's `user-agent`, `x-app`, `anthropic-beta` and `x-stainless-*` never
  reach the provider. Four adapters compensate ad hoc (P9a, P9c): `KimiCodeAdapter`, `BytePlusAdapter`
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

**The join.** `ConnectAttempt` records target and authentication status only, which is not enough
to identify a connection at both ends. Extend it to record the **proxy's outbound source port**
for each tunnel it opens; the recording upstream already records the peer port of each accepted
connection. Joining on that port identifies each upstream connection with the tunnel that
created it. An upstream connection with no matching tunnel port is a bypass, and it is the only
thing this assertion needs to catch.

(Peer *address* cannot do this job: with bridge, proxy and upstream on loopback in one process,
proxied and direct connections both present `127.0.0.1`.)

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
| `openai_subscription` — serving | `curl_cffi.AsyncSession` | `proxies=egress.proxies_dict()` |
| `openai_subscription` — OAuth token legs | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` |
| `bedrock` | boto3 / botocore | `BotoConfig(proxies=egress.proxies_dict())` |
| `ollama_cloud` | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` |

Three consequences the rest of §5 must not paper over:

1. **The local-bypass assertion in §5.2.2 is false for these paths.** They have no bypass, so a
   loopback or private destination *is* tunnelled. Arguably safer, but different — the design
   must say so rather than imply uniformity.
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
| `should_bypass` | Every address in a private range is bypassed · the §5.3 hostname property |
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
| **Wire-shape honesty** | For every adapter × representative model, assert the adapter's declared wire shape agrees with the shape the body is actually written in. Catches F5 (KBR-7). **Two boundaries, two owners.** The *hook* form — observing `translate_to_upstream`'s return value — landed with **KBR-7** as `tests/test_wire_shape_honesty.py`, together with the fix: it asserts the per-model declaration `upstream_wire_is_messages_api_for_model(model)` against the emitted body, and the bare property against an explicitly declared default-route model. It guards itself so it cannot rot — the classifier is pinned against known Messages, Chat Completions and Converse bodies (including a Converse body with no tools, since the tools axis is what separates Converse from Messages); every registry key must be represented; every route of a model-routing adapter must be represented; and the custom-transport set is asserted rather than narrated. The *wire* form — observing the body at the §3.2.3 boundary — is **T-G4 / KBR-80** and is **not** delivered: for the three `use_custom_transport` adapters nothing in the hook-level guard observes the bytes that ship. On `openai_subscription` `translate_to_upstream` is never invoked on the request path at all — `_cc_to_responses` builds the Responses body inside the transport (P13–P17) — so there the exemption is load-bearing. On `bedrock` and `ollama_cloud` the transport mutates the hook's body afterwards (P18, P19); neither mutation changes the body's *shape family*, so for those two the exemption is precautionary — a guard must observe the shipped bytes, not infer them. The hook form also does not cover adapters constructed with `provider_config`, native-passthrough requests, whether a non-Messages body is *well-formed* (the declaration is a boolean, so Chat Completions and "neither" are collapsed) — **T-G4 inherits that one**, because it is a property of the declaration and not of the boundary — or whether the routing table matches the provider's published endpoint table (**KBR-126**). The declaration stays boolean deliberately: its consumer is binary (`_repair_thinking_roundtrip` picks between exactly two carriers), so widening it to an enum would change the repair's contract rather than this declaration's. If a routing adapter ever gains a third wire, the boolean must be **replaced**, not extended — a `False` meaning "Responses" would be F5 again in a new costume. |
| **Bridge-introduced vendor token** | No content the bridge *introduces* into a request body or header contains `kitty` in any casing. Scoped by the projection diff (§3.3.3), never a flat scan of the serialized body — a flat scan would fail on a user legitimately writing the word, and "fixing" that would breach I1. Caught F3 (KBR-5). **F3 is now fixed, so this guard has no live positive fixture left**: its positive control is the synthetic historical M13 string held in `tests/bridge/test_vendor_token_guard.py`, which T-G5 inherits. That file is also the defect-scoped stand-in until T-G5 lands — it scans **source literals** against an allowlist, never traffic, so it does not fall into the flat-scan trap this row warns about. |
| **Start-path domination** | Every `BridgeServer(` construction is dominated by an `egress_block_reason(` call **at AST level**, not merely co-located in the same file. `cli/main.py` already holds two of the five start paths (§5.1 gap 3). **Necessary but not sufficient — see below.** |
| **Env-var register** | `_SETTINGS_ENV_OVERRIDE_KEYS` and `_CONFLICTING_ENV_VARS` (`launchers/claude.py`) match what `build_spawn_config` emits and what the README documents. |
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

  # One exempt assertion: header-subset, pending KBR-8. See the exemption registry in 8.
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
gates normally. TR-1c's remains, pending KBR-8.

The exemption covers **one assertion**, never the scenario. In TR-1c it is the header-subset
assertion (KBR-8). Every other step in
those scenarios — setup, the `Given` clauses, and any other `Then` — gates normally, so a broken
fixture or an unrelated regression still fails the build. An unexpected pass also fails, forcing
the exemption off when the defect closes. The full policy and the exemption registry are in §8;
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
baseline run, recorded and then ratcheted. **`_backend_context` isolation moved to L3** (§6.3.1)
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

### 7.3 Recording CONNECT proxy

**Already exists.** `tests/test_egress_https_proxy.py` contains `_ConnectProxy` (enforces Basic
auth, records every `CONNECT` as a `ConnectAttempt`) and `_TlsTarget`, with throwaway
certificates on ephemeral ports. Extend it rather than build a second:

- expose it as a shared fixture;
- add the ability to stop it mid-test, for §5.2.2 phase 2;
- **record the outbound source port** of each tunnel it opens, so §5.2.1 can join tunnels to the
  connections the recorders accept — `ConnectAttempt` carries target and auth status only today,
  which is not enough to identify a connection at both ends;
- resolve the harness hostname itself on the proxied leg, and expose a per-transport direct-route
  override for §5.2.2 phase 1.

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

**`@ratchet` exempts one named assertion, not a scenario.** A scenario-wide exemption is a
blanket amnesty: a broken fixture, a failure in a `Given` step, or an unrelated regression inside
that scenario all become invisible, indistinguishable from the known defect. That is a worse
outcome than the red gate it was meant to avoid.

The exemption is therefore narrow and accountable:

- It attaches to **one assertion**, named, with its expected failure condition and its issue key
  (TR-1c's header-subset assertion → KBR-8. TR-4's no-vendor-content assertion → KBR-5 was the
  only other entry and was **withdrawn on 2026-09-07** when KBR-5 shipped; the registry is now one
  row long, which is the length it is supposed to trend towards).
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
them. A **resource-availability** skip is not, and the suite has one today —
`tests/bridge/test_bridge_state.py` calls `pytest.skip("openssl not available")` inside a gating
job, which is the exact shape this rule forbids. Filed as KBR-132 rather than quietly
grandfathered — the rule is only worth stating if its one known breach has an owner.

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
today — `test_egress_https_proxy.py` foremost among them. Reclassifying them is correct and is
T-K6's business, together with the job that runs them; doing it earlier would remove them from
every gate. T-H1 must take that reclassification into account before it measures a mutation
baseline, because it selects on `l1`.

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

**Status convention.** A closed gap keeps its row — the reasoning is still worth reading — with
**CLOSED** in the *Gap* column, *Today* in the past tense, and Priority `—`, so a scan by priority
does not surface work that is done. `TEST_SUITE_IMPLEMENTATION_PLAN.md` §16 mirrors it.

| ID | Gap | Today | Target | Priority |
|---|---|---|---|---|
| ~~**G14**~~ | ~~**F3 — the vendor name goes upstream in the body (M13)** — KBR-5~~ | **CLOSED 2026-09-07** | Post-condition raises; handlers render a downstream 400; defect-scoped source-literal guard (`tests/bridge/test_vendor_token_guard.py`) stands in until T-G5 | — |
| **G15** | **F4 — `_effort` / `_thinking_adaptive` reach the wire** — KBR-6 | Live I1+I2 breach on every CC-wire provider | Add both to `_INTERNAL_KEYS`; internal-key completeness guard (§6.2.3); regression test at `BridgeServer._upstream_body_for`. Residual: `openai_subscription` alone builds its body independently of that boundary (allowlisted, hence never leaked); `bedrock` and `ollama_cloud` call `translate_to_upstream` inside their transports, so the assertion reaches their wire. Carried by T-G2 over T-D4–T-D9's captures | **0** |
| **G16** | **F5 — `OpenCodeGoAdapter` misdeclared its wire shape** — KBR-7 · **CLOSED** | Was a latent defect in the M8 path and a trap for the oracle | Done: the declaration is per-model, both repair sites branch on it, and the **hook-level** honesty guard landed with the fix. The **wire-level** guard remains T-G4 / KBR-80 | — |
| **G20** | **OpenCode Go's routing table does not match the provider** — KBR-126 | Found while fixing G16. As of 2026-09-07 (<https://opencode.ai/docs/go/>, "Endpoints") the provider serves eight models on `/v1/messages`; `_MESSAGES_MODELS` holds two, and a `/v1/responses` endpoint (four models) has no route at all. The wire-shape guard correctly reports the adapter *honest* — declaration and emitted body agree — because this is routing, not shape | Refresh the table against the provider's endpoint list; decide the Responses route; replace the stale `validation_model`. Consider a checked-in snapshot of the endpoint table, so the routing question gets an in-repo oracle | **1** |
| **G19** | Routing was outside the register and outside the oracle | The destination is built from the profile (M14, P20, P21); a body-only check cannot see a misrouted Azure deployment | §3.3.5 — whole-request oracle with an independently derived route | **1** |
| **G17** | Undecided behaviour for an irreducible final turn | Compaction emits an over-budget request, or (since KBR-5) the bridge refuses it downstream; neither was designed | Answer Q10, then align M3-M7, the 6.1 properties and TR-3 together | **2** |
| **G18** | P13-P19 - seven transport-level mutations, unregistered in the first draft | Necessary (the Codex backend and boto3 require them) but invisible above DEBUG, and unreachable by a guard placed at `translate_to_upstream` | Rows P13-P19; boundary corrected in 3.2.3; Q5 decides user visibility | **3** |
| **G1** | I1 is unstated and untested | No definition of "unchanged"; mutation sites discoverable only by reading 6,463 lines | Register (§3.2) + oracle (§3.3) | **1** |
| **G2** | No-bypass unproven **for the bridge's serving path**; no negative assertion; start-path guard is file-granular | `test_egress_https_proxy.py` proves the transports and drives `egress_cmd._probe` | Sealed-network harness (§5.2) per transport (§5.5) + AST start-path guard | **1** |
| **G3** | I2 partially breached (F1) — KBR-8 | Identity ad hoc per adapter; the subscription adapter reports two different versions in one request | Header contract + parity baseline, then a policy and a code fix | **2** |
| **G4** | L1 strength unmeasured | Line coverage only | `mutmut` ≥ 85% **per target group** on the §6.1 scope | **2** |
| **G8** | No corpus of real agent traffic | Synthetic fixtures encode our assumptions | Golden corpus (§7.1) | **2** |
| **G10** | Custom-transport containment untested | Proven at transport level, never through the bridge; ambient `HTTP_PROXY`/`NO_PROXY` untested; the OAuth leg untested | §5.5 + §6.2.4 | **2** |
| **G5** | No contract layer | No published schema; SSE grammar unchecked | OpenAPI + `schemathesis` + grammar state machine | **3** |
| **G6** | Docs drift undetected (F2) — KBR-9 | README endpoint table already wrong | README ⇄ code guards | **3** |
| **G7** | No property-based tests | All example-based | `hypothesis` on the §6.1 list | **3** |
| **G9** | C5 unmeasured | `force_close=True` gives a per-request connection pattern unlike the agent's | Connection-count baseline | **3** |
| **G11** | Dependency behaviour unpinned; `curl_cffi` unbounded and **botocore undeclared** | Containment rests on an undeclared transitive dependency | Dependency contract tests (§6.2.4) + declare botocore | **3** |
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
already owns a recording proxy across all three transport stacks. A second would duplicate the
hard part and risk the two drifting on exactly the behaviour they both exist to pin.

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
