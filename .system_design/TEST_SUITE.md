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

Twenty-two request-path rows — the original fourteen (M1–M11, M15, M16 and M17), two
id-synthesis rows added by KBR-195 (M18, M19), and six KBR-194 Gemini-side slot drops on
the Gemini inbound route (M20–M25) — one response-path row (M12), and the routing row
**M14** (§3.3.5), which is listed here because the destination is a mutation surface the body cannot
show. Twenty-five rows in all. The former substitution row M13 is **withdrawn** — KBR-5 replaced it
with a downstream error, so it mutates nothing — leaving **twenty-four live** bridge-level rows.

| # | Mutation | Site | Trigger | Why it is necessary |
|---|---|---|---|---|
| M1 | Replace `model` with the profile's model, then provider-normalise it | `BridgeServer._normalize_model` | Always, when the profile sets a model | This is the product. The agent asks for one model; the profile decides what actually runs. **"Always" was false on one route until KBR-160.** On `openai_subscription`'s Responses path the adapter built the shipped body out of `cc_request["_original_body"]` — the *raw inbound* body — so M1 ran, computed the profile's model, and had it discarded one layer down; the agent's model shipped, silently. `OpenAISubscriptionAdapter._prepare_responses_body` now reads `cc_request.get("model", "gpt-5.4")`, the same expression its sibling `_cc_to_responses` already used, which is why the other routes into that provider were never affected. The row's **site is still `_normalize_model` alone**: the adapter *consumes* M1's output, it does not perform M1's mutation, so this closed a discrepancy between the register and the code without changing a single register field. What the episode does show is that `paths=(envelope.model,)` is a claim about the §3.2.3 capture boundary that a downstream consumer can silently break, and that nothing but a test at that boundary notices. |
| M2 | Translate the agent's protocol → Chat Completions | `MessagesTranslator` / `ResponsesTranslator` / `GeminiTranslator` `.translate_request` | Provider has `use_native_messages == False`, or the agent speaks Responses/Gemini | The upstream speaks a different protocol. Skipped entirely for native-Anthropic providers such as `zai_coding`. **Since KBR-222 the Messages-ingress half of this row carries non-text user blocks instead of dropping them:** an `image` block ships as a CC `image_url` content part (a `data:` URI for a base64 source, the URL verbatim otherwise; any other source type still drops), a `document` block rides the `_documents` internal key to the Anthropic family (P1 below carries it; the restore is `AnthropicAdapter`'s), and non-tool blocks beside a `tool_result` become one trailing user message — non-tool blocks after the tool results, where the M9 fallback (`server._convert_native_to_cc_format`) keeps its own text-first placement. Both converters share one user-content builder (`build_user_content_message`), so the fallback's retry carries images and documents exactly as hop 1 does — KBR-178's lesson applied: a mapping hop 1 learns and the fallback misses is lost again on the tool_use format-error retry. The fallback stays otherwise minimal (KBR-224's precedent). Text-only turns are byte-identical to the pre-KBR-222 output — a joined string — so the common case changes no shape. The Responses and Gemini translators handle no image parts at all, so on those ingress routes nothing here changes. |
| M3 | Truncate a tool result over 50,000 chars (`_TOOL_RESULT_TRUNCATION_LIMIT`) | `_truncate_oversized_tool_results` / `_truncate_oversized_responses_outputs` | A single tool result exceeds the limit | A single oversized result can exceed the model's window on its own. On `openai_subscription`'s Responses path the shipped body is built from the raw inbound body, so the mutation is performed there by `_truncate_oversized_responses_outputs` on the Responses-shaped `input` itself — same limit, same notice text; the CC-shape site truncates a copy that never ships on that route (KBR-169). |
| M4 | Truncate a tool result over the same limit, again, inside compaction | `_compact_messages` step 1 | Compaction ran **and** a `role: "tool"` message's string content exceeds the limit | Second pass, CC-shape only. Distinct from M3: M3 is unconditional pre-processing, M4 fires only once compaction is already engaged. |
| M5 | Compact the message history | `_apply_compaction` → `_compact_messages` | Serialized messages exceed the model-derived budget | Without it the upstream rejects the request outright. On `openai_subscription`'s Responses path the decision is carried onto the wire by `_prune_compacted_responses_input`, which prunes the raw body's `input` to the surviving conversation — a consumer of the compaction, not a second policy, but it adds two selection rules of its own: a riding item (`reasoning` and types the translation skips) is owned by the next assistant message, and a riding-run + `function_call`-run + `function_call_output`-run group moves atomically, because the CC grouper can split one wire hop across the pruning boundary and the Responses API rejects both a call without its preceding reasoning item and a reasoning item without its following item (community-verified 2026-09-14, not live-probed). A final orphan sweep mirrors pairing validation. Survival is matched by message identity against the pre-compaction snapshot, so an edited message fails conservative — items drop, never an unprotected body ships (KBR-169). **Since KBR-222 the translated route's body can carry base64 media** (CC `image_url` parts and the `_documents` key), so media bytes now count against this budget and against the request-size guard as they always have on the native routes. Head+tail pruning of a media-bearing message silently reverts that message to the pre-KBR-222 loss — its `_documents` entry is addressed by message identity, so it forfeits rather than mis-attaching (the identity invariant is load-bearing: compaction must keep rebuilding `role == "tool"` dicts only, as it does today). G42 records the residue. |
| M6 | Re-compact at half budget and re-send the same backend | `_compact_with_tighter_budget`, called only from `_request_with_retry_balancing` | Upstream returned 400/413 **and** `_is_context_too_large_error` **and** `_is_oversized_request` | Recovery from a rejection kitty's own budget estimate failed to prevent. **Balancing profiles only** — `_request_with_retry` (single backend) has no compaction recovery. Also an I2 exception; see §4.3 C3. Still wire-ineffective on `openai_subscription`'s Responses path: the tighter recompaction rewrites only the CC copy, and the retry re-prepares the shipped body from `_original_body` — the one remaining row of the KBR-169 class on that route. |
| M7 | Drop orphan `tool_result` blocks | `_validate_tool_call_pairing` / `_drop_orphan_responses_tool_outputs` / `_prune_compacted_responses_input` | A `tool_result` has no matching `tool_use` after compaction | An orphan triggers upstream error 2013 and fails the turn. On `openai_subscription`'s Responses path the rule runs on the Responses-shaped `input` with the same order-sensitivity — an output ships only when a preceding `function_call` declared its `call_id`, and a missing `call_id` is undeclared (dropped, where the translator's `KeyError` used to render a 500) — and the selector re-runs it after compaction, because pruning an earlier wire group can orphan a later output (KBR-169). |
| M8 | Add a thinking-carrier block and re-send the same backend | `_repair_thinking_roundtrip` / `_with_thinking_carrier` | This backend rejected this transcript for a thinking round-trip mismatch (issue #32) | Avoids one rejected round-trip per turn against backends that require it. Also an I2 exception. |
| M9 | Convert a native Messages body to CC format and re-send the same backend | `_convert_native_to_cc_format`, then a re-run of `_normalize_model` and `normalize_request` | Upstream returned a `tool_use` format error on the native path | Fallback that keeps the session alive rather than failing the turn. Also an I2 exception. |
| M10 | Inject the model from the URL path into the body | `_handle_gemini` | Gemini protocol only | Gemini carries the model in the path, not the body; `_normalize_model` needs it in the body to override it. |
| M11 | Force `stream: False` | `_handle_gemini` | Gemini protocol, non-streaming `:generateContent` | The Gemini translator defaults `stream=True`; the non-streaming endpoint must not open an SSE stream. |
| M12 | Substitute fallback assistant text | `_EMPTY_ASSISTANT_FALLBACK_TEXT` in `bridge/messages/translator.py` **and** `bridge/responses/translator.py` | Upstream returned an empty response | **Response-side**, not part of the twelve request-path rows. **Never on a streamed `/v1/messages` reply from a Messages-wire upstream**, native or translated: that stream is forwarded as the upstream sent it, so there is no translator to substitute anything (KBR-227 — before it, the translated Anthropic-wire routes pushed Anthropic SSE through the Chat Completions chunk translator and Claude Code received an empty reply). Empty replies on that branch are KBR-155's to retry. The non-streaming reply and the other inbound protocols still translate, and still can fire this row. |
| ~~M13~~ | **Withdrawn — no longer a mutation.** Was: discard the conversation and substitute a `[Kitty Bridge: …]` user message. | `_compact_messages` / `_apply_compaction` post-condition | No non-system message survives | **Closed by KBR-5.** The post-condition now raises `CompactionFailedError` and the handler returns a protocol-native 400 downstream; nothing is substituted, so there is no mutation left to register. The row is kept struck through rather than deleted so a reader of finding F3 can still find it. **The trigger recorded here was wrong** — see F3. |
| M14 | **Replace the destination entirely** — scheme and host are built from the profile by `build_base_url()`; the path by `get_upstream_path(_route_model(cc_request))` — the **request's normalized model**, which is the normalized profile model when there is one and the agent's model when there is not. `_route_model` is the single place that answers this; the auth scheme (P9/P20), the thinking carrier and the choice to forward a `/v1/messages` stream unchanged (`BridgeServer._serves_messages_wire`, KBR-227) read it too, and the adapter reads the same key for the body (KBR-127 — it was the raw profile model, so path and body could route differently; and on `openai_subscription`'s Responses path the adapter read the *inbound* body's model instead until KBR-160, which was harmless for routing only because that provider posts to a fixed URL and derives no header from the model). Base and path are then **composed** by `ProviderAdapter.compose_upstream_url`, not concatenated (KBR-143). | `BridgeServer._build_upstream_url` | Always | The agent addressed a loopback bridge; the request has to reach the real provider. Listed because **the destination is a mutation surface the body cannot show**: on Azure an identical body sent to the wrong deployment path is a different request entirely (§3.3.5). **The query is part of the mutation, not a passenger** (KBR-143): the endpoint joins the *path* component and the two queries merge, the endpoint's parameters winning a name clash and the base URL's others surviving unaltered. A row naming only "path" would let an oracle derive `route.query` and still not know which side owns a clash. The base URL's fragment is carried through and never sent, since no HTTP client puts one on the wire — so an oracle deriving `route.*` from the profile must expect it on the composed URL and absent from the request line. **The composed URL is redacted before it is echoed** into the 404 diagnostic or a pre-flight failure (`redact_url_for_display`): query values and the fragment are masked, which is an I2-adjacent containment property, not a fidelity one — nothing about the request changes. The composition helper is shared with `kitty.validation.validate_api_key` and `OllamaCloudAdapter._build_url`, but **this row's site is the bridge alone**: pre-flight's probe is not a request the agent made, and the register describes what happens to the agent's request. |
| M15 | Rewrite a string `input` into the single-item list form `[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": <s>}]}]` | `normalize_responses_request` (`bridge/responses/translator.py`), called from `_handle_responses` before the body forks | Always | OpenAI's `CreateResponse` defines the two forms as the **same request**: `input` is `oneOf` a string (*"a text input to the model, equivalent to a text input with the `user` role"*) or an array, and everything downstream reads the array. Fires on every request reaching the handler; a body already in the array form meets the row with a **no-op** rather than avoiding it, so there is no complement state for §3.3.2 assertion 2 to arrange, which is why it is unconditional. Listed rather than omitted because the rewrite is real bytes at the `curl_cffi` boundary of §3.2.3, where `_original_body` **is** this body; the projection cannot express the difference, so the row takes §3.3.1a's escape for P16's reason. **KBR-144.** |
| M16 | **Strip every block-level `cache_control` cache breakpoint** — from tool declarations, from system blocks and from message content blocks | `MessagesTranslator.translate_request` (`bridge/messages/translator.py`) | The upstream wire is not native Messages — i.e. the provider does not declare `use_native_messages` | The translator rebuilds the body for Chat Completions and discards the breakpoint as it goes: system blocks are joined into one string, tools are rebuilt as `{name, description, parameters}`, and content blocks are rebuilt. **The discard is the translator's choice, not a limit of the format** — OpenRouter's Chat Completions dialect carries `cache_control` on content parts, and `openai/openai-openapi` puts `prompt_cache_breakpoint` on Chat Completions content parts (GPT-5.6+, with a request-wide TTL only, so it cannot say `ttl: 1h`). An earlier reading of OpenAI's guide took the latter to be Responses-only; KBR-199 checked the schema. **One carrier escapes the strip**: a breakpoint on a block nested inside a `tool_result`'s list content is copied through both hops — pinned at the translator by KBR-198 and across both hops by KBR-199. That is outside this row's paths by design — §3.3.1 residualises a nested breakpoint. Whether Anthropic honours one at that depth is not established: its SDK types accept `cache_control` on a block inside `tool_result` content, and its docs' sub-content rule names citations only. **This is the row whose cost is largest and least visible.** Anthropic prices a cache read at 0.1x base input, so a stripped breakpoint re-bills the agent's stable prefix — system prompt, tool definitions, history — at **at least** ten times its cached rate, on every turn, with nothing in the product saying so. Registered rather than left to the residual precisely so the oracle reports it as a *claimed* delta attributable to this site; §3.3.1 records why the declared-ignored alternative was rejected. **The trigger is not `Always`, and the row is still exempt from §3.3.2 assertion 2** — the same shape as **M2**, which carries this identical trigger. The native passthrough branch (`BridgeServer`, `use_native_messages`) shallow-copies the inbound body, so breakpoints do survive there — except on the `tool_use` format-error fallback, which re-converts through `server._convert_native_to_cc_format` and strips them (folded into KBR-200); but that complement is a property of the **route**, chosen by the profile, not of the request, and assertion 2 asks for an *input* that fails the trigger. A corpus entry cannot arrange a different provider. The native route's guarantee is therefore proven where it belongs — as product behaviour, in epic KBR-197 — rather than by a corpus complement nobody could author. Top-level `cache_control` (Anthropic's automatic caching) is **not** this row's: it lands in `envelope.extra[cache_control]` and P1's internal-key strip does not touch it. The translator does not copy it either, though, so on the translated route that `extra` delta is **claimed by no row** — gap G38. **KBR-228 part B restored the system carrier on the signature-binding routes**: on `anthropic`, `custom_anthropic` and `zai_anthropic` (`forwards_thinking_signature`) the adapter re-attaches the agent's system blocks — breakpoints included — from the internal carriage, so the system-path claim above no longer reaches those wires (the CC-intermediate strip is unchanged, and the tool and message carriers are not restored; `tests/providers/test_anthropic_cache_breakpoints.py` was rewritten to the new wire). **KBR-167**; the product-behaviour suite for the same defect is epic KBR-197.|
| M17 | **Strip the thinking Anthropic rejects and re-send the same backend** — `thinking` and `redacted_thinking` blocks at or before the turn the rejection names, plus any unsigned `thinking` block anywhere; up to three strips per serialized body, the third removing every thinking block | `_recover_rejected_thinking` / `_strip_thinking_blocks`, called from `_make_upstream_request` and `_stream_messages`, and — since KBR-232 — from the three non-Messages stream handlers `_stream_responses`, `_stream_chat_completions` and `_stream_gemini` | The upstream rejected this request's thinking signatures with a 4xx (`_is_thinking_signature_error`) and the body had thinking to strip | api.anthropic.com verifies every thinking block it is sent back, and kitty's history fails that check: the translator drops signatures, **P5e** injects unsigned blocks (empty or reasoning-bearing — toward a signature-checking upstream every P5e block triggers this row), and M3–M7 and M9 edit the prefix a signature is bound to; the M8 carrier is also unsigned, but it fires only on the DeepSeek/Kimi wording, so its part here is theoretical. **Probed live on 2026-09-13** against `claude-sonnet-5`, `claude-opus-4-6`, `claude-fable-5-1` and, in manual `enabled` mode, `claude-opus-4-6` and `claude-haiku-4-5`: a missing signature or a P5e block gets `400 ...thinking.signature: Field required`; an altered one, or an edited earlier message under the prefix check (default for accounts from 2026-08-31), gets ``400 ...Invalid `signature` in `thinking` block``; the stripped history succeeds, including a manual-mode `tool_result` tail. Through a real bridge on unfixed `main`, all four cases tried — `anthropic` translated and `custom_anthropic` native, each non-streaming and streaming — returned that 400 to the agent; with this row all four answered. **Why targeted, not everything (owner decision, 2026-09-13):** the damage is not transient. Claude Code re-sends the full history every turn and kitty re-compacts every turn, so a broken block returns every turn; stripping everything would erase the model's reasoning on every later turn, fresh reasoning included. A live probe on `claude-fable-5-1` showed a block produced *after* a strip is valid, and survives when only the older broken blocks are removed (removing thinking from the front of the history is allowed; re-sending the broken block fails the request). **Why counted per serialized body:** a failover rebuilds the body with its thinking restored, and the next backend must get its own recovery rather than a quarantine for kitty's history. **Known costs, recorded rather than hidden:** each request whose history is still broken pays up to three rejected round-trips — whether a rejected 400 counts toward rate limits is not documented; each switch between stripped and unstripped prefixes re-writes the prompt cache (1.25x); and the only operator signal is a WARNING log line plus the per-backend `thinking_stripped` counter in `/stats` (KBR-228). **After KBR-228 part B** the restore fixes the unsigned rebuild on the signature-binding routes, so a strip there means the signed prefix was *edited* (M3–M7, M9) — not that the carriage is broken. **The alternative not taken:** Anthropic's `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` (beta header `thinking-binding-controls-2026-08-01`) drops only failing blocks with no rejected round-trip — live-probed: it handles an edited prefix but still 400s on a missing or altered signature, so a strip is needed regardless, and it would add a beta header kitty does not send today. Gated by body shape, not by wire: a Chat Completions body carries no thinking blocks, so there the strip finds nothing and nothing is retried — which also means a Chat Completions gateway in front of Anthropic gets no recovery. Recovery exists on every non-streaming handler and on every Messages-wire stream: KBR-227's `/v1/messages` passthrough and, since KBR-232, the Responses, Chat Completions and Gemini streams (whose loop runs wider by the strip budget, so a strip gets its attempt back as on `_stream_messages`). The strip runs after the M8 carrier and removes it too, and on the streaming path the #32 repair is skipped for a stripped body, so the two cannot take turns. A turn left empty keeps `content: []`, which the API accepts, so message indices never shift. Not a backend fault: the backend is not marked unhealthy — neither after a strip nor when a rejection outlives recovery (the cap, a stream's last attempt, or nothing left to strip). That history fails on every pool member alike, so cooling one would only take a healthy backend out of rotation; a non-streaming pool re-selects a backend — possibly the same one, since none was cooled — and stops after two 400s, as for any bad body; a stream surfaces the error to the agent at once (PR #113 review). Also an I2 exception, like M8 (§4.3 C3). |
| M18 | Synthesise a fresh `call_<uuid>` id when the inbound Gemini `functionCall` carries no `id` | `GeminiTranslator._translate_content` (functionCall branch) | The inbound Gemini `functionCall` carries no `id` | KBR-195. Chat Completions requires a tool-call id, so when the client omits one the translator mints it — the delta is a synthetic id upstream where the Gemini reader projected absence. Conditional, because the complement (a corpus entry whose `functionCall` carries an `id`) is plainly writeable and arrives with T-D5. Kept distinguishable from M19 by `paths` (`id` vs `tool_use_id`) — the axis `test_no_two_rows_are_indistinguishable` keys on — not by the site, which both rows share. |
| M19 | Synthesise a fresh `call_<uuid>` tool-call id when the inbound Gemini `functionResponse` carries no `id` | `GeminiTranslator._translate_content` (functionResponse branch) | The inbound Gemini `functionResponse` carries no `id` | KBR-195. M18's tool-result twin: the synthesised id lands on the tool message's `tool_call_id`, not on the call's `id`, so the row anchors at the other field. Kept distinguishable from M18 by `paths`, not by the site. |
| M20 | Drop the Gemini `systemInstruction` `Content` role — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` — the same trigger M2 and M16 carry | KBR-194 gave the Gemini reader a slot for the role the source `Content` published, so the drop is now a positive delta at `conversation.system_role` rather than an invisible normalisation. §3.3.1a's path-table cell named M2 as the claiming row until KBR-195; M2 takes the escape and is never path-matched, so this row is the actual claim. |
| M21 | Drop Gemini's NON_BLOCKING calling toggle on a tool declaration (`behavior`) — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. `_translate_tools` discards the field. Anchored at the field, not the whole tool — a coarser anchor would claim a deleted tool description, one of §3.3.1's own falsification cases (§3.3.1a). |
| M22 | Drop Gemini's `thoughtSignature` on a `functionCall` part — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. M8 also lands a delta at this path, but with a RESPONSE trigger (a thinking round-trip rejection); on a plain Gemini→CC request M8's trigger is not met, so the two are distinguishable by trigger, site and the narrower field anchor here. |
| M23 | Drop Gemini's `functionResponse.scheduling` (the NON_BLOCKING response-side toggle) — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. The reader projects `ToolResult.scheduling`; the translation drops it. |
| M24 | Drop Gemini's part-level `videoMetadata` — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. The reader projects the slot on text, `inlineData` and `fileData` parts; the translation drops it. |
| M25 | Drop Gemini's image `displayName` — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. The reader projects `Image.display_name`; the translation drops it. |

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
added without a row is caught by C1's exact-set assertion failing, never by the oracle. The
deviations §9.2's G22 recorded are registered as P9d–P9h (KBR-148); the exact-set assertion itself
is still owed — T-G9.

| # | Mutation | Site | Trigger | Why it is necessary |
|---|---|---|---|---|
| P1 | Strip kitty's internal metadata keys | `ProviderAdapter._INTERNAL_KEYS` via `translate_to_upstream` | Always | These keys are kitty's own; forwarding them is both an I1 and an I2 breach. **The set is incomplete — see F4.** Note it also strips `base_url`, which is *not* kitty-internal: it is defence-in-depth against a URL override arriving in the body, and is the one entry that could discard a field a caller meant. `_documents` (KBR-222) is a member: the agent's `document` blocks, each entry addressed to the CC message dict it belongs to, which only `AnthropicAdapter` and its delegates restore (there, verbatim, into that message) — so on the non-Anthropic-family routes the strip is what keeps documents off the wire, and the drop it performs is deliberate residue (G28's shape; G42 records it). **Since KBR-228 the strip also reaches *message* level**: the carriage key `_thinking_blocks` rides on a message dict, inside `messages`, where the top-level comprehension cannot reach it (`_anthropic_system` rides at the top level, which the ordinary strip already covers), so a copy-on-write helper (`ProviderAdapter._strip_internal_message_keys`) runs at the passthrough and at every messages-forwarding override. The behavioural guard `tests/test_internal_keys_not_sent_upstream.py::TestNoMessageLevelInternalKeyReachesUpstream` asserts it per adapter and per route; its first run found the defect live on 23 route-cases across 14 adapters. |
| P2a | Inject `thinking: {"type": "enabled"}` | `_ZaiBase.translate_to_upstream` (`ZaiRegularAdapter`, `ZaiCodingAdapter`) | `_thinking_enabled` truthy, or a reasoning effort other than `none` | Z.AI's wire format for a signal the agent sent differently. |
| P2b | Inject `thinking: {"type": "disabled"}` | same | `_thinking_enabled is False` **or** effort `== "none"` | The `else` branch. Listed separately because the oracle must not treat one as covering the other. |
| P3 | Inject `reasoning: {"effort": …}` | `OpenRouterAdapter` | `_reasoning_effort` present | OpenRouter's spelling of the same signal. |
| P4 | Inject `reasoning_effort` | `OpenAIAdapter` | `_reasoning_effort` present | OpenAI's spelling. |
| P5a | Default `max_tokens` to `_DEFAULT_MAX_TOKENS` (4096) | `AnthropicAdapter.translate_to_upstream` | Agent omitted `max_tokens` | The Messages API requires it. |
| P5b | Join system blocks with `\n` | same | Multiple system blocks | Messages API takes one system string. **KBR-228 part B**: on the signature-binding routes (`anthropic`, `custom_anthropic`, `zai_anthropic` — the `forwards_thinking_signature` switch) the agent's original system value is restored verbatim from the `_anthropic_system` carriage, so the join's delta no longer reaches those wires; everywhere else, and for every Chat Completions origin, the join stands. |
| P5c | **Replace** the agent's `thinking` with `{"type": "enabled", "budget_tokens": …}` — the agent's own `budget_tokens` verbatim when it is valid (an int, `>= 1024` and `< max_tokens`; **KBR-225**), where the row is **trigger-met-but-inert**: nothing ships that the agent did not send, so no delta appears at `conversation.sampling[max_tokens]` or `envelope.extra[thinking]` (the M15 precedent of a trigger meeting a row with a no-op) — and otherwise the derived `max_tokens - 1`, **raising** `max_tokens` to at least 1025 with it. Carry the agent's `display` onto that object, or withhold it, as P5d describes | same | Thinking enabled | Anthropic requires `budget_tokens >= 1024` and `< max_tokens`. **User-visible** (fallback branch only): the derived budget increases the agent's own `max_tokens`. **The budget also costs the prompt cache, and that cost is why this row exists** (KBR-203). Anthropic renders the thinking configuration — `budget_tokens` included — into the prompt, and a change to it always invalidates message-level cache breakpoints, and tool and system ones on some models. Before **KBR-225** the shipped budget was always a function of `max_tokens`, not of what the agent sent, so two requests in one session that differ only in `max_tokens` carried the agent's one budget as two and could not share a cached prefix; the forward branch removes that failure, and the fallback keeps the old derivation for an absent or invalid budget, where deriving is the only way to ship a legal configuration. The fallback stays *stable* across turns whose `max_tokens` does not move, which is what keeps its residual cost small; and on the translated route **M16** already strips every breakpoint but the nested `tool_result` carrier, so that residual cost is **latent** — it becomes the failure that still charges the 1.25×/2× write premium without the 0.1× read once the KBR-197 epic restores breakpoints. `tests/providers/test_anthropic_thinking_cache_stability.py` pinned the pre-KBR-225 behaviour (KBR-203) and KBR-225 inverted those pins rather than deleting them. **Native passthrough does not meet this row on its first attempt**: `tests/bridge/test_native_thinking_passthrough.py` proves the agent's `thinking` reaches the wire unchanged there. **M9's fallback is the exception, and it is not this row's**: `_convert_native_to_cc_format` carries no thinking, no effort and no `output_config` at all (KBR-224 kept its field out of the fallback for the reason P5d's row records), so the retry ships no `thinking` — a configuration change between the native attempt and its retry, which invalidates the message cache by the same vendor rule. Only `MessagesTranslator` writes `_thinking_display`, `_thinking_budget_tokens` and `_output_config`; the fallback's missing keys are that converter's general loss, not an oversight here. |
| P5d | Map `_thinking_adaptive` → `thinking: {"type":"adaptive"}` and `_effort` → top-level `effort`; restore `_thinking_display` → `thinking.display` on the adaptive object here and on P5c's enabled object — **or withhold `display`** on adapters whose `forwards_thinking_display` is false (`minimax_token`, `opencode_go`, `zai_coding`); carry the agent's `output_config` → top-level `output_config` — **or withhold it** on adapters whose `forwards_output_config` is false (the same three; `custom_anthropic` restores it only when a balancing failover re-serializes a request translated for an earlier backend, since its own requests are native) | same | Those keys present, or `output_config` present | Passthrough of an agent signal, and verbatim is what keeps the thinking mode cache-safe: Anthropic renders it into the prompt, so any normalisation would cost the prompt cache the way P5c does. **`display` is a mutation only where it is withheld, where the agent's value is not one of the two GA values, or where it arrives with `disabled`.** Until KBR-203 the translator did not carry it at all, so `{"type":"adaptive","display":"summarized"}` shipped as `{"type":"adaptive"}` on every translated route. The translator now carries only `"summarized"` and `"omitted"`, and only with `adaptive` or `enabled`: the beta `"updates"` needs an `anthropic-beta` header kitty never forwards (§4.2 C1), and `disabled` rejects `display` outright, so carrying either would turn a request that works today into a 400. The adapter restores the value where Anthropic's Messages API defines it (`anthropic`; `custom_anthropic` only when a balancing failover re-serializes a request translated for an earlier backend, since its own requests are native) and withholds it where the upstream does not document it: MiniMax's Anthropic-compatible reference names `thinking` but not `display`, its translated route exists because MiniMax rejected Claude Code fields before, and `opencode_go`'s Messages route serves MiniMax and Qwen models. **Restoring it now shows the user the model's thinking**: KBR-227 forwards the streamed reply byte-for-byte, and KBR-228 part A carries the non-streaming reply's thinking blocks — signatures included — through the CC layer to `MessagesTranslator`, so `"summarized"` earns its latency instead of adding it; carrying the agent's signed blocks back upstream is KBR-228 part B. The withholding lands at the address the restore uses, `envelope.extra[thinking]`, which this row and P5c already claim under the same triggers, so no new row is needed — and because `extra` is keyed, never nested (§3.3.1a), `thinking.display` has no finer address to give one. **Native passthrough follows the opposite rule and that is deliberate, not a contradiction**: it forwards the agent's `thinking` untouched (the first-attempt guarantee P5c records), so `minimax_token`'s opt-in native mode and `zai_coding` do receive `display`, and an agent's beta `"updates"` reaches Anthropic there without its header and is expected to 400 — native requests are the agent's own, and rewriting them is out of this row's scope. That `display` survives on some adapters and not others is a **scope** fact the register does not carry (KBR-139). The register guards compare ids and conditionality only, so this prose is reviewed, not tested. Top-level `effort` is not Anthropic's documented spelling — `output_config.effort` is, and until KBR-224 the translated route dropped it entirely. The translator now carries `output_config` verbatim on `_output_config`, and the adapter restores it only where the upstream documents the field: Anthropic's GA schema defines it (SDK `OutputConfig`: optional, nullable, no beta header — verified 2026-09-13 **in isolation**; a body carrying *both* effort spellings has never shipped, and kitty does not arbitrate between two values the agent sent), MiniMax's endpoint rejects bodies carrying it, and `opencode_go`'s Messages models and `zai_coding` serve references that do not document it, so those withhold — `custom_anthropic`'s own requests are native and already forward the field untouched, and only its failover re-serialization goes through this restore. The register data gains no `envelope.extra[output_config]` path: no corpus entry carries the field, so no oracle run can see a withhold as a delta, and the row that claims the address arrives with the first corpus entry that does (§3.2's register maintenance; the withhold's per-destination scope is prose for the same reason `display`'s is, KBR-139). |
| P5e | Inject an empty `{"type":"thinking","thinking":""}` block into assistant messages | `AnthropicAdapter._translate_assistant_msg` | Assistant message lacks one while thinking is active, **and the adapter opts in** (`injects_placeholder_thinking`) | The Anthropic-path analogue of P8. **A message-content change**, not a parameter change. **KBR-228 part C made the injection a per-adapter opt-in, and the base class — which *is* the `anthropic` provider — opts out**: the live probe behind KBR-238 falsified the row's premise that Anthropic requires the block (the unsigned block itself is rejected with `400 ... thinking.signature: Field required`, while a history with no thinking block is accepted), so on `anthropic` the placeholder cost one rejected round-trip per thinking turn and M17's strip had to remove it again. The four subclasses opt in explicitly, so no other profile's wire changed without evidence. |
| P6 | Remove `model` from the body | `AzureOpenAIAdapter` | Always | Azure selects the model by deployment id in the URL; the body field is rejected. |
| P20 | **Encode the request's normalized model as the deployment id in the URL path** | `AzureOpenAIAdapter.get_upstream_path` | Always | The counterpart of P6: what P6 removes from the body reappears in the path. A register that records only P6 makes the model look *dropped* when it was *moved*, and leaves the move unchecked. Before KBR-127 this read *the profile's* model, so a profile written `azure/my-deploy` addressed a `/deployments/azure/my-deploy/` segment that cannot exist. |
| P21 | Encode `project_id` and `location` in the base URL | `VertexAIAdapter.build_base_url` | Always | Vertex addresses a project-scoped endpoint. Same class as P20: routing carried outside the body. |
| P7 | **Cap the agent's `max_tokens` at 4096** | `FireworksAdapter.normalize_request` | Non-streaming request with `max_tokens > 4096` | Fireworks rejects non-streaming requests above 4096. **User-visible** as shortened output. |
| P8 | Inject empty `reasoning_content` into assistant messages | `ProviderAdapter._inject_empty_reasoning_content`, called from `KimiCodeAdapter`, `_ZaiBase`, `CustomOpenAIAdapter` | Thinking signalled **or** inferred from prior `reasoning_content` via `_detect_thinking_from_messages` | Those providers reject the request without it. The *inferred* trigger matters: it fires with no signal from the agent at all. |
| P9a | Set `User-Agent` to `claude-code/1.0` | `KimiCodeAdapter`, `BytePlusAdapter`, `MimoAdapter` `.build_upstream_headers` | Always, on those three | Those providers 403 without a recognised coding-agent user-agent. Central to I2 — F1. |
| P9b | Remove `Authorization`, add `api-key` | `MimoAdapter.build_upstream_headers` | Always | MiMo does not use Bearer auth. An auth-**scheme** change §4.3 C1's exact-set assertion must encode. |
| P9c | Synthesise a Codex CLI `User-Agent` and a `version` header, and add `Accept: text/event-stream` | `OpenAISubscriptionAdapter._build_codex_headers` / `._build_user_agent` | Always | Impersonation required by the subscription endpoint; `Accept` is the header half of P17's forced streaming — the Codex backend is streaming-only. **The two versions disagree — see F1.** The conditional `ChatGPT-Account-Id` this site also sets is **P9d** |
| P9d | Add `ChatGPT-Account-Id` | `OpenAISubscriptionAdapter._build_codex_headers` | The profile's `id_token` yields an account id (`_extract_account_id`) | The Codex subscription request carries it. **Conditional**, so it owes §3.3.2 assertion 2 a complement: a corpus entry whose `id_token` carries no claim — L1-pinned at `tests/providers/test_openai_subscription.py`, the corpus fixture arrives with T-D5. The header is also absent when the token fails to parse (`except Exception`) or the claim is empty (`if account_id:` truthiness), so absence doubles as the unparseable-token signature |
| P9e | Replace `Authorization` with `x-api-key`, and add `anthropic-version` | `AnthropicAdapter.build_upstream_headers` (inherited by `custom_anthropic`, `minimax_token`), `OpenCodeGoAdapter.build_upstream_headers_for_model` on Messages models | Always, on those routes | Anthropic's Messages API authenticates by `x-api-key` and requires `anthropic-version`. An auth-**scheme** change §4.3 C1's exact-set assertion must encode — the P9b shape plus an addition. The lowercase `content-type` re-spelling has no address (`contract.header_path` lowercases), so it is recorded here and in the row's comment rather than claimed. `opencode_go` reaches the set by delegation on its Messages models; its default hook is the baseline Bearer set and is deliberately not a site |
| P9f | Add `anthropic-version` (auth stays `Authorization: Bearer`) | `ZaiAnthropicAdapter.build_upstream_headers` | Always | The same wire requirement as P9e, but this upstream takes Bearer auth — so no scheme change to claim. Kept apart from P9e because a row's paths must be true of every site it names; the lowercase `content-type` re-spelling is recorded, not claimed, for P9e's reason |
| P9g | Replace `Authorization` with `api-key` | `AzureOpenAIAdapter.build_upstream_headers` | The credential is not an Entra ID token | Azure's key-based auth. The Entra branch of the same hook sends the baseline `Authorization: Bearer`, which is why the trigger is named. The credential is profile config, not request content, so no corpus entry can vary it — unconditional in §3.3.2's sense, the M16 shape |
| P9h | Remove `Authorization` | `OllamaAdapter.build_upstream_headers` | Always | Local Ollama requires no auth and ignores the header — the P9b shape minus the addition. `OllamaCloudAdapter` overrides the hook and keeps Bearer, so it is deliberately not a site |
| P10 | Set `reasoning_split = True` | `MiniMaxAdapter.normalize_request` | **Unconditionally** | Makes MiniMax return thinking in `reasoning_details` instead of inline tags. Unconditional, so exempt from §3.3.2 assertion 2. |
| P11 | Translate CC → Bedrock Converse | `BedrockAdapter.translate_to_upstream` | Always, on `bedrock` | A third upstream wire format M2 does not name. Custom transport — see §3.3.4. |
| P12 | Translate CC → Ollama `/api/chat` | `OllamaCloudAdapter.translate_to_upstream` | Always, on `ollama_cloud` | A fourth wire format. Custom transport. |
| P13 | **Drop fourteen Chat-Completions-only parameters** — `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `response_format`, `stop`, `n`, `stream_options`, `seed`, `logit_bias` | `OpenAISubscriptionAdapter._cc_to_responses` | Always, on the **CC-origin** path | The Codex backend applies strict allowlist validation and rejects them with 400. **User-visible**: `max_tokens` and `temperature` silently do nothing on this provider. Logged at DEBUG. |
| P14 | **Drop every *sampling* parameter outside the Codex allowlist**, notably `max_output_tokens` | `OpenAISubscriptionAdapter._prepare_responses_body` | Always, on the **Responses-origin** path | Same backend restriction, different input shape — this path receives a Responses body, so the parameter is spelled `max_output_tokens`, not `max_tokens`. A single row cannot cover both paths; the sets differ. **The non-sampling half of the same drop is P23**, which lands at `envelope.extra[<wire key>]` rather than here: one mutation site, two rows, because the register addresses effects and not sites. |
| P23 | **Drop every non-sampling control field outside the Codex allowlist** — `background`, `context_management`, `conversation`, `max_tool_calls`, `metadata`, `moderation`, `previous_response_id`, `prompt`, `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, `safety_identifier`, `service_tier`, `text`, `truncation`, `user` | `OpenAISubscriptionAdapter._prepare_responses_body` | Always, on the **Responses-origin** path | **P14's other half, and its own row because the two land at different addresses.** `CreateResponse` (`openai/openai-openapi` v2.3.0) defines 31 top-level request fields and `_ALLOWED_RESPONSES_PARAMS` keeps ten, so 21 are dropped: five are sampling parameters P14 claims at the bare `conversation.sampling`, and these **sixteen** are declared control fields, which §3.3.1b sends to `envelope.extra[<wire key>]` — an address no row reached. Under-claiming is the direction §3.3.1a calls unrecoverable, so the first T-D5 corpus entry carrying, say, `text` or `truncation` would have failed the run on a deliberate, legitimate mutation. **KBR-171.** The row **enumerates** its sixteen addresses instead of anchoring at a bare `envelope.extra`; §3.3.1a records why, and what enumerating costs. ⚠️ **"Dropped" here means *never copied*.** The allowlist literal is read only by the DEBUG log; the shipped body is an explicit `if` chain, and six of its branches test truthiness rather than presence — so an *allowlisted* field with a falsy value (`include: []`, `reasoning: {}`) is dropped too, and this row does **not** claim it. That residue is **P25** (`G27`), because it is a conditional mutation with a trigger of its own and because claiming `envelope.extra[reasoning]` here would swallow P22's claim (KBR-149). **User-visible**: `truncation`, `text` and `previous_response_id` silently do nothing on this provider. |
| P22 | **Inject `reasoning: {"effort": …}`** from kitty's internal `_reasoning_effort` | `OpenAISubscriptionAdapter._prepare_responses_body`, `._cc_to_responses`, and the transports `.make_request` / `.stream_request` | `_reasoning_effort` present and not `none`; on the Responses-origin paths, only when the caller did not already send a truthy `reasoning` | The Codex spelling of the signal P3 and P4 carry, on the adapter whose request path never runs `translate_to_upstream` (§3.2.3), so P4 cannot fire. **Four sites, not the three the gap walk counted** — `_cc_to_responses`, P13's own site, injects from the same key and predates the ticket. A request-body mutation feeding §3.3.2 assertion 1: the first T-D5 corpus entry carrying an effort would have failed the run on a deliberate mutation. At effort `"none"` the trigger is met with no mutation (the P5c precedent), so the assertion-2 complement needs an entry carrying an effort, not a `"none"` one; and the Responses-origin half waits on T-A3 projecting the effort signal, which the CC-origin half's reader already has. **KBR-149.** Corpus trigger case and complement arrive with the T-C entries, as for P3 and P4. |
| P25 | **Drop allowlisted control fields whose value is falsy** | `OpenAISubscriptionAdapter._prepare_responses_body` | Allowlisted field present with a falsy value | The allowlist's residue: membership in `_ALLOWED_RESPONSES_PARAMS` is not what carries a field through — the shipped body is an explicit `if` chain, six of whose branches test truthiness (only `parallel_tool_calls` tests presence), so `include: []` and `reasoning: {}`, both legal under `CreateResponse`, are permitted by the allowlist and dropped anyway. The reader projects by presence, so each is a present-inbound, absent-upstream delta P14, P23 and P22 all miss. **Register, not fix — the owner's decision (2026-09-14, KBR-185):** an empty `include` and an empty `reasoning` carry no instruction, and forwarding them to a strictly-validating backend adds a rejection risk for no benefit; the defect was that the drop was undocumented, so the row claims it and the adapter is unchanged. `tool_choice` is truthiness-gated on the same chain and is an `envelope.extra` key, but its falsy form is not a legal `CreateResponse` value, so that branch is unreachable today — deliberately not claimed, recorded here so no future reader re-derives it. **P22's interaction:** the two triggers are predicates on different request fields and can co-occur (falsy `reasoning` beside a non-`none` effort); in that state the `elif` injects, the upstream projection carries a `reasoning` key with the injected value, and P22's injection claims the address — P25's drop is provably absent there, so the rows' claims on `envelope.extra[reasoning]` do not overlap. **Conditional**, so it owes §3.3.2 assertion 2 a complement: a corpus entry whose `include` and `reasoning` are absent or truthy, in which the drop is provably absent; the trigger case and complement arrive with the T-C entries, as for P22. Enumerated in data but not derived the way P23's paths are — the derivation would need "whose falsy form is legal", a vendor-schema judgment no artifact in the tree holds (G24's posture). **KBR-185.** |
| P15 | **Strip `strict` from every tool declaration** | `_prepare_responses_body` | Always, on the Responses-origin path | The Codex backend rejects it. A change to the **tool schema** the agent declared, not to a sampling parameter — a different kind of fidelity mutation and worth its own row. |
| P16 | Rewrite content types `input_text` → `output_text` | `_convert_content_types`, called from `_prepare_responses_body` | Always, on the Responses-origin path | The Codex backend validates content types strictly. **Message-content mutation.** |
| P17 | Inject `stream: True` and `store: False` | `_cc_to_responses` and the Responses-origin body builder | **Unconditionally**, both subscription paths | The Codex backend is streaming-only; kitty reassembles a non-streaming reply from the SSE. Note `stream: True` **overrides a non-streaming client request** — the subscription-path analogue of M11. |
| P18 | Remove `modelId` and `stream` from the Converse payload | `BedrockAdapter` transport (`make_request` / `stream_request`) | Always, on `bedrock` | boto3 takes the model id as a call argument and selects streaming by choosing `converse` vs `converse_stream`, so both must leave the body. Applied **in the transport, after `translate_to_upstream`**. |
| P19 | Overwrite `stream` | `OllamaCloudAdapter` transport (`make_request` sets `False`, `stream_request` sets `True`) | Always, on `ollama_cloud` | The transport, not the caller, decides which Ollama endpoint mode is used. Applied **after `translate_to_upstream`** has already set it from the request. |

**The KBR-228 thinking carriage is not a row.** The Messages → CC converters (`MessagesTranslator.translate_request` and `server._convert_native_to_cc_format`) carry the agent's verbatim thinking-family blocks under `_thinking_blocks` on the message dict and the original `system` value under `_anthropic_system` at the top level, and the adapters whose upstream honours Anthropic's thinking-binding contract (`forwards_thinking_signature`) put them back byte-identical; the response direction mirrors this onto the reply for `MessagesTranslator` to consume. A carriage is kitty-internal, consumed or stripped before any wire or client sees it, and restores content rather than altering it — there is no delta for a row to permit, which is why KBR-228's "register rows P5e and the new carriage" resolves to row updates only. The keys ride `ProviderAdapter._INTERNAL_KEYS` / `_INTERNAL_MESSAGE_KEYS` under P1's strip, and the message-level guard in `tests/test_internal_keys_not_sent_upstream.py` keeps them off every non-restoring wire.

**Conditional rows are the point.** Every row whose trigger is a condition must be provably
*inert* when that condition is absent — the sharpest form of "unless absolutely necessary", and
what §3.3.2 assertion 2 tests, with the trigger complements §3.3.4 requires. M1, M2, M10, M14, M15, M16, P1,
M20, M21, M22, M23, M24, M25,
P6, P9a, P9b, P9c, P9e, P9f, P9g, P9h, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21 and P23 are
unconditional by design and are exempt from that assertion.

**M14, P20 and P21 were missing from that list until KBR-26**, while their own trigger cells read
`Always`. A row that always fires has no complement, so assertion 2 would have demanded a corpus
entry nobody could ever write. The list is also written out id by id rather than abbreviated as a
range: `P9a–c` names three rows in one token, and §3.2.4's guard has to either guess at the
expansion or drop two rows from the comparison. It refuses the notation instead.

**The P family's one gap was a reservation, and it has landed.** `P22` was held for the
`reasoning` injection §9.2's G23 registers; P23 landed first (KBR-171) and did not take the
reserved id, because an id is how every ticket and document refers to a row. P22 has since
landed at the reserved id (KBR-149). One reservation remains: **P24** is held for §9.2's G26,
whose enumeration waits on the Chat Completions reader's control-field table (T-A2). The register
is ordered by **document position**, not numerically — §3.2.2 already interleaves P20 and P21
between P6 and P7 — so a row arriving out of numeric order costs nothing.

**Register maintenance.** The register is the specification. A pull request that adds a mutation
site without adding a row fails the L2 register guards (§6.2.3). **A row's table entry here and its
`MutationRow` in `register.py` are one change**, not two: the guard parses this document, so a
commit carrying only one half red-lines the suite — at that commit and at every later bisect
through it.

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
two by an L2 guard. Paths and sites are **not reconciled against the markdown**, because the
tables have no path column and their Site cells are prose (M4's reads "step 1", P17's "and the
Responses-origin body builder", P2b's simply "same"). Sites are checked against the **source
tree** instead, which is the stronger check: it catches a renamed mutation site, which no
comparison against a prose cell could.

**One row's paths are reconciled, and they are reconciled twice — against the source *and*
against the markdown.** Both exceptions belong to P23 and neither generalises:

1. **Against the source.** P23's sixteen addresses are the published control fields the Codex
   allowlist does not keep, and an L2 guard recomputes that difference from the adapter's
   allowlist literal and T-A3's control-field table. This buys less than it appears to, and the
   difference matters: a bidirectional set-equality makes widening the allowlist a **deliberate**
   edit to the row rather than a silent one — it does **not** make the row independent of the
   code, because once the code changes the only route back to green is to edit the row to match.
   It is the posture G24 records for the OpenCode endpoint snapshot: a green run proves
   self-consistency, not agreement with the vendor.
2. **Against the markdown**, which is the exception to the paragraph above and exists because
   P23's Mutation cell **enumerates its sixteen wire keys in prose**. "The tables have no path
   column" is the reason paths are not reconciled, and for this one row it stops being true: the
   cell is a path list in all but spelling, so it is a second copy, and a second copy nothing
   compares is one nobody will notice going stale. Measured before the guard existed — deleting
   one of the sixteen from the cell, and changing its count word, each left the whole suite green.

Every other row's paths stay reviewed rather than derived, and P23 is enumerated rather than
computed at import so that a reviewer still reads a list. **The obligation travels with the
shape, not with the row**: a future row that also spells its keys out in the markdown inherits
both exceptions, and one that does not inherits neither.

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
**absent**. The trigger's *kind* — how its condition is decided — is the second dimension that
shapes that question, and it is one of four:

* **`REQUEST`** — a property of the inbound request; the corpus entry that carries the request
  decides it. Only this kind can be varied by a corpus entry, so only REQUEST rows have a
  complement case the corpus can carry.
* **`ROUTE`** — a property of the adapter/route dispatch. Every request on that route meets it
  (or none does); a corpus entry cannot vary it. The complement is therefore off-route rather
  than on-route, and the rule the §3.2.2 unconditional list enforces (no complement to write) is
  why rows like M2 and M10 are exempt. `P13` ``CC_ORIGIN_PATH`` is the canonical case: under
  reading (2) — the body reaching ``_cc_to_responses``, regardless of inbound wire — it is met
  when ``provider.dispatch == "_cc_to_responses"``, decided by the adapter's routing. Its
  mirror-image ``P14``/``P15``/``P16``/``P23`` ``RESPONSES_ORIGIN_PATH`` is the deliberate
  asymmetric — ``inbound_format == RESPONSES`` is REQUEST, not ROUTE, because the inbound wire
  (not the provider's dispatch) decides it. The two names look symmetric; the asymmetry is
  load-bearing.
* **`RESPONSE`** — a property of the **upstream response**, arranged by a scripted recorder
  (``M6``, ``M8``, ``M9``, ``M12``, ``M17``). T-D8 discharges these from a named
  scripted-recorder test, not from the corpus.
* **`PROFILE`** — derived from the profile (the compaction budget from the profile's model, and
  on a balancing profile from the smallest context in the pool — ``M1`` additionally: the
  profile sets the model; ``P9g`` non-Entra credential, ``P9d`` ChatGPT account id present:
  the profile's configured credential or OAuth authentication token decides them).
  Declared at the call site that resolves the profile.

Compound triggers — `GEMINI_NON_STREAMING` is the canonical case (``GEMINI_PROTOCOL`` ROUTE
on the route, then the request's stream flag — REQUEST there) — classify on the
corpus-decidability of the discriminating component. A compound whose discriminating axis
cannot be varied on-route is ROUTE; one that can is REQUEST. This is the rule, not abstract
purity.

`ALWAYS` is the absence of a condition, carries no kind, and is unconditional by design.

**The T-D8 quantification rule.** Stated so the implementation does not have to
re-derive it: every `conditional` row with a `REQUEST` trigger needs both a trigger
case and a complement in the corpus; every `conditional` row with `RESPONSE` / `ROUTE`
/ `PROFILE` is discharged outside the corpus (scripted recorder / on-route
unconditionally met / declared at the call site that resolves the profile). The
classification on `Trigger` (KBR-186) lets `NOT_CORPUS_DECIDABLE` be derived from this
table, so the corpus rule and the register classification cannot drift.

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
| Data ⇄ the adapter's own drop set — P23's sixteen paths | L2 | **T-W3** | One module's AST plus T-A3's control-field table. No sockets |
| Data ⇄ §3.2.2's P23 cell — the same sixteen, spelled in prose | L2 | **T-W3** | Two files. No sockets |
| Register completeness — projected delta at the wire equals the triggered rows | **L3** | T-G2 | Captures from T-D4–T-D9 |

The first four are what make the register *well-formed*. Only the last makes it *true*, and it
cannot run until a recorder and an oracle exist. The third and fourth are narrower than the first
two — they check **one row**, for the two reasons §3.2.4 gives — and the third is the only guard
anywhere that reads a *value* out of `src/kitty` rather than a name.

**None of the well-formedness guards proves the register is *complete*.** They prove the data and
the document say the same thing, that every site named still exists, and that one row's paths
match one allowlist and one published cell. A mutation the product performs and
*neither* artifact records is invisible to all of them — only the wire-level guard can catch that,
and it needs a recorder and an oracle. Eleven such omissions are already known and filed: G26
(P13's CC-origin twin), G28
(`top_k` dropped off the Anthropic family), G29 (an empty `stop_sequences` omitted), G30
(a string-form `stop` rewritten into a list), G31 (`metadata` dropped off the Anthropic family),
G32 (tool-selection fields dropped where a wire has no field), G33 (Bedrock's `toolChoice: auto`
written when the body forces nothing), G34 (`disable_parallel_tool_use: false` omitted), G35
(a legal `tool_choice` omitted where carrying it would create a failure), G37 (the Anthropic
family dropping a Chat Completions request's breakpoints) and G38 (a top-level `cache_control`
dropped on the translated route). **None of the twelve was found by a guard** — the earlier ones
by walking the subscription request path by hand, G28-G30 by the design and code reviews of
KBR-178, G31-G35 by writing and reviewing KBR-214's requirements, G37 and G38 by probing inputs for
KBR-199 and its design review. That is the evidence for the sentence above, not a decoration on
it. (G36, found alongside G31-G35, is a gap in the harness rather than a mutation, so it is not
counted here.)

**A header row's `paths` are checked by none of the four well-formedness guards**, and the two
that do check paths check P23's alone. Under §3.2.2's header rule they are
consumed by §4.3 C1's exact-set assertion, which does not exist yet — so for the eight P9 rows only
the id, the conditionality and the sites are under test today. A header row's paths are a reviewed
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
  tools:    ordered [ ToolDecl(name, description, schema, strict, cache_control?) ]
  sampling: a CLOSED set of fifteen canonical keys (§3.3.1b) -- declared, may be absent

Part = Text(str, cache_control?)
     | ToolUse(name, arguments, id?, cache_control?)
     | ToolResult(content, tool_use_id?, is_error, cache_control?)
     | Thinking(text, signature?)                     -- no cache_control: see below
     | Image(digest?, media_type?, ref?, cache_control?)
     | Json(value)                                    -- no cache_control: see below
     | Opaque(kind, digest?, cache_control?)
```

**`cache_control` is a slot on six of the eight, and the two exclusions are the vendor's rule
rather than ours.** Claude Code sets a cache breakpoint on nearly every request, so without a slot
the field could only residualise, and a non-empty residual fails the run — the grammar would reject
essentially every real body (KBR-167). Anthropic's published "what can be cached" list permits a
breakpoint on tool declarations, on system blocks, on text, image and document blocks, and on
`tool_use` and `tool_result` blocks. The two exclusions have **different** reasons, and conflating
them is how a later reader talks themselves into "fixing" the asymmetry:

- **`Thinking`** — Anthropic states a thinking block "cannot be cached directly with
  `cache_control`". Note what that does *not* say: thinking blocks **can** be cached alongside
  other content when they appear in earlier assistant turns, so the exclusion is about the
  breakpoint, not about cacheability. Anthropic does not document *rejecting* such a body, and its
  nearest analogue goes the other way — a below-minimum prompt "will be processed without caching,
  and no error is returned". So the honest statement is that the vendor does not support a
  breakpoint there. The consequence is on the record: such a body **fails the run**, with the field
  named, and the fix at that point is a slot or the third outcome, decided then.
  **The exclusion is the `thinking` *type*, not thinking generally**: a
  `redacted_thinking` block is `Opaque`, so it carries the slot like any other unmodelled block,
  even though Anthropic gives it no `cache_control` field either. Left as is deliberately —
  `Opaque` is the catch-all, and a per-kind exclusion table would be an abstraction with one user.
- **`Json`** — not a sub-content rule, despite the neighbouring one about citations. §7.4.1 fixes
  `Json` as the part for a format carrying a structured value *natively* — Converse's
  `toolResult.content.json`, Gemini's `functionResponse.response`. **Neither format has a
  `cache_control` concept at all**, and the Anthropic reader can never emit a `Json` part. A slot
  there could not be filled by any reader.

**A breakpoint nested inside a `ToolResult` residualises**, and since KBR-199 on **one** reason, which
is this design's choice rather than a vendor rule: §3.3.1a defines **no path form reaching inside a
`ToolResult`**, so a breakpoint mapped onto a nested part would be a delta **M16** could never
claim — §3.3.1a's under-claiming direction, which manufactures a false I1 breach. The vendor half
this sentence used to lean on is uncertain there: Anthropic's docs say sub-content blocks cannot be
cached directly but name only citations, and its SDK types accept `cache_control` on a block inside
`tool_result` content. **The cost is on the record**: such a body fails the run, with the field
named, on every route — including the default `anthropic` route and native passthrough, which
KBR-199 measured carrying it intact. **Revisit trigger**: the first real Claude Code capture
carrying a nested breakpoint decides whether to add a path form inside `ToolResult`. The
`ToolResult` itself is cacheable and carries its breakpoint normally.

**The wire mapping is carried whole, not reduced to a boolean.** `{"type": "ephemeral"}` and
`{"type": "ephemeral", "ttl": "1h"}` are different products at different prices (a 1-hour write
costs 2x base input against 1.25x for the default five-minute one), so a boolean would make a
silently downgraded TTL invisible — the same class of loss the slot exists to expose.

**Only the Anthropic reader fills it today.** `cache_control` is Anthropic's spelling. Converse has
`cachePoint` — a *separate block* in the content list, not a field on one — and Gemini has a top-level
`cachedContent` reference, so against either the field is present on one side only, which is
exactly what M16 needs, and an author of the Converse or Gemini reader should not go looking for an
equivalent to map. A normalisation, if one is ever wanted, lands here with the first reader that
needs it — the same rule §7.4.1 already applies to `Opaque`'s cross-vendor alias table.

**Chat Completions is the exception, and an earlier draft got it wrong.** It said OpenAI caches
with no per-block marker and that its GPT-5.6 `prompt_cache_breakpoint` is Responses-only.
KBR-199 checked `openai/openai-openapi`: that field sits on Chat Completions content parts too, with
its TTL set request-wide by `prompt_cache_options`, so it cannot say `ttl: 1h`. And OpenRouter's Chat
Completions dialect carries Anthropic's own `cache_control` on content parts. So T-A2's reader *can*
meet a breakpoint. Whether it fills this slot from either spelling (the OpenAI one is a
value-mapping decision) or residualises it is gap **G37**; until then such a body residualises.

**Why a slot rather than §3.3.1's other outcome.** §3.3.1 offers "map it, or declare it ignored
with a reason", and a reader-side declared-ignored mechanism would also have stopped the run
failing. It was rejected deliberately: kitty's translated path **strips every block-level breakpoint** (one nested carrier escapes — see M16), so
under a declared-ignored rule the oracle would be blind, by construction, to a mutation that
re-bills the user's cached prefix at **at least** ten times its cached rate — a cache read is
0.1x base input on most models and 0.025x on Claude Fable 5.1 and Mythos 5.1, where the multiple
is forty. Register row **M16** claims
the strip instead, which keeps the cost visible and attributable. The declared-ignored mechanism
therefore still does not exist; §7.4.1 records that, and no field currently needs it.

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

**`Image.digest` is the lowercase hex SHA-256 of the decoded bytes** — or, when the payload
cannot be decoded, of the raw encoded bytes the wire carried (see §7.4 rule 7 row 3; the second
of `image_digest`'s recipes, KBR-192). `media_type` is excluded from it and carried separately,
so a changed media type is its own delta. Gemini's `fileData.fileUri` has no bytes: `digest`
is then absent and `ref` holds the URI. Unpinned, the Messages reader and the Chat Completions
reader would produce different digests for one image.

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
`conversation.sampling`. **M16 is the sharpest case** — three paths at once, each naming
`.cache_control` rather than the block it sits on, because `conversation.turns[*].parts[*]` would
also claim a *deleted part* and `conversation.tools[*]` a *deleted tool description*, which are two
of §3.3.1's own five oracle falsification cases. Without that, "claimed by a register row" is a judgement call rather
than a lookup.

#### 3.3.1a The path vocabulary

T-W2 owns the string form, because it has **two** consumers that must agree exactly: a delta the
oracle reports (§3.3.4), and the "projection field it touches" column of every register row
(T-W3). Neither can define it without the other agreeing. `headers[<name>]` is the one form with
only the second consumer; §3.2.2 says why.

| Path form | Names |
|---|---|
| `envelope.model` · `envelope.stream` · `envelope.store` | The named control fields |
| `envelope.extra[<wire key>]` | A format-specific control field — P2a `thinking`, P3 `reasoning`, P4 `reasoning_effort`, P10 `reasoning_split`, and P23's sixteen dropped Responses control fields. **The bare `envelope.extra` is not a legal anchor** — see below |
| `conversation.system[<i>]` | One system text part |
| `conversation.system_role` | The role a Gemini `systemInstruction` `Content` published — the first path form at conversation scope, added by KBR-194. `None` is the absent value, so a translated route that drops the role is a positive delta at this path. M20 on the Gemini inbound route claims it |
| `conversation.turns[<i>].role` · `.parts[<j>]` | A turn, or one part of it |
| `conversation.tools[<name>].description` · `.schema` · `.strict` · `.behavior` | A tool declaration, **by name**. `behavior` is Gemini's NON_BLOCKING calling toggle (KBR-194) |
| `conversation.sampling[<key>]` | One sampling parameter |
| `conversation.turns[<i>].parts[<j>].cache_control` · `conversation.tools[<name>].cache_control` · `conversation.system[<i>].cache_control` | One cache breakpoint — **M16**. ⚠️ **A coarser row can claim these first**: a pattern is a prefix, so P5b's bare `conversation.system` and M5/M6/M7's bare `conversation.turns` subsume the breakpoint paths beneath them whenever their own triggers are met — and P5b's `MULTIPLE_SYSTEM_BLOCKS` is met by most Claude Code bodies. M16 is therefore the row that fires only where no collection-level row does; on the system blocks that means the single-block case, since P5b changes the collection's length and no `system[i]` path survives it. The field addresses the same three carriers the grammar gives it a slot on; `system_path` and `part_path` take an optional field name for it, as `tool_path` already did for P15's `.strict` |
| `conversation.turns[<i>].parts[<j>].signature` | A vendor thinking signature — Anthropic's on a thought part, Gemini's `thoughtSignature` on either a thought part or a `functionCall` part (KBR-194 gave the latter its `ToolUse.signature` slot). **M8**'s carrier repair produces a delta at this path |
| `conversation.turns[<i>].parts[<j>].scheduling` | Gemini's ``functionResponse.scheduling`` — the NON_BLOCKING response-side toggle (KBR-194) |
| `conversation.turns[<i>].parts[<j>].video_metadata` · `conversation.turns[<i>].parts[<j>].display_name` | Gemini part modifiers carried at the part path. Video understanding (``videoMetadata``) and the blob/file ``displayName`` named to the model (KBR-194) |
| `conversation.turns` · `.system` · `.tools` · `.sampling` | A **whole collection** — M5, M6 and M7 rewrite the turns, P5b joins the system blocks, §3.3.1 pins P13/P14 to the bare `sampling` |
| `headers[<name>]` | A header — the P9 header rows, and §4.3 C1. **Not produced by the projection diff**: `Request` carries no headers and no inbound header is forwarded, so this form addresses a per-adapter *deviation from the base header set* (§3.2.2), never a delta between two projections |
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
writes `thinking` whole — the derived `{"type": "enabled", "budget_tokens": max_tokens - 1}` on its
fallback branch, the agent's own configuration verbatim on KBR-225's forward branch (a no-op over what
the agent sent) — so any delta is at the key. P2a, P2b, P3, P4, P5d and P10 are anchored the same way
for the same reason. Under the other spelling every one of those six rows would match nothing and
§3.3.2 assertion 1 would report a false I1 breach on six *registered* mutations — the under-claiming
direction this section warns is the unrecoverable one. **Two sites enforce the rule**, so a reader
cannot emit the nested form by accident at either: `extra_path()` rejects a dotted key on the
path-builder side, and `Envelope.__post_init__` (KBR-191) rejects one on the constructor side.
The constructor's guard raises because `Envelope` is constructed by harness readers, where a
violation is a reader bug rather than a vendor datum; §7.4.1 records why that direction is
right for keys, with leaves still residualising for the opposite reason.

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

⚠️ **`envelope.extra` is not a legal register anchor either**, which is why the table above lists
it only in its keyed form. `path_matches` would accept the bare spelling — a bracket-free pattern
segment claims a bracketed member of itself, so `envelope.extra` names `envelope.extra[text]` —
and that is exactly the problem: this is the one prohibition nothing else in the suite would
notice. **P23 (KBR-171) is the row that wanted it**, and the reason it may not have it is a
collision, not an aesthetic:

> `extra` is where the *injections* live — P2a, P2b, P3, P4, P10, and P22's `reasoning`
> injection (§9.2's G23) at `envelope.extra[reasoning]`. A P23 anchored at the bare collection and
> triggered on `RESPONSES_ORIGIN_PATH` would claim that delta too, **on the same route**, so
> P22 could be deleted and nothing would go red. One row silently absorbing another is the
> unrecoverable half of the asymmetry this section opens with.

**P13/P14's bare `conversation.sampling` is not a precedent for it**, and not for the reason a
first reading suggests. It is *not* that the bare anchor and an enumeration claim the same thing
there — `SAMPLING_KEYS` has fifteen members and P13 drops fourteen, so the bare anchor
additionally claims `conversation.sampling[top_k]`. Those rows anchor bare **deliberately**, so
that a fifteenth key added upstream is claimed by the same row (`register.py` says so at P13).
The difference is what the over-claim can swallow: `conversation.sampling` is a closed set that
`Conversation` enforces and that no row injects into, so the widest thing the bare anchor can
absorb is another sampling key. `envelope.extra` is open, and absorbs whole rows.

**The cost of enumerating is real, and it is not "already paid for".** A thirty-second published
field that the allowlist drops would be unclaimed, and the oracle would report a false breach on
it. Nothing in the suite detects a vendor revision — no test reads the published schema, by §8's
determinism rules, which is G24's shape rather than a solved problem. What the enumeration buys
is that the failure is **loud, local and one line to fix**, where the bare anchor's over-claim is
silent and costs a row. A bare anchor would not have detected the revision either; it would only
have hidden it. What *is* pinned is the harness side: T-A3's control-field table is asserted
against the published key count, so editing it goes red there and again in P23's derivation
guard.

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

  **A value that is not a string at all — including an already-decoded object — residualises
  too.** Accepting the object form would make a bridge that emitted it where the schema demands a
  string invisible to the oracle, which is the wire-format breach the readers exist to see. And
  what residualises is the **raw wire value, unmodified** — `"[1,2]"` stores the *string*, never
  the decoded list — because T-D8 diffs residual key sets across all six readers, and two
  renderings of one unreadable value would report a delta neither reader caused.

  Six readers cannot quietly disagree about what `arguments: ""` means, and it is not
  hypothetical — `openai_subscription.py:OpenAISubscriptionAdapter._cc_to_responses` writes
  `func.get("arguments", "")`. The residual *path* is format-specific and stays each reader's own;
  only the decode and the fail-closed policy are shared.

  **Pinned as code: `contract.decode_arguments(raw, path, residual)`** (KBR-174), beside
  `image_digest` and for the same reason. It takes the residual as a parameter rather than
  reporting a flag the caller must act on: `mypy` covers `src/kitty` only, so a reader that
  ignored such a flag would be caught by nothing.
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
| **Cache-breakpoint sites in `MessagesTranslator.translate_request`** — which of Anthropic's sites it destroys, including the top-level key M16 excludes, and the one it carries | L1 | `TestTranslateRequestCacheBreakpoints` in `tests/bridge/test_messages_translator.py` — **KBR-198** (epic KBR-197). A **characterisation**: nine sites expect no breakpoint anywhere in the output; a block nested inside `tool_result.content` expects its breakpoint unchanged and in place, because that list is forwarded as-is — pinned as behaviour, not endorsed. The eventual carry-through fix inverts the nine; that is the intended workflow, not a regression. Inputs come from `harness.cache_breakpoints`, whose module documentation records why it emits **one breakpoint per request** (Anthropic rejects more than four), why the breakpoint is the 1-hour value, why its detector matches a key *containing* `cache_control` or the value itself, and why the prefix is padded past 2 x 4,096 words. **Not covered:** `server._convert_native_to_cc_format`, the native route's format-error fallback, which strips the survivor too (folded into KBR-200) |
| On the default `anthropic` route the translation pair delivers none of the agent's block-level or request-level cache breakpoints, the loss is already in the Chat Completions intermediate, and the adapter adds none of its own; given a Chat Completions request that carries breakpoints, the adapter keeps those on user and tool content and drops them at the other sites recorded | L1 | `tests/providers/test_anthropic_cache_breakpoints.py` (KBR-199, CB-2). A **characterisation**: the product fix for M16 turns it red on purpose. It also pins the one survivor, a breakpoint nested in `tool_result` list content. It compares the translator with itself only to show breakpoints have no effect on its output, which is not the fidelity round-trip §3.3.1 forbids |
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
| **C1 — Request headers** | `build_upstream_headers()` constructs the set from scratch; no inbound agent header is forwarded. Four adapters supply a coding-agent `User-Agent` (P9a, P9c). The deviations from the base header set are registered: `x-api-key` + `anthropic-version` (+ the lowercase `content-type` re-spell, unaddressable — casing) on the Anthropic family (P9e: `anthropic`, `custom_anthropic`, `minimax_token`, and `opencode_go` on its Messages models); `anthropic-version` + the casing re-spell beside Bearer auth on `zai_coding` (P9f); `api-key` on Azure's non-Entra credential (P9g) and on Mimo (P9b); no `Authorization` at all on `ollama` (P9h); the conditional `ChatGPT-Account-Id` on `openai_subscription` (P9d). Every other adapter sends the baseline set. | **Gap.** The per-adapter expectation is reviewable against the register; the exact-set assertion itself is T-G9. F1's policy gap remains. |
| **C2 — Request body** | The register's mutations (§3.2), JSON key ordering produced by kitty's serialisation, ~~the literal string `[Kitty Bridge: …]` (M13)~~ **— fixed, KBR-5** — and **`_effort` / `_thinking_adaptive`, which are kitty-internal and reach the wire**. With M13 gone the only bridge-introduced literal left in the body is `[Tool output truncated — original size: N chars]` (M3/M4): still a viable fingerprint, it simply does not name the product. | **Still breached by F4.** F3 closed. |
| **C3 — Cross-attempt content and cadence** | Retries (`_MAX_RETRIES = 3`), failover, transport-blip re-connects, the empty-response ladder — and **five** paths that send a *different body* on a later attempt (M6, M8, M9, M17, and failover re-normalisation). | §4.3 C3. Five declared exceptions. |
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
`Authorization` beside lowercase `anthropic-version` and `content-type` (P9f), while
`AnthropicAdapter` — and `opencode_go` on its Messages models — sends `x-api-key` with the same
lowercase pair and no `Authorization` at all (P9e). `AzureOpenAIAdapter` swaps `Authorization` for
`api-key` on the non-Entra credential (P9g), `MimoAdapter` removes `Authorization` entirely (P9b),
and `OllamaAdapter` sends no auth header (P9h). A provider can fingerprint any of that.

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
a different body on a later attempt in **five** distinct situations, not one.

| Path | What changes between attempts | Same backend? |
|---|---|---|
| M6 — compaction recovery | Body re-compacted at half budget | Yes |
| M8 — thinking-carrier repair | `messages` rewritten in place | Yes |
| M9 — native→CC fallback | Whole body converted, then re-normalised | Yes |
| M17 — thinking strip | Thinking at or before the rejected turn, and unsigned thinking anywhere, removed; everything on the third strip | Yes |
| Backend failover | `_normalize_model` and `normalize_request` re-run, so a same-host different-key sibling still gets a different body | No (different backend, possibly same host) |

The assertions:

- *(i)* Transport-blip retries and empty-response retries — the two that repeat a request
  unchanged — must be **byte-identical** to the attempt they repeat: same body, same headers, no
  added retry-count or correlation header. **Quantified over the retry arm, not the balancing
  arm**: the latter re-runs `_normalize_model` and `normalize_request` and is the failover
  re-normalisation exception in the table above, not a repeat. §11 Q14(b) widens this row's
  population — once the preamble hold lands, empty-response retries become reachable on the
  native-passthrough adapters too, so **T-I8** must name that path.
- *(ii)* Each of the five paths above is a **declared exception**: assert each fires only under
  its own trigger and never otherwise. A provider that hashes bodies can see all five; whether to
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

`_session_for` and `should_bypass` govern **only** `BridgeServer`'s own aiohttp sessions. Five
other outbound paths exist, and each applies the proxy **unconditionally, without consulting
`should_bypass`**:

| Path | Client | How the proxy is applied | Who closes it |
|---|---|---|---|
| `openai_subscription` — serving | `curl_cffi.AsyncSession` | `proxies=` + `CURLOPT_NOPROXY` | `aclose()`, from `stop_async` |
| `openai_subscription` — OAuth **refresh** leg | its own `curl_cffi.AsyncSession` | `proxies=` + `CURLOPT_NOPROXY` | `aclose()`, from `stop_async` |
| `openai_subscription` — OAuth **login** leg (`kitty.auth.openai_oauth`) | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` | `run_oauth_flow`, in a `finally`, when it built the session itself |
| `bedrock` | boto3 / botocore | `BotoConfig(proxies=egress.proxies_dict())` | nobody — nothing is cached; `_get_boto3_client` builds one per request |
| `ollama_cloud` | its own `aiohttp.ClientSession` | `aiohttp_session_kwargs()` | `aclose()`, from `stop_async` |

**The fourth column is new, and it was empty until KBR-190.** `BridgeServer.stop_async`
closed the two sessions the bridge itself builds and nothing else, so a bridge started and
stopped **inside a living process** leaked one connection pool per cycle per custom-transport
adapter. Ordinary use never noticed — the bridge's lifetime is the process's and the OS
reclaims the sockets — but the suite starts and stops bridges in-process thousands of times,
and that is where it surfaced, as an `Unclosed client session` per test.

**The bridge owns the adapters it is given.** `ProviderAdapter.aclose()` is the hook: a no-op
by default, overridden by the adapters that own a client. The ownership is a contract rather
than an observation about today's call sites — `get_provider()` returns a **fresh instance per
call** and all five construction sites build their adapters immediately before the
`BridgeServer` that receives them, so a caller must not hand one adapter to two bridges whose
lifetimes overlap. `tests/test_wire_shape_honesty.py` sweeps the registry so a fourth
custom-transport adapter has to decide about its client as well as its wire shape.

**Three points where this could have gone wrong, and what settles each.**

1. **Closing a `curl_cffi` session under a live SSE stream is the one shape that has crashed.**
   `_curl_session`'s docstring recorded that as the reason it was never closed, citing
   lexiforest/curl_cffi **#675**. That issue was filed against **0.13.0** on the *synchronous*
   `Session` and was **closed 2026-07-18 as no longer reproducible**. That was read on
   **0.16.3**, the version resolved here — `pyproject.toml` declares `curl_cffi>=0.7` with no
   upper bound, the weakest pin in the repo, so this is measured rather than guaranteed. More
   to the point, `stop_async` closes adapters only **after**
   `_runner.cleanup()` has drained the in-flight handlers, so the precondition is not met on
   the ordinary path. Upstream **#845** still tracks **#751**, "active stream/session close
   lifecycle", so the risk is reduced, not zero — which is why a **failed drain deliberately
   skips the adapter close**. Leaking a pool beats crashing the process.
2. **A failure must not cascade, in either direction.** One adapter's `aclose()` failing is
   logged and contained, so the others still close — shutdown has no caller positioned to act
   on a provider's teardown error. Conversely the adapters close from a `finally`, so a failure
   in the bridge's *own* teardown cannot skip them, which would leak exactly what this fixes.
   And each adapter detaches its cached session **before** awaiting `close()`: the `curl_cffi`
   builders test for absence only, so a session left in place after a failed close would be
   handed back for the rest of the process's life — silent, under the containment above, and
   permanent.
3. **Teardown is not a request.** `_close_provider_transports` reads the backing fields
   `_provider` and `_backends`, never the `_active_*` properties, which resolve through a
   request-scoped `ContextVar`. Reading that at shutdown would close whichever backend the last
   request happened to select. It deduplicates by identity, because balancing mode is
   constructed with `provider=backends[0][0]`.

**Why the refresh leg has its own session rather than sharing the serving one (KBR-161).** Not for
identity — both come from one builder, `_new_curl_session`, so they cannot drift apart on
impersonation target, CA bundle or egress mapping. For two reasons that have nothing to do with what
the provider sees: an `AsyncSession` owns a bounded pool of curl handles (`max_clients`, 10) and a
streaming completion holds its handle for the life of the stream, so a refresh sharing that pool
would queue behind in-flight completions **while holding `OAuthSession._refresh_lock`**, blocking
every request for that account; and cookies are host-scoped, so one jar would hold
`auth.openai.com`'s cookies for the bridge's lifetime and replay them across every account the bridge
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
| **Env-var register** | `_SETTINGS_ENV_OVERRIDE_KEYS` and `_CONFLICTING_ENV_VARS` (`launchers/claude.py`) match what `build_spawn_config` emits. The set-equality is pinned by `tests/test_launcher_claude.py::TestInjectedKeyListsInSync::test_settings_and_cleanup_lists_are_identical` so a future addition to one list and not the other is caught without an external oracle (the README is silent on individual keys; the L2 contract lives in the test, not the prose). The `KITTY_*` half — the env vars the README *does* name — is a separate contract: every documented `KITTY_*` name must exist as an exact-match string constant under `src/kitty/`. Landed with **KBR-76** as `tests/test_readme_table_guards.py::TestEnvVarRegister`. Forward direction only; the README is silent on internal/operational keys (`KITTY_TMUX_WRAPPED`, `KITTY_THEME`, …) by design, so an exact-equality guard would force either documenting plumbing or renaming it in source, neither of which is the right answer. Exact-match is pinned against a substring deviation by its own falsification. |
| **Provider routing table ⇄ provider docs** | For every model the provider publishes, the adapter routes to the endpoint the provider serves it on. Landed with **KBR-126** as `tests/data/opencode_go_endpoints.json` (a snapshot of OpenCode Go's published endpoint table, carrying `source_url`, `verified_utc` and a note on what a keyed probe would add) plus `tests/test_opencode_endpoint_table.py`. The snapshot is an **oracle**, deliberately not the routing table itself: deriving `_MESSAGES_MODELS` from it at import would remove the duplication and add a worse failure mode, since a missing or corrupt data file would silently route everything to the default endpoint — the defect, reintroduced invisibly. The checker is a pure function (`check_routing`) so the negative cases can hand it a deliberate defect, and it compares **set equality in both directions**: a constant naming a model the provider has *stopped* serving on a route is exactly as wrong as one it never started routing, and that is the shape KBR-126 actually was. **Honest limit:** snapshot and constants are written in the same commit, so a green run proves self-consistency, not agreement with the provider; no stronger evidence is reachable without a paid key, because an unauthenticated probe of either endpoint returns `401 AuthError` (auth precedes dialect). Hence gap **G24**. |
| **`validation_model` reachability** | For every adapter that is not `use_custom_transport`, the path `validate_api_key` posts to and the bare `build_upstream_headers` agree on a dialect the key-check ping is written in. Landed with KBR-126 as `tests/test_validation_model_routing.py`. The ping body — `{model, messages, max_tokens, stream}` — is **simultaneously valid Chat Completions and valid Anthropic Messages**, which is why `anthropic`, `custom_anthropic`, `minimax_token` and `zai_coding` validate against `/v1/messages` and work. So the rule is *not* "`validation_model` must be Chat-Completions-routed": it is that path and auth must match. Pointing `opencode_go` at a Messages-routed model — the fix KBR-126's own ticket suggested — leaves the headers `Bearer` and fails every key check, and a path-only guard would pass it. |
| **Endpoint table** | The README endpoint table matches `_register_routes`. Catches F2 (KBR-9). Landed with **KBR-76** as `tests/test_readme_table_guards.py::TestEndpointTable` — AST-scanned by function name (bridge-mode branch plus the unconditional `/healthz`//`stats` registrations; launch mode out of scope), both directions, self-guarded. The agreement assertion is the `t-g1-endpoint-table` exemption (KBR-9): red at the base revision, and the row fails the job the day the README correction lands. |
| **Attribution-header table** | The README's `X-Kitty-*` table matches `_attribution_headers()`, and none of those names can reach any `build_upstream_headers()`. Landed with **KBR-76** as `tests/test_readme_table_guards.py::TestAttributionHeaders`. The "none can reach" half is discharged structurally: a raw-text scan (case-insensitive, deliberately not AST) asserts no `X-Kitty-*` literal exists in any `src/kitty/` file except `bridge/server.py` — a comment-side leak is caught too, and a falsification case pins that mechanism choice. Stated limit: a provider that *forwards* an inbound `X-Kitty-*` header without ever writing the literal stays invisible; that is a behavioural contract the structural scan cannot carry. |
| **Flag table** | The README logging-flag table matches the CLI parser. Landed with **KBR-76** as `tests/test_readme_table_guards.py::TestLoggingFlagTable` — the four flags probed through public `parse_known_args` (no argparse internals), the two default paths read from `_DEBUG_LOG_PATH` and a stub-constructed `BridgeServer` (mirroring `tests/test_cli_log_file.py`), and an unknown-flag negative control proving the accept-tests are not vacuous. |
| **CI capability inventory ⇄ the workflow tree** | §8.6's table of what the CI environment supplies matches `.github/`. Landed with **KBR-216** as `tests/test_ci_capability_inventory.py`. Five arms: forward (the document promises nothing CI lacks), reverse (a capability cannot reach CI undocumented — compared as **whole binding names**, so a new `secrets.KITTY_EGRESS` is not absorbed by the documented `secrets.KITTY_EGRESS_JSON`), version (the hand-maintained Claude Code pin, both ways), the logs the generated launcher writes, and the fork guard. The last two exist because a review round found four surviving mutants on them: the two log rows bind neither a secret nor a version, so every other arm was blind to their deletion, and the fork guard's removal changes no binding at all. Every row is covered by at least one *reverse* direction — that is the property, not the count. It lives in `tests/` rather than beside the review system's own suite in `.github/review/tests/` — §8.5's precedent — because its subject is `TEST_SUITE.md`, and that suite must stay runnable by a bare interpreter that cannot be asked to reason about the design documents. **Two stated limits:** the reverse arm scans the `KITTY_`-prefixed namespace only, so a non-kitty secret is deliberately not a row; and the forward arm is a substring scan for every artifact *except* the launcher, which is read by calling `wrapper_body` because that module's own prose defeats a scan of its source. |
| **Answered questions ⇄ dependent passages** | A question `TEST_SUITE.md` §11 marks **ANSWERED** is not described as open or blocking anywhere in either design document. Landed with **KBR-163** as `tests/test_answered_questions_are_settled.py`. Range mentions (`Q10-Q13`) are expanded, because the site that escaped the hand-written enumeration was a range; §15's blocking table is checked by **row content** rather than by phrase, since such a row states the block by position and contains no still-open word at all. The one exclusion — a question's own §11 entry, which keeps the original wording verbatim — carries its own bound assertion, per the rule below. **Two stated limits:** a passage must *name the number* to be seen, and a paraphrase that never does is a reading job, not a scan. |


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
| Cross-attempt content (§4.3 C3) | Transport-blip and empty-response retries byte-identical; each of M6, M8, M9, M17 and failover re-normalisation fires only under its own trigger |
| Connection lifecycle (§4.3 C5) | Distinct-connection count per session, against the native baseline |
| **Streaming recovery — content, not just grammar** (below) | Four injection points; no duplication, no replayed tool calls, no spliced arguments — and, per §11 Q14, exactly one upstream request at the recorder for the three post-emission points |
| Client disconnect during a stream | Upstream connection released; the backend not marked unhealthy for a client-side fault |
| All backends unhealthy | The 503 arrives in each protocol's native error envelope — but only once no recovery fits inside KBR-243's 300 s arrival-recovery window (`SYSTEM_DESIGN.md` §6.2); inside it the request is held and served |
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
| After text has been emitted | No text the client already received is repeated — **because there is no second attempt to repeat it from**: the recording upstream sees exactly one request, and the transcript reads as one message ending in a terminal error event. A clean failover to a second backend also satisfies "no repeated text", which is why the recorder assertion and not the transcript is what makes this row bite |
| Mid `input_json_delta`, tool arguments partly sent | Arguments are never a splice of two attempts, **and the turn ends there**: per Q14 the client receives no argument bytes from a second attempt, the partial `tool_use` block is closed, and one terminal error follows. The negatives still hold — no silent merge, no reused id across attempts — but they are no longer the whole oracle |
| After content, before the terminal event | Exactly one terminal outcome reaches the client, and per Q14 it is the **error** event rather than a `message_stop` synthesised from a second attempt; no `message_stop` follows the error, and none is duplicated. Where the finish chunk already arrived but its events were still buffered, the buffered `content_block_stop`s of blocks the client already saw open precede the error — never a stop for a block that chunk itself opened |

Each case asserts tool-call **identity** and a single terminal outcome. Only the first case has
two attempts to compare, so only there does "never reused **across** attempts" have content; for
the other three the stronger assertion is that no second attempt exists at all.

**The post-emission semantics are settled — §11, Q14, answered 2026-09-12.** Once a byte has
reached the client the bridge does not retry and does not fail over: it closes any half-open
block, emits one terminal error, and lets the agent retry the turn. So every row above has a full
acceptance oracle, and the four injection points divide cleanly — the first is pre-emission and
recovers silently, the other three are post-emission and terminate.

Since KBR-183 closed gap G26 (2026-09-13) that is the choice the bridge makes for **every**
post-emission failure: no retry and no failover on `/v1/messages`, on the Responses and Gemini
custom-transport branches, and inside `openai_subscription`'s own stream-reset retry. KBR-236
(2026-09-14) closed the site that claim was missing on `/v1/messages` itself: the translated
branch's empty-response retry (`translator.response_was_empty`) read no emission state, and
because an empty finish chunk resets the translator, content arriving after the verdict was
written live — so the ladder put a second attempt on the stream that already carried it (its
exhaustion failing over onto a Messages-wire backend ended in the "retry came back empty" error
the ticket reproduced). It now reads `sr` and ends the turn like every other post-emission
failure, under row 2's oracle; no backend is charged, keeping the empty ladder's no-quarantine
health model. One residual
differs in its *ending*, not in its recovery: a post-emission **transport** drop on the translated
`/v1/messages` path still closes with `end_turn` + `message_stop` rather than the error event
(KBR-183's decision D2, carried as a scope addition on KBR-99, so the second, third and fourth rows
above go red on a transport-drop injection until it is settled). It is also
the same one the real Anthropic API makes: its mid-stream failures arrive as an SSE `error` event on
an already-`200` response and are raised to the caller, never resumed. **I2** is why that matters
— a bridge that recovers where the provider gives up is observably not the provider.

The empty-stream case does not reach these rows at all: per Q14(b) the native passthrough holds
its leading events until the first content event, so a contentless reply is still pre-emission
when it is detected. That is KBR-155's to implement; the rows here assume it. Since KBR-241
(2026-09-14) neither does a pre-content `error` event: the hold records it and the attempt takes
the pre-emission ladder, so the post-emission rows keep governing only failures that follow
content that actually reached the client.

`/stats` remains authoritative for attribution, but not for the reason this paragraph used to give.
Since KBR-183 no stream switches backend after its first byte, and `/v1/messages` prepares its
response at the first content write, so there the `X-Kitty-*` headers name the backend that produced
every byte. The Responses, Gemini and Chat Completions handlers prepare their response before the
first upstream attempt, so after a legitimate **pre-emission** failover their headers name a backend
that produced nothing.

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
§8.6 supplies the binary and two facts this paragraph would otherwise mislead on: the CI
installer leaves `claude` at a path that is **not** on `PATH`, and the review wrapper passes
`--no-validate`.

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

Pinning it in CI was recorded as Q12 rather than assumed away — which distribution, how tightly
version-pinned, and whether redistribution inside a CI image is acceptable. **Q12 is answered
(2026-09-12, KBR-216) and the answer is in the tree:** the review workflow installs the CLI from
the official installer at an exact version, at run time, so nothing is redistributed and the
question of whether it may be does not arise. §8.6 is the inventory, including the two limits a
job planned around it has to respect — the pin is a hand-maintained pairing, and a fork pull
request gets no secrets. The hermetic smoke above needs **only the pinned CLI**, which a fork run
can also download, so it is a per-PR gate without an asterisk.

**Agent live — nightly.** `tests/integration/test_agent_e2e.py` as it exists: real binaries, real
credentials, real providers. §8.6 is where those credentials come from, and it carries the two
constraints this job inherits: the profile is a balancing pool, so no assertion may name a single
model; and kitty's debug log carries whole request bodies, so it **may never be uploaded as an
artifact** however convenient that would be for diagnosing a nightly failure. Keep it out of the
default run — it needs four agent CLIs and live network, and a suite that cannot run on a laptop
stops being trusted. Nightly with CI secrets,
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

#### 7.1.1 What T-W6 settled — the format, the scrubber and the loader

Delivered as `tests/harness/corpus.py`, `tests/corpus/` and `tests/corpus/README.md`, which carries
the capture procedure in full. **An entry is one request** — not a session and not a transcript, and
§3.2.4's "indexes captured sessions" should be read as "entries". The unit matters: M8's trigger
case is a *pair* (a request, an upstream rejection, the repaired retry), which a single-request
entry cannot express and which therefore belongs to the test that scripts the recorder.

**Owner and cadence, answered by the product owner (2026-09-12).** Owner: the repository
maintainer. Cadence: **re-capture when the pinned Claude Code version changes.** A version bump is
the only event that can invalidate a capture, so a calendar cadence is both too late (a release
lands the week after it runs) and wasted work (nothing changed), and refresh-on-demand is the
museum this section warns about. This is why the format **requires** `captured_from` on a captured
entry: a cadence tied to a version bump is unactionable if the entries do not say which version
they are.

**Scrub scope, answered by the product owner (2026-09-12).** Credentials **and** personal
identifiers — keys, tokens, auth headers and query parameters, plus home-directory paths,
usernames, e-mail addresses and hostnames. File contents and prompts are *not* synthesised; that
would destroy this section's own rationale. Human review before commit stays mandatory.

**Manifest plus sidecar body, not one file.** Byte-exactness is the obvious argument and the
weakest — JSON escaping is reversible. The two that decide it are **reviewability** (this section
makes human review mandatory, and a 50,000-character escaped one-liner is not reviewable in a
diff) and **greppability** (the lint reads the body as text, and so does GitHub's push protection,
which is the last line of defence when the scrubber misses). Base64 is byte-exact and removes that
second net entirely — which is the answer to any later proposal to "simplify" to one file.

**The committed body is byte-exact as committed, not as sent.** Scrubbing is the one bounded
departure from the wire bytes. It is free for I1, because the oracle diffs two projections of the
same scrubbed input. It is not free in two places, both recorded in the README: the **size-derived
triggers** (M3 and M5 are decided by length, and scrubbing shortens a body, so a declaration is
made against the committed artifact and never the capture), and **`content-length`**, the single
header the corpus does not preserve as captured — it is recomputed, because the alternative is a
manifest internally inconsistent with its own body. Nothing downstream reads the original value:
§4.3 C1 asserts on the headers the bridge *builds*, and C1b compares names.

**Captures carrying `content-encoding` or `transfer-encoding` are refused at write time.** The
corpus stores entity bodies, not wire octets. A compressed body is one the scrubber reads as noise
and reports clean — a false clean that no plaintext falsification case can ever detect, and the
worst failure available to this component.

**Triggers have three states: met, explicitly absent, and silent.** Silence is not absence. §3.3.2
assertion 2 is only as good as the complement it runs against, so a complement must be *claimed*;
if absence were inferred, every entry whose author never considered a trigger would be silently
offered as its complement and the assertion would run over entries nobody vetted.

**§3.2.4's binary, expanded to four kinds.** `register.py` now carries an
`ArrangingBy` enum (`REQUEST`/`ROUTE`/`RESPONSE`/`PROFILE`) on every non-`ALWAYS`
trigger, and `corpus.NOT_CORPUS_DECIDABLE` is *derived* from that classification
rather than hand-listed (KBR-186). Five RESPONSE rows (`M6`/`M8`/`M9`/`M12`/`M17`)
are discharged by a named scripted-recorder test, not by an entry; four ROUTE
rows (`M2`/`M16` non-native upstream wire, `M10` Gemini protocol, `P13` CC-origin
path) and five PROFILE rows (`M1` profile sets model, `M4`/`M5` compaction
budget, `P9g` non-Entra credential, `P9d` ChatGPT account id present) are
likewise declined in the manifest — every one is met at the route or profile
that resolves it, not by the inbound request. The loader refuses all thirteen
plus `ALWAYS` in both lists; the earlier hand list stopped at the five RESPONSE
rows plus `ALWAYS`, which is the drift the derivation removes. With the
classification in place, **T-D8 *can* now read corpus coverage** for every
conditional REQUEST row (the set T-C1–T-C6 populate), with the RESPONSE rows
discharged by the scripted-recorder test and ROUTE/PROFILE rows having no
corpus complement to find (the `P9d` row's complement is discharged by the
L1 pins in `tests/providers/test_openai_subscription.py`, not the corpus,
because the trigger is PROFILE). G21's over-declaration hazard is closed for
the whole vocabulary rather than the cases the repository happened to prove in
text.

**T-C7's connection-pattern baseline is a different artifact.** This format carries the *header*
half — a single request whose `host` is the real one. C5 counts distinct TCP connections across an
N-turn session, which needs session grouping, ordering and `CapturedRequest`'s `arrival` and
`peer_port`; the manifest carries none of those, deliberately. T-C7 defines that artifact. Do not
stretch this format to hold it.

**The lint fails on an empty corpus.** "No secrets found" is satisfied perfectly by having looked
at nothing (the rule §8's marker guard is built on). It also turns a quieter mistake into a loud
one: `load_corpus` takes a root, so a caller pointed at the wrong directory would otherwise report
the corpus clean forever. This is why one synthetic `format_example` entry ships with T-W6 — as a
worked example reviewable in one screen, excluded from every evidence query by `captured_only()`.

**No finding, message or `repr` ever carries a matched value, not even a prefix** — class and byte
offset only. A message that quoted what it matched would turn a contained authoring mistake into a
published one the moment CI logged it, and the remedy for a published credential is rotation, not
a better diff. The README carries that incident step.

**Every pattern is anchored, and the scan runs to a fixed point.** The two are one decision. An
unanchored rule matches inside ordinary words — `disk-usage-monitoring-service.py` redacts to
`di<redacted:openai_key>.py`, and file paths are the commonest payload in a Claude Code body, so an
unanchored table mangles legitimate content at scale. Anchoring alone then creates the opposite
defect: two secrets flush against each other leave the second with no boundary in front of it,
until the first is redacted — and by then a single-pass scan has moved on. That shipped briefly as
a live key left in a scrubbed body, with the fixed-point invariant asserted as load-bearing in
three documents and false in practice. Both halves therefore iterate, and both claim exactly the
span the rewriter replaces; dropping either rule made them disagree on 216 and 23 of 4,000
adversarial bodies. What is guaranteed is that no secret survives and a second scrub is a no-op,
asserted over every ordered pair of known shapes at three separations. What is *not* guaranteed is
that the finding count equals the placeholder count: one redaction can subsume a neighbour, which
names more than it needs to rather than less.

**The writer refuses exactly what the reader refuses — by running the reader's rules, not a copy.**
`write_entry` accepted entries `load_corpus` then rejected, which turns the capture procedure's last
step into a success that fails later in CI against a file already committed. It was closed one rule
at a time for three review rounds — the id, then provenance and triggers, then field types — and
each round found the writer's copy of the rules one short again. The fix that ended it was
structural: there is now **one** validator, `_entry_from_manifest`, pure over a manifest and its
body; the reader calls it on what it parsed and the writer calls it on the entry it was given, before
anything is scrubbed or written. A rule list maintained twice drifts; a rule list run twice cannot.
The test that holds it is adversarial — every malformed shape review found, fed through the writer —
because the earlier version used only well-typed entries and so could not fail on the very defect it
was named for.

The second asymmetry had the same shape — a rule enforced on one side only, so the gap looked like a
check that existed. And `scrub` scanned the body, the headers and the query but left
**`host` and `path`** alone: a path of `/v1/key/<token>/messages` was committed and linted clean,
and an internal hostname survived even when the operator named it in `extra` — while this document
already claimed hostnames were removed. Scanning the routing fields costs nothing the oracle needs,
because the patterns are shape-anchored and §3.3.5's evidence is structural: Azure's deployment
segment and Vertex's `projects/…/locations/…` are pinned by test against exactly this change.

**The manifest's prose is linted, not scrubbed.** `description` and `origin_note` are where a
maintainer writes what the policy exists to exclude, and they sat outside both the scrubber and the
lint. They are now reported and deliberately not rewritten: mangling a description makes an entry
harder to review rather than safer, and the author of the sentence is the right person to fix it.

**The scrubber matches shapes, never entropy.** A general high-entropy rule is the tempting
addition and was rejected on measurement: a real Claude Code body is full of long opaque strings
that are not secrets — thinking-block signatures, `toolu_` identifiers, base64 images — and §3.3.3
requires `Please explain how kitty-bridge works` to survive byte-identically. A scrubber that
mangles legitimate content breaks I1 in the act of defending the repository. The residue is the
review step's, and a per-entry `known_non_secrets` allow-list (each pair carrying a reason, each
failing the lint when it stops matching, per §6.2.3's rule for a stale exclusion) covers the case
this repository creates for itself: captures are taken while working on kitty-bridge, so a tool
result quotes this tree — and `tests/test_integration.py` alone contains `api_key = "sk-test-…"`.

**Entry size is left open.** T-C3's over-budget transcript is ~2.8 MB, which nobody reviews by
eye. The format imposes no ceiling, because the choice between committing the real thing and
synthesising a padded construction belongs to T-C3 and T-C4 — plan §6 already blesses synthesis
for those two — along with saying what replaces the review step either way.

### 7.2 Recording upstreams — one per transport

The bridge reaches upstream through **five** distinct client configurations (§3.2.3, §5.5), and
an aiohttp recorder observes only one of them. One recorder per configuration, all presenting the
same interface to the tests:

| Recorder | Serves | Observes |
|---|---|---|
| aiohttp server — bridge sessions | the 20 default-transport adapters | The primary; speaks Anthropic Messages and Chat Completions |
| aiohttp server — provider sessions | `ollama_cloud`, and the `openai_subscription` **OAuth login leg** | Those adapters build their own sessions and never touch `_session_for`, so the bridge recorder never sees them. The OAuth leg runs at startup, before anything else has been proven (§5.5). The **refresh** leg moved to `curl_cffi` in KBR-161 and is the row below's (§7.2.3) |
| curl_cffi-reachable server | `openai_subscription` serving path, and the OAuth **refresh** leg | Terminates TLS with the harness certificate and observes at socket level (§7.2.3); the only place `_cc_to_responses` output (P13, P17) can be seen, and the only recorder on the refresh leg's stack |
| botocore endpoint override | `bedrock` | Points the client at the local recorder rather than AWS; observes the Converse payload **after** the transport's `modelId`/`stream` pops (P18) |

Each records every request in full: method, scheme, host, path, **query**, **headers with
original casing and order**, raw body bytes, arrival timestamp, and — for containment — **the
peer port of the accepted connection**, which is the join key against the proxy's tunnel log
(§5.2.1). The routing fields are not decoration: §3.3.5 asserts on them, and on Azure they carry
the only difference between two otherwise identical requests. Each replays
scripted responses: SSE streams, error statuses, Cloudflare blocks, empty responses,
context-too-large rejections, and disconnects at each of §6.3.1's four injection points.

**A disconnect delivers what was written before it, then drops — on every platform.** Three of
§6.3.1's four points are *post-emission*: their oracle is what the client already received, so a
recorder whose disconnect loses the bytes before it would turn "after text has been emitted" into
"before any downstream byte" and test the wrong row. `Reply.abort()` therefore ends the connection
with `transport.close()`, whose asyncio contract is that queued data is flushed first, and **not**
`transport.abort()`, whose contract is that it is lost. The response is still left unfinished — no
chunked terminator, no `[DONE]` — so the client reads a truncated stream and then end-of-connection.

*Why this is not a POSIX detail.* The first version called `transport.abort()` and passed on Linux
and macOS, where a write reaches the kernel immediately and the transport's queue is empty when
the abort runs. Windows' Proactor loop sends each write as an overlapped operation completed on a
later loop iteration, so the body was still queued and the client saw headers and nothing else —
deterministically, on the Windows leg (KBR-189). The same loss is reproducible on Linux by queuing
more than the socket accepts: of a 32 MiB body, 23,764,432 bytes were still queued at the abort
and exactly that many never arrived. `test_an_abort_delivers_every_byte_written_before_it` builds
that state on every leg — it holds its client back from reading until the abort has run, so the
queue's size does not race a concurrent reader. *Why not "signal that the frame was written, then
abort"*, which the ticket first suggested: `await write()` has already returned by then — written
to the transport is not on the wire.

*Two things `abort()` no longer does on its own, stated so nothing is built on them.*

- **It does not refuse later writes — aiohttp does.** The response stays unfinished only because
  aiohttp will not write to a closing transport, so a responder writes nothing, through aiohttp or
  around it, after `abort()`. asyncio alone does not guarantee it: after `close()` the Proactor
  loop drops a later write, but the selector loop (Linux, macOS) still queues and delivers one if
  its queue is non-empty at that moment. `test_a_responder_can_abort_mid_stream` asserts the reply
  is incomplete as well as `[DONE]`-free, so a response aiohttp was allowed to finish fails it.
  It does **not** test aiohttp's check itself: that case's small writes leave the selector queue
  empty at the abort, where asyncio drops later writes on both loops without aiohttp's help.
- **Its return is not the disconnect.** The drop completes when the reader has taken the queued
  bytes. A scripted injection (T-B4) that releases a barrier on `abort()` returning must not make
  the reader wait on end-of-connection before that barrier.

*The cost, and why it is the opposite of §7.3's rule.* §7.3 aborts the proxy's connections
because that is **teardown**, where a flush serves nothing and a TLS `close()` waits out
`ssl_shutdown_timeout`. This is a **scripted disconnect inside a test**, where the flushed bytes are
the evidence. The price is that a peer which is alive but has stopped reading holds the connection
open — past `recorder.stop()`, whose force-close is a no-op on a transport already closing — until
it reads or disconnects. No reader of this recorder does that: the bridge's HTTP client reads to
the end, and the raw-socket probes close their socket, which fails the pending send and
force-closes the transport. A TLS recorder (T-B2) that reuses `Reply` inherits the wait for the
peer's `close_notify` on top — bounded at 30 s on 3.11+, unbounded on 3.10 — and should decide for
itself.

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

**One thing no bridge-side judgement checked — until KBR-155.** §4.3 C3's emptiness oracles are
keyed on the **upstream** format: `_is_empty_cc_response` for non-streaming,
`translator.response_was_empty` plus the `has_content` byte flag for Chat Completions streams —
and, when this recorder landed, **nothing at all** for a native Anthropic Messages stream, which
the native branch forwarded to the client byte-for-byte. That was a property of the product, not
of the harness: an upstream returning a well-formed but contentless Anthropic stream reached
Claude Code with none of the retry the Chat Completions path has. KBR-155 closed it with the
preamble hold of §11 Q14(b): `bridge/preamble_hold.py`'s `PreambleHold` is now the fourth oracle,
judging the native stream by its release rule before any byte is written. A recorder's scripted
Anthropic SSE success must therefore carry content — a contentless one is an empty reply and costs
the retry ladder. **The four oracles did not agree on every shape**, and this was recorded
rather than smoothed over: the non-streaming judgement `_is_empty_cc_response` called a reply of only
`server_tool_use` or `web_search_tool_result` blocks empty (D1 does not) and retried a `max_tokens`
reply with no content (D3 does not), and a translated stream that yields no chunk and no
`finish_reason` never reached its buffered-finish check, so it was written as an empty `200`. None
of the three was KBR-155's; each was a candidate for its own ticket. **All three closed 2026-09-14,
under KBR-235 for `/v1/messages` and KBR-250 for `/v1/responses` and Gemini**: the no-finish
translated stream takes the same ladder as any empty reply and ends it in the D4 error (owner
decision, recorded under Q14 below); `_is_empty_cc_response`'s Messages-shaped arm now mirrors
`PreambleHold`'s release rule — D1 judges by block type, D3 ends the non-streaming ladder at once
with the `400` at every gate the judgement feeds. The remainder is recorded rather than smoothed
over: the translated route's **exhaustion is split** — a contentless reply *with* `finish_reason`
still exhausts into the M12 fallback text while a no-finish stream exhausts into the D4 error,
until the owner unifies them; the translated route has **no streaming D3** — a `max_tokens`
finish-chunk empty stream is still retried and fallback-ized, the route's pre-existing
finish-chunk behaviour this change deliberately did not touch. KBR-250's per-route application
on `/v1/responses` and Gemini is in-stream SSE error events rather than a JSON 502 response (the
branches `sr.prepare(request)` at the top, so lazy-prepare is out of scope); the messages
branch's post-emission arm is deliberately NOT mirrored on these two routes because
`ResponsesTranslator` and `GeminiTranslator` set `response_was_empty` against the whole
response's accumulated content, not the finish chunk's content, making the post-emission arm
structurally unreachable — the design decision is recorded in the Q14 amendment below. The Chat Completions-shaped arm of `_is_empty_cc_response` keeps its `.strip()`
judgement — D1/D3 are decisions about Messages-format replies.

#### 7.2.2 What T-B1 settled — the provider-session recorder

**Delivered by T-B1 ([KBR-40]) in `tests/harness/provider_recorder.py` and
`tests/harness/provider_aiohttp.py`**, registered as `provider_aiohttp` with a `CONFORMANCE_CASES`
row naming `OLLAMA_CHAT` and the **Chat Completions** inbound route — named rather than derived,
because `OLLAMA_CHAT` has no inbound route of its own (§7.5.1).

**It subclasses T-W4's recorder rather than being a second server.** §7.2.1 catalogues six ways an
aiohttp recorder can look correct and lie, and a second implementation is a second chance to get
each of them wrong — §7.3's own argument, that "two proxy implementations is how two harnesses come
to disagree". What differs between the two recorders is **vocabulary**: the format served, the
suffix that selects a reply, and what a minimal success looks like. Those three are overridden and
nothing else, so both recorders are judged by §7.2.1's fourteen checks over the same capture path.
The claim that costs is `recorder_conformance`'s: running those checks against a subclass that
overrides none of the capture path proves the overrides did not break it, not that a second
implementation agrees. **What carries that weight instead is a falsification case against the
overrides themselves** — a Chat Completions body and an SSE stream, each driven through the real
adapter, which reads nothing out of either.

**The OAuth token leg is served by this recorder and is *not* an `UpstreamTransport`** — §7.5's open
question, decided. `bind()` must return `(adapter, provider_config)` and the conformance check drives
a request through a real `BridgeServer`; the login leg has neither an adapter nor a bridge, so a
transport for it could not satisfy the interface it joined. A seventh `WireFormat` was the other
option and would mutate a contract six Epic A readers consume (T-W2, [KBR-25]) to add a value **no
projection can read**: §3.3.1 pairs every format with a wire reader, and a form-encoded token grant
is not an LLM request. So the recorder dispatches the leg by path suffix, `WireFormat` stays closed
at six, and `oauth_token_endpoint()` is that leg's `bind()`.

**Scope, after KBR-161: the login leg here, the refresh leg in T-B2.** §5.5's table records that the
refresh leg now runs on the adapter's impersonating `curl_cffi` session, which an aiohttp recorder
cannot observe. Note the consequence §5.5 already states from the other side: the `curl_cffi` leg
"fires on every subsequent request, so that is the transport the harness must exercise first". T-B1
therefore covers the **less** urgent of the two legs, deliberately — it is the one its stack can see
— and T-B2 inherits the other. **Two constants named `OAUTH_TOKEN_URL` exist**, one per leg
(`kitty.auth.openai_oauth` and `kitty.auth.oauth_session`); they are identical strings and unrelated
variables, so a seam that swapped the wrong one would send a real request to `auth.openai.com`.

**A one-format transport has one teardown check, and that is measured.** §7.5.4's row 4 — a declared
format that was never under test — needs **two** served formats to stay silent: the recorder answers
by path suffix, so the adapter parses the reply and the capture list comes out complete. With one
served format the same mistake takes the fallback instead. Measured, against a transport whose
adapter posted elsewhere and against one that posted at the OAuth endpoint: **4 captures and a
10-second timeout each, 72 seconds including the teardown that waits out the retry ladder**. Both are
caught loudly by the conformance check's first assertion, so a second teardown pass here would be an
assertion no defect could falsify — which is what §7.5.4 found and removed in T-W8. The same
measurement is why the reply-shape falsification cases are driven through the adapter and not
through the bridge: the defect is caught either way, and one way costs milliseconds.

**The OAuth endpoint is excluded from the declared-format claim by name, not by silence.** A token
grant is answered before any format lookup — it has no `WireFormat` and must not be reported as a
fallback — so `assert_teardown_clean()` passes over it, and a test pins that exclusion so it cannot
be mistaken for a hole.

[KBR-40]: https://shelpuk.atlassian.net/browse/KBR-40
[KBR-25]: https://shelpuk.atlassian.net/browse/KBR-25

#### 7.2.3 What T-B2 settled — the curl_cffi recorder, harness TLS, and the refresh leg

**Delivered by T-B2 ([KBR-41]) in `tests/harness/curl_recorder.py` and
`tests/harness/curl_cffi.py`**, registered as `curl_cffi` with a `CONFORMANCE_CASES` row naming
`OPENAI_RESPONSES` and the `/v1/responses` inbound route — the tidy case KBR-31's final review
round opened by adding that mapping to `protocol_for`.

**It subclasses T-W4's recorder, for §7.2.2's same reason.** The vocabulary is overridden —
the served format (`OPENAI_RESPONSES`), the suffix table (`/responses` and `/oauth/token`), and
what a minimal success is, which for this transport is an OpenAI Responses SSE stream carrying at
least one `output_text.delta` (the adapter always sends `stream: true`, P17, so a non-streaming
JSON success would be the wrong shape and cost the ladder). Everything the conformance suite
judges is inherited unchanged.

**TLS is terminated by the recorder with the harness certificate.** The `certs` fixture and
`server_ssl_context` from :mod:`harness.connect_proxy` are reused, so one throwaway CA signs every
TLS endpoint the harness presents. The recorder's bind site takes the resulting `SSLContext`
directly; nothing in `src/kitty` learns about it, and the client side trusts the same CA through
the adapter's own `CODEX_CA_CERTIFICATE` seam (`_resolve_ca_cert_path`, matching Codex CLI's
`custom_ca.rs`).

**§7.2.1's connection-logging limitation is resolved at socket level, by owner decision.** The
recorder's `Server` subclass logs every accept in the **protocol factory**, which asyncio calls
before the TLS handshake begins — so a connection that opens and then fails negotiation still
produces a `ConnectionRecord`. The peer port is not visible to the factory (the accepted socket
has not yet been handed to anything), so the record is created with `peer_port = -1` and filled in
by `connection_made` on handshake success. A record that keeps `-1` is honest evidence: the
connection existed, no peer ever identified itself over TLS, and §5.2.1's bypass shape is a
connection that carries no request regardless of what it sent. `check_connection_logged` is not
weakened by this — the conformance driver's probes all complete the handshake, so every port it
opens is filled in before the check runs; the failed-handshake shape has its own falsification
case against `recorder.connections` directly.

**The OAuth refresh leg is served by this recorder and reaches it through a second seam.** The
refresh leg addresses :mod:`kitty.auth.oauth_session`'s `OAUTH_TOKEN_URL`, which is **a different
constant from the login leg's** (`kitty.auth.openai_oauth.OAUTH_TOKEN_URL`), so T-B1's
`oauth_token_endpoint` seam does not apply — §7.2.2's warning is load-bearing here.
`oauth_refresh_endpoint(recorder)` follows T-B1's shape: read the constant, swap, restore in a
``finally``, raise `AttributeError` when the name is gone. A transport that swapped the wrong
constant would send a real request to `auth.openai.com`, which is the failure this guard exists
to prevent.

**The serving leg reaches the recorder through a module-constant swap, not a base-URL redirect.**
`OpenAISubscriptionAdapter` never reads `provider_config["base_url"]` (§7.5.2's custom-transport
rule) — its upstream path is `_CODEX_BACKEND_URL`, a module constant on
:mod:`kitty.providers.openai_subscription`. `bind()` returns the recorder's own URL in place of
that constant, the same read-swap-restore shape as the refresh leg's seam, so a request the
product meant for `chatgpt.com` reaches the recorder instead. **`bind()` is therefore not
idempotent across recorder restarts**, and every call after the first returns the same adapter
instance (it owns two session pools whose lifetime :meth:`stop` must close — the reason
§7.5.2's "one adapter per transport, reused" applies doubly here).

**Redaction is the transport's decision, not the capture type's.** `CapturedRequest` stays a
raw-bytes carrier — T-C6's malformed body must reach a reader untouched — and the refresh leg's
`client_secret` and `refresh_token` are the only credentials this recorder's traffic carries in a
body. The masking lives on `CurlCffiTransport.captures`: applied per capture, returning a fresh
object, so the recorder's own list stays raw by construction. **Not a `redact_body` method on the
`UpstreamTransport` Protocol, deliberately**: the Protocol is `@runtime_checkable` and three
suites assert `isinstance(..., UpstreamTransport)` on instances, so a Protocol member must be
implemented by every instance — the two existing transports would each need a no-op definition,
which is the opposite of "inherit without change". The first draft's protocol-method shape was
falsified by that contract; what survived is the inlined helper.

[KBR-41]: https://shelpuk.atlassian.net/browse/KBR-41
#### 7.2.4 What T-B3 settled — the botocore endpoint-override recorder

**Delivered by T-B3 ([KBR-42]) in `tests/harness/botocore_recorder.py` and
`tests/harness/botocore.py`**, registered as `botocore` with a `CONFORMANCE_CASES`
row naming `BEDROCK_CONVERSE` and the **Chat Completions** inbound route — named
rather than derived, because the bridge *translates* and the Bedrock adapter is
a CC-wire adapter (§7.5.1, §3.2.3).

**It subclasses T-W4's recorder, for the same reason T-B1 does.** The six ways an
aiohttp recorder can look correct and lie (§7.2.1) are capture-path facts; a
second implementation is a second chance to get each of them wrong. The
differences from T-W4 / T-B1 are vocabulary only — the format served, the
suffix table that dispatches the reply, and what a minimal success looks like —
and that is the same three-override recipe. The capture path is inherited
unchanged; `check_peer_port` is satisfied because T-W4's `capture()` already
calls `_peer_port(request)`, and an override here would mirror the path T-B1
deliberately avoided. The conformance suite is run unchanged against the
subclass; what proves the overrides themselves is a hand-written round-trip
through the pinned `botocore.eventstream.EventStreamBuffer` parser, and the
`TestThroughARealBridge` class drives a real `BridgeServer` against the
recorder end-to-end. (A live `boto3.client(...).converse_stream(...)`
round-trip is **not** exercised in the suite — boto3's synchronous urllib3
blocks the asyncio loop the recorder's aiohttp server runs in, and every such
test times out at 60 s without reaching the recorder's capture method.)

**The product seam is `provider_config["endpoint_url"]`, not `AWS_ENDPOINT_URL`.**
The env var (added in boto3 ≥ 1.28) is process-global ambient state, and CI
runners carry user-set `AWS_ENDPOINT_URL` values that would hijack the test
fixture in the same way they hijack production traffic. The `provider_config`
channel is the bridge's own (§7.5.2): per-profile, visible in the schema, and
the same precedent `provider_config["base_url"]` already sets for T-B1. The
edit is five lines in `BedrockAdapter._get_boto3_client` and adds nothing when
the key is absent.

**The streaming reply is AWS EventStream binary, not SSE.** Bedrock's
`converse_stream` returns `application/vnd.amazon.eventstream`; boto3 parses
each frame's `:event-type` header to a member key (`messageStart`,
`contentBlockDelta`, `contentBlockStop`, `messageStop`, `metadata`) and parses
the JSON payload against the member's shape. There is no public serializer in
the pinned `botocore.eventstream` module — verified against the installed
source — so the harness ships its own encoder following the spec every AWS
SDK implements, with CRC32 per the pinned `binascii.crc32(data) & 0xFFFFFFFF`.
The encoder is validated by round-trip through the pinned parser, by a
bad-CRC falsification case (`test_a_streaming_reply_with_a_bad_message_crc_fails_to_decode`),
and by a bridge-driven end-to-end path (`test_a_streamed_request_via_the_bridge_yields_finish_reason`).

**The harness rule (§1.4) requires a falsification case against the
recorder's own overrides** — the conformance suite runs unchanged over the
subclass and so cannot tell a wrong encoder from a working one. The shipped
falsification is the bad-message-CRC test (`test_a_streaming_reply_with_a_bad_message_crc_fails_to_decode`)
plus an encoding-determinism check (`test_encoding_is_deterministic`) and a
construction-refusal check (`test_a_format_this_recorder_does_not_serve_is_refused_at_construction`).
All three run in the suite; the conformance suite judges the capture path,
these cases judge what the recorder writes back.

**A non-streaming reply carries content, not the empty-reply ladder.** §7.2.1
already names the 80-second `asyncio.sleep` cost that an empty or mis-shaped
upstream reply triggers; a minimal success that satisfied the bridge's emptiness
judgement is the contract every recorder obeys, and T-B3's is no exception.

**No `aclose` and no session.** `_get_boto3_client` builds a botocore client per
call (KBR-190) — `BotocoreTransport.stop` has nothing here to release, and the
`tests/test_provider_bedrock.py` regression pins that property for this adapter
specifically.

[KBR-42]: https://shelpuk.atlassian.net/browse/KBR-42
