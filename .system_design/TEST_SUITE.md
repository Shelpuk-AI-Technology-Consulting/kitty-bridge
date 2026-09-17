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

Twenty-three request-path rows — the original fourteen (M1–M11, M15, M16 and M17), two
id-synthesis rows added by KBR-195 (M18, M19), and six KBR-194 Gemini-side slot drops on
the Gemini inbound route (M20–M25), plus KBR-184's M26 (the Anthropic-family `metadata`
drop, G31) — one response-path row (M12), and the routing row **M14** (§3.3.5), which is
listed here because the destination is a mutation surface the body cannot show. Twenty-six rows
in all. The former substitution row M13 is **withdrawn** — KBR-5 replaced it with a downstream
error, so it mutates nothing — leaving **twenty-five live** bridge-level rows.

| # | Mutation | Site | Trigger | Why it is necessary |
|---|---|---|---|---|
| M1 | Replace `model` with the profile's model, then provider-normalise it | `BridgeServer._normalize_model` | Always, when the profile sets a model | This is the product. The agent asks for one model; the profile decides what actually runs. **"Always" was false on one route until KBR-160.** On `openai_subscription`'s Responses path the adapter built the shipped body out of `cc_request["_original_body"]` — the *raw inbound* body — so M1 ran, computed the profile's model, and had it discarded one layer down; the agent's model shipped, silently. `OpenAISubscriptionAdapter._prepare_responses_body` now reads `cc_request.get("model", "gpt-5.4")`, the same expression its sibling `_cc_to_responses` already used, which is why the other routes into that provider were never affected. The row's **site is still `_normalize_model` alone**: the adapter *consumes* M1's output, it does not perform M1's mutation, so this closed a discrepancy between the register and the code without changing a single register field. What the episode does show is that `paths=(envelope.model,)` is a claim about the §3.2.3 capture boundary that a downstream consumer can silently break, and that nothing but a test at that boundary notices. |
| M2 | Translate the agent's protocol → Chat Completions | `MessagesTranslator` / `ResponsesTranslator` / `GeminiTranslator` `.translate_request` | Provider has `use_native_messages == False`, or the agent speaks Responses/Gemini | The upstream speaks a different protocol. Skipped entirely for native-Anthropic providers such as `zai_coding`. **Since KBR-222 the Messages-ingress half of this row carries non-text user blocks instead of dropping them:** an `image` block ships as a CC `image_url` content part (a `data:` URI for a base64 source, the URL verbatim otherwise; any other source type still drops), a `document` block rides the `_documents` internal key to the Anthropic family (P1 below carries it; the restore is `AnthropicAdapter`'s), and non-tool blocks beside a `tool_result` become one trailing user message — non-tool blocks after the tool results, where the M9 fallback (`server._convert_native_to_cc_format`) keeps its own text-first placement. Both converters share one user-content builder (`build_user_content_message`), so the fallback's retry carries images and documents exactly as hop 1 does — KBR-178's lesson applied: a mapping hop 1 learns and the fallback misses is lost again on the tool_use format-error retry. The fallback stays otherwise minimal (KBR-224's precedent). Text-only turns are byte-identical to the pre-KBR-222 output — a joined string — so the common case changes no shape. The Responses and Gemini translators handle no image parts at all, so on those ingress routes nothing here changes. |
| M3 | Truncate a tool result over 50,000 chars (`_TOOL_RESULT_TRUNCATION_LIMIT`) | `_truncate_oversized_tool_results` / `_truncate_oversized_responses_outputs` | A single tool result exceeds the limit | A single oversized result can exceed the model's window on its own. On `openai_subscription`'s Responses path the shipped body is built from the raw inbound body, so the mutation is performed there by `_truncate_oversized_responses_outputs` on the Responses-shaped `input` itself — same limit, same notice text; the CC-shape site truncates a copy that never ships on that route (KBR-169). |
| M4 | Truncate a tool result over the same limit, again, inside compaction | `_compact_messages` step 1 | Compaction ran **and** a `role: "tool"` message's string content exceeds the limit | Second pass, CC-shape only. Distinct from M3: M3 is unconditional pre-processing, M4 fires only once compaction is already engaged. |
| M5 | Compact the message history | `_apply_compaction` → `_compact_messages` | Serialized messages exceed the model-derived budget | Without it the upstream rejects the request outright. On `openai_subscription`'s Responses path the decision is carried onto the wire by `_prune_compacted_responses_input`, which prunes the raw body's `input` to the surviving conversation — a consumer of the compaction, not a second policy, but it adds two selection rules of its own: a riding item (`reasoning` and types the translation skips) is owned by the next assistant message, and a riding-run + `function_call`-run + `function_call_output`-run group moves atomically, because the CC grouper can split one wire hop across the pruning boundary and the Responses API rejects both a call without its preceding reasoning item and a reasoning item without its following item (community-verified 2026-09-14, not live-probed). A final orphan sweep mirrors pairing validation. Survival is matched by message identity against the pre-compaction snapshot, so an edited message fails conservative — items drop, never an unprotected body ships (KBR-169). **Since KBR-222 the translated route's body can carry base64 media** (CC `image_url` parts and the `_documents` key), so media bytes now count against this budget and against the request-size guard as they always have on the native routes. Head+tail pruning of a media-bearing message silently reverts that message to the pre-KBR-222 loss — its `_documents` entry is addressed by message identity, so it forfeits rather than mis-attaching (the identity invariant is load-bearing: compaction must keep rebuilding `role == "tool"` dicts only, as it does today). G42 records the residue. |
| M6 | Re-compact at half budget and re-send the same backend | `_compact_with_tighter_budget`, called from `_request_with_retry_balancing` and, since KBR-256, from the four streaming ladders (`_stream_messages`, `_stream_responses`, `_stream_gemini`, `_stream_chat_completions`) | Upstream returned 400/413 **and** `_is_context_too_large_error` **and** `_is_oversized_request` | Recovery from a rejection kitty's own budget estimate failed to prevent. **The non-streaming arm is balancing-profiles only** — `_request_with_retry` (single backend) has no compaction recovery — **but since KBR-256 the streaming arms are not**: their gate keys on `recovery_retries < n_backends` with `n_backends = 1` in single-backend mode, so a single-profile streaming 413 recovers too (the row's conditionality holds on every shape; what differs per shape is the recovery's *fallback*, not its availability). Also an I2 exception; see §4.3 C3. **On the streaming ladders the recovery is pre-byte only** (KBR-256): it fires on the status-error branch before any byte reaches the client (`sr is None` on `_stream_messages`; position-as-guarantee on the three eager-prepared routes, where any client write terminates the attempt loop per KBR-183/Q14(a)), and a mid-stream SSE-embedded error keeps the existing PreambleHold break path — the recovery cannot take written bytes back. The streaming recovery is capped at one per attempt ordinal and `n_backends` per request (parity with the non-streaming ladder); a second 413 on the compacted re-POST falls to the standard mark-unhealthy arm; `CompactionFailedError` fails over without marking (KBR-5 parity). Still wire-ineffective on the custom-transport branches — non-streaming (`openai_subscription`'s Responses path) **and** streaming: the tighter recompaction rewrites only the CC copy, and the retry re-prepares the shipped body from `_original_body` (KBR-169 class; R7 of KBR-256 leaves those branches untouched). |
| M7 | Drop orphan `tool_result` blocks | `_validate_tool_call_pairing` / `_drop_orphan_responses_tool_outputs` / `_prune_compacted_responses_input` | A `tool_result` has no matching `tool_use` after compaction | An orphan triggers upstream error 2013 and fails the turn. On `openai_subscription`'s Responses path the rule runs on the Responses-shaped `input` with the same order-sensitivity — an output ships only when a preceding `function_call` declared its `call_id`, and a missing `call_id` is undeclared (dropped, where the translator's `KeyError` used to render a 500) — and the selector re-runs it after compaction, because pruning an earlier wire group can orphan a later output (KBR-169). |
| M8 | Add a thinking-carrier block and re-send the same backend | `_repair_thinking_roundtrip` / `_with_thinking_carrier` | This backend rejected this transcript for a thinking round-trip mismatch (issue #32) | Avoids one rejected round-trip per turn against backends that require it. Also an I2 exception. |
| M9 | Convert a native Messages body to CC format and re-send the same backend | `_convert_native_to_cc_format`, then a re-run of `_normalize_model` and `normalize_request` | Upstream returned a `tool_use` format error on the native path | Fallback that keeps the session alive rather than failing the turn. Also an I2 exception. |
| M10 | Inject the model from the URL path into the body | `_handle_gemini` | Gemini protocol only | Gemini carries the model in the path, not the body; `_normalize_model` needs it in the body to override it. |
| M11 | Force `stream: False` | `_handle_gemini` | Gemini protocol, non-streaming `:generateContent` | The Gemini translator defaults `stream=True`; the non-streaming endpoint must not open an SSE stream. |
| M12 | Substitute fallback assistant text | `_EMPTY_ASSISTANT_FALLBACK_TEXT` in `bridge/messages/translator.py` **and** `bridge/responses/translator.py` | Upstream returned an empty response | **Response-side**, not part of the twelve request-path rows. **Never on a streamed `/v1/messages` reply from a Messages-wire upstream**, native or translated: that stream is forwarded as the upstream sent it, so there is no translator to substitute anything (KBR-227 — before it, the translated Anthropic-wire routes pushed Anthropic SSE through the Chat Completions chunk translator and Claude Code received an empty reply). Empty replies on that branch are KBR-155's to retry. The non-streaming reply and the other inbound protocols still translate, and still can fire this row. |
| ~~M13~~ | **Withdrawn — no longer a mutation.** Was: discard the conversation and substitute a `[Kitty Bridge: …]` user message. | `_compact_messages` / `_apply_compaction` post-condition | No non-system message survives | **Closed by KBR-5.** The post-condition now raises `CompactionFailedError` and the handler returns a protocol-native 400 downstream; nothing is substituted, so there is no mutation left to register. The row is kept struck through rather than deleted so a reader of finding F3 can still find it. **The trigger recorded here was wrong** — see F3. |
| M14 | **Replace the destination entirely** — scheme and host are built from the profile by `build_base_url()`; the path by `get_upstream_path(_route_model(cc_request))` — the **request's normalized model**, which is the normalized profile model when there is one and the agent's model when there is not. `_route_model` is the single place that answers this; the auth scheme (P9/P20), the thinking carrier and the choice to forward a `/v1/messages` stream unchanged (`BridgeServer._serves_messages_wire`, KBR-227) read it too, and the adapter reads the same key for the body (KBR-127 — it was the raw profile model, so path and body could route differently; and on `openai_subscription`'s Responses path the adapter read the *inbound* body's model instead until KBR-160, which was harmless for routing only because that provider posts to a fixed URL and derives no header from the model). Base and path are then **composed** by `ProviderAdapter.compose_upstream_url`, not concatenated (KBR-143). | `BridgeServer._build_upstream_url` | Always | The agent addressed a loopback bridge; the request has to reach the real provider. Listed because **the destination is a mutation surface the body cannot show**: on Azure an identical body sent to the wrong deployment path is a different request entirely (§3.3.5). **The query is part of the mutation, not a passenger** (KBR-143): the endpoint joins the *path* component and the two queries merge, the endpoint's parameters winning a name clash and the base URL's others surviving unaltered. A row naming only "path" would let an oracle derive `route.query` and still not know which side owns a clash. The base URL's fragment is carried through and never sent, since no HTTP client puts one on the wire — so an oracle deriving `route.*` from the profile must expect it on the composed URL and absent from the request line. **The composed URL is redacted before it is echoed** into the 404 diagnostic or a pre-flight failure (`redact_url_for_display`): query values and the fragment are masked, which is an I2-adjacent containment property, not a fidelity one — nothing about the request changes. The composition helper is shared with `kitty.validation.validate_api_key` and `OllamaCloudAdapter._build_url`, but **this row's site is the bridge alone**: pre-flight's probe is not a request the agent made, and the register describes what happens to the agent's request. |
| M15 | Rewrite a string `input` into the single-item list form `[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": <s>}]}]` | `normalize_responses_request` (`bridge/responses/translator.py`), called from `_handle_responses` before the body forks | Always | OpenAI's `CreateResponse` defines the two forms as the **same request**: `input` is `oneOf` a string (*"a text input to the model, equivalent to a text input with the `user` role"*) or an array, and everything downstream reads the array. Fires on every request reaching the handler; a body already in the array form meets the row with a **no-op** rather than avoiding it, so there is no complement state for §3.3.2 assertion 2 to arrange, which is why it is unconditional. Listed rather than omitted because the rewrite is real bytes at the `curl_cffi` boundary of §3.2.3, where `_original_body` **is** this body; the projection cannot express the difference, so the row takes §3.3.1a's escape for P16's reason. **KBR-144.** |
| M16 | **Strip every `cache_control` cache breakpoint** — from tool declarations, from system blocks, from message content blocks, and the top-level automatic-caching form (projected to `envelope.extra[cache_control]`; KBR-263 closing G38) | `MessagesTranslator.translate_request` (`bridge/messages/translator.py`) | The upstream wire is not native Messages — i.e. the provider does not declare `use_native_messages` | The translator rebuilds the body for Chat Completions and discards the breakpoint as it goes: system blocks are joined into one string, tools are rebuilt as `{name, description, parameters}`, and content blocks are rebuilt. **The discard is the translator's choice, not a limit of the format** — OpenRouter's Chat Completions dialect carries `cache_control` on content parts, and `openai/openai-openapi` puts `prompt_cache_breakpoint` on Chat Completions content parts (GPT-5.6+, with a request-wide TTL only, so it cannot say `ttl: 1h`). An earlier reading of OpenAI's guide took the latter to be Responses-only; KBR-199 checked the schema. **One carrier escapes the strip**: a breakpoint on a block nested inside a `tool_result`'s list content is copied through both hops — pinned at the translator by KBR-198 and across both hops by KBR-199. That is outside this row's paths by design — §3.3.1 residualises a nested breakpoint. Whether Anthropic honours one at that depth is not established: its SDK types accept `cache_control` on a block inside `tool_result` content, and its docs' sub-content rule names citations only. **This is the row whose cost is largest and least visible.** Anthropic prices a cache read at 0.1x base input, so a stripped breakpoint re-bills the agent's stable prefix — system prompt, tool definitions, history — at **at least** ten times its cached rate, on every turn, with nothing in the product saying so. Registered rather than left to the residual precisely so the oracle reports it as a *claimed* delta attributable to this site; §3.3.1 records why the declared-ignored alternative was rejected. **The trigger is not `Always`, and the row is still exempt from §3.3.2 assertion 2** — the same shape as **M2**, which carries this identical trigger. The native passthrough branch (`BridgeServer`, `use_native_messages`) shallow-copies the inbound body, so breakpoints do survive there — except on the `tool_use` format-error fallback, which re-converts through `server._convert_native_to_cc_format` and strips most of them: the `system` carrier (on the `forwards_thinking_signature` adapters) and `document` blocks survive via the converter's `_anthropic_system` / `_documents` carriages and the rebuild's verbatim restores, while tool, image, message-text, `tool_use`, `tool_result` (nested included) and top-level breakpoints are lost — pinned by KBR-200; but that complement is a property of the **route**, chosen by the profile, not of the request, and assertion 2 asks for an *input* that fails the trigger. A corpus entry cannot arrange a different provider. The native route's guarantee is therefore proven where it belongs — as product behaviour, in epic KBR-197 — rather than by a corpus complement nobody could author. Top-level `cache_control` (Anthropic's automatic caching) is **now** this row's: it lands in `envelope.extra[cache_control]` and P1's internal-key strip does not touch it. The translator does not copy it either, though, so on the translated route that `extra` delta is **claimed (KBR-263), closing G38** — the product fix is still epic KBR-197's boundary. **KBR-228 part B restored the system carrier on the signature-binding routes**: on `anthropic`, `custom_anthropic` and `zai_anthropic` (`forwards_thinking_signature`) the adapter re-attaches the agent's system blocks — breakpoints included — from the internal carriage, so the system-path claim above no longer reaches those wires (the CC-intermediate strip is unchanged, and the tool and message carriers are not restored; `tests/providers/test_anthropic_cache_breakpoints.py` was rewritten to the new wire). **KBR-167**; the product-behaviour suite for the same defect is epic KBR-197.|
| M17 | **Strip the thinking Anthropic rejects and re-send the same backend** — `thinking` and `redacted_thinking` blocks at or before the turn the rejection names, plus any unsigned `thinking` block anywhere; up to three strips per serialized body, the third removing every thinking block | `_recover_rejected_thinking` / `_strip_thinking_blocks`, called from `_make_upstream_request` and `_stream_messages`, and — since KBR-232 — from the three non-Messages stream handlers `_stream_responses`, `_stream_chat_completions` and `_stream_gemini` | The upstream rejected this request's thinking signatures with a 4xx (`_is_thinking_signature_error`) and the body had thinking to strip | api.anthropic.com verifies every thinking block it is sent back, and kitty's history fails that check: the translator drops signatures, **P5e** injects unsigned blocks (empty or reasoning-bearing — toward a signature-checking upstream every P5e block triggers this row), and M3–M7 and M9 edit the prefix a signature is bound to; the M8 carrier is also unsigned, but it fires only on the DeepSeek/Kimi wording, so its part here is theoretical. **Probed live on 2026-09-13** against `claude-sonnet-5`, `claude-opus-4-6`, `claude-fable-5-1` and, in manual `enabled` mode, `claude-opus-4-6` and `claude-haiku-4-5`: a missing signature or a P5e block gets `400 ...thinking.signature: Field required`; an altered one, or an edited earlier message under the prefix check (default for accounts from 2026-08-31), gets ``400 ...Invalid `signature` in `thinking` block``; the stripped history succeeds, including a manual-mode `tool_result` tail. Through a real bridge on unfixed `main`, all four cases tried — `anthropic` translated and `custom_anthropic` native, each non-streaming and streaming — returned that 400 to the agent; with this row all four answered. **Why targeted, not everything (owner decision, 2026-09-13):** the damage is not transient. Claude Code re-sends the full history every turn and kitty re-compacts every turn, so a broken block returns every turn; stripping everything would erase the model's reasoning on every later turn, fresh reasoning included. A live probe on `claude-fable-5-1` showed a block produced *after* a strip is valid, and survives when only the older broken blocks are removed (removing thinking from the front of the history is allowed; re-sending the broken block fails the request). **Why counted per serialized body:** a failover rebuilds the body with its thinking restored, and the next backend must get its own recovery rather than a quarantine for kitty's history. **Known costs, recorded rather than hidden:** each request whose history is still broken pays up to three rejected round-trips — whether a rejected 400 counts toward rate limits is not documented; each switch between stripped and unstripped prefixes re-writes the prompt cache (1.25x); and the only operator signal is a WARNING log line plus the per-backend `thinking_stripped` counter in `/stats` (KBR-228). **After KBR-228 part B** the restore fixes the unsigned rebuild on the signature-binding routes, so a strip there means the signed prefix was *edited* (M3–M7, M9) — not that the carriage is broken. **The alternative not taken:** Anthropic's `thinking.block_binding.prefix_mismatch_behavior: "drop_block"` (beta header `thinking-binding-controls-2026-08-01`) drops only failing blocks with no rejected round-trip — live-probed: it handles an edited prefix but still 400s on a missing or altered signature, so a strip is needed regardless, and it would add a beta header kitty does not send today. Gated by body shape, not by wire: a Chat Completions body carries no thinking blocks, so there the strip finds nothing and nothing is retried — which also means a Chat Completions gateway in front of Anthropic gets no recovery. Recovery exists on every non-streaming handler and on every Messages-wire stream: KBR-227's `/v1/messages` passthrough and, since KBR-232, the Responses, Chat Completions and Gemini streams (whose loop runs wider by the strip budget, so a strip gets its attempt back as on `_stream_messages`). The strip runs after the M8 carrier and removes it too, and on the streaming path the #32 repair is skipped for a stripped body, so the two cannot take turns. A turn left empty keeps `content: []`, which the API accepts, so message indices never shift. Not a backend fault: the backend is not marked unhealthy — neither after a strip nor when a rejection outlives recovery (the cap, a stream's last attempt, or nothing left to strip). That history fails on every pool member alike, so cooling one would only take a healthy backend out of rotation; a non-streaming pool re-selects a backend — possibly the same one, since none was cooled — and stops after two 400s, as for any bad body; a stream surfaces the error to the agent at once (PR #113 review). Also an I2 exception, like M8 (§4.3 C3). |
| M18 | Synthesise a fresh `call_<uuid>` id when the inbound Gemini `functionCall` carries no `id` | `GeminiTranslator._translate_content` (functionCall branch) | The inbound Gemini `functionCall` carries no `id` | KBR-195. Chat Completions requires a tool-call id, so when the client omits one the translator mints it — the delta is a synthetic id upstream where the Gemini reader projected absence. Conditional, because the complement (a corpus entry whose `functionCall` carries an `id`) is plainly writeable and arrives with T-D5. Kept distinguishable from M19 by `paths` (`id` vs `tool_use_id`) — the axis `test_no_two_rows_are_indistinguishable` keys on — not by the site, which both rows share. |
| M19 | Synthesise a fresh `call_<uuid>` tool-call id when the inbound Gemini `functionResponse` carries no `id` | `GeminiTranslator._translate_content` (functionResponse branch) | The inbound Gemini `functionResponse` carries no `id` | KBR-195. M18's tool-result twin: the synthesised id lands on the tool message's `tool_call_id`, not on the call's `id`, so the row anchors at the other field. Kept distinguishable from M18 by `paths`, not by the site. |
| M20 | Drop the Gemini `systemInstruction` `Content` role — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` — the same trigger M2 and M16 carry | KBR-194 gave the Gemini reader a slot for the role the source `Content` published, so the drop is now a positive delta at `conversation.system_role` rather than an invisible normalisation. §3.3.1a's path-table cell named M2 as the claiming row until KBR-195; M2 takes the escape and is never path-matched, so this row is the actual claim. |
| M21 | Drop Gemini's NON_BLOCKING calling toggle on a tool declaration (`behavior`) — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. `_translate_tools` discards the field. Anchored at the field, not the whole tool — a coarser anchor would claim a deleted tool description, one of §3.3.1's own falsification cases (§3.3.1a). |
| M22 | Drop Gemini's `thoughtSignature` on a `functionCall` part — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. M8 also lands a delta at this path, but with a RESPONSE trigger (a thinking round-trip rejection); on a plain Gemini→CC request M8's trigger is not met, so the two are distinguishable by trigger, site and the narrower field anchor here. |
| M23 | Drop Gemini's `functionResponse.scheduling` (the NON_BLOCKING response-side toggle) — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. The reader projects `ToolResult.scheduling`; the translation drops it. |
| M24 | Drop Gemini's part-level `videoMetadata` — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. The reader projects the slot on text, `inlineData` and `fileData` parts; the translation drops it. |
| M25 | Drop Gemini's image `displayName` — Chat Completions has no equivalent slot | `GeminiTranslator.translate_request` | Provider does not declare `use_native_messages` | KBR-194. The reader projects `Image.display_name`; the translation drops it. |
| M26 | **Drop an agent's `metadata` off the Anthropic family** — the bridge carries Anthropic `metadata` on the internal key `_metadata`, which only the Anthropic family restores; every other route omits it — `audio`, `function_call`, `functions`, `metadata`, `modalities`, `moderation`, `prediction`, `prompt_cache_options`, `reasoning_effort`, `service_tier`, `user`, `verbosity`, `web_search_options` | `carry_tool_choice_and_metadata` (`kitty/bridge/messages/translator.py`) | Always | KBR-184 / G31 (KBR-214). The bridge-level row: seventeen routes drop the agent's `metadata` by design and permanently because Chat Completions' own `metadata` is a stored-completions tag map, and a bare mapping would put a new field on every request to sixteen third-party providers or reject every turn (product owner's decision, 2026-09-13). The eighteenth route, `openai_subscription`, also drops it — **G26 / P24** claims that one at the provider level. Site = the policy point (`carry_tool_choice_and_metadata` mints the internal key; the restore is per-adapter); the 17-route omission is named once here rather than as 17 per-adapter rows, G28's shape. ⚠️ The future Chat Completions reader (T-A2 / KBR-34) will land CC's own `metadata` (a stored-completions tag map, NOT the Anthropic user-id object) at the **same** address `envelope.extra[metadata]`. `envelope.extra` is keyed by wire key (§3.3.1b) so the address is shared and the meanings are not — the row title and the comment record the split so a future reader or oracle does not conflate the two fields |

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
| P5d | Map `_thinking_adaptive` → `thinking: {"type":"adaptive"}` and `_effort` → top-level `effort`; restore `_thinking_display` → `thinking.display` on the adaptive object here and on P5c's enabled object — **or withhold `display`** on adapters whose `forwards_thinking_display` is false (`minimax_token`, `opencode_go`, `zai_coding`) | same | Those keys present | Passthrough of an agent signal, and verbatim is what keeps the thinking mode cache-safe: Anthropic renders it into the prompt, so any normalisation would cost the prompt cache the way P5c does. **`display` is a mutation only where it is withheld, where the agent's value is not one of the two GA values, or where it arrives with `disabled`.** Until KBR-203 the translator did not carry it at all, so `{"type":"adaptive","display":"summarized"}` shipped as `{"type":"adaptive"}` on every translated route. The translator now carries only `"summarized"` and `"omitted"`, and only with `adaptive` or `enabled`: the beta `"updates"` needs an `anthropic-beta` header kitty never forwards (§4.2 C1), and `disabled` rejects `display` outright, so carrying either would turn a request that works today into a 400. The adapter restores the value where Anthropic's Messages API defines it (`anthropic`; `custom_anthropic` only when a balancing failover re-serializes a request translated for an earlier backend, since its own requests are native) and withholds it where the upstream does not document it: MiniMax's Anthropic-compatible reference names `thinking` but not `display`, its translated route exists because MiniMax rejected Claude Code fields before, and `opencode_go`'s Messages route serves MiniMax and Qwen models. **Restoring it now shows the user the model's thinking**: KBR-227 forwards the streamed reply byte-for-byte, and KBR-228 part A carries the non-streaming reply's thinking blocks — signatures included — through the CC layer to `MessagesTranslator`, so `"summarized"` earns its latency instead of adding it; carrying the agent's signed blocks back upstream is KBR-228 part B. The withholding lands at the address the restore uses, `envelope.extra[thinking]`, which this row and P5c already claim under the same triggers, so no new row is needed — and because `extra` is keyed, never nested (§3.3.1a), `thinking.display` has no finer address to give one. **Native passthrough follows the opposite rule and that is deliberate, not a contradiction**: it forwards the agent's `thinking` untouched (the first-attempt guarantee P5c records), so `minimax_token`'s opt-in native mode and `zai_coding` do receive `display`, and an agent's beta `"updates"` reaches Anthropic there without its header and is expected to 400 — native requests are the agent's own, and rewriting them is out of this row's scope. That `display` survives on some adapters and not others is a **scope** fact the register does not carry (KBR-139). The register guards compare ids and conditionality only, so this prose is reviewed, not tested. Top-level `effort` is not Anthropic's documented spelling — `output_config.effort` is; the field's restore is **P5f**'s claim under its own `OUTPUT_CONFIG_PRESENT` trigger, and P5f's notes carry the vendor-schema analysis. The two emissions are independent `if`s in the translator (`translator.py:425-426` vs `:438-439`), so a single trigger cannot cover both addresses (KBR-44). |
| P5f | Restore the agent's `output_config` → top-level `output_config` — or withhold it on adapters whose `forwards_output_config` is false (`minimax_token`, `opencode_go`, `zai_coding`; `custom_anthropic` restores it only when a balancing failover re-serializes a request translated for an earlier backend, since its own requests are native) | same | `output_config` present | Passthrough of an agent signal. The translator now carries `output_config` verbatim on `_output_config` (KBR-224), and the adapter restores it only where the upstream documents the field: Anthropic's GA schema defines it (SDK `OutputConfig`: optional, nullable, no beta header — verified 2026-09-13 **in isolation**; a body carrying *both* effort spellings has never shipped, and kitty does not arbitrate between two values the agent sent), MiniMax's endpoint rejects bodies carrying it, and `opencode_go`'s Messages models and `zai_coding` serve references that do not document it, so those withhold. **Separate row and trigger, not an extension of P5d**: the translator emits `_output_config` and `_effort` in independent `if`s (`translator.py:425-426` vs `:438-439`), so a request carrying `output_config` with no `thinking` and no top-level `effort` would have produced an unclaimed delta at `envelope.extra[output_config]` had P5d been extended under `ADAPTIVE_THINKING_KEYS_PRESENT` — the false I1 breach §3.3.1a warns about, and the deferred comment P5d carried since KBR-224 anticipated by naming the would-be trigger. That observation about Claude Code (output_config and thinking co-occur in every observed capture) is an empirical note about the agent, not a register invariant, and the design's own plan — also anticipated by §3.2.2's P5d trigger cell wording ("or output_config present", data-orphan until this row landed) — is the separate row. **KBR-44.** First corpus entry carrying `output_config`: `effort_configured` (T-C1); until a corpus entry carries the field, no oracle run can see the withhold as a delta, which is why this row and the entry land together (§3.3.1a's pairing rule, named the moment the conditional row exists). Per-destination scope is prose for the same reason `display`'s is on P5d (KBR-139): the register guards compare ids and conditionality only. |
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
| P26 | **Drop a top-level `cache_control` on the CC-origin translated route** — the Anthropic adapter family never reads the root-level key, so it does not reach the wire | `AnthropicAdapter.translate_to_upstream` (`kitty/providers/anthropic.py`) | The upstream wire is not native Messages — same trigger M2/M16 carry | KBR-258, G37's CC-origin half. The `envelope.extra[cache_control]` projection exists today on the Anthropic Messages reader (G38's remedy records the Messages-route twin); the Chat Completions reader does not yet consume the key into this address, so today such a body residualises and fails the run first — a named, honest failure. The row claims the address the moment the reader grows the slot (the G38 precedent: the row lands before the oracle can drive the body). All four adapters (`anthropic`, `minimax_token`, `zai_coding`, `custom_anthropic`) are covered by the one base-class site: the three subclasses short-circuit on `_native_messages_request` and otherwise delegate to `super().translate_to_upstream`, and the drops KBR-199 measured happen on the translated (non-native) branch only. `conditional=False` per KBR-186 (ROUTE triggers are not corpus-variable) and the `test_rows_sharing_a_trigger_agree_on_whether_it_is_conditional` guard (which already binds M2/M16/M20–M25 to `False` on this trigger). The native passthrough carrying the breakpoint is the observational complement, proven as product behaviour in epic KBR-197, not a corpus entry — the ticket's loose "each row owes a complement corpus entry" prose is reconciled in KBR-258's Jira comment. **KBR-167**; the product-behaviour suite for the same defect is epic KBR-197 |
| P27 | **Drop a system content-part `cache_control` on the CC-origin translated route** — the system-extraction loop joins system blocks into one string and discards breakpoints as it goes | `AnthropicAdapter.translate_to_upstream` (`kitty/providers/anthropic.py`) | The upstream wire is not native Messages — same trigger P26/M2/M16 carry | KBR-258, G37's CC-origin half. The Chat Completions reader projects a system content-part breakpoint onto `conversation.system[*].cache_control` today (KBR-34), so this row is **claimable now** — a body carrying one and routed through the Anthropic family would otherwise report an unclaimed I1 breach. Same site, trigger and conditionality as P26; same M16 precedent on the Messages twin (`system_path(*, cache_control)`); same scope note on the four adapters. `forwards_thinking_signature` does not change this: `_anthropic_system` is a Messages-ingress carriage stripped on the CC route by P1, so the join stands on every adapter |
| P28 | **Drop a user-message object-level `cache_control` on the CC-origin translated route** — the outer message loop reads each `{"role": …, "content": …}` for role and content only | `AnthropicAdapter.translate_to_upstream` (`kitty/providers/anthropic.py`) | The upstream wire is not native Messages — same trigger P26/P27/M2/M16 carry | KBR-258, G37's CC-origin half. Site covers the user-message and tool-message object case; the assistant-message object case rides P29's site — the oracle matches paths, not sites, so the address is claimed either way. **The kept half — a breakpoint on a user content part and on tool-message content (moved inside the `tool_result`) — is not this row's, and carries no row anywhere in this five-row set**: the adapter family preserves both on the CC route (KBR-199's measurement), the same boundary M16 draws on the Messages twin. The object-level rows (P28, P29) are anticipatory today — the Chat Completions reader residualises a message-dict-level `cache_control`, so such a body fails the run first, the honest named failure; the row lands now, before the reader grows the slot, on the G38 precedent. Anchor contingency: the row assumes the reader projects the message-object `cache_control` onto a Part rather than onto `Turn` itself (`conversation.turns[*].cache_control`, which §3.3.1a does not define today); if a future reader lands the slot on Turn, the anchor must move to match. Same trigger and conditionality as P26; same scope note on the four adapters |
| P29 | **Drop an assistant-message object-level `cache_control` and a tool_call-level `cache_control` on the CC-origin translated route** — `_translate_assistant_msg` rebuilds the assistant message and its `tool_calls` blocks into Anthropic `tool_use` blocks | `AnthropicAdapter._translate_assistant_msg` (`kitty/providers/anthropic.py`) | The upstream wire is not native Messages — same trigger P26/P27/P28/M2/M16 carry | KBR-258, G37's CC-origin half. Anchored at the same path as P28 because both project to a Part (the tool_use is a Part); distinguishable by site, the axis `test_no_two_rows_are_indistinguishable` explicitly allows (P3/P4 precedent). Same anchor contingency as P28; anticipatory today for the same reason (the reader residualises a tool-call-level `cache_control`), on the G38 precedent. Same trigger and conditionality as P26; same scope note on the four adapters |
| P30 | **Drop a tool-declaration `cache_control` on the CC-origin translated route** — `_translate_tools` rebuilds every tool declaration as `{name, description, input_schema}` | `AnthropicAdapter._translate_tools` (`kitty/providers/anthropic.py`) | The upstream wire is not native Messages — same trigger P26/P27/P28/P29/M2/M16 carry | KBR-258, G37's CC-origin half. The Chat Completions reader projects a tool-decl breakpoint onto `conversation.tools[<name>].cache_control` today (KBR-34), so this row is **claimable now** — a body carrying one and routed through the Anthropic family would otherwise report an unclaimed I1 breach. The Messages-route twin of this drop is M16's tool-decl path. Same trigger and conditionality as P26; same scope note on the four adapters |
| P24 | **Drop every non-sampling Chat Completions control field outside the CC builder's carries** — `audio`, `function_call`, `functions`, `metadata`, `modalities`, `moderation`, `prediction`, `prompt_cache_options`, `reasoning_effort`, `service_tier`, `user`, `verbosity`, `web_search_options` (**thirteen**) | `OpenAISubscriptionAdapter._cc_to_responses` | Always, on the **CC-origin** path | KBR-184 / G26. P13's CC-origin twin for the **non-sampling** half: `_cc_to_responses` builds the Responses body from scratch and ships `model`, `messages`→`input`, `stream`, `store`, `tools`, `tool_choice`, `parallel_tool_calls` and an injected `reasoning`; every other declared Chat Completions control field (T-A2's `_PUBLISHED_EXTRA_KEYS`) is dropped. P13 is anchored at the bare `conversation.sampling` and reaches none of these, so T-D5 would report a false I1 breach on the CC-origin route exactly as it would have on the Responses-origin one. ⚠️ **Thirteen, not the ticket's sixteen.** T-A2's `CreateChatCompletionRequest` table has fourteen published `extra` keys (retrieved 2026-09-14); `store` is rewritten (forced `False`) by the builder, not dropped — P17's territory. Three keys the ticket listed (`prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`) are not in T-A2's CC-surface extra table — the reader residualises them, the §3.3.2 "named, honest failure" shape, so no row is owed for them. ⚠️ Enumerated, NOT anchored at a bare `envelope.extra` — the bare form matches and would over-claim `extra[reasoning]` (P22), `extra[store]` (P17), and the canonical knob address (`extra[parallel_tool_calls]`, G36 / KBR-205). The derivation guard `TestP24ClaimsTheDroppedNonSamplingControlFields` recomputes the set from the AST; widening the reader table or the builder literal both turn the row red. **KBR-171** is the Responses-origin twin. **KBR-184** |
| P31 | **Drop `tool_choice` and `parallel_tool_calls` on the `ollama_cloud` route** — Ollama `/api/chat` defines neither a tool choice nor a parallel-tool-use knob, so the hook writes neither; the projected deltas land at `envelope.extra[tool_choice]` and `envelope.extra[parallel_tool_calls]` | `OllamaCloudAdapter.translate_to_upstream` (`kitty/providers/ollama_cloud.py`) | Always, on `ollama_cloud` | KBR-184 / G32. The ollama half: a third-party API with no field for either input. Not a defect in the adapter — nothing on that wire can carry the field. P9e/P9f's "paths must be true of every site" rule is why the bedrock parallel-knob twin (P32) gets its own row rather than sharing this one (bedrock carries tool choice, G33 / P33). The canonical knob address `envelope.extra[parallel_tool_calls]` is the G36 / KBR-205 form; P31 does not need a parallel-knob reader side to ship the wire key |
| P32 | **Drop `parallel_tool_calls` on the `bedrock` route** — Bedrock Converse's `ToolConfiguration` has no parallel-tool-use knob (botocore `bedrock-runtime`), so the hook writes no `parallelToolCalls`; the projected delta lands at `envelope.extra[parallel_tool_calls]` | `BedrockAdapter.translate_to_upstream` (`kitty/providers/bedrock.py`) | Always, on `bedrock` | KBR-184 / G32. The bedrock half. Site is `translate_to_upstream` (the hook-level decision to omit) rather than the boto3 transport, which mutates `modelId` / `stream` (P18) — the parallel knob is decided earlier, at the hook. The canonical knob address `envelope.extra[parallel_tool_calls]` is the G36 / KBR-205 form |
| P33 | **Rewrite `tool_choice` to `auto` whenever the Chat Completions body forces nothing** — the Bedrock hook writes Converse `toolChoice: {"auto": {}}` for any CC choice that is not `required` and not a named choice to an ordinary tool; absent choice, `none`, and G35's omitted forms are all rewritten here | `BedrockAdapter.translate_to_upstream` (`kitty/providers/bedrock.py`) | The CC body carries tools and the `tool_choice` value is absent, `none`, or any value except `required` / named function | KBR-184 / G33 (KBR-214). Converse's `ToolChoice` union has no `none`, and dropping `toolConfig` is unavailable once a transcript holds `toolUse` / `toolResult` blocks ("`toolConfig` must be defined when using toolUse and toolResult content blocks"), so absent or `none` on the CC side is rewritten to `auto` rather than omitted. Pre-existing; recorded now because the Converse reader (T-A5) maps `toolConfig.toolChoice` onto `envelope.extra[tool_choice]` (§3.3.1b) and P11 is `NOT_PROJECTABLE`, so nearly every Bedrock request with tools would have shown it. Shares `envelope.extra[tool_choice]` with P35; distinguishable by site (P3/P4 precedent, `test_no_two_rows_are_indistinguishable` explicitly allows). Both conditional, so the two rows cannot disagree about whether the address owes a complement. The trigger case + §3.3.2 assertion-2 complement (a `required` choice or a named choice to an ordinary tool, which the adapter must carry as `any` or `tool`) arrive with the T-D5 corpus entries |
| P34 | **Omit `disable_parallel_tool_use: false` rather than forward it** — the Anthropic→CC carry maps the flag only when `true`, onto `parallel_tool_calls: false`; an explicit `false` is omitted because `false` is the documented default on both wires | `carry_tool_choice_and_metadata` (`kitty/bridge/messages/translator.py`) | The inbound Anthropic body carries `disable_parallel_tool_use: false` | KBR-184 / G34 (KBR-214 D2). Writing it would add a second field some providers reject, and the omission is correct — must not be "fixed" by forwarding `false`, G29's exact shape. The Anthropic reader can nonetheless tell the two spellings apart, so an explicit `false` is a delta at `envelope.extra[parallel_tool_calls]`. The address exists as of KBR-205 (§3.3.1b); the row registers the omission now, with the trigger case + §3.3.2 complement (a body where the flag is `true` and carried) arriving with the T-D5 corpus |
| P35 | **Omit a legal `tool_choice` over no tools, or forcing an Anthropic-defined tool** — KBR-214 D9/D10 omits the choice where carrying it would create a failure the agent did not cause | `carry_tool_choice_and_metadata` (`kitty/bridge/messages/translator.py`) | The inbound Anthropic body carries a `tool_choice` over no tools, or forces a tool whose declared `type` is not absent / `null` / `"custom"` (e.g. Claude Code's `web_search_20250305`) | KBR-184 / G35 (KBR-214 D9/D10). Case (1) — choice over no tools — is legal Anthropic and "'tool_choice' is only allowed when 'tools' are specified" on OpenAI. Case (2) — forced call to an Anthropic-defined tool — flattens into a schema-less function nothing on the route can execute (anthropics/claude-code#56984; omitted on the product owner's decision, 2026-09-13). `{"type": "any"}` over only Anthropic-defined tools is carried: guarding it would reason over the whole tool list rather than one name. Forced calls to **undeclared** tools are not omitted — that is the agent's mistake and the provider's error names it. Shares `envelope.extra[tool_choice]` with P33; distinguishable by site (Messages-route omission vs bedrock auto-rewrite). The Anthropic reader carries `ToolDecl.type` as of KBR-205, so case (2)'s trigger is authorable today. The trigger case + §3.3.2 complement arrive with T-D5 |
| P36 | **Translate CC → OpenAI Responses body** — the whole-protocol rewrite for OpenCode Go's `/v1/responses` route; messages → `input` + `instructions` (system hoisted; per-message: user → `input_text`, assistant → `output_text`, tool → `function_call_output`); tools and tool_choice envelopes unwrapped; `response_format` moved to `text.format` | `OpenCodeGoAdapter._cc_to_responses` (`kitty/providers/opencode.py`) | Always, on the four `_RESPONSES_MODELS` routes | KBR-137. OpenAI Responses is the fifth wire format M2 names — after Messages, Chat Completions, Converse (P11), Ollama `/api/chat` (P12). Like P11/P12 the projection is what makes the formats comparable, so the whole-protocol translate names no field; the path is `NOT_PROJECTABLE` for the same reason. The per-message renames and envelope unwraps are part of this single claim — they have no path vocabulary, and the L1 suite in `tests/test_provider_opencode_responses.py` pins each directly. Replaces the KBR-126 refusal: the four models are now servable. The user accepts the spec default for `store` (Codex-only, P17). |
| P37 | **Drop eight CC sampling / control fields with no Responses equivalent** — `frequency_penalty`, `presence_penalty`, `seed`, `logit_bias`, `n`, `stop`, `logprobs`, `stream_options` | `OpenCodeGoAdapter._cc_to_responses` (`kitty/providers/opencode.py`) | Always, on the four `_RESPONSES_MODELS` routes | KBR-137. The eight fields are absent from the published `CreateResponse` schema (verified 2026-09-16 against `openai/openai-openapi` master v2.3.0). **`stream_options` is included** because Responses' `stream_options` has different semantics (`include_obfuscation`, not `include_usage`); the CC value is silently dropped, not translated. **User-visible**: those settings silently do nothing on these four models. Registered against the spec, not the builder — a future spec addition that adds one of these names turns the row red, and a builder addition that stops dropping one turns the L1 suite red. |
| P38 | **Rename `max_tokens` / `max_completion_tokens` → `max_output_tokens`** | `OpenCodeGoAdapter._cc_to_responses` (`kitty/providers/opencode.py`) | Either CC spelling present | KBR-137. Responses spells the token ceiling differently and counts reasoning tokens toward it. Precedence: `max_output_tokens` > `max_completion_tokens` > `max_tokens`; only the first present is forwarded. The two CC addresses are claimed because they disappear; the Responses address is not — it is the natural key the Responses API names, not a mutation. |
| P42 | **Inject `reasoning: {"effort": …}`** from kitty's internal `_reasoning_effort` | `OpenCodeGoAdapter._cc_to_responses` (`kitty/providers/opencode.py`) | `_reasoning_effort` present and not `"none"` | KBR-137. P3/P4-class passthrough in the target's spelling (mirrors the Codex P22 on the openai_subscription provider). The CC value rides `_reasoning_effort` (an internal key stripped by P1); the Responses address `envelope.extra[reasoning]` is created here. **Conditional**, so the §3.3.2 complement is owed: a corpus entry whose `_reasoning_effort` is absent or `"none"`, in which the injection is provably absent; the trigger case + complement arrive with the T-C entries, as for P22. |

**The KBR-228 thinking carriage is not a row.** The Messages → CC converters (`MessagesTranslator.translate_request` and `server._convert_native_to_cc_format`) carry the agent's verbatim thinking-family blocks under `_thinking_blocks` on the message dict and the original `system` value under `_anthropic_system` at the top level, and the adapters whose upstream honours Anthropic's thinking-binding contract (`forwards_thinking_signature`) put them back byte-identical; the response direction mirrors this onto the reply for `MessagesTranslator` to consume. A carriage is kitty-internal, consumed or stripped before any wire or client sees it, and restores content rather than altering it — there is no delta for a row to permit, which is why KBR-228's "register rows P5e and the new carriage" resolves to row updates only. The keys ride `ProviderAdapter._INTERNAL_KEYS` / `_INTERNAL_MESSAGE_KEYS` under P1's strip, and the message-level guard in `tests/test_internal_keys_not_sent_upstream.py` keeps them off every non-restoring wire.

**Conditional rows are the point.** Every row whose trigger is a condition must be provably
*inert* when that condition is absent — the sharpest form of "unless absolutely necessary", and
what §3.3.2 assertion 2 tests, with the trigger complements §3.3.4 requires. M1, M2, M10, M14, M15, M16, P1,
M20, M21, M22, M23, M24, M25, M26,
P6, P9a, P9b, P9c, P9e, P9f, P9g, P9h, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P23, P24, P26, P27, P28, P29, P30, P31, P32, P36, P37 and P38 are
unconditional by design and are exempt from that assertion.

**M14, P20 and P21 were missing from that list until KBR-26**, while their own trigger cells read
`Always`. A row that always fires has no complement, so assertion 2 would have demanded a corpus
entry nobody could ever write. The list is also written out id by id rather than abbreviated as a
range: `P9a–c` names three rows in one token, and §3.2.4's guard has to either guess at the
expansion or drop two rows from the comparison. It refuses the notation instead.

**The P family's reservations have landed.** `P22` was held for the
`reasoning` injection §9.2's G23 registers; P23 landed first (KBR-171) and did not take the
reserved id, because an id is how every ticket and document refers to a row. P22 has since
landed at the reserved id (KBR-149). `P24` was held for §9.2's G26, whose enumeration
waited on the Chat Completions reader's control-field table (T-A2 / KBR-34, landed
2026-09-15); P24 has now landed at the reserved id (KBR-184) with thirteen enumerated
paths, closing G26. No reservation remains in the P family. The register is ordered by
**document position**, not numerically — §3.2.2 already interleaves P20 and P21 between
P6 and P7 — so a row arriving out of numeric order costs nothing.

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

**Why a slot for `cache_control` rather than §3.3.1's other outcome.** §3.3.1 offers "map it, or
declare it ignored with a reason", and a reader-side declared-ignored mechanism would also have
stopped the run failing. It was rejected **for this field** deliberately: kitty's translated path
**strips every block-level breakpoint** (one nested carrier escapes — see M16), so under a
declared-ignored rule the oracle would be blind, by construction, to a mutation that re-bills the
user's cached prefix at **at least** ten times its cached rate — a cache read is 0.1x base input on
most models and 0.025x on Claude Fable 5.1 and Mythos 5.1, where the multiple is forty. Register
row **M16** claims the strip instead, which keeps the cost visible and attributable.

**The other outcome exists now, and its first users arrived with KBR-205.** Five Anthropic block
fields — `text.citations`, `image.transformations`, `tool_use.caller`, `tool_use.toolset_name` and
`tool_result.toolset_name` — needed the third outcome rather than a slot, because **none of them
carries a consequence a strip would hide**: they are optional, vendor-defined *descriptive* fields,
not control knobs, and a reader that drops one changes nothing the agent asked for. Slots for five
vendor spellings would have pushed one vendor's names into a form whose entire purpose is wire
independence (§3.3.1 declines that for P16's reason). The mechanism — path-keyed by
`(block wire type, field wire key)`, reason-required per entry, defined once for all six readers in
`tests/harness/contract.py` — is recorded in §7.4.1, which also names the two Anthropic tool
fields that are deliberately **mapped** rather than ignored because a mutation on them is a real
fidelity finding (KBR-214).

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
| `envelope.extra[<wire key>]` | A format-specific control field — P2a `thinking`, P3 `reasoning`, P4 `reasoning_effort`, P10 `reasoning_split`, P26 and **M16** `cache_control` (Messages-route twin of the KBR-258 P26 half — same address, different ingress route), and P23's sixteen dropped Responses control fields. **The bare `envelope.extra` is not a legal anchor** — see below |
| `conversation.system[<i>]` | One system text part |
| `conversation.system_role` | The role a Gemini `systemInstruction` `Content` published — the first path form at conversation scope, added by KBR-194. `None` is the absent value, so a translated route that drops the role is a positive delta at this path. M20 on the Gemini inbound route claims it |
| `conversation.turns[<i>].role` · `.parts[<j>]` | A turn, or one part of it |
| `conversation.tools[<name>].description` · `.schema` · `.strict` · `.behavior` · `.type` | A tool declaration, **by name**. `behavior` is Gemini's NON_BLOCKING calling toggle (KBR-194); `type` is the tool's discriminator — `"custom"` on a client tool, an Anthropic-defined dated spelling (e.g. `web_search_20250305`) on a server tool (KBR-205, closing G35). `ToolDecl.type` is `str | None`, so an absent type is its own absence rather than a coerced `""` |
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
- **The parallel-tool-use knob is its own address, not part of `tool_choice`** (KBR-205, closing
  G36). Anthropic nests an inverted flag on every `tool_choice` shape —
  `tool_choice.<shape>.disable_parallel_tool_use`, default `false` — and Chat Completions carries
  a top-level `parallel_tool_calls` boolean, default `true`, on its own wire. Same concept, two
  spellings, opposite polarity; the value must therefore be normalised too, so both wires meet at
  **`envelope.extra["parallel_tool_calls"]`, the Chat Completions spelling and polarity** —
  `true` when parallel calls are allowed, `false` when they are not. The reader writes the entry
  only when the wire carries a **non-default** value, mirroring the product's forwarding rule
  (KBR-214 forwards only `disable_parallel_tool_use: true` as `parallel_tool_calls: false`); an
  absent entry and an explicit default are one request on both wires, so writing both would
  invent a second field some providers reject and every comparison would carry. The Anthropic
  reader maps `disable_parallel_tool_use: true` onto `parallel_tool_calls = False`; the Chat
  Completions reader (T-A2) meets the same address and reads `parallel_tool_calls` directly.
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
that format's published shape and **importing nothing from `src/kitty`**. Six small
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

**Prose-claimed omissions on the Responses and Gemini tool-choice carriers (KBR-221,
KBR-139 precedent).** Since KBR-221 the Responses and Gemini ingress translators carry
`tool_choice` onto the CC body, but neither destination wire can express every shape the
source wires publish. The carriers keep the representable half and omit the rest; until a
corpus entry carries one of these shapes, prose claims the omission and **a register row
becomes due for each shape with the first corpus entry that carries it**:

- *Responses* — the eleven hosted choice types (the `ToolChoiceTypes` union, of
  which three are `Specific*` singletons — nothing on a translated route can
  execute them — D10's shape), `type: "custom"`
  and `type: "mcp"` choices (the hop degrades or does not proxy those tools), `function`
  choices without a string name, `allowed_tools` modes outside the published enum, non-dict
  non-string `tool_choice` values, and `parallel_tool_calls` values that are not booleans
  (the CC reader would residualise them).
- *Gemini* — `mode: VALIDATED` and `MODE_UNSPECIFIED` (no canonical mapping; the readers
  residualise them, so neither side writes the address), and the restriction half of
  `allowedFunctionNames`: multi-name `ANY`, `AUTO`/`NONE` beside names, an empty list, a
  non-list value, and lists with non-string members all carry the mode only — the name
  restriction has no CC home on any destination wire (the owner's decision: carry the
  mode, record the loss).

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
| **Cache-breakpoint sites in `MessagesTranslator.translate_request`** — which of Anthropic's sites it destroys, including the top-level key (now claimed by M16, KBR-263 closing G38), and the one it carries | L1 | `TestTranslateRequestCacheBreakpoints` in `tests/bridge/test_messages_translator.py` — **KBR-198** (epic KBR-197). A **characterisation**: nine sites expect no breakpoint anywhere in the output; a block nested inside `tool_result.content` expects its breakpoint unchanged and in place, because that list is forwarded as-is — pinned as behaviour, not endorsed. The eventual carry-through fix inverts the nine; that is the intended workflow, not a regression. Inputs come from `harness.cache_breakpoints`, whose module documentation records why it emits **one breakpoint per request** (Anthropic rejects more than four), why the breakpoint is the 1-hour value, why its detector matches a key *containing* `cache_control` or the value itself, and why the prefix is padded past 2 x 4,096 words. **Not covered:** `server._convert_native_to_cc_format`, the native route's format-error fallback, which strips the survivor too (folded into KBR-200) |
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
| **C1 — Request headers** | `build_upstream_headers()` constructs the set from scratch; no inbound agent header is forwarded. Four adapters supply a coding-agent `User-Agent` (P9a, P9c). The deviations from the base header set are registered: `x-api-key` + `anthropic-version` (+ the lowercase `content-type` re-spell, unaddressable — casing) on the Anthropic family (P9e: `anthropic`, `custom_anthropic`, `minimax_token`, and `opencode_go` on its Messages models); `anthropic-version` + the casing re-spell beside Bearer auth on `zai_coding` (P9f); `api-key` on Azure's non-Entra credential (P9g) and on Mimo (P9b); no `Authorization` at all on `ollama` (P9h); the conditional `ChatGPT-Account-Id` on `openai_subscription` (P9d). Every other adapter sends the baseline set. | **Closed (KBR-78, 2026-09-15).** Every wire route's exact header set — names, casing, value shape — is asserted over the same `wire_routes()` enumeration, and the AST-derived branch-arm meta-test proves the swept routes execute every `If` arm of every header builder. `bedrock`'s contract is observed on the botocore-prepared request (its hook never ships). The OAuth token POSTs (KBR-161's other half) assert their exact set per site; `originator: codex_cli_rs` lands on all four (probed against `auth.openai.com` 2026-09-15, indifferent). F1's policy gap remains — the identity each adapter *should* claim is Q1. |
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
guard; see its entry below and §6.2.3. **F2 has since been fixed under KBR-9**, together with the
exemption-row withdrawal; see its entry below and §6.2.3. F1's self-contradiction was fixed in
KBR-8; the policy half and F4 remain open.

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
  **T-G9 landed (2026-09-15, KBR-78)**: the same file now extends to the exact
  header set per wire route — names, casing, value shape, with an
  AST-derived branch-arm meta-test that proves the swept routes execute
  every `If` arm of every header builder.
  **What KBR-8 did not close:** identity is still ad hoc per adapter — three hard-coded
  `claude-code/1.0` strings, one synthesised Codex identity, and aiohttp's default everywhere else.
  That is the policy half of G3, and it waits on Q1.
- **F2 — FIXED (KBR-9, 2026-09-17).** The README's endpoint table did not match the router:
  it documented `POST /v1/gemini/generateContent` (no such route); `_register_routes` registered
  `/v1beta/models/{model:.*}:generateContent` and `:streamGenerateContent`, and the README omitted
  `GET /v1/models`. Exactly the drift the L2 docs⇄code layer exists to catch. The guard half
  landed with KBR-76 under the `t-g1-endpoint-table` exemption; the README correction landed with
  KBR-9, the exempt assertion passed, `UnexpectedExemptionPass` fired, and the row was withdrawn
  together with its `ratchet` plumbing in the same change — the mechanism's second payout
  (the KBR-188/KBR-208 narrative in `tests/exemptions.py` records the first). Gap G6 is closed.
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

**C4a — the OpenAI login leg, specifically (KBR-161, KBR-78).** The general C4 argument above does
**not** apply to this provider: `curl_cffi` is already a dependency, already constructed, and
already in use in this very adapter, so the cost that justifies accepting C4 elsewhere is absent.
KBR-161 moved the **recurring** OAuth traffic — `_refresh` and `_exchange_api_key`, which
`get_valid_api_key` drives on every Codex request — onto an impersonating `curl_cffi` session, and
all four token POSTs now carry the Codex `User-Agent` from `kitty.codex_identity`. **KBR-78 (2026-09-15)**
closed the residual header gap on the auth leg: every one of the four token POSTs now also carries
`originator: codex_cli_rs` — the one header the genuine Codex CLI sends and the legacy bridge
omitted. The decision rests on a live probe against `auth.openai.com` (four POST variants, no
credentials) which confirmed the auth host is indifferent to `originator` on its error path. The
strict-tool-validation warning in `OpenAISubscriptionAdapter._build_codex_headers`, which scoped
the omission, applies to the Codex **backend** (`api.openai.com/v1/responses`); KBR-78 scopes
that warning to the API leg.

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
guards the identity half and asserts the exact-set per site; it says explicitly that it asserts
nothing about the fingerprint.

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

**Two things are missing, and they are the gap** — a third (the file-granular start-path
guard) has since been closed, and is recorded below in the state it reached.

1. **`BridgeServer`'s own request path is untested.** The existing module drives
   `egress_cmd._probe` — the function behind `kitty egress test` — not `_session_for` /
   `_make_upstream_request`. The serving path, which carries every byte of every conversation,
   has no equivalent proof.
2. **No negative assertion anywhere.** Nothing asserts that with the proxy *down*, the upstream
   receives **zero** connections. Without it, a bridge that proxies most of the time and falls
   back to a direct route on error would pass every test in the suite.
3. **The start-path guard is now AST-level.** The original guard was file-granular
   (a file holding a `BridgeServer(` call only had to also *contain* a call to
   `egress_block_reason`; `cli/main.py` already held two start paths, so a third added to
   that file would have passed unguarded). Landed with **KBR-67** as
   `tests/test_egress_coverage.py::TestEveryStartPathIsGuarded` — every `BridgeServer(`
   construction is now checked for domination by an `egress_block_reason(` call in the
   same enclosing function scope.

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
| **Every** `BridgeServer` construction is dominated by an `egress_block_reason` call | L2 structural | **Landed with KBR-67** as `tests/test_egress_coverage.py::TestEveryStartPathIsGuarded` (§5.1 gap 3, now closed) |
| The guard's rejection is **enforced** — no server starts on a rejecting configuration | **L3** | §6.2.3. Structural domination proves the call, not the branch that acts on it |
| An `https://` proxy carries real traffic on all three transport stacks | L2/L3 | **Exists:** `tests/test_egress_https_proxy.py` |
| Proxy semantics under each dependency's version range | L2 | §6.2.4 |
| The destination is reachable directly with egress **disabled**, on every transport | **L3** | §5.2.2 phase 1 — the positive control. Without it the row below proves nothing (§5.3) |
| Nothing reaches upstream except via the proxy, **from the bridge's own serving path** | **L3** | Sealed-network harness (§5.2.2 phase 2b) |
| The harness detects a deliberately injected bypass | **L3** | §5.2.2 phase 3 — the falsification control. A containment harness never shown to fail is indistinguishable from one that cannot |
| Stopping the proxy stops the traffic — no direct fallback | **L3** | §5.2.2 phase 2 |
| Containment holds for each custom transport | **L3** | §5.5 |
| A loopback or `localhost` provider is reached directly with egress configured, on the bridge's own sessions — the local-bypass property the §5.2.2 phases deliberately do not exercise | **L3** | **T-E8:** `tests/harness/test_local_bypass_slice.py` |
| The transport asymmetry — the §5.5 paths apply the proxy unconditionally and consult no `should_bypass` — is pinned so a future fix cannot quietly add one | L1 + L2 structural | **T-E8:** `tests/test_egress_asymmetry.py`; the site inventory and the no-`should_bypass`-in-custom-transport guard in `tests/test_egress_coverage.py` |
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
- **Decided (KBR-170, folded into [KBR-71](https://shelpuk.atlassian.net/browse/KBR-71), 2026-09-15):**
  the overrides catalog keeps its top rank — a remote-synced entry still beats
  a profile's hand-written `provider_config["context_window"]`. What the owner
  authorised is the **visibility** the docstring's escape hatch did not provide:
  when the two disagree, `get_model_context_tokens` emits an `INFO` line once per
  `(provider, model, effective, ignored)`, naming the model and both values, so
  an operator whose setting is being outvoted by a file they do not control can
  discover that from the logs **at INFO level** — `INFO` (not `WARNING`) because
  the precedence is deliberate and this is visibility, not an alarm.
  Discoverability therefore depends on the operator's logging config emitting
  at INFO for the `kitty.providers.model_context` logger; handlers that filter
  INFO out will need to surface this logger's events explicitly. The docstring's
  "omit that model from the overrides file" advice remains true; the log makes
  the shadowing visible even when the operator did not set the file themselves.
  The behaviour itself (precedence + silent shadow) was unchanged. No register
  row moves.

**Mutation testing.** Line coverage cannot tell a real assertion from `assert result is not
None`. `mutmut` closes that gap.

- **Tool:** `mutmut` (3.x; requires `fork`, so it runs on Linux CI — on Windows it needs WSL).
  Configured in `pyproject.toml` under `[tool.mutmut]`, where `source_paths` and
  `pytest_add_cli_args_test_selection` take **arrays**.
- **Test selection:** `pytest_add_cli_args_test_selection = ["-m", "l1", "--ignore",
  "tests/test_internal_keys_not_sent_upstream.py"]`. Mutation testing measures the L1
  suite; letting it run L3 subsystem tests would make each mutant minutes long and attribute
  kills to the wrong layer. The one `--ignore` is mutmut-only — the L1 gate still runs the
  file — because mutmut's clean-run context trips it (see `.system_design/MUTATION_BASELINE.md`
  for the rationale); the gate's own selection is unchanged.
- **Scope — narrow, but it must include the code the rationale is about.** An earlier draft
  justified the subset by "a mutation surviving in the compactor means the suite would not notice
  kitty eating a tool result", then excluded `server.py`, where the compactor lives. Corrected
  scope, using `mutmut`'s function wildcards rather than whole modules:

  | Target | Why |
  |---|---|
  | `kitty.bridge.messages.*`, `kitty.bridge.responses.*`, `kitty.bridge.gemini.*`, `kitty.bridge.engine` | Translation — I1 |
  | `kitty.bridge.server._compact_messages*`, `_compact_with_tighter_budget*`, `_validate_tool_call_pairing*`, `_truncate_oversized_tool_results*`, `_apply_compaction*`, `_normalize_model*`, `_get_max_context_chars*` | Compaction and pairing — the I1 core, and the thing the rationale was always about |
  | `kitty.providers.*` `translate_to_upstream` / `normalize_request` / `build_upstream_headers` | The register's provider half — I1 and I2 |
  | `kitty.providers.base.ProviderAdapter._strip_endpoint_suffix` (KBR-134) | The four URL-shape mutations unguarded stripping missed during review; the guard's whole safety property is a one-line composition-and-recompare that a unit test would have to re-check for every shape |
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
  This triage pragma and [KBR-266](https://shelpuk.atlassian.net/browse/KBR-266)'s
  scope-narrowing markers on `server.py` are two different uses of the same
  pragma string: the triage use suppresses one known-equivalent mutant and
  carries its per-site "why" comment; the scope-narrowing use marks whole
  out-of-scope defs/classes so generation stays bounded (the "why" lives in
  `pyproject.toml`'s `[tool.mutmut]` comment and the marker-scheme guard's
  docstring, not at each of the 135 sites). A future reader grepping the
  source should not expect a per-site comment at every scope-narrowing
  marker.
- **Cadence — to be set by measurement, not assertion.** The full scoped run is nightly. Whether
  a **changed-code** mutation run also fits the per-PR gate is an open question with a numeric
  answer: measure the wall-clock of `mutmut run` restricted to functions touched by a
  representative PR, and adopt it per-PR if it lands inside the budget the fast gate can absorb.
  Rejecting per-PR mutation testing without that measurement is an assumption, not a decision.
  Tracked as Q11.

**Where the score lives.** Per-component scores (and the mutmut config that
produces them) are recorded in `.system_design/MUTATION_BASELINE.md`, with
the machine-readable scope in `tests/mutmut_scope.py`. The baseline is
**provisional** until [KBR-115](https://shelpuk.atlassian.net/browse/KBR-115)
(T-K6) reclassifies the socket/process modules §8.2 enumerates out of
`l1`; the current numbers are **optimistic**, since kills currently
credited through substantively-L3 tests vanish on re-measure. The
seventh group, `compaction_and_pairing`, moves from deferred to
measured with this change — see `MUTATION_BASELINE.md` for the row.
It was initially deferred because mutmut generates per-file and the
unscoped mutated copy of `server.py` reached 354 MB without finishing
generation; [KBR-266](https://shelpuk.atlassian.net/browse/KBR-266)
narrows generation by marking every def/class in `server.py` except
the seven `BridgeServer` methods the group names with
`# pragma: no mutate block`. The marker scheme is pinned by
`tests/test_mutmut_scope.py::test_server_py_pragma_scheme_marks_everything_but_the_seven`:
a stray block-level pragma on one of the seven's def headers (or on
the `BridgeServer` class itself) silently shrinks the measured I1
core, and a missing pragma elsewhere re-opens whole-file generation.
The guard inspects block-level defs/classes only — a pragma placed
*inside* one of the seven's bodies on a leading line of a nested
statement suppresses that branch's mutations without the guard
noticing; mutmut's config-level `do_not_mutate_patterns` (a regex on
source lines) is a parallel silent-shrink path the guard cannot see,
surfaced by the per-group TOTAL count in `MUTATION_BASELINE.md`
against the previous run.

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
| **Wire-shape honesty** | For every adapter × representative model, assert the adapter's declared wire shape agrees with the shape the body is actually written in. Catches F5 (KBR-7). **Two boundaries, two owners.** The *hook* form — observing `translate_to_upstream`'s return value — landed with **KBR-7** as `tests/test_wire_shape_honesty.py`, together with the fix: it asserts the per-model declaration against the emitted body, and the bare property against an explicitly declared default-route model. It guards itself so it cannot rot — the classifier is pinned against known Messages, Chat Completions, Converse and Responses bodies (including a Converse body with no tools, since the tools axis is what separates Converse from Messages, and a Responses body without `input`, since that axis separates Responses from the flat-tools lookalikes); every registry key must be represented; every route of a model-routing adapter must be represented; and the custom-transport set is asserted rather than narrated. The *wire* form — observing the body at the §3.2.3 boundary — lands with **KBR-80 / T-G4** as `tests/test_wire_shape_honesty_wire.py` (L2 contract guard). It captures the body handed to each custom-transport client — a stubbed boto3 client for `bedrock` (after the P18 pops of `modelId` and `stream`), a stubbed aiohttp session for `ollama_cloud` (after the P19 `stream` overwrite), and the body builders (`_cc_to_responses` CC-origin, `_prepare_responses_body` Responses-origin) plus a curl_cffi pass-through pin for `openai_subscription` — and pairs the captured classification against the declaration, per adapter. The exemptions named above are now closed: `bedrock` and `ollama_cloud`'s scalar mutations do not change the shape family, and the wire form observes them; `openai_subscription`'s hook is never the shipped body, so the wire form captures at the body builders. The wire form also extends the sweep to `provider_config`-constructed adapters (a `minimax_token` sweep that pairs the captured classification against `upstream_wire_shape_for_model` for both native routes) and to native-passthrough requests (a per-adapter `_native_messages_request` sweep that asserts both the wire shape and body preservation modulo `ProviderAdapter._INTERNAL_KEYS`) — the two areas the hook form deferred to the wire. Per §1.4 the first working version ships with a falsification case per capture, all defects adapter-side. The guard itself caught a live defect on landing: `OpenAISubscriptionAdapter` inherited `CHAT_COMPLETIONS` from `OpenAIAdapter` while every body it ships is OpenAI Responses — corrected to `RESPONSES` atomically with the guard (the KBR-7 pattern). All four `server.py` readers of the declaration are unreachable for custom-transport adapters, so the fix is behaviourally inert; the full reachability analysis (dispatch sites 4674 / 9973 / 3585 / 7493, converter call sites 3784 / 6423 / 7716) lives in the production docstring on `OpenAISubscriptionAdapter.upstream_wire_shape`. **KBR-137 replaced the boolean declaration** with the four-valued `WireShape` enum (Messages / Chat Completions / Responses / Other), exactly as this row's "replace, not extend" rule required: OpenCode Go now serves four models on the Responses endpoint, and a `False` meaning "Responses" would have been F5 in a new costume. The enum's consumer — the thinking round-trip repair — still makes a binary choice per body, because only `MESSAGES` and `CHAT_COMPLETIONS` define a carrier; `RESPONSES` and `OTHER` make the repair inert, and the retry site never flags a backend whose shape it cannot write a carrier for. Bedrock's Converse body now declares `OTHER` (a fifth wire M2 does not name) rather than inheriting a `CHAT_COMPLETIONS` default it does not match. |
| **Bridge-introduced vendor token** | No content the bridge *introduces* into a request body or header contains `kitty` in any casing. Scoped by the projection diff (§3.3.3), never a flat scan of the serialized body — a flat scan would fail on a user legitimately writing the word, and "fixing" that would breach I1. Caught F3 (KBR-5). **F3 is now fixed, so this guard has no live positive fixture left**: its positive control is the synthetic historical M13 string held in `tests/bridge/test_vendor_token_guard.py`, which T-G5 inherits. That file is also the defect-scoped stand-in until T-G5 lands — it scans **source literals** against an allowlist, never traffic, so it does not fall into the flat-scan trap this row warns about. |
| **Start-path domination** | Every `BridgeServer(` construction is dominated by an `egress_block_reason(` call **at AST level**, not merely co-located in the same file. **Necessary but not sufficient — see below.** Landed with **KBR-67** as `tests/test_egress_coverage.py::TestEveryStartPathIsGuarded`: every `BridgeServer(` construction in `src/kitty/**.py` (except the class definition in `bridge/server.py`) is walked to its innermost enclosing function and must find a preceding `egress_block_reason(` call **in that function's own body, not inside a nested helper scope or a generator expression** (whose body is deferred and would not dominate even if invoked). The construction matcher accepts both `Name` and `Attribute.attr` callees, so `mod.BridgeServer(...)` shapes are not invisible; an aliased-import guard refuses any `import … BridgeServer as X` so a future renaming cannot hide a call site from the scan. The matchers are shared helpers — `_iter_constructions_in_tree` and `_iter_aliased_bridge_server_imports_in_tree` — called by both the live-source scan and the synthetic-tree falsification probes, so a regression in the matching logic fails every caller rather than only the probe's own copy of the loop. Self-guards, per the §1.4 harness rule: a scan that finds fewer than five constructions across the three expected files fails; an in-test falsification fixture with **five** shapes — sibling-undominated, sibling-dominated, outer-guard with inner-construction, outer-construction with the only guard confined to a nested helper scope, and outer-construction with the only guard inside a generator expression — proves the walker classifies each correctly; a dotted-construction probe proves the `Attribute` widening is load-bearing; and the alias-import probe proves the import-rename safety net is live. |
| **Env-var register** | `_SETTINGS_ENV_OVERRIDE_KEYS` and `_CONFLICTING_ENV_VARS` (`launchers/claude.py`) match what `build_spawn_config` emits. The set-equality is pinned by `tests/test_launcher_claude.py::TestInjectedKeyListsInSync::test_settings_and_cleanup_lists_are_identical` so a future addition to one list and not the other is caught without an external oracle (the README is silent on individual keys; the L2 contract lives in the test, not the prose). The `KITTY_*` half — the env vars the README *does* name — is a separate contract: every documented `KITTY_*` name must exist as an exact-match string constant under `src/kitty/`. Landed with **KBR-76** as `tests/test_readme_table_guards.py::TestEnvVarRegister`. Forward direction only; the README is silent on internal/operational keys (`KITTY_TMUX_WRAPPED`, `KITTY_THEME`, …) by design, so an exact-equality guard would force either documenting plumbing or renaming it in source, neither of which is the right answer. Exact-match is pinned against a substring deviation by its own falsification. |
| **Provider routing table ⇄ provider docs** | For every model the provider publishes, the adapter routes to the endpoint the provider serves it on. Landed with **KBR-126** as `tests/data/opencode_go_endpoints.json` (a snapshot of OpenCode Go's published endpoint table, carrying `source_url`, `verified_utc` and a note on what a keyed probe would add) plus `tests/test_opencode_endpoint_table.py`. The snapshot is an **oracle**, deliberately not the routing table itself: deriving `_MESSAGES_MODELS` from it at import would remove the duplication and add a worse failure mode, since a missing or corrupt data file would silently route everything to the default endpoint — the defect, reintroduced invisibly. The checker is a pure function (`check_routing`) so the negative cases can hand it a deliberate defect, and it compares **set equality in both directions**: a constant naming a model the provider has *stopped* serving on a route is exactly as wrong as one it never started routing, and that is the shape KBR-126 actually was. **Honest limit:** snapshot and constants are written in the same commit, so a green run proves self-consistency, not agreement with the provider; no stronger evidence is reachable without a paid key, because an unauthenticated probe of either endpoint returns `401 AuthError` (auth precedes dialect). Hence gap **G24**. |
| **`validation_model` reachability** | For every adapter that is not `use_custom_transport`, the path `validate_api_key` posts to and the bare `build_upstream_headers` agree on a dialect the key-check ping is written in. Landed with KBR-126 as `tests/test_validation_model_routing.py`. The ping body — `{model, messages, max_tokens, stream}` — is **simultaneously valid Chat Completions and valid Anthropic Messages**, which is why `anthropic`, `custom_anthropic`, `minimax_token` and `zai_coding` validate against `/v1/messages` and work. So the rule is *not* "`validation_model` must be Chat-Completions-routed": it is that path and auth must match. Pointing `opencode_go` at a Messages-routed model — the fix KBR-126's own ticket suggested — leaves the headers `Bearer` and fails every key check, and a path-only guard would pass it. |
| **Endpoint table** | The README endpoint table matches `_register_routes`. Catches F2 (KBR-9). Landed with **KBR-76** as `tests/test_readme_table_guards.py::TestEndpointTable` — AST-scanned by function name (bridge-mode branch plus the unconditional `/healthz`//`stats` registrations; launch mode out of scope), both directions, self-guarded. The agreement assertion shipped as the `t-g1-endpoint-table` exemption (KBR-9) — red at the base revision — and was withdrawn 2026-09-17 with the README correction: the exempt assertion passed, `UnexpectedExemptionPass` fired, and the guard now gates normally (eight routes on each side, including the `{model:.*}` converter literal). |
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
| `SIGTERM` (forwarded to the child; kitty's `finally` restores) | Same byte-identical assertion. A SIGTERM landing in the pre-handler window (between the `_register_atexit_cleanup` call and the `signal.signal(SIGTERM, _forward_signal)` install, both inside `launch_async`) kills kitty with no cleanup — a known accepted window, recovered by `kitty cleanup` |
| `SIGKILL`, then `kitty cleanup` | Cleanup runs `run_cleanup` against the redirected home. The `_kitty_values_present` heuristic (`launchers/claude.py:124-148`) fires on **either** a loopback `ANTHROPIC_BASE_URL` or `ANTHROPIC_AUTH_TOKEN == "kitty-bridge-token"`; a negative control uses a **non-loopback** URL with no token |
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

**Three lines of defence keep the committed bytes LF.** The byte-exactness claim above holds only if
every contributor's checkout sees the bytes the writer produced, and the rest of the suite cannot
tell the difference if it does not: `core.autocrlf=true` on Windows silently rewrites LF to CRLF
inside a `.body` on checkout and would commit the result back, changing both the bytes §3.3.2
asserts on and the lengths M3 and M5 decide by (T-W6 / **KBR-29**). `.gitattributes` is the first
line — `tests/corpus/*.body -text` and `tests/corpus/*.json -text` stop git translating either
path — and is pinned by `TestTheCorpusIsProtectedFromLineEndingTranslation::test_gitattributes_still_covers_the_corpus`
in `tests/harness/test_corpus_lint.py`. The manifest digest in `harness.corpus.load_entry` is the
second, in case the patterns are ever renamed, the directory moves, or the file is removed.
**KBR-261** closed a third hole the digest could not see: `write_entry` used
`Path.write_text(...)` with default newline translation, which emits LF on Linux and CRLF on
Windows regardless of `.gitattributes`, so a Windows regen run produced a CRLF manifest and CI
reported the resulting LF-vs-committed byte diff as pure line-ending drift. The writer now passes
`newline="\n"` so its bytes are LF on every platform;
`TestTheRoundTrip::test_the_manifest_bytes_are_lf_only` (`tests/harness/test_corpus.py`) pins
the writer end, and `TestTheCorpusIsProtectedFromLineEndingTranslation::test_no_committed_manifest_carries_crlf`
pins the committed tree.

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

**Delivered by T-E1 ([KBR-61])** in `tests/harness/containment.py`, the same pattern — seam and
policy in separate modules — continued one layer up. What it provides:

- **`SealedNetwork`** — proxy + recording upstream stood up together, the upstream addressed by
  :data:`HARNESS_UPSTREAM_HOST` at its own ephemeral port and the proxy's ``resolve`` map carrying
  exactly that ``host:port`` → ``127.0.0.1:port`` binding. T-E2–T-E5 read the same
  :class:`~harness.connect_proxy.ConnectProxy` and recording-upstream objects the harness holds,
  so sibling slices do not need a second pair.
- **`monkeypatched_aiohttp_resolver`** — the **direct**-leg override for the bridge's own aiohttp
  sessions: ``socket.getaddrinfo`` mapped for the harness hostname, deferring every other name to
  the real resolver. It is ``getaddrinfo``, not an aiohttp ``Resolver`` instance, because
  ``_build_client_session`` builds its own ``TCPConnector`` with no injection point (§5.3). The
  default (no-``aiodns``) build selects ``ThreadedResolver`` as ``DefaultResolver``, which reaches
  ``getaddrinfo`` in a worker thread; if ``aiodns`` ≥ 3.2 (the version whose ``DNSResolver``
  exposes ``getaddrinfo``) ever becomes importable, ``AsyncResolver`` is chosen instead and this
  patch has no effect. **KBR-259** pins that premise in code —
  `test_containment.py::TestMonkeypatchedResolver::test_default_resolver_is_threaded` fails
  loudly, naming ``AsyncResolver`` and ``aiodns``, the day the flip happens — so the seam no
  longer rests on this prose alone. pyproject pins no ``aiodns`` extra, so the seam holds
  today. ``/etc/hosts`` stays out — no administrator rights on CI runners.
- **The per-transport capability report** — `CapabilityReport`, an in-process singleton over the
  four §5.5 transports, every entry initialised ``not_attempted``; ``proven``, ``unsupported``
  (with a reason) and ``failed`` are recorded by T-E2–T-E5, and T-E9's completeness gate reads
  what they wrote. T-E1 records nothing — shipping the report and shipping verdicts are separate
  deliveries, per plan §8's rule that a transport task is done when it records an outcome.
- **The containment transport extension interface** — a ``ContainmentTransport`` protocol plus
  registry, with the bridge-aiohttp route registered as the default. Its ``direct_route``
  member is the per-transport override seam T-E3–T-E5 satisfy; nothing in this module changes
  for them.

The containment tests live at the **l1 path default** for the reason `test_bridge.py` records:
§8.2 forbids moving a test to a layer no job selects, and the Subsystem job is T-K6's. §8.2
lists `tests/harness/test_containment.py` as its T-E1 bullet, so T-K6 inherits the relocation.

**A missing `openssl` fails, it does not skip.** `certs` is shared infrastructure, and §8's rule
is that a skip in a gating job is a failure: a suite that quietly stops proving containment
because a tool is absent is indistinguishable from one that proves it.

[KBR-28]: https://shelpuk.atlassian.net/browse/KBR-28

### 7.3.1 T-E2 delivery ([KBR-62](https://shelpuk.atlassian.net/browse/KBR-62)) — the aiohttp slice, complete and falsified

T-E2 closes phases 2, 2b and 3 of §5.2.2 for the bridge's own aiohttp serving path, and
records the slice's verdict. What it added to the shared T-E1 harness core, and what it had
to add to make the phases provable:

- **`RecordingUpstream.start(ssl_context=None)`** — accept an optional server-side TLS context.
  The KBR-61 design assumed the bridge's aiohttp client CONNECT-tunnels for a plain-HTTP
  upstream, which is not how aiohttp behaves: for ``http://`` targets it sends the request in
  absolute form (``POST http://upstream...``), which the harness's CONNECT-only proxy answers
  with 405. With TLS at the recorder, the bridge's outbound URL is ``https://...``, aiohttp
  CONNECTs through the proxy, and §5.2.1's source-port join holds. The harness's
  ``SealedNetwork.start()`` passes ``server_ssl_context(certs.target_cert, certs.target_key)``
  — the throwaway leaf cert ``TlsTarget`` already uses, with ``HARNESS_UPSTREAM_HOST`` in its
  SAN, so the same key material serves both servers without a second generation pass.
- **`SealedNetwork.upstream_base_url`** — now ``https://...`` (not ``http://...``). T-E1's
  comment "plain HTTP at the upstream" is no longer true and the test that named it
  (``TestBridgeAiohttpContainment``) takes the ``aiohttp_trusts_test_ca`` fixture so the
  bridge's aiohttp client trusts the harness CA on both hops.
- **`BridgeAiohttpContainment.drive_with_egress(harness, *, egress, monkeypatch)`** — the
  phases-2/2b/3 entry point: same shape as ``drive_phase_1`` but passes ``egress=`` through
  to ``BridgeServer`` so ``_session_for`` routes public destinations through the proxy.
- **`CapabilityReport.reset_for_test()`** — test-scoped seam returning the singleton to its
  every-``not_attempted`` state. The autouse fixture on ``TestCapabilityReport`` (and on this
  module) calls it before each test so the KBR-61 "initial state" contract survives the
  verdict T-E2 records at the end of the slice, regardless of test ordering.

Phase 2 (proxy down ⇒ zero connections), phase 2b (every peer port joins a tunnel source
port, 407 contributes nothing) and phase 3 (an injected ``should_bypass`` bypass makes the
harness fail) live in ``tests/harness/test_aiohttp_containment_slice.py``. Phase 3's
monkeypatch lands on ``kitty.bridge.server`` (where ``_session_for`` resolves its own module
binding) — a patch on ``kitty.egress`` is silently ignored, which is the exact fail-by-silence
the falsification exists to prevent. The phases skip below Python 3.11 where aiohttp cannot
do TLS-in-TLS over stdlib asyncio, matching ``test_egress_https_proxy.py``; phase 1
(direct-leg TLS to the recorder, no proxy) runs on every supported Python.

The slice records ``Outcome.PROVEN`` for ``bridge_aiohttp`` into ``harness.containment.instance()``
at teardown. T-E1's contract test — "singleton exposes every transport ``not_attempted``" — was
updated to use ``reset_for_test`` via autouse, so T-E1's assertion holds regardless of
whether T-E2's verdict has been written in the same process.

### 7.4 Wire projections and the transparency oracle

**The projections (§3.3.1)** are the load-bearing piece: one hand-written reader per wire format —
Anthropic Messages, Chat Completions, OpenAI Responses, Gemini, Bedrock Converse, Ollama
`/api/chat` — each mapping a serialized body to the common `Conversation` form and **importing
nothing from `src/kitty`**. They are test code that the whole of I1 rests on, so they get
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
   builds this mapping. The mapping itself is built by `contract.residual_key()` — one call
   covers both rules and is the only spelling the seven readers use (KBR-193), so a
   disagreement is structurally impossible at the call site.
2. A nested key is keyed by its path from the body root with **array positions as indices** —
   `tools[0].type`, `messages[2].content[0].x_vendor_marker`, `system[0].x_vendor_marker`. It does
   **not** inherit §3.3.1a's by-name tool addressing. That convention exists because "translators
   reorder and filter declarations", which is a property of a *comparison*; a residual key is
   never matched against a register pattern, so the reason does not apply and one rule is better
   than two. T-D8 diffs residual key sets across all six readers and index-here/name-there is
   exactly the drift this section exists to stop.

**`Opaque` consumes its block, and carries a payload digest.** A block type the grammar does not
model projects as `Opaque(kind=…, digest=…)` where:

- `kind` is **named through `contract.opaque_kind()`** — not restated by each reader. For most
  types that is the wire `type` itself; where two vendors spell one concept differently the table
  reconciles them, and where a type is not snake_case at all it raises rather than converting (see
  below). Anthropic's spellings (`document`, `search_result`,
  `redacted_thinking`, `server_tool_use`) are already canonical; a format-unique type keeps its own
  spelling, which is the deliberate exception to "never the wire's spelling".

  **The alias table arrived with T-A3, not T-A5.** This section deferred it to "the first reader
  that needs one"; that was wrong, and the cost was paid before it was noticed. T-A1 and T-A3
  landed first and named one concept two ways — Anthropic's `document` and Responses' `file` for
  an attached file — which is the permanent unclaimed delta the deferral was meant to avoid.
  `document` is canonical because Anthropic Messages **and** Bedrock Converse both spell it that
  way on the wire, so exactly one reader moved. The table is `contract.OPAQUE_ALIASES` and it is
  **enforced**: `Opaque` rejects any key of it, so a reader cannot quietly project a rival name.
  The lesson generalises — a shared vocabulary deferred to the reader that first needs it is
  deferred to the *second* reader, because the first has already answered it alone.

  **A non-snake_case wire type with no alias raises.** `opaque_kind` does not convert it: a
  camelCase splitter with no caller and no corpus is a second source of drift, not a cure for one,
  so the conversion belongs to the author who first meets a real one. That is **T-A5, nine times**
  — Converse's `ContentBlock` union is `text` `image` `document` `video` `audio` `toolUse`
  `toolResult` `guardContent` `cachePoint` `reasoningContent` `citationsContent` `searchResult`
  `toolAddition` `toolRemoval` (confirmed against the `bedrock-runtime` 2023-09-30 service model),
  of which nine are camelCase. A **reader** meeting such a type translates the `ValueError` into
  `UnreadableBodyError`: the body is not projectable, but the reader is not at fault, and
  `contract` defines a reader-raised `ValueError` as a reader bug. A new *snake_case* vendor type
  needs no decision and still projects, so a vendor release is not a harness outage.
- `digest` has **two recipes**, and which applies is a property of the content, not of the format:

  * `contract.opaque_digest(block)` for a block the grammar cannot model — exactly
    `hashlib.sha256(json.dumps(rest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()`,
    where `rest` is the block without `type` and without `cache_control`.
  * `contract.text_digest(text)` for content whose identity is a run of text — a refusal is the
    case. One recipe was considered and rejected on a checked fact: Chat Completions carries a
    refusal as a **bare string** on the message (`ChatCompletionResponseMessage.refusal` is
    `anyOf[string, null]`), so there is no block for T-A2 to hash and a single rule would force it
    to invent a wrapper — and the wrapper's shape would be a new thing six readers could disagree
    about, which is this rule's own defect one level down.

  Both are pinned in `contract.py` rather than restated here (KBR-174), because prose did not hold
  the first one: in T-A1 all three wrong spellings survived mutation testing until a test pinned a
  digest to an external literal.
- the block's every other key is **consumed** — nothing beneath it residualises — while
  `cache_control` maps to `Opaque.cache_control` exactly as it does on a modelled block (KBR-167;
  it residualised until the grammar gained the slot).

Each clause is load-bearing. A bare `Opaque("document")` makes two different documents project
identically, so a swapped or truncated document produces no delta at all — and §3.3.1 put
`digest` on `Opaque` precisely to keep unmodelled content "detectable". Residualising every
payload key instead would fail the run on every `document` block, which is not a defect signal but
the grammar's known limit. **Canonical JSON rather than the raw wire slice**, because a translator
that reorders keys must not change the digest; that is the whole reason the recipe is not
`sha256(raw_block_bytes)`. **`ensure_ascii` is pinned** because its default is `True` while the
surrounding prose says UTF-8: an author who "helpfully" passes `False` gets a different digest for
the same block, and it would surface only on non-ASCII content. **`cache_control` is excluded from
the digest** so that one field behaves the same way everywhere — inside the digest it would produce
a delta with no named cause, on a path where the same field on a modelled block produces a
diagnosis. That exclusion is unchanged now the field has a slot, and it is what lets **M16** claim
a stripped breakpoint on an unmodelled block by path: were it inside the digest, the strip would
show as an opaque digest change no row could name.

> **The cost, recorded so it is not discovered later.** The digest is over *that format's* JSON, so
> one document carried from Messages to Converse digests differently and shows a cross-format
> delta no mutation caused. Modelling six vendors' block zoos is what §3.3.1 declined to do, so
> the alternative is not on offer. **T-D9's cross-format matrix is where this will first bite**,
> and the fix, when it is needed, is a per-kind payload rule here — not six readers each
> inventing one.

**A wrongly-typed leaf residualises — it is neither coerced nor raised on — and the rule is
general.** It binds *every* optional leaf, not the ones a bug happened to be found in: the
contract validates only `Turn.role`, `Conversation.sampling`, `extra["tool_choice"]` and —
KBR-191 — that an `extra` key names no nested value, so an
unguarded leaf declared `str | None` carries a dict silently with an empty residual. A reader
should apply it through one helper, so the next field added inherits it. `str(7)` and
`dict(["ab", "cd"])` invent a value the agent never sent, and a silently nulled tool description is
indistinguishable from the deletion §3.3.1's own falsification set injects. Raising is the other
wrong answer: it blinds the oracle to everything else in a request it could otherwise diff, and
`verify_total` cannot see a nested coercion because `consumed` is top-level only. So the field
residualises at its own path, the projection carries the grammar's absent value in its place, and
the run fails with the field named. **The one exception is the leaf the contract itself validates
— the key-shape rule — and the two read in opposite directions deliberately.** A wrongly-typed
leaf is vendor input arriving through a reader's own parse, so residualising keeps the rest of
the request diffable. A dotted `extra` key is not vendor input at all: `Envelope` is constructed
by harness readers, so the violation is a reader bug before it is ever a fidelity finding, and
`contract` already defines a reader-raised `ValueError` as a reader bug. Residualising it would
land the unclaimed address on `residual[...]`, where no register row can ever reach it.

Structural failures are the exception and still raise `UnreadableBodyError` — a role outside
`user`/`assistant`, a content block with no type — because there is no partial projection to
salvage: the turn cannot be built at all.

**A tool-result string is always `Text`, never `Json`.** `Json` is for a format that carries a
structured value natively — Converse's `toolResult.content.json`, Gemini's
`functionResponse.response`. A Messages or Chat Completions tool result whose content is the
*string* `'{"price": 259.75}'` is text that happens to parse. Without this fixed once, one reader
parses and another does not, and every JSON-shaped tool result shows an unclaimed delta on the
Messages ↔ Chat Completions comparison the oracle rests on.

**The Chat Completions reader's rule is its role set, and the merge rule's clause 3 is vacuous
on it.** Chat Completions delivers tool results in their own ``role: tool`` messages, so a run is
delimited by the wire and clause 1 builds the merged ``ToolResult`` turn from contiguous
``tool`` messages; an immediately following ``role: user`` message is a separate wire entry clause 2
absorbs into the same turn; clause 3 has nothing to reorder. T-A1's Anthropic Messages reader
applies the same four clauses per message (results and text inside one message, the run has no
natural boundary there); T-A2's Chat Completions reader applies them per *role*, with clause 3
deliberately a no-op — the two readers agree on the projection because the rule is the rule, not
because they happen to converge. KBR-25's scope addition pins this in `test_reader_chat_completions.
TestConvergence`: the CC and Messages encodings of one worked tool exchange project to identical
``Conversation`` values, indices included.

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
   register row name this?" about fields no row names, such as `output_config` (claimed on its
   own row **P5f**, KBR-44), and to residualise them. (`context_management` stood beside it until
   **P23** claimed it — which is the point: membership of that set is a fact about the register
   on the day you read it.)

A key kitty emits that **no** register row names is an *unregistered* mutation. That is the defect
the oracle exists to find, and it must residualise.

**The third outcome §3.3.1 promises exists now.** §3.3.1 says adding a field to a wire format
"forces a deliberate decision: map it, or declare it ignored with a reason". The contract
implements all three; the third is the **`IGNORED_BLOCK_FIELDS` registry** in
`tests/harness/contract.py` — a single, shared mapping from `(block wire type, field wire key)` to
the reason the reader consumes the field without modelling its value. Built once, importable by
every reader, because seven authors would otherwise answer this question seven ways.

**Why it shipped with KBR-205, and not before.** `cache_control` could have used it — and was
deliberately rejected from it (§3.3.1), because kitty's translated path **strips every block-level
breakpoint** and a declared-ignored rule would have made that cost invisible to the oracle. The
five Anthropic fields that arrived with KBR-205 carry no comparable consequence: they are optional,
descriptive, vendor-defined fields the agent neither reads nor writes — `text.citations`,
`image.transformations`, `tool_use.caller`, `tool_use.toolset_name` and
`tool_result.toolset_name`. None of them costs the user anything to lose, so a strip would not be
hiding a cost; it would only be hiding a *description* the reader was never going to model
anyway. The mechanism is the right answer when the field is one the grammar will not model **and**
losing it does not change behaviour the agent asked for.

**Five rules the mechanism follows, all of them written for the next field the grammar cannot
carry.**

1. **Path-keyed, exact wire spellings only.** Each entry names the block's wire `type` (the
   discriminator Anthropic publishes, like `"text"` or `"tool_use"`) and the field's wire key.
   No wildcards, no prefixes, no fuzzy match — a re-spelled sibling (`CitationS`, `Citation`) is a
   different entry the registry does not know, residualises, and fails the run, which is exactly
   the unregistered-mutation case §3.3.1 says a dropped field would have hidden.
2. **A reason is required per entry**, enforced by `ignored_field_problems` the way
   `row_shape_problems` enforces `not_projectable_reason` — a `None`, `""` or whitespace-only
   reason is rejected. The reason is the audit trail: a future reader can answer "why does this
   field disappear?" without having to read the ticket.
3. **The reader consumes the field, never the residual.** A key in `IGNORED_BLOCK_FIELDS` is added
   to the reader's mapped-keys set, the way `cache_control` is on every modelled block; the value
   is dropped (no slot, no `Envelope.extra`, nothing downstream reads it). `verify_total` sees the
   field as accounted for, which is the property that lets it not fail the run.
4. **The mechanism is shared, not per-reader.** `contract.IGNORED_BLOCK_FIELDS` is the single
   source; every reader imports the same registry and asks it the same way. A reader that wanted
   to ignore a field its five siblings did not would have to propose a new entry — the audit trail
   the registry carries is also a coordination guard.
5. **A falsification case ships in the suite.** §1.4's harness rule: a mechanism never shown to
   fail is indistinguishable from one that cannot. The test suite pins (i) `verify_total` passes
   on a body carrying each declared-ignored field; (ii) a re-spelled sibling still residualises
   and fails the run; (iii) `ignored_field_problems` rejects a registry entry without a reason.

**Two Anthropic tool fields that are *not* declared-ignored** (KBR-214's design review of KBR-205):

- `tool_choice.disable_parallel_tool_use` — Anthropic's nested, inverted flag is mapped onto
  `envelope.extra["parallel_tool_calls"]` (one address, one polarity, fixed in §3.3.1b so the Chat
  Completions reader meets the same rule). Mapping it makes a real mutation detectable: a
  product that strips the flag would now show a fidelity finding on a request that asked for
  non-parallel tool calls. Declaring it ignored would have made that loss invisible.
- `tools[i].type` — the discriminator on a tool declaration carries information the oracle must
  see: KBR-214 D10 uses it to decide whether a forced call is to a client-declared or
  Anthropic-defined tool, and G35's register row (KBR-184) needs to read it to anchor its
  conditional on Anthropic-defined types (`web_search_20250305` and friends). Declaring it
  ignored would have made that decision invisible to the oracle.

Both are mapped rather than ignored for the same reason: a mutation on them changes what the
agent asked for, or what the product decides about the request — exactly the property a declared-
ignored mechanism is designed not to see.

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
| 2 — a nested control field flattens to its leaf published key | **T-A5**, whose `inferenceConfig` and `toolConfig` nest the same way, and **T-A6**, whose `options` is the same flattening pattern |
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
| a **required field** of a part — a tool `name`, `fileData.fileUri` | **residualise**, project the part: a tool `name` projects `ToolUse(name="")`; `fileData.fileUri` projects with the canonical-JSON digest of the blob as identity | §3.3.1b settles it in those words: "an absent `name` *does* residualise … a call nobody can name cannot be paired or addressed". `Image.__post_init__` now enforces XOR (KBR-192), so a missing `fileUri` cannot project `Image(ref=None)`; the part keeps its position with the `opaque_digest` recipe as identity, and the residual records the missing or wrongly-typed value. |
| a **payload** the reader cannot canonicalise — base64 that does not decode | **residualise the leaf**, project the part with the raw-bytes digest | `Image.__post_init__` enforces XOR (KBR-192), so a bare `None` digest is illegal. The reader digests the wire's own raw bytes (`image_digest(raw.encode("utf-8"))`), giving the part a distinguishing identity; §7.4.1: "raising is the other wrong answer: it blinds the oracle to everything else in a request it could otherwise diff". |
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

> **Reconciliation owed — paid 2026-09-14 (KBR-251).** The undecodable-base64 shape this rule names
> is now conformed across all three shipped readers. `reader_anthropic_messages._read_image`
> residualises the `…source.data` leaf and projects the part under
> `image_digest(raw.encode("utf-8"))` — the second of `image_digest`'s recipes (KBR-192) — with
> the media type and breakpoint carried as before. `reader_responses._read_image` never returns
> *no part*: both data-URL branches residualise the `…image_url` leaf (digest of the payload
> string where the base64 grammar parsed, digest of the payload after the first comma where it
> did not — with the media segment parsed from the URL prefix and carried separately, keeping
> `image_digest`'s "media type excluded" rule), and the no-`image_url`-no-`file_id` branch
> carries the `opaque_digest(part)` identity Gemini's missing-`fileUri` shape already used, the
> residual naming the absent or wrongly-typed key (a present-but-wrongly-typed `image_url`
> residualises even when a usable `file_id` carries the part). `reader_gemini.py` remains the
> reference implementation.
>
> **What still diverges, and is owed elsewhere.** Anthropic's missing/wrongly-typed
> `source.data` (`KeyError` / `TypeError` from `b64decode`) still escapes the reader. Two further
> items this note once named are **paid 2026-09-16 (KBR-179)**: the `input_file` sibling sweep
> (the part now carries `opaque_digest(entry)`, with the same no-op `_residualise` sweep the
> Anthropic reader's `_read_opaque` runs, so a mutated `detail` — a key the published
> `InputFileContentParam` schema lists but no Opaque slot covers — shows as a digest delta at the
> part path rather than vanishing), and the `_DATA_URL` regex (`[^;,]*` plus `re.IGNORECASE`, so
> `data:;base64,…` and a case-variant `;BASE64,` route to the base64 branch by design). KBR-179
> also aligned the Responses reader's structural boundary with the Messages reader's: an input
> item whose `type` is present but neither a string nor `null` raises `UnreadableBodyError`
> naming the path — `Opaque.kind` *is* its value, per §7.4.1.
>
> **One asymmetry, recorded so it is not mistaken for an oversight.** The Anthropic
> `_read_opaque` does *three* things where the Responses branches do *one*: the no-op sweep,
> the `cache_control` slot (`Opaque.cache_control = _read_cache_control(…)`), and the
> `opaque_kind` translation. The Responses branches match only the first — deliberately.
> Responses items and content parts do not carry `cache_control` (OpenAI's spelling is
> `prompt_cache_breakpoint`, on `InputTextContentParam` / `InputImageContentParam` /
> `InputFileContentParam`); a reader that filled `Opaque.cache_control` would invent a value
> the wire never sent. `prompt_cache_breakpoint` rides in the digest until gap **G37** lands a
> slot for it — visible as a part-level delta, not a clean residual entry, but visible. The
> broader sweep
> this note once implied — the Responses dispatcher's other `return None` paths for wrongly-
> typed text and refusal parts — stays tracked under KBR-196, consolidated here.

> **What this reader leaves on the record — and when it was paid.** Six fields the format
> publishes, real clients send, and the grammar could not carry residualised and so **failed the
> first oracle run** — the same shape as the `cache_control` deadline above, tracked as its own
> defect ([KBR-194](https://github.com/Shelpuk-AI-Technology-Consulting/kitty-bridge)). **KBR-194
> met the deadline by growing the six slots**: `ToolUse.signature` (the sharp one — Gemini 3
> *requires* clients to echo a functionCall's `thoughtSignature` back verbatim, and unlike
> `cache_control` the grammar nearly had the slot, `Thinking.signature`), `Conversation.system_role`
> for the role Google's SDKs set on `systemInstruction`, `ToolDecl.behavior` and
> `ToolResult.scheduling` for the two halves of NON_BLOCKING calling, and `Text.video_metadata` /
> `Image.video_metadata` / `Image.display_name` for the part modifiers. §3.3.1a grew
> `conversation.system_role` — the first path form at conversation scope — so a register row can
> anchor there. The Gemini reader's L1 inventory asserts all six flow into their slots.
>
> A second consequence, on the oracle rather than the reader: `GeminiTranslator` **discards** the
> inbound tool-call id and synthesises one per call (`_make_tool_call_id`). Now that the §3.3.1 correction above
> projects the wire id, every tool turn from a client that populates one shows a delta with no
> register row to claim it. That is a correct oracle finding, not a reader defect, and it needs a
> row or a ticket before T-D9 runs.

#### 7.4.3 What T-A5 settled — Converse-specific decisions, and the three §7.4.2 rules it inherits

The Bedrock Converse reader inherits three of §7.4.2's seven rules and settles three Converse-specific
decisions on top. They are recorded here for the same reason §7.4.2 names — six readers, path-keyed
residuals, and the same drift if any of them is restated per-reader.

**Rule 2 (inherited).** Converse nests ``inferenceConfig`` (maxTokens / temperature / topP /
stopSequences) and ``toolConfig.toolChoice`` the same way Gemini nests ``generationConfig`` /
``toolConfig.functionCallingConfig``, so the leaf-key-on-the-narrowest-anchor rule applies. The
reader maps each of the four sampling leaves to its canonical §3.3.1b spelling (``max_tokens``,
``temperature``, ``top_p``, ``stop``) under :attr:`~harness.contract.Conversation.sampling`; ``tool_choice``
unifies onto ``envelope.extra["tool_choice"]``. The container addresses are *not* emitted — the
test class :class:`TestExtraKeyDisjointness` pins the four flattening sources pairwise-disjoint
so a future schema revision that adds a sibling under ``inferenceConfig`` (or any other container)
fails that test, not the reader's contract.

**Rule 5 (inherited).** Converse's ``Tool`` union is three members: ``toolSpec`` (a client tool
spec, becomes :class:`~harness.contract.ToolDecl`), plus ``cachePoint`` and ``systemTool``. The
last two have no ``name`` to bind (cachePoint is a marker, systemTool is a server-side reference),
so :attr:`~harness.contract.Conversation.tools` cannot hold them — they project to
``envelope.extra["cachePoint"]`` and ``envelope.extra["systemTool"]`` verbatim. Two ``cachePoint``
entries inside ``toolConfig.tools`` are schema-legal (Converse allows up to four cachePoints per
request across system / messages / tools), so the reader takes the first value and residualises
the duplicate at its entry path (§7.4.2 rule 2's "loser overwrites winner with no residual"
hazard).

**Rule 7 (inherited).** §7.4.2's "no branch ever returns *no part*" applies to Converse too, and the
five reader helpers it names — ``_read_image``, ``_read_tool_use``, ``_read_tool_result``,
``_read_reasoning_content``, and the toolResult content dispatcher — are KBR-251-conformed:
when a leaf is wrongly typed, the leaf residualises AND the part is produced (identity from the
wire's own bytes via :func:`harness.contract.image_digest`'s second recipe (KBR-192), or
canonical-JSON digest for the helpers where bytes do not exist). A non-Mapping member
(``toolUse``, ``toolResult``, ``reasoningContent``, content block) raises
:class:`~harness.contract.UnreadableBodyError` per rule 7 row 2 — "the member itself is wrong,
the schema says an object and the wire sent a scalar."

**Three Converse-specific decisions:**

**1. ``cachePoint`` is an ``Opaque``, not a ``cache_control`` slot.** §3.3.1 records Converse's
``cachePoint`` as a separate block in the content list, not a field on one. The reader projects
the block itself to :class:`~harness.contract.Opaque` with the canonical kind ``cache_point``
(reached via :func:`harness.contract.opaque_kind`), and does **not** fill
:attr:`~harness.contract.Text.cache_control` or :attr:`~harness.contract.Opaque.cache_control`.
This is the design's deliberate exception for Converse — Anthropic carries ``cache_control``,
Converse does not, and forcing one slot for two spellings would put a vendor name into a
wire-independent value (§3.3.1's P16 objection). The same applies to ``cachePoint`` and
``guardContent`` inside :attr:`~harness.contract.Conversation.system`: text lifts into
:attr:`~harness.contract.Conversation.system`; the other two residualise at the entry path
so the totality check sees them. §11 entry ``Q-cache-control-converse`` records this.

**2. ``toolResult.content[i]`` may carry a ``json`` block.** Converse publishes a ``json``
ContentBlock inside tool results — the schema is ``{json: <Document>}``, where Document is
any JSON value (object, array, scalar). The reader maps ``json`` to
:class:`~harness.contract.Json`, which carries the value verbatim (object, array, or scalar);
``decode_arguments`` is the string-carrying-formats tool and would reject the object form on
this wire. A non-Mapping ``input`` on a ``toolUse`` (Converse's natively-object form)
residualises at its leaf AND the part is projected as ``ToolUse(arguments={})`` per §7.4.2
rule 7 row 2 (the inherited rule §7.4.3 records three lines above); the empty-mapping
default matches what §3.3.1b uses for absent arguments. ``decode_arguments`` still rejects a
value the wire could not carry.

**3. The reader consumes the URL.** :attr:`~harness.contract.CapturedRequest.path` carries
``/model/{modelId}/converse`` or ``/model/{modelId}/converse-stream`` — the model is a URI
parameter (NOT a wire-body key) and the streaming flag is the operation (NOT a body field). The
second reader to consume the URL after Gemini; ``TestRoute`` owns §3.3.5's claim that two
byte-identical bodies on ``converse`` and ``converse-stream`` yield different
``envelope.stream`` values.

### 7.5 The bridge fixture

`tests/harness/bridge.py` — plan task **T-W8** ([KBR-31]). The counterpart of §7.2 on the
inbound side: §7.2 says what a recording upstream must observe, this says how a **real
`BridgeServer`** is put in front of one. Around fifteen tasks across Epics D, E, G, I, J and K
need it, and `tests/conftest.py` offers only `unused_tcp_port` today. Measured in-repo: of the
**62** test modules under `tests/bridge/`, **38** start a real `BridgeServer` and most build
their own stub adapter, fake upstream and `post()` helper; 25 of those *also* intercept the
upstream with `aioresponses` rather than a real socket, and 24 start no server at all.

**This was the one `tests/harness/` module that imports the product, and since T-B1 it is one of
two** — an Epic B transport must build the adapter it binds, which is what `bind()` is for
(§7.2.2). `contract.py` and
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

### 7.6 The proven vertical slice

`tests/harness/test_vertical_slice.py` — plan task **T-W9** ([KBR-32]). §7.2 says what a
recording upstream must observe, §7.5 how the product is put in front of one, and §3.3.1 what a
projection reads. This is the first place all three are driven **together**: one request from a
real `BridgeServer`, into the recorder, and out into T-W2's declared types. Small, and it is the
moment the contracts become known to *compose* rather than merely to exist — six streams build
on that afterwards.

**Four claims, and the falsification that keeps each honest.**

| Claim | Assertion | The defect it catches |
|---|---|---|
| One inbound request, one upstream request | Exactly one capture; inbound status 200 | A fired retry ladder. Scoped against §7.5.4's identical-looking assertion: T-W8's is about **one binding**, this is the product claim for a **driven request** |
| The capture is complete | Each of T-W2's seven fields at its declared type and wire value; `arrival` and `peer_port` populated | A capture that silently stops carrying §5.2.1's join key, which nothing else would notice until T-E2 |
| The capture is **readable** | A minimal reader satisfies `Projection`, and `verify_total` passes over what it produces | A capture that type-checks and that no projection can consume — the composition failure this task exists to rule out |
| The query survives | The capture's `query` equals what the bridge sent, percent-encoding and duplicate names intact | A recorder that drops the query string, driven **end to end** |

**The default binding carries no query, so the falsification had to be made non-vacuous.**
Measured: `AiohttpTransport.bind()` returns `{"base_url": recorder.base_url}`, and the capture's
`query` is then `""` — against which a query-dropping recorder passes perfectly. The slice
therefore drives a binding whose `base_url` carries one, merged onto the adapter's endpoint path
by `ProviderAdapter.compose_upstream_url` (the KBR-143 rule). That is the **product's own
channel**, not a patch arranged for the test: a query the harness injected some other way would
prove the recorder reads `raw_query_string` and nothing about whether the bridge preserves a
query at all.

**Both properties of the query literal are load-bearing, and an earlier draft had neither.** It is
`kbr32=slice%20value&dup=1&dup=2`: the `%20` would become `+` under a `parse_qsl`/`urlencode`
round trip, and `dup` appears **twice** so a recorder that collapses duplicates is
distinguishable from one that preserves them. A first draft wrote `dup` once, which made the
duplicate half of the claim untestable — §1.4's shape again, caught in review.

**Byte-for-byte survival is specific to these two adapters.** `compose_upstream_url` *merges*
when both sides carry a query and drops base parameters whose name the endpoint also uses;
`custom_anthropic` and `custom_openai` contribute no endpoint query, so the base query survives
verbatim. The same assertion against Azure would be false, which is why it is stated here rather
than generalised.

**The transports this module defines are never registered.** They are constructed directly and
driven, following §7.5.4's rule that a shared registry must not hold things that are wrong on
purpose. It is also load-bearing for the gate: `test_bridge.py`'s meta-test asserts
`set(registered_transports()) == EXPECTED_TRANSPORTS`, and registration happens at **import
time** on module-global state — so a stray `register_transport` here would leave this module
green on its own and the full suite red. That is an order-dependent failure and a direct hit on
plan §1.3(5), "it lands on `main` alone".

**The `peer_port` assertion asserts the join, not the type.** §5.2.1 joins captures to tunnels on
that key, and an `isinstance(..., int)` check passes for a port matching no connection — which is
precisely the column T-E2 would then be joining against nothing. The slice asserts the capture's
port is among the ports of the connections the upstream actually accepted.

**§7.2.1's `has_content` claim is discharged here**, because it is the only place it can be. It is
a local flag inside `BridgeServer._stream_chat_completions`, not a callable, so no unit test
reaches it; T-W4's streams satisfy its precondition by construction. Driving one `stream: true`
request end to end and asserting a single capture is the evidence that the flag was set and the
empty-response retry never fired. Named by symbol, not by line: the plan's `server.py:5145`
anchor is already stale, and §7.2.1 states this in prose rather than in a table.

**That case asserts the capture count and nothing else, deliberately.** A downstream status
assertion there would be unfalsifiable: `_stream_chat_completions` commits the 200 with
`sr.prepare()` before it opens the upstream at all, so every outcome on that route is a 200 —
measured, a bridge with `has_content` forced false does not answer non-200, it fails to complete.
An assertion nothing can kill is what §7.5.4 removed from T-W8 and what §1.4 forbids, so it is
not shipped. The non-streaming case **does** assert the status, where §7.5.4's measured row 3 (an
upstream 400, one correct capture, a failed client) is reachable and T-W8's `_RefusingTransport`
is the defect that proves the assertion bites.

It is a claim about one **pair** of axes and not about either alone (§7.5.1): inbound
`chat_completions` over upstream `CHAT_COMPLETIONS`, streaming.

**A wall-clock bound, and why it is an assertion rather than a timeout.** `_EMPTY_RETRY_DELAYS`
+ `_EMPTY_FINAL_DELAYS` is 80 seconds of real `asyncio.sleep`, and a regression that reintroduces
the ladder must make the suite **red**, not merely slow — the defect commit `691e974` fixed that
once already. Three mechanisms were rejected before the one that ships:

- `pytest-timeout` is **not** in the dev extras.
- `asyncio.timeout` is **3.11+**, and the matrix is 3.10–3.13.
- `BridgeFixture.post`'s own client timeout **does not bound the slice**. Measured: a `post()`
  given 2 seconds against a still-sleeping ladder exited its block after **62 seconds**, because
  `BridgeFixture.stop` waits for in-flight upstream handlers rather than aborting them — §7.5's
  own documented property, and the reason §7.3 took the opposite decision for the proxy.

What ships is `time.monotonic()` around each driven request, asserted against a **4.0 s** budget:
200× the measured healthy time (0.01–0.02 s).

**What that budget catches is not "a ladder".** An earlier draft justified it as "strictly below
`_EMPTY_RETRY_DELAYS[0]`, so any ladder that sleeps trips it", and that is wrong twice over. Any
ladder that fires leaves an **extra capture**, so the one-capture assertion sees it first and sees
it deterministically, with no dependence on a runner's speed. And there are **two ladders, not
one**: `_request_with_retry_single` — the *non-streaming* helper — is the only site that
**sleeps** `_EMPTY_RETRY_DELAYS`, while every streaming path sleeps
`_BACKOFF_BASE * 2 ** (attempt % 4)`, i.e. 1, 2, 4, 8 s, reaching `_EMPTY_FINAL_DELAYS` only on
its last two attempts. (Say *sleeps*, not *reads*: one other site reads the list's **length**, to
report an attempt total. An earlier draft of this paragraph said "read at exactly one site",
which is simply false about the source tree — and a docstring citing another module's internals
as an invariant is a failure mode this repo has already been bitten by. It is also why T-W9's
monkeypatch changes those values and leaves the lengths alone.) A streaming
ladder can therefore fire **twice inside a 4-second budget**. That fact is recorded here because
it is non-obvious and the next author will otherwise repeat the mistake.

The honest division of labour, and the reason all three are kept:

| Mechanism | The band only it covers |
|---|---|
| The one-capture assertion | **Any** retry, at any sleep length, deterministically |
| The 4.0 s budget | **Slow with a single capture** — a connect grace, a wedged handler, a non-ladder regression |
| `post()`'s `DEFAULT_TIMEOUT` (10 s) | Everything above ten seconds, with a message naming the transport and both ladder costs |

The budget's failure message reports the capture count **first** and says so in words, because the
gate is ~18.5 minutes on runners this repo has already seen OOM-killed under parallel load: a
contended runner and a fired ladder must be distinguishable without a rerun. CI is the authority
for the number, across every leg — 3.10–3.13 on Linux plus the pinned Windows and macOS legs
§8.4 added after this module was written — and a platform- or version-dependent failure means
raising it, never skipping.

**The coarse clock reached this module twice, and only one of them was foreseen.** The second was
found by the Windows leg itself: `test_the_budget_is_enforced_on_every_driven_request` drove a
request with an "impossible" budget of literal `0.0`, on the reasoning that zero is over-budget
for any real request. On Windows `elapsed` measured **exactly** `0.0`, so `0.0 <= 0.0` held and
the case failed with "DID NOT RAISE" while all five other legs were green. The impossible budget
is **negative** now, which no measurement can satisfy at any resolution.

Worth separating from the rule two paragraphs above, because the remedy was **not** that rule.
"A platform-dependent failure means raising the budget, never skipping" governs the **4.0 s**
budget, whose margin is a judgement about runner speed. This was a different defect: an assertion
written so that it *could not fire* on a coarse clock. The fix was to make it
resolution-independent, not to widen a margin — and no §8.3 row was added, because the platform
dependence was removed rather than amnestied.

**The slice needs no Windows exemption row, and must not acquire one.** §8.3's registry is all
arrival *ordering*: Windows' clock cannot separate two adjacent requests, so every "arrival
increases" assertion is false there (KBR-188). This module asserts only that `arrival` is
**populated and typed**, which a coarse clock satisfies — ordering is T-W4's claim, not the
slice's. Tightening it into an ordering check here would add debt to §8.3 for a claim that is
already made, and made better, one layer down.

**The fired ladder is shown, and it is shown on the non-streaming path.** Plan §1.4 requires the
one-capture assertion to be caught detecting a real ladder rather than passing by construction,
so the module ships an upstream reply the bridge judges *empty*, with the ladder's delays
flattened to zero by `monkeypatch`: measured at **five captures in 0.008 s**, deterministic and
free. It is deliberately **not** done on the streaming path: with `has_content` mutated to never
become true, the driven request was measured **never completing at all**, so a streaming version
of this defect would hang the gate rather than fail it. Flattening the delays rather than waiting
them out is what keeps a falsification case affordable in a gate that already runs ~18.5 minutes
per Python version.

**The reader is local and minimal, and it is deliberately not T-A1's — which has now landed.**
T-W9's declared dependencies are T-W1, T-W2, T-W4 and T-W8 and **no reader task** (§7.5.4), and
that is worth keeping now that `reader_anthropic_messages.py` exists: a slice that went red
because a reader had a bug would mis-attribute the failure, and the claim here is only that the
recorder's output is readable **at all** by something shaped like a `Projection`. It is
correspondingly **not** a `reader_<format>.py` module — §7.4.1 closes that set at six, one per
`WireFormat` member, and this is evidence rather than a seventh projection. How a format *ought*
to be projected stays T-A1–T-A6's question.

**`consumed` is built from literal key names, and that is load-bearing.** `verify_total` computes
`set(source) - (consumed | residual)`, so a reader deriving `consumed = frozenset(source)` passes
unconditionally — the same shape §1.4 forbids. The deliberate consequence is that a product change
adding a new top-level key to the upstream body turns the slice **red**, which is the signal worth
having: an unregistered addition is exactly what §3.2's register exists to catch.

**The falsification cases live in this module rather than a sibling.** T-W4 and T-W8 each split
theirs out because each ships four or more deliberate defects. T-W9 ships two, and a separate
60-line module would cost a reader a file hop to reach the defect that falsifies the assertion
three lines above it. Recorded so the divergence reads as a decision and not as an oversight.

**What the falsification sweep killed, and what it could not.** §1.4 asks for the procedure;
what is worth recording is its result. Each of the recorder's capture fields was mutated in turn
and the module re-run:

- **Every** mutation is caught: wrong method, scheme, host, headers, body, arrival, a wrong or
  absent peer port, and a dropped, percent-decoded or duplicate-collapsing query. Most red
  exactly one case; a corrupted body reds three, and each query mutation reds two, because the
  field-completeness case asserts the query alongside the six other fields and the dedicated
  query case asserts it alone. That overlap is deliberate — one case is "the capture is
  complete", the other is "routing survives" — but it means the sweep shows *caught*, not
  *uniquely attributed*. An earlier draft of this paragraph claimed the stronger property, which
  the sweep's own output contradicts.
- Mutating `host` to aiohttp's `request.host` kills nothing, and that is an **equivalent
  mutant** rather than a hole: aiohttp returns the Host header whenever one is present and the
  bridge always sends one. Simulating the real §7.2.1 defect — the `socket.getfqdn()` fallback —
  reds exactly one case, which is what makes the `host` row honest.
- **`path` read percent-decoded killed nothing.** The adapter's own path is `/v1/messages`, which
  contains nothing encoded, so `request.path` and `rel_url.raw_path` are identical on this route
  and §7.2.1's trap was untestable here. The fix was to carry `%20` in the base URL's path
  component too, exactly as the query already did; the mutation then fails the field case. This
  is the second time in this task that a claim turned out to be vacuous because the *driven
  request* did not carry the thing the claim was about — the first being the query itself — and
  it is the argument for running the sweep rather than reasoning about it.
- **One assertion survived on purpose.** The non-streaming case's `status == 200` is falsified by
  nothing in this module. It is not redundant — §7.5.4's measured row 3 is an upstream 400, which
  yields one correct capture and a failed client — and T-W8's `_RefusingTransport` is the defect
  that proves it bites. Re-shipping that defect here would duplicate T-W8 rather than add
  evidence, so the assertion is kept with the argument written beside it rather than deleted or
  left unexplained.
- **One assertion is weaker than it looks.** `scheme` is built from a recorder *constant*, not
  from anything on the wire, so asserting `"http"` pins the recorder's own declaration. T-B2's
  TLS transport is where the field starts carrying information.

**What this task does not settle.** Corpus-wide quantification is **T-D1**'s ([KBR-51]); the
other five wire formats and the three custom transports are **T-B1–T-B3**'s; non-loopback
addressing stays with **T-E1**/**T-E2** (§5.3). Bounding fixture *teardown* against a fired
ladder is real — the 62 seconds above — but aborting in-flight handlers is a change to §7.5's
module and a decision §7.3 already took for the proxy; it is left to **T-K6** and **T-B4** rather
than taken here as a side effect.

[KBR-32]: https://shelpuk.atlassian.net/browse/KBR-32
[KBR-51]: https://shelpuk.atlassian.net/browse/KBR-51

---

## 8. CI cadence

One executable selection matrix. Every test carries exactly one layer marker
(`l1`, `l2`, `l3`, `acceptance`, `agent_smoke`, `agent_live`, `eval`, `load`), so a job is
defined by its marker expression and no test can fall between two jobs or into both.

| Job | Selection | Trigger | Gates a PR? | Gates a release? |
|---|---|---|---|---|
| **Fast** | `ruff`, `lint-imports`, `mypy src/kitty`, then `pytest -m "l1 or l2" -q` on Python 3.10–3.13 on Linux, and on one pinned version on Windows and macOS (§8.4) | push, PR | **Yes** | **Yes** |
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

**The "Gates a release" and "Gates a PR" columns are now enforced, not merely stated
(KBR-152).** Repository ruleset `21038306` carries a `required_status_checks` rule
that names `ci-required` and nothing else — the aggregate that `ci.yml`'s 🔴-marked
header says every job in that file feeds. Before KBR-152 the ruleset carried
`deletion`, `non_fast_forward` and `pull_request` only, and a pull request with a
four-version red test matrix could merge through the GitHub UI or API; the gate
computed a verdict nothing obliged anyone to honour. The four "gates a release"
rows above are also "gates a PR" rows by construction — the same matrix, the same
aggregate — so a release branch that bypasses the PR gate still passes through
`publish.yml`, which calls `tests.yml` for the same checks. The two product-
owner decisions the ruleset carries alongside the new rule (`required_approving_
review_count: 0` and the single `bypass_mode: always` actor) are outside KBR-152's
scope and untouched.

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
`acceptance`, `agent_smoke`, `eval` and `load` are selected by **no job at all**.
`agent_live` has one job, and it is **not** the nightly: `.github/workflows/tmux-disconnect.yml`
runs exactly one file, `tests/integration/test_tmux_disconnect.py`, on pull requests that touch
the `--tmux` wrap (`SYSTEM_DESIGN.md` §3.4). That job removed `agent_live` from
`PENDING_ACTIVATION_LAYERS`, so the registry no longer records that
`tests/integration/test_agent_e2e.py` still runs nowhere: **T-K11, the Agent-live nightly, is
still owed** (product owner, 2026-09-13).
What each of those jobs would need from the runner is §8.6; for `agent_smoke` and `agent_live`
the answer is that CI has had it all along, so what they are still waiting on is the tests, not
the resources. **`eval` is not in that sentence:** its runner needs are met too, but T-K12 waits
on T-K3, which waits on Q4 and Q13 — both still open. A resource being available does not close a
decision.

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
Twelve modules under `tests/` bind real sockets or spawn processes and are `l1` by default
today — `test_egress_https_proxy.py` foremost among them (an earlier draft said "roughly six";
the count has grown as Epic B, E and the KBR-132/144/176/220 fixes each landed a socket-binding
module, and the bullet list below is now the authoritative enumeration). Since T-W5 that file's
shared fixture plus `tests/harness/test_connect_proxy.py` must move **with** it: an
extraction and its own regression evidence landing in two different jobs would leave one proving
the other in a run that no longer includes it. Reclassifying them is correct and is T-K6's
business, together with the job that runs them; doing it earlier would remove them from every
gate. T-H1 must take that reclassification into account before it measures a mutation
baseline, because it selects on `l1`.

**Eleven modules are bulleted below — in nine bullets, since the T-W4 and T-W8 rows name two
modules each — and `tests/cli/test_stream_encoding.py` (KBR-10) is described after them, twelve in
all, named here so T-K6 inherits a list rather than a search** — the count
is what T-K6 and T-H1 plan against. (The bullet count and the KBR-10 paragraph were already
drifting apart before T-W8 added two; spelling out both is what stops the next addition
guessing which set it joins. T-W9 joins the **bulleted** set, not the paragraph above it, which
still names `tests/test_egress_https_proxy.py` and `tests/harness/test_connect_proxy.py`
separately.)

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
- **T-W9 (KBR-32):** `tests/harness/test_vertical_slice.py` starts a real `BridgeServer` and a
  recorder in nine of its ten cases. The module runs in **~0.5 seconds**, measured (0.42–0.54 s
  over three runs), which is the number the fast-gate budget should carry until T-K6 moves it.
  Its falsification cases are deliberately cheap: the empty-response ladder is flattened with
  `monkeypatch` rather than waited out, because an 80-second falsification case is the defect
  commit `691e974` fixed once already. Worth knowing while planning the move: when one of its
  cases goes red on a fired ladder, the *bound* catches the regression but does not bound the
  cost of catching it — teardown still waits out the in-flight handlers, measured at 62 seconds.
  T-B1 below measures the same property at 72 s through the adapter; they are the same 62-second
  teardown plus that path's own ladder, not two different findings.
- **T-B1 (KBR-40):** `tests/harness/test_provider_aiohttp.py` binds a recorder in most of its
  cases and a real `BridgeServer` in several, and drives the OpenAI login OAuth leg over loopback
  in four more. It runs in **~0.6 seconds**, measured, which is the number the fast-gate budget
  should carry until T-K6 moves it. It is one module rather than two because its falsification
  cases are defects in the transport it ships, not a separate harness; where T-W8 put four defect
  transports in their own file, a fifth file per Epic B ticket would be three more for T-K6 to
  move. Worth knowing while planning that move: a **wrong-shaped reply** costs 72 seconds here —
  10 s of retry ladder plus the teardown that waits it out — which is why §7.2.2's reply-shape
  falsification is driven through the adapter rather than through the bridge.
- **T-E1 (KBR-61):** `tests/harness/test_containment.py` drives a real `BridgeServer` against the
  sealed-network harness (`ConnectProxy` + recording upstream) in two cases — one green, one
  falsification. The other 23 cases (KBR-259 added this module's resolver pin test, a ~0.3 s
  case) run in **~0.9–2.1 s**, re-measured across three runs, of which
  the slowest is one `sealed_network` setup (recorder + proxy + TLS certs); the range comes from
  `openssl`-generated throwaway certs, whose cost varies with runner load. The falsification case
  (`test_drive_phase_1_with_a_broken_resolver_records_zero_connections`) takes **~30 s** because
  `_make_upstream_request`'s `_wait_out_transport_blip` (`server.py:8579`) sleeps through
  `_TRANSPORT_GRACE_DELAYS = (2.0, 4.0, 8.0, 16.0)` (`server.py:1030`) summing to
  `_TRANSPORT_GRACE_PERIOD = 30.0` (`server.py:1029`) when the resolver is closed, and
  `stop_async` drains the in-flight handlers. The cost is on the *failure* path; the green path
  completes in ~0.01 s. Reclassifying the module to `l3` is T-K6's call — the runtime figure is
  what the Subsystem job's budget must carry.
- **KBR-144:** `tests/bridge/test_responses_string_input.py` starts a real `BridgeServer` on an
  ephemeral port in four of its classes, following the existing convention of
  `tests/bridge/test_crash_resilience.py` rather than inventing a second one. The whole module
  runs in **~0.6 seconds**, measured, of which the socket-binding cases are ~0.1.
- **KBR-176:** `tests/bridge/test_bridge_management.py` spawns real child interpreters in four
  cases of `TestStartBridgeWithALoudChild`, and already bound loopback sockets in
  `TestBridgeReachable` before that, unlisted. It has no choice for the children: the defect is a
  child blocked inside `write()` on a pipe nobody reads, which only a real pipe and a real writer
  can show — a stand-in stream never blocks — and one case runs a whole second interpreter as the
  CLI, because how an interpreter exits around a thread still blocked on that pipe is platform
  behaviour. The four spawning cases take **~0.6 seconds** together and the whole module
  **~3 seconds**, measured. The child is a `python -c` script, never `kitty.bridge_runner`, which
  would refresh the model-context catalog over the network.
- **KBR-220:** `tests/cli/test_bridge_state_location.py` runs the real `kitty bridge start`,
  `status`, `restart` and `stop`, and a real `kitty.bridge_runner` started as a service would start
  it. It has no choice: the defect lived *between* two processes, each of which resolved its path
  correctly by its own logic. It is the **one** module allowed to spawn `kitty.bridge_runner`, and it
  keeps KBR-176's reason for the rule. The model-context catalog cache is seeded fresh in an isolated
  cache directory, so no fetch runs inside `start`'s 5-second window and no user cache is written.
  Its isolation is worth copying: `WIN_PD_OVERRIDE_LOCAL_APPDATA` (platformdirs ≥ 4.8), because
  plain `LOCALAPPDATA` does not redirect Windows. A child reports where kitty will look, and the
  fixture refuses before writing anything unless that is the temporary directory. Its three cases
  take **~6.5 seconds** on Linux, measured. They passed unskipped on the Windows and macOS legs of
  the first PR run (2026-09-13, run 34766656090), but the gate prints no per-test durations, so those
  legs have a result and no figure yet. One cost to know about: a bridge that misses the 5 s window fails the case
  with *"did not report ready"* rather than slowing it, so a slow runner shows up as a red leg,
  never as a quiet delay.

**One cross-cutting cost, added by KBR-188's fix.** Every conformance probe now begins by waiting
for the clock to report a new instant (§8.3). Measured at **80 calls** across the harness suite:
immeasurable on Linux and macOS, where the clock resolves in nanoseconds and the first look
returns, and a worst case of **~1.2 s** on the Windows leg, whose step is ~15.6 ms. It scales with
the number of raw-socket probes, so a future recorder adds to it in proportion to the probes it
drives, not to its test count.

KBR-10 added the largest one: `tests/cli/test_stream_encoding.py` spawns **38 child interpreters**
per run, on each of the gate's six legs (four Linux interpreters, plus the Windows and macOS legs
§8.4 added). It has no choice — the behaviour it proves is that kitty survives a
hostile *interpreter start-up encoding*, and `PYTHONIOENCODING` is read before any in-process test
exists, so a real child is the only oracle. Each spawn is short (the whole file runs in ~13s,
measured on Linux), but
T-H1 should note that mutation testing over `l1` will re-pay that cost per mutant, and may want to
deselect this file from the mutation baseline rather than from the gate.

**KBR-204 added three of the 38**, for the same Windows family by another route: an interactive
command whose stdin reports as a terminal while its stdout is a pipe. The children get a
pseudo-terminal as stdin on POSIX and `NUL` on Windows, so all six legs build the real asymmetry.
One cost to know about: a child that gets past **both** guards — the prompts' and the menus' —
with that stdin **blocks** waiting for keys nothing will type, so a regression there shows up as
the runner's 60-second `TimeoutExpired`, not as a fast assertion.

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

**KBR-164 gave the registry its first rows**, and they were not the row this section
anticipated. TR-1c's header-subset assertion (KBR-8) still belongs to an
acceptance scenario that does not exist yet (§6.4.1, delivered by T-J2 downstream of T-J1), and a
registry row for an assertion no test contains documents a fiction — so it is still unwritten.
What arrived first instead were the **Windows cells** of assertions that the platform legs
(§8.4) found to be false on Windows and true everywhere else: several over KBR-188 and one over
KBR-189.

**The KBR-188 rows are gone again, and why they could not have worked is the lesson.** KBR-188
was one defect — Windows' `time.monotonic` advances in ~15.6 ms steps, so two probes sent back to
back are stamped at the same instant and `check_arrival_increases` is false. Its four rows
exempted the assertions that observed it. But **whether a given pair collides is a race**, not a
platform property: a pair that straddles a step boundary is stamped at two instants and the
assertion *passes*. An exemption fails on an unexpected pass — deliberately, so debt cannot
outlive its defect — so the leg went red in **both** directions on alternate runs. Measured on
2026-09-12: one run failed `check_arrival_increases`, the next failed the exemption for passing.

The rule that follows, and it generalises past this defect: **an exemption is only sound over an
assertion that is deterministically false on the exempt platform.** Over a racing one it converts
a flaky assertion into a job that is red either way. A nondeterministic platform difference has to
be removed, not exempted.

Removed here by `recorder_conformance._advance_clock()`, which stands a probe apart from whatever
arrived before it by waiting — on a **condition**, that the clock has reported a new instant,
never on a fixed sleep. It lives in the shared driver, so §7.2's four recorders inherit it rather
than each carrying a row, and it costs nothing measurable where the clock is fine: 80 calls across
the harness suite, a worst case of ~1.2 s on Windows and immeasurable on Linux and macOS. Its own
bound is a **poll count, not a deadline**, because a deadline is computed from a clock and the case
it exists for is a clock that has stopped — the frozen-clock defect `check_arrival_increases` is
there to catch, and the first draft of the function hung the suite on it. The case that pins that
bound counts polls for the same reason: an earlier version asserted elapsed *wall time* and went
red on the macOS leg, where `time.sleep(0.001)` takes about eleven milliseconds. Sleep accuracy is
the platform's business; the count is the only part the driver promises.

**The wait goes before the probe, not after it, and the first fix got that wrong.** Waiting after
the reply separates a probe from the driver's own previous probe and from nothing else — so a
probe following a request the driver did **not** send still shares that request's instant. §6.3's
slow-body pair is exactly that shape: it hand-rolls its first request on a raw socket so that
arrival order and completion order differ, and the driven probe behind it collided anyway. The
Windows leg stayed red. Waiting at the *start* of `send_raw` is the general form — every probe is
separated from everything before it, however the earlier request was produced.

**A clock quantised from real time could not have caught that**, which is the sharper lesson. It
reproduces the platform but keeps the race: whether a given pair collides still depends on how
fast the runner is, so a case built on it passes against the defect most of the time. Measured:
the slow-body pair collided in four runs out of eight at a 50 ms quantum. `test_recorder.py`
therefore pins the *placement* against a clock that moves **only when something sleeps on it** —
no real time passes and every run agrees. That clock is sound only away from the event loop, which
reads `time.monotonic` for its own timers, so the end-to-end cases keep the quantised clock and
the placement case, which never opens a loop, uses the stepped one.

KBR-189's row was over a different defect — a mid-stream abort that lost the body on Windows —
and it was sound where KBR-188's were not, because that loss was deterministic there. It is gone
too, withdrawn together with its fix: KBR-189 changed `Reply.abort()` (§7.2), and the Windows leg
of the PR that removed the row is the evidence the assertion now holds there. **The registry is
empty again.**

All of those rows were the parametrised-cell shape above rather than whole-test exemptions, and the reason
is the rule this section opens with — the reason the next platform row should take that shape too.
A `skipif` would have been the obvious move and is the wrong one: §8 permits a platform skip for
behaviour that **does not exist** on a platform, and these assertions were not inapplicable on
Windows — they were **false** there, which is a defect with a ticket. Exempting the cell kept the
assertion gating on the four Linux legs and on macOS, kept the count of outstanding Windows defects
readable in one file, and failed the job the day Windows started passing. Skipping would have
bought a green leg by not looking.

That makes the registry-shape check itself vulnerable to §8's own "green because it stopped
looking": a validator run over zero rows passes perfectly. So `registry_violations` is proved
against a **fabricated malformed registry** rather than against the production one, and the
production registry is asserted clean as a separate, weaker claim.

### 8.4 The platform matrix

Delivered by [KBR-164](https://shelpuk.atlassian.net/browse/KBR-164). Until it landed, every
job in the repository ran on Linux, so **every Windows-only and macOS-only defect in the product
was reachable only by a user reporting it** — which is how all three platform bugs on epic
KBR-123 (KBR-1, KBR-4, KBR-10) were in fact found.

The Fast gate therefore runs on three platforms:

| Leg | Runner label | Python | Selection |
|---|---|---|---|
| Linux | `ubuntu-latest` | 3.10, 3.11, 3.12, 3.13 | the whole Fast gate |
| Windows | `windows-latest` | 3.12 | **identical** |
| macOS | `macos-latest` | 3.12 | **identical** |

**One job with an `os` matrix dimension, not a second job.** A separate platform job would
duplicate the five-step list, and the day the two copies differ the platform leg stops being
evidence about the gate and becomes evidence about a *similar* gate. This is the same argument
`tests.yml`'s header already makes for the release path — *"there is no second, weaker
definition to drift out of sync"* — applied across platforms instead of across events. It also
means the legs gate a pull request through `ci-required`'s existing `needs: [test, …]` with **no
change to `ci.yml`**, and gate a release through `publish.yml`, for free.

**The same tests, not a platform-dependent subset.** The ticket floated scoping the leg "to the
tests whose behaviour is actually platform-dependent". Rejected: that set is precisely what
nobody knows — a latent POSIX assumption is invisible until the test runs somewhere else — and
naming it would need a second marker axis, which collides with §8.1's exactly-one-layer-marker
rule. The whole `l1 or l2` expression runs on every leg.

**GitHub-hosted, and this is not a new decision.** `.github/review/rules/ci.md` § "Runner and
caps" already fixed it for every job in the repository: a self-hosted runner group carries an
*"Allow public repositories"* setting that is **off by default**, and this repository is public,
so a `[self-hosted, …]` label reaches no group at all and the job **queues for ever — no error,
no annotation, no timeout**. The platform legs inherit that unchanged.

**The cost objection in the ticket does not apply here, and the reason is worth recording
because it is the whole reason this was cheap.** KBR-164 was written expecting a large bill
("`windows-latest` minutes bill at 2×"). That multiplier is a **private**-repository rule.
`kitty-bridge` is public, and GitHub's runner reference states the case in one sentence: *"Use of
the standard GitHub-hosted runners is free and unlimited on public repositories."* `windows-latest`
and `macos-latest` are both in that table. **Confirmed 2026-09-12.** If this repository is ever
made private, this subsection is the one to revisit first — the legs keep working and start
billing at 2× and 10× respectively.

**One Python version per platform, and it is 3.12.** The suite is mostly platform-independent, so
a four-version Windows matrix would quadruple wall-clock and quadruple the first-run triage
surface to re-prove interpreter-version facts the Linux legs already prove. 3.12 rather than the
newest because `ci.yml`'s two review-system jobs already pin 3.12, so the repository names one
version in one place; and because a Windows-only defect is likelier to reach a user on a
mainstream version than on the newest. Interpreter-version questions stay the Linux matrix's job.

**`include:` entries, not an `os` × `python-version` product with `exclude:`.** The product form
needs six `exclude:` entries to remove six of twelve combinations, and `_matrix_values` in
`.github/review/tests/test_review_scripts.py` deliberately does **not** honour `exclude:` — it
over-approximates on purpose, which is the safe direction for a ceiling check but the wrong one
for a matrix that would then be mostly holes. ⚠️ The `include:` form carries its own subtlety and
it is where this construct is misread: an include object whose keys would **overwrite** a base
matrix value is not merged into the existing combinations — it becomes a **new** combination.
That is what produces the two platform legs, and a comment in `tests.yml` says so beside them.

**One integer `timeout-minutes` for the whole matrix job.** `DeclaredJobCapIsEnforceableTests`
parses the cap with `(\d+)`, so a `${{ matrix.… }}` expression there reads as **absent** and the
guard reports "declares no job-level `timeout-minutes:`" about a line that is plainly present. A
matrix job is held to the **lowest** platform ceiling among the labels its matrix can produce;
all three labels here are ordinary GitHub-hosted 4-CPU runners at **360 minutes**, so the cap is
bounded by measurement rather than by the platform.

**`openssl` is an environment prerequisite on all three images, and §8 already said so.** The
paragraph above on the resource-availability rule states the duty in advance of this change: *"a
non-Ubuntu matrix entry has to keep it, and the fix for a red gate is to install `openssl`, never
to reinstate the skip."* `tests/bridge/tls_certs.py` calls `pytest.fail` — not `skip` — when the
binary is absent, so a missing `openssl` is a red gating leg with no sanctioned recovery.
Discharged, and recorded rather than assumed: **confirmed 2026-09-12** against the
`actions/runner-images` image manifests — Windows Server 2025 ships **OpenSSL 3.6.4**, macOS 15
arm64 ships **OpenSSL 1.1.1w**, Ubuntu 24.04 ships **3.0.13**. A future image bump inherits that
duty. ⚠️ If the Windows TLS tests go red, check `-subj "/CN=localhost"` first: a leading-slash
argument is mangled by MSYS2 path conversion if the resolved `openssl.exe` is the Git-for-Windows
build rather than the native one.

**`fail-fast: false` is load-bearing now, and was merely tidy before.** With six legs it is the
only reason a Windows failure does not cancel the four Linux legs mid-run. Cancelling them would
destroy the evidence needed to tell "Windows is broken" from "this change is broken" — which is
the first question asked of every red platform leg. It predates this subsection; its importance
does not.

**A red platform leg blocks a release, deliberately.** `publish.yml` calls this same reusable
workflow, so macOS and Windows have just joined the release gate — which is not free, and the
cost is named rather than discovered later. `rules/ci.md` already notes that the release path
carries blast radius beyond itself; after this change a red or flaky platform leg stops a PyPI
release of a product whose platform behaviour was, until now, never exercised at all. That is the
correct trade — shipping a release known to be broken on Windows is the worse outcome — and the
break-glass is the same admin path `review/README.md` documents.

**The measurement, because the number beside it used to be fiction.** The comment that stood
beside `timeout-minutes: 30` claimed *"the suite runs in a couple of minutes"* — stale by an
order of magnitude, leaving the backstop only ~50% headroom on a leg nobody had timed. Measured
on the first six-leg run (2026-09-12, run 34689734864):

| Leg | Wall clock | Outcome |
|---|---|---|
| Linux × 4 | 19.5 – 19.7 min | passed |
| **macOS** | **20.6 min** | **passed, whole suite, first run** |
| Windows | 2.3 min | **not a measurement** — aborted early on KBR-180 |

The cap is **60**, which clears the slowest *completed* leg (macOS, 20.6) by ~3×. ⚠️ Windows is
**not yet timed**: its first run aborted 227 tests in, so the figure above measures a failure, not
the suite. Whoever next reads a green Windows leg should set this number from it. The cost of 60,
stated rather than hidden: a **hung Linux leg is noticed 30 minutes later than it was**. On a free
runner that is cheap, and a cap too low is worse — it kills a healthy leg and reads as a product
failure. **A platform leg killed at the cap is a cap problem until proven otherwise**, never
triaged as a hang; the platform ceiling is 360, so there is room to raise it.

**What the legs found on their first run, recorded because it is the argument for the whole
subsection.** macOS passed the entire suite immediately. Windows did not, and the failure was not
a latent POSIX assumption in a test — it was a **user-facing product defect**
([KBR-180](https://shelpuk.atlassian.net/browse/KBR-180)): `probe_pid` in `bridge/manage.py`
probes liveness with `os.kill(pid, 0)`, and `signal.CTRL_C_EVENT` **is** `0` on Windows, so that
call broadcasts a Ctrl+C to every process sharing the console instead of probing. It is reached by
`kitty bridge status`, `stop`, `start` and `restart`, so each of those interrupted the user's own
shell. Its docstring asserted the opposite — *"on Windows as well as POSIX — it does not terminate
the target"* — and the `except OSError` branch beneath it explained a Windows code path that call
never reaches. Both were written by reasoning about Windows rather than running there. **That is
the failure mode a platform leg exists to end**, and it was caught within two minutes of the leg
first existing.

**Skips are named, not counted in silence.** §8's rule — a gating job that goes green because it
ran nothing is the most expensive false confidence — is what a new platform leg is most likely to
breach, because a **platform** skip is the one kind §8 permits. So the gate's pytest invocation
carries **`-rsfE`**: every skipped test is listed in the log *with its reason*, on all six legs.
The flag is on the one shared invocation rather than on the platform legs alone, because a second
invocation in the file is exactly the second definition this subsection's first decision rejects.

🔴 **`fE` is not decoration, and the obvious spelling is a trap this section fell into before it
was corrected.** `-r` **stores** reportchars; it does not append to them. A bare `-rs` therefore
*replaces* pytest's default `fE` and **deletes the `FAILED …` summary** from the end of the run —
measured on pytest 9.1.1 with this exact command. On a suite of this size that summary is how a
red leg is read, and the first red platform leg is precisely when it is needed. The skip letter
must be **added** to the defaults, never substituted for them. An earlier draft of this paragraph
claimed the visibility "costs one flag"; it costs one flag *only* when the defaults are restated
alongside it. `tests/test_github_actions.py::…::test_the_gate_keeps_its_failure_summary_while_reporting_skips`
is the check, because a claim about a flag that nothing verifies is how the first spelling
survived review.

⚠️ **A platform skip is not a licence to skip a platform's defects.** §8 permits a skip for
behaviour that *does not exist* on a platform — no `SIGKILL`, no POSIX path semantics — and the
distinction matters most here, where a leg is new and red. A test that fails because a tool is
missing or behaves differently is a **resource-availability** skip in platform clothing: the
shape §8 forbids and KBR-132 closed. The fix for that is to provision the runner or make the test
platform-agnostic, never `skipif`. And a test that fails because the **product is broken on that
platform** is neither: it is a defect, it gets a ticket, and the leg stays red until the defect is
fixed. `tests/bridge/tls_certs.py` is the template for stating the difference in code.

**A consequence that is a product win, not a side effect.** Two things in the repository have
never executed even once:

- `tests/test_launcher_discovery.py`'s two `skipif(sys.platform != "win32")` cases — written for
  a leg that did not exist, skipped in every run that has ever happened;
- **every `if sys.platform == "win32"` branch in `src/kitty`, as far as `mypy` is concerned.**
  mypy resolves `sys.platform` against the platform it runs on, so the Windows bodies were
  invisible to the only type check we run — and §8's table credits mypy with four user-visible
  defects the suite could not find, one of them explicitly *"on a non-Linux OS"*.

Running `mypy src/kitty` on the Windows leg is therefore not redundant with the Linux legs; it is
a **new detector over code no check has ever read**. That is also why the platform legs run the
whole step list rather than pytest alone.

### 8.5 The review classifier's tier order

The gate is not the only thing that decides whether a pull request is reviewed.
`.github/review/scripts/interpret_claude_result.py` reads the wreckage of a failed review
attempt and answers a question no test in `tests/` asks: **who fixes this** — top up a
balance, or edit a workflow. It also decides whether the one automatic retry is spent. Its
entire rationale has lived in module comments, which is why three tickets (KBR-145, KBR-172,
KBR-166) each re-derived the same reasoning from scratch. The invariants are recorded here so
the next change to it has something to contradict.

**I-C1 — Tier order is a cost ordering, not a specificity ordering.** Tier 1 is consulted
first — since KBR-206, the context-management refusal and then `FATAL_PATTERNS`, and ahead of it
the unattributable-record check (I-C5, D3) — then quota, then credentials, then the generic-code
tier, then transients.
"Generic loses to everything more specific" reads well and is wrong: `EXHAUSTED_PATTERNS`
carries the bare words `timeout` and `capacity`, which a model can write in its own prose, so
yielding to them would let a billed rejection be retried at full price. The order is justified
by what each misclassification costs, and each entry's position is measured rather than
argued.

**I-C2 — A pattern weak enough to appear in ordinary prose is scoped by AUTHORSHIP, never by
a tighter regex — with one measured exception, below.** KBR-172 and KBR-166 each measured anchoring — on vendor vocabulary, on the
CLI's line shape, on proximity to a spending verb — and every version lost a real provider
body while still leaking model prose. The separable question is not what the text says but who
wrote it, and the execution record already answers it: `_provider_outcome_text` is the
narrower haystack, and the weak patterns read only that.

KBR-181 moved four more there — the Z.ai business codes `1308`, `1310`, `1113` and the phrase
`limit will reset` — and **anchored a fifth instead, which is the exception.** The exception is
a **verbatim prefix of a vendor's own error sentence, long enough that prose matching it is a
quotation of that vendor**: `you exceeded your current quota`. It stays in the full haystack
because OpenAI's spent quota arrives in production as CLI text in `result`, where no
provider-scoped pattern can see it, so the bare `exceeded your current` was that body's only
quota signal. Measured across the status matrix, moving or deleting the phrase kept every
status and retry but lost the quota reason and top-up advice on 10 of 11 statuses (the 11th is
402, which now names the quota itself); the anchor lost nothing. Its residual is that a
quotation still leaks, and that Gemini sends the same sentence for a per-minute rate limit, so a
free-tier throttle is advised to top up — status and retry still right. The rule is about the
tiers that can **promote** a record to `exhausted`. KBR-206 added a second exception on the same
terms — `your credit balance is too low`, the shared prefix of Anthropic's two spent-balance sentences
(*"…to access the Anthropic API"* and *"…to access the Claude API"*, both in public reports); the longer
prefix would lose one of them. It is needed for the same reason: the CLI's error line in `result` is
the carrier no provider-scoped pattern reads. `FATAL_PATTERNS` no longer carries a weak entry — see
I-C5. `CREDENTIAL_PATTERNS`'
`model not found` / `authentication_failed` words are the same exposure and are KBR-217's.

The four moved patterns kept every English body: Z.ai's own error table puts a stronger vendor
phrase beside every code and every `limit will reset`. They moved rather than being deleted
because a code survives translation and a phrase does not.

**I-C3 — A numeric outcome field is admitted to the provider-scoped haystack only.** KBR-182.
`api_error_status` is a JSON number and was discarded before any pattern saw it, so the field
whose purpose is to report the provider's status was dead weight while looking live. It is now
read — but it must never reach the haystack `_outcome_text` returns. When KBR-182 set this rule that
haystack was read first by tier 1, which carried `\b400\b`, and the victim was Anthropic's spent
balance, which is an HTTP 400. KBR-206 moved `\b400\b` to `FATAL_UNLESS_PROVIDER_NAMED_PATTERNS`,
below the quota group, and the rule still holds for a different harm: that tier is consulted
**before** `EXHAUSTED_PATTERNS` and `STRUCTURED_OUTPUT_PATTERNS`, so a status in the full haystack
would turn a transient server error or a structured-output give-up beside a 400 into `fatal` with
the re-run refused. `StatusMatrixTests.test_no_status_turns_a_record_into_a_workflow_fault` and
`test_a_bare_400_is_not_promoted_to_a_workflow_fault` are the rows. The bound is at the field's own
value: a number nested inside a provider's error object is a parameter, not a status.

⚠️ **One consequence, accepted by name.** A 400 that names no cause reaches different verdicts by
carrier: in the CLI's text (`API Error: 400`, bare or in `result`) the generic tier reads it and the
record is `fatal`; as `api_error_status` alone it is not read there, and the record falls through to
`exhausted` "no recognisable error" with a retry. Production usually writes both, so the text decides.

⚠️ Two qualifications, because the rule is easy to state more absolutely than it holds. It
governs **parseable** records: when `_parse_events` fails, `classify` searches the raw text
whole and always has, status text included — except that since KBR-206 (D3, I-C5) such a
record carrying a `result` key is decided before every tier. And `_provider_outcome_text` has a **second
consumer** — `_write_diagnostic`'s quota branch — so a numeric pattern added to
`QUOTA_WORD_PATTERNS` would fire the top-up paragraph off a bare status, including under a
`fatal` verdict. That is the door KBR-207 has to walk through carefully.

KBR-181 walked through it (KBR-207 folded into that ticket). `\b402\b` and the three Z.ai codes
joined `QUOTA_WORD_PATTERNS` — one provider-scoped tuple, so no consumer can read the words and
miss the codes — and `\b40[134]\b` replaced `\b40[13]\b`. **The 402 is read in the quota
group, above `CREDENTIAL_PATTERNS`, because of I-C4 rather than specificity.** A 402 beside a
named `authentication_error` is incoherent and either verdict is arguable, but the diagnostic
keys on evidence, so a 402 placed below the credential names would print a credential verdict
over a top-up paragraph. In the quota group the two agree.

Measured over the full cross product for the two disagreement kinds this change can create —
`fatal` over a top-up paragraph, and a credential verdict over one — exactly one new cell
appears, and it is **reachable**: a 402 whose body carries the number 400. OpenRouter's own
spent-credit message is *"You requested up to N tokens, but can only afford M"*, and at M=400
tier 1's `\b400\b` calls it a workflow fault while the diagnostic, reading the 402, advises a
top-up. That is KBR-206's family — tier 1 reading text that is not a workflow fault — and the
advice is the correct half, so it is pinned rather than hidden: gating the diagnostic on the
verdict would suppress correct advice there and in KBR-206's own row. A pre-existing
disagreement of a third kind — a quota verdict under the context-management paragraph — is not
touched by this change. **KBR-206 closed the 400-carried instance of the first kind, including
this 402 cell, and the whole of the third**; four cells of the first kind remain, and the second kind
was never measured in its sweep (I-C5).

**Residuals, stated so they read as decisions.** A non-English Z.ai body in `result` keeps
`exhausted` and its retry but loses the quota reason and advice. A provider that writes `0.402`
or `1308` in its own message is read as quota, as KBR-166 records for `quota`. The truncated-record
fallback `_provider_outcome_text` already documents — no `result` key, so model-authored
`message`/`content` is handed back — reaches 402 and 404 exactly as it reached 401 and 403.

**A decode failure degrades; it does not crash.** KBR-181 also widened the four `json.loads`
sites to `ValueError` and `RecursionError`: an integer over the interpreter's digit limit and a
deeply nested document raised through the script, so the run that already failed wrote no
status and no diagnostic. The cost is recorded: such a record takes the unparseable path, where
`classify` searches the text whole — a crash traded for an unscoped verdict — unless it carries a
`result` key, which since KBR-206 makes it unattributable (I-C5, D3).

**I-C4 — The verdict, the `retryable` flag and the diagnostic's advice must agree.** They are
computed by three different functions from three different inputs — pattern order, cost, and
the evidence text — so they can disagree without any one of them being obviously wrong.
KBR-145 was filed because they did: an operator was told to top up a balance and, one
paragraph up, that the workflow was broken. Any change to the tier order re-checks all three.

**I-C5 — Tier 1 names a workflow fault; it never carries a bare status or a generic code.** KBR-206.
Tier 1 is consulted before any provider-named cause, so whatever it matches decides the verdict
*and* refuses the retry. `\b400\b` sat there, and a 400 says the request was rejected, not **by what**:
Anthropic bills a spent balance as a 400 (*"Your credit balance is too low…"*), and the CLI writes the
status into its error line, so tier 1 called an empty account a broken workflow while the diagnostic,
reading evidence, advised a top-up. KBR-145 moved `invalid[_ ]request` down for the identical reason;
KBR-206 moves `\b400\b` beside it, into `FATAL_UNLESS_PROVIDER_NAMED_PATTERNS`, and adds Anthropic's
sentence prefix to `QUOTA_PATTERNS` (I-C2) so the `result` carrier is read.

*Why not simply demote it.* The context-management refusal is also a 400, and the wording this
repository has carried since upstream adds *"No quota was consumed"* — a sentence KBR-206 found in no
public report. The verbatim OpenRouter body (cc-switch#1929) carries no quota word, but its
`"code":400` is nested and never read, so in the error-object carrier nothing except its wording can
call it fatal. So the two 400s are separated by their **bodies**, not by tier order: `classify` reads
`CONTEXT_MANAGEMENT_REFUSAL` — the constant `_write_diagnostic`'s refusal branch already reads — first.
It is read **ahead of the schema patterns** because the diagnostic has a refusal branch and no schema
branch, so a record carrying both is told one thing. Its reason is a **fixed string**, not the match:
this pattern now decides the verdict, and echoing up to 80 characters of provider- or model-written
text into a `reason=` line of `$GITHUB_OUTPUT` is a surface the other reasons never had. ⚠️ "One
pattern" is not "one matcher": `classify` searches lowercased text and the diagnostic uses `re.I`, and
the two can differ on a character whose lowercase is longer (`İ`) inside the phrase's 80-character
window. The coverage of `CONTEXT_MANAGEMENT_REFUSAL` is what decides the verdict: a future refusal
wording matching neither alternative falls to the tiers below.

*The leak the move uncovered (D3).* An unparseable record is searched whole, tool results included.
A `400` anywhere in it used to reach tier 1 first; with the status demoted, every full-haystack tier
read what the reviewer read — `src/kitty/bridge/server.py` names `authentication_error`, the harness
quotes the refusal's dated slug, and prose says `timeout` — and each granted a paid retry or gave advice
drawn from it. The leak already existed for text without a `400`. **Scoping tiers one at a time was
built first and failed review:** it removed the quota and credential votes, and the record fell to
`EXHAUSTED_PATTERNS` or the fallthrough and was retried anyway, while a real spent balance in that
shape lost its top-up advice. So `_record_is_unattributable` — not JSON per `_parse_events`, and
carrying a `result` key — decides such a record before every tier: `fatal`, a fixed reason, no
automatic retry, and a diagnostic paragraph that gives no advice drawn from unattributable text.

*Measured*, `origin/main` @ `b902076` against the change: the 400-carrying module-level fixtures in
`test_review_scripts.py`, the inline KBR-206 rows, `QUOTA_FIXTURES`, both schema rejections, four
public-report bodies (the *"Claude API"* wording, claude-code#4283's malformed-request 400, a bare
`HTTP 400 Bad Request`, OpenRouter's verbatim refusal) and the status-matrix bodies, × three carriers
(bare CLI line, `result`, `error`) × every matrix status — 595 cells, 181 changed (reason-only changes
included). Verdict/advice disagreements of the two kinds the sweep checks (top-up advice against a
non-quota verdict; refusal advice against a non-`fatal` one) fell from **72 to 4**; the four are
pre-existing (a schema rejection beside a `402` status) and are the named I-C4 exclusion. No quota
fixture and no schema rejection changed verdict. The figure is a measurement, not a pinned test;
`FourHundredCarrierTests` pins the verdicts it rests on, and fails when a new 400-carrying
**module-level** fixture joins the file without a row.

*Owner decisions (2026-09-13)*, recorded because each flips behaviour that was pinned or relied on:

* **D1 — a 400 whose only cause evidence is a billing word is a spent balance.** Anthropic's real body
  has exactly that shape (`Plans & Billing`), so no rule separates it from a synthetic *"the billing
  account is not permitted to use this model"* except tier order, which this invariant forbids. The
  cost is at most one re-run of a request that was rejected before the model ran.
* **D2 — a record naming only the context-management slug is `fatal`.** Its advice already said
  "re-running unchanged will not help"; the verdict now agrees. The cost is that a reviewer quoting the
  dated slug in a non-schema-failure `result` loses its retry.
* **D3 — a transcript too broken to attribute is `fatal`, not retried, full stop.** Closed in this
  ticket rather than filed, because the move widened it. Chosen over deferring it to a ticket after
  the narrower version was measured failing. The cost, accepted by name: a spent balance or a
  transient outage in that shape is a workflow-level `fatal` and not retried — the PR notice, the job
  summary and the `::error::` line say the workflow needs fixing, and only the embedded diagnostic
  says the record was unreadable — and a spent balance whose top-up advice worked on `main` in that
  shape now gets none.

*Residuals.* **D3 is a text sentinel, and it does not reach three broken shapes.** (1) A transcript
cut off before its result event carries no `result` key, is indistinguishable from raw CLI output,
and is still searched whole — on `main` a `400` beside a quota phrase there was `fatal`, and it is now
a paid retry with top-up advice (`test_a_transcript_cut_before_its_result_event_still_reads_what_was_read`).
(2) A transcript whose result event is an error subtype — the Agent SDK's `SDKResultError` carries
`errors` and no `result` — is the same case; with the dated refusal slug in a tool result it moved
from `exhausted` to `fatal` with no retry (`test_an_error_subtype_transcript_is_not_unattributable`).
Widening the sentinel to `"type": "result"` would refuse the truncated-401 record
`test_a_result_value_is_not_a_result_key` keeps readable, so it is not done here. (3) `_parse_events`
treats a record as readable if any line decodes, so an NDJSON record whose result line is corrupt is
not unattributable and its lost result is never seen (`test_a_corrupt_ndjson_result_line_is_not_unattributable`).
Conversely a raw provider body passed through unescaped that carries its own `"result":` key — e.g.
Cloudflare's `{"result":null,"success":false,...}` wrapper — IS called unattributable; kitty's
translated error escapes it, so the usual route is unaffected. OpenRouter's *"can only afford 400"* message stays `fatal` with no advice unless a 402 is
in `api_error_status` or in the CLI's text outside `result`: a nested `"code":402` is never read, and
`API Error: 402 …` inside `result` is read only by the anchored tiers, which carry no phrase of that
message. A 401/402/403/404 status beside a malformed-request 400 body resolves to what the status
names — credentials, or quota with the top-up paragraph — with `retryable` **true**, the incoherent-pair
resolution `STATUS_MATRIX_MOVED` records for the generic code; that body comes from a run 27 turns
deep, so the retry is priced, and `FourHundredCarrierTests` pins it.

*Closed by KBR-217 (2026-09-14).* D3's three broken shapes are closed (KBR-217,
`fix/kbr-217-credential-prose-retry`): shape (1) — `<truncated` marker with no
result-event marker — and shape (3) — a dropped corrupt NDJSON result line — are now
decided before every tier as unattributable-fatal. Shape (2) needs no change: the
salvage the ticket sketched would BREAK the pinned refusal verdict (the parseable
variant loses the refusal because `_outcome_text` excludes `tool_result` content), so
the whole-record search path remains. The truncated-401 control
`test_a_result_value_is_not_a_result_key` stays unchanged. The Cloudflare wrapper
sentinel is tightened to require a string-opener (`r'"result"\s*:\s*"'`), excluding
`"result":null`. OpenRouter's *"but can only afford"* phrase is now read by the
provider-scoped `QUOTA_WORD_PATTERNS`; the `API Error: 402 … in result` carrier stays
`fatal`-with-no-advice and is the named residual (Z.ai's reset-time-in-result carrier
is the same boundary, closed by the reset-time capture via `RESET_PATTERN`).

**How it is proven.** `.github/review/tests/test_review_scripts.py`, run directly by `ci.yml`
rather than through `pytest`, so it is outside §8.1's marker matrix and carries no layer
marker. That is deliberate: the suite must stay runnable with a bare interpreter and no
installed dependencies, because a broken review workflow has to be diagnosable before an
environment is provisioned. Its content is L1 in kind. Behaviour-changing edits to the tier
order carry a before/after table over the full cross product of statuses and body fixtures —
incoherent pairs included, since a gateway's status need not match a passed-through upstream
body — and `StatusMatrixTests` is the pattern to copy.

### 8.6 What the CI environment already supplies

Recorded because this document spent its first draft reasoning about these resources as things
CI might one day be given. It has had all of them since the automated reviewer landed.
`.github/workflows/claude-code-review.yml` runs a real Claude Code through a released Kitty
Bridge on every same-repository pull request, which means the runner is already provisioned
with a complete kitty installation — and that is the single most consequential fact about what
this suite can afford to test.

| Capability | Artifact | Binding |
|---|---|---|
| Claude Code CLI, version-pinned | `.github/workflows/claude-code-review.yml` | `bash -s -- 2.1.238` |
| Kitty profiles — a balancing pool by default | `.github/workflows/claude-code-review.yml` | `vars.KITTY_PROFILES_JSON` |
| Kitty credentials | `.github/workflows/claude-code-review.yml` | `secrets.KITTY_CREDENTIALS_JSON` |
| Kitty egress gateway | `.github/workflows/claude-code-review.yml` | `secrets.KITTY_EGRESS_JSON` |
| Kitty bridge debug log | `.github/review/scripts/configure_kitty.py` | `kitty-bridge-debug.log` |
| Kitty launch stderr — a disjoint window | `.github/review/scripts/configure_kitty.py` | `kitty-bridge-stderr.log` |

`configure_kitty.py` materialises the three kitty documents at the paths kitty itself reads,
and generates the launcher that puts `kitty` in front of `claude`.

**A second consumer binds the same three settings:** `.github/workflows/tmux-disconnect.yml`
runs the pull request's own kitty against them for the live `--tmux` test (`SYSTEM_DESIGN.md`
§3.4), with the same Claude Code pin. The rows above keep naming the review workflow because a
duplicate row per consumer could be deleted with every arm green. Every job binding a `KITTY_*`
capability, in any workflow, must refuse a fork's pull request; `kitty_job_fork_discrepancies`
holds that.

**The profile is a balancing pool, and that changes what an assertion may say.** The workflow
derives the consequence three times over: *"the profile is a four-member balancing pool"*, so
*"no single name here could be true of the run"*. A `agent_live` or `eval` assertion may
therefore not name a model — it may assert membership, or a property every member has, and
nothing else. This is the half of the capability that is easiest to plan around wrongly.

**Both logs exist, and neither may leave the runner.** They cover disjoint windows:
`--debug-file` records everything after the bridge is up, and the stderr redirect catches the
launch failures kitty prints nowhere else — an egress refusal, a credential that will not
resolve — which reach `claude-code-action` as an empty execution record. Between them they are
the only evidence a failed run leaves. Two properties a test job must inherit rather than
rediscover:

- 🔴 **The debug log carries kitty's inbound request bodies — the bridge token and the entire
  prompt. It must never be uploaded as an artifact.** The obvious way to use a debug log in a
  smoke or e2e job is `actions/upload-artifact`, and that is the one thing it may not do. The
  review workflow keeps the log on the runner and uploads a filtered timeline instead.
- The logs are **per-runner, not per-run** — *on a persistent runner*. The workflow purges stale
  copies before launching, so at most one failed run's evidence is ever on a machine. The review
  job runs on `ubuntu-latest`, which is destroyed after the job, so the purge is a no-op there and
  is kept for the half of the invariant that survives a move back to a self-hosted fleet. A job
  that expects to find its own log by name, unqualified, is reading someone else's the moment the
  runner persists.

**Two limits, stated because a job planned without them will be wrong.**

**The CLI pin is a manual pairing, not a mechanism.** `anthropics/claude-code-action` is
referenced at a floating `@v1` tag and the literal above is kept equal to the version that
action pins internally, by hand, as one change when the action bumps. Nothing fails when they
diverge. `tests/test_ci_capability_inventory.py` holds this table to the workflow, which catches
the table going stale but cannot catch the action moving underneath both. Two more facts that
will surprise whoever writes the startup smoke: the wrapper passes `--no-validate`, because in
CI the run itself is the validation; and `~/.local/bin/claude` is deliberately **not** on `PATH`
— kitty's own fallback chain is the rung that finds it, so a job that assumes a plain `PATH`
lookup will not find a CLI that is present.

**A fork pull request does not get the secrets, and the profile variable is unmeasured.** Stated
split because only one half is established. GitHub documents that *"with the exception of
`GITHUB_TOKEN`, secrets are not passed to the runner when a workflow is triggered from a forked
repository"*, which covers the credential and egress stores. It documents **nothing** about
configuration variables, so whether `vars.KITTY_PROFILES_JSON` reaches a fork run is **unknown
here and was not measured**; no gating decision may assume either answer. Independently of both,
this workflow's job does not start on a fork at all, guarded by
`github.event.pull_request.head.repo.full_name == github.repository` — and the documented wrong
way to "fix" a skipped fork run is `pull_request_target`, which runs the fork's pull request in
the base branch's context *with* secrets. A new workflow inherits none of that guard and has to
write its own.

That asymmetry divides the L4 jobs, and the division is why the §8 cadence table reads as it
does:

| Job | Needs | Consequence |
|---|---|---|
| `agent_smoke` (hermetic, §6.4.2) | the pinned CLI only — no credentials, no variables, no network, no provider | The installer is a public download, so the fork question does not arise. A per-PR gate stays viable. |
| `agent_live`, `eval` | the kitty credential store, and a provider that bills | Withheld from a fork run by GitHub's own rule. Nightly on `schedule`, where the fork case does not arise — which is where §8 already puts them, now for a stated reason rather than by cadence preference. |

**The pinned CLI is an environment prerequisite of the Acceptance job, not something a test may
probe for** — §8's `openssl` reasoning applies unchanged, and this is the shape that makes a
per-PR `agent_smoke` gate compatible with "skips are failures in a gating job". The CLI arrives
by a live download from a third party on every run, so the *install step* owns the retry ladder
and the job must fail when it cannot install. That is **prescriptive for a future `agent_smoke`
job, not a description of the review workflow**, whose install step carries `continue-on-error:
true` on purpose: there an install failure composes into the review path and is classified, which
is the right answer for a job whose output is a review and the wrong one for a gate. One non-obvious reason that ladder is shaped as it is,
worth carrying into any job that copies it: without `set -o pipefail` inside the `bash -c`, a
curl 429 or 403 feeds `bash -s` empty stdin, which exits 0 — the retry loop then breaks on the
first attempt with **no CLI installed** and the transient download error reaches the classifier
as a fatal.

**What this unlocks that was not previously plannable.** The balancing pool means the nightly
`agent_live` job can exercise member selection and failover against real providers, which §4.3
C3's cross-attempt row otherwise proves only against recorded upstreams. The egress secret means
the same job can assert I3 containment end to end, through a real gateway, rather than only
through the local CONNECT proxy of §5.2. Neither is specified here; both are named so that T-K11
and T-I14 inherit them.

**Five directions, because fewer would not have caught this.**
`tests/test_ci_capability_inventory.py` checks forward, so the document cannot promise a
capability CI does not have; reverse, so a capability cannot be added to CI and left out of the
design; the version pin, both ways, because that pairing is maintained by hand; the logs the
generated launcher writes, because the two log rows bind neither a secret nor a version and a
review round found that both could be deleted from the table with every other arm green; and the
fork guard itself, whose removal changes no binding and would leave the paragraph above false.
Every row is covered by at least one *reverse* direction — that is the property, not the count.
A forward-only check would have gone green through the entire period this section describes —
the table would simply have been absent, which is what it was.

**One case is read rather than scanned, and the reason generalises.** The launcher's two log rows
are checked against the text `configure_kitty.py` *generates*, not against its source: the module
discusses `--debug-file` and both log names at length in its own docstring and comments, so a
substring scan over the file is satisfied by the prose after the launcher stops emitting either
flag — measured, in the same review round. It is the trap the fork arm was already built to avoid
by reading the parsed `on:` block instead of a file whose comments argue about
`pull_request_target`. When a matcher reads text the change under test does not control, scope it
by what the artifact *declares*.

---

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
| ~~**G26**~~ | **P13's CC-origin twin — the Codex body builder drops 13 non-sampling control fields, unregistered** — KBR-184 · **CLOSED 2026-09-16** | Was: found while closing KBR-171, which closed the identical defect on the Responses-origin path with P23. `_cc_to_responses` ships `model`, `messages`→`input`, `stream`, `store`, `tools`, `tool_choice`, `parallel_tool_calls` and an injected `reasoning`; against T-A2's `_PUBLISHED_EXTRA_KEYS` (14 published top-level Chat Completions control fields, retrieved 2026-09-14 from `openai/openai-openapi` master) that leaves **13** which are neither carried nor sampling — `metadata`, `user`, `service_tier` and the agent's own `reasoning_effort` among them. *(The ticket counted 17, then 16 after KBR-214 began carrying `parallel_tool_calls`. T-A2's table has since surfaced three keys the older 37-field count included — `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier` — as not CC-surface extra keys: the reader residualises them, the §3.3.2 "named, honest failure" shape, so no row is owed. `store` rides `extra[store]` but is rewritten, not dropped — P17's territory.)* P13 is anchored at the bare `conversation.sampling` and reaches none of these, so T-D5 would have reported a false I1 breach on the CC-origin route exactly as it would have on the Responses-origin one | Done: row **P24** — trigger `CC_ORIGIN_PATH`, conditional=False, enumerating the 13 `envelope.extra[<wire key>]` addresses (`audio`, `function_call`, `functions`, `metadata`, `modalities`, `moderation`, `prediction`, `prompt_cache_options`, `reasoning_effort`, `service_tier`, `user`, `verbosity`, `web_search_options`), with the derivation guard `TestP24ClaimsTheDroppedNonSamplingControlFields` that recomputes the set from the AST: the reader's `_PUBLISHED_EXTRA_KEYS` minus the builder's carries. The enumeration is reader-table-driven so widening either input turns the row red; widen the allowlist / drop a reader key / damage a builder literal are the falsification cases | — |
| ~~**G27**~~ | ~~**An allowlisted Codex control field is dropped when its value is falsy, unregistered** — KBR-185~~ · **CLOSED 2026-09-14** | Was: found by the design review of KBR-171 and confirmed by running the builder. `_ALLOWED_RESPONSES_PARAMS` is read only by `_prepare_responses_body`'s DEBUG log; the shipped body is an explicit `if` chain and six of its branches test **truthiness** (only `parallel_tool_calls` tests presence), so `include: []` and `reasoning: {}` are permitted by the allowlist and dropped anyway. The reader projects by presence, so each is an unclaimed `envelope.extra[...]` delta. P23 excludes both by construction (they are inside the allowlist), P14 reaches no `extra` path, and P22 (KBR-149) is conditional on `REASONING_EFFORT_PRESENT`, which this case does not meet | Done: row **P25** — trigger `ALLOWLISTED_FIELD_IS_FALSY`, conditional, anchored at `envelope.extra[include]` and `envelope.extra[reasoning]`, on `_prepare_responses_body`. Owner decision 2026-09-14 (KBR-185): register the drop, do not change the adapter — `include: []` and `reasoning: {}` carry no instruction, the allowlist permits them, the truthiness branches drop them, and the defect was that the drop was undocumented. `tool_choice` is truthiness-gated on the same chain and is an `envelope.extra` key, but its falsy form is not a legal `CreateResponse` value, so that branch is unreachable today — recorded in P25's Notes so no future reader re-derives it. P22's interaction: the two triggers are predicates on different request fields and can co-occur (falsy `reasoning` beside a non-`none` effort); in that state the `elif` injects and P22's injection claims the address, so the rows' claims on `envelope.extra[reasoning]` do not overlap. The corpus trigger case and the §3.3.2 assertion-2 complement arrive with the T-C entries, as for P22 | — |
| **G28** | **`top_k` is dropped on every non-Anthropic-family route, unregistered** — KBR-178 | Found by the design review of KBR-178. Chat Completions declares no `top_k` (`CreateChatCompletionRequest` has zero occurrences), so KBR-178 carries the agent's value on the internal key `_top_k` and only `AnthropicAdapter` and its four delegates restore it — **five** registry entries, `opencode_go` only on its Messages-routed models. The other **eighteen** routes therefore drop it **by design and permanently** — including `ollama_cloud`, which is the one adapter that accepts `options.top_k` and is now guaranteed never to receive it, and `bedrock`, whose Converse `InferenceConfiguration` has no `topK` member at all. `reader_anthropic_messages.py:84` projects the field and `Conversation` enforces it as one of the closed fifteen `SAMPLING_KEYS`, and no register row claims `conversation.sampling[top_k]` outside `openai_subscription` (P13/P14 reach it there via the bare `CONVERSATION_SAMPLING` anchor). So the moment T-D5 drives a corpus entry carrying a `top_k`, the oracle reports a **false** I1 breach on a deliberate drop — the under-claiming direction §3.3.1a calls unrecoverable, and structurally the same defect as G26 and G27 | Either widen the restore to every destination that accepts the field, or add a bridge-level row anchored at `conversation.sampling[top_k]`. **Unconditional** in P13's sense — it fires wherever the field is present, so it owes no §3.3.2 assertion-2 complement, which is what separates it from G29. **Deferred deliberately, not overlooked:** KBR-178 chose the internal-key route on the product owner's decision. The row itself waits on T-D5 for its trigger case, as G26 and G27 do. Recording the residue here is what stops the trade-off being mistaken for an accident. **Before T-D5** | **1** |
| **G29** | **An empty inbound `stop_sequences` is omitted rather than forwarded, unregistered** — KBR-178 | Found by the design review of KBR-178 and confirmed by running the reader. D6 omits an empty `stop_sequences` instead of forwarding it, because `StopConfiguration` in `openai/openai-openapi` declares **`minItems: 1`** — so `stop: []` is a schema-invalid Chat Completions body, and forwarding it would put one on the wire of the **sixteen** adapters that deliver a top-level `stop`. **The omission is correct and must not be "fixed" by forwarding `[]`.** But it is still an unclaimed delta: `reader_anthropic_messages.py:193-195` projects sampling **by presence with no emptiness guard** and `contract.py:1063` validates sampling **keys only**, so an inbound `stop_sequences: []` projects as `conversation.sampling[stop] = []` while the upstream projection has none. Measured: `_project({... 'stop_sequences': []})` yields `{'max_tokens': 8, 'stop': []}`. Claimed only by P13/P14's bare `CONVERSATION_SAMPLING` on `openai_subscription`; unclaimed on the other twenty-two routes, so §3.3.2 assertion 1 reports a **false** I1 breach on a deliberate, correct omission | A row of its own, and **conditional** — unlike G28 this depends on the *value*, not the route, so per G27's precedent it also owes §3.3.2 assertion 2 a **complement**: a corpus entry with a non-empty `stop_sequences` in which the omission is provably absent. Trigger `EMPTY_STOP_SEQUENCES`. Deferred with G28 and G30, for the reason G26/G27 are: the trigger case and the assertion-2 complement both need the T-D5 corpus. **Before T-D5** | **1** |
| **G30** | **The Chat Completions string-form `stop` is rewritten into a list, unregistered** — KBR-178 | Found by the code review of KBR-178. `StopConfiguration` declares `stop` as `oneOf` a string or an array, so `"END"` and `["END"]` are the same request; every wire kitty writes downstream takes only the array form. `server._normalize_cc_stop`, called from `_handle_chat_completions` before the body forks, wraps a non-empty string. That is a real rewrite of the agent's bytes at the §3.2.3 boundary, and **no register row claims it** — the same class as G22/G23/G26-G29 | Row **M17**, modelled exactly on **M15**, which is this same decision already taken for the Responses `input` field: `site=("server._normalize_cc_stop",)`, trigger always, `paths=(NOT_PROJECTABLE,)`, **unconditional** — an array-form body meets the row as a no-op, so there is no complement state for §3.3.2 assertion 2 — and `NOT_PROJECTABLE` for P16's reason: both forms project to one `Conversation`, and a projection that told them apart would be reading a vendor's spelling into a wire-independent form. **This binds the future Chat Completions reader (T-A2 / KBR-34) to read `stop: "END"` and `stop: ["END"]` into the identical `Request`; if it ever does not, this row needs a projectable anchor instead.** Without that clause the escape is void, which is the failure M15's own text warns about. Deferred for the reason G26 and G27 are: a register row needs its trigger case and, when conditional, its §3.3.2 assertion-2 complement, and the corpus that supplies both arrives with T-D5. *(An earlier draft gave the reason as "`register.py` is held by a concurrent session"; KBR-167 merged as PR #87 mid-task, so that reason is void — and it was a scheduling reason masquerading as a design one, which the design review said at the time.)* **Before T-D5** | **1** |
| ~~**G31**~~ | **An agent's `metadata` is dropped off the Anthropic family, unregistered** — KBR-214 · **CLOSED 2026-09-16** | Was: KBR-214 carried the Anthropic `metadata` (Claude Code sends `metadata.user_id` on every request) on the internal key `_metadata`, which only `AnthropicAdapter` and its four delegates restored — the same five registry entries as G28. So **seventeen** routes dropped it by design and permanently, plus `openai_subscription` (the eighteenth, which G26 / P24 now claims at the provider level). `reader_anthropic_messages.py` projected it to `envelope.extra[metadata]` and no row claimed that address, so T-D5 would have reported a false I1 breach — G28's shape. **Deliberate, on the product owner's decision (2026-09-13):** Chat Completions' own `metadata` is a 16-pair string map for stored completions, and its abuse field is `safety_identifier`; a bare mapping would put a new field on every request to sixteen third-party providers, and a strict one would reject every turn | Done: bridge-level row **M26** — trigger ALWAYS, anchored at `envelope.extra[metadata]`, site `carry_tool_choice_and_metadata`. The 17-route omission is the bridge's design decision and is named once here rather than as 17 per-adapter rows (G28's shape). The future CC reader (T-A2 / KBR-34) will land CC's own `metadata` (a stored-completions tag map, NOT the Anthropic user-id object) at the **same** address — the row title and comment record the split so a reader or oracle does not conflate the two meanings. ⚠️ Because §3.3.1b keys `extra` by wire key, the address is shared and the meanings are not | — |
| ~~**G32**~~ | **Tool-selection fields are dropped where the destination wire has no field for them, unregistered** — KBR-214 · **CLOSED 2026-09-16** | Was: Ollama `/api/chat` defined neither a tool choice nor a parallel-tool-use knob (`ollama/ollama` `docs/api.md`), so `OllamaCloudAdapter` wrote neither; Bedrock Converse's `ToolConfiguration` had no parallel knob (botocore `bedrock-runtime`), so `BedrockAdapter` did not write one. Each was an unclaimed `envelope.extra[tool_choice]` or parallel-knob delta on its route. Not a defect in the adapters — nothing on those wires can carry the field | Done: two unconditional provider rows — **P31** at `OllamaCloudAdapter.translate_to_upstream` claiming both `envelope.extra[tool_choice]` and `envelope.extra[parallel_tool_calls]`, and **P32** at `BedrockAdapter.translate_to_upstream` claiming `envelope.extra[parallel_tool_calls]`. Split into two rows because the bedrock hook carries tool choice (G33 / P33); P9e/P9f's "paths must be true of every site" rule forbids sharing a row whose paths are not true of every named site. The parallel-knob anchor address is the G36 / KBR-205 canonical form | — |
| ~~**G33**~~ | **Bedrock writes `toolChoice: {"auto": {}}` whenever the Chat Completions body forces nothing, unregistered** — KBR-214 · **CLOSED 2026-09-16** | Was: `BedrockAdapter.translate_to_upstream` always wrote `auto` whenever tools were present; KBR-214 mapped only `required` and the named form onto `any` and `tool`. Converse's `ToolChoice` union has no `none`, and dropping `toolConfig` was unavailable once a transcript held `toolUse`/`toolResult` blocks (*"`toolConfig` must be defined when using toolUse and toolResult content blocks"*) — conditional on history, which KBR-214 rejected as a new conditional mutation. Pre-existing; recorded now because the Converse reader (T-A5) maps `toolConfig.toolChoice` onto `envelope.extra[tool_choice]` (§3.3.1b) and P11 is `NOT_PROJECTABLE`, so nearly every Bedrock request with tools showed it | Done: conditional provider row **P33** — trigger `BEDROCK_FORCES_AUTO_TOOL_CHOICE` (REQUEST), anchored at `envelope.extra[tool_choice]`. The trigger covers an absent choice, `none`, and the choices G35 omits upstream of this adapter; its §3.3.2 assertion-2 complement is a `required` choice or a named choice to an ordinary tool, which the adapter must carry as `any` or `tool`. Shares the address with P35; distinguishable by site (P3/P4 precedent, `test_no_two_rows_are_indistinguishable` explicitly allows). Both conditional, so the two rows cannot disagree about whether the address owes a complement. The trigger case + §3.3.2 complement arrive with the T-D5 corpus entries, as for P22 / P25 | — |
| ~~**G34**~~ | **`disable_parallel_tool_use: false` is omitted rather than forwarded, unregistered** — KBR-214 · **CLOSED 2026-09-16** | Was: KBR-214 mapped the flag only when `true`, onto `parallel_tool_calls: false`, on the product owner's decision: `false` is Anthropic's documented default and `true` is `ParallelToolCalls`' default, so the two spellings are one request on both wires, and writing it would add a second field some providers reject. The Anthropic reader can nonetheless tell them apart, so an explicit `false` was a delta — **the omission is correct and must not be "fixed" by forwarding it**, G29's exact shape | Done: conditional provider row **P34** — trigger `ANTHROPIC_PARALLEL_FALSE_OMITTED` (REQUEST), anchored at `envelope.extra[parallel_tool_calls]`, site `carry_tool_choice_and_metadata`. The address exists as of KBR-205 (§3.3.1b); the row registers the omission now, with the trigger case + §3.3.2 complement (a body where the flag is `true` and carried) arriving with the T-D5 corpus, as for P22 / P25 | — |
| ~~**G35**~~ | **A legal inbound `tool_choice` is omitted in two cases, unregistered** — KBR-214 · **CLOSED 2026-09-16** | Was: KBR-214 carried the agent's choice except where carrying it would create a failure the agent did not cause: (1) a choice over no tools — legal on Anthropic, and *"'tool_choice' is only allowed when 'tools' are specified"* on OpenAI; (2) a forced call to an **Anthropic-defined** tool, meaning a declaration whose `type` is neither absent, `null` nor `"custom"` — Claude Code's WebSearch forces `web_search`, declared `type: "web_search_20250305"`, which the translator flattens into a schema-less function nothing on the route can execute (anthropics/claude-code#56984; omitted on the product owner's decision, 2026-09-13). A forced call to an **undeclared** tool is *not* omitted: it is the agent's mistake and the provider's error names it. Each case projected as `envelope.extra[tool_choice]` inbound and nothing upstream — or `auto` on `bedrock`, which is **G33**. `{"type": "any"}` over only Anthropic-defined tools was carried: nothing showed an agent sending it, and guarding it would mean reasoning about the whole tool list rather than one name | Done: conditional provider row **P35** — trigger `TOOL_CHOICE_OMITTED_AS_LEGAL_BUT_UNSUPPORTED` (REQUEST), anchored at `envelope.extra[tool_choice]`, site `carry_tool_choice_and_metadata`. Both case-1 and case-2 triggers are now authorable: the Anthropic reader carries `ToolDecl.type` as of KBR-205 (the blocker this row named is gone). Shares the address with P33; distinguishable by site (Messages-route omission vs bedrock auto-rewrite). The trigger case + §3.3.2 complement arrive with T-D5. Case (2) is also where a future capability — running server tools on a translated route — would remove the row rather than widen it | — |
| ~~**G36**~~ | ~~**The parallel-tool-use knob has no canonical address, so the oracle fails the run on it** — KBR-214~~ · **harness, not product** · **CLOSED 2026-09-14** | Was: found by the design review of KBR-214. `reader_anthropic_messages._read_tool_choice` sent `disable_parallel_tool_use` to the residual (pinned by its own test), and a non-empty residual fails the run before register matching — so every Messages body carrying the flag failed, whatever the product did. Fixing the reader alone was not enough: §3.3.1b unified `tool_choice` across four wire keys *"because four spellings name one concept"* and said nothing about the knob, so Anthropic's nested, inverted flag and Chat Completions' top-level `parallel_tool_calls` would have landed at two addresses with two polarities, and every route that carried the flag **correctly** would have shown a false delta. Excluded from §3.2.5's count, which is about mutations the product performs | Done: **§3.3.1b** gives the knob one address and one polarity — `envelope.extra["parallel_tool_calls"]`, the Chat Completions spelling and polarity (`true` parallel allowed, `false` parallel not allowed) — and the reader writes the entry only when the wire carries a non-default value, mirroring KBR-214's forwarding rule. The Anthropic reader maps `disable_parallel_tool_use: true` onto `parallel_tool_calls = False`; the Chat Completions reader (T-A2) reads `parallel_tool_calls` directly. `test_disable_parallel_tool_use_is_not_part_of_the_canonical_value` is inverted to assert the new mapping at the same path it used to pin as residual | — |
| ~~**G37**~~ | ~~**The Anthropic adapter family drops a Chat Completions request's cache breakpoints everywhere except user and tool content, unregistered** — KBR-199~~ · **CLOSED 2026-09-14** | Was: found while implementing KBR-199. `_handle_chat_completions` handed the body to `translate_to_upstream` without translating it, and a `cache_control` was then **dropped** at the top level, on a system content part and on a tool declaration (a system content part is one of OpenRouter's defined content parts; the top level and the tool declaration are the other two sites its Chat Completions dialect defines), and on a message object and a tool call (two sites no published dialect defines), and **kept** on a user content part and on tool-message content (moved inside the `tool_result`). Measured identical on `anthropic`, `minimax_token`, `zai_coding` and `custom_anthropic` (the last two are native only for Messages bodies). OpenAI's own spelling, `prompt_cache_breakpoint`, was not ignored: the same family forwarded it verbatim on user and tool content, onto an Anthropic wire whose schema does not define it, and dropped it elsewhere. No register row claimed any of it, and the oracle could not see it yet: no Chat Completions reader filled the `cache_control` slot (§3.3.1), so such a body residualised and failed the run first | Done: **§3.3.1b** (KBR-205 closing G36) gives the parallel knob one address and one polarity; **T-A2 (KBR-34)** (closing G37) fills the `cache_control` slot from **both** spellings, verbatim — ``cache_control`` and ``prompt_cache_breakpoint`` both project onto ``Part.cache_control`` (and ``ToolDecl.cache_control``) as the wire carries them, the value shape travels because the slot is ``Mapping[str, Any] | None``, and §3.3.1's "carried whole, not reduced" rule applies to a spelling with no TTL the same way it applies to one with. The reader's `_CACHE_KEYS` constant pins both spellings in one place; `test_reader_chat_completions.TestCacheBreakpoints` asserts both verbatim, the OpenAI `mode: "explicit"` shape on the Anthropic `cache_control` slot, and the residual-on-wrong-type case. The `cache_control` spelling is OpenRouter's CC dialect (Anthropic's own field); `prompt_cache_breakpoint` is OpenAI's GPT-5.6+ spelling with a request-wide TTL (`prompt_cache_options.ttl`, `30m` only). The dropped sites the row was owed — top level, message object, tool call — remain dropped in the adapter, and the register rows that name them are still owed to M16's twin *(CC-origin rows P26–P30 landed 2026-09-15, KBR-258; the phrase now scopes to the Messages-route twin rows, which remain owed)* | **2** |
| ~~**G38**~~ | ~~**A top-level `cache_control` is dropped on every translated route, and M16 disclaims it** — KBR-199~~ · **CLOSED 2026-09-16** | Was: found by the design review of KBR-199. Anthropic's automatic-caching form projects to `envelope.extra[cache_control]` (§3.3.1). `MessagesTranslator.translate_request` never copied it, so on every route that was not native passthrough the upstream projection had no such key. M16's paths named the system, part and tool carriers only and its text called the top-level form "not this row's"; M2 is `NOT_PROJECTABLE`; no row anchored on that `extra` key. So a body using automatic caching yielded an unclaimed delta on every translated route — §3.3.1a's under-claiming direction, a false I1 breach. The loss itself cost the user the same as M16's | Done: **KBR-263** extended M16's `paths` with `c.extra_path("cache_control")` (the keyed literal — §3.3.1a, no `_SHAPES` entry) — same site (`MessagesTranslator.translate_request`), same trigger (`NON_NATIVE_UPSTREAM_WIRE`), same `conditional=False` — and amended §3.2.1's row to claim the form. The translator still drops the marker; the product-side carry-through remains epic KBR-197's boundary. The M9-fallback converter (`server._convert_native_to_cc_format`) drops more carriers still, recorded on KBR-263 for separate triage. **Before T-D1 (KBR-51) drives a body with a top-level `cache_control`** — that deadline no longer binds: the address is claimed before any body carrying it arrives | — |
| ~~**G39**~~ | ~~**The translated Messages stream opens every parallel `tool_use` at block index 0** — KBR-226~~ · **CLOSED 2026-09-14** | Was: the tool-call branch of `MessagesTranslator.translate_stream_chunk` recorded `block_index: self._content_block_index` and emitted `content_block_start` at it but never advanced the counter, while the text and thinking branches advance theirs when they close. Two parallel tool calls therefore opened at index 0, both calls' argument deltas landed under index 0, and the finish path closed index 0 twice — every event valid on its own, the sequence not a sentence in the §6.2.2 grammar. Found by KBR-183's design review, confirmed by running the translator | Done: the counter advances when a tool block opens — the same address the text and thinking branches already use — so each call gets a distinct increasing index and a following text block opens the next free one; the finish path, `finalize_interrupted_stream` and `close_open_blocks` close by the recorded `meta["block_index"]`, so one stop per index follows without further change. **Decided stream shape:** blocks opened by parallel calls may overlap and close out of order — clients key blocks by index, and closing a tool block early would risk a delta after its stop — but each index opens once, closes once after its start, and carries no delta outside its window; the overlap itself is not new (pre-fix text-after-tools overlapped at the *colliding* index). **Scope-out:** a repeated id-chunk for an already-open Chat Completions tool-call index re-enters the open branch and stays malformed, as before this fix. **Advancing on open cannot flip the empty-stream fallbacks:** a tool-open always creates its `ToolCallBuffer`, so `had_any_content` already holds via the buffers term. Sibling check (the ticket's DoD item 3): Gemini is clean — positional `parts[]`, no index in its grammar; Responses has the same class through a different mechanism (no shared counter; text pinned to `output_index: 0`), filed as **KBR-240**. Regression tests: `tests/bridge/test_messages_translator.py::TestParallelToolCallBlockIndices`, `tests/bridge/test_parallel_tool_use_stream.py`. Red at base, evidence in the PR | — |
| ~~**G40**~~ | ~~**The translated Responses stream has no output-item counter, so text and the first function call claim one `output_index` slot** — KBR-240~~ · **CLOSED 2026-09-14** | Was: `ResponsesTranslator` pinned the reasoning and text items to `output_index: 0` at every event site and took a function call item's `output_index` raw from the Chat Completions `tool_calls[].index` — also `0` for the first call — so a stream carrying text and then a tool call, the common Codex shape, announced two items at slot 0 and both `output_item.done` events closed slot 0; reasoning vs text collided the same way. Every event valid on its own, the sequence not a sentence in the §6.2.2 grammar. Found by KBR-226's sibling check (its DoD item 3), confirmed by running the translator | Done: one shared output-item counter — each newly opened item (reasoning, text message, function call) allocates the next `output_index`, and every event referencing the item carries its recorded index on the chunk path, `_build_finish_events` and `synthesize_completed_events` alike; the CC `tool_calls[].index` remains a routing key only. The ticket's DoD audit of the close paths (its "audit the finish path and finalize/reset" item) then found `synthesize_completed_events` — the EOF-without-finish path timeouts and dropped connections reach — closing neither the reasoning item nor a text item whose text stripped to nothing: both close there now, at their recorded index, and **every opened item appears in `response.completed`'s `output`, so array position equals `output_index` throughout** (a text item whose content was all thinking tags closes with empty text and stays in the array — its done event already did). `response.function_call_arguments.delta/done` now carry the owning call's `output_index` — the vendor grammar defines the field as required (OpenAI SDK generated types, verified 2026-09-14); kitty omitted it. **Decided stream shape:** G39's — items may overlap and close out of order, but each index opens once, closes once after its start, and carries no delta outside its window. **Scope-out:** the non-streaming `translate_response` needs no allocation — the Responses body's `output` array is positional and carries no per-item index field; a repeated id-chunk for an already-open Chat Completions index re-enters the open branch and stays malformed, G39's scope-out shared. **Cannot flip the empty-stream fallbacks:** allocation happens where the buffers are created, and `response_was_empty` reads accumulated text, reasoning and the buffers — not indices (the synthesize path's own `was_empty` reads text and buffers only; unchanged either way). The design review caught the first draft under-testing the text *part/delta/done* event indices at non-zero slots — a mutant pinning those sites to 0 survived the first suite; the mixed-stream test now pins every text-addressed event to slot 1 and kills it. Regression tests: `tests/bridge/test_responses_translator.py::TestOutputItemIndices` (10 tests), `tests/bridge/test_responses_output_index_stream.py`. Red at base, evidence in the PR | — |
| ~~**G41**~~ | ~~**The translated Responses stream never opens with `response.created` — the server buffers the lifecycle start and never writes it** — KBR-242~~ · **CLOSED 2026-09-14** | Was: `BridgeServer._stream_responses` built `translate_stream_start`'s two events and never wrote them — the only references were the assignment and a `logger.debug` loop — so every translated `/v1/responses` stream began at `response.output_item.added`, mid-sentence. The buffering comment stated an intent (protect the client from a half-open lifecycle across an empty-response failover) the write side never implemented. Found by KBR-240's server-level walk, which had to weaken its "complete sentence" assertion to match the wire | Done: the owner picked the failover-clean shape. The lifecycle is translated **speculatively at the start of each attempt** — before any `translate_stream_chunk`, because both draw from the same `_seq` counter and a write-time call would put `sequence_number` 3 and 4 on the wire ahead of the chunk's 0, 1, 2 — and **written lazily** on the first real non-finish event, with the strings invalidated at every `translator.reset()` inside the loop (the in-stream-error failover and the empty-response ladder). A purely-empty attempt publishes nothing, so no half-open lifecycle crosses the failover and KBR-247's `events_emitted` model is preserved; the exhausted-ladder fallback stays an empty 200 (probed: zero bytes); the error paths — cloudflare, upstream 5xx, timeout, in-stream terminal — never open the lifecycle. Regression tests: `tests/bridge/test_responses_output_index_stream.py` (the walk now asserts the opening and exact `0..len-1` sequence numbers) and `tests/bridge/test_responses_stream_lifecycle.py` (empty-failover opening-once + the R4 negative controls). Decided shape recorded as **S7** in `.system_design/SYSTEM_DESIGN.md` §5.3 | — |
| **G42** | **User images and documents on the routes that still drop them, unregistered** — KBR-222 | KBR-222 carries an agent's `image` blocks as CC `image_url` parts and `document` blocks on the `_documents` internal key. `AnthropicAdapter` and its delegates restore both (the tool run and its trailing user message re-join into one user message there, the shape Anthropic's docs prescribe); the OpenAI-family verbatim forwarders ship `image_url` parts natively. The remaining routes keep a deliberate drop no row claims: `ollama_cloud`'s `_flatten_content` and `bedrock`'s user-branch flatten keep only the text (`bedrock`'s flatten exists because forwarding the parts list would newly fail boto3 validation on every Messages-route image turn — KBR-223 owns real Converse handling), and `openai_subscription`'s `_cc_to_responses` flattens a parts list to its text, an image-only list yielding no input item at all. M5's pruning forfeit is recorded in that row. `reader_anthropic_messages` and `reader_responses` already project images on their wires, so the moment the corpus carries one (KBR-45's), the oracle reports a **real** I1 breach on each route in this list — a true positive that is still unclaimed | One row per route's flatten/drop, or claim them in M2's prose per destination as KBR-223 and KBR-45 land. **Before T-D5 drives a corpus entry carrying an image or document** | **1** |
| ~~**G23**~~ | ~~**`openai_subscription` injects `reasoning` from `_reasoning_effort`, unregistered** — KBR-149~~ · **CLOSED 2026-09-14** | Was: four sites in `providers/openai_subscription.py` set `reasoning: {"effort": …}` from kitty's internal key. The gap walk had counted three and missed `_cc_to_responses`, the CC-origin builder P13 names, which injects from the same key and predates the ticket. Structurally identical to P3 and P4, and **P4 cannot cover it**: §3.2.3 records that `translate_to_upstream` never runs on this adapter's request path. Unlike G22 this is a **request-body** row feeding §3.3.2 assertion 1, so the moment T-D5 drives a corpus entry carrying a reasoning effort the oracle reports a *false* I1 breach on a deliberate mutation — the under-claiming direction §3.3.1a calls unrecoverable | Done: row **P22** — trigger `REASONING_EFFORT_PRESENT`, conditional, anchored at `envelope.extra[reasoning]`, naming all four sites. At effort `"none"` the trigger is met-but-inert (the P5c precedent), so the §3.3.2 assertion-2 complement needs a corpus entry carrying an effort, not a `"none"` one; the Responses-origin half waits on T-A3 projecting the effort signal, which the CC-origin half's reader already has. The corpus trigger case and complement arrive with the T-C entries, as for P3 and P4 | — |
| **G22** | **Register header coverage was partial and inconsistent** — KBR-148 · **CLOSED** | Was: rows for four adapters (P9a ×3, P9b, P9c) while six more deviated from the base header set with none. **Closed 2026-09-14** as five rows, P9d–P9h. The closing survey corrected the gap's own lumping: `ZaiAnthropicAdapter` keeps `Authorization: Bearer` — its deviation is `anthropic-version` plus the lowercase `content-type` re-spell (P9f), not the `x-api-key` scheme change (P9e) this survey first attributed to it — because a row's paths must be true of every site it names. `opencode_go` reaches P9e's set only on its Messages models, by delegation rather than inheritance; `OllamaCloudAdapter` keeps Bearer and takes no row | Done: **P9d** (conditional `ChatGPT-Account-Id`; its assertion-2 complement is a claimless-`id_token` corpus entry, L1-pinned at `tests/providers/test_openai_subscription.py`, the fixture owed to T-D5), **P9e** (the Anthropic-family auth scheme), **P9f** (`zai_coding`), **P9g** (Azure's non-Entra `api-key`), **P9h** (Ollama's dropped `Authorization`). §4.3 C1's per-adapter expectation is now *reviewable against the register* instead of written from scratch — which is what stops C1 reproducing the ad-hockery F1 names; the exact-set assertion itself remains T-G9 | — |
| **G21** | **A declared trigger is never verified** — KBR-140 | §7.4 hands the oracle `triggers_met` as an argument and §3.3.2 asserts only that a row is **absent** when its trigger is not met. Nothing asserts a trigger declared met actually fired, so a corpus entry that over-declares makes assertion 1 claim every delta — the oracle reports green on a bridge that is rewriting messages. The same author writes the entry and its trigger index (T-W6), so the mechanism has no second reader | Roughly fifteen triggers are decidable from the inbound request; give those an optional predicate and have T-D8 require the declaration to agree with it. M6, M8, M9 and M12 depend on an upstream response and stay declaration-only — the stated residual risk. Blocked on T-A1/T-A2, since a predicate needs a projected request to read | **1** |
| ~~**G20**~~ | ~~**OpenCode Go's routing table does not match the provider** — KBR-126~~ | **CLOSED 2026-09-11.** Was: eight models served on `/v1/messages` against two routed there, four `/v1/responses` models with no route, and a `validation_model` that had left the catalogue — ten of twenty-eight models broken, presenting to the user as a false auth failure because the provider answers an unsupported model with `401` | Done: table refreshed against the provider's published list (re-verified live 2026-09-11); `/v1/responses` models route truthfully and are refused at serialization with `UnsupportedModelError` naming KBR-137, which owns the route itself; `validation_model` replaced and the whole class guarded registry-wide; snapshot oracle checked in. The balancing pool is explicitly protected from the refusal, mirroring the `CompactionFailedError` precedent | — |
| **G19** | Routing was outside the register and outside the oracle | The destination is built from the profile (M14, P20, P21); a body-only check cannot see a misrouted Azure deployment | §3.3.5 — whole-request oracle with an independently derived route | **1** |
| **G17** | Undecided behaviour for an irreducible final turn | Compaction emits an over-budget request, or (since KBR-5) the bridge refuses it downstream; neither was designed | Answer Q10, then align M3-M7, the 6.1 properties and TR-3 together | **2** |
| **G18** | P13-P19 - seven transport-level mutations, unregistered in the first draft | Necessary (the Codex backend and boto3 require them) but invisible above DEBUG, and unreachable by a guard placed at `translate_to_upstream` | Rows P13-P19; boundary corrected in 3.2.3; Q5 decides user visibility | **3** |
| **G21** | §8's skip rule is stated in prose and nothing checks it — KBR-138 | Found while closing KBR-132. The known breach is fixed, and every skip left in a *gating* layer is a platform or interpreter one — but that is an observation, not a mechanism, and the next resource-availability skip written into `l1`, `l2`, `l3` or `acceptance` re-creates the same silent-green defect | A check over the collected suite that fails on a resource-availability skip in a gating layer, with a planted skip as its falsification case (§1.4). It must be **layer-aware**: `tests/integration/test_agent_e2e.py` holds three legitimate resource skips (missing credentials, profile, agent binary) that are legal only because they sit in `agent_live`, so a flat grep would report them and be turned off. Two further questions: static sweep or runtime hook, and whether a permitted skip is recognised by condition shape or declared by marker | **3** |
| **G1** | I1 is unstated and untested | No definition of "unchanged"; mutation sites discoverable only by reading 6,463 lines | Register (§3.2) + oracle (§3.3) | **1** |
| **G2** | **Partly closed (KBR-67).** The start-path guard is no longer file-granular: the AST-level domination guard landed as `tests/test_egress_coverage.py::TestEveryStartPathIsGuarded` (§6.2.3). **Still open:** no-bypass unproven **for the bridge's serving path**; no negative assertion | `test_egress_https_proxy.py` proves the transports and drives `egress_cmd._probe` | Sealed-network harness (§5.2) per transport (§5.5); AST start-path guard landed (KBR-67) | **1** |
| **G3** | I2 partially breached (F1) — KBR-8 · **fix landed, gap open** | Identity is still ad hoc per adapter. The subscription adapter no longer reports two different versions in one request: **KBR-8 fixed that on 2026-09-11**, and `tests/test_upstream_identity_consistency.py` guards both halves of §4.3 C1's F1 assertions across every registered adapter | Remaining: the parity baseline (T-C7, T-I12), then a policy — Q1. **KBR-78 (2026-09-15) closed the exact-set half** — the per-route contract, the AST branch-arm meta-test, the bedrock botocore observation, and the OAuth token POSTs' exact sets (incl. `originator`). | **1** |
| **G4** | L1 strength unmeasured | Line coverage only | `mutmut` ≥ 85% **per target group** on the §6.1 scope | **2** |
| **G8** | No corpus of real agent traffic | Synthetic fixtures encode our assumptions | Golden corpus (§7.1) | **2** |
| **G10** | Custom-transport containment untested | **Partly closed (KBR-161, KBR-64).** Ambient `NO_PROXY` on `curl_cffi` is now measured, closed with `CURLOPT_NOPROXY`, and pinned; the OAuth refresh leg is covered by the same builder and contract. **KBR-64 closed the botocore half:** the four-phase containment slice drives the bridge end to end over the bedrock adapter (phase 1 positive control, proxy-down zero-connection, §5.2.1 tunnel join, injected-bypass falsification), records `PROVEN` in the capability report on every supported interpreter (no Python ≥3.11 skip — botocore's urllib3 is synchronous, so bpo-44011 does not apply), and the §6.2.4 precedence probe measured `Config(proxies=)` **winning** over ambient `NO_PROXY`/`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` in both letter cases — the opposite of the curl_cffi finding, and now pinned by `tests/harness/test_botocore_transport_contract.py`. **Still open:** the aiohttp login leg is unproven end to end | §5.5 + §6.2.4 | **2** |
| **G5** | No contract layer | No published schema; SSE grammar unchecked | OpenAPI + `schemathesis` + grammar state machine | **3** |
| ~~**G6**~~ | ~~Docs drift undetected (F2) — KBR-9~~ · **CLOSED 2026-09-17** *(this row was the guard's landing record; the fix closed it)* | Was: README endpoint table wrong; the **guard** half landed with KBR-76 under the exemption path (`t-g1-endpoint-table`, issue KBR-9) — the README correction itself remained in KBR-9 | Done: README corrected (KBR-9) — eight bridge-mode routes documented with the `{model:.*}` converter literal; the exemption row fired `UnexpectedExemptionPass` and was withdrawn with its `ratchet` plumbing; `TestEndpointTable` gates normally | — |
| **G7** | No property-based tests | All example-based | `hypothesis` on the §6.1 list | **3** |
| **G9** | C5 unmeasured | `force_close=True` gives a per-request connection pattern unlike the agent's | Connection-count baseline | **3** |
| **G11** | Dependency behaviour unpinned; `curl_cffi` unbounded, **botocore undeclared** and the interpreter declared to the minor only | **Partly closed (KBR-64).** The `botocore` declaration now lands in `pyproject.toml` (`botocore>=1.34`), named explicitly in the §6.2.4 row — and the four transport contract tests have landed (KBR-161 `curl_cffi` precedence + no-bypass; KBR-64 botocore precedence + no-bypass). Still open: `curl_cffi` and `keyring` lack bounded versions, the interpreter is still declared to the minor only | Dependency contract tests (§6.2.4) + declare botocore | **3** |
| **G25** | **§6.2.4 contracts are only ever evaluated on the newest patch of each minor** — KBR-146 | `tests.yml` names bare minor versions and `actions/setup-python` resolves each to the newest patch. Every dependency contract therefore proves forward drift only; a value that differs on an older patch a user runs — the shape KBR-146 had — is invisible to the gate. Today that half rests on one L1 test that forces the property both ways, which works because the surrounding behaviour was measured stable, and does not generalise to a contract whose neighbours have not been | One job pinned to the oldest supported patch (`setup-python` accepts an exact version, so it is one job, not four). Deferred as a CI-spend decision, not a technical one | **3** |
| ~~**G26**~~ | ~~**Post-emission failover is reachable for the timeout class** — KBR-183~~ · **CLOSED 2026-09-13** *(this ID is also used by KBR-184's row above — a numbering collision, not the same gap)* | Was: §11 Q14(a) was honoured for the **transport** class only. `_is_transport_error` returns `False` for `asyncio.TimeoutError` by design and the failover arm after it carried **no** emission test, so a mid-stream `sock_read` timeout after emission selected another backend (or, without balancing, retried the same one up to six times) and wrote a second attempt onto the already-prepared response | Done: the arm reads `sr`; a post-emission failure closes the blocks the client saw open (from the translator, or from the unsent finish buffer) and ends in one error, charging the backend. The same guard landed on the Responses and Gemini custom-transport failovers (`_bytes_written`) and in `openai_subscription`'s stream-reset retry, and a pre-emission failover now resets the translator. Regression tests: `tests/bridge/test_post_emission_no_failover.py`, `TestCloseOpenBlocks`, `test_does_not_retry_a_reset_after_bytes_were_written`. T-I7 still owns the L3 form. Residual: the transport drop's `message_stop` ending (§6.3.1) | — |
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
per hour. **§8.6 does not move G12**, and the reason is worth stating so the next reader does not
re-derive it: the inventory removes the *provisioning* half of G12's cost — credentials, profiles
and a gateway are already on the runner — but the cost that put G12 last is token spend and
nondeterminism, and neither is changed by a resource being available.

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

Answers belong in this document. They are not invented here. Q10 and Q13 are prerequisites for
the implementation work they name — each blocks a test whose acceptance oracle depends on it. An
answered question keeps its place in the list and carries its answer in the heading.

Q12 was in that set until 2026-09-12 (KBR-216). Its entry below records the answer, and what the
answer does not settle.

**Q-image-digest-ref — ANSWERED by the product owner, 2026-09-14 (KBR-192).** Yes — `Image.__post_init__` enforces a strict XOR between `digest` and `ref`. Construction with both `None` raises (the blindness KBR-179 names for `Opaque`); construction with both set raises (a phantom delta two readers could populate the pair differently for one image and report on content neither altered). The legitimate reader paths that today produced neither — Gemini's undecodable inlineData base64, Gemini's missing or wrongly-typed `fileData.fileUri`, and Anthropic's missing or wrongly-typed `url` / `file_id` — now give the part identity: the raw-encoded-bytes digest for the first case (the second of `image_digest`'s recipes), and the `opaque_digest` canonical-JSON digest of the malformed blob for the others. The residual still records the bad/missing value, so the run fails visibly at the right path. Dependent passages re-derived: §3.3.1 line 599-602 (the `Image.digest` recipe paragraph, now mentions the raw-bytes second recipe); §7.4 rule 7 row 2 (line 3167) and row 3 (line 3168) (the absent-value table now describes the canonical-JSON and raw-bytes outcomes respectively). Out of scope: the reconciliation ticket for the two earlier reader divergences (§7.4 "Reconciliation owed" line 3183-3186) — KBR-192 unblocks it by giving both branches a compatible answer to the "what identity does an unreadable image carry?" question.

**Q-cache-control-converse — ANSWERED by T-A5 (KBR-37), 2026-09-15.** The Bedrock Converse reader does *not* fill :attr:`~harness.contract.Text.cache_control` (or the Opaque slot) from Converse's ``cachePoint``. Anthropic's ``cache_control`` is a field on a block; Converse's ``cachePoint`` is a *separate block* in the content list (§3.3.1's `cache_control` slot note names this exception explicitly). The reader projects ``cachePoint`` to :class:`~harness.contract.Opaque` with canonical kind ``cache_point`` (reached via :func:`harness.contract.opaque_kind` and the new ``OPAQUE_ALIASES`` entry for it) and leaves :attr:`~harness.contract.Text.cache_control` ``None``. The same decision applies to ``cachePoint`` inside :attr:`~harness.contract.Conversation.system`: text lifts into ``system``; the two non-text block types (``cachePoint``, ``guardContent``) residualise at the entry path. ``M16`` (which claims the cache-point strip on the Anthropic route) does not transfer — Converse's wire has no carrier for the strip to claim. The decision is recorded in §7.4.3 above and exercised by :class:`~harness.tests.harness.test_reader_bedrock_converse.TestResidualsExpectedOnRealTraffic`. *Original question:* does T-A5 fill the ``cache_control`` slot from Converse's ``cachePoint`` block (the way T-A2 filled it from both Chat Completions' ``cache_control`` and ``prompt_cache_breakpoint``), or leave such bodies residualising at the ``cachePoint`` block itself? The first would invent a slot on a format that has no slot concept; the second is what §3.3.1 already names as the deliberate exception.

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

**Q6 — Which of the five body-changing retry paths are acceptable (§4.3 C3)?** M6, M8, M9, M17 and
failover re-normalisation each send a different payload on a later attempt, and a provider that
hashes bodies sees all five. The alternative in each case is to fail the turn, which is worse for
the user. Declare all five as exceptions, or close some?

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

**Q12 — ANSWERED by the product owner, 2026-09-12 (KBR-216).** CI has had a version-pinned
Claude Code since the automated reviewer landed, together with kitty profiles, credentials and an
egress gateway. `.github/workflows/claude-code-review.yml` installs the CLI from the official
installer script at an exact version, at run time. §8.6 is the inventory and the reasoning; the
answer to each half of the original question is: the official distribution, pinned by an exact
version argument, and **redistribution does not arise** because nothing is baked into an image.
So the settings-precedence claim *does* have a per-PR proof available, and T-I5, T-I6 and T-K10
are unblocked.

**What this does not settle.** Three things, named so a task does not inherit them as surprises.
The pin is a **hand-maintained pairing** with a floating `@v1` action tag — `§8.6` holds the
document to the workflow, but nothing holds the workflow to the action, so a job asserting a
*specific* CLI version rests on a human having noticed. A **fork pull request receives no
secrets**, which costs the hermetic smoke nothing and rules out a credential-dependent per-PR
gate entirely. And the product owner's answer establishes that the resource **exists**, not that
the smoke job's three-sentinel design (§6.4.2) is the right shape — that remains T-I6's to prove.

*Original question:* how is a pinned Claude Code binary supplied to CI (§6.4.2)? The per-PR agent
smoke needs a real Claude Code, not an arbitrary child process, because the claim under test is
Claude Code's own settings precedence. Which distribution, pinned how, and is redistribution
inside a CI image acceptable? If it is not, the settings-precedence claim has no per-PR proof,
and that limitation should be stated rather than papered over.

**Q13 — What is the baseline for the compaction arm of the evals (§6.4.3)?** For an over-context
input the direct-provider arm returns a 400, so there is no answer to compare against.
Candidates: kitty against a larger-context model, or kitty with compaction relaxed. The choice
determines what a regression in that arm actually means.

**Q14 — ANSWERED by the product owner, 2026-09-12.** Two parts, and the second is what makes the
first affordable.

**(a) Post-emission, the bridge closes the turn and surfaces the error.** Once a byte has reached
the client there is no retry and no failover. The bridge closes any half-open content block,
emits one terminal error, and lets the agent retry the whole turn. The first candidate below —
abandon the partial block and re-open under a new id — is **rejected**.

**(b) An empty stream is kept out of that situation rather than recovered from inside it.** On
the native passthrough the bridge performs a **preamble hold**: it withholds the stream's *leading*
events until the first content event arrives, mirroring the buffer the translated path already
keeps (*"Buffer finish events to detect empty responses before writing"*). A contentless reply is
therefore still pre-emission when it is detected and keeps the ordinary retry ladder. This is the
KBR-155 remedy; KBR-163 records it, KBR-155 implements it.

**Amended by KBR-227 (2026-09-13): "the native passthrough" now means the Messages-wire passthrough.** **Confirmed by the product owner, 2026-09-13:** (b) and the KBR-155 decisions D1–D7 below cover the translated Messages-wire routes too — one behaviour for every upstream that speaks Anthropic Messages, accepting that their users watch the spinner rather than live thinking until content arrives.
`_stream_messages` forwards the upstream stream unchanged whenever `BridgeServer._serves_messages_wire`
holds — a native adapter, **or** a translated one whose upstream speaks Messages for the routed model
(`anthropic`, `minimax_token` by default, `opencode_go`'s Messages models). Those translated routes used
to push Anthropic SSE through the Chat Completions chunk translator, which discarded every event and sent
Claude Code an empty reply. So everything (b) says about the passthrough, and the preamble hold KBR-155
adds, applies to them too; every site that decides "Messages wire" must call that one method. What the
change traded, recorded here because each is a behaviour the translated branch had or lacked:

- **A pre-content `error` event is forwarded and the backend marked healthy**, as on native passthrough.
  The translated branch failed over on it, and on these routes that worked — while every successful
  stream arrived empty. Restoring failover is **KBR-233**; the preamble hold is its natural home.
  *(Resolved 2026-09-14: KBR-233 was closed NOT-fixed pending a fresh owner decision, and KBR-241
  supplied it — the hold now records a pre-content error event and the ladder runs, on every
  Messages-wire route; see D2's amendment.)*
- **Post-emission timeout failover (G26 / KBR-183) becomes reachable** on these routes, because bytes now
  reach the client.
- **A forwarded stream is counted as one completion** (`_log_usage(None)`), for native routes too, which
  counted none. Tokens stay 0 on every Messages-wire route: `_log_usage` reads Chat Completions usage keys.
  Under `--logging` this writes a zero-token usage-log line per turn. When the preamble hold lands, the
  count belongs **inside** its release test: a discarded empty attempt is not a completion.
- **Claude Code now receives signed thinking blocks**, and the request side still strips signatures, so
  with thinking on a signature-validating upstream rejects turn 2 and a balancing pool classifies that 400
  as `hard`. Not a regression — turn 1 used to be empty — and **KBR-228**'s to fix.
- **Non-streaming** `/v1/messages` on the same routes still translates: thinking dropped, stop reason
  mapped, M12 reachable. Aligning it is KBR-228's.
- An accidental Chat Completions stream from a Messages-wire upstream is no longer translated on
  `/v1/messages`. The other inbound protocols share the original defect and still translate: **KBR-232**.

**Completed by KBR-232 (2026-09-14):** the three non-Messages stream handlers now convert a
Messages-wire upstream's Anthropic SSE into Chat Completions chunks through the stateful
`AnthropicCCStreamConverter` (`kitty.providers.anthropic`) before their per-chunk logic, gated
per attempt on `_serves_messages_wire` — tool calls stream under a per-block `tool_calls`
index, thinking crosses as `reasoning_content`, the finish chunk carries accumulated Chat
Completions usage, and the Responses and Gemini empty-response ladders became reachable
(their finish events exist only since this conversion). M17's strip-and-retry covers these
three streams too.

**Completed by KBR-248 (2026-09-17):** the converted `/v1/chat/completions` route joined
them — the handler holds the converter's non-content lines (role chunk, finish chunk,
`[DONE]`) until the first content-bearing delta, so a content-less completion fires the
empty-response ladder there instead of reaching the client as a well-formed skeleton, and an
exhausted ladder ends in the route's D4 terminal error (`type: "empty_response"`) like the
siblings. The hold is converter-gated; raw Chat Completions upstreams keep their pre-KBR-248
behaviour.

Four things the implementer needs that the question itself did not settle, decided here so KBR-155
is writable:

- **What releases the hold.** The first `content_block_delta` of a non-thinking block. A
  thinking-only reply and a `message_delta` carrying `stop_reason: max_tokens` with no content are
  **not** content: the first is a reply the user cannot read, the second is a truncation no retry
  can improve — so the first releases nothing, and the second exhausts the ladder rather than
  restarting it.
- **The held bytes are replayed verbatim, never re-serialised.** §4.3 C2 keeps a byte-level
  key-order assertion that applies wherever kitty claims to be forwarding rather than translating.
  A hold that re-emitted parsed events would break that claim while looking identical downstream.
- **Ladder exhaustion.** When every attempt comes back empty the client receives a **terminal
  error** — not fallback text, and not the empty stream. Nothing has been emitted, so this is the
  ordinary pre-emission error path. Deliberately *not* the translated path's substituted fallback
  text: that is register row **M12**, whose Site column names the two translators, and the native
  path drives neither. Fallback text here would open a second place where the bridge puts words in
  the model's mouth and would require M12's row to be widened in the same change; an error opens
  none.
- **The accepted cost, stated rather than discovered.** Under the hold a stalled native stream
  produces *no* downstream bytes until `_STREAM_READ_TIMEOUT`, where today the client sees
  `message_start` within a round trip; and because `sr` stays `None`, that failure then surfaces as
  a pre-emission JSON error response rather than a `200` carrying an SSE `error` event. Both change
  the downstream contract, and §6.2.2's grammar suite must cover the second.

**Amended by the product owner, 2026-09-13, while implementing KBR-155.** Reading the four
points above against the Anthropic stream format, and a design review of the implementation,
found cases they did not reach and one they understated. Each is decided here, with its reason:

- **D1 — the release rule also counts a block that arrives whole.** Besides the first
  `content_block_delta` of a non-thinking block, the hold releases on a `content_block_start`
  whose block already carries content: any type other than `text`, `thinking` and
  `redacted_thinking` (`tool_use`, `server_tool_use`, `web_search_tool_result`, and any type not
  yet known), and a `text` block whose `text` is non-empty. *Why:* Anthropic's own result blocks
  have no delta at all — the official SDK's accumulator (`lib/streaming/_messages.py`) seeds its
  snapshot straight from `content_block_start` — and `bridge/tool_audit.py` records that shims
  handing back a pre-parsed tool call do the same. The delta-only rule would judge those replies
  empty, retry them, and then fail them.
- **D2 — an upstream `error` event before content is passed through.** It releases the hold and
  is forwarded verbatim, exactly as before the hold existed. *Why:* the provider's own error type
  (`overloaded_error` and the like) is what Claude Code is written against; treating it as an
  empty reply would replace it with the bridge's wording and add a retry policy nobody decided.
  The event is recognised by its SSE `event:` name as well as its `data.type`, because the
  official SDK (`_streaming.py`) raises on `sse.event == "error"` whatever the data holds, and
  fills a missing `data.type` from the event name. **Also as before:** a released error-only
  stream counts as a completed attempt, so its backend is marked healthy — the translated path
  quarantines on an in-stream error, and closing that difference is not this decision's.

  **Amended by the product owner, 2026-09-14 (KBR-241; supersedes KBR-233's "closed, NOT
  fixed"): a pre-content error event is no longer delivered — it is judged, and the attempt
  takes the ordinary pre-emission ladder.** KBR-233 had asked for exactly this and was closed on
  the recorded condition that revisiting it "needs a fresh owner decision amending D2 in §11
  first"; KBR-241, filed an hour after that closure after the owner's own session was stopped
  by this error a second time, is that decision — confirmed in conversation as recover
  **pre-emission only**, with Q14(a) untouched. What the amendment keeps of D2: the recognition
  rule (name or `data.type`, decided at the name line, as the SDK decides) and the exhaustion
  rationale — when the ladder runs out, the client receives the **provider's payload
  re-embedded** (`502`, with only `reason: "upstream_error"` added inside its `error` object),
  because the provider's error type is what the client is written against; this deliberately
  departs from Q9/D4's "names the product" precedent, so the body is not mistaken for a defect
  or "fixed" into kitty's own wording later. When the error was seen but its payload is too
  malformed to deliver (no `error` object, unparseable data), the marker still says
  `upstream_error` and the wording is kitty's own (Q9) — an errored ladder never reports
  `empty_response`, which would be false reporting against D4's own rationale for the marker.
  What it changes: the hold records the error
  (`error_seen`, `error_event`, `error_event_complete` — a chunk boundary may fall between the
  name line and its data line, and reading stops only when the event's lines have all arrived)
  and is deaf to everything after it (a later `message_delta` stop reason records nothing, so
  the judge's answer cannot depend on how the stream was chunked); the handler stops reading at
  the completed error event and runs the empty ladder. Two differences are **kept, not closed**:
  the health model stays the empty ladder's — no quarantine, unlike the CC-wire path's in-stream
  cooldown — and the two wires now also exhaust differently, the CC-wire generic body carrying
  neither the provider payload nor a reason marker. The `sr is not None` arm ends an
  already-open stream with the provider's payload raw, or kitty's upstream-error wording when
  the payload is unusable. Three accepted residues: error-then-content
  reaches the client only when both arrive in one chunk (the SDK raises on the event, so that
  content was wasted anyway); the D5 cap still fails open past an unjudged error, delivering as
  before this amendment; and a stream that stalls *inside* the error event (name line seen, no
  blank line) still ends at the read timeout, since judging it early would truncate the event.
- **D3 — a truncation before content fails at once with a `400`.** A reply whose stop reason is
  `max_tokens` or `model_context_window_exceeded` and that carried no content gets
  `invalid_request_error` with `reason: "<stop_reason>_before_content"`, and the ladder ends on
  that attempt. *Why:* the point above already says no retry can improve a `max_tokens`
  truncation, and the SDK's `StopReason` names the context-window stop as the same kind of
  ending; a `400` also stops the agent retrying it, where a `5xx` would invite exactly that.
  Every other stop reason before content is an empty reply.
- **D4 — the exhaustion error.** `502`, `api_error`, `reason: "empty_response"`, and a message
  that names the product, per Q9's precedent. The `reason` marker follows `compaction_failed`:
  without it the body is indistinguishable from any other `502`.
- **D5 — the hold is capped at 10 MiB and fails open.** Past the cap it releases and the stream
  continues as a plain passthrough. *Why:* the adapters on this path serve reasoning models, so the
  "preamble" is not a few events but the whole thinking phase — the downstream cost in the fourth
  point above includes it, and the user watches Claude Code's spinner rather than live thinking.
  An uncapped hold would make that memory unbounded, against every other bound the bridge keeps
  (F24's line cap, the auditor's). The price of the cap is stated rather than discovered: a
  thinking-only reply longer than 10 MiB cannot be retried. Worst-case memory per concurrent
  native request is therefore about 10 MiB for the hold, plus the auditor's own 10 MiB line bound,
  plus a set of at most 256 thinking-block indices (kilobytes). The cap is checked per upstream
  chunk — aiohttp's `StreamReader.__aiter__` is `AsyncStreamIterator(self.readline)`, one line at a
  time, and `readuntil` raises `LineTooLong` past the reader's high-water mark — so it can be
  overshot by at most that one line, and releasing copies the buffer once. The index set fails
  open past its bound, mirroring the auditor's bound on open `tool_use` blocks.
- **D6 — an empty text chunk is not content.** A `text_delta` whose `text` is `""` releases
  nothing. *Why:* the literal rule would release a reply that opens a text block, streams `""`
  and stops — a blank turn, the defect this ticket exists to fix — while D1 already declines to
  count a text block that *starts* empty.
- **D7 — a stall while held takes the ordinary pre-emission ladder; the cost above is corrected,
  not the policy.** The fourth point above says a stalled stream sends nothing "until
  `_STREAM_READ_TIMEOUT`" and then errors. That understates it: `asyncio.TimeoutError` is
  retryable and not a transport error, so each timeout while held is retried — failed over, with
  the backend quarantined, on a balancing profile — up to the whole attempt budget. On one
  backend the worst case is six 120-second reads plus the backoff and final delays, about
  thirteen minutes with no downstream byte. *Why keep it:* it is exactly the policy the bridge
  already applies to a provider that never answers at all, which is what Q14 meant by "the
  ordinary pre-emission path"; a separate stall rule would make the two cases differ for no
  product reason.
- **Implementation consequences recorded with the decision.** (i) A client that disconnects
  while held is no longer revealed by a failed write, and aiohttp does not cancel the handler, so
  the native branch checks the client connection on each held chunk and before each upstream
  attempt, and stops — leaving backend health alone, as issue #38 requires. Without that check a user who
  presses Esc during a long thinking phase would keep it billing upstream. (ii) Headers go out at
  release, so a discarded empty attempt never names its backend: the attribution headers name
  the backend whose bytes are the first the client receives — which is what the README already
  promises ("the backend that produced the first byte of the response"). (iii) Each discarded attempt is logged at WARNING with
  its held byte count and stop reason, and a bounded head of the held bytes at DEBUG — a `200`
  carrying a non-SSE body is otherwise undiagnosable, because nothing of it was written.
  (iv) The `sr is not None` arm that ends an already-open stream with one SSE `error` event
  instead of a JSON `502` is defence in depth, twice over. It was written for the pre-KBR-183
  route — a timeout *after* release starting another attempt — which KBR-183 closed; KBR-236
  (2026-09-14) closed the other route into it, the translated branch's empty-response retry
  failing over onto a Messages-wire backend (§6.3.1's recovery paragraph records the site).
  Kept rather than deleted because a future route that reached it must not fall through to the
  pre-emission ladder — that would put a second attempt on the open stream, the exact hazard
  the arm exists to stop — and its message now reports an empty response arriving after
  content, not "the retry came back empty": no such retry can exist.

**Extended to the translated Chat Completions route by KBR-235 (owner decision, 2026-09-14).**
The translated branch of `_stream_messages` judged an empty reply only when a chunk carrying
`finish_reason` had arrived, so a `200` whose stream was zero bytes, `[DONE]`-only, or otherwise
content-free with no finish chunk reached Claude Code as an empty turn — no retry, no failover,
and the backend marked healthy. Those streams now take the same ladder as any empty reply, and
**(b)'s preamble hold needs no translated counterpart**: the branch already withholds the finish
events, and the no-finish shapes write nothing, so an empty attempt is still pre-emission when
the gate judges it. The gate's no-finish arm also stands down once the request has written
anything (`sr is not None`): `events_emitted` counts one attempt, while `sr` counts the request,
so a request that has already written is post-emission — Q14(a)'s situation, and KBR-236's to
fix, not the D4 path's. The exhaustion outcome was this ticket's one owner decision, and the product
owner chose **D4, not M12**: a no-finish empty stream that exhausts the ladder ends in the same
`502` `api_error` `reason: "empty_response"` as the native route — a `200` carrying substituted
text is the one thing the route must never produce, and exhausting into an error opens no second
exhaustion vocabulary. Two boundaries recorded with the decision: the contentless-with-
`finish_reason` case keeps its M12 fallback exhaustion, so the route's exhaustion is deliberately
split until the owner unifies it; and the translated route has **no streaming D3** — a
`max_tokens` finish-chunk empty stream is still retried and fallback-ized, the route's
pre-existing finish-chunk behaviour this change deliberately did not touch. The non-streaming
judgement `_is_empty_cc_response` and the four ladders it gates were aligned with D1/D3 for
Messages-shaped replies in the same change (§7.2.1 above): D1 by block type, mirroring
`PreambleHold._block_start_releases`; D3 ends the ladder at once and the Messages handler returns
the `400` with the same body the native branch builds. The branch's tail-flush write path now
sets `events_emitted`, so content arriving in a final unterminated line counts as a write for
both the emptiness gate and FI-8.3's truncation guard. The post-emission retry a content delta
after the finish chunk can still trigger is KBR-236's, untouched here.

**Extended to `/v1/responses` and `/v1/gemini` by KBR-250 (owner decision, 2026-09-14,
per-route confirmation pending).** The no-finish empty-stream hole that KBR-235 closed on
`/v1/messages` had the same shape on `_stream_responses` (Codex CLI) and `_stream_gemini`
(Gemini CLI): the gate at `server.py` ~3766 / ~5677 still read
`translator.response_was_empty and finish_events`, so an upstream that answered `200` with
zero bytes or `[DONE]`-only passed both conditions' guard and reached the agent as an
empty turn with the backend marked healthy. The fix is the same on both routes: a no-finish
arm `empty_no_finish = not finish_events and not events_emitted` extends the gate to
`or empty_no_finish`, and the D4 exhaustion path writes an SSE error event into the open
`sr`, lets the post-loop synthesize `response.completed` with `status="incomplete"`
(responses) or `write_eof()` without the healthy-mark (gemini), and returns `sr` with
`200 text/event-stream`. The route-specific D4 discriminators carry the same semantic
KBR-235 chose for the messages branch: `responses_format_error({"code": "empty_response",
"message": ...}, seq=N)` producing a top-level `{type: "error", code: "empty_response",
...}` on the Responses wire; `{"error": {"code": 502, "message": ..., "reason":
"empty_response"}}` serialized as `data: {json}\n\n` on the Gemini wire. The owner is
asked in the PR to confirm this in-stream shape as the exhaustion form for Codex and
Gemini CLI.

**The post-emission arm and per-request `request_emitted` flag are deliberately NOT
mirrored on these two routes.** `ResponsesTranslator` and `GeminiTranslator` set
`response_was_empty` against the **whole response's accumulated content** (text + tools +
reasoning), not the finish chunk's content alone — the messages branch's
`MessagesTranslator` is chunk-scoped, so its post-emission arm fires there but is
structurally unreachable on responses and gemini. A stream that wrote content then received
a finish chunk with no content has accumulated content → `response_was_empty` is False →
the gate does not fire → no post-emission arm is reachable. The `empty_no_finish` arm
requires `not events_emitted` on the current attempt, which also cannot be true when
content has been written. The two routes therefore carry no `request_emitted` flag and no
post-emission guard; minimum code, no defensive scaffolding for an impossible scenario. The
shape is locked in by `TestResponsesInStreamErrorExhaustion::test_content_then_empty_finish_never_fires_the_empty_gate`
and the gemini mirror in `tests/bridge/test_empty_response_retry.py` — a regression that
makes the gate fire on a content-carrying stream fails them both. If a future translator
change makes the emptiness check chunk-scoped on these routes too, the post-emission arm is
the obvious next step — the same shape KBR-235 added on messages.

**JSON 502 alternative.** The branches `sr.prepare(request)` at the top of each handler,
before any upstream POST, so aiohttp cannot later replace the prepared `StreamResponse`
with a JSON `Response`. Lazy-preparing both branches (mirroring `_ensure_prepared()` on
messages) would carry attribution-header timing and `§11 Q14(ii)` "backend that produced
the first byte" consequences that need their own design pass — out of scope for KBR-250.
The in-stream SSE error event is the chosen shape; the JSON 502 alternative is recorded
here as the route the owner can take if a future decision prefers protocol-identical
behaviour to the messages branch.

**Why, and not the obvious alternative.** Three reasons, in decreasing order of how much they
would cost to be wrong about.

1. **It is what the provider being imitated does.** Verified against the official Anthropic Python
   SDK, `src/anthropic/_streaming.py` (the `sse.event == "error"` branch, sync and async): a
   mid-stream failure arrives as an SSE `error` event and is raised to the caller. No resumption,
   no re-opened block, no second attempt — the HTTP status was already `200` and the stream simply
   ends in an error. The claim this rests on is §1's ordinary-correctness promise that *protocol
   translation is faithful*, together with §6.2.2's SSE grammar — **not I2**, which the correction
   below shows is upstream-side and says nothing about what the client is handed. The client is
   Claude Code, and Claude Code is written against Anthropic's stream shape.
2. **It ratifies a policy already in force.** `bridge/server.py`'s streaming handler already makes
   exactly this choice for a mid-stream transport drop — *"Bytes already reached the client, so a
   restart on any backend would duplicate them. Close the message off instead"* — for exactly this
   reason, and calls it the same choice FI-8.3 makes for a clean truncation. Answering the other
   way would mean **changing working code to introduce a duplication hazard**.

   **It was the transport class only — gap G26, closed by KBR-183 on 2026-09-13.**
   `_is_transport_error` returns `False` for `asyncio.TimeoutError` deliberately, and the failover
   arm that followed carried no emission test, so a mid-stream `sock_read` timeout **after** bytes
   had reached the client marked the backend unhealthy, selected another and wrote a second attempt
   onto the already-prepared response — measured at 2 upstream requests with two backends and 6
   with one. KBR-183 guards that arm and three sites of the same class it found on the way: the
   Responses and Gemini custom-transport failovers, and `openai_subscription`'s in-provider retry
   of a stream reset. The transport drop's own *ending* still differs from (a) — `message_stop`
   rather than an error — and is recorded in §6.3.1 rather than ratified.
3. **The alternative is not soundly implementable.** Re-opening on a second backend lets the client
   receive the same sentence twice, or tool-call arguments spliced from two attempts. §6.3.1 states
   the consequence and it is not hypothetical: every SSE event stays syntactically valid while the
   conversation is corrupt, and Claude Code will act on a duplicated tool call. De-duplicating
   across attempts needs to know what the second backend was about to say.

**One correction to the framing this question was filed under.** KBR-163 argued that buffering the
passthrough "changes downstream latency and the observable timing that invariant **I2**
constrains". It does not: every I2 channel in §4.2 — C1 headers, C2 body, C3 cross-attempt content
and cadence, C4 transport fingerprint, C5 connection lifecycle — is **upstream-side**. What the
downstream client is handed, and when, is invisible to the provider. The only coupling is TCP
backpressure, and holding a bounded preamble makes the bridge read upstream *sooner*, not later,
which is what any promptly-reading client does. The real cost of (b) is downstream
time-to-first-token, which is a user-experience question and bounded by the preamble, not an I2
breach. That is why (b) is affordable and full buffering — unbounded, and growing with stream
length — still is not.

**What this does not decide.** The wording of the terminal error the client receives follows Q9's
precedent (downstream only, names the product); KBR-155 settled it as D4 above. The *zero-chunk*
case needs no separate rule, but the sentence first recorded here — that the ordinary
pre-emission ladder already applied to it — was not what the code did: the native branch set
`stream_ok` unconditionally, so a zero-byte stream skipped the ladder, fell to the handler's
"should not happen" `502` and marked the backend healthy. Under the hold it is simply an empty
reply and takes the ladder like any other.

*Original question:* what is a correct stream recovery after bytes have reached the client
(§6.3.1)? Failover
before the first downstream byte is unambiguous. After text has been emitted, or mid tool-call
arguments, there is no obvious right answer: abandon the partial block and re-open under a new
id, fail the turn and let the agent retry, or something else. Until this is decided the L3 row
can assert only the negatives — no duplicated text, no reused tool-call id across attempts, no
spliced arguments — which catches corruption but cannot confirm correct behaviour. This is the
one place in the design where a test is specified without a full acceptance oracle, and it is
recorded here rather than papered over.

**Q15 — ANSWERED by the product owner, 2026-09-14 (KBR-205).** §3.3.1's reader-side
declared-ignored mechanism is built, and the first five users are Anthropic block fields; the
two Anthropic tool-control fields named by KBR-214 are **mapped** rather than ignored. Recorded
in §3.3.1, §3.3.1b, §7.4.1, G35 and G36 in §9.2; the answer to each half:

1. **A field the grammar cannot carry is declared-ignored, not slotted, when a strip would not
   hide a cost.** §3.3.1 offers "map it, or declare it ignored with a reason". A slot is the right
   answer when a strip would have hidden a mutation the user pays for — `cache_control` is the
   worked example, because a stripped breakpoint re-bills the agent's cached prefix at **at
   least** ten times the cached rate. The five Anthropic fields — `text.citations`,
   `image.transformations`, `tool_use.caller`, `tool_use.toolset_name` and
   `tool_result.toolset_name` — are optional, descriptive, vendor-defined fields the agent
   neither reads nor writes; a strip would have hidden no cost, only a description the reader was
   never going to model anyway. Slots for five vendor spellings would also have put one vendor's
   names into a form whose entire purpose is wire independence, which §3.3.1 declines for P16's
   reason. The mechanism — path-keyed by `(block wire type, field wire key)`, reason-required per
   entry, defined once for all six readers in `tests/harness/contract.py` — is the right home for
   this class of field, and the audit trail every entry carries is also the coordination guard
   the seven-reader shape needs.
2. **`tool_choice.disable_parallel_tool_use` and `tools[i].type` are mapped, not ignored.**
   KBR-214's design review of KBR-205 named these as defects of the same shape (they were the
   cause the ticket expanded), and the same reasoning applies in reverse: a mutation on either
   changes what the agent asked for, or what the product decides about the request. Mapping them
   keeps that mutation detectable — declaring either ignored would have hidden the very
   fidelity finding the oracle exists to catch. The parallel knob gets one address and one
   polarity at `envelope.extra["parallel_tool_calls"]` (§3.3.1b), the Chat Completions spelling
   so the T-A2 reader meets the same rule; `tools[i].type` becomes `ToolDecl.type`
   (§3.3.1a), the discriminator the G35 register row needs to author.

**Scope note for KBR-34.** T-A2 is the first to **read** the canonical
`envelope.extra["parallel_tool_calls"]` directly — the Anthropic reader
writes it (from Anthropic's inverted `disable_parallel_tool_use` flag); the
Chat Completions reader carries it (the wire key is the canonical spelling,
routed through `PARALLEL_TOOL_CALLS_KEY` so the two spellings cannot drift).
A wrongly-typed value at the canonical address residualises at its own path,
the same shape §7.4.1's wrongly-typed-leaf rule gives every other optional
leaf — and is asserted in `test_reader_chat_completions.TestEnvelope`.
T-A3 (Responses) and T-A4 (Gemini) carry the same address when their
readers exist; the closed canonical spelling is the anchor.

**What this does not settle.** Two things, named so a future task does not inherit them as
surprises.

- **Per-reader entries the registry does not yet carry.** Five readers do not exist yet (T-A2
  through T-A6). The mechanism is shared, but the entries those readers will need — Converse's
  `cachePoint` discriminator, Gemini's ProtoJSON-only fields, Chat Completions' `prompt_cache_*`
  family, Responses' `reasoning`/`include` controls, Ollama's tool-shape extras — are author
  decisions the relevant tickets will make, with the registry as the agreed place to put them.
  A field that is *not* in the registry and is not on a modelled block residualises today, so
  nothing fails silently in the meantime.
- **The falsification case for the unregistered-mutation guard.** §1.4 requires a mechanism to
  ship with a falsification case proving it can fail. `test_reader_anthropic_messages.py` now
  asserts (i) a body carrying each declared-ignored field projects without residual, (ii) a
  re-spelled sibling still residualises and fails `verify_total`, and (iii) the
  `ignored_field_problems` reporter rejects a registry entry without a reason. A future reader
  that wants to declare-ignore a field that one of its siblings does not carries the same
  falsification cost.

*Original question:* when the projection grammar cannot carry a wire field, declare it ignored
or give it a slot (§3.3.1, §7.4.1)? The decision turns on whether a strip would hide a cost —
`cache_control` cost re-bills the cached prefix, so it got a slot; five Anthropic block-extras
carry no comparable cost, so they got the third outcome; two Anthropic tool-control fields
carry information a mutation could corrupt, so they got a mapping onto a shared canonical
address.

**Q16 — ANSWERED by the product owner's delegation to T-A2, 2026-09-14 (KBR-34).** The Chat
Completions reader fills `Part.cache_control` from **both** spellings — `cache_control`
(OpenRouter's CC dialect, Anthropic's own field) and `prompt_cache_breakpoint`
(OpenAI's GPT-5.6+ spelling, `{"mode": "explicit"}`, no per-part TTL) — **verbatim**. The
value shape travels because the slot is `Mapping[str, Any] | None`: the two spellings are
distinguishable on the slot by their own keys, and a future vendor addition to either shape
(a `ttl` on `prompt_cache_breakpoint`, an extra mode value) arrives as a projection delta
rather than as a silently-flattened loss.

**Why not normalise the two into one shape.** §3.3.1's "carried whole, not reduced" rule
applies to a spelling with no TTL the same way it applies to one with: a flattened form would
make a vendor's future TTL invisible (a 1-hour write costs 2x base input against 1.25x for
the default five-minute one, so a silently-downgraded TTL is a cost the user bears —
§3.3.1's `cache_control` rationale verbatim, applied to the slot's own carrier); a
normalised form would put a vendor spelling into a wire-independent value, which the
whole-slot rule exists to prevent.

**Why the same slot, and not a second one.** §3.3.1's slot is the canonical address for
"the cache breakpoint this block carries"; the wire's spelling is a property of the
dialect, not of the concept. Two slots would put the dialect into the wire-independent
form (§3.3.1's P16 objection), and the two spellings cannot co-occur on one block (the
schema forbids it; if one ever appears, the first non-null spelling in
`_CACHE_KEYS` order — `cache_control` first — fills the slot and the other
residualises at its own path; a silent drop is the shape M16 and G37 exist
to prevent, asserted in `test_reader_chat_completions.TestCacheBreakpoints.
test_when_both_cache_spellings_are_present_first_fills_the_slot_and_second_residualises`).

**Converse does not inherit this.** Bedrock Converse's `cachePoint` is a *separate block*
in the content list, not a field on one (§3.3.1 records the shape); a T-A5 reader that
meets it projects the `cachePoint` block itself as `Opaque(kind="cache_point")` and does
not fill the `cache_control` slot. §11 Q16 is the answer T-A2 shipped; T-A5's is its own
decision and belongs in a §11 entry of its own.

*Original question:* does the Chat Completions reader fill the `cache_control` slot from
`cache_control`, from `prompt_cache_breakpoint`, or leave such bodies residualising
(§3.3.1, KBR-199's comment)? T-A2 filled it from both, verbatim, and residualised a
wrongly-typed value at its own path — the answer §3.3.1's own slot rationale points to,
and the one that makes G37's closure a data point rather than a wording change.

