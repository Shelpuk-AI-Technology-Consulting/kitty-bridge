---
id: kbr287_custom_transport_empty_hold
depends_on: [KBR-248, KBR-254, KBR-276]
---

# KBR-287 — Empty-completion hold on the `_stream_chat_completions` `use_custom_transport` branch

## What

`BridgeServer._stream_chat_completions` (`src/kitty/bridge/server.py`; the
custom-transport segment inside the KBR-254 dispatch loop — grep anchor:
`# Custom-transport providers return Responses API SSE but CC clients`) gets
the same pre-emission discipline KBR-276 put on the plain-POST segment (grep
anchor: `KBR-248: pre-emission hold, widened to the whole route by`). The
branch builds the synthesised chunk list — role chunk, content delta?,
tool-call arg deltas?, finish chunk, `[DONE]` — and applies one up-front
verdict through the shared predicate `_cc_chunk_carries_content` (grep anchor:
`def _cc_chunk_carries_content`): content-bearing → write every line in
synthesis order (byte-identical wire output); content-free → write nothing,
the attempt stays pre-emission.

The empty ladder mirrors the plain-POST arm (grep anchor: `Check for empty
response (pass-through: no content bytes written)`): one **class-agnostic
`_select_backend()`** — selected provider custom → re-normalise + refresh
`_resolved_key`/`_provider_config` + `continue` (stay in branch); selected
provider plain → pop the three custom keys + `break` (the branch's
fall-through idiom — grep anchor: `Cross-mode failover: entering standard
streaming path`); pool-less `elif` → exponential backoff retry; exhaustion →
the route's `empty_response` D4 terminal (`_NATIVE_EMPTY_REPLY_MESSAGE` +
`type: "empty_response"` + `[DONE]`), uniform with the plain-POST twin. The
attempt loop widens to `range(n_backends + len(_EMPTY_FINAL_DELAYS))` with
the plain-POST final-delay prologue for attempts ≥ `n_backends`; the
exception path's `attempt < n_backends - 1` gate keeps its meaning.

## Why

The same defect class KBR-248 closed on the converted route and KBR-276
closed on the raw-CC route survives one branch over: the `use_custom_transport`
segment the KBR-254 cross-class re-dispatch routes into. A content-less
completion from such a backend currently delivers a well-formed Chat
Completions skeleton (role chunk → finish → `[DONE]`) and the empty-response
ladder cannot fire, so the client sees a successful turn with nothing in it,
the backend is marked healthy, and a balancing pool keeps routing to the
broken upstream. KBR-276 deliberately recorded this branch as a scope-out
(SYSTEM_DESIGN.md §5.4 KBR-276 paragraph, "Scope-out, deliberate"); KBR-287
retires that scope-out. Together KBR-248/276/287 cover every Chat Completions
emission path on the route.

Four decisions the review settled (2026-09-18), each against a plausible
alternative:

- **Judge-first, not an incremental hold-walk.** The plain-POST hold buffers
  because a streaming branch does not know the future when the first line
  arrives; this branch parses the entire upstream response before emitting
  anything, so there is no unknown future to buffer against. An up-front
  `any(_cc_chunk_carries_content(payload) ...)` verdict produces the same
  wire bytes with no `held`/`held_bytes` state, and the D5 `MAX_HELD_BYTES`
  cap is satisfied by construction (the only holdable lines are the
  synthesised role/finish/`[DONE]`, tiny against 10 MiB — a cap here would
  be dead policy code). This deliberately simplifies the ticket mechanism
  wording, which assumed the plain-POST line-arrival shape.
- **`empty_response`, not the ticket's `cross_class_exhaustion`.** The
  ticket's acceptance named the cap-hit terminal; the design review showed
  reusing it breaks §5.3 S8's promise that a client branching on the route's
  `type` can tell the crossing-cap hit (pathological ping-pong,
  configuration problem) apart from `empty_response` (upstream returned
  nothing, transient), and would split one failure into two discriminators
  by pool composition. `empty_response` + `_NATIVE_EMPTY_REPLY_MESSAGE` is
  route-uniform with the plain twin, needs no §5.3 S8 amendment, and still
  satisfies the ticket's core intent (a defined D4 terminal, not a bare
  empty 200). Flagged to the owner in the PR and the Jira comment as a
  deliberate correction.
- **One class-agnostic select, not a custom-first tier pair.** A
  custom-first ordering would class-lock the ladder: empties never mark a
  backend unhealthy, so a mixed pool [custom-empty, plain-good] would spend
  every attempt re-selecting among customs and never try the plain backend —
  worst exactly in the cross-class scenario this ticket exists for. The
  plain-POST idiom (select any healthy backend; cross or stay by what
  lands) restores symmetry and makes the crossing testable. The
  custom→plain crossing stays uncapped like the exception path's; termination
  is bounded by the reverse direction (only plain→custom crossings `continue`
  the dispatch loop, capped at `(2 * n_backends) + 1`; a custom→plain
  fall-through happens at most once per pass).
- **Attempt bound `n_backends + len(_EMPTY_FINAL_DELAYS)`, not the plain-POST
  `(_MAX_RETRIES + 1) * n_backends + len(...)`.** The plain-POST bound bakes
  in that branch's transport-error ladder (6 attempts on a single-backend
  pool with `_MAX_RETRIES = 3`); the custom branch's transport errors ladder
  within `n_backends` via its own exception path, so its "original" budget is
  the failover walk and only the empty ladder extends it (3 attempts,
  single-backend, all against the same provider — empties never mark a
  backend unhealthy).

Two structural facts the next reader needs: the synthesis projects only
`content` and `tool_calls` (neither parser surfaces `reasoning_content`), so
a reasoning-only completion synthesises the empty shape and ladders — the
projection gap is pre-existing and not this ticket's to close; and of
KBR-285's widening set only the **list-content** clause is reachable here
(refusal and legacy `function_call` are never projected). Usage logging is
log-on-release: the content path logs exactly as today (including on client
disconnect — the branch parses atomically, so usage is fully known
regardless of client state), a discarded attempt logs nothing.

## Plan

Implementation steps live in
`.requirements/20260918T210851Z_kbr287_custom_transport_empty_hold/REQUIREMENTS.md`
(Testing Plan + Implementation Plan). Summary: harness + AC-1 watch fail;
judge-first verdict + gated writes; ladder arms + exhaustion terminal;
byte-identity / tool-call / reasoning-only tests; mixed-pool crossing test;
SYSTEM_DESIGN.md §5.4 update (retire the KBR-276 scope-out, record this
paragraph's decisions) + TEST_SUITE.md amendment; full suite + CI.
