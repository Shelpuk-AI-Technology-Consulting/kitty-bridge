# Rule: the local bridge and protocol translation (`bridge/**`, `bridge_runner.py`, `cloudflare.py`)

This is the component the whole product is named after: a local HTTP server that
receives the agent's native request, translates it to the provider's dialect,
forwards it, and translates the response back. Everything the agent says and
everything the model says crosses this code.

## The invariant to defend hardest: transparency

**Kitty must not change the agent's messages unless translation makes a change
unavoidable, and the upstream provider must not be able to tell kitty is there.**

That is the product promise README makes — *"Just a bridge … It only proxies and
translates traffic"*, *"Kitty does not record your prompts"* — and it is the one
a reviewer has to hold, because nothing else does.

Concretely, treat each of these as a **critical** finding:

- **Content the agent did not send.** A system prompt, a preamble, an appended
  instruction, a reformatted tool description, a stripped field. If a diff adds
  text to a request body, the question is not "is it helpful" but "did the agent
  ask for it".
- **Content the model sent that the agent does not receive.** A dropped block, a
  swallowed `thinking` segment, a truncated tool call, a coalesced stream event
  that loses a boundary. Silent loss here looks to the user like the model being
  worse, and it is untraceable from the agent's side.
- **A header, user-agent, or parameter that identifies kitty to the provider.**
  The upstream must see traffic shaped like the agent's own. Attribution headers
  that already exist are the deliberate exception; adding a new identifying
  field, or widening an existing one, is a finding. Quote the line.
- **Anything written to disk that was not asked for.** Debug and usage logging
  are opt-in flags with documented paths. A new write, a new default-on log, or
  a broadened log level that starts recording request bodies breaks *"Does Kitty
  record my prompts?  No."*

A change that is a genuine, unavoidable consequence of translation is fine — say
so and move on. What is not fine is one that is *convenient*.

## Translation correctness

Three protocols are translated here (Anthropic Messages, OpenAI Responses,
Gemini), each with a sync and a streaming path, against one Chat Completions
target. The failure modes that actually occur:

- **The streaming path diverging from the sync path.** A field mapped in one and
  not the other. Ask, for every mapping the diff touches, whether its twin moved.
- **Event ordering and terminal events.** A stream that ends without its terminal
  event leaves the agent waiting; one that emits a terminal event early truncates
  the answer. Both look like a hang or a short response, not like a bug here.
- **Line buffering over SSE.** A chunk boundary is not a line boundary. A parser
  that assumes it drops or splits events under load and not in tests.
- **Tool-use round-trips.** A tool call translated out and its result translated
  back must reconstruct exactly what the agent expects to match against. The
  tool-use auditor exists for this; weakening it to make a diff pass is critical.
- **Thinking/reasoning blocks.** They round-trip through providers that do not
  have the concept. Repair and failover paths for these already exist and have
  tests; a change that shortcuts one needs to say why.

## Errors, retries and health

- **An upstream error must reach the agent in the agent's own error envelope**,
  not as a bridge-shaped error the agent cannot parse. Each protocol has its own
  shape; check the diff did not collapse them.
- **A retry must not amplify.** Look for a retry loop that can outlive the
  agent's own timeout, one that retries a non-retryable status, and one whose
  budget is per-attempt rather than per-request.
- **The circuit breaker and health state are shared mutable state** across
  concurrent requests. A change to either needs to be safe under concurrency, and
  "the tests pass" is not evidence of that — say when you cannot tell.
- A backend marked unhealthy must be able to become healthy again. A path that
  can only ever remove capacity wedges the bridge until restart.

## Resource lifecycle

The bridge is `async` throughout and holds sockets to both sides. Look for an
acquire with no release on the exception path, a response body not drained, a
client disconnect that leaks the upstream request, and an unbounded `gather`.

## Layering

`pyproject.toml` forbids `kitty.bridge` from importing `kitty.cli`,
`kitty.launchers`, `kitty.profiles`, `kitty.credentials` or `kitty.tui`. The
bridge receives what it needs as arguments. A new import that reaches sideways or
up is a finding even though `lint-imports` will also catch it — say which
contract it breaks.

## Severity

- **Critical** — any transparency breach above; a translation defect that loses
  or corrupts agent or model content; a leaked connection or wedged health state;
  a retry that amplifies a failure.
- **Warning** — a mapping changed on one path and not its twin; an error shape
  that degrades; a concurrency question you could not settle by reading.
- **Suggestion** — naming, structure, a comment that would save the next reader
  re-deriving an event ordering.
