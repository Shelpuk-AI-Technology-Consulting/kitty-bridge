---
id: kbr285_widen_cc_content_classifier
depends_on: [KBR-276, KBR-277]
---

# KBR-285 — Widen the CC content classifier (and non-streaming detector) to count refusal / legacy `function_call` / list content

Jira: [KBR-285](https://shelpuk.atlassian.net/browse/KBR-285). Parent: KBR-136.
Requirements: `.requirements/20260918T204318Z_kbr285_widen_cc_content_classifier/REQUIREMENTS.md`.
Design: `SYSTEM_DESIGN.md` §5.4 (the known-limit clause this step retires).

Depends on:

* **KBR-276** (Done, PR #221) — widened the pre-emission hold to raw-CC upstreams, which made
  the classifier's narrow three-shape content set binding for raw-CC chunks (the refusal-only
  regression this step fixes) and filed this ticket.
* **KBR-277** (Done, PR #220) — established the byte-for-byte mirror between
  `_cc_chunk_carries_content`'s last clause and `_is_empty_cc_response`'s CC arm; this step
  extends that mirror to the three new clauses.

## What

Extend both predicates — the streaming hold's release predicate
(`_cc_chunk_carries_content`, `server.py:1627`) and the non-streaming emptiness oracle's
Chat Completions arm (`BridgeServer._is_empty_cc_response`, `server.py:2622`) — from three
content shapes to six: the existing non-empty string `content`, non-empty `tool_calls` list,
non-empty string `reasoning_content`, plus non-empty string `refusal`, truthy dict legacy
`function_call`, and non-empty list `content` (multimodal parts). **Second half (owner
decision 2026-09-18, "fix both halves"):** extend `MessagesTranslator` so the translated
`/v1/messages` route carries the three shapes the widening releases — stream direction:
list `content` coerced via `_extract_text_content`, `refusal` emitted as text, legacy
`function_call` mapped onto the `tool_calls` machinery; reply direction: legacy
`function_call` mapped to one `tool_use` block. Both predicates move into the
mutation-measured scope (new registry group, pragmas removed, guard generalised,
baseline row recorded). §5.4's known-limit clause is retired; the handler comment and the
README empty-reply FAQ follow the widened set.

## Why

KBR-276's hold made the narrow set *binding*: an OpenAI refusal-only completion (a normal
moderation-path shape, `delta.refusal` with `content` null) is now held pre-emission, the
empty ladder fires, and the user waits ~80 s to receive an `empty_response` terminal instead
of the refusal text that reached them verbatim pre-KBR-276. Legacy dict `function_call` and
list-typed multimodal `content` misclassify the same way. Widening is a product decision on
what "content" means for every Chat Completions route — the same class of trade KBR-248
deferred and KBR-276 made for the gate; this ticket makes it for the classifier, matching the
KBR-277 precedent (the detector mirrors the classifier).

**Why mirrored clauses, not one shared function:** the recorded KBR-277 decision — the two
predicates read different wire shapes (`delta` vs `message`) with the documented `.strip()`
vs `!= ""` drift on the string-`content` clause that aligning would change product behaviour;
side-by-side mutation testing is the divergence guard.

**Accepted cost, recorded:** a whitespace-only `refusal` counts as content on both sides
(`!= ""` spelling — no new drift minted). The two halves are one fix: without the
translator extension the widening would regress the translated stream cells from "honest
retry terminal" to "silent empty turn / non-string on the wire", so the translator work is
not a separate ticket.
