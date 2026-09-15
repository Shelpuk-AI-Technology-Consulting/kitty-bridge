# The golden Claude Code transcript corpus

`.system_design/TEST_SUITE.md` §7.1 · plan task **T-W6** (KBR-29).

Real `POST /v1/messages` requests captured from actual Claude Code sessions, committed here as
fixtures. They are the **inbound** half of every fidelity comparison: the oracle (§3.3) projects
what Claude Code sent and what reached the provider, and asserts the two agree except for the
mutations on the register.

**Why real rather than hand-written.** A hand-written fixture encodes our belief about what Claude
Code sends. That belief is the thing most likely to be wrong, and it drifts every time Anthropic
ships a release. A captured body is evidence.

---

## Refresh — owner and cadence

> **Owner:** the repository maintainer (Sergii Shelpuk).
> **Cadence:** re-capture when the pinned Claude Code version changes.

Decided by the product owner on 2026-09-12. A version bump is the only event that can invalidate a
capture, so it is the only cadence that is neither late nor wasted: a calendar cadence misses a
release that lands the week after it runs, and re-capturing on a quarter where nothing shipped is
work for its own sake. Refresh-on-demand was rejected outright — it is precisely the museum §7.1
warns about, where a stale fixture passes happily while the protocol has moved on.

This is why every captured entry records `captured_from`, the Claude Code version it came from. A
cadence tied to a version bump is unactionable if the entries do not say which version they are;
the loader therefore **requires** the field on a captured entry rather than accepting it as
decoration.

---

## The format

One entry is **two files**:

| File | Holds |
|---|---|
| `<id>.json` | The manifest — everything about the request except its body |
| `<id>.body` | The raw request body, byte for byte as it arrived |

**Why the body is a sidecar and not a string inside the manifest.** Byte-exactness is the
argument that comes to mind first and it is the weakest one — JSON string escaping is reversible,
so a single-file manifest with an escaped body is byte-exact too. Two others actually decide it:

1. **Review.** §7.1 makes human review before commit mandatory. A 50,000-character body escaped
   into a one-line JSON string is not reviewable in a diff, and the entries that cross the
   compaction budget are far larger than that.
2. **The body must stay greppable as plain text.** Not only for the CI lint, but for GitHub's own
   push protection, which is the last line of defence when the scrubber misses something. Base64
   would be byte-exact and would remove that safety net — it defeats every scanner, ours and
   GitHub's alike. That is the reason not to "simplify" this to one file later.

**The committed body is byte-exact as *committed*, not as it left the agent.** Scrubbing is the
one bounded departure from the wire bytes, and it is safe for I1 because the oracle diffs two
projections of the *same* scrubbed input. The two places it is not free are named below: the
size-derived triggers, and `content-length`.

### Manifest keys

Every key is required to be present and no other key is accepted — a typo such as `trigers_met`
must fail the load, not silently mean "declares nothing".

| Key | Meaning |
|---|---|
| `id` | The entry's name. Must equal the filename stem |
| `description` | One line: what this entry exercises |
| `origin` | `captured` or `synthetic` |
| `origin_note` | Required when `synthetic`: why this could not be captured |
| `captured_from` | Required when `captured`: the Claude Code version, e.g. `claude-code/1.2.3` |
| `captured_at` | Required when `captured`: the ISO date of capture |
| `method`, `scheme`, `host`, `path`, `query` | The request line, query raw and unreordered |
| `headers` | Ordered `[name, value]` pairs — original casing and duplicates preserved, because §4.3 C1 asserts on the exact header set |
| `body_file` | The sidecar's filename |
| `body_sha256` | The sidecar's digest. `load_entry` verifies it — see *Byte-exactness* below |
| `wire_format` | The format the **inbound** body is written in, or `null` for a body no reader can classify. §7.4's oracle takes an `inbound_format`, and inbound is not always Anthropic Messages |
| `known_non_secrets` | `[literal, reason]` pairs a reviewer has cleared — see *What the scrubber removes* |
| `triggers_met` | Register triggers this entry, as an inbound request, establishes |
| `triggers_absent` | Register triggers this entry provably does **not** establish |

### Byte-exactness is enforced, not hoped for

`body_sha256` is checked on every load. The threat is not malice, it is
`core.autocrlf=true`: a contributor on Windows rewrites LF to CRLF inside a `.body` on checkout
and commits the result back, which changes both the bytes §3.3.2 asserts on and the length two
register triggers are decided by. The Windows CI leg (KBR-164) is not a substitute: it catches a
body rewritten while its manifest digest stays put, but a rewrite that updates **both** passes
everywhere. `.gitattributes` marks these paths `-text` so the rewrite never happens; the digest
catches it if that file is ever dropped or its patterns stop matching.

### Triggers have three states, not two

A trigger named in neither list means **this entry says nothing about it**. That third state is
deliberate and load-bearing.

§3.3.2 assertion 2 — "no mutation without its trigger" — is what stops M3 and M5 quietly becoming
unconditional, and it is only as good as the complement it runs against. If absence meant "not
met", every entry whose author never considered trigger *T* would be silently offered as *T*'s
complement, and the assertion would be quantified over entries nobody vetted. A complement must be
claimed, not inferred.

**A declaration is about the committed artifact, never the capture.** Scrubbing shrinks a body —
every home path and e-mail address in it gets shorter — so a tool result captured at 50,010
characters can cross back under M3's 50,000-character threshold on its way into the repository.
Declare triggers after `write_entry`, by reading what landed.

**And it is a claim about the entry *under the profile the test resolves*.** `M5`'s compaction
budget is derived from the profile's model (and on a balancing profile from the smallest context
in the pool), so one transcript is a trigger case on a 200K-context model and a complement on a
1M one. `M4` is worse: its trigger is a pipeline state, not a request property at all. `M1`
(`PROFILE_SETS_MODEL`) is profile-shaped too — the agent's model is the profile's. All three
are declared at the call site that resolves the profile, not in the manifest.

It also keeps entries honest about what they cannot know. `register.py` classifies every
non-`ALWAYS` trigger by how it is decided — one of four `ArrangingBy` kinds
(`REQUEST`/`ROUTE`/`RESPONSE`/`PROFILE`); only `REQUEST` can be varied by a corpus entry,
which is the kind the corpus is *for*. `RESPONSE` triggers (`M6`'s upstream 400, `M8`'s
rejected thinking round trip, `M9`'s upstream tool-use format error, `M12`'s empty upstream
response, `M17`'s rejected thinking signature) are properties of the upstream response,
arranged by a scripted recorder. `ROUTE` triggers (`M2`/`M16` non-native upstream wire, `M10`
Gemini protocol, `P13` CC-origin path) and `PROFILE` triggers (`M1` profile sets model, `M4`
compaction ran with oversized tool result, `M5` over compaction budget, `P9g` non-Entra
credential, `P9d` ChatGPT account id present) are decided at the route or the profile. The loader refuses all of them — `ALWAYS` and every non-`REQUEST`
trigger — in both lists. §9.2's gap **G21** is the reason — an over-declaring entry makes the
oracle's first assertion claim every delta and pass over a broken bridge, and "the same author
writes the entry and its trigger index, so the mechanism has no second reader". This is that
second reader, and the set it refuses is *derived* from the register's classification
(KBR-186) rather than hand-listed — so the two cannot drift.

---

## Capture procedure

### Inbound requests (T-C1–T-C6)

Point Claude Code at the recorder **directly, not through the bridge**. What the corpus needs is
what Claude Code sends; a request that has been through the bridge has already been mutated, and
comparing it against itself would prove only self-consistency.

1. Start `harness.recorder.RecordingUpstream` on a loopback port, serving
   `WireFormat.ANTHROPIC_MESSAGES` and scripted with whatever replies the shape you want requires —
   a `tool_use` reply to provoke the next turn's `tool_result`, a thinking block to provoke a
   thinking round trip, a large tool result to cross a compaction threshold.
2. Launch Claude Code against it:

   ```
   ANTHROPIC_BASE_URL=http://127.0.0.1:<port>
   ANTHROPIC_API_KEY=<a throwaway value — never a real key>
   ANTHROPIC_AUTH_TOKEN=kitty-bridge-token
   ```

3. Drive the session until the recorder has captured the request you are after.
4. Write each entry with `harness.corpus.write_entry(...)`, passing `extra=` with your username,
   hostname and anything else no pattern can recognise (each at least four characters — an
   unbounded substring replacement of a short literal shreds the body wherever those characters
   occur). `write_entry` scrubs on the way out, so the procedure cannot produce an unscrubbed
   fixture by forgetting a step.

   It **refuses** a capture carrying `content-encoding` or `transfer-encoding`. The corpus stores
   entity bodies, not wire octets: a compressed body is one the scrubber reads as noise and reports
   clean — a false clean no plaintext test can ever detect. If you hit this, capture the decoded
   body; the recorder already hands you one.
5. **Read every `.body` file.** The scrubber finds shaped secrets; it cannot find a password that
   looks like a word, a client name, or an unreleased feature discussed in a prompt.
6. Commit. The L2 lint (`tests/harness/test_corpus_lint.py`) gates the result.

**Do not capture from the bridge's debug log.** `server.py` logs the inbound body through
`json.dumps(body, indent=2)` — a re-serialisation. The bytes it prints are not the bytes that
arrived, which defeats the byte-level assertion the corpus exists to support.

### The native baseline (T-C7)

§7.1 also needs the headers and connection pattern Claude Code produces when it talks to
`api.anthropic.com` **directly** — the baselines design channels C1b and C5 compare against.

**This format carries half of that, deliberately.** The *header* baseline is a single request and
is an ordinary entry whose `host` is the real one. The *connection pattern* is not: C5 counts
distinct TCP connections across an N-turn session, which needs session grouping, ordering, and the
`arrival` and `peer_port` fields this manifest does not carry. That is a different artifact with a
different shape, and it is **T-C7's to define** — do not stretch this format to hold it. The
capture itself terminates TLS, which is also T-C7's problem.

---

## What the scrubber removes, and what it cannot

Scrub policy set by the product owner on 2026-09-12: **credentials and personal identifiers**.
File contents and prompts are *not* synthesised — that would destroy the reason for capturing them.

Scanned: the body, every header value, the query string, **and the host and path**. The routing
fields are scanned like everything else — a path of `/v1/key/<token>/messages` would otherwise
commit the token, and an internal hostname has no shape, so `extra` is the only thing that can
reach it. The patterns are shape-anchored precisely so a real route survives: an Azure deployment
segment and a Vertex `projects/…/locations/…` path are evidence §3.3.5 asserts on, and are pinned
by test against this.

Removed automatically, each replaced by a class-named placeholder:

| Class | Shape |
|---|---|
| `anthropic_key` | `sk-ant-…` |
| `openai_key` | `sk-…` |
| `gcp_key` | `AIza…` |
| `aws_key_id` | `AKIA…`, `ASIA…`, `A3T…`, `ABIA…`, `ACCA…` |
| `github_token` | `ghp_…`, `gho_…`, `ghu_…`, `ghs_…`, `ghr_…`, `github_pat_…` |
| `jwt` | `eyJ….….…` |
| `private_key` | The **whole** private-key block — header, key material and footer, including `OPENSSH`, `ENCRYPTED` and PGP's `PRIVATE KEY BLOCK`. A key truncated when the agent read it (no END line) is redacted to the end of the field. Public `CERTIFICATE` blocks are kept |
| `bearer_token` | `Bearer <opaque>` appearing in a **body** (a curl command in a prompt, an HAR file, a log the agent read) |
| `assigned_secret` | An `api_key`/`secret`/`token`/`password` assignment with a long value |
| `email` | An e-mail address. Excludes the retina-asset shape `name@2x.png` — by that **shape**, deliberately, not by extension: `.py` is Paraguay's ccTLD and `.md` is Moldova's, so an extension list stops scrubbing `admin@empresa.py` |
| `home_path` | The username in `/home/<user>/`, `/Users/<user>/`, `C:\Users\<user>\` |
| `literal` | Anything passed in `extra=` |

Credential **header** and **query-parameter** names are not listed again here: they come from
`harness.contract.REDACTED_HEADERS` and `REDACTED_QUERY_KEYS`, which already own that vocabulary.
A second spelling is a second thing to forget to update.

**The manifest's prose is checked but never rewritten.** `description` and `origin_note` are
scanned by the lint and reported — they are where a maintainer is most likely to write the thing
this policy exists to keep out ("captured on *<internal host>*", "the customer's key was in this
one"). They are not scrubbed automatically: silently mangling a description makes the entry harder
to review rather than safer, and the person who wrote the sentence is the right person to fix it.

**What it over-removes.** A credential's alphabet includes `/`, `.` and `=`, so a token written
flush against a path takes the path with it: `token=ZZZ…/v1/messages` redacts the path segment too.
The redaction errs safe but deletes evidence the corpus exists to preserve, so prefer a capture
where the value is quoted or newline-terminated — which is what a real transcript gives you.

**What it cannot find**, and why the review step in the procedure is mandatory rather than advisory:

- a secret with no shape — `PASSWORD=hunter2`, an internal hostname, a customer name;
- a secret this project has never seen — a provider whose key format is not in the table above;
- anything sensitive that is not a secret at all: an unreleased product discussed in a prompt, a
  file path that reveals a client, a code comment naming a person;
- **two token-shaped credentials concatenated with no separator at all** — `AKIA…34AKIA…34`, or an
  API key immediately followed by `Bearer …`. Every token pattern is anchored on a word boundary, and
  there is no boundary between the two, so neither rule's anchor holds; where a rule's tail runs on,
  the first swallows the start of the second and the remainder has no shape. The anchors are not
  negotiable — without them `disk-usage-monitoring-service.py` is redacted as an OpenAI key. A real
  capture is JSON with delimited strings, so this shape does not arise from one; if you ever see two
  credentials run together in a body, split them by hand before `write_entry`.

  The limitation is exactly that and no wider, and a test holds it there: any secret separated from
  its neighbour by a single space or comma is removed, and **a private key is removed at any
  separation, including none** — its rule runs first and anchors on its own delimiters.

The scrubber is a net under the review, not a replacement for it. A fixture file is as public as
the repository.

### If a credential reaches a commit

**Rotate it.** Immediately, before anything else. Rewriting history is not the remedy and should
not be attempted first: the value may already be in a fork, a CI log, a cache or a mirror, and
every minute spent on `git filter-repo` is a minute the credential is still live. Scrub the tree
afterwards, in an ordinary commit.

This is also why no finding, assertion message or `repr` this module produces ever contains a
matched value — not even a prefix. Findings carry a class and a byte offset. A message that
quoted the run it matched would turn a contained authoring mistake into a published one the moment
CI logged it.

### Two things this format does not decide

**Entry size.** T-C3's over-budget transcript is ~2.8 MB (the compaction threshold is 70% of a
4,000,000-character ceiling). Nobody reviews 2.8 MB by eye, which is in tension with the mandatory
review step above. The format imposes no ceiling, because capping it here would pre-empt a call
that belongs to the task capturing those entries: **T-C3 and T-C4 decide** whether to commit the
real thing or to synthesise a padded construction — plan §6 already blesses synthesis for exactly
those two entries — and to record what replaces the review step either way.

### Threshold-pair entries (T-C3) — what replaces the review step

T-C3 ships four synthetic entries pinned against two thresholds from `src/kitty/bridge/server.py`:

| Entry id | Measured property (CC-converted messages length) | M3 / M4 / M5 coverage |
|---|---|---|
| `tool_result_under_limit` | one `tool_result` string content of length exactly `_TOOL_RESULT_TRUNCATION_LIMIT` (= 50 000) — the largest non-triggering size (the bridge's three sites compare with strict `>`) | M3 complement; M4 has no oversized result either |
| `tool_result_over_limit` | one `tool_result` string content of length `_TOOL_RESULT_TRUNCATION_LIMIT + 1` (= 50 001) — the smallest triggering size | M3 trigger case |
| `compaction_budget_under` | filler alone, CC-converted, serialises to exactly `_COMPACTION_CHAR_THRESHOLD` (= 2 800 000) — the largest short-circuit size on the static fallback (`max_messages_chars=None`, `server.py:7012`) | M5 complement on the static fallback path and on profiles whose derived budget ≥ 2 800 000; M4 complement (no oversized tool result) |
| `compaction_budget_over` | filler alone, CC-converted, serialises to exactly `_COMPACTION_CHAR_THRESHOLD + 1` (= 2 800 001); an oversized tool_result rides on top, pushing the total to ~2 850 100 | M5 trigger on the static fallback path and on profiles whose derived budget is below this body; post-M3-truncation still over threshold so the pruning step fires; also M4 trigger (oversized tool result present) |

#### Why the budget pairs pin the CC-converted shape against the static constant

The bridge's runtime trigger is not the static constant — it is the **profile-derived**
`messages_budget = max_chars - overhead - 10_000`, where `max_chars = min(tokens_to_chars(context_tokens), _MAX_REQUEST_CHARS)`
(`src/kitty/bridge/server.py:7597`, `:7214-7238`). For a 200 K-token model the budget is roughly
790 K; for a 1 M-token model roughly 3.99 M; for an unknown model the budget falls back to
`_MAX_REQUEST_CHARS` (4 M). The static `_COMPACTION_CHAR_THRESHOLD` (2 800 000) is the constant
the bridge uses when `_compact_messages` is called with `max_messages_chars=None`
(`server.py:7012`).

Crucially, the bridge measures `len(json.dumps(messages, ensure_ascii=False))` on the
**CC-converted** messages (`server.py:6996`), not the Anthropic-Messages shape the fixture
commits. The two shapes differ by a constant ~92 chars for this layout, and on the
`use_native_messages=True` passthrough path the Anthropic shape is preserved verbatim. The
builder in `tests/harness/corpus_thresholds.py` therefore sizes filler against the CC-converted
shape, calling `MessagesTranslator.translate_request` directly to measure what
`_compact_messages` will see.

That gives three directional facts the fixtures satisfy, and one thing they do not claim:

* **`compaction_budget_under` is M5-complement on the static fallback.** The fixture's
  CC-length is exactly 2 800 000, which `_compact_messages` short-circuits on
  (`server.py:7014`). The fixture is also a complement on profiles whose derived budget is
  ≥ 2 800 000, and triggers M5 on profiles whose derived budget is smaller (e.g. the default
  200 K-token model gives ~790 K).
* **`compaction_budget_over` is M5-trigger on the static fallback, and post-M3 too.** Pre-M3
  the CC-length is ~2 850 100; post-M3-truncation it is ~2 800 174, still above the threshold,
  so M5's pruning step fires after M3's truncation. (An earlier build sized the filler at
  exactly `threshold + 1` counting the tool_result; M3's truncation collapsed the body back
  below the threshold and M5 short-circuited instead of firing. The filler alone crossing the
  boundary is what keeps M5 live post-truncation.)
* **On profiles whose derived budget exceeds the body's CC-length** (e.g. 1 M-token models at
  ~3.99 M), neither entry triggers M5. The fixture is calibrated to the static fallback
  threshold, and an oracle slice resolving a much larger profile builds its own fixture or
  accepts that neither boundary pair exercises M5 for that profile.
* The M5 trigger is therefore NOT declared in either manifest: it would be false on a profile
  whose derived budget doesn't match the calibration. §"Triggers have three states" above
  forbids declaring triggers whose truth depends on the profile, so M5 is declared at the call
  site that resolves the profile. M4 is similarly pipeline-state-dependent and not declared;
  `compaction_budget_over` carries M4's request-property half (oversized tool_result present),
  and an oracle slice supplies the compaction-engaged half at call time.

#### Why the bodies are padded construction

The two small pairs are real-shaped synthetic conversations; the two budget pairs are **padded
construction**, synthesised per plan §6 because a capture cannot be aimed at exactly the limit
(M3) or exactly the threshold (M5) and survive scrubbing at that size. The padded bodies
honour the README's guarantees:

* The skeleton (initial user turn, closing user turn, the optional tool_use/tool_result pair) is
  reviewable in one screen.
* Every filler line begins `Turn NNNN of TOTAL:` and embeds its turn index — a reviewer can read
  a sample line and recognise construction, and a unique index breaks ties across 400 turns.
* The filler is `lorem ipsum dolor sit amet, …` repeated, no special characters, no escape cost;
  the committed file's bytes are exactly what the builder produced, byte-for-byte. The over
  entry's embedded tool_use block carries a `/tmp/…` path (deliberately — not `/home/<user>/`)
  precisely so the scrubber's `home_path` rule does not rewrite the bytes between builder
  output and committed file.

The 2.8 MB file's mandatory review step is replaced by:

1. Review the **builder** (`tests/harness/corpus_thresholds.py`) — the generator is small and the
   only thing a reviewer needs to understand; the artifact is mechanical.
2. The L1 regeneration test in `tests/harness/test_corpus_thresholds.py` pins every committed
   `<id>.body` file's SHA-256 to what the builder produces right now, and asserts
   `scrub(captured) == captured` so a future scrubber pattern (or a future filler that triggers
   one) breaks the test before the committed file exists. A hand edit of the `.body` or `.json`
   breaks the same test.
3. The L2 lint (`tests/harness/test_corpus_lint.py`) runs the scrubber's `findings` over every
   committed entry, so a future scrubber rule that started matching filler text (e.g. on a
   word-boundary difference) would fail the gate rather than the padding slipping through.
4. The L2 README guard (`TestTheProcedureListsTheThresholdPairs`) reads the committed README and
   fails if any of the four entry ids disappears — replacing the "by read" verification with a
   committed guard.

The constants the pairs straddle are named in `src/kitty/bridge/server.py` (and the builder
imports them); if either constant ever changes, the regeneration test fails on the first CI run
after the change, and a maintainer updates the fixture alongside.

T-C4's decision (KBR-47): its three entries are **synthesised padded constructions**. Each
entry's `origin_note` records both that the size is constructed and what replaces the review step
— code-review of the builder, the scrubber's full-byte scan in CI, and a bounded head/tail human
read of the committed `.body` file. The builder itself ran during authoring (in the task's
requirements notes, which are not tracked); what the repository carries is the `origin_note`'s
description of the construction — a small hand-written core plus a deterministic filler unit —
and each manifest's `body_sha256`, which pins the committed bytes either way.

**Who reviewed.** A `reviewed_by` field was considered and rejected: an unverifiable
self-attestation creates the appearance of an audit trail without the substance. Git already
records who authored the commit that adds an entry, and the pull request records who approved it.
