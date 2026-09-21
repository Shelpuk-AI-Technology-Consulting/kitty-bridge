---
id: kbr90_extract_ollama_body
depends_on: [KBR-40]
---

# KBR-90 (T-H5) — Extract `_ollama_body` pure payload builder

Ticket: [KBR-90](https://shelpuk.atlassian.net/browse/KBR-90) (Epic H —
Mutation validation, parent KBR-20). Design: TEST_SUITE.md §6.1
("P19's `_ollama_body(...)` is still owed (KBR-90, T-H5)"),
plus §3.2.2 (P19 row) and §3.2.3 (provider-aiohttp). Baseline:
MUTATION_BASELINE.md (new `_pending_` group row).
Requirements: `.requirements/20260921T224046Z_extract_ollama_body_pure_builder/REQUIREMENTS.md`.

## What

Apply the KBR-89 pattern to `OllamaCloudAdapter`. Extract
`_ollama_body(self, cc_request, *, streaming: bool) -> dict` from
`make_request` and `stream_request`. The builder calls `translate_to_upstream`
then sets `stream` to the `streaming` flag — P19's load-bearing site moves from
the two network methods to the builder, so mutmut can reach it from the L1
selection. Both transport methods delegate to the builder and post the
returned body verbatim; no in-transport body mutation.

## Why the streaming flag (KBR-89 needed no equivalent)

Bedrock's pops are defensive (`stream` is *absent* regardless of whether
`translate_to_upstream` ever emitted it — the load-bearing assertion is the
`modelId` pop, and the `stream` assertion is vacuously true today). Ollama's
P19 is the opposite half: `stream` is a **conditional overwrite the transport
REALLY applies** (False for `make_request`, True for `stream_request`), and
the override must reach the wire even when `translate_to_upstream` already set
`stream` from the request. The builder's `streaming` parameter is the single
hook point that expresses "decide the mode regardless of what translate said".
A mutation that drops the assignment is the real kill (per the bedside reading
in KBR-90 comment 2).

## Companion-site sweep (KBR-90 comment 2 + the 7-site register memory)

Adding a mutmut scope row or re-targeting an existing register row touches
seven sites beyond the code change. Done in one PR, intentionally clustered:

* `tests/mutmut_scope.py::TARGET_GROUPS["ollama_transport"]` — new
  per-adapter group (sibling of `bedrock_transport`, NOT `provider_hooks`,
  whose description claims "every adapter implements these three").
* `tests/test_aggregate_mutation_baseline.py` — fixture-seed mirror in
  both per-group tests (`test_main_exits_zero_when_a_group_is_fully_tested`
  and `test_render_markdown_table_renders_but_does_not_fail_on_unmatched`).
* `tests/harness/register.py::MutationRow(id="P19")` — site retargeted to
  `OllamaCloudAdapter._ollama_body`, comment mirroring P18.
* `.system_design/TEST_SUITE.md` — §3.2.2 P19 row, §3.2.3 bullet + table,
  §6.1 target table + the "still owed" → landed note.
* `.system_design/MUTATION_BASELINE.md` — `ollama_transport` row, `_pending_`
  until KBR-91 re-measures (mirror of `bedrock_transport`'s row).
* `tests/test_wire_shape_honesty_wire.py` — streaming-half capture +
  adapter-side falsification (KBR-90 comment 1; closes the streaming-half
  stated limit flagged in KBR-80).
* `.system_design/steps/kbr90_extract_ollama_body.md` — this file.

The P19 references in `tests/harness/contract.py:1207` and
`tests/harness/test_register_agreement.py:285` are id-only or
shape-effect-still-accurate and need no edit (verified per the multi_round
review sweep rule).

## Why a new group instead of `provider_hooks`

The `provider_hooks` group's description names `translate_to_upstream`,
`normalize_request`, `build_upstream_headers` — the three every adapter
implements. `_ollama_body` is not one of those: only `bedrock_transport` and
`ollama_transport` carry it, and adding it to `provider_hooks` would either
violate the "every adapter" claim (correctness) or require every other
provider to grow an empty body-builder stub (false code). The per-adapter
group, mirroring KBR-89's `bedrock_transport`, is the established shape.

## Verification

11 L1 + 2 L2 tests on origin/main's branch (`fix/kbr-90-extract-ollama-body`):

* `tests/test_provider_ollama_cloud.py::TestOllamaCloudBody` — return-shape
  (stream True/False), injected-translate overwrite kill (the real P19 kill,
  with two separate `patch` contexts to dodge the shared-dict aliasing
  trap), no-aiohttp purity, body-matches-translate property (×6), sentinel-
  passthrough delegate tests for `make_request` and `stream_request`.
* `tests/test_wire_shape_honesty_wire.py` — non-streaming capture/falsification
  survived verbatim (per-side commentary updated); streaming capture +
  falsification added with an empty `mock_aiter` so the streamed-byte path is
  inert and the assertion is on `session.post(json=)`.
* `tests/mutmut_scope.py` — new `ollama_transport` group.
* `tests/test_aggregate_mutation_baseline.py` — fixture seeds in both
  per-group tests.
* `tests/harness/register.py::MutationRow(id="P19")` — site retargeted.

`CI mypy src/kitty` returns success across 94 files; the touched src is
clean. `tests/harness/` registers + `tests/test_mutmut_scope.py` +
`tests/test_aggregate_mutation_baseline.py` +
`tests/test_wire_shape_honesty_wire.py` + the L1 `test_provider_ollama_cloud.py`:
all green.

## Implementation notes (2026-09-22)

- The red-watch before the seeds landed the load-bearing mechanism the
  ticket-comment-2 warning flagged: with a new non-deferred group and no
  seed, the aggregator's `main()` exits 1 with "zero mutants matched its
  patterns" — exactly the silent-failure path the loader-only `only_mutate`
  guard does not catch.
- The first-pass test wrote `patch.object(..., return_value=injected)` for
  both calls inside one `with` — the builder mutates the returned dict in
  place, so the second call's `stream=True` aliased into the first call's
  captured body. Two separate `with` blocks (each taking a fresh
  `dict(injected)` copy) fixes it and matches the bedrock precedent of one
  call per patch context. Recorded in Serena memory
  `mutmut-elimination-shared-builders` so the next per-adapter extraction
  does not re-discover it.
- The first-pass property test added an assertion `body["stream"] !=
  translated["stream"]` to catch a hypothetical dropped-assignment mutant —
  but the assertion is logically wrong when the request-level value happens
  to equal the flag (e.g. `stream: False` request + `streaming=False`
  builder → `False != False` fails despite correct behaviour). The kill is
  already pinned by `test_overwrites_stream_even_when_translate_emits_it`;
  the property test's job is `body == expected`, which is sufficient.
- ProviderError wrap pin verified, per comment 2's last item: the pins at
  `tests/test_provider_ollama_cloud.py:486` (`match="401"`) and `:581`
  (`match="500"`) cover the message-substring shape; the error path stays
  inside `make_request` / `stream_request` (never in `_ollama_body`), so
  the refactor does not move it and no edit was needed. Recorded in Serena
  memory `kbr90-ollama-builder-extraction`.

