---
id: t_w8_bridge_fixture_core_and_transport_extension_interface
depends_on: [KBR-27]
---

# T-W8 — Bridge fixture core and transport extension interface (KBR-31)

Plan row: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §3 **T-W8**. Design:
`TEST_SUITE.md` §7.5. Jira: **KBR-31**.

Depends on T-W4 (KBR-27) — the primary aiohttp recorder, Done. T-W4 has no
step file of its own, so the D7 Jira-key form applies (the convention
`t_g4_wire_shape_guard.md` documents).

**This file lands retroactively (KBR-289).** T-W8 shipped 2026-09-12
(PR #76, merged as `7f4213d`) and no step file was committed for it, so
T-G6's `depends_on` entry pointed at a step id that did not resolve and
`scripts/regenerate_step_index.py` exited 1 for every developer. This file
supplies the honest target. The one-line alternative suggested with KBR-82
(rewriting t_g6's entry to `depends_on: [KBR-31]`) is superseded here: the
plan row records the dependency as T-W8, and a step-id edge keeps the graph
resolved and cycle-checked where a Jira-key cross-reference is accepted
verbatim without resolution.

## Delivered (2026-09-12)

`tests/harness/bridge.py`, beside T-W4's recorder rather than inside it:

* **`UpstreamTransport`** — the extension interface custom transports plug
  into (`name`, `format`, `start`, `stop`, `bind`, `captures`,
  `connections`, `assert_teardown_clean`), plus the registry
  (`register_transport`, `transport(name, fmt, *, responder=None)`,
  `registered_transports`). `AiohttpTransport` is the only transport
  registered here, over T-W4's `RecordingUpstream`.
* **`redirected(adapter, origin, provider_config)`** — re-hosts any
  default-transport adapter onto a recorder.
* **`profile_for` / `backend_for` / `pin_backend_order`** — valid `Profile`
  objects, `(adapter, key, profile)` backend triples, and the balancing
  determinism seam.
* **`InboundProtocol` / `inbound_path` / `minimal_inbound_body`** — the
  agent-facing axis.
* **`BridgeFixture`** — async context manager over a real `BridgeServer`,
  single-backend or balancing.
* **`assert_transport_reaches_its_recorder`** — the conformance check the
  fixture carries: pointed at its recorder → 200 with 1 capture; closed
  port → 500 after 30 s with 0 captures; a different live upstream → 200
  with 0 captures. The third row is why a check rather than a promise:
  nothing fails there, and every §6.3.1 claim built on the fixture would
  be quantified over an empty list.
* `tests/harness/test_bridge.py` (66 cases) and
  `tests/harness/test_bridge_falsification.py` (four deliberate defects,
  one per conformance assertion, each passing the other three); all 80
  assertions in the two modules negated one at a time — 0 survivors.
* `TEST_SUITE.md` §7.5 (§7 had a section per shared harness and none for
  this one) and §8.2 registration with measured runtimes.

## Decisions and why

* **Extension interface in the core, not "any registered adapter".** An
  earlier draft promised any adapter while depending only on T-W4, which
  would have pulled T-B1–T-B3 into the shared fixture. Defining the
  interface here lets recorder authors integrate without editing the core.
* **Re-hosting, never replacing.** `redirected()` overrides `build_base_url`
  — the one method the bridge calls for the destination — and substitutes
  scheme and authority only. Replacing the whole URL would delete Vertex's
  `/projects/{id}/locations/{loc}` (the account being billed, §3.3.5 P21),
  while Azure survives whole-URL replacement — so the obvious check would
  pass while the real row breaks.
* **Two axes, not one.** `WireFormat` is what a recorder serves;
  `InboundProtocol` is the bridge route an agent posts to. Two of the six
  formats have no inbound route at all.
* **Deliberately not included:** no `host=` seam and no pytest plugin
  (neither had a paid-for consumer; the plugin measured 0.72 s charged to
  every pytest run); the ~38 existing modules that build their own bridge
  are not migrated (§7.5 records the fixture as for new work, not debt);
  the Epic B recorder transports (T-B1–T-B3), §5.3 addressing (T-E1/E2),
  T-W9's obligations, and the capture-retention bound (T-K4) stay with
  their own tickets.

## Notes

* Landed as PR #76 (merged `7f4213d`, 2026-09-12): full suite 4361 passed /
  2 skipped on 3.10–3.13 in CI. Both new modules run at the `l1` path
  default per §8.2 — a test may not move to `l3` before the Subsystem job
  (T-K6) exists, or CI's `-m "l1 or l2"` runs it in no job at all.
  `BridgeFixture` teardown orders client-session close before
  `BridgeServer.stop_async`; that range is where the asyncio `wait_closed`
  behaviour splits, which is why the full 3.10–3.13 matrix is load-bearing
  for this fixture.
* Found while doing this: KBR-175 (closed) — `tests/conftest.py::sample_profile_dict`
  carried a UUIDv7 `auth_ref` where `Profile` requires v4, so it could not
  build the model it described; nothing read it. Deleted the same day per
  `TEST_SUITE_IMPLEMENTATION_PLAN.md` row KBR-175.
* **2026-09-18 (KBR-289):** this step file created retroactively to repair
  the step graph — see the paragraph above Delivered.
