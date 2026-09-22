---
id: t_j3_eg_scenarios
depends_on:
  - KBR-62
  - KBR-66
  - KBR-68
  - KBR-107
---

# T-J3 — EG acceptance scenarios (KBR-109)

## What

Four Gherkin acceptance scenarios for invariant I3 (Egress Containment) — EG-0, EG-1, EG-2, EG-3 — in `tests/acceptance/features/egress.feature`, with step definitions extended into `tests/acceptance/test_acceptance.py` and three fixtures added to `tests/acceptance/conftest.py` (`sealed_network`, `egress_for_sealed_network`, `refusing_profile_start_path`).

## Why

KBR-109 is the last link in the I3 chain: T-E2/E6/E8 deliver the L3 harness, T-J1 delivers the Gherkin↔L3 binding, and T-J3 is the user-visible journey written in Gherkin on top. KBR-135's closing condition (set 2026-09-11) is *"KBR-135 closes when EG-0 through EG-3 pass"*, so KBR-109's four scenarios are also KBR-135's acceptance criterion. The PR that lands T-J3 closes both tickets together.

## How

- **EG-0 (reachability control).** `BridgeAiohttpContainment.drive_phase_1` → `status == 200`, `len(captures) == 1`, `attempts == []`. The control that makes EG-2's "zero connections" assertion non-vacuous (TEST_SUITE.md §5.3 trap 2).
- **EG-1 (containment, healthy).** `BridgeAiohttpContainment.drive_with_egress` with `egress=EgressConfig(...)` from the harness → every recorder peer port joins a tunnel (`unattributable_peer_ports == []`), captures recorded.
- **EG-2 (containment, proxy down).** Stop the proxy mid-scenario, then drive once → `status != 200` AND the drive's snapshot of the recorder empty (`eg_drive.connections == []` and `eg_drive.captures == []`). The snapshot is what the L3 `Phase1Result` carries (a copy of `sealed_network.recorder.connections` and `.requests` taken inside `drive_with_egress`); the **recorder-direct** falsification sensitivity is inherited from T-E2's phase 3, not re-implemented here.
- **EG-3 (unproxyable profile).** The bridge-runner single-profile start path from `tests/test_egress_start_path.py`, replayed as a Gherkin step: `bridge_runner.main()` with the Bedrock-SSO profile → non-zero SystemExit, stderr names `bedrock-sso`, `BridgeServer.__init__` never reached.

**Why bind to L3 rather than re-implement (the §2.2 allocation rule).** The mechanical substance of every assertion in EG-0..EG-2 is already proven at L3 by `tests/harness/test_aiohttp_containment_slice.py` (phases 1/2/2b/3, the injected-bypass falsification in T-E2) and `tests/harness/test_local_bypass_slice.py` (T-E8). The L4 layer's job is the user-visible journey — what a developer running `kitty claude` sees, not what a TCP socket sees. Re-implementing at L4 would duplicate L3 and, worse, would drift from L3 on the next harness change.

**Per-transport coverage stays at L3.** EG-0's TEST_SUITE.md §6.4.1 wording is *"kitty sends a request on each supported transport"*. The L4 step exercises the bridge's own aiohttp serving path — the one the developer-session EG-1/EG-2 cover — and the per-transport matrix (curl_cffi, botocore, provider-aiohttp) stays at L3 in T-E3/T-E4/T-E5. The KBR-109 owner comment on 2026-09-17 explicitly directed this scoping. The narrowing is recorded in the feature file's preamble so a future contributor does not re-add the §6.4.1 wording thinking it satisfies R1.

**EG-2's "fails with a clear error" is intentionally `!= 200` only.** The user-visible failure shape is the proxy-down behaviour (`status < 0` and `text` carrying the connection-error repr), and the negative assertion EG-2 actually carries evidence on is the recorder-empty check (TEST_SUITE.md §5.2.2 row 2). A "clarity" assertion would need a product decision about the exact error body — that decision lives with T-E2's L3 contract, not with this L4 binding. Narrowing further here would re-prove L3.

**Falsification discipline is collective, not per-scenario.** TEST_SUITE_IMPLEMENTATION_PLAN.md §1.4 (the harness rule — note: this rule lives in the *implementation plan*, not in TEST_SUITE.md; TEST_SUITE.md §1 has no subsections) requires the first working version of every harness to ship with a falsification case. T-J3 inherits four:

1. The T-J1 wiring smoke (`tests/acceptance/features/wiring_smoke.feature`) — any future step text mistyped in the EG module fails pytest-bdd collection here too.
2. The T-E2 phase 3 injected bypass — any future change that makes the L3 negative assertion silently pass is caught independently of T-J3.
3. The T-E8 start-path falsification in `tests/test_egress_start_path.py::TestReturnDiscardedFalsification` — any future change that breaks the bridge_runner guard's enforcement is caught independently of T-J3.
4. **The T-E8 AC2.2 compliant control** (`tests/test_egress_start_path.py::TestCompliantControl::test_non_sso_aws_key_proceeds_past_the_guard_to_bridge_server`) — the positive control that proves EG-3's assertion surface is not vacuous. EG-3 deliberately omits an L4-level compliant-profile control (the conftest's `refusing_profile_start_path` carries `key_holder` for parity but does not exercise it): a fixture that hardcoded refusal would pass EG-3 vacuously, and only the L3 AC2.2 control owns that case. If that L3 test is ever renamed, skipped, or removed, EG-3 loses its positive control and must re-home it.

A separate T-J3-only structural test (e.g., "the .feature file declares exactly four scenarios") would be redundant: pytest-bdd's collection already fails on a malformed or mistyped step. The discipline is documented here so a future reviewer does not propose adding one.

**The `acceptance` layer default, and the T-K9 handoff (corrected after PR-review).** Per `tests/layers.py::_PATH_DEFAULTS`, every test file under `tests/acceptance/` carries the `acceptance` layer marker by default (T-J1 added the row). Two different pytest invocations govern which scenarios actually run:

* **A bare local `pytest`** uses the `addopts` expression (`pyproject.toml:144`): `not agent_smoke and not agent_live and not eval and not load`. `acceptance` is **not** in `RESOURCE_DEPENDENT_LAYERS`, so a bare local `pytest` collects and runs the EG scenarios.
* **The Fast gate** (`.github/workflows/tests.yml:110`) runs `pytest -m "l1 or l2" -q -rsfE --strict-markers --require-category=l1 --require-category=l2` — it *positively* selects L1 and L2 and *deselects every other layer*, `acceptance` included. So the Fast gate does **not** run the EG scenarios today.

An earlier draft of this paragraph conflated the two invocations and claimed the Fast gate runs these scenarios — it does not. A future change that breaks egress containment merges green today; the four scenarios enforce nothing but a developer's local run. KBR-135's 2026-09-11 closing condition ("EG-0 through EG-3 pass") is satisfied at the code level, but the implicit stronger reading ("and a CI gate enforces them") requires T-K9 (KBR-118), the Acceptance job activation that *positively* selects `acceptance`. **T-K9's obligation:** (a) add the Acceptance job to `tests.yml` (or its successor) with the right `--require-category=acceptance` plumbing, (b) drop the `acceptance` entry from `PENDING_ACTIVATION_LAYERS` in the same change (the two-directional check in `tests/layers.py::stale_pending_layers` fails otherwise). Until that lands, the `acceptance` entry is *acknowledged debt* — recorded here so a future reader does not mistake the local-green state for enforcement.

**Why this PR still merges.** T-J3's deliverable is the four scenarios themselves; their absence is what blocks KBR-135. The scenarios pass locally and falsify the harness through the inherited surfaces (T-J1 wiring smoke, T-E2 phase 3, T-E8 AC-2.2/AC-2.3). T-K9 turns local-green into CI-enforced; that is its own ticket and its own PR, scheduled by §8 / §13 of the implementation plan.

## Implementation notes (2026-09-20, this PR)

- One feature file, one extension of `test_acceptance.py`, one extension of `tests/acceptance/conftest.py`. No new module. Per the T-J1 docstring's promise (*"T-J2 and T-J3 add their `.feature` files and step definitions alongside this one's smoke; no further wiring is needed"*).
- The `sealed_network` fixture in `tests/acceptance/conftest.py` mirrors `bridge_session`'s loop-ownership discipline: function-scoped, the asyncio loop owned by the fixture (not pytest-asyncio), builds the harness fresh per scenario (phase-1's peer-port row would leak across scenarios if shared, defeating phase 2's "zero connections" assertion). It also takes `aiohttp_trusts_test_ca` — without it the bridge's outbound TLS handshake to the harness recorder fails with `ClientConnectorCertificateError` (measured, not assumed: the first run without it timed out through three blip retries).
- The `refusing_profile_start_path` fixture is a near-copy of `tests/test_egress_start_path.py::start_path`, scoped to the single EG-3 scenario. The duplication is intentional: tests/acceptance/ has no `l3` job, and importing the L3 fixture across the layer boundary would couple the binding to that fixture's monkeypatch lifetime.
- The EG-3 `When` step captures the `SystemExit` and the capsys drain itself: a `Then` step cannot retroactively wrap the `When`'s raise (pytest-bdd fails the scenario at the raise, so the `Then` steps would never run), and `capsys.readouterr()` empties the buffer, so a `Then` that read it would leave the second `Then` reading an empty stream. Both facts are measured, not assumed.
- EG-2's step asserts on `eg_drive.connections` and `eg_drive.captures` — the drive's snapshot of the recorder. The rationale ("a silently-empty assertion cannot pass on a non-empty leak") is satisfied at L4 by the snapshot being non-mediated by any EG-layer helper, and at L3 by T-E2 phase 3 reading the recorder directly.
- KBR-109 transitions To Do → Done in this PR (its deliverable is the four scenarios, delivered). KBR-135's closing condition ("EG-0 through EG-3 pass") is satisfied at the code level locally, but the implicit stronger reading ("and a CI gate enforces them") requires T-K9. The PO decides whether to close KBR-135 on this PR or on T-K9's activation — the closing comment on KBR-135 will note both readings.
- Step index regenerated 2026-09-20; dependency graph validates (`depends_on: KBR-62, KBR-66, KBR-68, KBR-107`, all Done); `scripts/regenerate_step_index.py` exits 0.
