---
id: t_i10_backend_context_isolation
depends_on: [t_w8_bridge_fixture_core_and_transport_extension_interface]
---

# T-I10 — `_backend_context` isolation under concurrency (KBR-102)

Plan row: `TEST_SUITE_IMPLEMENTATION_PLAN.md` §12 **T-I10**. Design:
`TEST_SUITE.md` §6.3.1 (the `_backend_context` isolation row). Jira:
**KBR-102**.

Depends on T-W8 (KBR-31) — the bridge fixture (`BridgeFixture`,
`backend_models`, `pin_backend_order`) the L3 form is built on. Landed as
PR #76 (merged `7f4213d`).

Requirements: `.requirements/20260921T224017Z_kbr102_t_i10_backend_context_isolation/REQUIREMENTS.md`
(rev r2, after three `system-design-reviewer` passes).

## Planned delivery (pending implementation)

The shape of the module to be written under R1–R8:

* `pytestmark = pytest.mark.l3` (path default for `tests/bridge/` is `l1`;
  the file overrides, matching `test_cross_attempt_content_l3.py` and
  `test_tc4_corpus_recovery_l3.py`).
* Drives a real balancing `BridgeServer` (two members via
  `BridgeFixture(backend_models=["m0", "m1"])`) through concurrent aiohttp
  requests, with `pin_backend_order` (T-W8's round-robin seam) making the
  distribution deterministic.
* **The per-request join:** each response's `X-Kitty-Model` /
  `X-Kitty-Backend` headers — read from `_backend_context` at
  `_attribution_headers()` (`server.py:3086`) — must name the member whose
  upstream capture body carries that request's marker. The capture's
  `"model"` JSON field is the second-source oracle on per-request
  identity, independent of the header. A peer's selection leaking via a
  shared context copy fails the join.
* **Both stamping paths:** one non-streaming case (the
  `_attribution_middleware` site, `server.py:3796`) and one streaming case
  (the StreamResponse construction site, `server.py:5471`, inside
  `_stream_messages` which begins at `server.py:5394`).
* **Deterministic per-task order:** a staggered-start case waits on
  `len(captures) == 1` before creating the second request, pinning the
  per-task list `[harness0, harness1]`. The isolation claim is owned by
  the two concurrent cases (real interleaving); this one orthogonally
  pins the per-task ordering the round-robin produces when task starts
  are sequenced explicitly.
* **Non-vacuous guard:** `len(captures) == N` asserted **before** any
  per-response pre-flight, so the guard is the first line that fires
  under the R6 recipe (§6.3.1's empty-list rule).

## Decisions and why

* **A separate L3 module, not a marker on the existing L1 file.**
  `tests/bridge/test_concurrent_backend_selection.py` (on `origin/main`
  since `a0c1a21`/`9dd2745`) proves the *mechanism* by calling
  `_select_backend()` and reading the ContextVar directly — a legitimate
  L1 unit claim. §6.3.1's row is the *boundary* claim (concurrent HTTP
  requests); a regression that made a handler read an instance field, or
  set the ContextVar outside the request's task context, would pass every
  L1 test and corrupt real traffic. Two layers, two files, both kept.

* **Headers read through a private client session, not a `BridgeFixture`
  extension.** `BridgeFixture.post` returns `(status, text)` and discards
  headers (`tests/harness/bridge.py:859`); the oracle lives in
  `X-Kitty-Backend`/`X-Kitty-Model`. Extending the fixture would be
  T-W8-scope needing its own step; dropping the fixture would lose the
  lifecycle and the recorder hand-off. A local `_post_returning_headers`
  helper posting to `fixture.base_url` keeps both.

* **`X-Kitty-*` headers as the oracle, not `/stats`.**
  `_attribution_headers()` reads the ContextVar in the request's own
  middleware chain, so it is per-request by construction. `/stats` counts
  every attempt session-wide and cannot express per-request attribution.
  If the header is wrong, client-visible attribution is wrong — which is
  what §6.3.1 protects.

* **`pin_backend_order`, not a `_get_next_backend` monkeypatch.** The L1
  file pins selection by patching `_get_next_backend` directly (fine for a
  unit test of the mechanism); at L3 that would bypass the selection path
  the claim is about. T-W8's seam patches `random.choices` — the one
  sanctioned determinism hook that leaves the selection code path intact.
  Note: the patch is process-wide for the duration of the test
  (`tests/harness/bridge.py:686`); inside these tests, do not call
  `random.choices` for any other purpose.

* **Marker-keyed captures for the join.** Concurrent requests make
  upstream arrival order nondeterministic; the `marker()` string in each
  request body is what pairs a capture to its request without relying on
  order. `asyncio.gather` preserves task order, so `results[i]`
  corresponds to `markers[i]` regardless of completion order.

* **Route sampling named, not hidden.** `/v1/messages` (the T-I8
  precedent). The ContextVar is a module-level singleton
  (`server.py:1934`) shared by all four handlers, so the property is
  universal across routes by construction; the sample is for
  observability, not coverage (sweep rule (a) recorded here).

* **Guard falsifications to be observed during development, not shipped as
  tests.** Two recipes; both run only as scratch experiments in the working
  tree:
  1. *Product-code defect (observed 2026-09-22):* scratch-revert the
     reader in `_attribution_headers` from `idx = ctx["idx"]` to
     `idx = self._backend_idx` (`server.py:3106`, working tree only) →
     the two concurrent tests go red with the message *"response for
     marker … names backend 'harness1' but the upstream capture for that
     request was served by profile 'harness0'. The ContextVar leaked: a
     peer's selection reached this response"*. The staggered test
     continues to pass under this defect (its timing window — A's
     middleware runs before B's `_select_backend` — does not exercise
     the reader-side swap). Revert; green. Verified in
     `tests/bridge/test_backend_context_isolation_l3.py` during the
     implementation. The `_active_*` properties (`server.py:2127-2172`)
     are *not* on the oracle path; `_attribution_headers` reads the
     ContextVar directly, so breaking the properties does not exercise
     the §6.3.1 boundary claim.
  2. *Test-guard defect:* patch `server._select_backend` to raise
     `AllBackendsUnhealthyError(retry_after=400)` (the
     `test_all_backends_unhealthy.py:139-168` idiom; beyond KBR-243's
     recovery window → immediate 503) → zero upstream captures, two 503
     responses, no `X-Kitty-Backend` on either → the `len(captures) == N`
     guard fires with a legible message instead of a confusing downstream
     failure. Recorded here rather than shipped: the 503 shape is already
     covered by `test_all_backends_unhealthy.py`, and re-testing it would
     duplicate that module's territory.

## Notes

* `l3` has no CI job yet (T-K6 owed,
  `tests/layers.py::PENDING_ACTIVATION_LAYERS`): the Fast gate does not
  collect this file; bare local `pytest` does. Same status as
  `test_tc4_corpus_recovery_l3.py` and `test_cross_attempt_content_l3.py`.
* Per-test ContextVar isolation across the suite is enforced by the
  autouse `_reset_backend_context` fixture (`tests/conftest.py:92-104`);
  the new module must not defeat it, and the bridge-directory slice
  (`pytest tests/bridge/ -q`) is the check.
