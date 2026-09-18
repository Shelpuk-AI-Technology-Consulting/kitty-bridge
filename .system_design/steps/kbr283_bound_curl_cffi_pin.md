---
id: kbr283_bound_curl_cffi_pin
depends_on: [KBR-85]
---

# KBR-283 — Bound the curl_cffi dependency pin

Plan: `REQUIREMENTS.md` at
`.requirements/20260918T200000Z_bound_curl_cffi_pin/REQUIREMENTS.md`.
Jira: **KBR-283** ([link](https://shelpuk.atlassian.net/browse/KBR-283)).
Design: `TEST_SUITE.md` §6.2.4 (the dependency-behaviour-contracts table and
its rule that where no stable neighbour exists the row must record the chosen
option), gap register **G11**.
Predecessor: **KBR-85** ([link](https://shelpuk.atlassian.net/browse/KBR-85),
PR #212) landed the 29-test curl_cffi dependency-contract suite and found the
0.16.3 Windows-build `HTTP_PROXY` divergence — the concrete drift evidence the
bound fences.

## What the task does

Replaces the only unbounded HTTP-transport declaration,
`curl_cffi>=0.7`, with `curl_cffi>=0.15,<0.17` — the product owner's
decision recorded 2026-09-18 (the bound option from the ticket's own
example; the written-rejection option was declined because `uv.lock` is
gitignored, so `pyproject.toml` is the only durable statement of the range
and no other upgrade gate exists). The contract suite pins the
known-critical surface; the bound forces unknown drift through a
deliberate, reviewed upgrade.

The change:

1. `pyproject.toml`: `"curl_cffi>=0.7",` becomes `"curl_cffi>=0.15,<0.17",`
   with an ASCII policy comment mirroring the botocore-comment pattern.
2. New L2 guard
   `tests/test_pypi_packaging.py::TestVersionConsistency::test_curl_cffi_pin_matches_recorded_policy`:
   character-exact assertion on the declared specifier (KBR-9 literal-
   agreement precedent), so a regression to unbounded fails the gate and a
   deliberate upgrade must touch the guard in the same PR.
3. Local `uv lock` regeneration (the file stays gitignored; locked version
   stays 0.16.3).
4. Prose sweep — every artifact claiming the pin is unbounded:
   TEST_SUITE.md §6.2.4 curl_cffi row (records the bound + decision), the
   #675 lifecycle passage, G11 (drops curl_cffi from still-open), the
   contract-test module docstring, the `openai_subscription`
   `_curl_session_instance` docstring, the `test_layer_selection.py`
   L2-registration comment, and the `test_token_transport.py` timeout
   docstring.
5. The full curl_cffi contract suite green on the new pin.

## Implementation notes

### What landed (2026-09-18)

1. `tests/test_pypi_packaging.py::TestVersionConsistency::test_curl_cffi_pin_matches_recorded_policy`
   -- the L2 pin guard: raw-line scan anchored on the `"curl_cffi` dependencies
   entry (immune to a policy comment restating the specifier), character-exact
   equality on `curl_cffi>=0.15,<0.17`, failure message names the upgrade
   procedure. TDD red at base (found `curl_cffi>=0.7`), green after (2).
2. `pyproject.toml`: `"curl_cffi>=0.7",` -> `"curl_cffi>=0.15,<0.17",` with the
   ASCII policy comment (botocore-comment pattern; the uv.lock-gitignored note
   included).
3. `uv lock`: specifier metadata updated, 0.16.3 kept; `uv.lock` stays
   gitignored. Environment note: the worktree venv then needed
   `uv sync --extra dev` (a bare `uv sync` strips the dev extra, and KBR-82's
   schemathesis test files collect-fail without it).
4. Prose sweep (all six sites): TEST_SUITE.md section 6.2.4 row (records the
   bound + the both-mechanisms choice), the #675 lifecycle passage, G11
   (curl_cffi out of still-open; keyring + interpreter remain), the
   contract-test module docstring, the `_curl_session` docstring, the
   `test_layer_selection.py` L2-registration comment, and the
   `test_token_transport.py` timeout docstring.

Gates: ruff clean on all touched files; the four touched test files together
80 passed / 0 errors (including the whole-suite coherence tests); contract
suite 29/29 on the new pin. The step-index validator exits 1 on a
PRE-EXISTING dangling dep shipped by KBR-82's merge to main
(`t_g6_openapi_schemathesis_conformance.md` ->
`t_w8_bridge_fixture_core_and_transport_extension_interface` does not exist);
verified pre-existing by removing this step file and re-running. Developer-side
only (not CI-gated, KBR-278 decision D2); flagged for a follow-up ticket.

Design review: system-design-reviewer ran one round on REQUIREMENTS.md
(4 concerns + 6 suggestions, all folded in before implementation); the
policy wording was reconciled to the decided range per concern 1.

## Status

Implemented (2026-09-18); PR open, awaiting CI + review.
