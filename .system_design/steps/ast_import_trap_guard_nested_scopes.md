---
id: ast_import_trap_guard_nested_scopes
depends_on:
  - ast_import_trap_guard
---

# Widen the tests-package import guard to nested scopes

KBR-282, Epic G. Second slice of the §6.2.3-style structural guard from
KBR-280: `tests/test_no_tests_package_imports.py` now flags every absolute
`import tests[.X]` / `from tests[.X] import …` statement not only at module
top level but inside any nested function, async-function, or class body, to
arbitrary depth.

## Why

KBR-280's matcher inspected `Module.body` only. The same bare-`pytest` trap
lives one scope level deeper: `def helper(): import tests.helper` passes the
top-level scan, passes CI at import time (the function body never executes
during collection), then raises `ModuleNotFoundError` under CI's bare
`pytest` the moment any test calls `helper()` — the identical KBR-84
failure shape that cost a full CI round. KBR-280's own module docstring
deferred widening as "a separate, conscious decision"; KBR-282 is that
decision. No live offender exists in the tree today (grep: zero
`import tests` / `from tests` under `tests/`, top-level or nested), but the
failure mode is identical to KBR-84's and the cheapest moment to adopt the
wider contract is before someone writes the trap by accident.

## The walk semantics — allowlist, not denylist

The ticket says "collecting every `Import`/`ImportFrom` of `tests` anywhere
in the tree" while also instructing to keep KBR-280's four out-of-scope
shapes. The two cannot both be read literally: a blanket `ast.walk` would
flag `if TYPE_CHECKING:` imports (a false positive) and author-handled
`try`/`except` wrappers (a contract break). The resolution: the walker
descends into exactly three node types — `FunctionDef`, `AsyncFunctionDef`,
`ClassDef` — and treats every other statement-bearing node as a leaf
(inspected for direct `Import`/`ImportFrom` statements, never descended
into). Python's own scope statements are exactly those three; everything
else is either a runtime guard (`If`, `Try`, `With`, `While`, `AsyncWith`,
`TryStar`, and the `orelse`/`finalbody` arms of each), a loop or `match`
body (`For`, `AsyncFor`, `Match`), or a non-statement node.

A denylist reading of the same instruction would have flagged `async with`
and `except*` guards — the same runtime-guard contract broken through its
own exception list. The skip set therefore falls out of the allowlist
rather than being enumerated negatively; the module docstring and the
TEST_SUITE.md §6.2.3 row state the resulting limits explicitly (whole-node
skip covers `else`/`finally` arms; `for`/`async for`/`while`/`match` bodies
and guard-wrapped scope statements are not visited; `importlib` dynamic
imports remain invisible) so the next widening starts from an honest
boundary instead of rediscovering it.

## Implementation notes

- Matcher renamed `_iter_top_level_tests_imports` →
  `_iter_tests_package_imports` (the old name asserted a scope the widened
  matcher no longer has); the recursion lives in a private
  `_tests_imports_in_scope(body)` helper that matches imports at its level
  and recurses into the three scope-node types. Absolute-import matching
  (`level == 0`) is unchanged, now pinned at function depth too.
- KBR-280's negative control `test_function_level_import_is_out_of_scope`
  was consciously rewritten into `test_function_level_import_is_flagged` —
  exactly the tripwire its own docstring named. Six positive controls now
  pin the nested positions: function, async function, class body, method,
  function-in-function, class-in-class.
- The runtime-guard exemption is now pinned at function depth by a
  parameterised negative control over all four base shapes (`try`/`except`,
  `if False:`, `with`, `while False:`) — so a regression deleting any one
  from the skip set turns the boundary red instead of moving silently.
  `from .tests import x` is likewise pinned at function depth.
- Live-guard test renamed
  `test_no_test_module_imports_the_tests_package_at_top_level` →
  `test_no_test_module_imports_the_tests_package` and its assertion
  message dropped the "at top-level" phrasing a maintainer reads in a red
  CI log; `_scan_tests_tree`'s docstring likewise. All in-file surfaces
  were swept in the same pass (test id, message, docstrings, matcher name)
  so none states a scope the code no longer has.
- Same file, same `pytestmark = pytest.mark.l2`; the file keeps its
  registration in `test_layer_selection.py`'s explicit-L2 set (that set
  keys on file path; only new L2 files require registration).
- Live-scan falsification: a throwaway scratch module with a
  function-level `import tests` (never executed at collection time) turned
  the live scan red naming `test_scratch_kbr282_probe.py:11 -- import
  tests`; removed afterwards, scan green again.

## Verification

- `pytest tests/test_no_tests_package_imports.py -q` green (25 tests:
  6 new positive controls, 5 new negative controls, 14 carried over).
- Positive controls watched failing against the pre-widening matcher for
  the right reason (`[]` at every nested position) before the walker was
  widened.
- Live-scan scratch-offender round trip red → green (above).
- `pytest -m "l1 or l2" -q -rsfE --strict-markers --require-category=l1
  --require-category=l2` (the exact fast-gate command) green.
- `ruff check` clean; `mypy src/kitty` clean.
- `TEST_SUITE.md` §6.2.3 row re-lettered (three out-of-scope shapes), the
  stated limits recorded; this step file is the widening's audit trail.

## Status

Implemented on branch `feat/kbr-282-nested-tests-package-import-guard`.
Note: `scripts/regenerate_step_index.py` exits 1 in this worktree because
of a pre-existing dangling dep carried over from `main`
(`t_g6_openapi_schemathesis_conformance` →
`t_w8_bridge_fixture_core_and_transport_extension_interface`, owned by
another ticket); this step introduces no new validation error.
