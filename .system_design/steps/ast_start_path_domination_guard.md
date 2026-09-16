---
id: ast_start_path_domination_guard
depends_on: []
---

# AST start-path domination guard

T-E7 of the test-suite implementation plan. Replaces the file-granular start-path coverage
guard with an AST-level structural guard: every `BridgeServer(` call in `src/kitty/**.py`
(except `bridge/server.py` itself) must be lexically dominated by an `egress_block_reason(`
call in its innermost enclosing function.

## Why

The existing file-granular guard at `tests/test_egress_coverage.py::TestEveryStartPathIsGuarded`
asks whether a file contains *both* `BridgeServer(` and `egress_block_reason(`. `cli/main.py`
already holds two start paths, so a third start path added to that same file without a guard
call would pass the existing test unguarded. The gap is described in `TEST_SUITE.md` §5.1
gap 3 and the fix is specified in §6.2.3 (Start-path domination row).

## Implementation notes

- The five live start paths — `bridge_runner.py` ×2 (lines 153, 185), `cli/main.py` ×2
  (lines 559, 661), `cli/launcher.py` (line 191) — are all already dominated by their
  companion `egress_block_reason(` call in the same function. The current codebase passes the
  AST-level guard unchanged.
- The helper walks `src/kitty/**.py` with `ast.parse` and excludes
  `bridge/server.py` (the class definition, line 1483).
- Falsification control per `TEST_SUITE.md` §1.4: an in-test `textwrap.dedent` source
  string constructs four `BridgeServer(` shapes that exercise every pairwise
  combination of {construction, guard} across {own scope, nested scope} — sibling-
  undominated, sibling-dominated, outer-guard with inner-construction (a guard does not
  cross into an inner function), and outer-construction with the only guard confined to
  a nested helper scope (a guard confined to a nested def does not dominate the
  enclosing scope). The fourth shape closes the "guard inside a nested def" blind spot
  a plain `ast.walk(func)` would leave open.
- Construction matcher matches `Name` and `Attribute.attr` spellings, so dotted
  `mod.BridgeServer(...)` constructions are not invisible to the scan; a separate
  test asserts no file aliases `BridgeServer` under a renamed binding, since a renamed
  import (`from kitty.bridge.server import BridgeServer as BS`) cannot be detected
  without import resolution — forcing a deliberate decision if one is added.
- Self-guard for known-positives (matches the pattern of
  `TestEveryHttpClientIsAccountedFor::test_the_scan_actually_finds_something`): asserts the
  scan finds at least five sites across exactly the three expected files.

## Verification

- `pytest tests/test_egress_coverage.py -q` green.
- `pytest -m l2 -q tests/test_egress_coverage.py` green (the layer marker is on the file's
  existing `pytestmark`).
- `ruff check` and `mypy src/kitty` clean.

## Status

Implemented in `feat/kbr-67-ast-start-path-domination`. PR pending.
