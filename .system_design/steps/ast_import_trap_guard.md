---
id: ast_import_trap_guard
depends_on: []
---

# AST guard: no test module may `import tests.…`

KBR-280, Epic G. A §6.2.3-style structural guard over `tests/**/*.py` that fails
the suite when any test module imports the `tests` package at module top-level.

## Why

CI's fast-gate job invokes bare `pytest` (`.github/workflows/tests.yml:110`), whose
import resolution differs from a developer's `python -m pytest` (which prepends
cwd). `tests/` is a pytest rootdir without `__init__.py`, so `import tests.X`
resolves under one invocation style and raises `ModuleNotFoundError: No module
named 'tests'` under the other — taking down the entire six-job CI matrix from a
single line. KBR-84 (PR #214) hit this on a contract test's self-guard and cost a
full CI round plus a fix commit. The guard prevents the class of bug, not just the
one occurrence; the falsification channel (reproducing the CI-style invocation
locally) is captured in memory but memories don't enforce.

The obvious alternative — `pythonpath = ["."]` in `pyproject.toml` — is rejected:
it would expose both `harness` (pytest's prepended rootdir) and `tests.harness`
as importable, a double-import hazard where one module has two identities. The
AST guard rejects the bad pattern explicitly and changes nothing else about import
resolution.

## Implementation notes

- `tests/test_no_tests_package_imports.py`, `pytestmark = pytest.mark.l2`
  (§6.2.3 home; §8.5/KBR-216 precedent for living in `tests/` rather than
  `.github/review/tests/`).
- The matcher `_iter_top_level_tests_imports(tree)` inspects `Module.body` only —
  **direct** top-level statements, not `ast.walk` — because the ticket scopes the
  guard to the top-level KBR-84 failure shape. `ast.Import` matches
  `alias.name == "tests"` or a `tests.`-prefixed name; `ast.ImportFrom` matches
  `node.module` the same way, absolute imports only (`level == 0` —
  `from .tests import x` carries `module == "tests"` but its `level == 1` names a
  sibling of the current package).
- Four adjacent shapes are deliberately out of scope, each with its reason:
  (a) function-level imports — same trap, widening is a separate conscious
  decision; (b) `if TYPE_CHECKING:`-guarded imports — runtime-dead under both
  styles, flagging them would be a false positive; (c) top-level
  `try: import tests.X / except ModuleNotFoundError: pass` and other
  runtime-guarded wrappers — the author has explicitly handled the failure mode,
  not the accidental KBR-84 shape; (d) dynamic
  `importlib.import_module("tests.X")` — invisible to an import-statement
  matcher; documented limit, no negative control possible. Boundaries (a)–(c)
  are pinned by negative-control tests.
- Shared-helper rule (§6.2.3 pattern): the synthetic falsification probes call the
  production matcher on `ast.parse`d inline sources, so a matcher regression fails
  every caller.
- Self-guards: the live scan asserts it walks ≥ 1 file and that the guard file
  itself is among the scanned population; the in-tree analogue of the synthetic
  positive was verified by a throwaway scratch module during development
  (`import tests.internal_key_scan` at top-level → guard red naming file+line).
- There is no live positive control in the tree — the KBR-84 fix removed the only
  real offender — so the synthetic inline case is the positive control, mirroring
  `tests/bridge/test_vendor_token_guard.py`'s historical-M13 pattern.
- **Stated limit:** the rationale assumes `tests/` remains a rootdir without
  `__init__.py`; adding one would break the bare-import convention
  (`import layers`, `import internal_key_scan`) the guard's rationale rests on
  and invalidate the KBR-84 hazard model. Policing the absence is out of scope.

## Verification

- `pytest tests/test_no_tests_package_imports.py -q` green.
- `pytest -m "l1 or l2" -q -rsfE --strict-markers --require-category=l1
  --require-category=l2` (the exact fast-gate command) green locally.
- Throwaway scratch module with a top-level `import tests.…` turns the guard red
  with the file and line named (then removed).
- `ruff check` clean; `mypy src/kitty` clean.
- `TEST_SUITE.md` §6.2.3 carries the new guard row and the
  "why an AST guard, not `pythonpath`" note.

## Status

In progress — branch `feat/kbr-280-no-tests-package-imports` (PR pending).
Note: `scripts/regenerate_step_index.py` referenced by `CLAUDE.md` does not exist
in the repo (known gap, see auto-memory); the dependency graph was validated by
inspection — `depends_on` is empty, this step is independent.
