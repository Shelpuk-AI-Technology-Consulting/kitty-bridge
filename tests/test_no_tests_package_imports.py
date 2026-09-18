"""Structural guard: no test module may import the ``tests`` package.

CI's fast-gate job invokes bare ``pytest`` (``.github/workflows/tests.yml``),
whose import resolution differs from a developer's ``python -m pytest`` (which
prepends the working directory to ``sys.path``). ``tests/`` is a pytest rootdir
without ``__init__.py``, so ``import tests.X`` resolves under one invocation
style and raises ``ModuleNotFoundError: No module named 'tests'`` under the
other -- one line taking down the entire six-job CI matrix. KBR-84 (PR #214)
paid a full CI round plus a fix commit for exactly this, on a contract test's
self-guard. Memories capture the falsification channel; memories do not
enforce. This guard does.

If this test fails, a test module imports the ``tests`` package. Import the
shared code the way every sibling does -- bare ``import layers``-style, via
pytest's ``prepend`` import mode putting ``tests/`` itself on ``sys.path`` --
or share it through a conftest fixture.

Scope: **direct top-level ``import`` / ``from-import`` statements** -- the
KBR-84 failure shape. Four adjacent shapes are deliberately out of scope:

* function-level imports, where the same trap lives but widening is a
  separate, conscious decision;
* ``if TYPE_CHECKING:``-guarded imports, which never execute at runtime under
  either invocation style and would be a false positive;
* top-level runtime-guarded wrappers around ``import tests.X`` --
  ``try: import tests.X / except ModuleNotFoundError: pass`` and similar
  ``if``/``with``/``while False:`` shapes -- where the author has explicitly
  handled the failure mode and the guard would be wrong to refuse;
* dynamic ``importlib.import_module("tests.X")``, which an import-statement
  matcher cannot see at all.

Boundaries 1--3 are pinned by negative-control tests below; boundary 4 is a
documented limit (no negative control is possible against code the matcher
does not parse).

Stated limit: the rationale above assumes ``tests/`` remains a rootdir without
``__init__.py``. Adding one would break the bare-import convention
(``import layers``, ``import internal_key_scan``) this guard's rationale rests
on and would invalidate the KBR-84 hazard model -- it is not this ticket's
job to police the absence.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# L2: the subject of this file is a structural scan of the suite's own source
# text -- an artifact edited separately from any code under ``src/kitty`` that
# the scan must agree with. It gates pull requests in the ``l1 or l2`` job; the
# marker records which half of that expression it answers to, and keeps a
# source-text scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

TESTS_DIR = Path(__file__).resolve().parent


def _iter_top_level_tests_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return every top-level import of the ``tests`` package in ``tree``.

    Args:
        tree: A parsed module (``ast.parse`` output).

    Returns:
        ``(lineno, rendered statement)`` for each top-level ``import tests`` /
        ``import tests.X`` and ``from tests import ...`` /
        ``from tests.X import ...``. Empty when the module is clean.

    Only ``tree.body`` is inspected -- **direct** top-level ``import`` /
    ``from-import`` statements, not ``ast.walk``. A top-level
    ``try: import tests.X / except ...`` is therefore not flagged: it is
    ``ast.Try`` in ``tree.body``, with the offending ``Import`` one level
    deeper, and the author has explicitly handled the failure mode (see the
    module docstring for the four out-of-scope shapes). ``ast.ImportFrom``
    matches only absolute imports (``level == 0``): ``from .tests import x``
    carries ``module == "tests"`` too, but its ``level == 1`` names a sibling
    of the *current* package, not the top-level ``tests`` package.
    """
    found: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tests" or alias.name.startswith("tests."):
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "tests" or module.startswith("tests.")):
                names = ", ".join(alias.name for alias in node.names)
                found.append((node.lineno, f"from {module} import {names}"))
    return found


def _scanned_files() -> list[Path]:
    """Return every ``tests/**/*.py`` path the live scan walks.

    Single source of the scan's population. ``_scan_tests_tree`` walks exactly
    these files, and the AC4 self-guards assert over the same list, so a
    regression in the walk (a narrowed glob, a broken root) fails every caller
    instead of passing vacuously -- the shared-helper property the §1.4
    harness rule requires.
    """
    return sorted(TESTS_DIR.rglob("*.py"))


def _scan_tests_tree() -> list[tuple[Path, int, str]]:
    """Scan every ``tests/**/*.py`` module for top-level ``tests`` imports.

    Returns:
        ``(path, lineno, statement)`` per offender, sorted by path; empty when
        the tree is clean. The guard file itself sits inside the scanned
        population, so a future ``import tests.X`` written here is flagged by
        the guard's own live scan.
    """
    offenders: list[tuple[Path, int, str]] = []
    for path in _scanned_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, statement in _iter_top_level_tests_imports(tree):
            offenders.append((path, lineno, statement))
    return offenders


def _format_offenders(offenders: list[tuple[Path, int, str]], root: Path) -> str:
    """Render offenders as ``relative/path.py:lineno -- statement`` lines.

    Args:
        offenders: The ``(path, lineno, statement)`` tuples the scan produced.
        root: The directory the paths are reported relative to.

    Returns:
        One line per offender, joined with newlines plus a two-space indent so
        the assertion message reads as a list. Empty string for no offenders.
    """
    return "\n  ".join(
        f"{path.relative_to(root).as_posix()}:{lineno} -- {statement}" for path, lineno, statement in offenders
    )


class TestNoTestModuleImportsTheTestsPackage:
    """The live guard: the tree must stay clean."""

    def test_no_test_module_imports_the_tests_package_at_top_level(self):
        """Any top-level ``tests`` import fails the suite, naming file and line."""
        offenders = _scan_tests_tree()

        assert not offenders, (
            "a test module imports the `tests` package at top-level. Bare `pytest` "
            "-- the CI fast-gate invocation -- cannot resolve it, and one line takes "
            "down the whole six-job matrix (KBR-84 paid a full CI round for this). "
            "Import the sibling module directly (`import layers`, not "
            "`import tests.layers`), or share code through a conftest fixture:\n  "
            + _format_offenders(offenders, TESTS_DIR)
        )


class TestTheFailureMessageNamesTheSite:
    """The report must be actionable without opening the scanner.

    A maintainer reading a red CI log needs file and line in the message; a
    guard that fails with "an import exists somewhere" costs a round-trip.
    Pinned mechanically on a synthetic offender so a formatting regression
    cannot silently degrade the report.
    """

    def test_the_formatted_report_carries_relative_path_line_and_statement(self):
        """One synthetic offender renders as ``rel/path.py:LINE -- STATEMENT``."""
        synthetic = (TESTS_DIR / "test_example.py", 7, "import tests.helper")

        rendered = _format_offenders([synthetic], TESTS_DIR)

        assert rendered == "test_example.py:7 -- import tests.helper"


class TestTheScanRunsOverTheRealTree:
    """Self-guards, per the §1.4 harness rule.

    A scan that could pass over an empty tree proves nothing: a broken glob or
    a renamed directory would turn the guard into a silent no-op while every
    assertion above stays green.
    """

    def test_the_scan_covers_a_non_empty_population(self):
        """The ``tests/`` glob must resolve to at least the guard file itself."""
        files = _scanned_files()

        assert len(files) >= 1, (
            f"the scan found no test modules under {TESTS_DIR}; the glob is broken "
            "and the guard above passed over an empty tree"
        )

    def test_the_guard_file_is_inside_the_scanned_population(self):
        """The guard polices itself: its own file must be among the scanned paths."""
        me = Path(__file__).resolve()

        assert me in set(_scanned_files()), (
            "the guard file is not inside the scanned population; the scan cannot police the tree it lives in"
        )


class TestTheMatcherCatchesTheTrap:
    """Positive controls, per the §1.4 harness rule.

    There is no live offender in the tree -- the KBR-84 fix removed the only
    real one -- so these synthetic cases are the guard's positive controls,
    mirroring ``tests/bridge/test_vendor_token_guard.py``'s historical-M13
    pattern. Each probe feeds a deliberate defect through the production
    matcher (shared-helper rule: a regression in the matcher fails every
    caller, not only these probes' copy of the loop).
    """

    def test_top_level_dotted_import_is_flagged(self):
        """``import tests.helper`` -- the KBR-84 shape -- is reported."""
        tree = ast.parse("import tests.helper\n")

        assert _iter_top_level_tests_imports(tree) == [(1, "import tests.helper")]

    def test_top_level_bare_package_import_is_flagged(self):
        """Even bare ``import tests`` -- which binds nothing usable -- is reported."""
        tree = ast.parse("import tests\n")

        assert _iter_top_level_tests_imports(tree) == [(1, "import tests")]

    def test_top_level_from_tests_import_is_flagged(self):
        """``from tests import helper`` is reported."""
        tree = ast.parse("from tests import helper\n")

        assert _iter_top_level_tests_imports(tree) == [(1, "from tests import helper")]

    def test_top_level_from_tests_subpackage_import_is_flagged(self):
        """``from tests.bridge import helpers`` is reported."""
        tree = ast.parse("from tests.bridge import helpers\n")

        assert _iter_top_level_tests_imports(tree) == [(1, "from tests.bridge import helpers")]

    def test_parenthesised_from_tests_import_is_flagged(self):
        """The multi-line parenthesised spelling resolves to the same statement."""
        source = "from tests import (\n    helper_a,\n    helper_b,\n)\n"
        tree = ast.parse(source)

        assert _iter_top_level_tests_imports(tree) == [(1, "from tests import helper_a, helper_b")]

    def test_both_spellings_in_one_module_are_all_reported(self):
        """Two offenders in one module are both surfaced, not just the first."""
        source = "import tests.helper\nimport ast\nfrom tests import other\n"
        tree = ast.parse(source)

        assert _iter_top_level_tests_imports(tree) == [
            (1, "import tests.helper"),
            (3, "from tests import other"),
        ]


class TestTheMatcherSpellsTheScopeBoundary:
    """What the guard deliberately does NOT flag, and why each is safe.

    Every negative control here is a boundary the module docstring states;
    pinning them in code keeps the docstring honest -- a future widening of
    the matcher must turn one of these red and force the decision into the
    open rather than arriving silently.
    """

    def test_unrelated_top_level_imports_are_not_flagged(self):
        """The sibling-import spelling the docstring recommends stays clean."""
        source = "import ast\nimport layers\nfrom internal_key_scan import scan\n"
        tree = ast.parse(source)

        assert _iter_top_level_tests_imports(tree) == []

    def test_function_level_import_is_out_of_scope(self):
        """The same trap inside a function body is not this guard's scope.

        The ticket scopes the guard to module top-level -- where the KBR-84
        failure lived. Widening to nested scopes is a deliberate follow-up
        decision, not a silent one; if it is ever taken, this test is the one
        that must be consciously rewritten.
        """
        source = "def helper():\n    import tests.helper\n    return tests.helper\n"
        tree = ast.parse(source)

        assert _iter_top_level_tests_imports(tree) == []

    def test_type_checking_guarded_import_is_not_flagged(self):
        """A runtime-dead ``TYPE_CHECKING`` import is harmless under both styles."""
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import tests.helper\n"
        tree = ast.parse(source)

        assert _iter_top_level_tests_imports(tree) == []

    def test_relative_import_of_a_tests_named_module_is_not_flagged(self):
        """``from .tests import x`` names a sibling of the current package.

        Its ``ast.ImportFrom.module`` reads ``"tests"`` but its ``level == 1``
        makes it a relative import -- not the top-level ``tests`` package this
        guard governs. (It is broken under both invocation styles anyway,
        because ``tests/`` carries no ``__init__.py``; that is a different
        defect with a different error.)
        """
        tree = ast.parse("from .tests import x\n")

        assert _iter_top_level_tests_imports(tree) == []

    def test_try_guarded_top_level_import_is_not_flagged(self):
        """A runtime-guarded import is the author's explicit failure handling.

        ``try: import tests.X / except ModuleNotFoundError: pass`` at top
        level is an ``ast.Try`` in ``Module.body`` with the ``Import`` one
        level deeper; the matcher walks ``Module.body`` only, so it does not
        flag this shape. Widening the walker into ``Try`` / ``If`` / ``With``
        bodies would change the guard's contract; the design doc states this
        boundary explicitly. (The shape is also self-documenting -- the
        ``except`` clause tells the next reader the failure was deliberate.)
        """
        source = "try:\n    import tests.helper\nexcept ModuleNotFoundError:\n    pass\n"
        tree = ast.parse(source)

        assert _iter_top_level_tests_imports(tree) == []
