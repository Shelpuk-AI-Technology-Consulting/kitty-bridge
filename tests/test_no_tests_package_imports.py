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

Scope: **every absolute ``import tests[.X]`` / ``from tests[.X] import ...``
statement** at module top level or inside any nested ``def``, ``async def``,
or ``class`` body. The walker descends into exactly those three Python
scope statements (an explicit allowlist, not a denylist); every other
statement-bearing node is a leaf, so an import inside a runtime guard,
loop, or ``match`` body is not visited. Three adjacent shapes are
deliberately out of scope:

* (a) ``if TYPE_CHECKING:``-guarded imports, which never execute at runtime
  under either invocation style and would be a false positive;
* (b) runtime-guarded wrappers around ``import tests.X`` -- the
  failure-avoiding spellings ``try``/``except``, ``if False:``,
  ``while False:`` and ``with contextlib.suppress(ModuleNotFoundError):``,
  with their ``async`` / ``*Star`` siblings -- where the author has
  explicitly handled the failure mode and the guard would be wrong to
  refuse. The KBR-280 contract pinned this exemption at top level; KBR-282
  extends it to function depth (the matcher's allowlist skips these nodes
  at every nesting level). A plain ``with`` whose manager does not
  suppress the failure is skipped by the same leaf rule but is not itself
  failure handling -- it is an executable position the walker does not
  visit, in the same category as stated limit 2;
* (c) dynamic ``importlib.import_module("tests.X")``, which an
  import-statement matcher cannot see at all.

Out-of-scope shapes (a) and (b) are pinned by negative-control tests
below; shape (c) is a documented limit (no negative control is possible
against code the matcher does not parse).

Stated limits -- shapes the walker does not see, with the reason for
each:

1. Whole-node skip on guard statements covers their ``orelse`` and
   ``finalbody`` arms, so an unconditionally executing ``import tests.X``
   inside ``finally:`` or ``else:`` is not flagged. The skip rationale is
   "the whole statement is a guard", not "inspect every arm"; descent
   into guard arms would re-implement a runtime-guard detector inside the
   matcher, out of scope for an ``import tests.*`` guard.
2. The three scope types are the only nodes the walker descends into; an
   ``import tests.X`` inside a ``for``, ``async for``, ``while``, or
   ``match`` body is not flagged. The ticket enumerates only the three
   scope types; widening further is a separate conscious decision. The
   next widening, when it comes, must add ``For``/``AsyncFor``/``Match``
   explicitly (and re-state the runtime-guard contract uniformly).
3. A scope statement itself wrapped in a guard (``if flag: def f():
   import tests.X``) is invisible: the walker never reaches the ``def``,
   because the ``If`` body is not descended into. Same limitation as (2).
4. ``importlib.import_module("tests.X")`` -- invisible to any
   import-statement matcher (unchanged).

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


def _tests_imports_in_scope(body: list[ast.stmt]) -> list[tuple[int, str]]:
    """Collect every ``tests``-package import in one scope's statement list.

    Args:
        body: The ``body`` list of an ``ast.Module``, ``FunctionDef``,
            ``AsyncFunctionDef``, or ``ClassDef``.

    Returns:
        ``(lineno, rendered statement)`` pairs for each ``import tests`` /
        ``import tests.X`` and ``from tests import ...`` /
        ``from tests.X import ...`` statement at this scope level, plus
        the same for any deeper scope reached by recursing into nested
        ``FunctionDef``, ``AsyncFunctionDef``, or ``ClassDef`` bodies.

    The walker descends into exactly the three Python scope statements
    (``FunctionDef``, ``AsyncFunctionDef``, ``ClassDef``) and into nothing
    else. Every other statement-bearing node -- ``If`` / ``For`` /
    ``AsyncFor`` / ``While`` / ``With`` / ``AsyncWith`` / ``Try`` /
    ``TryStar`` / ``Match`` and the ``orelse`` / ``finalbody`` arms of
    guard statements -- is a leaf inspected for direct ``Import`` /
    ``ImportFrom`` statements only (no descent). This is an explicit
    allowlist, not a denylist: a denylist reading would flag an
    ``async with`` or ``except*`` guard, breaking the runtime-guard
    contract (see the module docstring's stated-limit list for the
    shapes the walker therefore does not see). ``ast.ImportFrom``
    matches only absolute imports (``level == 0``): ``from .tests
    import x`` carries ``module == "tests"`` too, but its ``level == 1``
    names a sibling of the *current* package, not the top-level
    ``tests`` package.
    """
    found: list[tuple[int, str]] = []
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tests" or alias.name.startswith("tests."):
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "tests" or module.startswith("tests.")):
                names = ", ".join(alias.name for alias in node.names)
                found.append((node.lineno, f"from {module} import {names}"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.extend(_tests_imports_in_scope(node.body))
    return found


def _iter_tests_package_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return every ``tests``-package import statement in ``tree``.

    Args:
        tree: A parsed module (``ast.parse`` output).

    Returns:
        ``(lineno, rendered statement)`` for each ``import tests`` /
        ``import tests.X`` and ``from tests import ...`` /
        ``from tests.X import ...`` in any module, function, async-function,
        or class body the walker descends into. Empty when the module is
        clean.

    The walk is an explicit allowlist of three Python scope statements
    (``FunctionDef``, ``AsyncFunctionDef``, ``ClassDef``); every other
    statement-bearing node is a leaf. Runtime-guarded positions
    (``try``/``except``, ``if``, ``with``, ``while`` and their ``async``
    / ``*Star`` siblings), loop and ``match`` bodies, and the ``orelse`` /
    ``finalbody`` arms of guard statements are therefore never visited.
    See the module docstring for the four stated-limit shapes the walker
    does not see; the runtime-guard boundary is pinned by KBR-280's
    top-level negative controls and KBR-282's parameterised function-depth
    control.
    """
    return _tests_imports_in_scope(tree.body)


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
    """Scan every ``tests/**/*.py`` module for ``tests``-package imports.

    Returns:
        ``(path, lineno, statement)`` per offender, sorted by path; empty when
        the tree is clean. The guard file itself sits inside the scanned
        population, so a future ``import tests.X`` written here is flagged by
        the guard's own live scan.
    """
    offenders: list[tuple[Path, int, str]] = []
    for path in _scanned_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, statement in _iter_tests_package_imports(tree):
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

    def test_no_test_module_imports_the_tests_package(self):
        """Any ``tests``-package import, top-level or nested, fails the suite."""
        offenders = _scan_tests_tree()

        assert not offenders, (
            "a test module imports the `tests` package. Bare `pytest` "
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

        assert _iter_tests_package_imports(tree) == [(1, "import tests.helper")]

    def test_top_level_bare_package_import_is_flagged(self):
        """Even bare ``import tests`` -- which binds nothing usable -- is reported."""
        tree = ast.parse("import tests\n")

        assert _iter_tests_package_imports(tree) == [(1, "import tests")]

    def test_top_level_from_tests_import_is_flagged(self):
        """``from tests import helper`` is reported."""
        tree = ast.parse("from tests import helper\n")

        assert _iter_tests_package_imports(tree) == [(1, "from tests import helper")]

    def test_top_level_from_tests_subpackage_import_is_flagged(self):
        """``from tests.bridge import helpers`` is reported."""
        tree = ast.parse("from tests.bridge import helpers\n")

        assert _iter_tests_package_imports(tree) == [(1, "from tests.bridge import helpers")]

    def test_parenthesised_from_tests_import_is_flagged(self):
        """The multi-line parenthesised spelling resolves to the same statement."""
        source = "from tests import (\n    helper_a,\n    helper_b,\n)\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(1, "from tests import helper_a, helper_b")]

    def test_both_spellings_in_one_module_are_all_reported(self):
        """Two offenders in one module are both surfaced, not just the first."""
        source = "import tests.helper\nimport ast\nfrom tests import other\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [
            (1, "import tests.helper"),
            (3, "from tests import other"),
        ]

    def test_function_level_import_is_flagged(self):
        """``import tests.helper`` inside a function body is reported.

        KBR-280 scoped the guard to module top-level and deferred widening
        as "a separate, conscious decision"; KBR-282 is that decision. The
        function-level shape is the same KBR-84 trap one scope deeper: the
        module imports cleanly, then any call to the helper raises
        ``ModuleNotFoundError`` under CI's bare ``pytest``. This test is
        the conscious rewrite of KBR-280's negative control of the same
        shape -- its docstring names this rewrite as the tripwire.
        """
        source = "def helper():\n    import tests.helper\n    return tests.helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(2, "import tests.helper")]

    def test_class_body_import_is_flagged(self):
        """``from tests import helper`` in a class body is reported."""
        source = "class Fixture:\n    from tests import helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(2, "from tests import helper")]

    def test_async_function_level_import_is_flagged(self):
        """``import tests.helper`` in an ``async def`` body is reported."""
        source = "async def helper():\n    import tests.helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(2, "import tests.helper")]

    def test_import_inside_a_method_is_flagged(self):
        """A ``tests`` import in a method (class -> function) is reported."""
        source = "class Fixture:\n    def helper(self):\n        import tests.helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(3, "import tests.helper")]

    def test_import_inside_a_nested_function_is_flagged(self):
        """A ``tests`` import in a function nested in a function is reported."""
        source = "def outer():\n    def inner():\n        import tests.helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(3, "import tests.helper")]

    def test_import_inside_a_nested_class_is_flagged(self):
        """A ``tests`` import in a class nested in a class is reported."""
        source = "class Outer:\n    class Inner:\n        import tests.helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == [(3, "import tests.helper")]


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

        assert _iter_tests_package_imports(tree) == []

    def test_type_checking_guarded_import_is_not_flagged(self):
        """A runtime-dead ``TYPE_CHECKING`` import is harmless under both styles."""
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import tests.helper\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == []

    def test_relative_import_of_a_tests_named_module_is_not_flagged(self):
        """``from .tests import x`` names a sibling of the current package.

        Its ``ast.ImportFrom.module`` reads ``"tests"`` but its ``level == 1``
        makes it a relative import -- not the top-level ``tests`` package this
        guard governs. (It is broken under both invocation styles anyway,
        because ``tests/`` carries no ``__init__.py``; that is a different
        defect with a different error.)
        """
        tree = ast.parse("from .tests import x\n")

        assert _iter_tests_package_imports(tree) == []

    def test_relative_import_inside_a_function_is_not_flagged(self):
        """``from .tests import x`` at function depth stays unflagged.

        KBR-280 pinned this shape at top level (``level == 1`` names a
        sibling of the current package); KBR-282 extends the same
        exemption to nested scopes so the ``ImportFrom`` exemption is
        not silently depth-bound.
        """
        tree = ast.parse("def f():\n    from .tests import helper\n")

        assert _iter_tests_package_imports(tree) == []

    @pytest.mark.parametrize(
        "source",
        [
            "def f():\n    try:\n        import tests.helper\n    except ModuleNotFoundError:\n        pass\n",
            "def f():\n    if False:\n        import tests.helper\n",
            "import contextlib\n"
            "def f():\n    with contextlib.suppress(ModuleNotFoundError):\n        import tests.helper\n",
            "def f():\n    while False:\n        import tests.helper\n",
        ],
        ids=["try", "if_false", "with", "while_false"],
    )
    def test_runtime_guarded_import_inside_a_function_is_not_flagged(self, source: str):
        """A runtime-guarded ``tests`` import at function depth is not reported.

        Pinned for all four base guard shapes (``try``/``except``,
        ``if False:``, ``with contextlib.suppress(ModuleNotFoundError):``,
        ``while False:``) at function depth so a regression deleting any
        one from the skip set turns this boundary red and forces the
        decision into the open. All four spellings genuinely avoid the
        runtime failure, so the guard framing holds for every param.
        KBR-280 pinned the ``try`` shape at top level and documented the
        other three; KBR-282 extends the contract to function depth.
        """
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == []

    def test_try_guarded_top_level_import_is_not_flagged(self):
        """A runtime-guarded import is the author's explicit failure handling.

        ``try: import tests.X / except ModuleNotFoundError: pass`` at top
        level is an ``ast.Try`` in ``Module.body``; the walker treats
        ``Try`` as a leaf, so the ``Import`` one level deeper is never
        visited and the shape is not flagged. Descending into ``Try`` /
        ``If`` / ``With`` bodies would change the guard's contract; the
        design doc states this boundary explicitly. (The shape is also
        self-documenting -- the ``except`` clause tells the next reader the
        failure was deliberate.)
        """
        source = "try:\n    import tests.helper\nexcept ModuleNotFoundError:\n    pass\n"
        tree = ast.parse(source)

        assert _iter_tests_package_imports(tree) == []
