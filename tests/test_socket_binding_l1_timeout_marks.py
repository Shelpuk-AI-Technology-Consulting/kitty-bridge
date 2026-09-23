"""The socket-binding L1 timeout registry: every real-socket module is bounded.

``.system_design/TEST_SUITE.md`` section 8.2 enumerates the test modules that
bind real sockets or spawn real processes and so run under the ``l1`` path
default today. Several of them stalled mutmut's clean-test phase indefinitely
under workstation contention (KBR-266): a socket wait has no per-test bound
unless the module carries one. ``pytest-timeout`` supplies that bound, and this
module is its registry -- the machine-readable form of section 8.2, the same
shape ``tests/mutmut_scope.py`` is for section 6.1.

Every listed module must carry ``pytestmark = pytest.mark.timeout(120)``. The
value clears section 8.2's measured worst legitimate case (72 s: a fired
failure ladder plus the teardown that waits it out in
``tests/harness/test_provider_aiohttp.py``) with headroom for runner load; the
ticket's 60 s example sits below that case and would turn a legitimate
failure-path cost into a spurious timeout.

The sources this registry mirrors are two, because section 8.2 names the
socket-binding set in two places: the bullet list, and the KBR-10 paragraph
that follows it (``tests/cli/test_stream_encoding.py``, described separately
from the bullets ever since it landed). A module added to either place must
join this list, and a module on this list that loses its mark fails the suite
here rather than hanging a mutation run.

This is a docs-vs-code agreement guard over two real artifacts (the design
doc's enumeration and the test sources' AST), so it is a contract test: ``l2``
by marker, registered in ``tests/test_layer_selection.py``'s explicit set --
the same treatment ``tests/test_mutmut_scope.py`` gets.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from socket_binding_l1_modules import SOCKET_BINDING_L1_MODULES

pytestmark = pytest.mark.l2

REPO_ROOT = Path(__file__).resolve().parent.parent

# The section 8.2 set is imported from ``tests/socket_binding_l1_modules.py``:
# KBR-290 made that module the shared source of truth, read by this mark
# registry and by the mutmut ``--ignore`` exclusion guard
# (``tests/test_socket_binding_l1_mutation_exclusion.py``) alike, so a new
# socket-binding module joins both treatments with one edit. This file owns
# the mark half; the exclusion guard owns the ignore half.

REQUIRED_TIMEOUT_SECONDS = 120


def module_level_timeout(tree: ast.Module) -> tuple[int, int] | None:
    """Return the module-level timeout mark's ``(seconds, lineno)``, or ``None``.

    Args:
        tree: The parsed module AST.

    Returns:
        The timeout value and the assignment's line number when the module
        carries ``pytestmark = pytest.mark.timeout(N)`` (or the same mark
        inside a ``pytestmark`` list), otherwise ``None``. Module-level only:
        a per-test ``@pytest.mark.timeout`` decorator does not bound the
        hang-prone fixtures and teardowns this registry exists to bound.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        values: list[ast.expr]
        # A bare `pytestmark = pytest.mark.timeout(N)` and the list form
        # `pytestmark = [pytest.mark.timeout(N)]` are both accepted: either
        # applies the mark to every test in the module, which is the property
        # this registry cares about.
        values = node.value.elts if isinstance(node.value, ast.List) else [node.value]
        for value in values:
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "timeout"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "mark"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "pytest"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, int)
            ):
                return value.args[0].value, node.lineno
    return None


@pytest.mark.parametrize("module_path", SOCKET_BINDING_L1_MODULES)
def test_socket_binding_l1_module_carries_the_timeout_mark(module_path: str) -> None:
    """Every §8.2 socket-binding L1 module bounds its tests at 120 seconds.

    Args:
        module_path: Repo-relative path of the module under check.

    A missing mark is what KBR-266 ran on: mutmut's clean-test phase exercised
    these modules' real sockets with no bound and the run stalled. The failure
    names the fix (the registry entry to keep or the mark to restore) so a
    red gate is actionable without this file's context.
    """
    source_path = REPO_ROOT / module_path
    assert source_path.exists(), (
        f"{module_path}: listed in the socket-binding registry but the file "
        f"is gone -- remove the entry (with the §8.2 bullet) or restore the file"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=module_path)
    found = module_level_timeout(tree)
    assert found is not None, (
        f"{module_path}: no module-level `pytestmark = pytest.mark.timeout("
        f"{REQUIRED_TIMEOUT_SECONDS})` -- the module binds real sockets (§8.2) "
        f"and an unbounded socket wait here hangs mutmut's clean-test phase "
        f"(KBR-266). Restore the mark or argue the module out of the registry."
    )
    seconds, _ = found
    assert seconds == REQUIRED_TIMEOUT_SECONDS, (
        f"{module_path}: timeout mark is {seconds}s, registry requires "
        f"{REQUIRED_TIMEOUT_SECONDS}s -- the value clears §8.2's measured 72s "
        f"worst legitimate case; raising or lowering it is a registry-wide "
        f"decision (REQUIRED_TIMEOUT_SECONDS), not a per-module one"
    )


def test_a_module_that_loses_its_mark_is_reported() -> None:
    """The checker detects a module whose mark was removed or mistyped.

    A checker proven only on passing input cannot distinguish "green because
    the property holds" from "green because the check stopped looking" (the
    meta-suite's own rule, ``tests/test_layer_markers.py``). Feeding it the
    same module body with the mark present, stripped, and mistyped proves all
    three verdicts come from the parse and not from an accident of the
    fixture data.
    """
    header = "import pytest\n\n"
    marked = header + f"pytestmark = pytest.mark.timeout({REQUIRED_TIMEOUT_SECONDS})\n"
    stripped = header
    wrong_value = header + "pytestmark = pytest.mark.timeout(60)\n"
    listed_form = header + (
        f"pytestmark = [pytest.mark.timeout({REQUIRED_TIMEOUT_SECONDS})]\n"
    )

    assert module_level_timeout(ast.parse(marked)) == (
        REQUIRED_TIMEOUT_SECONDS,
        3,
    )
    assert module_level_timeout(ast.parse(listed_form)) == (
        REQUIRED_TIMEOUT_SECONDS,
        3,
    )
    assert module_level_timeout(ast.parse(stripped)) is None
    assert module_level_timeout(ast.parse(wrong_value)) == (60, 3)
