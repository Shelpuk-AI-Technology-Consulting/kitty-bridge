"""The route helpers must have exactly one source for the model they route on.

KBR-127.  ``BridgeServer._build_upstream_url`` and ``_build_upstream_headers``
resolved the upstream path and the auth scheme from ``self._active_model`` —
the profile's model, never normalized — while the body was built from
``cc_request["model"]``.  Two sources for one question, and they disagreed
whenever the profile model carried a provider prefix, or whenever there was no
profile model at all.

``tests/bridge/test_upstream_route_resolution.py`` proves the route resolves
correctly today.  This file proves it cannot quietly stop being, which is a
different claim: KBR-127 was not a wrong rule, it was a *second* source for a
rule that already existed, and a second source can be reintroduced by a change
that looks locally reasonable.

**What this guard does not prove.**  Three things, deliberately, so a green run
is not over-read:

1. **Semantics.**  It reads identifiers, not meaning.  A helper that reached the
   profile model through some path none of the forbidden names covers would pass.
   The three shapes on the list — the property, its backing field, and the
   backends table — are the ones reachable from ``BridgeServer`` today.
2. **Runtime replacement.**  ``tests/bridge/test_compaction_failure_response.py``
   binds the real method to a local name and monkeypatches a stand-in over it.
   Neither is a call this sweep can see; a stand-in with the wrong signature is
   caught by that test raising ``TypeError``, not by this file.
3. **That the route is right.**  Only that it is resolved from the request.  The
   behavioural half of that claim lives in the sibling file named above.

L2 rather than L1: ``tests/test_egress_coverage.py`` — the file
``.system_design/TEST_SUITE.md`` 6.2 names as the pattern to copy — classifies a
structural scan of source text as L2, and keeps such a scan out of the L1 set
that mutation testing judges.  The sibling guard this ticket pairs with,
``tests/test_wire_shape_honesty.py`` (KBR-7), is L2 for the same reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

# ── The structural guard ───────────────────────────────────────────────────
#
# The tests above prove the route is resolved correctly today. This proves it
# cannot quietly stop being, which is a different claim: KBR-127 was not a
# wrong rule, it was a *second* source for a rule that already existed.

_HELPERS = ("_build_upstream_url", "_build_upstream_headers")

# `_active_model` is a property over `_model`, and `_backends[idx][2].model`
# reaches the same profile string a third way. Reading any of them reinstates
# the defect, so the guard forbids all three rather than the one the bug used.
_FORBIDDEN_ATTRIBUTES = ("_active_model", "_model", "_backends")

_SERVER_MODULE = Path(__file__).resolve().parent.parent / "src" / "kitty" / "bridge" / "server.py"
_SCANNED_TREES = (
    Path(__file__).resolve().parent.parent / "src",
    Path(__file__).resolve().parent.parent / "tests",
)


def _function_defs(tree: ast.AST, names: tuple[str, ...]) -> dict[str, ast.FunctionDef]:
    """Collect the named function definitions from a parsed module.

    Args:
        tree: A parsed module.
        names: The function names to find.

    Returns:
        The definitions found, keyed by name.  A name that is absent is simply
        missing from the mapping; callers assert on that themselves, so the
        guard reports "the method was renamed" rather than a ``KeyError``.
    """
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node

    return found


def _self_attributes_read(function: ast.FunctionDef) -> set[str]:
    """Return the ``self.<name>`` attributes a function body reads.

    Args:
        function: The definition to inspect.

    Returns:
        Every attribute name accessed on ``self``.
    """
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }


def _reads_name(function: ast.FunctionDef, name: str) -> bool:
    """Return whether a function body reads a bare name.

    Args:
        function: The definition to inspect.
        name: The identifier to look for — here, the parameter that carries the
            request.

    Returns:
        True when the name is loaded anywhere in the body.
    """
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(function))


def _zero_argument_calls(tree: ast.AST) -> list[int]:
    """Return the line numbers of no-argument calls to either helper.

    Args:
        tree: A parsed module.

    Returns:
        One line number per offending call, in source order.
    """
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _HELPERS
        and not node.args
        and not node.keywords
    )


class TestRouteResolutionHasOneSourceOfTruth:
    """Both helpers must resolve the model from the request, and only from it.

    Each check asserts its own subject set and is exercised against a known
    defect, in the style of ``tests/test_egress_coverage.py``, so none can pass
    by having looked at nothing.  The module docstring records what the guard
    cannot see.
    """

    @pytest.fixture
    def helpers(self) -> dict[str, ast.FunctionDef]:
        """Parse ``server.py`` and return the two helper definitions.

        Returns:
            The definitions, keyed by method name.
        """
        tree = ast.parse(_SERVER_MODULE.read_text(encoding="utf-8"))
        found = _function_defs(tree, _HELPERS)

        assert set(found) == set(_HELPERS), (
            f"expected to find {list(_HELPERS)} in {_SERVER_MODULE.name}, found {sorted(found)}. "
            "A renamed method makes every check below vacuous."
        )
        return found

    @pytest.mark.parametrize("helper", _HELPERS)
    def test_the_helper_does_not_read_the_profile_model(self, helpers, helper: str) -> None:
        """Neither ``_active_model`` nor its backing field may appear (KBR-127).

        Args:
            helpers: The parsed helper definitions.
            helper: The method under test.
        """
        read = _self_attributes_read(helpers[helper])
        offending = sorted(read & set(_FORBIDDEN_ATTRIBUTES))

        assert not offending, (
            f"{helper} reads self.{' / self.'.join(offending)}. That is the second source of "
            "truth KBR-127 removed: the profile model is not normalized, so the route it "
            "resolves can disagree with the body translate_to_upstream builds."
        )

    @pytest.mark.parametrize("helper", _HELPERS)
    def test_the_helper_reads_the_request_it_was_given(self, helpers, helper: str) -> None:
        """Absence is not enough — the helper must read ``cc_request``.

        A helper that read nothing at all, returning a hardcoded route, would
        satisfy the absence check while being just as wrong.

        Args:
            helpers: The parsed helper definitions.
            helper: The method under test.
        """
        parameters = [arg.arg for arg in helpers[helper].args.args]

        assert "cc_request" in parameters, f"{helper} no longer takes cc_request; it takes {parameters}"
        assert _reads_name(helpers[helper], "cc_request"), (
            f"{helper} takes cc_request and never reads it, so the route is resolved from "
            "something other than the request it is for."
        )

    def test_no_call_site_resolves_a_route_without_naming_the_request(self) -> None:
        """No zero-argument call of either helper survives in ``src/`` or ``tests/``.

        CI type-checks ``src/kitty`` only, so a stale call in a test file is a
        ``TypeError`` raised whenever that branch happens to run.  Several of
        the call sites sit in deeply nested failover branches, which is exactly
        where "whenever that branch happens to run" means "in production".
        """
        offending: list[str] = []
        scanned = 0

        for tree_root in _SCANNED_TREES:
            for path in sorted(tree_root.rglob("*.py")):
                scanned += 1
                for lineno in _zero_argument_calls(ast.parse(path.read_text(encoding="utf-8"))):
                    offending.append(f"{path.relative_to(tree_root.parent)}:{lineno}")

        assert scanned > 100, f"the sweep found only {scanned} Python files; it is not reading the tree"
        assert not offending, "these call sites resolve a route without naming a request:\n  " + "\n  ".join(offending)

    @pytest.mark.parametrize(
        ("defect", "source"),
        [
            (
                "reads the profile model through the property",
                "def _build_upstream_url(self, cc_request):\n    return self._active_model or ''\n",
            ),
            (
                "reads the profile model through its backing field",
                "def _build_upstream_url(self, cc_request):\n    return self._model or ''\n",
            ),
            (
                "reads the profile model out of the backends table",
                "def _build_upstream_url(self, cc_request):\n"
                "    return self._backends[self._current_backend_idx][2].model\n",
            ),
        ],
    )
    def test_the_absence_check_detects_a_known_defect(self, defect: str, source: str) -> None:
        """The absence check must report each shape of the defect it exists to catch.

        Args:
            defect: What the synthetic function does wrong.
            source: A function carrying that defect.
        """
        function = _function_defs(ast.parse(source), _HELPERS)["_build_upstream_url"]

        assert _self_attributes_read(function) & set(_FORBIDDEN_ATTRIBUTES), (
            f"the absence check did not notice a helper that {defect}"
        )

    def test_the_presence_check_detects_a_helper_that_reads_nothing(self) -> None:
        """A hardcoded route passes the absence check, so the presence check must fail it."""
        source = "def _build_upstream_headers(self, cc_request):\n    return {'Authorization': 'Bearer x'}\n"
        function = _function_defs(ast.parse(source), _HELPERS)["_build_upstream_headers"]

        assert not _self_attributes_read(function) & set(_FORBIDDEN_ATTRIBUTES), "precondition: no forbidden read"
        assert not _reads_name(function, "cc_request"), (
            "the presence check did not notice a helper that ignores the request it was given"
        )

    def test_the_call_site_sweep_detects_a_zero_argument_call(self) -> None:
        """The sweep must report a no-argument call, or it is a no-op over a clean tree."""
        source = "url = self._build_upstream_url()\nheaders = self._build_upstream_headers(cc_request)\n"

        assert _zero_argument_calls(ast.parse(source)) == [1]
