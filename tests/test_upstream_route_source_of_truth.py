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
2. **A wrong argument.**  The sweep proves a call *names* a request, never that
   it names the *right* one.  ``self._build_upstream_url(body)`` or a stale dict
   passes this guard and passes ``mypy`` too — both are ``dict``.  That is the
   residual risk of a mechanical edit across 46 sites, and it is carried by
   reading the call sites, not by this file.
3. **Runtime replacement.**  ``tests/bridge/test_compaction_failure_response.py``
   monkeypatches a stand-in over ``_build_upstream_headers``.  The *alias* it
   binds first is covered — :func:`_helper_aliases` exists for that site — but a
   ``monkeypatch.setattr`` whose replacement takes no argument is not, because
   the helper's name reaches it only as a string literal.  Worse, that stand-in
   is never invoked at all: the same test replaces ``_make_upstream_request``,
   so nothing downstream of it runs.  So no test in the tree exercises that
   stand-in, and it is not this guard's job to pretend otherwise.
4. **That the route is right.**  Only that it is resolved from the request.  The
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

# The one function that answers "which model is this request for?". Both
# helpers must go through it; a helper that re-derives the answer inline is
# a second source of truth again, whatever it derives it from.
_ROUTE_MODEL = "_route_model"

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


def _calls(function: ast.FunctionDef, name: str) -> bool:
    """Return whether a function body calls a bare-named function.

    Args:
        function: The definition to inspect.
        name: The function name to look for.

    Returns:
        True when the body contains a call to that name.
    """
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(function)
    )


def _helper_aliases(tree: ast.AST) -> set[str]:
    """Return local names bound to either helper without calling it.

    ``real_headers = server._build_upstream_headers`` is how
    ``tests/bridge/test_compaction_failure_response.py`` reaches the real method
    before monkeypatching a stand-in over it.  A call through that name is a
    call to the helper, and a sweep that only matched attribute calls would step
    straight past the one site in the tree that takes this shape.

    Args:
        tree: A parsed module.

    Returns:
        Every name bound to an un-called helper attribute.
    """
    aliases: set[str] = set()

    for node in ast.walk(tree):
        # Both spellings: `real = obj.helper` and `real: Callable = obj.helper`.
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Attribute) or value.attr not in _HELPERS:
            continue
        aliases.update(t.id for t in targets if isinstance(t, ast.Name))

    return aliases


def _zero_argument_calls(tree: ast.AST) -> list[int]:
    """Return the line numbers of no-argument calls to either helper.

    Matches both shapes a call can take: directly on an object, and through a
    local name bound to the method (:func:`_helper_aliases`).

    Args:
        tree: A parsed module.

    Returns:
        One line number per offending call, in source order.
    """
    aliases = _helper_aliases(tree)

    def _is_helper(call: ast.Call) -> bool:
        """Return whether a call reaches either helper, by either shape."""
        if isinstance(call.func, ast.Attribute):
            return call.func.attr in _HELPERS
        return isinstance(call.func, ast.Name) and call.func.id in aliases

    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_helper(node) and not node.args and not node.keywords
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
    def test_the_helper_resolves_the_model_through_the_one_function_that_owns_it(
        self, helpers, helper: str
    ) -> None:
        """Absence is not enough — the helper must call :func:`_route_model`.

        A helper that read nothing at all and returned a hardcoded route would
        satisfy the absence check while being just as wrong.  Requiring the call
        is stricter than requiring a read of ``cc_request``, which any incidental
        lookup — ``cc_request.get("stream")`` — would satisfy.

        Args:
            helpers: The parsed helper definitions.
            helper: The method under test.
        """
        parameters = [arg.arg for arg in helpers[helper].args.args]

        assert "cc_request" in parameters, f"{helper} no longer takes cc_request; it takes {parameters}"
        assert _calls(helpers[helper], _ROUTE_MODEL), (
            f"{helper} does not call {_ROUTE_MODEL}(). The model every routing decision "
            "reads is defined in exactly one place; a helper that re-derives it inline is "
            "the second source of truth KBR-127 removed."
        )

    def test_no_call_site_resolves_a_route_without_naming_the_request(self) -> None:
        """No zero-argument call of either helper survives in ``src/`` or ``tests/``.

        CI type-checks ``src/kitty`` only, so a stale call in a test file is a
        ``TypeError`` raised whenever that branch happens to run.  Several of
        the call sites sit in deeply nested failover branches, which is exactly
        where "whenever that branch happens to run" means "in production".
        """
        offending: list[str] = []
        scanned: dict[str, int] = {}

        for tree_root in _SCANNED_TREES:
            paths = sorted(tree_root.rglob("*.py"))
            scanned[tree_root.name] = len(paths)
            for path in paths:
                for lineno in _zero_argument_calls(ast.parse(path.read_text(encoding="utf-8"))):
                    offending.append(f"{path.relative_to(tree_root.parent)}:{lineno}")

        # Per tree, not a total: an aggregate threshold stays satisfied by one
        # tree alone, so dropping the other from _SCANNED_TREES would go unseen.
        empty = sorted(name for name, count in scanned.items() if not count)
        assert not empty, f"the sweep read no Python files under {empty}; it is not reading the tree"
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

    @pytest.mark.parametrize(
        ("defect", "body"),
        [
            ("returns a hardcoded route", "    return {'Authorization': 'Bearer x'}"),
            ("re-derives the model inline", "    return cc_request.get('model', '')"),
        ],
    )
    def test_the_presence_check_detects_a_helper_that_bypasses_the_owner(self, defect: str, body: str) -> None:
        """Both bypass shapes must fail the presence check.

        The hardcoded one reads nothing and the inline one reads the right key
        the wrong way — a second copy of the rule, which is the thing KBR-127
        removed rather than the value it produced.

        Args:
            defect: What the synthetic helper does wrong.
            body: Its body.
        """
        source = f"def _build_upstream_headers(self, cc_request):\n{body}\n"
        function = _function_defs(ast.parse(source), _HELPERS)["_build_upstream_headers"]

        assert not _self_attributes_read(function) & set(_FORBIDDEN_ATTRIBUTES), "precondition: no forbidden read"
        assert not _calls(function, _ROUTE_MODEL), f"the presence check did not notice a helper that {defect}"

    def test_the_call_site_sweep_detects_a_zero_argument_call(self) -> None:
        """The sweep must report a no-argument call, or it is a no-op over a clean tree."""
        source = "url = self._build_upstream_url()\nheaders = self._build_upstream_headers(cc_request)\n"

        assert _zero_argument_calls(ast.parse(source)) == [1]

    @pytest.mark.parametrize(
        ("spelling", "binding"),
        [
            ("bare", "real_headers = server._build_upstream_headers"),
            ("annotated", "real_headers: Callable = server._build_upstream_headers"),
        ],
    )
    def test_the_call_site_sweep_detects_a_zero_argument_call_through_an_alias(
        self, spelling: str, binding: str
    ) -> None:
        """The aliased shape must be reported too, or the one site using it is invisible.

        Args:
            spelling: How the alias is bound.
            binding: The binding statement.
        """
        source = (
            f"{binding}\n"
            "captured.update(real_headers())\n"
            "kept = server._build_upstream_url\n"
            "url = kept(cc_request)\n"
        )

        assert _zero_argument_calls(ast.parse(source)) == [2], f"the {spelling} alias was not seen"
