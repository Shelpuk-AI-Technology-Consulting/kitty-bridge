"""KBR-256 — the four streaming recovery blocks stay in structural lockstep.

``.system_design/TEST_SUITE.md`` §3.2.1 row **M6** (as amended by KBR-256) ·
requirements ``.requirements/20260915T184115Z_kbr256_streaming_413_compact_retry/
REQUIREMENTS.md`` R1/R3.

The KBR-256 recovery is written once per streaming ladder — ``_stream_messages``,
``_stream_responses``, ``_stream_gemini`` and ``_stream_chat_completions`` each
carry their own inline copy, matching the repo's existing per-route duplication
of the mark-unhealthy arms. Behavioural L3 coverage in this suite reaches two of
the four copies (Messages and Chat Completions, via the aiohttp harness);
Responses and Gemini wait on the curl_cffi / botocore recorders (KBR-41/42).
That leaves the two untested copies enforced by nothing — until this module.

**What a source scan can and cannot prove.** Parsing ``src/kitty`` with
:mod:`ast` (the same discipline :mod:`tests.harness.test_register_agreement`
applies to the register's site list) pins each route's recovery *skeleton*:
the recovery call exists, it sits inside a ``try`` that catches
``CompactionFailedError`` (the no-mark fallback arm), the ladder's loop range
is extended by ``n_backends`` (the bounded budget), and the ``recovery_retries``
counter is initialised and subtracted from the attempt ordinal (the
thinking-strip shape). A deleted block, a dropped fallback, or a loop range
reverted to the pre-KBR-256 shape goes red here.

What it cannot prove is per-route *behavioural* shape — which SSE error
formatter each terminal arm uses, where each route's translator reset sits.
Those follow each route's own conventions and stay with the route's failover
arm tests and review of the verbatim copy; this module is the floor, not the
ceiling.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2

#: The parsed source. Read as text, never imported — a specification test that
#: imported the code would be satisfied by whatever the code happened to do.
SERVER_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "kitty" / "bridge" / "server.py"
).read_text(encoding="utf-8")

#: The four streaming handlers register row M6 names (tests/harness/register.py
#: carries the same four in ``MutationRow.site``). A fifth streaming surface or
#: a renamed one must update both places — the register guard and this tuple.
ROUTES: tuple[str, ...] = (
    "_stream_messages",
    "_stream_responses",
    "_stream_gemini",
    "_stream_chat_completions",
)


def _route_fn(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the module-level class method named ``name`` from the parsed source.

    Args:
        name: The method's name — one of :data:`ROUTES`.

    Returns:
        The function's AST node.

    Raises:
        AssertionError: When no function of that name exists — a renamed
            handler must update :data:`ROUTES` here and the register together.
    """
    tree = ast.parse(SERVER_SOURCE, filename=str(SERVER_SOURCE))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(
        f"{name!r} not found in server.py — if the handler was renamed, "
        "update ROUTES here and M6's site tuple in tests/harness/register.py together"
    )


def _caught_names(handler: ast.ExceptHandler) -> set[str]:
    """Return the exception names one ``except`` clause catches.

    Args:
        handler: The handler node.

    Returns:
        The caught names; empty for a bare ``except:``.
    """
    if handler.type is None:
        return set()
    return {node.id for node in ast.walk(handler.type) if isinstance(node, ast.Name)}


def _is_self_call(node: ast.AST, attr: str) -> bool:
    """Return True when ``node`` is a ``self.<attr>(...)`` call.

    Args:
        node: The AST node to inspect.
        attr: The method name to match.

    Returns:
        True when the node is such a call.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


class TestEveryStreamingRouteCarriesTheRecoverySkeleton:
    """The per-route structural pins — one test per route, parametrized."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_recovery_call_exists(self, route: str) -> None:
        """Each route calls ``_compact_with_tighter_budget`` in its ladder.

        A route whose recovery block was deleted (or whose call was renamed
        away from the M6 site the register names) fails here.

        Args:
            route: The streaming handler's name.
        """
        fn = _route_fn(route)
        calls = [node for node in ast.walk(fn) if _is_self_call(node, "_compact_with_tighter_budget")]
        assert calls, (
            f"{route}: no _compact_with_tighter_budget call found — the KBR-256 "
            "recovery block is missing or was renamed; register row M6 names "
            "this route as a recovery site"
        )

    @pytest.mark.parametrize("route", ROUTES)
    def test_recovery_sits_inside_a_compaction_failed_try(self, route: str) -> None:
        """The recovery call is guarded by a ``CompactionFailedError`` catch.

        The no-mark fallback arm (R3) exists only as this try/except: without
        it, a compaction exhaustion would propagate out of the ladder instead
        of failing over.

        Args:
            route: The streaming handler's name.
        """
        fn = _route_fn(route)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Try):
                continue
            if not any("CompactionFailedError" in _caught_names(h) for h in node.handlers):
                continue
            if any(_is_self_call(sub, "_compact_with_tighter_budget") for stmt in node.body for sub in ast.walk(stmt)):
                return
        raise AssertionError(
            f"{route}: no try/except CompactionFailedError wrapping a "
            "_compact_with_tighter_budget call — the no-mark fallback arm (R3) "
            "is missing"
        )

    @pytest.mark.parametrize("route", ROUTES)
    def test_ladder_loop_range_is_extended_by_n_backends(self, route: str) -> None:
        """The loop carrying the recovery extends its range by ``n_backends``.

        Ties the range assertion to the loop whose body actually contains the
        ``_compact_with_tighter_budget`` call — the plain-POST attempt ladder —
        because the same handler also carries pre-KBR-256 ladders that a
        name-only scan satisfies vacuously: the KBR-254 custom-transport
        dispatch loop is ``range(n_backends)`` and Messages' custom branch is
        ``range(max_attempts)`` (line 4444). Found by the negative control —
        with the ladder's extension deleted, the dispatch loop alone kept the
        first draft of this check green.

        Args:
            route: The streaming handler's name.
        """
        fn = _route_fn(route)
        recovery_loop_found = False
        for node in ast.walk(fn):
            if not isinstance(node, (ast.For, ast.AsyncFor)):
                continue
            if not any(
                _is_self_call(sub, "_compact_with_tighter_budget")
                for stmt in node.body
                for sub in ast.walk(stmt)
            ):
                continue
            recovery_loop_found = True
            is_range = (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            )
            assert is_range, f"{route}: the recovery-carrying loop does not iterate range(...)"
            arg_names = {
                name.id
                for arg in node.iter.args
                for name in ast.walk(arg)
                if isinstance(name, ast.Name)
            }
            assert "n_backends" in arg_names, (
                f"{route}: the recovery-carrying ladder's range does not name "
                "n_backends — the KBR-256 loop-range extension is missing"
            )
        assert recovery_loop_found, (
            f"{route}: no loop body contains a _compact_with_tighter_budget "
            "call — the recovery block is missing (the call-existence test "
            "pins this too)"
        )

    @pytest.mark.parametrize("route", ROUTES)
    def test_recovery_counter_is_initialised_and_subtracted(self, route: str) -> None:
        """``recovery_retries`` is initialised to 0 and subtracted from ``attempt``.

        The thinking-strip shape: the counter gives the compacted re-POST's
        attempt back, so the empty-response schedule does not pull forward.

        Args:
            route: The streaming handler's name.
        """
        fn = _route_fn(route)
        initialiser = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "recovery_retries" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value == 0
            for node in ast.walk(fn)
        )
        assert initialiser, f"{route}: recovery_retries is never initialised to 0"
        ordinal = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "attempt" for t in node.targets)
            and any(isinstance(n, ast.Name) and n.id == "recovery_retries" for n in ast.walk(node.value))
            for node in ast.walk(fn)
        )
        assert ordinal, (
            f"{route}: attempt arithmetic never subtracts recovery_retries — "
            "the compacted re-POST would consume a real failover attempt"
        )
