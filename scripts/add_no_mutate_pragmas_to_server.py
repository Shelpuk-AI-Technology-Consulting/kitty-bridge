#!/usr/bin/env python3
"""Mark every function/class in server.py with `# pragma: no mutate block`, except the
seven BridgeServer methods named by TEST_SUITE.md §6.1's compaction_and_pairing group.

Why a script: server.py is ~9,000 lines. Placing ~108 pragmas by hand drifts under merge
(line numbers change); AST is stable. The seven-method carve-out is encoded by symbol,
not by line, so a rename keeps the carve-out intact.

Placement: mutmut 3.x reads `# pragma: no mutate block` from a function/class body's
header (the whitespace before the first statement — see
``mutmut/mutation/pragma_handling.py`` ``_visit_compound_header`` and
``_scan_body_stmts``). The pragma is inserted on the line immediately before the
def/class's first body statement.

Carve-out: BridgeServer is NOT itself pragma'd (it would suppress the seven methods).
The script walks BridgeServer's body and pragmas each method that is not in the seven.

Scope is deliberately read-only outside the targeted edits: a single line of source
change per out-of-scope def, nothing else.
"""

from __future__ import annotations

import ast
import pathlib
import sys

SEVEN: frozenset[str] = frozenset(
    {
        "_compact_messages",
        "_compact_with_tighter_budget",
        "_validate_tool_call_pairing",
        "_truncate_oversized_tool_results",
        "_apply_compaction",
        "_normalize_model",
        "_get_max_context_chars",
    }
)
# SEVEN must stay in lockstep with TARGET_GROUPS["compaction_and_pairing"]
# in tests/mutmut_scope.py. The L2 guard compares server.py's unmarked set
# against the registry, so a registry edit trips the guard — but this
# script's re-run path consults only SEVEN, so update both together.


def collect_insertions(
    lines: list[str], tree: ast.Module
) -> list[tuple[int, str]]:
    """Return ``(0-indexed_insertion_point, indent)`` pairs in source order.

    Indent is derived from the body's first statement's leading whitespace — the
    pragma sits at the body's indent level, one line above the first statement.

    Args:
        lines: The source file split on newlines (unmutated).
        tree: The parsed AST of the same source.

    Returns:
        Insertion specs in ascending line order, one per out-of-scope
        def/class. Already-marked defs are skipped (idempotency).
    """
    insertions: list[tuple[int, str]] = []

    def mark(node: ast.AST) -> None:
        """Schedule a pragma insertion one line above ``node``'s first body statement.

        Idempotent: skips when the line above the first body statement already
        carries the pragma (a previous run inserted it). The check looks at
        ``body_first.lineno - 2`` (0-indexed) — the *existing* line above the
        insertion site, not the insertion site itself, since the insertion site
        in a re-run is the line the pragma was pushed to on the prior run.
        """
        body = getattr(node, "body", None)
        if not body:
            return  # no body — nothing to suppress
        body_first = body[0]
        if not isinstance(body_first, ast.stmt):
            return  # pragma must precede a statement
        insert_at = body_first.lineno - 1  # 0-indexed
        above = insert_at - 1
        if 0 <= above < len(lines) and lines[above].lstrip().startswith(
            "# pragma: no mutate"
        ):
            return  # already marked on a previous run
        cur = lines[body_first.lineno - 1]
        indent = cur[: len(cur) - len(cur.lstrip())]
        insertions.append((insert_at, indent))

    def walk_bridge_body(cls: ast.ClassDef) -> None:
        """Mark each BridgeServer child that's not one of the seven."""
        for child in cls.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name in SEVEN:
                    continue
                mark(child)
            elif isinstance(child, ast.ClassDef):
                mark(child)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in SEVEN:
                # All seven live on BridgeServer today; future-proofs hoisting.
                continue
            mark(node)
        elif isinstance(node, ast.ClassDef):
            if node.name == "BridgeServer":
                walk_bridge_body(node)
            else:
                mark(node)

    return insertions


def apply(lines: list[str], insertions: list[tuple[int, str]]) -> None:
    """Insert pragma lines into ``lines`` in reverse line order.

    Reversing ensures earlier insertions do not shift the indices of later
    ones.

    Args:
        lines: The source file split on newlines; mutated in place.
        insertions: ``(0-indexed_insertion_point, indent)`` pairs from
            :func:`collect_insertions`.
    """
    insertions.sort(key=lambda pair: pair[0], reverse=True)
    for index, indent in insertions:
        lines.insert(index, f"{indent}# pragma: no mutate block")


def main(path: pathlib.Path) -> int:
    """Mark every out-of-scope def/class in ``path`` with the pragma.

    Idempotent: a second run on an already-marked file is a no-op.

    Args:
        path: Path to the Python source file (``server.py``).

    Returns:
        0 when pragmas were written; 1 when the file was already fully
        marked (a no-op run).
    """
    src = path.read_text()
    lines = src.split("\n")
    tree = ast.parse(src)
    insertions = collect_insertions(lines, tree)
    apply(lines, insertions)
    new_src = "\n".join(lines)
    if new_src == src:
        print("no-op: no insertions produced", file=sys.stderr)
        return 1
    path.write_text(new_src)
    print(f"inserted {len(insertions)} pragmas into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
