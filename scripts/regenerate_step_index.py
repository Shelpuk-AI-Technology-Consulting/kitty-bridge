#!/usr/bin/env python3
"""Validate the ``.system_design/steps/`` dependency graph; write the index.

KBR-278. ``CLAUDE.md`` §"System-design discipline" mandates running this
script after adding a step file and treating a zero exit as the proof that
the ``.system_design/steps/`` dependency graph is honest. This is the
machine-checked half of that mandate; the developer-side half — running
the script after every change — is the discipline the script exists to
serve. The script is deliberately not wired into CI: a committed-tree
test would couple CI to sibling pull requests' step files (see the
REQUIREMENTS.md decision D2 in
``.requirements/20260917T185215Z_step_index_validator/``).

What the script does:

1. Walks ``.system_design/steps/*.md``, parses each file's YAML
   frontmatter, and type-checks it (``id`` is a string, ``depends_on``
   is a list of strings).
2. Validates the resulting step graph: every ``depends_on`` entry is
   either an existing step's ``id`` or a Jira-key cross-reference
   matching ``^KBR-\\d+$`` (the convention the KBR-80 step file
   documents); no two files share an ``id``; the graph is acyclic; every
   ``id`` matches ``^[a-z][a-z0-9_]*$``.
3. On a valid graph, writes the gitignored ``.system_design/INDEX.md``
   with one line per step (sorted by id, carrying the step's
   ``depends_on`` entries verbatim).
4. On any validation failure, writes nothing, prints one human-readable
   error per problem to stderr, and exits 1 — so a pre-existing index
   from a previous successful run is never clobbered by a bad graph.

The non-obvious calls — the Jira-key exemption, the id syntax (lowercase
identifiers with underscores, not kebab-case dashes), why the tests use
fixtures rather than the committed tree, and why the write happens only
after validation — are recorded in the same REQUIREMENTS.md.

Usage::

    python scripts/regenerate_step_index.py

Runs from any cwd (the repo root is derived from ``__file__``). Prints
each validation failure to stderr, one per line; on success it writes
``.system_design/INDEX.md`` and prints nothing.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STEPS_DIR = _REPO_ROOT / ".system_design" / "steps"

# A step id is lowercase ASCII, starting with a letter, then letters /
# digits / underscores. Underscores rather than kebab-case dashes: the
# KBR-278 ticket mandates the snake_case T-F1 id
# `hypothesis_and_transcript_strategies`, and all four shipped step files
# use underscores (REQUIREMENTS.md D1).
_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

# Jira-key cross-reference exemption: an entry naming a Jira ticket
# rather than a step id. The KBR-80 step file (t_g4_wire_shape_guard.md)
# documents the convention ("the plan-task IDs have no step files of
# their own"). The validator has no network access, so the key is
# accepted verbatim rather than resolved (REQUIREMENTS.md D7).
_JIRA_KEY_PATTERN = re.compile(r"KBR-\d+")


@dataclass(frozen=True)
class StepEntry:
    """One parsed step file.

    Attributes:
        path: The step file's path; used in error messages.
        id: The step id, already type-checked as a string and
            pattern-checked against :data:`_ID_PATTERN`.
        depends_on: The ``depends_on`` entries verbatim, already
            type-checked as a tuple of strings.
    """

    path: Path
    id: str
    depends_on: tuple[str, ...]


def _split_frontmatter(text: str) -> tuple[str | None, str | None]:
    """Split a step file's content into its frontmatter block.

    Args:
        text: The full file content.

    Returns:
        ``(frontmatter_text, None)`` on success; ``(None, message)``
        when the file has no leading ``---`` fence, no closing fence,
        or an empty frontmatter block. The message describes the
        problem for the caller to prefix with the file name.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, "no '---' frontmatter block"
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            block = "\n".join(lines[1:i])
            if not block.strip():
                return None, "frontmatter block is empty"
            return block, None
    return None, "no closing '---' fence for frontmatter"


def _parse_entry(path: Path, text: str) -> tuple[StepEntry | None, list[str]]:
    """Parse one step file's content into a :class:`StepEntry`.

    Type-checks and pattern-checks the two keys the validator owns
    (``id`` and ``depends_on``) before any graph logic runs; unknown
    frontmatter keys are ignored (REQUIREMENTS.md D4). Both keys are
    checked even when one fails, so a file with two problems reports
    both rather than hiding the second behind the first.

    Args:
        path: The step file's path, for error messages.
        text: The full file content.

    Returns:
        ``(entry, [])`` on success; ``(None, errors)`` with one
        human-readable error per problem found, each naming the file.
    """
    name = path.name
    block, fence_error = _split_frontmatter(text)
    if fence_error is not None:
        return None, [f"{name}: {fence_error}"]

    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        return None, [f"{name}: frontmatter is not valid YAML: {exc}"]
    if not isinstance(data, dict):
        return (
            None,
            [f"{name}: frontmatter must be a YAML mapping, got {type(data).__name__}"],
        )

    errors: list[str] = []

    # id: present, a string, and matching the lowercase-identifier pattern.
    # PyYAML infers bare scalars (123 -> int, no -> bool), so the type check
    # runs before the regex — otherwise a non-string id crashes with an
    # unhandled TypeError instead of a file-naming error (D8).
    raw_id = data.get("id")
    id_value: str | None = None
    if raw_id is None:
        errors.append(f"{name}: 'id' key missing from frontmatter")
    elif not isinstance(raw_id, str):
        errors.append(f"{name}: 'id' must be a string, got {type(raw_id).__name__}")
    elif not _ID_PATTERN.fullmatch(raw_id):
        errors.append(f"{name}: id '{raw_id}' does not match {_ID_PATTERN.pattern}")
    else:
        id_value = raw_id

    # depends_on: present, a list, and every entry a string. A bare
    # scalar string iterates as characters and a mapping iterates as
    # keys — both silently wrong semantics unless the shape is pinned
    # before iteration (D8).
    raw_deps = data.get("depends_on")
    deps: tuple[str, ...] = ()
    if raw_deps is None:
        errors.append(f"{name}: 'depends_on' key missing from frontmatter")
    elif not isinstance(raw_deps, list):
        errors.append(f"{name}: 'depends_on' must be a list of strings, got {type(raw_deps).__name__}")
    else:
        typed: list[str] = []
        for entry in raw_deps:
            if not isinstance(entry, str):
                errors.append(f"{name}: 'depends_on' entries must be strings, got {type(entry).__name__}")
            else:
                typed.append(entry)
        deps = tuple(typed)

    if errors or id_value is None:
        return None, errors
    return StepEntry(path=path, id=id_value, depends_on=deps), []


def collect_steps(steps_dir: Path) -> tuple[list[StepEntry], list[str]]:
    """Parse every ``steps_dir/*.md`` and return ``(entries, errors)``.

    Args:
        steps_dir: The directory of step files to parse.

    Returns:
        ``(entries, [])`` when every file parsed cleanly;
        ``(entries, errors)`` with one error line per frontmatter
        problem when some did not — the successfully parsed files are
        still returned so the caller's graph checks can proceed against
        the surviving subset.
    """
    entries: list[StepEntry] = []
    errors: list[str] = []
    for path in sorted(steps_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        entry, parse_errors = _parse_entry(path, text)
        errors.extend(parse_errors)
        if entry is not None:
            entries.append(entry)
    return entries, errors


def graph_errors(entries: Sequence[StepEntry]) -> list[str]:
    """Return graph-level errors: duplicates, resolution failures, cycles.

    Duplicate ids are reported naming both files; the first occurrence
    wins for resolution lookups, so a duplicate does not also produce
    spurious unresolved-dependency errors against itself. Each
    ``depends_on`` entry is checked next: either it matches a known step
    ``id``, or it matches the Jira-key pattern and is skipped (D7).
    Cycles are detected last, by depth-first search over the resolved
    (non-Jira) edges only, so an unresolvable edge does not mask or
    invent a cycle; one report per cycle, showing the cycle path.

    Args:
        entries: The step entries to check.

    Returns:
        One human-readable error per problem found; empty when the
        graph is honest.
    """
    errors: list[str] = []

    # Duplicate ids: name both files in one message so the maintainer
    # sees both sides of the collision without a second run.
    by_id: dict[str, StepEntry] = {}
    for entry in entries:
        if entry.id in by_id:
            errors.append(f"duplicate step id '{entry.id}' in {by_id[entry.id].path.name} and {entry.path.name}")
        else:
            by_id[entry.id] = entry

    # Resolution: a non-Jira entry must name a known step id.
    for entry in entries:
        for dep in entry.depends_on:
            if _JIRA_KEY_PATTERN.fullmatch(dep):
                continue
            if dep not in by_id:
                errors.append(f"{entry.path.name}: depends_on entry '{dep}' does not resolve to an existing step id")

    # Cycle detection: iterative-free DFS with the classic three colours.
    # A GRAY node on the stack meeting a GRAY dependency is a back edge —
    # the cycle is the stack slice from that dependency to the current
    # node, plus the dependency again to close the loop.
    _WHITE, _GRAY, _BLACK = 0, 1, 2
    colour: dict[str, int] = {entry.id: _WHITE for entry in entries}
    stack: list[str] = []
    position: dict[str, int] = {}

    def visit(node: str) -> None:
        colour[node] = _GRAY
        stack.append(node)
        position[node] = len(stack) - 1
        for dep in by_id[node].depends_on:
            if _JIRA_KEY_PATTERN.fullmatch(dep) or dep not in colour:
                continue
            if colour[dep] == _GRAY:
                cycle = stack[position[dep] :] + [dep]
                errors.append("dependency cycle: " + " -> ".join(cycle))
            elif colour[dep] == _WHITE:
                visit(dep)
        stack.pop()
        del position[node]
        colour[node] = _BLACK

    for node in sorted(colour):
        if colour[node] == _WHITE:
            visit(node)

    return errors


def write_index(entries: Sequence[StepEntry], index_path: Path) -> None:
    """Write the step index, one line per step, sorted by id.

    The caller has already verified the graph (write-on-valid-only,
    D5), so this function overwrites whatever a previous run left
    behind. Entries are carried verbatim — including Jira-key
    cross-references — so a reader grepping the index sees exactly what
    the validator saw.

    Args:
        entries: The validated step entries.
        index_path: Where to write the index.
    """
    lines = [
        "# Step index",
        "",
        "Generated by `scripts/regenerate_step_index.py`. Do not edit by hand.",
        "",
    ]
    for entry in sorted(entries, key=lambda item: item.id):
        deps = ", ".join(entry.depends_on) if entry.depends_on else "none"
        lines.append(f"- {entry.id} (depends_on: {deps})")
    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def main(steps_dir: Path = _STEPS_DIR) -> int:
    """Validate the graph under ``steps_dir`` and write the index.

    Args:
        steps_dir: The directory of step files. Defaults to the repo's
            ``.system_design/steps``. The index is written to
            ``steps_dir.parent / "INDEX.md"``.

    Returns:
        0 on success (index written); 1 on any validation failure
        (nothing written; one error line per problem on stderr).
    """
    entries, parse_errors = collect_steps(steps_dir)
    errors = [*parse_errors, *graph_errors(entries)]
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    write_index(entries, steps_dir.parent / "INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
