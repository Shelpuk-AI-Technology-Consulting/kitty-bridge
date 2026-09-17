"""Tests for ``scripts/regenerate_step_index.py``, the step-graph validator.

KBR-278. ``CLAUDE.md`` §"System-design discipline" mandates running the
script after adding a step file and treats a zero exit as the proof that
the ``.system_design/steps/`` dependency graph is honest. Before this
script existed, every step file shipped with ``depends_on: []`` because
the YAML could not be validated; this file pins the validator's contract
so the mandate is a machine-checked one.

The failure modes are exercised against **constructed fixture step
files** in ``tmp_path``, not the committed tree: a committed-tree test
would couple CI to sibling pull requests' step files (KBR-278's
REQUIREMENTS.md, decision D2, records the PR #218 case). The committed
tree is covered by the developer-side mandate itself — running the
script after adding a step — which is the discipline the script exists
to serve.

Layer: ``l2`` (explicit ``pytestmark``) — the subject is a config-like
artifact contract (frontmatter ⇄ graph), the same posture as
``tests/test_aggregate_mutation_baseline.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.l2


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the validator by file path rather than package name: `scripts/`
# is not a package and is deliberately NOT added to `sys.path` — one
# file, not a whole directory of script entry points. The module must
# be registered in ``sys.modules`` before ``exec_module`` — Python
# 3.13's ``@dataclass`` resolves the class's namespace through
# ``sys.modules`` while decorating ``StepEntry``.
_spec = importlib.util.spec_from_file_location(
    "regenerate_step_index",
    _REPO_ROOT / "scripts" / "regenerate_step_index.py",
)
assert _spec is not None and _spec.loader is not None
rsi = importlib.util.module_from_spec(_spec)
sys.modules["regenerate_step_index"] = rsi
_spec.loader.exec_module(rsi)
del _spec


def _steps_dir(tmp_path: Path) -> Path:
    """Return ``tmp_path / "steps"``, creating it on demand.

    The validator writes ``INDEX.md`` next to the steps directory
    (i.e., at ``tmp_path / "INDEX.md"``). Using a subdir isolates each
    test's index from its siblings — ``tmp_path.parent`` is shared across
    tests in the same pytest session.
    """
    path = tmp_path / "steps"
    path.mkdir(exist_ok=True)
    return path


def _step(tmp_path: Path, name: str, *, id_: str = "", depends_on: str = "[]", body: str = "") -> Path:
    """Write one step file under ``tmp_path / "steps"`` and return its path.

    Args:
        tmp_path: The test's scratch directory; the step is written under
            ``tmp_path / "steps"``.
        name: The file name (``.md`` appended when missing).
        id_: The ``id`` frontmatter value, written verbatim (so a caller can
            write a non-string scalar such as ``123`` or ``no``).
        depends_on: The ``depends_on`` frontmatter value, written verbatim.
        body: Markdown body below the frontmatter fence; ignored by the
            validator but present so fixtures read like real step files.

    Returns:
        The path of the written file.
    """
    if not name.endswith(".md"):
        name = f"{name}.md"
    path = _steps_dir(tmp_path) / name
    path.write_text(
        f"---\nid: {id_}\ndepends_on: {depends_on}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _raw_step(tmp_path: Path, name: str, text: str) -> Path:
    """Write a step file with hand-built content, fence and all.

    Args:
        tmp_path: The test's scratch directory; the step is written under
            ``tmp_path / "steps"``.
        name: The file name (``.md`` appended when missing).
        text: The exact file content.

    Returns:
        The path of the written file.
    """
    if not name.endswith(".md"):
        name = f"{name}.md"
    path = _steps_dir(tmp_path) / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# R2 — depends_on resolution (with the D7 Jira-key exemption)
# ---------------------------------------------------------------------------


def test_unresolved_step_id_dependency_is_reported(tmp_path: Path) -> None:
    """A depends_on entry naming a non-existent step id is an error.

    The error names both the file that references the id and the missing
    id itself, so a maintainer can fix the graph without re-deriving
    which side of the edge is broken.
    """
    _step(tmp_path, "a", id_="alpha_step", depends_on="[missing_step]")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    errors = rsi.graph_errors(entries)
    assert any("a.md" in e and "missing_step" in e for e in errors), errors


def test_jira_key_dependency_is_accepted_as_a_cross_reference(tmp_path: Path) -> None:
    """A ``KBR-<digits>`` depends_on entry passes without resolving.

    The KBR-80 step file (``t_g4_wire_shape_guard.md``) shipped this
    convention: plan-task ids that have no step files of their own are
    cross-referenced by Jira key. The validator cannot check Jira (no
    network), so the key is accepted verbatim (REQUIREMENTS.md D7).
    """
    _step(tmp_path, "a", id_="alpha_step", depends_on="[KBR-999]")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    assert rsi.graph_errors(entries) == []


# ---------------------------------------------------------------------------
# R3 — duplicate ids
# ---------------------------------------------------------------------------


def test_duplicate_step_id_names_both_files(tmp_path: Path) -> None:
    """Two files claiming one id is an error naming both files."""
    _step(tmp_path, "a", id_="same_id")
    _step(tmp_path, "b", id_="same_id")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    errors = rsi.graph_errors(entries)
    assert any("a.md" in e and "b.md" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R4 — cycles
# ---------------------------------------------------------------------------


def test_two_node_cycle_is_reported_with_its_path(tmp_path: Path) -> None:
    """A two-node cycle is an error that shows the cycle path."""
    _step(tmp_path, "a", id_="alpha_step", depends_on="[beta_step]")
    _step(tmp_path, "b", id_="beta_step", depends_on="[alpha_step]")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    errors = rsi.graph_errors(entries)
    assert any("alpha_step" in e and "beta_step" in e for e in errors), errors


def test_self_cycle_is_reported(tmp_path: Path) -> None:
    """A step depending on itself is a cycle, not a legal no-op."""
    _step(tmp_path, "a", id_="alpha_step", depends_on="[alpha_step]")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    errors = rsi.graph_errors(entries)
    assert any("alpha_step" in e for e in errors), errors


def test_each_cycle_is_reported_exactly_once(tmp_path: Path) -> None:
    """A cycle produced by multiple back-edges appears once in stderr.

    The reproducer: ``m -> n``, ``n -> z``, ``z -> [m, p, q]``, with
    ``p`` and ``q`` leaves. The iterative DFS re-scans ``z``'s dep
    list from the start every time a child subtree completes, so the
    back-edge to ``m`` is re-encountered after ``p`` completes and
    after ``q`` completes — two spurious duplications of the same
    cycle string. The one-report-per-cycle contract requires deduping
    by the full cycle string.
    """
    _step(tmp_path, "m", id_="m_step", depends_on="[n_step]")
    _step(tmp_path, "n", id_="n_step", depends_on="[z_step]")
    _step(tmp_path, "z", id_="z_step", depends_on="[m_step, p_step, q_step]")
    _step(tmp_path, "p", id_="p_step")
    _step(tmp_path, "q", id_="q_step")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    errors = rsi.graph_errors(entries)
    cycle_lines = [e for e in errors if e.startswith("dependency cycle:")]
    assert len(cycle_lines) == 1, cycle_lines


def test_acyclic_diamond_produces_no_cycle_error(tmp_path: Path) -> None:
    """A diamond (two steps sharing a dependency) is not a cycle.

    The falsification control for the cycle detector: a depth-first walk
    that forgets to colour completed nodes reports shared dependencies
    as cycles.
    """
    _step(tmp_path, "a", id_="alpha_step")
    _step(tmp_path, "b", id_="beta_step", depends_on="[alpha_step]")
    _step(tmp_path, "c", id_="gamma_step", depends_on="[alpha_step]")
    _step(tmp_path, "d", id_="delta_step", depends_on="[beta_step, gamma_step]")
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    assert rsi.graph_errors(entries) == []


# ---------------------------------------------------------------------------
# R5 — id syntax, typed-shape (D8), and frontmatter structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("id_", "why"),
    [
        ("Not-Valid", "uppercase anywhere is rejected"),
        ("123_underscores", "a leading digit is rejected"),
        ("has-dash", "kebab-case dashes are rejected"),
        ("''", "a literal empty-string id is rejected by the pattern check"),
    ],
)
def test_id_outside_the_pattern_is_reported(tmp_path: Path, id_: str, why: str) -> None:
    """An ``id`` outside ``^[a-z][a-z0-9_]*$`` is an error naming the file."""
    _step(tmp_path, "a", id_=id_)
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), f"{why}: {errors}"


@pytest.mark.parametrize("id_", ["123", "no", "true"])
def test_non_string_id_is_reported_not_crashed(tmp_path: Path, id_: str) -> None:
    """PyYAML infers bare scalars: ``123`` is an int, ``no``/``true`` are bools.

    A validator that goes straight to regex matching raises ``TypeError``
    on exactly the malformed inputs it exists to catch cleanly
    (REQUIREMENTS.md D8).
    """
    _step(tmp_path, "a", id_=id_)
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), errors


def test_missing_id_key_is_reported(tmp_path: Path) -> None:
    """A frontmatter mapping without ``id`` is an error naming the file."""
    _raw_step(tmp_path, "a", "---\ndepends_on: []\n---\n\n# a\n")
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), errors


def test_missing_depends_on_key_is_reported(tmp_path: Path) -> None:
    """A frontmatter mapping without ``depends_on`` is an error naming the file."""
    _raw_step(tmp_path, "a", "---\nid: alpha_step\n---\n\n# a\n")
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), errors


def test_frontmatter_that_is_not_yaml_is_reported(tmp_path: Path) -> None:
    """A frontmatter block PyYAML cannot parse is an error naming the file."""
    _raw_step(tmp_path, "a", "---\nid: [unclosed\n---\n\n# a\n")
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), errors


def test_file_without_a_frontmatter_fence_is_reported(tmp_path: Path) -> None:
    """A markdown file with no ``---`` block at all is an error, not a skip."""
    _raw_step(tmp_path, "a", "# just markdown\n\nno fence here\n")
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), errors


def test_non_utf8_step_file_is_reported_not_crashed(tmp_path: Path) -> None:
    """A step file saved with a non-UTF-8 encoding surfaces as a named error.

    The bare ``read_text`` would raise ``UnicodeDecodeError`` for any
    step file saved with a different encoding — same defect class the
    typed-shape contract (D8) guards against for bare YAML scalars,
    and it must surface as a file-naming error rather than a traceback.
    """
    path = _steps_dir(tmp_path) / "bad.md"
    # 0xC0 0xC1 are not legal UTF-8 start sequences.
    path.write_bytes(b"\xc0\xc1bad bytes\n")
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("bad.md" in e and "UTF-8" in e for e in errors), errors


def test_directory_in_place_of_step_file_surfaces_named_error(
    tmp_path: Path,
) -> None:
    """A ``steps/<name>.md/`` directory surfaces as a named error, not a traceback.

    ``Path.glob("*.md")`` matches directories whose name ends in
    ``.md``; reading one as text raises ``IsADirectoryError`` on POSIX
    (or ``PermissionError`` on Windows). Without the ``OSError`` catch
    a stray ``steps/archive.md/`` directory would surface as a Python
    traceback instead of the file-naming error the contract promises.
    """
    stray = _steps_dir(tmp_path) / "archive.md"
    stray.mkdir()
    _step(tmp_path, "a", id_="alpha_step")
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("archive.md" in e and "not readable" in e for e in errors), errors


def test_deep_chain_does_not_hit_recursion_limit(tmp_path: Path) -> None:
    """A dependency chain well past Python's recursion limit validates.

    The iterative DFS exists so the validator has no recursion-limit
    boundary: a deep *valid* chain succeeds where the recursive
    equivalent would ``RecursionError`` past ~1000 steps. The fixture
    is a **forward chain** — ``step_i`` depends on ``step_{i + 1}`` —
    so the first sorted start (``step_0000``) descends the full
    1500-level chain. A backward chain would stop at depth 1 because
    every visited node's successors are already BLACK, and a fully
    disconnected graph would never descend past depth 1 at all —
    neither would catch a reversion to the recursive form, which is
    the whole point of the case.
    """
    steps = _steps_dir(tmp_path)
    for i in range(1500):
        deps = f"[step_{i + 1:04d}]" if i < 1499 else "[]"
        (steps / f"step_{i:04d}.md").write_text(
            f"---\nid: step_{i:04d}\ndepends_on: {deps}\n---\n\n# step {i}\n",
            encoding="utf-8",
        )
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    assert len(entries) == 1500
    assert rsi.graph_errors(entries) == []


def test_unknown_frontmatter_keys_are_tolerated(tmp_path: Path) -> None:
    """Extra keys such as ``title`` and ``jira`` pass validation.

    PR #218's step file carries them; the validator's contract is the
    two keys it owns, not the file's whole frontmatter (REQUIREMENTS.md
    D4).
    """
    _raw_step(
        tmp_path,
        "a",
        "---\nid: alpha_step\ntitle: A step\njira: KBR-1\ndepends_on: []\n---\n\n# a\n",
    )
    entries, parse_errors = rsi.collect_steps(tmp_path / "steps")
    assert parse_errors == []
    assert rsi.graph_errors(entries) == []


# ---------------------------------------------------------------------------
# R6 — depends_on typed shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depends_on",
    [
        "alpha_step",  # a bare scalar string, not a list
        "{a: b}",  # a mapping
        "[42]",  # a list with a non-string entry
        "[no]",  # a list with a YAML-boolean entry
    ],
)
def test_depends_on_wrong_shape_is_reported(tmp_path: Path, depends_on: str) -> None:
    """A ``depends_on`` that is not a list of strings is a typed-shape error.

    Iterating a string yields characters and iterating a dict yields
    keys — both silently wrong semantics unless the shape is checked
    before iteration (REQUIREMENTS.md D8).
    """
    _step(tmp_path, "a", id_="alpha_step", depends_on=depends_on)
    _entries, errors = rsi.collect_steps(tmp_path / "steps")
    assert any("a.md" in e for e in errors), errors


# ---------------------------------------------------------------------------
# R7 — the index write, and write-on-valid-only (D5)
# ---------------------------------------------------------------------------


def test_valid_graph_writes_an_index_listing_every_step(tmp_path: Path) -> None:
    """A valid graph produces ``INDEX.md`` next to the steps directory.

    One line per step, sorted by id, with the step's ``depends_on``
    entries verbatim.
    """
    _step(tmp_path, "b", id_="beta_step", depends_on="[alpha_step]")
    _step(tmp_path, "a", id_="alpha_step")
    index_path = tmp_path / "INDEX.md"
    exit_code = rsi.main(tmp_path / "steps")
    assert exit_code == 0
    text = index_path.read_text(encoding="utf-8")
    assert "alpha_step" in text
    assert "beta_step" in text
    # Sorted by id: alpha's line comes first.
    assert text.index("alpha_step") < text.index("beta_step")
    # beta's line carries its dependency verbatim.
    beta_line = next(line for line in text.splitlines() if "beta_step" in line)
    assert "alpha_step" in beta_line


def test_invalid_graph_writes_no_index(tmp_path: Path) -> None:
    """A validation failure writes nothing (D5 — write-on-valid-only)."""
    _step(tmp_path, "a", id_="alpha_step", depends_on="[missing_step]")
    index_path = tmp_path / "INDEX.md"
    assert rsi.main(tmp_path / "steps") == 1
    assert not index_path.exists()


def test_invalid_graph_leaves_a_previous_index_untouched(tmp_path: Path) -> None:
    """A failure must not clobber the previous run's index (D5)."""
    index_path = tmp_path / "INDEX.md"
    index_path.write_text("previous run\n", encoding="utf-8")
    _step(tmp_path, "a", id_="alpha_step", depends_on="[missing_step]")
    assert rsi.main(tmp_path / "steps") == 1
    assert index_path.read_text(encoding="utf-8") == "previous run\n"


def test_valid_run_overwrites_a_stale_index(tmp_path: Path) -> None:
    """A successful run replaces whatever the previous run left behind."""
    index_path = tmp_path / "INDEX.md"
    index_path.write_text("stale content\n", encoding="utf-8")
    _step(tmp_path, "a", id_="alpha_step")
    assert rsi.main(tmp_path / "steps") == 0
    text = index_path.read_text(encoding="utf-8")
    assert "stale content" not in text
    assert "alpha_step" in text


def test_missing_steps_directory_fails_loud(tmp_path: Path) -> None:
    """A steps directory that does not exist is an error, not a vacuous pass.

    Without this floor, ``glob("*.md")`` returns nothing, no entries and
    no errors are found, the script writes an index with only a header,
    and exits 0 — the exact "silently validate" failure the validator
    exists to prevent (TEST_SUITE.md §6.2.3 self-guard posture).
    """
    index_path = tmp_path / "INDEX.md"
    assert rsi.main(tmp_path / "no-such-steps-dir") == 1
    assert not index_path.exists()


def test_empty_steps_directory_fails_loud(tmp_path: Path) -> None:
    """A steps directory holding no ``*.md`` files is the same vacuous pass.

    A repo whose step files were all deleted (or renamed away) must
    fail the validator rather than generate a header-only index with
    exit 0.
    """
    index_path = tmp_path / "INDEX.md"
    assert rsi.main(_steps_dir(tmp_path)) == 1
    assert not index_path.exists()
