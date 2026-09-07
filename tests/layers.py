"""Layer-marker vocabulary and the pure decisions the pytest hooks apply.

``.system_design/TEST_SUITE.md`` §8 defines every CI job as a marker expression
over eight layer names and nothing else.  This module owns that vocabulary and
the decisions taken over it:

* which layer a test file gets **by default**, from its path;
* whether a collected item's marker set is **legal** (exactly one layer);
* whether a job's **required categories** actually collected anything;
* which layers a marker expression can **positively select**, which is how a
  job's claim about itself is checked against the tests that exist.

Every function here is pure — no pytest objects, no filesystem, no clock — so
each can be handed a deliberate defect and asked whether it notices.  That is
required, not stylistic: the implementation plan's §1.4 harness rule says the
first working version of a harness ships with a falsification case it must
detect, and a decision entangled with the collection hook cannot be given one.

The hooks that apply these decisions live in :mod:`tests.conftest`.

**A note on units.** The design document counts 2,880 *test functions*; the
numbers here and in the tests count *collected items*, which is larger because
``parametrize`` expands.  The two are both right and are not comparable.

**Import note.** Imported as ``layers``, not ``tests.layers``: ``tests/`` has no
``__init__.py``, so pytest's ``prepend`` import mode puts that directory on
``sys.path``.  ``tests/internal_key_scan.py`` is imported the same way and the
same caveat applies — adding ``tests/__init__.py`` would break both.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# The vocabulary, in the order TEST_SUITE.md §8 lists it: the four layers of the
# model, then the four product-level cadences that are selected separately
# because they are slow, nondeterministic, or need credentials.
LAYER_MARKERS: tuple[str, ...] = (
    "l1",
    "l2",
    "l3",
    "acceptance",
    "agent_smoke",
    "agent_live",
    "eval",
    "load",
)

# Every enforcement failure this module feeds is prefixed with this, so a CI
# summary can tell "the suite is mis-labelled" apart from "somebody fat-fingered
# a flag in the workflow" -- pytest reports both as exit code 4.
ENFORCEMENT_PREFIX = "layer-selection:"

# Layers whose tests need a resource CI provides and a developer's machine may
# not: a pinned agent binary, live provider credentials, a load rig. A bare
# `pytest` excludes these by default, because the alternative is a local run
# that tries to launch four real agent CLIs.
#
# Membership is about the RESOURCE, not about whether the layer gates. Two of
# these gate a release; `agent_smoke` is here before it has a single test,
# because §6.4.2 launches a real pinned Claude Code binary and the task that
# makes that category live should not also have to discover this rule.
RESOURCE_DEPENDENT_LAYERS: tuple[str, ...] = (
    "agent_smoke",
    "agent_live",
    "eval",
    "load",
)

# Layers that have, or may have, tests but that no CI job runs yet, each mapped
# to the plan task that turns its job on.
#
# 🔴 This is a registry of ACKNOWLEDGED DEBT, not a permanent exemption, and it
# is checked in both directions. A layer here that a job now selects is a stale
# entry and fails; a populated layer that is neither selected by a job nor
# listed here also fails. Without it, splitting the suite by marker silently
# stops running any test written at a layer whose job has not been activated --
# which today runs in the one undivided `pytest -q`.
PENDING_ACTIVATION_LAYERS: Mapping[str, str] = {
    "l3": "the Subsystem job — plan task T-K6",
    "acceptance": "the Acceptance job — plan task T-K9",
    "agent_smoke": "the agent-smoke category — plan task T-K10",
    "agent_live": "the Agent-live nightly — plan task T-K11",
    "eval": "the Eval nightly — plan task T-K12",
    "load": "the Load workflow — plan task T-K5",
}

_FALLBACK_LAYER = "l1"

# Longest-prefix-first. One entry today; the shape is what matters, because
# every later plan task that adds a directory of tests adds a row here rather
# than a marker to each of its files.
_PATH_DEFAULTS: tuple[tuple[str, str], ...] = (("tests/integration/", "agent_live"),)

# A pytest invocation, anchored on the command position so that
# `pip install pytest-xdist` is not read as a test job.
_PYTEST_COMMAND = re.compile(r"(?:^|\s|&&|\|\||;)\s*pytest\s")
_MARKER_EXPRESSION = re.compile(r'-m\s+"([^"]*)"')
_REQUIRE_CATEGORY = re.compile(r"--require-category=([A-Za-z_0-9]+)")


def default_layer_for(relative_path: str) -> str:
    """Return the layer a test file gets when it declares none.

    The default is by path so that adding a test to an existing directory needs
    no ceremony, and so that the thousands of tests predating the marker scheme
    did not have to be edited one at a time.

    Args:
        relative_path: The test file's path relative to the repository root, in
            either POSIX or Windows form.

    Returns:
        One of :data:`LAYER_MARKERS`.  Never ``None`` — a test with no layer is
        a test that runs in no job.

    A path this function does not recognise falls back to ``l1``, the gating
    layer, rather than to nothing or to a nightly one.  A new corner of the tree
    joining the fast gate uninvited is a visible, cheap mistake; one joining a
    nightly job is invisible until something ships broken.  The case is
    reachable: ``.github/review/tests/`` matches pytest's ``python_files``.
    """
    # Normalise before matching. The hook builds this string from a Path, which
    # renders with backslashes on Windows; a prefix match against "tests/" would
    # otherwise miss every file and default the live-agent tests into the gate.
    normalised = relative_path.replace("\\", "/")

    for prefix, layer in _PATH_DEFAULTS:
        if normalised.startswith(prefix):
            return layer

    return _FALLBACK_LAYER


def default_marker_expression() -> str:
    """Return the ``-m`` expression a bare ``pytest`` run should carry.

    Returns:
        The conjunction excluding every resource-dependent layer, in
        :data:`LAYER_MARKERS` order.

    ``pyproject.toml`` carries this string literally, and a test asserts the two
    agree.  Two hand-maintained copies of one list is how a layer gets added to
    :data:`RESOURCE_DEPENDENT_LAYERS` and keeps running locally anyway.
    """
    return " and ".join(f"not {name}" for name in RESOURCE_DEPENDENT_LAYERS)


def layer_markers_in(marker_names: Iterable[str]) -> list[str]:
    """Return the layer markers among an item's full marker set.

    Args:
        marker_names: Every marker name on the item, layer and otherwise —
            ``asyncio`` and ``parametrize`` are the common non-layer members.

    Returns:
        The layer names present, in :data:`LAYER_MARKERS` order.  All of them,
        not the first: the violation this feeds is *having two*, so collapsing
        to one would hide exactly the case it exists to find.
    """
    present = set(marker_names)

    return [name for name in LAYER_MARKERS if name in present]


def find_layer_violations(records: Iterable[tuple[str, Iterable[str]]]) -> list[str]:
    """Return one human-readable line per item that does not carry exactly one layer.

    Args:
        records: ``(node id, marker names)`` pairs for the collected items.

    Returns:
        A message per offending item, naming the item and the layers found.
        Empty when every record is legal.  Every offender is reported, not the
        first: at ~18 minutes a CI round, a guard that surfaces one problem per
        round is a guard people turn off.
    """
    violations: list[str] = []

    for node_id, marker_names in records:
        layers = layer_markers_in(marker_names)
        if len(layers) == 1:
            continue

        # Name the layers found, not just the count. A maintainer reading this
        # in a CI log has no other way to see which two collided.
        found = ", ".join(layers) if layers else "none"
        violations.append(f"{node_id} carries {len(layers)} layer markers (found: {found})")

    return violations


def assert_no_layer_violations(records: Iterable[tuple[str, Iterable[str]]]) -> None:
    """Fail unless every record carries exactly one layer marker.

    Args:
        records: ``(node id, marker names)`` pairs for the collected items.

    Raises:
        AssertionError: When any record carries zero or more than one layer, or
            when ``records`` is empty.

    The empty case is an error rather than a pass, and that is the point of
    having this helper at all.  "No violations found" is satisfied perfectly by
    having looked at nothing, and a checker in that state is indistinguishable
    from a healthy one — the failure ``TEST_SUITE.md`` §8 calls green because it
    stopped looking.
    """
    materialised = [(node_id, list(names)) for node_id, names in records]

    assert materialised, (
        f"{ENFORCEMENT_PREFIX} the layer-marker check was handed no items. It "
        "cannot pass by having nothing to check; something upstream stopped "
        "collecting."
    )

    violations = find_layer_violations(materialised)

    assert not violations, (
        f"{ENFORCEMENT_PREFIX} every collected test must carry exactly one layer "
        f"marker ({', '.join(LAYER_MARKERS)}). {len(violations)} do not:\n  "
        + "\n  ".join(violations)
    )


def unknown_categories(required: Sequence[str]) -> list[str]:
    """Return the requested category names that are not layer markers.

    Args:
        required: Category names a job asked to be verified as non-empty.

    Returns:
        The unrecognised names, in the order given.

    A misspelled ``--require-category=acceptence`` would otherwise be a
    requirement that can never be met, which reads in CI as a permanently broken
    job; or, worse, if the check silently ignored unknown names, a job would
    claim a category it never verified.
    """
    return [name for name in required if name not in LAYER_MARKERS]


def missing_required_categories(
    counts: Mapping[str, int], required: Sequence[str]
) -> list[str]:
    """Return the required categories that collected nothing.

    Args:
        counts: Collected item count per layer, after deselection.
        required: Category names the job declared it runs.

    Returns:
        The names whose count is zero or absent, in the order given.

    This exists because a marker *expression* cannot express it.  ``-m
    "acceptance or agent_smoke"`` is satisfied by ``acceptance`` alone: once
    acceptance tests exist the job collects them and passes happily with zero
    agent-smoke coverage, reporting a category it never ran.
    """
    return [name for name in required if counts.get(name, 0) == 0]


def positively_selected_layers(expression: str) -> list[str]:
    """Return the layers a marker expression can select.

    Args:
        expression: A pytest ``-m`` expression over layer names.

    Returns:
        Every layer the expression admits, in :data:`LAYER_MARKERS` order.  An
        empty or whitespace-only expression selects all of them, matching
        pytest's own treatment of ``-m ""``.

    Raises:
        ValueError: When the expression is not a boolean combination of layer
            names — an unknown name, an unsupported operator, or a syntax error.

    Computed by evaluation, one layer at a time, rather than by scanning the
    string for names.  Scanning is correct for ``"l1 or l2"`` and wrong for
    every expression containing ``not``: it would read ``"not agent_live"`` as a
    job that claims to run the live-agent tests.  That distinction is the whole
    reason this function exists — the workflow check built on it asks "which
    categories does this job actually run", not "which words appear in it".
    """
    if not expression.strip():
        return list(LAYER_MARKERS)

    # Parse to a tree and walk it with an explicit node allowlist, rather than
    # calling `eval` behind an identifier check. An identifier check passes
    # `9**9**9` (which hangs) and `1/0` (which raises something the signature
    # does not document); the allowlist admits only `and`, `or`, `not` and a
    # layer name, so anything else is a ValueError before it can run.
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{ENFORCEMENT_PREFIX} marker expression {expression!r} does not parse: {exc}"
        ) from exc

    selected: list[str] = []
    for candidate in LAYER_MARKERS:
        # One layer true, the rest false: does a test carrying only this layer
        # survive the expression?
        bindings = {name: name == candidate for name in LAYER_MARKERS}
        if _evaluate_expression(tree.body, bindings, expression):
            selected.append(candidate)

    return selected


def _evaluate_expression(
    node: ast.expr, bindings: Mapping[str, bool], expression: str
) -> bool:
    """Evaluate one node of a parsed marker expression.

    Args:
        node: The node to evaluate.
        bindings: Truth value per layer name.
        expression: The original text, for error messages.

    Returns:
        The node's truth value.

    Raises:
        ValueError: When the node is not `and`, `or`, `not` or a layer name.
    """
    if isinstance(node, ast.BoolOp):
        values = [_evaluate_expression(v, bindings, expression) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate_expression(node.operand, bindings, expression)

    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise ValueError(
                f"{ENFORCEMENT_PREFIX} marker expression {expression!r} names "
                f"{node.id!r}, which is not a layer ({', '.join(LAYER_MARKERS)})."
            )
        return bindings[node.id]

    raise ValueError(
        f"{ENFORCEMENT_PREFIX} marker expression {expression!r} uses "
        f"{type(node).__name__}, which is not one of and / or / not / a layer name."
    )


def unaccounted_layers(
    populated: Iterable[str], run_by_a_job: Iterable[str], pending: Iterable[str]
) -> list[str]:
    """Return layers that hold tests, that no job runs, and that nothing acknowledges.

    Args:
        populated: Layers that at least one collected test carries.
        run_by_a_job: Layers some CI job's marker expression positively selects.
        pending: Layers the pending-activation registry names.

    Returns:
        The unaccounted layers, sorted.

    This is the arithmetic behind the hole the marker split creates: before it,
    one undivided ``pytest -q`` ran every test, so a test written at any layer
    ran in CI.  After it, a test at a layer whose job has not been activated
    runs nowhere, silently, because a job nobody wrote cannot go red.
    """
    return sorted(set(populated) - set(run_by_a_job) - set(pending))


def stale_pending_layers(run_by_a_job: Iterable[str], pending: Iterable[str]) -> list[str]:
    """Return layers listed as pending activation that a job already runs.

    Args:
        run_by_a_job: Layers some CI job's marker expression positively selects.
        pending: Layers the pending-activation registry names.

    Returns:
        The stale entries, sorted.

    Checked so that a registry entry cannot outlive its reason.  Without this
    direction the registry is a standing amnesty: every layer could be listed,
    and the check that nothing falls through would pass while nothing ran.
    """
    return sorted(set(run_by_a_job) & set(pending))


def workflow_pytest_invocations(workflows_dir: Path) -> list[str]:
    """Return every ``pytest`` command line any workflow job runs.

    Args:
        workflows_dir: The ``.github/workflows`` directory.

    Returns:
        One whitespace-collapsed command string per ``run:`` step that invokes
        pytest, across every job in every workflow.

    One implementation, shared by the two modules that need it, because they
    check opposite directions and a narrower sweep fails **open** in one of
    them: a job activating a layer in a file this missed would leave a stale
    registry entry undetected.

    Both ``*.yml`` and ``*.yaml`` — GitHub accepts either, and a ``.yaml`` added
    later must not slip past a ``.yml``-only sweep.  Whitespace is collapsed so
    a folded or literal block scalar reads the same as a single line.
    """
    # Imported here rather than at module scope: this module is imported by
    # `conftest.py` on every single pytest run, and only these two tests need
    # a YAML parser.
    import yaml

    invocations: list[str] = []

    for path in sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")]):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                run = " ".join((step.get("run") or "").split())
                # Anchored on the command position: a bare `\bpytest\b` also
                # matches `pip install pytest-xdist`, which would then be read
                # as a test job that names no layers.
                if _PYTEST_COMMAND.search(run):
                    invocations.append(run)

    return invocations


def marker_expression_of(command: str) -> str | None:
    """Return the ``-m`` expression a pytest command carries, if any.

    Args:
        command: A whitespace-collapsed pytest command line.

    Returns:
        The expression inside the quotes, or ``None`` when the command names no
        marker expression.
    """
    found = _MARKER_EXPRESSION.search(command)

    return found.group(1) if found else None


def required_categories_of(command: str) -> set[str]:
    """Return the categories a pytest command declares it verifies as non-empty.

    Args:
        command: A whitespace-collapsed pytest command line.

    Returns:
        Every name given to ``--require-category``.
    """
    return set(_REQUIRE_CATEGORY.findall(command))
