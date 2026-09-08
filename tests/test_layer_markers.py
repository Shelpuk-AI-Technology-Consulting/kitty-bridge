"""The layer-marker meta-suite: every test declares exactly one layer.

``.system_design/TEST_SUITE.md`` §8 defines a CI job as a marker expression and
nothing else.  That only divides the suite cleanly if every collected test
carries **exactly one** layer marker: a test with none falls between two jobs and
is never run, a test with two runs in both and is paid for twice.

This module holds that property three ways, and the three are deliberately
different in kind:

* the pure predicates in :mod:`tests.layers` are exercised against deliberate
  defects, so the checker is known to detect what it claims to detect;
* the assertion helper is handed an injected violation, so the *enforcement* is
  known to be the assertion rather than a call whose result is discarded;
* the meta-test judges the real, collected run, and first proves the list it is
  judging is the real one rather than an empty list.

The third point is the one worth stating twice.  A checker that reads an empty
collection and reports success is the failure §8 calls "green because it stopped
looking", and it is indistinguishable from a passing suite unless the checker is
made to say how much it looked at.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from layers import (
    LAYER_MARKERS,
    PENDING_ACTIVATION_LAYERS,
    RESOURCE_DEPENDENT_LAYERS,
    assert_no_layer_violations,
    default_layer_for,
    default_marker_expression,
    find_layer_violations,
    layer_markers_in,
    missing_required_categories,
    positively_selected_layers,
    stale_pending_layers,
    unaccounted_layers,
    unknown_categories,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _modules_applying_the_slow_marker(root: Path) -> list[str]:
    """Return every ``file:line`` under ``root`` that applies a ``slow`` marker.

    Args:
        root: Directory to scan recursively.

    Returns:
        One ``path:lineno`` string per occurrence, sorted by path.

    Matches the **attribute** ``.slow``, so it finds both a
    ``@pytest.mark.slow`` decorator and a ``pytestmark = pytest.mark.slow``
    assignment.  Parsed, never grepped: a text sweep for ``mark.slow`` matches
    this module's own docstrings and the design documents discussing the change
    — ``mem:test_design_traps`` #10 and #11 are that defect twice over.
    """
    offenders: list[str] = []

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "slow":
                offenders.append(f"{path.name}:{node.lineno}")

    return offenders

# This module compares the suite's declared metadata against the suite's actual
# metadata -- two artifacts edited separately, which is the L2 definition. It
# also sits at `tests/` root, where the path default is `l1`, so it is the first
# live exercise of "an explicit marker wins over the default".
pytestmark = pytest.mark.l2


class TestTheVocabulary:
    """The eight marker names, and their registration with pytest."""

    def test_the_eight_layers_are_exactly_the_ones_the_design_names(self) -> None:
        """Pin the vocabulary against ``TEST_SUITE.md`` §8.

        Written as an equality rather than a membership sweep: a ninth marker
        added without a job to run it is the same defect as a missing one, and
        only equality catches that direction.
        """
        assert LAYER_MARKERS == (
            "l1",
            "l2",
            "l3",
            "acceptance",
            "agent_smoke",
            "agent_live",
            "eval",
            "load",
        )

    def test_every_layer_marker_is_registered_with_pytest(
        self, pytestconfig: pytest.Config
    ) -> None:
        """Assert each layer name appears in the ``markers`` ini setting.

        Registration is what makes ``--strict-markers`` able to reject a typo.
        An unregistered marker still *works*, silently, which is why this is
        checked rather than assumed.
        """
        registered = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}

        assert set(LAYER_MARKERS) <= registered

    def test_the_superseded_slow_marker_is_gone(self, pytestconfig: pytest.Config) -> None:
        """Assert ``slow`` is no longer a registered marker.

        Two vocabularies for "do not run this now" is how a test ends up in
        neither job.  ``slow`` was the old one; its 32 users are ``agent_live``
        now, selected by expression instead of skipped in place.
        """
        registered = {line.split(":", 1)[0].strip() for line in pytestconfig.getini("markers")}

        assert "slow" not in registered


class TestDefaultLayerForPath:
    """The path → default-layer decision (R2.1, R2.4)."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("tests/integration/test_agent_e2e.py", "agent_live"),
            ("tests/integration/nested/test_deep.py", "agent_live"),
            ("tests/bridge/test_server.py", "l1"),
            ("tests/test_egress.py", "l1"),
            ("tests/tui/test_setup_wizard.py", "l1"),
        ],
    )
    def test_it_maps_a_path_to_its_layer(self, path: str, expected: str) -> None:
        """Map each representative path to the layer the design assigns it."""
        assert default_layer_for(path) == expected

    def test_it_accepts_a_windows_path_separator(self) -> None:
        """Normalise ``\\`` before matching, because the suite runs on Windows.

        The hook builds this string from a :class:`pathlib.Path`, which on
        Windows renders with backslashes.  A prefix match against ``"tests/"``
        would then miss every file and default the whole suite to ``l1`` --
        including the live-agent tests, which CI would then try to run.
        """
        assert default_layer_for("tests\\integration\\test_agent_e2e.py") == "agent_live"

    def test_a_path_outside_tests_still_gets_a_layer(self) -> None:
        """Never return ``None``: an unclassified test is one that runs nowhere.

        The default is deliberately the *gating* layer.  A path this function
        does not recognise is a new corner of the tree, and a new corner joining
        the fast gate uninvited is a visible, cheap mistake; one joining a
        nightly job is invisible until something ships broken.
        """
        assert default_layer_for("some/unexpected/place/test_x.py") == "l1"


class TestLayerMarkersIn:
    """Extracting the layer names from an item's full marker set (R3.3)."""

    def test_it_ignores_markers_that_are_not_layers(self) -> None:
        """Return only layer names, so ``parametrize`` and friends do not count."""
        assert layer_markers_in(["asyncio", "parametrize", "l1", "skipif"]) == ["l1"]

    def test_it_preserves_every_layer_marker_present(self) -> None:
        """Return *all* layers found, because the violation is having two."""
        assert layer_markers_in(["l1", "asyncio", "l3"]) == ["l1", "l3"]

    def test_it_returns_empty_when_no_layer_is_present(self) -> None:
        """Return an empty list rather than raising: zero is a reportable state."""
        assert layer_markers_in(["asyncio"]) == []


class TestFindLayerViolationsDetectsDeliberateDefects:
    """Falsification for the marker checker's predicate (R6.1).

    Plan §1.4: the first working version of a harness ships with a deliberate
    defect it must detect.  These are those defects.  Each case is the exact
    shape the meta-test exists to catch, handed to the checker directly.
    """

    def test_a_test_with_no_layer_marker_is_reported(self) -> None:
        """The falls-between-two-jobs defect."""
        violations = find_layer_violations([("tests/test_x.py::test_a", ["asyncio"])])

        assert len(violations) == 1
        assert "tests/test_x.py::test_a" in violations[0]

    def test_a_test_with_two_layer_markers_is_reported(self) -> None:
        """The runs-in-both-jobs defect."""
        violations = find_layer_violations([("tests/test_x.py::test_a", ["l1", "l3"])])

        assert len(violations) == 1
        assert "tests/test_x.py::test_a" in violations[0]

    def test_the_report_names_the_markers_it_found(self) -> None:
        """Name the offending markers, not just the test.

        A maintainer reading this failure in CI has no other way to see which
        two layers collided; "exactly one required" alone sends them to a
        3,000-test suite with no starting point.
        """
        violations = find_layer_violations([("tests/test_x.py::test_a", ["l1", "l3"])])

        assert "l1" in violations[0]
        assert "l3" in violations[0]

    def test_a_conforming_test_is_not_reported(self) -> None:
        """The positive control.

        A checker that reports everything detects both defects above and is
        still useless.  ``test_design_traps`` #3: a negative that never had a
        positive passes for the wrong reason.
        """
        assert find_layer_violations([("tests/test_x.py::test_a", ["l1", "asyncio"])]) == []

    def test_it_reports_every_offender_not_only_the_first(self) -> None:
        """Report all violations in one run.

        Fixing these one CI round-trip at a time, at ~18 minutes a round, is how
        a guard becomes something people disable.
        """
        violations = find_layer_violations(
            [
                ("tests/test_x.py::test_a", []),
                ("tests/test_y.py::test_b", ["l1"]),
                ("tests/test_z.py::test_c", ["l2", "load"]),
            ]
        )

        assert len(violations) == 2


class TestMissingRequiredCategoriesDetectsDeliberateDefects:
    """Falsification for the category checker's predicate (R6.3).

    :func:`find_layer_violations` got this treatment from the first draft and
    its twin got none — the end-to-end subprocess cases in
    ``test_layer_selection.py`` exercised the behaviour through a process, which
    is the enforcement half of plan §1.4 but not the predicate half.  The
    predicate is the part that can be handed a deliberate defect, which is
    exactly why §1.4 asks for it.
    """

    def test_a_required_category_that_collected_nothing_is_reported(self) -> None:
        """The defect: a job claims a category the run did not cover."""
        missing = missing_required_categories({"l1": 5, "acceptance": 0}, ["l1", "acceptance"])

        assert missing == ["acceptance"]

    def test_a_category_absent_from_the_counts_is_reported(self) -> None:
        """Absent and zero are the same claim.

        A layer nothing carries never appears as a key, so a check written as
        ``counts[name] == 0`` would raise instead of reporting -- and a check
        written as ``name in counts`` would miss the zero case.
        """
        assert missing_required_categories({"l1": 5}, ["acceptance"]) == ["acceptance"]

    def test_every_missing_category_is_reported_not_only_the_first(self) -> None:
        """Report them all, for the same reason the marker checker does."""
        missing = missing_required_categories({"l1": 5}, ["acceptance", "load"])

        assert missing == ["acceptance", "load"]

    def test_a_populated_category_is_not_reported(self) -> None:
        """The positive control.

        A predicate that reports everything detects the defects above and is
        useless; this is what makes the flag safe to put on a real job.
        """
        assert missing_required_categories({"l1": 5, "l2": 2}, ["l1", "l2"]) == []


class TestUnknownCategoriesDetectsDeliberateDefects:
    """Falsification for the ``--require-category`` name check (R4.1)."""

    def test_a_misspelled_layer_is_reported(self) -> None:
        """The defect a silent implementation would turn into a permanent red."""
        assert unknown_categories(["l1", "acceptence"]) == ["acceptence"]

    def test_every_real_layer_is_accepted(self) -> None:
        """The positive control: it rejects typos, not the vocabulary."""
        assert unknown_categories(list(LAYER_MARKERS)) == []


class TestTheActivationArithmetic:
    """Falsification for the composed R5.4 guard.

    The guard cannot fail against the repository as it stands — every non-gated
    layer is in the registry, and the only job runs ``l1 or l2`` — which is
    correct by design and leaves the set arithmetic that closes "the hole the
    split creates" otherwise unexercised.  These give it the defects it must
    detect.
    """

    def test_a_populated_layer_no_job_runs_and_nothing_lists_is_reported(self) -> None:
        """The hole itself: tests exist at a layer that runs nowhere."""
        unaccounted = unaccounted_layers(
            populated=["l1", "l3"], run_by_a_job=["l1"], pending=[]
        )

        assert unaccounted == ["l3"]

    def test_a_populated_layer_the_registry_acknowledges_is_not_reported(self) -> None:
        """Acknowledged debt is allowed; that is what the registry is for."""
        unaccounted = unaccounted_layers(
            populated=["l1", "l3"], run_by_a_job=["l1"], pending=["l3"]
        )

        assert unaccounted == []

    def test_a_layer_a_job_runs_is_not_reported(self) -> None:
        """The positive control for the forward direction."""
        assert unaccounted_layers(populated=["l1", "l2"], run_by_a_job=["l1", "l2"], pending=[]) == []

    def test_a_registry_entry_for_a_layer_a_job_now_runs_is_reported(self) -> None:
        """The reverse direction: an entry must not outlive its reason.

        When an activation task lands, this is what fails until its layer comes
        off the list -- which is how the debt is discharged rather than
        forgotten.
        """
        assert stale_pending_layers(run_by_a_job=["l1", "l3"], pending=["l3"]) == ["l3"]

    def test_a_registry_entry_for_a_layer_no_job_runs_is_not_reported(self) -> None:
        """The positive control for the reverse direction."""
        assert stale_pending_layers(run_by_a_job=["l1"], pending=["l3"]) == []


class TestTheEnforcementIsTheAssertion:
    """Falsification for the marker checker's enforcement path (R6.2).

    §1.4's third named trap is "a guard proving a function was *called* when the
    enforcement was the branch after it".  :func:`find_layer_violations` returning
    a populated list changes nothing on its own -- something has to fail.  These
    two tests pin that something.
    """

    def test_an_injected_violation_fails_the_assertion(self) -> None:
        """A violating record must raise, not merely be returned."""
        with pytest.raises(AssertionError) as excinfo:
            assert_no_layer_violations([("tests/test_x.py::test_a", ["l1", "l3"])])

        assert "tests/test_x.py::test_a" in str(excinfo.value)

    def test_a_conforming_record_set_does_not_raise(self) -> None:
        """The positive control for the enforcement path."""
        assert_no_layer_violations([("tests/test_x.py::test_a", ["l1"])])

    def test_an_empty_record_set_fails_rather_than_passing_vacuously(self) -> None:
        """Refuse to certify nothing.

        This is the "green because it stopped looking" case.  An empty list
        satisfies "no violations found" perfectly, so the helper must treat
        having been given nothing to check as the error it is.
        """
        with pytest.raises(AssertionError):
            assert_no_layer_violations([])


class TestTheRealCollectedSuite:
    """The meta-test: the property, asserted over this very run (R3.1, R3.2).

    Everything above proves the checker works on synthetic input.  This is the
    one that proves the suite is actually in the state the checker describes.
    """

    def test_the_records_describe_this_run_and_not_an_empty_list(
        self,
        collected_layer_markers: list[tuple[str, list[str]]],
        request: pytest.FixtureRequest,
    ) -> None:
        """Prove the judged list is the real one before judging it.

        Two claims that are one logical fact: the harness is looking at this
        run.  Without them every assertion below is satisfied by an empty list,
        and a suite that silently stopped collecting would report the same green
        as a healthy one.

        The node-id check is what makes it specific.  A non-empty list could
        still be a stale one from an earlier phase; a list containing *this
        test* can only have come from this collection.
        """
        node_ids = {node_id for node_id, _ in collected_layer_markers}

        assert collected_layer_markers, "no items were collected"
        assert request.node.nodeid in node_ids

        # Deliberately no floor on the count here. This assertion has to hold
        # when a developer runs one file, so it cannot also carry the
        # "the whole suite is still being collected" claim -- that one lives in
        # `test_layer_selection.py`, which collects the suite itself and is
        # therefore true regardless of how this session was invoked.

    def test_every_collected_test_carries_exactly_one_layer_marker(
        self, collected_layer_markers: list[tuple[str, list[str]]]
    ) -> None:
        """The property ``TEST_SUITE.md`` §8 depends on.

        A test with no layer is in no job and never runs.  A test with two runs
        in both and is paid for twice.  Either makes the marker expressions in
        the CI matrix a description of intent rather than of behaviour.
        """
        assert_no_layer_violations(collected_layer_markers)

    def test_this_run_collected_only_layers_its_own_expression_selects(
        self,
        collected_layer_markers: list[tuple[str, list[str]]],
        pytestconfig: pytest.Config,
    ) -> None:
        """Selection matches the expression the session was actually given.

        The practical claim is the one that replaced the ``--runslow`` silent
        skip: a bare ``pytest`` must not collect the live-agent tests. But
        asserting "no resource-dependent layer is present" outright is wrong in
        two ways at once, and an earlier draft did exactly that.

        It was **vacuous** in every invocation the project uses — under
        ``-m "l1 or l2"`` and under the bare default, a resource layer is
        deselected before the records are written, so the offender list is empty
        by construction. And it was **red** under ``pytest -m ""``, which the
        README and the command memory both document as "run everything": that
        collects the 32 live-agent tests legitimately, and the assertion failed
        with a message about layer markers.

        Comparing against the session's own expression fixes both. It is
        falsifiable under any invocation — a test carrying a layer this run did
        not ask for fails it — and it is true under all of them.
        """
        expression = pytestconfig.getoption("markexpr")
        selectable = set(positively_selected_layers(expression))

        offenders = [
            (node_id, layers)
            for node_id, layers in collected_layer_markers
            if not set(layers) <= selectable
        ]

        assert offenders == [], f"expression {expression!r} selects {sorted(selectable)}"


class TestPositivelySelectedLayers:
    """Which layers an expression can select -- by evaluation, not by grep.

    The workflow check is built on this, and the distinction it draws is the
    reason that check is worth anything: a job's ``-m`` string mentioning a
    layer and a job actually running that layer are different facts, and they
    come apart on every expression containing ``not``.
    """

    def test_an_or_expression_selects_both_sides(self) -> None:
        """The gating job's own expression."""
        assert positively_selected_layers("l1 or l2") == ["l1", "l2"]

    def test_a_single_name_selects_only_itself(self) -> None:
        """The shape every activation task will use."""
        assert positively_selected_layers("l3") == ["l3"]

    def test_a_negated_expression_selects_everything_it_does_not_exclude(self) -> None:
        """The case a string scan gets exactly backwards.

        ``not agent_live`` mentions ``agent_live`` and is precisely the
        expression of a job that does **not** run it.  A check that read names
        out of the string would demand the bare local run guarantee live-agent
        coverage.
        """
        selected = positively_selected_layers("not agent_smoke and not agent_live")

        assert "agent_live" not in selected
        assert "agent_smoke" not in selected
        assert "l1" in selected

    def test_the_projects_default_expression_excludes_exactly_the_resource_layers(
        self,
    ) -> None:
        """Tie the two halves of the exclusion together.

        The list and the expression are separate artifacts; this is the
        assertion that stops them drifting.
        """
        selected = positively_selected_layers(default_marker_expression())

        assert set(selected) == set(LAYER_MARKERS) - set(RESOURCE_DEPENDENT_LAYERS)

    def test_an_empty_expression_selects_everything(self) -> None:
        """Match pytest's own treatment of ``-m ""``.

        The whole-suite meta-check in ``test_layer_selection.py`` depends on
        this being the escape hatch from the default ``addopts`` filter.
        """
        assert positively_selected_layers("") == list(LAYER_MARKERS)

    def test_an_expression_naming_a_non_layer_is_rejected(self) -> None:
        """Refuse to guess.

        A job whose expression names something that is not a layer is a job
        nobody can reason about, and silently returning ``[]`` for it would make
        the workflow check pass by understanding nothing.
        """
        with pytest.raises(ValueError, match="acceptence"):
            positively_selected_layers("acceptence or l1")


class TestTheDefaultMarkerExpressionMatchesTheProjectConfig:
    """``pyproject.toml`` carries the derived expression verbatim (R5.3)."""

    def test_pyproject_addopts_carries_the_derived_expression(self) -> None:
        """Assert the config string equals what the module derives.

        Two hand-maintained copies of one list is how a layer gets added to
        ``RESOURCE_DEPENDENT_LAYERS`` and keeps running on developer machines
        anyway -- the exact failure the list exists to prevent.
        """
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        assert f'"-m", "{default_marker_expression()}"' in pyproject

    def test_strict_markers_is_not_in_addopts(self) -> None:
        """``--strict-markers`` must stay off the config file.

        pytest 9.0.x silently ignores it in ``addopts`` and 9.1.0 honours it
        (upstream issue 14442), and this project's floor is ``pytest>=8.0`` with
        no ceiling -- so a config-file placement means the gate is strict for one
        contributor and not for another, with nothing to say which. The workflow
        passes it on the command line instead, where every supported version
        honours it, and ``test_github_actions.py`` asserts that it does.

        Guarded in this direction because the natural tidy-up is to move it here.
        """
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        addopts = pyproject.split("addopts = ", 1)[1].split("\n", 1)[0]

        assert "--strict-markers" not in addopts, addopts

    def test_pyproject_stays_ascii(self) -> None:
        """No new non-ASCII byte in ``pyproject.toml``.

        ``tests/test_pypi_packaging.py`` reads this file with ``Path.read_text()``
        and no encoding, so on a machine whose locale is not UTF-8 a single
        non-ASCII byte fails the release-gating version check. The
        ``[tool.ruff]`` section says so in a comment, and nothing enforced it --
        which is how the first draft of this very change added four such lines.

        Line 8's em-dash is grandfathered; the count is what is pinned, so
        removing it is fine and adding another is not.
        """
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        offenders = [
            f"line {number}"
            for number, line in enumerate(text.splitlines(), start=1)
            if any(ord(character) > 127 for character in line)
        ]

        assert offenders == ["line 8"], offenders


class TestThePendingActivationRegistry:
    """The acknowledged-debt list is coherent with the vocabulary.

    Its *use* -- checking it against the jobs that exist and the tests that
    exist -- is in ``test_layer_selection.py``, which needs an unfiltered
    collection to do it.  These are the cheap structural claims.
    """

    def test_every_pending_layer_is_a_real_layer(self) -> None:
        """A registry entry for a nonexistent layer excuses nothing, forever."""
        assert set(PENDING_ACTIVATION_LAYERS) <= set(LAYER_MARKERS)

    def test_the_gating_layers_are_not_listed_as_pending(self) -> None:
        """``l1`` and ``l2`` have a job today, so they must not be excused.

        This is the direction that keeps the registry from quietly growing to
        cover the whole vocabulary, at which point splitting the suite by marker
        would mean nothing runs anywhere.
        """
        assert "l1" not in PENDING_ACTIVATION_LAYERS
        assert "l2" not in PENDING_ACTIVATION_LAYERS

    def test_every_entry_names_the_task_that_activates_it(self) -> None:
        """An entry with no owner is an exemption, not a debt."""
        for layer, reason in PENDING_ACTIVATION_LAYERS.items():
            assert "plan task T-" in reason, f"{layer} names no activation task"


class TestTheSupersededSlowVocabularyIsGone:
    """``slow`` and ``--runslow`` are gone from the code, not merely from prose.

    Checked by parsing, not by grepping.  A text sweep for ``mark.slow`` matches
    this module's own docstrings and the design documents that discuss the
    change -- ``mem:test_design_traps`` #10 and #11 are that same defect, a
    name-based scan over source text, twice.
    """

    def test_no_module_under_tests_still_applies_the_slow_marker(self) -> None:
        """Scan the whole `tests/` tree for a ``slow`` marker in any form.

        Every ``*.py``, not only ``test_*.py``: R1.3 says "anywhere", and a
        ``pytest.mark.slow`` in ``conftest.py`` or in a helper module would
        apply to real tests while sitting outside a ``test_*`` sweep.

        Both syntactic forms matter too — a ``@pytest.mark.slow`` decorator and
        a module-level ``pytestmark`` assignment — which is why the scan matches
        the attribute rather than the decorator: a scan finding only decorators
        would report a file clean that skips its whole module.
        """
        assert _modules_applying_the_slow_marker(REPO_ROOT / "tests") == []

    def test_the_scan_finds_a_slow_marker_when_one_is_present(self, tmp_path: Path) -> None:
        """The falsification case for the scan above (``TEST_SUITE.md`` §6.2).

        A structural guard that has rotted into matching nothing passes forever
        and silently.

        🔴 **This calls the same function the guard above calls.** An earlier
        version re-implemented the parse and walk over its own planted file, so
        it proved a *copy* of the scan worked while the real one could match
        nothing. That was confirmed by breaking the real scan's glob: both tests
        stayed green. Sharing the function is what makes this a self-check
        rather than a second, independent test of the same idea.
        """
        (tmp_path / "test_planted.py").write_text(
            "import pytest\n\n\n@pytest.mark.slow\ndef test_x():\n    pass\n",
            encoding="utf-8",
        )
        (tmp_path / "helper_planted.py").write_text(
            "import pytest\n\npytestmark = pytest.mark.slow\n",
            encoding="utf-8",
        )

        found = _modules_applying_the_slow_marker(tmp_path)

        assert len(found) == 2, found
