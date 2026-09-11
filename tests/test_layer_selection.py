"""The selection and enforcement machinery, exercised through a real pytest run.

``tests/test_layer_markers.py`` proves the decisions in :mod:`tests.layers` are
correct and that this run's markers satisfy them.  Neither of those proves the
part that matters operationally: that a **process** asked for a category it does
not have **exits non-zero**.

That distinction is the implementation plan's §1.4 harness rule, third case --
"a guard proving a function was *called* when the enforcement was the branch
after it".  :func:`missing_required_categories` returning a populated list is
inert until something raises, and only a real run can show that it does.

So every case here spawns ``pytest`` and asserts on its **exit code**.  Each one
is paired with a positive control: a negative that never had a positive passes
for the wrong reason, which is a trap this repository has already been caught by.

Each subprocess targets **one small file** and stops at ``--collect-only``.  The
claims are about collection and selection, collection is where the enforcement
branch fires, and a case that costs milliseconds is a case nobody deletes.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from layers import (
    LAYER_MARKERS,
    PENDING_ACTIVATION_LAYERS,
    assert_no_layer_violations,
    marker_expression_of,
    positively_selected_layers,
    stale_pending_layers,
    unaccounted_layers,
    workflow_pytest_invocations,
)

# L2: this module compares the suite's declared selection rules against what a
# real pytest process does with them -- two artifacts that must agree.
pytestmark = pytest.mark.l2

REPO_ROOT = Path(__file__).resolve().parent.parent

# A small, fast, all-`l2` target. Using this module's sibling rather than the
# whole suite keeps every case under a second and gives each exactly one reason
# to fail.
L2_TARGET = "tests/test_layer_markers.py"

# The live-agent target, for the two cases about the default exclusion.
AGENT_LIVE_TARGET = "tests/integration"

_TIMEOUT_SECONDS = 120


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``pytest`` in a subprocess from the repository root.

    Args:
        *args: Arguments appended after ``--collect-only -q``.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` captured as text.

    The child inherits the parent's ``sys.path`` through ``PYTHONPATH`` so it
    imports the same ``kitty`` this run did.  Without that, a run from a git
    worktree silently resolves the package from the main checkout and the child
    can disagree with its parent about what exists.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            # No cache writes: a test that leaves state behind is a test that
            # behaves differently on its second run.
            "-p",
            "no:cacheprovider",
            *args,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )


class TestRequiredCategoryEnforcement:
    """``--require-category`` fails the process, not just a predicate (R4.2, R6.4)."""

    def test_requiring_an_absent_category_exits_non_zero(self) -> None:
        """The deliberate defect this harness must detect.

        A run that collects only ``l2`` is asked to guarantee ``acceptance``.
        There are no acceptance tests in the repository, so the requirement
        cannot be met, and the run must say so by failing.

        This is the exact shape ``TEST_SUITE.md`` §8 warns about: ``-m
        "acceptance or agent_smoke"`` is satisfied by either side alone, so a
        job can report success for a category it never ran.
        """
        result = _run_pytest("-m", "l2", L2_TARGET, "--require-category=acceptance")

        assert result.returncode != 0
        assert "acceptance" in result.stdout + result.stderr

    def test_requiring_a_present_category_exits_zero(self) -> None:
        """The positive control for the case above.

        A checker that fails every run detects the defect above and is useless.
        Same command, same target, one category name changed.
        """
        result = _run_pytest("-m", "l2", L2_TARGET, "--require-category=l2")

        assert result.returncode == 0, result.stdout + result.stderr

    def test_requiring_several_categories_reports_every_missing_one(self) -> None:
        """Name all the unmet requirements, not the first.

        A job declares one requirement per category in its expression.  Learning
        about them one CI round-trip at a time, at ~18 minutes a round, is how a
        gate becomes something people switch off.
        """
        result = _run_pytest(
            "-m",
            "l2",
            L2_TARGET,
            "--require-category=acceptance",
            "--require-category=load",
        )

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "acceptance" in combined
        assert "load" in combined

    def test_a_misspelled_category_is_a_usage_error(self) -> None:
        """An unknown name must fail loudly rather than be quietly satisfied (R4.1).

        Both silent alternatives are worse than an error.  Ignoring the name
        lets a job claim a category it never verified; treating it as a real
        category that is always empty makes the job permanently, inexplicably
        red.
        """
        result = _run_pytest("-m", "l2", L2_TARGET, "--require-category=acceptence")
        combined = result.stdout + result.stderr

        assert result.returncode != 0
        # 🔴 Assert on WHICH error, not merely that one happened. Both branches
        # of the hook put the name in the message and both exit non-zero, so
        # `returncode != 0` plus the name is satisfied with the whole
        # `unknown_categories` check deleted -- the typo then falls through and
        # is reported as "that category is empty", which is precisely the
        # misleading diagnostic this requirement exists to prevent. Confirmed by
        # deleting the branch: the class stayed green.
        assert "unknown layers" in combined, combined

    def test_a_run_with_no_requirements_is_unaffected(self) -> None:
        """The second positive control: the flag is opt-in.

        Without it, nothing new can fail.  This is what makes the flag safe to
        add before the jobs that will use it exist.
        """
        result = _run_pytest("-m", "l2", L2_TARGET)

        assert result.returncode == 0, result.stdout + result.stderr


class TestTheDefaultExclusion:
    """A bare run excludes the non-gating categories (R5.3)."""

    def test_a_bare_run_does_not_collect_the_live_agent_tests(self) -> None:
        """The replacement for the ``--runslow`` silent skip.

        Every test under ``tests/integration/`` needs four real agent binaries
        and live provider credentials.  A bare ``pytest`` must not try to run
        them -- and must not *skip* them either, which is what it used to do.
        Exit code 5 is pytest's "no tests ran": everything the target holds was
        deselected, which is a statement about selection.
        """
        result = _run_pytest(AGENT_LIVE_TARGET)

        assert result.returncode == 5, result.stdout + result.stderr
        assert "deselected" in result.stdout

    def test_an_explicit_marker_expression_overrides_the_default(self) -> None:
        """The positive control, and the mechanism the nightly job depends on.

        ``addopts`` carries a default ``-m``; a command-line ``-m`` replaces it
        outright.  If it did not, the exclusion above would be a policy rather
        than a default and the live-agent tests could never be run at all --
        the deselection would have become a new silent skip.
        """
        result = _run_pytest("-m", "agent_live", AGENT_LIVE_TARGET)

        assert result.returncode == 0, result.stdout + result.stderr
        # A floor, not the exact count. The requirements argue at length that an
        # exact number fails on the next test anyone adds and then gets deleted;
        # the real set claim lives in
        # `test_live_agent_and_the_integration_directory_are_the_same_set`.
        assert "deselected" not in result.stdout
        assert "tests collected" in result.stdout or "/" in result.stdout


class TestStrictMarkers:
    """A misspelled marker is a collection error, not silence (R1.2)."""

    def test_an_unregistered_marker_fails_collection(self, tmp_path: Path) -> None:
        """Prove ``--strict-markers`` is in force, by tripping it.

        Without it, ``pytest.mark.acceptence`` is a warning and *no marker at
        all* -- so the test lands in whatever the path default is and the author
        is never told.  The layer meta-test would still catch it, one round
        later and with a message about a file the author did not touch.

        The bogus file is written to ``tmp_path``, never into ``tests/``, and
        ``-c`` points the run at the repository's own configuration so the
        ``addopts`` under test actually apply.
        """
        target = tmp_path / "test_bogus_marker.py"
        target.write_text(
            "import pytest\n\n\n@pytest.mark.deffinitely_not_a_marker\ndef test_x():\n    pass\n",
            encoding="utf-8",
        )

        result = _run_pytest("--strict-markers", "-c", "pyproject.toml", str(target))

        assert result.returncode != 0
        assert "deffinitely_not_a_marker" in result.stdout + result.stderr

    def test_a_registered_marker_collects_cleanly(self, tmp_path: Path) -> None:
        """The positive control: strictness rejects typos, not every marker."""
        target = tmp_path / "test_good_marker.py"
        target.write_text(
            "import pytest\n\n\n@pytest.mark.l2\ndef test_x():\n    pass\n",
            encoding="utf-8",
        )

        result = _run_pytest("--strict-markers", "-c", "pyproject.toml", str(target))

        assert result.returncode == 0, result.stdout + result.stderr


class TestTheRemovedRunslowFlag:
    """``--runslow`` is gone from the interface, not only from the code (R1.3)."""

    def test_the_flag_is_rejected(self) -> None:
        """Prove the option no longer exists, by asking for it.

        Asserting the string is absent from ``conftest.py`` would pass while an
        alias lingered somewhere else, and would fail on a docstring mentioning
        the removal.  Asking pytest is the claim itself.
        """
        result = _run_pytest("--runslow", L2_TARGET)

        assert result.returncode != 0
        assert "runslow" in result.stdout + result.stderr


def _declared_contract_guards() -> list[tuple[Path, str]]:
    """Return the test modules whose docstring opens by calling them a contract guard.

    Returns:
        ``(absolute path, repository-relative posix path)`` for each module
        whose docstring's **first line** declares it a contract guard.

    The first line only.  A module that cites §6.2 somewhere in its prose is
    referring to the design, not classifying itself, and treating the two the
    same produces a false positive on the first behavioural guard that explains
    where it asserts.
    """
    guards: list[tuple[Path, str]] = []

    for path in sorted(REPO_ROOT.joinpath("tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstring = ast.get_docstring(tree) or ""

        if docstring.splitlines()[:1] and "contract guard" in docstring.splitlines()[0].lower():
            guards.append((path, path.relative_to(REPO_ROOT).as_posix()))

    return guards


@pytest.fixture(scope="module")
def whole_suite_layers(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[str, list[str]]]:
    """Collect the **entire** suite, unfiltered, and return its layer records.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory.

    Returns:
        ``(node id, [layer names])`` for every test in the repository, including
        the ones a default run deselects.

    Raises:
        AssertionError: When the collection fails or reports implausibly little.

    ``-m ""`` is the escape hatch from the default ``addopts`` filter: without
    it there is no way to see the whole suite at once, because every invocation
    now carries a marker expression.  That matters because the in-session
    meta-test judges only what its own run selected -- so an ``l3`` test with a
    second, illegal marker would be deselected by the gating job's ``-m "l1 or
    l2"`` before anything looked at it.

    Module-scoped: one collection serves every check below, and it costs a
    couple of seconds.
    """
    report = tmp_path_factory.mktemp("layer-report") / "layers.json"

    result = _run_pytest("-m", "", f"--layer-report={report}")

    assert result.returncode == 0, result.stdout + result.stderr

    records = [
        (entry["nodeid"], entry["layers"])
        for entry in json.loads(report.read_text(encoding="utf-8"))
    ]

    # The floor lives here rather than in the in-session meta-test, which has to
    # stay true when a developer runs one file. A suite that quietly stopped
    # collecting two thousand tests would satisfy every assertion below.
    assert len(records) > 3000, f"only {len(records)} items collected for the whole suite"

    return records


class TestTheWholeSuiteIsCoherent:
    """Claims about the suite as a whole, not about one run's slice."""

    def test_every_test_in_the_repository_carries_exactly_one_layer(
        self, whole_suite_layers: list[tuple[str, list[str]]]
    ) -> None:
        """The invariant §8 rests on, checked over everything that exists.

        The in-session meta-test cannot make this claim.  It sees only what its
        own marker expression selected, and the items most likely to be
        mis-labelled are exactly the ones a gating expression deselects.
        """
        assert_no_layer_violations(whole_suite_layers)

    def test_the_layers_partition_the_suite(
        self, whole_suite_layers: list[tuple[str, list[str]]]
    ) -> None:
        """Per-layer counts sum to the total.

        An identity rather than a fixed number: it holds at any suite size, so
        it survives ordinary growth, while still failing the moment a test is
        counted twice or not at all.
        """
        per_layer = {
            layer: sum(1 for _, layers in whole_suite_layers if layer in layers)
            for layer in LAYER_MARKERS
        }

        assert sum(per_layer.values()) == len(whole_suite_layers)

    def test_live_agent_and_the_integration_directory_are_the_same_set(
        self, whole_suite_layers: list[tuple[str, list[str]]]
    ) -> None:
        """The path default that decides a test's CI cadence, checked where it is real.

        Asserted over an unfiltered collection, so both sides are populated.
        The equivalent assertion inside a default-filtered session is vacuous —
        the ``agent_live`` side is empty there by construction — and an earlier
        draft of this suite made exactly that mistake.
        """
        live = {node_id for node_id, layers in whole_suite_layers if "agent_live" in layers}
        integration = {
            node_id for node_id, _ in whole_suite_layers if node_id.startswith("tests/integration/")
        }

        assert live == integration
        assert live, "the live-agent set is empty; the path default stopped applying"

    def test_the_contract_files_are_the_only_ones_claiming_l2(
        self, whole_suite_layers: list[tuple[str, list[str]]]
    ) -> None:
        """Pin the explicit-marker list, in both directions.

        The forward direction stops a rename or a lost ``pytestmark`` from
        silently moving a structural guard into the L1 set that mutation testing
        will judge.  The reverse direction -- that nothing *else* claims ``l2``
        -- is what stops the list growing by accident, one convenient marker at
        a time, until the distinction means nothing.
        """
        expected = {
            "tests/test_pypi_packaging.py",
            "tests/test_model_context_packaged_catalog.py",
            "tests/test_review_workflow_cli_contract.py",
            "tests/test_internal_key_completeness.py",
            "tests/test_egress_coverage.py",
            "tests/test_wire_shape_honesty.py",
            "tests/test_github_actions.py",
            "tests/test_layer_markers.py",
            "tests/test_layer_selection.py",
            # T-W3 (KBR-26). The register's *agreement* guards read two real
            # artifacts -- TEST_SUITE.md §3.2 and the AST of `src/kitty` -- so
            # they are contract tests. Its *schema* tests read neither and stay
            # at the l1 default in `tests/harness/test_register.py`; splitting
            # the file is what let each carry the marker it earns, since a
            # module-level `pytestmark` cannot be overridden per test.
            "tests/harness/test_register_agreement.py",
        }

        actual = {
            node_id.split("::", 1)[0]
            for node_id, layers in whole_suite_layers
            if "l2" in layers
        }

        assert actual == expected

    def test_every_module_declaring_itself_a_contract_guard_is_marked_l2(
        self, whole_suite_layers: list[tuple[str, list[str]]]
    ) -> None:
        """A module that calls itself a contract guard must be filed as one.

        ``tests/test_wire_shape_honesty.py`` opens "Contract guard — ..." and was
        still left at the ``l1`` default in the first draft of this change.
        Nothing would have reported it, and mutation testing would have
        attributed its kills to the wrong layer.  The set equality above catches
        that file only because it is now named there; this catches the *next*
        one, which nobody will think to add.

        The predicate is the docstring's **first line**, not a citation anywhere
        in the module.  Scanning for a ``§6.2`` reference was tried and is the
        wrong shape: ``tests/test_internal_keys_not_sent_upstream.py`` cites that
        section to say *where* it asserts, and is a behavioural regression guard,
        correctly ``l1``.  A mention of a section is not a claim to be one --
        ``mem:test_design_traps`` #10 and #11, again.
        """
        mis_filed: list[str] = []

        for _path, relative in _declared_contract_guards():
            layers = {
                layer
                for node_id, item_layers in whole_suite_layers
                if node_id.split("::", 1)[0] == relative
                for layer in item_layers
            }
            if layers and layers != {"l2"}:
                mis_filed.append(f"{relative}: {sorted(layers)}")

        assert mis_filed == []

    def test_the_contract_guard_scan_finds_the_known_positive(self) -> None:
        """The self-check for the scan above (``TEST_SUITE.md`` §6.2).

        A structural guard that has rotted into matching nothing passes forever
        and silently.  ``tests/test_wire_shape_honesty.py`` is the known
        positive: it is the file whose misfiling motivated the scan, and if the
        scan stops finding it the scan has stopped working.
        """
        found = {relative for _, relative in _declared_contract_guards()}

        assert "tests/test_wire_shape_honesty.py" in found


class TestEveryPopulatedLayerIsRunSomewhere:
    """No layer may hold tests that no job runs and no registry acknowledges.

    This is the hole the marker split creates and that nothing else closes.
    Before it, ``tests.yml`` ran a bare ``pytest -q`` and every test ran.  After
    it, a job is a marker expression, and a test written at a layer whose job
    has not been activated yet runs **nowhere** — silently, because a job that
    was never written cannot go red.

    ``TEST_SUITE.md`` §8 asserts the opposite as a property of the design: "no
    test can fall between two jobs". That property is only true if something
    checks it, so this is that check.
    """

    @staticmethod
    def _layers_run_by_a_job() -> set[str]:
        """Return every layer some workflow job positively selects.

        Returns:
            The union over all jobs of the layers each job's marker expression
            admits.

        Built on the shared :func:`workflow_pytest_invocations`, not on a local
        sweep.  A narrower sweep fails **open** in
        :meth:`test_the_pending_registry_holds_no_layer_a_job_already_runs`: a
        job activating a layer in a ``.yaml`` file, or in a multi-line ``run:``
        block, would leave a stale registry entry undetected — the direction the
        design calls out as the one that must not outlive its reason.
        """
        return {
            layer
            for command in workflow_pytest_invocations(REPO_ROOT / ".github" / "workflows")
            for layer in positively_selected_layers(marker_expression_of(command) or "")
        }

    def test_the_workflow_sweep_actually_found_an_expression(self) -> None:
        """The self-check every structural scan in this repository carries.

        Without it, a change to how workflows invoke pytest turns the check
        below into a sweep over an empty list, which passes and proves nothing.
        ``TEST_SUITE.md`` §6.2 names this pattern and points at
        ``tests/test_egress_coverage.py`` as the one to copy.
        """
        assert workflow_pytest_invocations(REPO_ROOT / ".github" / "workflows"), (
            "no pytest invocation found in any workflow"
        )
        assert self._layers_run_by_a_job(), "no workflow job selects any layer"

    def test_each_populated_layer_is_selected_by_a_job_or_acknowledged_as_pending(
        self, whole_suite_layers: list[tuple[str, list[str]]]
    ) -> None:
        """Every layer holding tests is either run, or on the record as not run.

        The registry is the deliberate part.  ``agent_live`` holds 32 tests that
        no job runs today — true before this change too, when they were
        collected and skipped — and the honest response is to name it and the
        task that fixes it, not to pretend the split is complete.

        The arithmetic is :func:`unaccounted_layers`, which carries its own
        falsification cases in ``test_layer_markers.py``: this composition
        cannot fail against the repository as it stands, which is correct by
        design and would otherwise leave it unexercised.
        """
        populated = {layer for _, layers in whole_suite_layers for layer in layers}

        unaccounted = unaccounted_layers(
            populated, self._layers_run_by_a_job(), PENDING_ACTIVATION_LAYERS
        )

        assert unaccounted == [], (
            f"These layers hold tests that no CI job runs: {unaccounted}. Either "
            "activate a job for them or add them to PENDING_ACTIVATION_LAYERS "
            "with the plan task that will."
        )

    def test_the_pending_registry_holds_no_layer_a_job_already_runs(self) -> None:
        """A stale entry is a standing excuse; fail so it gets removed.

        Checked in this direction so the registry cannot outlive its reason.
        When an activation task lands, this test fails until its layer comes off
        the list -- which is how the debt gets discharged rather than forgotten.
        """
        stale = stale_pending_layers(self._layers_run_by_a_job(), PENDING_ACTIVATION_LAYERS)

        assert stale == [], (
            f"These layers are run by a job but still listed as pending: {stale}. "
            "Remove them from PENDING_ACTIVATION_LAYERS."
        )
