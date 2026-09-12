"""Tests for GitHub Actions workflow correctness.

Verifies that the CI/CD workflow files are valid, well-structured, and
implement the expected publish-on-tag pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from layers import (
    marker_expression_of,
    positively_selected_layers,
    required_categories_of,
    workflow_pytest_invocations,
)

# L2: the subject of this file is an artifact outside `src/kitty` Python code,
# or a structural scan of source text -- two things edited separately that must
# agree. It gates pull requests exactly as before, in the `l1 or l2` job; the
# marker records which half of that expression it answers to, and keeps a
# source-text scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

ROOT = Path(__file__).resolve().parent.parent
GITHUB_DIR = ROOT / ".github"
WORKFLOWS_DIR = GITHUB_DIR / "workflows"


def _load_workflow(name: str) -> dict:
    """Load and parse a GitHub Actions workflow YAML file.

    Handles PyYAML's quirk where ``on:`` is parsed as ``True`` (boolean).
    Normalizes the key to the string ``"on"`` for consistent test access.
    """
    path = WORKFLOWS_DIR / name
    if not path.exists():
        pytest.fail(f"Workflow file {name} not found at {path}")
    # Explicitly UTF-8: GitHub reads workflow files as UTF-8, and these carry
    # non-ASCII in their comments. Without this, `open()` uses the platform's
    # locale encoding and most cases in this module fail with a
    # UnicodeDecodeError on a Windows developer's machine while passing on the
    # Linux runner -- a suite that disagrees with itself by operating system.
    #
    # ⚠️ "most", not "every": this helper is not the module's only reader.
    # `TestTagVersionCheck` and `test_no_hardcoded_api_token` read files
    # directly, and the first version of this comment claimed to cover them.
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # PyYAML parses `on:` as boolean True. Normalize it.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _get_trigger(workflow: dict) -> dict:
    """Get the trigger configuration from a workflow."""
    return workflow.get("on", {})


def _resolve_jobs(workflow: dict) -> dict[str, dict]:
    """Expand jobs that delegate to a local reusable workflow.

    A job may either define ``steps`` itself or delegate the whole job to
    another workflow via ``uses: ./.github/workflows/<name>.yml``. Callers care
    about the work that actually runs, so delegated jobs are replaced by the
    jobs of the workflow they call.

    Args:
        workflow: A parsed workflow mapping.

    Returns:
        Job name to job mapping, with delegated jobs expanded in place. Expanded
        names are qualified as ``"<caller>/<callee>"``.
    """
    resolved: dict[str, dict] = {}
    for name, job in workflow.get("jobs", {}).items():
        uses = str(job.get("uses", ""))
        if uses.startswith("./.github/workflows/"):
            called = _load_workflow(Path(uses).name)
            for sub_name, sub_job in called.get("jobs", {}).items():
                resolved[f"{name}/{sub_name}"] = sub_job
        else:
            resolved[name] = job
    return resolved


def _get_all_steps(workflow: dict) -> list[dict]:
    """Flatten all steps from all jobs in a workflow, following reusable calls."""
    steps: list[dict] = []
    for job in _resolve_jobs(workflow).values():
        steps.extend(job.get("steps", []))
    return steps


# ── R1: CI workflow ───────────────────────────────────────────────────────


class TestCIWorkflow:
    """Verify the CI workflow runs on push/PR to main."""

    @pytest.fixture()
    def workflow(self) -> dict:
        return _load_workflow("ci.yml")

    def test_triggers_on_push_to_main(self, workflow: dict):
        trigger = _get_trigger(workflow)
        branches = trigger.get("push", {}).get("branches", [])
        assert "main" in branches, "CI must trigger on push to main"

    def test_triggers_on_pr_to_main(self, workflow: dict):
        trigger = _get_trigger(workflow)
        branches = trigger.get("pull_request", {}).get("branches", [])
        assert "main" in branches, "CI must trigger on PRs targeting main"

    def test_the_workflows_own_jobs_run_on_linux(self, workflow: dict):
        """Every job `ci.yml` declares itself runs on Linux.

        ⚠️ Narrowed from an assertion over every *resolved* job, which swept in
        the delegated test matrix and so forbade the Windows and macOS legs
        `.system_design/TEST_SUITE.md` §8.4 requires. A job whose `runs-on:`
        defers to a matrix is left to :class:`TestThePlatformLegsExist`, which
        reads the matrix the expression points at; the exact labels are held by
        `RUNNER_JOB_CEILING_MINUTES` in the review-scripts suite.
        """
        for job_name, job in _resolve_jobs(workflow).items():
            runs_on = str(job.get("runs-on", ""))
            if runs_on.startswith("${{"):
                continue
            assert "ubuntu" in runs_on, f"Job '{job_name}' must run on ubuntu"

    def test_checks_out_code(self, workflow: dict):
        steps = _get_all_steps(workflow)
        uses = [s.get("uses", "") for s in steps]
        assert any("actions/checkout" in u for u in uses), "Must use actions/checkout"

    def test_installs_python(self, workflow: dict):
        steps = _get_all_steps(workflow)
        uses = [s.get("uses", "") for s in steps]
        assert any("actions/setup-python" in u or "setup-python" in u for u in uses), "Must use setup-python action"

    def test_installs_dependencies(self, workflow: dict):
        steps = _get_all_steps(workflow)
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "pip install" in run_cmds, "Must install dependencies via pip"

    def test_runs_tests(self, workflow: dict):
        steps = _get_all_steps(workflow)
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "pytest" in run_cmds, "Must run pytest"

    def test_runs_lint(self, workflow: dict):
        steps = _get_all_steps(workflow)
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "ruff" in run_cmds, "Must run ruff lint"


# ── R2: Publish workflow ─────────────────────────────────────────────────


class TestPublishWorkflow:
    """Verify the publish workflow triggers on tag push and publishes to PyPI."""

    @pytest.fixture()
    def workflow(self) -> dict:
        return _load_workflow("publish.yml")

    def test_triggers_on_tag_push(self, workflow: dict):
        trigger = _get_trigger(workflow)
        tags = trigger.get("push", {}).get("tags", [])
        assert any("v*" in t for t in tags), "Must trigger on v* tag push"

    def test_does_not_trigger_on_main_push(self, workflow: dict):
        trigger = _get_trigger(workflow)
        branches = trigger.get("push", {}).get("branches", [])
        assert "main" not in branches, "Publish must NOT trigger on push to main"

    def test_has_pypi_environment(self, workflow: dict):
        jobs = workflow.get("jobs", {})
        publish_job = jobs.get("publish")
        assert publish_job is not None, "Must have a 'publish' job"
        env = publish_job.get("environment", {})
        assert isinstance(env, dict), "publish job must have an environment block"
        assert env.get("name") == "pypi", "publish job must use 'pypi' environment"

    def test_has_oidc_permission(self, workflow: dict):
        jobs = workflow.get("jobs", {})
        publish_job = jobs.get("publish")
        assert publish_job is not None, "Must have a 'publish' job"
        perms = publish_job.get("permissions", {})
        assert perms.get("id-token") == "write", "publish job must have id-token: write for OIDC"

    def test_has_contents_read_permission(self, workflow: dict):
        jobs = workflow.get("jobs", {})
        publish_job = jobs.get("publish")
        assert publish_job is not None, "Must have a 'publish' job"
        perms = publish_job.get("permissions", {})
        assert perms.get("contents") == "read", "publish job must have contents: read for checkout"

    def test_checks_out_code(self, workflow: dict):
        steps = _get_all_steps(workflow)
        uses = [s.get("uses", "") for s in steps]
        assert any("actions/checkout" in u for u in uses), "Must use actions/checkout"

    def test_installs_build_dependency(self, workflow: dict):
        steps = _get_all_steps(workflow)
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "pip install build" in run_cmds, "Must install build tool before building"

    def test_builds_package(self, workflow: dict):
        steps = _get_all_steps(workflow)
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "python -m build" in run_cmds, "Must build the package"

    def test_uses_pypi_publish_action(self, workflow: dict):
        steps = _get_all_steps(workflow)
        uses = [s.get("uses", "") for s in steps]
        assert any("pypa/gh-action-pypi-publish" in u for u in uses), "Must use pypa/gh-action-pypi-publish action"

    def test_no_hardcoded_api_token(self, workflow: dict):
        """Verify the workflow does not reference any secrets (OIDC only)."""
        path = WORKFLOWS_DIR / "publish.yml"
        # UTF-8 for the reason `_load_workflow` gives. Missed when the other
        # reads were fixed, which is why that comment no longer claims to cover
        # "every case in this module" -- it did not.
        content = path.read_text(encoding="utf-8")
        # OIDC workflows should not reference any secrets at all
        assert re.search(r"\$\{\{\s*secrets\.", content) is None, (
            "OIDC publish workflow must not reference ${{ secrets.* }}"
        )


# ── R3: YAML validity ────────────────────────────────────────────────────


def _all_workflow_names() -> list[str]:
    """Return every workflow file, by glob rather than by enumeration.

    Both extensions, because GitHub accepts either and a ``.yaml`` file added
    later must not slip past a ``.yml``-only sweep.

    Returns:
        The file names, sorted.
    """
    found = list(WORKFLOWS_DIR.glob("*.yml")) + list(WORKFLOWS_DIR.glob("*.yaml"))
    return sorted(path.name for path in found)


class TestYAMLValidity:
    """Verify workflow files are valid YAML.

    🔴 **Swept, not enumerated, and this class was enumerated until a broken
    workflow proved why that is not enough.** It named ``ci.yml`` and
    ``publish.yml``; a malformed block scalar landed in ``claude-code-review.yml``
    and every case here stayed green, because the file nobody listed was the file
    nobody parsed. GitHub would have reported it only as the workflow silently
    never running.
    """

    @pytest.mark.parametrize("name", _all_workflow_names())
    def test_workflow_is_valid_yaml(self, name: str):
        workflow = _load_workflow(name)
        assert isinstance(workflow, dict), f"{name} must parse as a dict"
        assert "on" in workflow, f"{name} must have 'on' trigger"
        assert "jobs" in workflow, f"{name} must have 'jobs' key"

    def test_the_sweep_actually_found_the_workflows(self):
        """The control: a glob that matched nothing parametrises zero cases.

        Every case above would then be collected zero times and the class would
        pass having checked nothing -- the same silent-skip shape the docstring
        describes, one level up.
        """
        names = _all_workflow_names()

        assert len(names) >= 4, f"the workflow sweep found only {names!r}"
        for expected in ("ci.yml", "publish.yml", "tests.yml", "claude-code-review.yml"):
            assert expected in names, f"{expected} was not swept"


# ── R4: Tag version matches pyproject.toml ────────────────────────────────


class TestTagVersionCheck:
    """Verify the publish workflow checks tag version matches pyproject.toml."""

    @pytest.fixture()
    def workflow(self) -> dict:
        return _load_workflow("publish.yml")

    def test_verifies_tag_version(self, workflow: dict):
        steps = _get_all_steps(workflow)
        run_cmds = " ".join(s.get("run", "") for s in steps)
        assert "TAG_VERSION" in run_cmds, "Must extract TAG_VERSION from git ref"
        assert "PYPROJECT_VERSION" in run_cmds, "Must extract version from pyproject.toml"
        assert "exit 1" in run_cmds, "Must exit on version mismatch"


# ── R5: the same suite gates pull requests and releases ───────────────────


class TestReleaseIsGatedOnTests:
    """A tag must not reach PyPI without passing the checks a PR must pass.

    A published version can never be reused, so an untested release is not
    recoverable by pushing a fix — it burns the version number.
    """

    def test_publish_runs_the_test_suite_before_publishing(self):
        workflow = _load_workflow("publish.yml")
        jobs = workflow.get("jobs", {})

        publish = jobs.get("publish")
        assert publish is not None, "Must have a 'publish' job"

        needs = publish.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert needs, "publish job must declare a dependency on the test job"

        for dep in needs:
            assert dep in jobs, f"publish needs unknown job {dep!r}"
            assert str(jobs[dep].get("uses", "")).startswith("./.github/workflows/"), (
                f"job {dep!r} must delegate to the shared reusable test workflow"
            )

    def test_publish_and_ci_call_the_same_test_workflow(self):
        """One definition, so the release gate cannot drift from the PR gate."""

        def called_workflows(name: str) -> set[str]:
            return {
                str(job["uses"])
                for job in _load_workflow(name).get("jobs", {}).values()
                if str(job.get("uses", "")).startswith("./.github/workflows/")
            }

        ci_called = called_workflows("ci.yml")
        publish_called = called_workflows("publish.yml")

        assert ci_called, "ci.yml must delegate its tests to the reusable workflow"
        assert ci_called == publish_called, (
            f"ci.yml calls {ci_called} but publish.yml calls {publish_called} — "
            "a release would run different checks than a pull request"
        )

    def test_reusable_workflow_is_callable(self):
        workflow = _load_workflow("tests.yml")
        assert "workflow_call" in _get_trigger(workflow), "tests.yml must be reusable via workflow_call"

    def test_reusable_workflow_runs_lint_imports_and_pytest(self):
        steps = _get_all_steps(_load_workflow("tests.yml"))
        run_cmds = " ".join(s.get("run", "") for s in steps)

        assert "pytest" in run_cmds, "Must run pytest"
        assert "ruff check" in run_cmds, "Must run ruff"
        assert "lint-imports" in run_cmds, "Must enforce the import layering contract"

    def test_blocking_steps_do_not_swallow_failures(self):
        """continue-on-error is only acceptable on steps documented as advisory."""
        advisory = {"Type check (informational)"}
        for step in _get_all_steps(_load_workflow("tests.yml")):
            if step.get("continue-on-error"):
                assert step.get("name") in advisory, (
                    f"step {step.get('name')!r} silently ignores failures but is not marked advisory"
                )

    def test_matrix_covers_every_supported_python(self):
        """The tested versions must match what pyproject.toml claims to support."""
        jobs = _load_workflow("tests.yml").get("jobs", {})
        tested = set()
        for job in jobs.values():
            tested.update(str(v) for v in job.get("strategy", {}).get("matrix", {}).get("python-version", []))

        # Explicitly UTF-8, for the reason `_load_workflow` gives: TOML is
        # defined as UTF-8, and reading it in the platform's locale encoding
        # fails on a Windows developer's machine while passing on the runner.
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        claimed = set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", pyproject))

        assert claimed, "expected Python version classifiers in pyproject.toml"
        assert claimed <= tested, f"classifiers claim {sorted(claimed - tested)} but CI never tests them"


# ── R5b: the gate runs on every platform the product ships to ─────────────


#: The platform families the Fast gate must cover, as substrings of a runner
#: label. Substrings rather than exact labels deliberately: `ubuntu-latest` and
#: `ubuntu-24.04` are the same platform for this file's question, and pinning
#: exact labels here would make a routine runner-image bump read as a lost leg.
#: The exact labels are pinned elsewhere and for a different reason, by
#: `RUNNER_JOB_CEILING_MINUTES` in `.github/review/tests/test_review_scripts.py`.
REQUIRED_PLATFORMS = ("ubuntu", "windows", "macos")


def _matrix_platforms(job: dict) -> set[str]:
    """Collect every ``os`` value a job's matrix can produce.

    Reads the top-level ``os`` list **and** every ``include:`` entry's ``os``.
    An include entry whose value would overwrite a base matrix value does not
    modify an existing combination -- GitHub creates a **new** one from it, and
    that is precisely how the Windows and macOS legs come to exist. An include
    entry read as a mere annotation would leave both legs invisible here.

    ``exclude:`` is deliberately not honoured: it only removes combinations, so
    ignoring it over-approximates, which is the safe direction for a question
    about whether a platform is *present*.

    Args:
        job: A parsed job mapping, as `tests.yml` declares it.

    Returns:
        The ``os`` values the matrix can produce. Empty when the job declares no
        matrix, or none this function can read -- see
        :meth:`TestThePlatformLegsExist.test_the_reader_reports_an_unreadable_matrix_as_empty`.
    """
    matrix = job.get("strategy", {}).get("matrix", {})
    platforms = {str(value) for value in matrix.get("os", [])}
    # An include entry without an `os` key annotates a combination rather than
    # naming a runner, so it contributes no platform.
    for entry in matrix.get("include", []):
        if "os" in entry:
            platforms.add(str(entry["os"]))
    return platforms


class TestThePlatformLegsExist:
    """The Fast gate must run on Linux, Windows and macOS.

    Until KBR-164 every job in the repository ran on Linux, so a Windows-only or
    macOS-only defect was reachable only by a user reporting one -- which is how
    all three platform bugs on epic KBR-123 were in fact found.
    ``.system_design/TEST_SUITE.md`` §8.4 records the matrix and its reasoning;
    this class is what stops a later edit quietly removing a leg.
    """

    @pytest.fixture()
    def test_job(self) -> dict:
        jobs = _load_workflow("tests.yml").get("jobs", {})
        assert "test" in jobs, "tests.yml must declare the `test` job"
        return jobs["test"]

    def test_the_reader_reports_an_unreadable_matrix_as_empty(self):
        """A matrix this reader cannot understand must read as no platforms.

        🔴 Fabricated input, not the production workflow, for the reason
        TEST_SUITE.md §8.3 gives about its own registry validator: a reader
        proved only against a matrix that works cannot show it fails *safe*. If
        an unreadable matrix returned something non-empty, the coverage
        assertions below would report on platforms nobody declared.
        """
        assert _matrix_platforms({}) == set()
        assert _matrix_platforms({"strategy": {"matrix": {"python-version": ["3.12"]}}}) == set()

    def test_the_reader_sees_a_leg_that_exists_only_as_an_include_entry(self):
        """The `include:` path is the one that produces both platform legs.

        Fabricated for the same reason as above, and pointed at the specific
        shape `tests.yml` uses: a reader that returned only the top-level `os`
        list would report a Linux-only matrix while two platform legs ran.
        """
        fabricated = {
            "strategy": {
                "matrix": {
                    "os": ["ubuntu-latest"],
                    "include": [{"os": "windows-latest", "python-version": "3.12"}],
                }
            }
        }
        assert _matrix_platforms(fabricated) == {"ubuntu-latest", "windows-latest"}

    def test_the_matrix_names_a_platform_at_all(self, test_job: dict):
        """Guard the guard: an empty read makes every assertion below vacuous."""
        assert _matrix_platforms(test_job), (
            "no `os` value could be read from the test matrix, so every platform "
            "assertion in this class would pass by asking nothing"
        )

    @pytest.mark.parametrize("platform", REQUIRED_PLATFORMS)
    def test_the_gate_runs_on_every_supported_platform(self, test_job: dict, platform: str):
        produced = _matrix_platforms(test_job)
        assert any(platform in label for label in produced), (
            f"the Fast gate never runs on {platform}; the matrix produces "
            f"{sorted(produced)}. Dropping a platform leg makes every "
            f"{platform}-only defect invisible until a user reports one -- see "
            f"`.system_design/TEST_SUITE.md` §8.4."
        )

    def test_runs_on_reads_the_matrix(self, test_job: dict):
        """A matrix of platforms is inert unless `runs-on:` defers to it.

        Named separately because the two halves fail independently: a matrix
        naming three platforms under a hard-coded `runs-on: ubuntu-latest` runs
        the Linux suite six times and reports three platforms' worth of green.
        """
        assert test_job.get("runs-on") == "${{ matrix.os }}", (
            f"the matrix names platforms but `runs-on:` is "
            f"{test_job.get('runs-on')!r}, so every leg runs on one machine"
        )

    def test_every_platform_leg_pins_a_supported_python(self, test_job: dict):
        """An `include:` leg pins its own version, and it must be one we claim.

        The base matrix is checked against the classifiers by
        :meth:`TestReleaseIsGatedOnTests.test_matrix_covers_every_supported_python`,
        which reads the `python-version` list and never sees an include entry.
        """
        # Explicitly UTF-8, for the reason `_load_workflow` gives: TOML is
        # defined as UTF-8, and the platform locale would differ on the very
        # Windows leg this class exists to add.
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        claimed = set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", pyproject))
        include = test_job.get("strategy", {}).get("matrix", {}).get("include", [])
        assert include, "the platform legs are include entries; none were found"
        for entry in include:
            pinned = str(entry.get("python-version", ""))
            assert pinned in claimed, (
                f"platform leg {entry.get('os')!r} pins Python {pinned!r}, which "
                f"pyproject.toml does not claim to support ({sorted(claimed)})"
            )


# ── R6: metadata refresh must work under branch protection ────────────────


class TestMetadataRefreshRespectsBranchProtection:
    """The refresh job must not push to a protected branch.

    `main` requires changes to arrive via pull request. The job used to run
    `git push` directly and had been failing on every push since that rule was
    enabled, so the bundled catalogue silently stopped being refreshed and
    unrelated pull requests started failing a freshness check nothing could
    satisfy.

    The job now lives in ``model-metadata.yml`` rather than ``ci.yml``. It was
    moved when ``ci.yml`` became the merge gate: ``ci-required`` must depend on
    every job in that file and compare each against ``success``, and this job
    skips on ``push`` -- a skipped dependency is not a success, so keeping it
    there would have reddened ``main`` on every merge. See that file's header.
    """

    WORKFLOW = "model-metadata.yml"

    @pytest.fixture()
    def steps(self) -> list[dict]:
        workflow = _load_workflow(self.WORKFLOW)
        return workflow["jobs"]["update-metadata"]["steps"]

    def test_no_step_pushes_to_the_default_branch(self, steps: list[dict]):
        offenders = [
            step.get("name")
            for step in steps
            if re.search(r"git push(?!\s+--force origin \"\$BRANCH\")\s*$", step.get("run", ""), re.M)
        ]

        assert not offenders, f"these steps push directly to main, which branch protection rejects: {offenders}"

    def test_push_does_not_use_a_lease_it_cannot_verify(self, steps: list[dict]):
        """``--force-with-lease`` fails outright on a shallow checkout.

        actions/checkout fetches only the default branch at depth 1, so the
        runner holds no remote-tracking ref for the refresh branch. git cannot
        evaluate the lease and rejects the push with "stale info" whenever that
        branch still exists on the remote -- turning main red and skipping the
        refresh. The lease protects nothing here: the branch is this job's own
        scratch space and is meant to be overwritten.
        """
        pushes = [
            line.strip()
            for step in steps
            for line in step.get("run", "").splitlines()
            if line.strip().startswith("git push")
        ]

        assert pushes, "expected the refresh to push its branch"
        assert not [p for p in pushes if "--force-with-lease" in p], pushes

    def test_refresh_does_not_run_on_every_push_to_main(self):
        """One refresh a week, not one per merge.

        The job opened a pull request on every push to main, and merging one of
        those is itself a push to main, so the refresh partly fed itself -- #9
        merged at 19:36:30 and #10 was opened two seconds later. On a busy day
        that is several near-empty pull requests against a catalogue that needs
        to be current to the week, not to the minute.
        """
        workflow = _load_workflow(self.WORKFLOW)

        # Since the move the exclusion is expressed by the workflow's own
        # triggers rather than by an `if:` on the job -- same outcome, one fewer
        # queued-and-skipped job per merge. Either spelling satisfies the claim;
        # what must not happen is the refresh running on every push to main.
        #
        # 🔴 The `if:` branch checks the CONDITION EXCLUDES push, not that the
        # word appears in it. The first version asserted `"push" in condition`,
        # which `if: github.event_name == 'push'` satisfies -- the exact opposite
        # of the claim. A guard that a negation and its inverse both pass is not
        # a guard.
        condition = str(workflow["jobs"]["update-metadata"].get("if", ""))
        triggers = _get_trigger(workflow)

        if "push" not in triggers:
            return

        normalised = condition.replace(" ", "")
        assert "github.event_name!='push'" in normalised, (
            "update-metadata is reachable on a push to main and its `if:` does "
            "not exclude that event, so merging its own pull request "
            f"re-triggers it; triggers={sorted(triggers)!r} if={condition!r}"
        )

    def test_gating_the_refresh_does_not_stop_main_being_tested(self):
        """Moving the refresh out must not have taken the gate's push trigger with it."""
        ci = _load_workflow("ci.yml")

        assert "main" in _get_trigger(ci).get("push", {}).get("branches", [])
        assert "if" not in ci["jobs"]["test"]

    def test_refresh_opens_a_pull_request(self, steps: list[dict]):
        run_commands = " ".join(step.get("run", "") for step in steps)

        assert "gh pr create" in run_commands, "the refresh must propose its change as a pull request"

    def test_refresh_has_permission_to_open_one(self):
        """Declared on the job now, not on the workflow.

        It used to be a workflow-level grant in `ci.yml`, which meant every job
        added to that file inherited a write token it had no use for -- and two
        such jobs were about to be added. The grant moved with the job and is now
        scoped to it.
        """
        job = _load_workflow(self.WORKFLOW)["jobs"]["update-metadata"]

        assert job.get("permissions", {}).get("pull-requests") == "write"
        assert job.get("permissions", {}).get("contents") == "write"

    def test_pull_requests_are_not_failed_by_metadata_drift(self, steps: list[dict]):
        """The check compares against a live API, so it can never be satisfiable.

        Enforcing it means unrelated pull requests go red whenever OpenRouter
        publishes a model between the last refresh and the pull request.
        """
        pr_steps = [s for s in steps if "pull_request'" in str(s.get("if", ""))]
        assert pr_steps, "expected a pull_request-scoped metadata step"

        for step in pr_steps:
            assert "--exit-code" not in step.get("run", ""), (
                f"step {step.get('name')!r} fails the build on metadata drift; it should report instead"
            )

# ── R5/R6: the type check is enforced, and not gamed ──────


class TestTypeCheckIsEnforced:
    """mypy must fail the build, not merely report.

    It was advisory while the codebase carried 98 errors. Those are now zero,
    and five of them turned out to be real defects, so the check earns a gate:
    nothing else stops the count climbing back.
    """

    def test_mypy_step_exists_and_is_blocking(self):
        steps = _get_all_steps(_load_workflow("tests.yml"))
        mypy_steps = [s for s in steps if "mypy" in s.get("run", "")]

        assert mypy_steps, "no mypy step found in the reusable test workflow"
        for step in mypy_steps:
            assert not step.get("continue-on-error"), (
                f"step {step.get('name')!r} runs mypy but swallows its failures"
            )

    def test_mypy_runs_before_the_test_suite(self):
        """A type error should surface in seconds, not after an 11-minute run."""
        commands = [s.get("run", "") for s in _get_all_steps(_load_workflow("tests.yml"))]
        mypy_at = next(i for i, c in enumerate(commands) if "mypy" in c)
        pytest_at = next(i for i, c in enumerate(commands) if "pytest" in c)

        assert mypy_at < pytest_at, "mypy runs after the suite, so type errors are reported last"


def _pytest_invocations() -> list[str]:
    """Return every ``pytest`` command line across all workflows.

    Returns:
        One string per ``run:`` step that invokes pytest, whitespace collapsed
        so a folded YAML scalar reads the same as a single line.

    Delegates to :func:`layers.workflow_pytest_invocations` so that this module
    and ``tests/test_layer_selection.py`` cannot disagree about which jobs
    exist.  They check opposite directions, and a sweep that misses a workflow
    fails **open** in one of them.
    """
    return workflow_pytest_invocations(WORKFLOWS_DIR)


class TestEveryJobVerifiesTheCategoriesItClaims:
    """A job must require every layer its own marker expression selects.

    ``-m "acceptance or agent_smoke"`` is satisfied by ``acceptance`` alone: the
    job collects, passes, and reports a category it never ran. ``TEST_SUITE.md``
    §8 records that an earlier draft of the design believed pytest would exit 5
    in that case and says plainly: "That is wrong."

    ``--require-category`` is the fix, and this is what stops a *future* job
    from being added without it -- which is why the sweep covers every workflow
    rather than naming the one job that exists today.
    """

    def test_the_sweep_actually_found_the_pytest_invocations(self):
        """The self-check: prove the scan still matches something.

        A structural guard that has rotted into matching nothing passes forever
        and silently. This is the pattern ``tests/test_egress_coverage.py``
        established and §6.2 requires of every guard here.
        """
        assert _pytest_invocations(), "no pytest invocation found in any workflow"

    def test_the_fast_job_selects_the_two_gating_layers(self):
        """Pin the divided command the whole plan schedules against.

        Parallel authors need the selection expression from day one; this is the
        artifact they read, so a change to it should be deliberate enough to
        update a test.
        """
        commands = " || ".join(_pytest_invocations())

        assert 'pytest -m "l1 or l2"' in commands

    def test_no_job_runs_pytest_without_naming_a_marker_expression(self):
        """Every job says which layers it runs.

        A bare ``pytest`` in a workflow inherits the default exclusion from
        ``addopts``, which is a developer-convenience setting. A CI job that
        depends on it is a job whose coverage can be changed by editing a
        comment about local runs.
        """
        undeclared = [
            cmd for cmd in _pytest_invocations() if marker_expression_of(cmd) is None
        ]

        assert undeclared == [], f"these CI pytest runs name no layers: {undeclared}"

    def test_the_gate_passes_strict_markers_on_the_command_line(self):
        """``--strict-markers`` belongs here, not in ``addopts``.

        pytest 9.0.x silently ignores it in ``addopts``; 9.1.0 honours it
        (upstream issue 14442). With ``pytest>=8.0`` and no upper bound, a
        config-file placement would make the gate strict for one contributor and
        not another. Asserted here, and asserted *absent* from ``addopts`` in
        ``tests/test_layer_markers.py`` -- the natural tidy-up is to move it
        there, and one assertion alone would not notice.
        """
        commands = " || ".join(_pytest_invocations())

        assert "--strict-markers" in commands

    def test_each_job_requires_every_layer_its_expression_selects(self):
        """The pairing that makes a job's claim about itself checkable.

        "Selects" is computed by evaluating the expression, never by looking for
        layer names in the string. The two differ on every expression containing
        ``not``: a string scan reads ``-m "not agent_live"`` as a job that runs
        the live-agent tests and would demand it guarantee them.
        """
        gaps: list[str] = []

        for command in _pytest_invocations():
            expression = marker_expression_of(command)
            if expression is None:
                continue

            unverified = set(positively_selected_layers(expression)) - required_categories_of(
                command
            )
            if unverified:
                gaps.append(f"{command!r} selects {sorted(unverified)} without requiring it")

        assert gaps == []
