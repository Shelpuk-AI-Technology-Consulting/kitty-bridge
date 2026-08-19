"""Tests for GitHub Actions workflow correctness.

Verifies that the CI/CD workflow files are valid, well-structured, and
implement the expected publish-on-tag pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

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
    with open(path) as f:
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

    def test_runs_on_ubuntu(self, workflow: dict):
        jobs = _resolve_jobs(workflow)
        for job_name, job in jobs.items():
            runs_on = job.get("runs-on", "")
            assert "ubuntu" in str(runs_on), f"Job '{job_name}' must run on ubuntu"

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
        content = path.read_text()
        # OIDC workflows should not reference any secrets at all
        assert re.search(r"\$\{\{\s*secrets\.", content) is None, (
            "OIDC publish workflow must not reference ${{ secrets.* }}"
        )


# ── R3: YAML validity ────────────────────────────────────────────────────


class TestYAMLValidity:
    """Verify workflow files are valid YAML."""

    @pytest.mark.parametrize("name", ["ci.yml", "publish.yml"])
    def test_workflow_is_valid_yaml(self, name: str):
        workflow = _load_workflow(name)
        assert isinstance(workflow, dict), f"{name} must parse as a dict"
        assert "on" in workflow, f"{name} must have 'on' trigger"
        assert "jobs" in workflow, f"{name} must have 'jobs' key"


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

        pyproject = (ROOT / "pyproject.toml").read_text()
        claimed = set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", pyproject))

        assert claimed, "expected Python version classifiers in pyproject.toml"
        assert claimed <= tested, f"classifiers claim {sorted(claimed - tested)} but CI never tests them"


# ── R6: metadata refresh must work under branch protection ────────────────


class TestMetadataRefreshRespectsBranchProtection:
    """The refresh job must not push to a protected branch.

    `main` requires changes to arrive via pull request. The job used to run
    `git push` directly and had been failing on every push since that rule was
    enabled, so the bundled catalogue silently stopped being refreshed and
    unrelated pull requests started failing a freshness check nothing could
    satisfy.
    """

    @pytest.fixture()
    def steps(self) -> list[dict]:
        workflow = _load_workflow("ci.yml")
        return workflow["jobs"]["update-metadata"]["steps"]

    def test_no_step_pushes_to_the_default_branch(self, steps: list[dict]):
        offenders = [
            step.get("name")
            for step in steps
            if re.search(r"git push(?!\s+--force-with-lease origin \"\$BRANCH\")\s*$", step.get("run", ""), re.M)
        ]

        assert not offenders, f"these steps push directly to main, which branch protection rejects: {offenders}"

    def test_refresh_opens_a_pull_request(self, steps: list[dict]):
        run_commands = " ".join(step.get("run", "") for step in steps)

        assert "gh pr create" in run_commands, "the refresh must propose its change as a pull request"

    def test_refresh_has_permission_to_open_one(self):
        workflow = _load_workflow("ci.yml")

        assert workflow.get("permissions", {}).get("pull-requests") == "write"

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
