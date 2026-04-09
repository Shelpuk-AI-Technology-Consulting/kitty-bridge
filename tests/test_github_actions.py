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


def _get_all_steps(workflow: dict) -> list[dict]:
    """Flatten all steps from all jobs in a workflow."""
    steps: list[dict] = []
    for job in workflow.get("jobs", {}).values():
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
        jobs = workflow.get("jobs", {})
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
        assert any("actions/setup-python" in u or "setup-python" in u for u in uses), (
            "Must use setup-python action"
        )

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
        assert any("pypa/gh-action-pypi-publish" in u for u in uses), (
            "Must use pypa/gh-action-pypi-publish action"
        )

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
