"""Tests for PyPI packaging correctness.

Verifies that the built package meets PyPI publication requirements:
- Metadata completeness
- Correct entry points
- Build artifact contents
- twine check passes
"""

from __future__ import annotations

import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def built_artifacts() -> dict[str, Path]:
    """Build once and return both wheel and sdist paths.

    Uses a single fixture to avoid a race condition where two session-scoped
    fixtures both trigger _build_dist() concurrently.
    """
    DIST_DIR.mkdir(exist_ok=True)
    if not list(DIST_DIR.glob("*.whl")):
        subprocess.run(
            ["python", "-m", "build", "--no-isolation"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    wheels = list(DIST_DIR.glob("*.whl"))
    sdists = list(DIST_DIR.glob("*.tar.gz"))
    assert len(wheels) == 1, f"Expected exactly 1 wheel, found: {wheels}"
    assert len(sdists) == 1, f"Expected exactly 1 sdist, found: {sdists}"
    return {"wheel": wheels[0], "sdist": sdists[0]}


@pytest.fixture(scope="session")
def built_wheel(built_artifacts: dict[str, Path]) -> Path:
    return built_artifacts["wheel"]


@pytest.fixture(scope="session")
def built_sdist(built_artifacts: dict[str, Path]) -> Path:
    return built_artifacts["sdist"]


@pytest.fixture(scope="session")
def sdist_names(built_sdist: Path) -> list[str]:
    """Read file names from the built sdist (tar.gz)."""
    with tarfile.open(built_sdist, "r:gz") as t:
        return t.getnames()


@pytest.fixture(scope="session")
def wheel_metadata(built_wheel: Path) -> str:
    """Read METADATA from the built wheel."""
    with zipfile.ZipFile(built_wheel) as z:
        for name in z.namelist():
            if name.endswith("/METADATA"):
                return z.read(name).decode("utf-8")
    pytest.fail("METADATA not found in wheel")


def _gitignore_lines() -> list[str]:
    """Read .gitignore as a list of stripped lines."""
    return [line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()]


# ── R1: Metadata completeness ────────────────────────────────────────────


class TestMetadataCompleteness:
    """Verify pyproject.toml metadata is complete and correct."""

    def test_package_name(self, wheel_metadata: str):
        assert "Name: kitty-bridge" in wheel_metadata

    def test_version_present(self, wheel_metadata: str):
        assert "Version:" in wheel_metadata

    def test_summary_present(self, wheel_metadata: str):
        assert "Summary:" in wheel_metadata

    def test_author_present(self, wheel_metadata: str):
        assert "Author:" in wheel_metadata or "Author-email:" in wheel_metadata

    def test_license_present(self, wheel_metadata: str):
        assert "License-File: LICENSE" in wheel_metadata

    def test_homepage_url(self, wheel_metadata: str):
        assert "Project-URL: Homepage" in wheel_metadata

    def test_repository_url(self, wheel_metadata: str):
        assert "Project-URL: Repository" in wheel_metadata

    def test_python_requires(self, wheel_metadata: str):
        assert "Requires-Python: >=3.10" in wheel_metadata

    def test_readme_in_metadata(self, wheel_metadata: str):
        """Long description (from README) must be present."""
        parts = wheel_metadata.split("\n\n", 1)
        assert len(parts) == 2, "METADATA must have a body (readme content)"
        body = parts[1]
        assert len(body) > 100, "Long description should contain README content"

    def test_classifiers_present(self, wheel_metadata: str):
        classifiers = [line for line in wheel_metadata.splitlines() if line.startswith("Classifier:")]
        assert len(classifiers) >= 4, f"Expected >= 4 classifiers, got {len(classifiers)}"

    def test_classifier_license(self, wheel_metadata: str):
        assert any("License :: OSI Approved :: MIT License" in line for line in wheel_metadata.splitlines())

    def test_classifier_python3(self, wheel_metadata: str):
        assert any("Programming Language :: Python :: 3" in line for line in wheel_metadata.splitlines())


# ── R2: twine check passes ───────────────────────────────────────────────


class TestTwineCheck:
    """Verify that twine check passes cleanly."""

    def test_twine_check(self, built_wheel: Path, built_sdist: Path):
        result = subprocess.run(
            ["twine", "check", str(built_wheel), str(built_sdist)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"twine check failed:\n{result.stdout}\n{result.stderr}"
        assert "WARNING" not in result.stdout, f"twine check warnings:\n{result.stdout}"


# ── R3: .gitignore covers build artifacts ─────────────────────────────────


class TestGitignore:
    """Verify .gitignore covers build artifacts."""

    def test_gitignore_covers_dist(self):
        lines = _gitignore_lines()
        assert any("dist" in line and not line.startswith("#") for line in lines)

    def test_gitignore_covers_build(self):
        lines = _gitignore_lines()
        assert any("build" in line and not line.startswith("#") for line in lines)

    def test_gitignore_covers_egg_info(self):
        lines = _gitignore_lines()
        assert any("egg-info" in line for line in lines)


# ── R4: Build artifact contents ──────────────────────────────────────────


class TestBuildArtifacts:
    """Verify the build output is correct."""

    def test_wheel_contains_kitty_package(self, built_wheel: Path):
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
        assert any("kitty/__init__.py" in n for n in names), "kitty package must be in wheel"

    def test_wheel_contains_entry_point(self, built_wheel: Path):
        with zipfile.ZipFile(built_wheel) as z:
            for name in z.namelist():
                if name.endswith("entry_points.txt"):
                    content = z.read(name).decode()
                    for line in content.splitlines():
                        if line.strip() == "kitty = kitty.cli.main:main":
                            return
                    pytest.fail("Entry point 'kitty = kitty.cli.main:main' not found")
        pytest.fail("entry_points.txt not found")

    def test_wheel_contains_license(self, built_wheel: Path):
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
        assert any(n.endswith("/LICENSE") or n == "LICENSE" for n in names), (
            "LICENSE must be in wheel (exact match)"
        )

    def test_wheel_contains_main_module(self, built_wheel: Path):
        with zipfile.ZipFile(built_wheel) as z:
            names = z.namelist()
        assert any("kitty/__main__.py" in n for n in names), "kitty/__main__.py must be in wheel"

    def test_sdist_contains_readme(self, sdist_names: list[str]):
        assert any("README.md" in n for n in sdist_names), "README.md must be in sdist"

    def test_sdist_contains_license(self, sdist_names: list[str]):
        assert any(n.endswith("/LICENSE") or n.endswith("LICENSE") for n in sdist_names), "LICENSE must be in sdist"

    def test_sdist_contains_pyproject(self, sdist_names: list[str]):
        assert any("pyproject.toml" in n for n in sdist_names), "pyproject.toml must be in sdist"

    def test_sdist_contains_source(self, sdist_names: list[str]):
        assert any("kitty/__init__.py" in n for n in sdist_names), "kitty source must be in sdist"


# ── R5: Version consistency ──────────────────────────────────────────────


class TestVersionConsistency:
    """Verify version is consistent across files."""

    def test_version_in_init_matches_pyproject(self):
        # Read version from source file to avoid importing stale installed package
        init_content = (ROOT / "src" / "kitty" / "__init__.py").read_text()
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_content)
        assert m, "__version__ not found in __init__.py"
        init_version = m.group(1)

        pyproject = (ROOT / "pyproject.toml").read_text()
        for line in pyproject.splitlines():
            if line.startswith("version ="):
                pyproject_version = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        else:
            pytest.fail("version not found in pyproject.toml")

        assert init_version == pyproject_version, (
            f"Version mismatch: __init__.py={init_version}, pyproject.toml={pyproject_version}"
        )
