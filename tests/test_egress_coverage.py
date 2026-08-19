"""Structural guard: every outbound HTTP client must be egress-aware.

The egress feature promises that when a proxy is configured, no provider-bound
request leaves from the machine's own address. Behavioural tests can only cover
code paths someone remembered to test; this file enumerates every HTTP client
construction in ``src/`` and fails when a new one appears that has not been
reviewed for egress.

If this test fails, you added an HTTP client. Either route it through the
egress proxy and add it to the allowlist below, or explain in the allowlist why
it must not be proxied.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "kitty"

#: Patterns that open a connection to somewhere the user does not control.
_CLIENT_PATTERNS = {
    "aiohttp session": re.compile(r"aiohttp\.ClientSession\("),
    "curl_cffi session": re.compile(r"curl_cffi\.requests\.AsyncSession\("),
    "boto3 client": re.compile(r"\.client\(\s*[\"']bedrock-runtime[\"']"),
    "urllib": re.compile(r"urllib\.request\.urlopen\("),
}

#: Every known client, and how it satisfies (or is exempt from) egress.
#:
#: Keyed by ``(relative posix path, kind)`` with the expected number of
#: occurrences, so a *new* client inside an already-listed file is still caught —
#: bridge/server.py is 5,000+ lines and is the likeliest home for the next one.
_EXPECTED_COUNTS: dict[tuple[str, str], int] = {
    ("bridge/server.py", "aiohttp session"): 1,
    ("providers/ollama_cloud.py", "aiohttp session"): 1,
    ("providers/openai_subscription.py", "aiohttp session"): 2,
    ("providers/openai_subscription.py", "curl_cffi session"): 1,
    ("providers/bedrock.py", "boto3 client"): 2,
    ("validation.py", "aiohttp session"): 1,
    ("auth/openai_oauth.py", "aiohttp session"): 1,
    ("cli/egress_cmd.py", "aiohttp session"): 1,
    ("bridge/manage.py", "urllib"): 1,
}

_ALLOWLIST: dict[str, str] = {
    # Proxied: session built with proxy=/proxy_auth= when egress is configured.
    "bridge/server.py": "two sessions; _session_for() picks proxied vs direct by destination",
    "providers/ollama_cloud.py": "session built with aiohttp_session_kwargs()",
    "providers/openai_subscription.py": "curl_cffi proxies= and OAuth sessions via aiohttp_session_kwargs()",
    "providers/bedrock.py": "botocore Config(proxies=...); SSO mode reports supports_egress()=False",
    "validation.py": "session built with aiohttp_session_kwargs(egress); fails closed under egress",
    "auth/openai_oauth.py": "session built with aiohttp_session_kwargs()",
    "cli/egress_cmd.py": "the gateway self-test; proxied by definition",
    # Exempt: never leaves the machine.
    "bridge/manage.py": "localhost /healthz poll — must stay direct, see should_bypass()",
}


def _iter_client_sites() -> list[tuple[str, int, str, str]]:
    """Find every HTTP client construction under ``src/kitty``.

    Returns:
        ``(relative_path, line_number, kind, line)`` for each match.
    """
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for kind, pattern in _CLIENT_PATTERNS.items():
                if pattern.search(line):
                    found.append((rel, lineno, kind, line.strip()))
    return found


class TestEveryHttpClientIsAccountedFor:
    """R14: no unreviewed HTTP client may exist in the source tree."""

    def test_no_unallowlisted_http_clients(self):
        offenders = [
            f"{rel}:{lineno} ({kind}) -> {line}"
            for rel, lineno, kind, line in _iter_client_sites()
            if rel not in _ALLOWLIST
        ]

        assert not offenders, (
            "New HTTP client(s) found that are not covered by the egress review:\n  "
            + "\n  ".join(offenders)
            + "\n\nRoute them through the egress proxy and add them to _ALLOWLIST in "
            "tests/test_egress_coverage.py, or document there why they must stay direct."
        )

    def test_allowlist_has_no_stale_entries(self):
        """A stale entry would silently excuse a file that no longer exists."""
        live = {rel for rel, _lineno, _kind, _line in _iter_client_sites()}
        stale = sorted(set(_ALLOWLIST) - live)

        assert not stale, f"_ALLOWLIST names files with no HTTP client any more: {stale}"

    def test_the_scan_actually_finds_something(self):
        """Guards against a broken regex quietly passing the suite."""
        assert len(_iter_client_sites()) >= len(_ALLOWLIST)

    def test_no_new_client_hides_inside_an_already_listed_file(self):
        """A per-file allowlist would wave through a second client in the same file."""
        actual: dict[tuple[str, str], int] = {}
        for rel, _lineno, kind, _line in _iter_client_sites():
            actual[(rel, kind)] = actual.get((rel, kind), 0) + 1

        added = {k: v for k, v in actual.items() if v != _EXPECTED_COUNTS.get(k)}
        removed = {k: v for k, v in _EXPECTED_COUNTS.items() if k not in actual}

        detail = []
        for (rel, kind), count in sorted(added.items()):
            expected = _EXPECTED_COUNTS.get((rel, kind), 0)
            lines = [ln for r, ln, k, _ in _iter_client_sites() if (r, k) == (rel, kind)]
            detail.append(f"{rel} ({kind}): expected {expected}, found {count} at lines {lines}")
        for (rel, kind), count in sorted(removed.items()):
            detail.append(f"{rel} ({kind}): expected {count}, found none")

        assert not detail, "HTTP client counts changed — review each for egress, then update " + (
            "_EXPECTED_COUNTS in tests/test_egress_coverage.py: " + "; ".join(detail)
        )


class TestNoProxyEnvironmentVariables:
    """R15: kitty configures clients explicitly, never through the environment.

    The three HTTP stacks disagree about proxy environment variables — aiohttp
    ignores them unless ``trust_env=True`` while curl_cffi and botocore honour
    them — so setting them would proxy some traffic and silently leak the rest,
    and would also tunnel the bridge's own localhost health check.
    """

    @pytest.mark.parametrize("var", ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"])
    def test_source_never_assigns_proxy_env_vars(self, var: str):
        assignment = re.compile(rf"""environ\[\s*["']{var}["']\s*\]\s*=|setenv\(\s*["']{var}["']""")
        offenders = [
            f"{path.relative_to(SRC).as_posix()}:{lineno}"
            for path in sorted(SRC.rglob("*.py"))
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if assignment.search(line)
        ]

        assert not offenders, f"kitty must not set {var}; found assignments at {offenders}"

    def test_no_session_trusts_the_environment(self):
        """``trust_env=True`` would reintroduce the inconsistency above.

        Parsed rather than grepped, so prose mentioning the flag in a docstring
        does not trip the check.
        """
        offenders: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if (
                        keyword.arg == "trust_env"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        offenders.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")

        assert not offenders, f"trust_env=True found at {offenders}"


class TestEveryStartPathIsGuarded:
    """R10 structurally: a new way to start a bridge must not skip the check.

    The first version of this feature wired the fail-closed guard into the agent
    launcher only, leaving foreground `kitty bridge` and the background runner
    able to start with a provider that cannot honour the proxy. Counting call
    sites catches that class of omission; a behavioural test of the guard
    function cannot.
    """

    @staticmethod
    def _files_calling(name: str) -> set[str]:
        """Return source files containing a call to ``name``."""
        pattern = re.compile(rf"\b{re.escape(name)}\(")
        return {
            path.relative_to(SRC).as_posix()
            for path in SRC.rglob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        }

    def test_every_file_constructing_a_bridge_also_checks_egress(self):
        constructors = self._files_calling("BridgeServer") - {"bridge/server.py"}
        guarded = self._files_calling("egress_block_reason")

        unguarded = sorted(constructors - guarded)

        assert not unguarded, (
            "these files start a BridgeServer without calling egress_block_reason, so a provider "
            f"that cannot be proxied would leak from them: {unguarded}"
        )

    def test_the_scan_finds_the_known_start_paths(self):
        """Guards against the regex silently matching nothing."""
        constructors = self._files_calling("BridgeServer") - {"bridge/server.py"}

        assert constructors == {"cli/launcher.py", "cli/main.py", "bridge_runner.py"}


class TestTypeSuppressionsAreSpecific:
    """R5: a clean type check must not be achieved by silencing it.

    With mypy now blocking, the cheapest way to make it pass is a blanket
    `# type: ignore`, which disables every check on that line — including the
    class of error that turned up four real defects. Each suppression must name
    the codes it silences, so it stops applying when the code changes.
    """

    @staticmethod
    def _suppressions() -> list[tuple[str, int, str]]:
        """Return every ``type: ignore`` in the source, with its location."""
        found = []
        for path in sorted(SRC.rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if "type: ignore" in line:
                    found.append((path.relative_to(SRC).as_posix(), lineno, line.strip()))
        return found

    def test_no_bare_type_ignore(self):
        bare = re.compile(r"type:\s*ignore(?!\[)")
        offenders = [f"{rel}:{lineno}" for rel, lineno, line in self._suppressions() if bare.search(line)]

        assert not offenders, (
            "these suppressions disable every check on their line; name the specific "
            f"error codes instead: {offenders}"
        )

    def test_no_blanket_file_level_suppression(self):
        """`# mypy: ignore-errors` would silence a whole module at once."""
        offenders = [
            path.relative_to(SRC).as_posix()
            for path in sorted(SRC.rglob("*.py"))
            if re.search(r"^#\s*mypy:\s*ignore-errors", path.read_text(encoding="utf-8"), re.M)
        ]

        assert not offenders, f"whole-file type suppression found in {offenders}"
