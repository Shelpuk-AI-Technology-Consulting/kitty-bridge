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

# L2: the subject of this file is an artifact outside `src/kitty` Python code,
# or a structural scan of source text -- two things edited separately that must
# agree. It gates pull requests exactly as before, in the `l1 or l2` job; the
# marker records which half of that expression it answers to, and keeps a
# source-text scan out of the L1 set that mutation testing will judge.
pytestmark = pytest.mark.l2

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
    ("providers/model_context_sync.py", "aiohttp session"): 1,
    ("providers/ollama_cloud.py", "aiohttp session"): 1,
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
    "providers/model_context_sync.py": "catalog-refresh session built with aiohttp_session_kwargs()",
    "providers/ollama_cloud.py": "session built with aiohttp_session_kwargs()",
    "providers/openai_subscription.py": "one builder, two sessions: proxies= and NOPROXY (KBR-161)",
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


class TestProxyApplicationSitesInventory:
    """R4 (T-E8): the asymmetry-pin inventory — site-loss and bypass-addition guards.

    **Scope (narrowed from the original R4 claim).** This class catches **source loss**
    (a site removed) and the **specific bypass-addition shape** of "consulting
    ``should_bypass`` inside a custom-transport file". It does **not** catch a new
    transport that adds a proxy-less client construction site — that is the
    ``TestEveryHttpClientIsAccountedFor`` registry sweep's responsibility and the
    per-adapter sweep in ``tests/test_wire_shape_honesty.py`` covers adapter additions.
    The two checks complement each other; both must hold.

    **Why a per-site keyword count.** Three distinct construction functions and three
    aiohttp consumers cover §5.5's five rows plus the sixth site the catalogue names.
    The count is the simplest signal a silent deletion cannot pass: removing a site
    drops the count to zero, this class fails, and the failure names the missing site.
    """

    #: Custom-transport files that **must not** reference ``should_bypass`` — adding a
    #: bypass here would silently route local traffic through a rented proxy that
    #: cannot reach it (the design's §5.5 consequence 1). ``validation.py`` is
    #: intentionally absent from this set: its ``aiohttp_session_kwargs`` use
    #: (``validation.py:152``) does consult ``should_bypass`` deliberately, as part of
    #: the pre-flight key check, and that bypass is by design.
    _BYPASS_FORBIDDEN_FILES: tuple[str, ...] = (
        "auth/openai_oauth.py",
        "providers/ollama_cloud.py",
        "providers/bedrock.py",
        "providers/openai_subscription.py",
        "providers/model_context_sync.py",
    )

    #: Per-file keyword counts for the proxy-application sites. Adding a site without
    #: updating this table is the failure mode this guard is designed to surface;
    #: removing a site (deleting the line) drops the count and the test fails. The
    #: ``_new_curl_session`` row is the **function definition** (not a call), since
    #: the function is invoked from a closed surface and a call-count scan would miss
    #: it after a regression that broke the builder without removing the calls.
    _EXPECTED_PROXY_APPLICATION_COUNTS: dict[tuple[str, str], int] = {
        ("auth/openai_oauth.py", "aiohttp_session_kwargs"): 1,
        ("providers/ollama_cloud.py", "aiohttp_session_kwargs"): 1,
        ("providers/model_context_sync.py", "aiohttp_session_kwargs"): 1,
        ("providers/openai_subscription.py", "_new_curl_session"): 1,
        ("providers/bedrock.py", "_BotoConfig_with_proxies"): 1,
    }

    @staticmethod
    def _iter_pattern(pattern: re.Pattern[str]) -> list[tuple[str, int, str]]:
        """Yield every ``(relative_path, line, text)`` matching ``pattern`` in source."""
        found: list[tuple[str, int, str]] = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    found.append((rel, lineno, line.strip()))
        return found

    def test_each_proxy_application_site_is_still_present(self) -> None:
        """AC4.1: each site still exists at its recorded location.

        Deleting one of the six sites drops its count below the expected floor and
        this test fails — naming the deleted site in the assertion message so the
        fix is local.
        """
        aiohttp_session_kwargs = re.compile(r"\baiohttp_session_kwargs\(")
        new_curl_session = re.compile(r"\bdef _new_curl_session\(")
        boto_config_with_proxies = re.compile(r"_BotoConfig\s*\(\s*proxies\s*=")

        patterns: dict[str, re.Pattern[str]] = {
            "aiohttp_session_kwargs": aiohttp_session_kwargs,
            "_new_curl_session": new_curl_session,
            "_BotoConfig_with_proxies": boto_config_with_proxies,
        }

        actual: dict[tuple[str, str], int] = {}
        for kind, pattern in patterns.items():
            for rel, _lineno, _line in self._iter_pattern(pattern):
                key = (rel, kind)
                actual[key] = actual.get(key, 0) + 1

        offenders: list[str] = []
        for (rel, kind), expected in self._EXPECTED_PROXY_APPLICATION_COUNTS.items():
            count = actual.get((rel, kind), 0)
            if count != expected:
                offenders.append(f"{rel} ({kind}): expected {expected}, found {count}")

        assert not offenders, (
            "proxy-application site inventory changed — review each for egress, "
            "then update _EXPECTED_PROXY_APPLICATION_COUNTS: " + "; ".join(offenders)
        )

    def test_no_custom_transport_file_consults_should_bypass(self) -> None:
        """R4 anti-bypass pin: a ``should_bypass`` reference inside a custom-transport file
        is the regression this guard exists to catch.

        A bypass added to ``auth/openai_oauth.py`` would route the OAuth-login leg
        through a rented proxy that cannot reach a private LAN, and the
        behavioural L3 tests in ``tests/harness/test_*_containment_slice.py`` do not
        drive the OAuth path at startup (R-3 in §5.5 consequence 3). Catching the
        defect at the structural layer is the cheapest line of defence.

        The companion ``tests/test_egress.py::TestShouldBypass`` pins the
        ``should_bypass`` function's classification; this guard pins *who calls it*,
        which is a different property.
        """
        pattern = re.compile(r"\bshould_bypass\b")
        offenders: list[str] = []
        for rel in self._BYPASS_FORBIDDEN_FILES:
            for found_rel, lineno, _text in self._iter_pattern(pattern):
                if found_rel == rel:
                    offenders.append(f"{rel}:{lineno} — a bypass was added to a custom-transport path")
                    break

        assert not offenders, (
            "these custom-transport files must not consult `should_bypass` "
            "(§5.5 consequence 1 — the proxy is unconditional here): " + "; ".join(offenders)
        )

    def test_the_scan_actually_finds_something(self) -> None:
        """A broken regex would silently pass the inventory check above."""
        aiohttp_session_kwargs = re.compile(r"\baiohttp_session_kwargs\(")
        aiohttp_matches = {
            rel
            for rel, _lineno, _line in self._iter_pattern(aiohttp_session_kwargs)
            if (rel, "aiohttp_session_kwargs") in self._EXPECTED_PROXY_APPLICATION_COUNTS
        }

        assert len(aiohttp_matches) == 3, (
            f"the aiohttp_session_kwargs scan found {len(aiohttp_matches)} sites, "
            f"expected 3 — the regex or the table is broken: {sorted(aiohttp_matches)}"
        )


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
            f"these suppressions disable every check on their line; name the specific error codes instead: {offenders}"
        )

    def test_no_blanket_file_level_suppression(self):
        """`# mypy: ignore-errors` would silence a whole module at once."""
        offenders = [
            path.relative_to(SRC).as_posix()
            for path in sorted(SRC.rglob("*.py"))
            if re.search(r"^#\s*mypy:\s*ignore-errors", path.read_text(encoding="utf-8"), re.M)
        ]

        assert not offenders, f"whole-file type suppression found in {offenders}"
