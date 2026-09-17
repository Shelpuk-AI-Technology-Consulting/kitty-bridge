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
import textwrap
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


#: The file that defines the ``BridgeServer`` class. Excluded from the
#: construction scan: the ``class BridgeServer:`` declaration is an
#: ``ast.ClassDef``, not an ``ast.Call``, so the class line itself is invisible
#: to the walker and needs no exclusion — but the file-level skip also covers
#: any ``BridgeServer(`` **call** the file might come to contain, and that
#: skip is only sound because
#: ``test_no_bridge_server_construction_in_definition_file`` asserts the file
#: holds none. A future factory or test helper inside it must surface here
#: rather than bypass the egress-guard check silently.
_BRIDGE_SERVER_DEFINITION_FILE = "bridge/server.py"


def _called_name(node: ast.Call) -> str | None:
    """Return the called function's short name, or ``None`` for other shapes.

    Args:
        node: The ``ast.Call`` node to inspect.

    Returns:
        The callee's identifier as a string — the ``Name.id`` for a bare call
        such as ``BridgeServer(...)`` or the ``Attribute.attr`` for a dotted
        call such as ``kitty.bridge.server.BridgeServer(...)``. ``None`` for
        any other shape (subscripts, calls, etc.).
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _enclosing_function(
    tree: ast.Module, lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the innermost function whose body covers ``lineno``.

    Args:
        tree: Parsed module to search.
        lineno: 1-based line number the covering function must span.

    Returns:
        The innermost (deepest-starting) ``FunctionDef`` or ``AsyncFunctionDef``
        whose span contains ``lineno``, or ``None`` when the line sits at
        module scope.
    """
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):
            best = node
    return best


def _iter_constructions_in_tree(tree: ast.AST) -> list[ast.Call]:
    """Yield every ``BridgeServer(`` ``Call`` node anywhere in ``tree``.

    Args:
        tree: Any parsed AST (module, function body, etc.).

    Returns:
        Every ``ast.Call`` whose callee is a ``Name`` or final ``Attribute``
        named ``"BridgeServer"``. The matcher is shared by the live-source
        scan and the synthetic-tree falsification probes so a regression to
        the matching logic is exercised by both.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == "BridgeServer"
    ]


def _iter_aliased_bridge_server_imports_in_tree(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Yield every aliased ``BridgeServer`` import in ``tree``.

    Args:
        tree: Any parsed AST (module, function body, etc.).

    Returns:
        ``(lineno, original_name, asname)`` for each ``import ... BridgeServer as
        <asname>`` (or ``from ... import BridgeServer as <asname>``) binding. A
        renamed import hides any construction site that uses the alias from
        the construction walker, so any aliased import is itself a guard
        failure. The matcher is shared by the live-source scan and the
        synthetic-tree falsification probe so a regression to the matching
        logic is exercised by both.
    """
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if alias.name.endswith("BridgeServer") and alias.asname:
                found.append((node.lineno, alias.name, alias.asname))
    return found


def _iter_bridge_constructions() -> list[tuple[str, int]]:
    """Find every ``BridgeServer(`` construction under ``src/kitty``.

    Returns:
        ``(relative_path, line_number)`` for each construction, excluding the
        class-definition file.
    """
    found: list[tuple[str, int]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == _BRIDGE_SERVER_DEFINITION_FILE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _iter_constructions_in_tree(tree):
            found.append((rel, node.lineno))
    return found


#: Nested scopes whose bodies do NOT execute when the enclosing function runs —
#: a guard call lexically inside any of these does not dominate a construction
#: in the enclosing function. ``GeneratorExp`` is included because generator
#: bodies are lazy (their ``elt`` only runs on iteration, not on the line where
#: the ``( ... for ... )`` expression appears); ``ListComp`` / ``SetComp`` /
#: ``DictComp`` are deliberately excluded because their elements execute
#: eagerly at the line where the comprehension appears.
_DEFERRED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.GeneratorExp,
)


def _calls_in_own_scope(scope: ast.AST):
    """Yield ``Call`` nodes in ``scope``'s own body, skipping deferred subtrees.

    Args:
        scope: An ``ast`` node whose direct body is executed when the scope
            runs (typically a function).

    Yields:
        Every ``ast.Call`` directly in the scope, plus any in eagerly-executed
        subexpressions (``ListComp``/``SetComp``/``DictComp`` elts, conditional
        expressions, etc.). Calls in deferred subtrees — nested ``def`` /
        ``class`` / ``lambda`` bodies, and generator-expression elts — are
        skipped entirely; those bodies do not run when the enclosing scope
        runs, so they cannot dominate anything here.
    """
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call):
            yield node
        if isinstance(node, _DEFERRED_SCOPES):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _is_dominated(tree: ast.Module, target_lineno: int) -> bool:
    """Report whether a ``BridgeServer`` construction is egress-guarded.

    A construction is dominated when its innermost enclosing function holds a
    call to ``egress_block_reason`` on an earlier line, in the function's own
    body (not in a nested helper scope, whose body is deferred and would not
    have run by the time the construction executes unless explicitly invoked).
    A construction at module scope has no enclosing function and is never
    dominated.

    Args:
        tree: Parsed module containing the construction.
        target_lineno: 1-based line number of the ``BridgeServer(`` call.

    Returns:
        ``True`` when a preceding ``egress_block_reason(`` call exists in the
        same function scope, ``False`` otherwise.
    """
    func = _enclosing_function(tree, target_lineno)
    if func is None:
        return False
    for node in _calls_in_own_scope(func):
        if (
            isinstance(node, ast.Call)
            and _called_name(node) == "egress_block_reason"
            and node.lineno < target_lineno
        ):
            return True
    return False


class TestEveryStartPathIsGuarded:
    """R10 structurally: a new way to start a bridge must not skip the check.

    The first version of this feature wired the fail-closed guard into the agent
    launcher only, leaving foreground `kitty bridge` and the background runner
    able to start with a provider that cannot honour the proxy. The first
    structural version of this test was file-granular — a file holding a
    `BridgeServer(` call needed only to contain an `egress_block_reason(` call
    anywhere. `cli/main.py` already holds two start paths, so a third added
    there without a guard call would have passed unguarded. Domination is now
    checked at AST level: the guard call must precede the construction in the
    same function scope.
    """

    def test_every_construction_is_dominated_by_a_guard_call(self):
        """Every `BridgeServer(` construction must be preceded by `egress_block_reason(`."""
        offenders: list[str] = []
        for rel, lineno in _iter_bridge_constructions():
            tree = ast.parse((SRC / rel).read_text(encoding="utf-8"), filename=str(SRC / rel))
            if not _is_dominated(tree, lineno):
                func = _enclosing_function(tree, lineno)
                where = func.name if func is not None else "<module scope>"
                offenders.append(f"{rel}:{lineno} in {where}()")

        assert not offenders, (
            "these BridgeServer constructions are not dominated by an egress_block_reason call "
            "in the same function scope, so a provider that cannot be proxied would leak from "
            f"them: {offenders}"
        )

    def test_no_bridge_server_construction_in_definition_file(self):
        """The class-definition file must hold no ``BridgeServer(`` calls.

        The file-level skip in ``_iter_bridge_constructions`` excludes
        ``bridge/server.py`` wholesale, so a ``BridgeServer(`` **call**
        introduced there — a factory, a test helper, anything of that shape —
        would be silently invisible to the domination guard. The class line
        itself is not the reason (a ``class BridgeServer:`` declaration is an
        ``ast.ClassDef``, not an ``ast.Call``, and the matcher never sees it);
        this test is what makes the file-level skip sound. If it fails, the
        new call is either a new start path that belongs outside this file or
        a helper that needs an explicit decision here.
        """
        definition_file = SRC / _BRIDGE_SERVER_DEFINITION_FILE
        tree = ast.parse(definition_file.read_text(encoding="utf-8"), filename=str(definition_file))
        calls = _iter_constructions_in_tree(tree)

        assert calls == [], (
            f"a `BridgeServer(...)` call inside {_BRIDGE_SERVER_DEFINITION_FILE} would be "
            "silently skipped by the file-level exclusion in `_iter_bridge_constructions`; "
            f"either move it or handle it explicitly. Found at lines "
            f"{[call.lineno for call in calls]}"
        )

    def test_the_scan_finds_the_known_start_paths(self):
        """Guards against a broken AST walk silently matching nothing."""
        sites = _iter_bridge_constructions()
        files = {rel for rel, _lineno in sites}

        assert len(sites) >= 5, (
            f"the scan found {len(sites)} BridgeServer constructions; at least five start "
            "paths are known to exist, so the AST walk is likely broken"
        )
        assert files == {"cli/launcher.py", "cli/main.py", "bridge_runner.py"}, (
            f"the set of files constructing BridgeServer changed: {sorted(files)}. Review each "
            "for egress domination before updating this assertion."
        )

    def test_undominated_construction_is_caught(self):
        """Falsification control: the walker must reject an unguarded construction.

        Without this, a broken `_is_dominated` that always returns True would
        pass every other test in this class while proving nothing. The five
        shapes exercise the pairwise combinations of {construction, guard}
        across {own scope, nested scope}, plus generator laziness:

        - **sibling-undominated** — guard absent (the degenerate case).
        - **sibling-dominated** — guard precedes construction in the same scope.
        - **outer-guard, inner-construction** — guards are not transitive into
          inner scopes; this closes the "innermost vs outermost enclosing
          function" blind spot. A walker that picked the outermost covering
          function would accept the inner construction here.
        - **outer-construction, inner-guard (never invoked)** — guards confined
          to nested scopes do not dominate a construction in the enclosing
          scope, because the nested helper is deferred and structurally
          indistinguishable from an unguarded code path. A walker that used
          plain ``ast.walk(func)`` over the entire enclosing function would
          wrongly accept this shape.
        - **outer-construction, guard inside a generator expression** —
          generator bodies are lazy (the ``elt`` runs only on iteration), so
          this is the same deferral shape as the nested-def case with the
          guard written inline rather than in a helper. A walker that omitted
          ``ast.GeneratorExp`` from ``_DEFERRED_SCOPES`` would wrongly accept
          this shape.
        """
        source = textwrap.dedent(
            """\
            from kitty.bridge.server import BridgeServer
            from kitty.egress_guard import egress_block_reason

            def unguarded():
                return BridgeServer()

            def guarded():
                egress_block_reason(None, None, None)
                return BridgeServer()

            def outer_with_inner_construction():
                egress_block_reason(None, None, None)
                def inner():
                    return BridgeServer()
                return inner

            def guard_confined_to_nested_def():
                def _helper():
                    egress_block_reason(None, None, None)
                return BridgeServer()

            def guard_inside_generator():
                # Generator elt is lazy; the guard does not execute until iteration.
                gen = (egress_block_reason(None, None, None) for _ in range(1))
                next(gen)
                return BridgeServer()
            """
        )
        tree = ast.parse(source)
        constructions = sorted(
            _iter_constructions_in_tree(tree), key=lambda n: n.lineno
        )
        assert len(constructions) == 5, "fixture must hold exactly five constructions"

        assert not _is_dominated(tree, constructions[0].lineno), (
            "sibling-undominated: a construction with no preceding guard must be reported"
        )
        assert _is_dominated(tree, constructions[1].lineno), (
            "sibling-dominated: a guard in the same scope must be accepted"
        )
        assert not _is_dominated(tree, constructions[2].lineno), (
            "outer-guard/inner-construction: a guard in the outer scope must not dominate "
            "a construction in an inner function"
        )
        assert not _is_dominated(tree, constructions[3].lineno), (
            "outer-construction/inner-guard: a guard confined to a nested helper scope "
            "must not dominate a construction in the enclosing function"
        )
        assert not _is_dominated(tree, constructions[4].lineno), (
            "outer-construction/inner-generator-guard: a guard inside a generator expression "
            "is lazy and must not dominate a sibling-level construction"
        )

    def test_dotted_construction_spellings_are_matched(self):
        """The ``Attribute`` branch is load-bearing: ``mod.BridgeServer(...)`` must be found.

        Without the ``Attribute`` branch in ``_called_name``, a construction
        written as ``kitty.bridge.server.BridgeServer(...)`` would be invisible
        to the construction scan and the guard would silently miss it. The
        scan against ``SRC`` is bound to the source tree, so this helper probe
        exercises the production matcher (``_iter_constructions_in_tree``) on
        a synthetic dotted call — a regression in the shared helper fails
        this test.
        """
        source = textwrap.dedent(
            """\
            import kitty.bridge.server

            def dotted_call():
                return kitty.bridge.server.BridgeServer()
            """
        )
        tree = ast.parse(source)
        matched = _iter_constructions_in_tree(tree)
        assert len(matched) == 1, (
            "a dotted `mod.BridgeServer(...)` call must be matched by the walker; the "
            "Attribute branch of `_called_name` may have regressed"
        )

    def test_no_aliased_bridge_server_imports(self):
        """An aliased import would hide a construction from the Name/Attribute matcher.

        The construction walker matches ``BridgeServer(...)`` as a ``Name`` or
        the final ``Attribute`` of ``mod.BridgeServer(...)``. It cannot see a
        construction under a renamed alias — ``from kitty.bridge.server
        import BridgeServer as BS; BS(...)`` — without resolving imports. No
        production file aliases it today; if any file starts to, the safety
        net is to fail this test and force a deliberate decision (extend the
        walker or rename the import back).
        """
        offenders: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for lineno, original, asname in _iter_aliased_bridge_server_imports_in_tree(tree):
                offenders.append(f"{rel}:{lineno}: `{original} as {asname}`")
        assert not offenders, (
            "BridgeServer is imported under an alias, so a construction call would be "
            f"invisible to the AST walker: {offenders}"
        )

    def test_aliased_bridge_server_import_is_detected(self):
        """Falsification control: a synthetic aliased import is flagged.

        The live tree has zero aliased ``BridgeServer`` imports, so the matcher
        is unproven by data — the test above is structurally incapable of
        failing on a regression like ``endswith("BS")``. This probe parses a
        small aliased import and exercises the **production** matcher
        (``_iter_aliased_bridge_server_imports_in_tree``) on it, so a
        regression in the shared helper fails this test rather than only its
        own copy of the loop.
        """
        source = textwrap.dedent(
            """\
            from kitty.bridge.server import BridgeServer as BS

            def aliased_call():
                return BS()
            """
        )
        tree = ast.parse(source)
        offenders = _iter_aliased_bridge_server_imports_in_tree(tree)
        formatted = [f"line {lineno}: `{original} as {asname}`" for lineno, original, asname in offenders]

        assert formatted == ["line 1: `BridgeServer as BS`"], (
            "the alias matcher must surface `from kitty.bridge.server import "
            f"BridgeServer as BS` as an offender; got {formatted}"
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
