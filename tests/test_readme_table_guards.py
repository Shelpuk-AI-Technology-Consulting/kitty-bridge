"""Contract guards — README tables agree with the code (KBR-76, plan task T-G1).

For a CLI tool the README *is* the interface specification: bridge-mode users
configure their IDEs and custom scripts against the endpoint table
(README §"Bridge Mode"), debug their routing against the ``X-Kitty-*`` header
table (§"Knowing which model actually served you"), enable logs with the
flag table (§"Logging"), and set proxy/summary paths via ``KITTY_*`` env
vars. ``TEST_SUITE.md`` §6.2.3 records four README tables this suite must
keep in agreement with the code, in the "Register and docs ⇄ code" row
that KBR-2's design work seeded as the broader category this file
delivers.

The four guards one at a time:

* **Endpoint table.** README "Available endpoints" table ↔
  ``BridgeServer._register_routes`` (bridge-mode branch only).
* **Attribution headers.** README ``X-Kitty-*`` table ↔
  ``BridgeServer._attribution_headers``, with a negative arm that no
  other file under ``src/kitty/`` may contain an ``X-Kitty-*`` literal
  (providers must never mint or forward one).
* **Env-var register.** Every ``KITTY_*`` env var the README names
  has a corresponding literal string constant somewhere under
  ``src/kitty/`` (forward direction only — the README is silent on
  internal/operational keys by design).
* **Logging-flag table.** README "Logging" flag table ↔ the four flags
  in ``kitty.cli.main._build_parser`` plus the two default paths the
  bridge uses: the ``BridgeServer._DEBUG_LOG_PATH`` module constant for
  debug, and the per-instance ``_usage_log_path`` default (derived from
  ``_DEBUG_LOG_DIR``) for usage.

**The one exemption.** KBR-9 documents the endpoint table's drift: the
README names ``POST /v1/gemini/generateContent`` (no such route) and
omits the two real Gemini routes plus ``GET /v1/models``. The endpoint
guard is wrapped in ``ratchet("t-g1-endpoint-table")`` so it can land
while the defect is open; when KBR-9's README correction lands, the
row fails the suite via ``UnexpectedExemptionPass`` and must be
deleted — the T-W7 mechanism for the day debt is paid. The other three
arms have no known drift and gate normally; if any guard surfaces drift
in the future, the smallest possible follow-up ticket is filed and the
corresponding exemption row added, rather than fix the README in this
PR.

**Verdict mechanism.** Every primary assertion uses
``assert not problems, "..." + "\n  ".join(problems)`` — matching
``tests/test_opencode_endpoint_table.py:151``. Only ``AssertionError``
is amnestied by ``ratchet(...)`` per
``tests/exemptions.py::outcome_for``; ``pytest.fail(...)`` raises
``Failed``, which ``ratchet`` lets propagate, silently defeating the
exemption. ``pytest.fail`` is fine outside a ``ratchet`` block; this
guard happens to live inside one.

**Scanning style.** Mirrors ``tests/test_egress_coverage.py``: a
source-text scan or a parser introspection over ``src/kitty``, not a
behavioural probe of a running server. AST is used where the
source-text shape is more reliable than a regex (the ``_register_routes``
function body mixes bridge-mode and launch-mode blocks at the same
indentation; an AST walk that matches the ``self._adapter is None``
test narrows the scan precisely). Regex is used where the contract is
about a literal that may sit inside a comment or a docstring
(``X-Kitty-*`` leaks; the AST walk would miss both).

Every guard asserts its own scan found known positives, so none can
rot into a vacuous "no problems" pass. Every pure checker has a
falsification case — a deliberate defect fed in under controlled
inputs — per the implementation plan's §1.4 harness rule.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

import pytest
from exemptions import ratchet

from kitty.bridge.server import _DEBUG_LOG_PATH, BridgeServer
from kitty.cli.main import _build_parser
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol

# L2 — the subject of every guard here is two artifacts edited separately,
# a markdown file and Python code, and the guard holds them in agreement.
# The pure checkers live at module scope so the falsification tests can
# hand them deliberate defects without a network or filesystem.
pytestmark = pytest.mark.l2

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_SRC = _ROOT / "src" / "kitty"
_SERVER_PY = _SRC / "bridge" / "server.py"


# ── README helpers ──────────────────────────────────────────────────────────


def readme_text() -> str:
    """Return the README's text content.

    Returns:
        The file's UTF-8 text, suitable for ``re`` matching against table
        rows or backticked env-var names.

    The README is treated as a text fixture; it is not checked for line
    endings on read because every guard is line-content-anchored and the
    re-scan below normalises neither end nor whitespace.
    """
    return _README.read_text(encoding="utf-8")


def readme_table_rows(text: str, anchor: str) -> list[list[str]]:
    """Return the cells of the README markdown table that contains *anchor*.

    Args:
        text: The full README text.
        anchor: A substring that appears in exactly one table — its header
            row or a data row.

    Returns:
        The list of row cells, each a list of stripped cell strings. The
        header row is included as the first entry; rows containing the
        alignment-separator (``---``) are dropped.

    Raises:
        AssertionError: When zero or more than one table contains the
            anchor, so a future refactor that splits or duplicates a table
            surfaces the ambiguity rather than silently matching the wrong
            one.
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in text.splitlines():
        # A line is in a table iff it starts with a non-empty cell. ``|``
        # at position 0 is the markdown prose split; `` \|`` is prose too
        # (escape). Lines starting with `` > |`` inside a blockquote do
        # start a table row — the README has none, so the simpler rule
        # holds.
        if raw_line.startswith("|"):
            current.append(_split_cells(raw_line))
            continue

        # First non-table line: close any open table.
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)

    matches = [
        table
        for table in tables
        if any(anchor in cell for row in table for cell in row)
    ]

    assert len(matches) == 1, (
        f"README table anchor {anchor!r} matches {len(matches)} tables; "
        "expected exactly one. Found tables of sizes "
        f"{[len(t) for t in tables]}; lines containing the anchor: "
        f"{[i for i, line in enumerate(text.splitlines(), start=1) if anchor in line]}"
    )

    rows = matches[0]
    return [row for row in rows if not all(set(cell) <= {"-", " "} for cell in row)]


def _split_cells(line: str) -> list[str]:
    """Split a markdown table line into its cells.

    Args:
        line: A line beginning and ending with ``|`` (a markdown table row).

    Returns:
        The trimmed cell strings, no outer padding ``|``.
    """
    parts = line.strip().split("|")
    # split produces one leading and one trailing empty string for a line
    # that begins and ends with ``|``; drop them.
    return [cell.strip() for cell in parts[1:-1]]


def _unwrap(cell: str) -> str:
    """Strip a single layer of surrounding backticks from a markdown cell.

    The README flags its code literals with backticks — `` `--logging` `` —
    so a side-by-side equivalence check between cell values and the
    parser's option strings needs the backticks removed first. Cells
    without backticks are returned unchanged.
    """
    if cell.startswith("`") and cell.endswith("`") and len(cell) >= 2:
        return cell[1:-1].strip()
    return cell


# ── Guard 1 — Endpoint table ────────────────────────────────────────────────


def endpoint_pairs(rows: list[list[str]]) -> set[tuple[str, str]]:
    """Extract ``(METHOD, path)`` pairs from the endpoint table rows.

    Args:
        rows: The rows of the README endpoint table; the header row is
            included and recognised as the row whose first cell does NOT
            contain ``GET /path`` in backticks.

    Returns:
        The set of ``(METHOD, path)`` pairs as printed in the README —
        code-side paths are matched to the README strings character by
        character (the AIO variable syntax in ``/v1beta/models/{model:.*}``
        is part of the contract a user sees when configuring a tool).

    Raises:
        AssertionError: When no data rows carry a ``(GET|POST) /...`` cell,
            so a future README restructure that removes the table's
            recognisable shape surfaces immediately rather than silently
            yielding an empty set.
    """
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        if not row or "`" not in row[0]:
            continue
        first = row[0]
        match = re.fullmatch(r"`(GET|POST|DELETE|PUT|PATCH) (\S+)`", first)
        if match is None:
            continue
        pairs.add((match.group(1), match.group(2)))

    assert pairs, (
        "the README endpoint table yielded no (METHOD, path) pairs — "
        "the guard's input is empty; the anchor or parser has drifted"
    )

    return pairs


def bridge_routes_from_source(source: str) -> set[tuple[str, str]]:
    """Return the routes ``_register_routes`` registers in bridge mode.

    Args:
        source: The full text of ``src/kitty/bridge/server.py``.

    Returns:
        The set of ``(METHOD, path)`` pairs added inside the
        ``if self._adapter is None:`` branch of ``_register_routes``,
        where METHOD is upper-cased (``"GET"``, ``"POST"``, …) and path
        is the literal first argument to ``app.router.add_<verb>(...)``.

    Raises:
        AssertionError: When the function ``_register_routes`` cannot be
        located, when its bridge-mode ``if`` branch cannot be located, or
        when the branch contains no ``app.router.add_*(...)`` calls. Each
        of those failures is the kind a future refactor would introduce;
        a silent empty set would make every endpoint-table assertion
        vacuously green.
    """
    tree = ast.parse(source)
    function = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_register_routes"),
        None,
    )
    assert function is not None, (
        "_register_routes is not defined in src/kitty/bridge/server.py; "
        "the AST scan cannot find its subject — the route registry has moved"
    )

    bridge_block = next(
        (stmt for stmt in function.body if isinstance(stmt, ast.If) and _is_adapter_none(stmt.test)),
        None,
    )
    assert bridge_block is not None, (
        "the `if self._adapter is None:` branch in _register_routes is "
        "missing; the AST scan has nothing to look at — a refactor "
        "relocated or renamed the bridge-mode route registration"
    )

    known_methods = {
        "add_get": "GET",
        "add_post": "POST",
        "add_put": "PUT",
        "add_delete": "DELETE",
        "add_patch": "PATCH",
        "add_head": "HEAD",
        "add_options": "OPTIONS",
    }

    # The README's table documents what BRIDGE mode serves. Two regions of
    # ``_register_routes`` are its subject: the statements BEFORE the
    # ``self._adapter is None`` branch (unconditional registrations such as
    # ``/healthz`` and ``/stats``) and the statements inside the branch
    # itself. The branch's ``orelse`` and everything after it are launch
    # mode — a different contract, deliberately out of scope.

    def _collect(from_statements: Iterable[ast.stmt]) -> None:
        for statement in from_statements:
            # Stop at the branch's ``return`` — anything past it is
            # launch-mode registration and is not the README's subject.
            if isinstance(statement, ast.Return):
                break
            for call in ast.walk(statement):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not isinstance(func, ast.Attribute) or func.attr not in known_methods:
                    continue
                if not call.args or not isinstance(call.args[0], ast.Constant):
                    continue
                literal = call.args[0].value
                if not isinstance(literal, str):
                    continue
                routes.add((known_methods[func.attr], literal))

    routes: set[tuple[str, str]] = set()
    bridge_block_index = function.body.index(bridge_block)
    _collect(function.body[:bridge_block_index])
    _collect(bridge_block.body)

    assert routes, (
        "the bridge-mode block of _register_routes registered no routes; "
        "either the block is empty or every add_*(...) call moved out of "
        "the bridge branch — the guard has no subject"
    )

    return routes


def _is_adapter_none(test: ast.expr) -> bool:
    """Return True iff *test* is the AST shape ``self._adapter is None``."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Attribute)
        and test.left.attr == "_adapter"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def check_endpoint_agreement(
    readme: set[tuple[str, str]], code: set[tuple[str, str]]
) -> list[str]:
    """Return one human-readable problem per disagreement between the sets.

    Args:
        readme: The endpoint pairs the README documents.
        code: The route pairs the bridge-mode branch registers.

    Returns:
        Every disagreement, each on its own line. Empty when the two agree.
        A vacuous set (empty on either side) is reported as one line —
        "no disagreements found" is exactly what a guard that has stopped
        looking also reports.
    """
    if not readme:
        return ["the README endpoint table parsed to zero rows; the guard's input is empty"]
    if not code:
        return ["the AST scan of _register_routes found zero bridge-mode routes; the guard's input is empty"]

    problems: list[str] = []
    for method, path in sorted(readme - code):
        problems.append(f"README documents `{method} {path}` but no bridge-mode route exists for it")
    for method, path in sorted(code - readme):
        problems.append(f"bridge mode registers `{method} {path}` but the README endpoint table omits it")
    return problems


# ── Guard 2 — Attribution headers ──────────────────────────────────────────


def attribution_header_names(source: str) -> set[str]:
    """Return the ``X-Kitty-*`` constants used inside ``_attribution_headers``.

    Args:
        source: The full text of ``src/kitty/bridge/server.py``.

    Returns:
        The set of string constants whose value starts with ``X-Kitty-``
        (the header-name prefix as a docstring promises, but the name is
        minted in code as a dict key, not as a substring of prose). The
        set is exact — a header key added without also updating both
        ``_attribution_headers`` and the README is detected because the
        code side moves.

    Raises:
        AssertionError: When ``_attribution_headers`` cannot be located
        or yields no ``X-Kitty-*`` string constants at all — either is the
        kind of refactor that should surface immediately, not silently.
    """
    tree = ast.parse(source)
    function = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_attribution_headers"),
        None,
    )
    assert function is not None, (
        "_attribution_headers is not defined in src/kitty/bridge/server.py; the AST scan cannot find its subject"
    )

    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        if node.value.startswith("X-Kitty-"):
            names.add(node.value)

    assert names, (
        "_attribution_headers produced no X-Kitty-* string constants; "
        "either the function is empty or the literal keys moved — the "
        "guard has no subject"
    )

    return names


def kitty_header_names_in_text(text: str) -> set[str]:
    """Return the case-normalised set of ``X-Kitty-*`` names in *text*.

    Args:
        text: Any source text (a single file, a synthetic fragment, a
            combined blob of a whole tree's worth of files).

    Returns:
        The distinct header names of the form ``X-Kitty-{suffix}``
        extracted from *text*, case-normalised to title case so that
        ``x-kitty-backend`` and ``X-Kitty-Backend`` are counted as one.
        The scan operates on raw text rather than AST string constants
        on purpose — a comment mentioning ``X-Kitty-Foo`` is exactly the
        regressive shape a refactor would take, and an AST walk would
        miss it.
    """
    return {match.group(0).title() for match in re.finditer(r"(?i)\bx-kitty-[a-z][a-z0-9-]*\b", text)}


def kitty_header_leaks_outside_server(source_per_path: dict[str, str]) -> dict[str, list[str]]:
    """Return the X-Kitty leaks per non-server source file.

    Args:
        source_per_path: Mapping of relative source-file path to its
            source text. The server file should be omitted.

    Returns:
        Each non-server file that contains one or more ``X-Kitty-*``
        literals, mapped to the sorted list of names found in that file.
        Empty when no leaks exist.
    """
    return {
        path: sorted(kitty_header_names_in_text(text))
        for path, text in sorted(source_per_path.items())
        if kitty_header_names_in_text(text)
    }


# ── Guard 3 — Env-var register ─────────────────────────────────────────────


def kitty_env_vars_in_readme(text: str) -> set[str]:
    """Return the ``KITTY_*`` env-var names appearing anywhere in *text*.

    Args:
        text: The README content (or any prose to be scanned).

    Returns:
        Every distinct name of the form ``KITTY_<UPPER>`` (uppercase
        letters and digits, beginning with a letter after the prefix).
        Words inside a bash block (``export KITTY_FOO=...``) and words
        inside backticks are both matched — the env-var reference is the
        same in either shape.

    Self-guard: an empty set asserts the README carries no ``KITTY_*``
    env-var name, which would mean the guard has nothing to verify and
    is vacuous; the assertion fails so that state surfaces.
    """
    names = set(re.findall(r"\bKITTY_[A-Z][A-Z0-9_]*\b", text))
    assert names, (
        "the README yields zero KITTY_* env-var names; either the anchor "
        "or the README's structure has drifted — the guard has no subject"
    )
    return names


def string_constants_in_src() -> set[str]:
    """Return the set of distinct string constants under ``src/kitty/``.

    Returns:
        Every ``ast.Constant`` whose ``value`` is a ``str`` from every
        ``*.py`` file under ``src/kitty/``. Exact-match against this
        set is the env-var-register oracle: ``"KITTY_SESSION_SUMMARY"``
        matches it, but the docstring sentence "The ``KITTY_SESSION_SUMMARY``
        env var writes the summary…" does not (the constant's value is
        a longer string, not the bare name). That is the discipline that
        keeps the guard honest on a future "rename the env var" refactor.
    """
    constants: set[str] = set()
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                constants.add(node.value)

    assert constants, (
        "the AST scan of src/kitty yielded zero string constants; the "
        "environment has changed in a way the scan cannot survive — the "
        "guard is blind"
    )

    return constants


def check_env_register(documented: Iterable[str], constants: set[str]) -> list[str]:
    """Return one problem per documented name with no constant match.

    Args:
        documented: The ``KITTY_*`` env-var names the README names.
        constants: The AST-derived set of string constants under
            ``src/kitty/`` (typically from
            :func:`string_constants_in_src`).

    Returns:
        One ``"documented but not read"`` problem per undocumented
        constant — the forward direction only. Empty when every
        documented name has at least one exact-match constant.
    """
    return [
        f"README documents `{name}` but no exact-match constant exists in src/kitty/"
        for name in sorted(set(documented))
        if name not in constants
    ]


# ── Guard 4 — Logging-flag table ────────────────────────────────────────────


_KNOWN_LOGGING_FLAGS = ("--logging", "--debug", "--log-file", "--debug-file")


def readme_flag_rows(rows: list[list[str]]) -> list[tuple[str, str, str]]:
    """Return ``(flag, default_path, custom_path_flag)`` from the flag table.

    Args:
        rows: The cells of the README logging-flag table.

    Returns:
        One triple per data row:
            ``(flag, default_path, custom_path_flag)``. Example::

                ("--logging", "~/.cache/kitty/usage.log", "--log-file PATH")

        The ``custom_path_flag`` column carries metavar text
        (``--log-file PATH``), used as a parse-arg hint for the accept-test.
    """
    triples: list[tuple[str, str, str]] = []
    for row in rows:
        if len(row) < 4:
            continue
        flag = _unwrap(row[0])
        if flag not in _KNOWN_LOGGING_FLAGS:
            continue
        default_path = _unwrap(row[2])
        custom_path = _unwrap(row[3])
        triples.append((flag, default_path, custom_path))

    assert triples, (
        "the README logging-flag table yielded no recognisable flag rows; "
        "the table shape has drifted — the guard has no subject"
    )

    return triples


def parser_accepts_flag(parser, flag: str, argv: list[str]) -> tuple[bool, set[str]]:
    """Parse *argv* and report whether *flag* bound and what stayed unknown.

    Args:
        parser: The argparse ``ArgumentParser`` to probe.
        flag: The option string whose acceptance is under test, e.g.
            ``"--log-file"``.
        argv: The argv to parse, including the subcommand
            (``"bridge"``) at the end.

    Returns:
        ``(bound, unknown)``: ``bound`` is True when the flag reached its
        argparse ``dest`` with a truthy value or a ``Path`` (the two
        shapes the logging flags take: ``store_true`` for ``--logging``
        and ``--debug``, ``Path`` for ``--log-file`` and
        ``--debug-file``); ``unknown`` is the set of unknown ``--``-led
        tokens ``parse_known_args`` left over.

    The ``dest`` is derived from the flag using argparse's documented
    default — strip the leading dashes, replace interior dashes with
    underscores — matching how ``tests/test_cli_main.py`` reads parsed
    namespaces. No ``parser._actions`` introspection: the acceptance
    contract is what ``parse_known_args`` reports, not the parser's
    private action list.
    """
    args, unknown = parser.parse_known_args(argv)
    dest = flag.lstrip("-").replace("-", "_")
    value = getattr(args, dest, None)
    bound = value is True or isinstance(value, Path)
    return bound, {token for token in unknown if token.startswith("--")}


def parser_rejects(parser, argv: list[str]) -> bool:
    """Return True iff ``parser.parse_args(argv)`` raises ``SystemExit``.

    Args:
        parser: The argparse ``ArgumentParser`` to probe.
        argv: The argv to attempt to parse.

    Returns:
        ``True`` on rejection (``SystemExit`` raised), ``False`` on
        acceptance. The standard argparse 3.x behaviour is
        ``SystemExit(2)`` on ``unrecognized arguments``.
    """
    try:
        parser.parse_args(argv)
    except SystemExit:
        return True
    return False


def bridge_default_usage_log_path() -> Path:
    """Return the ``BridgeServer``-default usage log path.

    Returns:
        ``server._usage_log_path`` of a freshly constructed
        ``BridgeServer`` with logging enabled and no custom path.
        Mirrors ``tests/test_cli_log_file.py::TestCustomUsageLogPath``.

    The stub launcher and provider are minimal — they satisfy the
    abstract base classes without doing any work. The constructor
    performs no I/O without ``logging_enabled=False`` and a
    `_log_usage` call; we neither log nor serve, so instantiation is
    cheap and side-effect-free.
    """
    server = BridgeServer(StubLauncher(), StubProvider(), "test-key", logging_enabled=True)
    return server._usage_log_path


def bridge_default_debug_log_path() -> Path:
    """Return the project-wide default debug-log path constant.

    Returns:
        ``BridgeServer._DEBUG_LOG_PATH`` — a module-level constant that
        is also what ``_setup_debug_logging`` returns when ``debug=True``
        with no custom path. Importing it directly is the standard
        pattern (``tests/test_debug_log_path.py`` does exactly this).
    """
    return _DEBUG_LOG_PATH


class StubLauncher(LauncherAdapter):
    """Minimal launcher stub for ``BridgeServer`` instantiation in tests."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(
        self,
        profile: Profile,
        bridge_port: int,
        resolved_key: str,
        *,
        context_tokens: int | None = None,
    ) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class StubProvider(ProviderAdapter):
    """Minimal provider stub for ``BridgeServer`` instantiation in tests."""

    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages}

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Tests ──────────────────────────────────────────────────────────────────


class TestEndpointTable:
    """Guard 1 — README's bridge-mode endpoint table matches ``_register_routes``."""

    def test_the_scans_find_their_known_subjects(self):
        """Both the README parser and the AST scan find a non-empty, known-positive set.

        Gates normally (no exemption) so a future refactor that empties
        either input surfaces as a regular test failure rather than a
        vacuous green.
        """
        readme = endpoint_pairs(readme_table_rows(readme_text(), "Protocol"))
        code = bridge_routes_from_source(_SERVER_PY.read_text(encoding="utf-8"))

        # README known positives: at least these five routes are documented today.
        assert ("GET", "/healthz") in readme
        assert ("GET", "/stats") in readme
        assert ("POST", "/v1/chat/completions") in readme
        assert ("POST", "/v1/messages") in readme
        assert ("POST", "/v1/responses") in readme

        # Code known positives: at least these six routes are registered in bridge mode.
        assert ("GET", "/healthz") in code
        assert ("GET", "/stats") in code
        assert ("POST", "/v1/chat/completions") in code
        assert ("POST", "/v1/messages") in code
        assert ("POST", "/v1/responses") in code
        assert ("GET", "/v1/models") in code
        assert len(code) >= 6

    def test_the_readme_endpoint_table_matches_the_registered_routes(self):
        """The README's documented endpoints are exactly the registered ones (KBR-9 drift exempted)."""
        readme = endpoint_pairs(readme_table_rows(readme_text(), "Protocol"))
        code = bridge_routes_from_source(_SERVER_PY.read_text(encoding="utf-8"))

        problems = check_endpoint_agreement(readme, code)

        # The exempt assertion: when KBR-9's README correction lands, this
        # row fails the suite via UnexpectedExemptionPass and must be deleted.
        with ratchet("t-g1-endpoint-table"):
            assert not problems, "README endpoint table disagrees with bridge mode:\n  " + "\n  ".join(problems)

    @pytest.mark.parametrize(
        "readme_set, code_set, expected_substring",
        [
            (set(), {("GET", "/healthz")}, "zero rows"),
            ({("GET", "/healthz")}, set(), "zero bridge-mode routes"),
            ({("GET", "/healthz")}, {("GET", "/stats")}, "no bridge-mode route exists"),
            ({("GET", "/healthz")}, {("GET", "/healthz"), ("POST", "/v1/x")}, "README endpoint table omits it"),
        ],
    )
    def test_check_endpoint_agreement_reports_a_deliberate_defect(self, readme_set, code_set, expected_substring):
        """The pure checker rejects every shape of disagreement — it cannot pass by having nothing to check."""
        problems = check_endpoint_agreement(readme_set, code_set)

        assert problems, "check_endpoint_agreement reported no disagreement on a deliberate defect"
        assert any(expected_substring in problem for problem in problems), (
            f"expected a problem mentioning {expected_substring!r}; got {problems!r}"
        )

    def test_check_endpoint_agreement_on_handcrafted_agreement_is_empty(self):
        """The pure checker returns an empty list when the two sets are exactly equal — its green baseline."""
        readme = {("GET", "/healthz"), ("POST", "/v1/messages")}
        code = set(readme)

        assert check_endpoint_agreement(readme, code) == []


class TestAttributionHeaders:
    """Guard 2 — README's ``X-Kitty-*`` table matches the bridge's minting site; no leak elsewhere."""

    def test_attribution_header_names_yields_the_documented_three(self):
        """The minting site defines exactly the three documented header names — the known-positive for the AST scan."""
        names = attribution_header_names(_SERVER_PY.read_text(encoding="utf-8"))

        assert "X-Kitty-Backend" in names
        assert "X-Kitty-Tier" in names
        assert "X-Kitty-Model" in names

    def test_readme_attribution_headers_match_the_code(self):
        """The README attribution-header table lists exactly the headers the bridge mints."""
        names = attribution_header_names(_SERVER_PY.read_text(encoding="utf-8"))
        readme_names = set()
        for row in readme_table_rows(readme_text(), "X-Kitty-Backend"):
            for cell in row:
                match = re.fullmatch(r"`(X-Kitty-[A-Za-z][A-Za-z0-9-]*)`", cell)
                if match is not None:
                    readme_names.add(match.group(1))

        assert readme_names, "the README attribution-header table yielded no X-Kitty-* names; the guard has no subject"
        assert readme_names == names, (
            f"README attribution-header table names {sorted(readme_names)}; "
            f"the bridge mints {sorted(names)}; the two sets must match"
        )

    def test_no_x_kitty_lives_outside_bridge_server(self):
        """Providers must never mint or forward an ``X-Kitty-*`` header — literal scan over every other source file."""
        other_files = {
            str(path.relative_to(_SRC)): path.read_text(encoding="utf-8")
            for path in sorted(_SRC.rglob("*.py"))
            if path != _SERVER_PY
        }

        offenders = kitty_header_leaks_outside_server(other_files)

        assert not offenders, (
            "X-Kitty-* literals must only live in src/kitty/bridge/server.py; "
            f"found leaks: {offenders}"
        )

    def test_kitty_header_leak_scan_finds_an_injected_header_in_code(self):
        """A provider file containing ``X-Kitty-Foo`` in code is reported (the leak-scan's known-positive)."""
        synthetic = {
            "providers/synthetic.py": (
                'class SyntheticProvider:\n'
                '    HEADERS = {"X-Kitty-Foo": "1"}\n'
            )
        }

        assert kitty_header_leaks_outside_server(synthetic) == {
            "providers/synthetic.py": ["X-Kitty-Foo"]
        }

    def test_kitty_header_leak_scan_catches_x_kitty_in_a_comment(self):
        """A comment mentioning ``X-Kitty-Foo`` is reported — pins the raw-text scan against AST-walk deviation."""
        synthetic = {
            "providers/synthetic.py": (
                "# TODO: the old bridge stamped X-Kitty-Foo; remove this proxy before migration is complete\n"
                "class SyntheticProvider:\n"
                '    HEADERS = {"Authorization": "Bearer x"}\n'
            )
        }

        assert kitty_header_leaks_outside_server(synthetic) == {
            "providers/synthetic.py": ["X-Kitty-Foo"]
        }

    def test_kitty_header_leak_scan_is_silent_on_a_clean_file(self):
        """The leak scan produces no offenders for a source file with no ``X-Kitty-*`` literal — its green baseline."""
        synthetic = {
            "providers/clean.py": (
                "class CleanProvider:\n"
                '    HEADERS = {"Authorization": "Bearer x", "content-type": "application/json"}\n'
            )
        }

        assert kitty_header_leaks_outside_server(synthetic) == {}


class TestEnvVarRegister:
    """Guard 3 — every ``KITTY_*`` env var the README names is read by the code."""

    def test_kitty_env_vars_in_readme_yields_known_subjects(self):
        """The README carries at least one known ``KITTY_*`` env-var name — the green baseline for the regex scanner."""
        names = kitty_env_vars_in_readme(readme_text())

        assert "KITTY_EGRESS_PROXY" in names
        assert "KITTY_SESSION_SUMMARY" in names

    def test_every_documented_kitty_env_var_is_read_by_the_code(self):
        """Forward direction: a documented name must appear as a string constant under ``src/kitty/``."""
        documented = kitty_env_vars_in_readme(readme_text())
        constants = string_constants_in_src()

        problems = check_env_register(documented, constants)

        assert not problems, (
            "README documents KITTY_* env vars that the code does not read as exact-match constants:\n  "
            + "\n  ".join(problems)
        )

    def test_check_env_register_reports_an_unread_documented_name(self):
        """The pure checker fails on a README-only name — its known-positive failure shape."""
        problems = check_env_register(["KITTY_DEFINITELY_NOT_READ"], {"OTHER_NAME"})

        assert problems
        assert any("KITTY_DEFINITELY_NOT_READ" in problem for problem in problems)

    def test_exact_match_not_substring_pins_the_constant_scan(self):
        """A docstring containing the name as a substring does NOT count as a reader.

        The string-constant scan uses exact equality, not substring
        matching. A future "loosen the matcher" refactor that switched
        to substring would let the docstring-only case through; this
        test pins the discipline.
        """
        # ``KITTY_SESSION_SUMMARY_NOTE`` contains ``KITTY_SESSION_SUMMARY``
        # as a substring but is not equal to it; a substring matcher would
        # report a reader where none exists.
        synthetic_constants = {"KITTY_SESSION_SUMMARY_NOTE", "OTHER_UNRELATED"}

        problems = check_env_register(["KITTY_SESSION_SUMMARY"], synthetic_constants)

        assert problems, (
            "check_env_register accepted 'KITTY_SESSION_SUMMARY' as read "
            "even though the constant 'KITTY_SESSION_SUMMARY_NOTE' only "
            "contains it as a substring; the exact-match contract is broken"
        )


class TestLoggingFlagTable:
    """Guard 4 — README's logging-flag table matches the parser and the default-path constants."""

    def test_each_documented_logging_flag_is_accepted_by_the_parser(self):
        """All four flags parse without an unknown-flag remainder.

        No exemption here: the negative case (an unknown flag must
        be rejected) is exercised by ``test_an_unknown_flag_is_rejected_by_the_parser``
        below. The positive case lives in this test.
        """
        parser = _build_parser()
        triples = readme_flag_rows(readme_table_rows(readme_text(), "Default path"))

        assert len(triples) == 2, (
            f"the README logging-flag table was expected to list two rows "
            f"(--logging, --debug); parsed {triples!r}"
        )

        for flag, _default, custom in triples:
            argv = [flag] + (["/tmp/example.log"] if custom else []) + ["bridge"]
            bound, unknown = parser_accepts_flag(parser, flag, argv)

            assert not unknown, (
                f"_build_parser rejected {flag!r} from argv={argv!r}; "
                f"unknown remainder: {sorted(unknown)!r}"
            )
            assert bound, (
                f"_build_parser accepted argv but did not bind {flag!r} to its dest; "
                f"unknown remainder: empty but the flag is not set"
            )

    def test_the_documented_default_paths_match_the_code(self):
        """The README's two default paths resolve to the bridge-server's constants — ``~`` expanded to ``$HOME``."""
        triples = readme_flag_rows(readme_table_rows(readme_text(), "Default path"))
        by_flag = {flag: default for flag, default, _custom in triples}

        # Bridge debug path: derived from the module constant (no instantiation
        # needed — the constant is the source of truth, mirroring how
        # ``tests/test_debug_log_path.py`` verifies it).
        debug_default = Path(by_flag["--debug"]).expanduser()
        assert debug_default == _DEBUG_LOG_PATH, (
            f"README documents --debug default {debug_default}; "
            f"src/kitty/bridge/server.py::_DEBUG_LOG_PATH is {bridge_default_debug_log_path()}"
        )

        # Usage default: derived from a freshly constructed BridgeServer
        # with logging enabled and no custom path — the same shape
        # ``tests/test_cli_log_file.py::TestCustomUsageLogPath`` verifies.
        usage_default = Path(by_flag["--logging"]).expanduser()
        assert usage_default == bridge_default_usage_log_path(), (
            f"README documents --logging default {usage_default}; "
            f"a fresh BridgeServer(_usage_log_path) is {bridge_default_usage_log_path()}"
        )

    def test_custom_path_flags_accept_a_path_argument(self):
        """``--log-file PATH`` and ``--debug-file PATH`` accept a path argument.

        The accept-test for the PATH metavar the README documents.
        """
        parser = _build_parser()

        for flag, _default, custom in readme_flag_rows(readme_table_rows(readme_text(), "Default path")):
            if not custom:
                continue
            argv = [flag, "/tmp/example.log", "bridge"]
            bound, unknown = parser_accepts_flag(parser, flag, argv)

            assert not unknown, (
                f"_build_parser rejected {flag!r} PATH from argv={argv!r}; "
                f"unknown remainder: {sorted(unknown)!r}"
            )
            assert bound, (
                f"_build_parser accepted argv but did not bind {flag!r} to its dest"
            )

    def test_an_unknown_flag_is_rejected_by_the_parser(self):
        """Negative control for the accept-tests: an undocumented flag must be rejected.

        Proves the accept-assertions above are not vacuous — a parser
        that silently swallows every flag would pass them all.
        """
        parser = _build_parser()

        assert parser_rejects(parser, ["--definitely-not-a-flag", "bridge"]), (
            "_build_parser accepted --definitely-not-a-flag; either the "
            "parser's wildcard-passing or argparse's semantics have changed — "
            "the accept-tests above are vacuous"
        )
