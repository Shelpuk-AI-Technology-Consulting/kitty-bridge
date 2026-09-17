"""The per-protocol registration matrix and the schema↔routes agreement.

``.system_design/TEST_SUITE.md`` §6.2.1 (L2) · Jira **KBR-82** (T-G6).

``_register_routes`` is a switch on ``bridge_protocol``. Bridge mode
(``self._adapter is None``) registers **all five** POST routes plus three
GETs; agent-launch mode registers only the routes matching the launcher's
protocol. §6.2.1 records why this matters: a conformance run against a
``kitty claude`` bridge would see a single route, pass, and leave the rest
unvalidated.

Two guards here:

* **The matrix** — for each ``BridgeProtocol`` value, the set of routes
  registered is exactly what §6.2.1 documents. This is a *static* claim
  about the method's code shape (AST-walked, the same pattern the KBR-9
  endpoint-table guard uses), falsifiable without starting a bridge.
* **The agreement** — the published OpenAPI document's ``paths`` agree with
  the bridge-mode route set, in both directions. Without this, a route
  dropped from the schema is silently unvalidated by the conformance run —
  the exact failure mode the "must target bridge mode" note warns about.

A third guard pins the **auth-off** precondition the conformance run relies
on: the bridge fixture starts its ``BridgeServer`` without a ``keys_file``,
so the auth middleware is a no-op. If a future fixture change turns auth on,
every fuzzed request would 401 and the conformance run would go green
vacuously — this guard makes that change visible instead.

aiohttp's path template uses regex converters (``{model:.*}``); OpenAPI 3.1
uses plain placeholders (``{model}``). The comparison normalises the
converter suffix before set-equality — the same character-exact-match trap
KBR-9's endpoint-table guard hit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from kitty.bridge.server import BridgeServer

pytestmark = pytest.mark.l2

SERVER_PATH = Path(__file__).parent.parent / "src" / "kitty" / "bridge" / "server.py"
SCHEMA_PATH = Path(__file__).parent.parent / "openapi" / "kitty-bridge.yaml"

#: The route set each mode must register, in aiohttp form — exactly what
#: `_register_routes` writes.
BRIDGE_MODE_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("get", "/healthz"),
        ("get", "/stats"),
        ("get", "/v1/models"),
        ("post", "/v1/chat/completions"),
        ("post", "/v1/messages"),
        ("post", "/v1/responses"),
        ("post", "/v1beta/models/{model:.*}:generateContent"),
        ("post", "/v1beta/models/{model:.*}:streamGenerateContent"),
    }
)

#: One POST per protocol for the three single-protocol launch modes, two for
#: Gemini. `/healthz` and `/stats` are unconditional and appear in every set.
LAUNCH_MODE_ROUTES: dict[str, frozenset[tuple[str, str]]] = {
    "RESPONSES_API": frozenset({("post", "/v1/responses")}),
    "MESSAGES_API": frozenset({("post", "/v1/messages")}),
    "CHAT_COMPLETIONS_API": frozenset({("post", "/v1/chat/completions")}),
    "GEMINI_API": frozenset(
        {
            ("post", "/v1beta/models/{model:.*}:generateContent"),
            ("post", "/v1beta/models/{model:.*}:streamGenerateContent"),
        }
    ),
}


def _register_routes_branches() -> tuple[list[ast.stmt], dict[str, list[ast.stmt]]]:
    """Locate `_register_routes`'s bridge-mode and launch-mode statement blocks.

    Bridge mode is the prefix ``/healthz`` + ``/stats`` registrations plus
    the body of ``if self._adapter is None:``. Launch mode is the
    ``if protocol == BridgeProtocol.X:`` dispatch that follows the
    bridge-mode ``if``'s ``return``.

    Returns:
        A pair ``(bridge_block, launch_branches)``: the statements that run
        in bridge mode, and a dict mapping each ``BridgeProtocol`` member
        name to the statements in its launch branch.

    Raises:
        AssertionError: When the function or its two structural halves are
            not where this guard expects them.
    """
    source = SERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SERVER_PATH))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_register_routes"):
            continue
        assert node.body, "_register_routes has an empty body"
        bridge_block: list[ast.stmt] = []
        bridge_if_index = -1
        for index, statement in enumerate(node.body):
            if isinstance(statement, ast.If) and _is_adapter_none_test(statement.test):
                bridge_if_index = index
                bridge_block.extend(statement.body)
                break
            bridge_block.append(statement)
        if bridge_if_index < 0:
            raise AssertionError("no `if self._adapter is None:` branch found in _register_routes")

        launch_branches = _launch_branches(node.body[bridge_if_index + 1 :])
        return bridge_block, launch_branches
    raise AssertionError("_register_routes is not defined in src/kitty/bridge/server.py")


def _launch_branches(statements_after_bridge_if: list[ast.stmt]) -> dict[str, list[ast.stmt]]:
    """Collect the per-protocol dispatch bodies from the launch-mode chain.

    The dispatch reads as a flat ``if/elif`` chain whose conditions name a
    ``BridgeProtocol`` member; each ``elif`` is encoded as the next ``If``
    in the AST's ``orelse`` of the previous one. Skips the
    ``protocol = self._adapter.bridge_protocol`` assignment between the
    bridge-mode ``if`` and the first launch branch — that statement is a
    discriminator read, not a route registration.

    Args:
        statements_after_bridge_if: The statements in `_register_routes`'s
            body that follow the bridge-mode ``if``.

    Returns:
        A ``{protocol_name: branch_body}`` mapping covering every branch
        the dispatch contains.

    Raises:
        AssertionError: When the dispatch is not where this guard expects
            it, or a branch's condition does not read as a
            ``BridgeProtocol`` member comparison.
    """
    branches: dict[str, list[ast.stmt]] = {}
    chain_head: ast.If | None = None
    for statement in statements_after_bridge_if:
        if isinstance(statement, ast.If):
            chain_head = statement
            break
    if chain_head is None:
        raise AssertionError("no `if protocol == BridgeProtocol.X:` dispatch found after the bridge-mode `if`")

    node: ast.stmt = chain_head
    while isinstance(node, ast.If):
        protocol_name = _protocol_member_from_test(node.test)
        branches[protocol_name] = node.body
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else:
            break
    return branches


def _protocol_member_from_test(test: ast.expr) -> str:
    """Extract the protocol name from a `protocol == BridgeProtocol.X` test.

    Args:
        test: The condition of an ``If`` branch in the launch dispatch.

    Returns:
        The ``BridgeProtocol.X`` member name, e.g. ``"RESPONSES_API"``.

    Raises:
        AssertionError: When the condition is not a
            ``protocol == BridgeProtocol.X`` comparison.
    """
    assert isinstance(test, ast.Compare), f"launch dispatch test is not a Compare: {ast.dump(test)}"
    assert len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq), (
        f"launch dispatch comparison is not ==: {ast.dump(test)}"
    )
    right = test.comparators[0]
    assert isinstance(right, ast.Attribute) and right.attr.isupper(), (
        f"launch dispatch compares against something other than `BridgeProtocol.X`: {ast.dump(test)}"
    )
    return right.attr


def _is_adapter_none_test(test: ast.expr) -> bool:
    """Return True when `test` reads as `self._adapter is None`.

    Args:
        test: The condition of an `If` statement.

    Returns:
        True when the condition is the bridge-mode discriminator.
    """
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    return isinstance(test.ops[0], ast.Is) and isinstance(test.comparators[0], ast.Constant)


def _registered_routes_from(statements: list[ast.stmt]) -> set[tuple[str, str]]:
    """Collect `(method, path)` pairs from the add_post/add_get calls in a block.

    Args:
        statements: AST statements to scan.

    Returns:
        The route set, in source order of the block.
    """
    routes: set[tuple[str, str]] = set()
    for statement in statements:
        for node in ast.walk(statement):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            method = node.func.attr
            if method not in ("add_post", "add_get"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            routes.add((method.removeprefix("add_"), str(node.args[0].value)))
    return routes


def _normalise_aiohttp_path(path: str) -> str:
    """Strip the aiohttp regex-converter suffix from every path placeholder.

    Args:
        path: A path template in aiohttp (`{model:.*}`) or OpenAPI (`{model}`) form.
            Placeholders may appear anywhere in the path (Gemini's are
            mid-path, followed by ``:generateContent``).

    Returns:
        The OpenAPI path-template form, with every `{name:regex}` rewritten
        to `{name}`.
    """
    out: list[str] = []
    index = 0
    while index < len(path):
        char = path[index]
        if char != "{":
            out.append(char)
            index += 1
            continue
        # Find the matching `}` accounting for nested braces (aiohttp regex
        # converters may contain `{` or `}` inside a quantifier).
        depth = 0
        for probe in range(index, len(path)):
            probe_char = path[probe]
            if probe_char == "{":
                depth += 1
            elif probe_char == "}":
                depth -= 1
                if depth == 0:
                    placeholder = path[index : probe + 1]
                    name = placeholder[1:-1].split(":", 1)[0]
                    out.append("{" + name + "}")
                    index = probe + 1
                    break
        else:
            # Unbalanced `{` — leave the rest untouched.
            out.append(path[index:])
            break
    return "".join(out)


def _registered_routes() -> set[tuple[str, str]]:
    """Collect every add_post/add_get route in `_register_routes`.

    Returns:
        All routes the method registers, across both branches.
    """
    source = SERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SERVER_PATH))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_register_routes"):
            continue
        return _registered_routes_from(node.body)
    return set()


class TestRegistrationMatrix:
    """Each mode registers exactly the route set §6.2.1 documents."""

    def test_bridge_mode_registers_all_routes(self) -> None:
        """Bridge mode: five POSTs plus three GETs."""
        bridge_block, _ = _register_routes_branches()
        routes = _registered_routes_from(bridge_block)
        assert routes == BRIDGE_MODE_ROUTES

    @pytest.mark.parametrize("protocol", sorted(LAUNCH_MODE_ROUTES))
    def test_launch_mode_registers_exactly_its_protocol(self, protocol: str) -> None:
        """Agent-launch mode: `protocol`'s dispatch registers exactly its route set.

        Args:
            protocol: A `BridgeProtocol` enum member name.
        """
        _bridge_block, launch_branches = _register_routes_branches()
        assert protocol in launch_branches, (
            f"no `if protocol == BridgeProtocol.{protocol}:` branch found in _register_routes; "
            f"dispatch contains: {sorted(launch_branches)}"
        )
        routes = _registered_routes_from(launch_branches[protocol])
        assert routes == LAUNCH_MODE_ROUTES[protocol], (
            f"{protocol}: registered {sorted(routes)}, expected exactly {sorted(LAUNCH_MODE_ROUTES[protocol])}"
        )

    def test_the_four_launch_sets_are_disjoint(self) -> None:
        """A route leaking between protocols would let a launch-mode bridge serve the wrong dialect."""
        sets = list(LAUNCH_MODE_ROUTES.values())
        for index, one in enumerate(sets):
            for other in sets[index + 1 :]:
                assert one.isdisjoint(other), f"launch-mode route sets overlap: {one} ∩ {other}"


class TestSchemaAgreesWithBridgeModeRoutes:
    """The published schema's `paths` match the bridge-mode route set, both ways."""

    def test_the_published_schema_has_every_bridge_mode_path(self) -> None:
        """A route missing from the schema is silently unvalidated by the conformance run."""
        with open(SCHEMA_PATH) as f:
            schema_paths = set(yaml.safe_load(f)["paths"])
        registered = {path for _method, path in _registered_routes()}
        normalised = {_normalise_aiohttp_path(path) for path in registered}
        assert normalised == schema_paths, (
            f"schema↔routes drift: registered={sorted(normalised)}, published={sorted(schema_paths)}"
        )


class TestBridgeFixtureStartsWithAuthOff:
    """The auth-off precondition the conformance run relies on, made a checkable claim."""

    def test_a_bridge_without_keys_file_allows_all(self) -> None:
        """`keys_file=None` leaves `_keys_entries` empty, so the auth middleware is a no-op.

        If a future change makes the fixture pass a `keys_file`, every fuzzed
        request the conformance run sends would 401 and `not_a_server_error`
        would pass vacuously — this test turns that into a red flag first.
        """
        from kitty.profiles.schema import Profile
        from kitty.providers.custom_anthropic import CustomAnthropicAdapter

        profile = Profile(
            name="auth-off-probe",
            provider="custom_anthropic",
            model="m",
            auth_ref="6f1d4f3e-2b9a-4c1d-8f7e-5a2b3c4d5e6f",
        )
        server = BridgeServer(
            None,  # type: ignore[arg-type]
            CustomAnthropicAdapter(),
            "probe-key",
            model=profile.model,
            provider_config={"base_url": "http://127.0.0.1:1"},
        )
        assert server._keys_entries == {}, (
            "a BridgeServer without keys_file must have an empty _keys_entries "
            "(the conformance run depends on the auth middleware being a no-op)"
        )
