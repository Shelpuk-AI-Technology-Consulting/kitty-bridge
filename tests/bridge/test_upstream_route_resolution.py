"""The upstream path and auth headers must resolve from the model the body was built from.

KBR-127.  ``BridgeServer._normalize_model`` writes the model the request is
actually *for* into ``cc_request["model"]`` — overriding it with the profile
model when there is one, then running it through the adapter's
``normalize_model_name``.  Every routing decision about the **body** reads that
key: :meth:`~kitty.providers.opencode.OpenCodeGoAdapter.translate_to_upstream`
does, and since KBR-7 so does the thinking-repair carrier in
``BridgeServer._upstream_body_for``.

The **path** and the **auth scheme** used to resolve from ``self._active_model``
instead — the profile's model, never normalized.  Two sources of truth for one
question, and they disagree in two reachable cases:

* a profile model carrying a provider prefix (``opencode/minimax-m2.5``), where
  the body is built for ``minimax-m2.5`` and the route for the prefixed name;
* no profile model at all, where both helpers resolved from ``""`` and every
  request took the adapter's default route regardless of what the agent asked
  for.

Both produce the same failure: an Anthropic Messages body posted to the Chat
Completions endpoint under Bearer auth, or the reverse.

**What these tests assert, and in what layer.**  Each case is a pure function of
(adapter, profile model, request model) and needs no socket, so it is L1 by
``.system_design/TEST_SUITE.md`` §2.2 — the layer marker comes from the path
default in ``tests/layers.py``.  Three cases (``glm-5`` in both configurations,
and Vertex) pass before the fix as well as after.  They are here deliberately:
the failure mode of this change is *inverting* the routing, and a suite that
asserts only the broken direction cannot see that.

**Why the body shape is re-derived here.**  ``tests/test_wire_shape_honesty.py``
already owns a classifier for the Messages/Chat-Completions distinction.  This
module deliberately does not import it, so the two guards fail independently: a
defect in that classifier must not be able to turn these tests green.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.azure import AzureOpenAIAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.opencode import OpenCodeGoAdapter
from kitty.providers.vertex import VertexAIAdapter

# A model OpenCode Go serves on /v1/messages, and one it serves on
# /v1/chat/completions. Kept as names rather than inlined so a routing-table
# change (KBR-126) lands in one place.
_MESSAGES_MODEL = "minimax-m2.5"
_CHAT_MODEL = "glm-5"


def _probe_request(model: str) -> dict:
    """Build a Chat Completions request that both wire dialects render differently.

    The body must carry a system message **and** a tool: those are the two axes
    :func:`_wire_shape` keys on, and a probe missing either is classified
    ``"other"`` regardless of which dialect the adapter emitted.

    Args:
        model: The model the agent is asking for, before normalization.

    Returns:
        A Chat Completions request dict.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a bridge."},
            {"role": "user", "content": "hello"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
    }


def _wire_shape(body: dict) -> str:
    """Classify an upstream body as ``"messages"``, ``"chat_completions"`` or ``"other"``.

    Keys on two axes that must agree — how a tool is described, and where the
    system prompt sits.  Anthropic hoists the system prompt to a top-level field
    and describes a tool with ``input_schema``; Chat Completions keeps a system
    turn and wraps each tool in a ``function`` envelope.  Requiring both means a
    body in neither dialect is reported as ``"other"`` rather than guessed at.

    Args:
        body: The dict an adapter's ``translate_to_upstream`` returned.

    Returns:
        The dialect name.
    """
    tools = body.get("tools")
    first = tools[0] if isinstance(tools, list) and tools and isinstance(tools[0], dict) else None
    roles = {m.get("role") for m in body.get("messages", []) if isinstance(m, dict)}

    if first is not None and "input_schema" in first and "system" not in roles:
        return "messages"
    if first is not None and "function" in first and "system" in roles:
        return "chat_completions"
    return "other"


def _server(
    provider: ProviderAdapter,
    *,
    model: str | None = None,
    provider_config: dict | None = None,
) -> BridgeServer:
    """Build an unstarted single-backend bridge over *provider*.

    Single-backend rather than a balancing pool, because ``Profile.model`` is a
    required non-empty string: the "no profile model" half of KBR-127 cannot be
    expressed through a pool at all, and the two halves must be built the same
    way to be comparable.

    Args:
        provider: The adapter under test.
        model: The profile model, or ``None`` for a bridge with none.
        provider_config: Provider configuration the adapter needs to build its
            base URL.

    Returns:
        An unstarted :class:`~kitty.bridge.server.BridgeServer`.
    """
    return BridgeServer(
        None,
        provider,
        "test-key",
        model=model,
        provider_config=provider_config,
    )


def _route(server: BridgeServer, request_model: str) -> tuple[str, dict[str, str], dict]:
    """Resolve the path, headers and body a request would go out with.

    Mirrors the order every handler uses: normalize first, then build the three
    outputs from the normalized request.  The point of KBR-127 is that all three
    must come from that one normalized model, so the test takes all three from
    one call rather than asserting them in separate cases.

    Args:
        server: The bridge under test.
        request_model: The model the agent asks for.

    Returns:
        The upstream URL, the upstream headers, and the upstream body.
    """
    cc_request = _probe_request(request_model)
    server._normalize_model(cc_request)

    return (
        server._build_upstream_url(cc_request),
        server._build_upstream_headers(cc_request),
        server._active_provider.translate_to_upstream(cc_request),
    )


class TestOpenCodeGoWithAProfileModel:
    """A profile model carrying a provider prefix must not split the route."""

    def test_a_prefixed_messages_model_routes_to_messages_on_all_three(self) -> None:
        """KBR-127(a): path, auth and body for ``opencode/minimax-m2.5``.

        Asserted together in one test because the defect is a *divergence*:
        each of the three was individually defensible, and only their
        disagreement was the bug.
        """
        server = _server(OpenCodeGoAdapter(), model=f"opencode/{_MESSAGES_MODEL}")

        url, headers, body = _route(server, "claude-sonnet-4-5")

        assert url.endswith("/v1/messages"), url
        assert headers.get("x-api-key") == "test-key"
        assert "Authorization" not in headers
        assert _wire_shape(body) == "messages"

    def test_a_prefixed_chat_model_still_routes_to_chat_completions(self) -> None:
        """The non-regression direction: ``opencode/glm-5`` must not move.

        Passes before the fix as well as after.  Without it, inverting the
        routing would satisfy every other case in this file.
        """
        server = _server(OpenCodeGoAdapter(), model=f"opencode/{_CHAT_MODEL}")

        url, headers, body = _route(server, "claude-sonnet-4-5")

        assert url.endswith("/v1/chat/completions"), url
        assert headers.get("Authorization") == "Bearer test-key"
        assert "x-api-key" not in headers
        assert _wire_shape(body) == "chat_completions"


class TestOpenCodeGoWithoutAProfileModel:
    """With no profile model, the route must follow what the agent asked for."""

    def test_a_messages_model_from_the_agent_routes_to_messages(self) -> None:
        """KBR-127(b): ``_active_model is None`` used to force the default route.

        Every OpenCode Go request took ``/v1/chat/completions`` with Bearer
        auth, whatever the agent asked for — while the body was built for the
        model it did ask for.
        """
        server = _server(OpenCodeGoAdapter())

        url, headers, body = _route(server, _MESSAGES_MODEL)

        assert url.endswith("/v1/messages"), url
        assert headers.get("x-api-key") == "test-key"
        assert "Authorization" not in headers
        assert _wire_shape(body) == "messages"

    def test_a_chat_model_from_the_agent_routes_to_chat_completions(self) -> None:
        """The other direction on the same server: ``glm-5`` stays on Chat Completions."""
        server = _server(OpenCodeGoAdapter())

        url, headers, body = _route(server, _CHAT_MODEL)

        assert url.endswith("/v1/chat/completions"), url
        assert headers.get("Authorization") == "Bearer test-key"
        assert "x-api-key" not in headers
        assert _wire_shape(body) == "chat_completions"


class TestAzureDeploymentPath:
    """Azure puts the model in the path, so an un-normalized name is a broken URL."""

    def test_a_prefixed_profile_model_does_not_leak_its_prefix_into_the_path(self) -> None:
        """``azure/my-deploy`` must address the deployment ``my-deploy``.

        Before the fix the prefix became a path segment of its own —
        ``/openai/deployments/azure/my-deploy/chat/completions`` — addressing a
        deployment that cannot exist.
        """
        server = _server(AzureOpenAIAdapter(), model="azure/my-deploy")

        url, _, _ = _route(server, "gpt-4o")

        assert "/openai/deployments/my-deploy/chat/completions" in url, url
        assert "/deployments/azure/" not in url, url

    def test_without_a_profile_model_the_agents_model_names_the_deployment(self) -> None:
        """A deliberate behaviour change, asserted rather than discovered.

        ``get_upstream_path`` falls back to the literal ``deployment`` when the
        model is empty, so before the fix every Azure request with no profile
        model addressed ``/openai/deployments/deployment/…`` — a placeholder no
        resource serves.  It now addresses the model the agent asked for, which
        is correct whenever the deployment is named after the model and no worse
        otherwise.
        """
        server = _server(AzureOpenAIAdapter())

        url, _, _ = _route(server, "my-deploy")

        assert "/openai/deployments/my-deploy/chat/completions" in url, url


class TestAnAdapterThatIgnoresTheModelIsUnaffected:
    """Most adapters do not route on the model, and must not start to."""

    @pytest.mark.parametrize("model", ["gemini-2.5-pro", "google/gemini-2.5-pro", ""])
    def test_vertex_resolves_one_path_for_every_model(self, model: str) -> None:
        """Vertex overrides ``get_upstream_path`` but ignores its argument.

        It is the third and only other override in the tree, so it is the case
        that proves the change is confined to adapters that actually route on
        the model.

        Args:
            model: A model name, prefixed, bare, or absent.
        """
        server = _server(VertexAIAdapter(), provider_config={"project_id": "proj"})

        url, _, _ = _route(server, model)

        assert url.endswith("/endpoints/openapi/chat/completions"), url


# ── The structural guard ───────────────────────────────────────────────────
#
# The tests above prove the route is resolved correctly today. This proves it
# cannot quietly stop being, which is a different claim: KBR-127 was not a
# wrong rule, it was a *second* source for a rule that already existed.

_HELPERS = ("_build_upstream_url", "_build_upstream_headers")

# `_active_model` is a property over `_model`; reading either reinstates the
# defect, so the guard forbids both names rather than the one the bug used.
_FORBIDDEN_ATTRIBUTES = ("_active_model", "_model")

_SERVER_MODULE = Path(__file__).resolve().parents[2] / "src" / "kitty" / "bridge" / "server.py"
_SCANNED_TREES = (
    Path(__file__).resolve().parents[2] / "src",
    Path(__file__).resolve().parents[2] / "tests",
)


def _function_defs(tree: ast.AST, names: tuple[str, ...]) -> dict[str, ast.FunctionDef]:
    """Collect the named function definitions from a parsed module.

    Args:
        tree: A parsed module.
        names: The function names to find.

    Returns:
        The definitions found, keyed by name.  A name that is absent is simply
        missing from the mapping; callers assert on that themselves, so the
        guard reports "the method was renamed" rather than a ``KeyError``.
    """
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node

    return found


def _self_attributes_read(function: ast.FunctionDef) -> set[str]:
    """Return the ``self.<name>`` attributes a function body reads.

    Args:
        function: The definition to inspect.

    Returns:
        Every attribute name accessed on ``self``.
    """
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }


def _reads_name(function: ast.FunctionDef, name: str) -> bool:
    """Return whether a function body reads a bare name.

    Args:
        function: The definition to inspect.
        name: The identifier to look for — here, the parameter that carries the
            request.

    Returns:
        True when the name is loaded anywhere in the body.
    """
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(function))


def _zero_argument_calls(tree: ast.AST) -> list[int]:
    """Return the line numbers of no-argument calls to either helper.

    Args:
        tree: A parsed module.

    Returns:
        One line number per offending call, in source order.
    """
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _HELPERS
        and not node.args
        and not node.keywords
    )


class TestRouteResolutionHasOneSourceOfTruth:
    """Both helpers must resolve the model from the request, and only from it.

    **What this guard does not prove.**  It reads names, not semantics, and it
    cannot see a helper replaced at runtime — ``tests/bridge/
    test_compaction_failure_response.py`` monkeypatches a stand-in over
    ``_build_upstream_headers``, and a stand-in with the wrong signature is
    caught by that test failing, not by this one.  Each check below asserts its
    own subject set and is exercised against a known defect, in the style of
    ``tests/test_egress_coverage.py``, so none can pass by having looked at
    nothing.
    """

    @pytest.fixture
    def helpers(self) -> dict[str, ast.FunctionDef]:
        """Parse ``server.py`` and return the two helper definitions.

        Returns:
            The definitions, keyed by method name.
        """
        tree = ast.parse(_SERVER_MODULE.read_text(encoding="utf-8"))
        found = _function_defs(tree, _HELPERS)

        assert set(found) == set(_HELPERS), (
            f"expected to find {list(_HELPERS)} in {_SERVER_MODULE.name}, found {sorted(found)}. "
            "A renamed method makes every check below vacuous."
        )
        return found

    @pytest.mark.parametrize("helper", _HELPERS)
    def test_the_helper_does_not_read_the_profile_model(self, helpers, helper: str) -> None:
        """Neither ``_active_model`` nor its backing field may appear (KBR-127).

        Args:
            helpers: The parsed helper definitions.
            helper: The method under test.
        """
        read = _self_attributes_read(helpers[helper])
        offending = sorted(read & set(_FORBIDDEN_ATTRIBUTES))

        assert not offending, (
            f"{helper} reads self.{' / self.'.join(offending)}. That is the second source of "
            "truth KBR-127 removed: the profile model is not normalized, so the route it "
            "resolves can disagree with the body translate_to_upstream builds."
        )

    @pytest.mark.parametrize("helper", _HELPERS)
    def test_the_helper_reads_the_request_it_was_given(self, helpers, helper: str) -> None:
        """Absence is not enough — the helper must read ``cc_request``.

        A helper that read nothing at all, returning a hardcoded route, would
        satisfy the absence check while being just as wrong.

        Args:
            helpers: The parsed helper definitions.
            helper: The method under test.
        """
        parameters = [arg.arg for arg in helpers[helper].args.args]

        assert "cc_request" in parameters, f"{helper} no longer takes cc_request; it takes {parameters}"
        assert _reads_name(helpers[helper], "cc_request"), (
            f"{helper} takes cc_request and never reads it, so the route is resolved from "
            "something other than the request it is for."
        )

    def test_no_call_site_resolves_a_route_without_naming_the_request(self) -> None:
        """No zero-argument call of either helper survives in ``src/`` or ``tests/``.

        CI type-checks ``src/kitty`` only, so a stale call in a test file is a
        ``TypeError`` raised whenever that branch happens to run.  Several of
        the call sites sit in deeply nested failover branches, which is exactly
        where "whenever that branch happens to run" means "in production".
        """
        offending: list[str] = []
        scanned = 0

        for tree_root in _SCANNED_TREES:
            for path in sorted(tree_root.rglob("*.py")):
                scanned += 1
                for lineno in _zero_argument_calls(ast.parse(path.read_text(encoding="utf-8"))):
                    offending.append(f"{path.relative_to(tree_root.parent)}:{lineno}")

        assert scanned > 100, f"the sweep found only {scanned} Python files; it is not reading the tree"
        assert not offending, "these call sites resolve a route without naming a request:\n  " + "\n  ".join(offending)

    @pytest.mark.parametrize(
        ("defect", "source"),
        [
            (
                "reads the profile model through the property",
                "def _build_upstream_url(self, cc_request):\n    return self._active_model or ''\n",
            ),
            (
                "reads the profile model through its backing field",
                "def _build_upstream_url(self, cc_request):\n    return self._model or ''\n",
            ),
        ],
    )
    def test_the_absence_check_detects_a_known_defect(self, defect: str, source: str) -> None:
        """The absence check must report each shape of the defect it exists to catch.

        Args:
            defect: What the synthetic function does wrong.
            source: A function carrying that defect.
        """
        function = _function_defs(ast.parse(source), _HELPERS)["_build_upstream_url"]

        assert _self_attributes_read(function) & set(_FORBIDDEN_ATTRIBUTES), (
            f"the absence check did not notice a helper that {defect}"
        )

    def test_the_presence_check_detects_a_helper_that_reads_nothing(self) -> None:
        """A hardcoded route passes the absence check, so the presence check must fail it."""
        source = "def _build_upstream_headers(self, cc_request):\n    return {'Authorization': 'Bearer x'}\n"
        function = _function_defs(ast.parse(source), _HELPERS)["_build_upstream_headers"]

        assert not _self_attributes_read(function) & set(_FORBIDDEN_ATTRIBUTES), "precondition: no forbidden read"
        assert not _reads_name(function, "cc_request"), (
            "the presence check did not notice a helper that ignores the request it was given"
        )

    def test_the_call_site_sweep_detects_a_zero_argument_call(self) -> None:
        """The sweep must report a no-argument call, or it is a no-op over a clean tree."""
        source = "url = self._build_upstream_url()\nheaders = self._build_upstream_headers(cc_request)\n"

        assert _zero_argument_calls(ast.parse(source)) == [1]
