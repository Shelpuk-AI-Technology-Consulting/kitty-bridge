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

The *structural* half of this claim — that neither helper can quietly reacquire a
second source for the model — is L2 and lives in
``tests/test_upstream_route_source_of_truth.py``.

**Why the body shape is re-derived here.**  ``tests/test_wire_shape_honesty.py``
already owns a classifier for the Messages/Chat-Completions distinction.  This
module deliberately does not import it, so the two guards fail independently: a
defect in that classifier must not be able to turn these tests green.
"""

from __future__ import annotations

import pytest

from kitty.bridge.server import BridgeServer, _route_model
from kitty.providers.azure import AzureOpenAIAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.opencode import OpenCodeGoAdapter
from kitty.providers.registry import _registry, get_provider
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


class TestTheRoutingKeyFallback:
    """Every shape that is not a usable model resolves to the same empty answer."""

    @pytest.mark.parametrize(
        ("label", "cc_request"),
        [
            ("absent", {}),
            ("null", {"model": None}),
            ("empty", {"model": ""}),
            ("not a string", {"model": 123}),
        ],
    )
    def test_an_unusable_model_falls_back_to_the_empty_string(self, label: str, cc_request: dict) -> None:
        """``_route_model`` must fold all of them to ``""``, as ``_active_model or ""`` did.

        Nothing validates this field's type, so a client can send any of these.
        The fallback is deliberately byte-identical to what the code KBR-127
        replaced produced: the request takes the adapter's default route and
        collects a proper error from the provider, rather than a 500 from the
        bridge or a route picked from a number.

        Args:
            label: What the request carries in place of a model.
            cc_request: The request.
        """
        assert _route_model(cc_request) == ""


class TestTheCustomUrlErrorPathStaysModelIndependent:
    """The 404 message rebuilds a route with no request in scope."""

    def test_no_custom_url_adapter_routes_on_the_model(self) -> None:
        """``_translate_upstream_error`` may only skip the model while this holds.

        KBR-134's 404 branch reports the URL the bridge asked for, and it runs
        where no ``cc_request`` exists. It is sound only because both adapters
        that set ``requires_custom_url`` inherit ``get_upstream_path``, which
        ignores its argument — so the reported route cannot depend on the model.

        An adapter that required a custom URL *and* routed per model would make
        that message name an endpoint the request never used, which on a 404 is
        the one moment the user is being told to go and check it. This fails
        first, so that lands as a decision rather than a wrong error string.
        """
        offenders = sorted(
            name
            for name in _registry
            if get_provider(name).requires_custom_url
            and type(get_provider(name)).get_upstream_path is not ProviderAdapter.get_upstream_path
        )

        assert not offenders, (
            f"{offenders} both require a custom URL and route on the model. The 404 branch in "
            "_translate_upstream_error builds its route from an empty request, so it would now "
            "report the wrong endpoint; give it the real model or narrow the branch."
        )

    def test_the_subject_set_is_not_empty(self) -> None:
        """The check above passes trivially if no adapter requires a custom URL."""
        requiring = sorted(name for name in _registry if get_provider(name).requires_custom_url)

        assert requiring == ["custom_anthropic", "custom_openai"], requiring


class TestTheRoutingKeySurvivesSerialization:
    """``translate_to_upstream`` must not remove ``model`` from the request it is given."""

    @pytest.mark.parametrize("provider_type", sorted(_registry))
    def test_no_adapter_strips_the_model_from_the_request_in_place(self, provider_type: str) -> None:
        """A failover re-reads the route from a request already serialized once.

        Within a single attempt the URL is built *before* the body, so the
        ordering looks safe site by site.  It is not safe across attempts: when
        an attempt fails, a later branch rebuilds the URL from the **same**
        ``cc_request`` object, which ``translate_to_upstream`` has already been
        handed.  Fifteen of the twenty-three ``_build_upstream_url`` sites sit
        after such a call with no intervening rebuild of the dict, and the path
        is thoroughly live rather than theoretical — instrumenting the helper
        by dict identity across ``tests/bridge`` recorded **57 of 303** URL
        builds landing on an already-translated request.

        So KBR-127 turned "the request keeps its ``model`` key" from an
        implementation detail into an invariant.  ``AzureOpenAIAdapter`` is the
        one adapter that removes ``model`` (register entry P6) and is safe only
        because it builds a copy; one that popped the key in place would send
        the failover attempt to the default route with an empty model —
        KBR-127(b), reappearing on the path hardest to reproduce.

        Args:
            provider_type: A key of the provider registry.
        """
        adapter = get_provider(provider_type)
        cc_request = _probe_request("kitty-test-model")

        adapter.translate_to_upstream(cc_request)

        assert cc_request.get("model") == "kitty-test-model", (
            f"{provider_type} removed or rewrote cc_request['model'] in place; every helper "
            "that re-reads the route after serialization would resolve the wrong one."
        )

    @pytest.mark.parametrize("provider_type", sorted(_registry))
    def test_no_adapter_rewrites_the_model_in_normalize_request(self, provider_type: str) -> None:
        """``normalize_request`` runs between ``_normalize_model`` and the helpers.

        At every one of the 46 call sites the order is normalize, then
        ``normalize_request``, then build the route.  So an adapter that
        rewrote ``model`` in that hook would hand the helpers a string
        ``_normalize_model`` never produced, and the body and the route would
        diverge again by a different door.

        Args:
            provider_type: A key of the provider registry.
        """
        adapter = get_provider(provider_type)
        cc_request = _probe_request("kitty-test-model")

        adapter.normalize_request(cc_request)

        assert cc_request.get("model") == "kitty-test-model", (
            f"{provider_type}.normalize_request changed cc_request['model']; the route would "
            "then resolve from a string _normalize_model never produced."
        )
