"""Contract guard — a provider adapter must declare the wire shape it actually emits.

`.system_design/TEST_SUITE.md` §6.2.3, "Wire-shape honesty".  Catches finding
**F5** (KBR-7): ``OpenCodeGoAdapter`` inherited ``upstream_wire_is_messages_api
== True`` from :class:`~kitty.providers.anthropic.AnthropicAdapter` while
emitting a Chat Completions body for every model outside ``_MESSAGES_MODELS``.
The bridge's thinking round-trip repair branches on that declaration, so a
wrong answer malforms the transcript it was sent to fix.

**What this guard does not prove.**  Five things, deliberately, so a green run
is not over-read:

1. **The serialization boundary.**  This observes ``translate_to_upstream``'s
   return value.  Three adapters set ``use_custom_transport`` and bypass
   ``_make_upstream_request``, and what a green result means differs for each:

   * ``openai_subscription`` — never calls the hook on the request path at all
     (``_cc_to_responses`` builds the Responses body inside the transport).
     The sweep checks its declaration against a body this adapter never sends,
     so here the exemption is **load-bearing**.
   * ``bedrock`` — calls the hook, then pops ``modelId`` and ``stream`` and
     reshapes into boto3 kwargs.
   * ``ollama_cloud`` — calls the hook, then overwrites ``stream``.

   For the latter two the mutation is scalar and does not change the body's
   **shape family**, so the exemption is precautionary rather than load-bearing
   — but a guard must observe the shipped bytes, not infer them.  Extending it
   to the wire needs the per-transport recorders and is **KBR-80 / T-G4**.
   :func:`test_custom_transport_adapters_are_the_known_exempt_set` pins the set
   so a fourth such adapter forces a decision rather than quietly inheriting a
   clean bill of health.
2. **Adapters constructed with ``provider_config``.**  ``get_provider`` forwards
   it to adapters that accept it, and at least one branches on it.  Default
   construction only, here.
3. **Native-passthrough requests.**  The probe is a Chat Completions request
   with no ``_native_messages_request`` flag.
4. **That the routing table matches the provider.**  This guard proves the
   declaration matches the *emitted body*.  It cannot prove that the models
   kitty routes to the Messages endpoint are the models the provider serves
   there — that oracle is the provider's published endpoint table, and it is
   stale today.  Tracked as **KBR-126**; see
   :func:`test_messages_routing_table_is_unchanged`.
5. **That a ``False`` body is well-formed Chat Completions.**  The declaration
   is a boolean, so the sweep asserts "Messages" against
   :attr:`WireShape.MESSAGES` and collapses ``CHAT_COMPLETIONS`` with ``OTHER``.
   A declared-``False`` adapter that began emitting a malformed body would
   still pass.  The failure direction is the safe one — a Messages body that
   degrades to ``OTHER`` turns a ``True``-declaring adapter red — but a green
   sweep is not a well-formedness check.

Every check below asserts its own subject set, in the style of
``tests/test_egress_coverage.py``, so none can rot into a no-op.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

import pytest

from kitty.providers.base import ProviderAdapter
from kitty.providers.opencode import _MESSAGES_MODELS
from kitty.providers.registry import _registry, get_provider


class WireShape(Enum):
    """The request dialects a provider adapter can put on the wire."""

    MESSAGES = "anthropic_messages"
    CHAT_COMPLETIONS = "chat_completions"
    OTHER = "other"


def classify_wire_shape(body: dict) -> WireShape:
    """Classify an upstream request body by its observable markers.

    Keys on **tool-entry shape plus system placement**, never on the presence of
    a top-level ``system`` key: Bedrock's Converse body also carries one and is
    not a Messages body.  Requiring both axes to agree means a body matching
    neither dialect — Converse, Responses — is reported as
    :attr:`WireShape.OTHER` rather than guessed at.

    The probe request must therefore carry a system message and a tool; see
    :func:`_probe_request`.  :func:`test_classifier_needs_the_tools_axis` pins
    that dependency.

    Args:
        body: The dict an adapter's ``translate_to_upstream`` returned.

    Returns:
        The dialect the body is written in.
    """
    tools = body.get("tools")
    first_tool = tools[0] if isinstance(tools, list) and tools and isinstance(tools[0], dict) else None
    roles = {m.get("role") for m in body.get("messages", []) if isinstance(m, dict)}

    # Anthropic hoists the system prompt to a top-level field and describes a
    # tool with `input_schema`; Chat Completions keeps a system turn and wraps
    # each tool in a `function` envelope.
    if first_tool is not None and "input_schema" in first_tool and "system" not in roles:
        return WireShape.MESSAGES
    if first_tool is not None and "function" in first_tool and "system" in roles:
        return WireShape.CHAT_COMPLETIONS
    return WireShape.OTHER


@dataclass(frozen=True)
class _Representation:
    """The models a provider is exercised with.

    Attributes:
        models: Every model class the adapter routes differently, plus enough
            ordinary cases to be meaningful.
        default_route_model: A model that takes the adapter's **default** route.
            This is what the bare property is measured against.  It is a field
            of its own and deliberately not ``validation_model``: that answers a
            different question (a model whose credential check will not be
            misread as an auth failure) and its value is scheduled to change.
    """

    models: tuple[str, ...]
    default_route_model: str


def _single_route(model: str = "kitty-test-model") -> _Representation:
    """Represent an adapter whose wire shape does not depend on the model."""
    return _Representation(models=(model,), default_route_model=model)


# Every registry key appears, so adding a provider forces a decision here.
REPRESENTATIVE_MODELS: dict[str, _Representation] = {
    "anthropic": _single_route(),
    "azure": _single_route(),
    "bedrock": _single_route(),
    "byteplus": _single_route(),
    "custom_anthropic": _single_route(),
    "custom_openai": _single_route(),
    "fireworks": _single_route(),
    "google_aistudio": _single_route(),
    "kimi": _single_route(),
    "mimo": _single_route(),
    "minimax": _single_route(),
    "minimax_token": _single_route(),
    "novita": _single_route(),
    "ollama": _single_route(),
    "ollama_cloud": _single_route(),
    "openai": _single_route(),
    "openai_subscription": _single_route(),
    # The one adapter that routes by model.  Both routes are represented, and
    # `test_a_routing_adapter_represents_both_of_its_routes` keeps it that way.
    "opencode_go": _Representation(
        models=("minimax-m2.5", "minimax-m2.7", "glm-5.2", "kimi-k2.7-code", "mimo-v2.5-pro", ""),
        default_route_model="glm-5.2",
    ),
    "openrouter": _single_route(),
    "vertex": _single_route(),
    "zai_coding": _single_route(),
    "zai_coding_cc": _single_route(),
    "zai_regular": _single_route(),
}

# The three adapters that bypass `_make_upstream_request`.  What the sweep
# proves for each differs — see non-claim 1 in the module docstring.  The set is
# asserted rather than narrated so a fourth one forces a decision.
CUSTOM_TRANSPORT_ADAPTERS = frozenset({"openai_subscription", "bedrock", "ollama_cloud"})


def _probe_request(model: str) -> dict:
    """Build the request every adapter is asked to translate.

    The system turn and the tool are load-bearing, not decoration:
    :func:`classify_wire_shape` reads both axes, and a probe missing either
    would make two dialects indistinguishable.
    """
    return {
        "model": model,
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "You are a reviewer."},
            {"role": "user", "content": "Review this diff"},
            {"role": "assistant", "content": "Reading the file."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def _provider_model_pairs() -> list[tuple[str, str]]:
    """Enumerate every (provider, model) case the sweep covers."""
    return [(name, model) for name, rep in sorted(REPRESENTATIVE_MODELS.items()) for model in rep.models]


# ── The sweep ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("provider_type", "model"), _provider_model_pairs())
def test_declared_wire_shape_matches_the_emitted_body(provider_type: str, model: str):
    """The per-model declaration agrees with what the adapter emits.

    This is the assertion that fails on the unfixed revision for
    ``opencode_go``'s Chat-Completions models.

    Note the declaration is a boolean, so a ``False`` here proves only "not a
    Messages body" — ``CHAT_COMPLETIONS`` and ``OTHER`` are not distinguished
    (non-claim 5).
    """
    adapter = get_provider(provider_type)
    body = adapter.translate_to_upstream(copy.deepcopy(_probe_request(model)))
    declared = adapter.upstream_wire_is_messages_api_for_model(model)

    assert declared is (classify_wire_shape(body) is WireShape.MESSAGES), (
        f"{provider_type} × {model!r} declares "
        f"{'Anthropic Messages' if declared else 'not Anthropic Messages'} "
        f"but emits {classify_wire_shape(body).value}"
    )


@pytest.mark.parametrize("provider_type", sorted(REPRESENTATIVE_MODELS))
def test_bare_property_matches_the_per_model_form_at_the_default_route_model(provider_type: str):
    """R6b — the bare property honestly reports the adapter's default route.

    Guards the half of KBR-7 the per-model form does not: without this,
    ``OpenCodeGoAdapter.upstream_wire_is_messages_api`` could drift back to
    ``True`` with every other test still green.
    """
    adapter = get_provider(provider_type)
    default_model = REPRESENTATIVE_MODELS[provider_type].default_route_model

    assert adapter.upstream_wire_is_messages_api is adapter.upstream_wire_is_messages_api_for_model(default_model)


# ── Guards on the guard ────────────────────────────────────────────────────


def test_every_registered_provider_is_represented():
    """R7b — a new provider cannot slip past the sweep."""
    assert set(REPRESENTATIVE_MODELS) == set(_registry)


def test_every_messages_route_model_is_represented():
    """R7c — a model added to the routing table cannot skip the sweep."""
    covered = set(REPRESENTATIVE_MODELS["opencode_go"].models)
    assert covered >= _MESSAGES_MODELS


def test_a_routing_adapter_represents_both_of_its_routes():
    """R7e — the representative table cannot be narrowed until it proves nothing.

    An adapter that overrides the per-model form has more than one route.  If
    its list covered only one of them, the sweep would go green while saying
    nothing about the other — the failure mode the sweep exists to catch.
    """
    checked = 0
    for provider_type, rep in sorted(REPRESENTATIVE_MODELS.items()):
        adapter = get_provider(provider_type)
        overrides = (
            type(adapter).upstream_wire_is_messages_api_for_model
            is not ProviderAdapter.upstream_wire_is_messages_api_for_model
        )
        if not overrides:
            continue
        declared = {adapter.upstream_wire_is_messages_api_for_model(m) for m in rep.models}
        assert declared == {True, False}, f"{provider_type} routes by model but only one route is represented"
        checked += 1

    # Without this the loop passes vacuously the moment no adapter overrides
    # the per-model form — the one check in this module that would otherwise
    # rot into the no-op its docstring promises it cannot become.
    assert checked, "no adapter overrides the per-model form — this check has nothing to guard"


def test_custom_transport_adapters_are_the_known_exempt_set():
    """R7d — a fourth custom-transport adapter forces a decision.

    For these three the hook body is not the shipped bytes — see non-claim 1
    in the module docstring for what a green result means for each.  Mechanized
    rather than narrated: a new one must be classified, not silently exempted.
    """
    observed = {name for name in _registry if get_provider(name).use_custom_transport}
    assert observed == CUSTOM_TRANSPORT_ADAPTERS


def test_messages_routing_table_is_unchanged():
    """Tripwire on ``_MESSAGES_MODELS`` — the routing table is stale (KBR-126).

    OpenCode Go serves eight models on ``/v1/messages`` today — ``minimax-m3``,
    ``minimax-m2.7``, ``minimax-m2.5``, ``qwen3.8-max``, ``qwen3.8-flash``,
    ``qwen3.7-max``, ``qwen3.7-plus`` and ``qwen3.6-plus`` — and four more on
    ``/v1/responses``, which this adapter has no route for at all.  kitty's
    table holds two of the eight.

    That is a **routing** defect and not KBR-7's — the declaration and the
    emitted body agree for those models, which is why the sweep above reports
    this adapter honest — so this asserts the table rather than correcting it.
    It fires the moment anyone edits the set, so KBR-126 has a red test waiting
    for it.
    """
    assert frozenset({"minimax-m2.5", "minimax-m2.7"}) == _MESSAGES_MODELS


# ── Known positives for the classifier ─────────────────────────────────────

_MESSAGES_BODY = {
    "model": "claude-opus-5",
    "max_tokens": 100,
    "system": "You are a reviewer.",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}],
}

_CHAT_COMPLETIONS_BODY = {
    "model": "gpt-4o",
    "max_tokens": 100,
    "messages": [{"role": "system", "content": "You are a reviewer."}, {"role": "user", "content": "hi"}],
    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
}

_CONVERSE_BODY = {
    "modelId": "anthropic.claude-v2",
    "system": [{"text": "You are a reviewer."}],
    "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    "inferenceConfig": {"maxTokens": 100},
    "toolConfig": {"tools": [{"toolSpec": {"name": "read_file", "inputSchema": {"json": {}}}}]},
}


def test_classifier_recognises_a_messages_body():
    assert classify_wire_shape(_MESSAGES_BODY) is WireShape.MESSAGES


def test_classifier_recognises_a_chat_completions_body():
    assert classify_wire_shape(_CHAT_COMPLETIONS_BODY) is WireShape.CHAT_COMPLETIONS


def test_classifier_reports_other_for_a_converse_body():
    """Converse is a third dialect, not a Messages body wearing a hat.

    It carries a top-level ``system`` exactly as Messages does, so a classifier
    keying on that alone would call Bedrock dishonest on every request.
    """
    assert classify_wire_shape(_CONVERSE_BODY) is WireShape.OTHER


def test_classifier_reports_other_for_a_converse_body_without_tools():
    """The tools axis must do the separating, not the system axis.

    Strip ``toolConfig`` and Converse and Messages agree on every other marker;
    only the absence of an ``input_schema`` tool keeps them apart.
    """
    body = {k: v for k, v in _CONVERSE_BODY.items() if k != "toolConfig"}
    assert classify_wire_shape(body) is WireShape.OTHER


def test_classifier_needs_the_tools_axis():
    """A probe without tools cannot be classified — hence R6's pinned shape.

    Asserted rather than assumed: if this ever returns ``MESSAGES``, the probe
    request may be weakened without anyone noticing the sweep went blind.
    """
    body = {k: v for k, v in _MESSAGES_BODY.items() if k != "tools"}
    assert classify_wire_shape(body) is WireShape.OTHER
