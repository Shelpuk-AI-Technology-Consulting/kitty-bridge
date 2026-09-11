"""Contract guard — every adapter's ``validation_model`` must be reachable by the key-check ping.

``.system_design/TEST_SUITE.md`` §6.2.3, "Register and docs ⇄ code".  Generalises
the half of **KBR-126** that was not about OpenCode Go: ``validation_model`` had
rotted to a model the provider no longer served, and the field exists precisely
so that a key check is never misread as an auth failure.

**What the ping actually is.**
:func:`kitty.validation.validate_api_key` posts
``{"model", "messages", "max_tokens", "stream"}`` with the **bare**
``build_upstream_headers`` to
``get_upstream_path(normalize_model_name(validation_model))``.  That body is a
coincidence worth writing down, because the whole guard rests on it: **it is
simultaneously a valid Chat Completions request and a valid Anthropic Messages
request.**  Four registered adapters — ``anthropic``, ``custom_anthropic``,
``minimax_token`` and ``zai_coding`` — validate against ``/v1/messages`` and work
for exactly that reason, so "``validation_model`` must be Chat-Completions-routed"
is *false* as a general rule and must not be asserted.

**What is actually required**, and what this module checks, is that the path and
the bare headers agree on a dialect the ping is written in:

* a Chat Completions path — any path ending ``/chat/completions``, query string
  permitted, since ``azure`` carries an ``api-version`` — is always fine; or
* ``/v1/messages``, but **only** if the bare headers are Anthropic-shaped.

The second clause is the one that catches the real defect.  Pointing
``opencode_go``'s ``validation_model`` at a Messages-routed model — the fix the
KBR-126 ticket originally suggested — leaves ``build_upstream_headers``
returning ``Bearer``, so the ping would post to ``/v1/messages`` with the wrong
auth and fail every key check.  A path-only check would pass it.

Anything else — an OpenAI Responses path, a Gemini ``:generateContent`` path —
fails, because the ping body is not written in those dialects at all.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from kitty.providers.registry import _registry, get_provider

# L2: a contract between `kitty.validation`'s request shape and each adapter's
# routing, two things edited independently.
pytestmark = pytest.mark.l2

CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
MESSAGES_PATH = "/v1/messages"

# Headers that mark a request as Anthropic-authenticated. Lower-cased at the
# call site: adapters disagree on casing (`content-type` vs `Content-Type`).
_ANTHROPIC_HEADERS = frozenset({"x-api-key", "anthropic-version"})

# Adapters `validate_api_key` returns early for, before building any request.
# Asserted against the live registry below rather than trusted, so a fourth one
# must be classified instead of silently inheriting an exemption.
CUSTOM_TRANSPORT_ADAPTERS = frozenset({"openai_subscription", "bedrock", "ollama_cloud"})


def validation_ping_problem(path: str, header_names: set[str]) -> str | None:
    """Return why the key-check ping would not be understood, or ``None``.

    Pure, so the negative tests below can hand it a deliberate defect and ask
    whether it notices — the rule ``tests/layers.py`` states.

    Args:
        path: The upstream path ``validate_api_key`` would post to.
        header_names: The header names the adapter's bare
            ``build_upstream_headers`` produces, lower-cased.

    Returns:
        A human-readable problem, or ``None`` when the ping is coherent.
    """
    # The query string is part of the route, not the dialect: `azure` appends
    # `?api-version=...` and still speaks Chat Completions.
    bare = urlsplit(path).path

    if bare.endswith(CHAT_COMPLETIONS_SUFFIX):
        return None

    if bare == MESSAGES_PATH:
        # The half a path-only check misses, and the half that matters: the ping
        # body is valid at /v1/messages, but only if the auth is too.
        if header_names & _ANTHROPIC_HEADERS:
            return None
        return (
            f"validation_model routes to {path} but build_upstream_headers is not "
            f"Anthropic-shaped (got {sorted(header_names)}) — the ping would carry the wrong auth"
        )

    return f"validation_model routes to {path}, a dialect the key-check ping is not written in"


def _validated_adapters() -> list[str]:
    """Return the adapters ``validate_api_key`` actually builds a request for."""
    return sorted(name for name in _registry if not get_provider(name).use_custom_transport)


@pytest.mark.parametrize("provider_type", _validated_adapters())
def test_the_key_check_ping_reaches_an_endpoint_that_understands_it(provider_type: str):
    """R16 — the path and the bare headers agree on a dialect the ping is written in."""
    adapter = get_provider(provider_type)
    path = adapter.get_upstream_path(adapter.normalize_model_name(adapter.validation_model))
    headers = {name.lower() for name in adapter.build_upstream_headers("sk-test")}

    problem = validation_ping_problem(path, headers)

    assert problem is None, f"{provider_type}: {problem}"


# ── Guards on the guard ────────────────────────────────────────────────────


def test_the_sweep_covers_most_of_the_registry():
    """The sweep must not shrink to nothing as adapters gain custom transports."""
    validated = _validated_adapters()

    assert len(validated) >= len(_registry) - len(CUSTOM_TRANSPORT_ADAPTERS)
    assert validated


def test_the_skipped_adapters_are_the_known_custom_transport_set():
    """A fourth custom-transport adapter must force a decision, not inherit a pass."""
    observed = {name for name in _registry if get_provider(name).use_custom_transport}

    assert observed == CUSTOM_TRANSPORT_ADAPTERS


def test_both_accepted_dialects_are_actually_exercised():
    """Neither branch may rot unused.

    Four adapters validate against ``/v1/messages`` today.  If that ever drops
    to zero, the Anthropic branch is dead code and this module's central claim —
    that the ping body is valid in two dialects — has stopped being tested.
    """
    paths = [
        urlsplit(
            get_provider(n).get_upstream_path(get_provider(n).normalize_model_name(get_provider(n).validation_model))
        ).path
        for n in _validated_adapters()
    ]

    assert any(p.endswith(CHAT_COMPLETIONS_SUFFIX) for p in paths)
    assert any(p == MESSAGES_PATH for p in paths)


def test_a_messages_route_with_bearer_auth_is_rejected():
    """The defect the KBR-126 ticket's own suggested fix would have introduced.

    The ticket proposed ``minimax-m2.7`` as OpenCode Go's replacement
    ``validation_model``.  It is Messages-routed, while that adapter's bare
    headers are ``Bearer`` — so every ``kitty auth`` check would have posted the
    wrong auth to ``/v1/messages`` and reported a bogus failure.
    """
    assert validation_ping_problem(MESSAGES_PATH, {"authorization", "content-type"}) is not None


def test_a_messages_route_with_anthropic_auth_is_accepted():
    """The complement, and the reason the rule is not "must be CC-routed".

    Four registered adapters are in exactly this state and work correctly.
    Without this case the guard could tighten to a path-only rule and turn
    them all red.
    """
    assert validation_ping_problem(MESSAGES_PATH, {"x-api-key", "anthropic-version"}) is None


def test_a_chat_completions_route_with_a_query_string_is_accepted():
    """Azure appends ``?api-version=``; the query is routing, not dialect."""
    assert validation_ping_problem("/openai/deployments/d/chat/completions?api-version=2024-10-21", {"api-key"}) is None


@pytest.mark.parametrize("path", ["/v1/responses", "/v1beta/models/gemini:generateContent", "/api/chat"])
def test_a_dialect_the_ping_is_not_written_in_is_rejected(path: str):
    """The ping body has no ``input`` and no ``contents`` — these cannot work."""
    assert validation_ping_problem(path, {"authorization"}) is not None
