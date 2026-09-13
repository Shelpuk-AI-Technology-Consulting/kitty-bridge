"""What the translated Messages route ships for the agent's thinking and effort (KBR-203).

Anthropic renders the thinking configuration — its mode, ``budget_tokens`` in
extended mode, and the effort value — into the prompt.  Changing any of them
between two requests invalidates the message-level prompt cache, and on some
models the tool and system caches too (``prompt-caching.md``, "What invalidates
the cache", fetched 2026-09-13).  So whatever kitty does to these fields must at
least be **stable**: the same agent configuration must ship the same bytes on
every turn.

Every test here is driven from an agent's **Messages body** through
:meth:`~kitty.bridge.messages.translator.MessagesTranslator.translate_request`
and then the adapter's ``translate_to_upstream`` — the translated route as the
bridge runs it.  Starting from a hand-built Chat Completions dict would skip the
half of the route that decides what is carried at all.

Register rows: P5c (the budget rewrite) and P5d (adaptive thinking, effort and
``display``), ``.system_design/TEST_SUITE.md`` §3.2.2.  The native-passthrough
counterpart lives in ``tests/bridge/test_native_thinking_passthrough.py``.
"""

from __future__ import annotations

import pytest

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.opencode import OpenCodeGoAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter

#: The agent's own budget, deliberately far from any ``max_tokens - 1`` used below,
#: so a shipped value equal to it cannot be a coincidence of the derivation.
_AGENT_BUDGET = 2048


def _messages_body(*, max_tokens: int, thinking: dict | None = None, effort: str | None = None) -> dict:
    """Build a minimal Anthropic Messages body as Claude Code would send it.

    Args:
        max_tokens: The agent's ``max_tokens``.
        thinking: The agent's ``thinking`` object, or ``None`` to omit it.
        effort: The agent's top-level ``effort``, or ``None`` to omit it.

    Returns:
        A fresh Messages request body.
    """
    body: dict = {
        "model": "claude-opus-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if thinking is not None:
        body["thinking"] = thinking
    if effort is not None:
        body["effort"] = effort
    return body


def _ship(adapter: ProviderAdapter, body: dict, *, model: str | None = None) -> dict:
    """Run *body* through the translated route and return the upstream body.

    Args:
        adapter: The provider adapter that serialises the upstream body.
        body: The agent's Messages request body.  Not mutated.
        model: When given, overrides the translated request's model — needed for
            an adapter that routes by model.

    Returns:
        The body ``translate_to_upstream`` produces.
    """
    cc_request = MessagesTranslator().translate_request(dict(body))
    if model is not None:
        cc_request["model"] = model
    return adapter.translate_to_upstream(cc_request)


class TestEnabledThinkingBudgetIsDerivedNotForwarded:
    """P5c: the shipped budget is kitty's, computed from ``max_tokens``.

    These pin **today's defect** so that it is on the record and cannot change
    silently.  KBR-225 fixes it by forwarding the agent's own budget, and that
    change must invert these tests rather than delete them.
    """

    def test_budget_moves_with_max_tokens_although_the_agent_sent_one_budget(self):
        """R1 — the falsifiable core of KBR-203.

        Two requests identical except for ``max_tokens``, carrying the same
        agent budget, ship two different budgets.  Anthropic renders the budget
        into the prompt, so these two requests cannot share a cached prefix.
        """
        thinking = {"type": "enabled", "budget_tokens": _AGENT_BUDGET}

        first = _ship(AnthropicAdapter(), _messages_body(max_tokens=8000, thinking=thinking))
        second = _ship(AnthropicAdapter(), _messages_body(max_tokens=16000, thinking=thinking))

        assert first["thinking"]["budget_tokens"] == 7999
        assert second["thinking"]["budget_tokens"] == 15999

    @pytest.mark.parametrize(
        ("max_tokens", "shipped_max_tokens", "shipped_budget"),
        [(8000, 8000, 7999), (512, 1025, 1024)],
        ids=["derived", "clamped-to-anthropic-minimum"],
    )
    def test_shipped_thinking_is_computed_not_the_agents(self, max_tokens, shipped_max_tokens, shipped_budget):
        """R2 — the agent's own ``thinking`` object is not what ships.

        Args:
            max_tokens: The agent's ``max_tokens``.
            shipped_max_tokens: The ``max_tokens`` kitty ships after P5c's raise.
            shipped_budget: The budget kitty computes.
        """
        agent_thinking = {"type": "enabled", "budget_tokens": _AGENT_BUDGET}

        shipped = _ship(AnthropicAdapter(), _messages_body(max_tokens=max_tokens, thinking=agent_thinking))

        assert shipped["thinking"] != agent_thinking
        assert shipped["thinking"] == {"type": "enabled", "budget_tokens": shipped_budget}
        assert shipped["max_tokens"] == shipped_max_tokens


class TestAdaptiveThinkingAndEffortAreStable:
    """P5d: the paths that do not deserve P5c's finding, pinned so they keep not deserving it."""

    @pytest.mark.parametrize("max_tokens", [8000, 16000])
    def test_bare_adaptive_ships_verbatim_whatever_max_tokens(self, max_tokens):
        """R3 — ``{"type": "adaptive"}`` is forwarded as sent, and ``max_tokens`` is not raised.

        Parametrised over two ``max_tokens`` so the pair is the same comparison
        R1 makes for the enabled branch, with the opposite outcome.

        Args:
            max_tokens: The agent's ``max_tokens``.
        """
        shipped = _ship(AnthropicAdapter(), _messages_body(max_tokens=max_tokens, thinking={"type": "adaptive"}))

        assert shipped["thinking"] == {"type": "adaptive"}
        assert shipped["max_tokens"] == max_tokens

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_effort_ships_verbatim(self, effort):
        """R4 — top-level ``effort`` is forwarded as sent.

        Verbatim is the property that keeps it cache-safe: any normalisation
        that could map one agent value to another would change the rendered
        prompt the way P5c's rewrite does.

        Args:
            effort: One of the documented effort levels.
        """
        shipped = _ship(AnthropicAdapter(), _messages_body(max_tokens=8000, effort=effort))

        assert shipped["effort"] == effort


class TestTranslatorCarriesThinkingDisplay:
    """R6 — ``MessagesTranslator`` carries the agent's ``display`` on ``_thinking_display``."""

    @pytest.mark.parametrize(
        "thinking",
        [
            {"type": "adaptive", "display": "summarized"},
            {"type": "enabled", "budget_tokens": _AGENT_BUDGET, "display": "omitted"},
        ],
        ids=["adaptive", "enabled"],
    )
    def test_display_is_carried_for_modes_that_accept_it(self, thinking):
        """Both modes that Anthropic lets carry ``display`` keep the agent's value.

        Args:
            thinking: The agent's thinking object, carrying a ``display``.
        """
        cc_request = MessagesTranslator().translate_request(_messages_body(max_tokens=8000, thinking=thinking))

        assert cc_request["_thinking_display"] == thinking["display"]

    @pytest.mark.parametrize(
        "thinking",
        [{"type": "adaptive"}, {"type": "disabled", "display": "summarized"}],
        ids=["absent", "disabled-mode"],
    )
    def test_display_is_not_carried_when_absent_or_invalid(self, thinking):
        """Nothing is invented, and ``display`` with ``disabled`` — which Anthropic rejects — is not carried.

        Args:
            thinking: An agent thinking object that must not yield the key.
        """
        cc_request = MessagesTranslator().translate_request(_messages_body(max_tokens=8000, thinking=thinking))

        assert "_thinking_display" not in cc_request


_WITH_DISPLAY = [
    {"type": "adaptive", "display": "summarized"},
    {"type": "enabled", "budget_tokens": _AGENT_BUDGET, "display": "summarized"},
]


class TestThinkingDisplayRestoredOnlyWhereTheUpstreamDocumentsIt:
    """R8 — ``display`` is restored on Anthropic's own wire, and withheld where it is undocumented.

    Dropping ``"summarized"`` hides every thinking token from the user on models
    whose default is ``"omitted"``.  Restoring it on MiniMax, whose
    Anthropic-compatible reference does not name the field, risks a 400 on every
    thinking request — so MiniMax-backed routes keep today's behaviour (decision
    D2, KBR-203).
    """

    @pytest.mark.parametrize("thinking", _WITH_DISPLAY, ids=["adaptive", "enabled"])
    @pytest.mark.parametrize("adapter", [AnthropicAdapter(), CustomAnthropicAdapter()], ids=lambda a: a.provider_type)
    def test_display_is_restored(self, adapter, thinking):
        """The agent's ``display`` ships on both thinking branches.

        Args:
            adapter: An adapter whose upstream is Anthropic's Messages API.
            thinking: The agent's thinking object, carrying a ``display``.
        """
        shipped = _ship(adapter, _messages_body(max_tokens=8000, thinking=thinking))

        assert shipped["thinking"]["display"] == "summarized"
        assert shipped["thinking"]["type"] == thinking["type"]

    @pytest.mark.parametrize("thinking", _WITH_DISPLAY, ids=["adaptive", "enabled"])
    @pytest.mark.parametrize(
        ("adapter", "model"),
        [
            (MiniMaxTokenAnthropicAdapter(), None),
            (OpenCodeGoAdapter(), "minimax-m2.7"),
            (ZaiAnthropicAdapter(), None),
        ],
        ids=["minimax_token", "opencode_go", "zai_coding"],
    )
    def test_display_is_withheld(self, adapter, model, thinking):
        """The field stays off upstreams that do not document it.

        Args:
            adapter: An Anthropic-family adapter whose upstream is not Anthropic.
            model: A Messages-routed model for an adapter that routes by model.
            thinking: The agent's thinking object, carrying a ``display``.
        """
        shipped = _ship(adapter, _messages_body(max_tokens=8000, thinking=thinking), model=model)

        assert shipped["thinking"]["type"] == thinking["type"]
        assert "display" not in shipped["thinking"]

    def test_the_internal_key_never_ships(self):
        """``_thinking_display`` itself is stripped on a route that does not read it."""
        body = _messages_body(max_tokens=8000, thinking=_WITH_DISPLAY[0])

        shipped = _ship(OpenCodeGoAdapter(), body, model="glm-5.1")

        assert "_thinking_display" not in shipped
