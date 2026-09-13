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

#: The agent's own budget, deliberately far from any ``max_tokens - 1`` used
#: below, so a shipped value equal to it cannot be a coincidence of the
#: derivation — the one deliberate exception being the floor-pin case, whose
#: ``max_tokens = budget + 1`` makes the old derivation agree with the agent.
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


class TestEnabledThinkingBudgetIsForwarded:
    """P5c, after KBR-225: the shipped budget is the agent's own.

    KBR-203 pinned the old defect here (the shipped budget was kitty's,
    computed from ``max_tokens``); KBR-225 inverts those pins rather than
    deleting them.  A valid agent budget — ``>= 1024`` and ``< max_tokens`` —
    ships verbatim, so two requests differing only in ``max_tokens`` share one
    thinking configuration and can share a cached prefix.  An invalid budget
    keeps the old derivation as the documented fallback.
    """

    def test_two_requests_differing_only_in_max_tokens_ship_one_identical_thinking(self):
        """R1 — the falsifiable core of KBR-225.

        Two requests identical except for ``max_tokens``, carrying the same
        agent budget, ship the same thinking configuration — the agent's own.
        Anthropic renders the budget into the prompt, so this is what lets the
        two requests share a cached prefix.

        """
        thinking = {"type": "enabled", "budget_tokens": _AGENT_BUDGET}

        first = _ship(AnthropicAdapter(), _messages_body(max_tokens=8000, thinking=thinking))
        second = _ship(AnthropicAdapter(), _messages_body(max_tokens=16000, thinking=thinking))

        assert first["thinking"] == thinking
        assert second["thinking"] == first["thinking"]
        assert first["max_tokens"] == 8000
        assert second["max_tokens"] == 16000

    @pytest.mark.parametrize(
        ("max_tokens", "shipped_max_tokens", "shipped_budget"),
        [
            # Discriminating case: the old derivation would ship 2999, so this
            # one is red until the agent's budget is forwarded.
            (3000, 3000, _AGENT_BUDGET),
            # Floor pin: ``max_tokens = budget + 1`` is the one input where the
            # old derivation already agreed with the agent (max(2049, 1025) - 1
            # == 2048).  It falsifies nothing; it characterises that the change
            # does not disturb the boundary.
            (2049, 2049, _AGENT_BUDGET),
            # Invalid budget (2048 >= 512): the derivation is kept as the
            # fallback, clamped to Anthropic's minimum.
            (512, 1025, 1024),
        ],
        ids=["forwarded", "floor-pin", "invalid-budget-falls-back-clamped"],
    )
    def test_shipped_budget_is_the_agents_when_valid_and_derived_when_not(
        self, max_tokens, shipped_max_tokens, shipped_budget
    ):
        """The shipped budget is the agent's own when valid, the derivation when not.

        Args:
            max_tokens: The agent's ``max_tokens``.
            shipped_max_tokens: The ``max_tokens`` kitty ships — as sent when
                the budget is valid (a valid budget implies ``max_tokens >=
                1025``, so P5c's old raise cannot trigger), clamped on the
                fallback.
            shipped_budget: The budget kitty ships — the agent's, or the
                derivation when the agent's is invalid.
        """
        agent_thinking = {"type": "enabled", "budget_tokens": _AGENT_BUDGET}

        shipped = _ship(AnthropicAdapter(), _messages_body(max_tokens=max_tokens, thinking=agent_thinking))

        assert shipped["thinking"] == {"type": "enabled", "budget_tokens": shipped_budget}
        assert shipped["max_tokens"] == shipped_max_tokens


class TestAdapterForwardsTheCarriedBudget:
    """Hand-built Chat Completions cases isolating the adapter hop.

    The through-the-route tests above cannot say *which* hop regressed when
    they go red.  These two pin the adapter's contract on its own: forward the
    carried key verbatim, fall back to the derivation when it is absent — the
    same house style as ``test_display_is_never_added_to_disabled_thinking``.
    """

    def test_carried_budget_is_forwarded_and_max_tokens_untouched(self):
        """A present ``_thinking_budget_tokens`` ships verbatim; ``max_tokens`` is not raised."""
        cc_request = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3000,
            "_thinking_enabled": True,
            "_thinking_budget_tokens": _AGENT_BUDGET,
        }

        shipped = AnthropicAdapter().translate_to_upstream(cc_request)

        assert shipped["thinking"] == {"type": "enabled", "budget_tokens": _AGENT_BUDGET}
        assert shipped["max_tokens"] == 3000

    def test_absent_budget_falls_back_to_the_derivation(self):
        """Without the key, the old derivation runs unchanged."""
        cc_request = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3000,
            "_thinking_enabled": True,
        }

        shipped = AnthropicAdapter().translate_to_upstream(cc_request)

        assert shipped["thinking"] == {"type": "enabled", "budget_tokens": 2999}
        assert shipped["max_tokens"] == 3000


class TestTranslatorCarriesThinkingBudget:
    """The translator carries the agent's ``budget_tokens`` on ``_thinking_budget_tokens``.

    The value is checked here, at the only write site, so the adapter can trust
    the key — the same division of labour KBR-203 established for
    ``_thinking_display``.
    """

    def test_budget_is_carried_when_valid(self):
        """A valid budget rides the internal key."""
        cc_request = MessagesTranslator().translate_request(
            _messages_body(max_tokens=8000, thinking={"type": "enabled", "budget_tokens": _AGENT_BUDGET})
        )

        assert cc_request["_thinking_budget_tokens"] == _AGENT_BUDGET

    @pytest.mark.parametrize(
        ("max_tokens", "thinking"),
        [
            (8000, {"type": "enabled", "budget_tokens": 1023}),
            (8000, {"type": "enabled", "budget_tokens": 8000}),
            (8000, {"type": "enabled", "budget_tokens": 9000}),
            (8000, {"type": "enabled", "budget_tokens": "2048"}),
            (8000, {"type": "enabled", "budget_tokens": True}),
            (8000, {"type": "enabled"}),
            (None, {"type": "enabled", "budget_tokens": _AGENT_BUDGET}),
            ("8000", {"type": "enabled", "budget_tokens": _AGENT_BUDGET}),
        ],
        ids=[
            "below-floor",
            "equals-max-tokens",
            "above-max-tokens",
            "string-budget",
            "bool-budget",
            "absent-budget",
            "max-tokens-absent",
            "max-tokens-not-an-int",
        ],
    )
    def test_budget_is_not_carried_when_absent_or_invalid(self, max_tokens, thinking):
        """Only an int budget strictly between 1023 and ``max_tokens`` is carried.

        A non-int ``max_tokens`` is treated as absent: comparing against one
        would move the malformed-input TypeError from the Anthropic-family
        adapter into the shared translator, crashing every provider.  The bool
        case cannot fail on its own (every bool is 0 or 1, under the 1024
        floor); it documents the outcome.

        Args:
            max_tokens: The agent's ``max_tokens``; ``None`` means the key is
                omitted from the body entirely, a string exercises the non-int
                guard.
            thinking: An agent thinking object that must not yield the key.
        """
        body = _messages_body(max_tokens=8000, thinking=thinking)
        if max_tokens is None:
            del body["max_tokens"]
        else:
            body["max_tokens"] = max_tokens

        cc_request = MessagesTranslator().translate_request(body)

        assert "_thinking_budget_tokens" not in cc_request


class TestAdaptiveThinkingAndEffortAreStable:
    """P5d: the paths that do not deserve P5c's finding, pinned so they keep not deserving it."""

    @pytest.mark.parametrize("max_tokens", [8000, 16000])
    def test_bare_adaptive_ships_verbatim_whatever_max_tokens(self, max_tokens):
        """R3 — ``{"type": "adaptive"}`` is forwarded as sent, and ``max_tokens`` is not raised.

        Parametrised over two ``max_tokens`` so the pair is the same comparison
        R1 makes for the enabled branch, with the opposite outcome.  The claim is
        about the shipped thinking *configuration* only; the assistant turns on
        this route change for other reasons (P5e).

        Args:
            max_tokens: The agent's ``max_tokens``.
        """
        shipped = _ship(AnthropicAdapter(), _messages_body(max_tokens=max_tokens, thinking={"type": "adaptive"}))

        assert shipped["thinking"] == {"type": "adaptive"}
        assert shipped["max_tokens"] == max_tokens

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_effort_ships_verbatim(self, effort):
        """R4 — the top-level ``effort`` key kitty copies (P5d) is forwarded as sent.

        That key is not in Anthropic's API reference: it is what Claude Code
        sends (TEST_SUITE.md §7.4.1).  The documented spelling is
        ``output_config.effort``, which this route drops (KBR-224), and the
        values here are borrowed from that field's enum.  So this pins
        verbatim copying and nothing more — it makes no claim about caching.

        Args:
            effort: A value from ``output_config.effort``'s enum.
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
        [
            {"type": "adaptive"},
            {"type": "disabled", "display": "summarized"},
            {"type": "adaptive", "display": "updates"},
            {"type": "enabled", "budget_tokens": _AGENT_BUDGET, "display": "verbose"},
            {"type": "adaptive", "display": {"mode": "summarized"}},
        ],
        ids=["absent", "disabled-mode", "beta-updates", "unknown-value", "non-string"],
    )
    def test_display_is_not_carried_when_absent_or_invalid(self, thinking):
        """Only a GA value on a mode that accepts it is carried.

        Nothing is invented when the agent sent none.  ``disabled`` rejects
        ``display`` outright.  The beta ``"updates"`` needs an ``anthropic-beta``
        header kitty never forwards, so carrying it would turn a request that
        works today into a 400; an unknown or malformed value would do the same.

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

    Restoring ``display`` keeps the request faithful to what the agent sent — the
    translated route does not yet return thinking to the user (KBR-227,
    KBR-228).  Restoring it on MiniMax, whose
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

    def test_display_is_never_added_to_disabled_thinking(self):
        """The adapter refuses ``display`` on ``disabled``, whoever wrote the key.

        Anthropic rejects ``display`` alongside ``disabled``, so this branch must
        hold on its own.  Checking the *value* is the translator's job.
        """
        cc_request = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "_thinking_enabled": False,
            "_thinking_display": "summarized",
        }

        shipped = AnthropicAdapter().translate_to_upstream(cc_request)

        assert shipped["thinking"] == {"type": "disabled"}
