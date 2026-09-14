"""What the translated Messages route ships for the agent's ``output_config`` (KBR-224).

``output_config`` is Anthropic's documented spelling of the effort control and
the home of structured output (``format``).  Until KBR-224 the translated route
dropped it entirely, so an agent asking for ``output_config.effort: "low"``
silently got the model's default.

Every test here is driven from an agent's **Messages body** through
:meth:`~kitty.bridge.messages.translator.MessagesTranslator.translate_request`
and then the adapter's ``translate_to_upstream`` — the translated route as the
bridge runs it, the same harness
``tests/providers/test_anthropic_thinking_cache_stability.py`` uses.  This
matrix is the serialization-boundary coverage, so R5-style strip cases
(``tests/test_internal_keys_not_sent_upstream.py``) are not duplicated: for
the rebuild adapters, ``translate_to_upstream`` *is* the boundary, and the
strip is implicit in the rebuild.

The field is carried verbatim: both published members are optional and nullable
in Anthropic's GA schema (SDK ``OutputConfig``, retrieved 2026-09-13), the field
needs no beta header, and an undocumented member is the agent's mistake, which
the upstream that documents the field reports better than kitty could.  The
restore is scoped per destination (KBR-203 decision-D2's pattern): MiniMax's
Anthropic-compatible endpoint rejects bodies carrying ``output_config``
(``minimax_token.py`` module docstring), and OpenCode's Messages route (MiniMax
and Qwen models) and Z.AI's endpoint do not document the field.
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

#: A value with structure in it, so a filtered or rebuilt copy cannot pass for
#: the verbatim carry the route promises.
_AGENT_OUTPUT_CONFIG = {
    "effort": "low",
    "format": {"type": "json_schema", "schema": {"type": "object"}},
}


def _messages_body(*, max_tokens: int = 8000, output_config: dict | None = None, effort: str | None = None) -> dict:
    """Build a minimal Anthropic Messages body as Claude Code would send it.

    Args:
        max_tokens: The agent's ``max_tokens``.
        output_config: The agent's ``output_config`` object, or ``None`` to omit
            it (or to send it as an explicit null — both must end key-absent).
        effort: The agent's top-level ``effort``, or ``None`` to omit it.

    Returns:
        A fresh Messages request body.
    """
    body: dict = {
        "model": "claude-opus-4-6",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if output_config is not None:
        body["output_config"] = output_config
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


class TestTranslatorCarriesOutputConfig:
    """Hop 1 — ``MessagesTranslator`` hands the field to the adapters on ``_output_config``."""

    def test_the_field_is_carried_verbatim(self):
        """The agent's object, structure included, reaches the internal key unchanged."""
        cc_request = MessagesTranslator().translate_request(_messages_body(output_config=_AGENT_OUTPUT_CONFIG))

        assert cc_request["_output_config"] == _AGENT_OUTPUT_CONFIG

    def test_no_field_is_invented_when_the_agent_sent_none(self):
        """A body without ``output_config`` gains no internal key."""
        cc_request = MessagesTranslator().translate_request(_messages_body())

        assert "_output_config" not in cc_request

    def test_an_explicit_null_is_omitted_not_carried(self):
        """``output_config: null`` means the same as absence, so it is not carried.

        The ``_metadata`` precedent (KBR-214): a nullable field is carried only
        when a value is present, so the adapter cannot emit a key whose value
        adds nothing.
        """
        cc_request = MessagesTranslator().translate_request(_messages_body(output_config=None))

        assert "_output_config" not in cc_request


class TestOutputConfigRestoredOnlyWhereTheUpstreamDocumentsIt:
    """Hop 2 — the field ships on Anthropic's own wire, and is withheld where it is undocumented.

    Restoring the field keeps the request faithful to what the agent sent.
    Restoring it on MiniMax, whose Anthropic-compatible endpoint rejects bodies
    carrying ``output_config``, would turn a request that works today into a
    400 — so MiniMax-backed routes keep today's behaviour, as does OpenCode's
    Messages route (MiniMax and Qwen models) and Z.AI's (decision D2's pattern,
    KBR-203).
    """

    @pytest.mark.parametrize("adapter", [AnthropicAdapter(), CustomAnthropicAdapter()], ids=lambda a: a.provider_type)
    def test_output_config_is_restored(self, adapter):
        """The agent's object ships verbatim on the upstreams that document the field.

        Args:
            adapter: An adapter whose upstream is Anthropic's Messages API.
        """
        shipped = _ship(adapter, _messages_body(output_config=_AGENT_OUTPUT_CONFIG))

        assert shipped["output_config"] == _AGENT_OUTPUT_CONFIG

    @pytest.mark.parametrize(
        ("adapter", "model"),
        [
            (MiniMaxTokenAnthropicAdapter(), None),
            (OpenCodeGoAdapter(), "minimax-m2.7"),
            (ZaiAnthropicAdapter(), None),
        ],
        ids=["minimax_token", "opencode_go", "zai_coding"],
    )
    def test_output_config_is_withheld(self, adapter, model):
        """The field stays off upstreams that reject or do not document it.

        Args:
            adapter: An Anthropic-family adapter whose upstream is not Anthropic.
            model: A Messages-routed model for an adapter that routes by model.
        """
        shipped = _ship(adapter, _messages_body(output_config=_AGENT_OUTPUT_CONFIG), model=model)

        assert "output_config" not in shipped

    @pytest.mark.parametrize(
        "adapter",
        [
            AnthropicAdapter(),
            CustomAnthropicAdapter(),
            MiniMaxTokenAnthropicAdapter(),
            ZaiAnthropicAdapter(),
        ],
        ids=lambda a: a.provider_type,
    )
    def test_no_field_is_invented_when_the_agent_sent_none(self, adapter):
        """A body without ``output_config`` ships without the key, on every adapter.

        Args:
            adapter: An Anthropic-family adapter under test.
        """
        shipped = _ship(adapter, _messages_body())

        assert "output_config" not in shipped

    def test_no_field_is_invented_on_the_messages_route_of_the_routing_adapter(self):
        """Same as the sweep above, for the one adapter that routes by model.

        ``OpenCodeGoAdapter`` needs a Messages-routed model to reach the branch
        this ticket touches, so it cannot join the parametrised sweep.
        """
        shipped = _ship(OpenCodeGoAdapter(), _messages_body(), model="minimax-m2.7")

        assert "output_config" not in shipped


class TestOutputConfigAndTopLevelEffortAreIndependent:
    """The documented and undocumented spellings ride side by side, unmerged.

    Top-level ``effort`` is what Claude Code sends today (KBR-168, P5d);
    ``output_config.effort`` is the documented spelling.  Kitty carries each
    verbatim and has no authority to pick a winner when an agent sends both.
    """

    def test_both_spellings_ship_together(self):
        """A body carrying both spellings ships both, each unmodified."""
        shipped = _ship(AnthropicAdapter(), _messages_body(output_config={"effort": "low"}, effort="high"))

        assert shipped["effort"] == "high"
        assert shipped["output_config"] == {"effort": "low"}

    @pytest.mark.parametrize(
        ("adapter", "model"),
        [
            (MiniMaxTokenAnthropicAdapter(), None),
            (OpenCodeGoAdapter(), "minimax-m2.7"),
            (ZaiAnthropicAdapter(), None),
        ],
        ids=["minimax_token", "opencode_go", "zai_coding"],
    )
    def test_effort_ships_but_output_config_is_withheld(self, adapter, model):
        """The asymmetric design, in one case: the undocumented spelling rides, the documented one is withheld.

        On these upstreams top-level ``effort`` is restored (P5d, unchanged by
        KBR-224) while ``output_config`` is withheld — the agent's effort choice
        keeps reaching them through the spelling they have always accepted, and
        a regression in either direction (restoring ``output_config`` here, or
        dropping ``effort``) trips this one named case.

        Args:
            adapter: An Anthropic-family adapter that withholds ``output_config``.
            model: A Messages-routed model for an adapter that routes by model.
        """
        shipped = _ship(adapter, _messages_body(output_config={"effort": "low"}, effort="high"), model=model)

        assert shipped.get("effort") == "high"
        assert "output_config" not in shipped
