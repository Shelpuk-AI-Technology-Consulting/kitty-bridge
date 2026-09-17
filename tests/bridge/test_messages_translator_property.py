"""The Messages→Chat Completions translator's semantic property, via the projections.

`.system_design/TEST_SUITE.md` §3.3.1, §6.1 · plan task **T-F6** (KBR-75).

The property (§6.1, ``MessagesTranslator`` row):

    project_messages(inbound) == project_cc(translate_request(inbound))

compared **through the projections** — never through a translator round-trip.
§3.3.1 records why the round-trip framing is wrong twice: the request and
response translators are not inverses (composing them maps a conversation to an
assistant reply), and even a true inverse would prove only self-consistency. The
projections here are the independent oracle — :mod:`harness.reader_anthropic_messages`
and :mod:`harness.reader_chat_completions` import nothing from ``src/kitty``.

**Semantic fidelity, not semantic aptness.** Like the L3 oracle it anticipates
(§3.3.4), the property answers one question: did the translation change anything
the agent asked for. It does not judge whether a translation *should* have
mapped a field the way it did.

**The property's honest shape — coverage and three named exemptions.** The
shared ``messages_request()`` strategy exercises ``model``, ``max_tokens``,
``messages`` (text / image / document / ``tool_use`` / ``tool_result``
blocks), ``tools`` and ``stream``; it does not generate ``system``,
``tool_choice``, ``metadata``, ``stop_sequences``, ``top_k``, ``effort``,
``output_config``, ``thinking`` or block-level ``cache_control``, and the
property is silent on what it does not exercise. Over its space the
projections agree on ``envelope`` and ``conversation`` up to three
*deliberate* asymmetries, all applied in :func:`_conversations_equivalent`
and nowhere else:

1. **Documents (M2 / KBR-222 → P1).** A document block projects to
   ``Opaque(kind="document", …)`` inbound, while the translator routes
   documents to the ``_documents`` internal key for the Anthropic-family
   adapters to restore at the provider hop (P1). Their fidelity claim
   lives at the provider layer
   (``tests/providers/test_anthropic_image_document_restore.py``), not here.
2. **``ToolDecl.type`` (translated-route asymmetry).** The Anthropic reader
   fills ``type`` from the wire (absent here → ``None``); the CC reader
   hardcodes ``type="function"`` for every function-wrapped tool, and the
   translator wraps unconditionally. The comparison maps ``None →
   "function"`` on the inbound side and leaves every other spelling
   untouched. Closing the asymmetry in either reader is a reader-contract
   decision, not this test's.
3. **Text-block joining on text-only turns (M2 / KBR-222).** A turn whose
   parts are all ``Text`` blocks collapses to one ``"\n"``-joined string on
   the CC wire — the projection cannot preserve per-block presence, so the
   comparison reduces both sides to ``(role, joined_text)``. The same
   reduction applies to both sides, so the join character itself is the
   property's variable, not its constant.

**What the signature deliberately does NOT exempt.** ``Turn.role`` rides on
every turn's signature — a translator that flips a turn's role is a real
fidelity defect. Mixed turns (any non-``Text`` part beside a ``Text`` part)
compare the full part tuple in wire order — neither the translator nor
either reader reorders parts, so per-part equality catches any wire-order
mutation (``[Image, Text]`` → ``[Text, Image]`` upstream is visible, not
silent). These two defences each carry their own falsification (§1.4).

**The §1.4 harness rule.** The first working version of every harness ships
with a falsification case it must detect. Four are provided: one per arm of
the original semantic claim (a translator that drops ``stream`` fails the
envelope arm — §3.3.1's worked example of what projection totality exists
for; one that drops ``max_tokens`` fails the conversation arm), plus one per
round-2 follow-up defence (a translator that flips a text-only user turn's
role; one that hoists text parts ahead of non-text in a mixed turn). Each
falsification test also asserts the healthy translator passes its arm, so a
green falsification test is unambiguously detection and not a stray
regression.

**Layer.** No ``pytestmark`` — ``tests/bridge/`` takes the ``l1`` path default,
so ``pytest -m l1`` selects every test here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from harness import contract as c
from harness import transcripts as t
from harness.reader_anthropic_messages import AnthropicMessagesProjection
from harness.reader_chat_completions import ChatCompletionsProjection
from hypothesis import HealthCheck, assume, given, settings

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.providers.base import ProviderAdapter

#: Request-level internal keys stripped before the CC projection reads the
#: translator's output, mirroring ``_INTERNAL_KEYS``' wire-time strip. Derived
#: by subtraction, not hand-curated: a new registry entry joins the strip
#: automatically, and :func:`test_request_level_internal_keys_match_the_provider_registry`
#: keeps the derivation honest. ``base_url`` is F15 defence-in-depth and never
#: a translator output; ``_thinking_blocks`` rides on *message* dicts under
#: ``ProviderAdapter._INTERNAL_MESSAGE_KEYS``, which the top-level strip never
#: sees.
_REQUEST_LEVEL_INTERNAL_KEYS: frozenset[str] = (
    ProviderAdapter._INTERNAL_KEYS - {"base_url", "_thinking_blocks"}
)


def _project_messages(body: Mapping[str, Any]) -> c.Request:
    """Project an Anthropic Messages body through its reader.

    Args:
        body: The parsed inbound Messages request.

    Returns:
        The wire-independent projection, total over the body.
    """
    return AnthropicMessagesProjection().read_request(
        c.CapturedRequest(
            method="POST",
            scheme="http",
            host="oracle.invalid",
            path="/v1/messages",
            query="",
            body=json.dumps(body).encode("utf-8"),
        )
    )


def _project_chat_completions(cc_body: Mapping[str, Any]) -> c.Request:
    """Project the translator's CC body through its reader, after the wire-time strip.

    Args:
        cc_body: The intermediate body :meth:`MessagesTranslator.translate_request`
            returned. The request-level internal keys it may carry are stripped
            here, mirroring ``_INTERNAL_KEYS``: the projection reads what ships,
            not the intermediate.

    Returns:
        The wire-independent projection, total over the stripped body.
    """
    wire_body = {key: value for key, value in cc_body.items() if key not in _REQUEST_LEVEL_INTERNAL_KEYS}
    return ChatCompletionsProjection().read_request(
        c.CapturedRequest(
            method="POST",
            scheme="http",
            host="oracle.invalid",
            path="/v1/chat/completions",
            query="",
            body=json.dumps(wire_body).encode("utf-8"),
        )
    )


def _conversations_equivalent(inbound: c.Conversation, cc: c.Conversation) -> bool:
    """Compare two projected conversations under the property's three named exemptions.

    Every field must compare equal except the three deliberate asymmetries the
    module docstring records:

    * **Documents** (M2 / KBR-222 → P1): ``Opaque(kind="document", …)`` parts
      are dropped from the inbound turns — the CC side carries them on the
      ``_documents`` internal key this property strips, and their restore is
      asserted at the provider layer
      (``tests/providers/test_anthropic_image_document_restore.py``).
    * **``ToolDecl.type``** (translated-route asymmetry): the CC reader
      hardcodes ``"function"`` where the Anthropic reader reads the wire's
      ``type`` leaf — ``None`` whenever absent. The inbound side maps
      ``None → "function"`` before comparison; every other spelling passes
      through untouched.
    * **Text-block joining** (M2 / KBR-222): adjacent text blocks in a
      turn collapse to one ``"\n"``-joined string on the CC wire — the
      projection cannot preserve per-block presence. Each turn is reduced
      to ``(non-text parts, "\n".join(text of Text parts))`` and the two
      sides compare on that reduction, symmetrically — so the join character
      itself is not hard-coded by the property.

    Args:
        inbound: The Anthropic projection's conversation.
        cc: The Chat Completions projection's conversation.

    Returns:
        ``True`` when the two conversations carry the same semantic content
        under the exemptions.
    """
    # Exemption 1: documents drop on the inbound side.
    inbound_turns = tuple(
        c.Turn(
            role=turn.role,
            parts=tuple(
                part
                for part in turn.parts
                if not (isinstance(part, c.Opaque) and part.kind == "document")
            ),
        )
        for turn in inbound.turns
    )

    # Exemption 2: ToolDecl.type — normalise None -> "function" on inbound only.
    inbound_tools = tuple(
        c.ToolDecl(
            name=tool.name,
            description=tool.description,
            schema=tool.schema,
            strict=tool.strict,
            cache_control=tool.cache_control,
            type="function" if tool.type is None else tool.type,
        )
        for tool in inbound.tools
    )

    # Exemption 3: text-block joining — and the per-shape handling. The
    # strategy's text-only turns (one or two ``Text`` blocks) collapse to one
    # CC ``Text("\n".join(...))`` per M2 / KBR-222; the projection cannot
    # preserve per-block presence, so the comparison reduces both sides to
    # ``(role, joined_text)``. Mixed turns (text plus ``Image`` / ``ToolUse`` /
    # etc.) keep the full part tuple — neither the translator nor either
    # reader reorders parts, so per-part equality with ``role`` catches any
    # wire-order mutation a translator might introduce. ``Turn.role`` rides
    # on the signature because a role swap is a real fidelity defect the
    # M2 / KBR-222 join clause does not cover. The same reduction applies
    # to both sides, so the join character itself is the property's variable,
    # not its constant — a translator regression on the join char surfaces
    # as a known delta rather than a self-consistency pass.
    def _turn_signature(turn: c.Turn) -> tuple:
        non_text = tuple(part for part in turn.parts if not isinstance(part, c.Text))
        if not non_text:
            # Text-only (after document strip): the M2 / KBR-222 join covers
            # per-block presence; reduce to ``(role, joined_text)``.
            return (
                turn.role,
                "\n".join(part.text for part in turn.parts if isinstance(part, c.Text)),
            )
        # Mixed or pure non-text: full tuple equality catches wire-order
        # mutations a translator could introduce (``Text, Image`` → ``Image,
        # Text`` would not be a no-op semantic change).
        return (turn.role, tuple(turn.parts))

    in_signatures = [_turn_signature(turn) for turn in inbound_turns]
    cc_signatures = [_turn_signature(turn) for turn in cc.turns]

    return (
        inbound.system == cc.system
        and inbound.sampling == cc.sampling
        and inbound_tools == cc.tools
        and in_signatures == cc_signatures
    )


# ── Property (T1) ───────────────────────────────────────────────────────────


@given(t.messages_request())
@settings(max_examples=200)
def test_messages_translator_semantic_round_trip_via_projections(body: dict) -> None:
    """The translator preserves the projected ``envelope`` and ``conversation``.

    Property (T-F6 / §6.1, §3.3.1): ``project_messages(inbound)`` equals
    ``project_cc(translate_request(inbound))`` through the projections, never
    through the translator pair. ``envelope`` and ``conversation`` carry the
    semantic claim; residual / consumed / source are reader bookkeeping, and a
    non-empty residual on either side fails loudly via :func:`~harness.contract.verify_total`
    (§3.3.1's totality rule applied where it can actually fire — the CC side,
    where the internal-key strip lives).
    """
    p_in = _project_messages(body)
    c.verify_total(p_in)

    cc_body = MessagesTranslator().translate_request(body)
    p_cc = _project_chat_completions(cc_body)
    c.verify_total(p_cc)

    assert p_in.envelope == p_cc.envelope
    assert _conversations_equivalent(p_in.conversation, p_cc.conversation)


# ── Strip snapshot (T5) ──────────────────────────────────────────────────────


def test_request_level_internal_keys_match_the_provider_registry() -> None:
    """The strip set is the registry's request-level entries, hand-pinned here.

    AC-2: the expected set is written out literally on purpose — a re-derived
    comparison would be a tautology (the same expression on both sides can
    never fail) and would detect nothing. A pinned snapshot makes every
    ``ProviderAdapter._INTERNAL_KEYS`` change fail here first, so the
    contributor confirms the strip still mirrors the wire-time one before the
    property can mis-diagnose the drift as a residual. When this test fails,
    the fix is either (a) the registry grew a request-level key the translator
    can emit — add it to the strip by extending the subtraction's exclusion
    set only if the new key is genuinely not a translator output — or (b) the
    registry re-shaped and the two exclusions below need re-deriving.

    The two exclusions: ``base_url`` is F15 defence-in-depth and never a
    translator output; ``_thinking_blocks`` rides on *message* dicts under
    ``ProviderAdapter._INTERNAL_MESSAGE_KEYS``, which the top-level strip
    never sees.
    """
    assert frozenset(
        {
            "_anthropic_system",
            "_documents",
            "_effort",
            "_metadata",
            "_native_messages_request",
            "_original_body",
            "_output_config",
            "_provider_config",
            "_reasoning_effort",
            "_resolved_key",
            "_thinking_adaptive",
            "_thinking_budget_tokens",
            "_thinking_display",
            "_thinking_enabled",
            "_top_k",
        }
    ) == _REQUEST_LEVEL_INTERNAL_KEYS


# ── Falsification subclasses (§1.4) ─────────────────────────────────────────


class _DropsStream(MessagesTranslator):
    """Falsification translator: pops ``stream`` from the translated CC body.

    §3.3.1 names a dropped ``stream`` as the worked example of what projection
    totality exists for. A property blind to this would let an envelope
    mutation through silently, so the falsification proves the envelope arm
    *does* see it.
    """

    def translate_request(self, messages_request: dict) -> dict:
        """Translate and drop the wire-level ``stream`` flag."""
        result = super().translate_request(messages_request)
        result.pop("stream", None)
        return result


class _DropsMaxTokens(MessagesTranslator):
    """Falsification translator: pops ``max_tokens`` from the translated CC body.

    The mirror of :class:`_DropsStream` for the conversation arm: a translator
    that drops a sampling parameter must surface as a ``conversation.sampling``
    delta so the property's conversation arm fails. ``max_tokens`` is the
    sampling key the strategy always carries (P13 row, §3.2.1).
    """

    def translate_request(self, messages_request: dict) -> dict:
        """Translate and drop the wire-level ``max_tokens``."""
        result = super().translate_request(messages_request)
        result.pop("max_tokens", None)
        return result


class _SwapsTextOnlyUserRoles(MessagesTranslator):
    """Falsification translator: flips ``user`` → ``assistant`` on text-only messages.

    Round-2 review follow-up: the property's turn comparison used to reduce
    to ``(non-text parts, joined text)`` and silently dropped ``Turn.role``,
    so a translator that flipped a turn's role would pass. The signature now
    carries ``role``; this falsification proves that defence is wired.

    The mutation is deliberately narrow — only messages whose ``content`` is
    a plain string — because a broader swap (assistant → user on a body with
    ``tool_calls``) produces a body the CC reader rejects with
    ``UnreadableBodyError``, which fails the falsification with the wrong
    signal rather than the property's assertion. A silent role flip on a
    plain text turn is still a real fidelity defect the projection must see.
    """

    def translate_request(self, messages_request: dict) -> dict:
        """Translate and flip ``user`` → ``assistant`` on string-content messages."""
        result = super().translate_request(messages_request)
        for message in result.get("messages", []):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                message["role"] = "assistant"
        return result


class _HoistsTextPartsFirst(MessagesTranslator):
    """Falsification translator: hoists every ``text`` content part before non-text.

    Round-2 review follow-up: the property's turn comparison used to split
    text from non-text parts and compare the two buckets separately, so a
    translator that reordered a mixed turn to text-first would pass. Mixed
    turns now compare the full part tuple, and this falsification proves
    that defence is wired: a strategy-generated mixed user turn
    (``[Image, Text]`` or ``[Text, Image, Text]``) reordered upstream is a
    real semantic mutation the projection must see.
    """

    def translate_request(self, messages_request: dict) -> dict:
        """Translate and reorder ``content`` lists to put ``text`` parts first."""
        result = super().translate_request(messages_request)
        for message in result.get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            message["content"] = [
                *sorted(content, key=lambda part: 0 if part.get("type") == "text" else 1)
            ]
        return result


# ── Falsification: envelope arm (T2) ────────────────────────────────────────


@given(t.messages_request())
@settings(max_examples=200)
def test_property_catches_a_translator_that_drops_stream(body: dict) -> None:
    """The envelope arm fails loudly when the translator drops ``stream``.

    Falsification (T-F6 / §1.4): a translator that drops a top-level envelope
    field must not pass the projection-equality property. The healthy
    translator passes; the defective one fails on a delta the projection
    names — the absent ``stream`` key defaults to ``None`` through
    ``_typed_leaf``, against the inbound ``bool``, so the two ``Envelope``
    values differ.
    """
    p_in = _project_messages(body)
    healthy_envelope = _project_chat_completions(MessagesTranslator().translate_request(body)).envelope
    assert p_in.envelope == healthy_envelope

    dropped_envelope = _project_chat_completions(_DropsStream().translate_request(body)).envelope
    assert p_in.envelope != dropped_envelope, (
        "dropping stream did not surface as an envelope delta — the property "
        "would let a real-world regression of this shape through"
    )


# ── Falsification: conversation arm (T3) ────────────────────────────────────


@given(t.messages_request())
@settings(max_examples=200)
def test_property_catches_a_translator_that_drops_max_tokens(body: dict) -> None:
    """The conversation arm fails loudly when the translator drops ``max_tokens``.

    Falsification (T-F6 / §1.4): symmetric with :func:`test_property_catches_a_translator_that_drops_stream`
    for the conversation arm. The healthy translator passes; the defective one
    fails on a delta the projection names (``max_tokens`` lives in
    ``conversation.sampling``).
    """
    p_in = _project_messages(body)
    healthy_conversation = _project_chat_completions(
        MessagesTranslator().translate_request(body)
    ).conversation
    assert _conversations_equivalent(p_in.conversation, healthy_conversation)

    dropped_conversation = _project_chat_completions(_DropsMaxTokens().translate_request(body)).conversation
    assert not _conversations_equivalent(p_in.conversation, dropped_conversation), (
        "dropping max_tokens did not surface as a conversation delta — the "
        "property would let a real-world regression of this shape through"
    )


# ── Falsification: role preservation (round-2 W1 follow-up) ─────────────────


@given(t.messages_request())
@settings(max_examples=200)
def test_property_catches_a_translator_that_swaps_a_text_only_user_role(body: dict) -> None:
    """The conversation arm fails loudly when the translator flips a turn's role.

    Falsification (T-F6 / §1.4, round-2 review follow-up): the turn comparison
    now carries ``Turn.role``; this test proves that defence is wired. The
    healthy translator passes; the role-flipping translator fails on a delta
    the projection names — a turn the agent sent as ``user`` arrives
    upstream as ``assistant``.

    ``assume`` focuses the test on bodies where the mutation actually fires:
    the subclass only flips text-only user messages, so a body with none of
    them is unchanged by the defect and the comparison would be vacuous
    there — the same narrowing the healthy-pass + defective-fail pair uses
    everywhere else.
    """
    translated = MessagesTranslator().translate_request(body)
    assume(
        any(
            message.get("role") == "user" and isinstance(message.get("content"), str)
            for message in translated.get("messages", [])
        )
    )

    p_in = _project_messages(body)
    healthy_conversation = _project_chat_completions(translated).conversation
    assert _conversations_equivalent(p_in.conversation, healthy_conversation)

    swapped_conversation = _project_chat_completions(_SwapsTextOnlyUserRoles().translate_request(body)).conversation
    assert not _conversations_equivalent(p_in.conversation, swapped_conversation), (
        "flipping a text-only user turn's role did not surface as a "
        "conversation delta — the property would let a real-world regression "
        "of this shape through"
    )


# ── Falsification: part-order preservation (round-2 W2 follow-up) ──────────

# `filter_too_much` is suppressed deliberately. The hoist only changes mixed
# content lists, and the strategy's space generates them at roughly one body
# in eight — so ~88% of raw examples are filtered by the `assume` below.
# That is not a distortion: mixed turns are exactly the inputs where the
# hoist could fire, and every filtered body was a vacuous run regardless.
# The role-swap falsification above does NOT need the same suppression —
# text-only user turns are drawn at ~50% per user turn, comfortably inside
# Hypothesis's default health-check budget.
@given(t.messages_request())
@settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
def test_property_catches_a_translator_that_hoists_text_parts_first(body: dict) -> None:
    """The conversation arm fails loudly when the translator reorders mixed parts.

    Falsification (T-F6 / §1.4, round-2 review follow-up): mixed turns now
    compare the full part tuple in wire order; this test proves that defence
    is wired. The healthy translator passes; the hoisting translator fails
    on a delta the projection names — a mixed user turn (``[Image, Text]``
    or ``[Text, Image, Text]``) is reordered upstream to text-first.

    ``assume`` focuses the test on bodies where the mutation actually fires:
    the hoist only changes mixed ``content`` lists that are not already
    text-first, so a body where every mixed list already has its text parts
    up front is unchanged by the defect and the comparison would be vacuous
    there. The filter compares the healthy and hoisted bodies directly, so
    the test's precondition is "the hoist moved at least one part".
    """
    translated = MessagesTranslator().translate_request(body)
    hoisted_body = _HoistsTextPartsFirst().translate_request(body)
    assume(
        any(
            hoisted_message.get("content") != healthy_message.get("content")
            for hoisted_message, healthy_message in zip(
                hoisted_body.get("messages", []),
                translated.get("messages", []),
                strict=True,
            )
        )
    )

    p_in = _project_messages(body)
    healthy_conversation = _project_chat_completions(translated).conversation
    assert _conversations_equivalent(p_in.conversation, healthy_conversation)

    hoisted_conversation = _project_chat_completions(_HoistsTextPartsFirst().translate_request(body)).conversation
    assert not _conversations_equivalent(p_in.conversation, hoisted_conversation), (
        "hoisting text parts first did not surface as a conversation delta — "
        "the property would let a real-world regression of this shape through"
    )
