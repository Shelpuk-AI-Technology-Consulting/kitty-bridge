"""The Permitted-Mutation Register, as data.

``.system_design/TEST_SUITE.md`` §3.2 · plan task **T-W3** (KBR-26).

Invariant I1 says the bridge forwards the agent's message content unchanged
*except* for mutations named on this register, each firing only under its stated
trigger.  §3.2 publishes the register as two markdown tables.  This module
publishes the same rows as data, because three consumers need to read them and a
prose table can only be read by a human:

* **T-D1**, the transparency oracle, which §3.3 says "takes ``register`` and
  ``triggers_met`` as arguments" and classifies every projected delta against
  them;
* **T-G2 / T-D8**, the coverage checks, which fail when a conditional row has no
  trigger case or no complement;
* **T-W6**, the corpus loader, which indexes captured sessions **by trigger** —
  so :class:`Trigger` is the vocabulary, defined here and nowhere else.

**It imports nothing from ``src/kitty``, and must not.**  The register is the
*specification* of what the code may do.  A specification that read the code
would be satisfied by whatever the code happened to do, which is the
self-consistency trap §3.3.1 rejects for the oracle and rejects here for the same
reason.  :func:`defined_symbols` reads ``src/kitty`` as **text**, through the
``ast`` module, never by importing it.

**Why a trigger is a name and not a callable.**  Plan §3 calls the column a
"trigger predicate", and a predicate over the inbound request would be the
obvious reading.  It cannot work: M6 fires on *an upstream 400*, M8 on *a
rejected thinking round-trip*, M9 on *an upstream tool-use format error*, M12 on
*an empty upstream response*.  None is a property of the request.  §3.3 settles
it — the oracle receives ``triggers_met`` as an argument, so the register's job
is to name the conditions, and the test that drove the request declares which
ones it arranged.

**Why ``conditional`` is a second field and not a consequence of the trigger.**
It answers exactly one question: *does §3.3.2 assertion 2 apply to this row* —
must a corpus entry exist in which this row's mutation is provably **absent**.
That is not the same as "the trigger cell says something".  P13's trigger is the
CC-origin path through ``openai_subscription``; every request on that route meets
it, so there is no complement to write and §3.2.2 lists the row as unconditional.
A trigger is arranged by one of four kinds — :class:`ArrangingBy` — and only
:attr:`~ArrangingBy.REQUEST` can be varied by a corpus entry (KBR-186).

**Scope is carried, and it names reachability, not survival (KBR-139).**
§6.2.3's completeness guard and T-D8's coverage check both need to know, per
adapter, which rows are reachable, so every row now states its ``scope``: the
provider registry keys on which at least one request path executes the row's
site, or :data:`ALL_PROVIDERS` when every adapter can reach it.  Reachability
is deliberately **not** "the row's effect survives to the capture boundary" —
KBR-160 caught M1 true at its site while a consumer one layer down discarded
its effect, with site, trigger, conditionality *and* scope all accurate.  A
wrong scope entry is falsifiable today: the guard below checks every key
against the registry read by AST, and the oracle run (KBR-51) fails a real
capture that contradicts the data.  The tempting shortcut — read the scope off
the site's class — is wrong twice:

* P8's site is ``ProviderAdapter._inject_empty_reasoning_content``, a **base
  class** method.  The site reads as all 23 adapters; only four call it
  (``kimi``, ``custom_openai``, ``zai_regular``, ``zai_coding_cc``).
  Over-scoping there would oblige T-D8 to demand nineteen complement cases that
  cannot exist.
* P5a–d's site is ``AnthropicAdapter.translate_to_upstream``, which four
  subclasses override *and conditionally delegate back to*, so the row is
  reachable on ``custom_anthropic``, ``zai_coding``, ``minimax_token`` and
  ``opencode_go`` as well.  No static rule over the class hierarchy finds that.

Scope is therefore populated by reading each override and delegation chain —
the per-adapter facts are recorded on the rows and in
``.system_design/steps/kbr139_register_scope.md`` — and held honest by the
guards in this module.  What scope still does not carry is the *survival*
question (which bridge-level rows survive to the capture boundary on the three
custom-transport adapters); that is an assertion at the §3.2.3 boundary, per
KBR-160, and is recorded in §3.2.4 as a follow-up rather than smuggled into
this data.

**A declared trigger is not a verified one.**  §7.4 hands the oracle
``triggers_met`` as an argument, and §3.3.2 asserts only in one direction: a
conditional row's mutation must be **absent** when its trigger is not met.
Nothing asserts that a trigger declared met actually fired, so a corpus entry
that over-declares would make assertion 1 claim every delta and the oracle pass
over a broken bridge.  Roughly fifteen triggers here are decidable from the
inbound request alone and could carry a predicate; they do not, because no
reader produces a projected request yet (Epic A), so every predicate would ship
unexercised — §1.4 again.  Recorded as gap G21 in §9.2 and carried by `KBR-140`.

**These guards prove the register is *well-formed*, never that it is *complete*.**  A mutation the
product performs that neither §3.2 nor this module records is invisible to all of them; only the
wire-level guard (§6.2.3, T-G2) can catch that.  One omission is already known and filed —
`KBR-184` (P13's CC-origin twin).  The other three on this list have since landed: `KBR-148`
(headers) closed with rows P9d–P9h, `KBR-149` (the `openai_subscription` reasoning injection)
with P22, and `KBR-185` (an allowlisted field dropped for being falsy) with P25.  The list is
kept to the still-open ticket so it does not disagree with §9.2's struck-through rows.  Every
one was found by reading the code by hand; none was found by a guard.  Do not read a green
suite as "the register is the whole truth".

⚠️ **Anchoring discipline.**  §3.3.1a: a path pattern is a **prefix**, claiming
its node and everything beneath it.  A row must therefore be anchored at the
**narrowest** path covering its effect.  Anchoring P15 at
``conversation.tools[*]`` rather than ``conversation.tools[*].strict`` would
claim a *deleted tool description* — one of §3.3.1's own five oracle
falsification cases.  The matcher cannot detect that, by construction, so every
row below carries a comment saying why its anchor is the narrowest one, and T-D3
carries the paired falsification case.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from harness import contract as c

# --------------------------------------------------------------------------
# The trigger vocabulary
# --------------------------------------------------------------------------


class ArrangingBy(Enum):
    """How a trigger's condition is decided.

    KBR-186. A trigger is one of four kinds:

    * :attr:`REQUEST` — a property of the inbound request; the corpus entry
      that carries the request decides it. Only REQUEST can be varied by a
      corpus entry, so only REQUEST triggers count toward the §3.3.2
      assertion-2 complement case.
    * :attr:`ROUTE` — a property of the adapter/route dispatch. Every request
      on the route meets it (or none does); a corpus entry cannot vary it.
      ``P13`` :attr:`~Trigger.CC_ORIGIN_PATH` is the canonical case: under
      reading (2) of its trigger ("the body reaching ``_cc_to_responses``,
      regardless of inbound wire"), it is met when
      ``provider.dispatch == "_cc_to_responses"``.
    * :attr:`RESPONSE` — a property of the upstream response, arranged by a
      scripted recorder (``M6``, ``M8``, ``M9``, ``M12``, ``M17``). T-D8
      reads these from a named scripted-recorder test, not from the corpus.
    * :attr:`PROFILE` — derived from the profile (``M1`` — profile model;
      ``M4`` / ``M5`` — compaction budget from profile model, and on a
      balancing profile from the smallest context in the pool). Declared
      at the call site that resolves the profile.

    The classification lets ``harness.corpus.NOT_CORPUS_DECIDABLE`` be
    *derived* from the register rather than hand-listed — that is the
    single-edit invariant this ticket exists to establish.

    The enum is deliberately named ``ArrangingBy`` so it matches the
    per-trigger attribute grammar (``Trigger.X.arranged_by``), which the
    ticket fixed. Renaming the enum to ``TriggerKind`` would require
    renaming the attribute too.
    """

    REQUEST = "request"
    ROUTE = "route"
    RESPONSE = "response"
    PROFILE = "profile"


class Trigger(Enum):
    """The conditions under which a registered mutation is permitted to fire.

    Closed deliberately, for the reason :class:`~harness.contract.WireFormat` is
    closed: T-W6 indexes the corpus by these names and T-D8 fails when a
    conditional row has no entry, so a free-form string would let two authors
    spell one condition two ways and leave a row silently uncovered.

    :attr:`ALWAYS` is the absence of a condition, not a condition — a row
    carrying it fires on every request that reaches its site. It carries no
    ``arranged_by``: the four kinds of :class:`ArrangingBy` are for the
    triggers that *are* conditions.

    Each non-``ALWAYS`` member carries an ``arranged_by`` of one of the four
    :class:`ArrangingBy` kinds (F1 in :mod:`tests.harness.test_register`).
    For compound triggers (e.g. :attr:`GEMINI_NON_STREAMING`) the
    corpus-decidability of the discriminating component picks the kind —
    that is the load-bearing choice, not abstract purity.
    """

    def __new__(cls, value: str, arranged_by: ArrangingBy | None = None) -> Trigger:
        """Construct a member, storing ``arranged_by`` alongside ``.value``.

        ``ALWAYS`` passes ``None`` and skips the attribute assignment so it
        carries no ``arranged_by`` (the test ``test_always_does_not_carry_an_arranging_by``
        is the guard).
        """
        obj = object.__new__(cls)
        obj._value_ = value
        if arranged_by is not None:
            obj.arranged_by = arranged_by
        return obj

    # Absence of a condition. No ``arranged_by`` — see the enum docstring.
    ALWAYS = ("always", None)

    # Bridge-level, request path.
    PROFILE_SETS_MODEL = ("profile_sets_model", ArrangingBy.PROFILE)
    NON_NATIVE_UPSTREAM_WIRE = ("non_native_upstream_wire", ArrangingBy.ROUTE)
    TOOL_RESULT_OVER_LIMIT = ("tool_result_over_limit", ArrangingBy.REQUEST)
    COMPACTION_RAN_WITH_OVERSIZED_TOOL_RESULT = (
        "compaction_ran_with_oversized_tool_result",
        ArrangingBy.PROFILE,
    )
    OVER_COMPACTION_BUDGET = ("over_compaction_budget", ArrangingBy.PROFILE)
    UPSTREAM_REJECTED_OVERSIZED_ON_BALANCING = (
        "upstream_rejected_oversized_on_balancing",
        ArrangingBy.RESPONSE,
    )
    ORPHAN_TOOL_RESULT = ("orphan_tool_result", ArrangingBy.REQUEST)
    THINKING_ROUNDTRIP_REJECTED = ("thinking_roundtrip_rejected", ArrangingBy.RESPONSE)
    THINKING_SIGNATURE_REJECTED = ("thinking_signature_rejected", ArrangingBy.RESPONSE)
    NATIVE_TOOL_USE_FORMAT_ERROR = ("native_tool_use_format_error", ArrangingBy.RESPONSE)
    GEMINI_PROTOCOL = ("gemini_protocol", ArrangingBy.ROUTE)
    GEMINI_NON_STREAMING = ("gemini_non_streaming", ArrangingBy.REQUEST)
    # The inbound Gemini functionCall/functionResponse carries no ``id``
    # (KBR-195). REQUEST — a property of the inbound body, decidable per
    # corpus entry — so §3.3.2 assertion 2 owes a complement, delivered with
    # T-D5 (Gemini corpus entries).
    GEMINI_INBOUND_ID_ABSENT = ("gemini_inbound_id_absent", ArrangingBy.REQUEST)

    # Bridge-level, response path.
    UPSTREAM_EMPTY_RESPONSE = ("upstream_empty_response", ArrangingBy.RESPONSE)

    # Provider-level.
    ZAI_THINKING_ENABLED = ("zai_thinking_enabled", ArrangingBy.REQUEST)
    ZAI_THINKING_DISABLED = ("zai_thinking_disabled", ArrangingBy.REQUEST)
    REASONING_EFFORT_PRESENT = ("reasoning_effort_present", ArrangingBy.REQUEST)
    MAX_TOKENS_ABSENT = ("max_tokens_absent", ArrangingBy.REQUEST)
    MULTIPLE_SYSTEM_BLOCKS = ("multiple_system_blocks", ArrangingBy.REQUEST)
    ANTHROPIC_THINKING_ENABLED = ("anthropic_thinking_enabled", ArrangingBy.REQUEST)
    ADAPTIVE_THINKING_KEYS_PRESENT = ("adaptive_thinking_keys_present", ArrangingBy.REQUEST)
    # KBR-44 (2026-09-14, B1-A): the row's deferred comment anticipated this
    # trigger. The translator emits `_output_config` and `_effort` in
    # independent `if`s (translator.py:425-426 vs :438-439), and the adapter
    # restores `output_config` on `_output_config is not None` alone
    # (anthropic.py:572-577) — so a request carrying `output_config` with no
    # `thinking` and no top-level `effort` produces a delta at
    # `envelope.extra[output_config]` with P5d's `ADAPTIVE_THINKING_KEYS_PRESENT`
    # unmet. A separate trigger is the design's own plan and §3.2.2's P5d
    # trigger-cell wording ("or output_config present") already anticipated it
    # as data-orphan until this row landed.
    OUTPUT_CONFIG_PRESENT = ("output_config_present", ArrangingBy.REQUEST)
    ASSISTANT_TURN_LACKS_THINKING_BLOCK = (
        "assistant_turn_lacks_thinking_block",
        ArrangingBy.REQUEST,
    )
    NON_STREAMING_MAX_TOKENS_OVER_4096 = (
        "non_streaming_max_tokens_over_4096",
        ArrangingBy.REQUEST,
    )
    THINKING_SIGNALLED_OR_INFERRED = ("thinking_signalled_or_inferred", ArrangingBy.REQUEST)
    CC_ORIGIN_PATH = ("cc_origin_path", ArrangingBy.ROUTE)
    RESPONSES_ORIGIN_PATH = ("responses_origin_path", ArrangingBy.REQUEST)
    # An allowlisted field whose value is falsy (`include: []`, `reasoning: {}`)
    # is dropped by the truthiness branches — decided by the inbound request's
    # own field values, so REQUEST (KBR-186's classification).
    ALLOWLISTED_FIELD_IS_FALSY = ("allowlisted_field_is_falsy", ArrangingBy.REQUEST)
    # Both decided by the resolved profile, not by the inbound request:
    # NON_ENTRA_CREDENTIAL reads the profile's configured `api_key`
    # (``AzureOpenAIAdapter.build_upstream_headers``); CHATGPT_ACCOUNT_ID_PRESENT
    # reads the OAuth `id_token` the openai_subscription profile authenticates
    # with (``OpenAISubscriptionAdapter._build_codex_headers`` → ``_extract_account_id``).
    # PROFILE per KBR-186's classification.
    NON_ENTRA_CREDENTIAL = ("non_entra_credential", ArrangingBy.PROFILE)
    CHATGPT_ACCOUNT_ID_PRESENT = ("chatgpt_account_id_present", ArrangingBy.PROFILE)
    # KBR-214 / KBR-184 (G33). Decided by the CC body's ``tool_choice`` value
    # and the presence of ``tools``: met when tools exist and the choice is
    # absent, ``none``, or any value except ``required`` and the named function
    # form (the two KBR-214 maps onto Converse ``any`` / ``tool``). REQUEST.
    BEDROCK_FORCES_AUTO_TOOL_CHOICE = (
        "bedrock_forces_auto_tool_choice",
        ArrangingBy.REQUEST,
    )
    # KBR-214 / KBR-184 (G34). Met when the inbound Anthropic body carries
    # ``disable_parallel_tool_use: false``; the flag is mapped only when
    # ``true`` (D2), so an explicit ``false`` is omitted. REQUEST.
    ANTHROPIC_PARALLEL_FALSE_OMITTED = (
        "anthropic_parallel_false_omitted",
        ArrangingBy.REQUEST,
    )
    # KBR-214 / KBR-184 (G35). Met when the inbound Anthropic body carries a
    # ``tool_choice`` that KBR-214 D9/D10 omits: a choice over no tools, or
    # a forced call to an Anthropic-defined tool (declared ``type`` other
    # than absent / ``null`` / ``"custom"``). REQUEST.
    TOOL_CHOICE_OMITTED_AS_LEGAL_BUT_UNSUPPORTED = (
        "tool_choice_omitted_as_legal_but_unsupported",
        ArrangingBy.REQUEST,
    )
    # KBR-55 (G28). Met when the inbound Anthropic body carries ``top_k`` —
    # the only field Anthropic defines and Chat Completions does not, so
    # KBR-178 carries it on the internal key ``_top_k`` and only the
    # Anthropic-family adapters restore it. REQUEST, decided by the inbound
    # body. Unconditional per §9.2 ("unconditional in P13's sense — fires
    # wherever the field is present, so it owes no §3.3.2 assertion-2
    # complement"): the off-state (no ``top_k`` on the inbound) is also the
    # row-absent state, so no corpus complement can prove anything the
    # trigger-absent case does not already prove.
    ANTHROPIC_TOP_K_PRESENT = ("anthropic_top_k_present", ArrangingBy.REQUEST)
    # KBR-55 (G29). Met when the inbound Anthropic body carries an empty
    # ``stop_sequences`` list — omitted rather than forwarded, because
    # ``StopConfiguration`` declares ``minItems: 1`` on Chat Completions and
    # ``stop: []`` is a schema-invalid body. REQUEST, decided by the inbound
    # body's value. Conditional: the non-empty case is the complement state —
    # a non-empty list is carried as-is, so §3.3.2 assertion 2 owes a
    # corpus entry that proves the omission's absence.
    EMPTY_STOP_SEQUENCES = ("empty_stop_sequences", ArrangingBy.REQUEST)


#: Docstrings on the two adapter-dispatch triggers — the asymmetry is
#: deliberately recorded because the names look like a symmetric pair and
#: are not. ``CC_ORIGIN_PATH`` is ROUTE because ``provider.dispatch`` decides
#: it (the body reaches ``_cc_to_responses`` regardless of inbound wire, per
#: KBR-186's second comment). ``RESPONSES_ORIGIN_PATH`` is REQUEST because the
#: inbound wire — Responses-shaped — decides it.
Trigger.CC_ORIGIN_PATH.__doc__ = (
    "Met when the adapter dispatches the body to ``_cc_to_responses``, "
    "regardless of inbound wire. A Messages-origin request on the "
    "``openai_subscription`` provider meets this trigger — the dispatch is "
    "decided by the provider's routing, not by the request's wire shape. "
    "Under KBR-186 this is a ROUTE property (the provider decides), not a "
    "REQUEST property. Reading (1) — ``the inbound wire was Chat "
    "Completions`` — would leave KBR-178's ``stop`` carry unclaimed on a "
    "Messages-origin entry and the oracle would report a false I1 breach on "
    "a deliberate mutation."
)
Trigger.RESPONSES_ORIGIN_PATH.__doc__ = (
    "Met when the inbound wire is Responses-shaped (``/v1/responses``), so "
    "``_original_body`` is set on the cc_request and ``_prepare_responses_body`` "
    "is the dispatch site. Unlike ``CC_ORIGIN_PATH`` (ROUTE), this trigger is "
    "decided by the request's own wire shape — REQUEST, not ROUTE. The "
    "asymmetry is deliberate and load-bearing: the two names read as a "
    "symmetric pair, but only one is decided by the provider's dispatch."
)


# --------------------------------------------------------------------------
# The row
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationRow:
    """One row of the Permitted-Mutation Register.

    Frozen, because the register is a specification: a test that could edit a row
    in place could make its own failure disappear.  The falsification cases build
    a modified *copy* with :func:`dataclasses.replace` instead.

    Attributes:
        id: The row's identifier as §3.2 publishes it — ``M1``, ``P9a``.  Stable;
            it is how every other document and ticket refers to the row.
        site: The symbols that perform the mutation, each as
            ``<path under src>:<qualified name>``, e.g.
            ``kitty/bridge/server.py:BridgeServer._normalize_model``.  More than
            one where the design names more than one.  Checked against the source
            tree by :func:`unresolved_sites`.
        trigger: The condition under which the mutation is permitted to fire.
        paths: The projection paths the mutation touches, as **patterns** in
            T-W2's vocabulary — or the single value
            :data:`~harness.contract.NOT_PROJECTABLE` when the projection
            deliberately does not model the effect.
        conditional: Whether §3.3.2 assertion 2 applies — whether a corpus entry
            must exist in which this row's mutation is provably absent.
        design_ref: Where the design document specifies the row.
        scope: The provider registry keys on which the row's site is **reachable**
            — at least one request path through that adapter executes the site —
            or the single-sentinel tuple ``(ALL_PROVIDERS,)`` when every adapter
            can reach it.  Reachability, **not** survival to §3.2.3's capture
            boundary (KBR-160: a row can be true at its site and false one layer
            down; survival lives at the boundary).  Validated by
            :func:`scope_problems`; no default, so a future row author must
            decide explicitly (the register's anti-silent-default culture).
        not_projectable_reason: Required when, and only when, ``paths`` is the
            escape.  §3.3.1a: "an empty cell would leave those rows silently
            unfalsifiable; an explicit value with a reason does not."
    """

    id: str
    site: tuple[str, ...]
    trigger: Trigger
    paths: tuple[str, ...]
    conditional: bool
    design_ref: str
    scope: tuple[str, ...]
    not_projectable_reason: str | None = None

    @property
    def is_projectable(self) -> bool:
        """Return whether the projection models this row's effect.

        Returns:
            ``False`` when the row carries the
            :data:`~harness.contract.NOT_PROJECTABLE` escape.
        """
        return self.paths != (c.NOT_PROJECTABLE,)


def row_is_in_scope(row: MutationRow, provider_key: str) -> bool:
    """Return whether ``row``'s site is reachable on the provider ``provider_key``.

    Reachability is the §3.2.4 definition: at least one request path through
    that adapter executes the row's site (KBR-160: not survival to §3.2.3's
    capture boundary). The :data:`ALL_PROVIDERS` sentinel matches every key;
    otherwise the key must be a member of ``row.scope``.

    Args:
        row: The register row to check.
        provider_key: A key of :data:`providers.registry._registry`.

    Returns:
        ``True`` if ``row.scope`` is the sentinel or contains ``provider_key``;
        ``False`` otherwise.  Does **not** validate that ``provider_key`` is a
        real registry key — that is :func:`scope_problems`' job, called once
        over the whole register rather than per row at consumer time.
    """
    return ALL_PROVIDERS in row.scope or provider_key in row.scope


def row_shape_problems(row: MutationRow) -> tuple[str, ...]:
    """Report every way one row's ``paths``/``reason`` pair is malformed.

    Pure, and separate from the loop that applies it, so a deliberately bad row
    can be handed to it — plan §1.4 requires the falsification case to run in the
    suite, and a rule enforced only inside a ``for`` over :data:`REGISTER` cannot
    be given one.  A ``__post_init__`` check was the alternative and is worse: it
    would make the malformed row unconstructable, and therefore untestable.

    Args:
        row: The row to check.

    Returns:
        One message per problem, empty when the row is well formed.
    """
    problems: list[str] = []

    if not row.paths:
        problems.append(f"{row.id}: names no path and does not take the escape")
    elif row.is_projectable:
        if row.not_projectable_reason is not None:
            problems.append(f"{row.id}: names paths and also a reason for having none")
        if c.NOT_PROJECTABLE in row.paths:
            problems.append(f"{row.id}: mixes the escape with real paths")
    elif not row.not_projectable_reason:
        problems.append(f"{row.id}: takes the escape without a reason (§3.3.1a)")

    return tuple(problems)


# Spelled once, so the data below reads as a table rather than as an argument list.
#
# `_SERVER` carries an assumption worth stating: every bridge-level row's site is
# in `server.py` today. Two are not -- M2's translators and M12's fallback
# constants -- and those spell their files in full rather than reaching for an
# alias. A row that moves out of `server.py` must do the same; silently keeping
# the alias would point it at a file that no longer defines it, and only
# `unresolved_sites` would notice, after the row was committed.
_SERVER = "kitty/bridge/server.py"
_BASE = "kitty/providers/base.py"
_SUBSCRIPTION = "kitty/providers/openai_subscription.py"
_OLLAMA_CLOUD = "kitty/providers/ollama_cloud.py"
_BEDROCK = "kitty/providers/bedrock.py"

#: The "every provider" sentinel for :attr:`MutationRow.scope`.  An entry of
#: this single element means at least one request path through **every**
#: registered provider reaches the row's site (KBR-139).
ALL_PROVIDERS: str = "*"

#: The three adapters whose ``use_native_messages`` is true — hardcoded or
#: profile-driven — and so the only ones whose request path can set
#: ``_native_messages_request`` (server.py:5048) and reach the M9 fallback
#: converter.  ``anthropic`` is *not* one: ``AnthropicAdapter`` inherits the
#: base property, which returns False.
_NATIVE_MESSAGES_ADAPTERS: tuple[str, ...] = (
    "custom_anthropic",
    "minimax_token",
    "zai_coding",
)

#: Every registry key except the two adapters that hardcode
#: ``use_native_messages = True`` (``custom_anthropic.py:77``,
#: ``zai_anthropic.py:72``): the /v1/messages handler skips
#: :class:`MessagesTranslator` on those two, so the Messages-translator rows
#: are unreachable there.  ``minimax_token`` stays in — its native flag is
#: profile-driven and defaults off, so Messages-inbound translates by default.
#: Twenty-one adapters.
_TRANSLATED_MESSAGES_ADAPTERS: tuple[str, ...] = (
    "anthropic",
    "azure",
    "bedrock",
    "byteplus",
    "custom_openai",
    "fireworks",
    "google_aistudio",
    "kimi",
    "mimo",
    "minimax",
    "minimax_token",
    "novita",
    "ollama",
    "ollama_cloud",
    "openai",
    "openai_subscription",
    "openrouter",
    "opencode_go",
    "vertex",
    "zai_coding_cc",
    "zai_regular",
)

#: The Anthropic adapter family for the P5 / P26 rows.  ``AnthropicAdapter``
#: defines ``translate_to_upstream``, ``_translate_assistant_msg`` and
#: ``_translate_tools``; four delegators reach the base class body on the
#: translated (non-native) branch — ``super().translate_to_upstream`` at
#: ``custom_anthropic.py:97``, ``minimax_token.py:135``, ``zai_anthropic.py:91``,
#: and the explicit ``AnthropicAdapter.translate_to_upstream(self, …)`` at
#: ``opencode.py:882`` for ``opencode_go``'s Messages-routed models.
#: KBR-258 measured four; ``opencode_go`` is the derived fifth (KBR-139).
_ANTHROPIC_FAMILY: tuple[str, ...] = (
    "anthropic",
    "custom_anthropic",
    "minimax_token",
    "zai_coding",
    "opencode_go",
)

#: The three Anthropic-family adapters that reach
#: :meth:`MessagesTranslator.translate_request` on the ``/v1/messages``
#: route and **restore** the agent's ``top_k`` (KBR-178's carry: the
#: other two, ``custom_anthropic`` and ``zai_coding``, are hardcoded
#: native and never run the translator). M27's scope is every translated
#: adapter *minus* these three — **eighteen** adapters (§9.2's count;
#: the naive arithmetic ``21 − 5`` is wrong because two of
#: :data:`_ANTHROPIC_FAMILY`'s five are not in
#: :data:`_TRANSLATED_MESSAGES_ADAPTERS` to begin with).
_TOP_K_RESTORING_TRANSLATED: tuple[str, ...] = (
    "anthropic",
    "minimax_token",
    "opencode_go",
)

#: M27 / M28's scope — every adapter whose ``/v1/messages`` route runs
#: :meth:`MessagesTranslator.translate_request` and whose downstream
#: adapter does not restore ``_top_k`` (M27) or non-empty ``stop``
#: (M28). Deliberately **not** ``(ALL_PROVIDERS,)``: the two
#: hardcoded-native adapters are unreachable by construction.
_TOP_K_DROPPED_SCOPE: tuple[str, ...] = tuple(
    key for key in _TRANSLATED_MESSAGES_ADAPTERS if key not in _TOP_K_RESTORING_TRANSLATED
)

_ALWAYS = Trigger.ALWAYS

#: The wire keys P23 claims: the declared `CreateResponse` control fields the
#: Codex allowlist never copies -- every published top-level field except the ten
#: the allowlist keeps and the five sampling parameters P14 claims, which is
#: 31 - 10 - 5 = 16. (The three conversation-carrying keys are *inside* the ten,
#: so they are not a third subtraction.)
#:
#: Enumerated rather than computed at import so a reviewer reads a list, not an
#: expression. Be honest about what that buys: `test_register_agreement`
#: recomputes the same difference from the adapter's allowlist and fails on a
#: disagreement, which makes widening the allowlist a **deliberate** edit here
#: rather than a silent one -- it does *not* make this row independent of the
#: code, because after the code changes the only way back to green is to edit
#: this tuple. §3.2.4 records that trade and calls it G24's posture: a green run
#: proves self-consistency, not agreement with the vendor.
#:
#: ⚠️ "Never copies" is the precise claim. The allowlist literal feeds only a
#: DEBUG log; the body is an explicit `if` chain, and six of its branches test
#: truthiness rather than presence (only `parallel_tool_calls` tests presence),
#: so an *allowlisted* field with a falsy value is dropped as well and is **not**
#: claimed here. That residue is claimed by P25 (G27 / `KBR-185`).
_CODEX_DROPPED_CONTROL_FIELDS: tuple[str, ...] = (
    "background",
    "context_management",
    "conversation",
    "max_tool_calls",
    "metadata",
    "moderation",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_options",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "text",
    "truncation",
    "user",
)

#: KBR-184 / P24 — the CC-origin twin of `_CODEX_DROPPED_CONTROL_FIELDS`. The
#: Chat Completions reader's `_PUBLISHED_EXTRA_KEYS` (T-A2 / KBR-34, retrieved
#: 2026-09-14 from `openai/openai-openapi` master) intersected with the
#: dropped set: every published top-level CC control field except `store`
#: (which `_cc_to_responses` rewrites to `False`, not drops — P17's territory)
#: and `parallel_tool_calls` (G36 / KBR-205 moved it to a canonical knob
#: address, and KBR-214 began carrying it on this route). 14 - 1 = 13 keys.
#:
#: Three keys the ticket listed (`prompt_cache_key`, `prompt_cache_retention`,
#: `safety_identifier`) are not in T-A2's CC-surface extra table — the reader
#: residualises them, which §3.3.2 names as a "named, honest failure" rather
#: than an unclaimed delta, so no row is owed for them.
#:
#: The derivation guard `TestP24ClaimsTheDroppedNonSamplingControlFields`
#: recomputes this set from the AST; widening the reader table or the builder
#: literal both turn the row red.
_CC_DROPPED_CONTROL_FIELDS: tuple[str, ...] = (
    "audio",
    "function_call",
    "functions",
    "metadata",
    "modalities",
    "moderation",
    "prediction",
    "prompt_cache_options",
    "reasoning_effort",
    "service_tier",
    "user",
    "verbosity",
    "web_search_options",
)

# --------------------------------------------------------------------------
# §3.2.1 — bridge-level rows
# --------------------------------------------------------------------------

_BRIDGE_ROWS: tuple[MutationRow, ...] = (
    MutationRow(
        id="M1",
        site=(f"{_SERVER}:BridgeServer._normalize_model",),
        trigger=Trigger.PROFILE_SETS_MODEL,
        # Unconditional despite a trigger that reads like a condition: a profile
        # without a model cannot be launched, so the complement state is
        # unreachable rather than merely untested. §3.4 leans on M1 always
        # firing. M2 and M10 are exempt for the same shape of reason — their
        # triggers are properties of the route, not of the request.
        #
        # That unreachability is load-bearing and is enforced elsewhere:
        # `Profile.model` is a required field whose validator rejects empty and
        # whitespace-only values, and every `BridgeServer` construction passes a
        # profile's model. Remove either and this flag is wrong — and so is
        # `OpenAISubscriptionAdapter._prepare_responses_body`'s `.get`, which
        # relies on the same invariant to treat an absent and a falsy model as
        # equivalent. Its docstring records why.
        #
        # KBR-160: the row was true at this site and false downstream of it —
        # `OpenAISubscriptionAdapter._prepare_responses_body` rebuilt the shipped
        # body from the raw inbound body and dropped the normalized model. Nothing
        # in this data was wrong, which is the point: a row's `paths` claim can be
        # broken by a consumer, and only an assertion at §3.2.3's capture boundary
        # sees it.
        #
        # The model, and nothing else. This is the row that makes a total
        # projection necessary at all (§3.3.1): a conversation-only projection
        # could not see the product's entire purpose.
        paths=(c.ENVELOPE_MODEL,),
        conditional=False,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M2",
        site=(
            "kitty/bridge/messages/translator.py:MessagesTranslator.translate_request",
            "kitty/bridge/responses/translator.py:ResponsesTranslator.translate_request",
            "kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",
        ),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
        not_projectable_reason=(
            "A whole-body protocol translation changes the wire format, not a field. The "
            "projection exists precisely so the two formats become comparable, so naming a path "
            "here would claim every delta on every translated request and make the oracle blind."
        ),
    ),
    MutationRow(
        id="M3",
        site=(
            f"{_SERVER}:BridgeServer._truncate_oversized_tool_results",
            # The Responses-subscription route ships the raw inbound body, so
            # the mutation is performed there on the Responses-shaped `input`
            # itself; the CC-shape site truncates a copy that never ships on
            # that route (KBR-169).
            f"{_SERVER}:BridgeServer._truncate_oversized_responses_outputs",
        ),
        trigger=Trigger.TOOL_RESULT_OVER_LIMIT,
        # The truncated content lives in one ToolResult part. Anchoring at
        # `conversation.turns` would claim a dropped turn as well.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M4",
        site=(f"{_SERVER}:BridgeServer._compact_messages",),
        trigger=Trigger.COMPACTION_RAN_WITH_OVERSIZED_TOOL_RESULT,
        # Same shape as M3 and deliberately a separate row. §3.2.1 calls M3
        # "unconditional pre-processing", meaning the *step* always runs — not
        # that the row is unconditional in this schema's sense; M3 mutates only
        # when a result exceeds the limit, so both rows are `conditional=True`.
        # What separates them is when: M4 fires only once compaction is engaged.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M5",
        site=(
            f"{_SERVER}:BridgeServer._apply_compaction",
            f"{_SERVER}:BridgeServer._compact_messages",
        ),
        trigger=Trigger.OVER_COMPACTION_BUDGET,
        # The whole collection, and this is the narrowest honest anchor:
        # compaction removes turns, which renumbers every index after the first
        # removal, so no per-turn path survives the mutation it describes.
        paths=(c.CONVERSATION_TURNS,),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M6",
        site=(
            f"{_SERVER}:BridgeServer._compact_with_tighter_budget",
            f"{_SERVER}:BridgeServer._request_with_retry_balancing",
            # KBR-256: the four streaming ladders engage the same recovery on a
            # pre-byte oversized 413 (position-as-guarantee for the three eager-
            # prepared routes; `sr is None` on `_stream_messages`).
            f"{_SERVER}:BridgeServer._stream_messages",
            f"{_SERVER}:BridgeServer._stream_responses",
            f"{_SERVER}:BridgeServer._stream_gemini",
            f"{_SERVER}:BridgeServer._stream_chat_completions",
        ),
        trigger=Trigger.UPSTREAM_REJECTED_OVERSIZED_ON_BALANCING,
        paths=(c.CONVERSATION_TURNS,),
        conditional=True,
        design_ref="§3.2.1 · §4.3 C3",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M7",
        site=(
            f"{_SERVER}:BridgeServer._validate_tool_call_pairing",
            # The Responses-subscription route applies the pairing rule to the
            # Responses-shaped `input`, and `_prune_compacted_responses_input`
            # re-runs it after compaction — pruning an earlier wire group can
            # orphan a later output (KBR-169).
            f"{_SERVER}:BridgeServer._drop_orphan_responses_tool_outputs",
            f"{_SERVER}:BridgeServer._prune_compacted_responses_input",
        ),
        trigger=Trigger.ORPHAN_TOOL_RESULT,
        # Dropping an orphan renumbers the parts after it and can empty a turn,
        # so the collection is again the narrowest anchor that survives.
        paths=(c.CONVERSATION_TURNS,),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M8",
        site=(
            f"{_SERVER}:_repair_thinking_roundtrip",
            f"{_SERVER}:_with_thinking_carrier",
        ),
        trigger=Trigger.THINKING_ROUNDTRIP_REJECTED,
        # §3.3.1a names this row as the reason a pattern is a prefix: the carrier
        # repair produces `conversation.turns[2].parts[0].signature`, which this
        # anchor claims and a bare `parts` index would not.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M9",
        site=(f"{_SERVER}:_convert_native_to_cc_format",),
        trigger=Trigger.NATIVE_TOOL_USE_FORMAT_ERROR,
        paths=(c.NOT_PROJECTABLE,),
        conditional=True,
        design_ref="§3.2.1",
        scope=_NATIVE_MESSAGES_ADAPTERS,
        not_projectable_reason=(
            "A whole-body conversion from native Messages to Chat Completions, followed by a "
            "re-run of model normalisation. Like M2 it changes the format rather than a field, "
            "and the projection is what makes the before and after comparable at all."
        ),
    ),
    MutationRow(
        id="M9a",
        # KBR-271: the M9 fallback's top-level twin of M16's fourth path
        # (KBR-263 / G38). At the KBR-271 measurement, `_convert_native_to_cc_format`
        # built its result dict from named keys and never copied
        # `body["cache_control"]`, so Anthropic's automatic-caching form --
        # projected to `envelope.extra[cache_control]` (§3.3.1) -- was dropped
        # by omission on the retried wire.
        #
        # KBR-296 (2026-09-23) amended this text: the converter now carries
        # `body["cache_control"]` onto the internal `_cache_control` carriage
        # and `AnthropicAdapter.translate_to_upstream` restores it verbatim,
        # so the retried WIRE carries the form again. The row remains the
        # register's record of the KBR-271 measurement and of the address
        # this site touches; its claim is dormant on any route where the
        # restore runs. Pinned at the wire by the KBR-200 CB-3 suite's
        # `top_level` site (`tests/bridge/test_native_passthrough_cache_breaks.py`).
        #
        # Two overlap facts, stated so no future reader re-derives them from
        # `oracle.py`. (1) Claim matching ignores sites, so whenever
        # `NON_NATIVE_UPSTREAM_WIRE` is met M16 -- same trigger, superset
        # paths -- claims this address; these rows are the register's record
        # of the M9 site's drops, not extra oracle coverage. (2) The rows are
        # anticipatory in P28's sense: the fallback is reachable only on the
        # native route (all four call sites gate on
        # `cc_request.get("_native_messages_request")`, set only on the
        # `use_native_messages` branch), so this ROUTE-kind trigger is false
        # on the only path that reaches the site today; no corpus entry
        # exercises the fallback, and any hypothetical fallback run fails
        # §4.3 C2 before assertion 1. The trigger is M16-family symmetry, not
        # a reachability claim -- the alternative (`NATIVE_TOOL_USE_FORMAT_ERROR`,
        # M9's own RESPONSE-kind trigger) would force `conditional=True`
        # (`test_rows_sharing_a_trigger_agree_on_whether_it_is_conditional`)
        # and owe a §3.3.2 assertion-2 complement no corpus entry can author.
        site=(f"{_SERVER}:_convert_native_to_cc_format",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Keyed literal -- no wildcard, so the `_SHAPES` test excludes it by
        # construction (P26's reason). The path stays narrow so it cannot
        # swallow a sibling row's `envelope.extra[<other>]` delta.
        paths=(c.extra_path("cache_control"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a · §9.2 G43",
        scope=_NATIVE_MESSAGES_ADAPTERS,
    ),
    MutationRow(
        id="M9b",
        # KBR-271: the M9 fallback's block-level twin of M16's three carrier
        # paths. At the KBR-271 measurement the rebuild flattened every block
        # it touched -- assistant text joined to a string, `tool_use` rebuilt
        # as a `tool_calls` entry, `tool_result.content` flattened to a string
        # *before* any carriage (the Messages translator preserves the nested
        # half; this converter defeated it), `tools` rebuilt from
        # `name`/`description`/`input_schema` only -- so a breakpoint on a
        # tool declaration or on any content part was dropped on the retried
        # wire.
        #
        # KBR-296 (2026-09-23) amended this text: the converter now carries
        # per-tool breakpoints onto the internal `_tool_cache_controls`
        # carriage (name-keyed, P30-aligned) and part-level breakpoints onto
        # the CC parts themselves (the adapter's restore side reads them via
        # the existing part-spread paths at `anthropic.py:714, 718-719`); the
        # assistant-text join carries a last-marked-wins breakpoint on the
        # message-level `_cache_control`; per-tool_use breakpoints ride the
        # message-level `_tool_call_cache_controls`; and `tool_result.content`
        # is now forwarded verbatim, restoring hop 1's KBR-198/KBR-199
        # nested preservation. The retried WIRE carries every carrier the
        # rebuild can express. The row remains the register's record of the
        # KBR-271 measurement; its claim is dormant on any route where the
        # restore runs. Pinned at the wire by the CB-3 suite's `tool`, `image`,
        # `user_text`, `assistant_text`, `tool_use`, `tool_result` and
        # `tool_result_nested` sites. See M9a for the two overlap facts
        # (site-blind matching against M16; anticipatory, native-route-only
        # today) and for the trigger choice.
        #
        # Deliberately outside these paths: the `system` carrier survives by
        # carriage on `zai_anthropic`/`custom_anthropic` (no drop, no row),
        # and a nested `tool_result` *block* residualises before register
        # matching (M16's "outside this row's paths by design" reason). Both,
        # plus the `minimax_token` system drop these rows' trigger cannot
        # reach (the adapter is native), are recorded on §9.2 G43.
        site=(f"{_SERVER}:_convert_native_to_cc_format",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Both patterns are already listed in `_SHAPES` for M16/P28 -- one
        # per carrier Anthropic permits a breakpoint on. The paths name the
        # field, never the block: a coarser anchor would also claim a deleted
        # part or tool description (two of §3.3.1's five falsification cases).
        paths=(
            c.tool_path(c.WILDCARD, "cache_control"),
            c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"),
        ),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a · §9.2 G43",
        scope=_NATIVE_MESSAGES_ADAPTERS,
    ),
    MutationRow(
        id="M10",
        site=(f"{_SERVER}:BridgeServer._handle_gemini",),
        trigger=Trigger.GEMINI_PROTOCOL,
        # Gemini carries the model in the URL path; M10 lifts it into the body so
        # M1 has something to override.
        paths=(c.ENVELOPE_MODEL,),
        conditional=False,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M11",
        site=(f"{_SERVER}:BridgeServer._handle_gemini",),
        trigger=Trigger.GEMINI_NON_STREAMING,
        paths=(c.ENVELOPE_STREAM,),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M12",
        site=(
            "kitty/bridge/messages/translator.py:_EMPTY_ASSISTANT_FALLBACK_TEXT",
            "kitty/bridge/responses/translator.py:_EMPTY_ASSISTANT_FALLBACK_TEXT",
        ),
        trigger=Trigger.UPSTREAM_EMPTY_RESPONSE,
        # The one response-path row, diffed by T-D10's Reply projection and never
        # by the request oracle. Anchored at index 0, not the wildcard: both
        # translators substitute one text part into a reply that was empty, so a
        # wildcard would also claim a delta at `reply.parts[5]` whenever this
        # trigger is declared met.
        #
        # KBR-99: the streamed translated /v1/messages route no longer DELIVERS
        # the substitution — its empty ladder exhausts into the D4 502 for both
        # empty shapes (SYSTEM_DESIGN §5.3 S11) — but the row stays live: the
        # translators still synthesise the fallback and every non-streaming
        # reply and other inbound protocol still writes it.
        paths=(c.reply_part_path(0),),
        conditional=True,
        design_ref="§3.2.1 · §3.3.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M14",
        site=(f"{_SERVER}:BridgeServer._build_upstream_url",),
        trigger=_ALWAYS,
        # §3.3.5: the destination is a mutation surface the body cannot show.
        # This site composes `build_base_url()` with `get_upstream_path()` through
        # `ProviderAdapter.compose_upstream_url`, which joins the endpoint to the
        # PATH and merges the two queries (KBR-143).
        #
        # `ROUTE_QUERY` is claimed here, which it was not before that change. The
        # earlier reasoning — "this site adds no query of its own, so where a
        # provider's path helper carries one, that provider's row claims it; P20 is
        # the case that does" — described concatenation, under which a query could
        # only ever arrive from the path helper. Now a query on the configured base
        # URL reaches `route.query` too, and P20 no longer accounts for every way a
        # parameter gets there. P20 keeps its own claim: both rows can contribute,
        # which is what the merge is.
        paths=(c.ROUTE_SCHEME, c.ROUTE_HOST, c.ROUTE_PATH, c.ROUTE_QUERY),
        conditional=False,
        design_ref="§3.2.1 · §3.3.5",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M15",
        site=("kitty/bridge/responses/translator.py:normalize_responses_request",),
        # Unconditional for M1's reason, not M2's. The trigger reads like a
        # condition on the request, but a body already in the array form meets this
        # row with a no-op rather than avoiding it, so there is no complement state
        # for §3.3.2 assertion 2 to arrange.
        trigger=_ALWAYS,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
        not_projectable_reason=(
            "OpenAI's `CreateResponse` declares a string `input` and the single-item array form to be "
            "one request, so the two project to one `Conversation` -- a single user turn carrying the "
            "text -- and a projection that told them apart would be reading a vendor's spelling into a "
            "wire-independent form. That is P16's reasoning. The rewrite is nonetheless real bytes at "
            "the curl_cffi boundary of section 3.2.3, where `_original_body` is this body, which is why "
            "it is a row and not an omission. **This binds the OpenAI-Responses reader, T-A3:** it must "
            "read a string `input` and every spelling of the equivalent single user message into the "
            "identical `Request`: the explicit `type: message` form this row mints, and both "
            "`EasyInputMessage` spellings (`content` as a plain string, and as an array of parts). "
            "Naming only one of the three would let a reader satisfy this literally and still "
            "project two equivalent bodies apart, which is the failure the escape is void on. If it "
            "ever does, M15 needs a projectable anchor."
        ),
    ),
    MutationRow(
        id="M16",
        site=("kitty/bridge/messages/translator.py:MessagesTranslator.translate_request",),
        # Unconditional for M2's reason, not M1's, and they share this trigger.
        # The complement is real -- the native passthrough branch shallow-copies
        # the inbound body, so breakpoints survive it -- but it is a property of
        # the *route*, chosen by the profile, and §3.3.2 assertion 2 asks for an
        # *input* that fails the trigger. No corpus entry can pick a provider.
        # The native route's guarantee is proven as product behaviour instead
        # (epic KBR-197), not by a complement nobody could author.
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Four anchors: three block-level carriers (Anthropic permits a
        # breakpoint at all three and Claude Code uses all three) plus the
        # top-level automatic-caching form (Anthropic projects it to
        # `envelope.extra[cache_control]`; KBR-263 closing G38 -- a literal
        # path, so no `_SHAPES` entry). Each names the **field**, never the
        # block: `conversation.turns[*].parts[*]` would also claim a deleted
        # part, and `conversation.tools[*]` a deleted tool description -- two of
        # §3.3.1's five oracle falsification cases. That is §3.3.1a's P15 lesson
        # applied to a second row.
        #
        # KBR-308 amendment: the translator now writes every carrier the rebuild
        # can express — top-level on ``_cache_control``, tools name-keyed on
        # ``_tool_cache_controls``, parts on the CC part (hop 1 ``carry_cache_control=True``),
        # assistant joined text last-marked-wins on message-level ``_cache_control``,
        # per-``tool_use`` index-keyed on ``_tool_call_cache_controls``, tool_result
        # message-level ``_cache_control`` — and the KBR-296 restore side picks them
        # up. The row stays as documentation-of-record for the KBR-199/KBR-258
        # measurement; the claim is dormant on any route where the carry runs
        # (system continues to ride ``_anthropic_system`` verbatim per KBR-228 part B,
        # document on ``_documents`` per KBR-222, and the nested-``tool_result``
        # content is forwarded verbatim per KBR-198). The user-message-object
        # case is the deliberate OD1 deferral (PO 2026-09-24).
        paths=(
            c.system_path(c.WILDCARD, "cache_control"),
            c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"),
            c.tool_path(c.WILDCARD, "cache_control"),
            c.extra_path("cache_control"),
        ),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a",
        scope=_TRANSLATED_MESSAGES_ADAPTERS,
    ),
    MutationRow(
        id="M17",
        site=(f"{_SERVER}:_recover_rejected_thinking", f"{_SERVER}:_strip_thinking_blocks"),
        trigger=Trigger.THINKING_SIGNATURE_REJECTED,
        # The strip removes whole `thinking` and `redacted_thinking` parts, so the
        # parts after them shift index: M8's prefix anchor over every part is the
        # only address that claims that, for the same §3.3.1a reason.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.1 · §3.3.1a · §4.3 C3",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M18",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator._translate_content",),
        trigger=Trigger.GEMINI_INBOUND_ID_ABSENT,
        # KBR-195, functionCall half. When the inbound Gemini functionCall
        # carries no ``id``, the translator synthesises a fresh
        # ``call_<uuid>`` — the Chat Completions wire requires one, and the
        # delta is real: the upstream projection carries a synthetic id
        # where the Gemini reader projected absence. Conditional, because
        # the complement (a corpus entry whose functionCall carries an id)
        # is plainly writeable and arrives with T-D5.
        #
        # Kept distinguishable from M19 by ``paths`` (``id`` vs
        # ``tool_use_id``) — the axis
        # ``test_no_two_rows_are_indistinguishable`` keys on — not by the
        # site, which both rows share.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "id"),),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M19",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator._translate_content",),
        trigger=Trigger.GEMINI_INBOUND_ID_ABSENT,
        # KBR-195, functionResponse half — M18's tool-result twin. The
        # synthesised id lands on the tool message's ``tool_call_id``, not
        # on the call's ``id``, so the row anchors at the other field.
        # Kept distinguishable from M18 by ``paths`` — the axis
        # ``test_no_two_rows_are_indistinguishable`` keys on — not by the
        # site.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "tool_use_id"),),
        conditional=True,
        design_ref="§3.2.1",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M20",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # KBR-194 gave the Gemini reader a slot for the role a
        # ``systemInstruction`` Content published; Chat Completions has no
        # equivalent, so the translation drops it and the reader's positive
        # value meets the upstream's absence at this path. §3.3.1a's
        # path-table cell used to name M2 as the claiming row; it names
        # this row since KBR-195 — M2 takes the escape and is never
        # path-matched.
        paths=(c.SYSTEM_ROLE_PATH,),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M21",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Gemini's NON_BLOCKING calling toggle on a function declaration
        # (KBR-194) has no Chat Completions equivalent, so
        # ``_translate_tools`` drops it. Anchored at the field, not the
        # whole tool — a coarser anchor would claim a deleted tool
        # description, one of §3.3.1's own falsification cases (§3.3.1a).
        paths=(c.tool_path(c.WILDCARD, "behavior"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M22",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Gemini's ``thoughtSignature`` on a ``functionCall`` part
        # (KBR-194) has no Chat Completions equivalent, so the translation
        # drops it. M8 also produces a delta at this path, but with a
        # RESPONSE trigger (a thinking round-trip rejection) — the two are
        # distinguishable by trigger, site and the narrower field anchor
        # here, and on a plain Gemini→CC request M8's trigger is not met.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "signature"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M23",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Gemini's ``functionResponse.scheduling`` — the NON_BLOCKING
        # response-side toggle (KBR-194) — has no Chat Completions
        # equivalent, so the translation drops it.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "scheduling"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M24",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Gemini part-level ``videoMetadata`` (KBR-194) has no Chat
        # Completions equivalent, so the translation drops it.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "video_metadata"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M25",
        site=("kitty/bridge/gemini/translator.py:GeminiTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # The blob/file ``displayName`` named to the model on an image part
        # (KBR-194) has no Chat Completions equivalent, so the translation
        # drops it.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "display_name"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
    ),
    MutationRow(
        id="M26",
        # KBR-184 (G31). The Messages→CC carry mints Anthropic ``metadata`` on
        # the internal key ``_metadata`` (so the Anthropic family can restore
        # it); the other seventeen routes omit it because Chat Completions' own
        # ``metadata`` is a stored-completions tag map and a bare mapping would
        # either put a new field on every request to sixteen third-party
        # providers or reject every turn (product owner's decision, 2026-09-13).
        # The eighteenth route, ``openai_subscription``, also drops it: G26 /
        # P24 claims that one at the provider level.
        #
        # ⚠️ The CC reader (T-A2 / KBR-34) projects CC's own ``metadata`` onto
        # ``envelope.extra[metadata]`` — the **same** address this row claims.
        # A reader or oracle that conflated the two meanings would mis-classify
        # any future corpus entry: the Anthropic user-id object and the CC tag
        # map are different fields. ``envelope.extra`` is keyed by wire key
        # (§3.3.1b), so the address is shared and the meanings are not.
        #
        # Site = the policy point: ``carry_tool_choice_and_metadata`` mints the
        # internal key; the restore is per-adapter. The 17-route omission is
        # the bridge's design decision and is named once here rather than as 17
        # per-adapter rows (the same shape as G28's parallel ``top_k`` gap).
        site=("kitty/bridge/messages/translator.py:carry_tool_choice_and_metadata",),
        trigger=_ALWAYS,
        paths=(c.extra_path("metadata"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1b · §9.2 G31",
        scope=_TRANSLATED_MESSAGES_ADAPTERS,
    ),
    MutationRow(
        id="M27",
        # KBR-55 / G28 — ``top_k`` dropped on the non-Anthropic-family
        # routes. KBR-178's carry mints the agent's value on the internal
        # ``_top_k`` (translator.py:439-440) and only AnthropicAdapter and
        # its two Messages-routed delegates — ``minimax_token`` (whose
        # native flag defaults off) and ``opencode_go`` (Messages models
        # only) — restore it. The other 18 adapters drop it by design.
        #
        # The ``_top_k`` itself rides on ``_INTERNAL_KEYS`` (P1 strips it
        # before the wire), so the row's *delta* is at
        # ``conversation.sampling[top_k]``: the Anthropic Messages
        # reader (``reader_anthropic_messages`` ``_SAMPLING_KEYS`` at
        # ``reader_anthropic_messages.py:84``) projects ``top_k`` onto
        # the sampling address and the upstream projection lacks it.
        # The address is keyed so the bare ``conversation.sampling``
        # anchor P13 uses on ``openai_subscription``'s CC-origin path
        # does not over-claim M27 there: the specificity rule in
        # ``oracle._conditional_violations`` keys on the narrower
        # anchor, and M27 is unconditional so assertion-2's
        # attribution never fires on either row.
        #
        # Trigger ``ANTHROPIC_TOP_K_PRESENT`` decides it; the field's
        # *presence* on the inbound is the condition, so
        # ``ArrangingBy.REQUEST``. Conditional=False per §9.2
        # ("unconditional in P13's sense — fires wherever the field
        # is present, so it owes no §3.3.2 assertion-2 complement"):
        # the off-state (no top_k on the inbound) is also the
        # row-absent state, so no corpus complement can prove anything
        # the trigger-absent case does not already prove.
        #
        # Scope is computed at module import:
        # ``_TOP_K_DROPPED_SCOPE = _TRANSLATED_MESSAGES_ADAPTERS
        # \ {anthropic, minimax_token, opencode_go}`` → **18** adapters
        # (§9.2's count). The naive arithmetic ``21 − 5`` would land at
        # 16 because two of :data:`_ANTHROPIC_FAMILY`'s five entries
        # (``custom_anthropic``, ``zai_coding``) are hardcoded native
        # and never reach the translator; the ``\`` operator above
        # already excludes them.
        site=(
            "kitty/bridge/messages/translator.py:MessagesTranslator.translate_request",
        ),
        trigger=Trigger.ANTHROPIC_TOP_K_PRESENT,
        paths=(c.sampling_path("top_k"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a · §9.2 G28",
        scope=_TOP_K_DROPPED_SCOPE,
    ),
    MutationRow(
        id="M28",
        # KBR-55 / G29 — empty ``stop_sequences`` omitted rather than
        # forwarded. The Messages-→-CC carry's truthy-only guard at
        # translator.py:431-433 (``if stop_sequences: result["stop"] =
        # stop_sequences``) skips an empty list deliberately, because
        # OpenAI's ``StopConfiguration`` declares ``minItems: 1`` and
        # ``stop: []`` is a schema-invalid Chat Completions body. The
        # omission is correct and must not be "fixed" by forwarding
        # ``[]``; the row is a register claim, not a fix.
        #
        # The Anthropic Messages reader projects ``stop_sequences``
        # onto ``conversation.sampling[stop]`` (the
        # ``stop_sequences → stop`` rename in
        # ``reader_anthropic_messages.py:85``) with no emptiness guard
        # (the reader "projects sampling by presence"), so an
        # empty ``stop_sequences`` projects as
        # ``conversation.sampling[stop] = []`` while the upstream
        # projection has no ``stop`` key. That is the row's delta.
        #
        # Trigger ``EMPTY_STOP_SEQUENCES`` (new enum member, REQUEST,
        # ``ArrangingBy.REQUEST``); ``conditional=True`` because the
        # non-empty value is the complement state — a non-empty list
        # is carried as-is, so §3.3.2 assertion 2 owes a corpus entry
        # proving the omission's absence. The trigger case and the
        # §3.3.2 complement arrive with T-D5 corpus entries.
        #
        # Scope mirrors M27: the same 18 adapters. The two
        # hardcoded-native adapters (``custom_anthropic``,
        # ``zai_coding``) never run the translator at all, and the
        # three Anthropic-family adapters whose translator runs do
        # carry the empty list as the array form (their downstream
        # needs the array form for their own ingestion).
        site=(
            "kitty/bridge/messages/translator.py:MessagesTranslator.translate_request",
        ),
        trigger=Trigger.EMPTY_STOP_SEQUENCES,
        paths=(c.sampling_path("stop"),),
        conditional=True,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a · §9.2 G29",
        scope=_TOP_K_DROPPED_SCOPE,
    ),
    MutationRow(
        id="M29",
        # KBR-55 / G30 — string-form ``stop`` rewritten into a list.
        # ``server._normalize_cc_stop`` (``server.py:1054``) wraps a
        # Chat Completions ``stop: "END"`` into ``stop: ["END"]`` at
        # the ``/v1/chat/completions`` ingress before the body forks.
        # OpenAI declares ``stop`` as ``oneOf`` a string or an array
        # of one to four strings, and every wire kitty writes downstream
        # takes only the array form.
        #
        # **Both forms project to one ``Conversation``**, so the row
        # takes §3.3.1a's ``NOT_PROJECTABLE`` escape with M15's exact
        # posture: a body already in the array form meets the row
        # with a no-op, so there is no complement state for §3.3.2
        # assertion 2 to arrange; ``conditional=False``. The
        # ``not_projectable_reason`` clause also binds T-A2 / KBR-34
        # — the future Chat Completions reader must read
        # ``stop: "END"`` and ``stop: ["END"]`` into the identical
        # ``Request`` (else the row's claim fails on the
        # indistinguishable projection — same clause M15 carries).
        #
        # Scope = ``(ALL_PROVIDERS,)`` because the
        # ``_normalize_cc_stop`` seam is on the Chat Completions
        # ingress route, not an adapter — every adapter reachable
        # through ``/v1/chat/completions`` runs it. KBR-139
        # reachability of the site.
        #
        # §9.2's stale text names "Row **M17**" for this row; that
        # id was reserved by KBR-232 for the thinking-strip row
        # (which closed G40 and is the actual M17), so M29 is the
        # landed id.
        site=("kitty/bridge/server.py:_normalize_cc_stop",),
        trigger=_ALWAYS,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a · §9.2 G30",
        scope=(ALL_PROVIDERS,),
        not_projectable_reason=(
            "OpenAI's ``StopConfiguration`` declares ``stop`` as ``oneOf`` a string or an array "
            "(G30's prose pin), and the Chat Completions and Anthropic readers both project the "
            "two forms onto one ``Conversation``: a single user turn with the stop sequence. A "
            "projection that told the two apart would be reading a vendor's spelling into a "
            "wire-independent form — the same defect M15 records. This binds T-A2 / KBR-34: it "
            "must read ``stop: \"END\"`` and ``stop: [\"END\"]`` into the identical ``Request``. "
            "Naming only one of the two forms would let a reader satisfy this literally and still "
            "project two equivalent bodies apart, which is the failure the escape is void on. If "
            "it ever does, M29 needs a projectable anchor."
        ),
    ),
    # KBR-309 rows appended after KBR-55's M27/M28/M29 (renamed to M30/M31
    # to keep ids unique; the KBR-271 "register-edit companion sites" lesson).
    MutationRow(
        id="M30",
        # KBR-309: drop an agent's ``context_management`` on the translated
        # Messages route. Anthropic's ``context_management`` is a **beta**
        # field — the documented endpoint accepts it only when the request
        # carries ``anthropic-beta: context-management-2025-06-27``
        # (platform.claude.com/docs/en/build-with-claude/context-editing,
        # fetched 2026-09-24). The bridge builds upstream headers from
        # scratch and forwards no inbound agent header (§4.2 C1), so a
        # KBR-224-style restore would 400 at the upstream. The drop is the
        # design posture, not a bug; beta-header carriage is a separate
        # product decision.
        #
        # P23's ``_CODEX_DROPPED_CONTROL_FIELDS`` lists ``context_management``
        # at the same wire-key address, but P23 is ``openai_subscription``-
        # specific (the Responses allowlist is the policy point there). The
        # Messages → CC translation has its own policy point — the
        # translator, which never reads the field — and this row names it.
        site=("kitty/bridge/messages/translator.py:MessagesTranslator.translate_request",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        paths=(c.extra_path("context_management"),),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1b",
        scope=_TRANSLATED_MESSAGES_ADAPTERS,
    ),
)

# --------------------------------------------------------------------------
# §3.2.2 — provider-level rows
# --------------------------------------------------------------------------

_PROVIDER_ROWS: tuple[MutationRow, ...] = (
    MutationRow(
        id="P1",
        site=(f"{_BASE}:ProviderAdapter._INTERNAL_KEYS",),
        trigger=_ALWAYS,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a",
        scope=(ALL_PROVIDERS,),
        not_projectable_reason=(
            "The keys P1 strips are kitty's own and no reader maps them, so the effect is a "
            "residual that fails the run before register matching ever happens — P1 can never "
            "show as a delta. A `residual` anchor would be worse than useless: a bare collection "
            "claims its members, P1's trigger is Always, so the row would claim every residual "
            "delta in the suite — including the injected `x-kitty-trace` field that is one of "
            "§3.3.1's five mandatory oracle falsification cases, and including a real internal-key "
            "leak, which is the defect this very row exists to prevent."
        ),
    ),
    MutationRow(
        id="P2a",
        site=("kitty/providers/zai.py:_ZaiBase.translate_to_upstream",),
        trigger=Trigger.ZAI_THINKING_ENABLED,
        paths=(c.extra_path("thinking"),),
        conditional=True,
        design_ref="§3.2.2",
        # Scope (KBR-139): the site is the shared _ZaiBase hook, so both zai CC
        # adapters reach it. The Anthropic-family zai_coding is a different
        # adapter with its own translate_to_upstream.
        scope=("zai_regular", "zai_coding_cc"),
    ),
    MutationRow(
        id="P2b",
        site=("kitty/providers/zai.py:_ZaiBase.translate_to_upstream",),
        trigger=Trigger.ZAI_THINKING_DISABLED,
        # Separate from P2a deliberately: §3.2.2 says the oracle must not treat
        # one branch as covering the other.
        paths=(c.extra_path("thinking"),),
        conditional=True,
        design_ref="§3.2.2",
        scope=("zai_regular", "zai_coding_cc"),
    ),
    MutationRow(
        id="P3",
        site=("kitty/providers/openrouter.py:OpenRouterAdapter.translate_to_upstream",),
        trigger=Trigger.REASONING_EFFORT_PRESENT,
        paths=(c.extra_path("reasoning"),),
        conditional=True,
        design_ref="§3.2.2",
        scope=("openrouter",),
    ),
    MutationRow(
        id="P4",
        site=("kitty/providers/openai.py:OpenAIAdapter.translate_to_upstream",),
        trigger=Trigger.REASONING_EFFORT_PRESENT,
        # OpenAI's spelling of the signal P3 carries; same trigger, different key.
        paths=(c.extra_path("reasoning_effort"),),
        conditional=True,
        design_ref="§3.2.2",
        # Scope (KBR-139): openai_subscription inherits OpenAIAdapter.translate_to_upstream
        # but its custom transport never calls it (§6.2.3) — so this row is openai only.
        scope=("openai",),
    ),
    MutationRow(
        id="P5a",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.MAX_TOKENS_ABSENT,
        paths=(c.sampling_path("max_tokens"),),
        conditional=True,
        design_ref="§3.2.2",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P5b",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.MULTIPLE_SYSTEM_BLOCKS,
        # Joining N blocks into one changes the length of the collection, so no
        # `system[i]` path survives the mutation.
        paths=(c.CONVERSATION_SYSTEM,),
        conditional=True,
        design_ref="§3.2.2",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P5c",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.ANTHROPIC_THINKING_ENABLED,
        # Two effects, two paths: it raises the agent's own max_tokens (the
        # user-visible half) and sets the thinking budget. Since KBR-225 these
        # apply only on the fallback branch -- a valid agent budget ships
        # verbatim and the row is met with no delta on either path (the M15
        # precedent).
        paths=(c.sampling_path("max_tokens"), c.extra_path("thinking")),
        conditional=True,
        design_ref="§3.2.2",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P5d",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.ADAPTIVE_THINKING_KEYS_PRESENT,
        paths=(c.extra_path("thinking"), c.extra_path("effort")),
        conditional=True,
        design_ref="§3.2.2",
        scope=_ANTHROPIC_FAMILY,
        # KBR-44 (2026-09-14): the `envelope.extra[output_config]` address,
        # deferred here since KBR-224, landed on the row below (P5f) under its
        # own trigger, because the translator emits `_output_config`
        # independently of `_effort` and of thinking (translator.py:425-426
        # vs :438-439) — extending this row under its existing trigger would
        # have left the output_config-only case unclaimed (the false I1
        # breach §3.3.1a warns about). P5d's trigger is now thinking/effort
        # only; see P5f for the output_config restore.
    ),
    MutationRow(
        id="P5f",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.OUTPUT_CONFIG_PRESENT,
        # KBR-224 / KBR-44: restore the agent's `output_config` (Anthropic's
        # documented spelling of the effort control) where the upstream
        # documents the field. Separate row and trigger, because the
        # translator's `output_config` and `effort` emissions are two
        # independent `if`s — the co-occurrence this row once assumed
        # (its KBR-186 deferred comment) is an observation about Claude Code's
        # behaviour, not a register invariant.
        paths=(c.extra_path("output_config"),),
        conditional=True,
        design_ref="§3.2.2",
        scope=_ANTHROPIC_FAMILY,
        # First corpus entry carrying `output_config`: KBR-44's
        # `effort_configured` capture (T-C1). Until a corpus entry carries the
        # field, no oracle run can see the withhold — the pairing rule
        # (§3.3.1a: a row whose conditional trigger is met but unclaimed
        # manufactures a false I1 breach) is what this row and the entry land
        # together.
    ),
    MutationRow(
        id="P5e",
        site=("kitty/providers/anthropic.py:AnthropicAdapter._translate_assistant_msg",),
        trigger=Trigger.ASSISTANT_TURN_LACKS_THINKING_BLOCK,
        # §3.2.2 calls this a message-content change, not a parameter change —
        # which is why it is anchored in the turns and P5a–d are not.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.2",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P6",
        site=("kitty/providers/azure.py:AzureOpenAIAdapter.translate_to_upstream",),
        trigger=_ALWAYS,
        # Paired with P20. §3.3.1b normalises Azure's deployment id onto
        # `envelope.model` so that what P6 removes and P20 re-adds reads as a
        # *moved* field rather than a dropped one.
        paths=(c.ENVELOPE_MODEL,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1b",
        scope=("azure",),
    ),
    MutationRow(
        id="P20",
        site=("kitty/providers/azure.py:AzureOpenAIAdapter.get_upstream_path",),
        trigger=_ALWAYS,
        # The query is not incidental: this site returns
        # `…/chat/completions?api-version={_API_VERSION}`, and the inbound
        # request to the loopback bridge carries no `api-version` at all. Under-
        # claiming here is the *loud* direction — §3.3.1a says an unclaimed delta
        # manufactures a false I1 breach — so the row names all three.
        paths=(c.ENVELOPE_MODEL, c.ROUTE_PATH, c.ROUTE_QUERY),
        conditional=False,
        design_ref="§3.2.2 · §3.3.5",
        scope=("azure",),
    ),
    MutationRow(
        id="P21",
        site=("kitty/providers/vertex.py:VertexAIAdapter.build_base_url",),
        trigger=_ALWAYS,
        # `location` selects the host, `project_id` and `location` both appear in
        # the path. The body is untouched, so no body path is named.
        paths=(c.ROUTE_HOST, c.ROUTE_PATH),
        conditional=False,
        design_ref="§3.2.2 · §3.3.5",
        scope=("vertex",),
    ),
    MutationRow(
        id="P7",
        site=("kitty/providers/fireworks.py:FireworksAdapter.normalize_request",),
        trigger=Trigger.NON_STREAMING_MAX_TOKENS_OVER_4096,
        paths=(c.sampling_path("max_tokens"),),
        conditional=True,
        design_ref="§3.2.2",
        scope=("fireworks",),
    ),
    MutationRow(
        id="P8",
        site=(f"{_BASE}:ProviderAdapter._inject_empty_reasoning_content",),
        trigger=Trigger.THINKING_SIGNALLED_OR_INFERRED,
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.2",
        # Scope (KBR-139): the base-class site reads as all 23 adapters, but
        # exactly four call it (kimi.py:95, custom_openai.py:105, zai.py:79 for
        # both zai CC subclasses) — the module docstring's own over-scoping
        # example, pinned by test_register.py::TestTheScopeColumn.
        scope=("kimi", "custom_openai", "zai_regular", "zai_coding_cc"),
    ),
    MutationRow(
        id="P9a",
        site=(
            "kitty/providers/kimi.py:KimiCodeAdapter.build_upstream_headers",
            "kitty/providers/byteplus.py:BytePlusAdapter.build_upstream_headers",
            "kitty/providers/mimo.py:MimoAdapter.build_upstream_headers",
        ),
        trigger=_ALWAYS,
        paths=(c.header_path("user-agent"),),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("kimi", "byteplus", "mimo"),
    ),
    MutationRow(
        id="P9b",
        site=("kitty/providers/mimo.py:MimoAdapter.build_upstream_headers",),
        trigger=_ALWAYS,
        # An auth-*scheme* change: one header leaves and another arrives, so both
        # are named. Naming only `api-key` would leave the removal unclaimed.
        paths=(c.header_path("authorization"), c.header_path("api-key")),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("mimo",),
    ),
    MutationRow(
        id="P9c",
        site=(
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._build_codex_headers",
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._build_user_agent",
        ),
        trigger=_ALWAYS,
        # `Accept: text/event-stream` is the header half of P17's forced
        # streaming — the Codex backend is streaming-only. `Content-Type` and
        # `Authorization` are *not* P9c effects: both match §3.2.2's base header
        # set in name, casing and value shape, and substituting the profile's
        # credential for the agent's is M14, not a per-adapter mutation. The
        # conditional `ChatGPT-Account-Id` this site also sets is P9d — its own
        # row because it is conditional, which P9c, an ALWAYS row, cannot be.
        paths=(c.header_path("user-agent"), c.header_path("version"), c.header_path("accept")),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P9d",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._build_codex_headers",),
        trigger=Trigger.CHATGPT_ACCOUNT_ID_PRESENT,
        # The header ships only when the profile's `id_token` yields an account
        # id. `_extract_account_id` returns None when the claim is absent and
        # when the token fails to parse (`except Exception`); an empty claim
        # survives extraction and is dropped by `if account_id:` in
        # `_build_codex_headers`. So the header's absence is also the
        # unparseable-token signature. The trigger is PROFILE under KBR-186's
        # classification — the resolved profile's OAuth `id_token` decides
        # it, not the inbound request — so the loader refuses it in both
        # manifest lists and §3.3.2 assertion 2 cannot find a corpus
        # complement. The complement is discharged by the L1 pins in
        # `tests/providers/test_openai_subscription.py`; `conditional=True`
        # here records that the row still needs a "mutant is absent" check
        # somewhere, just not via the corpus. T-D5's "corpus fixture arrives
        # with…" promise therefore does not apply for this row.
        paths=(c.header_path("chatgpt-account-id"),),
        conditional=True,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P9e",
        site=(
            "kitty/providers/anthropic.py:AnthropicAdapter.build_upstream_headers",
            "kitty/providers/opencode.py:OpenCodeGoAdapter.build_upstream_headers_for_model",
        ),
        trigger=_ALWAYS,
        # An auth-scheme change plus an addition: `Authorization` leaves and
        # `x-api-key` and `anthropic-version` arrive, so all three are named —
        # naming only the additions would leave the removal unclaimed (P9b's
        # rule). The lowercase `content-type` re-spelling has no address:
        # `contract.header_path` lowercases for matching, so a casing-only
        # difference is not claimable, and this comment is its record.
        # `custom_anthropic` and `minimax_token` inherit the first site;
        # `opencode_go` reaches the same set only on its Messages-routed
        # models — its default `build_upstream_headers` is the baseline Bearer
        # set, so that hook is deliberately not a site.
        paths=(
            c.header_path("authorization"),
            c.header_path("x-api-key"),
            c.header_path("anthropic-version"),
        ),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("anthropic", "custom_anthropic", "minimax_token", "opencode_go"),
    ),
    MutationRow(
        id="P9f",
        site=("kitty/providers/zai_anthropic.py:ZaiAnthropicAdapter.build_upstream_headers",),
        trigger=_ALWAYS,
        # Kept apart from P9e because a row's paths must be true of every site
        # it names: this adapter adds `anthropic-version` and re-spells
        # `content-type` lowercase, but its auth stays `Authorization: Bearer`
        # — the baseline shape — so claiming `authorization` or `x-api-key`
        # here would lie. The casing re-spelling has no address
        # (`contract.header_path` lowercases); this comment is its record.
        paths=(c.header_path("anthropic-version"),),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("zai_coding",),
    ),
    MutationRow(
        id="P9g",
        site=("kitty/providers/azure.py:AzureOpenAIAdapter.build_upstream_headers",),
        trigger=Trigger.NON_ENTRA_CREDENTIAL,
        # An auth-scheme change on the key-based credential: `Authorization`
        # leaves and `api-key` arrives, so both are named (P9b's rule). The
        # Entra branch of the same hook sends the baseline
        # `Authorization: Bearer`, which is why the trigger is named rather
        # than ALWAYS. The credential is profile config, not request content,
        # so no corpus entry can vary it — the row is unconditional in
        # §3.3.2's sense. Under KBR-186's four kinds that makes the trigger
        # PROFILE (decided by the resolved profile), not ROUTE: the adapter's
        # dispatch is the same hook on both branches; what differs is the
        # profile's configured credential.
        paths=(c.header_path("authorization"), c.header_path("api-key")),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("azure",),
    ),
    MutationRow(
        id="P9h",
        site=("kitty/providers/ollama.py:OllamaAdapter.build_upstream_headers",),
        trigger=_ALWAYS,
        # The P9b shape minus the addition: local Ollama requires no auth and
        # ignores the header, so only the removal is named. `OllamaCloudAdapter`
        # overrides the hook and keeps Bearer auth — the baseline set — so it
        # is deliberately not a site.
        paths=(c.header_path("authorization"),),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
        scope=("ollama",),
    ),
    MutationRow(
        id="P10",
        site=("kitty/providers/minimax.py:MiniMaxAdapter.normalize_request",),
        trigger=_ALWAYS,
        paths=(c.extra_path("reasoning_split"),),
        conditional=False,
        design_ref="§3.2.2",
        scope=("minimax",),
    ),
    MutationRow(
        id="P11",
        site=("kitty/providers/bedrock.py:BedrockAdapter.translate_to_upstream",),
        trigger=_ALWAYS,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.4",
        scope=("bedrock",),
        not_projectable_reason=(
            "A whole-body translation into Bedrock Converse — a third wire format M2 does not "
            "name. Same reason as M2: the projection is what makes the formats comparable, so "
            "the translation itself names no field."
        ),
    ),
    MutationRow(
        id="P12",
        site=("kitty/providers/ollama_cloud.py:OllamaCloudAdapter.translate_to_upstream",),
        trigger=_ALWAYS,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.4",
        scope=("ollama_cloud",),
        not_projectable_reason=(
            "A whole-body translation into Ollama's /api/chat — a fourth wire format. Same reason as M2 and P11."
        ),
    ),
    MutationRow(
        id="P13",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._cc_to_responses",),
        trigger=Trigger.CC_ORIGIN_PATH,
        # §3.3.1 pins P13 to the bare collection rather than to fourteen keys:
        # the row is "the Codex allowlist drops what it does not permit", and a
        # fifteenth key added upstream must be claimed by the same row.
        paths=(c.CONVERSATION_SAMPLING,),
        conditional=False,
        design_ref="§3.2.2 · §3.2.3 · §3.3.1",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P14",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._prepare_responses_body",),
        trigger=Trigger.RESPONSES_ORIGIN_PATH,
        # The same backend restriction reached from the other input shape, where
        # the parameter is spelled `max_output_tokens`. §3.3.1b maps that onto
        # `max_tokens`, so the two rows differ by trigger, not by path.
        paths=(c.CONVERSATION_SAMPLING,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1b",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P23",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._prepare_responses_body",),
        trigger=Trigger.RESPONSES_ORIGIN_PATH,
        # P14's other half. The allowlist drops 21 of `CreateResponse`'s 31
        # fields: five are sampling and P14 claims them at the bare collection,
        # and these sixteen are declared control fields, which §3.3.1b sends to
        # `envelope.extra[<wire key>]` -- an address no row reached (KBR-171).
        #
        # ⚠️ Enumerated, NOT anchored at a bare `envelope.extra`. The bare form
        # would match -- a bracket-free pattern segment claims a bracketed member
        # of itself -- and it is still the wrong anchor, because `extra` is where
        # the *injections* live: a bare anchor on this trigger would also claim
        # the `reasoning` injection P22 (§9.2's G23) claims at
        # `envelope.extra[reasoning]` on this same route, so P22 could be deleted
        # with nothing going red.
        #
        # P13/P14's bare `conversation.sampling` is not a precedent, and not
        # because bare and enumerated agree there -- they do not, `SAMPLING_KEYS`
        # has fifteen members and P13 drops fourteen. Those rows over-claim
        # deliberately. The difference is what the over-claim can absorb: a
        # closed set nothing injects into absorbs at worst another sampling key,
        # while `envelope.extra` absorbs whole rows.
        #
        # Enumerating costs something and the cost is not zero: a thirty-second
        # published field would be unclaimed until someone adds it here. Nothing
        # detects a vendor revision (§8's determinism rules; G24's shape), so the
        # trade is a loud one-line failure against a silent absorbed row.
        #
        # `P22` was skipped deliberately -- §9.2's G23 reserved it for KBR-149,
        # which has since landed it; an id is how every ticket refers to a row.
        paths=tuple(c.extra_path(key) for key in _CODEX_DROPPED_CONTROL_FIELDS),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1a · §3.3.1b",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P22",
        site=(
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._prepare_responses_body",
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._cc_to_responses",
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter.make_request",
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter.stream_request",
        ),
        trigger=Trigger.REASONING_EFFORT_PRESENT,
        # P3's signal in the Codex spelling, on the one adapter whose request
        # path never reaches `translate_to_upstream`, so P4 cannot cover it
        # (§3.2.3). Four sites, not the three the gap walk counted: the
        # CC-origin builder `_cc_to_responses` injects from the same key and
        # predates the ticket (KBR-149). At effort `"none"` the trigger is
        # met-but-inert (the P5c precedent), so the assertion-2 complement
        # needs a corpus entry carrying an effort, not a `"none"` one. The
        # enum carries the request-side clause only: the Responses-origin
        # precedence gate (a caller-sent truthy `reasoning` wins) lives in
        # the §3.2.2 trigger cell; the closed vocabulary has no member for it.
        paths=(c.extra_path("reasoning"),),
        conditional=True,
        design_ref="§3.2.2",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P25",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._prepare_responses_body",),
        trigger=Trigger.ALLOWLISTED_FIELD_IS_FALSY,
        # The allowlist's residue: membership is not what carries a field
        # through, so an *allowlisted* field whose value is falsy is dropped
        # anyway by the truthiness branches (`include: []`, `reasoning: {}` --
        # both legal under `CreateResponse`; the reader projects by presence,
        # so each is a present-inbound, absent-upstream delta). Enumerated in
        # data but NOT derived the way P23's are: that derivation recomputes
        # allowlist minus reader table, while this set needs "whose falsy form
        # is legal", which is a vendor-schema judgment no artifact in the tree
        # holds (G24's posture -- nothing detects a revision).
        #
        # `tool_choice` is truthiness-gated on the same chain and is an extra
        # key, but it is deliberately NOT claimed: `tool_choice: ""` is not a
        # legal `CreateResponse` value, so that branch is unreachable with a
        # falsy value today. The `instructions`/`input`/`tools` branches are
        # likewise truthiness-gated but project to no `envelope.extra` path.
        #
        # P22's interaction: the two triggers are predicates on different
        # request fields and can co-occur (falsy `reasoning` beside a
        # non-`none` effort). In that state the `elif` injects
        # `{"effort": ...}`, so the upstream projection carries a `reasoning`
        # key with the injected value and P22's injection claims the address;
        # P25's drop is provably absent there. The rows' claims on
        # `envelope.extra[reasoning]` do not overlap.
        paths=(c.extra_path("include"), c.extra_path("reasoning")),
        conditional=True,
        design_ref="§3.2.2",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P15",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._prepare_responses_body",),
        trigger=Trigger.RESPONSES_ORIGIN_PATH,
        # ⚠️ `.strict`, NOT `conversation.tools[*]`. §3.3.1a names this exact row:
        # the coarser anchor would claim a deleted tool description, which is one
        # of §3.3.1's five oracle falsification cases.
        paths=(c.tool_path(c.WILDCARD, "strict"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1a",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P16",
        site=(f"{_SUBSCRIPTION}:_convert_content_types",),
        trigger=Trigger.RESPONSES_ORIGIN_PATH,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1a",
        scope=("openai_subscription",),
        not_projectable_reason=(
            "The input_text/output_text tag is redundant with the turn's role, which the "
            "projection already carries. §3.3.1a names P16 as the example: modelling the tag "
            "would put one vendor's spelling into a wire-independent form."
        ),
    ),
    MutationRow(
        id="P17",
        site=(
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._cc_to_responses",
            f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._prepare_responses_body",
        ),
        trigger=_ALWAYS,
        # `stream: True` overrides a non-streaming client request — the
        # subscription-path analogue of M11 — and `store: False` is the second
        # field, so both are named.
        paths=(c.ENVELOPE_STREAM, c.ENVELOPE_STORE),
        conditional=False,
        design_ref="§3.2.2",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P18",
        site=(
            # KBR-89 (T-H2) extracted the body's modelId/stream pops from
            # the two transport methods into this pure builder. Both
            # transports now call `_bedrock_body`, so the pops' load-bearing
            # site is here — `make_request` and `stream_request` splat the
            # returned body verbatim and add no further mutations.
            "kitty/providers/bedrock.py:BedrockAdapter._bedrock_body",
        ),
        trigger=_ALWAYS,
        # §3.3.1b normalises Converse's `modelId` onto `envelope.model`, so the
        # pop is expressible. Applied in the transport, after the hook — §3.2.3
        # says the capture boundary is after this mutation, not before.
        paths=(c.ENVELOPE_MODEL, c.ENVELOPE_STREAM),
        conditional=False,
        design_ref="§3.2.2 · §3.2.3 · §3.3.1b",
        scope=("bedrock",),
    ),
    MutationRow(
        id="P19",
        site=(
            # KBR-90 (T-H5) extracted the body's stream overwrite from the
            # two transport methods into this pure builder. Both
            # transports now call `_ollama_body` (parametrised on a
            # `streaming` flag), so the overwrite's load-bearing site is
            # here — `make_request` and `stream_request` post the returned
            # body verbatim and add no further mutations.
            "kitty/providers/ollama_cloud.py:OllamaCloudAdapter._ollama_body",
        ),
        trigger=_ALWAYS,
        paths=(c.ENVELOPE_STREAM,),
        conditional=False,
        design_ref="§3.2.2 · §3.2.3",
        scope=("ollama_cloud",),
    ),
    # KBR-258 — the Anthropic adapter family drops a Chat Completions request's
    # cache breakpoints at five sites on the translated route, measured identical
    # on `anthropic`, `minimax_token`, `zai_coding` and `custom_anthropic`. These
    # five rows are P26..P30 — the CC-origin half of G37's "still owed" set.
    # `minimax_token` and `custom_anthropic` short-circuit on
    # `_native_messages_request` and otherwise delegate to
    # `super().translate_to_upstream`; `zai_anthropic.ZaiAnthropicAdapter` is no
    # exception (`zai_anthropic.py:85-91`). The drops KBR-199 measured happen on
    # the translated (non-native) branch only.
    #
    # Scope (KBR-139) names five adapters, not the four KBR-258 measured:
    # `opencode_go` is the derived fifth — its Messages-routed models call
    # `AnthropicAdapter.translate_to_upstream(self, …)` explicitly
    # (`opencode.py:882`), so the same sites run there. Measured ≠ exhaustive;
    # the delegation read, not a wire capture, is what adds the fifth.
    #
    # `conditional=False` is forced by three independent guards: KBR-186 makes
    # `NON_NATIVE_UPSTREAM_WIRE` (ArrangingBy.ROUTE) non-corpus-variable; M16's
    # comment records the same reasoning for the Messages twin; and
    # `test_register.py::test_rows_sharing_a_trigger_agree_on_whether_it_is_conditional`
    # is the structural guard — any `conditional=True` here would conflict with
    # M2/M16/M20-M25 already `conditional=False` on this trigger. The native
    # passthrough carrying the breakpoint is the observational complement,
    # proven as product behaviour in epic KBR-197, not a corpus entry. The
    # ticket's loose "each row owes a complement corpus entry" prose is
    # reconciled in KBR-258's Jira comment.
    MutationRow(
        id="P26",
        # The `translate_to_upstream` body is read whole for keys the
        # Anthropic wire defines; the root-level `cache_control` is never read,
        # so on the translated route this `envelope.extra[cache_control]` is
        # dropped by omission. M16's note on §3.3.1's "carried whole, not
        # reduced" rule does not apply — the reader does not currently
        # consume the key into this address; the row claims the address the
        # moment the reader grows the slot (the G38 precedent on the Messages
        # twin). Today such a body residualises, which is the honest named
        # failure: the field is present, nobody claims it.
        #
        # KBR-308 amendment: the adapter now reads a raw CC body's
        # `cache_control` as a fallback when the KBR-296 carriage is absent
        # (DQ-B: carriage wins, raw shape is fallback) and restores it onto
        # the wire's top-level slot — the KBR-199 drop is closed on the
        # CC-origin path too; this row stays as the measurement-of-record
        # and the claim is dormant where the carry runs.
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        # Keyed literal — no wildcard, so the `_SHAPES` test excludes it by
        # construction. The path stays narrow so an over-claim cannot swallow
        # a sibling row's `envelope.extra[<other>]` delta (the §3.3.1a
        # prohibition on the bare `envelope.extra` anchor).
        paths=(c.extra_path("cache_control"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · §9.2 G37",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P27",
        # The system-extraction loop joins system blocks into one string; a
        # `cache_control` on a system content part does not survive the join
        # (P5b's twin on this route — the join is the cause; the cache drop is
        # the side-effect we register). The CC reader projects a system
        # content-part breakpoint onto `conversation.system[*].cache_control`
        # today, so this row is **claimable now**, not anticipatory.
        # `forwards_thinking_signature` does not change this: the carriage
        # `_anthropic_system` is a Messages-ingress concern, set by
        # `MessagesTranslator`, stripped on the CC route by P1 — the join
        # stands.
        #
        # KBR-308 amendment: the rebuild now emits a blocks-form ``system``
        # (each text block carrying its marker) when any system part carries a
        # marker AND ``self.forwards_thinking_signature`` is True — the same
        # gate the ``_anthropic_system`` restore uses. On ``minimax_token`` and
        # ``opencode_go``'s Messages-routed models (``forwards_thinking_signature=False``;
        # MiniMax rejects ``cache_control`` on system blocks outright,
        # ``minimax_token.py:29-30``) the join stands and the G43 scope-out is
        # preserved on the CC-origin path too; the new CB-2 pin asserts it.
        # The row stays as the measurement-of-record; the claim is dormant
        # where the carve runs.
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        paths=(c.system_path(c.WILDCARD, "cache_control"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · §9.2 G37",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P28",
        # The outer message loop reads each `{"role": …, "content": …}` for
        # role and content only; a `cache_control` on the message dict itself
        # is dropped by omission. This site covers the *user-message* and
        # *tool-message* object case; the assistant-message object case
        # (which rebuilds inside `_translate_assistant_msg`) rides P29's
        # site — the oracle matches paths, not sites, so the address is
        # claimed either way.
        #
        # The kept half of G37's measurement — a breakpoint on a user content
        # part and on tool-message content (moved inside the `tool_result`) —
        # is NOT this row's and carries no row in this set: the adapter family
        # preserves both on the CC route, the same boundary M16 draws on the
        # Messages twin.
        #
        # Anticipatory today: the CC reader residualises a message-dict-level
        # `cache_control` (`_read_one_message`'s `_residualise` sets name no
        # cache key), so such a body fails the run on residual first — the
        # honest named failure. The row lands now, before the reader grows
        # the slot, on the G38 precedent.
        #
        # KBR-308 amendment (tool-message-object half): `_tool_result_block`
        # now reads a raw `messages[*].cache_control` as a fallback when the
        # message-level `_cache_control` carriage is absent (DQ-B) and
        # attaches it onto the rebuilt Anthropic `tool_result` block — the
        # KBR-199 tool-message drop is closed on the CC-origin path too. The
        # row stays live for the **user-message-object** half, deliberately
        # deferred (OD1, PO 2026-09-24 — Anthropic's wire has no message-level
        # slot, no published CC dialect defines the shape, and any placement
        # would invent a block choice with no fidelity basis); the CB-2 case
        # `user-message-object-deferred-dropped` pins the deferral.
        #
        # Anchor contingency: the §3.3.1a path vocabulary names
        # `conversation.turns[*].parts[*].cache_control` (the field-level
        # address), and this row's narrowest path assumes the reader will
        # project the message-object `cache_control` onto a Part rather than
        # onto Turn itself (`conversation.turns[*].cache_control`, not in the
        # vocabulary today). If a future reader lands the slot on Turn, the
        # row's anchor must move to match — the comment records it.
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · §9.2 G37",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P29",
        # `_translate_assistant_msg` rebuilds every `tool_calls` block into
        # an Anthropic `tool_use` block (`{"type": "tool_use", "id": …,
        # "name": …, "input": …}`); a `cache_control` on the tool_call dict
        # is dropped by omission. This site also owns the assistant-message
        # object case for the same reason — the assistant message is rebuilt
        # here, not in the outer loop.
        #
        # Anchored at the same path as P28 because both project to a Part
        # (the tool_use is a Part); distinguishable by site, the axis
        # `test_no_two_rows_are_indistinguishable` explicitly allows (P3/P4
        # precedent). Same anchor contingency as P28; anticipatory today for
        # the same reason — the CC reader residualises a tool_call-level
        # `cache_control` (`_read_tool_calls`'s `_residualise` set names no
        # cache key), so such a body fails the run on residual first, the
        # G38 precedent.
        #
        # KBR-308 amendment: `_translate_assistant_msg` now reads a raw
        # `tool_calls[i].cache_control` and an assistant-message-object
        # `messages[i].cache_control` as fallbacks when the index-keyed /
        # message-level carriages are absent (DQ-B) — the KBR-199
        # tool-call and assistant-message-object drops are closed on the
        # CC-origin path. The row stays as the measurement-of-record; the
        # claims are dormant where the fallback reads run.
        site=("kitty/providers/anthropic.py:AnthropicAdapter._translate_assistant_msg",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        paths=(c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · §9.2 G37",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P30",
        # `_translate_tools` rebuilds every tool declaration as
        # `{name, description, input_schema}`; a `cache_control` on the
        # CC tool's `function` member is dropped by omission. The CC reader
        # projects a tool-decl breakpoint onto
        # `conversation.tools[<name>].cache_control` today, so this row is
        # **claimable now** — like P27, not anticipatory. The Messages-route
        # twin of this drop is M16's tool-decl path.
        #
        # KBR-308 amendment: `_translate_tools` now reads a raw
        # `tools[i].cache_control` as a fallback when the name-keyed
        # `_tool_cache_controls` carriage lacks the entry (DQ-B) and
        # attaches it onto the rebuilt Anthropic tool declaration — the
        # KBR-199 tool-declaration drop is closed on the CC-origin path.
        # The row stays as the measurement-of-record; the claim is dormant
        # where the fallback read runs.
        site=("kitty/providers/anthropic.py:AnthropicAdapter._translate_tools",),
        trigger=Trigger.NON_NATIVE_UPSTREAM_WIRE,
        paths=(c.tool_path(c.WILDCARD, "cache_control"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · §9.2 G37",
        scope=_ANTHROPIC_FAMILY,
    ),
    MutationRow(
        id="P24",
        site=(f"{_SUBSCRIPTION}:OpenAISubscriptionAdapter._cc_to_responses",),
        trigger=Trigger.CC_ORIGIN_PATH,
        # KBR-184 (G26). P13's CC-origin twin for the **non-sampling** half:
        # ``_cc_to_responses`` builds the Responses body from scratch and ships
        # ``model``, ``messages``→``input``, ``stream``, ``store``, ``tools``,
        # ``tool_choice``, ``parallel_tool_calls`` and an injected ``reasoning``.
        # Every other declared Chat Completions control field (T-A2's
        # ``_PUBLISHED_EXTRA_KEYS``) is dropped. P13 is anchored at the bare
        # ``conversation.sampling`` and reaches none of these, so T-D5 would
        # report a false I1 breach on the CC-origin route exactly as it would
        # have on the Responses-origin one.
        #
        # ⚠️ Enumerated, NOT anchored at a bare ``envelope.extra`` — the bare
        # form matches and would over-claim ``extra[reasoning]`` (P22),
        # ``extra[store]`` (P17 rewrites it), and ``extra[parallel_tool_calls]``
        # (G36 / KBR-205 unified the knob address, and ``_cc_to_responses``
        # carries the field since KBR-214). The ``_CC_DROPPED_CONTROL_FIELDS``
        # constant is the reader's ``_PUBLISHED_EXTRA_KEYS`` minus what the
        # builder carries; the derivation guard
        # ``TestP24ClaimsTheDroppedNonSamplingControlFields`` recomputes it from
        # the AST, so widening the reader table or the builder literal both
        # turn the row red.
        #
        # The 13 keys are the reader's table minus the builder's carries,
        # which is ``_PUBLISHED_EXTRA_KEYS − {store}`` today (store is rewritten
        # to ``False``, not dropped — P17's territory).
        paths=tuple(c.extra_path(key) for key in _CC_DROPPED_CONTROL_FIELDS),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1a · §3.3.1b · §9.2 G26",
        scope=("openai_subscription",),
    ),
    MutationRow(
        id="P31",
        # KBR-184 (G32). Ollama ``/api/chat`` defines neither a tool choice
        # nor a parallel-tool-use knob, so ``OllamaCloudAdapter.translate_to_upstream``
        # writes neither. Each is an unclaimed ``envelope.extra[...]`` delta
        # on the ``ollama_cloud`` route — the same class as G26 / P13 / P23.
        #
        # ⚠️ ``paths`` must be true of every site (P9e/P9f rule), so this row
        # names only ``ollama_cloud`` — the bedrock parallel-knob twin gets
        # its own row (P32). The shared knob address
        # ``envelope.extra[parallel_tool_calls]`` is the G36 / KBR-205
        # canonical form; P31 does not need a parallel-knob reader side to
        # ship the wire key.
        site=(f"{_OLLAMA_CLOUD}:OllamaCloudAdapter.translate_to_upstream",),
        trigger=_ALWAYS,
        paths=(c.extra_path("tool_choice"), c.extra_path("parallel_tool_calls")),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1b · §9.2 G32",
        scope=("ollama_cloud",),
    ),
    MutationRow(
        id="P32",
        # KBR-184 (G32). Bedrock Converse's ``ToolConfiguration`` has no
        # parallel-tool-use knob (botocore ``bedrock-runtime``), so the
        # bedrock hook writes no ``parallelToolCalls``. The G36 / KBR-205
        # canonical knob address (``envelope.extra[parallel_tool_calls]``) is
        # unclaimed on this route without this row. Site is
        # ``translate_to_upstream`` rather than the boto3 transport because
        # that is where the hook-level decision to omit the field lives (the
        # transport mutates ``modelId`` / ``stream``, P18).
        site=(f"{_BEDROCK}:BedrockAdapter.translate_to_upstream",),
        trigger=_ALWAYS,
        paths=(c.extra_path("parallel_tool_calls"),),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1b · §9.2 G32",
        scope=("bedrock",),
    ),
    MutationRow(
        id="P33",
        # KBR-184 (G33). ``BedrockAdapter.translate_to_upstream`` has always
        # written ``toolChoice: {"auto": {}}`` whenever tools are present;
        # KBR-214 maps only ``required`` and the named form onto Converse
        # ``any`` / ``tool``. Converse's ``ToolChoice`` union has no ``none``,
        # and dropping ``toolConfig`` is unavailable once a transcript carries
        # ``toolUse`` / ``toolResult`` (``toolConfig must be defined...``).
        # The rewrite is conditional on the value: any CC choice that isn't
        # ``required`` and isn't a named choice to an ordinary tool is
        # rewritten to ``auto``.
        #
        # Shares ``envelope.extra[tool_choice]`` with P35; the two are
        # distinguishable by site (the P3/P4 precedent
        # ``test_no_two_rows_are_indistinguishable`` explicitly allows it).
        # Both are conditional, so the two rows cannot disagree about
        # whether the address owes a complement. The trigger case and §3.3.2
        # assertion-2 complement arrive with the T-D5 corpus entries, as for
        # P22 / P25.
        site=(f"{_BEDROCK}:BedrockAdapter.translate_to_upstream",),
        trigger=Trigger.BEDROCK_FORCES_AUTO_TOOL_CHOICE,
        paths=(c.extra_path("tool_choice"),),
        conditional=True,
        design_ref="§3.2.2 · §3.3.1b · §9.2 G33",
        scope=("bedrock",),
    ),
    MutationRow(
        id="P34",
        # KBR-184 (G34). KBR-214 maps ``disable_parallel_tool_use: true`` onto
        # ``parallel_tool_calls: false``; ``false`` is omitted because it is
        # the default on both Anthropic and Chat Completions, and writing it
        # would add a second field some providers reject (D2). The Anthropic
        # reader can nonetheless tell them apart, so an explicit ``false``
        # is a delta — the omission is **correct** and must not be "fixed" by
        # forwarding it, G29's exact shape.
        #
        # Conditional on the value: trigger met when the inbound Anthropic
        # body carries ``disable_parallel_tool_use: false``. The address
        # ``envelope.extra[parallel_tool_calls]`` exists as of KBR-205
        # (§3.3.1b); the row registers the omission now, with the trigger
        # case + §3.3.2 complement (a body where the flag is ``true`` and
        # carried) arriving with the T-D5 corpus, as for P25.
        site=("kitty/bridge/messages/translator.py:carry_tool_choice_and_metadata",),
        trigger=Trigger.ANTHROPIC_PARALLEL_FALSE_OMITTED,
        paths=(c.extra_path("parallel_tool_calls"),),
        conditional=True,
        design_ref="§3.2.2 · §3.3.1b · §9.2 G34",
        scope=_TRANSLATED_MESSAGES_ADAPTERS,
    ),
    MutationRow(
        id="P35",
        # KBR-184 (G35). KBR-214 omits a legal ``tool_choice`` only where
        # carrying it would create a failure the agent did not cause: (1) a
        # choice over no tools — legal on Anthropic and ``'tool_choice' is
        # only allowed when 'tools' are specified`` on OpenAI; (2) a forced
        # call to an **Anthropic-defined** tool (declared ``type`` other than
        # absent / ``null`` / ``"custom"``, e.g. Claude Code's
        # ``web_search_20250305``) — Anthropic flattens it into a schema-less
        # function nothing on the route can execute
        # (anthropics/claude-code#56984; omitted on the product owner's
        # decision, 2026-09-13). ``{"type": "any"}`` over only Anthropic-defined
        # tools is carried: guarding it would reason over the whole tool
        # list rather than one name. Forced calls to **undeclared** tools are
        # not omitted — that is the agent's mistake and the provider's error
        # names it.
        #
        # Shares ``envelope.extra[tool_choice]`` with P33; distinguishable by
        # site (bedrock auto-rewrite vs Messages-route omission). Case (2)'s
        # trigger is authorable now that the Anthropic reader carries
        # ``ToolDecl.type`` (KBR-205, closing G36); the corpus trigger case +
        # §3.3.2 complement arrive with T-D5.
        site=("kitty/bridge/messages/translator.py:carry_tool_choice_and_metadata",),
        trigger=Trigger.TOOL_CHOICE_OMITTED_AS_LEGAL_BUT_UNSUPPORTED,
        paths=(c.extra_path("tool_choice"),),
        conditional=True,
        design_ref="§3.2.2 · §3.3.1b · §9.2 G35",
        scope=_TRANSLATED_MESSAGES_ADAPTERS,
    ),
    # KBR-137 — OpenCode Go's four `/v1/responses` models are now servable.
    # The whole-protocol translate, the eight CC-only drops, the
    # max_tokens→max_output_tokens rename, and the reasoning injection are
    # the four mutations the Responses route performs; P39–P41 (tools
    # envelope unwrap, tool_choice envelope unwrap, response_format → text.format
    # nesting move) are part of P36's whole-protocol claim, like P22 was
    # part of the Codex P13–P17 set rather than a row of its own. The
    # projection has no path vocabulary for an envelope unwrap, and the L1
    # tests in ``tests/test_provider_opencode_responses.py`` pin each
    # mutation directly. ``UNCONDITIONAL`` per the test gate (§3.2.2 footer):
    # every request on the route meets the trigger.
    MutationRow(
        id="P36",
        site=("kitty/providers/opencode.py:OpenCodeGoAdapter._cc_to_responses",),
        trigger=Trigger.CC_ORIGIN_PATH,
        # Whole-body translation into OpenAI Responses — a fifth wire format
        # M2 does not name. Same reason as P11/P12: the projection is what
        # makes the formats comparable, so the translation itself names no
        # field. The per-message parts (system → instructions, user →
        # input_text, assistant → output_text, tool → function_call_output)
        # are part of this single claim, as the P22 message-level translation
        # was on the Responses-origin twin.
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.4 · KBR-137",
        scope=("opencode_go",),
        not_projectable_reason=(
            "KBR-137 adds a fifth wire format — OpenAI Responses — that M2 does "
            "not name. Like P11/P12, the projection is what makes the formats "
            "comparable, so the whole-protocol translate names no field. The "
            "per-message renames (system→instructions, user→input_text, "
            "assistant→output_text, tool→function_call_output), the tools and "
            "tool_choice envelope unwraps, and the response_format nesting move "
            "are all part of this single claim; they are pinned by the L1 tests "
            "in ``tests/test_provider_opencode_responses.py`` and not by §3.3.1, "
            "because no path vocabulary names an envelope unwrap."
        ),
    ),
    MutationRow(
        id="P37",
        site=("kitty/providers/opencode.py:OpenCodeGoAdapter._cc_to_responses",),
        trigger=Trigger.CC_ORIGIN_PATH,
        # The eight CC sampling / control fields absent from the OpenAI
        # Responses create-request schema (verified 2026-09-16 against
        # ``openai/openai-openapi`` master). Verified at the spec, not from
        # the builder's source — a future spec addition turns this row red.
        # ``stream_options`` is included because Responses' ``stream_options``
        # has different semantics (``include_obfuscation``, not
        # ``include_usage``); the CC value is silently dropped, not translated.
        paths=tuple(c.extra_path(key) for key in (
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "logit_bias",
            "n",
            "stop",
            "logprobs",
            "stream_options",
        )),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · KBR-137",
        scope=("opencode_go",),
    ),
    MutationRow(
        id="P38",
        site=("kitty/providers/opencode.py:OpenCodeGoAdapter._cc_to_responses",),
        trigger=Trigger.CC_ORIGIN_PATH,
        # Rename the CC token-budget spellings to the Responses spelling.
        # Precedence (max_output_tokens > max_completion_tokens > max_tokens)
        # lives in the builder; the row claims the two CC addresses that
        # disappear, not the Responses address that already lives at its
        # own name.
        paths=(c.extra_path("max_tokens"), c.extra_path("max_completion_tokens")),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · KBR-137",
        scope=("opencode_go",),
    ),
    MutationRow(
        id="P42",
        site=("kitty/providers/opencode.py:OpenCodeGoAdapter._cc_to_responses",),
        trigger=Trigger.REASONING_EFFORT_PRESENT,
        # Passthrough of an agent signal in the target's own spelling, P3/P4
        # class. The CC value rides on ``_reasoning_effort`` (an internal key,
        # stripped by P1's ``_INTERNAL_KEYS``); the Responses address
        # ``envelope.extra[reasoning]`` is created here.
        paths=(c.extra_path("reasoning"),),
        conditional=True,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · KBR-137",
        scope=("opencode_go",),
    ),
    MutationRow(
        id="P43",
        site=(
            "kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",
            "kitty/providers/bedrock.py:BedrockAdapter.translate_to_upstream",
            "kitty/providers/ollama_cloud.py:OllamaCloudAdapter.translate_to_upstream",
        ),
        trigger=_ALWAYS,
        # KBR-305: drop the six KBR-301 sampling keys on the rebuild-trio.
        # The keys reach the CC body at hop 1 (KBR-301) and the verbatim
        # forwarders ship them, but the three rebuild-from-allowlist
        # adapters read only what their destination wire accepts:
        # Anthropic Messages accepts none of the six; Bedrock
        # ``InferenceConfiguration`` is exactly
        # ``{maxTokens, temperature, topP, stopSequences}``; Ollama
        # ``ChatRequest`` + ``Options`` accepts five of the six on the
        # wire (``seed``, ``presence_penalty``, ``frequency_penalty`` under
        # ``options``; ``logprobs`` and ``top_logprobs`` top-level) but has
        # no ``n``/``num_choices`` field anywhere on Ollama's Go source.
        # The split is named on the row in TEST_SUITE.md §3.2.2 so a
        # reader does not mistake the row for a blanket drop.
        #
        # ⚠️ Departure from the P31/P32 "paths must be true of every
        # site" rule, recorded: on ``ollama_cloud`` only ``n`` is
        # actually dropped; the five carried keys' claims are dormant on
        # that adapter. The five carries live in the mutmut scope per
        # TEST_SUITE.md §6.1 (``kitty.providers.* translate_to_upstream``),
        # so the L1 carry suite is the discriminating guard on that
        # route; the owner confirmed the one-row shape on 2026-09-23.
        #
        # The exact tuple is pinned by
        # ``test_register.py::test_row_ids_are_unique`` — a data-side drift
        # turns that assertion red. The L2 agreement guard compares
        # ids/conditionality/order but not path content, and the markdown
        # row's path list is held by review; the pin is what makes a
        # future widening a deliberate edit to both halves rather than a
        # silent one.
        paths=tuple(c.sampling_path(key) for key in (
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
        )),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1 · §3.3.1a · KBR-305",
        scope=_ANTHROPIC_FAMILY + ("bedrock", "ollama_cloud"),
    ),
)

#: The register. Ordered as §3.2 publishes it — bridge rows, then provider rows —
#: so a reader can hold the data and the document side by side.
REGISTER: tuple[MutationRow, ...] = _BRIDGE_ROWS + _PROVIDER_ROWS


# --------------------------------------------------------------------------
# Reading the design document
# --------------------------------------------------------------------------


class RegisterMarkdownError(AssertionError):
    """Raised when §3.2 cannot be read as a register.

    Every failure mode here is a *silent* one if it returns an empty result
    instead: a heading that moved, a table that was reformatted, or a range
    notation the parser cannot expand would all leave the agreement checks
    passing over nothing.  §6.2 forbids a guard that can rot into a no-op.
    """


@dataclass(frozen=True)
class ParsedRegister:
    """What §3.2 says, read out of the markdown.

    Attributes:
        live_ids: The ids of the rows that are in force, in document order.
        struck_ids: The ids struck through with ``~~``, in document order — a row
            kept for the record but withdrawn.
        unconditional_ids: The ids §3.2.2's closing paragraph exempts from
            §3.3.2 assertion 2.
    """

    live_ids: tuple[str, ...]
    struck_ids: tuple[str, ...]
    unconditional_ids: tuple[str, ...]


_SECTION_HEADINGS = ("#### 3.2.1", "#### 3.2.2")

#: A table row's first cell: an id, optionally struck through.
_ROW_ID = re.compile(r"^\|\s*(~~)?([MP]\d+[a-z]?)(~~)?\s*\|")

#: A line that is a table row of any kind, so a data row whose id cell the
#: parser cannot read is *refused* rather than skipped. §3.2's own "register
#: maintenance" paragraph promises that adding a mutation site without adding a
#: row fails these guards; the tables bold cells freely, so a contributor
#: bolding a new id (``| **M15** | … |``) is a plausible way to slip past a
#: parser that simply ignores what it does not match.
_TABLE_ROW = re.compile(r"^\|")

#: The header and separator rows of a register table, which carry no id and must
#: not be reported as unreadable.
_NOT_A_DATA_ROW = re.compile(r"^\|\s*(#|-{2,}|:?-)")

#: The sentence in §3.2.2 that enumerates the unconditional rows. Whitespace is
#: matched loosely because the document is hard-wrapped and a reflow must not
#: turn this guard into the no-op §6.2 forbids.
_UNCONDITIONAL_SENTENCE = re.compile(r"([^.]*?)\s+are\s+unconditional\s+by\s+design", re.DOTALL)

#: An id as it appears in that sentence.
_ID_TOKEN = re.compile(r"\b([MP]\d+[a-z]?)\b")

#: A range such as ``P9a–c``. Expanding one is guesswork, and dropping it
#: silently would drop rows from the comparison, so it is refused outright.
_ID_RANGE = re.compile(r"[MP]\d+[a-z]?\s*[–—-]\s*[a-z0-9]")


def _section(text: str, heading: str) -> str:
    """Return the body of one ``####`` section of the design document.

    Args:
        text: The full document.
        heading: The section's heading line prefix, e.g. ``#### 3.2.1``.

    Returns:
        Everything between that heading and the next ``####`` heading.

    Raises:
        RegisterMarkdownError: When the heading is absent.
    """
    start = text.find(heading)
    if start < 0:
        raise RegisterMarkdownError(f"{heading} is not in the document — §3.2's headings have moved")

    end = text.find("\n#### ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def _scan_table(text: str, heading: str) -> tuple[list[str], list[str]]:
    """Read one register table's ids, in document order.

    Args:
        text: The full document.
        heading: The section's heading line prefix, e.g. ``#### 3.2.1``.

    Returns:
        The section's live ids and its struck-through ids, each in the order the
        document lists them.

    Raises:
        RegisterMarkdownError: When a table row's id cell cannot be read, or when
            the section yields no live row at all.
    """
    live: list[str] = []
    struck: list[str] = []

    for line in _section(text, heading).splitlines():
        match = _ROW_ID.match(line)
        if match is None:
            # A table row that is not the header or separator and yields no id is
            # a row this parser cannot see. Skipping it would make the guard
            # silent about exactly the edit it exists to catch.
            if _TABLE_ROW.match(line) and not _NOT_A_DATA_ROW.match(line):
                raise RegisterMarkdownError(
                    f"{heading} has a table row whose id cell cannot be read: {line[:60]!r}. "
                    "An id is a bare M- or P-number in the first cell; formatting it hides the "
                    "row from this guard. A second table in this section -- a legend or a "
                    "summary -- trips this too, and needs its own section or a widening here."
                )
            continue
        (struck if match.group(1) else live).append(match.group(2))

    # Counting *live* rows, not rows. A table reformatted into something this
    # parser cannot read would still show one match on the struck M13 row, and a
    # section whose live rows have all become unreadable is exactly the silent
    # no-op §6.2 forbids.
    if not live:
        raise RegisterMarkdownError(f"{heading} yielded no live register rows — the table's shape has changed")

    return live, struck


def _unconditional_ids(section: str) -> tuple[str, ...]:
    """Read §3.2.2's list of rows exempt from §3.3.2 assertion 2.

    Args:
        section: The body of §3.2.2.

    Returns:
        The ids the closing paragraph names, in the order it names them.

    Raises:
        RegisterMarkdownError: When the sentence is gone, or when it abbreviates
            ids as a range instead of naming each one.
    """
    sentence_match = _UNCONDITIONAL_SENTENCE.search(section)
    if sentence_match is None:
        raise RegisterMarkdownError("§3.2.2 no longer states which rows are unconditional")

    sentence = sentence_match.group(1)

    # `P9a–c` names three rows in one token. Expanding it is guesswork and
    # skipping it drops two rows from the comparison, so the notation is refused.
    range_match = _ID_RANGE.search(sentence)
    if range_match is not None:
        raise RegisterMarkdownError(
            f"§3.2.2's unconditional list abbreviates {range_match.group(0)!r} as a range; "
            "name every id, or the rows inside the range are dropped from the comparison"
        )

    return tuple(dict.fromkeys(_ID_TOKEN.findall(sentence)))


def parse_register_markdown(text: str) -> ParsedRegister:
    """Read the register out of ``.system_design/TEST_SUITE.md``.

    The parser is deliberately strict.  Every shape it cannot read raises rather
    than returning a shorter list, because a shorter list makes
    :func:`register_disagreements` pass over rows it never saw.

    Args:
        text: The full text of the design document.

    Returns:
        The ids §3.2 publishes, split into live, struck and unconditional.

    Raises:
        RegisterMarkdownError: When a heading is missing, when a table yields no
            live rows, when a row's id cell cannot be read, when an id is
            published twice, when the unconditional sentence is absent, or when
            that sentence uses a range notation instead of naming every id.
    """
    live: list[str] = []
    struck: list[str] = []

    # Both tables, read in document order so `live_ids` matches the published one.
    for heading in _SECTION_HEADINGS:
        section_live, section_struck = _scan_table(text, heading)
        live += section_live
        struck += section_struck

    # An id is how every document, ticket and test refers to a row, so a repeat
    # makes the register ambiguous. Refused here rather than left to
    # `register_disagreements`, which compares by set and would either miss it or
    # report it as a confusing ordering problem.
    repeated = sorted(row_id for row_id, count in Counter(live + struck).items() if count > 1)
    if repeated:
        raise RegisterMarkdownError(f"§3.2 publishes these ids more than once: {repeated}")

    return ParsedRegister(
        live_ids=tuple(live),
        struck_ids=tuple(struck),
        unconditional_ids=_unconditional_ids(_section(text, _SECTION_HEADINGS[1])),
    )


def _first_divergence(data_order: Sequence[str], published: Sequence[str]) -> str:
    """Return the first id at which two orderings of the register differ.

    Total by construction, including when the two differ only in length.
    :func:`register_disagreements` must *return* problems rather than raise —
    its callers hand it damaged artifacts on purpose — so this must not be a
    strict pairing that dies on a length mismatch.

    Args:
        data_order: The ids in :data:`REGISTER` order.
        published: The ids in §3.2's order.

    Returns:
        The id naming the divergence: the first position where the two disagree,
        or the first id past the end of the shorter one.
    """
    for mine, theirs in zip(data_order, published, strict=False):
        if mine != theirs:
            return mine

    # One is a prefix of the other, so the divergence is the first id past it.
    longer = data_order if len(data_order) > len(published) else published
    return longer[min(len(data_order), len(published))]


def register_disagreements(rows: tuple[MutationRow, ...], markdown: str) -> tuple[str, ...]:
    """Report every way the register data and the design document differ.

    Pure, and reported as a tuple of messages rather than raised, so a test can
    hand this function a deliberately damaged copy of either side and assert the
    damage is named.

    Args:
        rows: The register data, normally :data:`REGISTER`.
        markdown: The full text of ``.system_design/TEST_SUITE.md``.

    Returns:
        One message per disagreement, in a stable order.  Empty when the two
        agree.

    Raises:
        RegisterMarkdownError: When the document cannot be read at all.
    """
    parsed = parse_register_markdown(markdown)
    problems: list[str] = []

    by_id = {row.id: row for row in rows}
    published = set(parsed.live_ids)

    # Membership, both directions. A row in one artifact and not the other is the
    # failure plan §3 names as this task's falsification case.
    for row_id in parsed.live_ids:
        if row_id not in by_id:
            problems.append(f"{row_id}: published in TEST_SUITE.md §3.2 but missing from the register data")
    struck = set(parsed.struck_ids)
    for row in rows:
        # A struck row IS published, so the two branches are alternatives, not
        # both. Reporting "not published" alongside "struck through" named one
        # problem twice and the first message was false.
        if row.id in struck:
            problems.append(f"{row.id}: struck through in TEST_SUITE.md §3.2 but still live in the register data")
        elif row.id not in published:
            problems.append(f"{row.id}: in the register data but not published in TEST_SUITE.md §3.2")

    # Classification. §3.3.2 assertion 2 applies to exactly the rows the document
    # does *not* list as unconditional, so a disagreement here means a row is
    # owed a corpus complement that nobody knows to write.
    exempt = set(parsed.unconditional_ids)
    for row_id in sorted(exempt - published):
        problems.append(f"{row_id}: listed as unconditional in §3.2.2 but is not a live register row")
    for row in rows:
        if row.id not in published:
            continue
        if row.conditional == (row.id in exempt):
            document_says = "unconditional" if row.id in exempt else "conditional"
            data_says = "conditional" if row.conditional else "unconditional"
            problems.append(f"{row.id}: §3.2.2 calls it {document_says}, the register data calls it {data_says}")

    # Order, but only once membership agrees -- otherwise a single deleted row
    # renumbers everything after it and buries the real problem under noise.
    #
    # Not `zip(strict=True)`: this function's contract is to *return* problems so
    # a test can hand it a damaged artifact, and a raise breaks that contract.
    # Equal lengths follow from set equality plus uniqueness on both sides, but
    # deriving safety from an invariant proved elsewhere is how the guard starts
    # crashing the day one of them moves.
    data_order = tuple(row.id for row in rows)
    if not problems and data_order != parsed.live_ids:
        problems.append(
            f"{_first_divergence(data_order, parsed.live_ids)}: the register data is in a different "
            "order from TEST_SUITE.md §3.2 — the document interleaves P20 and P21 between P6 and "
            "P7, and the data follows it so the two can be read side by side"
        )

    return tuple(problems)


# --------------------------------------------------------------------------
# Reading the source tree
# --------------------------------------------------------------------------


def _collect_bindings(node: ast.AST, prefix: str, relative: str, into: set[str]) -> None:
    """Record the names one scope binds *directly*, recursing into nested scopes.

    **Directly** is the limitation worth stating: a ``def`` or assignment inside a
    module-level ``if TYPE_CHECKING:`` or ``try:`` is a child of that block, not
    of the module, so this walker does not see it.

    That is deliberate, and widening it would make the guard weaker rather than
    stronger.  Recursing through conditional blocks everywhere would add the
    **1,386** assignments that live inside ``if``/``try``/``with`` bodies in
    ``src/kitty`` — almost all of them function-local variables — and a register
    row could then "resolve" against a local. Restricting the widening to module
    and class level avoids that, and buys nothing: **zero** module- or
    class-level bindings in ``src/kitty`` are currently hidden inside such a
    block. Both numbers were measured, not assumed.

    The failure direction is the safe one if that ever changes. A row pointing at
    a conditionally-defined symbol is reported *unresolved* — a loud, false
    failure a maintainer can read — rather than silently accepted.

    Args:
        node: The scope to walk.
        prefix: The qualified name of that scope, empty at module level.
        relative: The file's path under ``src``, which prefixes every address.
        into: The set to add addresses to.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            qualified = f"{prefix}.{child.name}" if prefix else child.name
            into.add(f"{relative}:{qualified}")
            _collect_bindings(child, qualified, relative, into)
        elif isinstance(child, ast.Assign | ast.AnnAssign):
            # A register row may name a constant -- P1's site is `_INTERNAL_KEYS`.
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    qualified = f"{prefix}.{target.id}" if prefix else target.id
                    into.add(f"{relative}:{qualified}")


def defined_symbols(src_root: Path) -> frozenset[str]:
    """Return every symbol defined under ``src``, addressed as a register site.

    Read with :mod:`ast`, never by importing ``kitty``: the register specifies
    what the code may do, and a specification that imports its subject can only
    ever agree with it.

    Args:
        src_root: The ``src`` directory, whose children are the import roots.

    Returns:
        Names of the form ``<path under src>:<qualified name>`` — for example
        ``kitty/bridge/server.py:BridgeServer._normalize_model``.  Classes,
        functions, methods and module- or class-level assignments, since a
        register row may name a constant such as ``_INTERNAL_KEYS``.
    """
    symbols: set[str] = set()

    for path in sorted(src_root.rglob("*.py")):
        relative = path.relative_to(src_root).as_posix()
        _collect_bindings(ast.parse(path.read_text(encoding="utf-8")), "", relative, symbols)

    return frozenset(symbols)


def unresolved_sites(rows: tuple[MutationRow, ...], symbols: frozenset[str]) -> tuple[str, ...]:
    """Report every register site that names no symbol in the source tree.

    This is what makes :attr:`MutationRow.site` data rather than decoration.  A
    mutation site renamed without the register following it leaves a row pointing
    at nothing while the register goes on looking complete.

    Takes the symbol set rather than a path, so the comparison is pure and the
    tree is parsed once per test module instead of once per assertion.  The I/O
    lives in :func:`defined_symbols` alone.

    Args:
        rows: The register data, normally :data:`REGISTER`.
        symbols: The output of :func:`defined_symbols`.

    Returns:
        One message per unresolved site, in register order.  Empty when every
        site resolves.
    """
    return tuple(
        f"{row.id}: site {site!r} names no symbol in the source tree"
        for row in rows
        for site in row.site
        if site not in symbols
    )


# --------------------------------------------------------------------------
# The scope guard (KBR-139)
# --------------------------------------------------------------------------


class RegisterSourceError(AssertionError):
    """Raised when ``providers/registry.py`` cannot be read as a registry.

    Every failure mode here is a *silent* one if it returns an empty result
    instead: a renamed ``_registry``, a dict rebuilt by a function call, or a
    non-literal key would all leave :func:`scope_problems` passing over an
    empty key set — and an empty key set rejects *nothing*, which is exactly
    the no-op §6.2 forbids.  Mirrors :class:`RegisterMarkdownError`.
    """


def provider_registry(src_root: Path) -> dict[str, str]:
    """Return the provider registry as ``key -> adapter class name``, read by AST.

    The scope data claims knowledge of :data:`providers.registry._registry`'s
    keys, and a specification that imported its subject could only ever agree
    with it (the module docstring's rule, §3.3.1's independent-oracle rule) —
    so the dict is read as **text** through :mod:`ast`, exactly as
    :func:`defined_symbols` reads the rest of ``src/kitty``.

    Args:
        src_root: The ``src`` directory, whose children are the import roots.

    Returns:
        The registry mapping, e.g. ``{"anthropic": "AnthropicAdapter", …}``.

    Raises:
        RegisterSourceError: When ``kitty/providers/registry.py`` carries no
            module-level ``_registry`` dict literal, when a key or class name
            cannot be read as a literal, or when the dict is empty — never an
            empty result, which would make the scope guard vacuous.
    """
    path = src_root / "kitty" / "providers" / "registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_registry"
            and isinstance(node.value, ast.Dict)
        ):
            registry: dict[str, str] = {}
            for key_node, class_node in zip(node.value.keys, node.value.values, strict=True):
                if key_node is None:
                    raise RegisterSourceError(
                        "registry.py's _registry contains a **-unpacking entry "
                        "— every key must be a string literal so the AST reader "
                        "can give it to scope_problems"
                    )
                try:
                    key = ast.literal_eval(key_node)
                except ValueError as exc:
                    raise RegisterSourceError(
                        f"registry.py's _registry key {ast.dump(key_node)!r} is not "
                        f"a string literal ({exc.__class__.__name__})"
                    ) from exc
                if not isinstance(key, str):
                    raise RegisterSourceError(
                        f"registry key {key_node!r} is not a string literal"
                    )
                if not isinstance(class_node, ast.Name):
                    raise RegisterSourceError(
                        f"registry value for {key!r} is not a bare class name"
                    )
                registry[key] = class_node.id
            if not registry:
                raise RegisterSourceError("registry.py's _registry dict literal is empty")
            return registry

    raise RegisterSourceError(
        "kitty/providers/registry.py no longer carries a module-level "
        "_registry dict literal — the scope guard cannot read it"
    )


def scope_problems(
    rows: tuple[MutationRow, ...],
    registry: dict[str, str],
    symbols: frozenset[str],
) -> tuple[str, ...]:
    """Report every way the rows' ``scope`` values are malformed.

    Pure, and separate from the loop that applies it, so a deliberately bad row
    can be handed to it — plan §1.4 requires the falsification case to run in
    the suite.  Mirrors :func:`row_shape_problems` and :func:`unresolved_sites`:
    return problems, never raise, so a test can hand this function a damaged
    register on purpose.

    Three checks per row, in order:

    1. **Shape** — the scope names at least one entry, and the
       :data:`ALL_PROVIDERS` sentinel is never mixed with keys (a mixed tuple
       is two claims in one cell; whichever the reader honours, the other is a
       lie).
    2. **Key validity** — every non-sentinel entry is a real key of the
       AST-read registry.  This is the deliverable's guard: a scope entry
       naming a provider that does not exist is data nothing could ever
       contradict on a run.
    3. **Site ↔ scope** — every registry key whose adapter class is defined in
       a file one of the row's sites names must appear in the scope.  This is
       the cheap direction of the P8/P5a mismatch: a row whose site names
       ``mimo.py`` cannot silently claim a scope without ``mimo``.  The check
       is file-level; the class-aware tightening (a site naming one class of a
       multi-class file, e.g. ``zai.py``, forcing only that class's key) is
       recorded in the step file as a future change, declined while no live row
       exercises the divergence.

    Args:
        rows: The register rows to check, normally :data:`REGISTER` or a
            deliberately damaged subset.
        registry: The output of :func:`provider_registry` — ``key -> class
            name``.
        symbols: The output of :func:`defined_symbols`, used to locate which
            file defines each registry class.

    Returns:
        One message per problem, in row order.  Empty when every scope is well
        formed.
    """
    # Map each registry class name to the providers file that defines it, so a
    # site path like `kitty/providers/mimo.py:…` resolves to the keys that must
    # be in scope.  A class entry is `path:ClassName` with no dot; methods and
    # nested names carry dots and are skipped.
    file_of_key: dict[str, str] = {}
    # `sorted` so the mapping is deterministic even if two provider files
    # ever define the same class name — frozenset iteration order is
    # hash-seeded, and the guard's verdict must not vary with it.
    sorted_symbols = sorted(symbols)
    for key, class_name in registry.items():
        for symbol in sorted_symbols:
            path, separator, qualified = symbol.partition(":")
            if separator and qualified == class_name and path.startswith("kitty/providers/"):
                file_of_key[key] = path
                break

    problems: list[str] = []
    for row in rows:
        # Shape: non-empty, sentinel never mixed with keys.
        if not row.scope:
            problems.append(f"{row.id}: names no scope (KBR-139)")
            continue
        if ALL_PROVIDERS in row.scope and len(row.scope) > 1:
            problems.append(
                f"{row.id}: mixes the {ALL_PROVIDERS!r} sentinel with keys "
                f"{[k for k in row.scope if k != ALL_PROVIDERS]!r} — the sentinel "
                "is a single-element tuple or nothing"
            )

        # Key validity: every non-sentinel entry names a real registry key.
        for key in row.scope:
            if key != ALL_PROVIDERS and key not in registry:
                problems.append(
                    f"{row.id}: scope names {key!r}, which is no key of "
                    "providers.registry._registry"
                )

        # Site ↔ scope: keys whose class file a site names must be in scope.
        # The sentinel claims every adapter by construction, so a row carrying
        # it is exempt from the file-level subset by design — the subset would
        # be the whole key set.
        if ALL_PROVIDERS in row.scope:
            continue
        site_files = {site.partition(":")[0] for site in row.site}
        required = {
            key for key, path in file_of_key.items() if path in site_files
        }
        missing = sorted(required - set(row.scope))
        if missing:
            problems.append(
                f"{row.id}: site names {sorted(site_files & set(file_of_key.values()))!r}, "
                f"which define {missing!r}, but scope omits them"
            )

    return tuple(problems)
