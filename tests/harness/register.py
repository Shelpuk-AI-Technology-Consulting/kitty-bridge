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
A trigger is a *route* property or a *request* property, and only the second kind
can be varied by a corpus entry.

**There is deliberately no scope column, and the site does not supply one.**
§6.2.3's completeness guard and T-D8's coverage check both need to know, per
adapter, which rows are reachable.  The register does not carry it, and the
tempting shortcut — read the scope off the site's class — is wrong twice:

* P8's site is ``ProviderAdapter._inject_empty_reasoning_content``, a **base
  class** method.  The site reads as all 23 adapters; only four call it
  (``kimi``, ``custom_openai``, ``zai_regular``, ``zai_coding_cc``).
  Over-scoping there would oblige T-D8 to demand nineteen complement cases that
  cannot exist.
* P5a–d's site is ``AnthropicAdapter.translate_to_upstream``, which four
  subclasses override *and conditionally delegate back to*, so the row is
  reachable on ``custom_anthropic``, ``zai_coding``, ``minimax_token`` and
  ``opencode_go`` as well.  No static rule over the class hierarchy finds that.

Scope is therefore its own task, filed rather than guessed.  Authoring it here
would ship data that **nothing in this change could prove wrong** — there is no
wire-level capture yet to contradict a bad entry — which is exactly what plan
§1.4's harness rule forbids.  See `KBR-139`.

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
wire-level guard (§6.2.3, T-G2) can catch that.  Four omissions are already known and filed —
`KBR-148` (headers), `KBR-149` (`openai_subscription` injecting `reasoning` from
`_reasoning_effort`, which P4 cannot cover because `translate_to_upstream` never runs on that
adapter's request path), `KBR-184` (P13's CC-origin twin) and `KBR-185` (an allowlisted field
dropped for being falsy).  Every one was found by reading the code by hand; none was found by a
guard.  Do not read a green suite as "the register is the whole truth".

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


class Trigger(Enum):
    """The conditions under which a registered mutation is permitted to fire.

    Closed deliberately, for the reason :class:`~harness.contract.WireFormat` is
    closed: T-W6 indexes the corpus by these names and T-D8 fails when a
    conditional row has no entry, so a free-form string would let two authors
    spell one condition two ways and leave a row silently uncovered.

    :attr:`ALWAYS` is the absence of a condition, not a condition — a row
    carrying it fires on every request that reaches its site.
    """

    ALWAYS = "always"

    # Bridge-level, request path.
    PROFILE_SETS_MODEL = "profile_sets_model"
    NON_NATIVE_UPSTREAM_WIRE = "non_native_upstream_wire"
    TOOL_RESULT_OVER_LIMIT = "tool_result_over_limit"
    COMPACTION_RAN_WITH_OVERSIZED_TOOL_RESULT = "compaction_ran_with_oversized_tool_result"
    OVER_COMPACTION_BUDGET = "over_compaction_budget"
    UPSTREAM_REJECTED_OVERSIZED_ON_BALANCING = "upstream_rejected_oversized_on_balancing"
    ORPHAN_TOOL_RESULT = "orphan_tool_result"
    THINKING_ROUNDTRIP_REJECTED = "thinking_roundtrip_rejected"
    NATIVE_TOOL_USE_FORMAT_ERROR = "native_tool_use_format_error"
    GEMINI_PROTOCOL = "gemini_protocol"
    GEMINI_NON_STREAMING = "gemini_non_streaming"

    # Bridge-level, response path.
    UPSTREAM_EMPTY_RESPONSE = "upstream_empty_response"

    # Provider-level.
    ZAI_THINKING_ENABLED = "zai_thinking_enabled"
    ZAI_THINKING_DISABLED = "zai_thinking_disabled"
    REASONING_EFFORT_PRESENT = "reasoning_effort_present"
    MAX_TOKENS_ABSENT = "max_tokens_absent"
    MULTIPLE_SYSTEM_BLOCKS = "multiple_system_blocks"
    ANTHROPIC_THINKING_ENABLED = "anthropic_thinking_enabled"
    ADAPTIVE_THINKING_KEYS_PRESENT = "adaptive_thinking_keys_present"
    ASSISTANT_TURN_LACKS_THINKING_BLOCK = "assistant_turn_lacks_thinking_block"
    NON_STREAMING_MAX_TOKENS_OVER_4096 = "non_streaming_max_tokens_over_4096"
    THINKING_SIGNALLED_OR_INFERRED = "thinking_signalled_or_inferred"
    CC_ORIGIN_PATH = "cc_origin_path"
    RESPONSES_ORIGIN_PATH = "responses_origin_path"


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
    not_projectable_reason: str | None = None

    @property
    def is_projectable(self) -> bool:
        """Return whether the projection models this row's effect.

        Returns:
            ``False`` when the row carries the
            :data:`~harness.contract.NOT_PROJECTABLE` escape.
        """
        return self.paths != (c.NOT_PROJECTABLE,)


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
#: claimed here. That residue is G27 / `KBR-185`.
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
        not_projectable_reason=(
            "A whole-body protocol translation changes the wire format, not a field. The "
            "projection exists precisely so the two formats become comparable, so naming a path "
            "here would claim every delta on every translated request and make the oracle blind."
        ),
    ),
    MutationRow(
        id="M3",
        site=(f"{_SERVER}:BridgeServer._truncate_oversized_tool_results",),
        trigger=Trigger.TOOL_RESULT_OVER_LIMIT,
        # The truncated content lives in one ToolResult part. Anchoring at
        # `conversation.turns` would claim a dropped turn as well.
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.1",
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
    ),
    MutationRow(
        id="M6",
        site=(
            f"{_SERVER}:BridgeServer._compact_with_tighter_budget",
            f"{_SERVER}:BridgeServer._request_with_retry_balancing",
        ),
        trigger=Trigger.UPSTREAM_REJECTED_OVERSIZED_ON_BALANCING,
        paths=(c.CONVERSATION_TURNS,),
        conditional=True,
        design_ref="§3.2.1 · §4.3 C3",
    ),
    MutationRow(
        id="M7",
        site=(f"{_SERVER}:BridgeServer._validate_tool_call_pairing",),
        trigger=Trigger.ORPHAN_TOOL_RESULT,
        # Dropping an orphan renumbers the parts after it and can empty a turn,
        # so the collection is again the narrowest anchor that survives.
        paths=(c.CONVERSATION_TURNS,),
        conditional=True,
        design_ref="§3.2.1",
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
    ),
    MutationRow(
        id="M9",
        site=(f"{_SERVER}:_convert_native_to_cc_format",),
        trigger=Trigger.NATIVE_TOOL_USE_FORMAT_ERROR,
        paths=(c.NOT_PROJECTABLE,),
        conditional=True,
        design_ref="§3.2.1",
        not_projectable_reason=(
            "A whole-body conversion from native Messages to Chat Completions, followed by a "
            "re-run of model normalisation. Like M2 it changes the format rather than a field, "
            "and the projection is what makes the before and after comparable at all."
        ),
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
    ),
    MutationRow(
        id="M11",
        site=(f"{_SERVER}:BridgeServer._handle_gemini",),
        trigger=Trigger.GEMINI_NON_STREAMING,
        paths=(c.ENVELOPE_STREAM,),
        conditional=True,
        design_ref="§3.2.1",
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
        paths=(c.reply_part_path(0),),
        conditional=True,
        design_ref="§3.2.1 · §3.3.1",
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
        # Three anchors because Anthropic permits a breakpoint at three carriers
        # and Claude Code uses all three. Each names the **field**, never the
        # block: `conversation.turns[*].parts[*]` would also claim a deleted
        # part, and `conversation.tools[*]` a deleted tool description -- two of
        # §3.3.1's five oracle falsification cases. That is §3.3.1a's P15 lesson
        # applied to a second row.
        paths=(
            c.system_path(c.WILDCARD, "cache_control"),
            c.part_path(c.WILDCARD, c.WILDCARD, "cache_control"),
            c.tool_path(c.WILDCARD, "cache_control"),
        ),
        conditional=False,
        design_ref="§3.2.1 · §3.3.1 · §3.3.1a",
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
    ),
    MutationRow(
        id="P3",
        site=("kitty/providers/openrouter.py:OpenRouterAdapter.translate_to_upstream",),
        trigger=Trigger.REASONING_EFFORT_PRESENT,
        paths=(c.extra_path("reasoning"),),
        conditional=True,
        design_ref="§3.2.2",
    ),
    MutationRow(
        id="P4",
        site=("kitty/providers/openai.py:OpenAIAdapter.translate_to_upstream",),
        trigger=Trigger.REASONING_EFFORT_PRESENT,
        # OpenAI's spelling of the signal P3 carries; same trigger, different key.
        paths=(c.extra_path("reasoning_effort"),),
        conditional=True,
        design_ref="§3.2.2",
    ),
    MutationRow(
        id="P5a",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.MAX_TOKENS_ABSENT,
        paths=(c.sampling_path("max_tokens"),),
        conditional=True,
        design_ref="§3.2.2",
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
    ),
    MutationRow(
        id="P5c",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.ANTHROPIC_THINKING_ENABLED,
        # Two effects, two paths: it raises the agent's own max_tokens (the
        # user-visible half) and sets the thinking budget.
        paths=(c.sampling_path("max_tokens"), c.extra_path("thinking")),
        conditional=True,
        design_ref="§3.2.2",
    ),
    MutationRow(
        id="P5d",
        site=("kitty/providers/anthropic.py:AnthropicAdapter.translate_to_upstream",),
        trigger=Trigger.ADAPTIVE_THINKING_KEYS_PRESENT,
        paths=(c.extra_path("thinking"), c.extra_path("effort")),
        conditional=True,
        design_ref="§3.2.2",
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
    ),
    MutationRow(
        id="P7",
        site=("kitty/providers/fireworks.py:FireworksAdapter.normalize_request",),
        trigger=Trigger.NON_STREAMING_MAX_TOKENS_OVER_4096,
        paths=(c.sampling_path("max_tokens"),),
        conditional=True,
        design_ref="§3.2.2",
    ),
    MutationRow(
        id="P8",
        site=(f"{_BASE}:ProviderAdapter._inject_empty_reasoning_content",),
        trigger=Trigger.THINKING_SIGNALLED_OR_INFERRED,
        paths=(c.part_path(c.WILDCARD, c.WILDCARD),),
        conditional=True,
        design_ref="§3.2.2",
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
        # conditional `ChatGPT-Account-Id` this site also sets needs its own row
        # and its own complement fixture — gap G22.
        paths=(c.header_path("user-agent"), c.header_path("version"), c.header_path("accept")),
        conditional=False,
        design_ref="§3.2.2 · §4.3 C1",
    ),
    MutationRow(
        id="P10",
        site=("kitty/providers/minimax.py:MiniMaxAdapter.normalize_request",),
        trigger=_ALWAYS,
        paths=(c.extra_path("reasoning_split"),),
        conditional=False,
        design_ref="§3.2.2",
    ),
    MutationRow(
        id="P11",
        site=("kitty/providers/bedrock.py:BedrockAdapter.translate_to_upstream",),
        trigger=_ALWAYS,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.4",
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
        # the `reasoning` injection §9.2's G23 registers at
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
        # `P22` is skipped deliberately -- §9.2's G23 reserves it for KBR-149,
        # and an id is how every ticket refers to a row.
        paths=tuple(c.extra_path(key) for key in _CODEX_DROPPED_CONTROL_FIELDS),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1a · §3.3.1b",
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
    ),
    MutationRow(
        id="P16",
        site=(f"{_SUBSCRIPTION}:_convert_content_types",),
        trigger=Trigger.RESPONSES_ORIGIN_PATH,
        paths=(c.NOT_PROJECTABLE,),
        conditional=False,
        design_ref="§3.2.2 · §3.3.1a",
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
    ),
    MutationRow(
        id="P18",
        site=(
            "kitty/providers/bedrock.py:BedrockAdapter.make_request",
            "kitty/providers/bedrock.py:BedrockAdapter.stream_request",
        ),
        trigger=_ALWAYS,
        # §3.3.1b normalises Converse's `modelId` onto `envelope.model`, so the
        # pop is expressible. Applied in the transport, after the hook — §3.2.3
        # says the capture boundary is after this mutation, not before.
        paths=(c.ENVELOPE_MODEL, c.ENVELOPE_STREAM),
        conditional=False,
        design_ref="§3.2.2 · §3.2.3 · §3.3.1b",
    ),
    MutationRow(
        id="P19",
        site=(
            "kitty/providers/ollama_cloud.py:OllamaCloudAdapter.make_request",
            "kitty/providers/ollama_cloud.py:OllamaCloudAdapter.stream_request",
        ),
        trigger=_ALWAYS,
        paths=(c.ENVELOPE_STREAM,),
        conditional=False,
        design_ref="§3.2.2 · §3.2.3",
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
