"""Hypothesis strategies that generate valid Anthropic Messages, Chat Completions, and OpenAI Responses transcripts.

`.system_design/TEST_SUITE.md` §6.1 · plan **T-F1** (KBR-70) with the
OpenAI Responses addition for **T-F3** (KBR-72).

The L1 suite today is entirely example-based; the §6.1 property list (compaction,
pairing, truncation, egress, anomaly, translator) needs property-based tests
running against pure logic. Property tests for the *transcript-shaped* targets —
``MessagesTranslator`` (T-F6), ``_compact_messages`` (T-F2),
``_validate_tool_call_pairing`` and ``_truncate_oversized_tool_results`` (T-F3) —
compose scenarios from valid request bodies, and they need to vary far enough
through the space that one example test could not cover. The Responses
strategies ship with KBR-72 to support the L1 properties for the Responses
twins of pairing and truncation (``BridgeServer._drop_orphan_responses_tool_outputs``
and ``BridgeServer._truncate_oversized_responses_outputs``, register rows M7 / M3),
whose wire rule mirrors the CC / native pair and whose inputs are valid
Responses request bodies. This module provides the composable building blocks:
whole-request strategies, single-message strategies, single-block strategies,
and tool-definition strategies, plus a per-format ``problems()`` reporter that
states what "valid" means so a generated body can be handed back and asserted
well-formed.

**Out of scope (recording the boundary here so it does not get re-litigated).**

T-F4's egress properties (``should_bypass``, ``parse_proxy_url``, ``EgressConfig``)
and T-F5's ``describe_tool_input_anomaly`` property do **not** consume this
module — they need address / hostname / proxy-URL strategies (T-F4) and
JSON-schema-conforming input strategies (T-F5). Those tasks ship their own local
strategies; pre-empting their design here would couple unrelated work to a
substrate that does not exist when they start. The plan's row for T-F1 names
only ``hypothesis`` + shared transcript strategies, and the ticket's literal
"Done when" reads "the strategies are reusable by T-F2–T-F5 without each
redefining its own generators" — the relevant "generator" set for the
transcript-shaped targets is this module, and the others' targets open their
own design.

**What "valid" means here, in the way the cache_breakpoints fixture records its
own rule boundaries in its docstring.**

The reporter enforces shape, required fields, role alternation, tool-use / tool-
result pairing, and JSON-strict serialisability. It does **not** flag oversize
content or over-budget conversations — size constraints belong to the consuming
property test (T-F2, T-F3) and asserting them in the substrate would forbid the
"irreducible surviving set" case T-F2's property explicitly excludes (§6.1).
Invalid bodies — the orphan ``tool_result`` T-F3 exercises its pairing property
on — are produced downstream by **mutating** valid bodies; the shared strategies
never emit them.

**It imports nothing from ``src/kitty``**, the same posture
``cache_breakpoints.py`` and ``contract.py`` take. The §3.3.1 independence rule
is the single structural guarantee the I1 fidelity oracle rests on, and the
substrate for property tests stands or falls on the same rule: a property test
whose generator shared assumptions with the code under test would prove self-
consistency rather than the property.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import hypothesis.strategies as st

__all__ = [
    "ANTHROPIC_MODELS",
    "OPENAI_MODELS",
    "CONTENT_SHAPES",
    "ContentShape",
    # Anthropic Messages strategies
    "messages_text_block",
    "messages_image_block",
    "messages_document_block",
    "messages_tool_definition",
    "messages_tool_use_block",
    "messages_tool_result_block",
    "messages_user_message",
    "messages_assistant_message",
    "messages_tool_pair",
    "messages_request",
    "messages_problems",
    # Chat Completions strategies
    "cc_text_part",
    "cc_image_part",
    "cc_tool_definition",
    "cc_tool_call",
    "cc_tool_message",
    "cc_user_message",
    "cc_assistant_message",
    "cc_tool_pair",
    "cc_request",
    "cc_problems",
    # OpenAI Responses strategies (T-F3 / KBR-72)
    "responses_tool_definition",
    "responses_function_call_item",
    "responses_function_call_output_item",
    "responses_message_item",
    "responses_request",
    "responses_problems",
]


#: The two ``content`` shapes Anthropic Messages and Chat Completions accept:
#: a plain string (only valid for text-only turns) and a list of content
#: blocks / parts. The parameter is exposed on every message- and request-level
#: strategy, and AC-5 exercises both — the §6.1 ``_validate_tool_call_pairing``
#: row pins pairing against "both message shapes".
ContentShape = Literal["string", "blocks"]
CONTENT_SHAPES: tuple[ContentShape, ...] = ("string", "blocks")

#: Model identifiers the strategies pick from. Two per format, so the
#: generators vary the model at least as much as the property tests need —
#: every property test asserts the model's effect (e.g. compaction budget,
#: routing) so the model must move, but the identity is not the property; two
#: is enough variety.
ANTHROPIC_MODELS: tuple[str, ...] = ("claude-sonnet-5", "claude-opus-5")
OPENAI_MODELS: tuple[str, ...] = ("gpt-4o", "gpt-4o-mini")


# A real 1x1 RGBA PNG: its chunk CRCs were verified by hand
# (``harness/cache_breakpoints.py``), so the Anthropic and OpenAI APIs can
# decode it. The byte sequence is duplicated rather than imported across
# harness modules, because the two fixtures each carry their own promise and
# importing a leading-underscore name across modules would couple them.
_PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _short_ascii_text(min_size: int = 0, max_size: int = 16) -> st.SearchStrategy[str]:
    """Return a strategy over short ASCII text strings.

    Args:
        min_size: The minimum character count.
        max_size: The maximum character count.

    Returns:
        A Hypothesis strategy that yields ASCII strings of length in
        ``[min_size, max_size]``. Default bounds keep block text short enough
        that 200 property examples finish in milliseconds; a downstream test
        that wants oversize text draws directly from ``st.text``.
    """
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Zs"),
            max_codepoint=0x7E,
        ),
        min_size=min_size,
        max_size=max_size,
    )


def _name_strategy() -> st.SearchStrategy[str]:
    """Return a strategy over short tool-name identifiers.

    Returns:
        ``[A-Za-z0-9]{1,12}`` — the alphabet Anthropic and OpenAI both accept
        in ``name`` fields without escaping.
    """
    return st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"), max_codepoint=0x7E
        ),
        min_size=1,
        max_size=12,
    )


def _path_schema() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over small JSON-Schema tool-input shapes.

    Returns:
        One of two schemas, either ``{"path": <str>}`` or
        ``{"path": <str>, "line": <int>}``, with the ``required`` list drawn
        independently per schema variant. Both Anthropic and OpenAI accept this
        subset directly.
    """
    return st.one_of(
        st.builds(
            lambda required: (
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                }
                if required
                else {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                }
            ),
            required=st.booleans(),
        ),
        st.builds(
            lambda required: (
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer"},
                    },
                    "required": ["path"],
                }
                if required
                else {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": "integer"},
                    },
                }
            ),
            required=st.booleans(),
        ),
    )


# ── Anthropic Messages: block-level strategies ──────────────────────────────


def messages_text_block() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Anthropic text content blocks.

    Returns:
        A Hypothesis strategy that yields ``{"type": "text", "text": <str>}``.
    """
    return st.builds(
        lambda text: {"type": "text", "text": text},
        text=_short_ascii_text(),
    )


def messages_image_block() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Anthropic image content blocks carrying a real PNG.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": <b64>}}``.
    """
    return st.just(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _PNG_1X1,
            },
        }
    )


def messages_document_block() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Anthropic document content blocks.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": <str>}}``.
    """
    return st.builds(
        lambda text: {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": text,
            },
        },
        text=_short_ascii_text(),
    )


def messages_tool_definition() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Anthropic tool declarations.

    Returns:
        A Hypothesis strategy that yields
        ``{"name": <str>, "description": <str>, "input_schema": {...}}``.
    """
    return st.builds(
        lambda n, d, s: {"name": n, "description": d, "input_schema": s},
        n=_name_strategy(),
        d=_short_ascii_text(min_size=1),
        s=_path_schema(),
    )


def messages_tool_use_block(
    tools: list[dict[str, Any]],
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Anthropic ``tool_use`` blocks naming one of ``tools``.

    Args:
        tools: The request's tool declarations. The strategy picks ``name``
        uniformly from this list so the generated block is always addressable.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "tool_use", "id": <unique>, "name": <name>, "input": {}}``.

    Raises:
        ValueError: When ``tools`` is empty — an unaddressable ``tool_use`` is
            an I1 violation the substrate must not emit, so the strategy is
            honest about the precondition rather than picking a fake name.
    """
    if not tools:
        raise ValueError(
            "messages_tool_use_block requires at least one declared tool; an "
            "unaddressable tool_use is an I1 violation the substrate does not emit."
        )
    return st.builds(
        lambda tool, unique: {
            "type": "tool_use",
            "id": unique,
            "name": tool["name"],
            "input": {},
        },
        tool=st.sampled_from(tools),
        unique=st.uuids().map(str),
    )


def messages_tool_result_block(
    tool_use_id: str,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Anthropic ``tool_result`` blocks answering ``tool_use_id``.

    Args:
        tool_use_id: The ``id`` of the ``tool_use`` block this result answers.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "tool_result", "tool_use_id": <id>, "content": <str>}``.
        Content is a string (the simplest well-formed case the API accepts);
        richer shapes are produced by composing blocks downstream.
    """
    return st.builds(
        lambda text: {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": text,
        },
        text=_short_ascii_text(),
    )


# ── Anthropic Messages: message- and pair-level strategies ─────────────────


def messages_user_message(
    text_only: bool,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over plain user messages (no ``tool_use`` or ``tool_result``).

    Args:
        text_only: When ``True``, ``content`` is a plain string. When
        ``False``, ``content`` is a list of one to three text / image /
        document blocks. Anthropic accepts both shapes for plain user turns;
        this parameter is the AC-5 hook — pairing is asserted across both.

    Returns:
        A Hypothesis strategy that yields ``{"role": "user", "content": ...}``.
    """
    if text_only:
        return st.builds(
            lambda text: {"role": "user", "content": text},
            text=_short_ascii_text(),
        )
    block = st.one_of(messages_text_block(), messages_image_block(), messages_document_block())
    return st.builds(
        lambda blocks: {"role": "user", "content": blocks},
        blocks=st.lists(block, min_size=1, max_size=3),
    )


def messages_assistant_message(
    tools: list[dict[str, Any]] | None = None,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over plain assistant messages (no ``tool_use``).

    Args:
        tools: Unused. Present for symmetry with the pair-level strategy and
        so a caller that conditions on tools can pass them through.

    Returns:
        A Hypothesis strategy that yields
        ``{"role": "assistant", "content": [{"type": "text", "text": ...}]}``.
        Always a block list — Anthropic accepts both shapes for assistant
        turns, but the property tests for translators and compaction see
        ``tool_use`` blocks at this site, and using one shape for plain
        assistants keeps the ``content`` shape variation on the user side
        where §6.1 names it.
    """
    del tools  # see docstring
    return st.builds(
        lambda blocks: {"role": "assistant", "content": blocks},
        blocks=st.lists(messages_text_block(), min_size=1, max_size=2),
    )


def messages_tool_pair(
    tools: list[dict[str, Any]],
) -> st.SearchStrategy[list[dict[str, Any]]]:
    """Return a strategy over an (assistant ``tool_use``, user ``tool_result``) pair.

    The pair is a single drawn unit because the pairing invariant is *between*
    the two turns: the ``tool_result``'s ``tool_use_id`` must equal the
    ``tool_use``'s ``id``. Drawing them independently cannot honour this; the
    conversation-level composite composes them through this strategy.

    Args:
        tools: The request's tool declarations.

    Returns:
        A Hypothesis strategy that yields a two-element list
        ``[assistant_with_tool_use, user_with_tool_result]``.

    Raises:
        ValueError: When ``tools`` is empty (see :func:`messages_tool_use_block`).
    """
    if not tools:
        raise ValueError(
            "messages_tool_pair requires at least one declared tool; see "
            "messages_tool_use_block for the rationale."
        )

    def _draw_pair(tool: dict[str, Any], unique: str, result: str) -> list[dict[str, Any]]:
        """Assemble the paired (tool_use, tool_result) turns from one draw.

        Args:
            tool: The tool declaration the block names.
            unique: The shared ``tool_use.id`` / ``tool_result.tool_use_id``.
            result: The ``tool_result`` content text.

        Returns:
            ``[assistant_with_tool_use, user_with_tool_result]``.
        """
        return [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": unique, "name": tool["name"], "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": unique, "content": result}
                ],
            },
        ]

    return st.builds(
        _draw_pair,
        tool=st.sampled_from(tools),
        unique=st.uuids().map(str),
        result=_short_ascii_text(),
    )


# ── Anthropic Messages: request-level composite ────────────────────────────


def messages_request(
    content_shape: ContentShape | None = None,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over complete Anthropic Messages request bodies.

    The composite draws N message-rounds, alternating user / assistant,
    starting and ending with user, and appends a ``tool_use`` turn at each
    assistant turn with probability ½ when ``tools`` is non-empty — the
    following user turn then answers it, which keeps the conversation
    strictly alternating. The pairing invariant is owned here (R6), because
    the user-turn ``content`` shape depends on whether the turn is answering
    a tool_use (block list) or not (string or block list).

    Args:
        content_shape: When given, all user turns that are *not* answering a
        tool_use use this content shape. When ``None``, the shape is drawn
        per turn. The shape parameter is the AC-5 hook — pairing is asserted
        against both shapes.

    Returns:
        A Hypothesis strategy that yields a fresh, JSON-serialisable
        Anthropic Messages request body whose :func:`messages_problems`
        returns ``[]``.
    """
    return _build_messages_request(content_shape)


@st.composite
def _build_messages_request(
    draw: st.DrawFn,
    content_shape: ContentShape | None,
) -> dict[str, Any]:
    """Build one valid Anthropic Messages request body.

    Args:
        draw: The Hypothesis draw callable.
        content_shape: See :func:`messages_request`.

    Returns:
        A request body that satisfies every rule :func:`messages_problems`
        enforces.
    """
    tools = draw(st.lists(messages_tool_definition(), min_size=1, max_size=3))
    # Roles always alternate and always start and end on a user turn: n full
    # user/assistant pairs plus the closing user turn. Ending on user matters
    # beyond convention — every assistant turn must have a following user
    # iteration to answer a drawn tool_use, or the pairing invariant breaks.
    n_pairs = draw(st.integers(min_value=1, max_value=3))

    turns: list[dict[str, Any]] = []
    roles: list[str] = []
    for _ in range(n_pairs):
        roles.extend(["user", "assistant"])
    roles.append("user")

    for index, role in enumerate(roles):
        # When the prior assistant turn emitted tool_use blocks, the user turn
        # MUST carry tool_result blocks for all of them — and the API only
        # accepts a block list for that shape. We honor that here.
        if (
            role == "user"
            and turns
            and turns[-1]["role"] == "assistant"
            and any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in turns[-1].get("content") or []
            )
        ):
            prior_uses = [
                block for block in turns[-1]["content"]
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": use["id"],
                    "content": draw(_short_ascii_text()),
                }
                for use in prior_uses
            ]
            turns.append({"role": "user", "content": blocks})
            continue

        # Plain turn. Resolve the content shape for the user case; for the
        # assistant case the shape is always blocks (see messages_assistant_message).
        shape = content_shape
        if shape is None and role == "user":
            shape = draw(st.sampled_from(CONTENT_SHAPES))
        text_only = role == "user" and shape == "string"

        if role == "assistant" and tools and index > 0 and draw(st.booleans()):
            # Append only the assistant tool_use turn here. The next
            # user-role iteration sees ``turns[-1]`` is an assistant
            # tool_use turn and answers it via the first arm above, which
            # # keeps the conversation strictly alternating. ``messages_tool_pair``
            # (and its sibling ``cc_tool_pair``) remain exported for any
            # future consumer that needs a single drawn pair — the
            # request-level composites compose them here, and the
            # content-length cap (``_short_ascii_text``, max 16 chars) makes
            # them unsuitable for the T-F3 truncation property, which builds
            # its own oversize-content mutation strategies instead.
            pair = draw(messages_tool_pair(tools))
            turns.append(pair[0])
            continue

        if role == "user":
            turns.append(draw(messages_user_message(text_only)))
        else:
            turns.append(draw(messages_assistant_message(tools)))

    return {
        "model": draw(st.sampled_from(ANTHROPIC_MODELS)),
        "max_tokens": draw(st.integers(min_value=1, max_value=8192)),
        "messages": turns,
        "tools": tools,
        "stream": draw(st.booleans()),
    }


# ── Anthropic Messages: reporter ────────────────────────────────────────────


def messages_problems(body: object) -> list[str]:
    """Report every way an Anthropic Messages body breaks the rules this fixture promises.

    Args:
        body: The request body to inspect. May be any value; the reporter
        never raises.

    Returns:
        One message per violation. Empty list means the body is well-formed
        per the rules recorded in the module docstring: required fields, role
        alternation, tool pairing, and JSON-strict serialisability.

        Validity is independent of size and budget: oversize content or an
        over-budget conversation is *not* a problem here. Size constraints
        belong to the consuming property test.
    """
    problems: list[str] = []

    if not isinstance(body, dict):
        problems.append(f"body must be a dict, got {type(body).__name__}")
        return problems

    for required in ("model", "max_tokens", "messages"):
        if required not in body:
            problems.append(f"missing required field {required!r}")

    messages = body.get("messages")
    if not isinstance(messages, list):
        problems.append("'messages' must be a list")
        return problems
    if not messages:
        problems.append("'messages' must be a non-empty list")
        return problems

    # JSON-strict serialisability. NaN / Inf in any numeric field would
    # otherwise smuggle a body through that the wire rejects with 400.
    try:
        json.dumps(body, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"body is not JSON-strictly serialisable: {exc}")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    # ``roles`` may be empty when every message is non-dict; the per-index
    # loop below flags each of those, so the first-turn / alternation checks
    # only run on what is actually present.
    if roles and roles[0] != "user":
        problems.append(f"first turn must be a user turn, got {roles[0]!r}")

    # Role alternation — fixture rule, mirroring ``cache_breakpoints.py``. The
    # API merges consecutive same-role turns rather than rejecting them, so
    # keeping the rule is what makes the property tests' outcomes attributable
    # to one site.
    for index in range(1, len(roles)):
        if roles[index] == roles[index - 1]:
            problems.append(f"consecutive {roles[index]!r} turns at messages[{index}]")

    # Tool pairing. Every ``tool_use`` block must have a matching
    # ``tool_result`` (same id) in a later user turn; no ``tool_result`` may
    # appear without its ``tool_use``. Tools declared must include every
    # ``tool_use.name``.
    #
    # An explicit ``null`` is a violation, not an absence: the wire rejects
    # ``tools: null``, and a mutation that sets the key to ``None`` must show
    # up here rather than reading as a toolless request.
    raw_tools = body.get("tools")
    if raw_tools is None:
        if "tools" in body:
            problems.append("'tools' must be a list when present")
        tools_list: list[object] = []
    elif isinstance(raw_tools, list):
        tools_list = raw_tools
    else:
        problems.append("'tools' must be a list when present")
        tools_list = []
    declared_tool_names = {
        tool.get("name")
        for tool in tools_list
        if isinstance(tool, dict)
    }
    # Per-message required fields: every message must carry a 'role' and a
    # 'content'. A message dict missing either passes the alternation and
    # pairing checks silently (its role contributes None, its content is
    # skipped), so the vacuous-pass shape §1.4 warns about hides here unless
    # flagged explicitly. The same holds for ``content: None``: the key is
    # present so the missing-content check is satisfied, but the wire rejects
    # a null content on every role (Anthropic requires string or list) — a
    # T-F3 mutation setting the value to None must show up.
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if not isinstance(message.get("role"), str) or not message.get("role"):
            problems.append(f"messages[{index}] has no 'role'")
        if "content" not in message:
            problems.append(f"messages[{index}] has no 'content'")
        elif message.get("content") is None:
            problems.append(f"messages[{index}] content must not be None")

    # Per-turn tool_use / tool_result id sets, in message order — the input
    # to both the set-level orphan/unanswered check and the positional
    # next-turn check below.
    tool_use_ids_by_message: dict[int, set[str]] = {}
    tool_result_ids_by_message: dict[int, set[str]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"messages[{index}] is not a dict")
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                problems.append(f"messages[{index}] contains a non-dict block")
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_use_ids_by_message.setdefault(index, set()).add(block.get("id") or "")
                tool_name = block.get("name")
                if tool_name not in declared_tool_names:
                    problems.append(
                        f"messages[{index}] tool_use calls undeclared tool {tool_name!r}"
                    )
                if not block.get("id"):
                    problems.append(f"messages[{index}] tool_use has no 'id'")
            elif block_type == "tool_result":
                tool_result_ids_by_message.setdefault(index, set()).add(
                    block.get("tool_use_id") or ""
                )
                if not block.get("tool_use_id"):
                    problems.append(f"messages[{index}] tool_result has no 'tool_use_id'")

    use_ids = {tid for ids in tool_use_ids_by_message.values() for tid in ids}
    result_ids = {tid for ids in tool_result_ids_by_message.values() for tid in ids}
    for tid in result_ids - use_ids:
        problems.append(f"tool_result {tid!r} has no matching tool_use")
    for tid in use_ids - result_ids:
        problems.append(f"tool_use {tid!r} has no matching tool_result")

    # Positional pairing — fixture rule, mirroring ``cache_breakpoints.py``:
    # every tool call must be answered in the very next turn. A matching
    # tool_result anywhere *later* is still a violation ("answered too late"):
    # it is exactly the shape a wiring bug produces when the answer arrives
    # several turns downstream, and the wire misattributes it.

    def _result_positions(tid: str) -> list[int]:
        """Return every turn index whose tool_result set contains ``tid``."""
        return sorted(
            index
            for index, ids in tool_result_ids_by_message.items()
            if tid in ids
        )

    for index, uses in tool_use_ids_by_message.items():
        following = tool_result_ids_by_message.get(index + 1, set())
        for tid in uses:
            if tid in following:
                continue
            later = [pos for pos in _result_positions(tid) if pos > index + 1]
            if later:
                problems.append(
                    f"tool_use {tid!r} answered too late at messages[{later[0]}]"
                )

    def _use_positions(tid: str) -> list[int]:
        """Return every turn index whose tool_use set contains ``tid``."""
        return sorted(
            index
            for index, ids in tool_use_ids_by_message.items()
            if tid in ids
        )

    for index, results in tool_result_ids_by_message.items():
        preceding = tool_use_ids_by_message.get(index - 1, set())
        for tid in results:
            if tid in preceding:
                continue
            earlier = [pos for pos in _use_positions(tid) if pos < index - 1]
            if earlier:
                problems.append(
                    f"tool_result {tid!r} answered too late at messages[{index}]"
                )

    return problems


# ── Chat Completions: block-level strategies ────────────────────────────────


def cc_text_part() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Chat Completions text content parts.

    Returns:
        A Hypothesis strategy that yields ``{"type": "text", "text": <str>}``.
    """
    return st.builds(
        lambda text: {"type": "text", "text": text},
        text=_short_ascii_text(),
    )


def cc_image_part() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Chat Completions image content parts.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``.
    """
    return st.just(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_PNG_1X1}"},
        }
    )


def cc_tool_definition() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Chat Completions tool declarations.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}``.
    """
    return st.builds(
        lambda n, d, p: {
            "type": "function",
            "function": {"name": n, "description": d, "parameters": p},
        },
        n=_name_strategy(),
        d=_short_ascii_text(min_size=1),
        p=_path_schema(),
    )


def cc_tool_call(
    tools: list[dict[str, Any]],
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over a single Chat Completions ``tool_call`` entry.

    Args:
        tools: The request's tool declarations; the strategy picks ``name``
        uniformly from this list.

    Returns:
        A Hypothesis strategy that yields
        ``{"id": <unique>, "type": "function", "function": {"name": ..., "arguments": "{}"}}``.

    Raises:
        ValueError: When ``tools`` is empty (see :func:`messages_tool_use_block`).
    """
    if not tools:
        raise ValueError(
            "cc_tool_call requires at least one declared tool; an "
            "unaddressable tool_call is an I1 violation the substrate does not emit."
        )
    return st.builds(
        lambda tool, unique: {
            "id": unique,
            "type": "function",
            "function": {"name": tool["function"]["name"], "arguments": "{}"},
        },
        tool=st.sampled_from(tools),
        unique=st.uuids().map(str),
    )


def cc_tool_message(
    tool_call_id: str,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over Chat Completions ``role: tool`` messages.

    Args:
        tool_call_id: The ``id`` of the assistant ``tool_calls`` entry this
        message answers.

    Returns:
        A Hypothesis strategy that yields
        ``{"role": "tool", "tool_call_id": <id>, "content": <str>}``.
    """
    return st.builds(
        lambda text: {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": text,
        },
        text=_short_ascii_text(),
    )


# ── Chat Completions: message- and pair-level strategies ────────────────────


def cc_user_message(
    text_only: bool,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over plain Chat Completions user messages.

    Args:
        text_only: When ``True``, ``content`` is a plain string. When
        ``False``, ``content`` is a list of text / image parts.

    Returns:
        A Hypothesis strategy that yields ``{"role": "user", "content": ...}``.
    """
    if text_only:
        return st.builds(
            lambda text: {"role": "user", "content": text},
            text=_short_ascii_text(),
        )
    part = st.one_of(cc_text_part(), cc_image_part())
    return st.builds(
        lambda parts: {"role": "user", "content": parts},
        parts=st.lists(part, min_size=1, max_size=3),
    )


def cc_assistant_message(
    tools: list[dict[str, Any]] | None = None,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over plain Chat Completions assistant messages.

    Args:
        tools: Unused. Present for symmetry with the pair-level strategy.

    Returns:
        A Hypothesis strategy that yields
        ``{"role": "assistant", "content": "..."}``. ``content`` is a plain
        string here; tool_calls appear only via :func:`cc_tool_pair`.
    """
    del tools
    return st.builds(
        lambda text: {"role": "assistant", "content": text},
        text=_short_ascii_text(),
    )


def cc_tool_pair(
    tools: list[dict[str, Any]],
) -> st.SearchStrategy[list[dict[str, Any]]]:
    """Return a strategy over an (assistant ``tool_calls``, tool message) pair.

    Args:
        tools: The request's tool declarations.

    Returns:
        A Hypothesis strategy that yields a two-element list
        ``[assistant_with_tool_calls, tool_message_answering]``.

    Raises:
        ValueError: When ``tools`` is empty (see :func:`cc_tool_call`).
    """
    if not tools:
        raise ValueError(
            "cc_tool_pair requires at least one declared tool; see "
            "cc_tool_call for the rationale."
        )

    def _draw_pair(tool_call: dict[str, Any], content: str) -> list[dict[str, Any]]:
        """Assemble the paired (tool_calls, tool message) messages from one draw.

        Args:
            tool_call: The assistant's ``tool_calls`` entry.
            content: The answering tool message's content text.

        Returns:
            ``[assistant_with_tool_calls, tool_message_answering]``.
        """
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": content,
            },
        ]

    return st.builds(
        _draw_pair,
        tool_call=cc_tool_call(tools),
        content=_short_ascii_text(),
    )


# ── Chat Completions: request-level composite ──────────────────────────────


def cc_request(
    content_shape: ContentShape | None = None,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over complete Chat Completions request bodies.

    The composite owns the (assistant ``tool_calls`` → tool message) pairing
    invariant (R6), drawing message rounds, alternating user / assistant,
    starting and ending with user, and appending a ``tool_calls`` turn at
    each assistant turn with probability ½ when ``tools`` is non-empty — the
    answering ``role: tool`` message follows immediately.

    Args:
        content_shape: When given, all user turns that are *not* answering a
        tool call use this content shape. When ``None``, the shape is drawn
        per turn.

    Returns:
        A Hypothesis strategy that yields a fresh, JSON-serialisable
        Chat Completions request body whose :func:`cc_problems` returns ``[]``.
    """
    return _build_cc_request(content_shape)


@st.composite
def _build_cc_request(
    draw: st.DrawFn,
    content_shape: ContentShape | None,
) -> dict[str, Any]:
    """Build one valid Chat Completions request body.

    Args:
        draw: The Hypothesis draw callable.
        content_shape: See :func:`cc_request`.

    Returns:
        A request body that satisfies every rule :func:`cc_problems` enforces.
    """
    tools = draw(st.lists(cc_tool_definition(), min_size=1, max_size=3))
    # Same shape as the Messages composite: strict alternation, starting and
    # ending on a user message.
    n_pairs = draw(st.integers(min_value=1, max_value=3))

    messages: list[dict[str, Any]] = []
    roles: list[str] = []
    for _ in range(n_pairs):
        roles.extend(["user", "assistant"])
    roles.append("user")

    for role in roles:
        # Answer a prior assistant tool_calls entry, if any, before considering
        # the current role. Chat Completions forbids interleaving — every
        # tool_call must be followed by one or more ``role: tool`` messages.
        if (
            role == "user"
            and messages
            and messages[-1]["role"] == "assistant"
            and messages[-1].get("tool_calls")
        ):
            for tool_call in messages[-1]["tool_calls"]:
                messages.append(draw(cc_tool_message(tool_call["id"])))
            continue

        shape = content_shape
        if shape is None and role == "user":
            shape = draw(st.sampled_from(CONTENT_SHAPES))
        text_only = role == "user" and shape == "string"

        if (
            role == "assistant"
            and tools
            and messages
            and messages[-1]["role"] == "user"
            and draw(st.booleans())
        ):
            tool_call = draw(cc_tool_call(tools))
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": [tool_call]}
            )
            messages.append(draw(cc_tool_message(tool_call["id"])))
            continue

        if role == "user":
            messages.append(draw(cc_user_message(text_only)))
        else:
            messages.append(draw(cc_assistant_message(tools)))

    return {
        "model": draw(st.sampled_from(OPENAI_MODELS)),
        "messages": messages,
        "tools": tools,
        "stream": draw(st.booleans()),
    }


# ── Chat Completions: reporter ──────────────────────────────────────────────


def cc_problems(body: object) -> list[str]:
    """Report every way a Chat Completions body breaks the rules this fixture promises.

    Args:
        body: The request body to inspect. May be any value; the reporter
        never raises.

    Returns:
        One message per violation. Empty list means the body is well-formed
        per the rules recorded in the module docstring: required fields, role
        alternation, tool pairing, and JSON-strict serialisability.
    """
    problems: list[str] = []

    if not isinstance(body, dict):
        problems.append(f"body must be a dict, got {type(body).__name__}")
        return problems

    for required in ("model", "messages"):
        if required not in body:
            problems.append(f"missing required field {required!r}")

    messages = body.get("messages")
    if not isinstance(messages, list):
        problems.append("'messages' must be a list")
        return problems
    if not messages:
        problems.append("'messages' must be a non-empty list")
        return problems

    try:
        json.dumps(body, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"body is not JSON-strictly serialisable: {exc}")

    roles = [
        message.get("role")
        for message in messages
        if isinstance(message, dict)
    ]
    # ``roles`` may be empty when every message is non-dict; the per-index
    # loop below flags each of those, so the first-turn / alternation checks
    # only run on what is actually present.
    if roles and roles[0] != "user":
        problems.append(f"first message must be a user message, got {roles[0]!r}")

    # Role alternation — fixture rule, same posture as the Messages reporter.
    # ``tool`` messages are exempt: an assistant turn that made several calls
    # is answered by several consecutive ``role: tool`` messages, which the
    # wire requires and this rule must not reject.
    for index in range(1, len(roles)):
        if roles[index] == roles[index - 1] and roles[index] in ("user", "assistant"):
            problems.append(
                f"consecutive {roles[index]!r} messages at messages[{index}]"
            )

    # Explicit ``null`` is a violation, not an absence — same posture as the
    # Messages reporter's ``tools`` guard.
    raw_tools = body.get("tools")
    if raw_tools is None:
        if "tools" in body:
            problems.append("'tools' must be a list when present")
        tools_list: list[object] = []
    elif isinstance(raw_tools, list):
        tools_list = raw_tools
    else:
        problems.append("'tools' must be a list when present")
        tools_list = []
    declared_tool_names = {
        tool.get("function", {}).get("name")
        for tool in tools_list
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }

    # Per-message required fields. Every message must carry a 'role'. On
    # 'content', CC is the same as Messages except assistant: an assistant
    # turn may carry ``content: None`` only when ``tool_calls`` is present and
    # non-empty (the wire permits it; a tool-only assistant with missing
    # content is malformed). The ``content: None`` check is symmetric: the
    # key is present so the missing-content check is satisfied, but the
    # value is null and the wire rejects it.
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if not isinstance(message.get("role"), str) or not message.get("role"):
            problems.append(f"messages[{index}] has no 'role'")
        role = message.get("role")
        if role == "user" or role == "tool":
            if "content" not in message:
                problems.append(f"messages[{index}] has no 'content'")
            elif message.get("content") is None:
                problems.append(f"messages[{index}] content must not be None")
        elif role == "assistant":
            raw_calls = message.get("tool_calls")
            has_calls = isinstance(raw_calls, list) and len(raw_calls) > 0
            if "content" not in message and not has_calls:
                problems.append(
                    f"messages[{index}] assistant has no 'content' and no tool_calls"
                )
            elif message.get("content") is None and not has_calls:
                problems.append(
                    f"messages[{index}] assistant content must not be None without tool_calls"
                )

    # Tool-call / tool-message id sets per message index, in order — input to
    # both the set-level orphan/unanswered check and the positional next-turn
    # check below.
    tool_call_ids_by_message: dict[int, set[str]] = {}
    tool_message_ids_by_message: dict[int, set[str]] = {}

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"messages[{index}] is not a dict")
            continue
        role = message.get("role")
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if raw_calls is None:
                if "tool_calls" in message:
                    problems.append(
                        f"messages[{index}] 'tool_calls' must be a list when present"
                    )
                calls: list[object] = []
            elif isinstance(raw_calls, list):
                calls = raw_calls
            else:
                problems.append(
                    f"messages[{index}] 'tool_calls' must be a list when present"
                )
                calls = []
            for call in calls:
                if not isinstance(call, dict):
                    problems.append(
                        f"messages[{index}] tool_calls entry is not a dict"
                    )
                    continue
                call_id = call.get("id")
                if not call_id:
                    problems.append(f"messages[{index}] tool_call has no 'id'")
                else:
                    tool_call_ids_by_message.setdefault(index, set()).add(call_id)
                function = call.get("function")
                if not isinstance(function, dict):
                    problems.append(
                        f"messages[{index}] tool_call 'function' is not a dict"
                    )
                    continue
                name = function.get("name")
                if name not in declared_tool_names:
                    problems.append(
                        f"messages[{index}] tool_call targets undeclared tool {name!r}"
                    )
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                problems.append(f"messages[{index}] tool message has no 'tool_call_id'")
            else:
                tool_message_ids_by_message.setdefault(index, set()).add(call_id)

    tool_call_ids_ever = {
        tid for ids in tool_call_ids_by_message.values() for tid in ids
    }
    tool_message_ids_ever = {
        tid for ids in tool_message_ids_by_message.values() for tid in ids
    }
    for tid in tool_message_ids_ever - tool_call_ids_ever:
        problems.append(f"tool message {tid!r} has no matching tool_call")
    for tid in tool_call_ids_ever - tool_message_ids_ever:
        problems.append(f"tool_call {tid!r} has no matching tool message")

    # Positional pairing — the wire requires a tool message immediately after
    # the assistant turn that emitted its tool_calls, and CC's grammar forbids
    # anything between them. A matching tool message several turns later is a
    # wiring defect and must be flagged, mirroring the same rule the Messages
    # reporter applies (and which the sibling fixture enforces as well).
    def _message_positions(tid: str, by_message: dict[int, set[str]]) -> list[int]:
        return sorted(index for index, ids in by_message.items() if tid in ids)

    for index, call_set in tool_call_ids_by_message.items():
        # Walk forward collecting the immediately-following tool messages. The
        # collected ids must equal the assistant's call set.
        collected: set[str] = set()
        walk = index + 1
        while walk in tool_message_ids_by_message:
            collected |= tool_message_ids_by_message[walk]
            walk += 1
        for tid in call_set - collected:
            later = [
                pos for pos in _message_positions(tid, tool_message_ids_by_message)
                if pos > index + 1
            ]
            if later:
                problems.append(
                    f"tool_call {tid!r} answered too late at messages[{later[0]}]"
                )

    for index, m_ids in tool_message_ids_by_message.items():
        preceding_calls = tool_call_ids_by_message.get(index - 1, set())
        for tid in m_ids:
            if tid in preceding_calls:
                continue
            earlier = [
                pos for pos in _message_positions(tid, tool_call_ids_by_message)
                if pos < index - 1
            ]
            if earlier:
                problems.append(
                    f"tool_result {tid!r} answered too late at messages[{index}]"
                )

    return problems


# ── OpenAI Responses: block-level strategies ────────────────────────────────


def responses_tool_definition() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over OpenAI Responses ``tools`` declarations.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "function", "name": ..., "description": ..., "parameters": {...}}``.
        ``parameters`` is the same minimal JSON-schema object the CC and
        Messages tool definitions emit (``{"type": "object", ...}``) — the
        Responses API accepts richer schemas, but the substrate keeps tool
        payloads minimal so the property tests do not exercise a schema
        surface the reporters do not pin. The name uses the same alphabet as
        the Anthropic / Chat Completions tool definitions so strategies
        compose across formats when a future cross-format test needs it.
    """
    return st.builds(
        lambda n, d, p: {
            "type": "function",
            "name": n,
            "description": d,
            "parameters": p,
        },
        n=_name_strategy(),
        d=_short_ascii_text(min_size=1),
        p=_path_schema(),
    )


def responses_function_call_item(
    tools: list[dict[str, Any]],
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over a single Responses ``function_call`` item.

    Args:
        tools: The request's tool declarations; the strategy picks ``name``
            uniformly from this list.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "function_call", "call_id": <unique>, "name": ..., "arguments": "..."}``.

    Raises:
        ValueError: When ``tools`` is empty (see :func:`cc_tool_call` for the
            analogous rationale: an unaddressable ``function_call`` would be
            an I1 violation the substrate refuses to emit).
    """
    if not tools:
        raise ValueError(
            "responses_function_call_item requires at least one declared tool; "
            "an unaddressable function_call is an I1 violation the substrate does not emit."
        )
    return st.builds(
        lambda tool, unique: {
            "type": "function_call",
            "call_id": unique,
            "name": tool["name"],
            "arguments": "{}",
        },
        tool=st.sampled_from(tools),
        unique=st.uuids().map(str),
    )


def responses_function_call_output_item(
    call_id: str,
) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over a Responses ``function_call_output`` item.

    Args:
        call_id: The ``call_id`` of the ``function_call`` this output answers.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "function_call_output", "call_id": <id>, "output": <str>}``.
        ``output`` is always a string here; the structured (list) branch is
        exercised by the truncation property's constructed cases, since the
        substrate would otherwise have to pin a content-parts shape the
        reporters do not enforce.
    """
    return st.builds(
        lambda text: {
            "type": "function_call_output",
            "call_id": call_id,
            "output": text,
        },
        text=_short_ascii_text(),
    )


def responses_message_item(role: str) -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over a Responses ``message`` item.

    Args:
        role: The role the item carries. Must be one of ``"user"``,
            ``"assistant"``, or ``"system"``; the conversation composite is
            the only caller and always passes a fixture-rule role.

    Returns:
        A Hypothesis strategy that yields
        ``{"type": "message", "role": ..., "content": [parts]}``. User and
        system items carry one to two ``input_text`` parts; assistant items
        carry one ``output_text`` part. The Responses API also accepts
        ``content`` as a plain string for ``message`` items; the substrate
        emits the list shape to keep the property tests' iteration over parts
        shape-stable (the twin functions ignore message content shape).
    """
    if role not in {"user", "assistant", "system"}:
        raise ValueError(
            f"responses_message_item role must be user|assistant|system, got {role!r}"
        )

    if role == "assistant":
        return st.builds(
            lambda text: {
                "type": "message",
                "role": role,
                "content": [{"type": "output_text", "text": text}],
            },
            text=_short_ascii_text(),
        )
    return st.builds(
        lambda text, extra: {
            "type": "message",
            "role": role,
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_text", "text": extra},
            ],
        },
        text=_short_ascii_text(),
        extra=_short_ascii_text(),
    )


# ── OpenAI Responses: request-level composite ──────────────────────────────


def responses_request() -> st.SearchStrategy[dict[str, Any]]:
    """Return a strategy over complete OpenAI Responses request bodies.

    The composite draws N message-rounds, alternating user / assistant,
    starting and ending with a user message, and inserts a
    ``function_call`` → ``function_call_output`` pair at each assistant turn
    with probability ½ when ``tools`` is non-empty — the output item is
    absorbed immediately so call / output adjacency holds. The pairing
    invariant is owned here (R6): output items declare a ``call_id`` drawn
    from the immediately-prior call, so the input list is valid by
    construction.

    The composite's body carries the minimum required envelope (``model``,
    ``input``, ``tools``, ``stream``); optional Responses keys like
    ``instructions``, ``store``, ``metadata`` are out of scope — the twin
    functions under test read only ``input``, so adding more keys would
    expand the reporters without exercising new behaviour.

    Returns:
        A Hypothesis strategy that yields a fresh, JSON-serialisable
        OpenAI Responses request body whose :func:`responses_problems`
        returns ``[]``.
    """
    return _build_responses_request()


@st.composite
def _build_responses_request(draw: st.DrawFn) -> dict[str, Any]:
    """Build one valid OpenAI Responses request body.

    Args:
        draw: The Hypothesis draw callable.

    Returns:
        A request body that satisfies every rule :func:`responses_problems`
        enforces.
    """
    tools = draw(st.lists(responses_tool_definition(), min_size=1, max_size=3))
    # Same fixture rule as the CC / Messages composites: alternating user /
    # assistant message items, start and end on user, so the conversation is
    # structurally well-formed even after orphan injection from the
    # T-F3 mutation strategies.
    n_pairs = draw(st.integers(min_value=1, max_value=3))

    items: list[dict[str, Any]] = []
    roles: list[str] = []
    for _ in range(n_pairs):
        roles.extend(["user", "assistant"])
    roles.append("user")

    for role in roles:
        # Answer the immediately-prior function_call before considering the
        # current role. The Responses API does not require adjacency of
        # function_call and function_call_output, but the fixture enforces
        # it so the substrate's emitted pairing matches the wire rule the
        # properties consume ("output's call_id declared by a preceding
        # call") without surfacing a forward-reference edge case in the
        # composite itself.
        if items and items[-1].get("type") == "function_call":
            items.append(draw(responses_function_call_output_item(items[-1]["call_id"])))
            continue

        if role == "assistant" and draw(st.booleans()):
            items.append(draw(responses_function_call_item(tools)))
            continue

        items.append(draw(responses_message_item(role)))

    return {
        "model": draw(st.sampled_from(OPENAI_MODELS)),
        "input": items,
        "tools": tools,
        "stream": draw(st.booleans()),
    }


# ── OpenAI Responses: reporter ─────────────────────────────────────────────


def responses_problems(body: object) -> list[str]:
    """Report every way an OpenAI Responses body breaks the rules this fixture promises.

    Args:
        body: The request body to inspect. May be any value; the reporter
            never raises.

    Returns:
        One message per violation. Empty list means the body is well-formed
        per the rules recorded in the module docstring: required fields,
        message-item role alternation (fixture rule, mirroring the CC /
        Messages reporters), ``call_id`` pairing with preceding-call
        semantics, and JSON-strict serialisability. Size and content-string
        length are deliberately **not** flagged — size constraints belong to
        the consuming property test (T-F3's truncation property) and
        asserting them in the substrate would forbid the over-limit cases
        the property exercises.
    """
    problems: list[str] = []

    if not isinstance(body, dict):
        problems.append(f"body must be a dict, got {type(body).__name__}")
        return problems

    for required in ("model", "input"):
        if required not in body:
            problems.append(f"missing required field {required!r}")

    model = body.get("model")
    if "model" in body and not (isinstance(model, str) and model):
        problems.append("'model' must be a non-empty string")

    input_items = body.get("input")
    if not isinstance(input_items, list):
        problems.append("'input' must be a list")
        return problems
    if not input_items:
        problems.append("'input' must be a non-empty list")
        return problems

    try:
        json.dumps(body, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"body is not JSON-strictly serialisable: {exc}")

    # Declared tool names. The composite always emits ``tools``; a body
    # that omits it has no declared set and every function_call would flag
    # as undeclared — mirroring the Chat Completions reporter's posture.
    raw_tools = body.get("tools")
    if raw_tools is None:
        if "tools" in body:
            problems.append("'tools' must be a list when present")
        declared_tool_names: set[str] = set()
    elif isinstance(raw_tools, list):
        declared_tool_names = {
            tool.get("name")
            for tool in raw_tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
    else:
        problems.append("'tools' must be a list when present")
        declared_tool_names = set()

    call_ids_by_item: dict[int, set[str]] = {}
    output_ids_by_item: dict[int, set[str]] = {}

    last_role: str | None = None
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            problems.append(f"input[{index}] is not a dict")
            continue
        item_type = item.get("type")
        if not isinstance(item_type, str) or not item_type:
            problems.append(f"input[{index}] has no 'type'")
            continue
        if item_type not in {"message", "function_call", "function_call_output", "reasoning"}:
            problems.append(f"input[{index}] unknown input item type {item_type!r}")
            continue
        if item_type == "message":
            role = item.get("role")
            if not (isinstance(role, str) and role in {"user", "assistant", "system"}):
                problems.append(f"input[{index}] message has invalid 'role' {role!r}")
                role = None
            content = item.get("content")
            if not isinstance(content, list):
                problems.append(f"input[{index}] message 'content' must be a list")
            else:
                for part_index, part in enumerate(content):
                    if not isinstance(part, dict):
                        problems.append(
                            f"input[{index}].content[{part_index}] is not a dict"
                        )
                        continue
                    part_type = part.get("type")
                    if part_type not in {
                        "input_text",
                        "output_text",
                        "input_image",
                        "input_file",
                    }:
                        problems.append(
                            f"input[{index}].content[{part_index}] unknown part type {part_type!r}"
                        )
            # Fixture rule: consecutive message items may not repeat a role
            # — matches the CC / Messages fixture's alternation posture so
            # the conversation composite emits a stable shape.
            if role and last_role and role == last_role and role in {"user", "assistant"}:
                problems.append(
                    f"input[{index}] consecutive message items with role {role!r}"
                )
            if role:
                last_role = role
        elif item_type == "function_call":
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id):
                problems.append(f"input[{index}] function_call has no 'call_id'")
            else:
                call_ids_by_item.setdefault(index, set()).add(call_id)
            name = item.get("name")
            if not (isinstance(name, str) and name):
                problems.append(f"input[{index}] function_call has no 'name'")
            elif declared_tool_names and name not in declared_tool_names:
                problems.append(
                    f"input[{index}] function_call targets undeclared tool {name!r}"
                )
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                problems.append(
                    f"input[{index}] function_call 'arguments' must be a string"
                )
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id):
                problems.append(f"input[{index}] function_call_output has no 'call_id'")
            else:
                output_ids_by_item.setdefault(index, set()).add(call_id)
            # The published schema allows ``output`` as a plain string or as
            # an array of input_text / input_image / input_file parts. The
            # string branch is what the truncation twin reads; the list
            # branch must still be shape-checked or a malformed list output
            # slips past the reporter into a property that assumes strings
            # (and silently under-covers). Reasoning items are not
            # strategy-emitted, so this is the only item kind that needs a
            # union shape check today.
            if "output" not in item:
                problems.append(
                    f"input[{index}] function_call_output has no 'output'"
                )
            else:
                output_value = item["output"]
                if isinstance(output_value, list):
                    for part_index, part in enumerate(output_value):
                        if not isinstance(part, dict):
                            problems.append(
                                f"input[{index}].output[{part_index}] is not a dict"
                            )
                            continue
                        part_type = part.get("type")
                        if part_type not in {
                            "input_text",
                            "input_image",
                            "input_file",
                        }:
                            problems.append(
                                f"input[{index}].output[{part_index}] unknown output part type {part_type!r}"
                            )
                elif not isinstance(output_value, str):
                    problems.append(
                        f"input[{index}] function_call_output 'output' must be a string or a list of parts"
                    )

    call_ids_ever = {tid for ids in call_ids_by_item.values() for tid in ids}
    output_ids_ever = {tid for ids in output_ids_by_item.values() for tid in ids}
    for tid in output_ids_ever - call_ids_ever:
        problems.append(
            f"function_call_output {tid!r} has no matching function_call"
        )
    for tid in call_ids_ever - output_ids_ever:
        problems.append(
            f"function_call {tid!r} has no matching function_call_output"
        )

    # Forward-reference guard: an output whose FIRST declaring call appears
    # *after* it violates the wire rule "output's call_id declared by a
    # preceding call". The set-level checks above already catch the "no call
    # anywhere" and "no output anywhere" cases; this catches the wrong-order
    # case, which a property test's orphan injection can construct.
    first_call_index: dict[str, int] = {}
    for index in sorted(call_ids_by_item):
        for tid in call_ids_by_item[index]:
            first_call_index.setdefault(tid, index)
    for index in sorted(output_ids_by_item):
        for tid in output_ids_by_item[index]:
            declared_at = first_call_index.get(tid)
            if declared_at is None or declared_at >= index:
                problems.append(
                    f"input[{index}] function_call_output {tid!r} answered "
                    f"before its declaring function_call"
                )

    return problems
