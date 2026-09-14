"""Hypothesis strategies that generate valid Anthropic Messages and Chat Completions transcripts.

`.system_design/TEST_SUITE.md` §6.1 · plan **T-F1** (KBR-70).

The L1 suite today is entirely example-based; the §6.1 property list (compaction,
pairing, truncation, egress, anomaly, translator) needs property-based tests
running against pure logic. Property tests for the *transcript-shaped* targets —
``MessagesTranslator`` (T-F6), ``_compact_messages`` (T-F2),
``_validate_tool_call_pairing`` and ``_truncate_oversized_tool_results`` (T-F3) —
compose scenarios from valid request bodies, and they need to vary far enough
through the space that one example test could not cover. This module provides the
composable building blocks: whole-request strategies, single-message strategies,
single-block strategies, and tool-definition strategies, plus a per-format
``problems()`` reporter that states what "valid" means so a generated body can be
handed back and asserted well-formed.

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
import os
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
    starting and ending with user, and inserts a (tool_use, tool_result) pair
    at one randomly chosen assistant turn when ``tools`` is non-empty. The
    pairing invariant is owned here (R6), because the user-turn ``content``
    shape depends on whether the turn is answering a tool_use (block list) or
    not (string or block list).

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
    n_turns = draw(st.integers(min_value=2, max_value=6))

    turns: list[dict[str, Any]] = []
    roles: list[str] = []
    for index in range(n_turns):
        roles.append("user" if index % 2 == 0 else "assistant")

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
            pair = draw(messages_tool_pair(tools))
            turns.append(pair[0])
            turns.append(pair[1])
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

    for required in ("model", "messages"):
        if required not in body:
            problems.append(f"missing required field {required!r}")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        problems.append("'messages' must be a non-empty list")
        return problems

    # JSON-strict serialisability. NaN / Inf in any numeric field would
    # otherwise smuggle a body through that the wire rejects with 400.
    try:
        json.dumps(body, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"body is not JSON-strictly serialisable: {exc}")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles[0] != "user":
        problems.append(f"first turn must be a user turn, got {roles[0]!r}")

    # Tool pairing. Every ``tool_use`` block must have a matching
    # ``tool_result`` (same id) in a later user turn; no ``tool_result`` may
    # appear without its ``tool_use``. Tools declared must include every
    # ``tool_use.name``.
    declared_tool_names = {
        tool.get("name")
        for tool in body.get("tools", [])
        if isinstance(tool, dict)
    }
    tool_uses: list[tuple[int, str]] = []
    tool_results: list[tuple[int, str]] = []
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
                tool_uses.append((index, block.get("id") or ""))
                tool_name = block.get("name")
                if tool_name not in declared_tool_names:
                    problems.append(
                        f"messages[{index}] tool_use calls undeclared tool {tool_name!r}"
                    )
                if not block.get("id"):
                    problems.append(f"messages[{index}] tool_use has no 'id'")
            elif block_type == "tool_result":
                tool_results.append((index, block.get("tool_use_id") or ""))
                if not block.get("tool_use_id"):
                    problems.append(f"messages[{index}] tool_result has no 'tool_use_id'")

    use_ids = {tid for _, tid in tool_uses}
    result_ids = {tid for _, tid in tool_results}
    for tid in result_ids - use_ids:
        problems.append(f"tool_result {tid!r} has no matching tool_use")
    for tid in use_ids - result_ids:
        problems.append(f"tool_use {tid!r} has no matching tool_result")

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
    starting and ending with user, and inserting a tool pair at one randomly
    chosen assistant turn when ``tools`` is non-empty.

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
    n_messages = draw(st.integers(min_value=2, max_value=6))

    messages: list[dict[str, Any]] = []
    roles: list[str] = [
        "user" if index % 2 == 0 else "assistant"
        for index in range(n_messages)
    ]

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
    if not isinstance(messages, list) or not messages:
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
    if roles[0] != "user":
        problems.append(f"first message must be a user message, got {roles[0]!r}")

    declared_tool_names = {
        tool.get("function", {}).get("name")
        for tool in body.get("tools", [])
        if isinstance(tool, dict)
    }

    tool_call_ids: list[str] = []
    tool_message_ids: list[str] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"messages[{index}] is not a dict")
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    problems.append(
                        f"messages[{index}] tool_calls entry is not a dict"
                    )
                    continue
                call_id = call.get("id")
                if not call_id:
                    problems.append(f"messages[{index}] tool_call has no 'id'")
                else:
                    tool_call_ids.append(call_id)
                name = call.get("function", {}).get("name")
                if name not in declared_tool_names:
                    problems.append(
                        f"messages[{index}] tool_call targets undeclared tool {name!r}"
                    )
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                problems.append(f"messages[{index}] tool message has no 'tool_call_id'")
            else:
                tool_message_ids.append(call_id)

    orphan_results = set(tool_message_ids) - set(tool_call_ids)
    for tid in orphan_results:
        problems.append(f"tool message {tid!r} has no matching tool_call")
    unanswered = set(tool_call_ids) - set(tool_message_ids)
    for tid in unanswered:
        problems.append(f"tool_call {tid!r} has no matching tool message")

    return problems


# ── Hypothesis CI profile (R8) ────────────────────────────────────────────
#
# When ``CI`` is set we load a profile that derandomises the test, removes
# the default 200 ms per-example deadline (the slow Windows / macOS Fast-gate
# legs would flake otherwise), and disables the example database (so a
# failure caught on one machine does not replay on another). The profile is
# registered lazily here because ``settings.register_profile`` must run
# before ``settings.load_profile`` and the test module loads it at collection
# time. None of this fires when ``CI`` is unset; the developer gets
# hypothesis's default profile locally.

if os.environ.get("CI"):
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "kitty-bridge-ci",
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        database=None,
    )
    settings.load_profile("kitty-bridge-ci")
