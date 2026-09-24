"""CB-2: the Anthropic rebuild cannot recover a breakpoint the translator already dropped.

Claude Code puts ``cache_control`` breakpoints on nearly every request. On the
default ``anthropic`` provider almost all of them are lost before the wire. That
provider does not declare ``use_native_messages``, so its translation pair is
Messages → Chat Completions → Messages: :meth:`MessagesTranslator.translate_request`
builds a Chat Completions intermediate, and
:meth:`AnthropicAdapter.translate_to_upstream` rebuilds a Messages body from it.

This module records what that pair does, each fact a characterisation of today's
behaviour that the eventual product fix is meant to turn red. KBR-228 part B
turned one of them red on purpose and the test was rewritten: the adapter now
restores the agent's ``system`` value — breakpoints included — verbatim from the
internal carriage, so the system block's breakpoint reaches the wire again. The
remaining characterisations:

* a breakpoint on a tool, a message content block, or at the top level of the
  request never reaches the wire;
* the intermediate carries the agent's breakpoints only under the internal
  carriage keys, nowhere else;
* the one other survivor: a breakpoint nested inside a ``tool_result``'s list content
  is copied through both hops, onto a block where it is not established that
  Anthropic honours one;
* the adapter adds no breakpoint of its own;
* given a Chat Completions request that *does* carry breakpoints (in the shapes
  OpenRouter's Chat Completions dialect defines, plus two sites no published
  dialect defines), which kitty's ``/v1/chat/completions`` handler passes to the
  adapter without translating it, the adapter keeps them on user and tool
  content and drops them at every other site recorded here.

**The intermediate format is not the reason.** Chat Completions *can* carry a
breakpoint: OpenRouter's dialect puts ``cache_control`` on content parts, and
OpenAI's own schema has ``prompt_cache_breakpoint`` on content parts (GPT-5.6+,
its TTL set request-wide by ``prompt_cache_options``). The loss is the
translator's choice (it joins system and text blocks into strings and rebuilds
tools), not a limit of the format.

**What the loss costs, and where.** Anthropic bills a cache read at 0.1x base
input (0.025x on some models), so a lost breakpoint re-bills the agent's stable
prefix at *at least* ten times its cached rate on every turn. That cost is real
only on an Anthropic-shaped upstream.

**This is L1 and no upstream is contacted.** The Claude Code fixture is for
covering sites, not a valid request. It carries seven breakpoints where Anthropic
allows four; its top-level 5-minute breakpoint differs from the last block's
1-hour one, which Anthropic answers with a 400; and it places a 5-minute entry
before a 1-hour one, which Anthropic's docs forbid. Do not reuse it against a
live or realistic upstream. Nor does this module cover the server's steps
between the two hops (model normalisation, truncation, compaction): none of
them reads or writes ``cache_control``.

Jira: KBR-199 (epic KBR-197). Design: ``.system_design/TEST_SUITE.md`` §3.2.1
row M16, §3.4, and gaps G37 and G38.
"""

from __future__ import annotations

import pytest

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.registry import get_provider

# The two published breakpoint values. They are different prices (a 1-hour write
# costs 2x base input against 1.25x), so assertions compare whole values.
_FIVE_MINUTES = {"type": "ephemeral"}
_ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}


def _marked(block: dict, breakpoint: dict, *, breakpoints: bool) -> dict:
    """Return ``block`` carrying ``breakpoint``, or unchanged when breakpoints are off.

    Args:
        block: A Messages API block or tool declaration.
        breakpoint: The ``cache_control`` value to attach.
        breakpoints: Whether to attach it at all.

    Returns:
        A new dict with a fresh copy of ``breakpoint`` under ``cache_control``,
        or ``block`` itself when ``breakpoints`` is ``False``.
    """
    if not breakpoints:
        return block
    return {**block, "cache_control": dict(breakpoint)}


def _claude_code_body(*, breakpoints: bool) -> dict:
    """Build a Claude Code Messages body, with or without breakpoints.

    With breakpoints it carries one on each block kind CB-1 names (a tool, a
    system block, a text block, an image block, a ``tool_use`` and a
    ``tool_result``) plus the top-level automatic-caching form: seven in all,
    both TTLs represented, every one on a top-level block. The ``tool_result``
    content is a string on purpose; the nested list form is the one survivor
    and has its own test. The image now ships as an ``image_url`` part
    (KBR-222), which carries no breakpoint, so it shows nothing about
    breakpoints on its own; the other six do.

    Every call builds fresh nested objects, so the marked and unmarked bodies
    share nothing. Every ``tool_use`` carries an id, so the translator's
    ``uuid4`` fallback never runs and two translations are comparable.

    Args:
        breakpoints: Whether to place the seven breakpoints.

    Returns:
        A Messages API request body.
    """
    body: dict = {
        "model": "claude-sonnet-5",
        "max_tokens": 32000,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 16000},
        "system": [
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
            _marked(
                {"type": "text", "text": "<env>Working directory: /repo</env>"},
                _ONE_HOUR,
                breakpoints=breakpoints,
            ),
        ],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file from the local filesystem.",
                "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
            },
            _marked(
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
                _FIVE_MINUTES,
                breakpoints=breakpoints,
            ),
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    _marked(
                        {"type": "text", "text": "Why does this test fail?"},
                        _FIVE_MINUTES,
                        breakpoints=breakpoints,
                    ),
                    _marked(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                        },
                        _FIVE_MINUTES,
                        breakpoints=breakpoints,
                    ),
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Read the test first.", "signature": "sig"},
                    {"type": "text", "text": "Let me read it."},
                    _marked(
                        {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "t.py"}},
                        _FIVE_MINUTES,
                        breakpoints=breakpoints,
                    ),
                ],
            },
            {
                "role": "user",
                "content": [
                    _marked(
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "def test(): assert 1 == 2"},
                        _ONE_HOUR,
                        breakpoints=breakpoints,
                    ),
                ],
            },
        ],
    }
    if breakpoints:
        body["cache_control"] = dict(_FIVE_MINUTES)
    return body


def _breakpoints(value: object, path: str = "$") -> dict[str, object]:
    """Find every ``cache_control`` key at any depth of a JSON value.

    Args:
        value: A decoded JSON value: dict, list or scalar.
        path: The JSONPath-like location of ``value``, used as a prefix.

    Returns:
        A mapping from each ``cache_control`` key's path, such as
        ``$.messages[0].content[1].cache_control``, to its value. Empty when
        there is none.
    """
    found: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "cache_control":
                found[child_path] = child
            found.update(_breakpoints(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_breakpoints(child, f"{path}[{index}]"))
    return found


def _cc_request(messages: list[dict], **extra: object) -> dict:
    """Build a minimal Chat Completions request for the adapter.

    Args:
        messages: The Chat Completions ``messages`` list.
        **extra: Further top-level fields, such as ``tools`` or ``cache_control``.

    Returns:
        A Chat Completions request body.
    """
    return {"model": "claude-sonnet-5", "max_tokens": 1024, "messages": messages, **extra}


def _tool_call_turn(**call_extra: object) -> dict:
    """Build an assistant turn making one ``Read`` tool call with id ``call_1``.

    Args:
        **call_extra: Further fields on the tool call object itself.

    Returns:
        A Chat Completions assistant message.
    """
    call = {"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}, **call_extra}
    return {"role": "assistant", "content": None, "tool_calls": [call]}


# ── The fixture carries what the assertions below are about ─────────────────


def test_the_claude_code_body_carries_a_breakpoint_at_every_site() -> None:
    """The fixture places seven breakpoints, and the helper finds each one.

    Without this, every "no breakpoint survives" assertion in this module would
    pass just as happily over a fixture, or a helper, that had none to lose.
    """
    assert _breakpoints(_claude_code_body(breakpoints=True)) == {
        "$.system[1].cache_control": _ONE_HOUR,
        "$.tools[1].cache_control": _FIVE_MINUTES,
        "$.messages[0].content[0].cache_control": _FIVE_MINUTES,
        "$.messages[0].content[1].cache_control": _FIVE_MINUTES,
        "$.messages[1].content[2].cache_control": _FIVE_MINUTES,
        "$.messages[2].content[0].cache_control": _ONE_HOUR,
        "$.cache_control": _FIVE_MINUTES,
    }


def test_the_unmarked_twin_carries_no_breakpoint() -> None:
    """The comparison body really is unmarked.

    If it carried the breakpoints too, the intermediate-equality test below
    would compare a body with itself and prove nothing.
    """
    assert _breakpoints(_claude_code_body(breakpoints=False)) == {}


# ── The headline: Anthropic → Anthropic loses the agent's breakpoints ───────


def test_the_default_anthropic_provider_takes_the_translated_route() -> None:
    """The ``anthropic`` provider is ``AnthropicAdapter`` and is not native passthrough.

    ``BridgeServer._handle_messages`` branches on ``use_native_messages``, so
    this is what puts the default provider on the translation pair below. If
    the provider became native passthrough, the breakpoints would survive and
    the round-trip test would describe a route nobody takes.
    """
    provider = get_provider("anthropic")

    assert (type(provider), provider.use_native_messages) == (AnthropicAdapter, False)


def test_the_default_anthropic_route_delivers_every_breakpoint_the_rebuild_can_express() -> None:
    """Messages → Chat Completions → Messages ships all seven of the fixture's breakpoints, at full value.

    KBR-308 (with KBR-228 part B's system carriage and KBR-296's restore
    side beneath it) carries every site the rebuild can express: the tool
    declaration, both message blocks (text part and image part), the
    ``tool_use``, the ``tool_result``, the system block, and the top-level
    automatic-caching form. Each arrives at the value the agent chose — the
    one-hour and five-minute TTLs land exactly where they were placed — so
    the agent's stable prefix re-bills at the cached rate on every
    translated turn.
    """
    intermediate = MessagesTranslator().translate_request(_claude_code_body(breakpoints=True))

    wire = AnthropicAdapter().translate_to_upstream(intermediate)

    assert _breakpoints(wire) == {
        "$.system[1].cache_control": _ONE_HOUR,
        "$.tools[1].cache_control": _FIVE_MINUTES,
        "$.messages[0].content[0].cache_control": _FIVE_MINUTES,
        "$.messages[0].content[1].cache_control": _FIVE_MINUTES,
        "$.messages[1].content[2].cache_control": _FIVE_MINUTES,
        "$.messages[2].content[0].cache_control": _ONE_HOUR,
        "$.cache_control": _FIVE_MINUTES,
    }


def _tool_result(tool_use_id: str, *, marked: bool) -> dict:
    """Build a ``tool_result`` block whose content is a two-block list.

    Args:
        tool_use_id: The id of the ``tool_use`` this result answers.
        marked: Whether the first content block carries a 1-hour breakpoint.

    Returns:
        A Messages API ``tool_result`` block.
    """
    first = {"type": "text", "text": "def test():"}
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [_marked(first, _ONE_HOUR, breakpoints=marked), {"type": "text", "text": "    assert 1 == 2"}],
    }


@pytest.mark.parametrize(
    ("tool_results", "surviving_path"),
    [
        pytest.param(
            [_tool_result("toolu_01", marked=True)],
            "$.messages[2].content[0].content[0].cache_control",
            id="single-tool-result",
        ),
        pytest.param(
            [_tool_result("toolu_01", marked=False), _tool_result("toolu_02", marked=True)],
            "$.messages[3].content[0].content[0].cache_control",
            id="parallel-tool-results",
        ),
    ],
)
def test_a_breakpoint_nested_in_tool_result_list_content_passes_through_both_hops(
    tool_results: list[dict], surviving_path: str
) -> None:
    """The one breakpoint the translation pair delivers: inside a ``tool_result``'s list content.

    The translator copies a ``tool_result``'s list content verbatim into the
    tool message, on both of its branches (one result, or several in parallel,
    which Claude Code sends routinely), and the adapter copies it back verbatim.
    So the breakpoint arrives with its TTL intact, on a block nested inside the
    ``tool_result``. Whether Anthropic honours a breakpoint at that depth is not
    established: its API types accept one there, and its docs' sub-content rule
    names only citations. It is recorded so the headline's "none" is not
    over-read, and so a fix that changes it does so on purpose.

    Args:
        tool_results: The ``tool_result`` blocks of the final user turn.
        surviving_path: The wire path the marked block's breakpoint arrives at.
    """
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Read both files."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"toolu_0{index + 1}", "name": "Read", "input": {}}
                    for index in range(len(tool_results))
                ],
            },
            {"role": "user", "content": tool_results},
        ],
    }

    wire = AnthropicAdapter().translate_to_upstream(MessagesTranslator().translate_request(body))

    assert _breakpoints(wire) == {surviving_path: _ONE_HOUR}


# ── Where the loss happens: before the adapter runs ─────────────────────────


def test_the_intermediate_handed_to_the_adapter_carries_cache_control_only_in_cargo() -> None:
    """The intermediate carries every agent marker under cargo — carriages, or directly on user content parts.

    The local ``_breakpoints`` walker matches keys whose name contains the
    substring ``"cache_control"``, so it sees:
      * ``_anthropic_system[1].cache_control`` (the system carriage —
        KBR-228 part B;
        ``_anthropic_system`` is not in its own substring, so the OUTER
        name does not match, but the INNER ``cache_control`` does);
      * ``messages[1].content[0].cache_control`` and ``...content[1].cache_control``
        (KBR-308: user text and image parts carry the marker directly —
        parts-form via ``carry_cache_control=True`` at hop 1; the adapter's
        existing part-level restore at ``anthropic.py:740, 744-745`` reads
        them).

    Markers that ride the KBR-296 carriages (``_cache_control``,
    ``_tool_cache_controls``, ``_tool_call_cache_controls``, and
    message-level ``_cache_control``) are invisible to this walker because
    their parent key does not contain ``"cache_control"`` as a substring —
    that is the walker's blind spot, not a claim about cargo absentness.
    The harness detector at ``tests/harness/cache_breakpoints.py`` overrides
    this with a ``_KITTY_CARRIAGE_KEYS`` skip; this walker stays simple to
    keep its claim falsifiable from the source alone.

    The claim this assertion DOES pin: the agent's marker set reaches the
    intermediate only in cargo (carriage or part-level) — never as a
    synthesised new marker the translator invented. A regression here
    would surface as cache writes nobody asked for, at a TTL nobody chose.
    """
    intermediate = MessagesTranslator().translate_request(_claude_code_body(breakpoints=True))

    assert _breakpoints(intermediate) == {
        "$._anthropic_system[1].cache_control": _ONE_HOUR,
        "$.messages[1].content[0].cache_control": _FIVE_MINUTES,
        "$.messages[1].content[1].cache_control": _FIVE_MINUTES,
    }


def test_the_intermediate_tells_marked_from_unmarked_only_through_cargo() -> None:
    """The intermediates differ only in cargo (carriages + part-level markers).

    The marked and unmarked intermediates are equal once every cargo — the
    KBR-228 carriages (``_anthropic_system``, ``_thinking_blocks``), the
    KBR-296 carriages (``_cache_control``, ``_tool_cache_controls``,
    ``_tool_call_cache_controls``, message-level ``_cache_control``), and
    the KBR-308 part-level ``cache_control`` on user content parts — is
    stripped. The pin: no field under any *other* name carries the agent's
    breakpoints. The adapter recovers them from the cargo and nowhere else.

    This compares the translator with itself, which §3.3.1 forbids as an
    oracle for fidelity. It is sound here because the claim is not "the
    output is faithful" but "the breakpoints move only through cargo".
    """
    translator = MessagesTranslator()
    cargo = {
        "_anthropic_system",
        "_thinking_blocks",
        # KBR-296 carriages.
        "_cache_control",
        "_tool_cache_controls",
        "_tool_call_cache_controls",
    }

    def _without_cargo(body: dict) -> dict:
        stripped = {k: v for k, v in body.items() if k not in cargo}
        messages: list = []
        for message in stripped["messages"]:
            # Strip message-level carriages AND any ``cache_control`` on
            # user content parts — both are cargo, not agent-visible
            # differentiation outside the breaker.
            stripped_message = {k: v for k, v in message.items() if k not in cargo}
            content = stripped_message.get("content")
            if isinstance(content, list):
                cleaned: list = []
                for part in content:
                    if isinstance(part, dict) and "cache_control" in part:
                        cleaned.append({k: v for k, v in part.items() if k != "cache_control"})
                    else:
                        cleaned.append(part)
                stripped_message["content"] = cleaned
            messages.append(stripped_message)
        stripped["messages"] = messages
        return stripped

    marked = _without_cargo(translator.translate_request(_claude_code_body(breakpoints=True)))
    unmarked = _without_cargo(translator.translate_request(_claude_code_body(breakpoints=False)))

    assert marked == unmarked


# ── Criterion 1a: the adapter invents nothing ───────────────────────────────


def _breakpoint_free_cc_request() -> dict:
    """Build a Chat Completions request that carries no breakpoint.

    It is non-streaming and sets no thinking configuration, the opposite of the
    round-trip fixture, so a synthesised breakpoint gated on either condition is
    caught by one test or the other.

    Returns:
        A Chat Completions request body.
    """
    return _cc_request(
        [
            {"role": "system", "content": [{"type": "text", "text": "You are Claude Code."}]},
            {"role": "user", "content": [{"type": "text", "text": "Why does this test fail?"}]},
            {
                "role": "assistant",
                "content": "Let me read it.",
                "reasoning_content": "Read the test first.",
                "tool_calls": [
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"file_path": "t.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01", "content": [{"type": "text", "text": "assert 1 == 2"}]},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "Read", "description": "Read a file.", "parameters": {"type": "object"}},
            }
        ],
        stream=False,
    )


def test_the_breakpoint_free_request_reaches_every_structure_a_breakpoint_could_sit_on() -> None:
    """The adapter's output for that request has a system prompt, tools, and every block kind it builds.

    Without this, a later edit to the request could quietly stop exercising a
    site and leave the next test green for the wrong reason.
    """
    wire = AnthropicAdapter().translate_to_upstream(_breakpoint_free_cc_request())

    assert (
        "system" in wire,
        [tool["name"] for tool in wire.get("tools", [])],
        [(message["role"], [block["type"] for block in message["content"]]) for message in wire["messages"]],
    ) == (
        True,
        ["Read"],
        [("user", ["text"]), ("assistant", ["thinking", "text", "tool_use"]), ("user", ["tool_result"])],
    )


def test_the_adapter_adds_no_breakpoint_to_a_request_that_carries_none() -> None:
    """A breakpoint-free Chat Completions request produces a breakpoint-free wire body.

    An adapter that synthesised automatic caching, or invented a block-level
    breakpoint, fails here (for a non-streaming request with no thinking
    configuration) or in the round-trip test (for a streaming one with
    thinking). Commercial consequence of such a change: cache writes the agent
    never asked for, at a TTL it never chose.
    """
    wire = AnthropicAdapter().translate_to_upstream(_breakpoint_free_cc_request())

    assert _breakpoints(wire) == {}


# ── Criterion 1c: the R3 gate — markers on system blocks go only to verified upstreams ──


def test_system_content_part_drop_is_preserved_on_minimax_token() -> None:
    """On a ``forwards_thinking_signature=False`` adapter, a CC system content-part marker is dropped — joined string.

    ``MiniMaxTokenAnthropicAdapter``'s endpoint rejects ``cache_control``
    on system blocks outright (``minimax_token.py:29-30``), so the R3
    carve stays closed there: a marked system part joins to a plain
    string, the marker disappears, and the deliberate G43 scope-out is
    preserved on the CC-origin path too — symmetric with the CB-3 row
    that records the same shape at the wire.
    """
    from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter

    body = _cc_request(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "sys", "cache_control": dict(_ONE_HOUR)},
                ],
            },
            {"role": "user", "content": "hi"},
        ]
    )
    adapter = MiniMaxTokenAnthropicAdapter(native_messages=True)

    wire = adapter.translate_to_upstream(body)

    assert wire["system"] == "sys"
    assert _breakpoints(wire) == {}


# ── Criterion 1b: what the adapter does with a breakpoint it is given ───────

# One case per object the adapter reads from a Chat Completions request as a
# unit: the request, a message, a content part, a tool declaration, a tool call.
# Each carries exactly one 1-hour breakpoint, so a kept one also proves its TTL
# survived. Kept cases mark the first of two parts, so a moved breakpoint fails.
# Assistant list content is absent: the adapter nests that whole list inside a
# `text` field, which is invalid whatever happens to the breakpoint (KBR-34).
#
# KBR-308 update: the six KBR-199-measured drops flip to kept at full value;
# the user-message-object case is deferred (OD1) and renamed for the
# reduction. Two new kept cases join (assistant-message-object,
# tool-message-object — the same restore slots the KBR-296 M9 rebuilder
# writes, now read on the CC-origin path too).
_GIVEN_BREAKPOINT_CASES = [
    pytest.param(
        _cc_request([{"role": "user", "content": "hi"}], cache_control=dict(_ONE_HOUR)),
        "$.cache_control",
        {"$.cache_control": _ONE_HOUR},
        id="top-level-kept",
    ),
    pytest.param(
        _cc_request([{"role": "user", "content": "hi", "cache_control": dict(_ONE_HOUR)}]),
        "$.messages[0].cache_control",
        {},
        id="user-message-object-deferred-dropped",
    ),
    pytest.param(
        _cc_request(
            [
                {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": dict(_ONE_HOUR)}]},
                {"role": "user", "content": "hi"},
            ]
        ),
        "$.messages[0].content[0].cache_control",
        # KBR-308 carve emits a flat list of blocks on
        # ``forwards_thinking_signature=True`` adapters; the marker lands
        # at the block level, not nested one deeper in ``content``.
        {"$.system[0].cache_control": _ONE_HOUR},
        id="system-content-part-kept",
    ),
    pytest.param(
        _cc_request(
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "Read", "parameters": {"type": "object"}},
                    "cache_control": dict(_ONE_HOUR),
                }
            ],
        ),
        "$.tools[0].cache_control",
        {"$.tools[0].cache_control": _ONE_HOUR},
        id="tool-declaration-kept",
    ),
    pytest.param(
        _cc_request(
            [
                {"role": "user", "content": "read it"},
                _tool_call_turn(cache_control=dict(_ONE_HOUR)),
                {"role": "tool", "tool_call_id": "call_1", "content": "file"},
            ]
        ),
        "$.messages[1].tool_calls[0].cache_control",
        # No text block on this assistant turn, so the rebuilt ``tool_use``
        # is ``content[0]``.
        {"$.messages[1].content[0].cache_control": _ONE_HOUR},
        id="tool-call-kept",
    ),
    pytest.param(
        _cc_request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "cache_control": dict(_ONE_HOUR)},
                        {"type": "text", "text": "there"},
                    ],
                }
            ]
        ),
        "$.messages[0].content[0].cache_control",
        {"$.messages[0].content[0].cache_control": _ONE_HOUR},
        id="user-content-part-kept",
    ),
    pytest.param(
        _cc_request(
            [
                {"role": "user", "content": "read it"},
                _tool_call_turn(),
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "text", "text": "file", "cache_control": dict(_ONE_HOUR)},
                        {"type": "text", "text": "more"},
                    ],
                },
            ]
        ),
        "$.messages[2].content[0].cache_control",
        {"$.messages[2].content[0].content[0].cache_control": _ONE_HOUR},
        id="tool-message-content-part-relocated-into-tool-result",
    ),
    # KBR-308: the two NEW kept cases — assistant-message-object and
    # tool-message-object ride the same restore slots KBR-296 created for
    # the M9 rebuild (text-block restore / tool_result-block restore).
    pytest.param(
        _cc_request(
            [
                {"role": "user", "content": "ask"},
                {
                    "role": "assistant",
                    "content": "reply",
                    "cache_control": dict(_ONE_HOUR),
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file"},
            ]
        ),
        "$.messages[1].cache_control",
        {"$.messages[1].content[0].cache_control": _ONE_HOUR},
        id="assistant-message-object-kept",
    ),
    pytest.param(
        _cc_request(
            [
                {"role": "user", "content": "ask"},
                _tool_call_turn(),
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "file",
                    "cache_control": dict(_ONE_HOUR),
                },
            ]
        ),
        "$.messages[2].cache_control",
        {"$.messages[2].content[0].cache_control": _ONE_HOUR},
        id="tool-message-object-kept",
    ),
]


@pytest.mark.parametrize(("request_body", "placed", "surviving"), _GIVEN_BREAKPOINT_CASES)
def test_each_site_by_site_case_carries_its_one_breakpoint(
    request_body: dict, placed: str, surviving: dict[str, object]
) -> None:
    """Each case's request carries exactly the breakpoint its id names.

    Without this, a "dropped" case whose input lost its breakpoint (a mistyped
    key, a moved block) would pass while recording nothing.

    Args:
        request_body: A Chat Completions request carrying one breakpoint.
        placed: The request path where the breakpoint was placed.
        surviving: Unused here; the expected wire breakpoints for the next test.
    """
    assert _breakpoints(request_body) == {placed: _ONE_HOUR}


@pytest.mark.parametrize(("request_body", "placed", "surviving"), _GIVEN_BREAKPOINT_CASES)
def test_the_adapter_keeps_or_drops_a_given_breakpoint_site_by_site(
    request_body: dict, placed: str, surviving: dict[str, object]
) -> None:
    """Record, site by site, which Chat Completions breakpoints reach the Anthropic wire.

    The adapter rebuilds a user ``text`` part member-for-member (KBR-222 gave
    known parts Anthropic spellings, preserving their other members), so a
    breakpoint there survives with its position and TTL. The tool-message one is
    moved inside the ``tool_result`` block's ``content``, a depth at which it is
    not established that Anthropic honours a breakpoint, so a fix may change
    that case too. The adapter rebuilds the request, messages, ``system``,
    ``tools`` and tool calls, so a breakpoint there is dropped.

    Commercial consequence: a Chat Completions client on this provider still
    caches the tools and system prompt up to a kept user-turn breakpoint, since
    Anthropic caches the prefix in tools → system → messages order. What it
    loses is the separate cache entry at the end of the system prompt and tool
    definitions, and that entry's TTL. That entry is the one that still hits
    once later messages change.

    Args:
        request_body: A Chat Completions request carrying one breakpoint.
        placed: Unused here; the input path the previous test checks.
        surviving: Expected wire paths and values of the breakpoints that survive.
    """
    wire = AnthropicAdapter().translate_to_upstream(request_body)

    assert _breakpoints(wire) == surviving
