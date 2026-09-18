"""Property tests for the streaming/non-streaming content-set agreement (KBR-277, KBR-285).

``.system_design/TEST_SUITE.md`` §6.1, first property row. Owns two claims:

* For every ``reasoning_content`` shape the streaming predicate would see, and
  for every ``tool_calls`` shape that is consistent across both predicates,
  ``_cc_chunk_carries_content`` and ``BridgeServer._is_empty_cc_response``
  agree on whether the reply has content. Pinned by one Hypothesis property
  test (KBR-277).
* The same agreement holds over the three axes KBR-285 added — ``refusal``,
  legacy dict ``function_call``, and list-typed ``content`` — with the
  reasoning/tool_calls axes held at shapes the predicates already agree on.
  Pinned by a second Hypothesis property test (KBR-285).

The non-streaming detector (``_is_empty_cc_response``'s Chat Completions arm)
and the streaming hold (``_cc_chunk_carries_content``) are two predicates
that decide the same question — *does this reply carry content the user
would render?* — over two different input shapes. KBR-248 taught the
streaming side that ``reasoning_content`` is content; KBR-277 teaches the
non-streaming side the same lesson; KBR-285 widens both sides together
(refusal, legacy ``function_call``, multimodal list ``content``). The
properties pin their agreement.

**Why ``content`` (the string axis) is held constant.** The non-streaming
detector uses ``content.strip()`` (whitespace-only content is empty) while
the streaming predicate uses ``content != ""`` (whitespace-only content is
content). This pre-existing drift is documented in ``SYSTEM_DESIGN.md`` §5.4
(KBR-248 paragraph) and is not KBR-277's or KBR-285's decision to fix. To
keep the properties sound over the other axes, the string-``content`` axis
is held at one of two shapes the two predicates agree on — ``None`` and
``""`` — both of which return False from the streaming ``content`` clause
and from the non-streaming ``has_text`` check. The **list** axis is free to
vary because both sides apply the same non-empty-list rule; only its
whitespace-free string elements make the soundness argument, and none of
the strategies emit whitespace-only string content.

**Layer.** No ``pytestmark``, so this file falls back to ``l1`` via
``tests/layers.py:_FALLBACK_LAYER`` (same posture as the shipped sibling
``test_pairing_truncation_properties.py``).
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

from kitty.bridge.server import BridgeServer, _cc_chunk_carries_content

# ── Hypothesis strategies ────────────────────────────────────────────────────
#
# ``reasoning_content`` covers every shape the streaming predicate can see:
# string (with empty / whitespace / non-empty sub-shapes), non-string scalars,
# lists, and ``None``. The literal absent-key case is pinned by the
# ``test_missing_reasoning_content_key_is_empty`` unit test; the value-None
# case is exercised here and is behaviourally equivalent because both
# predicates read via ``.get(key)`` with default ``None``.
#
# ``tool_calls`` covers the three shapes the predicates both accept: absent
# (``None``), empty list, and a one-element list. A multi-element list adds
# no signal — both predicates check for non-empty truthiness.
#
# ``content`` is held to two shapes the two predicates agree on (``None``
# and ``""``). Both predicates' ``content`` clauses return False on these,
# so they don't influence the verdict — which is exactly what the property
# needs to be sound over the ``reasoning_content`` axis.

_REASONING_STRATEGY = st.one_of(
    st.sampled_from(["", " ", "   ", "thinking hard", "x" * 64]),
    st.text(max_size=64),
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    st.lists(st.integers(), max_size=5),
    st.none(),
)

_TOOL_CALLS_STRATEGY = st.sampled_from(
    [
        None,
        [],
        [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}],
    ]
)

_CONTENT_STRATEGY = st.sampled_from([None, ""])


def _delta_message_pair(content, tool_calls, reasoning_content):
    """Build the (delta, message) shape pair both predicates read from.

    Args:
        content: Held constant to one of the two shapes the predicates agree on.
        tool_calls: Varied over the three shapes both predicates accept.
        reasoning_content: Varied over its full shape space.

    Returns:
        A 2-tuple ``(delta, message)`` carrying the same ``content``,
        ``tool_calls``, and ``reasoning_content`` keys — the only fields
        either predicate reads.
    """
    fields = {"content": content, "tool_calls": tool_calls, "reasoning_content": reasoning_content}
    return fields, dict(fields)


@given(content=_CONTENT_STRATEGY, tool_calls=_TOOL_CALLS_STRATEGY, reasoning_content=_REASONING_STRATEGY)
def test_cc_chunk_carries_content_and_is_empty_cc_response_agree_on_reasoning(
    content, tool_calls, reasoning_content
):
    """The streaming and non-streaming predicates agree on the ``reasoning_content`` axis.

    KBR-277 AC-2. The non-streaming detector (``_is_empty_cc_response``'s
    Chat Completions arm) and the streaming hold's release predicate
    (``_cc_chunk_carries_content``) are two predicates that decide the same
    question over two different input shapes. After KBR-277, they agree on
    every ``reasoning_content`` shape the streaming predicate would see.

    Args:
        content: Held to one of the two shapes the predicates agree on
            (``None``, ``""``).
        tool_calls: Varied over the three shapes both predicates accept.
        reasoning_content: Varied over its full shape space — strings of
            various lengths and emptiness, non-string scalars, lists, ``None``.
    """
    delta, message = _delta_message_pair(content, tool_calls, reasoning_content)
    chunk = {"choices": [{"delta": delta}]}
    cc_response = {"choices": [{"message": message}]}
    # Property: the two predicates agree on every input the streaming predicate would see.
    assert _cc_chunk_carries_content(chunk) is (not BridgeServer._is_empty_cc_response(cc_response))


# ── KBR-285: the widened axes ────────────────────────────────────────────────
#
# ``refusal`` covers every shape the streaming predicate can see: strings
# (empty / whitespace / non-empty), non-string scalars, and ``None``.
#
# ``function_call`` covers the legacy dict (populated and name-only), the
# empty dict (falsy — neither predicate counts it), non-dict values, and
# ``None``.
#
# ``content`` here varies over the four shapes the two predicates agree on
# (``None``, ``""``, ``[]``, one populated parts list) — the whitespace
# string shapes stay out (documented drift) and the other three axes ride
# the reasoning/tool_calls constants both predicates already agree on.

_REFUSAL_STRATEGY = st.one_of(
    st.sampled_from(["", " ", "I can't help with that.", "x" * 64]),
    st.text(max_size=64),
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    st.none(),
)

_FUNCTION_CALL_STRATEGY = st.sampled_from(
    [
        None,
        {},
        {"name": "read_file", "arguments": '{"path": "a"}'},
        {"name": "", "arguments": ""},
        42,
        "read_file",
    ]
)

_WIDENED_CONTENT_STRATEGY = st.sampled_from(
    [
        None,
        "",
        [],
        [{"type": "text", "text": "here is the chart"}],
    ]
)


@given(
    content=_WIDENED_CONTENT_STRATEGY,
    refusal=_REFUSAL_STRATEGY,
    function_call=_FUNCTION_CALL_STRATEGY,
)
def test_cc_chunk_carries_content_and_is_empty_cc_response_agree_on_widened_shapes(
    content, refusal, function_call
):
    """The predicates agree on the ``refusal`` / legacy ``function_call`` / list-``content`` axes.

    KBR-285 AC. The widening touched both predicates in the same commit; the
    property pins that they stayed in agreement over the three new axes, with
    ``reasoning_content`` and ``tool_calls`` held at shapes the predicates
    already agreed on (absent, and ``[]`` respectively).

    Args:
        content: Varied over the four list/string shapes the predicates
            agree on (drift excluded, see the module docstring).
        refusal: Varied over its full shape space.
        function_call: Varied over the legacy dict's shape space.
    """
    delta = {"content": content, "refusal": refusal, "function_call": function_call}
    message = dict(delta)
    chunk = {"choices": [{"delta": delta}]}
    cc_response = {"choices": [{"message": message}]}
    # Property: the two predicates agree on every input the streaming predicate would see.
    assert _cc_chunk_carries_content(chunk) is (not BridgeServer._is_empty_cc_response(cc_response))
