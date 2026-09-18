"""Property tests for the streaming/non-streaming ``reasoning_content`` symmetry (KBR-277).

``.system_design/TEST_SUITE.md`` §6.1, first property row. Owns one claim:

* For every ``reasoning_content`` shape the streaming predicate would see, and
  for every ``tool_calls`` shape that is consistent across both predicates,
  ``_cc_chunk_carries_content`` and ``BridgeServer._is_empty_cc_response``
  agree on whether the reply has content. Pinned by one Hypothesis property
  test.

The non-streaming detector (``_is_empty_cc_response``'s Chat Completions arm)
and the streaming hold (``_cc_chunk_carries_content``) are two predicates
that decide the same question — *does this reply carry content the user
would render?* — over two different input shapes. KBR-248 taught the
streaming side that ``reasoning_content`` is content; KBR-277 teaches the
non-streaming side the same lesson. The property pins their agreement.

**Why ``content`` is held constant.** The non-streaming detector uses
``content.strip()`` (whitespace-only content is empty) while the streaming
predicate uses ``content != ""`` (whitespace-only content is content). This
pre-existing drift is documented in ``SYSTEM_DESIGN.md`` §5.4 (KBR-248
paragraph) and is not KBR-277's decision to fix. To keep the property
sound over the ``reasoning_content`` axis, ``content`` is held at one of
two shapes the two predicates agree on — ``None`` and ``""`` — both of
which return False from the streaming ``content`` clause and from the
non-streaming ``has_text`` check.

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
