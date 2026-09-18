"""Falsification suite for :mod:`harness.sse_grammar` — T-G7 (KBR-83).

``.system_design/TEST_SUITE.md` §6.2.2 · plan task **T-G7** · requirements in
`.requirements/20260917T120913Z_sse_grammar_state_machine/REQUIREMENTS.md`.

Hand-built sequences that prove the grammar state machine can do two things:

1. **Reject every structural violation.** Every negative case ends with a
   diagnostic naming the offending event kind and position. The cases pair
   with the bridge-driven success tests, so a regression that loosens the
   grammar turns one red.
2. **Accept the amended shapes that a naive grammar would reject.** The G39
   parallel-tool-call closes-out-of-order sequence, the KBR-242/KBR-250 D4
   Responses shape, the Gemini D4 single-error-frame terminal, and the
   native Messages post-emission shape (open block + trailing error) all
   classify as documented — the classification precedence lives here too.

The module imports nothing from ``src/kitty``. The guard below asserts it,
with the house-rule positive and negative controls so a silently-emptied
pattern cannot pass.

**Layer.** No ``pytestmark``: harness tests default to ``l1`` by path (§6.4.2
of the implementation plan). The suite stays under mutmut's l1 selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.sse_grammar import (
    AnthropicMessagesGrammar,
    ChatCompletionsGrammar,
    Classification,
    GeminiGrammar,
    OpenAIResponsesGrammar,
    StreamProtocol,
    classify_response,
    grammar_for,
)

# Reuse the existing import-discipline regex. Keeping a single source of truth
# is why the contract's guard and the failures library share it; this suite
# follows the same pattern.
from harness.test_contract import _KITTY_IMPORT

# ---------------------------------------------------------------------------
# Test helpers: one frame at a time, in two line-shape dialects.
# ---------------------------------------------------------------------------


def _anthropic_frame(event: str, data: dict) -> bytes:
    """One Anthropic Messages SSE frame.

    Args:
        event: The event kind.
        data: The JSON payload.

    Returns:
        The frame bytes, terminated by a blank line.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _responses_frame(event: str, data: dict) -> bytes:
    """One Responses SSE frame.

    Args:
        event: The event kind.
        data: The JSON payload.

    Returns:
        The frame bytes, terminated by a blank line.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _cc_chunk(data: dict) -> bytes:
    """One Chat Completions SSE chunk (no event line)."""
    return f"data: {json.dumps(data)}\n\n".encode()


def _cc_done() -> bytes:
    """The literal ``[DONE]`` sentinel."""
    return b"data: [DONE]\n\n"


def _gemini_frame(payload: dict) -> bytes:
    """One Gemini SSE frame (no event line)."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def _drain(grammar, payload: bytes) -> Classification:
    """Feed ``payload`` to ``grammar`` and finish it.

    Args:
        grammar: The grammar instance under test.
        payload: The full stream as one byte string.

    Returns:
        The classification ``grammar.finish()`` reports.
    """
    grammar.feed(payload)
    return grammar.finish()


# ---------------------------------------------------------------------------
# Anthropic Messages — negative cases (must be malformed with a named kind).
# ---------------------------------------------------------------------------


class TestAnthropicMessagesMalformed:
    """Every structural violation the grammar rejects, named."""

    def test_content_block_delta_before_message_start_is_malformed(self) -> None:
        """A delta with no prior start and no prior message_start."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            _anthropic_frame(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "content_block_delta" in grammar.diagnostic

    def test_message_stop_before_message_start_is_malformed(self) -> None:
        """A bare stop with no start."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(grammar, _anthropic_frame("message_stop", {"type": "message_stop"}))
        assert verdict is Classification.MALFORMED
        assert "message_stop" in grammar.diagnostic

    def test_two_message_starts_is_malformed(self) -> None:
        """A second message_start after the message already began."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "message_start" in grammar.diagnostic

    def test_content_block_stop_with_no_open_block_is_malformed(self) -> None:
        """A stop whose index was never opened."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "content_block_stop" in grammar.diagnostic

    def test_delta_outside_its_block_window_is_malformed(self) -> None:
        """A delta at an index whose block was already closed (window rule, G39)."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    _anthropic_frame(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                    _anthropic_frame(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
                    ),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "content_block_delta" in grammar.diagnostic

    def test_frame_after_message_stop_is_malformed(self) -> None:
        """Anything after the terminal is malformed — even a ping."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    _anthropic_frame(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                    _anthropic_frame("message_stop", {"type": "message_stop"}),
                    _anthropic_frame("ping", {"type": "ping"}),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "ping" in grammar.diagnostic

    def test_error_before_message_start_is_malformed(self) -> None:
        """A bare leading ``error`` is the KBR-241-amended-out shape.

        The bridge's own fallback never fires before ``message_start`` is on
        the wire (it requires ``sr is not None`` and the first byte is
        always ``message_start``), so a stream that begins with ``error`` is a
        KBR-241 regression the guard must catch.
        """
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            _anthropic_frame(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": "x"}},
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "error" in grammar.diagnostic


# ---------------------------------------------------------------------------
# Anthropic Messages — positive controls for the amended shapes.
# ---------------------------------------------------------------------------


class TestAnthropicMessagesPositive:
    """Shapes a naive grammar would reject; this grammar accepts."""

    def test_parallel_blocks_close_out_of_order_is_a_complete_sentence(self) -> None:
        """G39: blocks may overlap and close out of order."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    _anthropic_frame(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 1,
                            "content_block": {"type": "tool_use", "id": "t1", "name": "n", "input": {}},
                        },
                    ),
                    _anthropic_frame(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
                    ),
                    _anthropic_frame(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {"type": "input_json_delta", "partial_json": "{}"},
                        },
                    ),
                    # Close out of order: index 1 first.
                    _anthropic_frame(
                        "content_block_stop", {"type": "content_block_stop", "index": 1}
                    ),
                    _anthropic_frame(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                    _anthropic_frame(
                        "message_delta",
                        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}},
                    ),
                    _anthropic_frame("message_stop", {"type": "message_stop"}),
                ]
            ),
        )
        assert verdict is Classification.COMPLETE_SENTENCE, grammar.diagnostic

    def test_native_post_emission_shape_is_truncated(self) -> None:
        """The native Messages-wire close-out shape: open block + one trailing error.

        The bridge writes exactly ``[message_start, content_block_start(0),
        content_block_delta(0), error]`` when the upstream drops on the native
        passthrough (no translator to close the block with). Open block at
        finish wins (precedence rule 2): the client's view of that block is
        unclosed, and the trailing error does not rescue it.
        """
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    _anthropic_frame(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
                    ),
                    _anthropic_frame(
                        "error",
                        {
                            "type": "error",
                            "error": {"type": "api_error", "message": "Upstream connection dropped mid-stream"},
                        },
                    ),
                ]
            ),
        )
        assert verdict is Classification.TRUNCATED, grammar.diagnostic

    def test_empty_after_emission_shape_is_error_terminal(self) -> None:
        """KBR-236 shape: blocks closed + one trailing error → error_terminal."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    _anthropic_frame(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
                    ),
                    _anthropic_frame(
                        "content_block_stop", {"type": "content_block_stop", "index": 0}
                    ),
                    _anthropic_frame(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Kitty Bridge received an empty response from upstream after emission",
                            },
                        },
                    ),
                ]
            ),
        )
        assert verdict is Classification.ERROR_TERMINAL, grammar.diagnostic


# ---------------------------------------------------------------------------
# OpenAI Responses — negative + positive (KBR-242/KBR-250 D4).
# ---------------------------------------------------------------------------



def _responses_function_call_args_delta(output_index: int) -> dict:
    """The data payload for a Responses function-call arguments delta."""
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": output_index,
        "item_id": f"item_{output_index}",
        "call_id": "c",
        "delta": "{}",
    }


def _responses_text_delta(output_index: int, content_index: int = 0) -> bytes:
    """A Responses text-delta frame."""
    return _responses_frame(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"item_{output_index}",
            "output_index": output_index,
            "content_index": content_index,
            "delta": "x",
        },
    )


def _responses_text_done(output_index: int, content_index: int = 0) -> bytes:
    """A Responses text-done frame."""
    return _responses_frame(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "sequence_number": 1,
            "item_id": f"item_{output_index}",
            "output_index": output_index,
            "content_index": content_index,
            "text": "x",
        },
    )



def _responses_created(seq: int = 0) -> bytes:
    """A Responses ``response.created`` frame."""
    return _responses_frame(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": seq,
            "response": {"id": "r", "object": "response", "status": "in_progress"},
        },
    )


def _responses_completed(seq: int = 10, status: str = "completed") -> bytes:
    """A Responses ``response.completed`` frame."""
    return _responses_frame(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": seq,
            "response": {"id": "r", "object": "response", "status": status, "output": []},
        },
    )


def _responses_part_done(seq: int, output_index: int) -> bytes:
    """A Responses ``response.content_part.done`` frame."""
    return _responses_frame(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "sequence_number": seq,
            "item_id": f"item_{output_index}",
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "x"},
        },
    )


def _responses_item_added(output_index: int, item_type: str = "message") -> bytes:
    """A Responses output-item-added frame."""
    return _responses_frame(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": output_index,
            "item": {"type": item_type, "id": f"item_{output_index}", "status": "in_progress"},
        },
    )


def _responses_item_done(output_index: int) -> bytes:
    """A Responses output-item-done frame."""
    return _responses_frame(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": output_index,
            "item": {"type": "message", "id": f"item_{output_index}", "status": "completed"},
        },
    )


class TestOpenAIResponsesMalformed:
    """Responses-shape violations the grammar rejects."""

    def test_text_delta_before_any_open_item_is_malformed(self) -> None:
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(grammar, _responses_text_delta(0))
        assert verdict is Classification.MALFORMED
        assert "response.output_text.delta" in grammar.diagnostic

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("output_index", [0]),
            ("output_index", {"k": 0}),
            ("content_index", [0]),
            ("content_index", {"k": 0}),
        ],
    )
    def test_unhashable_index_values_classify_malformed_not_typeerror(
        self, field: str, value: object
    ) -> None:
        """A list-or-dict index must be ``malformed``, never a ``TypeError``.

        The grammar's contract is to classify hostile sequences; the first
        draft used these payload fields raw as dict keys / set members, and a
        schemathesis-shaped body raised ``TypeError`` (unhashable) from the
        lookup instead.
        """
        grammar = OpenAIResponsesGrammar()
        if field == "output_index":
            payload: dict = {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "item": {"type": "message", "id": "item_0", "status": "in_progress"},
                field: value,
            }
            frames = [_responses_frame("response.output_item.added", payload)]
        else:
            # content_index rides content_part.added, which requires an open
            # item — the item frames must come first so the failure lands on
            # the index validation, not the unknown-item one.
            payload = {
                "type": "response.content_part.added",
                "sequence_number": 3,
                "item_id": "item_0",
                "output_index": 0,
                "content_index": value,
                "part": {"type": "output_text", "text": ""},
            }
            frames = [
                _responses_created(0),
                _responses_item_added(0, item_type="message"),
                _responses_frame("response.content_part.added", payload),
            ]
        verdict = _drain(grammar, b"".join(frames))
        assert verdict is Classification.MALFORMED
        assert field in grammar.diagnostic

    def test_unhashable_output_index_through_open_item_is_malformed(self) -> None:
        """A delta whose ``output_index`` is a list must be ``malformed``, not ``TypeError``.

        The round-5 fix validated the index at ``output_item.added``/``done``
        but missed ``_open_item`` — the first touch for five event kinds. A
        well-formed frame whose ``output_index`` is ``[]`` goes through
        ``self._items.get(_index_key(...))`` there, which raised
        ``TypeError: unhashable type`` instead of classifying.
        """
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _responses_created(0),
                    _responses_frame(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "sequence_number": 1,
                            "item_id": "item_0",
                            "output_index": [],
                            "content_index": 0,
                            "delta": "x",
                        },
                    ),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "output_index" in grammar.diagnostic

    def test_content_part_done_with_unhashable_content_index_is_malformed(self) -> None:
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _responses_created(0),
                    _responses_item_added(0, item_type="message"),
                    _responses_frame(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "sequence_number": 1,
                            "item_id": "item_0",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        },
                    ),
                    _responses_frame(
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "sequence_number": 2,
                            "item_id": "item_0",
                            "output_index": 0,
                            "content_index": {"bad": "shape"},
                        },
                    ),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED
        assert "content_index" in grammar.diagnostic

    def test_unknown_response_kind_is_malformed(self) -> None:
        """An event the bridge does not emit — drift the guard must catch."""
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(grammar, _responses_frame("response.widget_added", {"type": "response.widget_added"}))
        assert verdict is Classification.MALFORMED
        assert "unknown event kind" in grammar.diagnostic

    def test_function_call_arguments_delta_against_a_message_item_is_malformed(self) -> None:
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _responses_created(0),
                    _responses_item_added(0, item_type="message"),
                    _responses_frame(
                        "response.function_call_arguments.delta",
                        _responses_function_call_args_delta(0),
                    ),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED


class TestOpenAIResponsesPositive:
    """The KBR-242/KBR-250 shapes a naive grammar rejects."""

    def test_d4_exhaustion_with_lazy_lifecycle_is_a_complete_sentence(self) -> None:
        """KBR-242 + KBR-250: no ``response.created``; ``[error, response.completed]`` is the terminal."""
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _responses_frame(
                        "error",
                        {"type": "error", "sequence_number": 1, "code": "empty_response", "message": "x"},
                    ),
                    _responses_completed(2, "incomplete")
                ]
            ),
        )
        assert verdict is Classification.COMPLETE_SENTENCE, grammar.diagnostic

    def test_overlapping_items_close_out_of_order(self) -> None:
        """G40: items may overlap and close out of order; all closed by completed."""
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _responses_created(0),
                    _responses_item_added(0, item_type="message"),
                    _responses_item_added(1, item_type="message"),
                    # Open the content parts for both items before the deltas —
                    # the OpenAI grammar requires content_part.added before any
                    # output_text.delta, and the Sub2API bug (missing
                    # content_part.added breaking Codex CLI) is the failure
                    # class this requirement guards.
                    _responses_frame(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "sequence_number": 1,
                            "item_id": "item_0",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        },
                    ),
                    _responses_frame(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "sequence_number": 2,
                            "item_id": "item_1",
                            "output_index": 1,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        },
                    ),
                    _responses_text_delta(0),
                    _responses_text_delta(1),
                    # Close out of order: index 1 first.
                    _responses_text_done(1),
                    _responses_part_done(3, 1),
                    _responses_item_done(1),
                    _responses_text_done(0),
                    _responses_part_done(4, 0),
                    _responses_item_done(0),
                    _responses_completed(10, "completed")
                ]
            ),
        )
        assert verdict is Classification.COMPLETE_SENTENCE, grammar.diagnostic

    def test_function_call_item_sequence(self) -> None:
        """Bridge's function-call item path (G40)."""
        grammar = OpenAIResponsesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _responses_created(0),
                    _responses_item_added(0, item_type="function_call"),
                    _responses_frame(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": 1,
                            "item_id": "item_0",
                            "output_index": 0,
                            "call_id": "c",
                            "delta": "{}",
                        },
                    ),
                    _responses_frame(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "sequence_number": 2,
                            "item_id": "item_0",
                            "output_index": 0,
                            "call_id": "c",
                            "arguments": "{}",
                        },
                    ),
                    _responses_item_done(0),
                    _responses_completed(10, "completed")
                ]
            ),
        )
        assert verdict is Classification.COMPLETE_SENTENCE, grammar.diagnostic


# ---------------------------------------------------------------------------
# Chat Completions — negative + finish_reason relaxation (S2).
# ---------------------------------------------------------------------------


class TestChatCompletionsMalformed:
    def test_chunk_after_done_is_malformed(self) -> None:
        grammar = ChatCompletionsGrammar()
        verdict = _drain(
            grammar,
            b"".join([_cc_done(), _cc_chunk({"id": "x", "choices": []})]),
        )
        assert verdict is Classification.MALFORMED

    def test_two_done_sentinels_is_malformed(self) -> None:
        grammar = ChatCompletionsGrammar()
        verdict = _drain(grammar, _cc_done() + _cc_done())
        assert verdict is Classification.MALFORMED

    def test_non_json_chunk_is_malformed(self) -> None:
        grammar = ChatCompletionsGrammar()
        verdict = _drain(grammar, b"data: not-json\n\n")
        assert verdict is Classification.MALFORMED


class TestChatCompletionsPositive:
    def test_done_without_finish_reason_is_a_complete_sentence(self) -> None:
        """S2: ``[DONE]`` is the terminal signal; ``finish_reason`` is not required."""
        grammar = ChatCompletionsGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _cc_chunk(
                        {"id": "x", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "hi"}}]}
                    ),
                    _cc_done(),
                ]
            ),
        )
        assert verdict is Classification.COMPLETE_SENTENCE

    def test_error_frame_instead_of_done_is_error_terminal(self) -> None:
        grammar = ChatCompletionsGrammar()
        verdict = _drain(
            grammar,
            _cc_chunk({"error": {"code": "upstream_error", "message": "x"}}),
        )
        assert verdict is Classification.ERROR_TERMINAL


# ---------------------------------------------------------------------------
# Gemini — negative + D4 positive.
# ---------------------------------------------------------------------------


class TestGeminiMalformed:
    def test_chunk_with_neither_candidates_nor_error_is_malformed(self) -> None:
        grammar = GeminiGrammar()
        verdict = _drain(grammar, _gemini_frame({"usageMetadata": {"tokenCount": 1}}))
        assert verdict is Classification.MALFORMED

    def test_frame_after_error_is_malformed(self) -> None:
        grammar = GeminiGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _gemini_frame({"error": {"code": 502, "message": "x", "reason": "empty_response"}}),
                    _gemini_frame({"candidates": [{"content": {"parts": [{"text": "x"}]}}]}),
                ]
            ),
        )
        assert verdict is Classification.MALFORMED


class TestGeminiPositive:
    def test_candidate_stream_that_ends_is_complete(self) -> None:
        """No ``[DONE]`` sentinel; ending on a candidate frame is the happy path."""
        grammar = GeminiGrammar()
        verdict = _drain(
            grammar,
            _gemini_frame({"candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}]}),
        )
        assert verdict is Classification.COMPLETE_SENTENCE

    def test_d4_single_error_frame_is_error_terminal(self) -> None:
        """KBR-250 Gemini D4: single ``{"error": … "reason": "empty_response"}`` frame + EOF."""
        grammar = GeminiGrammar()
        verdict = _drain(
            grammar,
            _gemini_frame({"error": {"code": 502, "message": "x", "reason": "empty_response"}}),
        )
        assert verdict is Classification.ERROR_TERMINAL


# ---------------------------------------------------------------------------
# Classification precedence (one grammar, multiple applicable rules).
# ---------------------------------------------------------------------------


class TestPrecedence:
    """Two rules can both apply to one stream; the precedence is fixed."""

    def test_open_block_plus_error_classifies_truncated_not_error_terminal(self) -> None:
        """Native post-emission shape: precedence rule 2 wins over rule 4."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    _anthropic_frame(
                        "error",
                        {"type": "error", "error": {"type": "api_error", "message": "x"}},
                    ),
                ]
            ),
        )
        # The earlier test in TestAnthropicMessagesPositive covers the same
        # shape. This one lives separately so the precedence rule reads as a
        # contract: the diagnostic is *not* the test.
        assert verdict is Classification.TRUNCATED

    def test_malformed_wins_over_truncated(self) -> None:
        """An illegal event anywhere in the stream is malformed regardless of structure."""
        grammar = AnthropicMessagesGrammar()
        verdict = _drain(
            grammar,
            b"".join(
                [
                    _anthropic_frame("message_start", {"type": "message_start", "message": {}}),
                    _anthropic_frame(
                        "content_block_start",
                        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ),
                    # Frame after message_start with no kind — illegal.
                    b"event:\ndata: {}\n\n",
                ]
            ),
        )
        assert verdict is Classification.MALFORMED


# ---------------------------------------------------------------------------
# classify_response — the entry point the bridge-driven tests call.
# ---------------------------------------------------------------------------


class TestClassifyResponse:
    """The whole-response helper covers ``json_error`` and the stream dispatch."""

    def test_stream_body_classifies_via_grammar(self) -> None:
        chunk = {"id": "x", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "hi"}}]}
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        verdict = classify_response(StreamProtocol.CHAT_COMPLETIONS, 200, body)
        assert verdict is Classification.COMPLETE_SENTENCE

    def test_json_error_body_is_json_error(self) -> None:
        body = json.dumps({"type": "error", "error": {"type": "api_error", "message": "x", "reason": "empty_response"}})
        verdict = classify_response(StreamProtocol.MESSAGES, 502, body)
        assert verdict is Classification.JSON_ERROR

    def test_non_json_non_stream_body_is_malformed(self) -> None:
        verdict = classify_response(StreamProtocol.MESSAGES, 200, "not a stream")
        assert verdict is Classification.MALFORMED

    def test_empty_body_with_4xx_is_json_error(self) -> None:
        """An empty body on an error status is the close-out envelope the bridge omits."""
        verdict = classify_response(StreamProtocol.MESSAGES, 500, "")
        assert verdict is Classification.JSON_ERROR

    def test_empty_body_with_2xx_is_truncated(self) -> None:
        """The other half of the empty-body split: no bytes on a 2xx is a stream cut before its first frame.

        Both halves are deliberate: an empty body alone cannot say whether the
        bridge meant an error envelope it never wrote (4xx → ``json_error``)
        or a stream that was cut before its first frame (2xx →
        ``truncated``). The status decides, and this test pins the 2xx side
        so a refactor that collapses the split is visible.
        """
        verdict = classify_response(StreamProtocol.MESSAGES, 200, "")
        assert verdict is Classification.TRUNCATED


class TestFinishEscapeHatch:
    """``feed(b"")`` must disarm finish()'s never-fed RuntimeError.

    The first draft's escape hatch was broken: an empty decode appends
    nothing, so position and buffer both stayed at zero and the same
    RuntimeError re-fired — the message prescribed a remedy that did
    nothing.
    """

    def test_feed_empty_bytes_then_finish_classifies_truncated(self) -> None:
        grammar = AnthropicMessagesGrammar()
        grammar.feed(b"")
        assert grammar.finish() is Classification.TRUNCATED

    def test_finish_without_any_feed_raises(self) -> None:
        grammar = AnthropicMessagesGrammar()
        with pytest.raises(RuntimeError, match="never fed"):
            grammar.finish()

    def test_feed_after_finish_raises(self) -> None:
        grammar = AnthropicMessagesGrammar()
        grammar.feed(b"")
        grammar.finish()
        with pytest.raises(RuntimeError, match="after finish"):
            grammar.feed(b"")


# ---------------------------------------------------------------------------
# grammar_for dispatch + the StreamProtocol/InboundProtocol contract.
# ---------------------------------------------------------------------------


class TestGrammarFor:
    def test_returns_correct_subclass_per_protocol(self) -> None:
        assert isinstance(grammar_for(StreamProtocol.MESSAGES), AnthropicMessagesGrammar)
        assert isinstance(grammar_for(StreamProtocol.RESPONSES), OpenAIResponsesGrammar)
        assert isinstance(grammar_for(StreamProtocol.CHAT_COMPLETIONS), ChatCompletionsGrammar)
        assert isinstance(grammar_for(StreamProtocol.GEMINI), GeminiGrammar)

    def test_unknown_protocol_raises(self) -> None:
        with pytest.raises(ValueError):
            # An Enum mismatch surfaces as KeyError then ValueError via the
            # ``raise … from None`` in grammar_for.
            grammar_for("nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Import discipline — F11.
# ---------------------------------------------------------------------------

#: Modules whose source must contain no ``from kitty`` / ``import kitty`` line.
#: Named once at module top per the review's S5: a third harness module is one
#: entry, not a third guard.
#: Modules whose source must contain no ``from kitty`` / ``import kitty`` line.
#: Named once at module top per the review's S5: a third harness module is one
#: entry, not a third guard. This test module itself is deliberately NOT in
#: the set — its positive control below must carry literal kitty-import
#: strings to prove the regex fires, so scanning it would flag the control as
#: an offence. The scan iterates this set, so an entry added here is scanned,
#: not merely declared.
_HARNESS_MODULES_UNDER_GUARD: frozenset[str] = frozenset(
    {"tests/harness/sse_grammar.py"}
)


class TestImportDiscipline:
    """The grammar suite must not import ``src/kitty``."""

    def test_every_module_under_guard_imports_nothing_from_kitty(self) -> None:
        """Scan every file the registry names, not one hardcoded path.

        The first draft declared a two-module frozenset and then hardcoded a
        single ``sse_grammar.py`` path in the scan — the registry was
        consulted only by the existence self-check, so a module added to the
        set was guarded in name only.
        """
        repo_root = Path(__file__).resolve().parents[2]
        for name in sorted(_HARNESS_MODULES_UNDER_GUARD):
            path = repo_root / name
            source = path.read_text(encoding="utf-8")
            # Self-check: a guard that passes on empty input is the house
            # rule ``tests/test_egress_coverage.py`` warns about.
            assert len(source) > 1000, f"{name} was empty; the guard would pass vacuously"

            offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]
            assert offending == [], f"{name} must not import kitty: {offending}"

    def test_the_import_guard_actually_fires_on_every_form_it_claims(self) -> None:
        """Positive control: the regex catches every form its docstring claims.

        The same control :mod:`harness.test_contract` carries for the contract
        module; replicated here so a third harness module's guard cannot drift
        out of sync silently.
        """
        forms = [
            "from kitty.bridge import server",
            "import kitty",
            "from  kitty import server",
            "import  kitty.bridge",
            "from src.kitty import server",
            'mod = importlib.import_module("kitty.bridge.server")',
            '__import__("kitty")',
            'mod = importlib.import_module("src.kitty.bridge.server")',
            '__import__("src.kitty")',
        ]
        undetected = [form for form in forms if not _KITTY_IMPORT.search(form)]
        assert undetected == [], f"the guard would miss these: {undetected}"

    def test_the_import_guard_does_not_fire_on_innocent_text(self) -> None:
        """Negative control: a too-broad pattern would pass the positive above."""
        innocent = [
            "# kitty-bridge is the product under test",
            "from harness import contract",
            "kitty = 1",
            "# never write `import kitty` in this module",
        ]
        assert [line for line in innocent if _KITTY_IMPORT.search(line)] == []
