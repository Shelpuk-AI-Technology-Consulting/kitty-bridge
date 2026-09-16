"""L1 tests for the Chat Completions reply reader (response direction).

`.system_design/TEST_SUITE.md` §3.3.1 (the response-direction bullet), §7.4 ·
plan task **T-A7** (KBR-39).

**Every body here comes from OpenAI's published schema** — the
``CreateChatCompletionResponse`` example and ``choices[].finish_reason`` enum
of ``openai/openai-openapi`` master, retrieved 2026-09-16 — and each carries a
comment naming which example.  §3.3.1's rule: validated against the published
schema, **never** against kitty's output.  Bodies this module *constructs* (an
injected key, a pairing-rule trip) are marked as such.

**Every falsification case is paired with a control**, per plan §1.4's harness
rule applied to a projection.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness import contract as c
from harness import reader_chat_completions as cc

# --------------------------------------------------------------------------
# Published fixtures
# --------------------------------------------------------------------------

# The ``CreateChatCompletionResponse`` schema's own ``example`` block
# (`openai/openai-openapi` master, retrieved 2026-09-16).  Reproduced verbatim.
PUBLISHED_FULL_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
    "object": "chat.completion",
    "created": 1741570283,
    "model": "gpt-6-astra",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "The image shows a wooden boardwalk path running through a lush green "
                    "field or meadow. The sky is bright blue with some scattered clouds, "
                    "giving the scene a serene and peaceful atmosphere. Trees and shrubs "
                    "are visible in the background."
                ),
                "refusal": None,
                "annotations": [],
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1117,
        "completion_tokens": 46,
        "total_tokens": 1163,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
    "service_tier": "default",
    "system_fingerprint": "fp_fc9f1d7035",
}

# The ``tool_calls`` shape the ``CreateChatCompletionResponse`` schema documents:
# a message carrying ``tool_calls[].function`` with JSON-string ``arguments``
# and a ``finish_reason`` of ``tool_calls``.  Same source and date.
PUBLISHED_TOOL_CALL_MESSAGE: dict[str, Any] = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_current_weather", "arguments": '{"location": "Boston, MA"}'},
        }
    ],
}


def _reply(body: dict[str, Any]) -> c.CapturedReply:
    """Wrap a body dict in the capture type the reader takes.

    Args:
        body: The reply body.

    Returns:
        The capture, with the body serialised the way the wire carries it.
    """
    return c.CapturedReply(status=200, headers=(), body=json.dumps(body).encode("utf-8"))


# --------------------------------------------------------------------------
# R1 — the published example projects with verify_total clean
# --------------------------------------------------------------------------


class TestPublishedExample:
    """The schema's own complete response example, end to end."""

    def test_projects_clean_with_verify_total(self) -> None:
        """R1 — the published example round-trips with an empty residual."""
        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)
        assert projected.source == PUBLISHED_FULL_RESPONSE

    def test_text_part_carries_the_published_content(self) -> None:
        """R2 — the message content projects as one ``Text`` part."""
        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Text)
        assert part.text.startswith("The image shows a wooden boardwalk path")

    def test_stop_reason_maps_straight(self) -> None:
        """R4 — ``stop`` maps to ``end_turn`` with ``stop_reason_raw`` unset."""
        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert projected.stop_reason == "end_turn"
        assert projected.stop_reason_raw is None

    def test_usage_is_carried_whole(self) -> None:
        """§3.3.1 — usage is carried but excluded from the T-D10 diff."""
        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert projected.usage == PUBLISHED_FULL_RESPONSE["usage"]

    def test_conforms_to_the_reply_projection_protocol(self) -> None:
        """R5 — the reader satisfies ``ReplyProjection`` under ``isinstance``."""
        reader = cc.ChatCompletionsReplyProjection()

        assert isinstance(reader, c.ReplyProjection)
        assert reader.wire_format is c.WireFormat.CHAT_COMPLETIONS


# --------------------------------------------------------------------------
# R2 — the format's content-shape vocabulary
# --------------------------------------------------------------------------


class TestContentShapes:
    """One case per message shape the published schema lists for a reply."""

    @staticmethod
    def _body_with_message(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
        """Return the published reply body with the first choice's message swapped.

        Args:
            message: The replacement ``message`` object.
            finish_reason: The matching ``finish_reason``.

        Returns:
            The body, with every other key the published example carries.
        """
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["choices"][0]["message"] = message
        body["choices"][0]["finish_reason"] = finish_reason
        return body

    def test_tool_call_projects_with_decoded_arguments(self) -> None:
        """R2 — JSON-string arguments decode through the shared rule (KBR-174)."""
        body = self._body_with_message(PUBLISHED_TOOL_CALL_MESSAGE, "tool_calls")

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert projected.stop_reason == "tool_use"
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_current_weather"
        assert part.id == "call_abc123"
        assert part.arguments == {"location": "Boston, MA"}

    def test_refusal_projects_as_text(self) -> None:
        """R2 — a refusal is content whose identity is a run of text (§7.4.1)."""
        message = {"role": "assistant", "content": None, "refusal": "I can't help with that."}
        body = self._body_with_message(message, "content_filter")

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert projected.stop_reason == "error"
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Text)
        assert part.text == "I can't help with that."

    def test_reasoning_content_projects_as_thinking(self) -> None:
        """R2 — ``reasoning_content`` (P8's complement) projects as ``Thinking``."""
        message = {"role": "assistant", "content": "Answer.", "reasoning_content": "Because 2+2=4."}
        body = self._body_with_message(message, "stop")

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        thinking = projected.parts[1]
        assert isinstance(thinking, c.Thinking)
        assert thinking.text == "Because 2+2=4."

    def test_extra_choice_residualises(self) -> None:
        """R2/§3.3.1 — a bridge reply with ``n > 1`` is a named anomaly."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["choices"].append(json.loads(json.dumps(body["choices"][0])))

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        assert "choices[1]" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="choices\\[1\\]"):
            c.verify_total(projected)


# --------------------------------------------------------------------------
# R4 — the finish-reason mapping, canonical and escaped
# --------------------------------------------------------------------------


class TestFinishReasonMapping:
    """Canonical values map straight; the rest escape through ``other``."""

    @pytest.mark.parametrize(
        ("wire_value", "expected"),
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("function_call", "tool_use"),
            ("content_filter", "error"),
        ],
    )
    def test_published_values_map_straight(self, wire_value: str, expected: str) -> None:
        """R4 — every published ``finish_reason`` maps onto the canonical set."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["choices"][0]["finish_reason"] = wire_value

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == expected
        assert projected.stop_reason_raw is None

    def test_null_finish_reason_projects_as_none(self) -> None:
        """R4 — the schema types ``finish_reason`` as nullable; ``null`` maps to ``None``."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["choices"][0]["finish_reason"] = None

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason is None
        assert projected.stop_reason_raw is None

    def test_unpublished_value_escapes_through_other(self) -> None:
        """R4 — a value outside the published enum escapes with the wire string."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["choices"][0]["finish_reason"] = "brand_new_reason"

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == "brand_new_reason"

    def test_pairing_rule_is_enforced_against_a_stale_raw(self) -> None:
        """R4 — a canonical stop reason beside ``stop_reason_raw`` is rejected.

        The rule lives in :meth:`~harness.contract.Reply.__post_init__` (T-W2);
        this test pins the *reader's* mapping to it.
        """
        with pytest.raises(ValueError, match="stop_reason_raw"):
            c.Reply(parts=(), stop_reason="end_turn", stop_reason_raw="stop")


# --------------------------------------------------------------------------
# R3 — the falsification pair
# --------------------------------------------------------------------------


class TestFalsification:
    """Plan §1.4: the harness must be shown to fail, with its control."""

    def test_injected_unknown_key_residualises_and_fails_totality(self) -> None:
        """R3 — one unrecognised top-level key produces a non-empty residual."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["x-kitty-trace"] = "injected"

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"x-kitty-trace": "injected"}
        with pytest.raises(c.ResidualFieldsError, match="x-kitty-trace"):
            c.verify_total(projected)

    def test_control_without_the_injected_key_is_clean(self) -> None:
        """R3 — the paired control: the same body without the defect is clean."""
        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)

    def test_unknown_message_key_residualises(self) -> None:
        """R3 — an unmodelled key inside the message fails closed at depth."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["choices"][0]["message"]["x_vendor_marker"] = 1

        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"choices[0].message.x_vendor_marker": 1}
        with pytest.raises(c.ResidualFieldsError, match="x_vendor_marker"):
            c.verify_total(projected)

    def test_unknown_message_key_control_is_clean(self) -> None:
        """R3 — control for the depth case: the published message is clean."""
        projected = cc.ChatCompletionsReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)
