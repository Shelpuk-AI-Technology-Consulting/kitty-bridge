"""L1 tests for the Anthropic Messages reply reader (response direction).

`.system_design/TEST_SUITE.md` §3.3.1 (the response-direction bullet), §7.4 ·
plan task **T-A7** (KBR-39).

**Every body here comes from Anthropic's published documentation**, and each
carries a comment naming which example and when it was retrieved.  That is the
rule §3.3.1 puts on all seven readers — "validated against the published schema,
**never** against kitty's output" — and it is only checkable later if the
provenance travels with the fixture.  Bodies this module *constructs* (an
injected key, a pairing-rule trip) are marked as such.

**Every falsification case is paired with a control.**  Without the control, a
reader that residualised unconditionally would satisfy every positive assertion
in this module.  Plan §1.4's harness rule, applied to a projection.

**The module asserts on** ``verify_total`` **, not only on shapes.**  A reply
that projects the right parts while quietly dropping a key is exactly the defect
:attr:`~harness.contract.Reply.consumed` exists to catch, and a test that only
compared parts would pass over it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness import contract as c
from harness import reader_anthropic_messages as am

# --------------------------------------------------------------------------
# Published fixtures
# --------------------------------------------------------------------------

# The reference's own complete response example, from the "Create a Message"
# endpoint's `Response (200)` block.  Retrieved 2026-09-16 from
# docs.claude.com/en/api/messages.md.  Reproduced verbatim (key order
# normalised to the module's style, values untouched).
PUBLISHED_FULL_RESPONSE: dict[str, Any] = {
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "container": {
        "id": "container_011CpZohnwH4vuy7gazohgSP",
        "expires_at": "2019-12-27T18:11:19.117Z",
        "skills": [{"skill_id": "pdf", "type": "anthropic", "version": "latest"}],
    },
    "content": [
        {
            "citations": [
                {
                    "cited_text": "The grass is green. The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "end_char_index": 0,
                    "file_id": "file_011CNha8iCJcU1wXNR6q4V8w",
                    "start_char_index": 0,
                    "type": "char_location",
                }
            ],
            "text": "Hi! My name is Claude.",
            "type": "text",
        }
    ],
    "model": "claude-opus-5",
    "role": "assistant",
    "stop_details": {
        "category": "cyber",
        "explanation": "This request was declined because it conflicts with Anthropic's Usage Policy.",
        "type": "refusal",
    },
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "type": "message",
    "usage": {
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "cache_creation_input_tokens": 2051,
        "cache_read_input_tokens": 2051,
        "inference_geo": "global",
        "input_tokens": 2095,
        "output_tokens": 503,
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": {"web_fetch_requests": 2, "web_search_requests": 0},
        "service_tier": "standard",
    },
}

# The `tool_use` content block the same reference shows the model producing for
# "What's the S&P 500 at today?" (same source and date).  Wrapped here in a
# complete reply body whose non-`content` keys are the published example's.
PUBLISHED_TOOL_USE_BLOCK: dict[str, Any] = {
    "type": "tool_use",
    "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "name": "get_stock_price",
    "input": {"ticker": "^GSPC"},
}

# The `thinking` and `redacted_thinking` block shapes the reference's block
# documentation lists (same source and date): `thinking` carries the reasoning
# text and the `signature` the API returned; `redacted_thinking` carries only
# the opaque `data` payload.
PUBLISHED_THINKING_BLOCK: dict[str, Any] = {
    "type": "thinking",
    "thinking": "The user is asking about the capital of France. Paris is the answer.",
    "signature": "EuYBCkQYAiJAVk",
}
PUBLISHED_REDACTED_THINKING_BLOCK: dict[str, Any] = {
    "type": "redacted_thinking",
    "data": "Encrypted redacted-thinking payload.",
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
    """The reference's own complete response example, end to end."""

    def test_projects_clean_with_verify_total(self) -> None:
        """R1 — the published example round-trips with an empty residual."""
        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)
        assert projected.source == PUBLISHED_FULL_RESPONSE

    def test_text_part_carries_the_published_text(self) -> None:
        """R2 — the single text block projects as one ``Text`` part."""
        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Text)
        assert part.text == "Hi! My name is Claude."

    def test_stop_reason_maps_straight(self) -> None:
        """R4 — ``end_turn`` is canonical, so ``stop_reason_raw`` stays ``None``."""
        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert projected.stop_reason == "end_turn"
        assert projected.stop_reason_raw is None

    def test_usage_is_carried_whole(self) -> None:
        """§3.3.1 — usage is carried but excluded from the T-D10 diff."""
        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert projected.usage == PUBLISHED_FULL_RESPONSE["usage"]

    def test_conforms_to_the_reply_projection_protocol(self) -> None:
        """R5 — the reader satisfies ``ReplyProjection`` under ``isinstance``."""
        reader = am.AnthropicMessagesReplyProjection()

        assert isinstance(reader, c.ReplyProjection)
        assert reader.wire_format is c.WireFormat.ANTHROPIC_MESSAGES


# --------------------------------------------------------------------------
# R2 — the format's content-shape vocabulary
# --------------------------------------------------------------------------


class TestContentShapes:
    """One case per block shape the published schema lists for a reply."""

    @staticmethod
    def _body_with_content(content: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the published reply body with ``content`` swapped.

        Args:
            content: The replacement ``content`` array.

        Returns:
            The body, with every other key the published example carries.
        """
        body = dict(PUBLISHED_FULL_RESPONSE)
        body["content"] = content
        return body

    def test_tool_use_block_projects_as_tool_use(self) -> None:
        """R2 — the published ``tool_use`` block projects with name, id, arguments."""
        body = self._body_with_content([PUBLISHED_TOOL_USE_BLOCK])
        body["stop_reason"] = "tool_use"

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert projected.stop_reason == "tool_use"
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_stock_price"
        assert part.id == "toolu_01D7FLrfh4GYq7yT1ULFeyMV"
        assert part.arguments == {"ticker": "^GSPC"}

    def test_thinking_block_projects_as_thinking_with_signature(self) -> None:
        """R2 — the published ``thinking`` block projects text and signature."""
        body = self._body_with_content([PUBLISHED_THINKING_BLOCK, PUBLISHED_FULL_RESPONSE["content"][0]])

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        thinking = projected.parts[0]
        assert isinstance(thinking, c.Thinking)
        assert thinking.text == PUBLISHED_THINKING_BLOCK["thinking"]
        assert thinking.signature == PUBLISHED_THINKING_BLOCK["signature"]

    def test_redacted_thinking_projects_as_opaque(self) -> None:
        """R2 — ``redacted_thinking`` is unmodelled, so it projects as ``Opaque``."""
        body = self._body_with_content([PUBLISHED_REDACTED_THINKING_BLOCK])

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Opaque)
        assert part.kind == "redacted_thinking"
        assert part.digest  # the payload digest is present — the block stays detectable


# --------------------------------------------------------------------------
# R4 — the stop-reason mapping, canonical and escaped
# --------------------------------------------------------------------------


class TestStopReasonMapping:
    """Canonical values map straight; the rest escape through ``other``."""

    @pytest.mark.parametrize(
        ("wire_value", "expected"),
        [
            ("end_turn", "end_turn"),
            ("max_tokens", "max_tokens"),
            ("stop_sequence", "stop_sequence"),
            ("tool_use", "tool_use"),
        ],
    )
    def test_canonical_values_map_straight(self, wire_value: str, expected: str) -> None:
        """R4 — a canonical wire value maps with ``stop_reason_raw = None``."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        body["stop_reason"] = wire_value

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == expected
        assert projected.stop_reason_raw is None

    @pytest.mark.parametrize(
        "wire_value", ["pause_turn", "refusal", "model_context_window_exceeded"]
    )
    def test_unlisted_values_escape_through_other(self, wire_value: str) -> None:
        """R4 — a published value outside the canonical set projects as ``other``."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        body["stop_reason"] = wire_value

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == wire_value

    def test_absent_stop_reason_projects_as_none(self) -> None:
        """R4 — the schema types ``stop_reason`` as nullable; absent maps to ``None``."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        del body["stop_reason"]

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason is None
        assert projected.stop_reason_raw is None

    def test_pairing_rule_is_enforced_against_a_stale_raw(self) -> None:
        """R4 — a canonical stop reason beside ``stop_reason_raw`` is rejected.

        The rule lives in :meth:`~harness.contract.Reply.__post_init__` (T-W2);
        this test pins the *reader's* mapping to it: a reader that returned a
        raw value beside a canonical reason would surface here as a
        ``ValueError``, not as a silently mis-shaped projection.
        """
        with pytest.raises(ValueError, match="stop_reason_raw"):
            c.Reply(parts=(), stop_reason="end_turn", stop_reason_raw="end_turn")


# --------------------------------------------------------------------------
# R3 — the falsification pair
# --------------------------------------------------------------------------


class TestFalsification:
    """Plan §1.4: the harness must be shown to fail, with its control."""

    def test_injected_unknown_key_residualises_and_fails_totality(self) -> None:
        """R3 — one unrecognised top-level key produces a non-empty residual."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        body["x-kitty-trace"] = "injected"

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"x-kitty-trace": "injected"}
        with pytest.raises(c.ResidualFieldsError, match="x-kitty-trace"):
            c.verify_total(projected)

    def test_control_without_the_injected_key_is_clean(self) -> None:
        """R3 — the paired control: the same body without the defect is clean."""
        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)

    def test_non_string_stop_reason_residualises(self) -> None:
        """R3 — a wrongly-typed ``stop_reason`` is residualised, never coerced."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        body["stop_reason"] = 42

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"stop_reason": 42}
        with pytest.raises(c.ResidualFieldsError, match="stop_reason"):
            c.verify_total(projected)

    def test_absent_content_residualises(self) -> None:
        """R3 — ``content`` is required by the schema; absent is a named break."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        del body["content"]

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        assert "content" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="content"):
            c.verify_total(projected)

    def test_non_string_stop_reason_control_is_clean(self) -> None:
        """R3 — control for the wrongly-typed case: the published value is clean."""
        body = dict(PUBLISHED_FULL_RESPONSE)
        body["stop_reason"] = "end_turn"

        projected = am.AnthropicMessagesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
