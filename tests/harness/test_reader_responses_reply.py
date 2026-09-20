"""L1 tests for the OpenAI Responses reply reader (response direction).

`.system_design/TEST_SUITE.md` §3.3.1 (the response-direction bullet), §7.4 ·
plan task **T-A7** (KBR-39).

**Every body here comes from OpenAI's published schema** — the
``CreateResponse`` example of ``openai/openai-openapi`` master, retrieved
2026-09-16 — and each carries a comment naming which example.  §3.3.1's rule:
validated against the published schema, **never** against kitty's output.
Bodies this module *constructs* (an injected key, a mid-stream status) are
marked as such.

**Every falsification case is paired with a control**, per plan §1.4's harness
rule applied to a projection.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness import contract as c
from harness import reader_responses as rsp

# --------------------------------------------------------------------------
# Published fixtures
# --------------------------------------------------------------------------

# The ``CreateResponse`` schema's own complete ``response`` example
# (`openai/openai-openapi` master, retrieved 2026-09-16).  Reproduced verbatim.
PUBLISHED_FULL_RESPONSE: dict[str, Any] = {
    "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
    "object": "response",
    "created_at": 1741476542,
    "status": "completed",
    "completed_at": 1741476543,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [
        {
            "type": "message",
            "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": (
                        "In a peaceful grove beneath a silver moon, a unicorn named Lumina "
                        "discovered a hidden pool that reflected the stars. As she dipped her "
                        "horn into the water, the pool began to shimmer, revealing a pathway "
                        "to a magical realm of endless night skies. Filled with wonder, Lumina "
                        "whispered a wish for all who dream to find their own hidden magic, "
                        "and as she glanced back, her hoofprints sparkled like stardust."
                    ),
                    "annotations": [],
                }
            ],
        }
    ],
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "reasoning": {"effort": None, "summary": None},
    "store": True,
    "temperature": 1.0,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1.0,
    "truncation": "disabled",
    "usage": {
        "input_tokens": 36,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 87,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 123,
    },
    "user": None,
    "metadata": {},
}

# The ``function_call`` and ``reasoning`` output-item shapes the same schema
# documents.  Same source and date.
PUBLISHED_FUNCTION_CALL_ITEM: dict[str, Any] = {
    "type": "function_call",
    "id": "fc_67caf9736fc481909c5ba73204aba0ff062296c4537dd958",
    "status": "completed",
    "call_id": "call_234kdga09jf",
    "name": "get_weather",
    "arguments": '{"location": "Paris"}',
}
PUBLISHED_REASONING_ITEM: dict[str, Any] = {
    "type": "reasoning",
    "id": "rs_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
    "summary": [{"type": "summary_text", "text": "Weighing two options."}],
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
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)
        assert projected.source == PUBLISHED_FULL_RESPONSE

    def test_text_part_carries_the_published_text(self) -> None:
        """R2 — the message's ``output_text`` block projects as one ``Text`` part."""
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Text)
        assert part.text.startswith("In a peaceful grove beneath a silver moon")

    def test_completed_status_maps_to_end_turn(self) -> None:
        """R4 — ``status = completed`` with no incomplete reason maps to ``end_turn``."""
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert projected.stop_reason == "end_turn"
        assert projected.stop_reason_raw is None

    def test_usage_is_carried_whole(self) -> None:
        """§3.3.1 — usage is carried but excluded from the T-D10 diff."""
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert projected.usage == PUBLISHED_FULL_RESPONSE["usage"]

    def test_conforms_to_the_reply_projection_protocol(self) -> None:
        """R5 — the reader satisfies ``ReplyProjection`` under ``isinstance``."""
        reader = rsp.ResponsesReplyProjection()

        assert isinstance(reader, c.ReplyProjection)
        assert reader.wire_format is c.WireFormat.OPENAI_RESPONSES

    def test_parallel_tool_calls_is_consumed_without_an_extra_address(self) -> None:
        """KBR-273 — the reply direction has no ``extra`` address to violate.

        ``Reply`` carries no ``envelope.extra``, so every echoed request
        control field — ``parallel_tool_calls`` among them — is consumed
        silently. The §3.3.1b non-default rule binds the *request* readers'
        ``extra`` writes; this assertion pins the asymmetry so a future
        maintainer cannot mistake the absence for an oversight.
        """
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        assert "parallel_tool_calls" in projected.consumed
        assert projected.residual == {}


# --------------------------------------------------------------------------
# R2 — the format's content-shape vocabulary
# --------------------------------------------------------------------------


class TestContentShapes:
    """One case per output-item shape the published schema lists for a reply."""

    @staticmethod
    def _body_with_output(output: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the published reply body with ``output`` swapped.

        Args:
            output: The replacement ``output`` array.

        Returns:
            The body, with every other key the published example carries.
        """
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["output"] = output
        return body

    def test_function_call_projects_with_decoded_arguments(self) -> None:
        """R2 — JSON-string arguments decode through the shared rule (KBR-174)."""
        body = self._body_with_output([PUBLISHED_FUNCTION_CALL_ITEM])

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_weather"
        assert part.id == "call_234kdga09jf"
        assert part.arguments == {"location": "Paris"}

    def test_reasoning_summary_projects_as_thinking(self) -> None:
        """R2 — ``reasoning`` ``summary_text`` joins into one ``Thinking``."""
        body = self._body_with_output([PUBLISHED_REASONING_ITEM, PUBLISHED_FULL_RESPONSE["output"][0]])

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        thinking = projected.parts[0]
        assert isinstance(thinking, c.Thinking)
        assert thinking.text == "Weighing two options."
        assert thinking.signature == PUBLISHED_REASONING_ITEM["id"]

    def test_refusal_content_block_projects_as_text(self) -> None:
        """R2 — a ``refusal`` content block is content whose identity is text."""
        message = {
            "type": "message",
            "id": "msg_refusal",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I can't help with that.", "annotations": []}],
        }
        body = self._body_with_output([message])

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert len(projected.parts) == 1
        assert isinstance(projected.parts[0], c.Text)
        assert projected.parts[0].text == "I can't help with that."

    def test_unmodelled_item_type_projects_as_opaque(self) -> None:
        """R2 — a ``web_search_call`` item stays detectable as ``Opaque`` by digest."""
        body = self._body_with_output(
            [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                PUBLISHED_FULL_RESPONSE["output"][0],
            ]
        )

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        first = projected.parts[0]
        assert isinstance(first, c.Opaque)
        assert first.kind == "web_search_call"
        assert first.digest


# --------------------------------------------------------------------------
# R4 — the status mapping, canonical and escaped
# --------------------------------------------------------------------------


class TestStatusMapping:
    """Terminal statuses map; incomplete reasons escape through ``other``."""

    def test_max_output_tokens_reason_maps_straight(self) -> None:
        """R4 — the published ``incomplete_details.reason`` maps to ``max_tokens``."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["incomplete_details"] = {"reason": "max_output_tokens"}

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "max_tokens"
        assert projected.stop_reason_raw is None

    def test_failed_status_maps_to_error(self) -> None:
        """R4 — ``status = failed`` maps to ``error`` with ``stop_reason_raw`` unset."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["status"] = "failed"

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "error"
        assert projected.stop_reason_raw is None

    def test_cancelled_status_escapes_through_other(self) -> None:
        """R4 — ``cancelled`` is terminal but escapes with the wire string."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["status"] = "cancelled"

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == "cancelled"

    def test_unpublished_reason_escapes_through_other(self) -> None:
        """R4 — an ``incomplete_details.reason`` outside the published set escapes."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["incomplete_details"] = {"reason": "content_filter"}

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == "content_filter"

    @pytest.mark.parametrize("status", ["in_progress", "queued"])
    def test_mid_stream_status_raises_unreadable(self, status: str) -> None:
        """R2 — a mid-stream snapshot is not a captured reply; raise, don't project."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["status"] = status

        with pytest.raises(c.UnreadableBodyError, match="status"):
            rsp.ResponsesReplyProjection().read_reply(_reply(body))

    def test_pairing_rule_is_enforced_against_a_stale_raw(self) -> None:
        """R4 — a canonical stop reason beside ``stop_reason_raw`` is rejected.

        The rule lives in :meth:`~harness.contract.Reply.__post_init__` (T-W2);
        this test pins the *reader's* mapping to it.
        """
        with pytest.raises(ValueError, match="stop_reason_raw"):
            c.Reply(parts=(), stop_reason="end_turn", stop_reason_raw="completed")


# --------------------------------------------------------------------------
# R3 — the falsification pair
# --------------------------------------------------------------------------


class TestFalsification:
    """Plan §1.4: the harness must be shown to fail, with its control."""

    def test_injected_unknown_key_residualises_and_fails_totality(self) -> None:
        """R3 — one unrecognised top-level key produces a non-empty residual."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["x-kitty-trace"] = "injected"

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"x-kitty-trace": "injected"}
        with pytest.raises(c.ResidualFieldsError, match="x-kitty-trace"):
            c.verify_total(projected)

    def test_control_without_the_injected_key_is_clean(self) -> None:
        """R3 — the paired control: the same body without the defect is clean."""
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)

    def test_unknown_output_item_key_residualises(self) -> None:
        """R3 — an unmodelled key inside an output item fails closed at depth."""
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["output"][0]["x_vendor_marker"] = 1

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert "output[0].x_vendor_marker" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="x_vendor_marker"):
            c.verify_total(projected)

    def test_unknown_output_item_key_control_is_clean(self) -> None:
        """R3 — control for the depth case: the published item is clean."""
        projected = rsp.ResponsesReplyProjection().read_reply(_reply(PUBLISHED_FULL_RESPONSE))

        c.verify_total(projected)

    @pytest.mark.parametrize(
        ("block_kind", "text_field", "bad_value"),
        [("output_text", "text", 42), ("refusal", "refusal", 42)],
    )
    def test_wrongly_typed_message_content_residualises(
        self, block_kind: str, text_field: str, bad_value: Any
    ) -> None:
        """R3 / W5 — a wrongly-typed ``text`` / ``refusal`` residualises at
        its own path (§7.4.1: the rule is general).
        """
        message = {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": block_kind, text_field: bad_value, "annotations": []}],
        }
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["output"] = [message]

        projected = rsp.ResponsesReplyProjection().read_reply(_reply(body))

        assert f"output[0].content[0].{text_field}" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match=text_field):
            c.verify_total(projected)


class TestNameRequired:
    """A ``function_call`` output item whose ``name`` is missing, empty, or not a string raises.

    ``contract.decode_arguments`` is explicit: ``""``
    for a name is **not** lossless — a call nobody can name cannot be paired
    with its result or addressed by a register row. KBR-279 landed the rule
    for the Ollama readers, KBR-281 for Chat Completions / Gemini / Anthropic;
    KBR-292 extends it here (§7.4.2 rule 7 row 2, seven strict readers).
    """

    @staticmethod
    def _body_with_function_call(name: Any) -> dict[str, Any]:
        """Return the published response whose one output item carries ``name``.

        The item is the published ``function_call`` example verbatim;
        ``name`` is spliced in verbatim, so ``None`` means the key is absent
        rather than an explicit ``null`` — the absent form is the one the
        published schema calls required and no published example shows.
        """
        item = dict(PUBLISHED_FUNCTION_CALL_ITEM)
        if name is None:
            del item["name"]
        else:
            item["name"] = name
        body = json.loads(json.dumps(PUBLISHED_FULL_RESPONSE))
        body["output"] = [item]
        return body

    def test_missing_function_call_name_raises(self) -> None:
        """A ``function_call`` output item with no ``name`` raises."""
        with pytest.raises(c.UnreadableBodyError, match=r"output\[0\].name"):
            rsp.ResponsesReplyProjection().read_reply(_reply(self._body_with_function_call(None)))

    def test_empty_function_call_name_raises(self) -> None:
        """A ``function_call`` output item with an empty ``name`` raises (KBR-292)."""
        with pytest.raises(c.UnreadableBodyError, match=r"output\[0\].name"):
            rsp.ResponsesReplyProjection().read_reply(_reply(self._body_with_function_call("")))

    def test_non_string_function_call_name_raises(self) -> None:
        """A ``function_call`` output item with a non-string ``name`` raises."""
        with pytest.raises(c.UnreadableBodyError, match=r"output\[0\].name"):
            rsp.ResponsesReplyProjection().read_reply(_reply(self._body_with_function_call(42)))

    def test_named_function_call_control_is_clean(self) -> None:
        """Control — the published named ``function_call`` projects cleanly."""
        projected = rsp.ResponsesReplyProjection().read_reply(
            _reply(self._body_with_function_call("get_weather"))
        )

        c.verify_total(projected)
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_weather"
        assert part.arguments == {"location": "Paris"}
        assert part.id == "call_234kdga09jf"
