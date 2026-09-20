"""L1 tests for the Ollama ``/api/chat`` reply reader (response direction).

`.system_design/TEST_SUITE.md` §3.3.1 (response-direction bullet), §7.4, §7.4.1 ·
plan task **T-A7 follow-up** ([KBR-267](https://shelpuk.atlassian.net/browse/KBR-267)).

**Every body here comes from Ollama's published schema** — the ``/api/chat``
response examples at
``https://github.com/ollama/ollama/blob/main/docs/api.md``, retrieved
2026-09-17 — and each carries a comment naming which example. §3.3.1's rule:
validated against the published schema, **never** against kitty's output.
Bodies this module *constructs* (the streaming-chunk falsification, the
``done_reason: "unload"`` pair, the thinking and images shapes, the
name-required falsifications) are marked as such.

**Every falsification case is paired with its control**, per plan §1.4's harness
rule applied to a projection.

**Object-form tool-call arguments.** Ollama publishes
``tool_calls[*].function.arguments`` as a JSON object (``{"city": "Tokyo"}``
in the docs/api.md with-tools example, lines 749-758) — same wire shape as
the request reader. The reply reader mirrors the request side; it is not a
caller of ``contract.decode_arguments`` (see ``reader_ollama.py``'s module
docstring and ``contract.py:942-945``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from harness import contract as c
from harness import reader_ollama as r

# --------------------------------------------------------------------------
# Published fixtures
# --------------------------------------------------------------------------

# The "Chat request (No streaming)" example at docs/api.md lines 686-702,
# retrieved 2026-09-17. Reproduced verbatim: six provider-reported
# usage/timing keys plus model, created_at, message, done. No `done_reason`
# field — exercises the absent-done_reason branch.
PUBLISHED_NO_STREAMING: dict[str, Any] = {
    "model": "llama3.2",
    "created_at": "2023-12-12T14:13:43.416799Z",
    "message": {
        "role": "assistant",
        "content": "Hello! How are you today?",
    },
    "done": True,
    "total_duration": 5191566416,
    "load_duration": 2154458,
    "prompt_eval_count": 26,
    "prompt_eval_duration": 383809000,
    "eval_count": 298,
    "eval_duration": 4799921000,
}

# The "Chat request (No streaming, with tools)" example at docs/api.md
# lines 749-758, retrieved 2026-09-17. Demonstrates the object-form
# arguments shape (`{"city": "Tokyo"}`), no id on tool_calls, the
# empty-content + tool_calls convergence rule's input, and the six
# provider-reported usage/timing keys.
PUBLISHED_TOOL_CALL_REPLY: dict[str, Any] = {
    "model": "llama3.2",
    "created_at": "2025-07-07T20:32:53.844124Z",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {
                        "city": "Tokyo",
                    },
                },
            }
        ],
    },
    "done_reason": "stop",
    "done": True,
    "total_duration": 3244883583,
    "load_duration": 2969184542,
    "prompt_eval_count": 169,
    "prompt_eval_duration": 141656333,
    "eval_count": 18,
    "eval_duration": 133293625,
}

# The "Load a model" chat example at docs/api.md lines 1119-1133, retrieved
# 2026-09-17. Demonstrates ``done_reason: "load"`` → ``other`` + raw, the
# no-usage/timing-fields case (``usage`` becomes ``{}``), and an empty-content
# message.
PUBLISHED_LOAD_REPLY: dict[str, Any] = {
    "model": "llama3.2",
    "created_at": "2024-09-12T21:17:29.110811Z",
    "message": {
        "role": "assistant",
        "content": "",
    },
    "done_reason": "load",
    "done": True,
}

# The "Unload a model" chat example at docs/api.md lines 1167-1170, retrieved
# 2026-09-17. Demonstrates ``done_reason: "unload"`` → ``other`` + raw.
PUBLISHED_UNLOAD_REPLY: dict[str, Any] = {
    "model": "llama3.2",
    "created_at": "2024-09-12T21:33:17.547535Z",
    "message": {
        "role": "assistant",
        "content": "",
    },
    "done_reason": "unload",
    "done": True,
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
    """The schema's own no-streaming example, end to end."""

    def test_projects_clean_with_verify_total(self) -> None:
        """R1 — the published example round-trips with an empty residual."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        c.verify_total(projected)
        assert projected.source == PUBLISHED_NO_STREAMING

    def test_text_part_carries_the_published_content(self) -> None:
        """R2 — the message content projects as one ``Text`` part."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.Text)
        assert part.text == "Hello! How are you today?"

    def test_done_reason_absent_projects_as_none(self) -> None:
        """R4 — absent ``done_reason`` projects as ``None`` / ``None``."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        assert projected.stop_reason is None
        assert projected.stop_reason_raw is None

    def test_usage_is_carried_whole(self) -> None:
        """§3.3.1 — usage is carried but excluded from the T-D10 diff."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        assert projected.usage == {
            "total_duration": 5191566416,
            "load_duration": 2154458,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 383809000,
            "eval_count": 298,
            "eval_duration": 4799921000,
        }

    def test_empty_usage_when_no_timing_fields_present(self) -> None:
        """R1 / §3.3.1 — usage is ``{}`` when the final object carries none.

        The load example at docs/api.md lines 1119-1133 carries no
        usage/timing fields; the projection's ``usage`` is therefore empty.
        """
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_LOAD_REPLY))

        assert projected.usage == {}

    def test_conforms_to_the_reply_projection_protocol(self) -> None:
        """R5 — the reader satisfies ``ReplyProjection`` under ``isinstance``."""
        reader = r.OllamaChatReplyProjection()

        assert isinstance(reader, c.ReplyProjection)
        assert reader.wire_format is c.WireFormat.OLLAMA_CHAT


# --------------------------------------------------------------------------
# R2 — the format's content-shape vocabulary
# --------------------------------------------------------------------------


class TestContentShapes:
    """One case per content shape the schema documents on the reply side."""

    @staticmethod
    def _body_with_message(message: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        """Return the no-streaming body with the message swapped.

        Args:
            message: The replacement ``message`` object.
            overrides: Top-level keys to overwrite on the body
                (``done_reason="load"``, ``total_duration=...``).

        Returns:
            The body with every other key the published example carries.
        """
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["message"] = message
        for key, value in overrides.items():
            body[key] = value
        return body

    def test_tool_call_projects_with_object_arguments(self) -> None:
        """R2 — object-form arguments decode through the request reader's helper.

        The published ``tool_calls`` example has ``arguments`` as a JSON object
        (``{"city": "Tokyo"}``); the projection keeps that shape verbatim, not
        decoded through ``contract.decode_arguments``.
        """
        body = self._body_with_message(PUBLISHED_TOOL_CALL_REPLY["message"], done_reason="stop")

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert projected.stop_reason == "end_turn"
        # Empty content + tool_calls → no Text part (the convergence rule,
        # mirrors the request reader and CC's `content: null → ()`).
        assert len(projected.parts) == 1
        part = projected.parts[0]
        assert isinstance(part, c.ToolUse)
        assert part.name == "get_weather"
        assert part.id is None
        assert part.arguments == {"city": "Tokyo"}

    def test_thinking_projects_as_thinking_part(self) -> None:
        """R2 — the ``message.thinking`` field projects as ``Thinking``.

        Constructed body: the published reply-side ``thinking`` field is never
        populated in ``docs/api.md`` (the field is documented at lines 499-506
        for the message shape but no published response carries a populated
        value). The request reader's helper handles the same shape identically,
        so this is a transitive coverage of the request reader's ``thinking``
        branch on the reply side.
        """
        message = {
            "role": "assistant",
            "content": "42.",
            "thinking": "The user asked a math question. Answer: 42.",
        }
        body = self._body_with_message(message)

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        # Part ordering: Text → Thinking (mirrors request reader).
        assert len(projected.parts) == 2
        assert isinstance(projected.parts[0], c.Text)
        assert projected.parts[0].text == "42."
        assert isinstance(projected.parts[1], c.Thinking)
        assert projected.parts[1].text == "The user asked a math question. Answer: 42."

    def test_images_projects_as_image_parts(self) -> None:
        """R2 — a message's ``images`` list projects as ``Image`` parts.

        Constructed body: the published reply-side ``images`` value is always
        ``null`` (the multimodal form is documented at lines 503-505 but no
        ``/api/chat`` **response** example carries a populated list). Built
        from the published message-field shape.
        """
        message = {
            "role": "assistant",
            "content": "An image.",
            "images": ["aGVsbG8="],  # base64("hello")
        }
        body = self._body_with_message(message)

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        # Text → Image ordering (mirrors request reader; no ToolUse, no Thinking).
        assert len(projected.parts) == 2
        assert isinstance(projected.parts[0], c.Text)
        assert projected.parts[0].text == "An image."
        assert isinstance(projected.parts[1], c.Image)
        assert projected.parts[1].digest


# --------------------------------------------------------------------------
# R4 — the done_reason mapping, canonical and escaped
# --------------------------------------------------------------------------


class TestDoneReasonMapping:
    """Canonical values map straight; the rest escape through ``other``."""

    @pytest.mark.parametrize(
        ("wire_value", "expected"),
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
        ],
    )
    def test_published_values_map_straight(self, wire_value: str, expected: str) -> None:
        """R4 — every canonical ``done_reason`` maps onto the canonical set."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["done_reason"] = wire_value

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == expected
        assert projected.stop_reason_raw is None

    @pytest.mark.parametrize("wire_value", ["load", "unload"])
    def test_load_and_unload_escape_through_other(self, wire_value: str) -> None:
        """R4 — ``load`` / ``unload`` (the contract docstring's motivating case)
        escape through ``other`` with the wire string kept."""
        fixture = PUBLISHED_LOAD_REPLY if wire_value == "load" else PUBLISHED_UNLOAD_REPLY

        projected = r.OllamaChatReplyProjection().read_reply(_reply(fixture))

        c.verify_total(projected)
        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == wire_value

    def test_unknown_value_escapes_through_other(self) -> None:
        """R4 — a value outside the published enum escapes with the wire string."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["done_reason"] = "brand_new_reason"

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason == "other"
        assert projected.stop_reason_raw == "brand_new_reason"

    def test_non_string_done_reason_residualises(self) -> None:
        """R4 / AC4 — a non-string ``done_reason`` residualises at its bare name.

        Pins the non-string row of the ``done_reason`` table: canonical
        ``None``/``None``, the value lands in the residual at ``done_reason``,
        and ``verify_total`` fails the run.
        """
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["done_reason"] = 42

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert projected.stop_reason is None
        assert projected.stop_reason_raw is None
        assert projected.residual == {"done_reason": 42}
        with pytest.raises(c.ResidualFieldsError, match="done_reason"):
            c.verify_total(projected)

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
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["x-kitty-trace"] = "injected"

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"x-kitty-trace": "injected"}
        with pytest.raises(c.ResidualFieldsError, match="x-kitty-trace"):
            c.verify_total(projected)

    def test_control_without_the_injected_key_is_clean(self) -> None:
        """R3 — the paired control: the same body without the defect is clean."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        c.verify_total(projected)

    def test_unknown_message_key_residualises(self) -> None:
        """R3 — an unmodelled key inside the message fails closed at depth."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["message"]["x_vendor_marker"] = 1

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert projected.residual == {"message.x_vendor_marker": 1}
        with pytest.raises(c.ResidualFieldsError, match="x_vendor_marker"):
            c.verify_total(projected)

    def test_unknown_message_key_control_is_clean(self) -> None:
        """R3 — control for the depth case: the published message is clean."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        c.verify_total(projected)

    def test_wrongly_typed_message_field_residualises(self) -> None:
        """R3 / §7.4.1 — a wrongly-typed optional leaf residualises at its own
        path."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["message"]["thinking"] = {"wrong": "type"}

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert "message.thinking" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match=r"message\.thinking"):
            c.verify_total(projected)

    def test_message_absent_residualises(self) -> None:
        """R3 / §7.4.1 — an absent ``message`` residualises at ``message``.

        Published bodies always carry ``message``; the absent case is the
        ``§7.4.1`` fail-closed rule applied to a top-level container field.
        """
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        del body["message"]

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        assert "message" in projected.residual
        with pytest.raises(c.ResidualFieldsError, match="message"):
            c.verify_total(projected)


# --------------------------------------------------------------------------
# R6 — streaming boundary
# --------------------------------------------------------------------------


class TestStreamingBoundary:
    """The reader reads one complete response object.

    Mid-stream chunks (carrying ``done: false``) and bodies without ``done``
    raise :class:`~harness.contract.UnreadableBodyError`. This closes the
    silent failure unique to Ollama: every published ``/api/chat`` chunk
    carries every key the accounting table handles, so without this guard
    a mid-stream chunk would project as a short legitimate reply and pass
    ``verify_total``.
    """

    def test_chunk_with_done_false_raises(self) -> None:
        """R6 — a mid-stream chunk (``done: false``) is not a complete reply.

        Constructed from the streaming-chunk example at docs/api.md lines
        552-562 (every key that example carries). The contract's reassembly
        boundary (``contract.py:1493-1496``) is the caller's responsibility;
        a chunk slipping through reassembly is the reader's signal that
        reassembly did not happen.
        """
        chunk: dict[str, Any] = {
            "model": "llama3.2",
            "created_at": "2023-08-04T08:52:19.385406455-07:00",
            "message": {
                "role": "assistant",
                "content": "The",
                "images": None,
            },
            "done": False,
        }

        with pytest.raises(c.UnreadableBodyError, match="done"):
            r.OllamaChatReplyProjection().read_reply(_reply(chunk))

    def test_chunk_control_published_no_streaming_is_accepted(self) -> None:
        """R6 — paired control: the published no-streaming example
        (``done: true``) is accepted cleanly."""
        projected = r.OllamaChatReplyProjection().read_reply(_reply(PUBLISHED_NO_STREAMING))

        c.verify_total(projected)

    def test_done_absent_raises(self) -> None:
        """R6 — absent ``done`` raises (every published response carries it)."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        del body["done"]

        with pytest.raises(c.UnreadableBodyError, match="done"):
            r.OllamaChatReplyProjection().read_reply(_reply(body))


# --------------------------------------------------------------------------
# R7 — name-required check on tool_calls
# --------------------------------------------------------------------------


class TestNameRequired:
    """A tool_call with a missing or empty ``function.name`` raises.

    ``contract.decode_arguments`` (``contract.py:935-941``) is explicit: ``""``
    for a name is **not** lossless — a call nobody can name cannot be paired
    with its result or addressed by a register row. Chat Completions raises
    :class:`~harness.contract.UnreadableBodyError` on an absent, empty, or
    non-string name (KBR-281); the reply reader mirrors that, and the request
    readers on both sides of the harness enforce the same non-empty-string
    form.
    """

    @staticmethod
    def _body_with_tool_call(function: dict[str, Any]) -> dict[str, Any]:
        """Return the published no-streaming body with one tool_call swapped in."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["message"] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": function}],
        }
        body["done_reason"] = "stop"
        return body

    def test_missing_function_name_raises(self) -> None:
        """R7 — a tool_call with no ``function.name`` raises."""
        body = self._body_with_tool_call({"arguments": {"city": "Tokyo"}})

        with pytest.raises(c.UnreadableBodyError, match="function.name"):
            r.OllamaChatReplyProjection().read_reply(_reply(body))

    def test_empty_function_name_raises(self) -> None:
        """R7 — a tool_call with an empty ``function.name`` raises."""
        body = self._body_with_tool_call({"name": "", "arguments": {"city": "Tokyo"}})

        with pytest.raises(c.UnreadableBodyError, match="function.name"):
            r.OllamaChatReplyProjection().read_reply(_reply(body))

    def test_non_string_function_name_raises(self) -> None:
        """R7 — a tool_call with a non-string ``function.name`` raises."""
        body = self._body_with_tool_call({"name": 42, "arguments": {}})

        with pytest.raises(c.UnreadableBodyError, match="function.name"):
            r.OllamaChatReplyProjection().read_reply(_reply(body))

    def test_named_function_control_is_clean(self) -> None:
        """R7 — paired control: the published with-tools example (with a name)
        is accepted cleanly."""
        body = json.loads(json.dumps(PUBLISHED_NO_STREAMING))
        body["message"] = PUBLISHED_TOOL_CALL_REPLY["message"]
        body["done_reason"] = "stop"

        projected = r.OllamaChatReplyProjection().read_reply(_reply(body))

        c.verify_total(projected)
        assert any(isinstance(p, c.ToolUse) and p.name == "get_weather" for p in projected.parts)
