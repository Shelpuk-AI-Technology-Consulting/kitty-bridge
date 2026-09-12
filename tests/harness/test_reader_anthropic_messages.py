"""L1 tests for the Anthropic Messages reader.

`.system_design/TEST_SUITE.md` §3.3.1, §7.4.1 · plan task **T-A1** (KBR-33).

**Every body here comes from Anthropic's published documentation**, and each
carries a comment naming which example and when it was retrieved.  That is the
rule §3.3.1 puts on all seven readers — "validated against the published schema,
**never** against kitty's output" — and it is only checkable later if the
provenance travels with the fixture.  Bodies this module *constructs* (a
malformed one, an injected key) are marked as such.

**Every falsification case is paired with a control.**  Without the control, a
reader that residualised unconditionally, or raised unconditionally, would
satisfy every positive assertion in this module.  Plan §1.4's harness rule,
applied to a projection.

**The module asserts on** ``verify_total`` **, not only on shapes.**  A body
that projects the right parts while quietly dropping a key is exactly the defect
:attr:`~harness.contract.Request.consumed` exists to catch, and a test that
only compared parts would pass over it.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from harness import contract as c
from harness import reader_anthropic_messages as am

# --------------------------------------------------------------------------
# Published fixtures
# --------------------------------------------------------------------------

# The reference's own complete request example, from the "Create a Message"
# cURL sample. Retrieved 2026-09-11 from docs.claude.com/en/api/messages.md.
PUBLISHED_FULL_REQUEST: dict[str, Any] = {
    "max_tokens": 1024,
    "messages": [{"content": "Hello, world", "role": "user"}],
    "model": "claude-opus-5",
    "stream": False,
    "system": [{"text": "Today's date is 2024-06-01.", "type": "text"}],
    "temperature": 1,
    "thinking": {"type": "adaptive"},
    "tools": [
        {
            "input_schema": {
                "type": "object",
                "properties": {"location": "bar", "unit": "bar"},
                "required": ["location"],
            },
            "name": "name",
        }
    ],
    "top_k": 5,
    "top_p": 0.7,
}

# The `messages` parameter's three published examples, same source and date.
PUBLISHED_SINGLE_USER = [{"role": "user", "content": "Hello, Claude"}]
PUBLISHED_MULTI_TURN = [
    {"role": "user", "content": "Hello there."},
    {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
    {"role": "user", "content": "Can you explain LLMs in plain English?"},
]
PUBLISHED_PREFILLED = [
    {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},
    {"role": "assistant", "content": "The best answer is ("},
]

# The `tools` parameter's published worked example and the tool_use/tool_result
# pair the same passage shows, same source and date.
PUBLISHED_TOOL = {
    "name": "get_stock_price",
    "description": "Get the current stock price for a given ticker symbol.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."}
        },
        "required": ["ticker"],
    },
}
PUBLISHED_TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "name": "get_stock_price",
    "input": {"ticker": "^GSPC"},
}
PUBLISHED_TOOL_RESULT = {
    "type": "tool_result",
    "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "content": "259.75 USD",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _captured(body: Any) -> c.CapturedRequest:
    """Wrap a body as the capture a projection is handed.

    Serialised here rather than stored as bytes so the fixtures above stay
    readable as the published examples they came from, while the reader is still
    exercised on ``bytes`` as :class:`~harness.contract.Projection` requires.

    Args:
        body: The request body, as a Python object, or raw ``bytes`` to pass
            through untouched for the malformed cases.

    Returns:
        A capture addressed at the Messages endpoint.
    """
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    return c.CapturedRequest(
        method="POST",
        scheme="https",
        host="api.anthropic.com",
        path="/v1/messages",
        query="",
        headers=(("content-type", "application/json"),),
        body=raw,
    )


def _read(body: Any) -> c.Request:
    """Project a body through the reader under test.

    Args:
        body: The request body, as a Python object or raw ``bytes``.

    Returns:
        The projection.
    """
    return am.AnthropicMessagesProjection().read_request(_captured(body))


def _minimal(**overrides: Any) -> dict[str, Any]:
    """Return the smallest valid published body, with overrides applied.

    The base is the published single-message example plus the two fields the
    API requires. Used as the **control** half of every falsification pair: the
    same body without the injected defect must project cleanly.

    Args:
        **overrides: Body keys to add or replace.

    Returns:
        A request body.
    """
    body: dict[str, Any] = {
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello, Claude"}],
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------
# R1 — module and protocol
# --------------------------------------------------------------------------


class TestProtocolConformance:
    """R1 — the reader is a `Projection`, and it is independent of kitty."""

    def test_the_reader_satisfies_the_projection_protocol(self) -> None:
        """R1.2 — T-D1 selects a reader by protocol, not by duck-typing at the call site."""
        assert isinstance(am.AnthropicMessagesProjection(), c.Projection)

    def test_the_reader_declares_the_anthropic_messages_wire_format(self) -> None:
        """R1.2 — the format-keyed lookup in the oracle is only as good as this member."""
        assert am.AnthropicMessagesProjection().wire_format is c.WireFormat.ANTHROPIC_MESSAGES

    def test_the_reader_imports_nothing_from_kitty(self) -> None:
        """R1.3 — §3.3.1's independent-oracle rule, the structural half.

        Reuses ``test_contract.py``'s regex rather than an AST walk: that pattern
        deliberately also catches ``from src.kitty ...`` and both dynamic import
        forms, which an import-node walk would miss.
        """
        from harness.test_contract import _KITTY_IMPORT

        source = Path(am.__file__).read_text(encoding="utf-8")

        assert _KITTY_IMPORT.search(source) is None


# --------------------------------------------------------------------------
# R2 — envelope
# --------------------------------------------------------------------------


class TestEnvelope:
    """R2 — routing and control fields."""

    def test_model_and_stream_reach_their_named_envelope_fields(self) -> None:
        """R2.1 — M1 replaces the model; it needs an address of its own."""
        projected = _read(PUBLISHED_FULL_REQUEST)

        assert projected.envelope.model == "claude-opus-5"
        assert projected.envelope.stream is False

    def test_store_is_absent_because_the_messages_format_has_no_such_field(self) -> None:
        """R2.2 — P17 injects `store` on the Responses path only."""
        assert _read(PUBLISHED_FULL_REQUEST).envelope.store is None

    @pytest.mark.parametrize(
        "key, value",
        [
            ("thinking", {"type": "adaptive"}),
            ("metadata", {"user_id": "13803d75"}),
            ("service_tier", "standard_only"),
            ("container", {"id": "container_011CpZohnwH4vuy7gazohgSP"}),
            ("cache_control", {"type": "ephemeral"}),
            ("inference_geo", "us"),
            ("output_config", {"effort": "medium"}),
            ("context_management", {"edits": [{"type": "clear_tool_uses_20250919"}]}),
            ("mcp_servers", [{"type": "url", "url": "https://example.com/sse", "name": "srv"}]),
        ],
    )
    def test_a_published_control_field_reaches_extra_under_its_wire_key(
        self, key: str, value: Any
    ) -> None:
        """R2.3 — keyed by the wire key so a register row can name `envelope.extra[thinking]`."""
        projected = _read(_minimal(**{key: value}))

        assert projected.envelope.extra[key] == value
        c.verify_total(projected)

    def test_all_nine_control_fields_survive_one_body_together(self) -> None:
        """R2.3 — parametrised one at a time cannot catch two keys overwriting each other.

        Also §3.3.1a's rule that `extra` is diffed one wire key at a time: no
        key may carry a dot, or the register's `envelope.extra[<key>]` anchor
        addresses nothing.
        """
        fields = {
            "thinking": {"type": "adaptive"},
            "metadata": {"user_id": "13803d75"},
            "service_tier": "standard_only",
            "container": {"id": "container_011CpZohnwH4vuy7gazohgSP"},
            "cache_control": {"type": "ephemeral"},
            "inference_geo": "us",
            "output_config": {"effort": "medium"},
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "mcp_servers": [{"type": "url", "url": "https://example.com/sse", "name": "srv"}],
        }
        projected = _read(_minimal(**fields))

        assert dict(projected.envelope.extra) == fields
        assert all("." not in key for key in projected.envelope.extra)
        c.verify_total(projected)

    def test_effort_reaches_extra_although_the_reference_does_not_list_it(self) -> None:
        """R2.6 — Claude Code sends it, so residualising it would fail every real request.

        §7.4.1's evidence rule: a key the client demonstrably sends is a
        recognised control field. ``MessagesTranslator`` reads ``effort`` off
        the inbound body, which is the evidence.
        """
        projected = _read(_minimal(effort="high"))

        assert projected.envelope.extra["effort"] == "high"
        c.verify_total(projected)

    def test_the_register_still_declares_the_path_effort_now_populates(self) -> None:
        """R2.6 — P5d's declared path and this reader's output must stay the same string.

        Asserts the register still names the path, not that a delta was
        produced: on an Anthropic-to-Anthropic route the translated path copies
        `_effort` back verbatim, so no delta is available to assert on.
        """
        from harness.register import REGISTER

        p5d = next(row for row in REGISTER if row.id == "P5d")

        assert any(c.path_matches(path, c.extra_path("effort")) for path in p5d.paths)

    @pytest.mark.parametrize(
        "choice, expected",
        [
            ({"type": "auto"}, "auto"),
            ({"type": "any"}, "any"),
            ({"type": "none"}, "none"),
            ({"type": "tool", "name": "get_weather"}, "tool:get_weather"),
        ],
    )
    def test_each_published_tool_choice_shape_normalises_to_its_canonical_value(
        self, choice: dict[str, Any], expected: str
    ) -> None:
        """R2.4 — four wire spellings, one concept, one canonical value (§3.3.1b)."""
        projected = _read(_minimal(tool_choice=choice))

        assert projected.envelope.extra["tool_choice"] == expected
        c.verify_total(projected)

    def test_disable_parallel_tool_use_is_not_part_of_the_canonical_value(self) -> None:
        """R2.4 — it is a separate knob; folding it in would make the value unmatched."""
        projected = _read(_minimal(tool_choice={"type": "auto", "disable_parallel_tool_use": True}))

        assert projected.envelope.extra["tool_choice"] == "auto"
        assert projected.residual == {"tool_choice.disable_parallel_tool_use": True}

    def test_a_stale_name_beside_a_non_tool_choice_residualises(self) -> None:
        """R4.10 — `name` is consumed only on the branch that read it.

        A translator that flips the type to `auto` and leaves the name behind is
        exactly the mutation the oracle should report; excluding `name`
        unconditionally dropped it silently.
        """
        projected = _read(_minimal(tool_choice={"type": "auto", "name": "ghost"}))

        assert projected.residual == {"tool_choice.name": "ghost"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_name_on_a_tool_choice_that_uses_it_does_not_residualise(self) -> None:
        """R4.10 — the paired control: on the `tool` branch the name is accounted for."""
        projected = _read(_minimal(tool_choice={"type": "tool", "name": "get_weather"}))

        assert projected.residual == {}
        c.verify_total(projected)

    @pytest.mark.parametrize("choice", [{"type": "magic"}, {"type": "tool"}])
    def test_an_unusable_tool_choice_is_an_unreadable_body_and_never_a_value_error(
        self, choice: dict[str, Any]
    ) -> None:
        """R2.5 — classify before constructing, or the diagnosis is wrong.

        ``Envelope`` accepts any string starting ``tool:``, so a missing name
        would yield the canonical-looking ``"tool:None"`` that no register row
        can interpret; an unrecognised type would escape as a bare
        ``ValueError``, which the contract defines as "the reader mis-routed a
        field" — a different defect from "this body is unreadable".
        """
        with pytest.raises(c.UnreadableBodyError):
            _read(_minimal(tool_choice=choice))

    def test_a_tool_choice_without_a_name_never_becomes_the_string_tool_none(self) -> None:
        """R2.5 — the assertion that actually bites for that half of the case."""
        try:
            projected = _read(_minimal(tool_choice={"type": "tool"}))
        except c.UnreadableBodyError:
            return

        pytest.fail(f"expected UnreadableBodyError, got {projected.envelope.extra!r}")


# --------------------------------------------------------------------------
# R3 — system
# --------------------------------------------------------------------------


class TestSystem:
    """R3 — system instructions lift into `Conversation.system` (§3.3.1b)."""

    def test_both_published_system_forms_project_identically(self) -> None:
        """R3.1 — a string is shorthand for one text block; the projection must agree."""
        as_string = _read(_minimal(system="Today's date is 2024-06-01."))
        as_blocks = _read(_minimal(system=[{"type": "text", "text": "Today's date is 2024-06-01."}]))

        assert as_string.conversation.system == (c.Text("Today's date is 2024-06-01."),)
        assert as_string.conversation.system == as_blocks.conversation.system

    def test_system_blocks_keep_their_order(self) -> None:
        """R3.1 — `conversation.system[<i>]` is positional, so order is part of the claim."""
        projected = _read(
            _minimal(system=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}])
        )

        assert projected.conversation.system == (c.Text("first"), c.Text("second"))

    def test_a_system_block_with_cache_control_residualises_and_fails_the_run(self) -> None:
        """R3.2 — the grammar has no slot for it, so it fails closed (KBR-167)."""
        projected = _read(
            _minimal(system=[{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}])
        )

        assert projected.residual == {"system[0].cache_control": {"type": "ephemeral"}}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_the_control_for_that_case_passes(self) -> None:
        """R3.2 — the paired control: without the injected key the same body is clean."""
        projected = _read(_minimal(system=[{"type": "text", "text": "hi"}]))

        assert projected.residual == {}
        c.verify_total(projected)

    def test_a_non_text_system_block_is_an_unreadable_body(self) -> None:
        """R3.3 — `Conversation.system` admits only `Text`, so there is nowhere else to put it."""
        with pytest.raises(c.UnreadableBodyError):
            _read(_minimal(system=[{"type": "image", "source": {"type": "url", "url": "u"}}]))


# --------------------------------------------------------------------------
# R4 — content blocks
# --------------------------------------------------------------------------

# A 1x1 PNG, constructed here rather than published: the reference documents the
# base64 source shape but ships no image payload.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0000003010100f4f4a0bb0000000049454e44ae426082"
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


class TestContentBlocks:
    """R4.1–R4.6 — one part per block, and nothing unmodelled vanishes."""

    def test_the_published_single_message_example_projects_one_user_turn(self) -> None:
        """R4.1, R4.2 — a string content is the format's shorthand for one text block."""
        projected = _read(_minimal(messages=PUBLISHED_SINGLE_USER))

        assert projected.conversation.turns == (c.Turn(role="user", parts=(c.Text("Hello, Claude"),)),)

    def test_the_published_multi_turn_example_keeps_its_three_turns_and_order(self) -> None:
        """R4.1 — turns are positional, so a reordering is a delta."""
        projected = _read(_minimal(messages=PUBLISHED_MULTI_TURN))

        assert [turn.role for turn in projected.conversation.turns] == ["user", "assistant", "user"]
        assert projected.conversation.turns[2].parts == (c.Text("Can you explain LLMs in plain English?"),)

    def test_the_published_prefilled_assistant_example_projects_as_an_assistant_turn(self) -> None:
        """R4.1 — a partially-filled response is still an assistant turn, not a special case."""
        projected = _read(_minimal(messages=PUBLISHED_PREFILLED))

        assert projected.conversation.turns[1] == c.Turn(
            role="assistant", parts=(c.Text("The best answer is ("),)
        )

    def test_a_base64_image_projects_the_digest_of_its_decoded_bytes(self) -> None:
        """R4.4 — pinned so six readers agree; Anthropic sends base64, Converse raw bytes."""
        block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        image = projected.conversation.turns[0].parts[0]

        assert image == c.Image(digest=hashlib.sha256(PNG_BYTES).hexdigest(), media_type="image/png")
        c.verify_total(projected)

    def test_changing_only_the_media_type_changes_the_media_type_and_not_the_digest(self) -> None:
        """R4.4 — the media type is excluded from the digest so it is its own delta."""
        as_png = _read(
            _minimal(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
                            }
                        ],
                    }
                ]
            )
        )
        as_webp = _read(
            _minimal(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/webp", "data": PNG_B64},
                            }
                        ],
                    }
                ]
            )
        )

        first = as_png.conversation.turns[0].parts[0]
        second = as_webp.conversation.turns[0].parts[0]

        assert first.digest == second.digest
        assert first.media_type != second.media_type

    @pytest.mark.parametrize(
        "source, expected_ref",
        [
            ({"type": "url", "url": "https://example.com/cat.png"}, "https://example.com/cat.png"),
            ({"type": "file", "file_id": "file_011CpZ"}, "file_011CpZ"),
        ],
    )
    def test_an_image_carrying_a_reference_instead_of_bytes_has_no_digest(
        self, source: dict[str, Any], expected_ref: str
    ) -> None:
        """R4.4 — a URL or file source carries no bytes, so `digest` is absent and `ref` holds it."""
        block = {"type": "image", "source": source}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.conversation.turns[0].parts[0] == c.Image(ref=expected_ref)
        c.verify_total(projected)

    def test_a_thinking_block_carries_its_signature_rather_than_residualising_it(self) -> None:
        """R4.3 — M8's carrier repair manipulates `signature`; a residual would hide it."""
        block = {"type": "thinking", "thinking": "step one", "signature": "sig-abc"}
        projected = _read(_minimal(messages=[{"role": "assistant", "content": [block]}]))

        assert projected.conversation.turns[0].parts[0] == c.Thinking(text="step one", signature="sig-abc")
        assert projected.residual == {}
        c.verify_total(projected)

    def test_an_empty_thinking_block_is_a_part_with_an_empty_string(self) -> None:
        """R4.3 — P5e injects an empty thinking block; §3.3.2 assertion 2 needs it observable."""
        block = {"type": "thinking", "thinking": "", "signature": "s"}
        projected = _read(_minimal(messages=[{"role": "assistant", "content": [block]}]))

        assert projected.conversation.turns[0].parts == (c.Thinking(text="", signature="s"),)

    def test_the_published_tool_use_example_projects_its_name_arguments_and_id(self) -> None:
        """R4.5 — Messages sends an object; Chat Completions a JSON string. Normalised here."""
        projected = _read(_minimal(messages=[{"role": "assistant", "content": [PUBLISHED_TOOL_USE]}]))

        assert projected.conversation.turns[0].parts[0] == c.ToolUse(
            name="get_stock_price", arguments={"ticker": "^GSPC"}, id="toolu_01D7FLrfh4GYq7yT1ULFeyMV"
        )
        c.verify_total(projected)

    def test_the_published_tool_result_example_projects_its_string_content_as_text(self) -> None:
        """R4.5 — and `is_error` defaults to False rather than to absent."""
        projected = _read(_minimal(messages=[{"role": "user", "content": [PUBLISHED_TOOL_RESULT]}]))

        assert projected.conversation.turns[0].parts[0] == c.ToolResult(
            content=(c.Text("259.75 USD"),),
            tool_use_id="toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            is_error=False,
        )
        c.verify_total(projected)

    def test_a_failed_tool_result_carries_its_error_flag(self) -> None:
        """R4.5 — only the default was asserted, so a reader hard-coding False would pass."""
        block = {"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": True}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.conversation.turns[0].parts[0].is_error is True

    def test_a_wrongly_typed_error_flag_residualises_rather_than_being_coerced(self) -> None:
        """R7.3 — `bool("false")` is `True`, so the coercion invents the opposite of the body.

        `False` is the absent value the grammar already carries, so there is one
        to fall back to — which is what makes this a wrongly-typed leaf rather
        than a structural failure.
        """
        block = {"type": "tool_result", "tool_use_id": "t", "content": "x", "is_error": "false"}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.conversation.turns[0].parts[0].is_error is False
        assert projected.residual == {"messages[0].content[0].is_error": "false"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_cache_control_on_a_tool_use_block_residualises(self) -> None:
        """R4.10 — Claude Code sets a breakpoint on tool blocks too, not only on text (KBR-167)."""
        block = dict(PUBLISHED_TOOL_USE, cache_control={"type": "ephemeral"})
        projected = _read(_minimal(messages=[{"role": "assistant", "content": [block]}]))

        assert projected.residual == {"messages[0].content[0].cache_control": {"type": "ephemeral"}}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_json_shaped_tool_result_string_is_text_and_never_json(self) -> None:
        """R4.5 — §7.4.1: `Json` is for a format carrying a structured value natively.

        Parsing it here would show an unclaimed delta against the Chat
        Completions reader on every JSON-shaped tool result.
        """
        block = {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"price": 259.75}'}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        result = projected.conversation.turns[0].parts[0]

        assert result.content == (c.Text('{"price": 259.75}'),)

    @pytest.mark.parametrize(
        "block, kind",
        [
            ({"type": "document", "source": {"type": "text", "data": "d", "media_type": "text/plain"}}, "document"),
            ({"type": "search_result", "source": "s", "title": "t", "content": []}, "search_result"),
            ({"type": "redacted_thinking", "data": "encrypted"}, "redacted_thinking"),
            ({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}, "server_tool_use"),
        ],
    )
    def test_an_unmodelled_block_projects_as_opaque_and_accounts_for_its_payload(
        self, block: dict[str, Any], kind: str
    ) -> None:
        """R4.6 — `Opaque` keeps what the grammar cannot express detectable rather than dropped."""
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        part = projected.conversation.turns[0].parts[0]

        assert isinstance(part, c.Opaque)
        assert part.kind == kind
        assert projected.residual == {}
        c.verify_total(projected)

    def test_two_documents_differing_only_in_payload_produce_different_opaque_values(self) -> None:
        """R4.6 — a bare `Opaque("document")` would make a swapped document invisible."""
        def project(data: str) -> c.Opaque:
            block = {"type": "document", "source": {"type": "text", "data": data, "media_type": "text/plain"}}
            return _read(_minimal(messages=[{"role": "user", "content": [block]}])).conversation.turns[0].parts[0]

        assert project("alpha") != project("beta")

    def test_two_identical_documents_compare_equal_whatever_their_key_order(self) -> None:
        """R4.6 — the digest is over canonical JSON, so a reordering is not a delta."""
        first = {"type": "document", "title": "t", "context": "ctx"}
        second = {"context": "ctx", "title": "t", "type": "document"}

        def project(block: dict[str, Any]) -> c.Opaque:
            return _read(_minimal(messages=[{"role": "user", "content": [block]}])).conversation.turns[0].parts[0]

        assert project(first) == project(second)

    def test_the_opaque_digest_is_the_pinned_recipe_down_to_ensure_ascii(self) -> None:
        """R4.6 — six readers must produce the *same* digest, so pin it to a literal.

        Every other digest assertion here compares two projections against each
        other, which cannot see a recipe change that stays internally
        consistent. The non-ASCII payload is what kills an `ensure_ascii=False`
        spelling; the two keys are what kill `separators` and `sort_keys`.
        """
        block = {"type": "document", "title": "café", "context": "ctx"}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        canonical = json.dumps(
            {"title": "café", "context": "ctx"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

        assert projected.conversation.turns[0].parts[0].digest == hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def test_cache_control_on_an_opaque_block_residualises_and_leaves_the_digest_alone(self) -> None:
        """R4.6 — one field must not behave two ways: it residualises on modelled blocks too."""
        bare = {"type": "document", "title": "t"}
        marked = {"type": "document", "title": "t", "cache_control": {"type": "ephemeral"}}

        plain = _read(_minimal(messages=[{"role": "user", "content": [bare]}]))
        cached = _read(_minimal(messages=[{"role": "user", "content": [marked]}]))

        assert cached.residual == {"messages[0].content[0].cache_control": {"type": "ephemeral"}}
        assert (
            cached.conversation.turns[0].parts[0].digest == plain.conversation.turns[0].parts[0].digest
        )

    def test_cache_control_on_a_text_block_residualises_and_fails_the_run(self) -> None:
        """R4.10 — fail closed at depth; KBR-167 is the contract decision that would close it."""
        block = {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.residual == {"messages[0].content[0].cache_control": {"type": "ephemeral"}}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_the_control_for_that_case_passes(self) -> None:
        """R4.10 — the paired control, without which a reader that always residualised would pass."""
        projected = _read(_minimal(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]))

        assert projected.residual == {}
        c.verify_total(projected)

    def test_an_unmodelled_block_inside_a_tool_result_is_accounted_for_exactly_once(self) -> None:
        """R4.6 — §7.4.1: an `Opaque` consumes its block, so nothing beneath it residualises.

        Reading the block first and re-reading it as opaque residualised its keys
        *and* digested them, accounting for one key twice.
        """
        inner = {"type": "thinking", "thinking": "t", "signature": "s", "surprise": 1}
        block = {"type": "tool_result", "tool_use_id": "t", "content": [inner]}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.residual == {}
        c.verify_total(projected)

    def test_a_block_whose_type_is_not_a_string_is_an_unreadable_body(self) -> None:
        """R4.6 — `Opaque.kind` is declared `str`; an object there matches no canonical name."""
        with pytest.raises(c.UnreadableBodyError):
            _read(_minimal(messages=[{"role": "user", "content": [{"type": {"a": 1}, "b": 2}]}]))

    def test_an_unknown_message_key_residualises_under_the_message_path(self) -> None:
        """R4.10 — the depth above the content blocks is guarded too."""
        projected = _read(_minimal(messages=[{"role": "user", "content": "hi", "unknown": 1}]))

        assert projected.residual == {"messages[0].unknown": 1}

    def test_an_unknown_key_inside_an_image_source_residualises_under_the_source_path(self) -> None:
        """R4.10 — and the depth below them."""
        block = {"type": "image", "source": {"type": "url", "url": "u", "unknown": 1}}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.residual == {"messages[0].content[0].source.unknown": 1}


# --------------------------------------------------------------------------
# R4.7–R4.9 — turn normalisation
# --------------------------------------------------------------------------


class TestTurnNormalisation:
    """R4.7–R4.9 — the two rules that make a Messages turn comparable with a CC one."""

    def test_two_consecutive_user_messages_merge_into_one_turn(self) -> None:
        """R4.7 — the format's own stated behaviour, and §3.3.1b's shared rule."""
        projected = _read(
            _minimal(messages=[{"role": "user", "content": "one"}, {"role": "user", "content": "two"}])
        )

        assert projected.conversation.turns == (
            c.Turn(role="user", parts=(c.Text("one"), c.Text("two"))),
        )

    def test_alternating_roles_do_not_merge(self) -> None:
        """R4.7 — the control: a merge that fired unconditionally would collapse everything."""
        projected = _read(_minimal(messages=PUBLISHED_MULTI_TURN))

        assert len(projected.conversation.turns) == 3

    def test_a_result_is_never_hoisted_past_text_the_agent_sent_before_it(self) -> None:
        """R4.8 — the case that proves a re-sort after the merge is wrong.

        `tool_result -> user(text) -> tool_result` must project as
        `[ToolResult, Text, ToolResult]`. A reader that re-sorted the merged turn
        would give `[ToolResult, ToolResult, Text]`, moving history the bridge
        did not move — and because paths are index-based, that invented delta
        lands on every part of the turn and every turn after it.
        """
        projected = _read(
            _minimal(
                messages=[
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "A"}]},
                    {"role": "user", "content": "in between"},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "B"}]},
                ]
            )
        )

        parts = projected.conversation.turns[0].parts

        assert [type(part).__name__ for part in parts] == ["ToolResult", "Text", "ToolResult"]
        assert [parts[0].tool_use_id, parts[2].tool_use_id] == ["a", "b"]

    def test_the_wire_order_of_parts_inside_one_turn_survives(self) -> None:
        """R4.8 — in this format a tool result already arrives inside a user turn.

        §3.3.1b's first three clauses are satisfied by the wire order, so the
        reader moves nothing; reordering here would manufacture a delta.
        """
        content = [
            {"type": "text", "text": "first text"},
            {"type": "tool_result", "tool_use_id": "a", "content": "A"},
            {"type": "text", "text": "second text"},
            {"type": "tool_result", "tool_use_id": "b", "content": "B"},
        ]
        projected = _read(_minimal(messages=[{"role": "user", "content": content}]))

        parts = projected.conversation.turns[0].parts

        assert [type(part).__name__ for part in parts] == ["Text", "ToolResult", "Text", "ToolResult"]

    def test_an_assistant_turn_keeps_its_order_too(self) -> None:
        """R4.8 — the rule was never role-specific once the re-sort is gone."""
        content = [{"type": "text", "text": "calling"}, {"type": "tool_result", "tool_use_id": "a", "content": "x"}]
        projected = _read(_minimal(messages=[{"role": "assistant", "content": content}]))

        assert projected.conversation.turns[0].parts[0] == c.Text("calling")

    def test_an_orphan_tool_result_still_projects(self) -> None:
        """R4.9 — M7 exists to drop orphans, so raising here would hide the delta that names it."""
        block = {"type": "tool_result", "tool_use_id": "toolu_nothing_matches", "content": "x"}
        projected = _read(_minimal(messages=[{"role": "user", "content": [block]}]))

        assert projected.conversation.turns[0].parts[0].tool_use_id == "toolu_nothing_matches"
        c.verify_total(projected)


# --------------------------------------------------------------------------
# R5 — tools and sampling
# --------------------------------------------------------------------------


class TestToolsAndSampling:
    """R5 — declarations by name, sampling on the canonical spelling."""

    def test_the_published_tool_example_projects_its_schema_unchanged(self) -> None:
        """R5.1 — one oracle falsification case deletes the description, so it must be carried."""
        projected = _read(_minimal(tools=[PUBLISHED_TOOL]))

        assert projected.conversation.tools == (
            c.ToolDecl(
                name="get_stock_price",
                description="Get the current stock price for a given ticker symbol.",
                schema=PUBLISHED_TOOL["input_schema"],
                strict=None,
            ),
        )
        c.verify_total(projected)

    def test_strict_is_absent_rather_than_false(self) -> None:
        """R5.1 — P15 strips it on the Responses path; presence and absence must differ."""
        assert _read(_minimal(tools=[PUBLISHED_TOOL])).conversation.tools[0].strict is None

    def test_a_wrongly_typed_schema_residualises_rather_than_being_fabricated(self) -> None:
        """R5.1 — `dict(["ab","cd"])` invents `{"a":"b","c":"d"}`, a schema the agent never sent.

        Residualised rather than raised: raising would blind the oracle to
        everything else in an otherwise diffable request, while the residual
        fails the run and names the field.
        """
        projected = _read(_minimal(tools=[{"name": "t", "input_schema": ["ab", "cd"]}]))

        assert projected.conversation.tools[0].schema is None
        assert projected.residual == {"tools[0].input_schema": ["ab", "cd"]}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_wrongly_typed_tool_arguments_residualise_rather_than_being_fabricated(self) -> None:
        """R4.5 — the same trap on `tool_use.input`, reachable from the upstream side.

        Chat Completions carries arguments as a JSON *string*, so an upstream
        body that failed to parse one back lands here.
        """
        block = {"type": "tool_use", "id": "t", "name": "n", "input": ["ab", "cd"]}
        projected = _read(_minimal(messages=[{"role": "assistant", "content": [block]}]))

        assert projected.conversation.turns[0].parts[0].arguments == {}
        assert projected.residual == {"messages[0].content[0].input": ["ab", "cd"]}

    def test_a_server_tool_residualises_its_type_under_an_indexed_key(self) -> None:
        """R5.2 — indexed, not by name (§7.4.1): a residual key is the body's own path."""
        projected = _read(_minimal(tools=[{"type": "web_search_20250305", "name": "web_search"}]))

        assert projected.residual == {"tools[0].type": "web_search_20250305"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_the_control_for_that_case_passes(self) -> None:
        """R5.2 — the paired control."""
        projected = _read(_minimal(tools=[PUBLISHED_TOOL]))

        assert projected.residual == {}
        c.verify_total(projected)

    def test_sampling_maps_onto_the_canonical_chat_completions_spelling(self) -> None:
        """R5.3 — `stop_sequences` becomes `stop`; `Conversation` rejects anything else."""
        projected = _read(
            _minimal(
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                stop_sequences=["\n\nHuman:"],
            )
        )

        assert projected.conversation.sampling == {
            "max_tokens": 1024,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "stop": ["\n\nHuman:"],
        }

    def test_the_anthropic_spelling_never_reaches_the_projection(self) -> None:
        """R5.3 — the half of the rename a positive assertion alone would not catch."""
        projected = _read(_minimal(stop_sequences=["x"]))

        assert "stop_sequences" not in projected.conversation.sampling


# --------------------------------------------------------------------------
# R6 — totality
# --------------------------------------------------------------------------


class TestTotality:
    """R6 — every key accounted for, and the falsification cases that prove it."""

    def test_the_published_full_request_is_total(self) -> None:
        """R6.1 — the reference's own complete example must project cleanly or nothing else matters."""
        projected = _read(PUBLISHED_FULL_REQUEST)

        assert projected.consumed == set(PUBLISHED_FULL_REQUEST)
        assert projected.residual == {}
        c.verify_total(projected)

    def test_source_is_the_parsed_body_the_reader_actually_read(self) -> None:
        """R6.2 — carried so `verify_total` needs no second parse that could disagree."""
        projected = _read(PUBLISHED_FULL_REQUEST)

        assert dict(projected.source) == PUBLISHED_FULL_REQUEST

    def test_an_injected_unrecognised_key_fails_the_run(self) -> None:
        """R6.4 — the ticket's named falsification case, and §3.3.1's fifth oracle case.

        Constructed, not published: `x-kitty-trace` is the injected field
        §3.3.1's falsification table names.
        """
        projected = _read(_minimal(**{"x-kitty-trace": "abc"}))

        assert projected.residual == {"x-kitty-trace": "abc"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_the_control_for_the_injected_key_passes(self) -> None:
        """R6.4 — without this, a reader that raised unconditionally would satisfy the case above."""
        projected = _read(_minimal())

        assert projected.residual == {}
        c.verify_total(projected)

    def test_a_residual_key_is_the_bare_body_path_and_never_the_delta_spelling(self) -> None:
        """R6.3 — the wrapped form misses `source` and raises `DroppedFieldsError`, naming the wrong defect."""
        projected = _read(_minimal(**{"x-kitty-trace": "abc"}))

        assert "x-kitty-trace" in projected.residual
        assert c.residual_path("x-kitty-trace") not in projected.residual


# --------------------------------------------------------------------------
# R7 — unreadable bodies
# --------------------------------------------------------------------------


class TestUnreadableBodies:
    """R7 — one exception type, so T-D1 can tell a bad body from an I1 breach."""

    @pytest.mark.parametrize(
        "case, body",
        [
            ("not JSON", b"{not json"),
            ("not an object", b"[1, 2, 3]"),
            ("messages not a list", _minimal(messages={"role": "user"})),
            ("message not an object", _minimal(messages=["hello"])),
            ("message lacks role", _minimal(messages=[{"content": "hi"}])),
            ("message lacks content", _minimal(messages=[{"role": "user"}])),
            ("role outside the two", _minimal(messages=[{"role": "system", "content": "hi"}])),
            ("content neither string nor list", _minimal(messages=[{"role": "user", "content": 7}])),
            ("system neither string nor list", _minimal(system=7)),
            ("system block not text", _minimal(system=[{"type": "image", "source": {}}])),
            ("tools not a list", _minimal(tools={"name": "x"})),
            ("tools entry not an object", _minimal(tools=["x"])),
            ("block not an object", _minimal(messages=[{"role": "user", "content": ["raw"]}])),
            ("block lacks type", _minimal(messages=[{"role": "user", "content": [{"text": "hi"}]}])),
            (
                "tool_result content member type not a string",
                _minimal(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "t", "content": [{"type": {"a": 1}, "b": 2}]}
                            ],
                        }
                    ]
                ),
            ),
            ("system block text not a string", _minimal(system=[{"type": "text", "text": {"a": 1}}])),
            (
                "thinking text not a string",
                _minimal(messages=[{"role": "assistant", "content": [{"type": "thinking", "thinking": 7}]}]),
            ),
            ("tool_choice unrecognised", _minimal(tool_choice={"type": "magic"})),
            ("tool_choice tool without name", _minimal(tool_choice={"type": "tool"})),
            ("system member not an object", _minimal(system=["hello"])),
            ("tools entry lacks name", _minimal(tools=[{"description": "d"}])),
            (
                "image lacks source",
                _minimal(messages=[{"role": "user", "content": [{"type": "image"}]}]),
            ),
            (
                "image source type unknown",
                _minimal(messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "ftp"}}]}]),
            ),
            (
                "tool_use lacks name",
                _minimal(messages=[{"role": "assistant", "content": [{"type": "tool_use", "id": "t"}]}]),
            ),

            (
                "tool_result content member not an object",
                _minimal(
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "t", "content": [7]}],
                        }
                    ]
                ),
            ),
        ],
    )
    def test_a_malformed_body_raises_unreadable_body_error(self, case: str, body: Any) -> None:
        """R7 — the enumerated cases, each raising with a message that is not empty.

        The non-empty check is the half a bare ``pytest.raises`` misses: a reader
        that raised ``UnreadableBodyError("")`` everywhere would satisfy every
        row here while telling a maintainer nothing.
        """
        with pytest.raises(c.UnreadableBodyError) as raised:
            _read(body)

        assert str(raised.value).strip()

    def test_an_unanticipated_structural_failure_is_still_an_unreadable_body(self) -> None:
        """R7 — the catch-all is the guarantee; the enumeration is only what is tested by name.

        A text block with no `text` key reaches a `KeyError` no branch above
        anticipates. The contract names three failure shapes, so an escaping
        `KeyError` would be an undefined fourth on the one path T-D1 uses to
        tell an unreadable body from an I1 breach.
        """
        body = _minimal(messages=[{"role": "user", "content": [{"type": "text"}]}])

        with pytest.raises(c.UnreadableBodyError):
            _read(body)

    def test_a_bad_role_never_escapes_as_a_value_error(self) -> None:
        """R7.1 — `Turn` would raise `ValueError`, which the contract defines as a reader bug."""
        body = _minimal(messages=[{"role": "system", "content": "hi"}])

        try:
            _read(body)
        except c.UnreadableBodyError:
            return
        except ValueError as exc:
            pytest.fail(f"reader constructed before classifying: {exc!r}")

        pytest.fail("expected UnreadableBodyError")

    def test_absent_required_fields_project_rather_than_raise(self) -> None:
        """R7.2 — the oracle catches a vanished field as a delta, a better diagnosis than a crash.

        A reader that rejected an incomplete body could not project the very
        mutation M5, M6 and M7 produce.
        """
        projected = _read({"max_tokens": 1024})

        assert projected.conversation.turns == ()
        assert projected.envelope.model is None
        c.verify_total(projected)
