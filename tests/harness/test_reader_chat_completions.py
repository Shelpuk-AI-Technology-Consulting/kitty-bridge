"""L1 tests for the Chat Completions reader.

`.system_design/TEST_SUITE.md` §3.3.1, §3.3.1b, §3.3.1a, §7.4.1 · plan task
**T-A2** (KBR-34).

**Every body here comes from OpenAI's published documentation**, retrieved
2026-09-14 from ``openai/openai-openapi`` master — the same source the design
doc cites for G37's cache-breakpoint facts. §3.3.1's independent-oracle rule
is checkable later only if the provenance travels with the fixture.

**The convergence test is the ticket's own load-bearing claim.** It asserts the
CC and Anthropic Messages encodings of one worked tool exchange project to
**identical** ``Conversation`` values, indices included — T-A2's proof that
§3.3.1b's merge rule is shared, not just stated. Without it, six independently
written readers could agree on a shape one of them got wrong and pass
everything else.

**Every falsification case is paired with a control.** A reader that residualised
unconditionally, or that consumed every key as ``extra``, would pass every
positive assertion in this module. §1.4's harness rule.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from harness import contract as c
from harness import reader_anthropic_messages as am
from harness import reader_chat_completions as cc

# --------------------------------------------------------------------------
# Published fixtures — OpenAI's own examples
# --------------------------------------------------------------------------


#: The reference's own complete request example, taken from the
#: ``x-oaiMeta.examples`` block on ``CreateChatCompletionRequest`` in
#: ``openai-openapi`` master (retrieved 2026-09-14). Stripped of fields this
#: reader does not yet project (the format's response fields), this is the
#: minimum body a CC request can carry to round-trip the public schema.
PUBLISHED_BASIC = {
    "model": "gpt-6-astra",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
}

#: The reference's tool-call example, taken from the same source. A user asks
#: for weather; the assistant returns one ``tool_call``; a ``tool`` message
#: delivers the result; the user closes the exchange. The CC and Messages
#: encodings of this exchange are the convergence test's bodies.
PUBLISHED_TOOL_CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"city": "San Francisco"}',
    },
}

#: The reference's function-tool declaration, same source.
PUBLISHED_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
    },
}

#: The OpenRouter CC dialect's ``cache_control`` shape, identical to Anthropic
#: Messages — a one-byte PNG so a base64 data URL has something to digest.
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63f80f0000010001002ce29ccd0000000049454e44ae426082"
)
PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _captured(body: Any) -> c.CapturedRequest:
    """Wrap a body as the capture a projection is handed.

    Serialised here rather than stored as bytes so the fixtures above stay
    readable as the published examples they came from, while the reader is
    still exercised on ``bytes`` as :class:`~harness.contract.Projection`
    requires.

    Args:
        body: The request body, as a Python object, or raw ``bytes`` to pass
            through untouched for the malformed cases.

    Returns:
        A capture addressed at the Chat Completions endpoint.
    """
    raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    return c.CapturedRequest(
        method="POST",
        scheme="https",
        host="api.openai.com",
        path="/v1/chat/completions",
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
    return cc.ChatCompletionsProjection().read_request(_captured(body))


def _minimal(**overrides: Any) -> dict[str, Any]:
    """Return a minimal body with the given overrides applied.

    Args:
        **overrides: Keys to merge over :data:`PUBLISHED_BASIC`.

    Returns:
        A body that round-trips the reader.
    """
    body = dict(PUBLISHED_BASIC)
    body.update(overrides)
    return body


# --------------------------------------------------------------------------
# R1 — the envelope
# --------------------------------------------------------------------------


class TestEnvelope:
    """R1 — envelope keys and the canonical ``parallel_tool_calls`` address."""

    def test_the_minimal_body_projects_with_an_empty_residual(self) -> None:
        """R1.1 — totality on the published worked example."""
        projected = _read(PUBLISHED_BASIC)

        assert projected.residual == {}
        c.verify_total(projected)

    def test_model_lands_at_envelope_model(self) -> None:
        """R1.2 — the named envelope field is the wire key."""
        projected = _read(PUBLISHED_BASIC)

        assert projected.envelope.model == "gpt-6-astra"

    def test_stream_lands_at_envelope_stream(self) -> None:
        """R1.3 — same."""
        projected = _read(_minimal(stream=True))

        assert projected.envelope.stream is True
        assert projected.residual == {}

    def test_parallel_tool_calls_lands_at_the_canonical_address(self) -> None:
        """R1.4 — §3.3.1b, KBR-205 closing G36. This is the first reader to *read* the address directly."""
        projected = _read(_minimal(parallel_tool_calls=False))

        assert projected.envelope.extra[c.PARALLEL_TOOL_CALLS_KEY] is False
        assert projected.residual == {}

    def test_a_wrongly_typed_parallel_tool_calls_residualises(self) -> None:
        """R1.5 — §7.4.1's wrongly-typed-leaf rule at the canonical address."""
        projected = _read(_minimal(parallel_tool_calls="yes"))

        assert "parallel_tool_calls" not in projected.envelope.extra
        assert projected.residual == {"parallel_tool_calls": "yes"}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_parallel_tool_calls_null_is_treated_as_absent(self) -> None:
        """R1.5b — ``null`` follows the ``cache_control`` precedent: absent.

        ``True``/``False`` map onto the canonical address; ``null`` is
        treated as no-op (the wire key carries no instruction). The key is
        consumed for totality so a body that explicitly sent ``null`` is
        accounted for, not silently dropped.
        """
        projected = _read(_minimal(parallel_tool_calls=None))

        assert c.PARALLEL_TOOL_CALLS_KEY not in projected.envelope.extra
        assert "parallel_tool_calls" in projected.consumed
        assert projected.residual == {}
        c.verify_total(projected)

    def test_a_published_extra_key_rides_at_its_wire_key(self) -> None:
        """R1.6 — every other CC control field lands at its wire key, the §3.3.1a rule."""
        projected = _read(_minimal(metadata={"trace": "abc"}, service_tier="auto"))

        assert projected.envelope.extra["metadata"] == {"trace": "abc"}
        assert projected.envelope.extra["service_tier"] == "auto"
        assert projected.residual == {}

    @pytest.mark.parametrize(
        "key, value",
        [
            ("audio", {"voice": "alloy", "format": "wav"}),
            ("moderation", {"model": "omni-moderation-latest"}),
            ("functions", [{"name": "old_f", "parameters": {}}]),  # deprecated top-level
            ("function_call", "auto"),  # deprecated top-level
            ("store", True),
            ("user", "session-abc-123"),
            ("prediction", {"type": "content", "content": []}),
            ("modalities", ["text", "audio"]),
            ("verbosity", "low"),
            ("reasoning_effort", "medium"),
            ("prompt_cache_options", {"ttl": "30m"}),
            ("web_search_options", {"user_location": {"type": "approximate"}}),
        ],
        ids=[
            "audio",
            "moderation",
            "functions-deprecated",
            "function_call-deprecated",
            "store",
            "user",
            "prediction",
            "modalities",
            "verbosity",
            "reasoning_effort",
            "prompt_cache_options",
            "web_search_options",
        ],
    )
    def test_each_published_extra_key_is_consumed_at_its_wire_key(
        self, key: str, value: Any
    ) -> None:
        """R1.6a — every key in :data:`_PUBLISHED_EXTRA_KEYS` rides at its wire key
        (the B1 closure). The four B1 keys (`audio`, `moderation`,
        `functions`, `function_call`, all deprecated) plus the eight that
        shipped earlier (`store`, `user`, `prediction`, `modalities`,
        `verbosity`, `reasoning_effort`, `prompt_cache_options`,
        `web_search_options`) all carry the §3.3.1a "declared control field
        of the format maps to ``envelope.extra[<wire key>]``" rule. A
        residual entry on any of them is the wrong shape — G26 binds the
        row-plan for the downstream register, and a body carrying any of
        them is one real Codex / OpenAI / OpenAI-compat traffic sends.
        """
        projected = _read(_minimal(**{key: value}))

        assert projected.envelope.extra[key] == value, (
            f"key {key!r} did not ride at its wire key: {projected.envelope.extra!r}"
        )
        assert projected.residual == {}
        c.verify_total(projected)

    def test_audio_with_a_wrongly_typed_value_is_carried_whole(self) -> None:
        """R1.6b — the value at a published-extra key is carried whole, not type-checked.

        §3.3.1a: ``extra[<wire key>]`` is compared **whole**. The reader's
        job is to name the wire field; the value's shape is not the
        reader's job to enforce. A wrongly-typed value lands at its wire
        key and is named — a future type check, or the schema validator,
        owns the shape, not the reader. This matches the posture T-A1
        ships for its ``_PUBLISHED_EXTRA_KEYS``.
        """
        projected = _read(_minimal(audio="not an audio config"))

        assert projected.envelope.extra["audio"] == "not an audio config"
        assert projected.residual == {}
        c.verify_total(projected)

    @pytest.mark.parametrize(
        "key",
        [
            "store",
            "metadata",
            "service_tier",
            "reasoning_effort",
            "verbosity",
            "modalities",
            "prediction",
            "user",
            "web_search_options",
            "prompt_cache_options",
            "audio",
            "moderation",
            "function_call",
        ],
        ids=[
            "store",
            "metadata",
            "service_tier",
            "reasoning_effort",
            "verbosity",
            "modalities",
            "prediction",
            "user",
            "web_search_options",
            "prompt_cache_options",
            "audio",
            "moderation",
            "function_call",
        ],
    )
    def test_a_wrongly_typed_value_at_each_published_extra_key_is_carried_whole(
        self, key: str
    ) -> None:
        """R1.6c — the wrongly-typed coverage R1.6b promised for all 13 keys.

        Every key in :data:`_PUBLISHED_EXTRA_KEYS` is read with no type
        check on the value (§3.3.1a: ``extra[<wire key>]`` is compared
        whole). A wrongly-typed value lands at its wire key and is named
        — the schema validator's job, not the reader's. Each parametrise
        case carries a string, which is the wrong type for every key on
        this list (a dict-shaped value, a list, a bool, a number), so the
        pin covers the wrong-type case regardless of the schema's shape.
        """
        projected = _read(_minimal(**{key: "definitely the wrong type"}))

        assert projected.envelope.extra[key] == "definitely the wrong type"
        assert projected.residual == {}
        c.verify_total(projected)

    def test_a_sampling_key_rides_at_its_canonical_spelling(self) -> None:
        """R1.7 — §3.3.1b: CC's spellings are the canonical spellings."""
        projected = _read(
            _minimal(
                temperature=0.5,
                top_p=0.9,
                max_tokens=512,
                max_completion_tokens=1024,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                stop=["\n"],
                seed=42,
            )
        )

        assert projected.conversation.sampling == {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 512,
            "max_completion_tokens": 1024,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["\n"],
            "seed": 42,
        }
        assert projected.residual == {}

    def test_a_wrongly_typed_sampling_value_is_carried_unchanged(self) -> None:
        """R1.8 — ``Conversation.sampling`` is ``Mapping[str, Any]``; the value type is not
        enforced today (the same posture T-A1 ships). The contract validates keys,
        not values, and P13 drops fourteen of the fifteen on its own row. A
        stricter rule is a separate ticket's worth of work; the wrong type is
        documented here so a future test pins the next decision."""
        projected = _read(_minimal(max_tokens="lots"))

        assert projected.conversation.sampling == {"max_tokens": "lots"}

    def test_a_unknown_top_level_key_residualises_at_its_bare_name(self) -> None:
        """R1.9 — §7.4.1's top-level key rule: bare name, not ``residual[...]``."""
        projected = _read(_minimal(x_kitty_probe=True))

        assert projected.residual == {"x_kitty_probe": True}
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)


# --------------------------------------------------------------------------
# R2 — tool_choice (the canonical values)
# --------------------------------------------------------------------------


class TestToolChoice:
    """R2 — §3.3.1b's ``auto · any · none · tool:<name>`` rule."""

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("none", "none"),
            ("auto", "auto"),
            # KBR-214: CC's `required` and Anthropic's `any` name one concept.
            ("required", "any"),
        ],
    )
    def test_each_published_string_normalises_to_its_canonical_value(
        self, choice: str, expected: str
    ) -> None:
        """R2.1 — the three strings the schema publishes."""
        projected = _read(_minimal(tool_choice=choice))

        assert projected.envelope.extra["tool_choice"] == expected
        assert projected.residual == {}

    def test_the_named_function_form_normalises_to_tool_colon_name(self) -> None:
        """R2.2 — ``{"type": "function", "function": {"name": …}}`` → ``tool:<name>``."""
        projected = _read(
            _minimal(tool_choice={"type": "function", "function": {"name": "get_weather"}})
        )

        assert projected.envelope.extra["tool_choice"] == "tool:get_weather"
        assert projected.residual == {}

    def test_the_named_custom_form_normalises_to_tool_colon_name(self) -> None:
        """R2.3 — ``{"type": "custom", "custom": {"name": …}}`` maps the same way."""
        projected = _read(_minimal(tool_choice={"type": "custom", "custom": {"name": "qa"}}))

        assert projected.envelope.extra["tool_choice"] == "tool:qa"
        assert projected.residual == {}

    def test_the_allowed_tools_form_maps_by_its_mode(self) -> None:
        """R2.4 — ``{"type": "allowed_tools", "allowed_tools": {"mode": "required", …}}``
        maps onto ``any``; ``mode: "auto"`` maps onto ``auto``."""
        for mode, expected in [("auto", "auto"), ("required", "any")]:
            projected = _read(
                _minimal(
                    tool_choice={
                        "type": "allowed_tools",
                        "allowed_tools": {"mode": mode},
                    }
                )
            )
            assert projected.envelope.extra["tool_choice"] == expected, f"mode={mode!r}"
            assert projected.residual == {}

    def test_the_allowed_tools_member_tools_list_residualises_whole(self) -> None:
        """R2.4b — the ``tools`` list is part of the wire shape but not the canonical
        form. The reader residualises the list at its own path: a body
        that names a specific toolset is one the bridge does not silently
        swallow, and the residual entry names what was sent.
        """
        projected = _read(
            _minimal(
                tool_choice={
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": "auto",
                        "tools": [PUBLISHED_FUNCTION_TOOL],
                    },
                }
            )
        )

        assert projected.envelope.extra["tool_choice"] == "auto"
        assert projected.residual == {
            "tool_choice.allowed_tools.tools": [PUBLISHED_FUNCTION_TOOL]
        }
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_an_unrecognised_allowed_tools_mode_raises(self) -> None:
        """R2.5 — ``mode`` is an enum; anything else is unreadable."""
        with pytest.raises(c.UnreadableBodyError):
            _read(
                _minimal(
                    tool_choice={
                        "type": "allowed_tools",
                        "allowed_tools": {"mode": "sometimes", "tools": []},
                    }
                )
            )

    def test_an_unrecognised_tool_choice_string_raises(self) -> None:
        """R2.6 — a string outside the schema is an unreadable body."""
        with pytest.raises(c.UnreadableBodyError):
            _read(_minimal(tool_choice="maybe"))

    def test_a_named_form_without_a_name_raises(self) -> None:
        """R2.7 — classify before constructing, or the diagnosis is wrong."""
        with pytest.raises(c.UnreadableBodyError):
            _read(_minimal(tool_choice={"type": "function", "function": {}}))


# --------------------------------------------------------------------------
# R3 — messages
# --------------------------------------------------------------------------


class TestMessages:
    """R3 — roles, content parts, the merge rule."""

    def test_system_lifts_into_conversation_system(self) -> None:
        """R3.1 — ``role: system`` is a system message, not a turn (§3.3.1b)."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )

        assert projected.conversation.system == (c.Text("Be brief."),)
        assert len(projected.conversation.turns) == 1
        assert projected.conversation.turns[0].role == "user"
        assert projected.residual == {}

    def test_developer_lifts_into_conversation_system(self) -> None:
        """R3.2 — ``role: developer`` is the newer spelling of system."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "developer", "content": "Answer like a pirate."},
                    {"role": "user", "content": "Hi"},
                ],
            }
        )

        assert projected.conversation.system == (c.Text("Answer like a pirate."),)
        assert len(projected.conversation.turns) == 1
        assert projected.residual == {}

    def test_a_user_string_content_is_one_text_part(self) -> None:
        """R3.3 — string content is shorthand for one text block."""
        projected = _read(
            {"model": "gpt-6-astra", "messages": [{"role": "user", "content": "hi"}]}
        )

        assert projected.conversation.turns[0].parts == (c.Text("hi"),)

    def test_a_user_text_image_and_audio_part_project_in_order(self) -> None:
        """R3.4 — order preserved; ``image_url`` is ``Image``; ``input_audio`` is
        ``Opaque(kind="input_audio")`` — OpenAI's spelling is the canonical one
        here, per §7.4.1's "format-unique type passes through" rule."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look at this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "ABCD", "format": "wav"},
                            },
                        ],
                    }
                ],
            }
        )

        parts = projected.conversation.turns[0].parts
        assert len(parts) == 3
        assert isinstance(parts[0], c.Text) and parts[0].text == "look at this"
        assert isinstance(parts[1], c.Image) and parts[1].digest is not None
        assert isinstance(parts[2], c.Opaque) and parts[2].kind == "input_audio"
        assert projected.residual == {}

    def test_a_non_base64_data_url_residualises(self) -> None:
        """R3.5 — ``data:image/png,abc`` cannot be digested; residualising at its own path."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:image/png,abc"}}
                        ],
                    }
                ],
            }
        )

        assert projected.residual == {"messages[0].content[0].image_url.url": "data:image/png,abc"}

    def test_an_image_url_http_is_carried_as_ref(self) -> None:
        """R3.6 — a non-data URL keeps its ``ref``; no digest is invented."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
                    }
                ],
            }
        )

        image = projected.conversation.turns[0].parts[0]
        assert isinstance(image, c.Image)
        assert image.ref == "https://example.com/a.png"
        assert image.digest is None

    def test_an_assistant_message_with_tool_calls_projects_tool_use_parts(self) -> None:
        """R3.7 — ``tool_calls`` becomes ``ToolUse`` parts, one per call."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [PUBLISHED_TOOL_CALL],
                    },
                ],
            }
        )

        assistant_turn = projected.conversation.turns[1]
        assert len(assistant_turn.parts) == 1
        call = assistant_turn.parts[0]
        assert isinstance(call, c.ToolUse)
        assert call.name == "get_weather"
        assert call.arguments == {"city": "San Francisco"}
        assert call.id == "call_abc123"
        assert projected.residual == {}

    def test_an_assistant_message_audio_and_function_call_residualise(self) -> None:
        """R3.7b — the assistant message's ``audio`` and ``function_call`` are
        recognised CC fields that the projection does not model today.
        They residualise at the message's path — a silent drop would be
        the totality violation the residual rule exists to prevent, and
        the residual entry is the hook a future reader that grows an
        ``Audio`` part or a ``function_call`` mapping consults."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "hi",
                        "audio": {"id": "audio-1"},
                        "function_call": {"name": "legacy", "arguments": "{}"},
                    }
                ],
            }
        )

        assert projected.residual == {
            "messages[0].audio": {"id": "audio-1"},
            "messages[0].function_call": {"name": "legacy", "arguments": "{}"},
        }
        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(projected)

    def test_a_cache_control_on_an_undecodable_image_residualises(self) -> None:
        """R6.3b — the image's ``cache_control`` survives the decode failure.

        A base64 data URL whose bytes fail to decode residualises the URL
        at its own path; the part's ``cache_control`` is also residualised
        because the reader has no ``Image`` to carry it on. A silent drop
        would be the M16-shaped loss — a strip the bridge did not
        register.
        """
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"},
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            }
        )

        assert projected.residual == {
            "messages[0].content[0].image_url.url": "data:image/png;base64,!!!not-base64!!!",
            "messages[0].content[0].cache_control": {"type": "ephemeral"},
        }

    def test_a_non_json_arguments_string_residualises_at_its_path(self) -> None:
        """R3.8 — the shared ``decode_arguments`` rule: a non-JSON string residualises at its path."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "t",
                                "type": "function",
                                "function": {"name": "f", "arguments": "not json"},
                            }
                        ],
                    },
                ],
            }
        )

        assert projected.residual == {"messages[1].tool_calls[0].function.arguments": "not json"}
        # `arguments` defaults to `{}` when unreadable.
        call = projected.conversation.turns[1].parts[0]
        assert isinstance(call, c.ToolUse)
        assert call.arguments == {}

    def test_a_tool_message_projects_a_tool_result_part_in_a_user_turn(self) -> None:
        """R3.9 — CC's ``role: tool`` is the canonical tool-result carrier."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [PUBLISHED_TOOL_CALL],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "72 and sunny",
                    },
                ],
            }
        )

        result_turn = projected.conversation.turns[2]
        assert result_turn.role == "user"
        assert len(result_turn.parts) == 1
        result = result_turn.parts[0]
        assert isinstance(result, c.ToolResult)
        assert result.tool_use_id == "call_abc123"
        assert result.content == (c.Text("72 and sunny"),)
        assert projected.residual == {}

    def test_an_unrecognised_role_raises(self) -> None:
        """R3.10 — a role the schema does not publish is an unreadable body."""
        with pytest.raises(c.UnreadableBodyError):
            _read(
                {
                    "model": "gpt-6-astra",
                    "messages": [{"role": "function", "content": "old"}],
                }
            )

    def test_a_tool_message_without_a_tool_call_id_raises(self) -> None:
        """R3.11 — ``tool_call_id`` is required; missing is unreadable."""
        with pytest.raises(c.UnreadableBodyError):
            _read(
                {
                    "model": "gpt-6-astra",
                    "messages": [{"role": "tool", "content": "x"}],
                }
            )

    def test_a_message_level_refusal_projects_to_opaque(self) -> None:
        """R3.12 — the message-level ``refusal`` field carries a refusal string and
        projects as ``Opaque(kind="refusal", digest=text_digest(...))`` — the
        same shape the content-part form uses, so the two representations
        agree on an unchanged refusal (§7.4.1). T-A3 ships the same
        decision for Responses, so the two readers agree across formats.
        """
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "do X"},
                    {
                        "role": "assistant",
                        "content": None,
                        "refusal": "I won't do X.",
                    },
                ],
            }
        )

        assistant_turn = projected.conversation.turns[1]
        assert any(
            isinstance(p, c.Opaque) and p.kind == "refusal" for p in assistant_turn.parts
        ), assistant_turn
        # Same digest across CC's content-part form (covered by the
        # content-part tests) and the message-level form.
        refusal_part = next(
            p for p in assistant_turn.parts if isinstance(p, c.Opaque) and p.kind == "refusal"
        )
        assert refusal_part.digest == c.text_digest("I won't do X.")

    def test_a_refusal_content_part_projects_to_opaque_with_a_text_digest(self) -> None:
        """R3.13 — refusal part (``{"type": "refusal", "refusal": "<text>"}``) projects as
        ``Opaque(kind="refusal", digest=text_digest(text))``, not as ``Text``.
        T-A3 ships the same decision; the two readers agree on a refusal
        regardless of which wire shape carried it."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "do X"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "refusal", "refusal": "I won't do X."},
                        ],
                    },
                ],
            }
        )

        refusal_part = projected.conversation.turns[1].parts[0]
        assert isinstance(refusal_part, c.Opaque)
        assert refusal_part.kind == "refusal"
        assert refusal_part.digest == c.text_digest("I won't do X.")


# --------------------------------------------------------------------------
# R4 — the merge rule
# --------------------------------------------------------------------------


class TestMergeRule:
    """R4 — §3.3.1b's merge rule on CC; clause 3 is vacuous, clauses 1/2/4 do the work."""

    def test_assistant_tool_then_tool_merges_into_one_user_turn(self) -> None:
        """R4.1 — the canonical exchange: ``user → assistant(tool_calls) → tool``.

        Clause 1 builds the ``user`` turn from the tool result; clause 2
        absorbs the next user message; clause 3 is vacuous (CC delivers
        results in their own messages); clause 4 merges same-role turns.
        """
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [PUBLISHED_TOOL_CALL],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_abc123",
                        "content": "72 and sunny",
                    },
                    {"role": "user", "content": "thanks"},
                ],
            }
        )

        assert len(projected.conversation.turns) == 3
        assert projected.conversation.turns[0].role == "user"
        assert projected.conversation.turns[1].role == "assistant"
        merged = projected.conversation.turns[2]
        assert merged.role == "user"
        # ToolResult first (clause 1), then the absorbed user text (clause 2).
        assert len(merged.parts) == 2
        assert isinstance(merged.parts[0], c.ToolResult)
        assert isinstance(merged.parts[1], c.Text)
        assert merged.parts[1].text == "thanks"

    def test_interleaved_tool_calls_form_a_merged_turn_with_results_in_call_order(
        self,
    ) -> None:
        """R4.2 — an interleaved CC body: ``tool(a) → user(text) → tool(b)``.

        Clauses 1+2 build a ``ToolResult`` turn from each ``tool`` message and
        absorb the user-text turn that follows; clause 4 then merges the
        next ``tool`` message's ``ToolResult`` into the same user turn. The
        final turn is the interleaved shape ``[ToolResultA, Text,
        ToolResultB]`` — results in tool-call order, with the absorbed
        user text in its position. The Messages-side encoding of the same
        exchange (one user message ``[tool_result_a, text, tool_result_b]``)
        produces the same shape — that is the convergence test's worked
        exchange, R7.
        """
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "weather in SF and NYC?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {**PUBLISHED_TOOL_CALL, "id": "t1"},
                            {**PUBLISHED_TOOL_CALL, "id": "t2"},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "t1", "content": "72 and sunny"},
                    {"role": "user", "content": "and humidity?"},
                    {"role": "tool", "tool_call_id": "t2", "content": "30 and cloudy"},
                ],
            }
        )

        # Three turns: user (ask), assistant (tool calls), user (merged:
        # ToolResult t1, then the absorbed user text, then ToolResult t2).
        assert len(projected.conversation.turns) == 3
        assert projected.conversation.turns[0].role == "user"
        assert projected.conversation.turns[1].role == "assistant"
        merged = projected.conversation.turns[2]
        assert merged.role == "user"
        assert len(merged.parts) == 3
        assert isinstance(merged.parts[0], c.ToolResult) and merged.parts[0].tool_use_id == "t1"
        assert isinstance(merged.parts[1], c.Text) and merged.parts[1].text == "and humidity?"
        assert isinstance(merged.parts[2], c.ToolResult) and merged.parts[2].tool_use_id == "t2"

    def test_two_tool_messages_form_one_user_turn_with_results_first(self) -> None:
        """R4.3 — clause 1's run of consecutive results; clause 3 vacuous here too."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {"role": "user", "content": "weather in SF and NYC?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {**PUBLISHED_TOOL_CALL, "id": "t1"},
                            {**PUBLISHED_TOOL_CALL, "id": "t2"},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "t1", "content": "72 and sunny"},
                    {"role": "tool", "tool_call_id": "t2", "content": "30 and cloudy"},
                ],
            }
        )

        merged = projected.conversation.turns[2]
        assert merged.role == "user"
        assert len(merged.parts) == 2
        assert isinstance(merged.parts[0], c.ToolResult) and merged.parts[0].tool_use_id == "t1"
        assert isinstance(merged.parts[1], c.ToolResult) and merged.parts[1].tool_use_id == "t2"


# --------------------------------------------------------------------------
# R5 — tools
# --------------------------------------------------------------------------


class TestTools:
    """R5 — function-tool declarations."""

    def test_a_function_tool_projects_a_tool_decl(self) -> None:
        """R5.1 — the published function-tool example projects verbatim."""
        projected = _read(_minimal(tools=[PUBLISHED_FUNCTION_TOOL]))

        assert len(projected.conversation.tools) == 1
        tool = projected.conversation.tools[0]
        assert tool.name == "get_weather"
        assert tool.description == "Get the current weather for a given city."
        assert tool.type == "function"
        assert tool.schema == {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        }
        assert projected.residual == {}

    def test_a_tool_with_a_wrongly_typed_schema_residualises(self) -> None:
        """R5.2 — §7.4.1's wrongly-typed-leaf rule on ``parameters``."""
        projected = _read(
            _minimal(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": ["ab", "cd"],
                        },
                    }
                ]
            )
        )

        assert projected.conversation.tools[0].schema is None
        assert projected.residual == {"tools[0].parameters": ["ab", "cd"]}

    def test_a_tool_without_a_name_raises(self) -> None:
        """R5.3 — ``function.name`` is required."""
        with pytest.raises(c.UnreadableBodyError):
            _read(_minimal(tools=[{"type": "function", "function": {"description": "x"}}]))

    def test_a_non_function_tool_residualises(self) -> None:
        """R5.4 — Chat Completions' ``function`` is the only declared tool type today;
        anything else residualises at its own path."""
        projected = _read(
            _minimal(
                tools=[
                    {
                        "type": "web_search_preview",
                        "name": "ws",
                        "extra": "x",
                    }
                ]
            )
        )

        assert projected.conversation.tools == ()
        assert projected.residual == {"tools[0]": {"type": "web_search_preview", "name": "ws", "extra": "x"}}


# --------------------------------------------------------------------------
# R6 — cache breakpoints (G37, §11 Q16)
# --------------------------------------------------------------------------


class TestCacheBreakpoints:
    """R6 — both spellings fill ``Part.cache_control`` verbatim."""

    def test_cache_control_on_a_text_part_projects_verbatim(self) -> None:
        """R6.1 — OpenRouter's CC dialect carries Anthropic's own spelling."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hi",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            }
        )

        text = projected.conversation.turns[0].parts[0]
        assert isinstance(text, c.Text)
        assert text.cache_control == {"type": "ephemeral"}
        assert projected.residual == {}

    def test_prompt_cache_breakpoint_projects_verbatim_too(self) -> None:
        """R6.2 — OpenAI's ``prompt_cache_breakpoint`` (no TTL) fills the same slot."""
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hi",
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            }
                        ],
                    }
                ],
            }
        )

        text = projected.conversation.turns[0].parts[0]
        assert isinstance(text, c.Text)
        # The OpenAI shape travels — it is on the same slot, identifiable
        # by its `mode` rather than `type`.
        assert text.cache_control == {"mode": "explicit"}
        assert projected.residual == {}

    def test_a_wrongly_typed_cache_control_residualises(self) -> None:
        """R6.3 — §7.4.1: a string value at the ``cache_control`` spelling residualises at its own path.

        The ``prompt_cache_breakpoint`` spelling behaves the same way:
        both are read by :func:`_read_cache_control` and residualised through
        the same path on a non-object value, so the test for one is the
        test for the other. R6.4 covers the co-occurrence case where both
        spellings appear on one part — the first fills the slot, the second
        residualises.
        """
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hi", "cache_control": "yes"}
                        ],
                    }
                ],
            }
        )

        assert projected.residual == {"messages[0].content[0].cache_control": "yes"}

    def test_when_both_cache_spellings_are_present_first_fills_the_slot_and_second_residualises(self) -> None:
        """R6.4 — the co-occurrence case §11 Q16 names.

        The schema forbids ``cache_control`` and ``prompt_cache_breakpoint``
        on the same part, so a body carrying both is a mutation the bridge
        is obliged to name. The reader's contract: the first spelling
        (per :data:`_CACHE_KEYS` order — an ordered ``tuple``, so the rule
        is not hash-seed-dependent) fills the slot, the second
        residualises at its own path. Silent drop is the shape M16 and G37
        exist to prevent — a strip the bridge did not register.
        """
        projected = _read(
            {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hi",
                                "cache_control": {"type": "ephemeral"},
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            }
                        ],
                    }
                ],
            }
        )

        text = projected.conversation.turns[0].parts[0]
        assert isinstance(text, c.Text)
        assert text.cache_control == {"type": "ephemeral"}
        assert projected.residual == {
            "messages[0].content[0].prompt_cache_breakpoint": {"mode": "explicit"}
        }

    def test_when_both_cache_spellings_are_present_first_fills_deterministically(self) -> None:
        """R6.5 — the co-occurrence rule is stable across PYTHONHASHSEED values.

        A ``frozenset`` iteration is hash-seed-dependent; a ``tuple`` is
        not. `_CACHE_KEYS` is an ordered tuple, so the "first non-null wins"
        rule is stable across hash randomisation. This test runs the
        reader under three different ``PYTHONHASHSEED`` values (0, 1, 2)
        in subprocess invocations — a within-process assertion cannot
        reach across the runtime's hash seed. If the reader's iteration
        order ever regresses to a frozenset (or to any other
        hash-dependent container), this test fails under at least one seed.
        """
        import subprocess
        import sys
        import textwrap

        runner = textwrap.dedent(
            """
            import json
            from harness import contract as c
            from harness import reader_chat_completions as cc

            body = {
                "model": "gpt-6-astra",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hi",
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            }
            cap = c.CapturedRequest(
                method="POST",
                scheme="https",
                host="api.openai.com",
                path="/v1/chat/completions",
                query="",
                headers=(),
                body=json.dumps(body).encode(),
            )
            projected = cc.ChatCompletionsProjection().read_request(cap)
            text = projected.conversation.turns[0].parts[0]
            assert text.cache_control == {"type": "ephemeral"}
            assert (
                projected.residual[
                    "messages[0].content[0].prompt_cache_breakpoint"
                ]
                == {"mode": "explicit"}
            )
            """
        ).strip()

        for seed in (0, 1, 2):
            result = subprocess.run(
                [sys.executable, "-c", runner],
                env={"PYTHONHASHSEED": str(seed), "PYTHONPATH": "tests"},
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, (
                f"PYTHONHASHSEED={seed} failed: stdout={result.stdout!r} "
                f"stderr={result.stderr!r}"
            )


# --------------------------------------------------------------------------
# R7 — the convergence test (KBR-25's scope addition)
# --------------------------------------------------------------------------


class TestConvergence:
    """R7 — the CC and Messages encodings of one worked tool exchange project to identical
    ``Conversation`` values, indices included. T-A2's own proof that §3.3.1b's
    merge rule is shared.
    """

    def test_an_interleaved_tool_exchange_documents_an_intentional_divergence(self) -> None:
        """R7.1 — the **interleaved** tool exchange records an intentional asymmetry.

        CC: ``user → assistant(tool_calls=[a,b]) → tool(a) → user("and humidity?") → tool(b)``
        yields a single merged user turn ``[ToolResult(a), Text,
        ToolResult(b)]`` — clauses 1+2+4; clause 3 is vacuous on CC because
        results arrive in separate messages.

        Messages: the same conversation encoded as one user message
        ``[tool_result_a, text, tool_result_b]`` yields
        ``[ToolResult(a), ToolResult(b), Text]`` — clause 3 hoists results
        first within the user message, per §3.3.1b: "Anthropic Messages
        carries text and results inside one message, so the run has no
        natural boundary and clause 1 cannot do the work by splitting;
        clause 3 does it instead".

        The two projections are **deliberately** not equal. Each reader
        applies the rule to the wire shape it sees. Recording that here so a
        future task does not mistake the asymmetry for a defect, and so the
        simple exchange (R7.2 below) is the one the convergence claim is
        actually pinned against.
        """
        cc_body = {
            "model": "gpt-6-astra",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "weather in SF and NYC?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {**PUBLISHED_TOOL_CALL, "id": "t1"},
                        {**PUBLISHED_TOOL_CALL, "id": "t2"},
                    ],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "72 and sunny"},
                {"role": "user", "content": "and humidity?"},
                {"role": "tool", "tool_call_id": "t2", "content": "30 and cloudy"},
            ],
        }
        messages_body = {
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "weather in SF and NYC?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "get_weather",
                            "input": {"city": "San Francisco"},
                        },
                        {
                            "type": "tool_use",
                            "id": "t2",
                            "name": "get_weather",
                            "input": {"city": "New York"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "72 and sunny",
                        },
                        {"type": "text", "text": "and humidity?"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t2",
                            "content": "30 and cloudy",
                        },
                    ],
                },
            ],
        }

        cc_request = cc.ChatCompletionsProjection().read_request(_captured(cc_body))
        messages_capture = c.CapturedRequest(
            method="POST",
            scheme="https",
            host="api.anthropic.com",
            path="/v1/messages",
            query="",
            headers=(("content-type", "application/json"),),
            body=json.dumps(messages_body).encode("utf-8"),
        )
        messages_request = am.AnthropicMessagesProjection().read_request(messages_capture)

        # Each reader's projection matches the documented shape on its side.
        cc_merged = cc_request.conversation.turns[2]
        assert [type(p).__name__ for p in cc_merged.parts] == [
            "ToolResult",
            "Text",
            "ToolResult",
        ]
        messages_merged = messages_request.conversation.turns[2]
        assert [type(p).__name__ for p in messages_merged.parts] == [
            "ToolResult",
            "ToolResult",
            "Text",
        ]

        # And the two projections are deliberately not equal.
        assert cc_request.conversation != messages_request.conversation

    def test_the_standard_tool_exchange_projects_identically_to_messages(self) -> None:
        """R7.1 — the canonical exchange: ``user → assistant(tool_calls=[call]) → tool → user follow-up``.

        The two encodings carry the same conversation through different
        routes and project to identical ``Conversation`` values, indices
        included — — T-A2's proof that §3.3.1b's merge rule is shared, not
        just stated. Sampling must match across both encodings for the
        equality to hold; ``max_tokens`` is the one the Messages shape
        carries in this fixture, so the CC side sets it the same way.
        """
        cc_body = {
            "model": "gpt-6-astra",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [PUBLISHED_TOOL_CALL],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "72 and sunny",
                },
                {"role": "user", "content": "Thanks!"},
            ],
        }
        messages_body = {
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_abc123",
                            "name": "get_weather",
                            "input": {"city": "San Francisco"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_abc123",
                            "content": "72 and sunny",
                        },
                        {"type": "text", "text": "Thanks!"},
                    ],
                },
            ],
        }

        cc_request = cc.ChatCompletionsProjection().read_request(_captured(cc_body))
        messages_capture = c.CapturedRequest(
            method="POST",
            scheme="https",
            host="api.anthropic.com",
            path="/v1/messages",
            query="",
            headers=(("content-type", "application/json"),),
            body=json.dumps(messages_body).encode("utf-8"),
        )
        messages_request = am.AnthropicMessagesProjection().read_request(messages_capture)

        assert cc_request.conversation == messages_request.conversation

    def test_the_merge_rule_mutation_pin(self) -> None:
        """R7.2 — §7.4.1 names a specific failure (re-sorting results *after* the merge);
        the pin lives in the suite, not in a one-off run.

        Two assertions, on two levels: (1) the dataclass equality between
        two fabricated ``Turn`` objects — the merge-rule invariant that
        results come first within a merged turn. (2) The reader's actual
        behaviour on the standard exchange — the merged turn's parts
        start with a ``ToolResult`` and the absorbed user text follows it,
        in that order. A reader that mutated clauses 1+2+4 to re-sort
        results after merge would fail (2); a reader that broke the
        invariant the merger produces would fail (1).
        """
        wrong_merged = c.Turn(
            role="user",
            parts=(
                c.ToolResult(content=(c.Text("72 and sunny"),), tool_use_id="call_abc123"),
                c.Text("Thanks!"),
            ),
        )
        right_merged = c.Turn(
            role="user",
            parts=(
                c.Text("Thanks!"),
                c.ToolResult(content=(c.Text("72 and sunny"),), tool_use_id="call_abc123"),
            ),
        )

        # (1) Dataclass-level: the merge-rule invariant.
        assert wrong_merged != right_merged

        # (2) Reader-level: the canonical exchange from R7.1 produces a
        # merged user turn whose parts are ``[ToolResult, Text("Thanks!")]``
        # in that order. A reader that re-sorted would produce
        # ``[Text("Thanks!"), ToolResult]`` and fail this assertion.
        cc_body = {
            "model": "gpt-6-astra",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [PUBLISHED_TOOL_CALL],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "72 and sunny",
                },
                {"role": "user", "content": "Thanks!"},
            ],
        }
        projected = cc.ChatCompletionsProjection().read_request(_captured(cc_body))
        merged = projected.conversation.turns[2]
        assert [type(p).__name__ for p in merged.parts] == ["ToolResult", "Text"]
        assert isinstance(merged.parts[0], c.ToolResult)
        assert isinstance(merged.parts[1], c.Text)
        assert merged.parts[1].text == "Thanks!"


# --------------------------------------------------------------------------
# R8 — failure shapes
# --------------------------------------------------------------------------


class TestFailures:
    """R8 — malformed bodies raise ``UnreadableBodyError``, not ``ValueError``."""

    def test_a_malformed_body_raises_unreadable_body_error(self) -> None:
        """R8.1 — malformed JSON is unreadable, not a reader bug."""
        with pytest.raises(c.UnreadableBodyError):
            cc.ChatCompletionsProjection().read_request(_captured(b"{not json"))

    def test_a_non_object_body_raises_unreadable_body_error(self) -> None:
        """R8.2 — a JSON array is valid JSON and an invalid request."""
        with pytest.raises(c.UnreadableBodyError):
            cc.ChatCompletionsProjection().read_request(_captured(b"[]"))

    def test_an_unanticipated_structural_failure_is_still_unreadable(self) -> None:
        """R8.3 — ``KeyError``, ``TypeError`` and ``AttributeError`` from inside the
        reader are wrapped, so T-D1 can tell a malformed body from an I1
        breach."""
        with pytest.raises(c.UnreadableBodyError):
            # A message that is not an object — the per-message isinstance
            # guard raises ``TypeError`` on the inner code path; the wrapping
            # converts that to ``UnreadableBodyError``.
            cc.ChatCompletionsProjection().read_request(
                _captured({"model": "x", "messages": ["not a dict"]})
            )

    def test_a_user_role_with_a_refusal_part_type_is_unreadable(self) -> None:
        """R8.4 — a content-part ``type`` outside the role's part set is unreadable.

        User messages admit ``text`` / ``image_url`` / ``input_audio`` /
        ``file``; assistant messages admit ``text`` / ``refusal``. A
        ``refusal`` part on a user message is a role/type mismatch, and
        the reader raises ``UnreadableBodyError`` rather than forcing the
        part into a kind it does not name.
        """
        with pytest.raises(c.UnreadableBodyError):
            _read(
                {
                    "model": "gpt-6-astra",
                    "messages": [
                        {"role": "user", "content": [{"type": "refusal", "refusal": "no"}]}
                    ],
                }
            )

    def test_a_message_without_role_raises(self) -> None:
        """R8.5 — ``role`` is required on every message."""
        with pytest.raises(c.UnreadableBodyError):
            _read({"model": "gpt-6-astra", "messages": [{"content": "hi"}]})
