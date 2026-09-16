"""L1 unit tests for KBR-137 — the OpenCode Go OpenAI Responses route.

Four contracts, one file, so the wire-shape declaration, the body builder,
the response translator and the stream converter fail together when one of
them drifts:

* :meth:`OpenCodeGoAdapter._cc_to_responses` — the Chat Completions →
  OpenAI Responses body builder.  Register rows P36–P42 name its mutations.
* :func:`kitty.providers.opencode._responses_to_cc` — the non-streaming
  reverse translator.
* :class:`OpenCodeGoResponsesCCStreamConverter` — the stateful SSE → CC
  chunk converter.  Mirrors :class:`~kitty.providers.anthropic.AnthropicCCStreamConverter`
  in shape and in test discipline: every event the Responses spec defines
  has a pinned outcome here, so an unhandled event cannot silently pass.
* ``kitty.bridge.server._repair_thinking_roundtrip`` — widened from a
  boolean to :class:`~kitty.providers.base.WireShape`; the four values each
  have a pinned outcome so the contract cannot quietly shrink.

The live Responses endpoint needs a paid OpenCode Go key, so these tests
rest on the published OpenAI Responses API specification (v2.3.0 master,
retrieved 2026-09-16 from ``openai/openai-openapi`` by curl from
raw.githubusercontent) rather than on traffic.  A key-holder can land a
live probe and amend the fixtures without changing a single assertion
here — the assertions pin kitty's contract, not the wire's.
"""

from __future__ import annotations

import json

import pytest

from kitty.bridge.server import _repair_thinking_roundtrip
from kitty.providers.base import WireShape
from kitty.providers.opencode import (
    _RESPONSES_MODELS,
    OpenCodeGoAdapter,
    OpenCodeGoResponsesCCStreamConverter,
    _responses_to_cc,
)

pytestmark = pytest.mark.l1

# ── Fixtures ───────────────────────────────────────────────────────────────

_MODEL = "grok-4.6"

_CC_BASE = {
    "model": _MODEL,
    "messages": [
        {"role": "system", "content": "You are a reviewer."},
        {"role": "user", "content": "Review this diff"},
        {"role": "assistant", "content": "Reading the file."},
    ],
}


def _adapter() -> OpenCodeGoAdapter:
    return OpenCodeGoAdapter()


# ── _cc_to_responses: body shape ──────────────────────────────────────────


class TestTranslateToResponsesBody:
    """KBR-137 — Chat Completions → OpenAI Responses body builder.

    Every parameter the Responses spec accepts has a carry assertion; every
    one it rejects has a drop assertion.  The spec-derived allow/drop split
    is pinned here rather than derived from the code, so a coordinated
    edit to the builder and the constant fails the assertion before it
    ships a wrong wire.
    """

    def setup_method(self):
        self.adapter = _adapter()

    def test_minimal_body_has_model_input_and_no_store(self):
        body = self.adapter._cc_to_responses({"model": _MODEL, "messages": [], "stream": False})

        assert body["model"] == _MODEL
        assert body["input"] == []
        # P17's ``store: False`` is Codex-specific and must not leak here —
        # OpenCode Go accepts the spec default.
        assert "store" not in body

    def test_stream_honoured_true(self):
        body = self.adapter._cc_to_responses({"model": _MODEL, "messages": [], "stream": True})
        assert body["stream"] is True

    def test_stream_honoured_false(self):
        body = self.adapter._cc_to_responses({"model": _MODEL, "messages": [], "stream": False})
        assert body["stream"] is False

    def test_stream_absent_omits_the_field(self):
        """Absent ``stream`` leaves the field absent — the spec default governs."""
        body = self.adapter._cc_to_responses({"model": _MODEL, "messages": []})
        assert "stream" not in body

    def test_system_message_becomes_instructions(self):
        body = self.adapter._cc_to_responses(
            {
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert body["instructions"] == "Be terse."
        # The system turn does not appear in `input`.
        assert all(item.get("role") != "system" for item in body["input"])

    def test_multiple_system_messages_join_with_blank_line(self):
        body = self.adapter._cc_to_responses(
            {
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": "First."},
                    {"role": "system", "content": "Second."},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert body["instructions"] == "First.\n\nSecond."

    def test_user_message_becomes_input_text_item(self):
        body = self.adapter._cc_to_responses({"model": _MODEL, "messages": [{"role": "user", "content": "hello"}]})
        assert body["input"] == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ]

    def test_assistant_message_becomes_output_text_item(self):
        body = self.adapter._cc_to_responses({"model": _MODEL, "messages": [{"role": "assistant", "content": "done"}]})
        assert body["input"] == [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}
        ]

    def test_tool_message_becomes_function_call_output(self):
        body = self.adapter._cc_to_responses(
            {"model": _MODEL, "messages": [{"role": "tool", "tool_call_id": "call_9", "content": "42"}]}
        )
        assert body["input"] == [{"type": "function_call_output", "call_id": "call_9", "output": "42"}]

    def test_tool_message_list_content_flattens_to_text(self):
        """Round-5 review — a list-form tool result must flatten to its text
        parts.  ``str(content)`` on a list produces the Python repr, and a
        tool result is agent-read content: shipping the repr would put
        ``[{'type': 'text', 'text': 'I read it'}]`` into the model's context
        as though that string were the tool's output.
        """
        body = self.adapter._cc_to_responses(
            {
                "model": _MODEL,
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_9",
                        "content": [{"type": "text", "text": "I read it"}, {"type": "text", "text": "twice"}],
                    }
                ],
            }
        )
        assert body["input"] == [
            {"type": "function_call_output", "call_id": "call_9", "output": "I read it\ntwice"}
        ]

    def test_assistant_message_list_content_flattens_to_text(self):
        """Round-5 review — a list-form assistant content must flatten to its
        text parts, the same discipline the user branch already applies
        (KBR-222).  ``str(content)`` would ship a Python repr as the model's
        own words.
        """
        body = self.adapter._cc_to_responses(
            {
                "model": _MODEL,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Reading"},
                            {"type": "text", "text": "now"},
                        ],
                    }
                ],
            }
        )
        assert body["input"] == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Reading\nnow"}],
            }
        ]

    def test_assistant_tool_calls_become_function_call_items(self):
        body = self.adapter._cc_to_responses(
            {
                "model": _MODEL,
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
                        ],
                    }
                ],
            }
        )
        assert body["input"] == [{"type": "function_call", "call_id": "call_1", "name": "t", "arguments": "{}"}]

    # ── Sampling parameters ───────────────────────────────────────────

    @pytest.mark.parametrize("key", ["temperature", "top_p", "top_logprobs", "parallel_tool_calls"])
    def test_responses_spec_fields_carried_through(self, key):
        body = self.adapter._cc_to_responses({**_CC_BASE, key: 7})
        assert body[key] == 7

    @pytest.mark.parametrize(
        "key",
        ["frequency_penalty", "presence_penalty", "seed", "logit_bias", "n", "stop", "logprobs", "stream_options"],
    )
    def test_cc_only_fields_dropped(self, key):
        """Register row P37 — the eight CC parameters absent from the spec."""
        body = self.adapter._cc_to_responses({**_CC_BASE, key: 1 if key != "logit_bias" else {"t": [1, 2]}})
        assert key not in body

    def test_max_tokens_renamed_to_max_output_tokens(self):
        body = self.adapter._cc_to_responses({**_CC_BASE, "max_tokens": 512})
        assert body["max_output_tokens"] == 512
        assert "max_tokens" not in body

    def test_max_completion_tokens_renamed_to_max_output_tokens(self):
        body = self.adapter._cc_to_responses({**_CC_BASE, "max_completion_tokens": 512})
        assert body["max_output_tokens"] == 512

    def test_max_output_tokens_wins_when_multiple_spellings_present(self):
        """P38 precedence — Responses spelling first, then the two CC spellings."""
        body = self.adapter._cc_to_responses(
            {**_CC_BASE, "max_tokens": 1, "max_completion_tokens": 2, "max_output_tokens": 3}
        )
        assert body["max_output_tokens"] == 3

    def test_max_completion_tokens_beats_max_tokens(self):
        body = self.adapter._cc_to_responses({**_CC_BASE, "max_tokens": 1, "max_completion_tokens": 2})
        assert body["max_output_tokens"] == 2

    # ── Tools ─────────────────────────────────────────────────────────

    def test_tools_envelope_unwrapped(self):
        """P36 — CC's function envelope becomes Responses' flat form."""
        body = self.adapter._cc_to_responses(
            {
                **_CC_BASE,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
        assert body["tools"] == [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def test_strict_is_carried_not_stripped(self):
        """P36 — ``strict`` is carried (P15's Codex strip does not apply here)."""
        body = self.adapter._cc_to_responses(
            {**_CC_BASE, "tools": [{"type": "function", "function": {"name": "t", "strict": True, "parameters": {}}}]}
        )
        assert body["tools"][0]["strict"] is True

    def test_no_tools_omits_the_field(self):
        body = self.adapter._cc_to_responses({**_CC_BASE})
        assert "tools" not in body

    # ── tool_choice ───────────────────────────────────────────────────

    @pytest.mark.parametrize("mode", ["none", "auto", "required"])
    def test_tool_choice_string_modes_pass_through(self, mode):
        body = self.adapter._cc_to_responses({**_CC_BASE, "tool_choice": mode})
        assert body["tool_choice"] == mode

    def test_tool_choice_named_function_unwrapped(self):
        """P36 — the CC envelope unwraps to the flat form."""
        tool_choice = {"type": "function", "function": {"name": "t"}}
        body = self.adapter._cc_to_responses({**_CC_BASE, "tool_choice": tool_choice})
        assert body["tool_choice"] == {"type": "function", "name": "t"}

    def test_tool_choice_absent_omits_the_field(self):
        body = self.adapter._cc_to_responses({**_CC_BASE})
        assert "tool_choice" not in body

    # ── response_format ───────────────────────────────────────────────

    def test_response_format_text_passes_through_under_text_format(self):
        body = self.adapter._cc_to_responses({**_CC_BASE, "response_format": {"type": "text"}})
        assert body["text"] == {"format": {"type": "text"}}

    def test_response_format_json_object_passes_through(self):
        body = self.adapter._cc_to_responses({**_CC_BASE, "response_format": {"type": "json_object"}})
        assert body["text"] == {"format": {"type": "json_object"}}

    def test_response_format_json_schema_hoists_one_level(self):
        """P36 — json_schema's nesting lifts one level."""
        schema = {"type": "object", "properties": {}}
        body = self.adapter._cc_to_responses(
            {
                **_CC_BASE,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "Answer", "schema": schema, "strict": True},
                },
            }
        )
        assert body["text"] == {"format": {"type": "json_schema", "name": "Answer", "schema": schema, "strict": True}}

    def test_response_format_absent_omits_text(self):
        body = self.adapter._cc_to_responses({**_CC_BASE})
        assert "text" not in body

    # ── reasoning ─────────────────────────────────────────────────────

    def test_reasoning_effort_injected_when_present(self):
        """Register row P42 — the agent signal passthrough in the target's spelling."""
        body = self.adapter._cc_to_responses({**_CC_BASE, "_reasoning_effort": "high"})
        assert body["reasoning"] == {"effort": "high"}

    def test_reasoning_effort_none_not_injected(self):
        body = self.adapter._cc_to_responses({**_CC_BASE, "_reasoning_effort": "none"})
        assert "reasoning" not in body

    def test_reasoning_effort_absent_omits_reasoning(self):
        body = self.adapter._cc_to_responses({**_CC_BASE})
        assert "reasoning" not in body


# ── _responses_to_cc: non-streaming reverse ────────────────────────────────


class TestResponsesToCC:
    """KBR-137 — Responses JSON → Chat Completions response translator."""

    def test_text_only_response_translated(self):
        cc = _responses_to_cc(
            {
                "object": "response",
                "model": _MODEL,
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }
        )
        assert cc["object"] == "chat.completion"
        assert cc["model"] == _MODEL
        assert cc["choices"][0]["message"]["role"] == "assistant"
        assert cc["choices"][0]["message"]["content"] == "Hi"
        assert cc["choices"][0]["finish_reason"] == "stop"
        assert cc["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        assert cc["id"].startswith("chatcmpl-")
        assert isinstance(cc["created"], int)

    def test_tool_only_response_translated(self):
        cc = _responses_to_cc(
            {
                "object": "response",
                "model": _MODEL,
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "{}",
                    }
                ],
            }
        )
        assert cc["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = cc["choices"][0]["message"]["tool_calls"]
        assert tool_calls == [
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        ]

    def test_mixed_response_allocates_indices_in_document_order(self):
        cc = _responses_to_cc(
            {
                "object": "response",
                "model": _MODEL,
                "status": "completed",
                "output": [
                    {"type": "function_call", "call_id": "a", "name": "f1", "arguments": "1"},
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "mid"}]},
                    {"type": "function_call", "call_id": "b", "name": "f2", "arguments": "2"},
                ],
            }
        )
        assert cc["choices"][0]["message"]["content"] == "mid"
        assert cc["choices"][0]["message"]["tool_calls"][0]["index"] == 0
        assert cc["choices"][0]["message"]["tool_calls"][1]["index"] == 1

    def test_incomplete_status_maps_to_length(self):
        cc = _responses_to_cc(
            {"object": "response", "model": _MODEL, "status": "incomplete", "output": []}
        )
        assert cc["choices"][0]["finish_reason"] == "length"

    def test_empty_output_sets_content_none(self):
        """A truly empty reply — no text, no tool calls — is preserved as None.

        The bridge's empty-response fallback keys on that None; collapsing it
        to ``""`` would hide the case from the fallback chain.
        """
        cc = _responses_to_cc({"object": "response", "model": _MODEL, "status": "completed", "output": []})
        assert cc["choices"][0]["message"]["content"] is None
        assert "tool_calls" not in cc["choices"][0]["message"]

    def test_absent_usage_yields_empty_usage(self):
        cc = _responses_to_cc({"object": "response", "model": _MODEL, "status": "completed", "output": []})
        assert cc["usage"] == {}

    def test_ids_are_unique_per_call(self):
        first = _responses_to_cc({"object": "response", "model": _MODEL, "status": "completed", "output": []})
        second = _responses_to_cc({"object": "response", "model": _MODEL, "status": "completed", "output": []})
        assert first["id"] != second["id"]

    def test_function_call_without_call_id_gets_a_synthesised_one(self):
        """A missing call_id would otherwise collide the tool call's id with
        the next one — Claude Code keys tool-call results by id."""
        cc = _responses_to_cc(
            {
                "object": "response",
                "model": _MODEL,
                "status": "completed",
                "output": [
                    {"type": "function_call", "name": "f", "arguments": "{}"},
                    {"type": "function_call", "name": "g", "arguments": "{}"},
                ],
            }
        )
        ids = [tc["id"] for tc in cc["choices"][0]["message"]["tool_calls"]]
        assert ids[0] != ids[1]


# ── OpenCodeGoResponsesCCStreamConverter ───────────────────────────────────


def _feed(converter: OpenCodeGoResponsesCCStreamConverter, event: dict) -> list[dict]:
    """Feed one event through the converter and return the parsed chunks."""
    raw = f"data: {json.dumps(event)}\n\n".encode()
    lines = converter.feed(raw)
    out = []
    for line in lines:
        if line.startswith(b"data: ") and line != b"data: [DONE]\n\n":
            out.append(json.loads(line[6:]))
    return out


def _feed_done(converter: OpenCodeGoResponsesCCStreamConverter, event: dict) -> list[bytes]:
    """Feed one event and return the raw lines (for the [DONE] sentinel)."""
    return converter.feed(f"data: {json.dumps(event)}\n\n".encode())


class TestOpenCodeGoResponsesCCStreamConverter:
    """KBR-137 — every Responses SSE event family has a pinned outcome."""

    def test_response_created_emits_assistant_role_chunk(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        chunks = _feed(converter, {"type": "response.created", "response": {"model": _MODEL}})
        assert len(chunks) == 1
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[0]["model"] == _MODEL

    def test_one_chunk_id_across_the_stream(self):
        """Chunks correlate by id; a fresh id per chunk would orphan every one.

        The id mirrors the Anthropic converter's shape: a ``chatcmpl-`` prefix
        on a truncated hex UUID, so a maintained prefix contract is asserted
        rather than a full UUID parse.
        """
        converter = OpenCodeGoResponsesCCStreamConverter()
        first = _feed(converter, {"type": "response.created", "response": {"model": _MODEL}})
        second = _feed(converter, {"type": "response.output_text.delta", "delta": "hi"})
        assert first[0]["id"] == second[0]["id"]
        # 12 hex chars, per the Anthropic converter's precedent.
        suffix = first[0]["id"].removeprefix("chatcmpl-")
        assert len(suffix) == 12
        int(suffix, 16)  # parses as hex

    def test_text_delta_emits_content_delta(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        chunks = _feed(converter, {"type": "response.output_text.delta", "delta": "hello"})
        assert chunks[0]["choices"][0]["delta"] == {"content": "hello"}

    def test_function_call_added_allocates_index_and_carries_name(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        chunks = _feed(
            converter,
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_a", "name": "f", "arguments": ""},
            },
        )
        assert chunks[0]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "f", "arguments": ""}}
        ]

    def test_two_function_calls_allocate_distinct_indices(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        _feed(
            converter,
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "a", "name": "f1", "arguments": ""},
            },
        )
        chunks = _feed(
            converter,
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_2", "type": "function_call", "call_id": "b", "name": "f2", "arguments": ""},
            },
        )
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1

    def test_arguments_delta_lands_under_the_right_index(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        _feed(
            converter,
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "a", "name": "f1", "arguments": ""},
            },
        )
        chunks = _feed(
            converter,
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"x":'},
        )
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0] == {
            "index": 0,
            "function": {"arguments": '{"x":'},
        }

    def test_arguments_done_after_deltas_does_not_duplicate_arguments(self):
        """KBR-137 review finding 2 — the spec sequence is deltas × N → .done →
        output_item.done, and the .done event carries the FULL arguments so a
        client that lost deltas can recover.  A client that received every
        delta must NOT also receive .done's full string, or the CC client
        concatenates both and the model sees its tool arguments doubled.
        """
        converter = OpenCodeGoResponsesCCStreamConverter()
        _feed(
            converter,
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "a", "name": "f", "arguments": ""},
            },
        )
        first = _feed(
            converter,
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"k":'},
        )
        second = _feed(
            converter,
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "1}"},
        )
        # Both deltas must cross; the client concatenates them into one body.
        assert first[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"k":'
        assert second[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "1}"

        # .done is a no-op: the full string is exactly what the deltas summed to.
        done_event = (
            b'data: {"type": "response.function_call_arguments.done", '
            b'"item_id": "fc_1", "arguments": "{\\"k\\":1}"}\n\n'
        )
        assert converter.feed(done_event) == []

    def test_output_text_done_with_no_prior_deltas_emits_carried_text(self):
        """KBR-137 review finding — ``response.output_text.done`` is the recovery
        channel for a backend that streams no ``response.output_text.delta``
        events and closes the content part with the full text.  When that
        happens, the converter must emit the carried text as a CC chunk — the
        CC client's view of the model is "empty" without it.
        """
        converter = OpenCodeGoResponsesCCStreamConverter()
        # created → done with text but no deltas.  The carried text must reach
        # the client.
        _feed(converter, {"type": "response.created", "response": {"model": "m"}})
        chunks = _feed(
            converter,
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "full reply in done",
            },
        )
        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["delta"]["content"] == "full reply in done"

    def test_output_text_done_after_deltas_does_not_duplicate_text(self):
        """Symmetric to the arguments case: a done event after deltas is a no-op
        for text content too.  Otherwise the client concatenates the carried
        full text with the deltas, doubling the model's reply.
        """
        converter = OpenCodeGoResponsesCCStreamConverter()
        _feed(converter, {"type": "response.created", "response": {"model": "m"}})
        _feed(converter, {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Check"})
        _feed(converter, {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "ing"})
        # Done after deltas is a no-op.
        chunks = _feed(
            converter,
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "Checking",
            },
        )
        assert chunks == []

    def test_arguments_delta_without_an_added_item_is_dropped(self):
        """An orphan delta has nowhere to land — dropping beats a crash or a
        fabricated index that would splice two calls' arguments together."""
        converter = OpenCodeGoResponsesCCStreamConverter()
        orphan = b'data: {"type": "response.function_call_arguments.delta", "item_id": "x", "delta": "1"}\n\n'
        assert converter.feed(orphan) == []

    def test_arguments_done_carries_whole_arguments_when_no_deltas_arrived(self):
        """KBR-137 review finding 8 — some backends ship arguments only on .done.

        Mirrors ``AnthropicCCStreamConverter._arguments_complete``: a done
        event with no preceding deltas carries the full arguments in one
        chunk; subsequent deltas are suppressed.
        """
        converter = OpenCodeGoResponsesCCStreamConverter()
        _feed(
            converter,
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "a", "name": "f", "arguments": ""},
            },
        )
        chunks = _feed(
            converter,
            {"type": "response.function_call_arguments.done", "item_id": "fc_1", "arguments": '{"whole": true}'},
        )
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"whole": true}'

        # The delta that arrives after a completed .done is suppressed —
        # otherwise the same fragment would land twice.
        assert converter.feed(
            b'data: {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "{}"}\n\n'
        ) == []

    def test_completed_emits_finish_chunk_with_usage_then_done_sentinel(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        converter.feed(b'data: {"type": "response.created", "response": {"model": "grok-4.6"}}\n\n')
        lines = _feed_done(
            converter,
            {
                "type": "response.completed",
                "response": {
                    "model": _MODEL,
                    "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
                },
            },
        )
        assert len(lines) == 2
        chunk = json.loads(lines[0][6:])
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        assert lines[1] == b"data: [DONE]\n\n"

    def test_incomplete_emits_length_finish_chunk(self):
        """KBR-137 review finding 7 — ``response.incomplete`` ends the stream
        with ``finish_reason="length"`` rather than truncating silently."""
        converter = OpenCodeGoResponsesCCStreamConverter()
        lines = _feed_done(
            converter,
            {
                "type": "response.incomplete",
                "response": {"model": _MODEL, "usage": {"input_tokens": 5, "output_tokens": 3}},
            },
        )
        chunk = json.loads(lines[0][6:])
        assert chunk["choices"][0]["finish_reason"] == "length"
        assert lines[1] == b"data: [DONE]\n\n"

    def test_incomplete_emits_usage_matching_the_completed_twin(self):
        """Round-5 review — ``response.incomplete`` is the truncated-tail twin
        of ``response.completed`` and carries usage the same way.  The finish
        chunk must carry the truncated reply's usage, or the CC client loses
        its accounting on every hit-the-ceiling reply."""
        converter = OpenCodeGoResponsesCCStreamConverter()
        converter.feed(b'data: {"type": "response.created", "response": {"model": "grok-4.6"}}\n\n')
        lines = _feed_done(
            converter,
            {
                "type": "response.incomplete",
                "response": {
                    "model": _MODEL,
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                },
            },
        )
        chunk = json.loads(lines[0][6:])
        assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        assert lines[1] == b"data: [DONE]\n\n"

    def test_failed_emits_an_error_chunk_the_detector_sees(self):
        """KBR-137 review finding 6 — ``response.failed`` must become a CC
        ``{"error": …}`` chunk, because that is the marker
        ``_is_upstream_stream_error`` keys on."""
        from kitty.bridge.server import BridgeServer

        converter = OpenCodeGoResponsesCCStreamConverter()
        failure = {"type": "response.failed", "response": {"error": {"code": "rate_limit", "message": "slow down"}}}
        raw = converter.feed(f"data: {json.dumps(failure)}\n\n".encode())
        chunk = json.loads(raw[0][6:])
        assert BridgeServer._is_upstream_stream_error(chunk) is True
        assert chunk["error"]["code"] == "rate_limit"

    def test_content_part_events_are_dropped(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        assert converter.feed(b'data: {"type": "response.content_part.added"}\n\n') == []
        assert converter.feed(b'data: {"type": "response.content_part.done"}\n\n') == []

    def test_output_item_done_for_other_types_is_dropped(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        assert converter.feed(b'data: {"type": "response.output_item.done"}\n\n') == []

    def test_in_progress_is_dropped(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        assert converter.feed(b'data: {"type": "response.in_progress"}\n\n') == []

    def test_unknown_event_type_is_dropped(self):
        """The drop-list is the complement of the pinned happy-path set — an
        unknown event that begins carrying content must turn red here, not
        silently pass."""
        converter = OpenCodeGoResponsesCCStreamConverter()
        assert converter.feed(b'data: {"type": "response.reasoning_text.delta", "delta": "thinking"}\n\n') == []

    def test_malformed_json_passthrough_is_returned_unchanged(self):
        """A non-JSON line is forwarded as-is so the caller's per-line error
        handling sees it (same discipline as the Anthropic converter)."""
        converter = OpenCodeGoResponsesCCStreamConverter()
        raw = b"data: not json at all\n\n"
        assert converter.feed(raw) == [raw]

    def test_blank_line_returns_empty(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        assert converter.feed(b"") == []

    def test_done_sentinel_passes_through_unchanged(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        assert converter.feed(b"data: [DONE]\n\n") == [b"data: [DONE]\n\n"]


# ── Happy-path end-to-end stream ───────────────────────────────────────────


class TestResponsesStreamHappyPath:
    """The full event sequence, exactly as a text + tool-call turn arrives."""

    def _events(self) -> list[dict]:
        return [
            {"type": "response.created", "response": {"model": _MODEL}},
            {"type": "response.in_progress", "response": {"model": _MODEL}},
            {"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "role": "assistant"}},
            {"type": "response.content_part.added", "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "delta": "Checking"},
            {"type": "response.output_text.done", "text": "Checking"},
            {"type": "response.content_part.done", "part": {"type": "output_text"}},
            {"type": "response.output_item.done", "item": {"id": "m1", "type": "message"}},
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_a", "name": "f", "arguments": ""},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"k":'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "1}"},
            {"type": "response.output_item.done", "item": {"id": "fc_1", "type": "function_call"}},
            {
                "type": "response.completed",
                "response": {
                    "model": _MODEL,
                    "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                },
            },
        ]

    def test_full_stream_produces_wellformed_cc_chunks_ending_in_done(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        lines: list[bytes] = []
        for event in self._events():
            lines.extend(converter.feed(f"data: {json.dumps(event)}\n\n".encode()))
        # Every line is a sentence in the CC SSE grammar.
        assert lines, "the happy path must emit chunks"
        for line in lines[:-1]:
            assert line.startswith(b"data: ")
            payload = json.loads(line[6:])
            assert payload["object"] == "chat.completion.chunk"
        # The final line is the [DONE] sentinel.
        assert lines[-1] == b"data: [DONE]\n\n"

    def test_full_stream_content_and_tool_calls_are_correct(self):
        converter = OpenCodeGoResponsesCCStreamConverter()
        content = ""
        tool_arguments: dict[int, str] = {}
        tool_names: dict[int, str] = {}
        finish_reason = None
        for event in self._events():
            for line in converter.feed(f"data: {json.dumps(event)}\n\n".encode()):
                if line == b"data: [DONE]\n\n":
                    continue
                payload = json.loads(line[6:])
                delta = payload["choices"][0]["delta"]
                if "content" in delta:
                    content += delta["content"]
                for tc in delta.get("tool_calls", []):
                    tool_names.setdefault(tc["index"], tc["function"].get("name", ""))
                    prior_args = tool_arguments.get(tc["index"], "")
                    tool_arguments[tc["index"]] = prior_args + tc["function"].get("arguments", "")
                if payload["choices"][0]["finish_reason"] is not None:
                    finish_reason = payload["choices"][0]["finish_reason"]

        assert content == "Checking"
        assert tool_names == {0: "f"}
        assert tool_arguments == {0: '{"k":1}'}
        assert finish_reason == "stop"


class TestRouteIsExercisedAcrossAllFourResponsesModels:
    """KBR-137 — every routed model reaches the Responses builder, and the
    response direction recognises a Responses body by its shape.

    Lives at L1 (not L3) so the four-model coverage runs in the gating
    ``l1 or l2`` job — see ``.system_design/TEST_SUITE.md`` §8 on layer
    activation.  A separate L3 subsystem round-trip can be added under T-K6
    once that layer activates; today the route contract is fully pinned
    by these assertions against the in-process adapter.
    """

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    @pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
    def test_cc_to_responses_returns_a_well_formed_body(self, model):
        body = self.adapter._cc_to_responses(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            }
        )
        assert body["model"] == model
        assert isinstance(body["input"], list)
        assert body["tools"] == [{"type": "function", "name": "f", "parameters": {}}]

    @pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
    def test_translate_from_upstream_uses_the_responses_route(self, model):
        responses_json = {
            "object": "response",
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
        }
        cc = self.adapter.translate_from_upstream(responses_json)
        assert cc["model"] == model
        assert cc["choices"][0]["message"]["content"] == "ok"
        assert cc["choices"][0]["finish_reason"] == "stop"


# ── _repair_thinking_roundtrip over WireShape ──────────────────────────────


class TestRepairThinkingRoundtripWireShape:
    """KBR-137 — the repair now takes a WireShape; the four values are pinned.

    The pre-KBR-137 boolean had exactly two behaviours; the enum must keep
    those two and must be structurally inert on the two new values, so a
    Responses backend cannot be flagged into a permanent repair loop.
    """

    def test_messages_shape_writes_the_anthropic_carrier(self):
        body = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert _repair_thinking_roundtrip(body, wire_shape=WireShape.MESSAGES) is True
        assert body["messages"][0]["content"][0]["type"] == "thinking"

    def test_chat_completions_shape_writes_reasoning_content(self):
        body = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert _repair_thinking_roundtrip(body, wire_shape=WireShape.CHAT_COMPLETIONS) is True
        assert "reasoning_content" in body["messages"][0]

    def test_responses_shape_is_inert(self):
        body = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert _repair_thinking_roundtrip(body, wire_shape=WireShape.RESPONSES) is False
        assert body["messages"][0] == {"role": "assistant", "content": "hi"}

    def test_other_shape_is_inert(self):
        body = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert _repair_thinking_roundtrip(body, wire_shape=WireShape.OTHER) is False
        assert body["messages"][0] == {"role": "assistant", "content": "hi"}

    def test_a_body_without_messages_is_inert_for_every_shape(self):
        for shape in WireShape:
            body = {"model": "m"}
            assert _repair_thinking_roundtrip(body, wire_shape=shape) is False
