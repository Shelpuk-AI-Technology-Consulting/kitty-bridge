"""Tests for bridge/responses/translator.py — Responses API <-> Chat Completions translation."""

import json
import uuid

import pytest

from kitty.bridge.responses.translator import ResponsesTranslator


def _v4() -> str:
    return str(uuid.uuid4())


def _parse_sse_event(raw: str) -> tuple[str, dict]:
    """Parse a raw SSE event string into (event_type, data_dict)."""
    lines = raw.strip().split("\n")
    event_type = lines[0].split(": ", 1)[1]
    data_line = lines[1].split(": ", 1)[1]
    return event_type, json.loads(data_line)


def _extract_event_types(events: list[str]) -> list[str]:
    """Extract the event types from a list of SSE event strings."""
    return [_parse_sse_event(e)[0] for e in events]


def _assert_has_type_and_seq(raw: str) -> dict:
    """Assert SSE event has 'type' and 'sequence_number'. Returns parsed data."""
    event_type, data = _parse_sse_event(raw)
    assert data["type"] == event_type, f"type mismatch for {event_type}"
    assert "sequence_number" in data
    return data


# ── translate_request ────────────────────────────────────────────────────────


class TestTranslateRequest:
    def setup_method(self):
        self.t = ResponsesTranslator()

    def test_extracts_model(self):
        req = {"model": "gpt-4o", "input": [{"role": "user", "content": "hi"}]}
        result = self.t.translate_request(req)
        assert result["model"] == "gpt-4o"

    def test_maps_input_to_messages(self):
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you?"},
            ],
        }
        result = self.t.translate_request(req)
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "hello"
        assert result["messages"][2]["content"] == "how are you?"

    def test_system_instructions_prepended(self):
        req = {
            "model": "gpt-4o",
            "instructions": "You are a helpful assistant.",
            "input": [{"role": "user", "content": "hi"}],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result["messages"][1]["role"] == "user"

    def test_max_output_tokens_to_max_tokens(self):
        req = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "hi"}],
            "max_output_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assert result["max_tokens"] == 1024

    def test_stream_flag_passed_through(self):
        req = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = self.t.translate_request(req)
        assert result["stream"] is True

    def test_tools_mapped_to_chat_completions_format(self):
        req = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "weather?"}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
        result = self.t.translate_request(req)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["required"] == ["city"]

    def test_function_call_mapped_to_tool_calls(self):
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_001",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            ],
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["tool_calls"] is not None
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "London"}'

    def test_function_call_output_mapped_to_tool_result(self):
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_001",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_001",
                    "output": "72F sunny",
                },
            ],
        }
        result = self.t.translate_request(req)
        tool_msg = result["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_001"
        assert tool_msg["content"] == "72F sunny"

    def test_input_text_content_parts_converted_to_string(self):
        """Responses API input_text parts must be converted to plain strings for CC."""
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "2+2"}]},
            ],
        }
        result = self.t.translate_request(req)
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "2+2"

    def test_multiple_input_text_parts_joined(self):
        """Multiple input_text parts are joined with newlines."""
        req = {
            "model": "gpt-4o",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello"},
                        {"type": "input_text", "text": "World"},
                    ],
                },
            ],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0]["content"] == "Hello\nWorld"

    def test_plain_string_content_passes_through(self):
        """Plain string content is unchanged."""
        req = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "hello"}],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0]["content"] == "hello"

    def test_empty_content_list_produces_empty_string(self):
        """Empty content list produces empty string."""
        req = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": []}],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0]["content"] == ""

    def test_developer_role_mapped_to_system(self):
        """Responses API 'developer' role must map to 'system' for provider compatibility."""
        req = {
            "model": "gpt-4o",
            "input": [{"role": "developer", "content": "You are an expert."}],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are an expert."

    def test_system_role_passes_through_unchanged(self):
        """'system' role should NOT be modified — only 'developer' is remapped."""
        req = {
            "model": "gpt-4o",
            "input": [{"role": "system", "content": "You are helpful."}],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_reasoning_input_item_mapped_to_reasoning_content(self):
        """Responses API 'reasoning' items must map to reasoning_content on the
        preceding assistant message for providers that require it."""
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "solve"},
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "summary": [{"type": "summary_text", "text": "Step 1: analyze..."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The answer is 42."}],
                },
            ],
        }
        result = self.t.translate_request(req)
        # Should produce user + assistant with reasoning_content
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["reasoning_content"] == "Step 1: analyze..."
        assert assistant_msgs[0]["content"] == "The answer is 42."

    def test_reasoning_item_before_function_call(self):
        """Reasoning items before function_call must attach reasoning_content
        to the function_call's assistant message."""
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "summary": [{"type": "summary_text", "text": "I need the weather tool"}],
                },
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_001",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            ],
        }
        result = self.t.translate_request(req)
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["reasoning_content"] == "I need the weather tool"
        assert len(assistant_msgs[0]["tool_calls"]) == 1

    def test_developer_role_with_instructions_merges_into_single_system(self):
        """When both 'instructions' and 'developer' role items exist, system messages are merged."""
        req = {
            "model": "gpt-4o",
            "instructions": "Base instructions.",
            "input": [
                {"role": "developer", "content": "Extra developer instructions."},
                {"role": "user", "content": "hi"},
            ],
        }
        result = self.t.translate_request(req)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert "Base instructions." in result["messages"][0]["content"]
        assert "Extra developer instructions." in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"

    def test_developer_role_with_content_parts(self):
        """developer role with input_text parts must also map to system."""
        req = {
            "model": "gpt-4o",
            "input": [{"role": "developer", "content": [{"type": "input_text", "text": "Expert mode."}]}],
        }
        result = self.t.translate_request(req)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Expert mode."

    def test_instructions_only_produces_single_system_message(self):
        """instructions alone should produce exactly one system message."""
        req = {
            "model": "gpt-4o",
            "instructions": "You are helpful.",
            "input": [{"role": "user", "content": "hi"}],
        }
        result = self.t.translate_request(req)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    def test_non_consecutive_system_messages_not_merged(self):
        """Non-consecutive system messages (separated by other roles) should NOT be merged."""
        req = {
            "model": "gpt-4o",
            "input": [
                {"role": "system", "content": "First system msg"},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "Second system msg"},
            ],
        }
        result = self.t.translate_request(req)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 2

    def test_strip_thinking_tags_from_content(self):
        """اخل thinking tags must be stripped from response content."""
        cc_response = {
            "id": "chatcmpl-123",
            "model": "MiniMax-M2.7",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<\u0627\u062e\u0644>thinking about math</\u0627\u062e\u0644>\n\nThe answer is 4.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = self.t.translate_response(cc_response)
        output_text = result["output"][0]["content"][0]["text"]
        assert "<\u0627\u062e\u0644>" not in output_text
        assert "thinking about math" not in output_text
        assert "The answer is 4." in output_text

    def test_strip_thinking_tags_entire_content(self):
        """When content is only thinking tags, fallback text is emitted."""
        cc_response = {
            "id": "chatcmpl-456",
            "model": "MiniMax-M2.7",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<\u0627\u062e\u0644>just thinking, no output</\u0627\u062e\u0644>",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = self.t.translate_response(cc_response)
        text_items = [o for o in result["output"] if o["type"] == "message"]
        assert len(text_items) == 1
        assert "retry" in text_items[0]["content"][0]["text"].lower()
        assert self.t.response_was_empty


# ── translate_response ──────────────────────────────────────────────────────


class TestTranslateResponse:
    def setup_method(self):
        self.t = ResponsesTranslator()

    def test_text_response(self):
        cc_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.t.translate_response(cc_response)
        assert result["object"] == "response"
        assert result["model"] == "gpt-4o"
        assert result["status"] == "completed"
        assert len(result["output"]) >= 1
        msg_item = result["output"][0]
        assert msg_item["type"] == "message"
        assert msg_item["role"] == "assistant"
        assert any(
            block.get("type") == "output_text" and block.get("text") == "Hello!"
            for block in msg_item.get("content", [])
        )
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_call_response(self):
        cc_response = {
            "id": "chatcmpl-456",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "London"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = self.t.translate_response(cc_response)
        fc_items = [o for o in result["output"] if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        fc = fc_items[0]
        assert fc["name"] == "get_weather"
        assert fc["arguments"] == '{"city": "London"}'
        assert fc["call_id"] == "call_abc"
        assert fc["status"] == "completed"

    def test_mixed_text_and_tool_call_response(self):
        cc_response = {
            "id": "chatcmpl-789",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "London"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = self.t.translate_response(cc_response)
        assert result["object"] == "response"
        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "message"
        assert result["output"][1]["type"] == "function_call"

    def test_response_with_reasoning_content_produces_reasoning_item(self):
        """CC responses with reasoning_content must produce a reasoning output item."""
        cc_response = {
            "id": "chatcmpl-reason",
            "model": "kimi-for-coding",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Step 1: analyze. Step 2: compute.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        result = self.t.translate_response(cc_response)
        reasoning_items = [o for o in result["output"] if o.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        summary_texts = [s["text"] for s in reasoning_items[0].get("summary", [])]
        assert "Step 1: analyze. Step 2: compute." in summary_texts
        # Text message should also be present
        msg_items = [o for o in result["output"] if o.get("type") == "message"]
        assert len(msg_items) == 1

    def test_response_with_only_reasoning_content(self):
        """CC response with only reasoning_content (no text) produces reasoning item only."""
        cc_response = {
            "id": "chatcmpl-reason-only",
            "model": "kimi-for-coding",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Deep thinking...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        result = self.t.translate_response(cc_response)
        reasoning_items = [o for o in result["output"] if o.get("type") == "reasoning"]
        assert len(reasoning_items) == 1


class TestKBR285WidenedShapes:
    """KBR-285 — refusal / list ``content`` / legacy ``function_call`` on /v1/responses."""

    def setup_method(self):
        self.t = ResponsesTranslator()

    @staticmethod
    def _function_call_items(output: list[dict]) -> list[dict]:
        return [o for o in output if o.get("type") == "function_call"]

    @staticmethod
    def _message_items(output: list[dict]) -> list[dict]:
        return [o for o in output if o.get("type") == "message"]

    # ── translate_response ──────────────────────────────────────────────

    def test_refusal_only_response_becomes_the_refusal_text(self):
        """Refusal-only CC reply renders the refusal text in an output_text part."""
        result = self.t.translate_response(
            {
                "id": "chatcmpl-refusal",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": None, "refusal": "I can't help with that."},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        msg_items = self._message_items(result["output"])
        assert len(msg_items) == 1
        assert any(
            block.get("text") == "I can't help with that."
            for block in msg_items[0].get("content", [])
        )

    def test_list_content_response_joins_text_parts(self):
        """List ``content`` joins to one output_text — no raw list on the wire."""
        result = self.t.translate_response(
            {
                "id": "chatcmpl-mm",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": [
                                {"type": "text", "text": "here is the chart"},
                                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        msg_items = self._message_items(result["output"])
        assert len(msg_items) == 1
        texts = [
            block["text"]
            for block in msg_items[0].get("content", [])
            if block.get("type") == "output_text"
        ]
        assert texts == ["here is the chart"]
        assert all(isinstance(t, str) for t in texts)

    def test_legacy_function_call_response_becomes_one_function_call_item(self):
        """Legacy dict ``function_call`` becomes one function_call output item."""
        result = self.t.translate_response(
            {
                "id": "chatcmpl-legacy",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": None,
                            "function_call": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        },
                        "finish_reason": "function_call",
                    }
                ],
            }
        )
        fc_items = self._function_call_items(result["output"])
        assert len(fc_items) == 1
        fc = fc_items[0]
        assert fc["name"] == "get_weather"
        assert fc["arguments"] == '{"city": "London"}'
        assert fc["status"] == "completed"

    def test_legacy_function_call_non_string_arguments_serialise_as_json(self):
        """A non-string ``arguments`` value ships as JSON, never a Python repr.

        The detector widening makes any truthy dict ``function_call`` count
        as content, so a reply like ``{"name": "x", "arguments": {"city": 1}}``
        is reachable; ``str()`` of the dict would put single-quoted
        non-JSON text on a wire typed ``arguments: string``.
        """
        result = self.t.translate_response(
            {
                "id": "chatcmpl-nonstr",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": None,
                            "function_call": {"name": "x", "arguments": {"city": 1}},
                        },
                        "finish_reason": "function_call",
                    }
                ],
            }
        )
        fc_items = self._function_call_items(result["output"])
        assert len(fc_items) == 1
        arguments = fc_items[0]["arguments"]
        assert isinstance(arguments, str)
        assert json.loads(arguments) == {"city": 1}
        # No Python repr leaked: single quotes are never valid JSON.
        assert "'" not in arguments

    # ── translate_stream_chunk ──────────────────────────────────────────

    @staticmethod
    def _output_text_deltas(events: list[str]) -> list[str]:
        """Extract the text payload of every output_text delta event."""
        deltas: list[str] = []
        for event in events:
            if "response.output_text.delta" not in event:
                continue
            payload = json.loads(event.split("data: ", 1)[1])
            deltas.append(payload["delta"])
        return deltas

    def test_refusal_delta_emits_the_refusal_text(self):
        """Refusal-only delta streams the refusal text on the wire."""
        chunk = {
            "choices": [
                {
                    "delta": {"content": None, "refusal": "I can't help with that."},
                    "finish_reason": None,
                }
            ],
        }
        events = self.t.translate_stream_chunk("resp_test", chunk)
        assert self._output_text_deltas(events) == ["I can't help with that."]

    def test_list_content_delta_emits_joined_text(self):
        """List ``content`` joins to a single output_text delta string."""
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": [
                            {"type": "text", "text": "here is the chart"},
                            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events = self.t.translate_stream_chunk("resp_test", chunk)
        assert self._output_text_deltas(events) == ["here is the chart"]

    def test_legacy_function_call_stream_accumulates_one_call(self):
        """Legacy dict ``function_call`` deltas open one function_call item, args accumulate."""
        open_chunk = {
            "choices": [
                {
                    "delta": {"function_call": {"name": "get_weather", "arguments": '{"city": '}},
                    "finish_reason": None,
                }
            ],
        }
        arg_chunk = {
            "choices": [
                {
                    "delta": {"function_call": {"arguments": '"London"}'}},
                    "finish_reason": None,
                }
            ],
        }
        finish_chunk = {
            "choices": [{"delta": {}, "finish_reason": "function_call"}],
        }
        self.t.translate_stream_chunk("resp_test", open_chunk)
        self.t.translate_stream_chunk("resp_test", arg_chunk)
        events = self.t.translate_stream_chunk("resp_test", finish_chunk)
        fc_items = [
            json.loads(event.split("data: ", 1)[1])["item"]
            for event in events
            if "response.output_item.done" in event
            and "function_call" in event
        ]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"
        assert json.loads(fc_items[0]["arguments"]) == {"city": "London"}

    def test_refusal_only_stream_judges_non_empty(self):
        """Refusal-only stream reaches the finish with ``response_was_empty`` False."""
        refusal = {
            "choices": [
                {
                    "delta": {"content": None, "refusal": "I can't help with that."},
                    "finish_reason": None,
                }
            ],
        }
        finish = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }
        self.t.translate_stream_chunk("resp_test", refusal)
        self.t.translate_stream_chunk("resp_test", finish)
        assert self.t.response_was_empty is False


# ── translate_stream_start ──────────────────────────────────────────────


class TestTranslateStreamStart:
    def setup_method(self):
        self.t = ResponsesTranslator()

    def test_emits_created_and_in_progress(self):
        events = self.t.translate_stream_start("resp_test", model="gpt-4o")
        types = _extract_event_types(events)
        assert types == ["response.created", "response.in_progress"]

    def test_events_have_type_and_sequence(self):
        events = self.t.translate_stream_start("resp_test", model="gpt-4o")
        for e in events:
            _assert_has_type_and_seq(e)

    def test_created_wraps_response_key(self):
        events = self.t.translate_stream_start("resp_test", model="gpt-4o")
        _, data = _parse_sse_event(events[0])
        assert "response" in data
        assert data["response"]["id"] == "resp_test"
        assert data["response"]["status"] == "in_progress"

    def test_in_progress_wraps_response_key(self):
        events = self.t.translate_stream_start("resp_test", model="gpt-4o")
        _, data = _parse_sse_event(events[1])
        assert "response" in data
        assert data["response"]["status"] == "in_progress"

    def test_sequence_numbers_increase(self):
        events = self.t.translate_stream_start("resp_test", model="gpt-4o")
        _, d0 = _parse_sse_event(events[0])
        _, d1 = _parse_sse_event(events[1])
        assert d1["sequence_number"] > d0["sequence_number"]


# ── translate_stream_chunk ─────────────────────────────────────────────────


class TestTranslateStreamChunk:
    def setup_method(self):
        self.t = ResponsesTranslator()

    def test_first_text_delta_emits_full_lifecycle_start(self):
        """First text delta should emit output_item.added + content_part.added + delta."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        events = self.t.translate_stream_chunk("resp_test", chunk)
        types = _extract_event_types(events)
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert "response.output_text.delta" in types
        # Should be in order: item added, part added, delta
        idx_item = types.index("response.output_item.added")
        idx_part = types.index("response.content_part.added")
        idx_delta = types.index("response.output_text.delta")
        assert idx_item < idx_part < idx_delta

    def test_subsequent_deltas_omit_lifecycle_start(self):
        """Second and later text deltas should only emit delta."""
        chunk1 = {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
        self.t.translate_stream_chunk("resp_test", chunk1)

        chunk2 = {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}
        events2 = self.t.translate_stream_chunk("resp_test", chunk2)
        types2 = _extract_event_types(events2)
        assert types2 == ["response.output_text.delta"]

    def test_text_delta_includes_item_id_and_indices(self):
        chunk = {"choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]}
        events = self.t.translate_stream_chunk("resp_test", chunk)
        # Find the delta event
        delta_events = [e for e in events if "response.output_text.delta" in e]
        assert len(delta_events) == 1
        _, data = _parse_sse_event(delta_events[0])
        assert "item_id" in data
        assert data["output_index"] == 0
        assert data["content_index"] == 0
        assert data["delta"] == "Hi"

    def test_finish_produces_full_trailing_lifecycle(self):
        """Finish chunk should emit: text.done → part.done → item.done → completed."""
        # First accumulate some text
        chunk1 = {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
        self.t.translate_stream_chunk("resp_test", chunk1)

        chunk2 = {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}
        self.t.translate_stream_chunk("resp_test", chunk2)

        # Finish
        chunk3 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk("resp_test", chunk3)
        types = _extract_event_types(events)

        assert "response.output_text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        assert "response.completed" in types

        # Order: text.done → part.done → item.done → completed
        idx_text_done = types.index("response.output_text.done")
        idx_part_done = types.index("response.content_part.done")
        idx_item_done = types.index("response.output_item.done")
        idx_completed = types.index("response.completed")
        assert idx_text_done < idx_part_done < idx_item_done < idx_completed

    def test_finish_includes_accumulated_text_in_completed(self):
        chunk1 = {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
        self.t.translate_stream_chunk("resp_test", chunk1)
        chunk2 = {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}
        self.t.translate_stream_chunk("resp_test", chunk2)

        chunk3 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk("resp_test", chunk3)
        # Find the completed event
        completed_events = [e for e in events if "response.completed" in e]
        assert len(completed_events) == 1
        _, data = _parse_sse_event(completed_events[0])
        assert data["response"]["status"] == "completed"
        output = data["response"]["output"]
        assert len(output) == 1
        assert output[0]["type"] == "message"
        assert any(block["type"] == "output_text" and block["text"] == "Hello world" for block in output[0]["content"])

    def test_finish_reason_length_sets_incomplete_status(self):
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk("resp_test", chunk)
        completed_events = [e for e in events if "response.completed" in e]
        _, data = _parse_sse_event(completed_events[0])
        assert data["response"]["status"] == "incomplete"

    def test_completed_event_includes_model_and_object(self):
        chunk = {
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        events = self.t.translate_stream_chunk("resp_test", chunk)
        completed_events = [e for e in events if "response.completed" in e]
        _, data = _parse_sse_event(completed_events[0])
        assert data["response"]["object"] == "response"
        assert data["response"]["model"] == "gpt-4o"

    def test_auto_reset_after_finish(self):
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        self.t.translate_stream_chunk("resp_1", chunk)
        assert self.t._tool_call_buffers == {}
        assert self.t._tool_call_meta == {}
        assert self.t._accumulated_text == ""
        assert not self.t._text_item_id
        assert self.t._text_started is False

    def test_finish_with_null_usage_does_not_crash(self):
        """MiniMax sends 'usage: null' in streaming chunks. Must not crash."""
        chunk1 = {"choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]}
        chunk2 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": None,
        }
        self.t.translate_stream_chunk("resp_test", chunk1)
        events = self.t.translate_stream_chunk("resp_test", chunk2)
        # Should produce completed event without crashing
        completed = [e for e in events if "response.completed" in e]
        assert len(completed) == 1
        _, data = _parse_sse_event(completed[0])
        assert data["response"]["status"] == "completed"

    def test_tool_call_with_null_usage_preserves_arguments(self):
        """MiniMax sends tool calls with 'usage: null' in the same chunk. Arguments must be preserved."""
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": '{"key": "value"}',
                                },
                                "index": 0,
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": None,
        }
        events = self.t.translate_stream_chunk("resp_test", chunk)
        # Should produce completed event with correct arguments
        completed = [e for e in events if "response.completed" in e]
        assert len(completed) == 1
        _, data = _parse_sse_event(completed[0])
        fc_items = [o for o in data["response"]["output"] if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["arguments"] == '{"key": "value"}'

    def test_tool_call_delta_produces_lifecycle_events(self):
        # First chunk: tool call name + id (no argument delta yet)
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events1 = self.t.translate_stream_chunk("resp_test", chunk1)
        types1 = _extract_event_types(events1)
        # First chunk emits output_item.added but no arguments delta (empty string)
        assert "response.output_item.added" in types1
        # item should be a function_call
        item_events = [e for e in events1 if "response.output_item.added" in e]
        _, data = _parse_sse_event(item_events[0])
        assert data["item"]["type"] == "function_call"
        assert data["item"]["name"] == "get_weather"

        # Second chunk: argument delta
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"city":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events2 = self.t.translate_stream_chunk("resp_test", chunk2)
        types2 = _extract_event_types(events2)
        assert "response.function_call_arguments.delta" in types2

    def test_streaming_tool_calls_included_in_completed_output(self):
        # Setup: tool call name
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self.t.translate_stream_chunk("resp_test", chunk1)

        # Argument delta
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"city": "London"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self.t.translate_stream_chunk("resp_test", chunk2)

        # Finish
        chunk3 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk("resp_test", chunk3)
        completed_events = [e for e in events if "response.completed" in e]
        _, data = _parse_sse_event(completed_events[0])
        fc_items = [o for o in data["response"]["output"] if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        fc = fc_items[0]
        assert fc["name"] == "get_weather"
        assert fc["arguments"] == '{"city": "London"}'
        assert fc["status"] == "completed"

        # Should also have arguments.done and output_item.done
        types = _extract_event_types(events)
        assert "response.function_call_arguments.done" in types
        assert "response.output_item.done" in types

        # arguments.done should come before output_item.done
        idx_args_done = types.index("response.function_call_arguments.done")
        idx_item_done = types.index("response.output_item.done")
        assert idx_args_done < idx_item_done

    def test_reset_clears_internal_state(self):
        chunk = {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}
        self.t.translate_stream_chunk("resp_1", chunk)
        self.t.reset()
        assert self.t._accumulated_text == ""
        assert not self.t._text_item_id
        assert self.t._text_started is False

        # After reset, should work with new state
        events = self.t.translate_stream_chunk("resp_2", chunk)
        assert len(events) >= 1


# ── strip_thinking_tags ────────────────────────────────────────────────────


class TestStripThinkingTags:
    def setup_method(self):
        self.t = ResponsesTranslator()

    def test_streaming_thinking_tags_stripped(self):
        """اخل tags in streaming content must be stripped from accumulated text."""
        # Simulate MiniMax streaming: thinking tag + actual content + finish
        chunk1 = {
            "choices": [{"index": 0, "delta": {"content": "<\u0627\u062e\u0644>thinking"}, "finish_reason": None}]
        }
        chunk2 = {"choices": [{"index": 0, "delta": {"content": " about math"}, "finish_reason": None}]}
        chunk3 = {
            "choices": [
                {"index": 0, "delta": {"content": "</\u0627\u062e\u0644>\n\nThe answer is 4."}, "finish_reason": None}
            ]
        }
        chunk4 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        self.t.translate_stream_chunk("resp_test", chunk1)
        self.t.translate_stream_chunk("resp_test", chunk2)
        self.t.translate_stream_chunk("resp_test", chunk3)
        events = self.t.translate_stream_chunk("resp_test", chunk4)

        # Find the completed event and check the output text
        completed = [e for e in events if "response.completed" in e]
        _, data = _parse_sse_event(completed[0])
        output_text = data["response"]["output"][0]["content"][0]["text"]
        assert "thinking about math" not in output_text
        assert "The answer is 4." in output_text

    def test_streaming_no_thinking_tags_passes_through(self):
        """Content without thinking tags passes through unchanged."""
        chunk1 = {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
        chunk2 = {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}
        chunk3 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

        self.t.translate_stream_chunk("resp_test", chunk1)
        self.t.translate_stream_chunk("resp_test", chunk2)
        events = self.t.translate_stream_chunk("resp_test", chunk3)

        completed = [e for e in events if "response.completed" in e]
        _, data = _parse_sse_event(completed[0])
        output_text = data["response"]["output"][0]["content"][0]["text"]
        assert output_text == "Hello world"


# ── synthesize_completed_events ────────────────────────────────────────────


class TestSynthesizeCompletedEvents:
    def setup_method(self):
        self.t = ResponsesTranslator()

    def test_returns_empty_list_when_no_content_accumulated(self):
        result = self.t.synthesize_completed_events("resp_test", "gpt-4o")
        assert result == []

    def test_synthesizes_completed_for_no_content_when_status_incomplete(self):
        events = self.t.synthesize_completed_events("resp_test", "gpt-4o", status="incomplete")
        types = _extract_event_types(events)
        assert types == ["response.completed"]
        _, data = _parse_sse_event(events[0])
        assert data["response"]["status"] == "incomplete"
        assert data["response"]["output"] == []

    def test_synthesizes_full_trailing_lifecycle(self):
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        self.t.translate_stream_chunk("resp_test", chunk)

        events = self.t.synthesize_completed_events("resp_test", "gpt-4o")
        types = _extract_event_types(events)

        assert "response.output_text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        assert "response.completed" in types

    def test_synthesize_includes_accumulated_text(self):
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}],
        }
        self.t.translate_stream_chunk("resp_test", chunk)

        events = self.t.synthesize_completed_events("resp_test", "gpt-4o")
        completed = [e for e in events if "response.completed" in e]
        _, data = _parse_sse_event(completed[0])
        assert data["response"]["status"] == "completed"
        output = data["response"]["output"]
        assert len(output) == 1
        assert output[0]["content"][0]["text"] == "Hello world"

    def test_resets_state_after_synthesize(self):
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        self.t.translate_stream_chunk("resp_test", chunk)
        self.t.synthesize_completed_events("resp_test", "gpt-4o")
        assert self.t._accumulated_text == ""
        assert not self.t._text_item_id

    def test_no_double_completed_when_finish_reason_already_sent(self):
        # Normal flow: finish_reason produces completed event
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        self.t.translate_stream_chunk("resp_test", chunk)
        # synthesize should return empty since state was reset by finish
        result = self.t.synthesize_completed_events("resp_test", "gpt-4o")
        assert result == []

    def test_sequence_numbers_are_monotonic_across_stream(self):
        events = self.t.translate_stream_start("resp_test", model="gpt-4o")
        seqs = []
        for e in events:
            _, d = _parse_sse_event(e)
            seqs.append(d["sequence_number"])

        chunk1 = {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
        for e in self.t.translate_stream_chunk("resp_test", chunk1):
            _, d = _parse_sse_event(e)
            seqs.append(d["sequence_number"])

        chunk2 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        for e in self.t.translate_stream_chunk("resp_test", chunk2):
            _, d = _parse_sse_event(e)
            seqs.append(d["sequence_number"])

        # All sequence numbers should be monotonically increasing
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"seq[{i}]={seqs[i]} not > seq[{i - 1}]={seqs[i - 1]}"

    def test_tool_call_arguments_not_empty_in_synthesized_completed(self):
        """synthesize_completed_events must call finalize() only once per buffer.

        Double finalize() bug: if finalize() is called twice, the second call
        returns '{}' because the buffer was already consumed. The response.completed
        output must contain the actual arguments.
        """
        # Feed a tool call chunk
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self.t.translate_stream_chunk("resp_test", chunk1)

        # Argument delta
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"city": "Paris"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        self.t.translate_stream_chunk("resp_test", chunk2)

        # synthesize_completed_events (no finish_reason chunk was sent)
        events = self.t.synthesize_completed_events("resp_test", "gpt-4o")
        types = _extract_event_types(events)

        # Must have function_call_arguments.done
        assert "response.function_call_arguments.done" in types

        # Must have response.completed
        completed = [e for e in events if "response.completed" in e]
        assert len(completed) == 1
        _, data = _parse_sse_event(completed[0])

        # CRITICAL: arguments must be the actual JSON, not '{}'
        fc_items = [o for o in data["response"]["output"] if o.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["arguments"] == '{"city": "Paris"}', (
            f"Expected actual arguments but got: {fc_items[0]['arguments']!r}. "
            "This indicates double-finalize() bug where buffer was already consumed."
        )


# ── output_index allocation (KBR-240) ───────────────────────────────────────


class TestOutputItemIndices:
    """Every opened output item gets its own ``output_index`` (KBR-240).

    The Responses SSE grammar positions output items by ``output_index``.
    Before the fix the translator had no counter at all: text and reasoning
    were pinned to 0 and a function call took the Chat Completions tool-call
    index raw (also 0 for the first call), so prose-then-tool-call — the
    common Codex shape — announced two items at slot 0 and closed slot 0
    twice. One shared counter, allocated when an item opens, mirrors the
    KBR-226 fix on the Messages wire; overlap and free close order stay
    permitted, as decided there.
    """

    def setup_method(self):
        """Give each test a fresh translator."""
        self.t = ResponsesTranslator()
        self.resp_id = "resp_test"

    def _feed(self, delta: dict, finish: str | None = None) -> list[tuple[str, dict]]:
        """Translate one Chat Completions chunk carrying ``delta``.

        Args:
            delta: The ``choices[0].delta`` object of the chunk.
            finish: The chunk's ``finish_reason``, if it is the final chunk.

        Returns:
            The emitted events, parsed as ``(event_name, data_dict)`` pairs.
        """
        chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return [_parse_sse_event(raw) for raw in self.t.translate_stream_chunk(self.resp_id, chunk)]

    def _finish(self) -> list[tuple[str, dict]]:
        """Translate the final chunk that closes the stream."""
        return self._feed({}, finish="tool_calls")

    @staticmethod
    def _added(events: list[tuple[str, dict]]) -> list[tuple[int, str]]:
        """Return ``(output_index, item_type)`` for every output_item.added.

        Args:
            events: Parsed ``(event_name, data_dict)`` pairs.

        Returns:
            One pair per ``response.output_item.added`` event, in wire order.
        """
        return [
            (d["output_index"], d["item"]["type"])
            for name, d in events
            if name == "response.output_item.added"
        ]

    @staticmethod
    def _done_indices(events: list[tuple[str, dict]]) -> list[int]:
        """Return the ``output_index`` of every output_item.done event.

        Args:
            events: Parsed ``(event_name, data_dict)`` pairs.

        Returns:
            The indices in wire order.
        """
        return [d["output_index"] for name, d in events if name == "response.output_item.done"]

    @staticmethod
    def _tool_call(cc_index: int, call_id: str, arguments: str = "") -> dict:
        """Build one Chat Completions ``tool_calls`` delta entry.

        Args:
            cc_index: The Chat Completions tool-call index the entry belongs to.
            call_id: The upstream call id; its presence marks a new call.
            arguments: The argument fragment the entry carries.

        Returns:
            A ``tool_calls`` delta entry in Chat Completions shape.
        """
        return {
            "index": cc_index,
            "id": call_id,
            "type": "function",
            "function": {"name": f"fn_{call_id}", "arguments": arguments},
        }

    def test_text_then_first_tool_call_get_distinct_indices(self):
        """Prose followed by a tool call must not share the output slot."""
        text_events = self._feed({"content": "Let me check."})
        call_events = self._feed({"tool_calls": [self._tool_call(0, "call_a", '{"path": "a"}')]})
        final_events = self._finish()

        assert self._added(text_events) == [(0, "message")]
        assert self._added(call_events) == [(1, "function_call")]
        assert self._done_indices(final_events) == [0, 1]

    def test_reasoning_text_and_parallel_calls_get_increasing_indices(self):
        """All three item kinds draw from one counter, in open order.

        Text sits at slot 1 here, so every text-addressed event — not just
        output_item.added/done — is pinned to the text item's own slot: a
        translator that pinned the delta/part events back to 0 fails here.
        """
        reasoning_events = self._feed({"reasoning_content": "thinking"})
        text_events = self._feed({"content": "Both, then the calls."})
        call_a_events = self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        call_b_events = self._feed({"tool_calls": [self._tool_call(1, "call_b")]})

        assert self._added(reasoning_events) == [(0, "reasoning")]
        assert self._added(text_events) == [(1, "message")]
        assert self._added(call_a_events) == [(2, "function_call")]
        assert self._added(call_b_events) == [(3, "function_call")]

        # The text item's part and delta events carry the text item's slot.
        assert [
            (name, d["output_index"])
            for name, d in text_events
            if name in ("response.content_part.added", "response.output_text.delta")
        ] == [("response.content_part.added", 1), ("response.output_text.delta", 1)]

        # The finish closes each item once, at its own recorded slot, and the
        # text done events (output_text.done, content_part.done) follow it.
        final_events = self._finish()
        assert self._done_indices(final_events) == [0, 1, 2, 3]
        assert [
            (name, d["output_index"])
            for name, d in final_events
            if name in ("response.output_text.done", "response.content_part.done")
        ] == [("response.output_text.done", 1), ("response.content_part.done", 1)]

    def test_text_after_calls_opens_at_next_free_slot(self):
        """Text following the calls opens a fresh slot; the calls stay open."""
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        self._feed({"tool_calls": [self._tool_call(1, "call_b")]})
        text_events = self._feed({"content": "Both calls are in."})

        # Overlap is the decided shape: the text item opens while both calls
        # are still open, and no done event may close a call early.
        assert self._added(text_events) == [(2, "message")]
        assert not [name for name, _ in text_events if name == "response.output_item.done"]

    def test_arguments_events_carry_the_owning_call_index(self):
        """``function_call_arguments.delta/done`` address the call's own slot.

        The vendor grammar defines ``output_index`` on both events as a
        required field; the translator omitted it entirely.
        """
        self._feed({"content": "Let me check."})
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        args_events = self._feed({"tool_calls": [{"index": 0, "function": {"arguments": '{"path"'}}]})
        final_events = self._finish()

        arg_deltas = [
            d["output_index"]
            for name, d in args_events
            if name == "response.function_call_arguments.delta"
        ]
        arg_dones = [
            d["output_index"]
            for name, d in final_events
            if name == "response.function_call_arguments.done"
        ]
        assert arg_deltas == [1]
        assert arg_dones == [1]

    def test_finish_closes_each_item_once_at_its_recorded_index(self):
        """The finish path closes text and both calls once, at their own slots."""
        self._feed({"content": "Let me check."})
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        self._feed({"tool_calls": [self._tool_call(1, "call_b")]})

        final_events = self._finish()
        assert self._done_indices(final_events) == [0, 1, 2]

    def test_synthesize_completed_events_closes_each_item_at_its_recorded_index(self):
        """The EOF-without-finish path closes by the recorded slot, like finish."""
        self._feed({"content": "Let me check."})
        self._feed({"tool_calls": [self._tool_call(0, "call_a", '{"path": "a"}')]})

        final_events = [
            _parse_sse_event(raw) for raw in self.t.synthesize_completed_events(self.resp_id, "m")
        ]
        assert self._done_indices(final_events) == [0, 1]

    def test_completed_output_ordered_by_index(self):
        """The completed response's output array follows slot order, not open order.

        Text opened first (slot 0), reasoning second (slot 1); the array must
        list the message before the reasoning item so a positional client
        aligning array position with ``output_index`` reads it correctly.
        """
        self._feed({"content": "The answer is 4."})
        self._feed({"reasoning_content": "checked the sum"})
        final_events = self._feed({}, finish="stop")

        completed = [
            d for name, d in final_events if name == "response.completed"
        ]
        output_types = [item["type"] for item in completed[0]["response"]["output"]]
        assert output_types == ["message", "reasoning"]

    def test_completed_array_includes_item_whose_text_stripped_to_nothing(self):
        """A text item opened but stripped empty still closes and appears.

        MiniMax interleaves thinking tags in content; when every character of
        the text was a tag, the item is closed with empty text (its done event
        already was) and stays in the completed array, so array position keeps
        equalling ``output_index``.
        """
        self._feed({"content": "<اخل>weighing it</اخل>"})
        self._feed({"tool_calls": [self._tool_call(0, "call_a", '{"path": "a"}')]})
        final_events = self._finish()

        text_done = [
            d
            for name, d in final_events
            if name == "response.output_item.done" and d["item"]["type"] == "message"
        ]
        assert [d["output_index"] for d in text_done] == [0]
        assert text_done[0]["item"]["content"][0]["text"] == ""

        completed = [d for name, d in final_events if name == "response.completed"]
        output_types = [item["type"] for item in completed[0]["response"]["output"]]
        assert output_types == ["message", "function_call"]
        assert completed[0]["response"]["output"][0]["content"][0]["text"] == ""

    def test_synthesize_closes_the_reasoning_item_at_its_recorded_index(self):
        """The EOF-without-finish path closes reasoning too, at its own slot.

        Upstream streams that end without a finish chunk (timeout, dropped
        connection) reach ``synthesize_completed_events`` with the reasoning
        item open; leaving it open would leave an added with no done.
        """
        self._feed({"reasoning_content": "working it out"})
        self._feed({"content": "The answer is 4."})

        final_events = [
            _parse_sse_event(raw) for raw in self.t.synthesize_completed_events(self.resp_id, "m")
        ]
        assert self._done_indices(final_events) == [0, 1]
        completed = [d for name, d in final_events if name == "response.completed"]
        output_types = [item["type"] for item in completed[0]["response"]["output"]]
        assert output_types == ["reasoning", "message"]
        assert completed[0]["response"]["output"][0]["summary"][0]["text"] == "working it out"

    def test_synthesize_closes_text_stripped_to_nothing_before_a_call(self):
        """EOF with thinking-tag-only text plus a tool call closes both.

        The synthesize path closes items by whether they *opened*, not by
        whether their text survived tag-stripping — the timeout-shaped twin of
        ``test_completed_array_includes_item_whose_text_stripped_to_nothing``,
        which walks the finish path. The two paths use different predicates
        and emit in different orders, so neither covers the other.
        """
        self._feed({"content": "<اخل>weighing it</اخل>"})
        self._feed({"tool_calls": [self._tool_call(0, "call_a", '{"path": "a"}')]})

        final_events = [
            _parse_sse_event(raw) for raw in self.t.synthesize_completed_events(self.resp_id, "m")
        ]
        assert self._done_indices(final_events) == [0, 1]
        completed = [d for name, d in final_events if name == "response.completed"]
        output_types = [item["type"] for item in completed[0]["response"]["output"]]
        assert output_types == ["message", "function_call"]
        assert completed[0]["response"]["output"][0]["content"][0]["text"] == ""

    def test_reset_restarts_allocation_at_zero(self):
        """A second stream on a reused translator allocates from 0 again."""
        self._feed({"content": "First stream."})
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        self._finish()

        text_events = self._feed({"content": "Second stream."})
        call_events = self._feed({"tool_calls": [self._tool_call(0, "call_b")]})
        assert self._added(text_events) == [(0, "message")]
        assert self._added(call_events) == [(1, "function_call")]


class TestResponsesToolChoice:
    """KBR-221: ``tool_choice`` and ``parallel_tool_calls`` survive the Responses -> CC hop.

    Responses and Chat Completions spell most values the same way; named choices move
    across via a spelling shift (``function.name`` -> ``function.function.name``), and
    ``allowed_tools`` / hosted / MCP / ``custom`` shapes that have no CC form are
    omitted (D5 / D10). ``parallel_tool_calls`` is forwarded only when the wire
    carries the explicit non-default ``false`` (D2) -- the rest of the request is
    unaffected.
    """

    def setup_method(self):
        self.t = ResponsesTranslator()

    def _req(self, **extra):
        """Build a minimal Responses request with one function tool, plus the case's fields."""
        req = {
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
        }
        req.update(extra)
        return req

    @pytest.mark.parametrize(
        "choice",
        ["auto", "none", "required"],
        ids=["auto", "none", "required"],
    )
    def test_tool_choice_string_is_carried(self, choice):
        """AC-1: each published string choice maps to the CC spelling (R1)."""
        result = self.t.translate_request(self._req(tool_choice=choice))
        assert result["tool_choice"] == choice

    def test_tool_choice_function_is_carried(self):
        """AC-1: named function choice moves ``name`` under ``function.function.name``."""
        result = self.t.translate_request(
            self._req(tool_choice={"type": "function", "name": "get_weather"})
        )
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_choice_custom_is_omitted(self):
        """AC-2 / D10: ``custom``-typed choice names a tool the hop degraded."""
        result = self.t.translate_request(
            self._req(tool_choice={"type": "custom", "name": "x"})
        )
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "kind",
        [
            "file_search",
            "web_search_preview",
            "computer",
            "computer_use_preview",
            "computer_use",
            "web_search_preview_2025_03_11",
            "image_generation",
            "code_interpreter",
            "programmatic_tool_calling",
            "apply_patch",
            "shell",
        ],
        ids=[
            "file_search",
            "web_search_preview",
            "computer",
            "computer_use_preview",
            "computer_use",
            "web_search_preview_2025_03_11",
            "image_generation",
            "code_interpreter",
            "programmatic_tool_calling",
            "apply_patch",
            "shell",
        ],
    )
    def test_tool_choice_hosted_is_omitted(self, kind):
        """AC-2: hosted built-in choices have no CC form and nothing downstream can run them (D5 / D10)."""
        result = self.t.translate_request(self._req(tool_choice={"type": kind}))
        assert "tool_choice" not in result

    def test_tool_choice_mcp_is_omitted(self):
        """AC-2: an MCP choice names a server tool the bridge does not proxy (D5 / D10)."""
        result = self.t.translate_request(
            self._req(tool_choice={"type": "mcp", "server_label": "srv", "name": "tool"})
        )
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "malformed",
        [7, 7.0, [], True],
        ids=["int", "float", "list", "bool"],
    )
    def test_tool_choice_malformed_is_omitted(self, malformed):
        """AC-2: a non-string non-dict shape has no CC home and is omitted (D5)."""
        result = self.t.translate_request(self._req(tool_choice=malformed))
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        ("mode", "cc"),
        [("auto", "auto"), ("required", "required"), ("none", "none")],
        ids=["auto", "required", "defensive-none"],
    )
    def test_allowed_tools_carries_mode_only(self, mode, cc):
        """AC-3: only the mode has a canonical home; the tools list is dropped."""
        result = self.t.translate_request(
            self._req(
                tool_choice={
                    "type": "allowed_tools",
                    "mode": mode,
                    "tools": [{"type": "function", "name": "get_weather"}],
                }
            )
        )
        assert result["tool_choice"] == cc

    @pytest.mark.parametrize(
        "bad_mode",
        ["bogus", None, 7],
        ids=["unknown-string", "null", "int"],
    )
    def test_allowed_tools_unknown_mode_is_omitted(self, bad_mode):
        """AC-3 / D5: a non-published or non-string mode inside ``allowed_tools`` has no canonical home."""
        result = self.t.translate_request(
            self._req(tool_choice={"type": "allowed_tools", "mode": bad_mode})
        )
        assert "tool_choice" not in result

    def test_parallel_tool_calls_false_is_carried(self):
        """AC-4: explicit non-default ``false`` is forwarded (R2)."""
        result = self.t.translate_request(self._req(parallel_tool_calls=False))
        assert result["parallel_tool_calls"] is False

    def test_parallel_tool_calls_true_is_omitted(self):
        """AC-5: explicit ``true`` matches the documented default on both wires (D2)."""
        result = self.t.translate_request(self._req(parallel_tool_calls=True))
        assert "parallel_tool_calls" not in result

    def test_parallel_tool_calls_absent_is_omitted(self):
        """AC-5: no instruction means no entry (R9)."""
        assert "parallel_tool_calls" not in self.t.translate_request(self._req())

    @pytest.mark.parametrize(
        "value",
        ["false", 0, None, []],
        ids=["string", "zero", "null", "empty-list"],
    )
    def test_parallel_tool_calls_non_bool_is_omitted(self, value):
        """AC-5b: a non-bool value would create an unclaimed residual delta (D5)."""
        result = self.t.translate_request(self._req(parallel_tool_calls=value))
        assert "parallel_tool_calls" not in result

    def test_no_tool_choice_invents_none(self):
        """R9: no inbound ``tool_choice`` and no inbound ``parallel_tool_calls`` -> no entries."""
        result = self.t.translate_request(self._req())
        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result

    def test_forced_call_to_custom_tool_is_omitted_beside_a_function(self):
        """AC-12a / D10: a function-typed choice naming a custom-declared tool is omitted.

        The declared function tool keeps the CC list non-empty, so the D9 gate
        passes and this test is decided by the D10 named-tool lookup -- which
        must walk the inbound list, the only place the degraded entry still
        lives (the hop has filtered it out of the CC list).
        """
        req = self._req(
            tools=[
                {"type": "function", "name": "get_weather", "parameters": {}},
                {"type": "custom", "name": "review", "format": {"type": "text"}},
            ],
            tool_choice={"type": "function", "name": "review"},
        )
        result = self.t.translate_request(req)
        assert "tool_choice" not in result

    def test_forced_call_to_hosted_tool_is_omitted_beside_a_function(self):
        """AC-12a / D10: the hosted twin -- choice naming a hosted entry beside a function tool."""
        req = self._req(
            tools=[
                {"type": "function", "name": "get_weather", "parameters": {}},
                {"type": "web_search_preview", "name": "search"},
            ],
            tool_choice={"type": "function", "name": "search"},
        )
        result = self.t.translate_request(req)
        assert "tool_choice" not in result

    def test_forced_call_to_undeclared_tool_is_carried(self):
        """AC-12b / D8: a choice naming no declared tool is the agent's mistake -- the provider's error names it."""
        result = self.t.translate_request(
            self._req(tool_choice={"type": "function", "name": "not_declared"})
        )
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "not_declared"},
        }

    @pytest.mark.parametrize(
        "bad",
        [
            {"type": "function"},
            {"type": "function", "name": 7},
        ],
        ids=["function-no-name", "function-non-string-name"],
    )
    def test_tool_choice_function_malformed_is_omitted(self, bad):
        """AC-2 / D5: a function-typed choice without a string name has no CC home."""
        result = self.t.translate_request(self._req(tool_choice=bad))
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "tools_value",
        [None, []],
        ids=["absent", "empty-list"],
    )
    def test_tool_choice_without_tools_is_omitted(self, tools_value):
        """AC-11 / D9: choice over no tools (key absent or empty list) is legal Responses and a 400 on CC."""
        req = self._req(tool_choice="required")
        if tools_value is None:
            del req["tools"]
        else:
            req["tools"] = tools_value
        result = self.t.translate_request(req)
        assert "tool_choice" not in result

    def test_hosted_tools_only_omits_tool_choice(self):
        """AC-11 / D9: an inbound list of only hosted tools -> CC list empty -> choice omitted (gate on CC list)."""
        req = self._req(
            tools=[{"type": "web_search_preview", "name": "search"}],
            tool_choice="required",
        )
        result = self.t.translate_request(req)
        assert "tool_choice" not in result

    def test_parallel_tool_calls_carries_without_tools(self):
        """AC-4 / D9 boundary: the knob is a standalone field and is not gated by D9.

        ``tool_choice`` nests no parallel knob on this wire (unlike Anthropic's
        ``disable_parallel_tool_use``), so an inbound ``false`` reaches the CC
        body even when no tools are present -- R2's mapping is unconditional.
        """
        req = self._req(parallel_tool_calls=False)
        del req["tools"]
        result = self.t.translate_request(req)
        assert result["parallel_tool_calls"] is False
        assert "tool_choice" not in result

    def test_full_composition_carries_all_three(self):
        """AC-14: tools + tool_choice + parallel_tool_calls reach the CC body together."""
        result = self.t.translate_request(
            self._req(
                tool_choice={"type": "function", "name": "get_weather"},
                parallel_tool_calls=False,
            )
        )
        assert [t["function"]["name"] for t in result["tools"]] == ["get_weather"]
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }
        assert result["parallel_tool_calls"] is False
