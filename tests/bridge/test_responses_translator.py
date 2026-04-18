"""Tests for bridge/responses/translator.py — Responses API <-> Chat Completions translation."""

import json
import uuid

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
        """When content is only thinking tags, no text output item is produced."""
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
        # No text output item since all content was thinking tags
        text_items = [o for o in result["output"] if o["type"] == "message"]
        assert len(text_items) == 0


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
        assert self.t._text_item_id is None
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
        assert self.t._text_item_id is None
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
        assert self.t._text_item_id is None

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
