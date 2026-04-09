"""Tests for bridge/messages/translator.py — Anthropic Messages API <-> Chat Completions translation."""

import json
import uuid

from kitty.bridge.messages.translator import MessagesTranslator


def _v4() -> str:
    return str(uuid.uuid4())


# ── translate_request ───────────────────────────────────────────────────────


class TestTranslateRequest:
    def setup_method(self):
        self.t = MessagesTranslator()

    def test_extracts_model(self):
        req = {"model": "claude-3-opus", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1024}
        result = self.t.translate_request(req)
        assert result["model"] == "claude-3-opus"

    def test_system_prompt_mapped(self):
        req = {
            "model": "claude-3-opus",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}

    def test_text_content_blocks_mapped_to_string(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assert result["messages"][-1]["content"] == "hello"

    def test_tool_use_blocks_mapped_to_tool_calls(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check."
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "toolu_001"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}

    def test_tool_result_blocks_mapped_to_tool_messages(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_001", "content": "72F sunny"},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        tool_msg = result["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_001"
        assert tool_msg["content"] == "72F sunny"

    def test_multiple_tool_results_produce_multiple_tool_messages(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                        {"type": "tool_use", "id": "toolu_002", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_001", "content": "72F sunny"},
                        {"type": "tool_result", "tool_use_id": "toolu_002", "content": "65F cloudy"},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        # Should produce two tool role messages
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "toolu_001"
        assert tool_msgs[1]["tool_call_id"] == "toolu_002"

    def test_tools_mapped_from_anthropic_to_cc_format(self):
        req = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
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

    def test_max_tokens_passthrough(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 512}
        result = self.t.translate_request(req)
        assert result["max_tokens"] == 512

    def test_stream_flag_passthrough(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": True}
        result = self.t.translate_request(req)
        assert result["stream"] is True

    def test_temperature_top_p_passthrough(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = self.t.translate_request(req)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_thinking_stripped_and_flag_set(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 4000},
        }
        assert not self.t.thinking_warned
        result = self.t.translate_request(req)
        assert "thinking" not in result
        assert self.t.thinking_warned

    def test_thinking_warning_only_once(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 4000},
        }
        self.t.translate_request(req)
        assert self.t.thinking_warned
        # Second call should not reset or re-warn
        self.t.translate_request(req)
        assert self.t.thinking_warned

    def test_unsupported_fields_stripped(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "metadata": {"user_id": "123"},
            "stop_sequences": ["\n"],
            "top_k": 50,
            "tool_choice": "auto",
        }
        result = self.t.translate_request(req)
        assert "metadata" not in result
        assert "stop_sequences" not in result
        assert "top_k" not in result
        assert "tool_choice" not in result


# ── translate_response ──────────────────────────────────────────────────────


class TestTranslateResponse:
    def setup_method(self):
        self.t = MessagesTranslator()

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
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["id"].startswith("msg_")
        assert result["model"] == "gpt-4o"
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
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
        assert result["stop_reason"] == "tool_use"
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        tb = tool_blocks[0]
        assert tb["name"] == "get_weather"
        assert tb["input"] == {"city": "London"}

    def test_stop_reason_mapping(self):
        for finish_reason, expected in [("stop", "end_turn"), ("tool_calls", "tool_use"), ("length", "max_tokens")]:
            cc_response = {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": finish_reason}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            result = self.t.translate_response(cc_response)
            assert result["stop_reason"] == expected, f"{finish_reason} -> {result['stop_reason']}, expected {expected}"

    def test_mixed_text_and_tool_call(self):
        cc_response = {
            "id": "chatcmpl-789",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = self.t.translate_response(cc_response)
        assert result["type"] == "message"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"


# ── translate_stream_chunk ─────────────────────────────────────────────────


class TestTranslateStreamChunk:
    def setup_method(self):
        self.t = MessagesTranslator()

    def _make_message_id(self) -> str:
        return f"msg_{uuid.uuid4().hex[:24]}"

    def test_text_delta_produces_content_block_events(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        # Should produce content_block_start + content_block_delta for text
        assert any("content_block_start" in e for e in events)
        assert any("content_block_delta" in e for e in events)
        # Check text_delta
        delta_events = [e for e in events if "content_block_delta" in e]
        assert '"text_delta"' in delta_events[0]
        assert '"Hello"' in delta_events[0]

    def test_finish_produces_message_delta_and_stop(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        assert any("message_delta" in e for e in events)
        assert any("message_stop" in e for e in events)
        # Check stop_reason
        delta_event = [e for e in events if "message_delta" in e][0]
        assert "end_turn" in delta_event

    def test_tool_call_delta_produces_input_json_delta(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        # First chunk: tool call name
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
        events1 = self.t.translate_stream_chunk(msg_id, model, chunk1)
        assert any("content_block_start" in e for e in events1)
        assert any("tool_use" in e for e in events1)

        # Second chunk: argument delta
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
                    "finish_reason": None,
                }
            ],
        }
        events2 = self.t.translate_stream_chunk(msg_id, model, chunk2)
        assert any("input_json_delta" in e for e in events2)

    def test_content_block_index_increments(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        # Text delta (index 0)
        chunk1 = {
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        events1 = self.t.translate_stream_chunk(msg_id, model, chunk1)
        # Verify text block at index 0
        start_events = [e for e in events1 if "content_block_start" in e]
        assert '"index": 0' in start_events[0]

        # Tool call (should be index 1)
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "fn", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events2 = self.t.translate_stream_chunk(msg_id, model, chunk2)
        start_events2 = [e for e in events2 if "content_block_start" in e]
        assert '"index": 1' in start_events2[0]

    def test_auto_reset_after_finish(self):
        msg_id = self._make_message_id()
        model = "m"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.t.translate_stream_chunk(msg_id, model, chunk)
        assert self.t._tool_call_buffers == {}
        assert self.t._content_block_index == 0
        assert self.t._text_block_opened is False

    def test_reset_clears_internal_state(self):
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        self.t.translate_stream_chunk("msg_1", "m", chunk)
        self.t.reset()
        assert self.t._tool_call_buffers == {}
        assert self.t._content_block_index == 0
        assert self.t._text_block_opened is False

    def test_finish_reason_length_maps_to_max_tokens(self):
        msg_id = self._make_message_id()
        model = "m"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        delta_event = [e for e in events if "message_delta" in e][0]
        assert "max_tokens" in delta_event
