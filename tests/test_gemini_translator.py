"""Tests for GeminiTranslator — request, response, and streaming translation."""

from __future__ import annotations

import json

from kitty.bridge.gemini.translator import GeminiTranslator


class TestTranslateRequestSimpleText:
    """Single user message with text parts."""

    def test_simple_user_text(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        cc = t.translate_request(gemini_req)
        assert cc["messages"][-1] == {"role": "user", "content": "Hello"}

    def test_model_key_not_in_request(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        cc = t.translate_request(gemini_req)
        assert "model" not in cc


class TestTranslateRequestMultiTurn:
    """Multi-turn conversation."""

    def test_user_model_user(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello!"}]},
                {"role": "user", "parts": [{"text": "How are you?"}]},
            ],
        }
        cc = t.translate_request(gemini_req)
        assert len(cc["messages"]) == 3
        assert cc["messages"][0] == {"role": "user", "content": "Hi"}
        assert cc["messages"][1] == {"role": "assistant", "content": "Hello!"}
        assert cc["messages"][2] == {"role": "user", "content": "How are you?"}


class TestTranslateRequestSystemInstruction:
    """systemInstruction → system message prepended."""

    def test_system_instruction_prepended(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": "You are a helper."}]},
        }
        cc = t.translate_request(gemini_req)
        assert cc["messages"][0] == {"role": "system", "content": "You are a helper."}
        assert cc["messages"][1] == {"role": "user", "content": "Hello"}


class TestTranslateRequestTools:
    """functionDeclarations → tools in Chat Completions format."""

    def test_tools_translated(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [{"role": "user", "parts": [{"text": "Weather?"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {"location": {"type": "STRING"}},
                                "required": ["location"],
                            },
                        }
                    ]
                }
            ],
        }
        cc = t.translate_request(gemini_req)
        assert len(cc["tools"]) == 1
        tool = cc["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather"
        assert "parameters" in tool["function"]


class TestTranslateRequestGenerationConfig:
    """generationConfig fields mapped to Chat Completions params."""

    def test_temperature_and_tokens(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 100, "topP": 0.9},
        }
        cc = t.translate_request(gemini_req)
        assert cc["temperature"] == 0.5
        assert cc["max_tokens"] == 100
        assert cc["top_p"] == 0.9


class TestTranslateRequestFunctionCall:
    """functionCall parts in model messages → tool_calls."""

    def test_model_function_call(self):
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {"role": "user", "parts": [{"text": "Weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}}]},
                {"role": "function", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp": "72F"}}}]},
                {"role": "user", "parts": [{"text": "Thanks"}]},
            ],
        }
        cc = t.translate_request(gemini_req)
        msgs = cc["messages"]
        # user, assistant with tool_calls, tool result, user
        assert len(msgs) == 4
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["tool_calls"]) == 1
        tc = msgs[1]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "NYC"}
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["content"] == '{"temp": "72F"}'
        assert msgs[3]["role"] == "user"


class TestTranslateResponseText:
    """Chat Completions text response → Gemini candidates."""

    def test_text_response(self):
        t = GeminiTranslator()
        cc_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        gemini_resp = t.translate_response(cc_resp)
        assert len(gemini_resp["candidates"]) == 1
        assert gemini_resp["candidates"][0]["content"]["role"] == "model"
        assert gemini_resp["candidates"][0]["content"]["parts"] == [{"text": "Hello!"}]
        assert gemini_resp["candidates"][0]["finishReason"] == "STOP"


class TestTranslateResponseToolCalls:
    """Chat Completions tool_calls → Gemini functionCall parts."""

    def test_tool_call_response(self):
        t = GeminiTranslator()
        cc_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        gemini_resp = t.translate_response(cc_resp)
        parts = gemini_resp["candidates"][0]["content"]["parts"]
        assert len(parts) == 1
        assert parts[0]["functionCall"]["name"] == "get_weather"
        assert parts[0]["functionCall"]["args"] == {"location": "NYC"}


class TestTranslateResponseFinishReasons:
    """finish_reason → Gemini finishReason mapping."""

    def test_stop(self):
        t = GeminiTranslator()
        resp = t.translate_response({
            "id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        assert resp["candidates"][0]["finishReason"] == "STOP"

    def test_length(self):
        t = GeminiTranslator()
        resp = t.translate_response({
            "id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        assert resp["candidates"][0]["finishReason"] == "MAX_TOKENS"

    def test_tool_calls_maps_to_stop(self):
        t = GeminiTranslator()
        resp = t.translate_response({
            "id": "x",
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        assert resp["candidates"][0]["finishReason"] == "STOP"


class TestTranslateResponseUsage:
    """usage → usageMetadata mapping."""

    def test_usage_metadata(self):
        t = GeminiTranslator()
        resp = t.translate_response({
            "id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })
        assert resp["usageMetadata"]["promptTokenCount"] == 10
        assert resp["usageMetadata"]["candidatesTokenCount"] == 20
        assert resp["usageMetadata"]["totalTokenCount"] == 30


class TestStreamingTextDeltas:
    """CC stream chunks → Gemini SSE data events."""

    def test_text_delta(self):
        t = GeminiTranslator()
        chunk = {
            "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}],
        }
        events = t.translate_stream_chunk(chunk)
        assert len(events) == 1
        data = json.loads(events[0].removeprefix("data: ").removesuffix("\n\n"))
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "Hi"

    def test_finish_chunk(self):
        t = GeminiTranslator()
        chunk = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = t.translate_stream_chunk(chunk)
        assert len(events) == 1
        data = json.loads(events[0].removeprefix("data: ").removesuffix("\n\n"))
        assert data["candidates"][0]["finishReason"] == "STOP"
        assert data["usageMetadata"]["totalTokenCount"] == 15


class TestStreamingToolCallDeltas:
    """CC stream tool_call argument deltas → accumulated functionCall."""

    def test_tool_call_accumulated(self):
        t = GeminiTranslator()
        # First chunk: tool call starts
        chunk1 = {
            "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": None}],
        }
        events1 = t.translate_stream_chunk(chunk1)
        # No events yet — buffering starts
        assert events1 == []

        # Argument delta
        chunk2 = {
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}, "finish_reason": None}],
        }
        events2 = t.translate_stream_chunk(chunk2)
        assert events2 == []

        # More argument delta
        chunk3 = {
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ation":"NYC"}'}}]}, "finish_reason": None}],
        }
        events3 = t.translate_stream_chunk(chunk3)
        assert events3 == []

        # Finish
        chunk4 = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        events4 = t.translate_stream_chunk(chunk4)
        # Should include the functionCall part + finish
        assert len(events4) >= 1
        # First event should be the functionCall
        data = json.loads(events4[0].removeprefix("data: ").removesuffix("\n\n"))
        fc = data["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert fc["name"] == "get_weather"
        assert fc["args"] == {"location": "NYC"}


class TestReset:
    """reset() clears all streaming state."""

    def test_reset_clears_state(self):
        t = GeminiTranslator()
        # Simulate some streaming state
        chunk = {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
        t.translate_stream_chunk(chunk)
        t.reset()
        # After reset, translator should work cleanly
        chunk2 = {"choices": [{"delta": {"content": "New"}, "finish_reason": None}]}
        events = t.translate_stream_chunk(chunk2)
        assert len(events) == 1
        data = json.loads(events[0].removeprefix("data: ").removesuffix("\n\n"))
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "New"
