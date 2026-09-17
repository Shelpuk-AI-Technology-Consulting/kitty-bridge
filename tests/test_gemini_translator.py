"""Tests for GeminiTranslator — request, response, and streaming translation."""

from __future__ import annotations

import json

import pytest

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
                {
                    "role": "function",
                    "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp": "72F"}}}],
                },
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

    def test_assistant_message_with_thought_mapped_to_reasoning_content(self):
        """Assistant messages with thought parts must map to reasoning_content."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"text": "I will solve this", "thought": True},
                        {"text": "The answer is 42"},
                    ],
                },
            ],
        }
        cc = t.translate_request(gemini_req)
        msg = cc["messages"][0]
        assert msg["reasoning_content"] == "I will solve this"
        assert msg["content"] == "The answer is 42"

    def test_model_function_call_with_thought(self):
        """Assistant messages with thought + functionCall must map to reasoning_content + tool_calls."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"text": "Thinking about weather...", "thought": True},
                        {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}},
                    ],
                },
            ],
        }
        cc = t.translate_request(gemini_req)
        msg = cc["messages"][0]
        assert msg["reasoning_content"] == "Thinking about weather..."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


class TestTranslateRequestToolCallIdPairing:
    """Echo the inbound Gemini wire ``id``; synthesise only when absent.

    KBR-195 — the production translator was discarding the inbound
    ``functionCall.id`` and ``functionResponse.id`` and minting two
    independent synthetic ids, so a call/result pair that arrived correctly
    matched came out paired by nothing. The fix preserves the wire id when
    present and falls back to synthesis only when it is absent.
    """

    def test_function_call_id_is_echoed_when_present(self):
        """Inbound ``functionCall.id`` must reach the CC ``tool_calls[].id``."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {"role": "user", "parts": [{"text": "Weather?"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "shared-123",
                                "name": "get_weather",
                                "args": {"location": "NYC"},
                            }
                        }
                    ],
                },
            ],
        }
        cc = t.translate_request(gemini_req)
        assistant = next(m for m in cc["messages"] if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"] == "shared-123"

    def test_function_response_id_is_echoed_when_present(self):
        """Inbound ``functionResponse.id`` must reach the CC ``tool_call_id``."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {
                    "role": "function",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "shared-123",
                                "name": "get_weather",
                                "response": {"temp": "72F"},
                            }
                        }
                    ],
                }
            ],
        }
        cc = t.translate_request(gemini_req)
        tool_msg = next(m for m in cc["messages"] if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "shared-123"

    def test_function_call_id_is_synthesised_when_absent(self):
        """No wire id → fall back to ``call_<uuid>`` (current behaviour preserved)."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "get_weather", "args": {}}}],
                }
            ],
        }
        cc = t.translate_request(gemini_req)
        assistant = next(m for m in cc["messages"] if m["role"] == "assistant")
        assert assistant["tool_calls"][0]["id"].startswith("call_")

    def test_function_response_id_is_synthesised_when_absent(self):
        """No wire id on the response → fall back to ``call_<uuid>`` (current behaviour preserved)."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {
                    "role": "function",
                    "parts": [{"functionResponse": {"name": "get_weather", "response": {}}}],
                }
            ],
        }
        cc = t.translate_request(gemini_req)
        tool_msg = next(m for m in cc["messages"] if m["role"] == "tool")
        assert tool_msg["tool_call_id"].startswith("call_")

    def test_pairing_survives_when_wire_ids_match(self):
        """The KBR-195 reproduction inverted: a matched wire pair stays matched upstream."""
        t = GeminiTranslator()
        gemini_req = {
            "contents": [
                {"role": "user", "parts": [{"text": "Weather?"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "shared-123",
                                "name": "get_weather",
                                "args": {"location": "NYC"},
                            }
                        }
                    ],
                },
                {
                    "role": "function",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "shared-123",
                                "name": "get_weather",
                                "response": {"temp": "72F"},
                            }
                        }
                    ],
                },
            ],
        }
        cc = t.translate_request(gemini_req)
        assistant = next(m for m in cc["messages"] if m["role"] == "assistant")
        tool_msg = next(m for m in cc["messages"] if m["role"] == "tool")
        call_id = assistant["tool_calls"][0]["id"]
        result_id = tool_msg["tool_call_id"]
        assert call_id == "shared-123"
        assert result_id == "shared-123"
        assert call_id == result_id


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

    def test_response_with_reasoning_content(self):
        """CC response with reasoning_content must map to Gemini thought parts."""
        t = GeminiTranslator()
        cc_resp = {
            "id": "chatcmpl-reason",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        gemini_resp = t.translate_response(cc_resp)
        parts = gemini_resp["candidates"][0]["content"]["parts"]
        # Should be [thought part, text part]
        assert len(parts) == 2
        assert parts[0] == {"text": "Let me think...", "thought": True}
        assert parts[1] == {"text": "The answer is 42."}


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


class TestTranslateResponseToolCallIdEcho:
    """Echo the upstream CC ``tool_calls[].id``; omit when absent.

    KBR-257 — the response-direction mirror of KBR-195. The production
    translator was reading only ``tc["function"]["name"]`` and
    ``tc["function"]["arguments"]`` on the sync path and storing only
    ``{"name": ...}`` in ``_tool_call_meta`` on the streaming path, so the
    upstream ``tool_calls[].id`` never reached the emitted ``functionCall``
    part. The Gemini client therefore had no id to echo back, and KBR-195's
    request-side ``or``-echo never activated for clients whose only id source
    is the bridge's response.
    """

    def test_function_call_id_is_echoed_when_present(self):
        """Upstream CC ``tool_calls[].id`` must reach the emitted ``functionCall.id``."""
        t = GeminiTranslator()
        cc_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_up1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        gemini_resp = t.translate_response(cc_resp)
        function_call = gemini_resp["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert function_call.get("id") == "call_up1"

    def test_function_call_id_is_omitted_when_absent(self):
        """No upstream ``id`` → emit a ``functionCall`` with no ``id`` key.

        Regression guard: KBR-195 synthesises on the request side because CC
        requires a tool-call id; the response side has no such requirement
        (Gemini ``FunctionCall.id`` is optional per v1beta). Absence stays
        absence — synthesising here would defeat KBR-195's request-side
        echo by minting a value the upstream never sent.
        """
        t = GeminiTranslator()
        cc_resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        gemini_resp = t.translate_response(cc_resp)
        function_call = gemini_resp["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert "id" not in function_call

    def test_function_call_id_is_carried_across_streamed_chunks(self):
        """The id from the opening delta must survive into the finish emit."""
        t = GeminiTranslator()

        # Opening delta: id + name open the slot, no arguments yet.
        chunk1 = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        assert t.translate_stream_chunk(chunk1) == []

        # Argument deltas only — id must NOT be re-read or overwritten.
        chunk2 = {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}, "finish_reason": None}
            ],
        }
        assert t.translate_stream_chunk(chunk2) == []

        chunk3 = {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ation":"NYC"}'}}]},
                    "finish_reason": None,
                }
            ],
        }
        assert t.translate_stream_chunk(chunk3) == []

        # Finish chunk — emits the buffered functionCall then the finish event.
        chunk4 = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        events4 = t.translate_stream_chunk(chunk4)
        function_call_event = json.loads(events4[0].removeprefix("data: ").removesuffix("\n\n"))
        function_call = function_call_event["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert function_call.get("id") == "call_1"
        assert function_call["name"] == "get_weather"
        assert function_call["args"] == {"location": "NYC"}

    def test_multiple_indices_carry_their_own_ids(self):
        """Each tool-call index must carry its own id, not the previous index's."""
        t = GeminiTranslator()

        # Both tools open on the same delta.
        open_chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_A", "function": {"name": "f_a", "arguments": ""}},
                            {"index": 1, "id": "call_B", "function": {"name": "f_b", "arguments": ""}},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        assert t.translate_stream_chunk(open_chunk) == []

        args_chunk = {
            "choices": [
                {
                    "delta": {"tool_calls": [
                        {"index": 0, "function": {"arguments": '{"x":1}'}},
                        {"index": 1, "function": {"arguments": '{"y":2}'}},
                    ]},
                    "finish_reason": None,
                }
            ],
        }
        assert t.translate_stream_chunk(args_chunk) == []

        events = t.translate_stream_chunk(
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        )
        function_calls = []
        for event in events:
            data = json.loads(event.removeprefix("data: ").removesuffix("\n\n"))
            for part in data["candidates"][0]["content"]["parts"]:
                if "functionCall" in part:
                    function_calls.append(part["functionCall"])

        assert len(function_calls) == 2
        assert function_calls[0].get("id") == "call_A"
        assert function_calls[0]["args"] == {"x": 1}
        assert function_calls[1].get("id") == "call_B"
        assert function_calls[1]["args"] == {"y": 2}


class TestTranslateResponseFinishReasons:
    """finish_reason → Gemini finishReason mapping."""

    def test_stop(self):
        t = GeminiTranslator()
        resp = t.translate_response(
            {
                "id": "x",
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        assert resp["candidates"][0]["finishReason"] == "STOP"

    def test_length(self):
        t = GeminiTranslator()
        resp = t.translate_response(
            {
                "id": "x",
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        assert resp["candidates"][0]["finishReason"] == "MAX_TOKENS"

    def test_tool_calls_maps_to_stop(self):
        t = GeminiTranslator()
        resp = t.translate_response(
            {
                "id": "x",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        assert resp["candidates"][0]["finishReason"] == "STOP"


class TestTranslateResponseUsage:
    """usage → usageMetadata mapping."""

    def test_usage_metadata(self):
        t = GeminiTranslator()
        resp = t.translate_response(
            {
                "id": "x",
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
        )
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
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events1 = t.translate_stream_chunk(chunk1)
        # No events yet — buffering starts
        assert events1 == []

        # Argument delta
        chunk2 = {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}, "finish_reason": None}
            ],
        }
        events2 = t.translate_stream_chunk(chunk2)
        assert events2 == []

        # More argument delta
        chunk3 = {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ation":"NYC"}'}}]},
                    "finish_reason": None,
                }
            ],
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


class TestGeminiToolChoice:
    """KBR-221: ``toolConfig.functionCallingConfig`` survives the Gemini -> CC hop.

    Gemini's ``mode`` (case-insensitive AUTO / ANY / NONE) maps onto CC strings;
    ``ANY`` plus a single ``allowedFunctionNames`` entry maps onto the CC named-function
    form; irreducible shapes (multi-name, AUTO / NONE + names, VALIDATED,
    MODE_UNSPECIFIED, non-list, empty) carry the mode only -- the restriction has no CC
    home (D5). Gemini carries no parallel-tool-use knob (KBR-205 / G36).
    """

    def setup_method(self):
        self.t = GeminiTranslator()

    def _req(self, **extra):
        """Build a minimal Gemini request with one tool, plus the case's fields."""
        req = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": {}}]}],
        }
        req.update(extra)
        return req

    @pytest.mark.parametrize(
        ("mode", "cc"),
        [("AUTO", "auto"), ("ANY", "required"), ("NONE", "none")],
        ids=["auto", "any", "none"],
    )
    def test_mode_is_carried(self, mode, cc):
        """AC-6: each published Gemini mode maps onto its CC spelling."""
        result = self.t.translate_request(
            self._req(toolConfig={"functionCallingConfig": {"mode": mode}})
        )
        assert result["tool_choice"] == cc

    @pytest.mark.parametrize(
        ("mode", "cc"),
        [
            ("auto", "auto"), ("Auto", "auto"), ("AUTO", "auto"), ("aUtO", "auto"),
            ("any", "required"), ("Any", "required"), ("ANY", "required"), ("aNy", "required"),
            ("none", "none"), ("NONE", "none"), ("NoNe", "none"),
        ],
        ids=[
            "auto-lower", "auto-title", "auto-upper", "auto-mixed",
            "any-lower", "any-title", "any-upper", "any-mixed",
            "none-lower", "none-upper", "none-mixed",
        ],
    )
    def test_mode_is_case_insensitive(self, mode, cc):
        """AC-6: every published Gemini mode is matched case-insensitively."""
        result = self.t.translate_request(
            self._req(toolConfig={"functionCallingConfig": {"mode": mode}})
        )
        assert result["tool_choice"] == cc

    def test_any_with_single_name_is_named_form(self):
        """AC-7: ANY + one name maps onto the CC named-function form."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["get_weather"],
                    }
                }
            )
        )
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_any_with_multi_names_carries_mode_only(self):
        """AC-8: ANY + >1 names carries the mode only (no CC form for a forced set)."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["get_weather", "search"],
                    }
                }
            )
        )
        assert result["tool_choice"] == "required"

    def test_any_with_empty_names_carries_mode_only(self):
        """AC-7b: ANY + empty names -> mode only (no specific name to carry)."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [],
                    }
                }
            )
        )
        assert result["tool_choice"] == "required"

    @pytest.mark.parametrize(
        "names",
        ["get_weather", 7, {"name": "get_weather"}],
        ids=["bare-string", "number", "object"],
    )
    def test_any_with_non_list_names_carries_mode_only(self, names):
        """AC-7c: ANY + non-list names carries the mode only (D5; reader residualises non-list values)."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": names,
                    }
                }
            )
        )
        assert result["tool_choice"] == "required"

    def test_auto_with_names_carries_mode_only(self):
        """AC-9: AUTO + any names carries the mode only."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "AUTO",
                        "allowedFunctionNames": ["get_weather", "search"],
                    }
                }
            )
        )
        assert result["tool_choice"] == "auto"

    def test_none_with_names_carries_mode_only(self):
        """AC-9: NONE + names carries the mode only (defensive)."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "NONE",
                        "allowedFunctionNames": ["get_weather"],
                    }
                }
            )
        )
        assert result["tool_choice"] == "none"

    def test_any_with_mixed_type_names_carries_mode_only(self):
        """AC-7c / AC-8 boundary: a names list with non-string members carries the mode only.

        The CC named-function form requires a string name; the representable
        half on this input is the mode, so the carrier mirrors the multi-name
        and non-list dispositions and emits ``required``.
        """
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["get_weather", 7],
                    }
                }
            )
        )
        assert result["tool_choice"] == "required"

    def test_any_with_single_non_string_name_carries_mode_only(self):
        """AC-7c / D5: a single-element non-string name carries the mode only.

        Kills a mutant that drops the ``isinstance(names[0], str)`` guard: with
        ``[7]``, ``len == 1`` would still pass the ``len(names) == 1`` check but
        ``names[0]`` is not a string, so the carrier must fall through to the
        mode-only branch.
        """
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [7],
                    }
                }
            )
        )
        assert result["tool_choice"] == "required"

    @pytest.mark.parametrize(
        "mode",
        ["VALIDATED", "MODE_UNSPECIFIED"],
        ids=["validated", "mode-unspecified"],
    )
    def test_validated_and_unspecified_are_omitted(self, mode):
        """AC-10: VALIDATED / MODE_UNSPECIFIED have no canonical mapping and are omitted (D5)."""
        result = self.t.translate_request(
            self._req(toolConfig={"functionCallingConfig": {"mode": mode}})
        )
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "mode",
        [7, None, []],
        ids=["int", "null", "list"],
    )
    def test_non_string_mode_is_omitted(self, mode):
        """AC-10: a non-string mode has no canonical mapping and is omitted (D5)."""
        result = self.t.translate_request(
            self._req(toolConfig={"functionCallingConfig": {"mode": mode}})
        )
        assert "tool_choice" not in result

    def test_absent_tool_config_invents_no_choice(self):
        """R9: no ``toolConfig`` in the inbound body -> no entry."""
        result = self.t.translate_request(self._req())
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "tool_config",
        [None, {}, {"reasoning": "x"}],
        ids=["null", "empty-object", "no-function-calling-config"],
    )
    def test_degenerate_tool_config_invents_no_choice(self, tool_config):
        """AC-10 / R9: a ``toolConfig`` without a readable ``functionCallingConfig`` carries nothing."""
        result = self.t.translate_request(self._req(toolConfig=tool_config))
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "tools_value",
        [None, [], [{"functionDeclarations": []}]],
        ids=["absent", "empty-list", "empty-declarations"],
    )
    def test_tool_choice_without_tools_is_omitted(self, tools_value):
        """AC-11 / D9: choice over no tools -- the gate reads the CC list, not the inbound key."""
        req = self._req(toolConfig={"functionCallingConfig": {"mode": "ANY"}})
        if tools_value is None:
            del req["tools"]
        else:
            req["tools"] = tools_value
        result = self.t.translate_request(req)
        assert "tool_choice" not in result

    def test_full_composition_carries_tools_and_config(self):
        """AC-14: tools + toolConfig reach the CC body together."""
        result = self.t.translate_request(
            self._req(
                toolConfig={
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["get_weather"],
                    }
                }
            )
        )
        assert [t["function"]["name"] for t in result["tools"]] == ["get_weather"]
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }
