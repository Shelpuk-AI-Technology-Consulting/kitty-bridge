"""Tests for providers/opencode.py — OpenCodeGoAdapter with auto-routing."""

import json

from kitty.providers.opencode import OpenCodeGoAdapter

# ── CC format samples ──────────────────────────────────────────────────────

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_CC_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "glm-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenCode Go"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
}

SAMPLE_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "kimi-k2.5",
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

# ── Anthropic-format samples ───────────────────────────────────────────────

CC_MESSAGES_BASIC = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

CC_MESSAGES_TOOLS = [
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "15°C, cloudy",
    },
]

CC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

ANTHROPIC_RESPONSE_TEXT = {
    "id": "msg_01ABC",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from OpenCode Go Messages"}],
    "model": "minimax-m2.7",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 25, "output_tokens": 10},
}

ANTHROPIC_RESPONSE_TOOL_USE = {
    "id": "msg_tool123",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_01ABC",
            "name": "get_weather",
            "input": {"city": "London"},
        },
    ],
    "model": "minimax-m2.7",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 50, "output_tokens": 30},
}

ANTHROPIC_RESPONSE_MAX_TOKENS = {
    "id": "msg_max",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Cut off..."}],
    "model": "minimax-m2.7",
    "stop_reason": "max_tokens",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 100},
}


# ═══════════════════════════════════════════════════════════════════════════
# Properties and identity
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoAdapterProperties:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "opencode_go"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://opencode.ai/zen/go"

    def test_default_upstream_path_is_chat_completions(self):
        assert self.adapter.upstream_path == "/v1/chat/completions"


class TestOpenCodeGoNormalizeModelName:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_returns_unchanged(self):
        assert self.adapter.normalize_model_name("glm-5") == "glm-5"
        assert self.adapter.normalize_model_name("kimi-k2.5") == "kimi-k2.5"

    def test_strips_provider_prefix(self):
        assert self.adapter.normalize_model_name("opencode/glm-5") == "glm-5"


# ═══════════════════════════════════════════════════════════════════════════
# Auto-routing: get_upstream_path
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoAutoRouting:
    """Verify that the adapter routes to the correct endpoint based on model name."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_glm5_routes_to_chat_completions(self):
        assert self.adapter.get_upstream_path("glm-5") == "/v1/chat/completions"

    def test_glm51_routes_to_chat_completions(self):
        assert self.adapter.get_upstream_path("glm-5.1") == "/v1/chat/completions"

    def test_kimi_routes_to_chat_completions(self):
        assert self.adapter.get_upstream_path("kimi-k2.5") == "/v1/chat/completions"

    def test_mimo_pro_routes_to_chat_completions(self):
        assert self.adapter.get_upstream_path("mimo-v2-pro") == "/v1/chat/completions"

    def test_mimo_omni_routes_to_chat_completions(self):
        assert self.adapter.get_upstream_path("mimo-v2-omni") == "/v1/chat/completions"

    def test_minimax_m25_routes_to_messages(self):
        assert self.adapter.get_upstream_path("minimax-m2.5") == "/v1/messages"

    def test_minimax_m27_routes_to_messages(self):
        assert self.adapter.get_upstream_path("minimax-m2.7") == "/v1/messages"

    def test_unknown_model_routes_to_chat_completions(self):
        """Unknown models default to Chat Completions."""
        assert self.adapter.get_upstream_path("some-future-model") == "/v1/chat/completions"


# ═══════════════════════════════════════════════════════════════════════════
# Auto-routing: build_upstream_headers_for_model
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoHeadersRouting:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_default_headers_are_bearer(self):
        headers = self.adapter.build_upstream_headers("sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_chat_completions_model_gets_bearer(self):
        headers = self.adapter.build_upstream_headers_for_model("sk-test", "glm-5")
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers

    def test_messages_model_gets_x_api_key(self):
        headers = self.adapter.build_upstream_headers_for_model("sk-test", "minimax-m2.7")
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"
        assert "Authorization" not in headers

    def test_minimax_m25_gets_messages_headers(self):
        headers = self.adapter.build_upstream_headers_for_model("sk-test", "minimax-m2.5")
        assert headers["x-api-key"] == "sk-test"
        assert "Authorization" not in headers


# ═══════════════════════════════════════════════════════════════════════════
# Chat Completions passthrough
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoChatCompletionsPassthrough:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_translate_to_upstream_passthrough_for_cc_model(self):
        cc = {"model": "glm-5", "messages": SAMPLE_MESSAGES, "stream": True}
        result = self.adapter.translate_to_upstream(cc)
        assert result == cc

    def test_translate_from_upstream_passthrough_for_cc_response(self):
        resp = SAMPLE_CC_RESPONSE
        result = self.adapter.translate_from_upstream(resp)
        assert result is resp

    def test_translate_upstream_stream_event_passthrough_for_cc(self):
        chunk = b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}\n\n'
        result = self.adapter.translate_upstream_stream_event(chunk)
        assert result == [chunk]


class TestOpenCodeGoBuildRequest:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="glm-5",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["model"] == "glm-5"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        result = self.adapter.build_request(
            model="kimi-k2.5",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools


class TestOpenCodeGoParseResponse:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_parse_cc_response(self):
        result = self.adapter.parse_response(SAMPLE_CC_RESPONSE)
        assert result["content"] == "Hello from OpenCode Go"
        assert result["finish_reason"] == "stop"

    def test_parse_tool_call_response(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_CALL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"


class TestOpenCodeGoMapError:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_error_format(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert "OpenCode Go error 401" in str(exc)


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic Messages translation (auto-routed for minimax models)
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoMessagesTranslateToUpstream:
    """CC request → Anthropic Messages API (auto-routed by model name)."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_minimax_model_triggers_anthropic_translation(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_BASIC, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        # Anthropic format has system at top level, not in messages
        assert "system" in result
        assert result["system"] == "You are helpful."

    def test_extracts_system_message(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_BASIC, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assert result["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in result["messages"])

    def test_no_system_message(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "system" not in result

    def test_max_tokens_default(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 4096

    def test_tools_translated(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": CC_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["tools"]) == 1
        assert "input_schema" in result["tools"][0]
        assert "parameters" not in result["tools"][0]

    def test_assistant_tool_calls_become_content_blocks(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_TOOLS, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][1]
        tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"

    def test_tool_result_becomes_user_tool_result_block(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_TOOLS, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"

    def test_temperature_passthrough(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["temperature"] == 0.5

    def test_cc_model_is_not_translated(self):
        """Non-messages models should pass through without Anthropic translation."""
        cc = {"model": "glm-5", "messages": CC_MESSAGES_BASIC, "stream": True}
        result = self.adapter.translate_to_upstream(cc)
        assert result == cc  # same content, passthrough


class TestOpenCodeGoMessagesTranslateFromUpstream:
    """Anthropic Messages API response → CC response."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_text_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["choices"][0]["message"]["content"] == "Hello from OpenCode Go Messages"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["usage"]["prompt_tokens"] == 25
        assert result["usage"]["completion_tokens"] == 10
        assert result["usage"]["total_tokens"] == 35

    def test_tool_use_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TOOL_USE)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Let me check."
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_max_tokens_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_MAX_TOKENS)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_cc_response_passthrough(self):
        """CC format responses should pass through without translation."""
        result = self.adapter.translate_from_upstream(SAMPLE_CC_RESPONSE)
        assert result is SAMPLE_CC_RESPONSE


class TestOpenCodeGoMessagesStreamEventAutoDetection:
    """Stream events auto-detected by format (not model name)."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_anthropic_text_delta_translated(self):
        raw = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["delta"]["content"] == "Hello"

    def test_anthropic_message_start_translated(self):
        raw = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01",'
            b'"role":"assistant","content":[],"model":"minimax-m2.7","stop_reason":null}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["delta"]["role"] == "assistant"

    def test_anthropic_message_delta_translated(self):
        raw = (
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
            b'"stop_sequence":null},"usage":{"output_tokens":15}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_anthropic_ping_ignored(self):
        raw = b'event: ping\ndata: {"type":"ping"}\n\n'
        assert self.adapter.translate_upstream_stream_event(raw) == []

    def test_anthropic_content_block_start_ignored(self):
        raw = (
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
        )
        assert self.adapter.translate_upstream_stream_event(raw) == []

    def test_anthropic_content_block_stop_ignored(self):
        raw = b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        assert self.adapter.translate_upstream_stream_event(raw) == []

    def test_anthropic_message_stop_yields_done(self):
        raw = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert b"".join(chunks).strip() == b"data: [DONE]"

    def test_cc_chunk_passthrough(self):
        """Chat Completions SSE events should pass through unchanged."""
        raw = b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == [raw]

    def test_done_event_passthrough(self):
        raw = b"data: [DONE]\n\n"
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == [raw]

    def test_empty_bytes(self):
        assert self.adapter.translate_upstream_stream_event(b"") == []

    def test_tool_use_delta_no_crash(self):
        raw = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"London\\"}"}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert isinstance(chunks, list)
