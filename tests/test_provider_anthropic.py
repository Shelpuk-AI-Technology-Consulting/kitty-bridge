"""Tests for providers/anthropic.py — AnthropicAdapter."""

import json

from kitty.providers.anthropic import AnthropicAdapter

# ── CC format samples (what the bridge produces internally) ─────────────────

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

# ── Anthropic response samples ─────────────────────────────────────────────

ANTHROPIC_RESPONSE_TEXT = {
    "id": "msg_01ABC",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from Claude"}],
    "model": "claude-sonnet-4-6",
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
    "model": "claude-sonnet-4-6",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 50, "output_tokens": 30},
}

ANTHROPIC_RESPONSE_MAX_TOKENS = {
    "id": "msg_max",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Cut off..."}],
    "model": "claude-sonnet-4-6",
    "stop_reason": "max_tokens",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 100},
}


class TestAnthropicAdapterProperties:
    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "anthropic"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.anthropic.com"

    def test_upstream_path(self):
        assert self.adapter.upstream_path == "/v1/messages"


class TestAnthropicBuildUpstreamHeaders:
    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_uses_x_api_key(self):
        headers = self.adapter.build_upstream_headers("sk-ant-test123")
        assert headers["x-api-key"] == "sk-ant-test123"

    def test_includes_anthropic_version(self):
        headers = self.adapter.build_upstream_headers("sk-ant-test123")
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"

    def test_includes_content_type(self):
        headers = self.adapter.build_upstream_headers("sk-ant-test123")
        assert headers["content-type"] == "application/json"

    def test_no_bearer_auth(self):
        headers = self.adapter.build_upstream_headers("sk-ant-test123")
        assert "Authorization" not in headers


class TestAnthropicTranslateToUpstream:
    """CC request → Anthropic Messages API request."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_extracts_system_message(self):
        cc = {"model": "claude-sonnet-4-6", "messages": CC_MESSAGES_BASIC, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assert result["system"] == "You are helpful."
        # system message removed from messages array
        assert all(m["role"] != "system" for m in result["messages"])

    def test_no_system_message_passes_messages_through(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "system" not in result
        assert len(result["messages"]) == 1

    def test_max_tokens_default_when_missing(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 4096

    def test_max_tokens_preserved_when_set(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 2048,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 2048

    def test_string_content_stays_string(self):
        """Anthropic accepts string content — no conversion needed."""
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0]["content"] == "Hello"

    def test_tools_translated_to_anthropic_format(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": CC_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["name"] == "get_weather"
        assert "input_schema" in tool
        assert "parameters" not in tool

    def test_assistant_tool_calls_become_content_blocks(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": CC_MESSAGES_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        # Assistant message should have content blocks with tool_use
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"
        assert tool_use_blocks[0]["input"] == {"city": "London"}

    def test_tool_result_message_becomes_user_with_tool_result_block(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": CC_MESSAGES_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        # Tool result message should be user with tool_result content block
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_abc"
        assert tool_msg["content"][0]["content"] == "15°C, cloudy"

    def test_stream_passthrough(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["stream"] is True

    def test_temperature_passthrough(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["temperature"] == 0.5

    def test_top_p_passthrough(self):
        cc = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "top_p": 0.9,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["top_p"] == 0.9


class TestAnthropicTranslateFromUpstream:
    """Anthropic Messages API response → CC response."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_text_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["choices"][0]["message"]["content"] == "Hello from Claude"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_text_response_usage(self):
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
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "London"}'
        assert tc["id"] == "toolu_01ABC"

    def test_max_tokens_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_MAX_TOKENS)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_response_id_passthrough(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["id"] == "msg_01ABC"

    def test_model_passthrough(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["model"] == "claude-sonnet-4-6"


class TestAnthropicTranslateUpstreamStreamEvent:
    """Anthropic SSE events → CC SSE events for streaming pass-through.

    For the initial implementation, Anthropic streaming goes through the
    translate_to_upstream / translate_from_upstream path, converting
    the entire non-streaming flow.  The stream event translator is used
    when the bridge needs to translate individual SSE chunks.
    """

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_text_delta_becomes_cc_chunk(self):
        raw = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert len(chunks) >= 1
        # Should produce CC-format SSE data
        combined = b"".join(chunks)
        assert b"data:" in combined
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["delta"]["content"] == "Hello"

    def test_message_start_yields_role(self):
        raw = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01",'
            b'"role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["delta"]["role"] == "assistant"

    def test_message_delta_yields_finish_reason(self):
        raw = (
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
            b'"stop_sequence":null},"usage":{"output_tokens":15}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_ping_ignored(self):
        raw = b'event: ping\ndata: {"type":"ping"}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == []

    def test_content_block_start_ignored(self):
        raw = (
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == []

    def test_content_block_stop_ignored(self):
        raw = b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == []

    def test_message_stop_yields_done(self):
        raw = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        assert combined.strip() == b"data: [DONE]"

    def test_tool_use_delta_accumulates(self):
        raw = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"London\\"}"}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        # Tool use deltas are buffered until content_block_stop
        # For now they may be suppressed or buffered — we verify no crash
        assert isinstance(chunks, list)


class TestAnthropicMapError:
    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_401_error(self):
        exc = self.adapter.map_error(
            401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        )
        assert "401" in str(exc)

    def test_429_error(self):
        exc = self.adapter.map_error(
            429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        )
        assert "429" in str(exc)

    def test_500_error(self):
        exc = self.adapter.map_error(500, {"type": "error", "error": {"type": "api_error", "message": "overloaded"}})
        assert "500" in str(exc)


class TestAnthropicBuildRequest:
    """build_request should return a CC-format request (used by registry layer)."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        assert result["model"] == "claude-sonnet-4-6"
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["stream"] is False

    def test_build_request_with_max_tokens(self):
        result = self.adapter.build_request(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            max_tokens=8192,
        )
        assert result["max_tokens"] == 8192


class TestAnthropicParseResponse:
    """parse_response should parse a CC-format response (used by registry layer)."""

    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_parse_cc_response(self):
        cc_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"


class TestAnthropicNormalizeModelName:
    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_returns_unchanged(self):
        assert self.adapter.normalize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_strips_provider_prefix(self):
        """If someone passes openrouter-style prefix, strip it."""
        assert self.adapter.normalize_model_name("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"
