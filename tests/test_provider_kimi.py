"""Tests for providers/kimi.py — KimiCodeAdapter."""

from kitty.providers.kimi import KimiCodeAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "kimi-for-coding",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Kimi Code"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
}

SAMPLE_TOOL_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "kimi-for-coding",
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
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/test.py"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}


class TestKimiCodeAdapter:
    def setup_method(self):
        self.adapter = KimiCodeAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "kimi"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.kimi.com/coding/v1"

    def test_validation_model_is_real_model(self):
        assert self.adapter.validation_model == "kimi-for-coding"

    def test_build_upstream_headers_includes_user_agent(self):
        headers = self.adapter.build_upstream_headers("sk-kimi-test123")
        assert headers["User-Agent"] == "claude-code/1.0"
        assert headers["Authorization"] == "Bearer sk-kimi-test123"
        assert headers["Content-Type"] == "application/json"

    def test_normalize_model_name_strips_kimi_prefix(self):
        assert self.adapter.normalize_model_name("kimi/kimi-for-coding") == "kimi-for-coding"

    def test_normalize_model_name_strips_kimi_prefix_case_insensitive(self):
        assert self.adapter.normalize_model_name("Kimi/Kimi-K2-Thinking") == "Kimi-K2-Thinking"

    def test_normalize_model_name_no_prefix_unchanged(self):
        assert self.adapter.normalize_model_name("kimi-for-coding") == "kimi-for-coding"

    def test_normalize_model_name_empty_after_prefix(self):
        assert self.adapter.normalize_model_name("kimi/") == "kimi/"

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="kimi-for-coding",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["model"] == "kimi-for-coding"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        result = self.adapter.build_request(
            model="kimi-for-coding",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools

    def test_build_request_with_optional_params(self):
        result = self.adapter.build_request(
            model="kimi-for-coding",
            messages=SAMPLE_MESSAGES,
            stream=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=32768,
        )
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 32768

    def test_build_request_with_custom_base_url(self):
        # F15: base_url must NOT be leaked into the Chat Completions body
        # — the CC upstream ignores the field and it can confuse third-party
        # request loggers.  The override is applied at the HTTP layer instead.
        result = self.adapter.build_request(
            model="kimi-for-coding",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.api.example.com/v1",
        )
        assert result.get("base_url") is None

    def test_parse_response(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from Kimi Code"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 8
        assert result["usage"]["completion_tokens"] == 4
        assert result["usage"]["total_tokens"] == 12

    def test_parse_response_with_tool_calls(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "read_file"

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert isinstance(exc, Exception)
        assert "Kimi Code error 401" in str(exc)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": "rate limited"})
        assert isinstance(exc, Exception)
        assert "Kimi Code error 429" in str(exc)

    def test_map_error_nested_error_object(self):
        exc = self.adapter.map_error(400, {"error": {"message": "bad request", "code": 1234}})
        assert isinstance(exc, Exception)
        assert "Kimi Code error 400" in str(exc)

    def test_map_error_empty_body(self):
        exc = self.adapter.map_error(500, {})
        assert isinstance(exc, Exception)
        assert "Kimi Code error 500" in str(exc)

    def test_parse_response_empty_choices(self):
        result = self.adapter.parse_response({"choices": [], "usage": {}})
        assert result["content"] is None
        assert result["finish_reason"] is None
        assert result["usage"] == {}

    def test_parse_response_usage_null(self):
        result = self.adapter.parse_response(
            {
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": None,
            }
        )
        assert result["content"] == "hi"
        assert result["usage"] == {}

    def test_build_request_temperature_zero(self):
        result = self.adapter.build_request(
            model="kimi-for-coding",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=0,
        )
        assert result["temperature"] == 0


class TestKimiThinkingEnabledToolCallGap:
    """Regression tests for: when thinking mode is enabled, assistant messages
    with tool_calls but no reasoning_content must get an empty reasoning_content
    injected.  Kimi rejects requests with:
      'thinking is enabled but reasoning_content is missing in assistant tool
       call message at index N'
    """

    def setup_method(self):
        self.adapter = KimiCodeAdapter()

    def test_thinking_enabled_tool_call_without_reasoning_gets_empty_reasoning(self):
        """Assistant message with tool_calls but no reasoning_content gets
        empty reasoning_content injected when _thinking_enabled is True."""
        cc = {
            "model": "kimi-for-coding",
            "messages": [
                {"role": "user", "content": "check the weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_001", "content": "Sunny, 22C"},
                {"role": "user", "content": "thanks"},
            ],
            "stream": True,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)

        # Internal keys must be stripped
        assert "_thinking_enabled" not in result

        # Assistant message with tool_calls must now have reasoning_content
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert "reasoning_content" in assistant_msg
        assert assistant_msg["reasoning_content"] == ""
        assert assistant_msg["tool_calls"] is not None

    def test_thinking_enabled_text_only_assistant_gets_empty_reasoning(self):
        """Assistant message with text content but no reasoning_content also
        gets empty reasoning_content when _thinking_enabled is True."""
        cc = {
            "model": "kimi-for-coding",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "bye"},
            ],
            "stream": False,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)

        assistant_msg = result["messages"][1]
        assert assistant_msg["reasoning_content"] == ""

    def test_existing_reasoning_content_not_overwritten(self):
        """Assistant messages that already have reasoning_content are left as-is."""
        cc = {
            "model": "kimi-for-coding",
            "messages": [
                {"role": "user", "content": "think about it"},
                {
                    "role": "assistant",
                    "content": "The answer is 42",
                    "reasoning_content": "I analyzed the problem...",
                },
                {"role": "user", "content": "thanks"},
            ],
            "stream": False,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)

        assistant_msg = result["messages"][1]
        assert assistant_msg["reasoning_content"] == "I analyzed the problem..."

    def test_no_injection_when_thinking_disabled(self):
        """No reasoning_content is injected when _thinking_enabled is not set."""
        cc = {
            "model": "kimi-for-coding",
            "messages": [
                {"role": "user", "content": "check"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_002",
                            "type": "function",
                            "function": {"name": "run", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_002", "content": "done"},
                {"role": "user", "content": "ok"},
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)

        assistant_msg = result["messages"][1]
        assert "reasoning_content" not in assistant_msg

    def test_non_assistant_messages_not_modified(self):
        """User and tool messages are never modified by reasoning_content injection."""
        cc = {
            "model": "kimi-for-coding",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi", "tool_calls": []},
                {"role": "tool", "tool_call_id": "t1", "content": "result"},
            ],
            "stream": False,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)

        assert result["messages"][0]["role"] == "user"
        assert "reasoning_content" not in result["messages"][0]
        assert result["messages"][2]["role"] == "tool"
        assert "reasoning_content" not in result["messages"][2]


class TestKimiDetectThinkingFromMessages:
    """F18: Kimi must detect prior reasoning_content and enable thinking automatically."""

    def test_detects_reasoning_in_prior_assistant_message(self):
        """When a prior assistant message has reasoning_content, thinking should be enabled."""
        adapter = KimiCodeAdapter()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "I thought about it"},
            {"role": "user", "content": "follow up"},
        ]
        request = adapter.build_request(model="test", messages=messages, stream=False)
        assert request.get("_thinking_enabled") is True

    def test_no_thinking_when_no_prior_reasoning(self):
        """When no prior message has reasoning_content, thinking should NOT be auto-enabled."""
        adapter = KimiCodeAdapter()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "follow up"},
        ]
        request = adapter.build_request(model="test", messages=messages, stream=False)
        assert "_thinking_enabled" not in request or request.get("_thinking_enabled") is False

    def test_detects_reasoning_in_middle_message(self):
        """reasoning_content in any assistant turn triggers detection."""
        adapter = KimiCodeAdapter()
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "r1"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        request = adapter.build_request(model="test", messages=messages, stream=False)
        assert request.get("_thinking_enabled") is True

    def test_explicit_thinking_enabled_not_overridden(self):
        """If thinking is already explicitly enabled, detection should not interfere."""
        adapter = KimiCodeAdapter()
        messages = [
            {"role": "user", "content": "hello"},
        ]
        request = adapter.build_request(
            model="test", messages=messages, stream=False,
            thinking_enabled=True,
        )
        assert request.get("_thinking_enabled") is True

    def test_empty_reasoning_content_does_not_trigger(self):
        """Empty string reasoning_content should not trigger thinking."""
        adapter = KimiCodeAdapter()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": ""},
            {"role": "user", "content": "follow up"},
        ]
        request = adapter.build_request(model="test", messages=messages, stream=False)
        assert "_thinking_enabled" not in request or request.get("_thinking_enabled") is False
