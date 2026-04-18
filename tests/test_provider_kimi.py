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
        result = self.adapter.build_request(
            model="kimi-for-coding",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.api.example.com/v1",
        )
        assert result["base_url"] == "https://custom.api.example.com/v1"

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
        result = self.adapter.parse_response({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": None,
        })
        assert result["content"] == "hi"
        assert result["usage"] == {}

    def test_build_request_temperature_zero(self):
        result = self.adapter.build_request(
            model="kimi-for-coding", messages=SAMPLE_MESSAGES, stream=False, temperature=0,
        )
        assert result["temperature"] == 0
