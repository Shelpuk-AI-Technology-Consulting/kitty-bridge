"""Tests for providers/zai.py — ZaiRegularAdapter and ZaiCodingAdapter."""

from kitty.providers.zai import ZaiCodingAdapter, ZaiRegularAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

SAMPLE_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1700000000,
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

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class TestZaiRegularAdapter:
    def setup_method(self):
        self.adapter = ZaiRegularAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "zai_regular"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.z.ai/api/paas/v4"

    def test_endpoint_url_default_base(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "gpt-4o"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False
        assert result.get("base_url") is None

    def test_endpoint_url_custom_base(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.api.example.com/v1",
        )
        assert result["base_url"] == "https://custom.api.example.com/v1"

    def test_request_building_includes_tools(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=SAMPLE_TOOLS,
        )
        assert result["tools"] == SAMPLE_TOOLS

    def test_parse_response_extracts_content(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_parse_response_extracts_tool_calls(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_CALL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["usage"]["total_tokens"] == 30

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert isinstance(exc, Exception)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": "rate limited"})
        assert isinstance(exc, Exception)

    def test_map_error_500(self):
        exc = self.adapter.map_error(500, {"error": "internal server error"})
        assert isinstance(exc, Exception)

    def test_normalize_model_name_strips_zai_prefix(self):
        assert self.adapter.normalize_model_name("zai/glm-5.1") == "glm-5.1"

    def test_normalize_model_name_strips_z_ai_prefix(self):
        assert self.adapter.normalize_model_name("z-ai/glm-5.1") == "glm-5.1"

    def test_normalize_model_name_case_insensitive(self):
        assert self.adapter.normalize_model_name("ZAI/glm-5.1") == "glm-5.1"
        assert self.adapter.normalize_model_name("Z-AI/glm-5.1") == "glm-5.1"

    def test_normalize_model_name_no_prefix_unchanged(self):
        assert self.adapter.normalize_model_name("glm-5.1") == "glm-5.1"


class TestZaiCodingAdapter:
    def setup_method(self):
        self.adapter = ZaiCodingAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "zai_coding"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.z.ai/api/coding/paas/v4"

    def test_endpoint_url_default_base(self):
        result = self.adapter.build_request(
            model="claude-3.5-sonnet",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["model"] == "claude-3.5-sonnet"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True
        assert result.get("base_url") is None

    def test_endpoint_url_custom_base(self):
        result = self.adapter.build_request(
            model="claude-3.5-sonnet",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.api.example.com/v1",
        )
        assert result["base_url"] == "https://custom.api.example.com/v1"

    def test_request_building_includes_tools(self):
        result = self.adapter.build_request(
            model="claude-3.5-sonnet",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=SAMPLE_TOOLS,
        )
        assert result["tools"] == SAMPLE_TOOLS

    def test_parse_response_extracts_content(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 15

    def test_parse_response_extracts_tool_calls(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_CALL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert isinstance(exc, Exception)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": "rate limited"})
        assert isinstance(exc, Exception)

    def test_map_error_500(self):
        exc = self.adapter.map_error(500, {"error": "internal server error"})
        assert isinstance(exc, Exception)

    def test_normalize_model_name_inherits_from_base(self):
        assert self.adapter.normalize_model_name("zai/glm-5.1") == "glm-5.1"
        assert self.adapter.normalize_model_name("glm-5.1") == "glm-5.1"
