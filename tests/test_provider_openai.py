"""Tests for providers/openai.py — OpenAIAdapter."""

from kitty.providers.openai import OpenAIAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenAI"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

SAMPLE_TOOL_RESPONSE = {
    "id": "chatcmpl-tool123",
    "object": "chat.completion",
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
                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


class TestOpenAIAdapter:
    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "openai"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.openai.com/v1"

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "gpt-4o"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False
        assert "base_url" not in result

    def test_build_request_with_streaming(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["stream"] is True

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools

    def test_build_request_with_temperature(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=0.7,
        )
        assert result["temperature"] == 0.7

    def test_build_request_with_max_tokens(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            max_tokens=4096,
        )
        assert result["max_tokens"] == 4096

    def test_build_request_with_top_p(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            top_p=0.9,
        )
        assert result["top_p"] == 0.9

    def test_build_request_omits_none_kwargs(self):
        result = self.adapter.build_request(
            model="gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=None,
        )
        assert "temperature" not in result

    def test_parse_response_text(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from OpenAI"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert "tool_calls" not in result

    def test_parse_response_tool_calls(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": {"message": "invalid api key", "type": "invalid_request_error"}})
        assert "401" in str(exc)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": "rate limited"})
        assert "429" in str(exc)

    def test_map_error_500(self):
        exc = self.adapter.map_error(500, {"error": "internal server error"})
        assert "500" in str(exc)

    def test_normalize_model_name_returns_unchanged(self):
        assert self.adapter.normalize_model_name("gpt-4o") == "gpt-4o"

    def test_normalize_model_name_strips_prefix(self):
        """OpenAI adapter strips provider prefix."""
        assert self.adapter.normalize_model_name("openai/gpt-4o") == "gpt-4o"
        assert self.adapter.normalize_model_name("openai/o3") == "o3"

    def test_normalize_model_name_no_prefix(self):
        """Model names without prefix pass through."""
        assert self.adapter.normalize_model_name("gpt-4.1") == "gpt-4.1"

    def test_normalize_request_does_nothing(self):
        """OpenAI is the canonical API — no request mutation needed."""
        cc = {"model": "gpt-4o", "messages": SAMPLE_MESSAGES, "stream": False}
        self.adapter.normalize_request(cc)
        assert cc == {"model": "gpt-4o", "messages": SAMPLE_MESSAGES, "stream": False}
