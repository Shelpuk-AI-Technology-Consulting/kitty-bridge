"""Tests for providers/openrouter.py — OpenRouterAdapter."""

from kitty.providers.openrouter import OpenRouterAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "openai/gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenRouter"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
}

SAMPLE_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "openai/gpt-4o",
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


class TestOpenRouterAdapter:
    def setup_method(self):
        self.adapter = OpenRouterAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "openrouter"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://openrouter.ai/api/v1"

    def test_build_request_passes_model_through(self):
        result = self.adapter.build_request(
            model="openai/gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["model"] == "openai/gpt-4o"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True
        assert result.get("base_url") is None

    def test_build_request_with_custom_base(self):
        result = self.adapter.build_request(
            model="openai/gpt-4o",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.example.com/v1",
        )
        assert result["base_url"] == "https://custom.example.com/v1"

    def test_build_request_with_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        result = self.adapter.build_request(
            model="anthropic/claude-3.5-sonnet",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools

    def test_build_request_with_temperature(self):
        result = self.adapter.build_request(
            model="google/gemma-3-12b-it",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=0.7,
        )
        assert result["temperature"] == 0.7

    def test_parse_response_extracts_content(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from OpenRouter"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 8
        assert result["usage"]["total_tokens"] == 23

    def test_parse_response_extracts_tool_calls(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_CALL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["usage"]["total_tokens"] == 30

    def test_map_error(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert isinstance(exc, Exception)

    def test_normalize_model_name_returns_unchanged(self):
        """OpenRouter needs the full provider/model format."""
        assert self.adapter.normalize_model_name("openai/gpt-4o") == "openai/gpt-4o"
        assert self.adapter.normalize_model_name("minimax/minimax-m2.7") == "minimax/minimax-m2.7"
        assert self.adapter.normalize_model_name("gpt-4o") == "gpt-4o"
