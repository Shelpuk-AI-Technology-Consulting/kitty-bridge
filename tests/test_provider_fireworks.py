"""Tests for providers/fireworks.py — FireworksAdapter."""

from kitty.providers.fireworks import FireworksAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Fireworks"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

SAMPLE_TOOL_RESPONSE = {
    "id": "chatcmpl-tool123",
    "object": "chat.completion",
    "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
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
                        "function": {"name": "read_file", "arguments": '{"path": "/tmp/test.py"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


class TestFireworksAdapter:
    def setup_method(self):
        self.adapter = FireworksAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "fireworks"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.fireworks.ai/inference/v1"

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "accounts/fireworks/routers/kimi-k2p5-turbo"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False
        assert "base_url" not in result

    def test_build_request_with_streaming(self):
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["stream"] is True

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools

    def test_build_request_with_temperature(self):
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=0.7,
        )
        assert result["temperature"] == 0.7

    def test_build_request_with_max_tokens(self):
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=False,
            max_tokens=4096,
        )
        assert result["max_tokens"] == 4096

    def test_build_request_with_top_p(self):
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=False,
            top_p=0.9,
        )
        assert result["top_p"] == 0.9

    def test_build_request_omits_none_kwargs(self):
        result = self.adapter.build_request(
            model="accounts/fireworks/routers/kimi-k2p5-turbo",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=None,
        )
        assert "temperature" not in result

    def test_parse_response_text(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from Fireworks"
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
        assert result["tool_calls"][0]["function"]["name"] == "read_file"

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
        assert (
            self.adapter.normalize_model_name("accounts/fireworks/routers/kimi-k2p5-turbo")
            == "accounts/fireworks/routers/kimi-k2p5-turbo"
        )

    def test_normalize_model_name_passthrough(self):
        """Fireworks uses full model paths — no prefix stripping needed."""
        assert self.adapter.normalize_model_name("some-model") == "some-model"

    def test_normalize_request_does_nothing_when_no_max_tokens(self):
        """No mutation when max_tokens is absent."""
        cc = {"model": "accounts/fireworks/routers/kimi-k2p5-turbo", "messages": SAMPLE_MESSAGES, "stream": False}
        self.adapter.normalize_request(cc)
        assert cc == {
            "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
            "messages": SAMPLE_MESSAGES,
            "stream": False,
        }

    def test_normalize_request_caps_max_tokens_non_streaming(self):
        """max_tokens > 4096 is capped for non-streaming requests."""
        cc = {
            "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
            "messages": SAMPLE_MESSAGES,
            "stream": False,
            "max_tokens": 16384,
        }
        self.adapter.normalize_request(cc)
        assert cc["max_tokens"] == 4096

    def test_normalize_request_no_cap_when_streaming(self):
        """max_tokens is not capped for streaming requests."""
        cc = {
            "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
            "messages": SAMPLE_MESSAGES,
            "stream": True,
            "max_tokens": 16384,
        }
        self.adapter.normalize_request(cc)
        assert cc["max_tokens"] == 16384

    def test_normalize_request_no_cap_when_max_tokens_below_limit(self):
        """max_tokens <= 4096 is not touched."""
        cc = {
            "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
            "messages": SAMPLE_MESSAGES,
            "stream": False,
            "max_tokens": 2048,
        }
        self.adapter.normalize_request(cc)
        assert cc["max_tokens"] == 2048

    def test_normalize_request_no_cap_when_max_tokens_exactly_4096(self):
        """max_tokens == 4096 is not touched."""
        cc = {
            "model": "accounts/fireworks/routers/kimi-k2p5-turbo",
            "messages": SAMPLE_MESSAGES,
            "stream": False,
            "max_tokens": 4096,
        }
        self.adapter.normalize_request(cc)
        assert cc["max_tokens"] == 4096
