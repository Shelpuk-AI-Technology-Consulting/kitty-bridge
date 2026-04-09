"""Tests for providers/minimax.py — MiniMaxAdapter."""

from kitty.providers.minimax import MiniMaxAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "MiniMax-Text-01",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from MiniMax"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
}


class TestMiniMaxAdapter:
    def setup_method(self):
        self.adapter = MiniMaxAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "minimax"

    def test_default_base_url_global(self):
        assert self.adapter.default_base_url == "https://api.minimax.io/v1"

    def test_endpoint_url_global_region(self):
        result = self.adapter.build_request(
            model="MiniMax-Text-01",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "MiniMax-Text-01"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False
        # Global region should use the default base_url
        assert result.get("base_url") is None

    def test_endpoint_url_cn_region(self):
        result = self.adapter.build_request(
            model="MiniMax-Text-01",
            messages=SAMPLE_MESSAGES,
            stream=False,
            provider_config={"region": "cn"},
        )
        assert result["model"] == "MiniMax-Text-01"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["base_url"] == "https://api.minimaxi.com/v1"

    def test_request_building_with_custom_base(self):
        result = self.adapter.build_request(
            model="MiniMax-Text-01",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.api.example.com/v1",
        )
        assert result["base_url"] == "https://custom.api.example.com/v1"

    def test_request_building(self):
        result = self.adapter.build_request(
            model="MiniMax-Text-01",
            messages=SAMPLE_MESSAGES,
            stream=True,
            temperature=0.7,
        )
        assert result["model"] == "MiniMax-Text-01"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True
        assert result["temperature"] == 0.7

    def test_parse_response(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from MiniMax"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 8
        assert result["usage"]["completion_tokens"] == 4
        assert result["usage"]["total_tokens"] == 12

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert isinstance(exc, Exception)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": "rate limited"})
        assert isinstance(exc, Exception)

    def test_map_error_500(self):
        exc = self.adapter.map_error(500, {"error": "internal server error"})
        assert isinstance(exc, Exception)

    def test_map_error_400(self):
        exc = self.adapter.map_error(400, {"error": "bad request"})
        assert isinstance(exc, Exception)

    def test_normalize_model_name_strips_minimax_prefix(self):
        assert self.adapter.normalize_model_name("minimax/minimax-m2.7") == "minimax-m2.7"

    def test_normalize_model_name_strips_minimax_prefix_case_insensitive(self):
        assert self.adapter.normalize_model_name("MiniMax/MiniMax-M2.7") == "MiniMax-M2.7"

    def test_normalize_model_name_no_prefix_unchanged(self):
        assert self.adapter.normalize_model_name("MiniMax-Text-01") == "MiniMax-Text-01"

    def test_normalize_model_name_empty_after_prefix(self):
        assert self.adapter.normalize_model_name("minimax/") == "minimax/"

    def test_normalize_request_adds_reasoning_split(self):
        cc = {"model": "minimax-m2.7", "messages": [], "stream": True}
        self.adapter.normalize_request(cc)
        assert cc["reasoning_split"] is True

    def test_normalize_request_preserves_existing_keys(self):
        cc = {"model": "minimax-m2.7", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        self.adapter.normalize_request(cc)
        assert cc["model"] == "minimax-m2.7"
        assert len(cc["messages"]) == 1
        assert cc["reasoning_split"] is True
