"""Tests for providers/novita.py — NovitaAdapter."""

from kitty.providers.novita import NovitaAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "meta-llama/llama-3.1-8b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Novita"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}


class TestNovitaAdapter:
    def setup_method(self):
        self.adapter = NovitaAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "novita"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.novita.ai/openai/v1"

    def test_endpoint_url_default_base(self):
        result = self.adapter.build_request(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "meta-llama/llama-3.1-8b-instruct"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False
        assert result.get("base_url") is None

    def test_endpoint_url_validated_override(self):
        result = self.adapter.build_request(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.novita.example.com/v1",
        )
        assert result["base_url"] == "https://custom.novita.example.com/v1"

    def test_request_building(self):
        result = self.adapter.build_request(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=SAMPLE_MESSAGES,
            stream=True,
            temperature=0.5,
        )
        assert result["model"] == "meta-llama/llama-3.1-8b-instruct"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True
        assert result["temperature"] == 0.5

    def test_parse_response(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from Novita"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 6
        assert result["usage"]["total_tokens"] == 18

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

    def test_normalize_model_name_strips_novita_prefix(self):
        assert self.adapter.normalize_model_name("novita/deepseek-r1-671b") == "deepseek-r1-671b"

    def test_normalize_model_name_strips_novita_prefix_case_insensitive(self):
        assert self.adapter.normalize_model_name("Novita/deepseek-r1-671b") == "deepseek-r1-671b"

    def test_normalize_model_name_no_prefix_unchanged(self):
        assert self.adapter.normalize_model_name("deepseek-r1-671b") == "deepseek-r1-671b"

    def test_normalize_model_name_empty_after_prefix(self):
        assert self.adapter.normalize_model_name("novita/") == "novita/"
