"""Tests for providers/google_aistudio.py — GoogleAIStudioAdapter."""

from kitty.providers.google_aistudio import GoogleAIStudioAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "test-id",
    "object": "chat.completion",
    "created": 1749110144,
    "model": "gemini-2.5-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Gemini"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}


class TestGoogleAIStudioAdapter:
    def setup_method(self):
        self.adapter = GoogleAIStudioAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "google_aistudio"

    def test_default_base_url(self):
        assert (
            self.adapter.default_base_url
            == "https://generativelanguage.googleapis.com/v1beta/openai"
        )

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="gemini-2.5-flash",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "gemini-2.5-flash"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False

    def test_build_request_with_stream(self):
        result = self.adapter.build_request(
            model="gemini-3-flash-preview",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["stream"] is True

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        result = self.adapter.build_request(
            model="gemini-2.5-flash",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools

    def test_build_request_with_optional_params(self):
        result = self.adapter.build_request(
            model="gemini-2.5-flash",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=0.5,
            top_p=0.9,
            max_tokens=2048,
        )
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 2048

    def test_build_request_ignores_none_params(self):
        result = self.adapter.build_request(
            model="gemini-2.5-flash",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=None,
            top_p=None,
            max_tokens=None,
        )
        assert "temperature" not in result
        assert "top_p" not in result
        assert "max_tokens" not in result

    def test_build_request_ignores_base_url_override(self):
        result = self.adapter.build_request(
            model="gemini-2.5-flash",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://custom.example.com/v1",
        )
        assert "base_url" not in result

    def test_parse_response(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from Gemini"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 6
        assert result["usage"]["total_tokens"] == 18

    def test_parse_response_with_tool_calls(self):
        response = {
            "id": "test-id",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(response)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "read_file"

    def test_parse_response_null_usage(self):
        response = {
            "id": "test-id",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,
        }
        result = self.adapter.parse_response(response)
        assert result["usage"] == {}

    def test_parse_response_missing_message_in_choice(self):
        response = {
            "id": "test-id",
            "choices": [{"index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = self.adapter.parse_response(response)
        assert result["content"] is None
        assert result["finish_reason"] == "stop"

    def test_normalize_request_does_nothing(self):
        cc = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        self.adapter.normalize_request(cc)
        assert cc == {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

    def test_translate_to_upstream_is_passthrough(self):
        cc = {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}
        assert self.adapter.translate_to_upstream(cc) == cc

    def test_translate_from_upstream_is_passthrough(self):
        raw = {"id": "test", "choices": [], "usage": {}}
        assert self.adapter.translate_from_upstream(raw) == raw

    def test_map_error_400(self):
        exc = self.adapter.map_error(400, {"error": {"message": "bad request"}})
        assert isinstance(exc, Exception)
        assert "Google AI Studio" in str(exc)
        assert "400" in str(exc)

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": {"message": "unauthorized"}})
        assert isinstance(exc, Exception)
        assert "Google AI Studio" in str(exc)
        assert "401" in str(exc)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": {"message": "rate limited"}})
        assert isinstance(exc, Exception)
        assert "Google AI Studio" in str(exc)
        assert "429" in str(exc)

    def test_map_error_500(self):
        exc = self.adapter.map_error(500, {"error": {"message": "internal server error"}})
        assert isinstance(exc, Exception)

    def test_map_error_string_error(self):
        exc = self.adapter.map_error(500, {"error": "plain string error"})
        assert isinstance(exc, Exception)
        assert "plain string error" in str(exc)

    def test_map_error_dict_without_message(self):
        exc = self.adapter.map_error(500, {"error": {"code": "1214", "details": "something"}})
        assert isinstance(exc, Exception)
        assert "Unknown error" in str(exc)

    def test_map_error_no_error_field(self):
        exc = self.adapter.map_error(500, {})
        assert isinstance(exc, Exception)
        assert "Unknown error" in str(exc)

    def test_normalize_model_name_strips_prefix(self):
        assert (
            self.adapter.normalize_model_name("google_aistudio/gemini-2.5-flash")
            == "gemini-2.5-flash"
        )

    def test_normalize_model_name_strips_prefix_case_insensitive(self):
        assert (
            self.adapter.normalize_model_name("GOOGLE_AISTUDIO/gemini-pro")
            == "gemini-pro"
        )

    def test_normalize_model_name_no_prefix_unchanged(self):
        assert self.adapter.normalize_model_name("gemini-2.5-flash") == "gemini-2.5-flash"

    def test_normalize_model_name_empty_after_prefix(self):
        assert self.adapter.normalize_model_name("google_aistudio/") == "google_aistudio/"

    def test_upstream_path(self):
        assert self.adapter.upstream_path == "/chat/completions"

    def test_validation_model(self):
        assert self.adapter.validation_model == "gemini-2.5-flash"

    def test_build_upstream_headers(self):
        headers = self.adapter.build_upstream_headers("sk-test-key")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"
