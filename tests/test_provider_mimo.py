"""Tests for providers/mimo.py — MimoAdapter."""

from kitty.providers.mimo import MimoAdapter

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1749110144,
    "model": "mimo-v2-pro",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from MiMo"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}


class TestMimoAdapter:
    def setup_method(self):
        self.adapter = MimoAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "mimo"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://api.xiaomimimo.com/v1"

    def test_validation_model(self):
        assert self.adapter.validation_model == "mimo-v2-pro"

    def test_upstream_path(self):
        assert self.adapter.upstream_path == "/chat/completions"

    # --- build_upstream_headers ---

    def test_build_upstream_headers_uses_api_key_header(self):
        headers = self.adapter.build_upstream_headers("sk-test-key")
        assert headers["api-key"] == "sk-test-key"
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers
        # MiMo, like BytePlus and Kimi, requires a coding-agent User-Agent
        assert headers["User-Agent"] == "claude-code/1.0"

    def test_build_upstream_headers_with_token_plan_key(self):
        headers = self.adapter.build_upstream_headers("tp-test-key")
        assert headers["api-key"] == "tp-test-key"
        assert headers["User-Agent"] == "claude-code/1.0"

    # --- build_request ---

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="mimo-v2-pro",
            messages=SAMPLE_MESSAGES,
            stream=False,
        )
        assert result["model"] == "mimo-v2-pro"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is False

    def test_build_request_ignores_base_url_override(self):
        result = self.adapter.build_request(
            model="mimo-v2-pro",
            messages=SAMPLE_MESSAGES,
            stream=False,
            base_url="https://token-plan-cn.xiaomimimo.com/v1",
        )
        assert "base_url" not in result

    def test_build_request_with_stream_and_temperature(self):
        result = self.adapter.build_request(
            model="mimo-v2-pro",
            messages=SAMPLE_MESSAGES,
            stream=True,
            temperature=0.7,
        )
        assert result["stream"] is True
        assert result["temperature"] == 0.7

    def test_build_request_with_optional_params(self):
        result = self.adapter.build_request(
            model="mimo-v2-pro",
            messages=SAMPLE_MESSAGES,
            stream=True,
            temperature=0.5,
            top_p=0.95,
            max_tokens=4096,
        )
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.95
        assert result["max_tokens"] == 4096

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        result = self.adapter.build_request(
            model="mimo-v2-pro",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools

    def test_build_request_ignores_none_params(self):
        result = self.adapter.build_request(
            model="mimo-v2-pro",
            messages=SAMPLE_MESSAGES,
            stream=False,
            temperature=None,
            top_p=None,
        )
        assert "temperature" not in result
        assert "top_p" not in result

    # --- parse_response ---

    def test_parse_response(self):
        result = self.adapter.parse_response(SAMPLE_RESPONSE)
        assert result["content"] == "Hello from MiMo"
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
            "choices": [{"index": 0}],
        }
        result = self.adapter.parse_response(response)
        assert result["content"] is None
        assert result["finish_reason"] is None
        assert result["usage"] == {}

    def test_parse_response_empty_choices(self):
        """Upstream returns HTTP 200 with empty choices list — return empty result."""
        response = {
            "id": "test-id",
            "choices": [],
        }
        result = self.adapter.parse_response(response)
        assert result["content"] is None
        assert result["finish_reason"] is None
        assert result["usage"] == {}

    # --- map_error ---

    def test_map_error_400(self):
        exc = self.adapter.map_error(400, {"error": {"message": "bad request"}})
        assert isinstance(exc, Exception)
        assert "MiMo" in str(exc)
        assert "400" in str(exc)

    def test_map_error_401(self):
        exc = self.adapter.map_error(401, {"error": {"message": "unauthorized"}})
        assert isinstance(exc, Exception)
        assert "MiMo" in str(exc)
        assert "401" in str(exc)

    def test_map_error_429(self):
        exc = self.adapter.map_error(429, {"error": {"message": "rate limited"}})
        assert isinstance(exc, Exception)
        assert "MiMo" in str(exc)
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

    # --- normalize_model_name ---

    def test_normalize_model_name_strips_mimo_prefix(self):
        assert self.adapter.normalize_model_name("mimo/mimo-v2-pro") == "mimo-v2-pro"

    def test_normalize_model_name_strips_prefix_case_insensitive(self):
        assert self.adapter.normalize_model_name("MiMo/mimo-v2-pro") == "mimo-v2-pro"
        assert self.adapter.normalize_model_name("MIMO/mimo-v2-pro") == "mimo-v2-pro"

    def test_normalize_model_name_no_prefix_unchanged(self):
        assert self.adapter.normalize_model_name("mimo-v2-pro") == "mimo-v2-pro"

    def test_normalize_model_name_empty_after_prefix(self):
        assert self.adapter.normalize_model_name("mimo/") == "mimo/"

    # --- inherited passthrough methods ---

    def test_translate_from_upstream_passthrough(self):
        response = {"status_code": 200, "body": SAMPLE_RESPONSE}
        result = self.adapter.translate_from_upstream(response)
        assert result == response
