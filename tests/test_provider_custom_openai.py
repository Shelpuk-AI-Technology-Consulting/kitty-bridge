"""Tests for Custom OpenAI-compatible provider adapter."""

from kitty.providers.custom_openai import CustomOpenAIAdapter, ProviderError


class TestCustomOpenAIAdapter:
    """Test suite for CustomOpenAIAdapter."""

    def test_instantiation(self):
        adapter = CustomOpenAIAdapter()
        assert adapter is not None

    def test_provider_type(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.provider_type == "custom_openai"

    def test_default_base_url(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.default_base_url == "https://api.openai.com/v1"

    def test_upstream_path(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.upstream_path == "/chat/completions"

    def test_requires_custom_url(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.requires_custom_url is True

    def test_use_custom_transport_false(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.use_custom_transport is False


class TestCustomOpenAIBuildBaseUrl:
    def test_returns_config_url(self):
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url({"base_url": "https://api.deepseek.com/v1"})
        assert url == "https://api.deepseek.com/v1"

    def test_returns_config_url_http(self):
        """HTTP URLs allowed via provider_config (unlike Profile.base_url which is HTTPS-only)."""
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url({"base_url": "http://localhost:8000/v1"})
        assert url == "http://localhost:8000/v1"

    def test_falls_back_to_default(self):
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url({})
        assert url == "https://api.openai.com/v1"

    def test_falls_back_on_none_config(self):
        adapter = CustomOpenAIAdapter()
        url = adapter.build_base_url(None)
        assert url == "https://api.openai.com/v1"


class TestCustomOpenAIHeaders:
    def test_bearer_auth(self):
        adapter = CustomOpenAIAdapter()
        headers = adapter.build_upstream_headers("sk-test-key-123")
        assert headers["Authorization"] == "Bearer sk-test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_different_keys_different_headers(self):
        adapter = CustomOpenAIAdapter()
        h1 = adapter.build_upstream_headers("key-a")
        h2 = adapter.build_upstream_headers("key-b")
        assert h1["Authorization"] != h2["Authorization"]


class TestCustomOpenAIBuildRequest:
    def test_basic_request(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request("deepseek-chat", [{"role": "user", "content": "hi"}])
        assert req == {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

    def test_streaming(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request("deepseek-chat", [], stream=True)
        assert req["stream"] is True

    def test_optional_params(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request(
            "deepseek-chat",
            [],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        assert req["temperature"] == 0.7
        assert req["top_p"] == 0.9
        assert req["max_tokens"] == 100

    def test_tools(self):
        adapter = CustomOpenAIAdapter()
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = adapter.build_request("deepseek-chat", [], tools=tools)
        assert req["tools"] == tools

    def test_omits_none_params(self):
        adapter = CustomOpenAIAdapter()
        req = adapter.build_request("deepseek-chat", [], temperature=None, max_tokens=None)
        assert "temperature" not in req
        assert "max_tokens" not in req


class TestCustomOpenAIParseResponse:
    def test_text_response(self):
        adapter = CustomOpenAIAdapter()
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = adapter.parse_response(resp)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_tool_calls(self):
        adapter = CustomOpenAIAdapter()
        tool_calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        resp = {
            "choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": "tool_calls"}],
        }
        result = adapter.parse_response(resp)
        assert result["tool_calls"] == tool_calls

    def test_missing_choices(self):
        adapter = CustomOpenAIAdapter()
        result = adapter.parse_response({})
        assert result["content"] is None

    def test_empty_choices(self):
        adapter = CustomOpenAIAdapter()
        result = adapter.parse_response({"choices": []})
        assert result["content"] is None


class TestCustomOpenAITranslation:
    def test_translate_to_upstream_strips_internal_keys(self):
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-chat",
            "messages": [],
            "_resolved_key": "secret",
            "_provider_config": {},
            "_original_body": {},
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(req)
        for key in adapter._INTERNAL_KEYS:
            assert key not in result
        assert "model" in result
        assert "messages" in result

    def test_translate_to_upstream_preserves_other_fields(self):
        adapter = CustomOpenAIAdapter()
        req = {"model": "m", "messages": [], "temperature": 0.5, "tools": []}
        result = adapter.translate_to_upstream(req)
        assert result == {"model": "m", "messages": [], "temperature": 0.5, "tools": []}

    def test_translate_to_upstream_injects_empty_reasoning_content_when_thinking_enabled(self):
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "assistant", "content": "done", "reasoning_content": "thoughts"},
            ],
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": ""},
            {"role": "assistant", "content": "done", "reasoning_content": "thoughts"},
        ]

    def test_translate_to_upstream_injects_when_reasoning_effort_only(self):
        """Activation via _reasoning_effort without explicit _thinking_enabled."""
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "assistant", "content": "hello"}],
            "_reasoning_effort": "high",
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [{"role": "assistant", "content": "hello", "reasoning_content": ""}]

    def test_translate_to_upstream_does_not_inject_reasoning_content_when_thinking_disabled(self):
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "assistant", "content": "hello"}],
            "_reasoning_effort": "none",
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_translate_to_upstream_no_injection_without_thinking_signal(self):
        """No injection when neither _thinking_enabled nor _reasoning_effort is set."""
        adapter = CustomOpenAIAdapter()
        req = {
            "model": "deepseek-v4-pro",
            "messages": [{"role": "assistant", "content": "hello"}],
        }
        result = adapter.translate_to_upstream(req)
        assert result["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_translate_from_upstream_passthrough(self):
        adapter = CustomOpenAIAdapter()
        resp = {"choices": [{"message": {"content": "hi"}}]}
        assert adapter.translate_from_upstream(resp) is resp

    def test_translate_upstream_stream_event_passthrough(self):
        adapter = CustomOpenAIAdapter()
        chunk = b'data: {"choices": []}\n\n'
        assert adapter.translate_upstream_stream_event(chunk) == [chunk]


class TestCustomOpenAIErrorMapping:
    def test_dict_error(self):
        adapter = CustomOpenAIAdapter()
        err = adapter.map_error(401, {"error": {"message": "invalid api key"}})
        assert isinstance(err, ProviderError)
        assert "401" in str(err)
        assert "invalid api key" in str(err)

    def test_nested_error(self):
        adapter = CustomOpenAIAdapter()
        err = adapter.map_error(429, {"error": {"message": "rate limited", "type": "rate_limit"}})
        assert isinstance(err, ProviderError)
        assert "429" in str(err)
        assert "rate limited" in str(err)

    def test_non_dict_body(self):
        adapter = CustomOpenAIAdapter()
        err = adapter.map_error(500, "internal server error")
        assert isinstance(err, ProviderError)
        assert "500" in str(err)


class TestCustomOpenAINormalizeModel:
    def test_passthrough(self):
        adapter = CustomOpenAIAdapter()
        assert adapter.normalize_model_name("deepseek-chat") == "deepseek-chat"
        assert adapter.normalize_model_name("Qwen/Qwen3-235B-A22B") == "Qwen/Qwen3-235B-A22B"

    def test_no_prefix_stripping(self):
        """Custom provider does not strip prefixes — model names are user-specified."""
        adapter = CustomOpenAIAdapter()
        assert adapter.normalize_model_name("custom/deepseek-chat") == "custom/deepseek-chat"

    def test_normalize_request_noop(self):
        adapter = CustomOpenAIAdapter()
        req = {"model": "deepseek-chat", "messages": []}
        adapter.normalize_request(req)
        assert req == {"model": "deepseek-chat", "messages": []}
