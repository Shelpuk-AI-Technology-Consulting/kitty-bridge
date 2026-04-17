"""Tests for Ollama provider adapter."""

from kitty.providers.ollama import OllamaAdapter, ProviderError


class TestOllamaAdapter:
    """Test suite for OllamaAdapter."""

    def test_instantiation(self):
        """OllamaAdapter can be instantiated."""
        adapter = OllamaAdapter()
        assert adapter is not None

    def test_provider_type(self):
        """OllamaAdapter has correct provider_type."""
        adapter = OllamaAdapter()
        assert adapter.provider_type == "ollama"

    def test_default_base_url(self):
        """OllamaAdapter has correct default base URL (HTTP localhost)."""
        adapter = OllamaAdapter()
        assert adapter.default_base_url == "http://localhost:11434"

    def test_upstream_path(self):
        """OllamaAdapter uses OpenAI-compatible endpoint path."""
        adapter = OllamaAdapter()
        assert adapter.upstream_path == "/v1/chat/completions"

    def test_build_upstream_headers(self):
        """OllamaAdapter builds minimal headers (auth not required locally)."""
        adapter = OllamaAdapter()
        headers = adapter.build_upstream_headers("test-key")
        assert headers == {"Content-Type": "application/json"}

    def test_build_upstream_headers_ignores_key(self):
        """OllamaAdapter ignores API key value (local deployment)."""
        adapter = OllamaAdapter()
        headers1 = adapter.build_upstream_headers("key1")
        headers2 = adapter.build_upstream_headers("ollama")
        assert headers1 == headers2

    def test_normalize_model_name_strips_prefix(self):
        """OllamaAdapter strips provider prefix."""
        adapter = OllamaAdapter()
        assert adapter.normalize_model_name("ollama/llama3.2") == "llama3.2"
        assert adapter.normalize_model_name("ollama/mistral") == "mistral"

    def test_normalize_model_name_no_prefix(self):
        """Model names without prefix pass through."""
        adapter = OllamaAdapter()
        assert adapter.normalize_model_name("llama3.2") == "llama3.2"
        assert adapter.normalize_model_name("codellama:7b") == "codellama:7b"

    def test_normalize_request_noop(self):
        """OllamaAdapter normalize_request is a no-op."""
        adapter = OllamaAdapter()
        req = {"model": "llama3.2", "messages": []}
        adapter.normalize_request(req)
        assert req == {"model": "llama3.2", "messages": []}

    def test_build_request(self):
        """OllamaAdapter builds OpenAI-compatible request."""
        adapter = OllamaAdapter()
        req = adapter.build_request("llama3.2", [{"role": "user", "content": "hi"}])
        assert req == {"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}], "stream": False}

    def test_build_request_streaming(self):
        """OllamaAdapter supports streaming requests."""
        adapter = OllamaAdapter()
        req = adapter.build_request("llama3.2", [{"role": "user", "content": "hi"}], stream=True)
        assert req["stream"] is True

    def test_build_request_with_optional_params(self):
        """OllamaAdapter passes through temperature, top_p, max_tokens."""
        adapter = OllamaAdapter()
        req = adapter.build_request(
            "llama3.2",
            [{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        assert req["temperature"] == 0.7
        assert req["top_p"] == 0.9
        assert req["max_tokens"] == 100

    def test_build_request_with_tools(self):
        """OllamaAdapter supports tools parameter."""
        adapter = OllamaAdapter()
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = adapter.build_request("llama3.2", [], tools=tools)
        assert req["tools"] == tools

    def test_parse_response(self):
        """OllamaAdapter parses OpenAI-compatible response."""
        adapter = OllamaAdapter()
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = adapter.parse_response(resp)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_parse_response_with_tool_calls(self):
        """OllamaAdapter parses response with tool_calls."""
        adapter = OllamaAdapter()
        tool_calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        resp = {
            "choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}, "finish_reason": "tool_calls"}],
        }
        result = adapter.parse_response(resp)
        assert result["tool_calls"] == tool_calls

    def test_parse_response_missing_choices_graceful(self):
        """OllamaAdapter handles missing or empty choices gracefully."""
        adapter = OllamaAdapter()
        result = adapter.parse_response({})
        assert result["content"] is None
        result = adapter.parse_response({"choices": []})
        assert result["content"] is None

    def test_translate_to_upstream_strips_metadata(self):
        """OllamaAdapter translate_to_upstream strips internal metadata fields."""
        adapter = OllamaAdapter()
        req = {"model": "llama3.2", "messages": [], "_resolved_key": "secret", "_provider_config": {}}
        result = adapter.translate_to_upstream(req)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert "model" in result

    def test_translate_from_upstream_passthrough(self):
        """OllamaAdapter translate_from_upstream is passthrough."""
        adapter = OllamaAdapter()
        resp = {"choices": [{"message": {"content": "hi"}}]}
        assert adapter.translate_from_upstream(resp) is resp

    def test_translate_upstream_stream_event_passthrough(self):
        """OllamaAdapter translate_upstream_stream_event passes through SSE chunks."""
        adapter = OllamaAdapter()
        chunk = b'data: {"choices": []}\n\n'
        assert adapter.translate_upstream_stream_event(chunk) == [chunk]

    def test_map_error_dict(self):
        """OllamaAdapter maps errors to ProviderError."""
        adapter = OllamaAdapter()
        err = adapter.map_error(500, {"error": {"message": "model not found"}})
        assert isinstance(err, ProviderError)
        assert "500" in str(err)
        assert "model not found" in str(err)

    def test_map_error_non_dict(self):
        """OllamaAdapter map_error handles non-dict body gracefully."""
        adapter = OllamaAdapter()
        err = adapter.map_error(400, "bad request")
        assert isinstance(err, ProviderError)
        assert "bad request" in str(err)

    def test_build_base_url_override(self):
        """OllamaAdapter supports custom base_url via provider_config."""
        adapter = OllamaAdapter()
        url = adapter.build_base_url({"base_url": "http://192.168.1.100:11434"})
        assert url == "http://192.168.1.100:11434"

    def test_build_base_url_default(self):
        """OllamaAdapter defaults to localhost when no config."""
        adapter = OllamaAdapter()
        url = adapter.build_base_url({})
        assert url == "http://localhost:11434"

    def test_use_custom_transport_false(self):
        """OllamaAdapter does not use custom transport (uses standard HTTP)."""
        adapter = OllamaAdapter()
        assert adapter.use_custom_transport is False
