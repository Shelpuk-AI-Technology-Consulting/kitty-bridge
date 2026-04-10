"""Tests for Ollama provider adapter."""

from __future__ import annotations

import pytest

from kitty.providers.ollama import OllamaAdapter, ProviderError


class TestOllamaAdapter:
    """Test suite for OllamaAdapter."""

    def test_adapter_instantiation(self) -> None:
        """OllamaAdapter can be instantiated."""
        adapter = OllamaAdapter()
        assert adapter is not None

    def test_provider_type(self) -> None:
        """OllamaAdapter has correct provider_type."""
        adapter = OllamaAdapter()
        assert adapter.provider_type == "ollama"

    def test_default_base_url(self) -> None:
        """OllamaAdapter has correct default base URL (HTTP localhost)."""
        adapter = OllamaAdapter()
        assert adapter.default_base_url == "http://localhost:11434"

    def test_upstream_path(self) -> None:
        """OllamaAdapter uses OpenAI-compatible endpoint path."""
        adapter = OllamaAdapter()
        assert adapter.upstream_path == "/v1/chat/completions"

    def test_build_upstream_headers_minimal(self) -> None:
        """OllamaAdapter builds minimal headers (auth not required locally)."""
        adapter = OllamaAdapter()
        headers = adapter.build_upstream_headers("any-key")
        assert headers == {"Content-Type": "application/json"}

    def test_build_upstream_headers_ignores_api_key(self) -> None:
        """OllamaAdapter ignores API key value (local deployment)."""
        adapter = OllamaAdapter()
        headers1 = adapter.build_upstream_headers("")
        headers2 = adapter.build_upstream_headers("ollama")
        headers3 = adapter.build_upstream_headers("fake-key-123")
        assert headers1 == headers2 == headers3

    def test_normalize_model_name_passthrough(self) -> None:
        """OllamaAdapter passes model names through unchanged."""
        adapter = OllamaAdapter()
        assert adapter.normalize_model_name("llama3.2") == "llama3.2"
        assert adapter.normalize_model_name("mistral") == "mistral"
        assert adapter.normalize_model_name("codellama:7b") == "codellama:7b"

    def test_normalize_request_no_op(self) -> None:
        """OllamaAdapter normalize_request is a no-op."""
        adapter = OllamaAdapter()
        request = {"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]}
        original = request.copy()
        adapter.normalize_request(request)
        assert request == original

    def test_build_request_basic(self) -> None:
        """OllamaAdapter builds OpenAI-compatible request."""
        adapter = OllamaAdapter()
        request = adapter.build_request(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert request["model"] == "llama3.2"
        assert request["messages"] == [{"role": "user", "content": "Hello"}]
        assert request["stream"] is False

    def test_build_request_with_streaming(self) -> None:
        """OllamaAdapter supports streaming requests."""
        adapter = OllamaAdapter()
        request = adapter.build_request(
            model="mistral",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        assert request["stream"] is True

    def test_build_request_with_parameters(self) -> None:
        """OllamaAdapter passes through temperature, top_p, max_tokens."""
        adapter = OllamaAdapter()
        request = adapter.build_request(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        assert request["temperature"] == 0.7
        assert request["top_p"] == 0.9
        assert request["max_tokens"] == 100

    def test_build_request_with_tools(self) -> None:
        """OllamaAdapter supports tools parameter."""
        adapter = OllamaAdapter()
        tools = [{"type": "function", "function": {"name": "test"}}]
        request = adapter.build_request(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert request["tools"] == tools

    def test_parse_response(self) -> None:
        """OllamaAdapter parses OpenAI-compatible response."""
        adapter = OllamaAdapter()
        response_data = {
            "choices": [{
                "message": {"content": "Hello!", "role": "assistant"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = adapter.parse_response(response_data)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_parse_response_with_tool_calls(self) -> None:
        """OllamaAdapter parses response with tool_calls."""
        adapter = OllamaAdapter()
        response_data = {
            "choices": [{
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "function": {"name": "test"}}],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        result = adapter.parse_response(response_data)
        assert result["content"] is None
        assert result["tool_calls"] == [{"id": "call_1", "function": {"name": "test"}}]

    def test_parse_response_raises_when_choices_missing(self) -> None:
        """OllamaAdapter raises ProviderError when choices is missing or empty."""
        adapter = OllamaAdapter()
        with pytest.raises(ProviderError, match="missing 'choices'"):
            adapter.parse_response({"usage": {}})
        with pytest.raises(ProviderError, match="missing 'choices'"):
            adapter.parse_response({"choices": [], "usage": {}})

    def test_translate_to_upstream_strips_internal_fields(self) -> None:
        """OllamaAdapter translate_to_upstream strips internal metadata fields."""
        adapter = OllamaAdapter()
        cc_request = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "_resolved_key": "secret-api-key",
            "_provider_config": {"base_url": "http://192.168.1.100:11434"},
        }
        result = adapter.translate_to_upstream(cc_request)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert result["model"] == "llama3.2"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["stream"] is False

    def test_translate_from_upstream_passthrough(self) -> None:
        """OllamaAdapter translate_from_upstream is passthrough."""
        adapter = OllamaAdapter()
        raw_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Hello!"}}],
        }
        result = adapter.translate_from_upstream(raw_response)
        assert result == raw_response

    def test_translate_upstream_stream_event_passthrough(self) -> None:
        """OllamaAdapter translate_upstream_stream_event passes through SSE chunks."""
        adapter = OllamaAdapter()
        raw = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        result = adapter.translate_upstream_stream_event(raw)
        assert result == [raw]

    def test_map_error(self) -> None:
        """OllamaAdapter maps errors to ProviderError."""
        adapter = OllamaAdapter()
        body = {"error": {"message": "Model not found"}}
        exc = adapter.map_error(404, body)
        assert isinstance(exc, ProviderError)
        assert "404" in str(exc)
        assert "Model not found" in str(exc)

    def test_map_error_with_non_dict_body(self) -> None:
        """OllamaAdapter map_error handles non-dict body gracefully."""
        adapter = OllamaAdapter()
        exc = adapter.map_error(500, "raw error")
        assert isinstance(exc, ProviderError)
        assert "500" in str(exc)
        assert "raw error" in str(exc)

    def test_custom_base_url_via_provider_config(self) -> None:
        """OllamaAdapter supports custom base_url via provider_config."""
        adapter = OllamaAdapter()
        # Default without config
        assert adapter.build_base_url({}) == "http://localhost:11434"
        # With custom base_url
        custom_url = adapter.build_base_url({"base_url": "http://192.168.1.100:11434"})
        assert custom_url == "http://192.168.1.100:11434"

    def test_use_custom_transport_false(self) -> None:
        """OllamaAdapter does not use custom transport (uses standard HTTP)."""
        adapter = OllamaAdapter()
        assert adapter.use_custom_transport is False
