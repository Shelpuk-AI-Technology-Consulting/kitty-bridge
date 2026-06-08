"""Tests for Custom Anthropic-compatible provider adapter."""

import json

from kitty.providers.base import ProviderError
from kitty.providers.custom_anthropic import CustomAnthropicAdapter


class TestCustomAnthropicAdapter:
    """Test suite for CustomAnthropicAdapter — basic properties."""

    def test_instantiation(self):
        adapter = CustomAnthropicAdapter()
        assert adapter is not None

    def test_provider_type(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.provider_type == "custom_anthropic"

    def test_default_base_url(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.default_base_url == "https://api.anthropic.com"

    def test_upstream_path(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.upstream_path == "/v1/messages"

    def test_requires_custom_url(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.requires_custom_url is True

    def test_use_native_messages(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.use_native_messages is True

    def test_use_custom_transport_false(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.use_custom_transport is False


class TestCustomAnthropicBuildBaseUrl:
    """Test build_base_url — returns provider_config['base_url'] when present."""

    def test_returns_config_url(self):
        adapter = CustomAnthropicAdapter()
        url = adapter.build_base_url({"base_url": "https://my-anthropic-proxy.internal"})
        assert url == "https://my-anthropic-proxy.internal"

    def test_returns_config_url_http(self):
        """HTTP URLs allowed via provider_config (unlike Profile.base_url which is HTTPS-only)."""
        adapter = CustomAnthropicAdapter()
        url = adapter.build_base_url({"base_url": "http://localhost:8080"})
        assert url == "http://localhost:8080"

    def test_falls_back_to_default(self):
        adapter = CustomAnthropicAdapter()
        url = adapter.build_base_url({})
        assert url == "https://api.anthropic.com"

    def test_falls_back_on_none_config(self):
        adapter = CustomAnthropicAdapter()
        url = adapter.build_base_url(None)
        assert url == "https://api.anthropic.com"


class TestCustomAnthropicHeaders:
    """Test build_upstream_headers — x-api-key auth inherited from AnthropicAdapter."""

    def test_x_api_key_auth(self):
        adapter = CustomAnthropicAdapter()
        headers = adapter.build_upstream_headers("sk-ant-test-key-123")
        assert headers["x-api-key"] == "sk-ant-test-key-123"
        assert headers["content-type"] == "application/json"

    def test_anthropic_version_header(self):
        adapter = CustomAnthropicAdapter()
        headers = adapter.build_upstream_headers("test-key")
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"

    def test_different_keys_different_headers(self):
        adapter = CustomAnthropicAdapter()
        h1 = adapter.build_upstream_headers("key-a")
        h2 = adapter.build_upstream_headers("key-b")
        assert h1["x-api-key"] != h2["x-api-key"]


class TestCustomAnthropicTranslateToUpstream:
    """Test translate_to_upstream — passthrough when native, CC↔Messages when not."""

    def test_native_messages_passthrough_strips_internal_keys(self):
        """When _native_messages_request is True, strip internal keys, keep all Messages API fields."""
        adapter = CustomAnthropicAdapter()
        req = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_native_messages_request": True,
            "_resolved_key": "secret",
            "_provider_config": {},
            "_original_body": {},
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(req)
        for key in adapter._INTERNAL_KEYS:
            assert key not in result
        assert result["model"] == "claude-sonnet-4-6"
        assert result["max_tokens"] == 4096
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["stream"] is True

    def test_native_messages_preserves_thinking(self):
        """Native passthrough preserves thinking config from Messages API."""
        adapter = CustomAnthropicAdapter()
        req = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "_native_messages_request": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_native_messages_preserves_system_prompt(self):
        """Native passthrough preserves system prompt from Messages API."""
        adapter = CustomAnthropicAdapter()
        req = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are a helpful assistant.",
            "_native_messages_request": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["system"] == "You are a helpful assistant."

    def test_native_messages_preserves_tools(self):
        """Native passthrough preserves tools from Messages API."""
        adapter = CustomAnthropicAdapter()
        tools = [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}]
        req = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "_native_messages_request": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["tools"] == tools

    def test_non_native_falls_back_to_cc_translation(self):
        """When _native_messages_request is NOT set, delegate to super() which translates CC → Messages."""
        adapter = CustomAnthropicAdapter()
        req = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = adapter.translate_to_upstream(req)
        # The AnthropicAdapter translates CC format → Messages format
        # CC "messages" with role+content → Anthropic messages with role+content
        assert "messages" in result
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["model"] == "claude-sonnet-4-6"
        assert result["stream"] is True


class TestCustomAnthropicTranslateFromUpstream:
    """Test translate_from_upstream — passthrough when CC-format, translate when Messages format."""

    def test_cc_format_passthrough(self):
        """When response is already in Chat Completions format, return unchanged."""
        adapter = CustomAnthropicAdapter()
        resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        }
        result = adapter.translate_from_upstream(resp)
        assert result is resp

    def test_messages_format_translated(self):
        """Anthropic Messages API response is translated to CC format by AnthropicAdapter."""
        adapter = CustomAnthropicAdapter()
        resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = adapter.translate_from_upstream(resp)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"


class TestCustomAnthropicTranslateStreamEvent:
    """Test translate_upstream_stream_event — SSE event detection and passthrough."""

    def _make_sse(self, data: dict) -> bytes:
        return f"data: {json.dumps(data)}\n\n".encode()

    def test_message_start_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse({"type": "message_start", "message": {"id": "msg_1", "model": "claude"}})
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_content_block_delta_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
        )
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_content_block_start_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        )
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_content_block_stop_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse({"type": "content_block_stop", "index": 0})
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_message_delta_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_message_stop_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse({"type": "message_stop"})
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_ping_passthrough(self):
        adapter = CustomAnthropicAdapter()
        event = self._make_sse({"type": "ping"})
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_done_passthrough(self):
        adapter = CustomAnthropicAdapter()
        chunk = b"data: [DONE]\n\n"
        result = adapter.translate_upstream_stream_event(chunk)
        assert result == [chunk]

    def test_empty_bytes_skipped(self):
        adapter = CustomAnthropicAdapter()
        result = adapter.translate_upstream_stream_event(b"")
        assert result == []

    def test_whitespace_only_skipped(self):
        adapter = CustomAnthropicAdapter()
        result = adapter.translate_upstream_stream_event(b"  \n  ")
        assert result == []

    def test_non_data_line_passthrough(self):
        """Lines without data: prefix are returned unchanged."""
        adapter = CustomAnthropicAdapter()
        chunk = b"event: ping\n"
        result = adapter.translate_upstream_stream_event(chunk)
        assert result == [chunk]

    def test_non_json_data_passthrough(self):
        adapter = CustomAnthropicAdapter()
        chunk = b"data: not-json\n\n"
        result = adapter.translate_upstream_stream_event(chunk)
        assert result == [chunk]

    def test_unknown_event_type_falls_back_to_cc_translation(self):
        """Unknown event types from Chat Completions stream are translated via AnthropicAdapter."""
        adapter = CustomAnthropicAdapter()
        # A CC-format chunk that Anthropic SSE doesn't recognize
        chunk = b'data: {"choices": [{"index": 0, "delta": {"content": "Hello"}}]}\n\n'
        result = adapter.translate_upstream_stream_event(chunk)
        # Should fall back to AnthropicAdapter's SSE translation (which keeps it)
        assert len(result) > 0


class TestCustomAnthropicErrorMapping:
    """Test map_error — descriptive ProviderError messages."""

    def test_dict_error(self):
        adapter = CustomAnthropicAdapter()
        err = adapter.map_error(401, {"error": {"message": "invalid x-api-key"}})
        assert isinstance(err, ProviderError)
        assert "Custom Anthropic" in str(err)
        assert "401" in str(err)
        assert "invalid x-api-key" in str(err)

    def test_nested_error(self):
        adapter = CustomAnthropicAdapter()
        err = adapter.map_error(429, {"error": {"message": "rate limited", "type": "rate_limit_error"}})
        assert isinstance(err, ProviderError)
        assert "Custom Anthropic" in str(err)
        assert "429" in str(err)
        assert "rate limited" in str(err)

    def test_non_dict_body(self):
        adapter = CustomAnthropicAdapter()
        err = adapter.map_error(500, "internal server error")
        assert isinstance(err, ProviderError)
        assert "Custom Anthropic" in str(err)
        assert "500" in str(err)
        assert "internal server error" in str(err)

    def test_error_object_without_message_key(self):
        """When error object has no 'message' key, use str() of the object."""
        adapter = CustomAnthropicAdapter()
        err = adapter.map_error(400, {"error": {"type": "invalid_request", "code": "missing_field"}})
        assert isinstance(err, ProviderError)
        assert "Custom Anthropic" in str(err)
        assert "400" in str(err)


class TestCustomAnthropicNormalizeModel:
    """Test normalize_model_name — inherited from AnthropicAdapter."""

    def test_strips_provider_prefix(self):
        """AnthropicAdapter strips prefixes like 'anthropic/'."""
        adapter = CustomAnthropicAdapter()
        assert adapter.normalize_model_name("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_normalizes_version_separators(self):
        """AnthropicAdapter replaces '.' with '-' for model versions."""
        adapter = CustomAnthropicAdapter()
        assert adapter.normalize_model_name("claude-3.5-sonnet") == "claude-3-5-sonnet"

    def test_passthrough_clean_name(self):
        adapter = CustomAnthropicAdapter()
        assert adapter.normalize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"


class TestCustomAnthropicBuildRequest:
    """Test build_request — inherited from AnthropicAdapter."""

    def test_basic_request(self):
        adapter = CustomAnthropicAdapter()
        req = adapter.build_request("claude-sonnet-4-6", [{"role": "user", "content": "hi"}])
        assert req == {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

    def test_streaming(self):
        adapter = CustomAnthropicAdapter()
        req = adapter.build_request("claude-sonnet-4-6", [], stream=True)
        assert req["stream"] is True

    def test_optional_params(self):
        adapter = CustomAnthropicAdapter()
        req = adapter.build_request(
            "claude-sonnet-4-6",
            [],
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        assert req["temperature"] == 0.7
        assert req["top_p"] == 0.9
        assert req["max_tokens"] == 100

    def test_tools(self):
        adapter = CustomAnthropicAdapter()
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = adapter.build_request("claude-sonnet-4-6", [], tools=tools)
        assert req["tools"] == tools

    def test_omits_none_params(self):
        adapter = CustomAnthropicAdapter()
        req = adapter.build_request("claude-sonnet-4-6", [], temperature=None, max_tokens=None)
        assert "temperature" not in req
        assert "max_tokens" not in req
