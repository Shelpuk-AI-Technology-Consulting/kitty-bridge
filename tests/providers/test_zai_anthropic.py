"""Tests for ZaiAnthropicAdapter — native Anthropic Messages API passthrough."""

from __future__ import annotations

import pytest

from kitty.providers.zai_anthropic import ZaiAnthropicAdapter


@pytest.fixture
def adapter() -> ZaiAnthropicAdapter:
    return ZaiAnthropicAdapter()


def _native_body(**overrides) -> dict:
    """Build a request body that signals native Messages format."""
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "_native_messages_request": True,
    }
    body.update(overrides)
    return body


# ── Adapter identity ─────────────────────────────────────────────────────


class TestZaiAnthropicIdentity:
    def test_provider_type(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.provider_type == "zai_coding"

    def test_default_base_url(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.default_base_url == "https://api.z.ai/api/anthropic"

    def test_upstream_path(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.upstream_path == "/v1/messages"

    def test_use_native_messages(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.use_native_messages is True


# ── Auth headers ─────────────────────────────────────────────────────────


class TestZaiAnthropicHeaders:
    def test_bearer_auth(self, adapter: ZaiAnthropicAdapter) -> None:
        headers = adapter.build_upstream_headers("sk-test-key")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"

    def test_no_x_api_key(self, adapter: ZaiAnthropicAdapter) -> None:
        headers = adapter.build_upstream_headers("sk-test-key")
        assert "x-api-key" not in headers


# ── translate_to_upstream: native Messages passthrough ──────────────────


class TestZaiAnthropicTranslateUpstreamNative:
    """Tests for the native Messages passthrough path (_native_messages_request=True)."""

    def test_strips_internal_keys(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(
            max_tokens=4096,
            _reasoning_effort="high",
            _thinking_enabled=True,
            _resolved_key="sk-secret",
            _provider_config={"foo": "bar"},
            _original_body={},
        )
        result = adapter.translate_to_upstream(body)
        assert "_reasoning_effort" not in result
        assert "_thinking_enabled" not in result
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert "_original_body" not in result
        assert "_native_messages_request" not in result

    def test_preserves_model(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(model="claude-sonnet-4-6")
        result = adapter.translate_to_upstream(body)
        assert result["model"] == "claude-sonnet-4-6"

    def test_preserves_messages(self, adapter: ZaiAnthropicAdapter) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        body = _native_body(messages=messages)
        result = adapter.translate_to_upstream(body)
        assert result["messages"] == messages

    def test_preserves_system(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(system="You are helpful.")
        result = adapter.translate_to_upstream(body)
        assert result["system"] == "You are helpful."

    def test_preserves_effort(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(effort="xhigh")
        result = adapter.translate_to_upstream(body)
        assert result["effort"] == "xhigh"

    def test_preserves_thinking_config(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(thinking={"type": "adaptive"})
        result = adapter.translate_to_upstream(body)
        assert result["thinking"] == {"type": "adaptive"}

    def test_preserves_thinking_with_budget(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(thinking={"type": "enabled", "budget_tokens": 10000})
        result = adapter.translate_to_upstream(body)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    def test_preserves_tools(self, adapter: ZaiAnthropicAdapter) -> None:
        tools = [{"name": "bash", "description": "run", "input_schema": {}}]
        body = _native_body(tools=tools)
        result = adapter.translate_to_upstream(body)
        assert result["tools"] == tools

    def test_preserves_max_tokens(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(max_tokens=16384)
        result = adapter.translate_to_upstream(body)
        assert result["max_tokens"] == 16384

    def test_preserves_stream(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(stream=True)
        result = adapter.translate_to_upstream(body)
        assert result["stream"] is True

    def test_preserves_temperature(self, adapter: ZaiAnthropicAdapter) -> None:
        body = _native_body(temperature=0.7)
        result = adapter.translate_to_upstream(body)
        assert result["temperature"] == 0.7


# ── translate_from_upstream: native Messages passthrough ────────────────


class TestZaiAnthropicTranslateFromUpstreamNative:
    def test_messages_response_translated_to_cc(self, adapter: ZaiAnthropicAdapter) -> None:
        """Anthropic Messages responses are always translated to CC format.

        The server skips translate_from_upstream entirely for the native
        Messages path, so the adapter always does Messages→CC.
        """
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = adapter.translate_from_upstream(response)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_cc_response_passthrough(self, adapter: ZaiAnthropicAdapter) -> None:
        """Already-CC responses are passed through unchanged."""
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = adapter.translate_from_upstream(response)
        assert result is response


# ── translate_upstream_stream_event: Anthropic SSE passthrough ──────────


class TestZaiAnthropicStreamEvent:
    def test_anthropic_event_passthrough(self, adapter: ZaiAnthropicAdapter) -> None:
        event = (
            b'data: {"type":"message_start","message":{"id":"msg_1","type":"message"'
            b',"role":"assistant","content":[],"model":"m","stop_reason":null'
            b',"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        )
        result = adapter.translate_upstream_stream_event(event)
        assert result == [event]

    def test_empty_passthrough(self, adapter: ZaiAnthropicAdapter) -> None:
        result = adapter.translate_upstream_stream_event(b"")
        assert result == []


# ── normalize_model_name ────────────────────────────────────────────────


class TestZaiAnthropicModelName:
    def test_strips_prefix(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.normalize_model_name("zai/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_strips_z_ai_prefix(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.normalize_model_name("z-ai/claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_no_prefix(self, adapter: ZaiAnthropicAdapter) -> None:
        assert adapter.normalize_model_name("claude-sonnet-4-6") == "claude-sonnet-4-6"


# ── Registry ────────────────────────────────────────────────────────────


class TestZaiAnthropicRegistry:
    def test_registry_returns_zai_anthropic(self) -> None:
        from kitty.providers.registry import get_provider

        provider = get_provider("zai_coding")
        assert isinstance(provider, ZaiAnthropicAdapter)

    def test_registry_keeps_zai_coding_cc(self) -> None:
        from kitty.providers.registry import _registry
        from kitty.providers.zai import ZaiCodingAdapter

        assert "zai_coding_cc" in _registry
        assert isinstance(_registry["zai_coding_cc"](), ZaiCodingAdapter)


# ── CC format fallback (non-Claude Code clients) ───────────────────────


class TestZaiAnthropicCCFallback:
    """Verify CC→Messages translation works for Responses/CC API clients.

    When a non-Claude Code client sends a request, the server translates it
    to CC format. The adapter must translate CC→Messages for the upstream.
    """

    def test_cc_request_translates_to_messages(self, adapter: ZaiAnthropicAdapter) -> None:
        cc_body = {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
            ],
            "tools": [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {}}}],
            "max_tokens": 4096,
            "stream": True,
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(cc_body)
        # Should be in Anthropic Messages format
        assert "system" in result
        assert result["system"] == "You are helpful."
        # No CC-style system message
        assert not any(m.get("role") == "system" for m in result["messages"])
        # Tools should be in Anthropic format (name, input_schema)
        assert result["tools"][0]["name"] == "bash"
        assert "input_schema" in result["tools"][0]
        # Thinking should be enabled
        assert result["thinking"]["type"] == "enabled"

    def test_cc_response_translates_from_messages(self, adapter: ZaiAnthropicAdapter) -> None:
        """An Anthropic Messages response should be translated to CC format."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = adapter.translate_from_upstream(anthropic_response)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_cc_stream_event_uses_anthropic_translation(self, adapter: ZaiAnthropicAdapter) -> None:
        """CC SSE events should be passed through super().translate_upstream_stream_event."""
        # This is a CC-format SSE event — should be delegated to AnthropicAdapter
        cc_event = (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk"'
            b',"created":0,"model":"m","choices":[{"index":0'
            b',"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        )
        result = adapter.translate_upstream_stream_event(cc_event)
        assert len(result) == 1
