"""Tests for MiniMax Token Plan Anthropic provider adapter."""

import json

from kitty.providers.base import ProviderError
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter


class TestMiniMaxTokenAnthropicAdapter:
    """Test suite for MiniMaxTokenAnthropicAdapter — basic properties."""

    def test_instantiation(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter is not None

    def test_provider_type(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.provider_type == "minimax_token"

    def test_default_base_url(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.default_base_url == "https://api.minimax.io/anthropic"

    def test_upstream_path(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.upstream_path == "/v1/messages"

    def test_use_native_messages_default_false(self):
        """Default behavior is the translated (CC→Messages) path, not native passthrough.

        Native passthrough forwards the raw Claude Code body, which MiniMax's
        Anthropic-compatible endpoint rejects with HTTP 400 because it sends
        fields the translator would have stripped (``context_management``,
        ``output_config``, ``thinking: {type: "adaptive"}``,
        ``cache_control`` on system blocks, advanced ``metadata``). The
        translated path produces a clean Anthropic Messages body that the
        upstream accepts.
        """
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.use_native_messages is False

    def test_use_custom_transport_false(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.use_custom_transport is False

    def test_use_native_messages_opt_in_via_kwarg(self):
        """Opt-in to native passthrough via the ``native_messages`` keyword.

        Some users may need to forward the raw Claude Code body for fields
        the translator strips (e.g. advanced ``cache_control``,
        ``context_management``). The opt-in is explicit and per-instance.
        """
        adapter = MiniMaxTokenAnthropicAdapter(native_messages=True)
        assert adapter.use_native_messages is True

    def test_use_native_messages_opt_in_via_provider_config(self):
        """Opt-in via ``provider_config`` so the profile wizard / settings
        can drive the flag without constructing the adapter by hand.
        """
        adapter = MiniMaxTokenAnthropicAdapter(
            provider_config={"native_messages": True},
        )
        assert adapter.use_native_messages is True

    def test_provider_config_without_flag_keeps_default(self):
        """``provider_config`` that does not include ``native_messages``
        falls back to the default (translated path).
        """
        adapter = MiniMaxTokenAnthropicAdapter(provider_config={"region": "cn"})
        assert adapter.use_native_messages is False

    def test_use_native_messages_opt_in_for_stream_event(self):
        """The stream-event translator also routes through the override so
        that opt-in to native passthrough forwards raw SSE bytes
        (``message_start``/``content_block_delta``/...) instead of trying
        to convert upstream CC chunks into Anthropic SSE.
        """
        adapter = MiniMaxTokenAnthropicAdapter(native_messages=True)
        sse = b'data: {"type": "message_start", "message": {"id": "msg_1"}}\n\n'
        assert adapter.translate_upstream_stream_event(sse) == [sse]


class TestMiniMaxTokenBuildBaseUrl:
    """Test build_base_url — global vs CN region."""

    def test_global_default(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        url = adapter.build_base_url(None)
        assert url == "https://api.minimax.io/anthropic"

    def test_global_empty_config(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        url = adapter.build_base_url({})
        assert url == "https://api.minimax.io/anthropic"

    def test_cn_region(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        url = adapter.build_base_url({"region": "cn"})
        assert url == "https://api.minimaxi.com/anthropic"

    def test_non_cn_region_uses_global(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        url = adapter.build_base_url({"region": "us"})
        assert url == "https://api.minimax.io/anthropic"


class TestMiniMaxTokenHeaders:
    """Test build_upstream_headers — x-api-key auth inherited from AnthropicAdapter."""

    def test_x_api_key_auth(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        headers = adapter.build_upstream_headers("sk-cp-test-key-123")
        assert headers["x-api-key"] == "sk-cp-test-key-123"
        assert headers["content-type"] == "application/json"

    def test_anthropic_version_header(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        headers = adapter.build_upstream_headers("test-key")
        assert "anthropic-version" in headers
        assert headers["anthropic-version"] == "2023-06-01"


class TestMiniMaxTokenNormalizeModel:
    """Test normalize_model_name — identity passthrough."""

    def test_passthrough_unmodified(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.normalize_model_name("MiniMax-M3") == "MiniMax-M3"

    def test_passthrough_with_dots(self):
        """Dots are NOT replaced — user provides exact model name."""
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.normalize_model_name("MiniMax-M3.5") == "MiniMax-M3.5"

    def test_passthrough_with_prefix(self):
        """Provider prefixes are NOT stripped — user provides exact model name."""
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.normalize_model_name("minimax/MiniMax-M3") == "minimax/MiniMax-M3"


class TestMiniMaxTokenTranslateToUpstream:
    """Test translate_to_upstream — passthrough when native, CC→Messages when not."""

    def test_native_messages_passthrough_strips_internal_keys(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        req = {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_native_messages_request": True,
            "_resolved_key": "secret",
            "_provider_config": {},
            "_reasoning_effort": "high",
            "_thinking_enabled": True,
        }
        result = adapter.translate_to_upstream(req)
        for key in adapter._INTERNAL_KEYS:
            assert key not in result
        assert result["model"] == "MiniMax-M3"
        assert result["max_tokens"] == 4096
        assert result["stream"] is True

    def test_native_messages_preserves_thinking(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        req = {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "_native_messages_request": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_native_messages_preserves_system_and_tools(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        tools = [{"name": "read", "input_schema": {"type": "object"}}]
        req = {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are helpful.",
            "tools": tools,
            "_native_messages_request": True,
        }
        result = adapter.translate_to_upstream(req)
        assert result["system"] == "You are helpful."
        assert result["tools"] == tools

    def test_non_native_falls_back_to_cc_translation(self):
        """When _native_messages_request is NOT set, delegate to super() for CC→Messages."""
        adapter = MiniMaxTokenAnthropicAdapter()
        req = {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = adapter.translate_to_upstream(req)
        assert "messages" in result
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["model"] == "MiniMax-M3"

    def test_tool_use_id_preserved_through_default_path(self):
        """tool_use_id ↔ tool_call_id pairing must be preserved through the
        default (translated) path so MiniMax's
        ``tool call result does not follow tool call (2013)`` validation
        does not fire on a broken pairing.

        Round-trip: CC ``assistant(tool_calls=[{id: "toolu_abc"}])`` + CC
        ``tool(tool_call_id="toolu_abc")`` → Anthropic assistant
        ``tool_use(id="toolu_abc")`` + Anthropic user
        ``tool_result(tool_use_id="toolu_abc")``.
        """
        adapter = MiniMaxTokenAnthropicAdapter()
        req = {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "read /etc/hostname"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_abc",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps({"path": "/etc/hostname"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_abc",
                    "content": "kitty-bridge-host",
                },
            ],
        }
        result = adapter.translate_to_upstream(req)
        # Assistant tool_use block carries the same id
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        tool_use_blocks = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["id"] == "toolu_abc"
        # Following user tool_result block carries the same tool_use_id
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "user"
        tool_result_blocks = [b for b in tool_msg["content"] if b.get("type") == "tool_result"]
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["tool_use_id"] == "toolu_abc"

    def test_thinking_block_does_not_displace_tool_use(self):
        """An assistant message with both thinking AND tool_use must keep
        both content blocks in the default path so that the
        tool_use → tool_result pairing is not silently broken.
        """
        adapter = MiniMaxTokenAnthropicAdapter()
        req = {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me read the file first.",
                    "reasoning_content": "I need to call read.",
                    "tool_calls": [
                        {
                            "id": "toolu_xyz",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
            ],
        }
        result = adapter.translate_to_upstream(req)
        assistant_msg = result["messages"][0]
        block_types = [b.get("type") for b in assistant_msg["content"]]
        assert "thinking" in block_types
        assert "text" in block_types
        assert "tool_use" in block_types


class TestMiniMaxTokenTranslateFromUpstream:
    """Test translate_from_upstream — passthrough when CC-format, translate when Messages."""

    def test_cc_format_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        }
        result = adapter.translate_from_upstream(resp)
        assert result is resp

    def test_messages_format_translated(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "MiniMax-M3",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = adapter.translate_from_upstream(resp)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"


class TestMiniMaxTokenStreamEvent:
    """Test translate_upstream_stream_event — SSE event detection and passthrough."""

    def _make_sse(self, data: dict) -> bytes:
        return f"data: {json.dumps(data)}\n\n".encode()

    def test_message_start_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        event = self._make_sse({"type": "message_start", "message": {"id": "msg_1"}})
        assert adapter.translate_upstream_stream_event(event) == [event]

    def test_content_block_delta_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        event = self._make_sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
        )
        assert adapter.translate_upstream_stream_event(event) == [event]

    def test_message_delta_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        event = self._make_sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
        assert adapter.translate_upstream_stream_event(event) == [event]

    def test_message_stop_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        event = self._make_sse({"type": "message_stop"})
        assert adapter.translate_upstream_stream_event(event) == [event]

    def test_ping_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        event = self._make_sse({"type": "ping"})
        assert adapter.translate_upstream_stream_event(event) == [event]

    def test_done_passthrough(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        chunk = b"data: [DONE]\n\n"
        assert adapter.translate_upstream_stream_event(chunk) == [chunk]

    def test_empty_bytes_skipped(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.translate_upstream_stream_event(b"") == []

    def test_whitespace_only_skipped(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        assert adapter.translate_upstream_stream_event(b"  \n  ") == []


class TestMiniMaxTokenErrorMapping:
    """Test map_error — descriptive ProviderError messages."""

    def test_dict_error(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        err = adapter.map_error(401, {"error": {"message": "invalid api key"}})
        assert isinstance(err, ProviderError)
        assert "MiniMax Token Plan" in str(err)
        assert "401" in str(err)
        assert "invalid api key" in str(err)

    def test_non_dict_body(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        err = adapter.map_error(500, "internal server error")
        assert isinstance(err, ProviderError)
        assert "MiniMax Token Plan" in str(err)
        assert "500" in str(err)

    def test_nested_error(self):
        adapter = MiniMaxTokenAnthropicAdapter()
        err = adapter.map_error(429, {"error": {"message": "rate limited", "type": "rate_limit_error"}})
        assert isinstance(err, ProviderError)
        assert "MiniMax Token Plan" in str(err)
        assert "rate limited" in str(err)


class TestMiniMaxTokenRegistry:
    """Test registry and schema integration."""

    def test_registry_lookup(self):
        from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
        from kitty.providers.registry import get_provider

        provider = get_provider("minimax_token")
        assert isinstance(provider, MiniMaxTokenAnthropicAdapter)

    def test_schema_list(self):
        from kitty.profiles.schema import PROVIDER_LIST

        assert "minimax_token" in PROVIDER_LIST

    def test_schema_label(self):
        from kitty.profiles.schema import PROVIDER_LABELS

        assert PROVIDER_LABELS["minimax_token"] == "MiniMax Token Plan"

    def test_schema_section(self):
        from kitty.profiles.schema import PROVIDER_SECTIONS

        coding_plans = [s for s in PROVIDER_SECTIONS if s[0] == "-- Coding Plans / Subscriptions --"]
        assert len(coding_plans) == 1
        assert "minimax_token" in coding_plans[0][1]
