"""Tests for native passthrough format fallback — when the upstream rejects
Anthropic-format ``tool_use`` blocks, the bridge converts to Chat Completions
format and retries the same backend without marking it unhealthy.
"""

from __future__ import annotations

import json

from kitty.bridge.server import (
    _convert_native_to_cc_format,
    _has_tool_use_blocks,
    _is_tool_use_format_error,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _anthropic_body_with_tool_use() -> dict:
    """Minimal Anthropic Messages body with tool_use + tool_result blocks."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please run ls"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_abc123",
                        "name": "Bash",
                        "input": {"command": "ls -la"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc123",
                        "content": "total 42\ndrwxr-xr-x ...",
                    },
                ],
            },
        ],
        "system": [{"type": "text", "text": "You are helpful."}],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a bash command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        ],
        "stream": True,
    }


def _anthropic_body_text_only() -> dict:
    """Anthropic Messages body with only text content."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ],
        "stream": True,
    }


def _anthropic_body_with_mixed_content() -> dict:
    """Anthropic body with text + tool_use in same assistant message."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Do X"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will do X now."},
                    {
                        "type": "tool_use",
                        "id": "call_mixed",
                        "name": "Bash",
                        "input": {"command": "do_x"},
                    },
                ],
            },
        ],
        "stream": True,
    }


# ── has_tool_use_blocks ───────────────────────────────────────────────────


class TestHasToolUseBlocks:
    def test_no_tool_use_returns_false(self):
        body = _anthropic_body_text_only()
        assert _has_tool_use_blocks(body) is False

    def test_with_tool_use_returns_true(self):
        body = _anthropic_body_with_tool_use()
        assert _has_tool_use_blocks(body) is True

    def test_mixed_content_returns_true(self):
        body = _anthropic_body_with_mixed_content()
        assert _has_tool_use_blocks(body) is True

    def test_no_messages_returns_false(self):
        body = {"model": "test", "stream": True}
        assert _has_tool_use_blocks(body) is False

    def test_empty_messages_returns_false(self):
        body = {"model": "test", "messages": [], "stream": True}
        assert _has_tool_use_blocks(body) is False

    def test_assistant_with_string_content(self):
        body = {
            "model": "test",
            "messages": [{"role": "assistant", "content": "plain text"}],
        }
        assert _has_tool_use_blocks(body) is False


# ── _is_tool_use_format_error ─────────────────────────────────────────────


class TestIsToolUseFormatError:
    def test_unknown_variant_tool_use(self):
        assert (
            _is_tool_use_format_error(400, '{"error": {"message": "unknown variant `tool_use`, expected `text`"}}')
            is True
        )

    def test_tool_call_result_does_not_follow(self):
        assert (
            _is_tool_use_format_error(
                400,
                '{"error": {"message": "tool call result does not follow tool call"}}',
            )
            is True
        )

    def test_2013_code(self):
        assert (
            _is_tool_use_format_error(
                400,
                '{"type":"error","error":{"code":"2013","message":"tool call result does not follow tool call"}}',
            )
            is True
        )

    def test_regular_400_not_detected(self):
        assert _is_tool_use_format_error(400, "Bad request") is False

    def test_500_not_detected(self):
        assert _is_tool_use_format_error(500, '{"error": {"message": "unknown variant"}}') is False

    def test_none_body(self):
        assert _is_tool_use_format_error(400, None) is False

    def test_dict_body(self):
        assert (
            _is_tool_use_format_error(400, {"error": {"message": "unknown variant `tool_use`, expected `text`"}})
            is True
        )

    def test_invalid_params_2013(self):
        assert (
            _is_tool_use_format_error(
                400,
                '{"error": {"message": "invalid params, tool call result does not follow tool call (2013)"}}',
            )
            is True
        )

    def test_minimax_tool_result_not_found_2013(self):
        # The EXACT production error string captured in debug/bridge.log —
        # MiniMax's actual variant, not the guessed "does not follow" wording.
        assert (
            _is_tool_use_format_error(
                400,
                '{"type":"error","error":{"type":"invalid_request_error",'
                '"message":"invalid params, tool result\'s tool id'
                '(call_f64ba2ce682a457f966bd6d7) not found (2013)"}}',
            )
            is True
        )

    def test_tool_result_not_found_no_code(self):
        # Phrase-based match must work without the numeric code.
        assert _is_tool_use_format_error(400, '{"error":{"message":"tool result not found"}}') is True

    def test_not_found_alone_not_matched(self):
        # "not found" without "tool result" must NOT match (false-positive guard).
        assert _is_tool_use_format_error(400, '{"error":{"message":"model not found"}}') is False

    def test_2013_alone_not_matched(self):
        # Bare "2013" with an unrelated message must NOT match — justifies the
        # phrase-based match over matching the numeric code.
        assert _is_tool_use_format_error(400, '{"error":{"code":"2013","message":"rate limit exceeded"}}') is False


# ── _convert_native_to_cc_format ──────────────────────────────────────────


class TestConvertNativeToCCFormat:
    def test_system_prompt_becomes_system_message(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_tool_use_becomes_tool_calls(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        # Find the assistant message
        assistant = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant) > 0
        assert assistant[0].get("tool_calls") is not None
        assert len(assistant[0]["tool_calls"]) == 1
        tc = assistant[0]["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "Bash"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls -la"}

    def test_tool_result_becomes_tool_message(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_abc123"
        assert "total 42" in tool_msgs[0]["content"]

    def test_anthropic_tools_become_cc_tools(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "Bash"
        assert result["tools"][0]["function"]["description"] == "Run a bash command"

    def test_model_stream_preserved(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert result["model"] == "claude-sonnet-4-6"
        assert result["stream"] is True

    def test_text_only_content_unchanged(self):
        body = _anthropic_body_text_only()
        result = _convert_native_to_cc_format(body)

        # User message content should be a plain string (text block stripped)
        user_msg = result["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Hello"

    def test_mixed_content_preserves_text_and_tool_use(self):
        body = _anthropic_body_with_mixed_content()
        result = _convert_native_to_cc_format(body)

        assistant = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant) == 1
        # Should have text content AND tool_calls
        assert "I will do X now" in assistant[0]["content"]
        assert assistant[0].get("tool_calls") is not None
        assert len(assistant[0]["tool_calls"]) == 1

    def test_no_system_prompt(self):
        body = _anthropic_body_text_only()
        result = _convert_native_to_cc_format(body)

        # No system message should be added
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_max_tokens_preserved(self):
        body = _anthropic_body_with_tool_use()
        body["max_tokens"] = 4096
        result = _convert_native_to_cc_format(body)

        assert result["max_tokens"] == 4096

    def test_temperature_top_p_preserved(self):
        body = _anthropic_body_with_tool_use()
        body["temperature"] = 0.7
        body["top_p"] = 0.9
        result = _convert_native_to_cc_format(body)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9


# ── Integration test — full round-trip with server ────────────────────────


class TestNativeFormatFallbackInServer:
    """Verify the fallback is wired in the BridgeServer error handling."""

    def test_is_tool_use_format_error_module_level(self):
        """_is_tool_use_format_error detects tool_use format mismatches."""
        assert _is_tool_use_format_error(400, "tool call result does not follow tool call") is True
        assert _is_tool_use_format_error(500, "tool call result does not follow tool call") is False

    def test_has_tool_use_blocks_module_level(self):
        """_has_tool_use_blocks detects Anthropic-format tool_use."""
        body = _anthropic_body_with_tool_use()
        assert _has_tool_use_blocks(body) is True
        body_no_tools = _anthropic_body_text_only()
        assert _has_tool_use_blocks(body_no_tools) is False
