"""Tests for bridge/engine.py — TranslationEngine and ToolCallBuffer."""

import json
import uuid

import pytest

from kitty.bridge.engine import (
    ToolCallBuffer,
    ToolCallBufferError,
    TranslationEngine,
)

# ── TranslationEngine ──────────────────────────────────────────────────────


class TestMapFinishReason:
    """map_finish_reason: Chat Completions finish_reason -> protocol stop reason."""

    def test_stop_to_end_turn(self):
        assert TranslationEngine.map_finish_reason("stop") == "end_turn"

    def test_tool_calls_to_tool_use(self):
        assert TranslationEngine.map_finish_reason("tool_calls") == "tool_use"

    def test_length_to_max_tokens(self):
        assert TranslationEngine.map_finish_reason("length") == "max_tokens"

    def test_none_to_end_turn(self):
        assert TranslationEngine.map_finish_reason(None) == "end_turn"

    def test_unknown_passthrough(self):
        assert TranslationEngine.map_finish_reason("content_filter") == "content_filter"


class TestMapStopReasonToFinishReason:
    """map_stop_reason_to_finish_reason: protocol stop reason -> Chat Completions finish_reason."""

    def test_end_turn_to_stop(self):
        assert TranslationEngine.map_stop_reason_to_finish_reason("end_turn") == "stop"

    def test_tool_use_to_tool_calls(self):
        assert TranslationEngine.map_stop_reason_to_finish_reason("tool_use") == "tool_calls"

    def test_max_tokens_to_length(self):
        assert TranslationEngine.map_stop_reason_to_finish_reason("max_tokens") == "length"

    def test_unknown_passthrough(self):
        assert TranslationEngine.map_stop_reason_to_finish_reason("unknown") == "unknown"


class TestBuildToolCall:
    def test_returns_expected_structure(self):
        result = TranslationEngine.build_tool_call("get_weather", '{"city": "London"}')
        assert "id" in result
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["arguments"] == '{"city": "London"}'

    def test_id_is_call_prefixed_uuid(self):
        result = TranslationEngine.build_tool_call("test", "{}")
        assert result["id"].startswith("call_")
        # Everything after "call_" should be a valid UUID
        uuid.UUID(result["id"][5:])


class TestBuildUsage:
    def test_returns_expected_structure(self):
        result = TranslationEngine.build_usage(100, 50)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150


# ── ToolCallBuffer ─────────────────────────────────────────────────────────


class TestToolCallBufferHappyPath:
    def test_single_chunk_finalize(self):
        buf = ToolCallBuffer()
        buf.append('{"city": "London"}')
        assert buf.finalize() == '{"city": "London"}'

    def test_incremental_chunks(self):
        buf = ToolCallBuffer()
        buf.append('{"ci')
        buf.append('ty": "')
        buf.append('London"}')
        assert buf.finalize() == '{"city": "London"}'

    def test_finalize_validates_json(self):
        buf = ToolCallBuffer()
        buf.append('{"valid": true}')
        result = buf.finalize()
        assert json.loads(result) == {"valid": True}

    def test_finalize_resets_total_len(self):
        buf = ToolCallBuffer()
        buf.append('{"a": 1}')
        buf.finalize()
        assert buf._total_len == 0
        assert buf._chunks == []

    def test_reset_clears_total_len(self):
        buf = ToolCallBuffer()
        buf.append("some text")
        buf.reset()
        assert buf._total_len == 0
        assert buf._chunks == []


class TestToolCallBufferMaxSize:
    def test_max_size_exceeded_raises(self):
        buf = ToolCallBuffer(max_size=10)
        with pytest.raises(ToolCallBufferError):
            buf.append("x" * 11)

    def test_exact_max_size_ok(self):
        buf = ToolCallBuffer(max_size=10)
        buf.append("x" * 10)
        # Should not raise on append — exactly at limit
        # finalize will fail on invalid JSON but that's a different error
        with pytest.raises(ToolCallBufferError):
            buf.finalize()  # "xxxxxxxxxx" is not valid JSON


class TestToolCallBufferInvalidJson:
    def test_invalid_json_on_finalize_raises(self):
        buf = ToolCallBuffer()
        buf.append("{invalid json")
        with pytest.raises(ToolCallBufferError):
            buf.finalize()


class TestToolCallBufferReset:
    def test_reset_clears_state(self):
        buf = ToolCallBuffer()
        buf.append('{"a": 1}')
        buf.reset()
        with pytest.raises(ToolCallBufferError):
            buf.finalize()

    def test_reset_allows_reuse(self):
        buf = ToolCallBuffer()
        buf.append('{"first": true}')
        buf.reset()
        buf.append('{"second": true}')
        result = buf.finalize()
        assert json.loads(result) == {"second": True}


class TestToolCallBufferEdgeCases:
    def test_empty_string_append(self):
        buf = ToolCallBuffer()
        buf.append("")
        buf.append('{"ok": true}')
        result = buf.finalize()
        assert json.loads(result) == {"ok": True}

    def test_double_finalize_without_reset_fails(self):
        buf = ToolCallBuffer()
        buf.append('{"a": 1}')
        buf.finalize()
        # Second finalize without reset should fail (buffer was cleared)
        with pytest.raises(ToolCallBufferError):
            buf.finalize()
