"""Tests for bridge/messages/events.py — SSE event formatters for the Messages API bridge."""

import json

from kitty.bridge.messages.events import (
    format_content_block_delta_event,
    format_content_block_start_event,
    format_content_block_stop_event,
    format_error_event,
    format_message_delta_event,
    format_message_start_event,
    format_message_stop_event,
    format_ping_event,
)


def _parse_sse(raw: str) -> tuple[str, dict]:
    """Parse a raw SSE string into (event_type, data_dict)."""
    lines = raw.strip().split("\n")
    event_type = lines[0].split(": ", 1)[1]
    data_line = "\n".join(lines[1:])
    data_line = data_line.split(": ", 1)[1]
    return event_type, json.loads(data_line)


class TestMessageStartEvent:
    def test_format(self):
        message_data = {
            "id": "msg_abc123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-opus",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 0},
        }
        raw = format_message_start_event(message_data)
        event_type, data = _parse_sse(raw)
        assert event_type == "message_start"
        assert data["type"] == "message_start"
        assert data["message"]["id"] == "msg_abc123"
        assert data["message"]["usage"]["input_tokens"] == 25

    def test_ends_with_double_newline(self):
        raw = format_message_start_event({"id": "msg_test", "type": "message", "content": []})
        assert raw.endswith("\n\n")


class TestContentBlockStartEvent:
    def test_text_block(self):
        raw = format_content_block_start_event(0, {"type": "text", "text": ""})
        event_type, data = _parse_sse(raw)
        assert event_type == "content_block_start"
        assert data["type"] == "content_block_start"
        assert data["index"] == 0
        assert data["content_block"]["type"] == "text"

    def test_tool_use_block(self):
        block = {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {}}
        raw = format_content_block_start_event(1, block)
        event_type, data = _parse_sse(raw)
        assert event_type == "content_block_start"
        assert data["index"] == 1
        assert data["content_block"]["type"] == "tool_use"
        assert data["content_block"]["name"] == "get_weather"

    def test_ends_with_double_newline(self):
        raw = format_content_block_start_event(0, {"type": "text", "text": ""})
        assert raw.endswith("\n\n")


class TestContentBlockDeltaEvent:
    def test_text_delta(self):
        raw = format_content_block_delta_event(0, {"type": "text_delta", "text": "Hello"})
        event_type, data = _parse_sse(raw)
        assert event_type == "content_block_delta"
        assert data["type"] == "content_block_delta"
        assert data["index"] == 0
        assert data["delta"]["type"] == "text_delta"
        assert data["delta"]["text"] == "Hello"

    def test_input_json_delta(self):
        raw = format_content_block_delta_event(1, {"type": "input_json_delta", "partial_json": '{"city":'})
        event_type, data = _parse_sse(raw)
        assert event_type == "content_block_delta"
        assert data["index"] == 1
        assert data["delta"]["type"] == "input_json_delta"
        assert data["delta"]["partial_json"] == '{"city":'

    def test_ends_with_double_newline(self):
        raw = format_content_block_delta_event(0, {"type": "text_delta", "text": "x"})
        assert raw.endswith("\n\n")


class TestContentBlockStopEvent:
    def test_format(self):
        raw = format_content_block_stop_event(0)
        event_type, data = _parse_sse(raw)
        assert event_type == "content_block_stop"
        assert data["type"] == "content_block_stop"
        assert data["index"] == 0

    def test_ends_with_double_newline(self):
        raw = format_content_block_stop_event(0)
        assert raw.endswith("\n\n")


class TestMessageDeltaEvent:
    def test_format(self):
        raw = format_message_delta_event(
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 42},
        )
        event_type, data = _parse_sse(raw)
        assert event_type == "message_delta"
        assert data["type"] == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"
        assert data["usage"]["output_tokens"] == 42

    def test_ends_with_double_newline(self):
        raw = format_message_delta_event({"stop_reason": "end_turn"}, {"output_tokens": 0})
        assert raw.endswith("\n\n")


class TestMessageStopEvent:
    def test_format(self):
        raw = format_message_stop_event()
        event_type, data = _parse_sse(raw)
        assert event_type == "message_stop"
        assert data["type"] == "message_stop"

    def test_ends_with_double_newline(self):
        raw = format_message_stop_event()
        assert raw.endswith("\n\n")


class TestPingEvent:
    def test_format(self):
        raw = format_ping_event()
        event_type, data = _parse_sse(raw)
        assert event_type == "ping"
        assert data["type"] == "ping"

    def test_ends_with_double_newline(self):
        raw = format_ping_event()
        assert raw.endswith("\n\n")


class TestErrorEvent:
    def test_format(self):
        error_payload = {"type": "error", "error": {"type": "overloaded_error", "message": "API is overloaded"}}
        raw = format_error_event(error_payload)
        event_type, data = _parse_sse(raw)
        assert event_type == "error"
        assert data["type"] == "error"
        assert data["error"]["type"] == "overloaded_error"
        assert data["error"]["message"] == "API is overloaded"

    def test_ends_with_double_newline(self):
        raw = format_error_event({"type": "error", "error": {"type": "x", "message": "y"}})
        assert raw.endswith("\n\n")


class TestSSEFormatValidity:
    """All events must be valid SSE: 'event: <type>\\ndata: <json>\\n\\n'."""

    def test_all_events_have_valid_format(self):
        events = [
            format_message_start_event({"id": "msg_1", "type": "message", "content": []}),
            format_content_block_start_event(0, {"type": "text", "text": ""}),
            format_content_block_delta_event(0, {"type": "text_delta", "text": "hi"}),
            format_content_block_stop_event(0),
            format_message_delta_event({"stop_reason": "end_turn"}, {"output_tokens": 5}),
            format_message_stop_event(),
            format_ping_event(),
            format_error_event({"type": "error", "error": {"type": "x", "message": "y"}}),
        ]
        for raw in events:
            assert raw.startswith("event: "), f"Missing 'event:' prefix: {raw!r}"
            assert raw.endswith("\n\n"), f"Missing double-newline ending: {raw!r}"
            lines = raw.split("\n")
            data_lines = [line for line in lines if line.startswith("data: ")]
            assert len(data_lines) == 1, f"Expected 1 data line, got {len(data_lines)} in: {raw!r}"
