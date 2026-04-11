"""Tests for bridge/responses/events.py — SSE event formatters for the Responses API bridge."""

import json

from kitty.bridge.responses.events import (
    format_content_part_added_event,
    format_content_part_done_event,
    format_error_event,
    format_function_call_arguments_delta_event,
    format_function_call_arguments_done_event,
    format_output_item_added_event,
    format_output_item_done_event,
    format_output_text_delta_event,
    format_output_text_done_event,
    format_response_completed_event,
    format_response_created_event,
    format_response_in_progress_event,
)


def _parse_sse(raw: str) -> tuple[str, dict]:
    """Parse a raw SSE string into (event_type, data_dict)."""
    lines = raw.strip().split("\n")
    event_type = lines[0].split(": ", 1)[1]
    data_line = lines[1].split(": ", 1)[1]
    return event_type, json.loads(data_line)


def _assert_required_fields(raw: str, expected_type: str) -> dict:
    """Assert that an SSE event has type and sequence_number fields. Returns parsed data."""
    event_type, data = _parse_sse(raw)
    assert event_type == expected_type
    assert data["type"] == expected_type, f"Missing/incorrect 'type' field in {expected_type}"
    assert "sequence_number" in data, f"Missing 'sequence_number' in {expected_type}"
    assert isinstance(data["sequence_number"], int)
    return data


class TestResponseCreatedEvent:
    def test_format(self):
        raw = format_response_created_event("resp_abc123", seq=0, model="gpt-4o")
        data = _assert_required_fields(raw, "response.created")
        assert data["sequence_number"] == 0
        assert data["response"]["id"] == "resp_abc123"
        assert data["response"]["object"] == "response"
        assert data["response"]["status"] == "in_progress"
        assert data["response"]["model"] == "gpt-4o"
        assert data["response"]["output"] == []

    def test_ends_with_double_newline(self):
        raw = format_response_created_event("resp_test", seq=0)
        assert raw.endswith("\n\n")


class TestResponseInProgressEvent:
    def test_format(self):
        raw = format_response_in_progress_event("resp_abc", seq=1, model="gpt-4o")
        data = _assert_required_fields(raw, "response.in_progress")
        assert data["sequence_number"] == 1
        assert data["response"]["id"] == "resp_abc"
        assert data["response"]["status"] == "in_progress"


class TestOutputItemAddedEvent:
    def test_format(self):
        item = {"id": "msg_abc", "type": "message", "status": "in_progress", "content": [], "role": "assistant"}
        raw = format_output_item_added_event(seq=2, output_index=0, item=item)
        data = _assert_required_fields(raw, "response.output_item.added")
        assert data["output_index"] == 0
        assert data["item"]["id"] == "msg_abc"
        assert data["item"]["type"] == "message"


class TestContentPartAddedEvent:
    def test_format(self):
        part = {"type": "output_text", "text": ""}
        raw = format_content_part_added_event(
            seq=3,
            item_id="msg_abc",
            output_index=0,
            content_index=0,
            part=part,
        )
        data = _assert_required_fields(raw, "response.content_part.added")
        assert data["item_id"] == "msg_abc"
        assert data["output_index"] == 0
        assert data["content_index"] == 0
        assert data["part"]["type"] == "output_text"


class TestOutputTextDeltaEvent:
    def test_format(self):
        raw = format_output_text_delta_event(
            seq=4,
            response_id="resp_abc",
            item_id="msg_abc",
            output_index=0,
            content_index=0,
            delta="Hello",
        )
        data = _assert_required_fields(raw, "response.output_text.delta")
        assert data["response_id"] == "resp_abc"
        assert data["item_id"] == "msg_abc"
        assert data["output_index"] == 0
        assert data["content_index"] == 0
        assert data["delta"] == "Hello"

    def test_ends_with_double_newline(self):
        raw = format_output_text_delta_event(
            seq=0,
            response_id="r",
            item_id="m",
            output_index=0,
            content_index=0,
            delta="x",
        )
        assert raw.endswith("\n\n")


class TestOutputTextDoneEvent:
    def test_format(self):
        raw = format_output_text_done_event(
            seq=10,
            item_id="msg_abc",
            output_index=0,
            content_index=0,
            text="Hello world",
        )
        data = _assert_required_fields(raw, "response.output_text.done")
        assert data["item_id"] == "msg_abc"
        assert data["text"] == "Hello world"


class TestContentPartDoneEvent:
    def test_format(self):
        part = {"type": "output_text", "text": "Hello world"}
        raw = format_content_part_done_event(
            seq=11,
            item_id="msg_abc",
            output_index=0,
            content_index=0,
            part=part,
        )
        data = _assert_required_fields(raw, "response.content_part.done")
        assert data["part"]["text"] == "Hello world"


class TestOutputItemDoneEvent:
    def test_format(self):
        item = {
            "id": "msg_abc",
            "type": "message",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello world"}],
            "role": "assistant",
        }
        raw = format_output_item_done_event(seq=12, output_index=0, item=item)
        data = _assert_required_fields(raw, "response.output_item.done")
        assert data["item"]["status"] == "completed"


class TestFunctionCallArgumentsDeltaEvent:
    def test_format(self):
        raw = format_function_call_arguments_delta_event(
            seq=5,
            response_id="resp_abc",
            item_id="fc_001",
            call_id="call_001",
            delta='{"city":',
        )
        data = _assert_required_fields(raw, "response.function_call_arguments.delta")
        assert data["response_id"] == "resp_abc"
        assert data["item_id"] == "fc_001"
        assert data["call_id"] == "call_001"
        assert data["delta"] == '{"city":'


class TestFunctionCallArgumentsDoneEvent:
    def test_format(self):
        raw = format_function_call_arguments_done_event(
            seq=8,
            response_id="resp_abc",
            item_id="fc_001",
            call_id="call_001",
            arguments='{"city":"London"}',
        )
        data = _assert_required_fields(raw, "response.function_call_arguments.done")
        assert data["arguments"] == '{"city":"London"}'
        assert data["call_id"] == "call_001"


class TestResponseCompletedEvent:
    def test_format(self):
        response_data = {
            "object": "response",
            "model": "gpt-4o",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        raw = format_response_completed_event("resp_abc", seq=13, response_data=response_data)
        data = _assert_required_fields(raw, "response.completed")
        assert data["response"]["id"] == "resp_abc"
        assert data["response"]["status"] == "completed"
        assert data["response"]["usage"]["total_tokens"] == 15

    def test_ends_with_double_newline(self):
        raw = format_response_completed_event("resp_x", seq=0, response_data={"output": []})
        assert raw.endswith("\n\n")


class TestErrorEvent:
    def test_format(self):
        raw = format_error_event({"code": "server_error", "message": "upstream timeout"}, seq=99)
        data = _assert_required_fields(raw, "error")
        assert data["code"] == "server_error"
        assert data["message"] == "upstream timeout"
        assert data["sequence_number"] == 99

    def test_caller_cannot_overwrite_required_fields(self):
        """format_error_event must protect 'type' and 'sequence_number' from caller overwrite."""
        raw = format_error_event(
            {"type": "evil_event", "sequence_number": 9999, "code": "test", "message": "hi"},
            seq=5,
        )
        data = _assert_required_fields(raw, "error")
        assert data["type"] == "error"
        assert data["sequence_number"] == 5


class TestSSEFormatValidity:
    """All events must be valid SSE: 'event: <type>\\ndata: <json>\\n\\n'."""

    def test_all_events_have_valid_sse_structure(self):
        events = [
            format_response_created_event("r1", seq=0),
            format_response_in_progress_event("r1", seq=1),
            format_output_item_added_event(seq=2, output_index=0, item={"id": "m1", "type": "message"}),
            format_content_part_added_event(
                seq=3, item_id="m1", output_index=0, content_index=0, part={"type": "output_text", "text": ""}
            ),
            format_output_text_delta_event(
                seq=4, response_id="r1", item_id="m1", output_index=0, content_index=0, delta="hi"
            ),
            format_output_text_done_event(seq=5, item_id="m1", output_index=0, content_index=0, text="hi"),
            format_content_part_done_event(
                seq=6, item_id="m1", output_index=0, content_index=0, part={"type": "output_text", "text": "hi"}
            ),
            format_output_item_done_event(
                seq=7, output_index=0, item={"id": "m1", "type": "message", "status": "completed"}
            ),
            format_response_completed_event("r1", seq=8, response_data={"output": []}),
            format_function_call_arguments_delta_event(
                seq=3, response_id="r1", item_id="fc_1", call_id="call_1", delta="{}"
            ),
            format_function_call_arguments_done_event(
                seq=4, response_id="r1", item_id="fc_1", call_id="call_1", arguments="{}"
            ),
            format_error_event({"code": "x", "message": "y"}),
        ]
        for raw in events:
            assert raw.startswith("event: "), f"Missing 'event:' prefix: {raw!r}"
            assert raw.endswith("\n\n"), f"Missing double-newline ending: {raw!r}"
            lines = raw.split("\n")
            data_lines = [line for line in lines if line.startswith("data: ")]
            assert len(data_lines) == 1, f"Expected 1 data line, got {len(data_lines)} in: {raw!r}"
