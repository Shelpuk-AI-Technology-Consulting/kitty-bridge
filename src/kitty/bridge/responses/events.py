"""SSE event formatters for the OpenAI Responses API bridge.

Every event data payload includes:
- ``type``: matches the SSE event name (e.g. ``"response.created"``)
- ``sequence_number``: monotonically increasing per-stream counter
"""

from __future__ import annotations

import json

__all__ = [
    "format_content_part_added_event",
    "format_content_part_done_event",
    "format_error_event",
    "format_function_call_arguments_delta_event",
    "format_function_call_arguments_done_event",
    "format_output_item_added_event",
    "format_output_item_done_event",
    "format_output_text_delta_event",
    "format_output_text_done_event",
    "format_response_completed_event",
    "format_response_created_event",
    "format_response_in_progress_event",
]


def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def format_response_created_event(response_id: str, seq: int, model: str = "") -> str:
    """Emit ``response.created`` when a new response starts."""
    return _sse(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": seq,
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
                "usage": None,
            },
        },
    )


def format_response_in_progress_event(response_id: str, seq: int, model: str = "") -> str:
    """Emit ``response.in_progress`` after creation."""
    return _sse(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "sequence_number": seq,
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
                "usage": None,
            },
        },
    )


def format_output_item_added_event(
    seq: int,
    output_index: int,
    item: dict,
) -> str:
    """Emit ``response.output_item.added`` when a new output item starts."""
    return _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": seq,
            "output_index": output_index,
            "item": item,
        },
    )


def format_content_part_added_event(
    seq: int,
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict,
) -> str:
    """Emit ``response.content_part.added`` when a text content part starts."""
    return _sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "sequence_number": seq,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def format_output_text_delta_event(
    seq: int,
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> str:
    """Emit ``response.output_text.delta`` for incremental text."""
    return _sse(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": seq,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        },
    )


def format_output_text_done_event(
    seq: int,
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> str:
    """Emit ``response.output_text.done`` with the full accumulated text."""
    return _sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "sequence_number": seq,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
        },
    )


def format_content_part_done_event(
    seq: int,
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict,
) -> str:
    """Emit ``response.content_part.done`` when a content part is finalized."""
    return _sse(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "sequence_number": seq,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def format_output_item_done_event(
    seq: int,
    output_index: int,
    item: dict,
) -> str:
    """Emit ``response.output_item.done`` when an output item is finalized."""
    return _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": seq,
            "output_index": output_index,
            "item": item,
        },
    )


def format_function_call_arguments_delta_event(
    seq: int,
    response_id: str,
    item_id: str,
    call_id: str,
    delta: str,
) -> str:
    """Emit ``response.function_call_arguments.delta`` for tool-call argument chunks."""
    return _sse(
        "response.function_call_arguments.delta",
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": seq,
            "response_id": response_id,
            "item_id": item_id,
            "call_id": call_id,
            "delta": delta,
        },
    )


def format_function_call_arguments_done_event(
    seq: int,
    response_id: str,
    item_id: str,
    call_id: str,
    arguments: str,
) -> str:
    """Emit ``response.function_call_arguments.done`` with finalized arguments."""
    return _sse(
        "response.function_call_arguments.done",
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": seq,
            "response_id": response_id,
            "item_id": item_id,
            "call_id": call_id,
            "arguments": arguments,
        },
    )


def format_response_completed_event(response_id: str, seq: int, response_data: dict) -> str:
    """Emit ``response.completed`` when the full response is ready."""
    return _sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": seq,
            "response": {
                **response_data,
                "id": response_id,
            },
        },
    )


def format_error_event(error_data: dict, seq: int = 0) -> str:
    """Emit ``error`` when something goes wrong."""
    payload = {**error_data, "type": "error", "sequence_number": seq}
    return _sse("error", payload)
