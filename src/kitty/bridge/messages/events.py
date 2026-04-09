"""SSE event formatters for the Anthropic Messages API bridge."""

from __future__ import annotations

import json

__all__ = [
    "format_content_block_delta_event",
    "format_content_block_start_event",
    "format_content_block_stop_event",
    "format_error_event",
    "format_message_delta_event",
    "format_message_start_event",
    "format_message_stop_event",
    "format_ping_event",
]


def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def format_message_start_event(message_data: dict) -> str:
    """Emit ``message_start`` with the initial message object."""
    return _sse(
        "message_start",
        {
            "type": "message_start",
            "message": message_data,
        },
    )


def format_content_block_start_event(index: int, content_block: dict) -> str:
    """Emit ``content_block_start`` to open a content block."""
    return _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        },
    )


def format_content_block_delta_event(index: int, delta: dict) -> str:
    """Emit ``content_block_delta`` for incremental content."""
    return _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        },
    )


def format_content_block_stop_event(index: int) -> str:
    """Emit ``content_block_stop`` to close a content block."""
    return _sse(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


def format_message_delta_event(delta: dict, usage: dict) -> str:
    """Emit ``message_delta`` with stop_reason and cumulative usage."""
    return _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": delta,
            "usage": usage,
        },
    )


def format_message_stop_event() -> str:
    """Emit ``message_stop`` as the final streaming event."""
    return _sse("message_stop", {"type": "message_stop"})


def format_ping_event() -> str:
    """Emit ``ping`` keepalive event."""
    return _sse("ping", {"type": "ping"})


def format_error_event(error_data: dict) -> str:
    """Emit ``error`` when something goes wrong."""
    return _sse("error", error_data)
