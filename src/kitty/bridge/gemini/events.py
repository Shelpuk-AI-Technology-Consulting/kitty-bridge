"""SSE event formatters for the Gemini API bridge.

Gemini streaming uses plain ``data: <json>\\n\\n`` SSE events — no event-type
prefix (unlike Responses API which uses ``event: <type>\\ndata: <json>\\n\\n``).
"""

from __future__ import annotations

import json

__all__ = ["format_gemini_sse"]


def format_gemini_sse(data: dict) -> str:
    """Format a single Gemini SSE event string.

    Gemini streaming SSE uses ``data: {json}\\n\\n`` without an event-type line.
    """
    return f"data: {json.dumps(data)}\n\n"
