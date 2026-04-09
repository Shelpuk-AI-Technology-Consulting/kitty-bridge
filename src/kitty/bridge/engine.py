"""Shared translation primitives for bridge protocol adapters.

Provides:
- TranslationEngine: static methods for reason mapping, tool-call and usage builders.
- ToolCallBuffer: incremental JSON argument buffering for streaming tool calls.
"""

from __future__ import annotations

import json
import uuid

__all__ = ["ToolCallBuffer", "ToolCallBufferError", "TranslationEngine"]

# ── Finish-reason mappings (shared across both protocols) ──────────────────

_FINISH_REASON_MAP: dict[str | None, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    None: "end_turn",
}

_STOP_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


class TranslationEngine:
    """Stateless utility class with shared translation helpers."""

    @staticmethod
    def map_finish_reason(finish_reason: str | None) -> str:
        """Map a Chat Completions ``finish_reason`` to a protocol stop reason.

        Unknown values pass through unchanged.
        """
        return _FINISH_REASON_MAP.get(finish_reason, finish_reason)  # type: ignore[return-value]

    @staticmethod
    def map_stop_reason_to_finish_reason(stop_reason: str) -> str:
        """Map a protocol ``stop_reason`` back to a Chat Completions ``finish_reason``.

        Unknown values pass through unchanged.
        """
        return _STOP_REASON_MAP.get(stop_reason, stop_reason)

    @staticmethod
    def build_tool_call(tool_name: str, arguments_json: str) -> dict:
        """Construct a normalized tool-call dict."""
        return {
            "id": f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments_json,
            },
        }

    @staticmethod
    def build_usage(input_tokens: int, output_tokens: int) -> dict:
        """Construct a usage dict."""
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


# ── Tool-call buffering ───────────────────────────────────────────────────


class ToolCallBufferError(Exception):
    """Raised when ToolCallBuffer exceeds limits or contains invalid JSON."""


class ToolCallBuffer:
    """Incremental buffer for partial JSON tool-call arguments from SSE streams.

    Callers append chunks and finalize once the stream ends. On error the caller
    is responsible for emitting a protocol-specific SSE ``error`` event.
    """

    def __init__(self, max_size: int = 1_000_000) -> None:
        self._max_size = max_size
        self._chunks: list[str] = []
        self._total_len: int = 0

    def append(self, chunk: str) -> None:
        """Buffer a partial JSON argument string.

        ``max_size`` counts **characters**, not UTF-8 bytes. This is adequate for
        typical tool-call argument JSON (small, mostly ASCII).

        Raises:
            ToolCallBufferError: If accumulated size exceeds ``max_size``.
        """
        self._total_len += len(chunk)
        if self._total_len > self._max_size:
            raise ToolCallBufferError(f"Tool call arguments exceeded max size ({self._max_size} chars)")
        self._chunks.append(chunk)

    def finalize(self) -> str:
        """Return the complete buffered JSON string.

        Raises:
            ToolCallBufferError: If buffer is empty or accumulated text is not valid JSON.
        """
        if not self._chunks:
            raise ToolCallBufferError("Tool call buffer is empty on finalize")
        text = "".join(self._chunks)
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolCallBufferError(f"Invalid JSON in tool call arguments: {exc}") from exc
        # Clear state after successful finalize
        self._chunks = []
        self._total_len = 0
        return text

    def reset(self) -> None:
        """Clear buffer state."""
        self._chunks = []
        self._total_len = 0
