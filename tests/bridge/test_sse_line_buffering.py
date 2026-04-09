"""Tests for SSE line buffering — validates that upstream TCP chunks are
properly reassembled into complete SSE lines before parsing."""

import json

import pytest


def parse_sse_chunks(chunks: list[bytes]) -> list[dict]:
    """Simulate the bridge's upstream SSE parsing with line buffering.

    This is the CORRECT parsing approach: accumulate a buffer, extract
    complete lines, then parse data: lines as JSON. After all chunks
    are consumed, flush any remaining data in the buffer.
    """
    buffer = ""
    received: list[dict] = []
    done = False

    for raw in chunks:
        if done:
            break
        buffer += raw.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                done = True
                break
            try:
                received.append(json.loads(data_str))
            except json.JSONDecodeError:
                pass

    # Flush remaining buffer (last chunk without trailing \n)
    if not done and buffer.strip():
        line = buffer.strip()
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str.strip() != "[DONE]":
                try:
                    received.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass

    return received


class TestSSELineBuffering:
    def test_complete_lines(self):
        """Each upstream chunk is a complete SSE line."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 2
        assert result[0]["choices"][0]["delta"]["content"] == "Hi"

    def test_split_across_chunks(self):
        """SSE data line split across two TCP chunks — the core bug scenario."""
        chunks = [
            b'data: {"choices":[{"delta":{"con',
            b'tent":"Hi"},"finish_reason":null}]}\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 1
        assert result[0]["choices"][0]["delta"]["content"] == "Hi"

    def test_multiple_lines_in_one_chunk(self):
        """Multiple SSE lines arrive in a single TCP chunk."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"A"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"B"},"finish_reason":null}]}\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 2
        assert result[0]["choices"][0]["delta"]["content"] == "A"
        assert result[1]["choices"][0]["delta"]["content"] == "B"

    def test_done_sentinel(self):
        """[DONE] sentinel is detected and stops parsing."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 1

    def test_three_way_split(self):
        """JSON data split across 3 chunks."""
        chunks = [
            b'data: {"choices":[{"de',
            b'lta":{"content":"hel',
            b'lo"},"finish_reason":null}]}\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 1
        assert result[0]["choices"][0]["delta"]["content"] == "hello"

    def test_empty_lines_ignored(self):
        """Empty SSE lines (event separators) are silently skipped."""
        chunks = [
            b'\n',
            b'data: {"choices":[{"delta":{"content":"X"},"finish_reason":null}]}\n',
            b'\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 1

    def test_no_newline_at_end(self):
        """Partial chunk without trailing newline is buffered until more arrives."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"X"}',
            # No newline yet — should buffer
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 0  # Not complete yet

        # Now deliver the rest
        chunks2 = chunks + [
            b',"finish_reason":null}]}\n\n',
        ]
        result2 = parse_sse_chunks(chunks2)
        assert len(result2) == 1

    def test_last_chunk_without_trailing_newline_is_flushed(self):
        """After all chunks are consumed, remaining buffer must be flushed."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',  # no trailing \n
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 2
        assert result[1]["choices"][0]["finish_reason"] == "stop"

    def test_done_sentinel_stops_parsing(self):
        """[DONE] sentinel stops parsing, remaining data is ignored."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            b'data: [DONE]\n\n',
            b'data: {"choices":[{"delta":{"content":"AfterDone"},"finish_reason":null}]}\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 1  # Only the first event, [DONE] stops parsing
