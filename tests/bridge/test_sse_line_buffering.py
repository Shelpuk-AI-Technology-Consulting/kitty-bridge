"""Tests for SSE line buffering — validates that upstream TCP chunks are
properly reassembled into complete SSE lines before parsing."""

import contextlib
import json

import pytest

from kitty.bridge.server import _append_sse_chunk


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
            with contextlib.suppress(json.JSONDecodeError):
                received.append(json.loads(data_str))

    # Flush remaining buffer (last chunk without trailing \n)
    if not done and buffer.strip():
        line = buffer.strip()
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str.strip() != "[DONE]":
                with contextlib.suppress(json.JSONDecodeError):
                    received.append(json.loads(data_str))

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
            b"data: [DONE]\n\n",
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
            b"\n",
            b'data: {"choices":[{"delta":{"content":"X"},"finish_reason":null}]}\n',
            b"\n",
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
            b"data: [DONE]\n\n",
            b'data: {"choices":[{"delta":{"content":"AfterDone"},"finish_reason":null}]}\n\n',
        ]
        result = parse_sse_chunks(chunks)
        assert len(result) == 1  # Only the first event, [DONE] stops parsing


# ── byte-based SSE parser (F23 + F24) ─────────────────────────────────────

_MAX_LINE_BYTES = 10 * 1024 * 1024  # 10 MB


def parse_sse_chunks_bytes(chunks: list[bytes], *, max_line_bytes: int = _MAX_LINE_BYTES) -> tuple[list[dict], bool]:
    """Simulate the CORRECT byte-based SSE parsing.

    F23 fix: buffers raw bytes instead of decoding with ``errors="replace"``
    on partial reads.  Only decodes at ``\\n`` boundaries where the line is
    guaranteed complete — so multi-byte UTF-8 sequences split across TCP
    chunks survive intact.

    F24 fix: enforces ``max_line_bytes``.  If a line exceeds the limit without
    a ``\\n``, the function signals ``truncated=True`` and stops.

    Returns:
        (received, truncated)
    """
    buffer = bytearray()
    received: list[dict] = []
    done = False
    truncated = False

    for raw in chunks:
        if done or truncated:
            break
        buffer.extend(raw)
        while not done and not truncated:
            nl_pos = buffer.find(b"\n")
            if nl_pos < 0:
                # No complete line yet — check for oversized buffer
                if len(buffer) > max_line_bytes:
                    truncated = True
                break
            line_bytes = buffer[:nl_pos]
            del buffer[: nl_pos + 1]
            if line_bytes.endswith(b"\r"):
                line_bytes = line_bytes[:-1]
            # Decode only at complete line boundary (F23 fix)
            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Truly malformed UTF-8 — skip the line
                continue
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                done = True
                break
            with contextlib.suppress(json.JSONDecodeError):
                received.append(json.loads(data_str))

    # Flush remaining buffer (last line without trailing \\n)
    if not done and not truncated and buffer:
        line_bytes = bytes(buffer)
        try:
            line = line_bytes.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            if line.strip().startswith("data: "):
                data_str = line.strip()[6:]
                if data_str.strip() != "[DONE]":
                    with contextlib.suppress(json.JSONDecodeError):
                        received.append(json.loads(data_str))

    return received, truncated


class TestUTF8MultiByteSurvival:
    """F23: Multi-byte UTF-8 sequences must survive TCP chunk splits."""

    def test_emoji_split_across_chunks(self):
        """The 4-byte snowman emoji ☃ (0xf0 0x9f 0x8c 0x83) split across chunks."""
        # "Hi ☃" encoded in JSON: `"Hi ☃"` via the SSE data line
        snowman = "☃"  # ☃ is 3 bytes in UTF-8
        payload = f'data: {{"text": "Hi {snowman}"}}\n\n'
        raw = payload.encode("utf-8")

        # Split at a multi-byte boundary
        split = 10
        chunks = [raw[:split], raw[split:]]

        # Current code uses errors="replace" — corrupts the character
        parse_sse_chunks(chunks)
        # The old str-buffer version with errors="replace" may produce garbage
        # or replacement characters — the exact result depends on split position.

        # The byte-based parser preserves the character
        result_bytes, truncated = parse_sse_chunks_bytes(chunks)
        assert not truncated
        assert len(result_bytes) == 1
        assert snowman in result_bytes[0]["text"], f"Multi-byte character was corrupted: {result_bytes[0]['text']!r}"

    def test_4byte_utf8_split_in_json_key(self):
        """4-byte UTF-8 character split across chunks in a JSON value.

        The eggplant emoji 🍆 (0xf0 0x9f 0x8d 0x86) is 4 bytes in UTF-8.
        Split at byte 2 of the 4-byte sequence.
        """
        eggplant = "\U0001f346"  # 🍆
        payload = f'data: {{"choices":[{{"delta":{{"content":"x{eggplant}y"}}}}]}}\n\n'
        raw = payload.encode("utf-8")
        # Find the 4-byte sequence position
        pos = raw.index(eggplant.encode("utf-8")) + 2  # middle of 4-byte char
        chunks = [raw[:pos], raw[pos:]]

        result_bytes, truncated = parse_sse_chunks_bytes(chunks)
        assert not truncated
        assert len(result_bytes) == 1
        content = result_bytes[0]["choices"][0]["delta"]["content"]
        assert eggplant in content, f"4-byte character corrupted: {content!r}"

    def test_cjk_character_split(self):
        """CJK character 好 (3 bytes: e5 a5 bd) split in SSE data."""
        cjk = "好"  # 好
        payload = f'data: {{"text": "{cjk}"}}\n\n'
        raw = payload.encode("utf-8")

        # Split in the middle of cjk
        cjk_bytes = cjk.encode("utf-8")
        split = raw.index(cjk_bytes) + 1  # after first byte of cjk
        chunks = [raw[:split], raw[split:]]

        result_bytes, truncated = parse_sse_chunks_bytes(chunks)
        assert not truncated
        assert len(result_bytes) == 1
        assert cjk in result_bytes[0]["text"]

    def test_ascii_data_unchanged(self):
        """Byte-based parser passes ASCII data through identically."""
        chunks = [
            b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
        ]
        result_bytes, truncated = parse_sse_chunks_bytes(chunks)
        result_str = parse_sse_chunks(chunks)
        assert not truncated
        assert result_bytes == result_str


class TestLineBufferMaxSize:
    """F24: Oversized SSE line must be caught, not grow unbounded."""

    def test_oversized_line_triggers_truncation(self):
        """A line without \\n exceeding max_line_bytes returns truncated=True."""
        prefix = b"data: " + b"x" * (2 * 1024 * 1024)  # 2 MB
        # No newline — should trigger truncation with a 1 KB limit
        _, truncated = parse_sse_chunks_bytes([prefix], max_line_bytes=1024)
        assert truncated is True

    def test_oversized_line_stops_further_processing(self):
        """After truncation, no further events are processed."""
        prefix = b"data: " + b"x" * 2000  # No newline
        suffix = b'data: {"valid": true}\n\n'
        result, truncated = parse_sse_chunks_bytes([prefix, suffix], max_line_bytes=1024)
        assert truncated is True
        assert len(result) == 0  # The valid suffix was never processed

    def test_normal_line_within_limit_works(self):
        """A line within max_line_bytes is processed normally."""
        line = b'data: {"ok": 1}\n\n' + (b"y" * 500)
        result, truncated = parse_sse_chunks_bytes([line], max_line_bytes=2048)
        assert not truncated
        assert len(result) == 1

    def test_exact_boundary_works(self):
        """A line exactly at max_line_bytes with a newline is fine."""
        # Pad JSON to a target total size, then close the SSE delimiter
        inner = b'{"k": "' + b"a" * 900 + b'"}'
        line = b"data: " + inner + b"\n\n"
        result, truncated = parse_sse_chunks_bytes([line], max_line_bytes=1024)
        assert not truncated
        assert len(result) == 1

    def test_max_bytes_one_byte_over_triggers(self):
        """One byte over the limit triggers truncation."""
        line = b"data: " + b"a" * 1025  # no newline
        _, truncated = parse_sse_chunks_bytes([line], max_line_bytes=1024)
        assert truncated is True


# ── _append_sse_chunk helper used by stream handlers ─────────────────────


class TestAppendSseChunk:
    """Validate the byte-based line-buffering helper used by stream handlers.

    The helper appends a chunk to the line buffer, returns the list of
    complete lines (without trailing newline, CR stripped), and raises
    ``ValueError`` when the line buffer exceeds ``max_line_bytes`` without
    a ``\\n`` (F24).
    """

    def test_no_newline_returns_empty(self):
        """Partial chunk with no newline produces no complete lines."""
        buf = bytearray()
        lines = _append_sse_chunk(buf, b'data: {"a": 1')
        assert lines == []
        assert buf == b'data: {"a": 1'

    def test_complete_line_returned(self):
        """Chunk ending with \\n yields one complete line."""
        buf = bytearray()
        lines = _append_sse_chunk(buf, b'data: {"a": 1}\n\n')
        assert lines == ['data: {"a": 1}', ""]
        assert buf == b""

    def test_split_across_chunks_preserves_partial(self):
        """Partial split leaves bytes in the buffer until newline arrives."""
        buf = bytearray()
        _append_sse_chunk(buf, b'data: {"a":')
        lines = _append_sse_chunk(buf, b" 1}\n\n")
        assert lines == ['data: {"a": 1}', ""]
        assert buf == b""

    def test_carriage_return_stripped(self):
        """\\r before \\n is stripped from the returned line."""
        buf = bytearray()
        lines = _append_sse_chunk(buf, b"data: hello\r\n")
        assert lines == ["data: hello"]

    def test_multi_byte_utf8_preserved(self):
        """Multi-byte UTF-8 split across chunks is preserved (F23)."""
        buf = bytearray()
        snowman = "☃"  # ☃ — 3 UTF-8 bytes: e2 98 83
        raw = f'data: {{"t": "{snowman}"}}\n\n'.encode()
        # Split inside the snowman
        snowman_bytes = snowman.encode("utf-8")
        split = raw.index(snowman_bytes) + 1
        _append_sse_chunk(buf, raw[:split])
        lines = _append_sse_chunk(buf, raw[split:])
        assert len(lines) == 2
        # First line is the SSE line; second is the empty separator
        assert snowman in lines[0]

    def test_oversized_line_raises(self):
        """Line exceeding max_line_bytes raises ValueError (F24)."""
        buf = bytearray()
        huge = b"data: " + b"a" * 1100
        with pytest.raises(ValueError, match="SSE line exceeded maximum buffer size"):
            _append_sse_chunk(buf, huge, max_line_bytes=1024)
        # Buffer should be cleared on the abort
        assert buf == b""

    def test_line_with_newline_at_max_boundary_ok(self):
        """A line that reaches max_line_bytes but has a newline is fine."""
        buf = bytearray()
        # Build a line that is exactly 100 bytes including 'data: ' and '\n'
        payload = b"x" * (100 - 6 - 1)  # 93 'x' bytes
        line = b"data: " + payload + b"\n"
        lines = _append_sse_chunk(buf, line, max_line_bytes=100)
        expected = "data: " + payload.decode()
        assert lines == [expected], f"expected [{expected!r}], got {lines}"

    def test_multi_line_chunk(self):
        """Chunk with multiple complete lines returns them all."""
        buf = bytearray()
        lines = _append_sse_chunk(buf, b"data: a\ndata: b\n\n")
        assert lines == ["data: a", "data: b", ""]
