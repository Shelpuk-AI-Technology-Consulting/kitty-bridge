"""Unit tests for the native-passthrough preamble hold (KBR-155).

The native Messages path forwards upstream bytes verbatim, so before the hold
existed an empty reply reached Claude Code before anything could judge it.
:class:`~kitty.bridge.preamble_hold.PreambleHold` withholds the stream's leading
bytes until content arrives. These tests pin the release rule decided in
``.system_design/TEST_SUITE.md`` §11 Q14(b) and its amendments D1 and D5 — with
D2 as amended by KBR-241: an ``error`` event before content is recorded for the
caller to retry on, not delivered — and the verbatim-replay guarantee that keeps
the path a passthrough.

Covers ``.requirements/20260913T134250Z_native_preamble_hold`` R1, R2, R3 and the
stop-reason half of R6, plus
``.requirements/20260913T234738Z_kbr241_preemission_error_recovery`` R1 and R2.
"""

from __future__ import annotations

import json

import pytest

from kitty.bridge.preamble_hold import MAX_HELD_BYTES, PreambleHold


def _sse(event_type: str, payload: dict) -> bytes:
    """Render one Anthropic SSE event exactly as an upstream would.

    Args:
        event_type: The SSE ``event:`` name.
        payload: The JSON object carried on the ``data:`` line.

    Returns:
        The encoded event, terminated by a blank line.
    """
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


MESSAGE_START = _sse(
    "message_start",
    {
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "m"},
    },
)
PING = _sse("ping", {"type": "ping"})
EMPTY_TEXT_START = _sse(
    "content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
)
TEXT_DELTA = _sse(
    "content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}
)
BLOCK_STOP = _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
END_TURN = _sse(
    "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}
)
MAX_TOKENS = _sse(
    "message_delta", {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 9}}
)
MESSAGE_STOP = _sse("message_stop", {"type": "message_stop"})
THINKING_START = _sse(
    "content_block_start",
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
)
THINKING_DELTA = _sse(
    "content_block_delta",
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
)
SIGNATURE_DELTA = _sse(
    "content_block_delta",
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
)
ERROR_EVENT = _sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

# An empty reply as a well-formed stream: every event of the grammar, no content.
CONTENTLESS = MESSAGE_START + PING + EMPTY_TEXT_START + BLOCK_STOP + END_TURN + MESSAGE_STOP


def _feed_all(hold: PreambleHold, chunks: list[bytes]) -> bytes:
    """Feed ``chunks`` in order and return everything the hold let through.

    Args:
        hold: The hold under test.
        chunks: Upstream chunks, in arrival order.

    Returns:
        The concatenation of every byte string ``feed`` returned.
    """
    return b"".join(hold.feed(chunk) for chunk in chunks)


class TestHolding:
    """R1 — a stream with no content yet writes nothing."""

    def test_contentless_stream_is_withheld_entirely(self):
        hold = PreambleHold()
        assert _feed_all(hold, [CONTENTLESS]) == b""
        assert hold.released is False

    @pytest.mark.parametrize(
        "event",
        [MESSAGE_START, PING, EMPTY_TEXT_START, BLOCK_STOP, END_TURN, MESSAGE_STOP],
        ids=["message_start", "ping", "empty_text_start", "block_stop", "message_delta", "message_stop"],
    )
    def test_no_single_grammar_event_without_content_releases(self, event):
        hold = PreambleHold()
        assert hold.feed(event) == b""
        assert hold.released is False

    def test_thinking_only_reply_is_withheld(self):
        """Q14(b): a reply the user cannot read is not content."""
        hold = PreambleHold()
        stream = MESSAGE_START + THINKING_START + THINKING_DELTA + SIGNATURE_DELTA + BLOCK_STOP + END_TURN
        assert hold.feed(stream) == b""
        assert hold.released is False

    def test_thinking_delta_releases_nothing_even_without_its_block_start(self):
        """The delta type alone marks thinking, so a shim that skips the start is judged the same."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + THINKING_DELTA + SIGNATURE_DELTA) == b""
        assert hold.released is False

    def test_unparseable_data_line_is_held_not_fatal(self):
        hold = PreambleHold()
        assert hold.feed(b"event: message_start\ndata: {not json\n\n") == b""
        assert hold.released is False

    def test_malformed_index_does_not_raise(self):
        """Upstream bytes are untrusted; an unhashable index must not escape as an exception."""
        hold = PreambleHold()
        odd = _sse("content_block_delta", {"type": "content_block_delta", "index": [0], "delta": "x"})
        assert hold.feed(MESSAGE_START + THINKING_START + odd) != b""

    def test_empty_text_delta_is_not_content(self):
        """D6: a text block that streams only "" is a blank reply, however it got there."""
        hold = PreambleHold()
        empty = _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ""}},
        )
        assert hold.feed(MESSAGE_START + EMPTY_TEXT_START + empty + BLOCK_STOP + END_TURN + MESSAGE_STOP) == b""
        assert hold.released is False

    def test_held_size_and_head_describe_the_withheld_bytes(self):
        hold = PreambleHold()
        hold.feed(MESSAGE_START + PING)
        assert hold.held_size == len(MESSAGE_START + PING)
        assert hold.head(10) == MESSAGE_START[:10]

    @pytest.mark.parametrize(
        "payload",
        [b'{"type":"message_delta","n":' + b"9" * 5000 + b"}", b"[" * 200_000],
        ids=["integer_past_the_digit_limit", "nesting_past_the_recursion_limit"],
    )
    def test_hostile_json_is_held_not_raised(self, payload):
        """Untrusted bytes: json.loads raises ValueError or RecursionError here, not JSONDecodeError."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b"data: " + payload + b"\n\n") == b""
        assert hold.released is False

    def test_zero_chunks_leave_the_hold_unreleased(self):
        assert PreambleHold().released is False


class TestReleaseRule:
    """R2 — each release trigger, and only those."""

    def test_text_delta_releases(self):
        hold = PreambleHold()
        hold.feed(MESSAGE_START + EMPTY_TEXT_START)
        assert hold.feed(TEXT_DELTA) != b""
        assert hold.released is True

    def test_delta_on_a_non_thinking_block_after_thinking_releases(self):
        text_at_1 = _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi"}},
        )
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + THINKING_START + THINKING_DELTA + BLOCK_STOP) == b""
        assert hold.feed(text_at_1) != b""

    def test_non_thinking_delta_type_on_a_thinking_index_releases_nothing(self):
        """A block opened as thinking stays thinking whatever its deltas claim."""
        hold = PreambleHold()
        odd = _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        )
        assert hold.feed(MESSAGE_START + THINKING_START + odd) == b""

    def test_redacted_thinking_block_start_releases_nothing(self):
        hold = PreambleHold()
        redacted = _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "redacted_thinking", "data": "x"}},
        )
        assert hold.feed(MESSAGE_START + redacted) == b""

    def test_input_json_delta_releases(self):
        hold = PreambleHold()
        delta = _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ""}},
        )
        assert hold.feed(MESSAGE_START + delta) != b""

    @pytest.mark.parametrize(
        "block",
        [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "a_block_type_not_yet_invented"},
        ],
        ids=["tool_use", "server_tool_use", "web_search_tool_result", "unknown_type"],
    )
    def test_block_that_arrives_whole_releases_on_its_start(self, block):
        """D1: these carry their content in content_block_start and may send no delta at all."""
        hold = PreambleHold()
        start = _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block})
        assert hold.feed(MESSAGE_START + start) != b""

    def test_text_block_started_with_text_releases(self):
        """D1: a shim that pre-fills the text block sends no delta."""
        hold = PreambleHold()
        start = _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Hello"}},
        )
        assert hold.feed(MESSAGE_START + start) != b""

    def test_error_event_is_recorded_not_released(self):
        """D2 as amended by KBR-241: a pre-content error is judged, not delivered."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + ERROR_EVENT) == b""
        assert hold.released is False
        assert hold.error_seen is True
        assert hold.error_event == {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

    def test_error_type_records_without_an_event_name(self):
        """A shim that sends only ``data:`` lines still has its error recognised."""
        hold = PreambleHold()
        bare = b'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
        assert hold.feed(MESSAGE_START + bare) == b""
        assert hold.released is False
        assert hold.error_seen is True
        assert hold.error_event == {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

    def test_error_event_name_records_even_when_its_data_is_not_json(self):
        """The SDK raises on ``event: error`` by name, so a shim's unparseable error body is still an error."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b"event: error\ndata: upstream exploded\n\n") == b""
        assert hold.released is False
        assert hold.error_seen is True
        assert hold.error_event is None

    def test_error_event_data_line_is_recorded_as_received(self):
        """The data line fills the payload whatever it holds; only its ``error`` dict is usable later."""
        hold = PreambleHold()
        odd = b'event: error\ndata: {"type":"content_block_delta","index":0}\n\n'
        assert hold.feed(MESSAGE_START + odd) == b""
        assert hold.released is False
        assert hold.error_seen is True
        assert hold.error_event == {"type": "content_block_delta", "index": 0}

    def test_second_error_overwrites_the_first(self):
        """Last error wins, as the last stop_reason does."""
        hold = PreambleHold()
        second = _sse("error", {"type": "error", "error": {"type": "api_error", "message": "later"}})
        assert hold.feed(MESSAGE_START + ERROR_EVENT + second) == b""
        assert hold.error_event == {"type": "error", "error": {"type": "api_error", "message": "later"}}

    def test_error_event_completes_only_at_its_blank_line(self):
        """KBR-241: a chunk boundary between the name line and the data line must not truncate the payload."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b"event: error\n") == b""
        assert hold.error_seen is True
        assert hold.error_event_complete is False
        assert hold.feed(b'data: {"type":"error","error":{"type":"overloaded_error"}}\n\n') == b""
        assert hold.error_event == {"type": "error", "error": {"type": "overloaded_error"}}
        assert hold.error_event_complete is True

    def test_error_event_completes_at_a_following_event_header(self):
        """A shim that omits the blank line still ends the error event at the next ``event:`` line."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b'event: error\ndata: {"type":"error"}\nevent: ping\n') == b""
        assert hold.error_event_complete is True

    def test_bare_data_error_completes_at_its_blank_line(self):
        """The type-only error shape has no name line: its blank line alone must complete it."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b'data: {"type":"error","error":{"type":"overloaded_error"}}\n') == b""
        assert hold.error_seen is True
        assert hold.error_event_complete is False
        assert hold.feed(b"\n") == b""
        assert hold.error_event == {"type": "error", "error": {"type": "overloaded_error"}}
        assert hold.error_event_complete is True

    def test_bare_data_error_completes_at_a_following_event_header(self):
        """And a following header completes it too — `_error_seen` alone gates, not the header's name."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b'data: {"type":"error","error":{"type":"overloaded_error"}}\n') == b""
        assert hold.error_event_complete is False
        assert hold.feed(b"event: ping\n\n") == b""
        assert hold.error_event_complete is True

    def test_missing_data_type_falls_back_to_the_event_name(self):
        """The SDK fills a missing ``type`` from the SSE event name; a type-less delta is judged as one."""
        hold = PreambleHold()
        typeless = b'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        assert hold.feed(MESSAGE_START + typeless) != b""

    def test_error_event_name_records_with_no_data_line(self):
        """The SDK's SSE decoder dispatches a data-less ``event: error`` and raises on it."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + b"event: error\n\n") == b""
        assert hold.released is False
        assert hold.error_seen is True
        assert hold.error_event is None

    def test_event_name_does_not_leak_past_its_blank_line(self):
        """A type-less data line after an event has ended is not judged by that event's name."""
        hold = PreambleHold()
        stream = b"event: content_block_delta\n\n" + b'data: {"index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        assert hold.feed(MESSAGE_START + stream) == b""

    def test_exceeding_the_cap_releases(self):
        """D5: the hold fails open past its bound rather than growing without limit."""
        hold = PreambleHold(max_held_bytes=len(MESSAGE_START) + 10)
        assert hold.feed(MESSAGE_START) == b""
        assert hold.feed(THINKING_START) != b""
        assert hold.released is True

    def test_reaching_the_cap_exactly_does_not_release(self):
        hold = PreambleHold(max_held_bytes=len(MESSAGE_START))
        assert hold.feed(MESSAGE_START) == b""

    def test_too_many_thinking_blocks_release(self):
        """The thinking-index set is bounded like the byte buffer, and fails open the same way."""
        hold = PreambleHold()
        starts = [
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}},
            )
            for i in range(257)
        ]
        assert hold.feed(MESSAGE_START + b"".join(starts[:256])) == b""
        assert hold.feed(starts[256]) != b""

    def test_default_cap_is_ten_mebibytes(self):
        assert MAX_HELD_BYTES == 10 * 1024 * 1024

    def test_unterminated_line_past_the_cap_releases(self):
        """An upstream that never sends a newline is bounded by the same cap."""
        hold = PreambleHold(max_held_bytes=64)
        assert hold.feed(b"data: " + b"x" * 100) != b""


class TestVerbatimReplay:
    """R3 — what the client receives is exactly what upstream sent."""

    def test_release_returns_every_held_byte_then_the_trigger(self):
        hold = PreambleHold()
        preamble = MESSAGE_START + PING + EMPTY_TEXT_START
        assert hold.feed(preamble) == b""
        assert hold.feed(TEXT_DELTA) == preamble + TEXT_DELTA

    def test_after_release_each_chunk_passes_through_as_received(self):
        hold = PreambleHold()
        hold.feed(MESSAGE_START + TEXT_DELTA)
        assert hold.feed(b"not even SSE") == b"not even SSE"
        assert hold.feed(END_TURN) == END_TURN

    def test_non_canonical_bytes_survive_the_hold(self):
        """§4.3 C2: a hold that re-serialised parsed events would change key order and spacing."""
        odd_start = (
            b'event:message_start\r\ndata:{"message":{"role":"assistant","id":"m"},  "type":"message_start"}\r\n\r\n'
        )
        odd_delta = b'data: {"index":0,"delta":{"text":"Hi","type":"text_delta"},"type":"content_block_delta"}\n\n'
        stream = odd_start + odd_delta + MESSAGE_STOP
        hold = PreambleHold()
        assert _feed_all(hold, [stream]) == stream
        assert hold.released is True

    @pytest.mark.parametrize(
        "trigger",
        [
            TEXT_DELTA,
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "Read"}},
            ),
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "Hi"}},
            ),
        ],
        ids=["text_delta", "tool_use_start", "prefilled_text_start"],
    )
    def test_every_split_point_replays_the_stream_exactly(self, trigger):
        """Chunks arrive on arbitrary byte boundaries; each trigger may straddle any of them."""
        stream = MESSAGE_START + EMPTY_TEXT_START + trigger + BLOCK_STOP + END_TURN + MESSAGE_STOP
        for cut in range(len(stream) + 1):
            hold = PreambleHold()
            out = _feed_all(hold, [stream[:cut], stream[cut:]])
            assert out == stream, f"split at byte {cut} changed the bytes"
            assert hold.released is True, f"split at byte {cut} never released"

    @pytest.mark.parametrize(
        "trigger",
        [
            b"event: error\ndata: upstream exploded\n\n",
            b'data: {"type":"error","error":{"type":"overloaded_error"}}\n\n',
        ],
        ids=["error_by_name", "error_by_type"],
    )
    def test_error_trigger_is_held_with_its_bytes_verbatim(self, trigger):
        """D2 as amended: an error ends nothing — the hold keeps every byte for the caller to judge."""
        stream = MESSAGE_START + EMPTY_TEXT_START + trigger
        for cut in range(len(stream) + 1):
            hold = PreambleHold()
            out = _feed_all(hold, [stream[:cut], stream[cut:]])
            assert out == b"", f"split at byte {cut} released an error stream"
            assert hold.released is False, f"split at byte {cut} released on an error"
            assert hold.error_seen is True, f"split at byte {cut} missed the error"
            assert hold.head(len(stream)) == stream, f"split at byte {cut} changed the held bytes"

    def test_content_after_an_error_still_releases(self):
        """R2: the judge is unchanged for content — an error first does not eat a later release."""
        hold = PreambleHold()
        out = hold.feed(MESSAGE_START + ERROR_EVENT + TEXT_DELTA)
        assert out == MESSAGE_START + ERROR_EVENT + TEXT_DELTA
        assert hold.released is True

    def test_cap_fails_open_even_after_an_error_trigger(self):
        """D5 boundary: the cap still fails open past an unjudged error, delivering as today."""
        hold = PreambleHold(max_held_bytes=len(MESSAGE_START + ERROR_EVENT))
        assert hold.feed(MESSAGE_START + ERROR_EVENT) == b""
        assert hold.feed(THINKING_START) != b""
        assert hold.released is True
        assert hold.error_seen is True

    def test_release_happens_only_once_the_trigger_line_is_complete(self):
        stream = MESSAGE_START + TEXT_DELTA
        newline = len(stream) - 2  # the data line's terminating newline, before the blank line
        hold = PreambleHold()
        assert hold.feed(stream[:newline]) == b""
        assert hold.feed(stream[newline : newline + 1]) == stream[: newline + 1]


class TestStopReason:
    """R6 — the hold remembers why a contentless stream stopped."""

    def test_max_tokens_stop_reason_is_recorded(self):
        hold = PreambleHold()
        hold.feed(MESSAGE_START + THINKING_START + THINKING_DELTA + BLOCK_STOP + MAX_TOKENS + MESSAGE_STOP)
        assert hold.released is False
        assert hold.stop_reason == "max_tokens"

    def test_end_turn_stop_reason_is_recorded(self):
        hold = PreambleHold()
        hold.feed(CONTENTLESS)
        assert hold.stop_reason == "end_turn"

    def test_no_message_delta_means_no_stop_reason(self):
        hold = PreambleHold()
        hold.feed(MESSAGE_START + MESSAGE_STOP)
        assert hold.stop_reason is None

    def test_stop_reason_after_an_error_is_not_recorded(self):
        """KBR-241: the error is terminal, so a later stop reason is noise — in any chunk framing."""
        hold = PreambleHold()
        assert _feed_all(hold, [MESSAGE_START + ERROR_EVENT + MAX_TOKENS + MESSAGE_STOP]) == b""
        assert hold.stop_reason is None
        assert hold.error_seen is True

    def test_stop_reason_before_an_error_is_recorded(self):
        """A truncation that preceded the error stands, and the 400 will govern at the held end."""
        hold = PreambleHold()
        assert hold.feed(MESSAGE_START + MAX_TOKENS + ERROR_EVENT) == b""
        assert hold.stop_reason == "max_tokens"
        assert hold.error_seen is True
