"""Unit tests for the SSE-side ``tool_use`` assembler.

The native-passthrough path forwards upstream bytes to the client without
parsing them, so the only way to record what was forwarded is to read the
stream alongside it.  These tests pin that reconstruction, and — just as
importantly — pin that it can never break the stream it observes.

Covers ``.requirements/20260828T210855Z_tool_use_response_audit`` FR-3, FR-4
and FR-5.
"""

from __future__ import annotations

import json
import logging

import pytest

from kitty.bridge.tool_audit import (
    _MAX_BUFFERED_ARG_CHARS,
    _MAX_LINE_BYTES,
    _MAX_OPEN_BLOCKS,
    ToolUseAuditor,
)

from .test_tool_use_audit import STRUCTURED_OUTPUT_SCHEMA

AUDIT_LOGGER = "kitty.bridge.tool_audit"


def _sse(event_type: str, payload: dict) -> bytes:
    """Render one Anthropic SSE event exactly as an upstream would."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _start(name: str, index: int = 0) -> bytes:
    """A ``content_block_start`` opening a tool_use block."""
    return _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}},
        },
    )


def _delta(partial: str, index: int = 0) -> bytes:
    """An ``input_json_delta`` carrying a fragment of the argument JSON."""
    return _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        },
    )


def _stop(index: int = 0) -> bytes:
    """A ``content_block_stop`` closing a block."""
    return _sse("content_block_stop", {"type": "content_block_stop", "index": index})


def _tool_use_stream(name: str, argument_json: str, index: int = 0) -> bytes:
    """A minimal SSE stream carrying one complete tool_use block."""
    return _start(name, index) + _delta(argument_json, index) + _stop(index)


class TestAssembly:
    """AC-3.1 — the assembler reconstructs the input that was forwarded."""

    def test_tool_use_input_is_logged(self, caplog):
        auditor = ToolUseAuditor({}, backend="minimax-large")
        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("Read", '{"path":"a.py"}'))
            auditor.finish()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Read" in m and "a.py" in m for m in messages), messages

    @pytest.mark.parametrize("size", [1, 3, 7, 64])
    def test_assembly_is_independent_of_chunk_boundaries(self, size, caplog):
        """Upstream chunks are arbitrary TCP bytes — not lines, not events.

        Size 1 splits every multi-byte UTF-8 sequence and every JSON token.
        """
        stream = _tool_use_stream("Read", '{"path":"café.py"}')
        auditor = ToolUseAuditor({}, backend="b")
        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            for start in range(0, len(stream), size):
                auditor.feed(stream[start : start + size])
            auditor.finish()

        assert any("café.py" in r.getMessage() for r in caplog.records), f"chunk size {size}"

    def test_arguments_split_across_deltas_are_joined(self, caplog):
        """Real upstreams deliver argument JSON one fragment at a time."""
        auditor = ToolUseAuditor({}, backend="b")
        stream = _start("Read") + b"".join(_delta(f) for f in ('{"pa', 'th":"', 'a.py"}')) + _stop()

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(stream)
            auditor.finish()

        assert any("a.py" in r.getMessage() for r in caplog.records)

    def test_text_only_stream_reports_no_tool_use(self, caplog):
        """A stream with no tool_use must not invent one."""
        auditor = ToolUseAuditor({}, backend="b")
        stream = (
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            + _stop()
        )
        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(stream)
            auditor.finish()

        assert not any("tool_use" in r.getMessage() for r in caplog.records)

    def test_parallel_tool_calls_are_reported_separately(self, caplog):
        """AC-4.3 — two blocks, two reports, correctly de-interleaved by index."""
        auditor = ToolUseAuditor({}, backend="b")
        stream = (
            _start("Read", 0)
            + _start("Write", 1)
            + _delta('{"path":"a.py"}', 0)
            + _delta('{"path":"b.py"}', 1)
            + _stop(0)
            + _stop(1)
        )
        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(stream)
            auditor.finish()

        messages = [r.getMessage() for r in caplog.records]
        assert any("Read" in m and "a.py" in m for m in messages), messages
        assert any("Write" in m and "b.py" in m for m in messages), messages


class TestWarnsOnAnomaly:
    """AC-4.1 / AC-4.2 — the warning fires exactly when the shape contradicts the schema."""

    def test_envelope_wrapped_input_warns_once(self, caplog):
        auditor = ToolUseAuditor({"StructuredOutput": STRUCTURED_OUTPUT_SCHEMA}, backend="minimax-large")
        payload = '{"result":{"findings":[],"conversation_notes":"x"}}'

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("StructuredOutput", payload))
            auditor.finish()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        message = warnings[0].getMessage()
        assert "minimax-large" in message
        assert "result" in message
        assert "findings" in message

    def test_clean_input_does_not_warn(self, caplog):
        auditor = ToolUseAuditor({"StructuredOutput": STRUCTURED_OUTPUT_SCHEMA}, backend="b")

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("StructuredOutput", '{"findings":[],"conversation_notes":"x"}'))
            auditor.finish()

        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_undeclared_tool_is_logged_but_not_warned(self, caplog):
        """A tool the client never declared cannot be judged."""
        auditor = ToolUseAuditor({}, backend="b")
        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("ServerSideThing", '{"anything":1}'))
            auditor.finish()

        assert any("ServerSideThing" in r.getMessage() for r in caplog.records)
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class TestRedaction:
    """The WARNING is the one line that can reach an embedder's own handler.

    Tool inputs routinely carry file contents and user data, so the warning
    must name keys and never values.  A maintainer "helpfully" appending the
    input to it would break this quietly, which is why it is pinned.
    """

    def test_warning_names_keys_but_never_values(self, caplog):
        secret = "s3cret-api-token-do-not-log"
        auditor = ToolUseAuditor({"StructuredOutput": STRUCTURED_OUTPUT_SCHEMA}, backend="b")
        payload = json.dumps({"result": {"findings": [secret], "conversation_notes": secret}})

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("StructuredOutput", payload))
            auditor.finish()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "result" in message, "key names are the point of the warning"
        assert secret not in message, "the warning must never carry tool input values"

    def test_invalid_json_warning_reports_length_not_bytes(self, caplog):
        """A truncated payload's bytes are still payload — report only its size."""
        secret = "s3cret-in-a-truncated-payload"
        auditor = ToolUseAuditor({}, backend="b")

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("Read", '{"path":"' + secret))
            auditor.finish()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert secret not in warnings[0].getMessage()


class TestPreParsedToolCall:
    """A shim that hands back a parsed tool call sends no deltas at all.

    Anthropic always opens with ``input: {}`` and streams fragments, but the
    suspected cause in #33 is a non-Anthropic shim — and a shim that already
    parsed the arguments typically populates ``content_block_start`` instead.
    Ignoring that field would make the audit blind to the very shape it exists
    to catch.
    """

    def test_input_on_content_block_start_is_audited(self, caplog):
        auditor = ToolUseAuditor({"StructuredOutput": STRUCTURED_OUTPUT_SCHEMA}, backend="b")
        stream = (
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t",
                        "name": "StructuredOutput",
                        "input": {"result": {"findings": [], "conversation_notes": "x"}},
                    },
                },
            )
            + _stop()
        )

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(stream)
            auditor.finish()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "result" in warnings[0].getMessage()

    def test_deltas_win_over_a_seeded_input(self, caplog):
        """Anthropic sends an empty seed then deltas; the deltas are the truth."""
        auditor = ToolUseAuditor({}, backend="b")
        stream = _start("Read") + _delta('{"path":"from-delta.py"}') + _stop()

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(stream)
            auditor.finish()

        assert any("from-delta.py" in r.getMessage() for r in caplog.records)


class TestAuditorIsTotal:
    """FR-5 — auditing is diagnostics and must never break the stream."""

    def test_invalid_argument_json_warns_but_does_not_raise(self, caplog):
        """AC-5.2 — malformed serialization is itself a signal worth having.

        The literal boundary-token leak from MiniMax-M3#31.
        """
        auditor = ToolUseAuditor({}, backend="b")
        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("Read", '{"path": ]<]minimax[>['))
            auditor.finish()

        assert any("not valid JSON" in r.getMessage() for r in caplog.records)

    def test_oversized_arguments_are_abandoned_without_raising(self):
        """AC-5.3 — a runaway upstream must not exhaust memory through the auditor."""
        auditor = ToolUseAuditor({}, backend="b")
        auditor.feed(_start("Read"))
        auditor.feed(_delta("x" * (_MAX_BUFFERED_ARG_CHARS + 1)))
        auditor.finish()

        assert auditor._open == {}

    def test_detector_exception_is_contained(self, monkeypatch, caplog):
        """AC-5.4 — a bug in the detector must not surface as a broken response."""
        import kitty.bridge.tool_audit as module

        def boom(*args, **kwargs):
            raise RuntimeError("detector bug")

        monkeypatch.setattr(module, "describe_tool_input_anomaly", boom)
        auditor = ToolUseAuditor({"Read": STRUCTURED_OUTPUT_SCHEMA}, backend="b")

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("Read", '{"path":"a.py"}'))
            auditor.finish()

        assert any("auditing failed" in r.getMessage() for r in caplog.records)

    def test_non_sse_noise_is_ignored(self):
        """Bytes that are not SSE at all must not raise."""
        auditor = ToolUseAuditor({}, backend="b")
        auditor.feed(b"\x00\xff not sse at all\n\ndata: {broken\n\nevent: ping\n\n")
        auditor.finish()

    def test_two_anomalous_blocks_produce_two_warnings(self, caplog):
        """AC-4.3 — the composition, not just the two halves separately."""
        auditor = ToolUseAuditor({"StructuredOutput": STRUCTURED_OUTPUT_SCHEMA}, backend="b")
        payload = '{"result":{"findings":[],"conversation_notes":"x"}}'
        stream = _tool_use_stream("StructuredOutput", payload, 0) + _tool_use_stream(
            "StructuredOutput", payload, 1
        )

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(stream)
            auditor.finish()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER]
        assert len(warnings) == 2, [r.getMessage() for r in warnings]

    def test_a_large_chunk_of_complete_lines_does_not_disable(self):
        """The line bound is for an *unterminated* line, not for volume.

        Checking the whole buffer before draining would disable the auditor on
        one big chunk of perfectly good SSE — and then silently miss every tool
        call in the rest of the response.
        """
        auditor = ToolUseAuditor({}, backend="b")
        filler = b"".join(
            _sse("content_block_delta", {"type": "content_block_delta", "index": 9, "delta": {}})
            for _ in range(20000)
        )
        assert len(filler) > _MAX_LINE_BYTES // 8, "filler should be substantial"

        auditor.feed(filler * 8)
        auditor.feed(_tool_use_stream("Read", '{"path":"a.py"}'))
        auditor.finish()

        assert auditor._disabled is False

    def test_tool_use_after_a_long_text_stream_is_still_detected(self, caplog):
        """The cheap pre-filter must not hide a tool call that comes later.

        Events are skipped without parsing only while no block is open, so a
        turn that talks for a long time and then calls a tool must still be
        caught.
        """
        auditor = ToolUseAuditor({"StructuredOutput": STRUCTURED_OUTPUT_SCHEMA}, backend="b")
        chatter = b"".join(
            _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello "}},
            )
            for _ in range(500)
        )

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.feed(chatter)
            auditor.feed(_tool_use_stream("StructuredOutput", '{"result":{"findings":[],"conversation_notes":"x"}}', 1))
            auditor.finish()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == AUDIT_LOGGER]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]

    def test_line_buffer_bound_disables_the_auditor(self, caplog):
        """FR-5 — an upstream that never sends a newline must not grow it forever."""
        auditor = ToolUseAuditor({}, backend="b")

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(b"data: " + b"x" * (_MAX_LINE_BYTES + 1))

        assert auditor._disabled is True
        assert auditor._line_buffer == bytearray()
        assert any("disabled for this response" in r.getMessage() for r in caplog.records)

    def test_open_block_bound_disables_the_auditor(self, caplog):
        """Per-block argument bounds do not bound the number of blocks."""
        auditor = ToolUseAuditor({}, backend="b")

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            # Open far more blocks than the cap, never closing any.
            auditor.feed(b"".join(_start("Read", i) for i in range(_MAX_OPEN_BLOCKS + 5)))

        assert auditor._disabled is True
        assert auditor._open == {}

    def test_containment_logs_once_not_once_per_chunk(self, monkeypatch, caplog):
        """A poisoned auditor must not spend a DEBUG line on every later chunk."""
        import kitty.bridge.tool_audit as module

        def boom(*args, **kwargs):
            raise RuntimeError("detector bug")

        monkeypatch.setattr(module, "describe_tool_input_anomaly", boom)
        auditor = ToolUseAuditor({"Read": STRUCTURED_OUTPUT_SCHEMA}, backend="b")

        with caplog.at_level(logging.DEBUG, logger=AUDIT_LOGGER):
            auditor.feed(_tool_use_stream("Read", '{"path":"a.py"}'))
            for _ in range(20):
                auditor.feed(_tool_use_stream("Read", '{"path":"b.py"}'))
            auditor.finish()

        failures = [r for r in caplog.records if "auditing failed" in r.getMessage()]
        assert len(failures) == 1, [r.getMessage() for r in failures]

    def test_truncated_stream_reports_the_partial_block(self, caplog):
        """An upstream dying mid-tool-call is exactly what we want to see."""
        auditor = ToolUseAuditor({}, backend="b")
        auditor.feed(_start("Read") + _delta('{"path":'))

        with caplog.at_level(logging.WARNING, logger=AUDIT_LOGGER):
            auditor.finish()

        assert any("not valid JSON" in r.getMessage() for r in caplog.records)
