"""The stateful Anthropic-SSE → Chat-Completions stream converter (KBR-232).

One upstream stream, one converter instance.  These tests pin the event→chunk
mapping the three non-Messages stream handlers rely on: text, tool calls
(``tool_use`` start + ``input_json_delta``), thinking as ``reasoning_content``,
the stop-reason map, usage accumulation, and the ``[DONE]`` sentinel.  They
also pin the statefulness the old per-event translator could not have: tool
indices allocated per block and one chunk id per stream.
"""

from __future__ import annotations

import json

import pytest

from kitty.providers.anthropic import AnthropicCCStreamConverter


def _raw(event: dict) -> bytes:
    """Render one Anthropic event as the SSE line the handler would read.

    Args:
        event: The event payload, carrying its ``type``.

    Returns:
        The ``event:``/``data:`` SSE bytes for that one event.
    """
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _chunks(converter: AnthropicCCStreamConverter, event: dict) -> list[dict]:
    """Feed one event and return the CC chunk payloads the converter emitted.

    Args:
        converter: The converter instance under test.
        event: The Anthropic event payload.

    Returns:
        The decoded ``data:`` payloads, in order.  The ``[DONE]`` sentinel is
        reported as the string ``"[DONE]"``.
    """
    out: list[dict | str] = []
    for line in converter.feed(_raw(event)):
        text = line.decode().removeprefix("data: ").strip()
        out.append("[DONE]" if text == "[DONE]" else json.loads(text))
    return [chunk for chunk in out if isinstance(chunk, dict)]


def _deltas(chunks: list[dict]) -> list[dict]:
    """Return every non-null choice delta from a list of CC chunks.

    Args:
        chunks: Decoded CC chunk payloads.

    Returns:
        The ``choices[0].delta`` objects that are not ``{}``.
    """
    return [c["choices"][0]["delta"] for c in chunks if c["choices"][0]["delta"]]


def _message_start(model: str = "claude-opus-4-6", input_tokens: int = 5) -> dict:
    """Return a ``message_start`` event with usage.

    Args:
        model: The upstream model name.
        input_tokens: The input-token count the event reports.

    Returns:
        The event payload.
    """
    return {
        "type": "message_start",
        "message": {
            "id": "msg_upstream",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }


def _message_delta(stop_reason: str = "end_turn", output_tokens: int = 9) -> dict:
    """Return a ``message_delta`` event with usage.

    Args:
        stop_reason: The upstream stop reason.
        output_tokens: The output-token count the event reports.

    Returns:
        The event payload.
    """
    return {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": output_tokens}}


class TestEventMapping:
    """One event in, the documented chunk(s) out."""

    def setup_method(self) -> None:
        """Give every test a fresh converter.

        Returns:
            Nothing.
        """
        self.converter = AnthropicCCStreamConverter()

    def test_message_start_yields_role_and_model(self) -> None:
        """A ``message_start`` opens the stream as an assistant role chunk."""
        (chunk,) = _chunks(self.converter, _message_start(model="claude-opus-4-6"))
        assert chunk["choices"][0]["delta"] == {"role": "assistant"}
        assert chunk["model"] == "claude-opus-4-6"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_text_delta_becomes_content(self) -> None:
        """A ``text_delta`` becomes a ``content`` delta."""
        (delta,) = _deltas(_chunks(self.converter, {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        }))
        assert delta == {"content": "hello"}

    def test_tool_use_start_opens_a_tool_call(self) -> None:
        """A ``tool_use`` block start opens a ``tool_calls`` entry with id and name."""
        event = {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        }
        (delta,) = _deltas(_chunks(self.converter, event))
        assert delta == {
            "tool_calls": [
                {"index": 0, "id": "toolu_1", "type": "function", "function": {"name": "Read", "arguments": ""}}
            ]
        }

    def test_input_json_delta_continues_the_same_tool_call(self) -> None:
        """An ``input_json_delta`` extends the arguments of its block's tool call."""
        self.converter.feed(_raw({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        }))
        (delta,) = _deltas(_chunks(self.converter, {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "a"}'},
        }))
        (call,) = delta["tool_calls"]
        assert call["index"] == 0
        assert call["function"]["arguments"] == '{"path": "a"}'

    def test_tool_arguments_accumulate_across_deltas(self) -> None:
        """Arguments from several ``input_json_delta`` events concatenate to the full JSON.

        Concatenation is the client's job (the Chat Completions grammar has no
        done marker), so the converter's claim is that it forwards every
        fragment under one index — an overwrite here would drop fragments.
        """
        self.converter.feed(_raw({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
        }))
        deltas = []
        for fragment in ('{"path": ', '"a", "mode": ', '"r"}'):
            (delta,) = _deltas(_chunks(self.converter, {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": fragment},
            }))
            deltas.append(delta)
        arguments = "".join(d["tool_calls"][0]["function"]["arguments"] for d in deltas)
        assert json.loads(arguments) == {"path": "a", "mode": "r"}
        assert {d["tool_calls"][0]["index"] for d in deltas} == {0}

    def test_parallel_tool_blocks_get_distinct_indices(self) -> None:
        """Two ``tool_use`` blocks get CC indices 0 and 1, and deltas keep their own."""
        for index in (0, 1):
            self.converter.feed(_raw({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": f"toolu_{index}", "name": "Read", "input": {}},
            }))
        first = _deltas(_chunks(self.converter, {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "b"}'},
        }))
        assert first[0]["tool_calls"][0]["index"] == 1

    def test_populated_input_at_start_is_emitted_and_later_deltas_suppressed(self) -> None:
        """A ``tool_use`` start carrying a populated ``input`` emits it as the arguments.

        Anthropic itself streams arguments only through ``input_json_delta``
        and starts the block with ``input: {}``; some compatible providers
        send the whole object at once.  Emitting it at the start and then
        forwarding fragments again would double the arguments, so the
        fragment stream for that block is suppressed.
        """
        (start_delta,) = _deltas(_chunks(self.converter, {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a"}},
        }))
        assert start_delta["tool_calls"][0]["function"]["arguments"] == json.dumps({"path": "a"})

        after = _chunks(self.converter, {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "a"}'},
        })
        assert after == []

    def test_thinking_delta_becomes_reasoning_content(self) -> None:
        """A ``thinking_delta`` becomes a ``reasoning_content`` delta."""
        (delta,) = _deltas(_chunks(self.converter, {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "look first"},
        }))
        assert delta == {"reasoning_content": "look first"}

    def test_signature_delta_is_dropped(self) -> None:
        """A ``signature_delta`` yields nothing: signatures cannot cross the CC wire."""
        assert _chunks(self.converter, {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM"},
        }) == []

    def test_block_starts_for_text_and_thinking_yield_nothing(self) -> None:
        """Text and thinking block starts need no chunk — CC has no counterpart."""
        for block_type in ("text", "thinking"):
            assert _chunks(self.converter, {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": block_type},
            }) == []

    def test_content_block_stop_yields_nothing(self) -> None:
        """A block close yields nothing on its own."""
        assert _chunks(self.converter, {"type": "content_block_stop", "index": 0}) == []

    def test_ping_yields_nothing(self) -> None:
        """A ping yields nothing."""
        assert _chunks(self.converter, {"type": "ping"}) == []

    @pytest.mark.parametrize(
        ("stop_reason", "finish_reason"),
        [("end_turn", "stop"), ("tool_use", "tool_calls"), ("max_tokens", "length"), ("stop_sequence", "stop")],
    )
    def test_message_delta_maps_the_stop_reason(self, stop_reason: str, finish_reason: str) -> None:
        """A ``message_delta`` closes the stream with the mapped finish reason."""
        converter = AnthropicCCStreamConverter()
        (chunk,) = _chunks(converter, _message_delta(stop_reason))
        assert chunk["choices"][0]["finish_reason"] == finish_reason

    def test_message_stop_yields_the_done_sentinel(self) -> None:
        """``message_stop`` becomes ``data: [DONE]``."""
        converter = AnthropicCCStreamConverter()
        (line,) = converter.feed(_raw({"type": "message_stop"}))
        assert line.strip() == b"data: [DONE]"

    def test_error_event_passes_through_unchanged(self) -> None:
        """An ``error`` event rides through so the handlers' in-stream error path sees it."""
        event = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        raw = _raw(event)
        converter = AnthropicCCStreamConverter()
        assert converter.feed(raw) == [raw]


class TestStreamState:
    """State lives for the stream, not the event."""

    def test_every_chunk_carries_one_stable_id(self) -> None:
        """All chunks of a stream share one ``chatcmpl`` id."""
        converter = AnthropicCCStreamConverter()
        events = (
            _message_start(),
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        )
        seen = [c["id"] for e in events for c in _chunks(converter, e)]
        assert len(seen) == 2
        assert len(set(seen)) == 1
        assert seen[0].startswith("chatcmpl-")
        # Sanity: a fresh converter draws a fresh id, not a module constant.
        assert AnthropicCCStreamConverter().feed(_raw(_message_start())) != b""

    def test_usage_accumulates_across_the_stream(self) -> None:
        """The finish chunk carries Chat Completions usage built from both ends."""
        converter = AnthropicCCStreamConverter()
        converter.feed(_raw(_message_start(input_tokens=5)))
        converter.feed(_raw({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}))
        (chunk,) = _chunks(converter, _message_delta(output_tokens=9))
        assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_events_without_a_message_start_still_translate(self) -> None:
        """A stream that opens mid-flight (no ``message_start``) still yields chunks."""
        converter = AnthropicCCStreamConverter()
        (delta,) = _deltas(_chunks(converter, {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }))
        assert delta == {"content": "hi"}

    def test_malformed_data_line_passes_through(self) -> None:
        """A non-JSON data line is passed through, not swallowed."""
        converter = AnthropicCCStreamConverter()
        raw = b"data: not-json\n\n"
        assert converter.feed(raw) == [raw]

    def test_done_sentinel_passes_through(self) -> None:
        """A ``[DONE]`` line is passed through unchanged.

        On a converted Chat Completions stream the handler's ``[DONE]`` branch
        re-enters the per-event translator with the converter's sentinel line;
        the identity here is what makes that delegation harmless (KBR-232).
        """
        converter = AnthropicCCStreamConverter()
        raw = b"data: [DONE]\n\n"
        assert converter.feed(raw) == [raw]

    def test_empty_input_yields_nothing(self) -> None:
        """Empty bytes yield nothing."""
        assert AnthropicCCStreamConverter().feed(b"") == []

    def test_a_fresh_converter_draws_a_unique_id(self) -> None:
        """Two converters do not share a chunk id."""
        first = AnthropicCCStreamConverter()
        second = AnthropicCCStreamConverter()
        (a,) = _chunks(first, _message_start())
        (b,) = _chunks(second, _message_start())
        assert a["id"] != b["id"]
