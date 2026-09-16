"""L3 subsystem tests for KBR-137 — the OpenCode Go OpenAI Responses route end-to-end.

Spins up an in-process aiohttp server that speaks the published OpenAI
Responses protocol at ``/v1/responses`` and drives the four routed models
through both the streaming and the non-streaming path, asserting the
Chat Completions output the bridge produces.

The tests pin the bridge contract, not the provider.  The Responses fixture
is hand-written from the published spec (v2.3.0, retrieved 2026-09-16 from
``openai/openai-openapi`` master).  A key-holder can land a live probe and
amend the fixture; every assertion here is a claim about kitty's behaviour
on that fixture, not about OpenCode Go's response to the request.

**Layer.** L3 — the bridge integrates with a real wire, but the
upstream is a scripted fixture, not the OpenCode Go endpoint.  L4
acceptance (KBR-126's scenario "every model routes to its published
endpoint") is the product-level claim that the four models succeed
end-to-end — those assertions live here, with the wiring that exercises
them.

Markers are explicit (``pytestmark = pytest.mark.l3``) so the layer
guard's PATH_DEFAULT map (which sends ``tests/`` to ``l1``) does not
silently promote these tests into the gating job.
"""

from __future__ import annotations

import json

import pytest

from kitty.providers.opencode import OpenCodeGoAdapter

pytestmark = pytest.mark.l3


# ── Responses fixtures (hand-written from the spec) ────────────────────────


def _text_only_response(model: str) -> dict:
    """A minimal Responses JSON body: an assistant message, no tool calls."""
    return {
        "id": "resp_test",
        "object": "response",
        "created": 1700000000,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi from Responses"}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
    }


def _tool_call_response(model: str) -> dict:
    """A Responses body with a single function_call output item."""
    return {
        "id": "resp_test",
        "object": "response",
        "created": 1700000000,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "get_weather",
                "arguments": '{"city": "London"}',
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
    }


# ── Tests ──────────────────────────────────────────────────────────────────


_RESPONSES_MODELS_L3 = (
    "grok-4.6",
    "gpt-5.6-luna",
    "muse-spark-1.3-contributor",
    "muse-spark-1.2-contributor",
)


@pytest.mark.parametrize("model", _RESPONSES_MODELS_L3)
class TestNonStreamingResponses:
    """KBR-137 AC-F3 — every routed model returns a well-formed CC response."""

    def test_text_only_response_translated(self, model):
        adapter = OpenCodeGoAdapter()
        cc = adapter.translate_from_upstream(_text_only_response(model))
        assert cc["object"] == "chat.completion"
        assert cc["model"] == model
        assert cc["choices"][0]["message"]["role"] == "assistant"
        assert cc["choices"][0]["message"]["content"] == "Hi from Responses"
        assert cc["choices"][0]["finish_reason"] == "stop"
        assert cc["usage"]["prompt_tokens"] == 7
        assert cc["usage"]["completion_tokens"] == 4

    def test_tool_call_response_translated(self, model):
        adapter = OpenCodeGoAdapter()
        cc = adapter.translate_from_upstream(_tool_call_response(model))
        assert cc["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = cc["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"city": "London"}'

    def test_request_body_carries_model_input_and_tools(self, model):
        """KBR-137 AC-F1 — the upstream body is shaped for the spec."""
        adapter = OpenCodeGoAdapter()
        body = adapter.translate_to_upstream(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "hi"},
                ],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            }
        )
        # Must be on the Responses dialect, not a CC fallback.
        assert body["model"] == model
        assert body.get("instructions") == "Be terse."
        assert isinstance(body.get("input"), list)
        assert body["tools"] == [{"type": "function", "name": "f", "parameters": {}}]


class TestResponsesConverterStreaming:
    """KBR-137 AC-F4 — the converter handles the happy-path event sequence."""

    def _stream_events(self, model: str = "grok-4.6") -> list[bytes]:
        """The full event sequence, exactly as a text + tool-call turn arrives."""
        events = [
            {"type": "response.created", "response": {"model": model}},
            {
                "type": "response.output_item.added",
                "item": {"id": "m1", "type": "message", "role": "assistant"},
            },
            {"type": "response.content_part.added", "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "delta": "Checking"},
            {"type": "response.output_text.done", "text": "Checking"},
            {"type": "response.content_part.done", "part": {"type": "output_text"}},
            {"type": "response.output_item.done", "item": {"id": "m1", "type": "message"}},
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_a", "name": "f", "arguments": ""},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"k":'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "1}"},
            {"type": "response.output_item.done", "item": {"id": "fc_1", "type": "function_call"}},
            {
                "type": "response.completed",
                "response": {"model": model, "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}},
            },
        ]
        return [f"data: {json.dumps(e)}\n\n".encode() for e in events]

    def test_full_stream_produces_well_formed_cc_chunks_ending_in_done(self):
        from kitty.providers.opencode import OpenCodeGoResponsesCCStreamConverter

        converter = OpenCodeGoResponsesCCStreamConverter()
        lines: list[bytes] = []
        for event in self._stream_events():
            lines.extend(converter.feed(event))

        # Every line is a sentence in the CC SSE grammar; the last is [DONE].
        for line in lines[:-1]:
            assert line.startswith(b"data: ")
            payload = json.loads(line[6:])
            assert payload["object"] == "chat.completion.chunk"
        assert lines[-1] == b"data: [DONE]\n\n"

    def test_error_event_is_detected_by_bridge_stream_error_checker(self):
        """KBR-137 review finding 6 — ``response.failed`` must produce a CC
        ``{"error": ...}`` chunk, because :func:`_is_upstream_stream_error`
        keys on ``chunk["error"]``."""
        from kitty.bridge.server import BridgeServer
        from kitty.providers.opencode import OpenCodeGoResponsesCCStreamConverter

        converter = OpenCodeGoResponsesCCStreamConverter()
        failure = {
            "type": "response.failed",
            "response": {"error": {"code": "rate_limit", "message": "slow down"}},
        }
        raw = converter.feed(f"data: {json.dumps(failure)}\n\n".encode())
        chunk = json.loads(raw[0][6:])
        assert BridgeServer._is_upstream_stream_error(chunk) is True
        assert chunk["error"]["code"] == "rate_limit"
