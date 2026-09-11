"""Tests for providers/opencode.py — OpenCodeGoAdapter with auto-routing."""

import json

import pytest

from kitty.providers.base import ProviderError
from kitty.providers.opencode import (
    _MESSAGES_MODELS,
    _RESPONSES_MODELS,
    OpenCodeGoAdapter,
    UnsupportedModelError,
)

# ── CC format samples ──────────────────────────────────────────────────────

SAMPLE_MESSAGES = [{"role": "user", "content": "hello"}]

SAMPLE_CC_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "glm-5.2",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenCode Go"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
}

SAMPLE_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "kimi-k2.6",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}

# ── Anthropic-format samples ───────────────────────────────────────────────

CC_MESSAGES_BASIC = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

CC_MESSAGES_TOOLS = [
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc",
        "content": "15°C, cloudy",
    },
]

CC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

ANTHROPIC_RESPONSE_TEXT = {
    "id": "msg_01ABC",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from OpenCode Go Messages"}],
    "model": "minimax-m2.7",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 25, "output_tokens": 10},
}

ANTHROPIC_RESPONSE_TOOL_USE = {
    "id": "msg_tool123",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_01ABC",
            "name": "get_weather",
            "input": {"city": "London"},
        },
    ],
    "model": "minimax-m2.7",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 50, "output_tokens": 30},
}

ANTHROPIC_RESPONSE_MAX_TOKENS = {
    "id": "msg_max",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Cut off..."}],
    "model": "minimax-m2.7",
    "stop_reason": "max_tokens",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 100},
}


# ═══════════════════════════════════════════════════════════════════════════
# Properties and identity
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoAdapterProperties:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "opencode_go"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://opencode.ai/zen/go"

    def test_default_upstream_path_is_chat_completions(self):
        assert self.adapter.upstream_path == "/v1/chat/completions"

    def test_validation_model(self):
        """Pin the literal, alongside the two structural checks it is paired with.

        ``tests/test_opencode_endpoint_table.py`` asserts this name is in the
        provider's published catalogue and
        ``tests/test_validation_model_routing.py`` asserts the key-check ping can
        reach it; both are stronger than this line. It is here so the value
        cannot change silently — the previous one, ``glm-5``, left the catalogue
        without anything noticing (KBR-126).
        """
        assert self.adapter.validation_model == "mimo-v2.5"


class TestOpenCodeGoNormalizeModelName:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_returns_unchanged(self):
        assert self.adapter.normalize_model_name("glm-5.2") == "glm-5.2"
        assert self.adapter.normalize_model_name("kimi-k2.6") == "kimi-k2.6"

    def test_strips_provider_prefix(self):
        assert self.adapter.normalize_model_name("opencode/glm-5.2") == "glm-5.2"


# ═══════════════════════════════════════════════════════════════════════════
# Auto-routing: get_upstream_path
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoAutoRouting:
    """Routing per model, against the provider's published endpoint table (KBR-126).

    The table itself is the oracle and lives in
    ``tests/data/opencode_go_endpoints.json``; the whole-catalogue sweep is
    ``tests/test_opencode_endpoint_table.py``.  What is proven here is the
    *predicate* — one model per route, plus the two inputs that name no model —
    so the 28-model agreement claim is asserted in exactly one place.
    """

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("minimax-m3", "/v1/messages"),
            ("qwen3.8-max", "/v1/messages"),
            ("grok-4.6", "/v1/responses"),
            ("muse-spark-1.3-contributor", "/v1/responses"),
            ("glm-5.2", "/v1/chat/completions"),
        ],
    )
    def test_one_literal_model_per_route(self, model, expected):
        """Literals, not the routing sets — otherwise the assertion is circular.

        Parametrising over ``_MESSAGES_MODELS`` and asserting the Messages path
        is true for *any* content of that set, including an empty or wrong one:
        it restates the implementation rather than checking it.  Spelled-out
        names catch a reversed or mis-ordered branch here, at L1, and not only
        in the snapshot sweep.
        """
        assert self.adapter.get_upstream_path(model) == expected

    def test_responses_models_report_the_endpoint_the_provider_serves_them_on(self):
        """The truthful path, even though no request is ever sent there.

        ``translate_to_upstream`` refuses these models, so reporting
        ``/v1/chat/completions`` here would cost nothing at runtime — and would
        put the lie back in the routing table that KBR-126 exists to remove, and
        force an exemption list into the snapshot guard.
        """
        paths = {self.adapter.get_upstream_path(m) for m in _RESPONSES_MODELS}

        assert paths == {"/v1/responses"}

    @pytest.mark.parametrize("model", ["glm-5.2", "kimi-k2.7-code", "mimo-v2.5", "hy3"])
    def test_chat_completions_models_take_the_default_route(self, model):
        assert self.adapter.get_upstream_path(model) == "/v1/chat/completions"

    @pytest.mark.parametrize("model", ["", "some-future-model", "glm-5"])
    def test_an_unknown_model_falls_through_to_chat_completions(self, model):
        """``glm-5`` is here on purpose: a retired name *is* the unknown case."""
        assert self.adapter.get_upstream_path(model) == "/v1/chat/completions"

    def test_the_two_routing_sets_are_disjoint(self):
        """A model in both sets would make the routing order decide the dialect."""
        assert not (_MESSAGES_MODELS & _RESPONSES_MODELS)

    def test_both_routing_sets_are_populated(self):
        """Neither sweep above may pass by having nothing to iterate."""
        assert _MESSAGES_MODELS and _RESPONSES_MODELS


# ═══════════════════════════════════════════════════════════════════════════
# Auto-routing: build_upstream_headers_for_model
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoHeadersRouting:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_default_headers_are_bearer(self):
        headers = self.adapter.build_upstream_headers("sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_chat_completions_model_gets_bearer(self):
        headers = self.adapter.build_upstream_headers_for_model("sk-test", "glm-5.2")
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers

    def test_messages_model_gets_x_api_key(self):
        headers = self.adapter.build_upstream_headers_for_model("sk-test", "minimax-m2.7")
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"
        assert "Authorization" not in headers

    def test_minimax_m25_gets_messages_headers(self):
        headers = self.adapter.build_upstream_headers_for_model("sk-test", "minimax-m2.5")
        assert headers["x-api-key"] == "sk-test"
        assert "Authorization" not in headers

    @pytest.mark.parametrize("model", sorted(_MESSAGES_MODELS))
    def test_every_messages_model_gets_anthropic_auth(self, model):
        """R5 — all eight, not just the two that were already routed."""
        headers = self.adapter.build_upstream_headers_for_model("sk-test", model)
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers

    @pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
    def test_responses_models_get_bearer(self, model):
        """R5 — the Responses API is Bearer-authenticated, like Chat Completions.

        Unreachable in practice, since ``translate_to_upstream`` refuses these
        models before any request is built.  Asserted anyway so the answer is
        correct on the day KBR-137 makes it reachable, rather than silently
        wrong.
        """
        headers = self.adapter.build_upstream_headers_for_model("sk-test", model)
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers


# ═══════════════════════════════════════════════════════════════════════════
# Chat Completions passthrough
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Auto-routing: upstream_wire_is_messages_api (KBR-7)
# ═══════════════════════════════════════════════════════════════════════════
#
# The model names here are the ones OpenCode Go serves today.  The older
# classes above still name `glm-5`, `kimi-k2.5` and `mimo-v2-*`, which the
# provider has since retired; refreshing them is the deferred catalogue ticket
# recorded in this task's REQUIREMENTS.md §3.1(b), not this change.


# Models the adapter routes to Chat Completions, plus the two inputs that have
# no model at all.  `glm-5` is kept deliberately: a retired name is now exactly
# the "unknown model" case, and the default must stay Chat Completions.
_CHAT_COMPLETIONS_MODELS = ["glm-5.2", "glm-5.1", "kimi-k2.7-code", "mimo-v2.5-pro", "glm-5", "", "some-future-model"]


class TestOpenCodeGoWireShapeDeclaration:
    """The declared wire shape must agree with what ``translate_to_upstream`` emits.

    KBR-7: the adapter inherited ``upstream_wire_is_messages_api == True`` from
    :class:`AnthropicAdapter` while emitting Chat Completions for every model
    outside ``_MESSAGES_MODELS``.  The bridge's thinking round-trip repair
    branches on that declaration, so a wrong answer writes an Anthropic
    ``thinking`` block into a Chat Completions body.
    """

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    @pytest.mark.parametrize("model", sorted(_MESSAGES_MODELS))
    def test_declares_messages_for_each_messages_model(self, model):
        assert self.adapter.upstream_wire_is_messages_api_for_model(model) is True

    @pytest.mark.parametrize("model", _CHAT_COMPLETIONS_MODELS)
    def test_declares_chat_completions_for_non_messages_models(self, model):
        assert self.adapter.upstream_wire_is_messages_api_for_model(model) is False

    @pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
    def test_declares_not_messages_for_responses_models(self, model):
        """R6 — ``False`` here means "not a Messages body", and that is honest.

        ``TEST_SUITE.md`` §6.2.3 says a routing adapter that gains a third wire
        must *replace* this boolean rather than overload ``False``.  That
        obligation belongs to KBR-137, not here: the declaration describes the
        body ``translate_to_upstream`` emits, and for these models it emits
        none — it raises.  There is no third wire to declare yet, only a refusal.
        """
        assert self.adapter.upstream_wire_is_messages_api_for_model(model) is False

    def test_bare_property_reports_the_default_chat_completions_route(self):
        """The bare property answers for the default route, like its neighbours.

        ``upstream_path`` and ``build_upstream_headers`` both report the Chat
        Completions default with a per-model form alongside; this declaration
        now does the same instead of inheriting Anthropic's ``True``.
        """
        assert self.adapter.upstream_wire_is_messages_api is False

    def test_declaration_does_not_call_translate_to_upstream(self):
        """The declaration is a predicate, not an observation.

        Implementing it by classifying ``translate_to_upstream``'s output would
        make the registry-wide guard a tautology: the declaration would carry no
        information, and an oracle must not ask the code under test what shape
        it emitted.
        """

        def _explode(cc_request):
            raise AssertionError("the declaration must not serialize a request")

        self.adapter.translate_to_upstream = _explode  # type: ignore[method-assign]  # deliberate tripwire
        assert self.adapter.upstream_wire_is_messages_api_for_model("minimax-m2.5") is True
        assert self.adapter.upstream_wire_is_messages_api_for_model("glm-5.2") is False

    @pytest.mark.parametrize("model", ["minimax-m2.5", "glm-5.2"])
    def test_declaration_matches_the_body_translate_to_upstream_returns(self, model):
        """Assert against the emitted body, not against a restated constant.

        A test that repeated the routing predicate would keep passing through
        the very drift KBR-7 is about.
        """
        body = self.adapter.translate_to_upstream(
            {
                "model": model,
                "max_tokens": 100,
                "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
            }
        )
        # An Anthropic Messages body hoists the system prompt out of `messages`
        # and carries `input_schema` tools; a Chat Completions body does neither.
        emitted_messages_api = "system" in body and "input_schema" in body["tools"][0]
        assert self.adapter.upstream_wire_is_messages_api_for_model(model) is emitted_messages_api


class TestOpenCodeGoResponsesRefusal:
    """R4 — a model served on ``/v1/responses`` is refused, not mis-serialized.

    KBR-126: these four were previously given a Chat Completions body and sent
    to ``/v1/chat/completions``.  The provider answers an unsupported model with
    ``401``, and ``BridgeServer`` classifies ``401`` as an auth failure — so the
    user was told their API key was bad.  Refusing here makes the real reason
    reach them instead.  KBR-137 replaces the refusal with a working route.
    """

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    @pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
    def test_translate_to_upstream_refuses_every_responses_model(self, model):
        with pytest.raises(UnsupportedModelError):
            self.adapter.translate_to_upstream({"model": model, "messages": SAMPLE_MESSAGES})

    @pytest.mark.parametrize("model", sorted(_RESPONSES_MODELS))
    def test_the_message_names_the_model_the_endpoint_and_the_ticket(self, model):
        """The message is the whole user-facing artifact — it must be actionable.

        On three of the four inbound protocols the SSE response is already
        committed with status 200 before the body is built, so this text is all
        the user gets.  Asserted rather than trusted for that reason.
        """
        with pytest.raises(UnsupportedModelError) as excinfo:
            self.adapter.translate_to_upstream({"model": model, "messages": SAMPLE_MESSAGES})

        message = str(excinfo.value)
        assert model in message
        assert "/v1/responses" in message
        assert "KBR-137" in message

    def test_the_error_is_a_provider_error(self):
        """Existing ``except ProviderError`` handlers must keep catching it."""
        with pytest.raises(ProviderError):
            self.adapter.translate_to_upstream({"model": "grok-4.6", "messages": SAMPLE_MESSAGES})

    def test_the_error_carries_no_http_status(self):
        """A ``400`` here would reach the bridge's context-too-large branch.

        ``_provider_error_failure_kind`` inspects ``http_status == 400`` for an
        oversized-context message and can route such a failure into the
        compaction-retry path.  A configuration error must not land there.
        """
        with pytest.raises(UnsupportedModelError) as excinfo:
            self.adapter.translate_to_upstream({"model": "grok-4.6", "messages": SAMPLE_MESSAGES})

        assert excinfo.value.http_status == 0

    @pytest.mark.parametrize("model", ["glm-5.2", "minimax-m2.7", "", "some-future-model"])
    def test_no_other_model_is_refused(self, model):
        """The positive control: the refusal is scoped, not a blanket failure.

        Asserts the returned body carries the conversation, not merely that
        something truthy came back — an adapter returning ``{"x": 1}`` would
        satisfy a truthiness check while having lost the request.
        """
        body = self.adapter.translate_to_upstream({"model": model, "messages": SAMPLE_MESSAGES})

        assert body["model"] == model
        assert body["messages"] == SAMPLE_MESSAGES


class TestOpenCodeGoChatCompletionsPassthrough:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_translate_to_upstream_passthrough_for_cc_model(self):
        cc = {"model": "glm-5.2", "messages": SAMPLE_MESSAGES, "stream": True}
        result = self.adapter.translate_to_upstream(cc)
        assert result == cc

    def test_translate_from_upstream_passthrough_for_cc_response(self):
        resp = SAMPLE_CC_RESPONSE
        result = self.adapter.translate_from_upstream(resp)
        assert result is resp

    def test_translate_upstream_stream_event_passthrough_for_cc(self):
        chunk = b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}\n\n'
        result = self.adapter.translate_upstream_stream_event(chunk)
        assert result == [chunk]


class TestOpenCodeGoBuildRequest:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="glm-5.2",
            messages=SAMPLE_MESSAGES,
            stream=True,
        )
        assert result["model"] == "glm-5.2"
        assert result["messages"] == SAMPLE_MESSAGES
        assert result["stream"] is True

    def test_build_request_with_tools(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        result = self.adapter.build_request(
            model="kimi-k2.6",
            messages=SAMPLE_MESSAGES,
            stream=False,
            tools=tools,
        )
        assert result["tools"] == tools


class TestOpenCodeGoParseResponse:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_parse_cc_response(self):
        result = self.adapter.parse_response(SAMPLE_CC_RESPONSE)
        assert result["content"] == "Hello from OpenCode Go"
        assert result["finish_reason"] == "stop"

    def test_parse_tool_call_response(self):
        result = self.adapter.parse_response(SAMPLE_TOOL_CALL_RESPONSE)
        assert result["content"] is None
        assert result["finish_reason"] == "tool_calls"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"


class TestOpenCodeGoMapError:
    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_error_format(self):
        exc = self.adapter.map_error(401, {"error": "unauthorized"})
        assert "OpenCode Go error 401" in str(exc)


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic Messages translation (auto-routed for minimax models)
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenCodeGoMessagesTranslateToUpstream:
    """CC request → Anthropic Messages API (auto-routed by model name)."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_minimax_model_triggers_anthropic_translation(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_BASIC, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        # Anthropic format has system at top level, not in messages
        assert "system" in result
        assert result["system"] == "You are helpful."

    def test_extracts_system_message(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_BASIC, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assert result["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in result["messages"])

    def test_no_system_message(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "system" not in result

    def test_max_tokens_default(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["max_tokens"] == 4096

    def test_tools_translated(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": CC_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert len(result["tools"]) == 1
        assert "input_schema" in result["tools"][0]
        assert "parameters" not in result["tools"][0]

    def test_assistant_tool_calls_become_content_blocks(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_TOOLS, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][1]
        tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"

    def test_tool_result_becomes_user_tool_result_block(self):
        cc = {"model": "minimax-m2.7", "messages": CC_MESSAGES_TOOLS, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"

    def test_temperature_passthrough(self):
        cc = {
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["temperature"] == 0.5

    def test_cc_model_is_not_translated(self):
        """Non-messages models should pass through without Anthropic translation."""
        cc = {"model": "glm-5.2", "messages": CC_MESSAGES_BASIC, "stream": True}
        result = self.adapter.translate_to_upstream(cc)
        assert result == cc  # same content, passthrough


class TestOpenCodeGoMessagesTranslateFromUpstream:
    """Anthropic Messages API response → CC response."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_text_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["choices"][0]["message"]["content"] == "Hello from OpenCode Go Messages"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TEXT)
        assert result["usage"]["prompt_tokens"] == 25
        assert result["usage"]["completion_tokens"] == 10
        assert result["usage"]["total_tokens"] == 35

    def test_tool_use_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_TOOL_USE)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Let me check."
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_max_tokens_response(self):
        result = self.adapter.translate_from_upstream(ANTHROPIC_RESPONSE_MAX_TOKENS)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_cc_response_passthrough(self):
        """CC format responses should pass through without translation."""
        result = self.adapter.translate_from_upstream(SAMPLE_CC_RESPONSE)
        assert result is SAMPLE_CC_RESPONSE


class TestOpenCodeGoMessagesStreamEventAutoDetection:
    """Stream events auto-detected by format (not model name)."""

    def setup_method(self):
        self.adapter = OpenCodeGoAdapter()

    def test_anthropic_text_delta_translated(self):
        raw = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["delta"]["content"] == "Hello"

    def test_anthropic_message_start_translated(self):
        raw = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01",'
            b'"role":"assistant","content":[],"model":"minimax-m2.7","stop_reason":null}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["delta"]["role"] == "assistant"

    def test_anthropic_message_delta_translated(self):
        raw = (
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
            b'"stop_sequence":null},"usage":{"output_tokens":15}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        combined = b"".join(chunks)
        parsed = json.loads(combined.split(b"data:", 1)[1].strip())
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_anthropic_ping_ignored(self):
        raw = b'event: ping\ndata: {"type":"ping"}\n\n'
        assert self.adapter.translate_upstream_stream_event(raw) == []

    def test_anthropic_content_block_start_ignored(self):
        raw = (
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
        )
        assert self.adapter.translate_upstream_stream_event(raw) == []

    def test_anthropic_content_block_stop_ignored(self):
        raw = b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        assert self.adapter.translate_upstream_stream_event(raw) == []

    def test_anthropic_message_stop_yields_done(self):
        raw = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert b"".join(chunks).strip() == b"data: [DONE]"

    def test_cc_chunk_passthrough(self):
        """Chat Completions SSE events should pass through unchanged."""
        raw = b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}\n\n'
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == [raw]

    def test_done_event_passthrough(self):
        raw = b"data: [DONE]\n\n"
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert chunks == [raw]

    def test_empty_bytes(self):
        assert self.adapter.translate_upstream_stream_event(b"") == []

    def test_tool_use_delta_no_crash(self):
        raw = (
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
            b'"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"London\\"}"}}\n\n'
        )
        chunks = self.adapter.translate_upstream_stream_event(raw)
        assert isinstance(chunks, list)
