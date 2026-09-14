"""Tests for bridge/messages/translator.py — Anthropic Messages API <-> Chat Completions translation."""

import json
import uuid

import pytest
from harness import cache_breakpoints as cb

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.providers.anthropic import AnthropicAdapter


def _v4() -> str:
    return str(uuid.uuid4())


# ── translate_request ───────────────────────────────────────────────────────


class TestTranslateRequest:
    def setup_method(self):
        self.t = MessagesTranslator()

    def test_extracts_model(self):
        req = {"model": "claude-3-opus", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1024}
        result = self.t.translate_request(req)
        assert result["model"] == "claude-3-opus"

    def test_system_prompt_mapped(self):
        req = {
            "model": "claude-3-opus",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}

    def test_system_prompt_as_content_blocks_flattened_to_string(self):
        """Anthropic allows system as array of content blocks — must flatten to string."""
        req = {
            "model": "claude-3-opus",
            "system": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assert result["messages"][0] == {"role": "system", "content": "Part one.\nPart two."}

    def test_text_content_blocks_mapped_to_string(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assert result["messages"][-1]["content"] == "hello"

    def test_tool_use_blocks_mapped_to_tool_calls(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check."
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "toolu_001"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}

    def test_tool_result_blocks_mapped_to_tool_messages(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_001", "content": "72F sunny"},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        tool_msg = result["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_001"
        assert tool_msg["content"] == "72F sunny"

    def test_multiple_tool_results_produce_multiple_tool_messages(self):
        req = {
            "model": "claude-3-opus",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                        {"type": "tool_use", "id": "toolu_002", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_001", "content": "72F sunny"},
                        {"type": "tool_result", "tool_use_id": "toolu_002", "content": "65F cloudy"},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.t.translate_request(req)
        # Should produce two tool role messages
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "toolu_001"
        assert tool_msgs[1]["tool_call_id"] == "toolu_002"

    def test_tools_mapped_from_anthropic_to_cc_format(self):
        req = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
        result = self.t.translate_request(req)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["required"] == ["city"]

    def test_max_tokens_passthrough(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 512}
        result = self.t.translate_request(req)
        assert result["max_tokens"] == 512

    def test_stream_flag_passthrough(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "stream": True}
        result = self.t.translate_request(req)
        assert result["stream"] is True

    def test_temperature_top_p_passthrough(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = self.t.translate_request(req)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_thinking_stripped_and_flag_set(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 4000},
        }
        assert not self.t.thinking_warned
        result = self.t.translate_request(req)
        assert "thinking" not in result
        assert self.t.thinking_warned

    def test_thinking_warning_only_once(self):
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 4000},
        }
        self.t.translate_request(req)
        assert self.t.thinking_warned
        # Second call should not reset or re-warn
        self.t.translate_request(req)
        assert self.t.thinking_warned

    def test_stop_sequences_mapped_to_stop(self):
        """KBR-178: the Messages `stop_sequences` becomes the CC `stop`."""
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stop_sequences": ["\n\nHuman:", "END"],
        }
        result = self.t.translate_request(req)
        assert result["stop"] == ["\n\nHuman:", "END"]
        assert "stop_sequences" not in result

    def test_no_stop_sequences_means_no_stop(self):
        """A client that sends no stop sequences gets no `stop` key invented."""
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        result = self.t.translate_request(req)
        assert "stop" not in result

    def test_empty_stop_sequences_is_omitted(self):
        """An empty list carries no instruction, and `stop: []` is schema-invalid.

        ``StopConfiguration`` in the OpenAI schema declares ``minItems: 1``, so
        forwarding ``[]`` would put an invalid body on fifteen passthrough
        adapters.  Omitting it is behaviourally identical.  See D6.
        """
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stop_sequences": [],
        }
        result = self.t.translate_request(req)
        assert "stop" not in result

    def test_top_k_carried_as_internal_key(self):
        """KBR-178: Chat Completions has no `top_k`, so it travels as `_top_k`."""
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "top_k": 40,
        }
        result = self.t.translate_request(req)
        assert result["_top_k"] == 40
        assert "top_k" not in result

    def test_top_k_zero_is_carried(self):
        """`top_k: 0` is a value the user sent, not an absent field."""
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "top_k": 0,
        }
        result = self.t.translate_request(req)
        assert result["_top_k"] == 0

    def test_no_top_k_means_no_internal_key(self):
        """No inbound `top_k` means no `_top_k` is invented."""
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        result = self.t.translate_request(req)
        assert "_top_k" not in result

    def test_unmapped_fields_stripped(self):
        """Fields the translator has no mapping for do not reach the CC body.

        ``stop_sequences`` and ``top_k`` were listed here until KBR-178 gave
        them mappings, and ``tool_choice`` and ``metadata`` until KBR-214 did;
        all four are now asserted positively elsewhere.  ``service_tier`` stands
        in as a published Messages field that is still genuinely dropped.
        """
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "service_tier": "auto",
        }
        result = self.t.translate_request(req)
        assert "service_tier" not in result

    def test_thinking_block_mapped_to_reasoning_content(self):
        """Thinking blocks in assistant messages must be mapped to reasoning_content
        for Chat Completions, not stripped. Required by Kimi Code and other
        providers with thinking mode."""
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me analyze this..."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                },
            ],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["reasoning_content"] == "Let me analyze this..."
        assert assistant_msg["content"] == "The answer is 42."

    def test_thinking_block_with_tool_use_maps_to_reasoning_content(self):
        """Thinking blocks with tool_use must include reasoning_content.
        Kimi Code errors with 400 if reasoning_content is missing from
        assistant tool call messages when thinking mode is enabled."""
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I need to check the weather..."},
                        {"type": "tool_use", "id": "toolu_001", "name": "get_weather", "input": {"city": "London"}},
                    ],
                },
            ],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["reasoning_content"] == "I need to check the weather..."
        assert assistant_msg["content"] is None
        assert len(assistant_msg["tool_calls"]) == 1

    def test_multiple_thinking_blocks_concatenated(self):
        """Multiple thinking blocks should be concatenated into a single reasoning_content."""
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Part one."},
                        {"type": "thinking", "thinking": "Part two."},
                        {"type": "text", "text": "Result."},
                    ],
                },
            ],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["reasoning_content"] == "Part one.\nPart two."

    def test_no_thinking_block_no_reasoning_content(self):
        """Assistant messages without thinking blocks should not have reasoning_content."""
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Just a plain response.",
                },
            ],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert "reasoning_content" not in assistant_msg


class TestToolChoiceAndMetadata:
    """KBR-214: ``tool_choice`` and ``metadata`` survive the Messages -> CC hop.

    ``tool_choice`` is a constraint, not a hint: dropping ``{"type": "any"}``
    lets the model answer in prose where the agent required a tool call, and
    dropping ``{"type": "none"}`` lets it call a tool the agent forbade.  The
    two wires spell the *values* differently, so this is a translation table,
    not a rename.  ``metadata`` rides the internal ``_metadata`` key and is
    restored only by Anthropic-family adapters (D1).
    """

    def setup_method(self):
        self.t = MessagesTranslator()

    def _req(self, **extra):
        """Build a minimal Messages request with one tool, plus the case's fields."""
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "tools": [{"name": "get_weather", "input_schema": {}}],
        }
        req.update(extra)
        return req

    @pytest.mark.parametrize(
        ("anthropic", "cc"),
        [
            ({"type": "auto"}, "auto"),
            ({"type": "any"}, "required"),
            ({"type": "none"}, "none"),
            ({"type": "tool", "name": "get_weather"}, {"type": "function", "function": {"name": "get_weather"}}),
        ],
        ids=["auto", "any", "none", "tool"],
    )
    def test_tool_choice_value_is_translated(self, anthropic, cc):
        """Each published Anthropic shape maps onto its Chat Completions spelling (R1)."""
        result = self.t.translate_request(self._req(tool_choice=anthropic))
        assert result["tool_choice"] == cc

    @pytest.mark.parametrize(
        "malformed",
        [None, "auto", {"type": "bogus"}, {"type": "tool"}, {"type": "tool", "name": 7}],
        ids=["null", "string", "unknown-type", "tool-without-name", "tool-with-non-string-name"],
    )
    def test_malformed_tool_choice_is_omitted(self, malformed):
        """Anthropic publishes four object shapes and no string form (D5).

        Kitty has no authority to invent a reading, so anything else is left out
        rather than repaired into a value the agent never sent.
        """
        result = self.t.translate_request(self._req(tool_choice=malformed))
        assert "tool_choice" not in result

    @pytest.mark.parametrize("tools", [None, []], ids=["absent", "empty"])
    def test_tool_choice_without_tools_is_omitted(self, tools):
        """Legal on Anthropic, a 400 on Chat Completions -- so not carried (D9)."""
        req = self._req(tool_choice={"type": "any", "disable_parallel_tool_use": True})
        if tools is None:
            del req["tools"]
        else:
            req["tools"] = tools
        result = self.t.translate_request(req)
        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result

    def test_forced_server_tool_is_not_carried(self):
        """Claude Code's WebSearch forces ``web_search``, a server tool (D10).

        This hop flattens the server tool into a schema-less function, so
        forcing it would force a call nothing on the route can execute.  The
        rest of the request -- the tool list included -- is unaffected.
        """
        req = self._req(
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            tool_choice={"type": "tool", "name": "web_search", "disable_parallel_tool_use": True},
        )
        result = self.t.translate_request(req)
        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result
        assert [t["function"]["name"] for t in result["tools"]] == ["web_search"]

    @pytest.mark.parametrize(
        "declaration",
        [
            {"name": "get_weather", "input_schema": {}},
            {"type": "custom", "name": "get_weather", "input_schema": {}},
            {"type": None, "name": "get_weather", "input_schema": {}},
        ],
        ids=["untyped", "custom", "type-null"],
    )
    def test_forced_ordinary_tool_is_carried_beside_a_server_tool(self, declaration):
        """Only the server tool is exempt; an ordinary tool next to one is still forced (D10)."""
        req = self._req(
            tools=[{"type": "web_search_20250305", "name": "web_search"}, declaration],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        result = self.t.translate_request(req)
        assert result["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_forcing_an_undeclared_tool_is_still_carried(self):
        """A choice naming no declared tool is the agent's mistake, not kitty's to hide (D8).

        The provider rejects it with an error that names the problem; omitting it
        would quietly answer a request the agent did not make.
        """
        result = self.t.translate_request(self._req(tool_choice={"type": "tool", "name": "not_declared"}))
        assert result["tool_choice"] == {"type": "function", "function": {"name": "not_declared"}}

    def test_a_non_string_tool_name_is_omitted_even_when_a_tool_has_it(self):
        """The name check is its own guard, not a side effect of the declaration lookup (D5)."""
        req = self._req(tools=[{"name": 7, "input_schema": {}}], tool_choice={"type": "tool", "name": 7})
        assert "tool_choice" not in self.t.translate_request(req)

    def test_a_truthy_non_boolean_disable_parallel_tool_use_is_not_carried(self):
        """R2 carries the flag only when it is exactly ``true``."""
        result = self.t.translate_request(self._req(tool_choice={"type": "any", "disable_parallel_tool_use": 1}))
        assert "parallel_tool_calls" not in result

    def test_empty_metadata_is_still_carried(self):
        """``{}`` is a value the agent sent; only ``None`` means absent (R3)."""
        assert self.t.translate_request(self._req(metadata={}))["_metadata"] == {}

    def test_no_tool_choice_invents_none(self):
        """No inbound ``tool_choice`` means neither key outbound (R9)."""
        result = self.t.translate_request(self._req())
        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result

    @pytest.mark.parametrize(
        "choice",
        [
            {"type": "auto"},
            {"type": "any"},
            {"type": "tool", "name": "get_weather"},
            {"type": "tool", "name": "not_declared"},
        ],
        ids=["auto", "any", "tool-declared", "tool-undeclared"],
    )
    def test_disable_parallel_tool_use_true_maps_to_parallel_tool_calls_false(self, choice):
        """The knob is inverted between the wires, and carried when it is on (R2)."""
        result = self.t.translate_request(self._req(tool_choice={**choice, "disable_parallel_tool_use": True}))
        assert result["parallel_tool_calls"] is False

    def test_disable_parallel_tool_use_false_is_omitted(self):
        """``false`` is both vendors' default, so it adds nothing to the body (D2)."""
        result = self.t.translate_request(self._req(tool_choice={"type": "any", "disable_parallel_tool_use": False}))
        assert result["tool_choice"] == "required"
        assert "parallel_tool_calls" not in result

    def test_disable_parallel_tool_use_on_none_is_ignored(self):
        """``ToolChoiceNone`` declares no such field, so it cannot produce the key (R2)."""
        result = self.t.translate_request(self._req(tool_choice={"type": "none", "disable_parallel_tool_use": True}))
        assert result["tool_choice"] == "none"
        assert "parallel_tool_calls" not in result

    def test_metadata_rides_the_internal_key(self):
        """``metadata`` is carried unmodified on ``_metadata``, never bare (R3)."""
        result = self.t.translate_request(self._req(metadata={"user_id": "u-123"}))
        assert result["_metadata"] == {"user_id": "u-123"}
        assert "metadata" not in result

    def test_null_metadata_invents_no_internal_key(self):
        """A ``null`` or absent ``metadata`` carries nothing (R3, R9)."""
        assert "_metadata" not in self.t.translate_request(self._req(metadata=None))
        assert "_metadata" not in self.t.translate_request(self._req())

    @pytest.mark.parametrize(
        "choice",
        [
            {"type": "auto"},
            {"type": "any"},
            {"type": "none"},
            {"type": "tool", "name": "get_weather"},
            {"type": "tool", "name": "get_weather", "disable_parallel_tool_use": True},
        ],
        ids=["auto", "any", "none", "tool", "tool-disable-parallel"],
    )
    def test_round_trip_to_an_anthropic_upstream(self, choice):
        """The ticket's reproduction: Messages -> CC -> Messages, unchanged (R1b, R2).

        This is the assertion that survives a refactor moving the mapping
        between the two hops.
        """
        cc = self.t.translate_request(self._req(tool_choice=choice, metadata={"user_id": "u-123"}))
        upstream = AnthropicAdapter().translate_to_upstream(cc)
        assert upstream["tool_choice"] == choice
        assert upstream["metadata"] == {"user_id": "u-123"}


# ── translate_request: cache breakpoints (KBR-198) ─────────────────────────


class TestTranslateRequestCacheBreakpoints:
    """Characterise which prompt-cache breakpoints ``translate_request`` destroys, site by site.

    KBR-198 (CB-1, epic KBR-197). Each body carries one one-hour breakpoint from
    :func:`harness.cache_breakpoints.build_request`, whose own tests prove it is
    really there. These tests pin today's behaviour, not the correct one: the
    carry-through fix is expected to turn the nine absence tests red and invert
    them.

    The cost stated in each docstring applies on upstreams that honour
    ``cache_control`` — Anthropic-compatible providers, and OpenRouter's Chat
    Completions API. Anthropic bills a cache read at 0.1x base input, so a lost
    breakpoint re-bills its prefix at roughly 10x on every turn. Where caching is
    implicit (OpenAI) the loss changes nothing billable.
    """

    def setup_method(self):
        """Give each test a fresh translator."""
        self.t = MessagesTranslator()

    def _translate(self, site: str) -> dict:
        """Translate the fixture body carrying a breakpoint at ``site``.

        Args:
            site: One of :data:`harness.cache_breakpoints.SITES`.

        Returns:
            The Chat Completions body ``translate_request`` emits.
        """
        return self.t.translate_request(cb.build_request(site))

    def test_tool_definition_breakpoint_is_destroyed(self):
        """Tool definitions are re-billed uncached every turn: the tool is rebuilt as a bare function."""
        assert cb.find_breakpoints(self._translate("tool")) == []

    def test_system_block_breakpoint_is_destroyed(self):
        """The system prompt is re-billed uncached every turn: its blocks are joined into one string."""
        assert cb.find_breakpoints(self._translate("system")) == []

    def test_user_text_block_breakpoint_is_destroyed(self):
        """History up to a user turn is re-billed uncached: text blocks are joined into one string."""
        assert cb.find_breakpoints(self._translate("user_text")) == []

    def test_assistant_text_block_breakpoint_is_destroyed(self):
        """History up to an assistant turn is re-billed uncached: text blocks are joined into one string."""
        assert cb.find_breakpoints(self._translate("assistant_text")) == []

    def test_image_block_breakpoint_is_destroyed(self):
        """A cached image prefix is re-billed uncached: the image block is dropped whole (KBR-222)."""
        assert cb.find_breakpoints(self._translate("image")) == []

    def test_document_block_breakpoint_is_destroyed(self):
        """A cached document prefix is re-billed uncached: the document block is dropped whole (KBR-222)."""
        assert cb.find_breakpoints(self._translate("document")) == []

    def test_tool_use_block_breakpoint_is_destroyed(self):
        """History up to a tool call is re-billed uncached: the block is rebuilt as ``tool_calls``."""
        assert cb.find_breakpoints(self._translate("tool_use")) == []

    def test_tool_result_block_breakpoint_is_destroyed(self):
        """History up to a tool result (Claude Code's usual last breakpoint) is re-billed uncached."""
        assert cb.find_breakpoints(self._translate("tool_result")) == []

    def test_top_level_automatic_caching_breakpoint_is_destroyed(self):
        """Automatic caching is switched off entirely: the top-level breakpoint appears nowhere in the output."""
        assert cb.find_breakpoints(self._translate("top_level")) == []

    def test_top_level_cache_control_is_not_a_re_emitted_key(self):
        """Automatic caching is switched off entirely: ``cache_control`` is not among the keys re-emitted.

        The ticket asks for this key-set form explicitly; the case above is
        strictly stronger, since it searches the whole body.
        """
        result = self._translate("top_level")

        assert "cache_control" not in result

    def test_breakpoint_nested_in_tool_result_content_survives_in_place(self):
        """The one breakpoint carried through, since ``tool_result.content`` is forwarded as-is: pinned, not endorsed.

        Whether Anthropic honours a breakpoint at that depth is not established:
        its SDK types accept one inside ``tool_result`` content, but its docs'
        sub-content rule names citations only (KBR-199). On an upstream that
        ignores ``cache_control`` it is an Anthropic field leaking into a Chat
        Completions ``tool`` message. The test exists so that this site, too,
        cannot start losing its breakpoint unnoticed.
        """
        result = self._translate("tool_result_nested")
        content = result["messages"][-1]["content"]

        assert isinstance(content, list), f"tool_result content was flattened: {content!r}"
        assert content[0]["cache_control"] == cb.BREAKPOINT
        assert cb.find_breakpoints(result) == [cb.BREAKPOINT]


# ── translate_response ──────────────────────────────────────────────────────


class TestTranslateResponse:
    def setup_method(self):
        self.t = MessagesTranslator()

    def test_text_response(self):
        cc_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.t.translate_response(cc_response)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["id"].startswith("msg_")
        assert result["model"] == "gpt-4o"
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_call_response(self):
        cc_response = {
            "id": "chatcmpl-456",
            "model": "gpt-4o",
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
        result = self.t.translate_response(cc_response)
        assert result["stop_reason"] == "tool_use"
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        tb = tool_blocks[0]
        assert tb["name"] == "get_weather"
        assert tb["input"] == {"city": "London"}

    def test_stop_reason_mapping(self):
        for finish_reason, expected in [("stop", "end_turn"), ("tool_calls", "tool_use"), ("length", "max_tokens")]:
            cc_response = {
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": finish_reason}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            result = self.t.translate_response(cc_response)
            assert result["stop_reason"] == expected, f"{finish_reason} -> {result['stop_reason']}, expected {expected}"

    def test_mixed_text_and_tool_call(self):
        cc_response = {
            "id": "chatcmpl-789",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = self.t.translate_response(cc_response)
        assert result["type"] == "message"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_empty_assistant_output_emits_fallback_text_block(self):
        cc_response = {
            "id": "chatcmpl-empty",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42},
        }

        result = self.t.translate_response(cc_response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"].strip() != ""
        assert "retry" in result["content"][0]["text"].lower()
        assert "/clear" in result["content"][0]["text"]

    def test_empty_assistant_output_prefers_refusal_text(self):
        cc_response = {
            "id": "chatcmpl-empty-refusal",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "refusal": "Policy refusal from upstream"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42},
        }

        result = self.t.translate_response(cc_response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Policy refusal from upstream"

    def test_empty_assistant_output_with_context_includes_provider_model(self):
        cc_response = {
            "id": "chatcmpl-empty",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42},
        }

        result = self.t.translate_response(
            cc_response,
            context={"provider": "openai", "model": "gpt-4o", "attempts": 3},
        )
        text = result["content"][0]["text"]
        assert "retry" in text.lower()
        assert "/clear" in text
        assert "openai" in text.lower()
        assert "gpt-4o" in text

    def test_empty_assistant_output_without_context_keeps_default(self):
        cc_response = {
            "id": "chatcmpl-empty",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42},
        }

        result = self.t.translate_response(cc_response)
        text = result["content"][0]["text"]
        assert text == (
            "Upstream model returned an empty response. Please retry. "
            "If the context is full, use /clear to reset the conversation."
        )

    def test_response_with_reasoning_content_mapped_to_thinking_block(self):
        """Chat Completions responses with reasoning_content must be mapped
        to thinking blocks in Messages API."""
        cc_response = {
            "id": "chatcmpl-reason",
            "model": "kimi-k2",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Step 1: Analyze. Step 2: Compute.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        result = self.t.translate_response(cc_response)
        # content should be [thinking block, text block]
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Step 1: Analyze. Step 2: Compute."
        assert result["content"][1]["type"] == "text"
        assert result["content"][1]["text"] == "The answer is 42."

    def test_response_with_only_reasoning_content_mapped_to_thinking_block(self):
        cc_response = {
            "id": "chatcmpl-reason-only",
            "model": "kimi-k2",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Deep thinking...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        result = self.t.translate_response(cc_response)
        # thinking-only response must include fallback text block
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Deep thinking..."
        assert result["content"][1]["type"] == "text"
        assert "retry" in result["content"][1]["text"].lower()


# ── translate_response: the thinking carriage (KBR-228 part A) ─────────────


class TestTranslateResponseThinkingCarriage:
    """Carried signed thinking blocks reach the Messages client verbatim.

    An Anthropic-family upstream's reply arrives with its thinking blocks under
    the internal ``_thinking_blocks`` key of the CC message; the Messages
    client must receive them in wire order, signatures included, instead of an
    unsigned rebuild from ``reasoning_content``.  Without the key, behaviour is
    unchanged.
    """

    def setup_method(self):
        self.t = MessagesTranslator()

    @staticmethod
    def _cc_response(message: dict) -> dict:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "claude-sonnet-4-6",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }

    def test_carried_blocks_are_emitted_verbatim_before_text(self):
        result = self.t.translate_response(
            self._cc_response(
                {
                    "content": "It is 18C.",
                    "_thinking_blocks": [
                        {"type": "thinking", "thinking": "I know this.", "signature": "sig-1"},
                        {"type": "redacted_thinking", "data": "opaque"},
                    ],
                }
            )
        )
        assert result["content"] == [
            {"type": "thinking", "thinking": "I know this.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "It is 18C."},
        ]
        assert self.t.response_was_empty is False

    def test_carriage_wins_over_reasoning_content(self):
        """The same reasoning rides both slots on this upstream; only one block may come out."""
        result = self.t.translate_response(
            self._cc_response(
                {
                    "content": "It is 18C.",
                    "reasoning_content": "I know this.",
                    "_thinking_blocks": [
                        {"type": "thinking", "thinking": "I know this.", "signature": "sig-1"},
                    ],
                }
            )
        )
        thinking_blocks = [b for b in result["content"] if b["type"] == "thinking"]
        assert thinking_blocks == [
            {"type": "thinking", "thinking": "I know this.", "signature": "sig-1"},
        ]

    def test_thinking_only_reply_still_takes_the_fallback(self):
        """Carried thinking does not make a reply non-empty: M12 still fires (KBR-155's decision)."""
        result = self.t.translate_response(
            self._cc_response(
                {
                    "content": None,
                    "_thinking_blocks": [
                        {"type": "thinking", "thinking": "Truncated mid-thought.", "signature": "sig-1"},
                    ],
                }
            )
        )
        assert self.t.response_was_empty is True
        assert [b["type"] for b in result["content"]] == ["thinking", "text"]
        assert "retry" in result["content"][1]["text"].lower()


# ── translate_request: the request carriage (KBR-228 part B) ───────────────


class TestRequestThinkingCarriage:
    """The agent's signed thinking blocks ride the CC request verbatim.

    ``translate_request`` used to flatten thinking to ``reasoning_content`` —
    signature and ``redacted_thinking`` lost — and to join the system block
    list into one string, which is exactly the history api.anthropic.com's
    signature binding rejects on turn 2 (KBR-228).  The original blocks and
    the original ``system`` value now ride internal keys so the Anthropic
    adapters can restore them verbatim; every other wire strips the keys.
    """

    def setup_method(self):
        self.t = MessagesTranslator()

    def test_assistant_thinking_blocks_are_carried_verbatim_in_order(self):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Reasoning.", "signature": "sig-1"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "Answer."},
                    ],
                },
            ],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assistant_msg = result["messages"][-1]
        assert assistant_msg["_thinking_blocks"] == [
            {"type": "thinking", "thinking": "Reasoning.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
        ]
        # The CC-facing halves stay: reasoning_content feeds providers that
        # display it, content/tool_calls feed every CC provider.
        assert assistant_msg["reasoning_content"] == "Reasoning."
        assert assistant_msg["content"] == "Answer."

    def test_assistant_without_thinking_carries_no_key(self):
        req = {
            "model": "m",
            "messages": [{"role": "assistant", "content": "Plain."}],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assert "_thinking_blocks" not in result["messages"][-1]

    def test_system_block_list_is_carried_verbatim_with_breakpoints(self):
        system = [
            {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Use the tools."},
        ]
        req = {"model": "m", "system": system, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        result = self.t.translate_request(req)
        assert result["_anthropic_system"] == system
        # The CC intermediate still carries the joined system message.
        assert result["messages"][0]["role"] == "system"
        assert "coding agent" in result["messages"][0]["content"]

    def test_system_string_is_carried_verbatim(self):
        req = {
            "model": "m",
            "system": "Plain system prompt.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }
        result = self.t.translate_request(req)
        assert result["_anthropic_system"] == "Plain system prompt."

    def test_no_system_means_no_system_carriage(self):
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        result = self.t.translate_request(req)
        assert "_anthropic_system" not in result


# ── translate_stream_chunk ─────────────────────────────────────────────────


class TestTranslateStreamChunk:
    def setup_method(self):
        self.t = MessagesTranslator()

    def _make_message_id(self) -> str:
        return f"msg_{uuid.uuid4().hex[:24]}"

    def test_text_delta_produces_content_block_events(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        # Should produce content_block_start + content_block_delta for text
        assert any("content_block_start" in e for e in events)
        assert any("content_block_delta" in e for e in events)
        # Check text_delta
        delta_events = [e for e in events if "content_block_delta" in e]
        assert '"text_delta"' in delta_events[0]
        assert '"Hello"' in delta_events[0]

    def test_finish_produces_message_delta_and_stop(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        assert any("message_delta" in e for e in events)
        assert any("message_stop" in e for e in events)
        # Check stop_reason
        delta_event = [e for e in events if "message_delta" in e][0]
        assert "end_turn" in delta_event

    def test_finish_without_content_emits_fallback_text_events(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }

        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        event_blob = "\n".join(events)
        assert "message_start" in event_blob
        assert "content_block_start" in event_blob
        assert "content_block_delta" in event_blob
        assert "/clear" in event_blob
        assert "retry" in event_blob.lower()
        assert "message_stop" in event_blob

    def test_finalize_interrupted_stream_closes_open_text_block(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }

        initial_events = self.t.translate_stream_chunk(msg_id, model, chunk)
        final_events = self.t.finalize_interrupted_stream()
        final_blob = "\n".join(final_events)

        assert any("content_block_delta" in event for event in initial_events)
        assert "content_block_stop" in final_blob
        assert "message_delta" in final_blob
        assert '"stop_reason": "end_turn"' in final_blob
        assert "message_stop" in final_blob
        assert "error" not in final_blob
        assert self.t._text_block_opened is False
        assert self.t._message_started is False

    def test_finalize_interrupted_stream_with_message_start_only_emits_fallback_block(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        events: list[str] = []
        self.t._emit_message_start_if_needed(events, msg_id, model)

        final_events = self.t.finalize_interrupted_stream()
        final_blob = "\n".join(final_events)

        assert "message_start" not in final_blob
        assert "content_block_start" in final_blob
        assert "content_block_delta" in final_blob
        assert "/clear" in final_blob
        assert "content_block_stop" in final_blob
        assert "message_delta" in final_blob
        assert "message_stop" in final_blob

    def test_finalize_interrupted_stream_closes_open_tool_block(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"query":'},
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
        }

        initial_events = self.t.translate_stream_chunk(msg_id, model, chunk)
        final_events = self.t.finalize_interrupted_stream()
        final_blob = "\n".join(final_events)

        assert any("tool_use" in event for event in initial_events)
        assert "content_block_stop" in final_blob
        assert "message_delta" in final_blob
        assert "message_stop" in final_blob
        assert self.t._tool_call_buffers == {}

    def test_finalize_interrupted_stream_without_started_message_is_noop(self):
        assert self.t.finalize_interrupted_stream() == []

    def test_tool_call_delta_produces_input_json_delta(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        # First chunk: tool call name
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events1 = self.t.translate_stream_chunk(msg_id, model, chunk1)
        assert any("content_block_start" in e for e in events1)
        assert any("tool_use" in e for e in events1)

        # Second chunk: argument delta
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
                    "finish_reason": None,
                }
            ],
        }
        events2 = self.t.translate_stream_chunk(msg_id, model, chunk2)
        assert any("input_json_delta" in e for e in events2)

    def test_content_block_index_increments(self):
        msg_id = self._make_message_id()
        model = "claude-3-opus"
        # Text delta (index 0)
        chunk1 = {
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        events1 = self.t.translate_stream_chunk(msg_id, model, chunk1)
        # Verify text block at index 0
        start_events = [e for e in events1 if "content_block_start" in e]
        assert '"index": 0' in start_events[0]

        # Tool call (should be index 1)
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_001",
                                "type": "function",
                                "function": {"name": "fn", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events2 = self.t.translate_stream_chunk(msg_id, model, chunk2)
        start_events2 = [e for e in events2 if "content_block_start" in e]
        assert '"index": 1' in start_events2[0]

    def test_auto_reset_after_finish(self):
        msg_id = self._make_message_id()
        model = "m"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.t.translate_stream_chunk(msg_id, model, chunk)
        assert self.t._tool_call_buffers == {}
        assert self.t._content_block_index == 0
        assert self.t._text_block_opened is False

    def test_reset_clears_internal_state(self):
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        self.t.translate_stream_chunk("msg_1", "m", chunk)
        self.t.reset()
        assert self.t._tool_call_buffers == {}
        assert self.t._content_block_index == 0
        assert self.t._text_block_opened is False

    def test_finish_reason_length_maps_to_max_tokens(self):
        msg_id = self._make_message_id()
        model = "m"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        delta_event = [e for e in events if "message_delta" in e][0]
        assert "max_tokens" in delta_event

    def test_empty_choices_returns_empty_events(self):
        """SSE chunks with empty choices list (e.g. Fireworks usage chunks) must not crash."""
        msg_id = self._make_message_id()
        model = "m"
        chunk = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        assert events == []

    def test_missing_choices_returns_empty_events(self):
        """SSE chunks with no choices key must not crash."""
        msg_id = self._make_message_id()
        model = "m"
        chunk = {"id": "chatcmpl-1", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        assert events == []

    def test_duplicate_finish_chunk_does_not_emit_extra_message(self):
        """Duplicate finish_reason chunks (emitted by some models/providers like Gemma/OpenRouter)
        must not produce a second message lifecycle with fallback text.

        Reproduces: https://github.com/QwenLM/qwen-code/issues/2402
        Some models emit two consecutive finish_reason chunks. After the first finishes
        and self.reset() is called, the second (empty) finish chunk must be silently ignored
        rather than triggering the fallback-text path.
        """
        msg_id = self._make_message_id()
        model = "m"

        # First chunk: content delta with finish
        chunk1 = {
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}],
        }
        events1 = self.t.translate_stream_chunk(msg_id, model, chunk1)

        # Should produce: message_start, content_block_start(text), content_block_delta("hello"),
        # content_block_stop, message_delta, message_stop
        content_block_deltas = [e for e in events1 if "content_block_delta" in e]
        assert len(content_block_deltas) == 1
        assert '"text": "hello"' in content_block_deltas[0]

        message_stops = [e for e in events1 if "message_stop" in e]
        assert len(message_stops) == 1

        # Second chunk: duplicate finish (empty delta) — must be ignored
        chunk2 = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        events2 = self.t.translate_stream_chunk(msg_id, model, chunk2)

        # Must produce ZERO events — no second message lifecycle, no fallback text
        assert events2 == [], f"Duplicate finish chunk should produce no events, got: {events2}"

    def test_translator_reuse_across_requests(self):
        """A new message_id must reset _finished so the translator can handle multiple requests."""
        msg_id1 = self._make_message_id()
        msg_id2 = self._make_message_id()
        model = "m"

        # First request finishes normally
        chunk1 = {
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        }
        events1 = self.t.translate_stream_chunk(msg_id1, model, chunk1)
        assert any("message_stop" in e for e in events1)

        # Second request (different message_id) must also finish properly
        chunk2 = {
            "choices": [{"index": 0, "delta": {"content": "bye"}, "finish_reason": "stop"}],
        }
        events2 = self.t.translate_stream_chunk(msg_id2, model, chunk2)
        assert any("message_stop" in e for e in events2), "Second request should emit message_stop"
        content_deltas = [e for e in events2 if "content_block_delta" in e]
        assert any('"bye"' in e for e in content_deltas), "Second request should emit its content"

    def test_reasoning_delta_produces_thinking_block(self):
        """Streaming reasoning_content deltas must map to thinking blocks in Messages API."""
        msg_id = self._make_message_id()
        model = "kimi-k2"

        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Let me think..."},
                    "finish_reason": None,
                }
            ],
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk)
        # Should produce content_block_start for thinking + content_block_delta
        start_events = [e for e in events if "content_block_start" in e]
        assert len(start_events) == 1
        assert '"thinking"' in start_events[0]

        delta_events = [e for e in events if "content_block_delta" in e]
        assert len(delta_events) == 1
        assert '"thinking_delta"' in delta_events[0]
        assert "Let me think..." in delta_events[0]

    def test_reasoning_then_text_content_block_increments(self):
        """Thinking block at index 0, then text block at index 1."""
        msg_id = self._make_message_id()
        model = "kimi-k2"

        # Reasoning chunk
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Analyzing..."},
                    "finish_reason": None,
                }
            ],
        }
        events1 = self.t.translate_stream_chunk(msg_id, model, chunk1)
        thinking_starts = [e for e in events1 if "content_block_start" in e]
        assert '"index": 0' in thinking_starts[0]

        # Text chunk
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "The answer."},
                    "finish_reason": None,
                }
            ],
        }
        events2 = self.t.translate_stream_chunk(msg_id, model, chunk2)
        text_starts = [e for e in events2 if "content_block_start" in e]
        assert len(text_starts) == 1
        assert '"index": 1' in text_starts[0]

    def test_thinking_only_finish_emits_fallback_text(self):
        """When the stream produces only thinking and no text, fallback text must be added."""
        msg_id = self._make_message_id()
        model = "kimi-k2"

        # Reasoning chunk
        chunk1 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Deep thinking..."},
                    "finish_reason": None,
                }
            ],
        }
        self.t.translate_stream_chunk(msg_id, model, chunk1)

        # Finish chunk with no text
        chunk2 = {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        events = self.t.translate_stream_chunk(msg_id, model, chunk2)
        event_blob = "\n".join(events)

        # Must contain a fallback text block after thinking
        assert "retry" in event_blob.lower() or "/clear" in event_blob

    def test_thinking_only_finalized_interrupted_stream_emits_fallback_text(self):
        """When an interrupted stream has only a thinking block open, fallback text must be added."""
        msg_id = self._make_message_id()
        model = "kimi-k2"

        # Reasoning chunk to open a thinking block
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "Thinking..."},
                    "finish_reason": None,
                }
            ],
        }
        self.t.translate_stream_chunk(msg_id, model, chunk)

        # Finalize the interrupted stream
        events = self.t.finalize_interrupted_stream()
        event_blob = "\n".join(events)

        # Must contain fallback text
        assert "retry" in event_blob.lower() or "/clear" in event_blob


# ── close_open_blocks (KBR-183) ─────────────────────────────────────────────


def _parse_events(events: list[str]) -> list[tuple[str, dict]]:
    """Split formatted SSE event strings into ``(event name, data)`` pairs.

    Args:
        events: Strings as returned by the translator, one SSE event each.

    Returns:
        The event name and decoded JSON payload of every event, in order.
    """
    parsed = []
    for event in events:
        name = next(line[len("event: ") :] for line in event.splitlines() if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in event.splitlines() if line.startswith("data: "))
        parsed.append((name, json.loads(data)))
    return parsed


class TestCloseOpenBlocks:
    """A post-emission failure closes what the client has open and nothing more.

    §11 Q14(a): the turn ends in one terminal error, so this must not
    synthesise the normal ending (``message_delta`` / ``message_stop``) or put
    fallback words in the model's mouth the way
    :meth:`MessagesTranslator.finalize_interrupted_stream` does.
    """

    def setup_method(self):
        """Give each test a fresh translator."""
        self.t = MessagesTranslator()
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def _feed(self, delta: dict) -> None:
        """Translate one non-final Chat Completions chunk carrying ``delta``.

        Args:
            delta: The ``choices[0].delta`` object of the chunk.
        """
        self.t.translate_stream_chunk(
            self.msg_id, "m", {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        )

    def test_close_open_blocks_before_any_message_returns_nothing(self):
        """Nothing reached the client, so there is nothing to close."""
        assert self.t.close_open_blocks() == []

    def test_close_open_blocks_stops_an_open_text_block_only(self):
        """An open text block gets exactly its own stop event."""
        self._feed({"content": "Hello"})

        assert _parse_events(self.t.close_open_blocks()) == [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ]

    def test_close_open_blocks_stops_an_open_thinking_block(self):
        """A reply that is still thinking has its thinking block closed."""
        self._feed({"reasoning_content": "hmm"})

        assert _parse_events(self.t.close_open_blocks()) == [
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ]

    def test_close_open_blocks_stops_a_tool_block_with_partial_arguments(self):
        """A tool_use whose arguments were cut off is closed at its own index."""
        self._feed({"content": "Let me look."})
        self._feed(
            {
                "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"q":'}}
                ]
            }
        )

        assert _parse_events(self.t.close_open_blocks()) == [
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ]

    def test_close_open_blocks_twice_closes_nothing_the_second_time(self):
        """A block is closed once; a repeat call must not emit a second stop."""
        self._feed({"content": "Hello"})
        self.t.close_open_blocks()

        assert self.t.close_open_blocks() == []


class TestParallelToolCallBlockIndices:
    """Parallel ``tool_use`` calls each open their own Anthropic content block.

    KBR-226: the tool-call branch recorded ``block_index: self._content_block_index``
    but never advanced the counter, so a translated stream carrying two tool calls
    opened both at index 0 and the finish path closed index 0 twice. The Anthropic
    stream grammar identifies a block by its index, so Claude Code saw the second
    call reuse the first one's slot while it was still open. Text that follows the
    calls opens at the next free index; the tool blocks stay open until the finish
    closes each at its own index, so a late argument delta can never land after
    its block's stop.
    """

    def setup_method(self):
        """Give each test a fresh translator."""
        self.t = MessagesTranslator()
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def _feed(self, delta: dict) -> list[tuple[str, dict]]:
        """Translate one non-final Chat Completions chunk carrying ``delta``.

        Args:
            delta: The ``choices[0].delta`` object of the chunk.

        Returns:
            The emitted events, parsed as ``(event_name, data_dict)`` pairs.
        """
        return _parse_events(
            self.t.translate_stream_chunk(
                self.msg_id, "m", {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            )
        )

    @staticmethod
    def _tool_call(cc_index: int, call_id: str, arguments: str = "") -> dict:
        """Build one Chat Completions ``tool_calls`` delta entry.

        Args:
            cc_index: The Chat Completions tool-call index the entry belongs to.
            call_id: The upstream call id; its presence marks a new call.
            arguments: The argument fragment the entry carries.

        Returns:
            A ``tool_calls`` delta entry in Chat Completions shape.
        """
        return {
            "index": cc_index,
            "id": call_id,
            "type": "function",
            "function": {"name": f"fn_{call_id}", "arguments": arguments},
        }

    def test_two_parallel_tool_calls_open_distinct_increasing_indices(self):
        """The second call must not reuse the first one's open block slot."""
        events1 = self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        events2 = self._feed({"tool_calls": [self._tool_call(1, "call_b")]})

        starts = [d for name, d in [*events1, *events2] if name == "content_block_start"]
        assert [d["index"] for d in starts] == [0, 1]
        assert all(d["content_block"]["type"] == "tool_use" for d in starts)

    def test_interleaved_argument_deltas_route_to_their_own_block(self):
        """Argument chunks are routed by the per-call meta, not by arrival order.

        The interleaved order (id0, id1, args1, args0) is what pins the routing:
        a translator that tracked "the current" index instead would swap the two.
        """
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        self._feed({"tool_calls": [self._tool_call(1, "call_b")]})
        events3 = self._feed({"tool_calls": [{"index": 1, "function": {"arguments": '{"b":'}}]})
        events4 = self._feed({"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]})

        deltas = [
            (d["index"], d["delta"]["partial_json"])
            for name, d in [*events3, *events4]
            if name == "content_block_delta"
        ]
        assert deltas == [(1, '{"b":'), (0, '{"a":')]

    def test_finish_emits_one_stop_per_opened_index(self):
        """The finish closes each tool block once, at that block's own index."""
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        self._feed({"tool_calls": [self._tool_call(1, "call_b")]})
        final = _parse_events(
            self.t.translate_stream_chunk(
                self.msg_id,
                "m",
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            )
        )

        stops = [d["index"] for name, d in final if name == "content_block_stop"]
        assert stops == [0, 1]
        tail = [name for name, _ in final]
        assert tail[-2:] == ["message_delta", "message_stop"]

    def test_text_after_tool_calls_opens_at_next_free_index(self):
        """Text following the calls opens a fresh block; the tool blocks stay open."""
        self._feed({"tool_calls": [self._tool_call(0, "call_a")]})
        self._feed({"tool_calls": [self._tool_call(1, "call_b")]})
        events3 = self._feed({"content": "Both calls are in."})

        starts = [d for name, d in events3 if name == "content_block_start"]
        assert len(starts) == 1
        assert starts[0]["index"] == 2
        assert starts[0]["content_block"]["type"] == "text"
        # Overlap is the decided shape: no stop may close a tool block early,
        # because a late argument delta would land after its block's stop.
        assert not [name for name, _ in events3 if name == "content_block_stop"]

    def test_finalize_interrupted_stream_closes_each_tool_block_at_its_own_index(self):
        """An interrupted stream closes every open tool block exactly once."""
        self._feed({"tool_calls": [self._tool_call(0, "call_a", arguments='{"a":')]})
        self._feed({"tool_calls": [self._tool_call(1, "call_b", arguments='{"b":')]})

        final = _parse_events(self.t.finalize_interrupted_stream())
        stops = [d["index"] for name, d in final if name == "content_block_stop"]
        assert stops == [0, 1]
