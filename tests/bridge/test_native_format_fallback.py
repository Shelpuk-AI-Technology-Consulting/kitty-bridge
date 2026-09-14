"""Tests for native passthrough format fallback — when the upstream rejects
Anthropic-format ``tool_use`` blocks, the bridge converts to Chat Completions
format and retries the same backend without marking it unhealthy.
"""

from __future__ import annotations

import json

import pytest

from kitty.bridge.messages.translator import MessagesTranslator
from kitty.bridge.server import (
    _convert_native_to_cc_format,
    _has_tool_use_blocks,
    _is_tool_use_format_error,
    _normalize_cc_stop,
)
from kitty.providers.anthropic import AnthropicAdapter

# ── Helpers ────────────────────────────────────────────────────────────────


def _anthropic_body_with_tool_use() -> dict:
    """Minimal Anthropic Messages body with tool_use + tool_result blocks."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please run ls"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_abc123",
                        "name": "Bash",
                        "input": {"command": "ls -la"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc123",
                        "content": "total 42\ndrwxr-xr-x ...",
                    },
                ],
            },
        ],
        "system": [{"type": "text", "text": "You are helpful."}],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a bash command",
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        ],
        "stream": True,
    }


def _anthropic_body_text_only() -> dict:
    """Anthropic Messages body with only text content."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ],
        "stream": True,
    }


def _anthropic_body_with_mixed_content() -> dict:
    """Anthropic body with text + tool_use in same assistant message."""
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Do X"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will do X now."},
                    {
                        "type": "tool_use",
                        "id": "call_mixed",
                        "name": "Bash",
                        "input": {"command": "do_x"},
                    },
                ],
            },
        ],
        "stream": True,
    }


# ── has_tool_use_blocks ───────────────────────────────────────────────────


class TestHasToolUseBlocks:
    def test_no_tool_use_returns_false(self):
        body = _anthropic_body_text_only()
        assert _has_tool_use_blocks(body) is False

    def test_with_tool_use_returns_true(self):
        body = _anthropic_body_with_tool_use()
        assert _has_tool_use_blocks(body) is True

    def test_mixed_content_returns_true(self):
        body = _anthropic_body_with_mixed_content()
        assert _has_tool_use_blocks(body) is True

    def test_no_messages_returns_false(self):
        body = {"model": "test", "stream": True}
        assert _has_tool_use_blocks(body) is False

    def test_empty_messages_returns_false(self):
        body = {"model": "test", "messages": [], "stream": True}
        assert _has_tool_use_blocks(body) is False

    def test_assistant_with_string_content(self):
        body = {
            "model": "test",
            "messages": [{"role": "assistant", "content": "plain text"}],
        }
        assert _has_tool_use_blocks(body) is False


# ── _is_tool_use_format_error ─────────────────────────────────────────────


class TestIsToolUseFormatError:
    def test_unknown_variant_tool_use(self):
        assert (
            _is_tool_use_format_error(400, '{"error": {"message": "unknown variant `tool_use`, expected `text`"}}')
            is True
        )

    def test_tool_call_result_does_not_follow(self):
        assert (
            _is_tool_use_format_error(
                400,
                '{"error": {"message": "tool call result does not follow tool call"}}',
            )
            is True
        )

    def test_2013_code(self):
        assert (
            _is_tool_use_format_error(
                400,
                '{"type":"error","error":{"code":"2013","message":"tool call result does not follow tool call"}}',
            )
            is True
        )

    def test_regular_400_not_detected(self):
        assert _is_tool_use_format_error(400, "Bad request") is False

    def test_500_not_detected(self):
        assert _is_tool_use_format_error(500, '{"error": {"message": "unknown variant"}}') is False

    def test_none_body(self):
        assert _is_tool_use_format_error(400, None) is False

    def test_dict_body(self):
        assert (
            _is_tool_use_format_error(400, {"error": {"message": "unknown variant `tool_use`, expected `text`"}})
            is True
        )

    def test_invalid_params_2013(self):
        assert (
            _is_tool_use_format_error(
                400,
                '{"error": {"message": "invalid params, tool call result does not follow tool call (2013)"}}',
            )
            is True
        )

    def test_minimax_tool_result_not_found_2013(self):
        # The EXACT production error string captured in debug/bridge.log —
        # MiniMax's actual variant, not the guessed "does not follow" wording.
        assert (
            _is_tool_use_format_error(
                400,
                '{"type":"error","error":{"type":"invalid_request_error",'
                '"message":"invalid params, tool result\'s tool id'
                '(call_f64ba2ce682a457f966bd6d7) not found (2013)"}}',
            )
            is True
        )

    def test_tool_result_not_found_no_code(self):
        # Phrase-based match must work without the numeric code.
        assert _is_tool_use_format_error(400, '{"error":{"message":"tool result not found"}}') is True

    def test_not_found_alone_not_matched(self):
        # "not found" without "tool result" must NOT match (false-positive guard).
        assert _is_tool_use_format_error(400, '{"error":{"message":"model not found"}}') is False

    def test_2013_alone_not_matched(self):
        # Bare "2013" with an unrelated message must NOT match — justifies the
        # phrase-based match over matching the numeric code.
        assert _is_tool_use_format_error(400, '{"error":{"code":"2013","message":"rate limit exceeded"}}') is False


# ── _convert_native_to_cc_format ──────────────────────────────────────────


class TestConvertNativeToCCFormat:
    def test_carries_signed_thinking_and_system_verbatim(self):
        """KBR-228 part B: the fallback converter is a second Messages -> CC hop.

        It owes the same carriage as ``MessagesTranslator.translate_request``:
        the assistant's signed thinking blocks and the original ``system`` ride
        internal keys so the Anthropic adapters can restore them verbatim on
        the retry.  Without them, a failover retry rebuilds the history
        unsigned and api.anthropic.com rejects the turn again.
        """
        body = {
            "model": "claude-sonnet-4-6",
            "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Reply briefly.", "signature": "sig-9"},
                        {"type": "text", "text": "Hello."},
                    ],
                },
            ],
        }
        result = _convert_native_to_cc_format(body)

        assert result["_anthropic_system"] == body["system"]
        assistant = [m for m in result["messages"] if m["role"] == "assistant"]
        assert assistant[0]["_thinking_blocks"] == [
            {"type": "thinking", "thinking": "Reply briefly.", "signature": "sig-9"},
        ]
        # The CC-facing translation is unchanged.
        assert assistant[0]["content"] == "Hello."

    def test_system_prompt_becomes_system_message(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_tool_use_becomes_tool_calls(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        # Find the assistant message
        assistant = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant) > 0
        assert assistant[0].get("tool_calls") is not None
        assert len(assistant[0]["tool_calls"]) == 1
        tc = assistant[0]["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "Bash"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls -la"}

    def test_tool_result_becomes_tool_message(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_abc123"
        assert "total 42" in tool_msgs[0]["content"]

    def test_anthropic_tools_become_cc_tools(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "Bash"
        assert result["tools"][0]["function"]["description"] == "Run a bash command"

    def test_model_stream_preserved(self):
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert result["model"] == "claude-sonnet-4-6"
        assert result["stream"] is True

    def test_text_only_content_unchanged(self):
        body = _anthropic_body_text_only()
        result = _convert_native_to_cc_format(body)

        # User message content should be a plain string (text block stripped)
        user_msg = result["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Hello"

    def test_image_beside_tool_result_is_carried(self):
        """KBR-222: the fallback owes hop 1's mappings — an image sibling survives the retry.

        KBR-178's stop sequences were lost on exactly this retry path until the
        converter was taught hop 1's mappings; images are the same class. The
        fallback's text-first placement is kept: the non-tool blocks come
        before the tool messages.
        """
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aWNvbg=="},
        }
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "run it"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                        image,
                    ],
                },
            ],
        }
        result = _convert_native_to_cc_format(body)

        sibling = result["messages"][2]
        assert sibling["role"] == "user"
        assert sibling["content"] == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aWNvbg=="}},
        ]
        assert result["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "ok"}

    def test_document_rides_the_documents_key_addressed_to_its_message(self):
        """KBR-222: a document in the retried history travels on ``_documents``, not dropped."""
        document = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRm"},
        }
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "summarise"}, document],
                }
            ],
        }
        result = _convert_native_to_cc_format(body)

        message = result["messages"][0]
        assert message["content"] == "summarise"
        assert len(result["_documents"]) == 1
        assert result["_documents"][0]["blocks"] == [document]
        assert result["_documents"][0]["message"] is message

    def test_image_only_message_becomes_parts_not_verbatim(self):
        """KBR-222: an image-only turn converts to a parts message, not a raw passthrough.

        The old converter appended the native block list verbatim, which only
        delivered on an adapter that forwards user content unchanged; the
        shared builder gives the retry the same CC spelling hop 1 produces.
        """
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/cat.png"},
                        }
                    ],
                }
            ],
        }
        result = _convert_native_to_cc_format(body)

        assert result["messages"][0] == {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }
        assert "_documents" not in result

    def test_empty_list_user_message_passes_through_verbatim(self):
        """A ``content: []`` user turn keeps the pre-KBR-222 passthrough, message count included.

        M17's stripping leaves such turns behind and counts on message
        indices not moving; the shared builder has nothing to build from an
        empty list, so the original message must pass through instead of
        vanishing from the retry.
        """
        body = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "before"},
                {"role": "user", "content": []},
            ],
        }
        result = _convert_native_to_cc_format(body)

        assert result["messages"][1] == {"role": "user", "content": []}
        assert len(result["messages"]) == 2

    def test_mixed_content_preserves_text_and_tool_use(self):
        body = _anthropic_body_with_mixed_content()
        result = _convert_native_to_cc_format(body)

        assistant = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant) == 1
        # Should have text content AND tool_calls
        assert "I will do X now" in assistant[0]["content"]
        assert assistant[0].get("tool_calls") is not None
        assert len(assistant[0]["tool_calls"]) == 1

    def test_no_system_prompt(self):
        body = _anthropic_body_text_only()
        result = _convert_native_to_cc_format(body)

        # No system message should be added
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_max_tokens_preserved(self):
        body = _anthropic_body_with_tool_use()
        body["max_tokens"] = 4096
        result = _convert_native_to_cc_format(body)

        assert result["max_tokens"] == 4096

    def test_temperature_top_p_preserved(self):
        body = _anthropic_body_with_tool_use()
        body["temperature"] = 0.7
        body["top_p"] = 0.9
        result = _convert_native_to_cc_format(body)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_stop_sequences_and_top_k_preserved(self):
        """KBR-178: the fallback converter must not re-drop what hop 1 carries.

        ``_convert_native_to_cc_format`` is a second, partial Messages -> CC
        converter.  Without this mapping the ``tool_use`` retry loses the
        user's stop sequences on exactly the Anthropic-family adapters the
        first-hop fix exists to serve.
        """
        body = _anthropic_body_with_tool_use()
        body["stop_sequences"] = ["A", "B"]
        body["top_k"] = 40
        result = _convert_native_to_cc_format(body)

        assert result["stop"] == ["A", "B"]
        assert result["_top_k"] == 40
        assert "stop_sequences" not in result
        assert "top_k" not in result

        # Two of the four call sites pass the already-normalised `cc_request`
        # rather than the pristine `body` (server.py:4402, 5318), so the mapping
        # is exercised against a body carrying the guard flag too.  This is a
        # shape check, not a pipeline check: it does not re-run _normalize_model,
        # normalize_request, truncation or compaction.
        mutated = _anthropic_body_with_tool_use()
        mutated["stop_sequences"] = ["A", "B"]
        mutated["top_k"] = 40
        mutated["_native_messages_request"] = True
        from_mutated = _convert_native_to_cc_format(mutated)
        assert from_mutated["stop"] == ["A", "B"]
        assert from_mutated["_top_k"] == 40

    def test_no_stop_sequences_or_top_k_invents_nothing(self):
        """Neither field inbound means neither key outbound."""
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert "stop" not in result
        assert "_top_k" not in result

    def test_empty_stop_sequences_is_omitted(self):
        """The empty-list rule holds at this converter too — see D6."""
        body = _anthropic_body_with_tool_use()
        body["stop_sequences"] = []
        result = _convert_native_to_cc_format(body)

        assert "stop" not in result

    def test_fallback_body_reaches_anthropic_upstream_with_stop_sequences(self):
        """End to end over the seam: fallback body -> Anthropic Messages body.

        This is the assertion that fails if either half of the pair is missing,
        and it is the one that names the defect in user terms — a stop sequence
        survives the ``tool_use`` retry.
        """
        body = _anthropic_body_with_tool_use()
        body["stop_sequences"] = ["A", "B"]
        cc = _convert_native_to_cc_format(body)
        upstream = AnthropicAdapter().translate_to_upstream(cc)

        assert upstream["stop_sequences"] == ["A", "B"]

    @pytest.mark.parametrize(
        "choice",
        [
            {"type": "auto"},
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "any", "disable_parallel_tool_use": False},
            {"type": "none", "disable_parallel_tool_use": True},
            {"type": "tool", "name": "Bash", "disable_parallel_tool_use": True},
            {"type": "tool"},
            "auto",
            None,
        ],
        ids=["auto", "any-disable", "any-keep", "none-disable", "tool-disable", "tool-nameless", "string", "null"],
    )
    @pytest.mark.parametrize("guarded", [False, True], ids=["pristine-body", "normalised-cc-request"])
    def test_tool_choice_and_metadata_agree_with_the_translator(self, choice, guarded):
        """KBR-214 R4: both Messages -> CC converters produce the same three keys.

        ``_convert_native_to_cc_format`` is a second, partial copy of hop 1, and a
        drifted second copy is how KBR-178's field was lost on the ``tool_use``
        retry.  Parity over legal, off-by-default and malformed shapes is what
        catches this converter's table *diverging*; a faithful copy is caught by
        the call test below instead.  Both argument shapes are exercised: the
        live Messages-ingress site (``_stream_messages``) passes the pristine
        body, and the three sites in the other stream handlers pass a
        ``cc_request`` carrying the guard flag.
        """
        body = _anthropic_body_with_tool_use()
        body["tool_choice"] = choice
        body["metadata"] = {"user_id": "u-123"}
        from_translator = MessagesTranslator().translate_request(body)
        if guarded:
            body["_native_messages_request"] = True
        from_fallback = _convert_native_to_cc_format(body)

        for key in ("tool_choice", "parallel_tool_calls", "_metadata"):
            assert from_fallback.get(key, "<absent>") == from_translator.get(key, "<absent>"), key

    def test_the_fallback_converter_calls_the_shared_helper(self, monkeypatch):
        """R4 is "the same helper", not "an equal table" (D3).

        A faithful copy of the value table passes every parity case above and
        still reintroduces the drift KBR-178 paid for, so the call itself is the
        claim.
        """
        import kitty.bridge.server as server

        seen: list[tuple[dict, dict]] = []
        monkeypatch.setattr(server, "carry_tool_choice_and_metadata", lambda body, cc: seen.append((body, cc)))
        body = _anthropic_body_with_tool_use()
        result = _convert_native_to_cc_format(body)

        assert len(seen) == 1
        assert seen[0][0] is body
        assert seen[0][1] is result

    def test_no_tool_choice_or_metadata_invents_nothing(self):
        """Neither field inbound means none of the three keys outbound (R9)."""
        result = _convert_native_to_cc_format(_anthropic_body_with_tool_use())

        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result
        assert "_metadata" not in result

    def test_fallback_body_reaches_anthropic_upstream_with_tool_choice_and_metadata(self):
        """End to end over the seam, in user terms: a forced tool survives the retry."""
        body = _anthropic_body_with_tool_use()
        body["tool_choice"] = {"type": "tool", "name": "Bash", "disable_parallel_tool_use": True}
        body["metadata"] = {"user_id": "u-123"}
        upstream = AnthropicAdapter().translate_to_upstream(_convert_native_to_cc_format(body))

        assert upstream["tool_choice"] == {"type": "tool", "name": "Bash", "disable_parallel_tool_use": True}
        assert upstream["metadata"] == {"user_id": "u-123"}


class TestNormalizeCCStop:
    """KBR-178 R11: a string `stop` is the same request as a one-item list.

    ``StopConfiguration`` declares ``stop`` as ``oneOf`` a string or an array, so
    the bridge owes the two forms identical treatment.  Normalised once at the
    Chat Completions ingress rather than at each rebuild seam, for the reason
    ``normalize_responses_request`` exists (KBR-144).
    """

    def test_a_string_becomes_a_one_item_list(self):
        """The load-bearing case: without this, Anthropic gets a 400."""
        cc = {"model": "m", "messages": [], "stop": "END"}
        _normalize_cc_stop(cc)
        assert cc["stop"] == ["END"]

    def test_a_list_is_left_unchanged(self):
        """The array form meets the rewrite as a no-op — why it is unconditional."""
        cc = {"model": "m", "messages": [], "stop": ["A", "B"]}
        _normalize_cc_stop(cc)
        assert cc["stop"] == ["A", "B"]

    def test_no_stop_is_left_unchanged(self):
        """No `stop` invents none."""
        cc = {"model": "m", "messages": []}
        _normalize_cc_stop(cc)
        assert "stop" not in cc

    def test_an_empty_string_is_not_wrapped(self):
        """`[""]` is rejected by Bedrock's NonEmptyString and can never match.

        Left falsy so the seams omit it, exactly as they omit ``[]``.
        """
        cc = {"model": "m", "messages": [], "stop": ""}
        _normalize_cc_stop(cc)
        assert cc["stop"] == ""

    def test_a_null_stop_is_left_alone(self):
        """`stop` is nullable in Chat Completions; null stays null for R8 to drop."""
        cc = {"model": "m", "messages": [], "stop": None}
        _normalize_cc_stop(cc)
        assert cc["stop"] is None

    def test_normalisation_is_idempotent(self):
        """Running it twice must not nest the list."""
        cc = {"model": "m", "messages": [], "stop": "END"}
        _normalize_cc_stop(cc)
        _normalize_cc_stop(cc)
        assert cc["stop"] == ["END"]

    def test_the_string_form_reaches_an_anthropic_upstream_as_a_list(self):
        """End to end over the seam the finding was about.

        Before R11 this produced ``stop_sequences: "END"``, which Anthropic
        rejects — a hard failure where the field had previously been dropped
        silently.
        """
        cc = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stop": "END"}
        _normalize_cc_stop(cc)
        upstream = AnthropicAdapter().translate_to_upstream(cc)
        assert upstream["stop_sequences"] == ["END"]


# ── Integration test — full round-trip with server ────────────────────────


class TestNativeFormatFallbackInServer:
    """Verify the fallback is wired in the BridgeServer error handling."""

    def test_is_tool_use_format_error_module_level(self):
        """_is_tool_use_format_error detects tool_use format mismatches."""
        assert _is_tool_use_format_error(400, "tool call result does not follow tool call") is True
        assert _is_tool_use_format_error(500, "tool call result does not follow tool call") is False

    def test_has_tool_use_blocks_module_level(self):
        """_has_tool_use_blocks detects Anthropic-format tool_use."""
        body = _anthropic_body_with_tool_use()
        assert _has_tool_use_blocks(body) is True
        body_no_tools = _anthropic_body_text_only()
        assert _has_tool_use_blocks(body_no_tools) is False
