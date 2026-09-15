"""Tests for providers/bedrock.py — BedrockAdapter."""

from unittest.mock import MagicMock, patch

import pytest

from kitty.providers.base import ProviderError
from kitty.providers.bedrock import BedrockAdapter

# ── CC format samples ────────────────────────────────────────────────────

CC_MESSAGES_BASIC = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
]

CC_MESSAGES_WITH_TOOLS = [
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

# ── Bedrock Converse response samples ────────────────────────────────────

BEDROCK_RESPONSE_TEXT = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello from Bedrock"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 25, "outputTokens": 10},
    "metrics": {"latencyMs": 500},
}

BEDROCK_RESPONSE_TOOL_USE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "Let me check."},
                {
                    "toolUse": {
                        "toolUseId": "toolu_01ABC",
                        "name": "get_weather",
                        "input": {"city": "London"},
                    }
                },
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 50, "outputTokens": 30},
}

BEDROCK_RESPONSE_MAX_TOKENS = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Cut off..."}],
        }
    },
    "stopReason": "max_tokens",
    "usage": {"inputTokens": 10, "outputTokens": 100},
}


# ── Properties ───────────────────────────────────────────────────────────


class TestBedrockAdapterProperties:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "bedrock"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"

    def test_upstream_path(self):
        # upstream_path not used for custom transport, but should exist
        assert self.adapter.upstream_path == "/chat/completions"

    def test_use_custom_transport(self):
        assert self.adapter.use_custom_transport is True


# ── CC → Bedrock request translation ─────────────────────────────────────


class TestBedrockTranslateToUpstream:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_extracts_system_message(self):
        cc = {"model": "anthropic.claude-sonnet-4-20250514", "messages": CC_MESSAGES_BASIC, "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assert result["system"] == [{"text": "You are helpful."}]
        assert all(m["role"] != "system" for m in result["messages"])

    def test_no_system_message(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "system" not in result

    def test_user_content_string_becomes_text_block(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0] == {"role": "user", "content": [{"text": "Hello"}]}

    def test_user_content_parts_list_flattens_to_its_text(self):
        """A parts-list user turn degrades to the single text block every pre-KBR-222 turn had.

        Hop 1 (KBR-222) ships image-bearing turns as CC content parts. Converse
        has no mapping for them here (KBR-223's territory), so the list must
        flatten — forwarding it would fail boto3 validation on every
        Messages-route image turn, a regression this fix would otherwise
        manufacture.
        """
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aWNvbg=="},
                        },
                    ],
                }
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0] == {"role": "user", "content": [{"text": "look at this"}]}

    def test_user_content_image_only_list_flattens_to_the_pre_fix_empty_text(self):
        """An image-only list flattens to ``[{"text": ""}]`` — the pre-fix shape for that turn."""
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aWNvbg=="},
                        }
                    ],
                }
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0] == {"role": "user", "content": [{"text": ""}]}

    def test_max_tokens_inference_config(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 2048,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["inferenceConfig"]["maxTokens"] == 2048

    def test_default_max_tokens(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["inferenceConfig"]["maxTokens"] == 4096

    def test_temperature_inference_config(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["inferenceConfig"]["temperature"] == 0.5

    def test_top_p_inference_config(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "top_p": 0.9,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["inferenceConfig"]["topP"] == 0.9

    def test_tools_translated_to_tool_config(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": CC_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "toolConfig" in result
        tool = result["toolConfig"]["tools"][0]
        assert "toolSpec" in tool
        assert tool["toolSpec"]["name"] == "get_weather"
        assert "inputSchema" in tool["toolSpec"]
        assert "json" in tool["toolSpec"]["inputSchema"]

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [
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
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        tool_use_blocks = [b for b in assistant_msg["content"] if "toolUse" in b]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["toolUse"]["name"] == "get_weather"
        assert tool_use_blocks[0]["toolUse"]["input"] == {"city": "London"}
        assert not any("reasoningContent" in b for b in assistant_msg["content"])

    def test_assistant_reasoning_content_becomes_reasoning_block(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should check the weather tool first.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        }
                    ],
                },
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][1]
        assert assistant_msg["content"][0] == {
            "reasoningContent": {"text": "I should check the weather tool first."},
        }
        assert "toolUse" in assistant_msg["content"][1]

    def test_tool_result_becomes_tool_result_block(self):
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": CC_MESSAGES_WITH_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "user"
        assert "toolResult" in tool_msg["content"][0]
        assert tool_msg["content"][0]["toolResult"]["toolUseId"] == "call_abc"
        assert tool_msg["content"][0]["toolResult"]["status"] == "success"

    def test_model_in_body(self):
        cc = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["modelId"] == "us.anthropic.claude-sonnet-4-20250514"


# ── Bedrock → CC response translation ────────────────────────────────────


class TestBedrockStopSequences:
    """KBR-178: the CC `stop` reaches Converse as `inferenceConfig.stopSequences`."""

    def setup_method(self):
        self.adapter = BedrockAdapter()

    def _cc(self, **extra):
        """Build a minimal CC request, plus whatever the case under test adds."""
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        cc.update(extra)
        return cc

    def test_stop_mapped_to_inference_config_stop_sequences(self):
        """Converse spells it `stopSequences`, inside `inferenceConfig`."""
        result = self.adapter.translate_to_upstream(self._cc(stop=["A"]))
        assert result["inferenceConfig"]["stopSequences"] == ["A"]
        assert "stop" not in result

    def test_no_stop_means_no_stop_sequences(self):
        """No `stop` invents no `stopSequences`."""
        result = self.adapter.translate_to_upstream(self._cc())
        assert "stopSequences" not in result["inferenceConfig"]

    def test_null_stop_is_omitted(self):
        """`stop: null` is a legal CC value and must not be forwarded — see D6."""
        result = self.adapter.translate_to_upstream(self._cc(stop=None))
        assert "stopSequences" not in result["inferenceConfig"]

    def test_empty_stop_is_omitted(self):
        """An empty stop list is semantically void — see D6."""
        result = self.adapter.translate_to_upstream(self._cc(stop=[]))
        assert "stopSequences" not in result["inferenceConfig"]

    def test_top_k_never_reaches_the_converse_body(self):
        """Converse's InferenceConfiguration has no `topK` member — see D4.

        The botocore service model declares exactly ``maxTokens``,
        ``temperature``, ``topP`` and ``stopSequences``.  Converse accepts
        ``top_k`` only under ``additionalModelRequestFields``, which this
        change does not open.
        """
        result = self.adapter.translate_to_upstream(self._cc(top_k=40, _top_k=40))
        assert "topK" not in result["inferenceConfig"]
        assert "top_k" not in result
        assert "_top_k" not in result


class TestBedrockToolChoice:
    """KBR-214: the CC ``tool_choice`` reaches Converse's ``toolConfig.toolChoice``."""

    def setup_method(self):
        self.adapter = BedrockAdapter()

    def _cc(self, **extra):
        """Build a minimal CC request with one tool, plus the case's fields."""
        cc = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        }
        cc.update(extra)
        return cc

    @pytest.mark.parametrize(
        ("cc", "converse"),
        [
            ("required", {"any": {}}),
            ({"type": "function", "function": {"name": "get_weather"}}, {"tool": {"name": "get_weather"}}),
            ("auto", {"auto": {}}),
        ],
        ids=["required", "named", "auto"],
    )
    def test_tool_choice_value_is_translated(self, cc, converse):
        """The two forcing values stop being downgraded to ``auto`` (R7)."""
        result = self.adapter.translate_to_upstream(self._cc(tool_choice=cc))
        assert result["toolConfig"]["toolChoice"] == converse
        assert "tool_choice" not in result

    @pytest.mark.parametrize(
        "value",
        ["none", None, "bogus", {"type": "function", "function": {}}],
        ids=["none", "null", "unrecognised", "named-without-name"],
    )
    def test_values_converse_cannot_express_keep_todays_auto(self, value):
        """Converse's ``ToolChoice`` union has no ``none``, so today's value stands (D7, G33)."""
        result = self.adapter.translate_to_upstream(self._cc(tool_choice=value))
        assert result["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_no_tool_choice_keeps_todays_auto(self):
        """No CC ``tool_choice`` leaves the existing default untouched (R9)."""
        result = self.adapter.translate_to_upstream(self._cc())
        assert result["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_no_tools_means_no_tool_config(self):
        """A choice without tools builds no ``toolConfig``, exactly as before (R7)."""
        cc = self._cc(tool_choice="required")
        del cc["tools"]
        result = self.adapter.translate_to_upstream(cc)
        assert "toolConfig" not in result

    def test_parallel_tool_calls_does_not_reach_the_converse_body(self):
        """Converse has no parallel-tool-use field; nothing is invented (G32)."""
        result = self.adapter.translate_to_upstream(self._cc(tool_choice="required", parallel_tool_calls=False))
        assert "parallel_tool_calls" not in result
        assert set(result["toolConfig"]) == {"tools", "toolChoice"}


class TestBedrockTranslateFromUpstream:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_text_response(self):
        result = self.adapter.translate_from_upstream(BEDROCK_RESPONSE_TEXT)
        assert result["choices"][0]["message"]["content"] == "Hello from Bedrock"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self):
        result = self.adapter.translate_from_upstream(BEDROCK_RESPONSE_TEXT)
        assert result["usage"]["prompt_tokens"] == 25
        assert result["usage"]["completion_tokens"] == 10
        assert result["usage"]["total_tokens"] == 35

    def test_tool_use_response(self):
        result = self.adapter.translate_from_upstream(BEDROCK_RESPONSE_TOOL_USE)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Let me check."
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "London"}'
        assert tc["id"] == "toolu_01ABC"

    def test_max_tokens_response(self):
        result = self.adapter.translate_from_upstream(BEDROCK_RESPONSE_MAX_TOKENS)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_model_passthrough(self):
        result = self.adapter.translate_from_upstream(BEDROCK_RESPONSE_TEXT)
        assert "model" in result


# ── Credential resolution ────────────────────────────────────────────────


class TestBedrockCredentialResolution:
    def test_parse_aws_credentials(self):
        """Colon-separated key:secret format."""
        adapter = BedrockAdapter()
        access_key, secret_key = adapter.parse_aws_credentials(
            "AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        assert access_key == "AKIAIOSFODNN7EXAMPLE"
        assert secret_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

    def test_parse_aws_credentials_with_session_token(self):
        """Colon-separated key:secret:token format."""
        adapter = BedrockAdapter()
        parts = adapter.parse_aws_credentials("AKID:SECRET:TOKEN123")
        assert parts == ("AKID", "SECRET", "TOKEN123")

    def test_parse_aws_credentials_invalid_format(self):
        adapter = BedrockAdapter()
        with pytest.raises(ProviderError, match="Invalid AWS credentials"):
            adapter.parse_aws_credentials("justakey")
        with pytest.raises(ProviderError, match="Invalid AWS credentials"):
            adapter.parse_aws_credentials(":secret")

    def test_is_sso_mode(self):
        adapter = BedrockAdapter()
        assert adapter.is_sso_mode("sso") is True
        assert adapter.is_sso_mode("") is True
        assert adapter.is_sso_mode("AKID:SECRET") is False

    def test_get_region_from_provider_config(self):
        adapter = BedrockAdapter()
        region = adapter.get_region({"region": "eu-west-1"})
        assert region == "eu-west-1"

    def test_get_region_default(self):
        adapter = BedrockAdapter()
        region = adapter.get_region({})
        assert region == "us-east-1"

    def test_get_profile_name(self):
        adapter = BedrockAdapter()
        name = adapter.get_profile_name({"profile_name": "my-sso-profile"})
        assert name == "my-sso-profile"

    def test_get_profile_name_default(self):
        adapter = BedrockAdapter()
        name = adapter.get_profile_name({})
        assert name is None


# ── Model name normalization ─────────────────────────────────────────────


class TestBedrockNormalizeModelName:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_strips_prefix(self):
        """Bedrock adapter strips provider prefix."""
        assert (
            self.adapter.normalize_model_name("bedrock/anthropic.claude-sonnet-4-20250514")
            == "anthropic.claude-sonnet-4-20250514"
        )
        assert (
            self.adapter.normalize_model_name("bedrock/us.anthropic.claude-sonnet-4-20250514")
            == "us.anthropic.claude-sonnet-4-20250514"
        )

    def test_no_prefix(self):
        """Model names without prefix pass through."""
        assert (
            self.adapter.normalize_model_name("anthropic.claude-sonnet-4-20250514")
            == "anthropic.claude-sonnet-4-20250514"
        )
        assert (
            self.adapter.normalize_model_name("us.anthropic.claude-sonnet-4-20250514")
            == "us.anthropic.claude-sonnet-4-20250514"
        )


# ── Error mapping ────────────────────────────────────────────────────────


class TestBedrockMapError:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_400_error(self):
        exc = self.adapter.map_error(400, {"message": "validation error"})
        assert "400" in str(exc)

    def test_403_error(self):
        exc = self.adapter.map_error(403, {"message": "access denied"})
        assert "403" in str(exc)

    def test_429_error(self):
        exc = self.adapter.map_error(429, {"message": "throttled"})
        assert "429" in str(exc)

    def test_500_error(self):
        exc = self.adapter.map_error(500, {"message": "internal error"})
        assert "500" in str(exc)


# ── build_request / parse_response (standard adapter interface) ──────────


class TestBedrockBuildRequest:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_build_request_basic(self):
        result = self.adapter.build_request(
            model="anthropic.claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        assert result["model"] == "anthropic.claude-sonnet-4-20250514"
        assert result["stream"] is False


class TestBedrockParseResponse:
    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_parse_cc_response(self):
        cc_resp = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.adapter.parse_response(cc_resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"


# ── boto3 transport (mocked) ─────────────────────────────────────────────


class TestBedrockMakeRequest:
    """Test make_request() with mocked boto3 client."""

    @pytest.mark.asyncio
    async def test_non_streaming_call(self):
        adapter = BedrockAdapter()
        cc_request = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        mock_client = MagicMock()
        mock_client.converse.return_value = BEDROCK_RESPONSE_TEXT

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            result = await adapter.make_request(cc_request)

        assert result["choices"][0]["message"]["content"] == "Hello from Bedrock"
        assert result["choices"][0]["finish_reason"] == "stop"
        mock_client.converse.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_model_id(self):
        adapter = BedrockAdapter()
        cc_request = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        mock_client = MagicMock()
        mock_client.converse.return_value = BEDROCK_RESPONSE_TEXT

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            await adapter.make_request(cc_request)

        call_kwargs = mock_client.converse.call_args
        assert call_kwargs[1]["modelId"] == "us.anthropic.claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_bedrock_error_raises(self):
        adapter = BedrockAdapter()
        cc_request = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        mock_client = MagicMock()
        mock_client.converse.side_effect = Exception("ThrottlingException")

        with (
            patch.object(adapter, "_get_boto3_client", return_value=mock_client),
            pytest.raises(Exception, match="ThrottlingException"),
        ):
            await adapter.make_request(cc_request)


class TestBedrockStreamRequest:
    """Test stream_request() with mocked boto3 client."""

    @pytest.mark.asyncio
    async def test_streaming_yields_sse_events(self):
        adapter = BedrockAdapter()
        cc_request = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        # Mock the converse_stream response
        stream_events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " there"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ]

        mock_response = {"stream": iter(stream_events)}
        mock_client = MagicMock()
        mock_client.converse_stream.return_value = mock_response

        chunks: list[bytes] = []

        async def write_cb(data: bytes):
            chunks.append(data)

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            await adapter.stream_request(cc_request, write_cb)

        # Should have emitted SSE chunks
        assert len(chunks) > 0
        # First chunk should have role
        first = chunks[0].decode()
        assert "assistant" in first
        # Should have text content
        combined = b"".join(chunks).decode()
        assert "Hi" in combined
        assert "there" in combined
        # Should have [DONE]
        assert "[DONE]" in combined

    @pytest.mark.asyncio
    async def test_streaming_tool_use(self):
        adapter = BedrockAdapter()
        cc_request = {
            "model": "anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
        }

        stream_events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": ""}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Checking."}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "toolu_123", "name": "get_weather"}},
                }
            },
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city":'}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"London"}'}}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 20, "outputTokens": 15}}},
        ]

        mock_response = {"stream": iter(stream_events)}
        mock_client = MagicMock()
        mock_client.converse_stream.return_value = mock_response

        chunks: list[bytes] = []

        async def write_cb(data: bytes):
            chunks.append(data)

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            await adapter.stream_request(cc_request, write_cb)

        combined = b"".join(chunks).decode()
        # Should have text content
        assert "Checking." in combined
        # Should have tool call with correct name and arguments
        assert "get_weather" in combined
        assert "London" in combined
        # Should have tool_calls finish reason
        assert "tool_calls" in combined
        assert "[DONE]" in combined

    @pytest.mark.asyncio
    async def test_stream_uses_model_id(self):
        adapter = BedrockAdapter()
        cc_request = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        mock_response = {"stream": iter([])}
        mock_client = MagicMock()
        mock_client.converse_stream.return_value = mock_response

        async def noop(data: bytes):
            pass

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client):
            await adapter.stream_request(cc_request, noop)

        call_kwargs = mock_client.converse_stream.call_args
        assert call_kwargs[1]["modelId"] == "us.anthropic.claude-sonnet-4-20250514"


class TestThinkingEnabledToolCallGap:
    """Regression tests for: thinking enabled but assistant tool-call messages lack reasoning_content.

    When _thinking_enabled is True, the upstream Bedrock API requires every assistant message
    to contain a reasoningContent block.  Historical tool-call messages that predate the
    thinking turn will not have reasoning_content, so the bridge must inject an empty one.
    """

    def setup_method(self):
        self.adapter = BedrockAdapter()

    def test_thinking_enabled_tool_call_without_reasoning_gets_empty_reasoning(self):
        cc = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [
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
                {"role": "tool", "tool_call_id": "call_abc", "content": "15°C"},
                {"role": "user", "content": "Thanks"},
                {
                    "role": "assistant",
                    "content": "You're welcome!",
                    "reasoning_content": "No further tool use needed.",
                },
            ],
            "stream": False,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)

        assistant_with_tools = result["messages"][1]
        assert assistant_with_tools["role"] == "assistant"
        reasoning_blocks = [b for b in assistant_with_tools["content"] if "reasoningContent" in b]
        assert len(reasoning_blocks) == 1, "Should inject empty reasoningContent when thinking enabled"
        assert reasoning_blocks[0]["reasoningContent"]["text"] == ""

    def test_thinking_enabled_text_only_assistant_without_reasoning_gets_empty_reasoning(self):
        cc = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Thanks"},
                {
                    "role": "assistant",
                    "content": "You're welcome!",
                    "reasoning_content": "Simple greeting exchange.",
                },
            ],
            "stream": False,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)
        first_assistant = result["messages"][1]
        reasoning_blocks = [b for b in first_assistant["content"] if "reasoningContent" in b]
        assert len(reasoning_blocks) == 1, "Should inject empty reasoningContent for text-only assistant"
        assert reasoning_blocks[0]["reasoningContent"]["text"] == ""

    def test_thinking_not_enabled_no_injection(self):
        cc = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [
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
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][1]
        reasoning_blocks = [b for b in assistant_msg["content"] if "reasoningContent" in b]
        assert len(reasoning_blocks) == 0, "No reasoningContent injected when thinking not enabled"

    def test_thinking_enabled_existing_reasoning_preserved(self):
        cc = {
            "model": "us.anthropic.claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Existing reasoning here.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                        }
                    ],
                },
            ],
            "stream": False,
            "_thinking_enabled": True,
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][0]
        reasoning_blocks = [b for b in assistant_msg["content"] if "reasoningContent" in b]
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0]["reasoningContent"]["text"] == "Existing reasoning here."


class TestBedrockStreamErrorEvents:
    """F17: Bedrock converse_stream error events must raise ProviderError."""

    def _make_adapter(self):
        return BedrockAdapter()

    def test_internal_server_exception_raises_provider_error(self):
        adapter = self._make_adapter()
        event = {"internalServerException": {"message": "Internal failure"}}
        with pytest.raises(ProviderError, match="Internal failure"):
            adapter._translate_stream_event(event, "id", {})

    def test_model_stream_error_exception_raises_provider_error(self):
        adapter = self._make_adapter()
        event = {"modelStreamErrorException": {"message": "Model overloaded"}}
        with pytest.raises(ProviderError, match="Model overloaded"):
            adapter._translate_stream_event(event, "id", {})

    def test_throttling_exception_raises_provider_error(self):
        adapter = self._make_adapter()
        event = {"throttlingException": {"message": "Rate exceeded"}}
        with pytest.raises(ProviderError, match="Rate exceeded"):
            adapter._translate_stream_event(event, "id", {})

    def test_validation_exception_raises_provider_error(self):
        adapter = self._make_adapter()
        event = {"validationException": {"message": "Invalid request"}}
        with pytest.raises(ProviderError, match="Invalid request"):
            adapter._translate_stream_event(event, "id", {})

    def test_service_unavailable_exception_raises_provider_error(self):
        adapter = self._make_adapter()
        event = {"serviceUnavailableException": {"message": "Service unavailable"}}
        with pytest.raises(ProviderError, match="Service unavailable"):
            adapter._translate_stream_event(event, "id", {})

    def test_unknown_event_type_returns_empty(self):
        """Unknown event types (not errors) should return empty chunks."""
        adapter = self._make_adapter()
        event = {"unknownFutureEvent": {"data": "something"}}
        chunks = adapter._translate_stream_event(event, "id", {})
        assert chunks == []


class TestItCachesNoTransport:
    """KBR-190 — why bedrock needs no ``aclose`` override.

    The other two custom-transport adapters cache a client on the instance and
    leak it when the bridge stops.  This one does not, and that is the reason it
    is exempt from the sweep rather than an oversight.  If caching is ever added,
    this fails and sends the author to ``OpenAISubscriptionAdapter.aclose``.
    """

    def test_get_boto3_client_returns_a_new_client_each_call(self):
        adapter = BedrockAdapter()
        args = ("AKIAEXAMPLE:secret-key", {"region": "us-east-1"})

        first = adapter._get_boto3_client(*args)
        second = adapter._get_boto3_client(*args)

        assert first is not second

    def test_no_client_is_stored_on_the_instance(self):
        """The identity check above cannot see a client kept but not reused.

        Both calls pass identical arguments, so an argument-keyed cache fails
        that one too.  This catches the other shape — a client assigned to the
        adapter and returned fresh each time — which is still state a stopping
        bridge would have to release.
        """
        adapter = BedrockAdapter()
        before = set(vars(adapter))

        adapter._get_boto3_client("AKIAEXAMPLE:secret-key", {"region": "us-east-1"})

        assert set(vars(adapter)) == before


class TestTheEndpointUrlSeam:
    """KBR-42 — ``provider_config["endpoint_url"]`` is the test-harness seam.

    T-B3's transport points the botocore client at the local recorder by
    setting this key.  Production profiles do not carry it (so the kwarg
    is opt-in), and a regression that dropped the new key from
    ``_get_boto3_client`` while keeping the rest of the method intact would
    still be caught by the harness integration — but only at the **flow**
    level, not the **kwarg** level.  These tests pin the kwarg at the layer
    where the production change lives, so a future refactor cannot silently
    regress it.
    """

    def test_endpoint_url_is_passed_to_the_boto3_client_when_set(self) -> None:
        adapter = BedrockAdapter()
        sentinel = MagicMock()
        captured_kwargs: dict = {}

        def _capture(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            return sentinel

        with patch("boto3.Session") as session_cls:
            session_cls.return_value.client.side_effect = _capture
            adapter._get_boto3_client(
                "AKIAEXAMPLE:secret",
                {"endpoint_url": "http://recorder:9", "region": "us-east-1"},
            )

        assert captured_kwargs.get("endpoint_url") == "http://recorder:9", (
            "the test-harness seam was not forwarded to the boto3 client; "
            "the botocore endpoint-override recorder (T-B3) cannot point at the loopback"
        )

    def test_endpoint_url_is_omitted_when_provider_config_lacks_it(self) -> None:
        """Profiles without the key must behave as before KBR-42.

        Production profiles do not carry ``endpoint_url``; passing it
        through to ``session.client(..., endpoint_url=None)`` raises on some
        botocore versions and is silently ignored on others, so the seam
        must consume the key only when truthy.
        """
        adapter = BedrockAdapter()
        captured_kwargs: dict = {}

        def _capture(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch("boto3.Session") as session_cls:
            session_cls.return_value.client.side_effect = _capture
            adapter._get_boto3_client("AKIAEXAMPLE:secret", {"region": "us-east-1"})

        assert "endpoint_url" not in captured_kwargs, (
            "endpoint_url must not be passed when the profile does not set it; "
            "production profiles do not, and some botocore versions raise on None"
        )

    def test_endpoint_url_reaches_the_sso_branch_when_set(self) -> None:
        """The SSO half of the ``if/else`` shares the same ``client_kwargs``.

        The two tests above drive the credentials branch
        (``parse_aws_credentials`` → ``boto3.Session(aws_access_key_id=…,
        …)``); this one drives the SSO branch
        (``boto3.Session(profile_name=…, region_name=…)``), which shares
        the same ``client_kwargs`` the seam mutates. ``endpoint_url``
        sits **above** the ``if/else`` today, so both branches see it —
        a refactor that moved the lines into one branch only would pass
        the credentials tests while silently breaking SSO profiles,
        which is why the SSO case is pinned too.
        """
        adapter = BedrockAdapter()
        captured_kwargs: dict = {}
        captured_session_kwargs: dict = {}

        def _capture(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            return MagicMock()

        session_mock = MagicMock()
        session_mock.client.side_effect = _capture

        def _session_capture(*args: object, **kwargs: object) -> MagicMock:
            captured_session_kwargs.update(kwargs)
            return session_mock

        with patch("boto3.Session") as session_cls:
            session_cls.side_effect = _session_capture
            adapter._get_boto3_client(
                "sso",
                {"endpoint_url": "http://recorder:9", "region": "us-east-1", "profile_name": "harness-profile"},
            )

        assert captured_session_kwargs.get("profile_name") == "harness-profile", (
            "the SSO branch was not taken; if the if/else moved to "
            "is_sso_mode=False for resolved_key='sso', the credentials branch's "
            "parse_aws_credentials would raise ProviderError on the colonless key "
            "before this assertion ran, surfacing the regression even louder than this assert"
        )
        assert captured_kwargs.get("endpoint_url") == "http://recorder:9", (
            "the test-harness seam did not reach the SSO branch's boto3 client; "
            "the seam sits inside one arm of the if/else and the other arm lost it"
        )

    def test_the_harness_key_is_not_an_sso_marker(self) -> None:
        """The harness key routes through the credentials branch.

        ``HarnessBedrockAdapter``'s override of ``parse_aws_credentials``
        resolves the harness key to the fake pair — but only when the key
        is **not** an SSO marker. ``is_sso_mode`` intercepts ``""`` and
        ``"sso"`` *before* ``parse_aws_credentials`` runs, so a harness
        key that matched either would silently fall into the SSO branch
        and use whatever ambient AWS credentials the test machine has.
        Pinning the harness key's non-membership here makes that
        brittleness a checkable claim rather than a docstring promise.
        """
        from kitty.providers.bedrock import BedrockAdapter

        adapter = BedrockAdapter()
        assert not adapter.is_sso_mode("harness-key"), (
            "the harness key matched an SSO marker; HarnessBedrockAdapter's "
            "parse_aws_credentials override would be silently bypassed and "
            "the test would use whatever ambient AWS credentials the machine has"
        )
