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
            "messages": CC_MESSAGES_WITH_TOOLS,
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        tool_use_blocks = [b for b in assistant_msg["content"] if "toolUse" in b]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["toolUse"]["name"] == "get_weather"
        assert tool_use_blocks[0]["toolUse"]["input"] == {"city": "London"}

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

    def test_returns_unchanged(self):
        assert (
            self.adapter.normalize_model_name("anthropic.claude-sonnet-4-20250514")
            == "anthropic.claude-sonnet-4-20250514"
        )

    def test_cross_region_prefix_preserved(self):
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

        with patch.object(adapter, "_get_boto3_client", return_value=mock_client), pytest.raises(
            Exception, match="ThrottlingException"
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
