"""Tests for OllamaCloudAdapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kitty.providers.base import ProviderError
from kitty.providers.ollama_cloud import OllamaCloudAdapter

# ── Properties ───────────────────────────────────────────────────────────────


class TestOllamaCloudAdapterProperties:
    """Test basic adapter properties."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_provider_type(self):
        assert self.adapter.provider_type == "ollama_cloud"

    def test_default_base_url(self):
        assert self.adapter.default_base_url == "https://ollama.com"

    def test_upstream_path(self):
        assert self.adapter.upstream_path == "/api/chat"

    def test_use_custom_transport(self):
        assert self.adapter.use_custom_transport is True


# ── Headers ──────────────────────────────────────────────────────────────────


class TestOllamaCloudHeaders:
    """Test upstream header construction."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_bearer_auth(self):
        headers = self.adapter.build_upstream_headers("sk-test-key")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_different_keys(self):
        h1 = self.adapter.build_upstream_headers("key-a")
        h2 = self.adapter.build_upstream_headers("key-b")
        assert h1["Authorization"] != h2["Authorization"]


# ── Model normalization ─────────────────────────────────────────────────────


class TestOllamaCloudNormalizeModelName:
    """Test model name normalization."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_strips_ollama_cloud_prefix(self):
        assert self.adapter.normalize_model_name("ollama_cloud/gpt-oss:120b") == "gpt-oss:120b"

    def test_strips_ollama_prefix(self):
        assert self.adapter.normalize_model_name("ollama/gpt-oss:120b") == "gpt-oss:120b"

    def test_no_prefix_passthrough(self):
        assert self.adapter.normalize_model_name("gpt-oss:120b") == "gpt-oss:120b"

    def test_slash_model_without_known_prefix_preserved(self):
        assert self.adapter.normalize_model_name("library/qwen:7b") == "library/qwen:7b"

    def test_colon_model_name_preserved(self):
        assert self.adapter.normalize_model_name("qwen3:8b") == "qwen3:8b"


# ── translate_to_upstream ───────────────────────────────────────────────────


class TestOllamaCloudTranslateToUpstream:
    """Test CC → Ollama request translation."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_basic_request(self):
        cc = {"model": "gpt-oss:120b", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        result = self.adapter.translate_to_upstream(cc)
        assert result["model"] == "gpt-oss:120b"
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        assert result["stream"] is False

    def test_system_message_extracted(self):
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            "stream": False,
        }
        result = self.adapter.translate_to_upstream(cc)
        assert any(m["role"] == "system" for m in result["messages"])

    def test_strips_internal_keys(self):
        cc = {
            "model": "gpt-oss:120b",
            "messages": [],
            "_resolved_key": "secret",
            "_provider_config": {},
            "_thinking_enabled": True,
            "_reasoning_effort": "high",
        }
        result = self.adapter.translate_to_upstream(cc)
        assert "_resolved_key" not in result
        assert "_provider_config" not in result
        assert "_thinking_enabled" not in result
        assert "_reasoning_effort" not in result

    def test_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        cc = {"model": "gpt-oss:120b", "messages": [], "tools": tools}
        result = self.adapter.translate_to_upstream(cc)
        assert result["tools"] == tools

    def test_tool_result_message_translated(self):
        """CC tool results use tool_call_id; Ollama uses tool_name."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "tool", "tool_call_id": "call_123", "name": "get_weather", "content": "sunny"},
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        tool_msg = result["messages"][0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_name"] == "get_weather"
        assert tool_msg["content"] == "sunny"
        assert "tool_call_id" not in tool_msg

    def test_assistant_tool_calls_translated(self):
        """CC assistant tool_calls: arguments JSON string → Ollama dict."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                        }
                    ],
                }
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        assistant_msg = result["messages"][0]
        assert assistant_msg["role"] == "assistant"
        # Arguments should be converted from JSON string to dict for Ollama
        tc = assistant_msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"city": "NYC"}

    def test_assistant_tool_calls_dict_args_preserved(self):
        """If arguments are already a dict, they pass through unchanged."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": {"expr": "1+1"}},
                        }
                    ],
                }
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"expr": "1+1"}

    def test_options_from_temperature(self):
        """CC temperature maps to Ollama options.temperature."""
        cc = {"model": "gpt-oss:120b", "messages": [], "temperature": 0.7}
        result = self.adapter.translate_to_upstream(cc)
        assert result["options"]["temperature"] == 0.7

    def test_options_empty_when_no_extras(self):
        cc = {"model": "gpt-oss:120b", "messages": []}
        result = self.adapter.translate_to_upstream(cc)
        assert "options" not in result

    def test_content_blocks_flattened_to_string(self):
        """CC content as list of text blocks → Ollama plain string."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]},
            ],
        }
        result = self.adapter.translate_to_upstream(cc)
        assert result["messages"][0]["content"] == "hello\nworld"

    def test_none_content_becomes_empty_string(self):
        """CC content None → Ollama empty string (not None)."""
        cc = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "assistant", "content": None, "tool_calls": []}],
        }
        result = self.adapter.translate_to_upstream(cc)
        # content is set only if not None, so it should be absent
        assert "content" not in result["messages"][0] or result["messages"][0].get("content") == ""


# ── translate_from_upstream ─────────────────────────────────────────────────


class TestOllamaCloudTranslateFromUpstream:
    """Test Ollama response → CC response translation."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_text_response(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_usage_mapping(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        usage = result["usage"]
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_tool_calls_in_response(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                    }
                ],
            },
            "done": True,
            "done_reason": "stop",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_done_reason_length(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "cut off"},
            "done": True,
            "done_reason": "length",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_missing_usage_defaults_to_zero(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0

    def test_cc_response_structure(self):
        ollama_resp = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
        }
        result = self.adapter.translate_from_upstream(ollama_resp)
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-oss:120b"
        assert "id" in result


# ── map_error ────────────────────────────────────────────────────────────────


class TestOllamaCloudMapError:
    """Test error mapping."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_dict_error(self):
        err = self.adapter.map_error(400, {"error": "model not found"})
        assert isinstance(err, ProviderError)
        assert "400" in str(err)
        assert "model not found" in str(err)

    def test_429_error(self):
        err = self.adapter.map_error(429, {"error": "too many requests"})
        assert isinstance(err, ProviderError)
        assert "429" in str(err)

    def test_500_error(self):
        err = self.adapter.map_error(500, {"error": "internal error"})
        assert isinstance(err, ProviderError)
        assert "500" in str(err)

    def test_non_dict_body(self):
        err = self.adapter.map_error(400, "bad request")
        assert isinstance(err, ProviderError)
        assert "bad request" in str(err)

    def test_nested_error_object(self):
        err = self.adapter.map_error(401, {"error": {"message": "invalid API key"}})
        assert isinstance(err, ProviderError)
        assert "invalid API key" in str(err)

    def test_http_status_set(self):
        err = self.adapter.map_error(429, {"error": "rate limited"})
        assert err.http_status == 429


# ── build_request / parse_response ──────────────────────────────────────────


class TestOllamaCloudBuildRequest:
    """Test build_request returns CC-format dict."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_basic(self):
        req = self.adapter.build_request("gpt-oss:120b", [{"role": "user", "content": "hi"}])
        assert req["model"] == "gpt-oss:120b"
        assert req["messages"] == [{"role": "user", "content": "hi"}]

    def test_with_stream(self):
        req = self.adapter.build_request("gpt-oss:120b", [], stream=True)
        assert req["stream"] is True

    def test_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = self.adapter.build_request("gpt-oss:120b", [], tools=tools)
        assert req["tools"] == tools


class TestOllamaCloudParseResponse:
    """Test parse_response returns normalized CC dict."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_basic(self):
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = self.adapter.parse_response(resp)
        assert result["content"] == "hi"
        assert result["finish_reason"] == "stop"


# ── make_request (custom transport) ──────────────────────────────────────────


class TestOllamaCloudMakeRequest:
    """Test non-streaming custom transport."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    @pytest.mark.asyncio
    async def test_non_streaming_call(self):
        """make_request translates CC → Ollama, calls HTTP, translates back."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }

        ollama_response = {
            "model": "gpt-oss:120b",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=ollama_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with patch.object(self.adapter, "_get_session", return_value=mock_session):
            result = await self.adapter.make_request(cc_request)

        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_upstream_error_raises(self):
        """make_request raises ProviderError on upstream failure."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [],
            "_resolved_key": "bad-key",
            "_provider_config": {},
        }

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={"error": "invalid API key"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch.object(self.adapter, "_get_session", return_value=mock_session),
            pytest.raises(ProviderError, match="401"),
        ):
            await self.adapter.make_request(cc_request)


# ── stream_request (custom transport) ────────────────────────────────────────


class TestOllamaCloudStreamRequest:
    """Test streaming custom transport: NDJSON → CC SSE."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    @pytest.mark.asyncio
    async def test_streaming_yields_sse_events(self):
        """stream_request reads NDJSON lines and writes CC SSE chunks."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }

        ndjson_lines = [
            b'{"model":"gpt-oss:120b","message":{"role":"assistant","content":"Hel"},"done":false}\n',
            b'{"model":"gpt-oss:120b","message":{"role":"assistant","content":"lo!"},"done":false}\n',
            b'{"model":"gpt-oss:120b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}\n',
        ]

        async def mock_aiter():
            for line in ndjson_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_aiter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        written_chunks: list[bytes] = []

        async def mock_write(data: bytes):
            written_chunks.append(data)

        with patch.object(self.adapter, "_get_session", return_value=mock_session):
            await self.adapter.stream_request(cc_request, mock_write)

        # Should have written SSE chunks
        assert len(written_chunks) > 0

        # Parse written SSE data
        all_data = b"".join(written_chunks).decode()
        assert "data:" in all_data
        assert "[DONE]" in all_data

        # Verify text content made it through
        sse_events = [line for line in all_data.split("\n") if line.startswith("data:")]
        contents = []
        for event in sse_events:
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                continue
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                contents.append(delta["content"])
        assert "Hel" in contents
        assert "lo!" in contents

    @pytest.mark.asyncio
    async def test_streaming_upstream_error(self):
        """stream_request raises ProviderError on upstream HTTP error."""
        cc_request = {
            "model": "gpt-oss:120b",
            "messages": [],
            "_resolved_key": "key",
            "_provider_config": {},
        }

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"error": "internal server error"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch.object(self.adapter, "_get_session", return_value=mock_session),
            pytest.raises(ProviderError, match="500"),
        ):
            await self.adapter.stream_request(cc_request, AsyncMock())

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(self):
        """stream_request translates Ollama tool_calls in NDJSON chunks to CC SSE."""
        cc_request = {
            "model": "qwen3-coder-next",
            "messages": [{"role": "user", "content": "2+2"}],
            "stream": True,
            "_resolved_key": "test-key",
            "_provider_config": {},
        }

        ndjson_lines = [
            b'{"model":"qwen3-coder-next","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_abc","function":{"index":0,"name":"calc","arguments":{"expr":"2+2"}}}]},"done":false}\n',
            b'{"model":"qwen3-coder-next","message":{"role":"assistant","content":""},"done":false}\n',
            b'{"model":"qwen3-coder-next","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}\n',
        ]

        async def mock_aiter():
            for line in ndjson_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = mock_aiter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        written_chunks: list[bytes] = []

        async def mock_write(data: bytes):
            written_chunks.append(data)

        with patch.object(self.adapter, "_get_session", return_value=mock_session):
            await self.adapter.stream_request(cc_request, mock_write)

        all_data = b"".join(written_chunks).decode()
        # Verify tool_calls appear in SSE output
        sse_events = [line for line in all_data.split("\n") if line.startswith("data:")]
        found_tool_calls = False
        for event in sse_events:
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                continue
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "tool_calls" in delta:
                found_tool_calls = True
                assert delta["tool_calls"][0]["function"]["name"] == "calc"
        assert found_tool_calls, "Expected tool_calls in streaming SSE output"


# ── parse_stream_to_cc_response ──────────────────────────────────────────────


class TestOllamaCloudParseStream:
    """Test CC SSE stream → CC response dict parsing."""

    def setup_method(self):
        self.adapter = OllamaCloudAdapter()

    def test_text_stream(self):
        raw = (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{"content":"lo!"},"finish_reason":null}]}\n\n'
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"qwen3-coder-next",'
            b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        result = self.adapter.parse_stream_to_cc_response(raw)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["model"] == "qwen3-coder-next"

    def test_empty_stream(self):
        raw = b"data: [DONE]\n\n"
        result = self.adapter.parse_stream_to_cc_response(raw)
        assert result["choices"][0]["message"]["content"] is None

    def test_tool_calls_in_stream(self):
        raw = (
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"m",'
            b'"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function",'
            b'"function":{"name":"test","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
            b"data: [DONE]\n\n"
        )
        result = self.adapter.parse_stream_to_cc_response(raw)
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1


# ── Registry ─────────────────────────────────────────────────────────────────


class TestOllamaCloudRegistry:
    """Test provider registry integration."""

    def test_registered(self):
        from kitty.providers.registry import get_provider

        adapter = get_provider("ollama_cloud")
        assert isinstance(adapter, OllamaCloudAdapter)

    def test_registry_key(self):
        from kitty.providers.registry import _registry

        assert "ollama_cloud" in _registry
