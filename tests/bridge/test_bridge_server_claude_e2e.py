"""End-to-end tests simulating Claude Code requests through the Messages API bridge.

These tests send realistic Anthropic Messages API payloads (exactly as Claude Code
would) to the bridge, mock the upstream Chat Completions provider, and verify the
full round-trip translation is correct.
"""

import asyncio
import json

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.profiles.schema import Profile
from kitty.providers.base import ProviderAdapter
from kitty.types import BridgeProtocol


# ── Stub adapters ────────────────────────────────────────────────────────────


class _StubLauncher(LauncherAdapter):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def binary_name(self) -> str:
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile: Profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


class _StubProvider(ProviderAdapter):
    @property
    def provider_type(self) -> str:
        return "stub"

    @property
    def default_base_url(self) -> str:
        return "https://api.example.com/v1"

    def build_request(self, model: str, messages: list[dict], **kwargs) -> dict:
        request = {"model": model, "messages": messages, "stream": kwargs.get("stream", False)}
        for key in ("tools", "temperature", "top_p", "max_tokens"):
            if key in kwargs and kwargs[key] is not None:
                request[key] = kwargs[key]
        return request

    def parse_response(self, response_data: dict) -> dict:
        return response_data

    def map_error(self, status_code: int, body: dict) -> Exception:
        return Exception(f"Upstream error {status_code}: {body}")


# ── Upstream response fixtures ───────────────────────────────────────────────

UPSTREAM_TEXT_RESPONSE = {
    "id": "chatcmpl-e2e-text",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from the upstream provider!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
}

UPSTREAM_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-e2e-tool",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_e2e_001",
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
    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
}

UPSTREAM_TOOL_RESULT_RESPONSE = {
    "id": "chatcmpl-e2e-result",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "The weather in London is 15°C and sunny."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95},
}


# ── Realistic Claude Code request payloads ───────────────────────────────────


def _claude_code_simple_request():
    """Simulates Claude Code sending a simple user message."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "Hello, can you help me write a function?"},
        ],
    }


def _claude_code_large_request(content_size: int) -> dict:
    """Claude Code request with a large but valid JSON message body."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "x" * content_size},
        ],
    }


def _claude_code_request_with_system():
    """Claude Code with a system prompt (includes project context)."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": "You are Claude Code, an AI coding assistant. You help with software engineering tasks.",
        "messages": [
            {"role": "user", "content": "Create a hello world function in Python"},
        ],
    }


def _claude_code_tool_call_request():
    """Claude Code after receiving a tool call — sends tool_result back."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check the weather."},
                    {
                        "type": "tool_use",
                        "id": "toolu_e2e_001",
                        "name": "get_weather",
                        "input": {"city": "London"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_e2e_001",
                        "content": "15°C, sunny",
                    },
                ],
            },
        ],
    }


def _claude_code_request_with_tools():
    """Claude Code registering tools for the model to use."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": "Read the file main.py"},
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        ],
    }


def _claude_code_request_with_thinking():
    """Claude Code with extended thinking enabled (should be stripped)."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8096,
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": "Explain recursion"},
        ],
        "thinking": {"type": "enabled", "budget_tokens": 4000},
    }


def _claude_code_request_with_content_blocks():
    """Claude Code sending user message as content blocks (not plain string)."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is my code:"},
                    {"type": "text", "text": "def hello(): print('hi')"},
                ],
            },
        ],
    }


def _claude_code_multiple_tool_results_request():
    """Claude Code sending multiple tool_results back."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "Check both files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_e2e_001",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_e2e_002",
                        "name": "read_file",
                        "input": {"path": "b.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_e2e_001",
                        "content": "print('a')",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_e2e_002",
                        "content": "print('b')",
                    },
                ],
            },
        ],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_server() -> BridgeServer:
    return BridgeServer(_StubLauncher(), _StubProvider(), "test-api-key")


UPSTREAM_URL = "https://api.example.com/v1/chat/completions"


# ── E2E Tests ────────────────────────────────────────────────────────────────


class TestClaudeCodeSimpleRoundTrip:
    """Simulate Claude Code sending a basic text message and receiving a response."""

    @pytest.mark.asyncio
    async def test_simple_text_request_response(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_simple_request(),
                        headers={
                            "content-type": "application/json",
                            "anthropic-version": "2023-06-01",
                            "x-api-key": "test-api-key",
                        },
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()

                        # Verify Messages API response format
                        assert data["type"] == "message"
                        assert data["role"] == "assistant"
                        assert data["model"] == "test-model"
                        assert data["stop_reason"] == "end_turn"
                        assert data["stop_sequence"] is None
                        assert data["id"].startswith("msg_")

                        # Content blocks
                        assert len(data["content"]) == 1
                        assert data["content"][0]["type"] == "text"
                        assert data["content"][0]["text"] == "Hello from the upstream provider!"

                        # Usage
                        assert data["usage"]["input_tokens"] == 25
                        assert data["usage"]["output_tokens"] == 10
        finally:
            await server.stop_async()


class TestClaudeCodeWithSystemPrompt:
    """Verify system prompt is correctly translated to system message."""

    @pytest.mark.asyncio
    async def test_system_prompt_translated(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_system(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["type"] == "message"

                        # Verify the upstream received the system message
                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        assert len(request_list) == 1
                        sent_json = request_list[0].kwargs["json"]
                        assert sent_json["messages"][0] == {
                            "role": "system",
                            "content": "You are Claude Code, an AI coding assistant. You help with software engineering tasks.",
                        }
        finally:
            await server.stop_async()


class TestClaudeCodeToolUseRoundTrip:
    """Simulate a full tool use cycle: model calls tool -> user returns result -> model responds."""

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """Claude Code receives a tool_call response from the model."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TOOL_CALL_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_tools(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()

                        assert data["type"] == "message"
                        assert data["stop_reason"] == "tool_use"

                        # Should have text + tool_use content blocks
                        assert len(data["content"]) == 2
                        assert data["content"][0]["type"] == "text"
                        assert data["content"][0]["text"] == "Let me check the weather for you."

                        tool_block = data["content"][1]
                        assert tool_block["type"] == "tool_use"
                        assert tool_block["name"] == "get_weather"
                        assert tool_block["input"] == {"city": "London"}
                        assert tool_block["id"]  # Must have an id
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_tool_result_sent_correctly(self):
        """Claude Code sends tool_result back — verify it becomes a 'tool' role message."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TOOL_RESULT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_tool_call_request(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["type"] == "message"
                        assert data["stop_reason"] == "end_turn"

                        # Verify the upstream received the tool message
                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]

                        # Find the tool role message
                        tool_msgs = [m for m in sent_json["messages"] if m["role"] == "tool"]
                        assert len(tool_msgs) == 1
                        assert tool_msgs[0]["tool_call_id"] == "toolu_e2e_001"
                        assert tool_msgs[0]["content"] == "15°C, sunny"
        finally:
            await server.stop_async()


class TestClaudeCodeMultipleToolResults:
    """Verify multiple tool_results in a single user message produce multiple tool messages."""

    @pytest.mark.asyncio
    async def test_multiple_tool_results_produce_multiple_tool_messages(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_multiple_tool_results_request(),
                    ) as resp:
                        assert resp.status == 200

                        # Verify the upstream received two tool messages
                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]

                        tool_msgs = [m for m in sent_json["messages"] if m["role"] == "tool"]
                        assert len(tool_msgs) == 2
                        assert tool_msgs[0]["tool_call_id"] == "toolu_e2e_001"
                        assert tool_msgs[0]["content"] == "print('a')"
                        assert tool_msgs[1]["tool_call_id"] == "toolu_e2e_002"
                        assert tool_msgs[1]["content"] == "print('b')"
        finally:
            await server.stop_async()


class TestClaudeCodeThinkingStripped:
    """Verify thinking config is stripped before forwarding upstream."""

    @pytest.mark.asyncio
    async def test_thinking_not_forwarded_to_upstream(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_thinking(),
                    ) as resp:
                        assert resp.status == 200

                        # Verify thinking was NOT forwarded
                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]
                        assert "thinking" not in sent_json
        finally:
            await server.stop_async()


class TestClaudeCodeContentBlocks:
    """Verify content blocks (text arrays) are correctly concatenated."""

    @pytest.mark.asyncio
    async def test_text_blocks_concatenated(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_content_blocks(),
                    ) as resp:
                        assert resp.status == 200

                        # Verify the upstream received concatenated text
                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]
                        user_msg = sent_json["messages"][0]
                        assert user_msg["role"] == "user"
                        # Text blocks should be joined with newline
                        assert "Here is my code:" in user_msg["content"]
                        assert "def hello(): print('hi')" in user_msg["content"]
        finally:
            await server.stop_async()


class TestClaudeCodeStreamingRoundTrip:
    """Verify streaming SSE translation works end-to-end for Claude Code."""

    @pytest.mark.asyncio
    async def test_streaming_text_response(self):
        """Simulate Claude Code receiving a streaming text response."""
        server = _make_server()
        port = await server.start_async()
        try:
            upstream_chunks = [
                b'data: {"id":"chatcmpl-stream1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"model":"test-model","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
                b"data: [DONE]\n\n",
            ]

            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    UPSTREAM_URL,
                    body=b"".join(upstream_chunks),
                    headers={"Content-Type": "text/event-stream"},
                )

                request = _claude_code_simple_request()
                request["stream"] = True

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=request,
                    ) as resp:
                        assert resp.status == 200
                        assert resp.headers.get("Content-Type") == "text/event-stream"

                        body = await resp.read()
                        body_str = body.decode("utf-8")

                        # Verify SSE events
                        assert "event: message_start" in body_str
                        assert "event: content_block_start" in body_str
                        assert "event: content_block_delta" in body_str
                        assert "event: content_block_stop" in body_str
                        assert "event: message_delta" in body_str
                        assert "event: message_stop" in body_str

                        # Verify text deltas
                        assert '"Hello"' in body_str
                        assert '" world"' in body_str

                        # Verify stop_reason
                        assert "end_turn" in body_str
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_streaming_tool_call_response(self):
        """Simulate Claude Code receiving a streaming tool call response."""
        server = _make_server()
        port = await server.start_async()
        try:
            upstream_chunks = [
                b'data: {"id":"chatcmpl-stream2","choices":[{"index":0,"delta":{"content":"Let me"},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream2","choices":[{"index":0,"delta":{"content":" check."},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_001","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":"}}]},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"main.py\\"}"}}]},"finish_reason":null}],"model":"test-model"}\n\n',
                b'data: {"id":"chatcmpl-stream2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"model":"test-model","usage":{"prompt_tokens":50,"completion_tokens":30,"total_tokens":80}}\n\n',
                b"data: [DONE]\n\n",
            ]

            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    UPSTREAM_URL,
                    body=b"".join(upstream_chunks),
                    headers={"Content-Type": "text/event-stream"},
                )

                request = _claude_code_request_with_tools()
                request["stream"] = True

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=request,
                    ) as resp:
                        assert resp.status == 200
                        body = await resp.read()
                        body_str = body.decode("utf-8")

                        # Verify text content was streamed
                        assert '"Let me"' in body_str
                        assert '" check."' in body_str

                        # Verify tool_use content block was opened
                        assert "tool_use" in body_str
                        assert "read_file" in body_str

                        # Verify input_json_delta for arguments
                        assert "input_json_delta" in body_str

                        # Verify stop_reason = tool_use
                        assert "tool_use" in body_str

                        # Verify message lifecycle events
                        assert "event: message_start" in body_str
                        assert "event: message_stop" in body_str
        finally:
            await server.stop_async()


class TestClaudeCodeErrorHandling:
    """Verify error responses follow Anthropic Messages API error format."""

    @pytest.mark.asyncio
    async def test_upstream_401_returns_api_error(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(
                    UPSTREAM_URL,
                    status=401,
                    payload={"error": {"message": "Invalid API key"}},
                )
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_simple_request(),
                    ) as resp:
                        assert resp.status == 401
                        data = await resp.json()
                        assert data["type"] == "error"
                        assert "error" in data
                        assert data["error"]["type"] == "api_error"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    data=b"not json",
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    assert resp.status == 400
                    data = await resp.json()
                    assert data["type"] == "error"
                    assert data["error"]["type"] == "invalid_request_error"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_large_valid_json_payload_is_accepted(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                content_size = 2 * 1024 * 1024
                request_payload = _claude_code_large_request(content_size=content_size)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=request_payload,
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["type"] == "message"
                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        assert len(request_list) == 1
                        sent_json = request_list[0].kwargs["json"]
                        assert sent_json["model"] == request_payload["model"]
                        assert len(sent_json["messages"][0]["content"]) == content_size
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_oversized_json_payload_returns_body_too_large_error(self):
        server = _make_server()
        port = await server.start_async()
        try:
            async with aiohttp.ClientSession() as session:
                request_payload = _claude_code_large_request(content_size=(16 * 1024 * 1024) + 1024)
                async with session.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json=request_payload,
                ) as resp:
                    assert resp.status == 400
                    data = await resp.json()
                    assert data["type"] == "error"
                    assert data["error"]["type"] == "invalid_request_error"
                    assert "too large" in data["error"]["message"].lower()
        finally:
            await server.stop_async()


class TestClaudeCodeUpstreamRequestFormat:
    """Verify the bridge sends correctly formatted Chat Completions requests upstream."""

    @pytest.mark.asyncio
    async def test_tools_translated_to_cc_format(self):
        """Anthropic tool definitions become Chat Completions function tools."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_tools(),
                    ) as resp:
                        assert resp.status == 200

                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]

                        # Verify tools are in Chat Completions format
                        assert "tools" in sent_json
                        assert len(sent_json["tools"]) == 2
                        assert sent_json["tools"][0]["type"] == "function"
                        assert sent_json["tools"][0]["function"]["name"] == "read_file"
                        assert sent_json["tools"][0]["function"]["parameters"]["required"] == ["path"]
                        assert sent_json["tools"][1]["function"]["name"] == "write_file"
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_assistant_tool_use_translated_to_tool_calls(self):
        """Assistant messages with tool_use blocks become tool_calls in CC format."""
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_tool_call_request(),
                    ) as resp:
                        assert resp.status == 200

                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]

                        # Find assistant message
                        assistant_msgs = [m for m in sent_json["messages"] if m["role"] == "assistant"]
                        assert len(assistant_msgs) == 1
                        asst = assistant_msgs[0]
                        assert asst["content"] == "Let me check the weather."
                        assert len(asst["tool_calls"]) == 1
                        tc = asst["tool_calls"][0]
                        assert tc["id"] == "toolu_e2e_001"
                        assert tc["function"]["name"] == "get_weather"
                        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_authorization_header_sent(self):
        """Bridge must send the resolved API key in Authorization header upstream."""
        server = BridgeServer(_StubLauncher(), _StubProvider(), "sk-resolved-key-12345")
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_simple_request(),
                    ) as resp:
                        assert resp.status == 200

                        # Verify the server builds the correct upstream headers
                        headers = server._build_upstream_headers()
                        assert headers["Authorization"] == "Bearer sk-resolved-key-12345"
                        assert headers["Content-Type"] == "application/json"
        finally:
            await server.stop_async()


class TestClaudeCodeTemperaturePassthrough:
    """Verify temperature and top_p are forwarded upstream."""

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                request = _claude_code_simple_request()
                request["temperature"] = 0.5
                request["top_p"] = 0.9
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=request,
                    ) as resp:
                        assert resp.status == 200

                        request_list = m.requests[("POST", aiohttp.client.URL(UPSTREAM_URL))]
                        sent_json = request_list[0].kwargs["json"]
                        assert sent_json["temperature"] == 0.5
                        assert sent_json["top_p"] == 0.9
        finally:
            await server.stop_async()


class TestClaudeCodeMissingUsage:
    """Verify the bridge handles upstream responses with missing or null usage."""

    @pytest.mark.asyncio
    async def test_missing_usage_defaults_to_zero(self):
        server = _make_server()
        port = await server.start_async()
        try:
            upstream_no_usage = {
                "id": "chatcmpl-no-usage",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "No usage info"},
                        "finish_reason": "stop",
                    }
                ],
            }
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=upstream_no_usage)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_simple_request(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["type"] == "message"
                        assert data["usage"]["input_tokens"] == 0
                        assert data["usage"]["output_tokens"] == 0
        finally:
            await server.stop_async()


class TestClaudeCodeEmptyToolArguments:
    """Verify the bridge handles tool calls with empty arguments."""

    @pytest.mark.asyncio
    async def test_empty_arguments_default_to_empty_object(self):
        server = _make_server()
        port = await server.start_async()
        try:
            upstream_empty_args = {
                "id": "chatcmpl-empty-args",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_empty",
                                    "type": "function",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=upstream_empty_args)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_tools(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["stop_reason"] == "tool_use"
                        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
                        assert len(tool_blocks) == 1
                        assert tool_blocks[0]["name"] == "list_files"
                        assert tool_blocks[0]["input"] == {}
        finally:
            await server.stop_async()

    @pytest.mark.asyncio
    async def test_invalid_json_arguments_default_to_empty_object(self):
        server = _make_server()
        port = await server.start_async()
        try:
            upstream_bad_json = {
                "id": "chatcmpl-bad-json",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "not-valid-json{",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, payload=upstream_bad_json)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_request_with_tools(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
                        assert len(tool_blocks) == 1
                        # Invalid JSON should gracefully default to empty dict
                        assert tool_blocks[0]["input"] == {}
        finally:
            await server.stop_async()


class TestClaudeCodeRetryOnUpstreamError:
    """Verify Messages API bridge retries on retryable upstream errors."""

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        server = _make_server()
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as m:
                m.post(UPSTREAM_URL, status=429)
                m.post(UPSTREAM_URL, payload=UPSTREAM_TEXT_RESPONSE)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=_claude_code_simple_request(),
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["type"] == "message"
        finally:
            await server.stop_async()
