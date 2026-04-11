"""Tests for bridge server integration with Anthropic provider."""

import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter
from kitty.providers.anthropic import AnthropicAdapter
from kitty.types import BridgeProtocol


class _FakeLauncher(LauncherAdapter):
    """Minimal launcher that uses RESPONSES_API for testing."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def binary_name(self) -> str:
        return "fake"

    @property
    def agent_name(self) -> str:
        return "fake"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        return BridgeProtocol.RESPONSES_API

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, model=None):
        return {}

    def prepare_launch(self, spawn_config):
        pass

    def cleanup_launch(self, spawn_config):
        pass


@pytest.fixture
def server():
    provider = AnthropicAdapter()
    adapter = _FakeLauncher()
    srv = BridgeServer(adapter, provider, "sk-ant-test123", host="127.0.0.1", port=0)
    return srv


class TestBridgeAnthropicURL:
    def test_upstream_url_uses_messages_path(self, server):
        url = server._build_upstream_url()
        assert url == "https://api.anthropic.com/v1/messages"

    def test_upstream_url_not_chat_completions(self, server):
        url = server._build_upstream_url()
        assert "/chat/completions" not in url


class TestBridgeAnthropicHeaders:
    def test_uses_x_api_key(self, server):
        headers = server._build_upstream_headers()
        assert headers["x-api-key"] == "sk-ant-test123"

    def test_no_bearer_auth(self, server):
        headers = server._build_upstream_headers()
        assert "Authorization" not in headers

    def test_has_anthropic_version(self, server):
        headers = server._build_upstream_headers()
        assert "anthropic-version" in headers


class TestBridgeAnthropicRequestTranslation:
    """Verify the bridge applies translate_to_upstream before sending upstream."""

    @pytest.mark.asyncio
    async def test_non_streaming_sends_anthropic_format(self, server):
        """Bridge should translate CC → Anthropic Messages before upstream call."""
        anthropic_response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "4"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.anthropic.com/v1/messages", payload=anthropic_response)

            await server.start_async()
            try:
                # Simulate a Responses API request that gets translated to CC internally
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/responses",
                        json={
                            "model": "claude-sonnet-4-6",
                            "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
                    body = await resp.json()
                    # Should contain the translated response
                    assert body is not None
            finally:
                await server.stop_async()
