"""Tests for bridge server integration with Z.AI Anthropic provider."""

import json

import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter
from kitty.types import BridgeProtocol


class _FakeLauncher(LauncherAdapter):
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
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, model=None):
        return {}

    def prepare_launch(self, spawn_config):
        pass

    def cleanup_launch(self, spawn_config):
        pass


@pytest.fixture
def server():
    provider = ZaiAnthropicAdapter()
    adapter = _FakeLauncher()
    srv = BridgeServer(adapter, provider, "sk-zai-test123", host="127.0.0.1", port=0)
    return srv


class TestBridgeZaiAnthropicURL:
    def test_upstream_url_uses_messages_path(self, server):
        url = server._build_upstream_url()
        assert url == "https://api.z.ai/api/anthropic/v1/messages"


class TestBridgeZaiAnthropicHeaders:
    def test_uses_bearer_auth(self, server):
        headers = server._build_upstream_headers()
        assert headers["Authorization"] == "Bearer sk-zai-test123"

    def test_has_anthropic_version(self, server):
        headers = server._build_upstream_headers()
        assert "anthropic-version" in headers

    def test_no_x_api_key(self, server):
        headers = server._build_upstream_headers()
        assert "x-api-key" not in headers


class TestBridgeZaiAnthropicNonStreaming:
    """Verify native Messages passthrough for non-streaming requests."""

    @pytest.mark.asyncio
    async def test_effort_passthrough(self, server):
        """The effort parameter must reach Z.AI unchanged."""
        upstream_response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Done"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        sent_body = {}

        def capture_body(url, **kwargs):
            nonlocal sent_body
            sent_body = kwargs.get("json", {})
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(upstream_response),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.z.ai/api/anthropic/v1/messages", callback=capture_body)

            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "claude-sonnet-4-6",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 4096,
                            "effort": "xhigh",
                            "thinking": {"type": "adaptive"},
                        },
                    ) as resp,
                ):
                    body = await resp.json()
                    assert resp.status == 200, body
                    assert body["content"][0]["text"] == "Done"
            finally:
                await server.stop_async()

        # Verify the upstream received effort and thinking directly
        assert sent_body.get("effort") == "xhigh"
        assert sent_body.get("thinking") == {"type": "adaptive"}
        # Verify no CC-format artifacts leaked through
        assert "choices" not in sent_body
        assert "_reasoning_effort" not in sent_body

    @pytest.mark.asyncio
    async def test_thinking_with_budget_passthrough(self, server):
        """The thinking budget_tokens must reach Z.AI unchanged."""
        upstream_response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hmm"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        sent_body = {}

        def capture_body(url, **kwargs):
            nonlocal sent_body
            sent_body = kwargs.get("json", {})
            return CallbackResult(
                status=200,
                content_type="application/json",
                body=json.dumps(upstream_response),
            )

        with aioresponses(passthrough=["http://127.0.0.1"]) as m:
            m.post("https://api.z.ai/api/anthropic/v1/messages", callback=capture_body)

            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/messages",
                        json={
                            "model": "claude-sonnet-4-6",
                            "messages": [{"role": "user", "content": "think"}],
                            "max_tokens": 16000,
                            "thinking": {"type": "enabled", "budget_tokens": 10000},
                        },
                    ) as resp,
                ):
                    body = await resp.json()
                    assert resp.status == 200, body
            finally:
                await server.stop_async()

        assert sent_body.get("thinking") == {"type": "enabled", "budget_tokens": 10000}
        assert sent_body.get("max_tokens") == 16000
