"""Tests for bridge server integration with Bedrock provider (custom transport)."""

from unittest.mock import AsyncMock, patch

import pytest

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter
from kitty.providers.bedrock import BedrockAdapter
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
        return BridgeProtocol.RESPONSES_API

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, model=None):
        return {}

    def prepare_launch(self, spawn_config):
        pass

    def cleanup_launch(self, spawn_config):
        pass


BEDROCK_RESPONSE_TEXT = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "4"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 5},
}


class TestBridgeBedrockCustomTransport:
    """Verify the bridge delegates to custom transport for Bedrock."""

    @pytest.mark.asyncio
    async def test_non_streaming_uses_custom_transport(self):
        """Bridge should call provider.make_request() instead of aiohttp."""
        provider = BedrockAdapter()
        adapter = _FakeLauncher()
        server = BridgeServer(
            adapter,
            provider,
            "AKID:SECRET",
            host="127.0.0.1",
            port=0,
            provider_config={"region": "us-east-1"},
        )

        # Mock make_request to avoid real boto3 call
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_make_request = AsyncMock(return_value=mock_response)
        with patch.object(provider, "make_request", mock_make_request):
            await server.start_async()
            try:
                import aiohttp

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{server.port}/v1/responses",
                        json={
                            "model": "anthropic.claude-sonnet-4-20250514",
                            "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
                        },
                    ) as resp,
                ):
                    assert resp.status == 200
            finally:
                await server.stop_async()

        # Verify make_request was called (not aiohttp upstream)
        mock_make_request.assert_called_once()
        # Verify cc_request contains the provider metadata
        call_args = mock_make_request.call_args[0][0]
        assert call_args["_resolved_key"] == "AKID:SECRET"
        assert call_args["_provider_config"] == {"region": "us-east-1"}
