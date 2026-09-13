"""Native Messages passthrough ships the agent's thinking configuration untouched (KBR-203, R5).

``tests/providers/test_anthropic_thinking_cache_stability.py`` pins what the
**translated** route does to ``thinking``: kitty replaces the agent's budget with
one derived from ``max_tokens`` (register row P5c), which changes the rendered
prompt and so the prompt cache.  This file establishes that the defect is
specific to that route — on native passthrough the agent's ``thinking`` and
``effort`` reach the upstream unchanged as JSON values.  The cache cost P5c
records is therefore confined to the translated route and does not erode the
native route, the one KBR-197's CB-3 relies on to keep cache breakpoints.

Driven through a real in-process :class:`~kitty.bridge.server.BridgeServer`
rather than the adapter hook, because the native route is decided in
``_handle_messages`` (``use_native_messages``) before any adapter runs.  The
request is non-streaming; the streaming branch forks *after* the same
``cc_request`` is built, so it ships the same body.
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from aioresponses import CallbackResult, aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter
from kitty.providers.anthropic import AnthropicAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter
from kitty.types import BridgeProtocol

#: Everything the translated route rewrites or drops: an agent budget far from
#: ``max_tokens - 1``, and a ``display`` two of these adapters withhold there.
_AGENT_THINKING = {"type": "enabled", "budget_tokens": 2048, "display": "summarized"}
_AGENT_EFFORT = "high"


class _FakeLauncher(LauncherAdapter):
    """A Messages-API launcher that spawns nothing."""

    @property
    def name(self) -> str:
        """Return the launcher name.

        Returns:
            A fixed placeholder.
        """
        return "fake"

    @property
    def binary_name(self) -> str:
        """Return the launcher binary name.

        Returns:
            A fixed placeholder.
        """
        return "fake"

    @property
    def agent_name(self) -> str:
        """Return the agent name.

        Returns:
            A fixed placeholder.
        """
        return "fake"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol the bridge serves to this launcher.

        Returns:
            :attr:`BridgeProtocol.MESSAGES_API`, so ``/v1/messages`` is served.
        """
        return BridgeProtocol.MESSAGES_API

    def build_spawn_config(self, profile, bridge_port, resolved_key, *, model=None):
        """Return an empty spawn configuration.

        Args:
            profile: Unused.
            bridge_port: Unused.
            resolved_key: Unused.
            model: Unused.

        Returns:
            An empty dict.
        """
        return {}

    def prepare_launch(self, spawn_config):
        """Do nothing.

        Args:
            spawn_config: Unused.
        """

    def cleanup_launch(self, spawn_config):
        """Do nothing.

        Args:
            spawn_config: Unused.
        """


def _anthropic_message() -> dict:
    """Return a minimal successful Anthropic Messages response.

    Returns:
        A non-streaming ``message`` object.
    """
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
        "model": "claude-opus-4-6",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


async def _upstream_body(provider: AnthropicAdapter) -> dict:
    """POST an agent request through a real bridge and return the body the upstream received.

    Args:
        provider: A native-passthrough adapter.

    Returns:
        The JSON body captured at the upstream.
    """
    server = BridgeServer(_FakeLauncher(), provider, "sk-test", host="127.0.0.1", port=0)
    # The server composes the upstream URL itself, so the mock matches what it will call.
    upstream_url = server._build_upstream_url({"model": "claude-opus-4-6"})
    sent: dict = {}

    def capture(url, **kwargs):
        """Record the upstream JSON body and answer with a canned message.

        Args:
            url: The upstream URL aioresponses matched.
            **kwargs: The request's keyword arguments; ``json`` is the body.

        Returns:
            A 200 JSON response carrying a minimal Messages reply.
        """
        sent.update(kwargs["json"])
        return CallbackResult(status=200, content_type="application/json", body=json.dumps(_anthropic_message()))

    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        mocked.post(upstream_url, callback=capture)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{server.port}/v1/messages",
                    json={
                        "model": "claude-opus-4-6",
                        "max_tokens": 8000,
                        "messages": [{"role": "user", "content": "hi"}],
                        "thinking": _AGENT_THINKING,
                        "effort": _AGENT_EFFORT,
                    },
                ) as resp,
            ):
                assert resp.status == 200, await resp.text()
        finally:
            await server.stop_async()

    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_factory",
    [ZaiAnthropicAdapter, CustomAnthropicAdapter, lambda: MiniMaxTokenAnthropicAdapter(native_messages=True)],
    ids=["zai_coding", "custom_anthropic", "minimax_token-native"],
)
async def test_native_route_ships_the_agents_thinking_and_effort_verbatim(provider_factory):
    """R5 — ``thinking`` (budget and ``display`` included) and ``effort`` arrive as the agent sent them.

    Args:
        provider_factory: Builds a native-passthrough adapter.
    """
    provider = provider_factory()
    assert provider.use_native_messages, "precondition: this test is about the native route"

    sent = await _upstream_body(provider)

    assert sent["thinking"] == _AGENT_THINKING
    assert sent["effort"] == _AGENT_EFFORT
    assert sent["max_tokens"] == 8000
