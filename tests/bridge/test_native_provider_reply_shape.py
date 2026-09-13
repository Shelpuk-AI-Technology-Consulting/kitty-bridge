"""A non-streaming reply reaches each agent in its own protocol on a native Anthropic provider (KBR-237).

A native-passthrough adapter (``custom_anthropic``, ``zai_coding``, opt-in
``minimax_token``) answers in Anthropic Messages format.  ``_make_upstream_request``
used to hand that body back untranslated whenever the *provider* was native, so
Chat Completions, Responses and Gemini clients received a Messages object they
cannot parse.  Only Claude Code's own ``/v1/messages`` request may take the body
as-is, and the Messages handler must decide from the reply's shape — a native
provider answers in Chat Completions form when the request was not marked native.

Every test drives a real in-process :class:`~kitty.bridge.server.BridgeServer`
against an ``aioresponses`` upstream.
"""

from __future__ import annotations

import aiohttp
import pytest
from aioresponses import aioresponses

from kitty.bridge.server import BridgeServer
from kitty.launchers.base import LauncherAdapter, SpawnConfig
from kitty.providers.base import ProviderAdapter
from kitty.providers.custom_anthropic import CustomAnthropicAdapter
from kitty.providers.minimax_token import MiniMaxTokenAnthropicAdapter
from kitty.providers.zai_anthropic import ZaiAnthropicAdapter
from kitty.types import BridgeProtocol

#: A minimal Anthropic Messages reply, the shape every native adapter's upstream returns.
_MESSAGES_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-6",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


class _StubLauncher(LauncherAdapter):
    """A launcher that selects a chosen bridge protocol and spawns nothing."""

    def __init__(self, protocol: BridgeProtocol) -> None:
        """Remember the protocol the bridge should serve.

        Args:
            protocol: The inbound protocol under test.
        """
        self._protocol = protocol

    @property
    def name(self) -> str:
        """Return the launcher name.

        Returns:
            A fixed placeholder.
        """
        return "stub"

    @property
    def binary_name(self) -> str:
        """Return the launcher binary name.

        Returns:
            A fixed placeholder.
        """
        return "stub"

    @property
    def bridge_protocol(self) -> BridgeProtocol:
        """Return the protocol the bridge serves.

        Returns:
            The protocol given at construction.
        """
        return self._protocol

    def build_spawn_config(self, profile, bridge_port: int, resolved_key: str) -> SpawnConfig:
        """Return an empty spawn configuration.

        Args:
            profile: Unused.
            bridge_port: Unused.
            resolved_key: Unused.

        Returns:
            A spawn configuration that changes nothing.
        """
        return SpawnConfig(env_overrides={}, env_clear=[], cli_args=[])


async def _post(
    protocol: BridgeProtocol,
    path: str,
    payload: dict,
    *,
    provider: ProviderAdapter | None = None,
    reply: dict | None = None,
) -> tuple[int, dict]:
    """POST one non-streaming request through a native bridge and return what the agent received.

    Args:
        protocol: The inbound protocol the bridge serves.
        path: The bridge route to post to.
        payload: The agent's request body.
        provider: The native adapter to serve with; ``custom_anthropic`` when omitted.
        reply: The body the mocked upstream returns; a Messages reply when omitted.

    Returns:
        The HTTP status and the decoded JSON body the agent received.
    """
    provider = provider or CustomAnthropicAdapter()
    server = BridgeServer(_StubLauncher(protocol), provider, "sk-test", host="127.0.0.1", port=0)
    upstream_url = server._build_upstream_url({"model": "claude-opus-4-6"})
    with aioresponses(passthrough=["http://127.0.0.1"]) as mocked:
        mocked.post(upstream_url, payload=reply or _MESSAGES_REPLY)
        await server.start_async()
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(f"http://127.0.0.1:{server.port}{path}", json=payload) as resp,
            ):
                return resp.status, await resp.json()
        finally:
            await server.stop_async()


_NATIVE_ADAPTERS = [
    pytest.param(CustomAnthropicAdapter, id="custom_anthropic"),
    pytest.param(ZaiAnthropicAdapter, id="zai_coding"),
    pytest.param(lambda: MiniMaxTokenAnthropicAdapter(native_messages=True), id="minimax_token-native"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_factory", _NATIVE_ADAPTERS)
async def test_a_chat_completions_client_gets_a_chat_completion(provider_factory):
    """R1 — ``/v1/chat/completions`` receives a ``chat.completion``, not a Messages object.

    Args:
        provider_factory: Builds a native-passthrough adapter.
    """
    provider = provider_factory()
    assert provider.use_native_messages, "precondition: this test is about native adapters"

    status, body = await _post(
        BridgeProtocol.CHAT_COMPLETIONS_API,
        "/v1/chat/completions",
        {"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]},
        provider=provider,
    )

    assert status == 200
    assert body.get("object") == "chat.completion", body
    assert body["choices"][0]["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_a_responses_client_gets_a_response():
    """R2 — ``/v1/responses`` receives a ``response`` carrying the reply text."""
    status, body = await _post(
        BridgeProtocol.RESPONSES_API,
        "/v1/responses",
        {"model": "claude-opus-4-6", "input": "hi"},
    )

    assert status == 200
    assert body.get("object") == "response", body
    assert "hello" in str(body.get("output"))


@pytest.mark.asyncio
async def test_a_gemini_client_gets_candidates():
    """R3 — ``:generateContent`` receives Gemini ``candidates`` carrying the reply text."""
    status, body = await _post(
        BridgeProtocol.GEMINI_API,
        "/v1beta/models/claude-opus-4-6:generateContent",
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )

    assert status == 200
    assert "candidates" in body, body
    assert "hello" in str(body["candidates"])


@pytest.mark.asyncio
async def test_claude_code_still_gets_the_upstream_messages_reply():
    """R4 — the native route's own client keeps receiving the upstream body as-is."""
    status, body = await _post(
        BridgeProtocol.MESSAGES_API,
        "/v1/messages",
        {"model": "claude-opus-4-6", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert status == 200
    assert body == _MESSAGES_REPLY


@pytest.mark.asyncio
async def test_the_messages_handler_translates_a_chat_completions_reply_even_on_a_native_provider():
    """R5 — ``/v1/messages`` decides from the reply's shape, not from the provider.

    A native-provider endpoint can answer in Chat Completions form — the
    adapters' own ``translate_from_upstream`` passes such a body through — so
    Claude Code must still receive a Messages reply rather than the raw object.
    """
    cc_reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "other",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }

    status, body = await _post(
        BridgeProtocol.MESSAGES_API,
        "/v1/messages",
        {"model": "claude-opus-4-6", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        reply=cc_reply,
    )

    assert status == 200
    assert body.get("type") == "message", body
    assert body["content"][0]["text"] == "hello"
