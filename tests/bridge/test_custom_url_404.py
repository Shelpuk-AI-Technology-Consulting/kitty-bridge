"""KBR-134 — the 404 diagnostic, wired to a real server rather than a bare function.

`tests/bridge/test_upstream_error_translation.py` proves the message text.  This
file proves the other half: that a ``BridgeServer`` built from a profile actually
reaches that text, with the URL it would really have requested, and that a
fixed-endpoint provider is left alone.

L1 by the path default in ``tests/layers.py``.  It needs no sockets — the same
construction the rest of ``tests/bridge/`` uses — and `TEST_SUITE.md` §2.2 puts a
claim at the lowest layer that can prove it, which for "the instance method reads
the active provider" is here.
"""

from __future__ import annotations

import aiohttp
import pytest

from kitty.bridge.server import BridgeServer
from kitty.providers.custom_openai import CustomOpenAIAdapter
from kitty.providers.openai import OpenAIAdapter


def _server(provider, provider_config: dict | None) -> BridgeServer:
    """Build a bridge server for one provider without starting it.

    Args:
        provider: The provider adapter under test.
        provider_config: The profile's provider configuration.

    Returns:
        An unstarted :class:`BridgeServer`.
    """
    return BridgeServer(
        None,  # type: ignore[arg-type]
        provider,
        "test-key",
        model="mistral-large-latest",
        provider_config=provider_config,
    )


class TestCustomUrl404Wiring:
    """The message reaches a server-formatted error, carrying the real URL."""

    def test_reporters_configuration_produces_the_diagnostic(self):
        """The reporter's stored profile, had it not been normalised, explains itself.

        The base URL here is deliberately the *already-normalised* root: the point
        is that whatever URL the bridge composes is the one the user is shown, so
        a future doubling is visible rather than reported as a missing model.
        """
        server = _server(CustomOpenAIAdapter(), {"base_url": "https://api.mistral.ai/v1"})

        message = server._translate_upstream_error(404, {"detail": "Not Found"})

        assert "https://api.mistral.ai/v1/chat/completions" in message
        assert "must end at the API root" in message
        assert '"/chat/completions"' in message

    def test_message_names_the_doubled_path_when_one_survives(self):
        """A base URL normalisation cannot repair still shows the address in full."""
        server = _server(
            CustomOpenAIAdapter(),
            {"base_url": "https://gw.example/v1/chat/completions?tenant=x"},
        )

        message = server._translate_upstream_error(404, {"detail": "Not Found"})

        # The query-bearing URL is left alone by normalisation (D10), so the
        # malformed composition is what the user sees -- which is the point.
        assert "https://gw.example/v1/chat/completions?tenant=x/chat/completions" in message

    def test_userinfo_never_reaches_the_message(self):
        """Credentials in the configured URL are stripped before it is echoed."""
        server = _server(
            CustomOpenAIAdapter(),
            {"base_url": "https://someone:hunter2@gw.example/v1"},
        )

        message = server._translate_upstream_error(404, {"detail": "Not Found"})

        assert "hunter2" not in message
        assert "someone" not in message
        assert "https://gw.example/v1/chat/completions" in message

    def test_fixed_endpoint_provider_is_untouched(self):
        """A provider whose URL the user never set keeps the old passthrough."""
        server = _server(OpenAIAdapter(), None)

        message = server._translate_upstream_error(404, {"detail": "Not Found"})

        assert "base URL" not in message
        assert message == '{"detail": "Not Found"}'


class TestCustomUrl404ReachesTheClient:
    """The diagnostic must survive into what the agent actually receives.

    The class above proves the message is built correctly.  This proves the far
    more important thing: that a real request, against a real upstream returning
    404, puts that text in front of the user.  Without it the feature could be
    perfect and invisible.
    """

    @staticmethod
    def _messages_request(stream: bool) -> dict:
        """Return a minimal Anthropic Messages request.

        Args:
            stream: Whether to ask for an SSE response.

        Returns:
            The JSON body to POST to ``/v1/messages``.
        """
        return {
            "model": "mistral-large-latest",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "2+2"}],
            "stream": stream,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
    async def test_404_diagnostic_reaches_the_agent(self, stream: bool):
        """A 404 from a user-configured endpoint reaches the agent as the §2.3 text.

        The base URL here is the *normalised* one the reporter's profile now
        resolves to, because that is what the bridge really requests after this
        change — a test pinned to the doubled path would be asserting against a
        URL the fix has already eliminated.  What is proven is the half the unit
        tests cannot see: the message survives the handler and lands in the
        client's payload.

        Args:
            stream: Whether the agent asked for SSE, parametrised because the
                streaming and non-streaming handlers format errors separately.
        """
        from aioresponses import aioresponses

        requested = "https://api.mistral.ai/v1/chat/completions"
        server = _server(CustomOpenAIAdapter(), {"base_url": "https://api.mistral.ai/v1"})
        port = await server.start_async()
        try:
            with aioresponses(passthrough=["http://127.0.0.1"]) as upstream:
                upstream.post(requested, status=404, body=b'{"detail": "Not Found"}', repeat=True)
                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        f"http://127.0.0.1:{port}/v1/messages",
                        json=self._messages_request(stream),
                        headers={"content-type": "application/json"},
                    ) as resp,
                ):
                    payload = await resp.text()
        finally:
            await server.stop_async()

        assert requested in payload, payload
        assert "must end at the API root" in payload, payload
        # Not the quoted path itself: the payload is JSON (or SSE-wrapped JSON),
        # so the quotes around it are escaped and matching them is brittle.
        assert "to the base URL itself" in payload, payload
