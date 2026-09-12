"""Bridge-level tests for the ChatGPT-subscription provider's Responses route.

These cover one claim that no unit test can make: that the model
:meth:`~kitty.bridge.server.BridgeServer._normalize_model` computes — register
row **M1**, the profile's model — actually survives the trip to
:class:`~kitty.providers.openai_subscription.OpenAISubscriptionAdapter` and onto
the wire.  It did not (KBR-160): the handler attaches the raw inbound body as
``cc_request["_original_body"]`` and the adapter built the shipped body out of
*that*, so M1 ran and its result was thrown away.

**Why the measurement is taken at the transport.** The wiring under test is
exactly what a unit test replaces, so stubbing the adapter's ``make_request`` —
the pattern :mod:`tests.bridge.test_bridge_server_bedrock` uses — would stub out
the defect.  These tests therefore drive the real adapter and mock one layer
lower, at the ``curl_cffi`` session it posts through, and read the shipped body
off the recorded call.

**Layer.** ``tests/layers.py`` maps only ``tests/integration/`` away from the
``l1`` fallback, so this file collects as ``l1`` alongside every other loopback
test in this directory.  Marking it ``l3`` would be worse than wrong: ``l3`` is
in ``PENDING_ACTIVATION_LAYERS``, so no CI job selects it and these tests would
never run.
"""

from __future__ import annotations

import contextlib
import json
import time
import unittest.mock
from collections.abc import Iterator
from pathlib import Path

import aiohttp
import pytest

from kitty.auth.oauth_session import OAuthSession
from kitty.bridge.server import BridgeServer
from kitty.providers.openai_subscription import OpenAISubscriptionAdapter

#: The model the profile sets, and therefore the only model that may ship.
_PROFILE_MODEL = "profile-model-XYZ"

#: The model the agent asks for.  Deliberately different from the profile's, and
#: deliberately not a real model name, so a passing assertion cannot be an
#: accident of the two happening to agree.
_CLIENT_MODEL = "client-model-ABC"

#: A well-formed inbound Responses body in the array form.
_INBOUND_INPUT = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello"}],
    }
]

#: A complete Codex SSE response, enough for the non-streaming collector to
#: produce a response rather than raise on an empty body.
_SSE_BODY = (
    b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
    b'data: {"type":"response.completed","response":{"model":"' + _PROFILE_MODEL.encode() + b'",'
    b'"status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}\n\n'
    b"data: [DONE]\n\n"
)


def _write_oauth_session(tmp_path: Path) -> Path:
    """Write an OAuth session file whose tokens are not due to refresh.

    An hour of headroom keeps the refresh path — and with it the clock — out of
    these tests entirely.

    Args:
        tmp_path: The pytest-provided temporary directory to write into.

    Returns:
        The path to the saved session file, which is what the bridge carries as
        its resolved key for this provider.
    """
    now = time.time()
    session = OAuthSession(
        client_id="app_test",
        access_token="at_fresh",
        refresh_token="rt_fresh",
        id_token="eyJhbGciOiJIUzI1NiJ9.e30.fake_sig",
        api_key=None,
        access_token_expires_at=now + 3600,
        api_key_expires_at=now + 3600,
        _file_path=str(tmp_path / "oauth_session.json"),
    )
    session.save()
    return Path(session._file_path)


@contextlib.contextmanager
def _recording_transport(response: object) -> Iterator[unittest.mock.MagicMock]:
    """Replace the adapter's curl session with one that records what it is sent.

    The adapter also opens a short-lived ``aiohttp`` session for OAuth refresh.
    That one is left real and unpatched: the fixture's token is fresh, so it is
    never used, and patching ``aiohttp.ClientSession`` globally would replace the
    test's own HTTP client along with it.

    Args:
        response: The mock ``curl_cffi`` response the recorded ``post`` returns.

    Yields:
        The mock session, whose ``post`` call carries the shipped body.
    """
    mock_session = unittest.mock.AsyncMock()
    mock_session.post = unittest.mock.AsyncMock(return_value=response)
    mock_session.close = unittest.mock.MagicMock()

    with unittest.mock.patch.object(
        OpenAISubscriptionAdapter,
        "_curl_session",
        new_callable=unittest.mock.PropertyMock,
        return_value=mock_session,
    ):
        yield mock_session


def _non_streaming_response() -> unittest.mock.MagicMock:
    """Build a mock curl response carrying a complete SSE body.

    Returns:
        A mock whose ``status_code``, ``text`` and ``content`` are what the
        non-streaming collector reads.
    """
    resp = unittest.mock.MagicMock()
    resp.status_code = 200
    resp.content = _SSE_BODY
    resp.text = _SSE_BODY.decode()
    return resp


def _streaming_response() -> unittest.mock.MagicMock:
    """Build a mock curl response that yields its SSE body in chunks.

    Returns:
        A mock exposing the async ``aiter_content`` the streaming path consumes.
    """
    resp = unittest.mock.MagicMock()
    resp.status_code = 200
    resp.content = _SSE_BODY
    resp.text = _SSE_BODY.decode()

    async def _aiter_content():
        yield _SSE_BODY

    resp.aiter_content = _aiter_content
    return resp


def _make_server(session_path: Path) -> BridgeServer:
    """Build a bridge-mode server whose profile sets a model of its own.

    Args:
        session_path: The OAuth session file standing in for the resolved key.

    Returns:
        A server bound to an ephemeral port, not yet started.
    """
    return BridgeServer(
        adapter=None,
        provider=OpenAISubscriptionAdapter(),
        resolved_key=str(session_path),
        model=_PROFILE_MODEL,
        provider_config={},
        host="127.0.0.1",
        port=0,
    )


def _shipped_body(mock_session: unittest.mock.MagicMock) -> dict:
    """Return the body the adapter posted to the Codex backend.

    Args:
        mock_session: The recording session yielded by :func:`_recording_transport`.

    Returns:
        The ``json=`` payload of the single recorded ``post`` call.

    Raises:
        AssertionError: If the adapter never reached the transport, which would
            make any assertion about the shipped body vacuous.
    """
    assert mock_session.post.await_count == 1, (
        f"expected exactly one upstream POST, got {mock_session.post.await_count}"
    )
    return mock_session.post.await_args.kwargs["json"]


class TestTheProfileModelReachesTheProvider:
    """Register row M1 must hold on both of the Responses route's two paths."""

    @pytest.mark.asyncio()
    async def test_non_streaming_responses_ships_the_profile_model(self, tmp_path: Path) -> None:
        """Ship the profile's model, not the agent's, on the non-streaming path

        This is the path where ``_handle_responses`` attaches ``_original_body``
        before calling the provider.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_non_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": _INBOUND_INPUT,
                            "stream": False,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
            finally:
                await server.stop_async()

        assert _shipped_body(mock_session)["model"] == _PROFILE_MODEL

    @pytest.mark.asyncio()
    async def test_streaming_responses_ships_the_profile_model(self, tmp_path: Path) -> None:
        """Ship the profile's model, not the agent's, on the streaming path

        Codex always streams — register row P17 forces ``stream: True`` — so
        this is the path a real subscription user actually takes, and it
        attaches ``_original_body`` at its own separate site.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "input": _INBOUND_INPUT,
                            "stream": True,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    await resp.read()
            finally:
                await server.stop_async()

        assert _shipped_body(mock_session)["model"] == _PROFILE_MODEL

    @pytest.mark.asyncio()
    async def test_the_agents_model_is_the_only_thing_replaced(self, tmp_path: Path) -> None:
        """Carry the agent's own request through untouched apart from the model

        The fix takes one field from the normalized request.  Everything else on
        the shipped body must still come from the body the agent sent, or the
        cure is worse than KBR-160.
        """
        server = _make_server(_write_oauth_session(tmp_path))
        with _recording_transport(_streaming_response()) as mock_session:
            port = await server.start_async()
            try:
                async with (
                    aiohttp.ClientSession() as client,
                    client.post(
                        f"http://127.0.0.1:{port}/v1/responses",
                        json={
                            "model": _CLIENT_MODEL,
                            "instructions": "You are helpful.",
                            "input": _INBOUND_INPUT,
                            "stream": True,
                        },
                    ) as resp,
                ):
                    assert resp.status == 200, await resp.text()
                    await resp.read()
            finally:
                await server.stop_async()

        body = _shipped_body(mock_session)
        assert body["instructions"] == "You are helpful."
        assert json.dumps(body["input"]).count("Hello") == 1
